//! Kernel-syscall wrappers for `flock(2)` acquire/release.
//!
//! Three entry points, each gated through
//! [`super::fs_filter::reject_remote_fs`] so a misconfigured lockfile
//! path on NFS / CIFS / SMB2 / CephFS / AFS / FUSE surfaces actionably
//! at open time rather than silently returning an unserialized fd:
//!
//!  - [`materialize`] — create the lockfile inode without acquiring
//!    a lock. Used by the DISCOVER phase of
//!    `acquire_llc_plan` so the snapshot pass has a target inode
//!    for the subsequent `/proc/locks` match without contending
//!    with live acquirers.
//!  - [`try_flock`] — non-blocking acquire. Returns `Ok(None)` on
//!    `EWOULDBLOCK` so the caller can decide whether to retry, poll,
//!    or surface contention.
//!  - [`block_flock`] — blocking acquire. Parks the calling thread
//!    in the kernel until the lock is available. Used after
//!    [`try_flock`] returns `None` for callers that want to wait
//!    indefinitely; callers with a deadline use
//!    [`super::acquire::acquire_flock_with_timeout`] instead.
//!
//! All three open with `O_CREAT | O_RDWR | O_CLOEXEC | 0o666` so the
//! resulting fd matches the rest of the crate's lockfile contract:
//!
//!  - `O_CLOEXEC` keeps the lock from leaking across `exec(2)` into
//!    spawned subprocesses (cargo subcommands, build pipeline,
//!    initramfs compressor) where the parent's `OwnedFd::drop`
//!    would not release a child-held flock.
//!  - 0o666 mode matches a peer first-acquire so the file's owner
//!    and permissions don't depend on creation order.

use anyhow::Result;
use std::os::fd::OwnedFd;
use std::path::Path;

use super::FlockMode;
use super::fs_filter::reject_remote_fs;

/// Open a lockfile with the crate-wide flock contract: refuses
/// remote filesystems via [`reject_remote_fs`], then opens with
/// `O_CREAT | O_RDWR | O_CLOEXEC | 0o666`. The three module entry
/// points ([`materialize`], [`try_flock`], [`block_flock`]) share
/// this open shape; centralizing it here means a future flag change
/// (or an addition to the remote-fs deny-list) lands in one place
/// instead of drifting across three call sites.
///
/// `O_CLOEXEC` is mandatory: a leaked fd across `exec(2)` (cargo
/// subcommand, build-pipeline subprocess, initramfs compressor)
/// would keep the lock alive in the child after the parent's
/// `OwnedFd::drop`, producing phantom holders the next acquirer
/// would blame on the wrong pid.
///
/// 0o666 mode matches a peer first-acquire so the file's owner and
/// permissions don't depend on creation order.
fn open_lockfile(path: &Path) -> Result<OwnedFd> {
    use rustix::fs::{Mode, OFlags, open};

    reject_remote_fs(path)?;
    open(
        path,
        OFlags::CREATE | OFlags::RDWR | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o666),
    )
    .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))
}

/// Ensure the lockfile exists on disk without acquiring a lock.
/// Used by the DISCOVER phase of `acquire_llc_plan` (see
/// `discover_llc_snapshots` in `crate::vmm::host_topology`): the
/// snapshot pass needs every per-LLC lockfile's inode to exist so a
/// subsequent `/proc/locks` match has a target, but DISCOVER itself
/// must not contend with peer acquires.
///
/// Opens through [`open_lockfile`] so the resulting inode and fd
/// mode match what a first-time acquirer would create. Immediately
/// closes the fd — `OwnedFd::drop` releases the open-file
/// description and (since no flock was ever taken on this fd)
/// cannot release a lock held by a peer fd.
pub(crate) fn materialize<P: AsRef<Path>>(path: P) -> Result<()> {
    let fd = open_lockfile(path.as_ref())?;
    drop(fd);
    Ok(())
}

/// Open a lock file and attempt `flock` with `LOCK_NB`.
///
/// Creates the file with mode 0o666 if absent. Returns
/// `Ok(Some(fd))` on successful acquire, `Ok(None)` on
/// `EWOULDBLOCK` (peer already holds an incompatible lock), and
/// propagates other errors. The returned fd owns the open-file
/// description; dropping it closes the fd AND releases the kernel
/// flock (the kernel releases `flock(2)` only when the last fd
/// referring to its OFD closes — `OwnedFd::drop` is what makes that
/// work).
///
/// `O_CLOEXEC` is mandatory: a leaked fd across `exec(2)` (cargo
/// subcommand, build-pipeline subprocess, initramfs compressor) would
/// keep the lock alive in the child process after the parent's
/// `OwnedFd::drop` runs, producing phantom holders the next acquirer
/// would blame on the wrong pid.
///
/// Calls `super::fs_filter::reject_remote_fs` before the open to
/// fail-fast on NFS / CIFS / SMB2 / CEPH / AFS / FUSE — see the
/// module-level rationale.
///
/// Accepts any `AsRef<Path>` so `&str`, `&Path`, `&PathBuf`, and
/// `String` callers all work without string-ifying round trips. LLC
/// lockfile paths are built as `String` via `format!` and cache
/// lockfile paths are built as `PathBuf` via `Path::join` — both
/// pass straight through.
pub fn try_flock<P: AsRef<Path>>(path: P, mode: FlockMode) -> Result<Option<OwnedFd>> {
    use rustix::fs::{FlockOperation, flock};

    let path = path.as_ref();
    let fd = open_lockfile(path)?;
    let op = match mode {
        FlockMode::Exclusive => FlockOperation::NonBlockingLockExclusive,
        FlockMode::Shared => FlockOperation::NonBlockingLockShared,
    };
    match flock(&fd, op) {
        Ok(()) => Ok(Some(fd)),
        Err(e) if e == rustix::io::Errno::WOULDBLOCK => Ok(None),
        Err(e) => anyhow::bail!("flock {}: {e}", path.display()),
    }
}

/// Blocking variant of [`try_flock`]. Opens the lockfile (creating
/// it if absent), then issues a blocking `flock(2)` that parks the
/// caller in the kernel until the lock is available. Use after
/// [`try_flock`] returns `None` to wait for a live peer to finish.
pub fn block_flock<P: AsRef<Path>>(path: P, mode: FlockMode) -> Result<OwnedFd> {
    use rustix::fs::{FlockOperation, flock};

    let path = path.as_ref();
    let fd = open_lockfile(path)?;
    let op = match mode {
        FlockMode::Exclusive => FlockOperation::LockExclusive,
        FlockMode::Shared => FlockOperation::LockShared,
    };
    flock(&fd, op).map_err(|e| anyhow::anyhow!("flock (blocking) {}: {e}", path.display()))?;
    Ok(fd)
}

/// RT signal used to interrupt a deadline-bounded blocking `flock(2)`
/// ([`block_flock_deadline`]). `SIGRTMIN+4`: `SIGRTMIN` itself is the
/// vCPU kick (`crate::vmm::vcpu::vcpu_signal`); +4 leaves headroom for
/// future vCPU-adjacent signals without colliding. The handler is a
/// no-op — the entire point is the `EINTR` the delivery forces out of
/// the blocked `flock`.
fn flock_deadline_signal() -> libc::c_int {
    libc::SIGRTMIN() + 4
}

/// Install the no-op handler for [`flock_deadline_signal`] exactly
/// once, process-wide. `sa_flags` deliberately OMITS `SA_RESTART`:
/// the kernel must NOT transparently restart the interrupted `flock`,
/// because the `EINTR` return is the deadline mechanism. The signal
/// is delivered thread-directed (`SIGEV_THREAD_ID` targeting the
/// blocked thread), so no other thread observes it.
fn install_flock_deadline_handler() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        extern "C" fn noop(_: libc::c_int) {}
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = noop as *const () as usize;
            sa.sa_flags = 0; // no SA_RESTART — EINTR is load-bearing.
            libc::sigemptyset(&mut sa.sa_mask);
            let rc = libc::sigaction(flock_deadline_signal(), &sa, std::ptr::null_mut());
            assert_eq!(
                rc,
                0,
                "install_flock_deadline_handler: sigaction(SIGRTMIN+4) failed: {} — \
                 deadline-bounded flock waits would park forever",
                std::io::Error::last_os_error(),
            );
        }
    });
}

/// Prepare the process-wide no-op handler used to interrupt a blocking
/// `flock(2)` without restarting it.
///
/// The progress-aware host-topology queue installs this before it starts its
/// monitor thread. Doing so closes the race where the monitor could reach its
/// deadline before the waiter enters `flock`: the signal is harmless before
/// the syscall, and the monitor re-fires until the waiter acknowledges it.
pub(crate) fn prepare_flock_interrupt_handler() {
    install_flock_deadline_handler();
}

/// Interrupt `target_tid`'s blocking flock immediately.
///
/// This is the progress monitor's deadline wake and its fail-safe if inotify
/// itself fails. It is intentionally thread-directed; no unrelated ktstr
/// thread should observe the EINTR used to unwind the queue wait.
pub(crate) fn interrupt_flock_thread(target_tid: libc::pid_t) -> Result<()> {
    install_flock_deadline_handler();
    let rc = unsafe {
        libc::syscall(
            libc::SYS_tgkill,
            libc::getpid(),
            target_tid,
            flock_deadline_signal(),
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        anyhow::bail!(
            "tgkill for progress-aware flock: {}",
            std::io::Error::last_os_error()
        )
    }
}

/// Block once on a single open file description until the flock is granted or
/// [`flock_deadline_signal`] interrupts the syscall.
///
/// Keeping the same fd and issuing exactly one blocking syscall is important:
/// an interrupted/re-opened loop loses its place in the kernel's flock wait
/// queue. The host-topology queue's inotify monitor sends the interrupt only
/// after a full no-progress deadline and a final holder-identity check.
pub(crate) fn block_flock_interruptible<P: AsRef<Path>>(
    path: P,
    mode: FlockMode,
) -> Result<FlockWait> {
    use rustix::fs::{FlockOperation, flock};

    install_flock_deadline_handler();
    let path = path.as_ref();
    let fd = open_lockfile(path)?;
    let op = match mode {
        FlockMode::Exclusive => FlockOperation::LockExclusive,
        FlockMode::Shared => FlockOperation::LockShared,
    };
    match flock(&fd, op) {
        Ok(()) => Ok(FlockWait::Granted(fd)),
        Err(e) if e == rustix::io::Errno::INTR => Ok(FlockWait::DeadlineExpired),
        Err(e) => anyhow::bail!("flock (interruptible) {}: {e}", path.display()),
    }
}

/// RAII POSIX per-thread interval timer that delivers
/// [`flock_deadline_signal`] to the CREATING thread. Armed with a
/// one-shot expiry at the deadline plus a periodic re-fire, so a
/// signal that lands in the race window between the deadline check
/// and the next blocking `flock` entry cannot strand the waiter —
/// the next interval tick interrupts it again.
struct DeadlineTimer {
    id: libc::timer_t,
}

impl DeadlineTimer {
    /// Create and arm a timer targeted at the calling thread: first
    /// fire at `tick` (clamped to the remaining deadline budget),
    /// re-firing every `tick` after.
    fn arm(deadline: std::time::Instant, tick: std::time::Duration) -> Result<Self> {
        install_flock_deadline_handler();
        let remaining = deadline
            .saturating_duration_since(std::time::Instant::now())
            .min(tick)
            .max(std::time::Duration::from_millis(1));
        unsafe {
            let mut sev: libc::sigevent = std::mem::zeroed();
            sev.sigev_notify = libc::SIGEV_THREAD_ID;
            sev.sigev_signo = flock_deadline_signal();
            sev.sigev_notify_thread_id = libc::gettid();
            let mut id: libc::timer_t = std::mem::zeroed();
            if libc::timer_create(libc::CLOCK_MONOTONIC, &mut sev, &mut id) != 0 {
                anyhow::bail!(
                    "timer_create for deadline-bounded flock: {}",
                    std::io::Error::last_os_error()
                );
            }
            let spec = libc::itimerspec {
                it_value: libc::timespec {
                    tv_sec: remaining.as_secs() as libc::time_t,
                    tv_nsec: remaining.subsec_nanos() as libc::c_long,
                },
                // Periodic re-fire closes the check-then-block race:
                // if a fire lands between the clock check and flock
                // entry, the next tick still interrupts.
                it_interval: libc::timespec {
                    tv_sec: tick.as_secs() as libc::time_t,
                    tv_nsec: tick.subsec_nanos() as libc::c_long,
                },
            };
            if libc::timer_settime(id, 0, &spec, std::ptr::null_mut()) != 0 {
                let e = std::io::Error::last_os_error();
                libc::timer_delete(id);
                anyhow::bail!("timer_settime for deadline-bounded flock: {e}");
            }
            Ok(DeadlineTimer { id })
        }
    }
}

impl Drop for DeadlineTimer {
    fn drop(&mut self) {
        // SAFETY: `id` came from a successful timer_create and is
        // deleted exactly once (Drop).
        unsafe {
            libc::timer_delete(self.id);
        }
    }
}

/// Outcome of one [`block_flock_step`] blocking cycle.
#[derive(Debug)]
pub enum FlockWait {
    /// Lock granted before the deadline; the fd holds it (RAII).
    Granted(OwnedFd),
    /// The periodic tick fired before the deadline: the caller gets
    /// control back to re-evaluate (re-plan, re-scan alternatives)
    /// and decide whether to call the step again. The kernel
    /// wait-queue position is released with the step's fd.
    Tick,
    /// The deadline passed while blocked.
    DeadlineExpired,
}

/// One deadline-bounded BLOCKING `flock(2)` cycle with a periodic
/// re-evaluation tick.
///
/// Parks the calling thread in the kernel's per-inode flock wait
/// queue — the wake-latency primitive the polling loops lacked: the
/// kernel wakes the waiter at release time, so a freed lock is
/// picked up in microseconds, not at the next poll interval.
///
/// Returns:
///  - [`FlockWait::Granted`] when the lock is granted,
///  - [`FlockWait::Tick`] when `tick` elapses first (caller
///    re-evaluates live state — e.g. a queue head re-scanning for a
///    fully-free alternative target — then typically calls again),
///  - [`FlockWait::DeadlineExpired`] when `deadline` passes,
///  - `Err` on open / unexpected-errno failures.
///
/// Mechanism: a thread-targeted `CLOCK_MONOTONIC` POSIX timer
/// (`DeadlineTimer`) delivers `SIGRTMIN+4` (no-op handler, no
/// `SA_RESTART`) to this thread every `tick` (clamped to the
/// remaining deadline budget), forcing the blocked `flock` to return
/// `EINTR`; the step maps that to `Tick` or `DeadlineExpired` by the
/// clock. The interval re-fire means a signal landing in the gap
/// between arming and blocking cannot strand the waiter.
pub fn block_flock_step<P: AsRef<Path>>(
    path: P,
    mode: FlockMode,
    deadline: std::time::Instant,
    tick: std::time::Duration,
) -> Result<FlockWait> {
    use rustix::fs::{FlockOperation, flock};

    if std::time::Instant::now() >= deadline {
        return Ok(FlockWait::DeadlineExpired);
    }
    let path = path.as_ref();
    let fd = open_lockfile(path)?;
    let op = match mode {
        FlockMode::Exclusive => FlockOperation::LockExclusive,
        FlockMode::Shared => FlockOperation::LockShared,
    };
    let _timer = DeadlineTimer::arm(deadline, tick)?;
    match flock(&fd, op) {
        Ok(()) => Ok(FlockWait::Granted(fd)),
        Err(e) if e == rustix::io::Errno::INTR => {
            if std::time::Instant::now() >= deadline {
                Ok(FlockWait::DeadlineExpired)
            } else {
                Ok(FlockWait::Tick)
            }
        }
        Err(e) => anyhow::bail!("flock (blocking, deadline) {}: {e}", path.display()),
    }
}

/// Deadline-bounded BLOCKING `flock(2)` with no tick observation:
/// loops [`block_flock_step`] until granted or the deadline expires.
/// Used where the caller has nothing to re-evaluate mid-wait — e.g.
/// a contended acquirer parking on the acquisition-queue ticket.
///
/// Returns `Ok(Some(fd))` when granted before `deadline`, `Ok(None)`
/// on expiry, `Err` on open / unexpected-errno failures.
pub fn block_flock_deadline<P: AsRef<Path>>(
    path: P,
    mode: FlockMode,
    deadline: std::time::Instant,
) -> Result<Option<OwnedFd>> {
    let path = path.as_ref();
    loop {
        match block_flock_step(path, mode, deadline, std::time::Duration::from_millis(500))? {
            FlockWait::Granted(fd) => return Ok(Some(fd)),
            FlockWait::Tick => continue,
            FlockWait::DeadlineExpired => return Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`try_flock`] sets `O_CLOEXEC` on the returned fd. Earlier
    /// revisions missed this flag, which leaked flock-held fds
    /// through `execve` into child processes — the child inherited
    /// the lock, broke assumptions about RAII scope, and
    /// manifested as phantom holders in `/proc/locks` long after
    /// the parent had dropped its guard.
    ///
    /// Verifies the bit directly via `fcntl(F_GETFD)` rather than
    /// asserting via a side-effect (forking an exec'd child is
    /// noisier and harder to match). Failure mode: if the bit is
    /// cleared by a future refactor that re-opens the fd without
    /// re-applying O_CLOEXEC, this test fails the build.
    #[test]
    fn try_flock_sets_cloexec_on_returned_fd() {
        use std::os::fd::AsRawFd;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("cloexec.lock");
        let fd = try_flock(&path, FlockMode::Exclusive)
            .expect("try_flock must succeed on fresh tempfile")
            .expect("EX must acquire on clean pool");

        // SAFETY: fd is a valid OwnedFd — fcntl F_GETFD is a pure
        // accessor, no concurrent modification, no ownership move.
        let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFD) };
        assert!(
            flags >= 0,
            "fcntl F_GETFD must succeed on our fd; got errno={}",
            std::io::Error::last_os_error(),
        );
        assert_eq!(
            flags & libc::FD_CLOEXEC,
            libc::FD_CLOEXEC,
            "FD_CLOEXEC must be set on try_flock-returned fd; \
             flags=0x{flags:x}. Without it, exec'd children \
             inherit the flock and produce phantom holders.",
        );

        drop(fd);
    }

    /// `block_flock_step` tri-state contract: while a peer holds
    /// `LOCK_EX`, a step with a short tick returns `Tick` (the
    /// thread-targeted timer interrupted the blocked flock via EINTR —
    /// this is the empirical pin for the SIGRTMIN+4 deadline
    /// machinery), a step whose deadline lapses returns
    /// `DeadlineExpired`, and once the peer releases, a step returns
    /// `Granted` carrying the lock.
    #[test]
    fn block_flock_step_tick_expiry_and_grant() {
        use tempfile::TempDir;
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("step.lock");
        let peer = try_flock(&path, FlockMode::Exclusive)
            .expect("open")
            .expect("peer EX on fresh file");

        // Tick: deadline far, tick near — the timer must interrupt
        // the blocked flock rather than leaving it parked.
        let start = std::time::Instant::now();
        let step = block_flock_step(
            &path,
            FlockMode::Exclusive,
            std::time::Instant::now() + std::time::Duration::from_secs(30),
            std::time::Duration::from_millis(150),
        )
        .expect("step must not error");
        assert!(
            matches!(step, FlockWait::Tick),
            "near tick + far deadline must interrupt as Tick; got {step:?}",
        );
        assert!(
            start.elapsed() < std::time::Duration::from_secs(5),
            "the tick must fire near its interval, not hang; elapsed={:?}",
            start.elapsed(),
        );

        // DeadlineExpired: deadline nearer than the tick.
        let step = block_flock_step(
            &path,
            FlockMode::Exclusive,
            std::time::Instant::now() + std::time::Duration::from_millis(150),
            std::time::Duration::from_secs(30),
        )
        .expect("step must not error");
        assert!(
            matches!(step, FlockWait::DeadlineExpired),
            "lapsed deadline must report DeadlineExpired; got {step:?}",
        );

        // Granted: kernel wakeup when the peer releases mid-block.
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            drop(peer);
        });
        let start = std::time::Instant::now();
        let step = block_flock_step(
            &path,
            FlockMode::Exclusive,
            std::time::Instant::now() + std::time::Duration::from_secs(30),
            std::time::Duration::from_secs(30),
        )
        .expect("step must not error");
        releaser.join().expect("releaser");
        assert!(
            matches!(step, FlockWait::Granted(_)),
            "release must grant the blocked step via kernel wakeup; got {step:?}",
        );
        assert!(
            start.elapsed() >= std::time::Duration::from_millis(150),
            "grant must come from the release, not an instant win; elapsed={:?}",
            start.elapsed(),
        );
    }

    /// `block_flock_deadline` wraps the step loop: bounded `None` on a
    /// never-released peer, `Some(fd)` when the peer releases within
    /// the deadline.
    #[test]
    fn block_flock_deadline_bounds_and_grants() {
        use tempfile::TempDir;
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("deadline.lock");
        let peer = try_flock(&path, FlockMode::Exclusive)
            .expect("open")
            .expect("peer EX on fresh file");

        let got = block_flock_deadline(
            &path,
            FlockMode::Exclusive,
            std::time::Instant::now() + std::time::Duration::from_millis(250),
        )
        .expect("must not error");
        assert!(got.is_none(), "held lock past the deadline must yield None");

        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(150));
            drop(peer);
        });
        let got = block_flock_deadline(
            &path,
            FlockMode::Exclusive,
            std::time::Instant::now() + std::time::Duration::from_secs(30),
        )
        .expect("must not error");
        releaser.join().expect("releaser");
        assert!(got.is_some(), "release within the deadline must grant");
    }
}
