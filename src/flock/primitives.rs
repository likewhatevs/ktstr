//! Kernel-syscall wrappers for `flock(2)` acquire/release.
//!
//! Five entry points, each gated through
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
//!  - [`try_flock_with_witness`] — the same non-blocking acquire while
//!    retaining the still-open writable fd on contention. Admission
//!    callers may use that witness to order its `IN_CLOSE_WRITE` after
//!    publishing their blocked state, avoiding an unnecessary UNKNOWN
//!    observation and re-probe.
//!  - `probe_flock_existing_read_only` — non-creating, read-only
//!    observation of an existing lockfile. Its fd closes with
//!    `IN_CLOSE_NOWRITE`, so a resource-release watcher can ignore
//!    observation traffic.
//!  - [`block_flock`] — blocking acquire. Parks the calling thread
//!    in the kernel until the lock is available. Used after
//!    [`try_flock`] returns `None` for callers that want to wait
//!    indefinitely; callers with a deadline use
//!    [`super::acquire::acquire_flock_with_timeout`] instead.
//!
//! The creating/writable entry points open with
//! `O_CREAT | O_RDWR | O_CLOEXEC | 0o666` so the resulting fd matches
//! the rest of the crate's lockfile contract:
//!
//!  - `O_CLOEXEC` keeps the lock from leaking across `exec(2)` into
//!    spawned subprocesses (cargo subcommands, build pipeline,
//!    initramfs compressor) where the parent's `OwnedFd::drop`
//!    would not release a child-held flock.
//!  - 0o666 mode matches a peer first-acquire so the file's owner
//!    and permissions don't depend on creation order.

use anyhow::{Context, Result};
use std::marker::PhantomData;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;
use std::rc::Rc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicUsize, Ordering};
use std::thread::JoinHandle;

use super::FlockMode;
use super::fs_filter::reject_remote_fs;

/// Open a lockfile with the crate-wide flock contract: refuses
/// remote filesystems via [`reject_remote_fs`], then opens with
/// `O_CREAT | O_RDWR | O_CLOEXEC | 0o666`. The three module entry
/// points ([`materialize`], [`try_flock_with_witness`] / [`try_flock`],
/// and [`block_flock`]) share this open shape; centralizing it here
/// means a future flag change (or an addition to the remote-fs
/// deny-list) lands in one place instead of drifting across call sites.
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

/// Result of a nonblocking flock attempt that preserves the opened fd in
/// either case.
///
/// A contended fd owns no flock: the kernel rejected the requested operation
/// with `EWOULDBLOCK`. Keeping that writable fd open is nevertheless useful to
/// admission protocols. They may publish the corresponding blocked state
/// first, then drop the witness so its `IN_CLOSE_WRITE` avoids an intervening
/// UNKNOWN state and re-probe. Correctness does not depend on retaining it when
/// the admission model durably records pending UNKNOWN resources.
#[derive(Debug)]
pub(crate) enum TryFlockOutcome {
    /// The fd owns the requested flock until its last open-file-description
    /// reference closes.
    Acquired(OwnedFd),
    /// The fd remains open but owns no flock.
    Contended(OwnedFd),
}

/// Issue the common nonblocking flock operation against an already-open fd.
fn try_flock_fd(path: &Path, fd: OwnedFd, mode: FlockMode) -> Result<TryFlockOutcome> {
    use rustix::fs::{FlockOperation, flock};

    let op = match mode {
        FlockMode::Exclusive => FlockOperation::NonBlockingLockExclusive,
        FlockMode::Shared => FlockOperation::NonBlockingLockShared,
    };
    match flock(&fd, op) {
        Ok(()) => Ok(TryFlockOutcome::Acquired(fd)),
        Err(e) if e == rustix::io::Errno::WOULDBLOCK => Ok(TryFlockOutcome::Contended(fd)),
        Err(e) => anyhow::bail!("flock {}: {e}", path.display()),
    }
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

/// Open a lockfile read-write and attempt a nonblocking flock without dropping
/// the fd on contention.
///
/// This is the evidence-preserving form of [`try_flock`]. Both variants create
/// a missing lockfile with the crate-wide `O_CREAT | O_RDWR | O_CLOEXEC`
/// contract. The only difference is that this function returns
/// [`TryFlockOutcome::Contended`] with the writable fd still open, allowing a
/// caller to order its eventual close after a cross-process state publication
/// when doing so avoids a conservative UNKNOWN re-probe.
pub(crate) fn try_flock_with_witness<P: AsRef<Path>>(
    path: P,
    mode: FlockMode,
) -> Result<TryFlockOutcome> {
    let path = path.as_ref();
    let fd = open_lockfile(path)?;
    try_flock_fd(path, fd, mode)
}

/// Observe an existing lockfile with a read-only, nonblocking flock attempt.
///
/// Returns `Ok(None)` if the path does not exist and never creates it. An
/// existing path returns the same acquired/contended outcome as
/// [`try_flock_with_witness`], but the fd was opened `O_RDONLY | O_CLOEXEC`.
/// Dropping either outcome therefore emits `IN_CLOSE_NOWRITE`, not
/// `IN_CLOSE_WRITE`; coordinator watches interested in real holder releases can
/// omit observation traffic at the kernel filter.
///
/// An `Acquired` result is an observational proof flock. A caller may retain it
/// through publication of the proven state so that state remains true at the
/// publication boundary, then drop it; it should not use this read-only probe
/// as a long-lived resource reservation.
#[cfg(test)]
pub(crate) fn probe_flock_existing_read_only<P: AsRef<Path>>(
    path: P,
    mode: FlockMode,
) -> Result<Option<TryFlockOutcome>> {
    use rustix::fs::{Mode, OFlags, open};

    let path = path.as_ref();
    reject_remote_fs(path)?;
    let fd = match open(path, OFlags::RDONLY | OFlags::CLOEXEC, Mode::empty()) {
        Ok(fd) => fd,
        Err(e) if e == rustix::io::Errno::NOENT => return Ok(None),
        Err(e) => return Err(anyhow::anyhow!("open existing {}: {e}", path.display())),
    };
    try_flock_fd(path, fd, mode).map(Some)
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
    match try_flock_with_witness(path, mode)? {
        TryFlockOutcome::Acquired(fd) => Ok(Some(fd)),
        TryFlockOutcome::Contended(witness) => {
            // Preserve the historical Option API: callers that do not
            // participate in publication ordering intentionally discard the
            // contended writable witness immediately.
            drop(witness);
            Ok(None)
        }
    }
}

/// Nonblockingly convert an already-held flock to `mode`.
///
/// Linux implements a flock conversion as an unlock followed by a new lock,
/// so the conversion is not atomic. A caller whose old and new modes carry
/// admission semantics must provide an independent fence across this call.
/// `false` means another holder won the conversion window; the fd then owns no
/// lock and the caller must discard the whole reservation attempt.
pub(crate) fn try_convert_flock(fd: &OwnedFd, mode: FlockMode) -> Result<bool> {
    use rustix::fs::{FlockOperation, flock};

    let operation = match mode {
        FlockMode::Exclusive => FlockOperation::NonBlockingLockExclusive,
        FlockMode::Shared => FlockOperation::NonBlockingLockShared,
    };
    match flock(fd, operation) {
        Ok(()) => Ok(true),
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => Ok(false),
        Err(error) => Err(std::io::Error::from_raw_os_error(error.raw_os_error()))
            .context("convert held flock mode"),
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
    flock(&fd, op)
        .map_err(|errno| std::io::Error::from_raw_os_error(errno.raw_os_error()))
        .with_context(|| format!("flock (blocking) {}", path.display()))?;
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

/// Cached signal number for async-signal-safe wake delivery.
///
/// On glibc `SIGRTMIN()` is a function call, not a compile-time constant.
/// Resolve it while installing the handler in normal context so
/// [`wake_interruptible_flock_waiter`] only performs atomic operations and
/// async-signal-safe syscalls.
static FLOCK_WAKE_SIGNAL: AtomicI32 = AtomicI32::new(0);

/// Install the no-op handler for [`flock_deadline_signal`] exactly once,
/// process-wide.
///
/// `sa_flags` deliberately omits `SA_RESTART`: both deadline ticks and an
/// explicit interrupt wake must force a blocked `flock(2)` / `poll(2)` back
/// into normal Rust context. Delivery is thread-directed, so unrelated threads
/// never observe the wake signal.
fn install_flock_deadline_handler() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        extern "C" fn noop(_: libc::c_int) {}
        unsafe {
            let signal = flock_deadline_signal();
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = noop as *const () as usize;
            sa.sa_flags = 0; // no SA_RESTART — EINTR is load-bearing.
            libc::sigemptyset(&mut sa.sa_mask);
            let rc = libc::sigaction(signal, &sa, std::ptr::null_mut());
            assert_eq!(
                rc,
                0,
                "install_flock_deadline_handler: sigaction(SIGRTMIN+4) failed: {} — \
                 deadline-bounded flock waits would park forever",
                std::io::Error::last_os_error(),
            );
            FLOCK_WAKE_SIGNAL.store(signal, Ordering::SeqCst);
        }
    });
}

const NO_INTERRUPTIBLE_WAITER: u32 = 0;
const CLOSING_INTERRUPTIBLE_WAITER: u32 = u32::MAX;
const NO_BROKER_EVENTFD: RawFd = -1;
const CLOSING_BROKER_EVENTFD: RawFd = -2;
const INTERRUPT_WAKE_RETRY: std::time::Duration = std::time::Duration::from_millis(2);

/// The current cancellation-aware registry ticket or coordinator.
///
/// The public identity is a generation token, not the Linux TID. A retry
/// queued for an old waiter can therefore never signal a later registration
/// that happens to run on the same thread. TID is published before the live
/// generation and cleared before that generation returns to zero.
static INTERRUPTIBLE_WAITER_ID: AtomicU32 = AtomicU32::new(NO_INTERRUPTIBLE_WAITER);
static INTERRUPTIBLE_WAITER_TID: AtomicI32 = AtomicI32::new(0);
static NEXT_INTERRUPTIBLE_WAITER_ID: AtomicU32 = AtomicU32::new(1);

/// Broker iterations that passed the generation check and may still load/use
/// the registered TID. Waiter teardown changes the generation to CLOSING and
/// drains this count before blocking and draining the private RT signal.
static INTERRUPTIBLE_BROKER_READERS: AtomicUsize = AtomicUsize::new(0);

/// Signal-handler writers that may have loaded either the waiter generation or
/// broker eventfd. Both waiter teardown and broker shutdown hide their
/// respective object and drain this count before allowing reuse/close.
static INTERRUPTIBLE_HANDLER_WRITERS: AtomicUsize = AtomicUsize::new(0);

/// Eventfd handoff from the async handler to the normal-context broker.
static INTERRUPTIBLE_BROKER_EVENTFD: AtomicI32 = AtomicI32::new(NO_BROKER_EVENTFD);
static INTERRUPTIBLE_BROKER_REQUEST: AtomicU32 = AtomicU32::new(NO_INTERRUPTIBLE_WAITER);
static INTERRUPTIBLE_BROKER_STOPPING: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static INTERRUPTIBLE_BROKER_SIGNAL_COUNT: AtomicUsize = AtomicUsize::new(0);

struct InterruptibleFlockBroker {
    eventfd: OwnedFd,
    thread: JoinHandle<()>,
}

static INTERRUPTIBLE_BROKER: Mutex<Option<InterruptibleFlockBroker>> = Mutex::new(None);

fn broker_state() -> std::sync::MutexGuard<'static, Option<InterruptibleFlockBroker>> {
    INTERRUPTIBLE_BROKER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Start the normal-context broker used to wake cancellation-aware flock
/// waiters.
///
/// Lifecycle ownership belongs to cargo-ktstr: start immediately before
/// entering its cleanup-owning phase and stop after all reservation state has
/// dropped. The broker is restartable after a matching stop, which also keeps
/// unit tests isolated.
pub fn start_interruptible_flock_broker() -> Result<()> {
    let mut state = broker_state();
    if state.is_some() {
        anyhow::bail!("interruptible flock broker is already running");
    }
    if INTERRUPTIBLE_WAITER_ID.load(Ordering::SeqCst) != NO_INTERRUPTIBLE_WAITER {
        anyhow::bail!("cannot start interruptible flock broker with a live waiter");
    }

    // SAFETY: eventfd returns a new owned descriptor on success.
    let raw_fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    if raw_fd < 0 {
        anyhow::bail!(
            "create interruptible flock broker eventfd: {}",
            std::io::Error::last_os_error(),
        );
    }
    let eventfd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

    // Prime the exact libc write entry used by the signal handler while in
    // normal context, then drain the probe before making the fd visible.
    let one = 1_u64;
    let wrote = unsafe {
        libc::write(
            raw_fd,
            (&one as *const u64).cast(),
            std::mem::size_of::<u64>(),
        )
    };
    if wrote != std::mem::size_of::<u64>() as isize {
        anyhow::bail!(
            "prime interruptible flock broker eventfd write: {}",
            std::io::Error::last_os_error(),
        );
    }
    let mut drained = 0_u64;
    let read = unsafe {
        libc::read(
            raw_fd,
            (&mut drained as *mut u64).cast(),
            std::mem::size_of::<u64>(),
        )
    };
    if read != std::mem::size_of::<u64>() as isize || drained != one {
        anyhow::bail!(
            "drain interruptible flock broker eventfd prime: {}",
            std::io::Error::last_os_error(),
        );
    }

    INTERRUPTIBLE_BROKER_REQUEST.store(NO_INTERRUPTIBLE_WAITER, Ordering::SeqCst);
    INTERRUPTIBLE_BROKER_STOPPING.store(false, Ordering::SeqCst);
    #[cfg(test)]
    INTERRUPTIBLE_BROKER_SIGNAL_COUNT.store(0, Ordering::SeqCst);
    let thread = std::thread::Builder::new()
        .name("ktstr-flock-wake".into())
        .spawn(move || interruptible_flock_broker_loop(raw_fd))
        .map_err(|error| anyhow::anyhow!("spawn interruptible flock broker: {error}"))?;

    // Publish only after the broker thread exists and owns its read loop.
    INTERRUPTIBLE_BROKER_EVENTFD.store(raw_fd, Ordering::SeqCst);
    *state = Some(InterruptibleFlockBroker { eventfd, thread });
    Ok(())
}

/// Stop and join the interruptible-flock broker.
///
/// Hiding the eventfd and draining handler writers before the owned wake/close
/// ensures an async handler can never write through a recycled fd number.
pub fn stop_interruptible_flock_broker() {
    let mut state = broker_state();
    let Some(broker) = state.take() else {
        return;
    };

    INTERRUPTIBLE_BROKER_STOPPING.store(true, Ordering::SeqCst);
    let raw_fd = broker.eventfd.as_raw_fd();
    let hidden = INTERRUPTIBLE_BROKER_EVENTFD.compare_exchange(
        raw_fd,
        CLOSING_BROKER_EVENTFD,
        Ordering::SeqCst,
        Ordering::SeqCst,
    );
    assert_eq!(
        hidden,
        Ok(raw_fd),
        "interruptible flock broker eventfd changed before stop",
    );
    while INTERRUPTIBLE_HANDLER_WRITERS.load(Ordering::SeqCst) != 0 {
        std::hint::spin_loop();
    }

    // Wake the broker through the still-owned fd after handlers can no longer
    // load it. Failure is harmless only if the broker already observed STOPPING
    // after a prior handler wake.
    let one = 1_u64;
    unsafe {
        libc::write(
            raw_fd,
            (&one as *const u64).cast(),
            std::mem::size_of::<u64>(),
        );
    }
    broker
        .thread
        .join()
        .expect("interruptible flock broker thread panicked");
    drop(broker.eventfd);

    INTERRUPTIBLE_BROKER_REQUEST.store(NO_INTERRUPTIBLE_WAITER, Ordering::SeqCst);
    INTERRUPTIBLE_BROKER_EVENTFD.store(NO_BROKER_EVENTFD, Ordering::SeqCst);
    INTERRUPTIBLE_BROKER_STOPPING.store(false, Ordering::SeqCst);
}

fn interruptible_flock_broker_loop(eventfd: RawFd) {
    loop {
        let mut pollfd = libc::pollfd {
            fd: eventfd,
            events: libc::POLLIN,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut pollfd, 1, -1) };
        if ready < 0 {
            if std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            std::thread::yield_now();
            continue;
        }

        drain_interruptible_broker_eventfd(eventfd);
        if INTERRUPTIBLE_BROKER_STOPPING.load(Ordering::SeqCst) {
            break;
        }

        let waiter_id =
            INTERRUPTIBLE_BROKER_REQUEST.swap(NO_INTERRUPTIBLE_WAITER, Ordering::SeqCst);
        if waiter_id == NO_INTERRUPTIBLE_WAITER || waiter_id == CLOSING_INTERRUPTIBLE_WAITER {
            continue;
        }

        // Re-fire only after cancellation was handed to the broker. Healthy
        // queue contention therefore retains one FIFO flock request, while a
        // signal that landed just before syscall entry cannot strand it.
        loop {
            if INTERRUPTIBLE_BROKER_STOPPING.load(Ordering::SeqCst)
                || !broker_signal_waiter_if_live(waiter_id)
            {
                break;
            }
            std::thread::sleep(INTERRUPT_WAKE_RETRY);
        }
    }
}

fn drain_interruptible_broker_eventfd(eventfd: RawFd) {
    loop {
        let mut value = 0_u64;
        let read = unsafe {
            libc::read(
                eventfd,
                (&mut value as *mut u64).cast(),
                std::mem::size_of::<u64>(),
            )
        };
        if read == std::mem::size_of::<u64>() as isize {
            continue;
        }
        if read < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
            continue;
        }
        break;
    }
}

fn broker_signal_waiter_if_live(waiter_id: u32) -> bool {
    INTERRUPTIBLE_BROKER_READERS.fetch_add(1, Ordering::SeqCst);
    let live = INTERRUPTIBLE_WAITER_ID.load(Ordering::SeqCst) == waiter_id;
    if live {
        let tid = INTERRUPTIBLE_WAITER_TID.load(Ordering::SeqCst);
        let signal = FLOCK_WAKE_SIGNAL.load(Ordering::SeqCst);
        if tid > 0 && signal > 0 {
            unsafe {
                libc::syscall(libc::SYS_tgkill, libc::getpid(), tid, signal);
            }
            #[cfg(test)]
            INTERRUPTIBLE_BROKER_SIGNAL_COUNT.fetch_add(1, Ordering::SeqCst);
        }
    }
    INTERRUPTIBLE_BROKER_READERS.fetch_sub(1, Ordering::SeqCst);
    live
}

/// Current cancellation-aware flock waiter generation, or zero when none is
/// registered.
///
/// This atomic-only query is safe to call from a signal handler.
pub fn interruptible_flock_waiter_id() -> u32 {
    let waiter_id = INTERRUPTIBLE_WAITER_ID.load(Ordering::SeqCst);
    if waiter_id == CLOSING_INTERRUPTIBLE_WAITER {
        NO_INTERRUPTIBLE_WAITER
    } else {
        waiter_id
    }
}

#[cfg(test)]
pub(crate) fn interruptible_flock_broker_signal_count() -> usize {
    INTERRUPTIBLE_BROKER_SIGNAL_COUNT.load(Ordering::SeqCst)
}

/// Hand cancellation of an exact waiter generation to the broker.
///
/// The signal-handler path performs only lock-free atomics and one nonblocking
/// eventfd write. The caller must publish its cancellation flag first.
pub fn wake_interruptible_flock_waiter(waiter_id: u32) {
    if waiter_id == NO_INTERRUPTIBLE_WAITER || waiter_id == CLOSING_INTERRUPTIBLE_WAITER {
        return;
    }

    INTERRUPTIBLE_HANDLER_WRITERS.fetch_add(1, Ordering::SeqCst);
    if INTERRUPTIBLE_WAITER_ID.load(Ordering::SeqCst) == waiter_id {
        let eventfd = INTERRUPTIBLE_BROKER_EVENTFD.load(Ordering::SeqCst);
        if eventfd >= 0 {
            INTERRUPTIBLE_BROKER_REQUEST.store(waiter_id, Ordering::SeqCst);
            let one = 1_u64;
            unsafe {
                libc::write(
                    eventfd,
                    (&one as *const u64).cast(),
                    std::mem::size_of::<u64>(),
                );
            }
        }
    }
    INTERRUPTIBLE_HANDLER_WRITERS.fetch_sub(1, Ordering::SeqCst);
}

fn next_interruptible_waiter_id() -> u32 {
    loop {
        let waiter_id = NEXT_INTERRUPTIBLE_WAITER_ID.fetch_add(1, Ordering::SeqCst);
        if waiter_id != NO_INTERRUPTIBLE_WAITER && waiter_id != CLOSING_INTERRUPTIBLE_WAITER {
            return waiter_id;
        }
    }
}

/// RAII registration for one cancellation-aware registry ticket/coordinator.
///
/// The broker targets this registration's Linux TID only while its generation
/// remains live. This value is deliberately `!Send`: its saved signal-mask
/// state must be restored on the registering thread.
pub(crate) struct InterruptibleFlockWaiter {
    waiter_id: u32,
    wake_was_blocked: bool,
    _not_send: PhantomData<Rc<()>>,
}

impl InterruptibleFlockWaiter {
    pub(crate) fn register() -> Result<Self> {
        if INTERRUPTIBLE_BROKER_EVENTFD.load(Ordering::SeqCst) < 0 {
            anyhow::bail!("interruptible flock broker is not running");
        }
        install_flock_deadline_handler();
        let signal = FLOCK_WAKE_SIGNAL.load(Ordering::SeqCst);
        let tid = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t };

        if INTERRUPTIBLE_WAITER_ID
            .compare_exchange(
                NO_INTERRUPTIBLE_WAITER,
                CLOSING_INTERRUPTIBLE_WAITER,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_err()
        {
            anyhow::bail!("another interruptible flock waiter is already registered");
        }

        let mut wake_set: libc::sigset_t = unsafe { std::mem::zeroed() };
        let mut previous_mask: libc::sigset_t = unsafe { std::mem::zeroed() };
        let mask_rc = unsafe {
            libc::sigemptyset(&mut wake_set);
            libc::sigaddset(&mut wake_set, signal);
            libc::pthread_sigmask(libc::SIG_UNBLOCK, &wake_set, &mut previous_mask)
        };
        if mask_rc != 0 {
            INTERRUPTIBLE_WAITER_ID.store(NO_INTERRUPTIBLE_WAITER, Ordering::SeqCst);
            return Err(anyhow::anyhow!(
                "unblock interruptible flock wake signal: {}",
                std::io::Error::from_raw_os_error(mask_rc),
            ));
        }
        let wake_was_blocked = unsafe { libc::sigismember(&previous_mask, signal) == 1 };
        let waiter_id = next_interruptible_waiter_id();

        // TID becomes visible before the generation that authorizes readers to
        // use it.
        INTERRUPTIBLE_WAITER_TID.store(tid, Ordering::SeqCst);
        INTERRUPTIBLE_WAITER_ID.store(waiter_id, Ordering::SeqCst);

        Ok(Self {
            waiter_id,
            wake_was_blocked,
            _not_send: PhantomData,
        })
    }
}

impl Drop for InterruptibleFlockWaiter {
    fn drop(&mut self) {
        let transitioned = INTERRUPTIBLE_WAITER_ID.compare_exchange(
            self.waiter_id,
            CLOSING_INTERRUPTIBLE_WAITER,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
        assert_eq!(
            transitioned,
            Ok(self.waiter_id),
            "interruptible flock waiter registration changed before drop",
        );

        while INTERRUPTIBLE_HANDLER_WRITERS.load(Ordering::SeqCst) != 0
            || INTERRUPTIBLE_BROKER_READERS.load(Ordering::SeqCst) != 0
        {
            std::hint::spin_loop();
        }

        // No new broker reader can pass the generation check after CLOSING.
        // Block and drain a signal already sent by an old reader before
        // allowing either the TID or generation slot to be reused.
        unsafe {
            let signal = FLOCK_WAKE_SIGNAL.load(Ordering::SeqCst);
            let mut wake_set: libc::sigset_t = std::mem::zeroed();
            libc::sigemptyset(&mut wake_set);
            libc::sigaddset(&mut wake_set, signal);
            libc::pthread_sigmask(libc::SIG_BLOCK, &wake_set, std::ptr::null_mut());
            let zero = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            loop {
                let drained = libc::sigtimedwait(&wake_set, std::ptr::null_mut(), &zero);
                if drained == signal {
                    continue;
                }
                assert!(
                    drained < 0,
                    "sigtimedwait returned unexpected signal {drained} while \
                     draining private wake signal {signal}",
                );
                let error = std::io::Error::last_os_error();
                match error.raw_os_error() {
                    // No private RT signal remains pending on this thread.
                    Some(libc::EAGAIN) => break,
                    // An unrelated unblocked signal ran while draining. Retry
                    // until EAGAIN so an old wake cannot escape into a later
                    // same-thread registration.
                    Some(libc::EINTR) => continue,
                    _ => panic!("drain interruptible flock wake signal: {error}"),
                }
            }

            INTERRUPTIBLE_WAITER_TID.store(0, Ordering::SeqCst);
            INTERRUPTIBLE_WAITER_ID.store(NO_INTERRUPTIBLE_WAITER, Ordering::SeqCst);
            if !self.wake_was_blocked {
                libc::pthread_sigmask(libc::SIG_UNBLOCK, &wake_set, std::ptr::null_mut());
            }
        }
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
///    re-evaluates live state — e.g. a coordinator re-scanning for a
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

    fn fd_flags(fd: &OwnedFd) -> libc::c_int {
        use std::os::fd::AsRawFd;

        // SAFETY: `fd` is live for the duration of the accessor and F_GETFD
        // neither mutates nor assumes ownership of it.
        let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFD) };
        assert!(
            flags >= 0,
            "fcntl F_GETFD must succeed on a live fd; got errno={}",
            std::io::Error::last_os_error(),
        );
        flags
    }

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

    #[test]
    fn flock_mode_conversion_retains_the_same_reservation_fd() {
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("convert.lock");
        let exclusive = try_flock(&path, FlockMode::Exclusive)
            .expect("acquire initial EX")
            .expect("fresh EX lock");
        assert!(
            try_flock(&path, FlockMode::Shared)
                .expect("probe SH against EX")
                .is_none(),
            "EX must initially fence a shared peer",
        );

        assert!(
            try_convert_flock(&exclusive, FlockMode::Shared).expect("convert EX to SH"),
            "uncontended conversion must retain the reservation",
        );
        let shared_peer = try_flock(&path, FlockMode::Shared)
            .expect("probe SH after conversion")
            .expect("converted lock must coexist with SH");
        assert!(
            try_flock(&path, FlockMode::Exclusive)
                .expect("probe EX against converted SH")
                .is_none(),
            "converted lock must continue to fence EX",
        );
        drop(shared_peer);
        drop(exclusive);
    }

    /// The evidence-preserving acquire keeps a real O_RDWR fd alive after
    /// EWOULDBLOCK, with CLOEXEC set, but that witness owns no flock. Releasing
    /// the real holder while the witness stays open must therefore let a third
    /// fd acquire immediately. Dropping the witness cannot release the third
    /// fd's independent flock.
    #[test]
    fn contended_witness_is_writable_cloexec_and_owns_no_flock() {
        use std::os::fd::AsRawFd;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("witness.lock");
        let peer = try_flock(&path, FlockMode::Exclusive)
            .expect("open peer")
            .expect("peer EX on fresh file");
        let witness =
            match try_flock_with_witness(&path, FlockMode::Exclusive).expect("contended probe") {
                TryFlockOutcome::Contended(fd) => fd,
                TryFlockOutcome::Acquired(_) => panic!("peer EX must force a contended witness"),
            };

        assert_eq!(
            fd_flags(&witness) & libc::FD_CLOEXEC,
            libc::FD_CLOEXEC,
            "a retained witness must not leak through exec",
        );
        // SAFETY: witness is a live O_RDWR fd. Writing one byte demonstrates
        // the access mode directly without changing fd ownership.
        let byte = b"x";
        let written = unsafe {
            libc::write(
                witness.as_raw_fd(),
                byte.as_ptr().cast::<libc::c_void>(),
                byte.len(),
            )
        };
        assert_eq!(
            written,
            byte.len() as libc::ssize_t,
            "the contended witness must remain open and writable: {}",
            std::io::Error::last_os_error(),
        );

        drop(peer);
        let successor = try_flock(&path, FlockMode::Exclusive)
            .expect("successor probe")
            .expect("an open contended witness owns no flock");

        drop(witness);
        assert!(
            try_flock(&path, FlockMode::Exclusive)
                .expect("probe against successor")
                .is_none(),
            "dropping a non-owning witness must not release the successor's flock",
        );
        drop(successor);
    }

    /// A read-only observation never creates a missing lockfile. On an
    /// existing contended inode it returns a CLOEXEC O_RDONLY witness, which is
    /// the flag-level guarantee that its close is IN_CLOSE_NOWRITE and cannot
    /// masquerade as a resource-holder release.
    #[test]
    fn read_only_existing_probe_does_not_create_and_uses_read_only_cloexec_fd() {
        use std::os::fd::AsRawFd;
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("observe.lock");
        assert!(
            probe_flock_existing_read_only(&path, FlockMode::Exclusive)
                .expect("missing observation")
                .is_none(),
            "a missing existing-only probe must report no inode",
        );
        assert!(
            !path.exists(),
            "a read-only existing-file probe must never create the lockfile",
        );

        let peer = try_flock(&path, FlockMode::Exclusive)
            .expect("open peer")
            .expect("peer EX on fresh file");
        let witness = match probe_flock_existing_read_only(&path, FlockMode::Exclusive)
            .expect("existing observation")
            .expect("peer materialized the inode")
        {
            TryFlockOutcome::Contended(fd) => fd,
            TryFlockOutcome::Acquired(_) => panic!("peer EX must force read-only contention"),
        };
        assert_eq!(
            fd_flags(&witness) & libc::FD_CLOEXEC,
            libc::FD_CLOEXEC,
            "read-only observation fd must not leak through exec",
        );
        // SAFETY: F_GETFL is a read-only query on a live fd.
        let status = unsafe { libc::fcntl(witness.as_raw_fd(), libc::F_GETFL) };
        assert!(
            status >= 0,
            "F_GETFL failed: {}",
            std::io::Error::last_os_error()
        );
        assert_eq!(
            status & libc::O_ACCMODE,
            libc::O_RDONLY,
            "observation fd must be O_RDONLY so close is IN_CLOSE_NOWRITE",
        );

        drop(peer);
        let successor = try_flock(&path, FlockMode::Exclusive)
            .expect("successor probe")
            .expect("read-only contended witness owns no flock");
        drop(witness);
        drop(successor);
    }

    /// Linux permits an exclusive flock on an O_RDONLY descriptor. The
    /// admission observer relies on that property to prove a free existing
    /// resource without opening it writable (and therefore without emitting
    /// IN_CLOSE_WRITE when the proof fd is dropped).
    #[test]
    fn read_only_existing_probe_acquires_exclusive_on_free_local_inode() {
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("read-only-exclusive.lock");
        materialize(&path).expect("materialize existing local inode");

        let proof = probe_flock_existing_read_only(&path, FlockMode::Exclusive)
            .expect("read-only exclusive probe")
            .expect("materialized inode must exist");
        match proof {
            TryFlockOutcome::Acquired(fd) => drop(fd),
            TryFlockOutcome::Contended(_) => {
                panic!("a free local inode must grant EX through an O_RDONLY fd")
            }
        }
    }

    /// Pin the flock compatibility matrix used when admission falls back to
    /// read-only observation of LLC availability:
    ///
    /// - a live SH holder admits another SH proof but rejects EX;
    /// - a live EX holder rejects both SH and EX.
    ///
    /// Each probe uses its own open-file description, so the result exercises
    /// kernel compatibility rather than recursive locking on one fd.
    #[test]
    fn read_only_existing_probe_matches_shared_exclusive_llc_matrix() {
        use tempfile::TempDir;

        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("read-only-matrix.lock");

        let shared_peer = try_flock(&path, FlockMode::Shared)
            .expect("open shared peer")
            .expect("first SH must acquire");
        let shared_proof = probe_flock_existing_read_only(&path, FlockMode::Shared)
            .expect("read-only SH probe")
            .expect("shared peer materialized the inode");
        let shared_proof = match shared_proof {
            TryFlockOutcome::Acquired(fd) => fd,
            TryFlockOutcome::Contended(_) => {
                panic!("SH must remain compatible with a peer SH holder")
            }
        };
        match probe_flock_existing_read_only(&path, FlockMode::Exclusive)
            .expect("read-only EX probe against SH")
            .expect("inode exists")
        {
            TryFlockOutcome::Contended(fd) => drop(fd),
            TryFlockOutcome::Acquired(_) => {
                panic!("EX must conflict with live SH holders")
            }
        }
        drop(shared_proof);
        drop(shared_peer);

        let exclusive_peer = try_flock(&path, FlockMode::Exclusive)
            .expect("open exclusive peer")
            .expect("EX must acquire after every SH proof drops");
        for mode in [FlockMode::Shared, FlockMode::Exclusive] {
            match probe_flock_existing_read_only(&path, mode)
                .expect("read-only probe against EX")
                .expect("inode exists")
            {
                TryFlockOutcome::Contended(fd) => drop(fd),
                TryFlockOutcome::Acquired(_) => {
                    panic!("{mode:?} must conflict with a live EX holder")
                }
            }
        }
        drop(exclusive_peer);
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
