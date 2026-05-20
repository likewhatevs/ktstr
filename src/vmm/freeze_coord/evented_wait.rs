//! Evented wait primitives shared by the scheduler-lifecycle paths.
//!
//! Both [`kernfs_evented_wait`] and [`pidfd_wait_exit`] follow the
//! same shape: subscribe to one or more kernel-evented sources,
//! poll(2) for any of them with a deadline-bounded timeout, and
//! re-check the source-of-truth predicate on every wake. The
//! cadence cap on each poll(2) iteration bounds wake latency
//! against the narrow cases where neither evented source fires
//! (verified per-call-site rationale in each caller — see
//! `poll_scx_attached` and `wait_for_scx_disabled` in
//! `src/vmm/rust_init.rs` and `src/scenario/ops/mod.rs`).
//!
//! Built on `nix::sys::inotify` and `nix::poll` instead of raw
//! `libc::inotify_*` / `libc::poll`, so callers contain zero
//! `unsafe` blocks for the wait loop. The pidfd primitive still
//! goes through `libc::syscall(SYS_pidfd_open)` because nix 0.31
//! does not wrap it; once nix adds a wrapper, the `unsafe` here
//! collapses too.

use std::os::fd::{AsFd, FromRawFd, OwnedFd};
use std::time::{Duration, Instant};

use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};

/// Outcome of [`kernfs_evented_wait`].
#[derive(Debug)]
pub(crate) enum KernfsWaitOutcome<T> {
    /// `check_done` returned `Some(value)` at some wake.
    Done(T),
    /// Deadline elapsed without `check_done` returning `Some`.
    Timeout,
    /// Both the attribute fd open AND the inotify watch failed —
    /// no evented source available, no point looping. Caller
    /// surfaces this as the appropriate kernel-defect error.
    NoEventedSource,
}

/// Wait for `check_done` to return `Some`, polling two evented
/// sources concurrently:
///
/// 1. inotify on `parent_dir` for `event_mask` — fires on
///    directory entry create/delete/etc events. Best-effort: if
///    `inotify_init1` or `inotify_add_watch` fails (kernel
///    CONFIG_INOTIFY_USER off, watch slot exhaustion), this
///    source is skipped.
///
/// 2. POLLPRI on the attribute fd at `attr_path` — kernfs fires
///    POLLPRI when the producer calls `sysfs_notify`. Best-effort:
///    if the attribute doesn't exist yet OR `open(2)` fails, this
///    source is skipped. The fd is opened ONCE here; the caller's
///    `check_done` closure typically re-reads the attribute via
///    its own `File::open` since reusing the same fd across reads
///    requires `lseek(0)` (which kernfs may not honor cleanly).
///
/// If BOTH sources fail to subscribe, returns `NoEventedSource`
/// immediately — no polling fallback per the project rule "no
/// polling fallbacks for evented paths".
///
/// `cadence` caps each `poll(2)` wait so the upper bound on wake
/// latency stays bounded when neither evented source has anything
/// to signal (e.g. kernfs attribute content transitions without
/// a `sysfs_notify`). Per the project rule "guard rail
/// deadlines/timeouts are ok", this is a deadline guard rail on
/// the evented wait, not a degraded polling path.
///
/// `check_done` runs ONCE at entry (fast-path), then after every
/// poll wake. Returning `Some(value)` exits with `Done(value)`.
pub(crate) fn kernfs_evented_wait<T, P, A>(
    parent_dir: P,
    event_mask: AddWatchFlags,
    attr_path: Option<A>,
    cadence: Duration,
    deadline: Instant,
    mut check_done: impl FnMut() -> Option<T>,
) -> KernfsWaitOutcome<T>
where
    P: AsRef<std::path::Path>,
    A: AsRef<std::path::Path>,
{
    // Fast-path: predicate may already be Done before any wake.
    if let Some(v) = check_done() {
        return KernfsWaitOutcome::Done(v);
    }

    // Subscribe to evented sources. Each is best-effort.
    let inotify = Inotify::init(InitFlags::IN_NONBLOCK | InitFlags::IN_CLOEXEC)
        .ok()
        .and_then(|inot| {
            let parent = parent_dir.as_ref();
            match inot.add_watch(parent, event_mask) {
                Ok(_wd) => Some(inot),
                Err(_) => None,
            }
        });

    let attr_fd: Option<OwnedFd> = attr_path.and_then(|path| {
        // nix has no wrapper for `open` that returns OwnedFd
        // ergonomically without exposing a raw flag set; fall
        // back to libc::open + OwnedFd::from_raw_fd. The single
        // unsafe block here is contained and the SAFETY contract
        // is "raw is a fresh fd we just opened".
        let c_path = match std::ffi::CString::new(path.as_ref().as_os_str().as_encoded_bytes()) {
            Ok(c) => c,
            Err(_) => return None,
        };
        // SAFETY: c_path is a NUL-terminated C string; libc::open
        // returns -1 on error or a valid fd we wrap immediately
        // in OwnedFd for RAII cleanup.
        let raw = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC) };
        if raw < 0 {
            None
        } else {
            // SAFETY: raw is a non-negative fd we just opened
            // and have exclusive ownership of.
            Some(unsafe { OwnedFd::from_raw_fd(raw) })
        }
    });

    if inotify.is_none() && attr_fd.is_none() {
        return KernfsWaitOutcome::NoEventedSource;
    }

    // PollFds borrow from the OwnedFds for their lifetime. Build
    // a small Vec of BorrowedFd + flags pairs, then construct
    // PollFds for each poll(2) call (PollFd::new takes a
    // BorrowedFd by ref-equivalent and the slice must outlive
    // the poll call).
    loop {
        let now = Instant::now();
        if now >= deadline {
            return KernfsWaitOutcome::Timeout;
        }
        let remaining = deadline - now;
        let wait_ms = remaining.min(cadence).as_millis().min(i32::MAX as u128) as i32;
        let timeout = PollTimeout::try_from(wait_ms).unwrap_or(PollTimeout::ZERO);

        // Build the pollfd set: max 2 entries. nix's PollFd::new
        // takes a BorrowedFd<'fd> that lives as long as the slice
        // — borrow from OwnedFd / Inotify here, scope to this
        // iteration.
        let mut pfds: Vec<PollFd<'_>> = Vec::with_capacity(2);
        if let Some(ref fd) = attr_fd {
            pfds.push(PollFd::new(fd.as_fd(), PollFlags::POLLPRI));
        }
        if let Some(ref inot) = inotify {
            pfds.push(PollFd::new(inot.as_fd(), PollFlags::POLLIN));
        }

        // Ignore poll return — caller's check_done is the source
        // of truth. Errors (EINTR, etc.) are benign: the loop's
        // next iteration re-checks the deadline and re-polls.
        let _ = poll(&mut pfds, timeout);

        // Drain inotify events so the fd doesn't stay readable
        // across iterations and spin the next poll. We don't
        // care about the event contents — the predicate re-read
        // is the source of truth.
        if let Some(ref inot) = inotify {
            // Ignore EAGAIN (no events) and any read error.
            let _ = inot.read_events();
        }

        if let Some(v) = check_done() {
            return KernfsWaitOutcome::Done(v);
        }
    }
}

/// Wait for `pid` to exit, polling the kernel-evented pidfd. The
/// kernel fires POLLIN on the pidfd when the task enters
/// EXIT_ZOMBIE (`do_notify_pidfd` from `exit_notify` in
/// `kernel/exit.c`). Returns `true` the first time `still_alive`
/// returns `false` (the pidfd POLLIN may fire before our re-check
/// resolves, but the re-check is the source of truth). Returns
/// `false` if the deadline elapses with the pid still alive.
///
/// Panics on non-ESRCH `pidfd_open` failure: pidfd_open shipped
/// unconditionally in Linux 5.3, so a non-ESRCH error
/// (ENOMEM, ENFILE) is a catastrophic environment defect that
/// polling cannot recover from. Same for nix's poll(2) failures
/// beyond EINTR (already handled internally by the syscall
/// retry).
pub(crate) fn pidfd_wait_exit(
    pid: u32,
    deadline: Instant,
    mut still_alive: impl FnMut() -> bool,
) -> bool {
    // Fast-path: predicate may already say dead.
    if !still_alive() {
        return true;
    }

    // pidfd_open(2). nix 0.31 has no wrapper; libc::syscall is
    // the only path. The single unsafe block is contained.
    // SAFETY: SYS_pidfd_open + (pid_t, u32 flags=0) is the
    // documented signature; return value is -1 on error or a
    // valid fd we wrap immediately in OwnedFd for RAII.
    let raw =
        unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::c_int, 0u32) as libc::c_int };
    if raw < 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ESRCH) {
            // pidfd_open observed the pid gone before we attached
            // a waiter — already dead.
            return true;
        }
        // Non-ESRCH failure. Re-probe truth; if /proc says alive,
        // panic loud — env defect, polling fallback would silently
        // lose detection.
        if !still_alive() {
            return true;
        }
        panic!(
            "evented_wait::pidfd_wait_exit: pidfd_open(pid={pid}) returned {err} \
             (errno {:?}); pidfd_open is unconditional from Linux 5.3 so a \
             non-ESRCH failure indicates env breakage — check ulimit -n / \
             memory pressure / cgroup pids.max",
            err.raw_os_error()
        );
    }
    // SAFETY: pidfd_open returned a non-negative fd we own.
    let pidfd = unsafe { OwnedFd::from_raw_fd(raw) };

    loop {
        // Re-check predicate at the top of the loop: the exit edge
        // may have fired between any prior step and now.
        if !still_alive() {
            return true;
        }
        let now = Instant::now();
        if now >= deadline {
            // Final re-probe to catch a race where the pid exited
            // between the last poll-return and the deadline check.
            return !still_alive();
        }
        let remaining = deadline - now;
        let wait_ms = remaining.as_millis().min(i32::MAX as u128) as i32;
        let timeout = PollTimeout::try_from(wait_ms).unwrap_or(PollTimeout::ZERO);

        let mut pfds = [PollFd::new(pidfd.as_fd(), PollFlags::POLLIN)];
        // EINTR / other transient errors fall through to the next
        // iteration; predicate re-check is the source of truth.
        let _ = poll(&mut pfds, timeout);
    }
}
