//! Evented wait primitives shared by the scheduler-lifecycle paths.
//!
//! Both [`kernfs_evented_wait`] and [`pidfd_wait_exit`] follow the
//! same shape: subscribe to one or more kernel-evented sources,
//! poll(2) for any of them with a deadline-bounded timeout, and
//! inspect the returned revents before deciding the outcome.
//! Kernfs wakes re-check their source-of-truth predicate; pidfd
//! POLLIN/POLLHUP is itself authoritative for the referenced task.
//! The cadence cap on each kernfs poll(2) iteration bounds wake
//! latency against the narrow cases where neither evented source
//! fires (verified per-call-site rationale in each caller — see
//! `poll_scx_attached` in `src/vmm/rust_init/scheduler.rs` and
//! `wait_for_scx_disabled` in `src/scenario/ops/dispatch.rs`).
//!
//! Built on `nix::sys::inotify` and `nix::poll` instead of raw
//! `libc::inotify_*` / `libc::poll`, so callers contain zero
//! `unsafe` blocks for the wait loop. The pidfd primitive still
//! goes through `libc::syscall(SYS_pidfd_open)` because nix 0.31
//! does not wrap it; once nix adds a wrapper, the `unsafe` here
//! collapses too.

use std::os::fd::{AsFd, FromRawFd, OwnedFd};
use std::time::{Duration, Instant};

use nix::errno::Errno;
use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};

/// Outcome of [`kernfs_evented_wait`].
#[derive(Debug)]
pub(crate) enum KernfsWaitOutcome<T> {
    /// `check_done` returned `Some(value)` at some wake.
    Done(T),
    /// Deadline elapsed without `check_done` returning `Some`.
    Timeout,
    /// Both the attribute fd and the inotify watch are unavailable
    /// after one final source-of-truth predicate sample. Caller
    /// surfaces this as the appropriate kernel-defect error.
    NoEventedSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KernfsAttrPollDisposition {
    Keep,
    Drop,
}

fn revents_or_panic(pfd: &PollFd<'_>, source: &str) -> PollFlags {
    pfd.revents()
        .unwrap_or_else(|| panic!("evented_wait: {source} returned unknown poll revents"))
}

/// kernfs attributes report `sysfs_notify()` as POLLERR|POLLPRI,
/// so POLLERR is a notification rather than an observer failure.
/// POLLHUP means the attribute vanished and this best-effort source
/// can no longer contribute; POLLNVAL means our owned fd became
/// invalid and is an internal invariant violation.
fn kernfs_attr_poll_disposition(revents: PollFlags) -> KernfsAttrPollDisposition {
    if revents.contains(PollFlags::POLLNVAL) {
        panic!("evented_wait: kernfs attribute fd returned POLLNVAL");
    }
    if revents.contains(PollFlags::POLLHUP) {
        KernfsAttrPollDisposition::Drop
    } else {
        // POLLPRI and POLLERR both cause the source-of-truth
        // predicate to be sampled below. Empty revents are also
        // valid when the cadence timeout expires.
        KernfsAttrPollDisposition::Keep
    }
}

fn validate_inotify_revents(revents: PollFlags) {
    let rejected = revents & (PollFlags::POLLERR | PollFlags::POLLNVAL | PollFlags::POLLHUP);
    if !rejected.is_empty() {
        panic!("evented_wait: inotify fd returned invalid revents {rejected:?}");
    }
}

fn pidfd_revents_mean_exit(pid: u32, revents: PollFlags) -> bool {
    let rejected = revents & (PollFlags::POLLERR | PollFlags::POLLNVAL);
    if !rejected.is_empty() {
        panic!("evented_wait: pidfd for pid {pid} returned invalid revents {rejected:?}");
    }

    revents.intersects(PollFlags::POLLIN | PollFlags::POLLHUP)
}

/// Resolve a deadline edge without letting the guard rail hide a
/// simultaneously-satisfied source-of-truth predicate.
///
/// The caller can be descheduled between its preceding predicate
/// check and its deadline check. Once it resumes at or after the
/// deadline, it must sample the predicate one final time before
/// reporting `Timeout`.
fn outcome_at_deadline<T>(
    now: Instant,
    deadline: Instant,
    check_done: &mut impl FnMut() -> Option<T>,
) -> Option<KernfsWaitOutcome<T>> {
    if now < deadline {
        return None;
    }

    Some(match check_done() {
        Some(value) => KernfsWaitOutcome::Done(value),
        None => KernfsWaitOutcome::Timeout,
    })
}

/// Convert a positive duration to poll's millisecond timeout without
/// truncating a sub-millisecond remainder to an immediate return.
fn poll_timeout_ceil(duration: Duration) -> PollTimeout {
    let has_submillisecond_remainder = !duration.subsec_nanos().is_multiple_of(1_000_000);
    let millis = duration
        .as_millis()
        .saturating_add(u128::from(has_submillisecond_remainder))
        .min(i32::MAX as u128);
    PollTimeout::try_from(millis).expect("millisecond timeout was clamped to i32::MAX")
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
/// If BOTH sources fail to subscribe, samples `check_done` once
/// more to close the subscription race and then returns
/// `NoEventedSource` — no polling fallback per the project rule
/// "no polling fallbacks for evented paths".
///
/// `cadence` caps each `poll(2)` wait so the upper bound on wake
/// latency stays bounded when neither evented source has anything
/// to signal (e.g. kernfs attribute content transitions without
/// a `sysfs_notify`). Per the project rule "guard rail
/// deadlines/timeouts are ok", this is a deadline guard rail on
/// the evented wait, not a degraded polling path.
///
/// `check_done` runs once at entry (fast-path), after every poll
/// wake, and once more before an elapsed deadline is reported.
/// Returning `Some(value)` exits with `Done(value)`.
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

    let mut attr_fd: Option<OwnedFd> = attr_path.and_then(|path| {
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
        return match check_done() {
            Some(value) => KernfsWaitOutcome::Done(value),
            None => KernfsWaitOutcome::NoEventedSource,
        };
    }

    // PollFds borrow from the OwnedFds for their lifetime. Build
    // a small Vec of BorrowedFd + flags pairs, then construct
    // PollFds for each poll(2) call (PollFd::new takes a
    // BorrowedFd by ref-equivalent and the slice must outlive
    // the poll call).
    loop {
        let now = Instant::now();
        if let Some(outcome) = outcome_at_deadline(now, deadline, &mut check_done) {
            return outcome;
        }
        let remaining = deadline - now;
        let timeout = poll_timeout_ceil(remaining.min(cadence));

        // Build the pollfd set: max 2 entries. nix's PollFd::new
        // takes a BorrowedFd<'fd> that lives as long as the slice
        // — borrow from OwnedFd / Inotify here, scope to this
        // iteration.
        let mut pfds: Vec<PollFd<'_>> = Vec::with_capacity(2);
        let attr_index = attr_fd.as_ref().map(|fd| {
            let index = pfds.len();
            pfds.push(PollFd::new(fd.as_fd(), PollFlags::POLLPRI));
            index
        });
        let inotify_index = inotify.as_ref().map(|inot| {
            let index = pfds.len();
            pfds.push(PollFd::new(inot.as_fd(), PollFlags::POLLIN));
            index
        });

        // EINTR is a normal wake: the predicate remains the source
        // of truth and the next iteration recomputes the remaining
        // deadline. Other errors mean the subscribed event source
        // cannot be trusted, so fail loudly rather than silently
        // spinning until a misleading timeout.
        match poll(&mut pfds, timeout) {
            Ok(_) => {}
            Err(Errno::EINTR) => {
                if let Some(value) = check_done() {
                    return KernfsWaitOutcome::Done(value);
                }
                continue;
            }
            Err(err) => {
                panic!("evented_wait::kernfs_evented_wait: poll failed: {err}");
            }
        }

        let attr_disposition = attr_index
            .map(|index| {
                kernfs_attr_poll_disposition(revents_or_panic(&pfds[index], "kernfs attribute fd"))
            })
            .unwrap_or(KernfsAttrPollDisposition::Drop);
        let inotify_readable = inotify_index
            .map(|index| {
                let revents = revents_or_panic(&pfds[index], "inotify fd");
                validate_inotify_revents(revents);
                revents.contains(PollFlags::POLLIN)
            })
            .unwrap_or(false);

        // PollFds borrow the owned descriptors. End those borrows
        // before dropping a hung-up best-effort attribute source.
        drop(pfds);
        if attr_disposition == KernfsAttrPollDisposition::Drop {
            attr_fd = None;
        }

        // Drain inotify events so the fd doesn't stay readable
        // across iterations and spin the next poll. We don't
        // care about the event contents — the predicate re-read
        // is the source of truth.
        if inotify_readable {
            let inot = inotify
                .as_ref()
                .expect("readable inotify index requires a live inotify fd");
            match inot.read_events() {
                Ok(_) | Err(Errno::EAGAIN) => {}
                Err(err) => {
                    panic!("evented_wait::kernfs_evented_wait: read inotify events: {err}");
                }
            }
        }

        if let Some(v) = check_done() {
            return KernfsWaitOutcome::Done(v);
        }
        if inotify.is_none() && attr_fd.is_none() {
            return KernfsWaitOutcome::NoEventedSource;
        }
    }
}

/// Wait for `pid` to exit, polling the kernel-evented pidfd. The
/// kernel fires POLLIN on the pidfd when the task enters
/// EXIT_ZOMBIE (`do_notify_pidfd` from `exit_notify` in
/// `kernel/exit.c`). POLLIN or POLLHUP is authoritative for the
/// process represented by the pidfd; it returns `true` directly
/// instead of re-reading a numeric PID that may already have been
/// reused. `still_alive` closes the races before `pidfd_open` and
/// at the deadline. Returns `false` if neither source observes exit
/// by the deadline.
///
/// Panics on non-ESRCH `pidfd_open` failure: pidfd_open shipped
/// unconditionally in Linux 5.3, so a non-ESRCH error
/// (ENOMEM, ENFILE) is a catastrophic environment defect that
/// polling cannot recover from. Same for nix's poll(2) failures
/// beyond EINTR; EINTR is treated as a normal wake and retries
/// after the source-of-truth predicate is sampled.
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
        // The fallback predicate closes pre-poll races. A ready
        // pidfd below remains authoritative even if this numeric
        // PID is subsequently reused.
        if !still_alive() {
            return true;
        }
        let now = Instant::now();
        if now >= deadline {
            // Sample the pidfd once at the boundary before the
            // fallback predicate. This catches an exit whose pid
            // was already reused before /proc could be re-read.
            let mut pfds = [PollFd::new(pidfd.as_fd(), PollFlags::POLLIN)];
            match poll(&mut pfds, PollTimeout::ZERO) {
                Ok(_) => {
                    let revents = revents_or_panic(&pfds[0], "pidfd");
                    if pidfd_revents_mean_exit(pid, revents) {
                        return true;
                    }
                }
                Err(Errno::EINTR) => {}
                Err(err) => {
                    panic!("evented_wait::pidfd_wait_exit: poll(pid={pid}) failed: {err}");
                }
            }
            return !still_alive();
        }
        let remaining = deadline - now;
        let timeout = poll_timeout_ceil(remaining);

        let mut pfds = [PollFd::new(pidfd.as_fd(), PollFlags::POLLIN)];
        match poll(&mut pfds, timeout) {
            Ok(_) => {
                let revents = revents_or_panic(&pfds[0], "pidfd");
                if pidfd_revents_mean_exit(pid, revents) {
                    return true;
                }
            }
            Err(Errno::EINTR) => {}
            Err(err) => {
                panic!("evented_wait::pidfd_wait_exit: poll(pid={pid}) failed: {err}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_subscriptions_recheck_predicate_before_no_source() {
        let tmp = tempfile::TempDir::new().expect("create temporary directory");
        let missing = tmp.path().join("missing");
        let mut checks = 0;

        let outcome = kernfs_evented_wait(
            &missing,
            AddWatchFlags::IN_CREATE,
            Some(&missing),
            Duration::from_secs(1),
            Instant::now() + Duration::from_secs(1),
            || {
                checks += 1;
                (checks == 2).then_some(29)
            },
        );

        assert!(matches!(outcome, KernfsWaitOutcome::Done(29)));
        assert_eq!(checks, 2);
    }

    #[test]
    fn real_pipe_hup_drops_kernfs_attribute_source() {
        let (read_fd, write_fd) = nix::unistd::pipe().expect("create pipe");
        drop(write_fd);

        let mut pfds = [PollFd::new(read_fd.as_fd(), PollFlags::POLLIN)];
        assert_eq!(
            poll(&mut pfds, PollTimeout::from(100u16)).expect("poll pipe"),
            1
        );
        let revents = revents_or_panic(&pfds[0], "test pipe");
        assert!(revents.contains(PollFlags::POLLHUP));
        assert_eq!(
            kernfs_attr_poll_disposition(revents),
            KernfsAttrPollDisposition::Drop
        );
    }

    #[test]
    fn kernfs_pollerr_is_a_notification_not_an_error() {
        assert_eq!(
            kernfs_attr_poll_disposition(PollFlags::POLLERR | PollFlags::POLLPRI),
            KernfsAttrPollDisposition::Keep
        );
    }

    #[test]
    fn inotify_rejects_terminal_poll_states() {
        for revents in [PollFlags::POLLERR, PollFlags::POLLNVAL, PollFlags::POLLHUP] {
            assert!(
                std::panic::catch_unwind(|| validate_inotify_revents(revents)).is_err(),
                "{revents:?} must be rejected"
            );
        }
    }

    #[test]
    fn pidfd_readiness_is_authoritative_without_proc_recheck() {
        let mut child = std::process::Command::new("/bin/true")
            .spawn()
            .expect("spawn short-lived child");
        let pid = child.id();
        let mut liveness_checks = 0;

        let exited = pidfd_wait_exit(pid, Instant::now() + Duration::from_secs(5), || {
            liveness_checks += 1;
            // Model a reused numeric PID: a /proc lookup keeps
            // reporting alive even after the original task exits.
            true
        });

        assert!(exited);
        assert!(
            liveness_checks >= 1,
            "test must exercise the misleading fallback predicate"
        );
        child.wait().expect("reap short-lived child");
    }

    #[test]
    fn pidfd_hup_is_exit_but_error_states_are_rejected() {
        assert!(pidfd_revents_mean_exit(7, PollFlags::POLLHUP));
        for revents in [PollFlags::POLLERR, PollFlags::POLLNVAL] {
            assert!(
                std::panic::catch_unwind(|| pidfd_revents_mean_exit(7, revents)).is_err(),
                "{revents:?} must be rejected"
            );
        }
    }

    #[test]
    fn deadline_edge_gives_done_precedence_over_timeout() {
        let deadline = Instant::now();
        let mut checks = 0;

        let outcome = outcome_at_deadline(deadline, deadline, &mut || {
            checks += 1;
            Some(17)
        });

        assert!(matches!(outcome, Some(KernfsWaitOutcome::Done(17))));
        assert_eq!(checks, 1);
    }

    #[test]
    fn elapsed_deadline_rechecks_before_timeout() {
        let deadline = Instant::now();
        let now = deadline + Duration::from_secs(1);
        let mut checks = 0;

        let outcome = outcome_at_deadline(now, deadline, &mut || {
            checks += 1;
            None::<()>
        });

        assert!(matches!(outcome, Some(KernfsWaitOutcome::Timeout)));
        assert_eq!(checks, 1);
    }

    #[test]
    fn live_deadline_does_not_sample_predicate() {
        let now = Instant::now();
        let deadline = now + Duration::from_secs(1);
        let mut checks = 0;

        let outcome = outcome_at_deadline(now, deadline, &mut || {
            checks += 1;
            Some(())
        });

        assert!(outcome.is_none());
        assert_eq!(checks, 0);
    }

    #[test]
    fn poll_timeout_rounds_positive_submillisecond_waits_up() {
        assert_eq!(
            poll_timeout_ceil(Duration::from_nanos(1)).as_millis(),
            Some(1)
        );
        assert_eq!(
            poll_timeout_ceil(Duration::from_micros(999)).as_millis(),
            Some(1)
        );
        assert_eq!(
            poll_timeout_ceil(Duration::from_micros(1_001)).as_millis(),
            Some(2)
        );
    }

    #[test]
    fn poll_timeout_preserves_zero_and_exact_milliseconds() {
        assert_eq!(poll_timeout_ceil(Duration::ZERO), PollTimeout::ZERO);
        assert_eq!(
            poll_timeout_ceil(Duration::from_millis(37)).as_millis(),
            Some(37)
        );
    }
}
