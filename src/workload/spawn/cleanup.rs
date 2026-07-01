//! Forked-child cleanup helpers shared across [`super::SpawnGuard`]'s
//! drop path, mid-spawn early-bail, and `stop_and_collect`'s and
//! `WorkloadHandle::sigkill_workers`'s per-child SIGKILL escalation.
//!
//! Each helper centralises a duplicated cleanup pattern from `mod.rs`
//! so a future addition (new signal, new fd-class to close, new errno
//! to swallow) lands in one place instead of drifting across six call
//! sites. Every errno is swallowed silently — the caller contexts are
//! mid-cleanup-on-error paths where there is nothing to assert against
//! beyond not leaking. See each helper's doc for the specific
//! swallow rationale.

use std::os::fd::RawFd;

use nix::sys::signal::Signal;
use nix::unistd::Pid;

/// Silently close every fd in `fds` that is `>= 0`. Negative-fd
/// entries (sentinel values for "this slot was never opened" inside
/// `ForkedChild::report_fd` / `start_fd`) are skipped so callers can
/// mix opened and never-opened fds in the same slice without
/// special-casing each one. `EBADF` from a double-close is swallowed
/// because the cleanup paths share fd state with the success path
/// that may have already closed inherited copies — and from the
/// child side of a fork that has already closed its inherited copy
/// before the parent's cleanup runs.
pub(super) fn close_fds_silently(fds: &[RawFd]) {
    for &fd in fds {
        if fd >= 0 {
            let _ = nix::unistd::close(fd);
        }
    }
}

/// Send `signal` to `pid` AND to the process group led by `pid` via
/// `killpg(2)`. Used in `stop_and_collect`'s per-child SIGKILL
/// escalation: the worker fork tree includes any grandchildren the
/// worker spawned (e.g. a `WorkType::Custom` body that forks helpers), and the
/// framework's `setpgid(0, 0)` boundary (each worker becomes its own
/// process-group leader) means a
/// direct `kill(pid, ...)` reaches only the leader. The paired
/// `killpg` fans out to every process in the leader's process group
/// so the fork tree dies as a unit, not "leader gone, orphans
/// reparented to init."
///
/// Both calls swallow errors:
/// - `ESRCH` from `kill` fires when the leader has already exited
///   (whether running-then-exited or auto-reaped under the
///   SIGCHLD=SIG_IGN convention — either way the pid slot is gone
///   from the kernel's process table). The `killpg` still reaches
///   surviving group members.
/// - `ESRCH` from `killpg` fires on an empty group (last member
///   exited) — nothing to do.
/// - `EPERM` is not expected under normal teardown: ktstr runs as
///   root (CAP_KILL always present per the always-root project
///   contract), so the only ways `kill` refuses are non-root
///   callers or PID 1 / kernel threads, neither of which apply to
///   forked workers. The `let _ =` ignore covers the EPERM case
///   anyway as a defensive belt against a future non-root run
///   mode.
pub(super) fn kill_and_killpg(pid: Pid, signal: Signal) {
    let _ = nix::sys::signal::kill(pid, signal);
    let _ = nix::sys::signal::killpg(pid, signal);
}
