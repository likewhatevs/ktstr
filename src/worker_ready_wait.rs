//! Test-side poll helper for the worker ready marker. Separated from
//! [`crate::worker_ready`] because this helper references
//! [`crate::scenario::payload_run::PayloadHandle`]; the bin crate
//! `ktstr-jemalloc-alloc-worker` pulls `worker_ready.rs` in via
//! `#[path]` and must stay dependency-free (see that module's doc for
//! why). This module is library-only.

use crate::scenario::payload_run::PayloadHandle;
use crate::worker_ready::{WORKER_READY_MARKER_PREFIX, worker_ready_marker_path};

/// Poll for the worker's ready marker with a deadline, returning
/// early if the worker exits before writing the marker or after
/// writing but before the caller's subsequent dispatch.
///
/// Event-driven via `inotify` on the marker's parent directory
/// (`IN_CREATE | IN_MOVED_TO`). The wait wakes on the actual file-
/// create edge — kernel-scheduling-tick latency — instead of a
/// 10 ms poll tail. inotify is set up BEFORE the initial existence
/// probe so a marker that lands between probe and watch-add still
/// fires the watch on a subsequent edge; the loop body re-checks
/// existence on every wake so a stale watch edge from an unrelated
/// file in the same directory doesn't false-fire.
///
/// The caller supplies `role` (e.g. `"worker"`, `"churn worker"`) and
/// `exit_code_legend` (a variant-specific decoder for the
/// worker-binary exit codes the caller wants printed in the error
/// message). Worker cleanup on timeout happens via
/// [`PayloadHandle::drop`] when the caller's `Result` error
/// propagates — calling `PayloadHandle::kill(self)` here would take
/// the handle by value, which we can't do behind an `&mut` borrow.
///
/// Consolidates what used to be two near-identical 20-line poll
/// loops in `tests/jemalloc_probe_tests.rs` — a rename of the marker
/// path, a change in poll interval, or a new early-exit shape now
/// edits one site instead of two.
pub fn wait_for_worker_ready(
    worker: &mut PayloadHandle,
    worker_pid: u32,
    timeout: std::time::Duration,
    role: &str,
    exit_code_legend: &str,
) -> anyhow::Result<()> {
    use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
    use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};
    use std::os::unix::io::AsFd;

    let ready_path = worker_ready_marker_path(worker_pid);
    let deadline = std::time::Instant::now() + timeout;
    // Parent directory of the marker. WORKER_READY_MARKER_PREFIX
    // encodes the dir + filename prefix; the marker itself lives
    // under /tmp so the watch attaches there. Pulling the dir from
    // the prefix (rather than hardcoding `/tmp`) keeps the wait
    // correct if the prefix moves.
    let marker_dir = std::path::Path::new(WORKER_READY_MARKER_PREFIX)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("/tmp"));
    // Set up the inotify watch BEFORE the existence probe so a
    // marker that lands between probe and watch is still observable
    // on a subsequent unrelated edge (the loop re-checks existence
    // on every wake). `IN_NONBLOCK` so read_events returns EAGAIN
    // when the queue is empty instead of blocking — we drive
    // wake-vs-timeout via poll(2). Fall back to the legacy 10 ms
    // sleep when inotify_init / add_watch fail (a stripped kernel
    // without CONFIG_INOTIFY_USER is the only known failure mode
    // and worker tests must not be blocked by it).
    let inotify_result =
        Inotify::init(InitFlags::IN_CLOEXEC | InitFlags::IN_NONBLOCK).and_then(|i| {
            i.add_watch(
                marker_dir,
                AddWatchFlags::IN_CREATE | AddWatchFlags::IN_MOVED_TO,
            )?;
            Ok(i)
        });
    while !std::path::Path::new(&ready_path).exists() {
        if let Some((_, metrics)) = worker.try_wait()? {
            anyhow::bail!(
                "{role} pid={worker_pid} exited before creating ready marker \
                 {ready_path} (exit_code={} — see stderr; worker exit codes: \
                 {exit_code_legend})",
                metrics.exit_code,
            );
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            anyhow::bail!(
                "{role} pid={worker_pid} did not create ready marker {ready_path} \
                 within {timeout:?}",
            );
        }
        let remaining_ms = deadline
            .duration_since(now)
            .as_millis()
            .min(u16::MAX as u128) as u16;
        match inotify_result.as_ref() {
            Ok(inotify) => {
                let fd = inotify.as_fd();
                let mut pollfds = [PollFd::new(fd, PollFlags::POLLIN)];
                let _ = poll(&mut pollfds, PollTimeout::from(remaining_ms));
                // Drain pending events so the next poll doesn't
                // re-fire on the same edge. We don't inspect the
                // event payload — the path-exists check at the top
                // of the loop is the source of truth.
                let _ = inotify.read_events();
            }
            Err(_) => {
                // CONFIG_INOTIFY_USER unavailable. Fall back to the
                // legacy 10 ms sleep so the wait still completes —
                // covered by the deadline check above.
                std::thread::sleep(
                    std::time::Duration::from_millis(10).min(deadline.duration_since(now)),
                );
            }
        }
    }
    // Narrow-race close: the worker may have written the marker and
    // then died between the write and the caller's next probe
    // dispatch (unusual — the worker is supposed to park — but a
    // fatal Drop or kernel SIGKILL could still fire). One more
    // try_wait surfaces that case with an actionable error instead
    // of letting the caller burn wall-time on a dead pid.
    if let Some((_, metrics)) = worker.try_wait()? {
        anyhow::bail!(
            "{role} pid={worker_pid} exited after writing ready marker but \
             before the caller's next dispatch (exit_code={} — see stderr)",
            metrics.exit_code,
        );
    }
    Ok(())
}
