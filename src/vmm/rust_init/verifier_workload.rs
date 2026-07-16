//! Guest-side verifier workload: a scheduler-agnostic "did the scheduler
//! dispatch?" probe run at [`super::init`]'s Phase 5 when the guest was
//! booted with `--ktstr-verifier-workload` (the `cargo ktstr verifier`
//! sweep path — a VM with no `#[ktstr_test]` body).
//!
//! After the scheduler has attached (Phase 3) and Phase 5 has emitted the
//! `PayloadStarting` frame, this spawns a SpinWait workload — as
//! SCHED_EXT tasks, so the BPF scheduler dispatches them under any switch
//! mode (full or `SCX_OPS_SWITCH_PARTIAL`) — sized to the guest's online
//! CPU count, waits (by BLOCKING on a shared first-iteration eventfd the
//! workers signal, not a blind fixed sleep or a polled counter) until
//! every worker has advanced at least one
//! iteration or a bounded deadline elapses, then stops the workload. The
//! workers are configured to park (`park_after_iterations(Some(1))`)
//! after their first counted iteration rather than spinning until stop:
//! one dispatch under SCHED_EXT is the whole proof, and a per-CPU spin
//! across the concurrent CI cells would dominate the host load and
//! starve other cells' bring-up. A parked worker still publishes its
//! final count and responds to stop, so the wait and the dispatch verdict
//! are unchanged. It
//! emits a [`LifecyclePhase::WorkloadDispatched`] frame
//! only for a worker that BOTH advanced non-zero `iterations` AND had its
//! SCHED_EXT set succeed (`sched_policy_error` is None) — so a fair-class
//! fallback cannot false-confirm — proof the scheduler actually
//! dispatched a task onto a CPU. The deadline scales with the worker
//! count ([`dispatch_deadline`]): wall clock is not guest CPU time on
//! a wide, host-time-sliced topology, so a flat window gives a
//! 64+-CPU guest under CI concurrency almost no per-vCPU compute
//! before the probe gives up on a working scheduler. The host
//! verdict ([`crate::verifier::collect_verifier_output`]) PASSes a cell
//! only when BOTH `PayloadStarting` (attached) AND `WorkloadDispatched`
//! (dispatched) frames arrive.
//!
//! On any failure — workload spawn error, or zero progress within the
//! deadline — NO frame is emitted and the function returns quietly. The
//! host reads the absence, given `PayloadStarting`, as "attached but did
//! not dispatch": a distinct, worse failure than a failed attach. It
//! never panics; a guest panic reboots via `panic=-1` (an i8042 reset →
//! `ExitAction::Shutdown`), which would strand the run with no clean
//! verdict signal.
//!
//! Safe from guest init because the fork workers' orphan self-exit guard
//! (`getppid() == 1` under pid-1 driver) is suppressed by the
//! `KTSTR_GUEST_INIT` env var, which [`super::init`] sets before Phase 5
//! and which is inherited across fork.

use crate::vmm::wire::LifecyclePhase;
use crate::workload::{SchedPolicy, WorkType, WorkloadConfig, WorkloadHandle};
use std::time::{Duration, Instant};

/// Base wait for the scheduler to dispatch the probe workers before
/// concluding it did not. Bounded so a wedged scheduler fails the cell
/// quickly rather than hanging until the host watchdog.
const DISPATCH_DEADLINE_BASE: Duration = Duration::from_secs(5);

/// Per-worker widening of the dispatch deadline. The probe spawns one
/// worker per online guest CPU and waits for EVERY worker to advance;
/// on a wide guest (128+ CPUs) whose vCPU threads are time-sliced on
/// the host, a flat 5s of wall clock is not 5s of guest CPU — the
/// widest topologies need proportionally more wall time for all
/// workers to fork, set SCHED_EXT, and take a first dispatch.
const DISPATCH_DEADLINE_PER_WORKER: Duration = Duration::from_millis(100);

/// Ceiling on the scaled deadline so a genuinely wedged scheduler
/// still fails in bounded time. Must stay below the host-side
/// `VERIFIER_WORKLOAD_BUDGET` (60s post-attach, see
/// `test_support::runtime`) with room for teardown, so the probe
/// verdict — not the watchdog — decides the cell.
const DISPATCH_DEADLINE_CAP: Duration = Duration::from_secs(30);

// NOTE: no inter-poll interval — the wait is evented. Each fork worker
// signals a shared first-iteration eventfd (`signal_first_iteration`), and
// `WorkloadHandle::wait_first_iteration_all` blocks on that fd until every
// worker has advanced, replacing the prior 5ms `snapshot_iterations` poll.

/// Dispatch deadline for `workers` probe workers: base plus a
/// per-worker term, capped.
fn dispatch_deadline(workers: usize) -> Duration {
    DISPATCH_DEADLINE_BASE
        .saturating_add(DISPATCH_DEADLINE_PER_WORKER.saturating_mul(workers as u32))
        .min(DISPATCH_DEADLINE_CAP)
}

/// Run the SpinWait probe and, on confirmed dispatch, emit
/// [`LifecyclePhase::WorkloadDispatched`]. See the module docs.
pub(crate) fn run_and_confirm_dispatch() {
    let workers = super::topology::count_online_cpus().unwrap_or(1).max(1) as usize;
    let cfg = WorkloadConfig::default()
        .work_type(WorkType::SpinWait)
        .workers(workers)
        // Run the probe workers as SCHED_EXT so the scheduler under test
        // dispatches them even if it uses SCX_OPS_SWITCH_PARTIAL (which
        // leaves SCHED_OTHER tasks in EEVDF/fair). A SCHED_OTHER worker
        // under a partial-switch scheduler would advance `iterations`
        // while running in fair — falsely confirming dispatch; SCHED_EXT
        // forces `task_should_scx` true via the policy arm under BOTH
        // switch modes, so non-zero iterations prove the BPF scheduler
        // dispatched the worker. The set can be rejected with EACCES if
        // the scheduler set `scx.disallow` on the worker (leaving it
        // SCHED_OTHER); the dispatch check below excludes any worker
        // whose `sched_policy_error` is set, so a fair-class fallback
        // cannot false-confirm dispatch.
        .sched_policy(SchedPolicy::Ext)
        // Park each worker the instant it counts its first iteration:
        // one counted iteration under SCHED_EXT is the whole proof the
        // probe needs, and a spinning SpinWait worker per online CPU
        // across ~30 concurrent CI cells is the dominant background
        // load starving other cells' bring-up. Parked workers still
        // publish their final count and respond to stop, so the
        // wait-for-all loop and the dispatch gate below are unchanged.
        .park_after_iterations(Some(1))
        // Signal a shared eventfd after each worker's first counted
        // iteration so the wait below BLOCKS on the dispatch edge instead
        // of polling `snapshot_iterations`. Independent of the park knob:
        // a worker signals once BEFORE parking. Fork workers only (the
        // probe spawns CloneMode::Fork), which is exactly this path.
        .signal_first_iteration(true);
    let mut handle = match WorkloadHandle::spawn(&cfg) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(error = %e, "verifier workload: spawn failed; no dispatch frame");
            return;
        }
    };
    handle.start();

    // Block until EVERY worker has signalled its first outer-loop
    // iteration, or the deadline elapses. We wait for ALL workers, not the
    // first: the eventfd counter sums per-worker signals but cannot see
    // which workers actually became SCHED_EXT, so waking on the FIRST
    // signal could stop while a fair-class fallback worker (whose SCHED_EXT
    // set was rejected under a subset-`scx.disallow` scheduler) had
    // progressed but a real SCHED_EXT worker was still at zero — the
    // dispatch gate below would then find no qualifying worker and
    // false-FAIL a working scheduler. Waiting for all workers guarantees
    // every SCHED_EXT worker has had its chance before we stop; a working
    // scheduler dispatches all of them quickly, and the deadline bounds one
    // that starves a worker (the gate still confirms via any qualifying
    // worker that did advance).
    let deadline = Instant::now() + dispatch_deadline(workers);
    handle.wait_first_iteration_all(deadline, workers);

    // `stop_and_collect` returns authoritative post-run reports (a worker
    // that never reported back is a zeroed sentinel: `iterations == 0`,
    // no `sched_policy_error`). A worker proves dispatch only when it BOTH
    // (a) actually became SCHED_EXT — `sched_policy_error` is None, so the
    // BPF scheduler owns it even under SCX_OPS_SWITCH_PARTIAL — AND (b)
    // advanced `iterations` (the SpinWait arm increments it
    // unconditionally per loop pass). A worker whose SCHED_EXT set was
    // rejected stayed SCHED_OTHER; its progress could be fair-class work
    // under a partial-switch scheduler, so it is NOT counted.
    let reports = handle.stop_and_collect();
    if reports
        .iter()
        .any(|r| r.iterations >= 1 && r.sched_policy_error.is_none())
    {
        crate::vmm::guest_comms::send_lifecycle(LifecyclePhase::WorkloadDispatched, "");
    } else {
        tracing::warn!(
            "verifier workload: no worker confirmed dispatched (0 iterations, or the \
             SCHED_EXT set was rejected leaving the worker in fair)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deadline scaling: base for a 1-worker probe, linear growth in
    /// the mid-range, and the cap binding on the widest topologies —
    /// while staying under the host's post-attach
    /// `VERIFIER_WORKLOAD_BUDGET` so the probe verdict, not the
    /// watchdog, decides the cell.
    #[test]
    fn dispatch_deadline_scales_and_caps() {
        assert_eq!(
            dispatch_deadline(1),
            DISPATCH_DEADLINE_BASE + DISPATCH_DEADLINE_PER_WORKER,
        );
        assert_eq!(dispatch_deadline(64), Duration::from_millis(11_400));
        assert_eq!(dispatch_deadline(128), Duration::from_millis(17_800));
        assert_eq!(dispatch_deadline(512), DISPATCH_DEADLINE_CAP, "cap binds");
        assert!(
            DISPATCH_DEADLINE_CAP < crate::test_support::runtime::VERIFIER_WORKLOAD_BUDGET,
            "probe cap must leave post-attach room for teardown inside \
             the host's workload budget",
        );
    }
}
