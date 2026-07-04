//! Guest-side verifier workload: a scheduler-agnostic "did the scheduler
//! dispatch?" probe run at [`super::init`]'s Phase 5 when the guest was
//! booted with `--ktstr-verifier-workload` (the `cargo ktstr verifier`
//! sweep path — a VM with no `#[ktstr_test]` body).
//!
//! After the scheduler has attached (Phase 3) and Phase 5 has emitted the
//! `PayloadStarting` frame, this spawns a SpinWait workload — as
//! SCHED_EXT tasks, so the BPF scheduler dispatches them under any switch
//! mode (full or `SCX_OPS_SWITCH_PARTIAL`) — sized to the guest's online
//! CPU count, waits (by polling live per-worker iteration counts, not a
//! blind fixed sleep) until every worker has advanced at least one
//! iteration or a bounded deadline elapses, then stops the workload. It
//! emits a [`LifecyclePhase::WorkloadDispatched`] frame
//! only for a worker that BOTH advanced non-zero `iterations` AND had its
//! SCHED_EXT set succeed (`sched_policy_error` is None) — so a fair-class
//! fallback cannot false-confirm — proof the scheduler actually
//! dispatched a task onto a CPU. The host
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

/// Longest we wait for the scheduler to dispatch a worker before
/// concluding it did not. Bounded so a wedged scheduler fails the cell
/// quickly rather than hanging until the host watchdog (120s).
const DISPATCH_DEADLINE: Duration = Duration::from_secs(5);

/// Inter-poll pause. `snapshot_iterations` is a polled shared-memory
/// counter (no eventfd to block on), so the wait is a bounded poll that
/// breaks the instant a worker advances — not a fixed sleep-then-stop.
const POLL_INTERVAL: Duration = Duration::from_millis(5);

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
        .sched_policy(SchedPolicy::Ext);
    let mut handle = match WorkloadHandle::spawn(&cfg) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(error = %e, "verifier workload: spawn failed; no dispatch frame");
            return;
        }
    };
    handle.start();

    // Poll live per-worker progress until EVERY worker has advanced at
    // least one outer-loop iteration, or the deadline elapses. We wait for
    // ALL workers, not the first: `snapshot_iterations` reports only
    // per-index counts and cannot see which workers actually became
    // SCHED_EXT, so breaking on the FIRST advance could stop while a
    // fair-class fallback worker (whose SCHED_EXT set was rejected under a
    // subset-`scx.disallow` scheduler) had progressed but a real SCHED_EXT
    // worker was still at zero — the dispatch gate below would then find no
    // qualifying worker and false-FAIL a working scheduler. Waiting for all
    // workers guarantees every SCHED_EXT worker has had its chance before
    // we stop; a working scheduler dispatches all of them quickly, and the
    // deadline bounds one that starves a worker (the gate still confirms
    // via any qualifying worker that did advance).
    let deadline = Instant::now() + DISPATCH_DEADLINE;
    loop {
        let iters = handle.snapshot_iterations();
        if !iters.is_empty() && iters.iter().all(|&it| it >= 1) {
            break;
        }
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(POLL_INTERVAL);
    }

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
