//! e2e: `survives_storm` is actively enforced on a NON-`execute_*` scenario.
//!
//! The companion `tests/scheduler_survives_storm_e2e.rs` drives an
//! `execute_steps` scenario, whose in-hold liveness probe records scheduler
//! death. This test deliberately HAND-ROLLS the workload via `WorkloadHandle`
//! (no `execute_*` driver, so no in-hold probe runs) to pin the guest-side
//! post-function probe (`enforce_survives_storm_liveness`): scx-ktstr survives
//! the hold, so the probe finds the scheduler still `enabled` and the run
//! PASSES — proving the probe is wired on a hand-rolled path and does not
//! false-fire on a healthy scheduler.
//!
//! The death direction — the probe folding a `Scheduler*` fail when the
//! scheduler is down — is pinned host-side in
//! `src/test_support/probe_tests.rs`
//! (`enforce_survives_storm_liveness_folds_only_on_unattributed_death`), the
//! CI-runnable invariant that proves the fold logic without a `/dev/kvm` VM, so
//! this VM-gated e2e SKIP cannot read as a vacuous PASS.

use ktstr::prelude::*;

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// `#[ktstr_test(survives_storm)]` over a HAND-ROLLED workload (a
/// `WorkloadHandle` held for the scenario duration, NOT an `execute_*` step
/// driver) against the scx-ktstr fixture. Because no `execute_*` driver runs,
/// the only survival enforcement is the guest-side post-function probe. scx-ktstr
/// has no crashing handler, so it survives the hold: the probe finds the
/// scheduler still `enabled`, leaves the passing result untouched, and the run
/// is EXIT_PASS. Had the scheduler died, the probe would fold a
/// `DetailKind::Scheduler*` fail, the eval layer would attach
/// `SurvivesStormViolated`, and dispatch would force EXIT_FAIL with the
/// survival explainer.
///
/// `num_workers = 2` on `cores = 4`: SpinWait has no blocking call, so more
/// workers than spare vCPUs starves the parent test thread + scx-ktstr monitor
/// (the same vCPU-saturation flake `tests/worker_thread_integration.rs`
/// documents); two workers keep two vCPUs free for the runtime.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    survives_storm,
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 1024,
    duration_s = 5,
    watchdog_timeout_s = 15
)]
fn survives_storm_non_execute_e2e(ctx: &Ctx) -> Result<AssertResult> {
    let config = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::SpinWait,
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    // The workload run window the scheduler must survive — not a
    // synchronization wait. Mirrors the hand-rolled-workload pattern in
    // tests/worker_thread_integration.rs.
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    if reports.is_empty() {
        return Ok(AssertResult::fail_msg(
            "hand-rolled workload produced no worker reports; spawn broken".to_string(),
        ));
    }
    // A clean survival is a bare pass; the post-function survives_storm probe
    // is what would turn a mid-run scheduler death into EXIT_FAIL.
    Ok(AssertResult::pass())
}
