//! Reproducer for a reported scx_mitosis periodic-capture
//! 0-fire bug. The claim: when a scheduler runs with
//! `--cell-parent-cgroup /path` in `sched_args` AND the test sets
//! `workload_root_cgroup = "/path"` matching, periodic captures
//! silently 0-fire even after the workload_root_cgroup + scheduler
//! _cgroup_parent wiring (commits 0bdd5e62, 9e830f26).
//!
//! Test uses scx-ktstr (NOT cell-aware) but supplies the same
//! `--cell-parent-cgroup` argv to isolate whether the 0-fire is
//! framework-side (any scheduler with the flag in argv hits it) or
//! scx_mitosis-specific (cell-mode CellManager interference).

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{VmResult, WorkType};
use ktstr::scenario::ops::{HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const CELL_PARENT_SCHED: Scheduler = Scheduler::named("ktstr_sched_cellparent")
    .binary(SchedulerSpec::Discover("scx-ktstr"))
    .sched_args(&["--cell-parent-cgroup", "/ktstr"]);

fn assert_periodic_fired(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.periodic_target == 2,
        "periodic_target must mirror num_snapshots = 2, got {}",
        result.periodic_target,
    );
    anyhow::ensure!(
        result.periodic_fired >= 1,
        "periodic_fired = {} of {} — sibling's bug reproduces: \
         scheduler with `--cell-parent-cgroup` in sched_args + \
         matching `workload_root_cgroup` does not fire any \
         periodic boundary",
        result.periodic_fired,
        result.periodic_target,
    );
    Ok(())
}

#[ktstr_test(
    scheduler = CELL_PARENT_SCHED,
    workload_root_cgroup = "/ktstr",
    duration_s = 6,
    watchdog_timeout_s = 12,
    num_snapshots = 2,
    auto_repro = false,
    post_vm = assert_periodic_fired,
)]
fn periodic_fires_with_explicit_cell_parent_cgroup(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            ctx.cgroup_def("cg_0")
                .workers(64)
                .work_type(WorkType::Bursty {
                    burst_duration: std::time::Duration::from_millis(1),
                    sleep_duration: std::time::Duration::from_millis(2),
                }),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    let mut result = execute_steps(ctx, steps)?;
    // Telemetry-decoupled-from-assertions symptom pin: this scenario
    // configures NO worker check (execute_steps passes none), yet 64 workers
    // ran. Per-cgroup telemetry is pure measurement and MUST populate
    // regardless of whether any worker-check assertion was requested.
    // Pre-fix collect_handles gated stats.cgroups population behind
    // checks.has_worker_checks(), so this came back [] despite the workload
    // running — the empty-cgroups symptom. This is NOT
    // --cell-parent-specific; the cell-parent layout is incidental here.
    anyhow::ensure!(
        !result.stats.cgroups.is_empty(),
        "per-cgroup telemetry empty (stats.cgroups == []) with no worker \
         check configured, despite 64 workers running — telemetry must not be \
         gated behind assertions",
    );
    anyhow::ensure!(
        result.stats.cgroups.iter().any(|c| c.num_workers > 0),
        "a populated cgroup must report a non-zero worker count; got {:?}",
        result
            .stats
            .cgroups
            .iter()
            .map(|c| c.num_workers)
            .collect::<Vec<_>>(),
    );
    result.note(
        "6s workload with --cell-parent-cgroup explicit + workload_root_cgroup match + 64 workers",
    );
    Ok(result)
}
