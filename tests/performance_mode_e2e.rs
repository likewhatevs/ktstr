//! End-to-end demonstration of the per-phase schbench metric path across a
//! real scheduler detach. Boots with scx-ktstr as the boot scheduler, runs a
//! backdrop schbench workload, holds (scx phase), then Op::DetachScheduler
//! (kernel falls back to EEVDF) and holds again (EEVDF phase). The per-phase
//! schbench percentiles are attributed to each phase's PhaseBucket and read
//! back in post_vm via VmResult::better_across_phases — the user-facing
//! "is scheduler X better than EEVDF?" ergonomic.
//!
//! The pass/fail gate is the FRAMEWORK INVARIANT, not a hard "scx beats EEVDF":
//! scx-ktstr is a trivial test fixture, not a performance scheduler, so it will
//! not reliably win on schbench latency. The test proves (1) both phases produce
//! the per-phase metric across the real detach, and (2) the comparator resolves
//! to a definite Pass/Fail (reads real data + orients by polarity) — never
//! gating on which scheduler won.

use anyhow::Result;
use ktstr::assert::{AssertResult, Phase, Verdict};
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::test_support::{Scheduler, SchedulerSpec};

const PERF_SCX: Scheduler =
    Scheduler::named("perf_scx").binary(SchedulerSpec::Discover("scx-ktstr"));

/// The per-phase metric the demo reads across the detach. A LowerBetter latency
/// percentile (the registry polarity orients the comparator).
const METRIC: &str = "wakeup_p99_latency_us";

/// post_vm: the robust framework-invariant gate (see the file doc). Proves the
/// per-phase schbench metrics survive a real boot + scheduler detach and that
/// the cross-phase comparator reads + orients them — not that scx wins.
fn assert_perphase_metrics_survive_detach(result: &VmResult) -> Result<()> {
    let scx = Phase::step(0); // step_index 1 = scx (boot scheduler)
    let eevdf = Phase::step(1); // step_index 2 = post-DetachScheduler EEVDF

    // Gate 1: both phases produced the per-phase schbench metric — proves the
    // per-phase attribution survives a real VM boot + scheduler detach.
    anyhow::ensure!(
        result.phase_metric(scx, METRIC).is_some(),
        "scx phase (Step[0]) produced no {METRIC} — per-phase carrier missing"
    );
    anyhow::ensure!(
        result.phase_metric(eevdf, METRIC).is_some(),
        "EEVDF phase (Step[1]) produced no {METRIC} — per-phase carrier missing across the detach"
    );

    // Gate 2: the comparator RESOLVES to Pass or Fail (not Inconclusive) —
    // proves it read both phases and oriented by registry polarity. Does NOT
    // gate on which scheduler won (scx-ktstr is a fixture, not a perf scheduler).
    let mut v = Verdict::new();
    result
        .better_across_phases(&mut v, eevdf, scx, METRIC)
        .better_than();
    anyhow::ensure!(
        !v.is_inconclusive(),
        "better_across_phases was inconclusive — could not read both phases or orient by polarity"
    );
    Ok(())
}

// duration_s budgets the in-scenario hold window: run_step clamps each Fixed
// hold to `(scenario_start + duration_s) - now` (ops/mod.rs), so it must cover
// Step0 (4s) + the detach op + Step1 (4s). Op::DetachScheduler waits up to the
// 10s SCX_DISABLED grace (SCHED_LIFECYCLE_KILL_GRACE, dispatch.rs:1645), so a
// slow detach would otherwise truncate the EEVDF hold to ~0 and starve Gate 1.
// 24s covers 4 + 10(worst detach) + 4 with margin. watchdog_timeout_s lifts the
// VM hard deadline -- max(watchdog_timeout, duration) plus boot headroom
// (vm_timeout_from_entry, test_support/runtime.rs) -- so the 35s base covers
// boot + the full worst-case scenario with slack.
#[ktstr_test(
    scheduler = PERF_SCX,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 24,
    watchdog_timeout_s = 35,
    cleanup_budget_ms = 5000,
    num_snapshots = 3,
    post_vm = assert_perphase_metrics_survive_detach,
)]
fn performance_mode_perphase_metrics_across_detach(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::backdrop::Backdrop;
    use ktstr::scenario::ops::{CgroupDef, HoldSpec, Op, Step, execute_scenario};
    use ktstr::workload::{SchbenchConfig, WorkType};
    use std::time::Duration;

    // Backdrop schbench cgroup: persists across BOTH steps (survives step
    // teardown), so the engine spans scx-hold -> detach -> EEVDF-hold and emits
    // one per-phase snapshot per epoch. WorkType::Schbench mandates a single
    // ktstr worker (the message/worker topology lives inside that worker).
    let backdrop = Backdrop::new().push_cgroup(
        CgroupDef::named("sched_bench")
            .work_type(WorkType::schbench(SchbenchConfig::default()))
            .workers(1),
    );

    let steps = vec![
        // Phase step_index 1: scx (boot scheduler) holds while schbench records.
        Step::new(vec![], HoldSpec::fixed(Duration::from_secs(4))),
        // Phase step_index 2: detach scx -> kernel EEVDF, hold again. The epoch
        // bumps 1->2 at this StepStart so the engine splits its histograms.
        Step::new(
            vec![Op::detach_scheduler()],
            HoldSpec::fixed(Duration::from_secs(4)),
        ),
    ];

    execute_scenario(ctx, backdrop, steps)
}
