//! End-to-end exercise of scx-ktstr's `ktstr_yield` (ops.yield).
//!
//! A [`WorkType::YieldHeavy`] workload — each worker loops
//! `std::thread::yield_now()`, which is `sched_yield(2)` on Linux —
//! runs under scx-ktstr for 10 s with three periodic captures.
//! `sched_yield(2)` lands in `yield_task_scx` (kernel/sched/ext.c) and
//! is routed to the registered `ktstr_yield` handler, which bumps the
//! `.bss` `nr_yielded` counter. The host-side `post_vm` check reads
//! `nr_yielded` from every periodic sample and asserts it advanced
//! (the `any_progress` floor) and never regressed (`nondecreasing`).
//!
//! Without a registered ops.yield the counter would stay 0 — the
//! kernel's default slice-zero fallback does not touch any scheduler
//! counter — so the `any_progress` floor catches a missing or
//! unregistered handler, and `nondecreasing` catches a counter wrap or
//! a stale/duplicated capture in the snapshot pipeline.

use anyhow::Result;
use ktstr::assert::{AssertResult, Verdict};
use ktstr::ktstr_test;
use ktstr::prelude::{SampleSeries, VmResult, WorkType};
use ktstr::scenario::ops::{HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Host-side check: `nr_yielded` advanced at least once and is
/// monotone across the periodic samples.
fn assert_yielded_advanced(result: &VmResult) -> Result<()> {
    let series = SampleSeries::from_drained_typed(
        result.snapshot_bridge.drain_ordered_with_stats(),
        result.monitor.clone(),
    )
    .periodic_only();
    anyhow::ensure!(
        series.len() >= 2,
        "need at least 2 periodic samples for nondecreasing to be \
         non-vacuous, got {} (periodic_target={}, periodic_fired={})",
        series.len(),
        result.periodic_target,
        result.periodic_fired,
    );

    // any_progress floor: ops.yield must fire at least once under the
    // YieldHeavy workload. A counter stuck at 0 means `ktstr_yield` was
    // never invoked — i.e. the handler is missing from `SCX_OPS_DEFINE`
    // or no worker reached `sched_yield`. Mirrors the `any_progress`
    // pattern in temporal_assertions_e2e.rs.
    let yielded = series.bpf("nr_yielded", |snap| snap.var("nr_yielded").as_u64());
    let any_progress = yielded
        .iter_full()
        .any(|(_, _, slot)| matches!(slot, Ok(v) if *v > 0));
    anyhow::ensure!(
        any_progress,
        "nr_yielded read 0 across every periodic sample — ops.yield \
         never fired under the YieldHeavy workload (is ktstr_yield \
         registered in SCX_OPS_DEFINE?)",
    );

    // Cumulative counter: `__sync_fetch_and_add(&nr_yielded, 1)` in
    // `ktstr_yield` means the host-side .bss read at every freeze
    // boundary observes a value at or above the prior sample's.
    let mut verdict = Verdict::new();
    series
        .bpf("nr_yielded", |snap| snap.var("nr_yielded").as_u64())
        .nondecreasing(&mut verdict);

    let r = verdict.into_result();
    if r.is_fail() {
        let lines: Vec<String> = r
            .failure_details()
            .map(|d| format!("  [{:?}] {}", d.kind, d.message))
            .collect();
        anyhow::bail!(
            "nr_yielded monotonicity failed across {} sample(s):\n{}",
            series.len(),
            lines.join("\n"),
        );
    }
    Ok(())
}

/// A `YieldHeavy` cgroup runs under scx-ktstr for 10 s with three
/// periodic captures; each worker's `sched_yield(2)` drives
/// `ktstr_yield`, so the `.bss` `nr_yielded` counter advances at every
/// freeze boundary. The cgroup holds workers across the full duration
/// so the yield callback fires continuously.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 10,
    watchdog_timeout_s = 15,
    num_snapshots = 3,
    auto_repro = false,
    post_vm = assert_yielded_advanced,
)]
fn yield_op_increments_nr_yielded(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![ctx.cgroup_def("cg_0").work_type(WorkType::YieldHeavy)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
