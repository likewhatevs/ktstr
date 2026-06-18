//! End-to-end coverage for the step-LOCAL `iteration_rate` model.
//!
//! Boots `custom_cgroup_load_oscillation`, whose four Steps each declare
//! FRESH step-local `cg_0` / `cg_1` CgroupDefs — the per-Step auto-
//! teardown at every boundary stops the prior step's workers and respawns
//! new ones. Every `StepStart[k]` therefore reads ~0 cumulative worker
//! iterations.
//!
//! That respawn-per-step shape is exactly what the pre-fix cross-step
//! model got wrong: pairing `StepStart[k]` -> `StepStart[k+1]` reads
//! `0 -> 0` and yields no rate, so EVERY fresh-per-step phase except the
//! last (which the scenario-end terminal rescued) silently reported no
//! throughput. Pairing each step's own `StepStart[k]` -> `StepEnd[k]`
//! (the per-step end-of-hold frame, sent while that step's workers are
//! still alive) recovers each step's real start-to-end-of-hold rate.
//!
//! The post_vm callback reads the framework-built per-phase buckets via
//! [`VmResult::phase_buckets`] and asserts at least two fresh-per-step
//! phases report a positive step-local `iteration_rate`. The pre-fix
//! cross-step model can give AT MOST ONE fresh-per-step phase a rate (the
//! last, via the terminal), so `>= 2` is reachable only through the
//! step-local `StepStart[k]` -> `StepEnd[k]` pairing — a decisive,
//! host-load-robust discriminator (it never pins WHICH steps, only that
//! more than one measured a rate).

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

fn assert_step_local_iteration_rates(result: &VmResult) -> Result<()> {
    let buckets = result.phase_buckets();
    // Step phases only — step_index 0 is the pre-Step BASELINE.
    let step_rates: Vec<(u16, Option<f64>)> = buckets
        .iter()
        .filter(|b| b.step_index >= 1)
        .map(|b| (b.step_index, b.get("iteration_rate")))
        .collect();
    anyhow::ensure!(
        !step_rates.is_empty(),
        "post_vm: no Step phase buckets — periodic captures never \
         attributed to a scenario step; phase_buckets = {:?}",
        buckets
            .iter()
            .map(|b| (b.step_index, b.label.clone()))
            .collect::<Vec<_>>(),
    );
    // Fresh-per-step phases reporting a measured, positive step-local rate.
    let positive = step_rates
        .iter()
        .filter(|(_, r)| r.is_some_and(|v| v > 0.0))
        .count();
    anyhow::ensure!(
        positive >= 2,
        "expected >= 2 fresh-per-step phases to report a positive \
         step-local iteration_rate (the pre-fix cross-step model caps at \
         one, via the scenario-end terminal); got {positive}. \
         per-step (step_index, iteration_rate) = {step_rates:?}",
    );
    Ok(())
}

/// Four-phase load oscillation with fresh step-local workers per Step.
/// Mirrors `cover_cgroup_load_oscillation`'s monitor config (workload-
/// generation coverage, not scheduler correctness — rate ceilings raised)
/// and adds periodic captures + the step-local-rate post_vm assertion.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 2048,
    duration_s = 12,
    watchdog_timeout_s = 25,
    num_snapshots = 12,
    sustained_samples = 25,
    max_keep_last_rate = 1000000000.0,
    max_fallback_rate = 1000000000.0,
    auto_repro = false,
    post_vm = assert_step_local_iteration_rates,
)]
fn step_local_iteration_rate_per_phase_e2e(ctx: &Ctx) -> Result<AssertResult> {
    ktstr::scenario::interaction::custom_cgroup_load_oscillation(ctx)
}
