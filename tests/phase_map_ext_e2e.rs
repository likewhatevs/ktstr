//! End-to-end coverage for the [`PhaseMapExt`] cross-phase
//! comparator + [`pickers::max_by_counter_value`] picker that
//! shrink the swap-counter-A/B-test pattern down to a 6-line
//! post_vm callback.
//!
//! Boots a single 2-Step scenario with periodic captures across
//! both Steps so the resulting [`SampleSeries`] carries Step[0]
//! and Step[1] samples. The post_vm callback exercises the full
//! pipeline the way a real A/B test does:
//!
//! 1. Project a cumulative BPF counter via
//!    `series.bpf("...", |snap| snap.var("...").as_u64())`.
//! 2. Reduce per phase via
//!    [`SeriesField::counter_delta_per_phase`].
//! 3. Compose with another projection via
//!    [`PhaseMapExt::zip_per_phase`].
//! 4. Compare across phases via
//!    [`PhaseMapExt::ratio_across_phases`]`.at_most(...)`.
//! 5. Confirm the verdict carries either an info note (pass arm)
//!    or a failure detail mentioning the comparator (fail arm).
//!
//! The test does NOT pin which arm fires because scheduler
//! activity across the window depends on host load — but it
//! pins the comparator chain produces a recorded verdict
//! mutation either way, end-to-end through the real VM.

use anyhow::Result;
use ktstr::assert::{AssertResult, Phase, PhaseMapExt, Verdict};
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Generous ceiling for the cross-phase ratio. scx-ktstr's
/// dispatch counter grows monotonically across the window so the
/// step(0)→step(1) ratio is unbounded above; a ceiling of 10⁹
/// keeps the comparator on the pass arm regardless of host load
/// while still exercising the pass-arm info-note path.
const RATIO_CEILING: f64 = 1_000_000_000.0;

fn assert_phase_map_ext_pipeline(result: &VmResult) -> Result<()> {
    // Environmental starvation gate: zero real captures under a
    // witnessed-contended host is a non-verdict (the readiness-gated
    // capture chain was starved past the workload window), not a
    // capture regression — SKIP instead of failing the assertions
    // below. A quiet-host zero-capture run still falls through and
    // fails with the specific diagnosis. See `periodic_starvation_gate`.
    ktstr::prelude::periodic_starvation_gate(result, 2)?;
    let periodic_fired = result.periodic_fired;
    let periodic_target = result.periodic_target;
    anyhow::ensure!(
        periodic_fired >= 2,
        "periodic_fired = {periodic_fired} of {periodic_target} — \
         counter_delta_per_phase needs at least 2 Ok samples per phase \
         to produce a non-zero delta",
    );

    let series = result.periodic_series();

    // counter_delta_per_phase requires the SeriesField projection to
    // be sample-axis stamped with per-phase tags (which periodic_series
    // does via the build_series_field phase-threading path). Project
    // the framework's BPF dispatch counter once per "sub-projection"
    // so we exercise zip_per_phase on real data: same / cross-axes of
    // dispatch.
    //
    // scx-ktstr doesn't actually carry per-LLC same/cross axes (it's
    // a single-bss counter), so we synthesize an A/B shape by reading
    // the same counter twice and zipping with itself. The result is
    // identical per-phase delta values; the test's load-bearing
    // claim is that the chain composes + records a verdict, NOT
    // that the metric value is meaningful here.
    let a_delta = series
        .bpf("nr_dispatched_a", |snap| {
            snap.var("nr_dispatched").as_u64().map(|v| v as f64)
        })
        .counter_delta_per_phase();
    let b_delta = series
        .bpf("nr_dispatched_b", |snap| {
            snap.var("nr_dispatched").as_u64().map(|v| v as f64)
        })
        .counter_delta_per_phase();

    // Step pins: at least one Step phase produced a delta on the
    // self-zip. If both maps are empty, the phase-stamp pipeline
    // is broken upstream — surface that loudly rather than letting
    // the comparator silently no-op on an empty zip result.
    anyhow::ensure!(
        a_delta.contains_key(&Phase::step(0)) || a_delta.contains_key(&Phase::step(1)),
        "counter_delta_per_phase produced no Step buckets despite \
         periodic_fired = {periodic_fired}; either the phase pipeline \
         isn't stamping samples, or every sample within both phases is \
         an Err. a_delta keys = {:?}",
        a_delta.keys().collect::<Vec<_>>(),
    );

    // zip_per_phase exercises the intersection-only fold with a
    // by-value closure (Copy bound on f64) — pins the trait isn't
    // forcing deref noise on the composition body.
    let synthetic_frac = a_delta.zip_per_phase(&b_delta, |a, b| {
        let total = a + b;
        if total == 0.0 { 0.0 } else { a / total }
    });
    anyhow::ensure!(
        !synthetic_frac.is_empty(),
        "zip_per_phase yielded an empty result map despite both \
         input maps having Step phases; intersection of a_delta keys = \
         {:?} with b_delta keys = {:?} should not be empty",
        a_delta.keys().collect::<Vec<_>>(),
        b_delta.keys().collect::<Vec<_>>(),
    );
    // self-zip cross_frac = a / (a + a) = 0.5 for any non-zero phase.
    for (phase, value) in &synthetic_frac {
        if *value > 0.0 {
            anyhow::ensure!(
                (value - 0.5).abs() < 1e-9,
                "self-zip cross_frac for {phase:?} = {value}, expected 0.5",
            );
        }
    }

    // ratio_across_phases takes the two-phase ratio path only when BOTH
    // Step(0) and Step(1) are populated; a missing phase records an
    // Inconclusive (neither an info note nor a failure detail), which the
    // assertion below reads as "the chain didn't fire". The
    // periodic_starvation_gate above only floors the TOTAL capture count,
    // so a starved run whose few captures all landed in one Step passes
    // the gate yet leaves the other Step empty. Under witnessed host
    // contention that skew is environmental, not a comparator regression
    // — SKIP. On a quiet host a missing phase is a real bug and falls
    // through to the assertion's specific diagnosis below.
    let both_steps = synthetic_frac.contains_key(&Phase::step(0))
        && synthetic_frac.contains_key(&Phase::step(1));
    if !both_steps && let Some(d) = ktstr::prelude::capture_starvation_witness(result) {
        return Err(ktstr::prelude::post_vm_skip(format!(
            "periodic captures populated only one Step bucket under \
             witnessed host contention (D={d:.2}, step0={}, step1={}): \
             ratio_across_phases cannot take the two-phase ratio path — \
             environmental non-verdict",
            synthetic_frac.contains_key(&Phase::step(0)),
            synthetic_frac.contains_key(&Phase::step(1)),
        )));
    }

    // PhaseMapExt::ratio_across_phases lands the verdict. With both
    // Step(0) and Step(1) populated the comparator takes the ratio path:
    // a pass info note when the ratio is within the ceiling, or a
    // failure detail when it exceeds the ceiling / is non-finite. Pin
    // that EITHER a pass note OR a failure detail mentioning the label
    // landed, proving the comparator ran.
    let mut verdict = Verdict::new();
    synthetic_frac
        .ratio_across_phases(
            &mut verdict,
            "synthetic_frac",
            Phase::step(0),
            Phase::step(1),
        )
        .at_most(RATIO_CEILING);
    let r = verdict.into_result();
    let recorded_pass = r
        .info_notes
        .iter()
        .any(|n| n.message.contains("synthetic_frac"));
    let recorded_fail = r
        .failure_details()
        .any(|d| d.message.contains("synthetic_frac"));
    anyhow::ensure!(
        recorded_pass || recorded_fail,
        "PhaseMapExt::ratio_across_phases produced NEITHER a pass info \
         note NOR a failure detail mentioning the supplied label — \
         the chain didn't fire. info_notes = {:?}, details = {:?}",
        r.info_notes,
        r.failure_details().collect::<Vec<_>>(),
    );

    Ok(())
}

/// 2-Step scenario with periodic captures across both Steps —
/// mirrors `phase_pipeline_two_step_e2e` so the CURRENT_STEP atomic
/// advances at the boundary and periodic samples land in both
/// `Phase::step(0)` and `Phase::step(1)` buckets that
/// `counter_delta_per_phase` reduces over.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 10,
    watchdog_timeout_s = 20,
    num_snapshots = 6,
    auto_repro = false,
    post_vm = assert_phase_map_ext_pipeline,
)]
fn phase_map_ext_compose_zip_then_ratio_across_phases_e2e(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step {
            setup: vec![CgroupDef::named("cg_step0").workers(2)].into(),
            ops: vec![],
            hold: HoldSpec::frac(0.5),
        },
        Step {
            setup: vec![CgroupDef::named("cg_step1").workers(2)].into(),
            ops: vec![],
            hold: HoldSpec::frac(0.5),
        },
    ];
    execute_steps(ctx, steps)
}
