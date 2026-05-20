//! End-to-end coverage for the API-gap fixes that landed in the
//! mid-2026-05-18 batch: [`VmResult::periodic_series`] sugar, the
//! [`SeriesField::ratio_across_phases`] cross-phase comparator,
//! [`SeriesField::value_at_phase`] / [`SeriesField::last_per_phase`]
//! per-phase reductions, and [`Snapshot::live_var_via`] live-pick
//! disambiguator.
//!
//! Boots a single 2-Step scenario with periodic captures across
//! both Steps so the resulting [`SampleSeries`] carries samples
//! stamped with both `Phase::step(0)` and `Phase::step(1)`. One
//! `post_vm` callback covers every helper in one drain:
//!
//! * `result.periodic_series()` returns a non-empty series — pins
//!   the sugar drains the same bridge `SampleSeries::from_drained_typed`
//!   would have, with `periodic_only()` applied.
//! * `series.bpf(...).value_at_phase(Phase::step(0))` returns
//!   `Some(_)` and `last_per_phase()` carries the same key —
//!   pins the two phase-reduction helpers are wired to the
//!   periodic-axis phase stamps the framework emits.
//! * `series.bpf(...).ratio_across_phases(verdict, Step[0], Step[1])
//!   .at_most(...)` lands either pass (records info note) or fail
//!   (records temporal detail) — pins the comparator chain
//!   propagates the verdict mutation regardless of which arm fires.
//!
//! Vacuity: the test does NOT pin which arm of `ratio_across_phases`
//! fires (pass vs fail) because the underlying counter value depends
//! on scheduler activity across the window. It pins the helper
//! returns a recorded verdict mutation either way.

use anyhow::Result;
use ktstr::assert::{AssertResult, Phase, Verdict};
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Generous ceiling for the cross-phase ratio. scx-ktstr's
/// dispatch counter grows monotonically across the window; the
/// step(0)→step(1) ratio can easily exceed 1.0 (later phase
/// observes a larger cumulative value). A ceiling of 10⁹ keeps
/// the comparator on the pass arm regardless of host-load
/// variation while still exercising the pass-arm note record path.
const RATIO_CEILING: f64 = 1_000_000_000.0;

fn assert_api_gap_helpers(result: &VmResult) -> Result<()> {
    let periodic_target = result.periodic_target;
    let periodic_fired = result.periodic_fired;
    anyhow::ensure!(
        periodic_fired >= 1,
        "periodic_fired = {periodic_fired} of {periodic_target} — the freeze \
         coordinator did not produce a capture; the API-gap helpers \
         cannot be exercised without at least one periodic sample",
    );

    // gap 5: VmResult::periodic_series sugar drains the bridge.
    let series = result.periodic_series();
    anyhow::ensure!(
        !series.is_empty(),
        "VmResult::periodic_series returned an empty series despite \
         periodic_fired = {periodic_fired}; the sugar's drain path \
         diverges from `from_drained_typed(...).periodic_only()`",
    );
    anyhow::ensure!(
        series.len() == periodic_fired as usize,
        "VmResult::periodic_series length {} != periodic_fired {} — \
         either the periodic-only filter dropped non-periodic captures \
         (which this test does NOT generate), or the bridge double-counted",
        series.len(),
        periodic_fired,
    );

    // Project to f64 because [`SeriesField::ratio_across_phases`]
    // requires `T: Copy + Into<f64> + Display`, and u64→f64 is a
    // lossy conversion the stdlib does not expose through `Into`.
    // `as f64` lossy-converts at the projection boundary; the
    // f64 range easily holds any plausible dispatch count.
    let bpf_dispatched = series.bpf("nr_dispatched", |snap| {
        snap.var("nr_dispatched").as_u64().map(|v| v as f64)
    });

    // gap 6: value_at_phase + last_per_phase consume the same series.
    // The framework's phase pipeline labels the FIRST Step
    // step_index = 1 and the SECOND Step step_index = 2; the
    // assert::Phase encoding is 0-indexed (Phase::step(0) =
    // step_index 1 = "Step[0]"). Pin that at least one of the two
    // Step phases produced a value — captures may land in either
    // phase depending on the freeze-coordinator's stride across
    // the 10s window.
    let v0 = bpf_dispatched.value_at_phase(Phase::step(0));
    let v1 = bpf_dispatched.value_at_phase(Phase::step(1));
    anyhow::ensure!(
        v0.is_some() || v1.is_some(),
        "value_at_phase returned None for BOTH Step phases despite \
         periodic_fired = {periodic_fired}; the periodic samples \
         landed under no Step phase, which means the CURRENT_STEP \
         atomic + phase pipeline aren't producing phase stamps \
         (every sample classified as BASELINE)",
    );

    let last_map = bpf_dispatched.last_per_phase();
    anyhow::ensure!(
        !last_map.is_empty(),
        "last_per_phase produced an empty map despite the series \
         carrying {} periodic samples; either the phase stamps are \
         all None (regression in the periodic-axis phase plumbing) \
         or every sample carried Err (which the test floor rules out)",
        series.len(),
    );

    // gap 6: ratio_across_phases records either pass or fail —
    // pin that the verdict mutation lands either way. Skip when
    // only one phase observed a sample (the ratio is undefined
    // and would record a known-failure "needs both phases" detail
    // that doesn't exercise the comparison arm).
    if v0.is_some() && v1.is_some() {
        let mut verdict = Verdict::new();
        bpf_dispatched
            .ratio_across_phases(&mut verdict, Phase::step(0), Phase::step(1))
            .at_most(RATIO_CEILING);
        let r = verdict.into_result();
        let recorded_pass_note = r
            .info_notes
            .iter()
            .any(|n| n.message.contains("ratio_across_phases"));
        let recorded_fail_detail = r
            .failure_details()
            .any(|d| d.message.contains("ratio_across_phases"));
        anyhow::ensure!(
            recorded_pass_note || recorded_fail_detail,
            "ratio_across_phases.at_most produced NEITHER a pass info \
             note NOR a failure detail mentioning 'ratio_across_phases' \
             — the comparator silently no-op'd. info_notes = {:?}, \
             details = {:?}",
            r.info_notes,
            r.failure_details().collect::<Vec<_>>(),
        );
    }

    Ok(())
}

/// 2-Step scenario with periodic captures across both Steps —
/// mirrors the `phase_pipeline_two_step_e2e` shape so the
/// framework's CURRENT_STEP advance + periodic-axis phase
/// stamping produces samples in both `Phase::step(0)` and
/// `Phase::step(1)` buckets that the API-gap helpers consume.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 10,
    watchdog_timeout_s = 20,
    num_snapshots = 4,
    auto_repro = false,
    post_vm = assert_api_gap_helpers,
)]
fn api_gaps_periodic_series_and_phase_helpers_e2e(ctx: &Ctx) -> Result<AssertResult> {
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
