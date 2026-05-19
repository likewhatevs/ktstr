//! End-to-end coverage of the phase-aware metric pipeline.
//!
//! The phase pipeline threads `step_index` from a per-VM
//! `Arc<AtomicU16>` (published by the Step loop on every Step
//! transition) through the host-side capture path (bridge stamps
//! every periodic capture with the live step_index, sample
//! conversion preserves it, `SampleSeries::by_phase` partitions
//! by it) into [`crate::assert::build_phase_buckets`] which folds
//! per-phase samples into the rendered [`crate::assert::PhaseBucket`]
//! vec the operator sees on `result.stats.phases`. The unit tests
//! at `src/assert/tests_phase_bucket.rs` pin each stage in
//! isolation against synthetic fixtures; this e2e test boots a
//! real guest under scx-ktstr, drives a 2-Step scenario with
//! periodic captures, and asserts the actual pipeline flows
//! step_index through the bridge into the rendered phases vec
//! end-to-end.
//!
//! ## Why this test exists when 38 unit tests already cover the surface
//!
//! Each stage of the pipeline has a unit test (wiring,
//! compare-pass, dual-gate, matches_phase / passes_delta_
//! threshold projection, rel_threshold resolution). This e2e is
//! the integration sentinel that catches gaps BETWEEN those
//! stages: a CURRENT_STEP atomic that's published but never read,
//! a bridge that stamps every capture as `step_index = 0`
//! regardless of guest state, a `SampleSeries::by_phase`
//! BTreeMap that silently drops samples whose step_index doesn't
//! match the bridge's expected encoding. The unit tests pass
//! because each stage is correct in isolation; the e2e test
//! catches when wiring between them breaks.
//!
//! ## Test design
//!
//! 1. Boot scx-ktstr with a 2-LLC-with-2-CPUs topology, 10 s
//!    duration, `num_snapshots = 4` (enough to land samples in
//!    both Step phases across the 10 s window).
//! 2. Run a 2-Step scenario, each Step holding for half the
//!    duration with a single cgroup workload to keep
//!    scx-ktstr's enqueue/dispatch paths advancing across
//!    both phases.
//! 3. In `post_vm`, drain the bridge BEFORE
//!    [`crate::test_support::eval::evaluate_vm_result`]
//!    consumes it, build a [`SampleSeries`], fold through
//!    `build_phase_buckets`, and assert the load-bearing
//!    pipeline contracts (assertions A1-A4 below).
//!
//! ## Vacuity sibling — `phase_pipeline_no_periodic_samples_yields_empty_phases`
//!
//! Separate test runs the same shape with `num_snapshots = 0`
//! and asserts `phases.is_empty()` — the vacuity guard that
//! catches a framework regression where `build_phase_buckets`
//! synthesizes phantom phases even without periodic samples
//! (which would invalidate A1's "non-empty phases" assertion
//! by vacuously passing).
//!
//! ## Assertion ledger
//!
//! - **A1 (CRITICAL)**: phases vec contains at least one Step
//!   bucket (step_index >= 1) — proves CURRENT_STEP advanced
//!   past 0 at least once. Note: ">= 2 buckets" would
//!   over-constrain because BASELINE may legitimately capture
//!   zero samples before the first Step transition (the
//!   framework's pre_buffer + boundary timing can land every
//!   periodic fire inside a Step window); the load-bearing
//!   invariant is that CURRENT_STEP advanced.
//! - **A2 (CRITICAL)**: per-phase label encoding follows the
//!   1-indexed convention (step_index = 0 → "BASELINE",
//!   step_index = k → "Step[k-1]"). Catches: label generator
//!   out-of-sync with framework convention; raw step_index
//!   leaking into label text.
//! - **A3 (MUST-HAVE)**: sample_count across phases sums to
//!   exactly the drained entry count — pinned via captured
//!   `drained_len` before the drain Vec is consumed. Catches:
//!   samples lost between bridge drain and aggregator (sum <
//!   drained_len) AND samples double-counted (sum >
//!   drained_len).
//! - **A4 (MUST-HAVE)**: at least one wired metric key
//!   (`max_dsq_depth`) populated on at least one non-BASELINE
//!   bucket. Catches: per-metric extraction breaks through the
//!   real bridge serialization roundtrip even when synthetic
//!   fixtures (unit tests) pass.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{SampleSeries, VmResult};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Drain the bridge, fold through `build_phase_buckets`, and
/// assert the phase pipeline contracts A1-A4. Runs in `post_vm`
/// before `evaluate_vm_result` would auto-drain; the test owns
/// the drain explicitly so the assertions exercise the same
/// pipeline path the framework uses.
fn assert_phase_pipeline(result: &VmResult) -> Result<()> {
    // The bridge must have captured something across the 2-Step
    // run with `num_snapshots = 4` — `periodic_fired >= 1` is
    // the lower bound the periodic-capture e2e already pins;
    // the phase pipeline builds on that floor.
    anyhow::ensure!(
        result.periodic_fired >= 1,
        "periodic_fired = {} of {} — the freeze coordinator did \
         not produce a capture; the phase pipeline cannot be \
         exercised end-to-end without at least one capture",
        result.periodic_fired,
        result.periodic_target,
    );

    let drained = result.snapshot_bridge.drain_ordered_with_stats();
    anyhow::ensure!(
        !drained.is_empty(),
        "snapshot_bridge.drain_ordered_with_stats returned an \
         empty Vec despite periodic_fired = {}; the bridge drain \
         path is the integration link between freeze-coordinator \
         capture and the phase aggregator",
        result.periodic_fired,
    );

    // Capture drained_len BEFORE the Vec is consumed by
    // from_drained_typed so A3 can pin sum == drained_len. The
    // double-equality form catches BOTH the silent-drop class
    // (sum < drained_len) AND the double-count class (sum >
    // drained_len) — `sum > 0` alone would let a regression
    // that visits each sample twice through by_phase silently
    // produce 2× the sample_count.
    let drained_len = drained.len();
    let series = SampleSeries::from_drained_typed(drained, result.monitor.clone());
    let phases = ktstr::assert::build_phase_buckets(&series);

    // A1 — at least one Step bucket present (step_index >= 1)
    // alongside whatever BASELINE bucket may have landed.
    // `>= 2` would over-constrain since BASELINE may legitimately
    // capture zero samples if no boundary fires before the first
    // Step transition; the load-bearing invariant is that
    // CURRENT_STEP advanced past 0 at least once.
    anyhow::ensure!(
        phases.iter().any(|p| p.step_index >= 1),
        "phases vec lacks any Step buckets (step_index >= 1); \
         every captured sample classified as BASELINE. The \
         CURRENT_STEP atomic was either never updated by the \
         Step loop or never read by the freeze coordinator at \
         periodic-fire site. phases = {:?}",
        phases
            .iter()
            .map(|p| (p.step_index, p.label.as_str()))
            .collect::<Vec<_>>(),
    );

    // A2 — label encoding contract: 1-indexed Step ordinals.
    // step_index = 0 → "BASELINE", step_index = k → "Step[k-1]".
    for phase in &phases {
        let expected = if phase.step_index == 0 {
            "BASELINE".to_string()
        } else {
            format!("Step[{}]", phase.step_index - 1)
        };
        anyhow::ensure!(
            phase.label == expected,
            "phase step_index = {} carries label {:?} but the \
             1-indexed convention requires {:?}. A regression in \
             the label generator (e.g. 0-indexed Step ordinals, \
             raw step_index leak, BASELINE renamed) surfaces here.",
            phase.step_index,
            phase.label,
            expected,
        );
    }

    // A3 — bucket sample_count sums to exactly the drained
    // entry count. The double-equality form catches BOTH the
    // silent-drop class (sum < drained_len) AND the
    // double-count class (sum > drained_len). A bare sum > 0
    // would let a by_phase regression that visits each sample
    // twice silently produce 2× the sample_count and still pass.
    let total_samples: usize = phases.iter().map(|p| p.sample_count).sum();
    anyhow::ensure!(
        total_samples == drained_len,
        "phases vec sum(sample_count) = {} but {} entries were \
         drained from the bridge. Mismatch: \
         - sum < drained: SampleSeries::by_phase / aggregator \
           dropped samples (silent-data-loss class). \
         - sum > drained: by_phase counted samples in multiple \
           buckets (double-count class). phases = {:?}",
        total_samples,
        drained_len,
        phases
            .iter()
            .map(|p| (p.step_index, p.sample_count))
            .collect::<Vec<_>>(),
    );

    // A4 — at least one Step bucket carries a wired metric key.
    // `max_dsq_depth` (Peak/LowerBetter) is the simplest wire
    // to assert: it reads from `DsqState.nr` filtered by
    // `origin.starts_with("local cpu ")`; scx-ktstr produces
    // local-cpu DSQs as a baseline, so every healthy capture
    // populates the entry.
    //
    // BASELINE may legitimately have an empty metrics map if
    // it captured zero samples before the first Step transition;
    // assert against the Step buckets only.
    let metric_present = phases
        .iter()
        .filter(|p| p.step_index >= 1)
        .any(|p| p.metrics.contains_key("max_dsq_depth"));
    anyhow::ensure!(
        metric_present,
        "no Step bucket carries the `max_dsq_depth` metric — \
         per-metric extraction works in unit tests but \
         broke through the real bridge / sample roundtrip. \
         Step buckets: {:?}",
        phases
            .iter()
            .filter(|p| p.step_index >= 1)
            .map(|p| (p.step_index, p.metrics.keys().collect::<Vec<_>>()))
            .collect::<Vec<_>>(),
    );

    Ok(())
}

/// 2-Step scenario with periodic captures across both Steps.
/// Each Step holds for half the 10 s duration; the framework's
/// CURRENT_STEP atomic advances at the boundary between the two
/// (Step[0] = step_index 1, Step[1] = step_index 2), and the
/// periodic-capture loop fires across both windows producing
/// captures stamped with both step_index values.
///
/// Workload: bare `CgroupDef::named.workers(N)` without an
/// explicit payload. A4 (max_dsq_depth assertion) depends on
/// scx-ktstr's per-CPU local DSQ populating from cgroup workers
/// regardless of payload; per phd recon, scx-ktstr always
/// produces "local cpu N" DSQ origins from its
/// enqueue/dispatch architecture, so the worker-only scenario
/// is sufficient. If a future scx-ktstr refactor changes when
/// local DSQs populate, A4 may need an explicit
/// `payload(WorkType::CpuSpin)` instead.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 10,
    watchdog_timeout_s = 20,
    num_snapshots = 4,
    auto_repro = false,
    post_vm = assert_phase_pipeline,
)]
fn phase_pipeline_two_step_e2e(ctx: &Ctx) -> Result<AssertResult> {
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

/// Vacuity sibling — same scenario shape with periodic captures
/// disabled (`num_snapshots = 0`). The bridge ends the run with
/// no `periodic_*` entries, the aggregator builds an empty
/// phases vec, and the test asserts the vacuity contract:
/// `build_phase_buckets` does NOT synthesize phantom phases
/// when no samples exist.
///
/// Without this sibling, a regression where the aggregator
/// always returns `vec![PhaseBucket::default(), ...]` regardless
/// of input would let the happy-path test pass vacuously
/// (phases.len() >= 1 + step_index = 0 BASELINE label both
/// pass) without exercising the real pipeline.
fn assert_phase_pipeline_vacuity(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.periodic_target == 0,
        "vacuity test must run with periodic_target = 0; got {}",
        result.periodic_target,
    );
    anyhow::ensure!(
        result.periodic_fired == 0,
        "vacuity test had {} periodic fires despite num_snapshots = 0 \
         — the periodic-capture path fired without a configured \
         target, which is a framework defect",
        result.periodic_fired,
    );

    let drained = result.snapshot_bridge.drain_ordered_with_stats();
    let series = SampleSeries::from_drained_typed(drained, result.monitor.clone());
    let phases = ktstr::assert::build_phase_buckets(&series);
    anyhow::ensure!(
        phases.is_empty(),
        "build_phase_buckets produced {} phantom buckets from \
         an empty sample series — the aggregator synthesizes \
         output without input, masking the silent-data-loss \
         class for every other phase test. phases = {:?}",
        phases.len(),
        phases
            .iter()
            .map(|p| (p.step_index, p.label.as_str(), p.sample_count))
            .collect::<Vec<_>>(),
    );

    Ok(())
}

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 5,
    watchdog_timeout_s = 15,
    num_snapshots = 0,
    auto_repro = false,
    post_vm = assert_phase_pipeline_vacuity,
)]
fn phase_pipeline_no_periodic_samples_yields_empty_phases(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_vacuity").workers(2)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
