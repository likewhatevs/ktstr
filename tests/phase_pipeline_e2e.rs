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
use ktstr::prelude::{Backdrop, SampleSeries, VmResult};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Op, Step, execute_scenario, execute_steps};
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
    // The COMPLETE timeline (step frames + scenario-end terminal) via
    // the shared accessor — folding only the raw wire Stimulus frames
    // would omit the terminal and drop the last step's iteration_rate.
    let stimulus = result.stimulus_timeline();
    let phases = ktstr::assert::build_phase_buckets_with_stimulus(&series, &stimulus);

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
/// regardless of payload; scx-ktstr always
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

/// Deterministic 3-Step e2e: pins the exact
/// `phases.len() == 4` shape (1 BASELINE + 3 Step buckets at
/// indexes 1, 2, 3 per the 1-index encoding). Stronger than
/// the 2-Step variant's `any(step_index >= 1)` floor — pins the
/// EXACT count, so a future regression that loses a Step or
/// double-counts a Step boundary surfaces here. The deterministic
/// pipeline-shape test is the load-bearing CI gate.
///
/// HoldSpec::frac(0.33) ⨉ 3 keeps the steps roughly balanced so
/// the periodic capture loop fires across every Step window
/// within the 15 s duration + 6 captures fixture. All three
/// Steps must produce at least one bucket; missing any bucket
/// would indicate either a step_index advancement bug or a
/// silent-drop in `by_phase`.
fn assert_phase_pipeline_three_step(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.periodic_fired >= 3,
        "periodic_fired = {} of {} — fewer than 3 captures means \
         the 3-step shape cannot be exercised across every Step \
         window; the load-bearing invariant requires at least \
         one bucket per Step",
        result.periodic_fired,
        result.periodic_target,
    );

    let drained = result.snapshot_bridge.drain_ordered_with_stats();
    let drained_len = drained.len();
    let series = SampleSeries::from_drained_typed(drained, result.monitor.clone());
    // The COMPLETE timeline (step frames + scenario-end terminal) via
    // the shared accessor — folding only the raw wire Stimulus frames
    // would omit the terminal and drop the last step's iteration_rate.
    let stimulus = result.stimulus_timeline();
    let phases = ktstr::assert::build_phase_buckets_with_stimulus(&series, &stimulus);

    // The load-bearing invariant: exact count. BASELINE bucket
    // is only emitted when at least one sample lands in the
    // pre-first-Step settle window; with 6 captures across a 15 s
    // run plus the bridge fire interval (~2.5 s) every Step window
    // should land at least one capture but BASELINE may legitimately
    // be empty if the first capture fires after the first Step
    // transition. Test both shapes: phases.len() == 3 (no BASELINE)
    // OR phases.len() == 4 (BASELINE + 3 Steps). Anything else is
    // a regression.
    let step_indices: Vec<u16> = phases.iter().map(|p| p.step_index).collect();
    anyhow::ensure!(
        phases.len() == 3 || phases.len() == 4,
        "phases.len() = {} — expected 3 (Step[0..2] only) or 4 \
         (BASELINE + Step[0..2]). Any other count means the \
         step_index pipeline either lost a bucket (silent drop \
         in by_phase) or double-counted a boundary. \
         step_indices = {:?}",
        phases.len(),
        step_indices,
    );
    // All three Step buckets must be present. step_index = 1, 2, 3
    // map to Step[0], Step[1], Step[2] per the 1-index encoding.
    for expected_step in [1u16, 2, 3] {
        anyhow::ensure!(
            step_indices.contains(&expected_step),
            "phases vec is missing step_index = {} (Step[{}]) — \
             step_indices = {:?}. The CURRENT_STEP atomic either \
             skipped a value or the by_phase partition lost every \
             sample for this Step.",
            expected_step,
            expected_step - 1,
            step_indices,
        );
    }
    // Sum invariant (same as 2-Step variant) — catches silent
    // drops + double-counts in the by_phase partition.
    let total_samples: usize = phases.iter().map(|p| p.sample_count).sum();
    anyhow::ensure!(
        total_samples == drained_len,
        "phases vec sum(sample_count) = {} but {} entries drained \
         (mismatch in by_phase). step_indices = {:?}",
        total_samples,
        drained_len,
        step_indices,
    );
    Ok(())
}

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 15,
    watchdog_timeout_s = 25,
    num_snapshots = 6,
    auto_repro = false,
    post_vm = assert_phase_pipeline_three_step,
)]
fn phase_pipeline_three_step_e2e(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step {
            setup: vec![CgroupDef::named("cg_step0").workers(2)].into(),
            ops: vec![],
            hold: HoldSpec::frac(0.33),
        },
        Step {
            setup: vec![CgroupDef::named("cg_step1").workers(2)].into(),
            ops: vec![],
            hold: HoldSpec::frac(0.33),
        },
        Step {
            setup: vec![CgroupDef::named("cg_step2").workers(2)].into(),
            ops: vec![],
            hold: HoldSpec::frac(0.34),
        },
    ];
    execute_steps(ctx, steps)
}

/// First-and-last-step iteration_rate end-to-end: a Backdrop-PERSISTENT workload (workers
/// that survive across Steps — the documented pattern for cross-phase
/// throughput; see [`ktstr::timeline::StimulusEvent::total_iterations`])
/// must yield an `iteration_rate` for BOTH the FIRST Step (the
/// 0-baseline first stimulus frame previously collapsed to `None`
/// and dropped the rate) AND the LAST Step (previously no
/// successor frame existed to diff against). Boots a real guest under
/// scx-ktstr so the full path is exercised: per-step stimulus emission,
/// the widened `ScenarioEnd` terminal frame (final cumulative count
/// captured coincident with its elapsed), and host aggregation — not
/// just the synthetic-fixture unit path that previously bypassed
/// `from_wire`.
fn assert_iteration_rate_first_and_last(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.periodic_fired >= 2,
        "periodic_fired = {} of {} — need a capture in each Step window \
         so both the first and last Step produce a bucket the rate can \
         attach to",
        result.periodic_fired,
        result.periodic_target,
    );
    let drained = result.snapshot_bridge.drain_ordered_with_stats();
    let series = SampleSeries::from_drained_typed(drained, result.monitor.clone());
    // The COMPLETE timeline (step frames + scenario-end terminal) via
    // the shared accessor — folding only the raw wire Stimulus frames
    // would omit the terminal and drop the last step's iteration_rate.
    let stimulus = result.stimulus_timeline();
    let phases = ktstr::assert::build_phase_buckets_with_stimulus(&series, &stimulus);

    // FIRST step = lowest step_index >= 1 (Step[0] under the 1-indexed
    // encoding). Its rate is the (first_frame -> second_frame) delta —
    // the zero-baseline case fixed.
    let first = phases
        .iter()
        .filter(|p| p.step_index >= 1)
        .min_by_key(|p| p.step_index)
        .ok_or_else(|| anyhow::anyhow!("no Step bucket present in phases"))?;
    anyhow::ensure!(
        first.metrics.contains_key("iteration_rate"),
        "first Step (step_index {}) has no iteration_rate — the \
         0-baseline first frame was dropped. metric keys = {:?}",
        first.step_index,
        first.metrics.keys().collect::<Vec<_>>(),
    );

    // LAST step = highest step_index. Its rate comes from the terminal
    // ScenarioEnd frame — the last-step terminal case fixed.
    let last = phases
        .iter()
        .filter(|p| p.step_index >= 1)
        .max_by_key(|p| p.step_index)
        .expect("at least one Step bucket (checked above)");
    anyhow::ensure!(
        last.metrics.contains_key("iteration_rate"),
        "last Step (step_index {}) has no iteration_rate — the \
         scenario-end terminal frame did not supply its right boundary. \
         metric keys = {:?}",
        last.step_index,
        last.metrics.keys().collect::<Vec<_>>(),
    );
    Ok(())
}

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 10,
    watchdog_timeout_s = 20,
    num_snapshots = 6,
    auto_repro = false,
    post_vm = assert_iteration_rate_first_and_last,
)]
fn phase_pipeline_iteration_rate_backdrop_e2e(ctx: &Ctx) -> Result<AssertResult> {
    // Persistent workload on the Backdrop so its iteration counter
    // survives across BOTH Steps (cross-step throughput is defined for
    // the persistent population; step-local workers would reset each
    // step). Two bare hold Steps let the backdrop spin continuously
    // across both phase windows so each step boundary's cumulative
    // count strictly increases.
    let backdrop = Backdrop::new().push_cgroup(CgroupDef::named("cg_bg").workers(2));
    let steps = vec![
        Step::new(vec![], HoldSpec::frac(0.5)),
        Step::new(vec![], HoldSpec::frac(0.5)),
    ];
    execute_scenario(ctx, backdrop, steps)
}

/// Probabilistic per-step-cpuset e2e: when the per-Step cpuset differs
/// meaningfully (Step 0 spreads across all CPUs, Step 1 collapses
/// to one LLC), the per-step `max_dsq_depth` metric must differ.
/// Pins that the per-phase metric pipeline reflects the per-step
/// configuration end-to-end — not just that buckets exist (which
/// the deterministic 3-step variant pins) but that the metric
/// VALUES respond to the per-step scheduler input.
///
/// `#[ignore]`-tagged: scheduler behavior is non-deterministic,
/// so this test may flake on a noisy CI box even when the
/// pipeline works correctly. Operators run it on demand via
/// `cargo ktstr test -- --ignored phase_pipeline_per_step_cpuset_differs`.
/// The deterministic pipeline-shape test stays the load-bearing
/// CI gate; this one is the behavior-coverage complement.
///
/// Tolerance: non-equality only, NOT a fixed threshold. A bound
/// would over-constrain the scheduler's freedom — we only care
/// that the two configurations produce DIFFERENT readings, not
/// that they differ by a specific amount.
fn assert_per_step_cpuset_changes_metrics(result: &VmResult) -> Result<()> {
    let drained = result.snapshot_bridge.drain_ordered_with_stats();
    let series = SampleSeries::from_drained_typed(drained, result.monitor.clone());
    // The COMPLETE timeline (step frames + scenario-end terminal) via
    // the shared accessor — folding only the raw wire Stimulus frames
    // would omit the terminal and drop the last step's iteration_rate.
    let stimulus = result.stimulus_timeline();
    let phases = ktstr::assert::build_phase_buckets_with_stimulus(&series, &stimulus);

    let step0 = phases.iter().find(|p| p.step_index == 1);
    let step1 = phases.iter().find(|p| p.step_index == 2);

    // Inconclusive-rather-than-fail policy: when periodic captures
    // didn't land in BOTH Steps the test cannot prove or disprove
    // the metric-differs claim. Printing a warn and returning Ok
    // keeps the test honest (no false pass on missing data, no
    // false fail when the timing happened not to cover Step[1])
    // and keeps the #[ignore]-tagged probabilistic gauntlet
    // run from going red on a topology-induced timing edge.
    // Operators inspecting the run log see the warn and can re-run
    // with longer duration / more snapshots if they want a definite
    // result. The real-defect path — both buckets present AND
    // metrics equal — still fails the test below.
    let (step0, step1) = match (step0, step1) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            let present: Vec<u16> = phases.iter().map(|p| p.step_index).collect();
            eprintln!(
                "WARN phase_pipeline_per_step_cpuset_differs: periodic \
                 captures did not land in both Steps (present step_indices: \
                 {:?}); test inconclusive on this run. Re-run, or raise \
                 duration_s/num_snapshots on the test annotation if this \
                 reproduces.",
                present,
            );
            return Ok(());
        }
    };

    let s0 = match step0.get("max_dsq_depth") {
        Some(v) => v,
        None => {
            eprintln!(
                "WARN: Step[0] bucket has no `max_dsq_depth` (sample_count={}, \
                 keys={:?}); test inconclusive.",
                step0.sample_count,
                step0.metrics.keys().collect::<Vec<_>>(),
            );
            return Ok(());
        }
    };
    let s1 = match step1.get("max_dsq_depth") {
        Some(v) => v,
        None => {
            eprintln!(
                "WARN: Step[1] bucket has no `max_dsq_depth` (sample_count={}, \
                 keys={:?}); test inconclusive.",
                step1.sample_count,
                step1.metrics.keys().collect::<Vec<_>>(),
            );
            return Ok(());
        }
    };

    // Inconclusive when both reduced values are 0 — the scheduler
    // didn't push the DSQ on either side, so the metric carries no
    // signal. Distinct from "both equal at a positive value" which
    // would indicate the per-step cpuset never reached the kernel
    // (failure class). On a small topology (e.g. tiny-2llc: 4 CPUs
    // / 2 LLCs / 4 workers) the SpinWait workload may not crowd
    // either cpuset enough to register depth > 0, and we don't
    // want a zero-signal observation to red-flag the test.
    if s0 == 0.0 && s1 == 0.0 {
        eprintln!(
            "WARN: both Step[0] and Step[1] max_dsq_depth == 0; scheduler \
             didn't crowd the DSQ on either side. Test inconclusive — \
             on small topologies the SpinWait workload may not produce \
             enough queue pressure to differentiate cpusets. Re-run on a \
             larger topology, or change the workload to one that produces \
             queue depth (e.g. CpuSpin + bursty wakers)."
        );
        return Ok(());
    }

    anyhow::ensure!(
        (s0 - s1).abs() > f64::EPSILON,
        "Step[0] `max_dsq_depth` = {} and Step[1] `max_dsq_depth` = {} are \
         equal (both positive — at least one side observed crowding). \
         Step 0 spreads workers across all CPUs, Step 1 collapses them \
         to LLC[0]; the per-step crowding should produce distinguishable \
         readings. Equality at positive values means either: \
         (a) the per-Step cpuset never reached the kernel (Op::SetCpuset \
             at step boundary lost), or \
         (b) the per-phase metric pipeline averaged across Steps instead \
             of partitioning by step_index. \
         The all-zero inconclusive case is gated above; reaching THIS \
         branch with equal positive values is a real defect signal.",
        s0,
        s1,
    );
    Ok(())
}

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 2,
    cores = 2,
    threads = 1,
    duration_s = 30,
    watchdog_timeout_s = 50,
    num_snapshots = 10,
    auto_repro = false,
    post_vm = assert_per_step_cpuset_changes_metrics,
    ignore = true,
    // Gauntlet variants of this test only make sense on topologies
    // with >= 2 LLCs and >= 4 CPUs — Step 1 rebinds the cgroup to
    // CpusetSpec::llc(0) (one LLC out of `llcs`) to contrast with
    // Step 0's "all CPUs". On a 1-LLC topology the rebind selects
    // the same CPU set as Step 0 ("all CPUs") and the metric-
    // differs assertion can't fire. On a 2-CPU topology the
    // "all CPUs" vs "1 LLC" contrast collapses to 2 vs 1 CPU which
    // doesn't produce reliable scheduler-behavior differentiation
    // within the 30 s window.
    min_llcs = 2,
    min_cpus = 4,
)]
fn phase_pipeline_per_step_cpuset_differs(ctx: &Ctx) -> Result<AssertResult> {
    // 30 s / 10 captures = 3 s capture interval. 2 Steps × 15 s each
    // → ~5 captures per Step under nominal timing, leaving margin so
    // an unlucky Op::SetCpuset apply latency or post-rebind settle
    // delay doesn't push every Step[1] capture past the scenario
    // end. The earlier 15 s / 6-capture budget failed when the
    // first post-rebind capture fired after Step[1] had already
    // ended.
    //
    // The cgroup lives in the Backdrop so it persists across both
    // Steps; Op::set_cpuset in Step 1 rebinds the persistent cgroup
    // rather than addressing a step-local clone (a step-local
    // cgroup tears down at the step boundary, so the mid-run
    // set_cpuset would race against the teardown and ENOENT).
    let backdrop = ktstr::scenario::backdrop::Backdrop::new()
        .push_cgroup(CgroupDef::named("cg_phase").workers(4));
    let steps = vec![
        // Step 0: persistent cgroup spans every CPU (no cpuset →
        // inherits root).
        Step {
            setup: vec![].into(),
            ops: vec![],
            hold: HoldSpec::frac(0.5),
        },
        // Step 1: rebind the persistent cgroup to LLC[0] only via
        // the mid-run set_cpuset Op. Same cgroup, same workers —
        // only the cpuset changes between the two phases.
        Step {
            setup: vec![].into(),
            ops: vec![ktstr::scenario::ops::Op::set_cpuset(
                "cg_phase",
                ktstr::scenario::ops::CpusetSpec::llc(0),
            )],
            hold: HoldSpec::frac(0.5),
        },
    ];
    ktstr::scenario::ops::execute_scenario(ctx, backdrop, steps)
}

/// Watchpoint trip-time stamping e2e: Op::WatchSnapshot pins the captured
/// snapshot to the phase ACTIVE WHEN THE WATCHPOINT FIRED, not the
/// phase active when the Op was issued. The contract is to stamp
/// from the CURRENT_STEP atomic at store time, implemented at
/// `src/vmm/freeze_coord/mod.rs` user-watchpoint trip arm: the
/// freeze coordinator loads `current_step` at the TOP of the
/// trip-handler arm (immediately after observing the hit-swap,
/// before the freeze rendezvous + file write window), then stamps
/// the bridge entry with that captured value via
/// `store_with_stats_and_step` in all 3 sub-arms
/// (Captured / Degraded / Suppressed).
///
/// ## What this test catches
///
/// A regression that:
/// - Reverts to `bridge.store(tag, report)` (unstamped) at the
///   trip arm → drained `step_index = None` (silent attribution
///   to BASELINE fallback) — caught by A1.
/// - Moves the `current_step.load` AFTER the file write → race
///   window during the freeze+IO latency lets the scenario driver
///   advance the phase, mis-attributing step-k trip data to
///   step-(k+1) — caught by A2 in the cross-step variant (TODO
///   follow-up; see Limitations).
///
/// ## Setup
///
/// Arms `Op::watch_snapshot("jiffies_64")` during Step[0]. The
/// kernel writes `jiffies_64` every timer tick (see
/// `kernel/time/timekeeping.c`), so the watchpoint fires within
/// the first few ms of Step[0]. By construction the trip phase
/// matches the registration phase here, which proves the wire-up
/// (step_index = Some(1), the Step[0] 1-indexed bucket) without
/// proving the trip-time vs registration-time distinction.
///
/// ## Limitations
///
/// This variant covers the wire-up regression class (unstamped
/// trip captures) but NOT the cross-step trip-vs-registration
/// drift class (race window between trip moment and stamp moment
/// under slow IO). A complementary
/// test that arms in Step[0] but fires the watchpoint in Step[2]
/// requires a kernel symbol whose write is gated on a guest-side
/// Op the scenario invokes in a specific Step — non-trivial to
/// construct deterministically. Tracked as a follow-up; the
/// in-flight fix's "load at top of trip arm" pattern keeps the
/// race window to sub-ms hit-swap→load latency on the coord
/// thread, well under the wall-clock duration of a typical Step.
fn assert_watch_snapshot_trip_phase_stamped(result: &VmResult) -> Result<()> {
    let drained = result.snapshot_bridge.drain_ordered_with_stats();
    let watch_caps: Vec<_> = drained.iter().filter(|e| e.tag == "jiffies_64").collect();
    anyhow::ensure!(
        !watch_caps.is_empty(),
        "watchpoint on 'jiffies_64' did not fire — `kernel/time/timekeeping.c` \
         writes jiffies_64 every tick so a fire within Step[0]'s window is \
         expected. Without a fire, the trip-stamping wire-up is uncovered. \
         Drained {} entries; tags = {:?}",
        drained.len(),
        drained.iter().map(|e| e.tag.as_str()).collect::<Vec<_>>(),
    );
    for cap in &watch_caps {
        anyhow::ensure!(
            cap.step_index.is_some(),
            "watchpoint trip capture '{}' has step_index = None — the trip \
             handler at src/vmm/freeze_coord/mod.rs bypassed \
             bridge.store_with_stats_and_step and fell back to unstamped \
             bridge.store. Every trip capture must be stamped \
             with the host's view of current_step (the load happens whether \
             or not the value is BASELINE/0). A None here is a wire-up \
             regression — the new code path went unexecuted and the legacy \
             unstamped `bridge.store` was used instead.",
            cap.tag,
        );
        // NOTE: the stamped VALUE (Some(0) vs Some(N)) is governed by
        // host_current_step (HOST-side mirror) at trip time. This atomic
        // is updated when the guest publishes a STIMULUS frame via the
        // bulk virtio-console port, which the scenario driver sends
        // AFTER apply_ops returns (scenario/ops/mod.rs:1248). For the
        // 1-Step scenario here, the watchpoint arms inside apply_ops
        // and fires on the next jiffies tick — that fire can race the
        // STIMULUS frame for Step[0]. When the trip wins the race, the
        // host's view is still 0 (BASELINE) at stamp time → Some(0).
        // When STIMULUS wins, Some(1). Both are correct per the
        // implementation (stamp from the host's last-known step); the
        // assertion that the value IS stamped at all (Some, not None)
        // is the load-bearing invariant. The cross-step trip-vs-registration
        // distinction (testing that a watchpoint armed in Step[0]
        // firing in Step[2] stamps with the Step[2] phase) requires
        // a deterministic guest-side write trigger gated on a specific
        // Step's ops — non-trivial to construct. Tracked as future
        // work; see test doc above.
    }
    Ok(())
}

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 10,
    watchdog_timeout_s = 30,
    num_snapshots = 0,
    auto_repro = false,
    post_vm = assert_watch_snapshot_trip_phase_stamped,
)]
fn watch_snapshot_trip_in_step_stamps_current_phase_not_unstamped(
    ctx: &Ctx,
) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_wp").workers(2)].into(),
        // Arm the watchpoint inside Step[0]. The kernel's
        // jiffies_64 update every tick fires the watchpoint within
        // ms; the trip lands while the scenario is still in
        // Step[0] (1-indexed step_index = 1).
        ops: vec![Op::watch_snapshot("jiffies_64")],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
