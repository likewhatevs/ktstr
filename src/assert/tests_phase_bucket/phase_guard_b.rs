use super::*;

/// Hostile input: a corrupt StepEnd[k] BEFORE its StepStart[k] yields an
/// inverted synthesized window (end_ms < start_ms). fold_monitor_into_bucket
/// folds nothing (the half-open `m >= start_ms && m < end_ms` is vacuously
/// false) and the bucket is present without panic — no over-fold, no crash.
#[test]
fn build_phase_buckets_with_stimulus_synthesized_inverted_window_folds_nothing() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    use crate::scenario::snapshot::DrainedSnapshotEntry;
    use crate::timeline::StimulusEvent;
    let cpu = |nr: u32| CpuSnapshot {
        nr_running: nr,
        ..Default::default()
    };
    let mon = MonitorReport {
        samples: vec![MonitorSample::new(1500, vec![cpu(6), cpu(2)])],
        ..Default::default()
    };
    let samples = SampleSeries::from_drained_typed(Vec::<DrainedSnapshotEntry>::new(), Some(mon));
    let ev = |elapsed_ms: u64, k: u16, is_step_end: bool| StimulusEvent {
        elapsed_ms,
        label: format!("Step{}[{k}]", if is_step_end { "End" } else { "Start" }),
        op_kind: None,
        detail: None,
        total_iterations: Some(0),
        step_index: Some(k),
        is_terminal: false,
        is_step_end,
    };
    // StepEnd[1] at 1000 is BEFORE StepStart[1] at 2000 (corrupt wire).
    let stimulus = vec![ev(2000, 1, false), ev(1000, 1, true)];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("synthesized step 1 bucket present despite inverted window");
    assert!(
        step1.start_ms > step1.end_ms,
        "window is inverted: start {} end {}",
        step1.start_ms,
        step1.end_ms,
    );
    assert!(
        !step1.metrics.contains_key("avg_imbalance_ratio"),
        "an inverted window folds no monitor samples; got {:?}",
        step1.metrics,
    );
}

/// Math-protocol boundary coverage for the synthesize end_ms terminal-clamp
/// arm: a last step with no StepEnd and no successor but a ScenarioEnd
/// terminal present clamps the window to the terminal (not u64::MAX), so a
/// monitor sample AFTER the terminal is excluded from the fold (the
/// pass-4 over-fold guard).
#[test]
fn build_phase_buckets_with_stimulus_synthesized_end_ms_clamps_to_terminal() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    use crate::scenario::snapshot::DrainedSnapshotEntry;
    use crate::timeline::StimulusEvent;
    let cpu = |nr: u32| CpuSnapshot {
        nr_running: nr,
        ..Default::default()
    };
    // s1 @1000 (anchor -> monitor_clock_offset 0), imbalance 2/2 = 1.0,
    // IN [1000, 3000). s2 @4000 (AFTER the terminal 3000), imbalance
    // 10/2 = 5.0, must be EXCLUDED by the terminal clamp.
    let mon = MonitorReport {
        samples: vec![
            MonitorSample::new(1000, vec![cpu(2), cpu(2)]),
            MonitorSample::new(4000, vec![cpu(10), cpu(2)]),
        ],
        ..Default::default()
    };
    let samples = SampleSeries::from_drained_typed(Vec::<DrainedSnapshotEntry>::new(), Some(mon));
    let start = StimulusEvent {
        elapsed_ms: 1000,
        label: "StepStart[1]".to_string(),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(1),
        is_terminal: false,
        is_step_end: false,
    };
    let terminal = StimulusEvent {
        elapsed_ms: 3000,
        label: "ScenarioEnd".to_string(),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: None,
        is_terminal: true,
        is_step_end: false,
    };
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &[start, terminal]);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("synthesized step 1 bucket present");
    assert_eq!(
        step1.end_ms, 3000,
        "end_ms clamps to the terminal (3000), not u64::MAX",
    );
    assert_eq!(
        step1.metrics.get("avg_imbalance_ratio").copied(),
        Some(1.0),
        "only the in-window pre-terminal sample folds (imbalance 1.0); the \
         post-terminal sample (5.0) is excluded — avg is 1.0, not 3.0",
    );
}

/// The public SampleSeries grouping methods: by_stamped_phase
/// COLLAPSES a deferred-fire burst (every capture stamped the same late
/// step) into one phase, while by_stimulus_phase re-derives the correct
/// per-phase grouping from each sample's timing-independent
/// boundary_offset_ms. Pins both new public entry points and the
/// difference that motivates by_stimulus_phase.
#[test]
fn by_stimulus_phase_separates_what_by_stamped_phase_collapses() {
    use crate::scenario::sample::SampleSeries;
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    // All four captures stamp the SAME late step (the burst) but their
    // SCHEDULED offsets fall in distinct step windows.
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(3),
    };
    let drained = vec![
        mk("p0", 500),   // before step 1 start (1000) -> BASELINE (0)
        mk("p1", 1_500), // step 1 window
        mk("p2", 2_500), // step 2 window
        mk("p3", 3_500), // step 3 window
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let stim = |elapsed_ms: u64, k: u16| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let stimulus = vec![stim(1000, 1), stim(2000, 2), stim(3000, 3)];

    // by_stamped_phase: all four collapse into the single stamped key 3.
    let stamped = samples.by_stamped_phase();
    assert_eq!(stamped.keys().copied().collect::<Vec<_>>(), vec![3]);
    assert_eq!(stamped[&3].len(), 4, "stamped grouping collapses the burst");

    // by_stimulus_phase: re-derived from boundary_offset -> 4 phases.
    let by_stim = samples.by_stimulus_phase(&stimulus);
    assert_eq!(
        by_stim.keys().copied().collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    for k in [0u16, 1, 2, 3] {
        assert_eq!(
            by_stim[&k].len(),
            1,
            "phase {k} should hold exactly one sample"
        );
    }
}

/// Fallback + synthesize interaction: a capture with no
/// `boundary_offset_ms` (on-demand / fixture) keeps its stamped
/// step_index even when a stimulus timeline is present — the offset
/// remap only overrides captures that carry a scheduled offset, so
/// legacy / non-periodic entries are untouched. The step-5 StepStart
/// does NOT pull the None-offset captures to step 5 (they stay at their
/// stamped 1 / 2); it does, per the synthesize seam, produce its OWN
/// capture-free step-5 bucket — a StepStart marks a step that ran, so
/// every StepStart-step gets a bucket even with zero captures.
#[test]
fn build_phase_buckets_with_stimulus_none_offset_falls_back_to_stamped_step() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, step: u16| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(100),
        boundary_offset_ms: None,
        step_index: Some(step),
    };
    let drained = vec![mk("periodic_000", 1), mk("periodic_001", 2)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    // A step-5 stimulus is present: it must NOT pull the None-offset
    // captures to step 5 (they keep their stamped 1 / 2), but it DOES
    // synthesize its own capture-free step-5 bucket.
    let stimulus = vec![StimulusEvent {
        elapsed_ms: 0,
        label: "StepStart[5]".to_string(),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(5),
        is_terminal: false,
        is_step_end: false,
    }];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let idxs: Vec<u16> = phases.iter().map(|p| p.step_index).collect();
    assert_eq!(
        idxs,
        vec![1, 2, 5],
        "None-offset captures keep their stamped step_index (1, 2), NOT \
         remapped to the step-5 stimulus; step 5's StepStart synthesizes its \
         own capture-free bucket; got {idxs:?}",
    );
    // The captures stayed in buckets 1/2 (one each) — NOT collapsed into a
    // single step-5 bucket. Step 5 is the synthesized, capture-free one.
    let count = |k: u16| {
        phases
            .iter()
            .find(|p| p.step_index == k)
            .map(|p| p.sample_count)
    };
    assert_eq!(
        count(1),
        Some(1),
        "capture stamped 1 stays in its own bucket"
    );
    assert_eq!(
        count(2),
        Some(1),
        "capture stamped 2 stays in its own bucket"
    );
    assert_eq!(
        count(5),
        Some(0),
        "step 5 is the synthesized capture-free bucket"
    );
}

/// Iteration-rate attribution regression: in the production shape (periodic captures carry
/// workload-relative boundary offsets in the step INTERIOR + stimulus
/// events carry step_index), iteration_rate must attach to the step the
/// rate was measured DURING — by step_index, NOT by a timestamp-window
/// match against the capture-derived (interior) bucket window. The
/// step-START (prev.elapsed_ms) falls in the inter-step gap, inside no
/// interior bucket window, so the old window match dropped every rate.
/// This pins the step_index attribution; it FAILS on the window match.
#[test]
fn build_phase_buckets_with_stimulus_iteration_rate_attaches_by_step_not_interior_window() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    // One capture per step, each at the step INTERIOR (offset strictly
    // greater than the step-start), mirroring compute_periodic_boundaries_ns
    // (10-90% interior). Stamped step_index intentionally wrong (all 9) so
    // the OFFSET drives grouping and the stimulus event's step_index drives
    // rate attribution.
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    let drained = vec![
        mk("periodic_000", 1_500), // step 1 interior (window [1000,2000))
        mk("periodic_001", 2_500), // step 2 interior (window [2000,3000))
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    // Step-starts at 1000/2000/3000; cumulative iterations 0/1000/3000.
    let stim = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let stimulus = vec![stim(1000, 1, 0), stim(2000, 2, 1000), stim(3000, 3, 3000)];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    // Step 1 bucket (step_index==1): rate for (step1@1000 -> step2@2000),
    // iters 0->1000 over 1s = 1000/s. Its window is the single interior
    // capture [1500,1500], which does NOT contain the step-start 1000 —
    // the old window match dropped this rate.
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(1000.0),
        "step 1 iteration_rate must attach by step_index to the interior \
         bucket; got {:?} (start_ms={}, end_ms={})",
        step1.metrics.get("iteration_rate"),
        step1.start_ms,
        step1.end_ms,
    );
    // Step 2 bucket: rate for (step2@2000 -> step3@3000), iters
    // 1000->3000 over 1s = 2000/s.
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("Step[1] bucket present");
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        Some(2000.0),
        "step 2 iteration_rate must attach by step_index; got {:?}",
        step2.metrics.get("iteration_rate"),
    );
}

/// Each step's `iteration_rate` is the STEP-LOCAL
/// `StepStart[k]` -> `StepEnd[k]` delta (its OWN workers, start-to-end of
/// hold), NOT the cross-step `StepStart[k]` -> `StepStart[k+1]` delta.
/// Workers respawned per step read ~0 at every StepStart, so the
/// cross-step delta is `0 - 0` and yields no rate (the old cross-step bug:
/// every fresh-per-step phase silently had no throughput). Pairing each
/// step's own StepStart -> StepEnd recovers the real per-step rate. The
/// elapsed-sorted `windows(2)` walk pairs `StepStart[k]` -> `StepEnd[k]`
/// first (both carry step_index `k`); `or_insert` keeps that step-local
/// rate, and the intervening `StepEnd[k]` -> `StepStart[k+1]` pair reads
/// `0 <= end` so `rate_to` returns None and never overwrites.
#[test]
fn build_phase_buckets_with_stimulus_pairs_step_local_for_respawned_workers() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    let drained = vec![
        mk("periodic_000", 1_500), // step 1 interior (window [1000,2100))
        mk("periodic_001", 2_500), // step 2 interior (window [2100,..))
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let start = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let end = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepEnd[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: true,
    };
    // Step 1: 0 -> 10000 over its 1s hold (10000/s). Step 2 workers are
    // RESPAWNED, so StepStart[2] reads 0 again; step 2: 0 -> 5000 over 1s
    // (5000/s). 1000ms windows keep the division exact (1.0s denominator).
    let stimulus = vec![
        start(1_000, 1, 0),
        end(2_000, 1, 10_000),
        start(2_100, 2, 0),
        end(3_100, 2, 5_000),
        StimulusEvent::terminal(3_200, 5_000),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket present");
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("step 2 bucket present");
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(10_000.0),
        "step 1 must report its step-local StepStart->StepEnd rate, not the \
         cross-step 0->0 delta (the old cross-step silent None); got {:?}",
        step1.metrics.get("iteration_rate"),
    );
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        Some(5_000.0),
        "step 2 (respawned workers) must report its step-local rate; got {:?}",
        step2.metrics.get("iteration_rate"),
    );
}

/// A PERSISTENT (Backdrop) population keeps iterating through
/// the inter-step teardown, so `StepStart[k+1]` reads MORE than
/// `StepEnd[k]` and the cross-step `StepEnd[k]` -> `StepStart[k+1]` pair
/// WOULD yield a rate. But that pair has an `is_step_end` `prev`, so the
/// attribution loop's guard skips it before `rate_components`/`or_insert` ever
/// run — the cross-step rate is never even computed, and each step
/// reports only its own step-local `StepStart[k]` -> `StepEnd[k]`
/// throughput. (`or_insert` is a redundant secondary safety here, not
/// the operative mechanism — the guard is.)
#[test]
fn build_phase_buckets_with_stimulus_step_local_wins_over_persistent_cross_step() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    let drained = vec![mk("periodic_000", 1_500), mk("periodic_001", 2_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let start = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let end = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepEnd[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: true,
    };
    // Step 1 local: 0 -> 10000 over 1s (10000/s). Persistent workers add
    // 500 in the 100ms teardown gap, so StepStart[2] reads 10500 — the
    // cross-step StepEnd[1] -> StepStart[2] pair would compute 500/0.1s =
    // 5000/s, but its is_step_end prev is skipped by the guard so that
    // rate is never computed for step 1.
    // Step 2 local: 10500 -> 15500 over 1s (5000/s).
    let stimulus = vec![
        start(1_000, 1, 0),
        end(2_000, 1, 10_000),
        start(2_100, 2, 10_500),
        end(3_100, 2, 15_500),
        StimulusEvent::terminal(3_200, 15_500),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket present");
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("step 2 bucket present");
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(10_000.0),
        "step-local rate must win over the 5000/s persistent cross-step \
         delta (the is_step_end guard skips the cross-step pair); got {:?}",
        step1.metrics.get("iteration_rate"),
    );
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        Some(5_000.0),
        "step 2 must report its own start-to-end-of-hold rate; got {:?}",
        step2.metrics.get("iteration_rate"),
    );
}

/// A STALLED step (its own `StepStart[k] -> StepEnd[k]` delta is zero)
/// must report its MEASURED-ZERO rate `Some(0.0)` — it must NOT leak the
/// inter-step teardown-gap rate that the `StepEnd[k] -> StepStart[k+1]`
/// pair would otherwise produce. StepEnd[k] carries step_index `k`, so
/// without the `is_step_end` guard in the attribution loop that cross-step
/// pair (prev = StepEnd[k], also step_index `k`) would `or_insert` a gap
/// rate into bucket `k`. This pins the guard: bucket `k` is sourced ONLY
/// by its own StepStart -> StepEnd pair (which here is a measured zero).
#[test]
fn build_phase_buckets_with_stimulus_stalled_step_reports_measured_zero() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    let drained = vec![mk("periodic_000", 1_500), mk("periodic_001", 2_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let start = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let end = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepEnd[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: true,
    };
    // Step 1 STALLED: StepStart[1] == StepEnd[1] == 0, so its step-local
    // pair is rate_components (0.0, secs), derived iteration_rate 0.0 —
    // measured zero throughput. A persistent
    // population then advances 500 during the 100ms teardown gap, so the
    // cross-step StepEnd[1](0) -> StepStart[2](500) pair WOULD compute
    // 500/0.1s = 5000/s — which must NOT land in bucket 1 (it is
    // guard-skipped). Step 2 runs normally: 500 -> 1500 over 1s = 1000/s.
    let stimulus = vec![
        start(1_000, 1, 0),
        end(2_000, 1, 0),
        start(2_100, 2, 500),
        end(3_100, 2, 1_500),
        StimulusEvent::terminal(3_200, 1_500),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket present");
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("step 2 bucket present");
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(0.0),
        "a stalled step reports measured-zero throughput, not the leaked \
         5000/s teardown gap rate from the StepEnd[1] -> StepStart[2] pair; \
         got {:?}",
        step1.metrics.get("iteration_rate"),
    );
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        Some(1_000.0),
        "step 2 still reports its own step-local rate; got {:?}",
        step2.metrics.get("iteration_rate"),
    );
}

/// Pins the ms→s unit conversion + end-to-end component emission for the
/// iteration_rate Rate: a step over a 2000ms window with a 1000-iteration
/// delta must emit total_phase_duration_sec == 2.0 (NOT 2000 — the /1000
/// lives in the component because derive_rate_metrics does a bare num/den)
/// and total_phase_iterations == 1000, so the re-derived iteration_rate is
/// 1000 / 2.0 = 500/s. A regression leaving the denominator in ms would
/// derive 1000/2000 = 0.5 and pass every type/naming gate silently.
#[test]
fn build_phase_buckets_with_stimulus_emits_rate_components_in_seconds() {
    use crate::timeline::StimulusEvent;
    // No capture samples for step 1 → the synthesized-bucket seam creates
    // it and the attribution loop fills the components from the stimulus.
    let samples = SampleSeries::from_drained_typed(vec![], None);
    let stimulus = vec![
        StimulusEvent {
            elapsed_ms: 1_000,
            label: "StepStart[1]".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: Some(0),
            step_index: Some(1),
            is_terminal: false,
            is_step_end: false,
        },
        StimulusEvent {
            elapsed_ms: 3_000, // 2000ms = 2.0s window
            label: "StepEnd[1]".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: Some(1_000), // 1000-iteration delta
            step_index: Some(1),
            is_terminal: false,
            is_step_end: true,
        },
        StimulusEvent::terminal(3_100, 1_000),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket synthesized from stimulus");
    assert_eq!(
        step1.metrics.get("total_phase_duration_sec").copied(),
        Some(2.0),
        "duration component is SECONDS (2000ms / 1000), not ms; got {:?}",
        step1.metrics.get("total_phase_duration_sec"),
    );
    assert_eq!(
        step1.metrics.get("total_phase_iterations").copied(),
        Some(1_000.0),
        "iteration component is the 1000-iteration delta",
    );
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(500.0),
        "derived iteration_rate = 1000 iters / 2.0 s = 500/s (NOT 0.5 from a \
         ms denominator); got {:?}",
        step1.metrics.get("iteration_rate"),
    );
}

/// The LAST step has no successor step event, so its
/// iteration_rate needs the terminal scenario-end boundary. The
/// terminal supplies that boundary (last step's rate = delta to the
/// terminal count) and must NOT seed a phantom bucket — the bucket set
/// stays equal to the sample-derived phases.
#[test]
fn build_phase_buckets_with_stimulus_terminal_gives_last_step_rate_no_phantom_bucket() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    // Two captures: step 1 interior, step 2 interior.
    let drained = vec![mk("periodic_000", 1_500), mk("periodic_001", 2_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let stim = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    // Step starts at 1000/2000 (iters 0/1000); terminal at 3000 with
    // final iters 3000 — the right boundary the LAST step (step 2)
    // needs. The terminal carries step_index None + is_terminal so it
    // seeds no bucket.
    let stimulus = vec![
        stim(1000, 1, 0),
        stim(2000, 2, 1000),
        StimulusEvent::terminal(3000, 3000),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let idxs: Vec<u16> = phases.iter().map(|p| p.step_index).collect();
    assert_eq!(
        idxs,
        vec![1, 2],
        "terminal must not add a phantom bucket; got {idxs:?}",
    );
    // Step 2 is the LAST step: its rate comes from the terminal,
    // 1000 -> 3000 over 1s = 2000/s. Without the terminal it would be
    // None (the bug this fixes).
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("step 2 bucket present");
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        Some(2000.0),
        "last step's iteration_rate must come from the terminal boundary",
    );
}

/// First-step zero-baseline rate, through the production aggregator + the FULL from_wire
/// path (unit tests previously injected Some(0) directly, masking the
/// wire 0->None collapse). The first step frame reads 0 cumulative
/// iterations; after dropping the sentinel the first step's bucket gets
/// a rate.
#[test]
fn build_phase_buckets_with_stimulus_first_step_zero_baseline_from_wire() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    let drained = vec![mk("periodic_000", 1_500), mk("periodic_001", 2_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let wire = |elapsed_ms: u32, step_index: u16, iters: u64| crate::vmm::wire::StimulusEvent {
        elapsed_ms,
        step_index,
        op_count: 0,
        op_kinds: 0,
        cgroup_count: 0,
        worker_count: 1,
        total_iterations: iters,
    };
    // First step frame reads 0 cumulative iters (workers just spawned).
    let stimulus: Vec<StimulusEvent> = [wire(1000, 1, 0), wire(2000, 2, 2000)]
        .iter()
        .map(StimulusEvent::from_wire)
        .collect();
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket present");
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(2000.0),
        "first step's iteration_rate must compute from the 0 baseline",
    );
}

/// Loop-hold attribution: once the guest emits a start frame
/// for a Loop step (the run_step Loop arm), a scenario ending on a Loop
/// step must attribute the final window's throughput to the LOOP step,
/// NOT graft it onto the prior step. This pins the host-side contract
/// the guest fix relies on: with the loop step's own start frame
/// present, (loop_start -> terminal) lands on the loop step's bucket and
/// (prior_start -> loop_start) lands on the prior step's bucket.
#[test]
fn build_phase_buckets_with_stimulus_loop_step_rate_no_prior_graft() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    // Prior step (step 1) interior + Loop step (step 2) interior.
    let drained = vec![mk("periodic_000", 1_500), mk("periodic_001", 2_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let stim = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    // step1@1000 (iters 0), loop step2@2000 (iters 1000, its OWN start
    // frame — the loop-hold attribution fix), terminal@3000 (iters 3000).
    let stimulus = vec![
        stim(1000, 1, 0),
        stim(2000, 2, 1000),
        StimulusEvent::terminal(3000, 3000),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket");
    let loop_step = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("loop step bucket");
    // Prior step gets ONLY its own window (0 -> 1000 over 1s = 1000/s),
    // NOT the loop window grafted on.
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(1000.0),
        "prior step must not absorb the loop step's window",
    );
    // Loop step gets the (loop_start -> terminal) rate: 1000 -> 3000
    // over 1s = 2000/s.
    assert_eq!(
        loop_step.metrics.get("iteration_rate").copied(),
        Some(2000.0),
        "loop step must get its own throughput from its start frame + terminal",
    );
}

/// `populate_run_ext_metrics` on a populated series produces
/// the expected entries. Without this, the empty test and the
/// no-overwrite test pass vacuously and the load-bearing happy
/// path is uncovered.
#[test]
fn populate_run_ext_metrics_populated_series_inserts_expected_keys() {
    use crate::monitor::dump::FailureDumpReport;
    use crate::monitor::scx_walker::DsqState;
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    let mk_entry = |tag: &str, ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            dsq_states: vec![DsqState {
                origin: "local cpu 0".to_string(),
                nr: 5,
                ..Default::default()
            }],
            ..Default::default()
        },
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(ms),
        boundary_offset_ms: None,
        step_index: Some(0),
    };
    let drained = vec![mk_entry("periodic_000", 100), mk_entry("periodic_001", 200)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let mut target = std::collections::BTreeMap::new();
    crate::assert::populate_run_ext_metrics(&samples, &mut target);
    // avg_dsq_depth has no typed GauntletRow field → populated.
    // mean of (5, 5) = 5.0.
    let avg = target
        .get("avg_dsq_depth")
        .copied()
        .expect("avg_dsq_depth populated for populated series");
    assert!(
        (avg - 5.0).abs() < f64::EPSILON,
        "expected avg_dsq_depth=5.0, got {avg}",
    );
    // max_dsq_depth has a typed field → skipped by populate.
    assert!(
        !target.contains_key("max_dsq_depth"),
        "max_dsq_depth has a typed GauntletRow field; must not leak into ext_metrics",
    );
}

/// populate_run_ext_metrics_from_phases folds per-phase metrics that
/// have no read_sample dispatch (e.g. avg_imbalance_ratio) by a
/// Gauge(Avg) weighted mean across phases. (iteration_rate, also without
/// read_sample dispatch, is a MetricKind::Rate — re-pooled from its
/// summed components, not weighted-mean folded.)
#[test]
fn populate_run_ext_metrics_from_phases_folds_per_phase_keys() {
    use crate::assert::PhaseBucket;
    use std::collections::BTreeMap;
    let mut m0 = BTreeMap::new();
    m0.insert("avg_imbalance_ratio".to_string(), 2.0);
    let mut m1 = BTreeMap::new();
    m1.insert("avg_imbalance_ratio".to_string(), 4.0);
    let phases = vec![
        PhaseBucket {
            per_cgroup: Default::default(),
            step_index: 1,
            label: "Step[0]".to_string(),
            start_ms: 0,
            end_ms: 100,
            sample_count: 5,
            metrics: m0,
        },
        PhaseBucket {
            per_cgroup: Default::default(),
            step_index: 2,
            label: "Step[1]".to_string(),
            start_ms: 100,
            end_ms: 200,
            sample_count: 15,
            metrics: m1,
        },
    ];
    let mut target = BTreeMap::new();
    crate::assert::populate_run_ext_metrics_from_phases(&phases, &mut target);
    // avg_imbalance_ratio is Gauge(Avg) — weighted mean by
    // sample_count: (2.0*5 + 4.0*15) / 20 = 70/20 = 3.5.
    let avg = target
        .get("avg_imbalance_ratio")
        .copied()
        .expect("avg_imbalance_ratio folded from per-phase");
    assert!(
        (avg - 3.5).abs() < f64::EPSILON,
        "expected weighted mean 3.5, got {avg}",
    );
}

/// PerPhaseDeltaSum (`system_time_ns` / `user_time_ns`) folds cross-PHASE by
/// SUM, NOT the weighted mean a Gauge(Avg) uses: the phases partition the run
/// timeline, so the run-level total is the sum of the disjoint per-phase
/// CPU-time deltas. Distinct sample_counts (5, 15) make the three candidate
/// folds numerically distinct — SUM = 8000, weighted-mean =
/// (3000*5+5000*15)/20 = 4500, unweighted-mean = 4000 — so this pins SUM
/// specifically (a regression to the old Gauge(Avg) weighted mean would read
/// 4500, and routing through aggregate_finite's cross-RUN arm would read 4000).
#[test]
fn populate_run_ext_metrics_from_phases_sums_per_phase_delta_kind() {
    use crate::assert::PhaseBucket;
    use std::collections::BTreeMap;
    let mut m0 = BTreeMap::new();
    m0.insert("system_time_ns".to_string(), 3000.0);
    let mut m1 = BTreeMap::new();
    m1.insert("system_time_ns".to_string(), 5000.0);
    let phases = vec![
        PhaseBucket {
            per_cgroup: Default::default(),
            step_index: 1,
            label: "Step[0]".to_string(),
            start_ms: 0,
            end_ms: 100,
            sample_count: 5,
            metrics: m0,
        },
        PhaseBucket {
            per_cgroup: Default::default(),
            step_index: 2,
            label: "Step[1]".to_string(),
            start_ms: 100,
            end_ms: 200,
            sample_count: 15,
            metrics: m1,
        },
    ];
    let mut target = BTreeMap::new();
    crate::assert::populate_run_ext_metrics_from_phases(&phases, &mut target);
    let sys = target
        .get("system_time_ns")
        .copied()
        .expect("system_time_ns folded from per-phase");
    assert!(
        (sys - 8000.0).abs() < f64::EPSILON,
        "PerPhaseDeltaSum must SUM disjoint per-phase deltas (3000+5000=8000), \
         not weighted-mean (4500) or unweighted-mean (4000); got {sys}",
    );
}

/// Run-level guard: populate_run_ext_metrics_from_phases must SKIP keys with
/// a typed GauntletRow field (TYPED_FIELD_NAMES) so the phase fold never
/// re-injects them into ext_metrics. The monitor fold writes max_imbalance_ratio +
/// stuck_count onto CAPTURED buckets; both are typed-backed (their accessor
/// wins on read), so writing them to ext would be unread bloat AND, for
/// stuck_count, a redundant-or-divergent value: the ext per-phase fold sum
/// is `<=` the typed whole-run count (equal when no dropped window is stuck;
/// strictly below otherwise — they share the is_cpu_stuck predicate but the
/// run-level count windows the full sample stream). avg_imbalance_ratio
/// (genuinely ext-only) must still fold.
#[test]
fn populate_run_ext_metrics_from_phases_skips_typed_backed_keys() {
    use crate::assert::PhaseBucket;
    use std::collections::BTreeMap;
    let mut m = BTreeMap::new();
    m.insert("avg_imbalance_ratio".to_string(), 2.0); // ext-only -> folded
    m.insert("max_imbalance_ratio".to_string(), 3.0); // typed-backed -> skipped
    m.insert("stuck_count".to_string(), 2.0); // typed-backed -> skipped
    let phases = vec![PhaseBucket {
        per_cgroup: Default::default(),
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 5,
        metrics: m,
    }];
    let mut target = BTreeMap::new();
    crate::assert::populate_run_ext_metrics_from_phases(&phases, &mut target);
    assert!(
        target.contains_key("avg_imbalance_ratio"),
        "avg_imbalance_ratio is ext-only and must be folded into ext_metrics",
    );
    assert!(
        !target.contains_key("max_imbalance_ratio"),
        "max_imbalance_ratio is typed-backed; must NOT leak into ext_metrics from the phase fold",
    );
    assert!(
        !target.contains_key("stuck_count"),
        "stuck_count is typed-backed; must NOT leak into ext_metrics (the ext per-phase fold sum is <= the typed whole-run count, never a guaranteed duplicate)",
    );
}

/// A synthesized zero-capture phase (sample_count==0) still folds into
/// the run aggregate — its capture-independent iteration_rate is
/// INCLUDED, not dropped. iteration_rate is now a MetricKind::Rate, so
/// inclusion is via its Counter components (total_phase_iterations /
/// total_phase_duration_sec) SUMMING across phases (weights ignored — see
/// aggregate_finite) and the rate re-deriving as Σiters/Σseconds. A
/// regression dropping the synthesized phase would silently re-drop its
/// iterations from the sidecar aggregate: the run-level variant of the
/// synthesized-bucket bug. Unequal phase durations make the re-pool (450) distinct from both
/// a mean-of-ratios (500) and a dropped-synthesized result (400).
#[test]
fn populate_run_ext_metrics_repools_synthesized_zero_capture_phase() {
    use crate::assert::PhaseBucket;
    use std::collections::BTreeMap;
    // Captured phase: 1200 iters over 3s = 400/s.
    let cap = BTreeMap::from([
        ("total_phase_iterations".to_string(), 1200.0),
        ("total_phase_duration_sec".to_string(), 3.0),
    ]);
    // Synthesized zero-capture phase: 600 iters over 1s = 600/s.
    let synth = BTreeMap::from([
        ("total_phase_iterations".to_string(), 600.0),
        ("total_phase_duration_sec".to_string(), 1.0),
    ]);
    let phases = vec![
        PhaseBucket {
            per_cgroup: Default::default(),
            step_index: 1,
            label: "Step[0]".to_string(),
            start_ms: 0,
            end_ms: 100,
            sample_count: 5,
            metrics: cap,
        },
        PhaseBucket {
            per_cgroup: Default::default(),
            step_index: 2,
            label: "Step[1]".to_string(),
            start_ms: 100,
            end_ms: 200,
            sample_count: 0, // synthesized zero-capture step
            metrics: synth,
        },
    ];
    let mut target = BTreeMap::new();
    crate::assert::populate_run_ext_metrics_from_phases(&phases, &mut target);
    // Re-pool: Σiters / Σseconds = (1200 + 600) / (3 + 1) = 1800/4 = 450/s.
    // The synthesized phase's 600 iters are SUMMED in (Counters ignore the
    // sample_count weight), so 450 — NOT 400 (synthesized dropped, 1200/3)
    // and NOT 500 (mean of the two ready ratios 400 and 600).
    let r = target
        .get("iteration_rate")
        .copied()
        .expect("synthesized zero-capture phase's components must re-pool into iteration_rate");
    assert!(
        (r - 450.0).abs() < f64::EPSILON,
        "expected re-pooled 450/s (synthesized 600 iters summed in), NOT \
         400 (dropped) or 500 (mean of ratios); got {r}",
    );
    // The summed components survive for any further re-derivation.
    assert_eq!(
        target.get("total_phase_iterations").copied(),
        Some(1800.0),
        "components sum across phases (weights ignored for Counters)",
    );
}
