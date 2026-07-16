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

/// Stimulus iteration deltas must not resurrect the former wall-denominated
/// rate, even when interior captures and step indices provide enough data to
/// attribute those deltas to a phase. The canonical rate additionally needs
/// the matching guest CPU-time carrier.
#[test]
fn build_phase_buckets_with_stimulus_interior_windows_do_not_create_wall_rate() {
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
    // Step 1's stimulus delta is 1000 iterations over 1 wall second.
    // It must remain absent because no matching guest CPU-time carrier exists.
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        None,
        "stimulus wall time must not produce the CPU-denominated iteration_rate; \
         got {:?} (start_ms={}, end_ms={})",
        step1.metrics.get("iteration_rate"),
        step1.start_ms,
        step1.end_ms,
    );
    // The same holds for Step 2's 2000-iteration wall delta.
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("Step[1] bucket present");
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        None,
        "stimulus wall time must not produce iteration_rate; got {:?}",
        step2.metrics.get("iteration_rate"),
    );
}

/// Stimulus boundary deltas must not create `iteration_rate`. Respawned
/// workers make wall-clock boundary arithmetic especially misleading; the
/// canonical rate requires matching guest iteration and CPU-time carriers.
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
        None,
        "step 1 has no delivered-CPU carrier, so iteration_rate stays absent; got {:?}",
        step1.metrics.get("iteration_rate"),
    );
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        None,
        "step 2 has no delivered-CPU carrier, so iteration_rate stays absent; got {:?}",
        step2.metrics.get("iteration_rate"),
    );
}

/// Persistent workers may advance during inter-step teardown. None of those
/// wall-clock boundary deltas may become `iteration_rate`; only guest carrier
/// CPU accounting is authoritative.
#[test]
fn build_phase_buckets_with_stimulus_never_uses_persistent_cross_step_wall_delta() {
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
    // The event stream contains both large step-local deltas and an inter-step
    // teardown delta. None may become the CPU-denominated metric.
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
        None,
        "stimulus deltas must not synthesize the CPU-denominated rate; got {:?}",
        step1.metrics.get("iteration_rate"),
    );
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        None,
        "stimulus deltas must not synthesize the CPU-denominated rate; got {:?}",
        step2.metrics.get("iteration_rate"),
    );
}

/// A stalled wall-clock step and a busy teardown gap still provide no
/// delivered-CPU rate without a guest carrier.
#[test]
fn build_phase_buckets_with_stimulus_stalled_wall_step_has_no_cpu_rate() {
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
    // Step 1 is stalled, then a persistent population advances during the
    // teardown gap. Neither the flat hold nor the inter-step wall delta can
    // produce a delivered-CPU rate without a carrier.
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
        None,
        "a wall-only stalled step has no delivered-CPU rate; got {:?}",
        step1.metrics.get("iteration_rate"),
    );
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        None,
        "a wall-only step has no delivered-CPU rate; got {:?}",
        step2.metrics.get("iteration_rate"),
    );
}

/// A wall-clock stimulus delta must emit neither the former wall-rate
/// components nor `iteration_rate`.
#[test]
fn build_phase_buckets_with_stimulus_does_not_emit_wall_rate_components() {
    use crate::timeline::StimulusEvent;
    // No capture samples for step 1: the synthesized-bucket seam creates it,
    // but the wall-only stimulus stream contributes no rate components.
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
        None,
        "wall duration is no longer an iteration-rate component; got {:?}",
        step1.metrics.get("total_phase_duration_sec"),
    );
    assert_eq!(
        step1.metrics.get("total_phase_iterations").copied(),
        None,
        "stimulus iteration deltas are no longer metric components",
    );
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        None,
        "no guest CPU-time carrier means no iteration_rate; got {:?}",
        step1.metrics.get("iteration_rate"),
    );
}

/// A terminal scenario event must not seed a phantom bucket or synthesize a
/// wall-denominated rate for the last step.
#[test]
fn build_phase_buckets_with_stimulus_terminal_adds_no_phantom_or_wall_rate() {
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
    // The terminal carries step_index None + is_terminal, so it seeds no
    // bucket and its wall delta is not a metric source.
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
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("step 2 bucket present");
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        None,
        "terminal wall deltas must not produce a delivered-CPU rate",
    );
}

/// A first-step zero iteration sentinel traversing the full wire path still
/// must not create a CPU rate without delivered-CPU evidence.
#[test]
fn build_phase_buckets_with_stimulus_wire_zero_does_not_create_cpu_rate() {
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
        None,
        "wire iteration counts without CPU time must not produce iteration_rate",
    );
}

/// Loop-step boundary events also remain wall-only and cannot graft a rate
/// onto either the loop or its preceding step.
#[test]
fn build_phase_buckets_with_stimulus_loop_events_do_not_create_cpu_rate() {
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
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        None,
        "prior step has no CPU-time carrier",
    );
    assert_eq!(
        loop_step.metrics.get("iteration_rate").copied(),
        None,
        "loop step has no CPU-time carrier",
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

/// Run-level double-source guard: populate_run_ext_metrics_from_phases
/// must SKIP avg_nr_running. Its authoritative run-level value is
/// MonitorSummary::avg_nr_running (fold_run_level_ext); fold_monitor_into_bucket
/// also writes it per-phase for rendering, so without the skip the per-phase
/// re-pool would claim the run-level key — and in VmResult::run_metric the
/// re-pool runs BEFORE fold_run_level_ext, whose `or_insert` would then no-op,
/// silently replacing the whole-run value with the per-phase mean. The ext-only
/// avg_imbalance_ratio must still fold (control).
#[test]
fn populate_run_ext_metrics_from_phases_skips_avg_nr_running() {
    use crate::assert::PhaseBucket;
    use std::collections::BTreeMap;
    let mut m = BTreeMap::new();
    m.insert("avg_imbalance_ratio".to_string(), 2.0); // ext-only -> folded
    m.insert("avg_nr_running".to_string(), 5.0); // MonitorSummary-fold authority -> skipped
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
        !target.contains_key("avg_nr_running"),
        "avg_nr_running run-level value comes from fold_run_level_ext; the per-phase \
         re-pool must NOT claim it (the fold's or_insert would no-op in \
         VmResult::run_metric, replacing the whole-run value with the per-phase mean)",
    );
}

/// A synthesized zero-capture phase can still carry guest iteration and CPU
/// components into the run aggregate. The rate re-pools as
/// Σiterations/ΣCPU-seconds; capture count is irrelevant to guest CPU
/// accounting. Unequal denominators distinguish this from averaging ratios or
/// dropping the zero-capture carrier.
#[test]
fn populate_run_ext_metrics_repools_synthesized_zero_capture_phase() {
    use crate::assert::PhaseBucket;
    use std::collections::BTreeMap;
    // Captured phase: 1200 iters over 3s = 400/s.
    let cap = BTreeMap::from([
        ("total_iterations_pooled".to_string(), 1200.0),
        ("total_cpu_time_sec".to_string(), 3.0),
    ]);
    // Synthesized zero-capture phase: 600 iters over 1s = 600/s.
    let synth = BTreeMap::from([
        ("total_iterations_pooled".to_string(), 600.0),
        ("total_cpu_time_sec".to_string(), 1.0),
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
        target.get("total_iterations_pooled").copied(),
        Some(1800.0),
        "components sum across phases (weights ignored for Counters)",
    );
}
