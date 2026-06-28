use super::*;

// ---------- PhaseGuard (RAII auto-stamp) ----------

#[test]
fn phase_guard_outside_scope_returns_none() {
    // No guard installed → current_phase_label is None and a
    // freshly-constructed AssertDetail inherits None.
    assert!(crate::assert::current_phase_label().is_none());
    let d = crate::assert::AssertDetail::new(crate::assert::DetailKind::Other, "no guard");
    assert!(
        d.phase.is_none(),
        "AssertDetail constructed outside any PhaseGuard must stamp phase=None"
    );
}

#[test]
fn phase_guard_install_step_sets_active_label() {
    let _g = crate::assert::PhaseGuard::install_step(0);
    assert_eq!(
        crate::assert::current_phase_label().as_deref(),
        Some("Step[0]"),
    );
    let d = crate::assert::AssertDetail::new(crate::assert::DetailKind::Other, "under Step[0]");
    assert_eq!(d.phase.as_deref(), Some("Step[0]"));
}

#[test]
fn phase_guard_install_baseline_sets_active_label() {
    let _g = crate::assert::PhaseGuard::install_baseline();
    assert_eq!(
        crate::assert::current_phase_label().as_deref(),
        Some("BASELINE"),
    );
}

#[test]
fn phase_guard_drop_restores_prior_label() {
    {
        let _outer = crate::assert::PhaseGuard::install_step(0); // "Step[0]"
        assert_eq!(
            crate::assert::current_phase_label().as_deref(),
            Some("Step[0]"),
        );
        {
            let _inner = crate::assert::PhaseGuard::install_step(2); // "Step[2]"
            assert_eq!(
                crate::assert::current_phase_label().as_deref(),
                Some("Step[2]"),
            );
        } // inner drops → restore Step[0]
        assert_eq!(
            crate::assert::current_phase_label().as_deref(),
            Some("Step[0]"),
            "inner guard's Drop must restore the outer guard's label",
        );
    } // outer drops → restore None
    assert!(
        crate::assert::current_phase_label().is_none(),
        "outermost guard's Drop must restore None",
    );
}

#[test]
fn phase_guard_passdetail_binary_auto_stamps() {
    let _g = crate::assert::PhaseGuard::install_step(1);
    let p = crate::assert::PassDetail::binary("metric", "ge", "10.0", "5.0");
    assert_eq!(p.phase.as_deref(), Some("Step[1]"));
}

#[test]
fn phase_guard_passdetail_unary_auto_stamps() {
    let _g = crate::assert::PhaseGuard::install_step(2);
    let p = crate::assert::PassDetail::unary("metric", "is_finite", "42.0");
    assert_eq!(p.phase.as_deref(), Some("Step[2]"));
}

#[test]
fn phase_guard_infonote_auto_stamps() {
    let _g = crate::assert::PhaseGuard::install_baseline();
    let n = crate::assert::InfoNote::new("settle observed");
    assert_eq!(n.phase.as_deref(), Some("BASELINE"));
}

#[test]
fn phase_guard_with_phase_builder_overrides_auto_stamp() {
    let _g = crate::assert::PhaseGuard::install_step(0); // "Step[0]"
    let d = crate::assert::AssertDetail::new(crate::assert::DetailKind::Other, "override")
        .with_phase("explicit_override");
    assert_eq!(
        d.phase.as_deref(),
        Some("explicit_override"),
        "with_phase builder must override the auto-stamp default",
    );
}

/// `populate_run_ext_metrics` is a no-op for an empty SampleSeries:
/// `read_sample` returns `None` for every registered metric on the
/// empty fixture (no DSQ states, no event counters), so nothing
/// lands in `ext_metrics`. Pins the contract that the helper
/// never synthesises sentinel zeros from no-data input.
#[test]
fn populate_run_ext_metrics_empty_series_inserts_nothing() {
    let samples = SampleSeries::from_drained_typed(Vec::new(), None);
    let mut target = std::collections::BTreeMap::new();
    crate::assert::populate_run_ext_metrics(&samples, &mut target);
    assert!(
        target.is_empty(),
        "no input samples must produce no ext_metrics entries, got {target:?}",
    );
}

/// `populate_run_ext_metrics` never overwrites a key already
/// present in `target` — a typed GauntletRow field that produced
/// a value via the MetricDef accessor stays untouched. Pins the
/// "fill the gap, never clobber" contract: cross-RUN comparison
/// expects the typed-field value when present and the
/// helper-computed value only when not.
#[test]
fn populate_run_ext_metrics_does_not_overwrite_existing_keys() {
    let samples = SampleSeries::from_drained_typed(Vec::new(), None);
    let mut target = std::collections::BTreeMap::new();
    target.insert("avg_dsq_depth".to_string(), 42.0);
    crate::assert::populate_run_ext_metrics(&samples, &mut target);
    assert_eq!(
        target.get("avg_dsq_depth").copied(),
        Some(42.0),
        "existing key must survive populate_run_ext_metrics",
    );
}

/// `build_phase_buckets` populates `avg_imbalance_ratio` from
/// MonitorSamples windowed by phase. Synthesised samples land in
/// the Step[0] window; the per-phase mean of their
/// `imbalance_ratio()` readings (max(nr_running)/max(1, min(nr_running))
/// per CPU) is stamped on PhaseBucket.metrics. Confirms the F-A
/// fix wiring: imbalance now flows through PhaseBucket per-phase
/// rather than only at the run-aggregate MonitorSummary level.
#[test]
fn build_phase_buckets_avg_imbalance_ratio_from_monitor_samples() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    // Three monitor samples covering [50..250 ms]. Sample CPU vecs
    // produce known imbalance ratios:
    //   s_50:  cpus=[nr=2, nr=2] -> ratio = 2 / max(1, 2) = 1.0
    //   s_100: cpus=[nr=4, nr=2] -> ratio = 4 / max(1, 2) = 2.0
    //   s_200: cpus=[nr=6, nr=2] -> ratio = 6 / max(1, 2) = 3.0
    // Mean across all three = (1.0 + 2.0 + 3.0) / 3 = 2.0
    let cpu = |nr: u32| CpuSnapshot {
        nr_running: nr,
        ..Default::default()
    };
    let mon = MonitorReport {
        samples: vec![
            MonitorSample::new(50, vec![cpu(2), cpu(2)]),
            MonitorSample::new(100, vec![cpu(4), cpu(2)]),
            MonitorSample::new(200, vec![cpu(6), cpu(2)]),
        ],
        ..Default::default()
    };
    // Two snapshot bridge entries fence the Step[0] window at
    // elapsed_ms [50..250]; all three monitor samples land inside.
    let drained = vec![
        fixture_entry("periodic_000", 1, 50),
        fixture_entry("periodic_001", 1, 250),
    ];
    let samples = SampleSeries::from_drained_typed(drained, Some(mon));
    let phases = crate::assert::build_phase_buckets(&samples);
    assert_eq!(phases.len(), 1, "single phase from two same-step samples");
    let step0 = &phases[0];
    let avg = step0
        .metrics
        .get("avg_imbalance_ratio")
        .copied()
        .expect("avg_imbalance_ratio must be populated from MonitorSamples");
    assert!(
        (avg - 2.0).abs() < f64::EPSILON,
        "expected mean = 2.0, got {avg}",
    );
}

/// MonitorSamples whose elapsed_ms falls OUTSIDE the phase window
/// (`[start_ms, end_ms]`) are excluded from the avg_imbalance_ratio
/// reduction. A sample at elapsed_ms = 9999 with a wildly
/// different imbalance must not contaminate the in-window mean.
#[test]
fn build_phase_buckets_avg_imbalance_excludes_out_of_window_monitor_samples() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    let cpu = |nr: u32| CpuSnapshot {
        nr_running: nr,
        ..Default::default()
    };
    let mon = MonitorReport {
        samples: vec![
            MonitorSample::new(100, vec![cpu(4), cpu(2)]),
            MonitorSample::new(150, vec![cpu(4), cpu(2)]),
            MonitorSample::new(200, vec![cpu(4), cpu(2)]),
            MonitorSample::new(9999, vec![cpu(100), cpu(2)]),
        ],
        ..Default::default()
    };
    let drained = vec![
        fixture_entry("periodic_000", 1, 100),
        fixture_entry("periodic_001", 1, 200),
    ];
    let samples = SampleSeries::from_drained_typed(drained, Some(mon));
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = &phases[0];
    let avg = step0
        .metrics
        .get("avg_imbalance_ratio")
        .copied()
        .expect("avg_imbalance_ratio populated");
    assert!(
        (avg - 2.0).abs() < f64::EPSILON,
        "out-of-window sample must not contaminate in-window mean (got {avg})",
    );
    // max_imbalance_ratio now folds on captured buckets too, so the
    // elapsed=9999 outlier (imbalance 100/2 = 50) is out-of-window and must
    // be excluded, so the per-phase Peak is the in-window max (all 2.0), NOT
    // 50. Pins that the windowing gate applies to the newly-folded key.
    let max_imb = step0
        .metrics
        .get("max_imbalance_ratio")
        .copied()
        .expect("max_imbalance_ratio populated on captured bucket");
    assert!(
        (max_imb - 2.0).abs() < f64::EPSILON,
        "out-of-window outlier (50) must be excluded from max_imbalance_ratio (got {max_imb})",
    );
}

/// A CAPTURED bucket (`sample_count > 0`) must carry
/// `max_imbalance_ratio` (Peak) and `stuck_count` (Counter) folded from its
/// in-window monitor samples. Neither has a `read_sample` dispatch arm (both
/// fall to `_ => None`), so before the gate fix they surfaced ONLY on
/// synthesized buckets and a captured (common-case) phase dropped them.
/// max_imbalance = max sample ratio; stuck_count = consecutive
/// frozen-`rq_clock` non-idle CPU stalls.
#[test]
fn build_phase_buckets_captured_bucket_carries_max_imbalance_and_stuck() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    // cpu(nr, clk): nr_running + a frozen rq_clock so two consecutive
    // samples sharing the SAME non-zero clock register a per-CPU stall.
    let cpu = |nr: u32, clk: u64| CpuSnapshot {
        nr_running: nr,
        rq_clock: clk,
        ..Default::default()
    };
    // s_60:  [nr=4, nr=2] -> imbalance 4 / max(1,2) = 2.0; rq_clock 1000
    // s_120: [nr=6, nr=2] -> imbalance 6 / max(1,2) = 3.0; rq_clock 1000
    //   frozen vs s_60 + both CPUs non-idle -> stall_count = 2.
    let mon = MonitorReport {
        samples: vec![
            MonitorSample::new(60, vec![cpu(4, 1000), cpu(2, 1000)]),
            MonitorSample::new(120, vec![cpu(6, 1000), cpu(2, 1000)]),
        ],
        ..Default::default()
    };
    // Two snapshot captures fence the Step[0] window at [50..250] and make
    // the bucket CAPTURED (sample_count == 2), not synthesized.
    let drained = vec![
        fixture_entry("periodic_000", 1, 50),
        fixture_entry("periodic_001", 1, 250),
    ];
    let samples = SampleSeries::from_drained_typed(drained, Some(mon));
    let phases = crate::assert::build_phase_buckets(&samples);
    assert_eq!(phases.len(), 1, "single phase");
    let step0 = &phases[0];
    assert!(
        step0.sample_count > 0,
        "bucket must be CAPTURED (not synthesized) for this test to mean anything",
    );
    let max_imb = step0
        .metrics
        .get("max_imbalance_ratio")
        .copied()
        .expect("captured bucket must now carry max_imbalance_ratio");
    assert!(
        (max_imb - 3.0).abs() < f64::EPSILON,
        "max imbalance = max(2.0, 3.0) = 3.0, got {max_imb}",
    );
    let stuck = step0
        .metrics
        .get("stuck_count")
        .copied()
        .expect("captured bucket must now carry stuck_count");
    assert!(
        (stuck - 2.0).abs() < f64::EPSILON,
        "two frozen-clock non-idle CPUs => stall_count 2, got {stuck}",
    );
}

/// Boundary: `stuck_count` needs `windows(2)` of monitor samples, so
/// a phase with a single in-window monitor sample has NO stall to report —
/// `stuck_count` must be ABSENT (not present as 0), while `max_imbalance_ratio`
/// (computed from the single sample) is still folded. Pins that the
/// `if pm.stall_count > 0` gate leaves the key out rather than writing a
/// misleading zero.
#[test]
fn build_phase_buckets_single_in_window_sample_has_no_stuck_count() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    let cpu = |nr: u32, clk: u64| CpuSnapshot {
        nr_running: nr,
        rq_clock: clk,
        ..Default::default()
    };
    // ONE in-window sample: imbalance 4/2 = 2.0; no second sample => no
    // windows(2) => stall_count 0 => stuck_count not written.
    let mon = MonitorReport {
        samples: vec![MonitorSample::new(100, vec![cpu(4, 1000), cpu(2, 1000)])],
        ..Default::default()
    };
    let drained = vec![
        fixture_entry("periodic_000", 1, 50),
        fixture_entry("periodic_001", 1, 250),
    ];
    let samples = SampleSeries::from_drained_typed(drained, Some(mon));
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = &phases[0];
    let max_imb = step0
        .metrics
        .get("max_imbalance_ratio")
        .copied()
        .expect("single in-window sample still yields max_imbalance_ratio");
    assert!(
        (max_imb - 2.0).abs() < f64::EPSILON,
        "max imbalance = 2.0 from the single sample, got {max_imb}",
    );
    assert!(
        !step0.metrics.contains_key("stuck_count"),
        "no consecutive-sample stall => stuck_count must be ABSENT, not 0: {:?}",
        step0.metrics.get("stuck_count"),
    );
}

/// Boundary: stuck_count scopes to in-window samples. The in_window
/// filter runs BEFORE compute_metrics' `windows(2)` stall detection, so a
/// stall pair never forms across the phase edge — an out-of-window sample
/// cannot pair with the last in-window sample. Two frozen-clock in-window
/// samples plus one far-out sample must yield stuck_count = 2 (the single
/// fully-in-window pair x 2 CPUs), NOT 4. Guards against a regression that
/// windowed AFTER stall computation and mis-attributed a cross-phase stall.
#[test]
fn build_phase_buckets_stuck_count_excludes_cross_window_stall_pair() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    let cpu = |nr: u32, clk: u64| CpuSnapshot {
        nr_running: nr,
        rq_clock: clk,
        ..Default::default()
    };
    // s_100 + s_200 are in [50,250); their frozen rq_clock pairs them ->
    // stall_count 2 (both CPUs). s_9999 is OUT of window; it is filtered
    // before compute_metrics, so the (s_200, s_9999) pair never forms. If
    // windowing ran AFTER stall detection, that pair would add 2 more.
    let mon = MonitorReport {
        samples: vec![
            MonitorSample::new(100, vec![cpu(4, 1000), cpu(2, 1000)]),
            MonitorSample::new(200, vec![cpu(6, 1000), cpu(2, 1000)]),
            MonitorSample::new(9999, vec![cpu(8, 1000), cpu(2, 1000)]),
        ],
        ..Default::default()
    };
    let drained = vec![
        fixture_entry("periodic_000", 1, 50),
        fixture_entry("periodic_001", 1, 250),
    ];
    let samples = SampleSeries::from_drained_typed(drained, Some(mon));
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = &phases[0];
    let stuck = step0
        .metrics
        .get("stuck_count")
        .copied()
        .expect("in-window stall pair yields stuck_count");
    assert!(
        (stuck - 2.0).abs() < f64::EPSILON,
        "only the fully-in-window pair counts => stuck_count 2 (not 4 with a cross-window pair), got {stuck}",
    );
}

/// Tester B14 BLOCKING: avg_dsq_depth end-to-end pin through
/// the registry → build_phase_buckets → PhaseBucket.metrics
/// path. Without this, a regression where the read_sample
/// dispatch arm at src/stats.rs returns None silently produces
/// an empty per-phase entry — operator-visible drop. Synthetic
/// Snapshot DSQ states produce a known mean across local-cpu
/// entries.
#[test]
fn build_phase_buckets_avg_dsq_depth_from_snapshot_dsq_states() {
    use crate::monitor::dump::FailureDumpReport;
    use crate::monitor::scx_walker::DsqState;
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    // Two periodic captures, each carrying 3 local-cpu DSQ
    // states with depths 2/4/6 → per-sample mean 4.0. Two
    // identical samples → per-phase mean 4.0.
    let mk_entry = |tag: &str, ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            dsq_states: vec![
                DsqState {
                    origin: "local cpu 0".to_string(),
                    nr: 2,
                    ..Default::default()
                },
                DsqState {
                    origin: "local cpu 1".to_string(),
                    nr: 4,
                    ..Default::default()
                },
                DsqState {
                    origin: "local cpu 2".to_string(),
                    nr: 6,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(ms),
        boundary_offset_ms: None,
        step_index: Some(1),
    };
    let drained = vec![mk_entry("periodic_000", 100), mk_entry("periodic_001", 200)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    let avg = step0
        .metrics
        .get("avg_dsq_depth")
        .copied()
        .expect("avg_dsq_depth populated from local-cpu DSQ states");
    assert!(
        (avg - 4.0).abs() < f64::EPSILON,
        "expected per-phase avg of mean(2,4,6)=4.0, got {avg}",
    );
    // Also verify max_dsq_depth shipped correctly through the
    // same DSQ-walker axis.
    let max = step0
        .metrics
        .get("max_dsq_depth")
        .copied()
        .expect("max_dsq_depth populated alongside avg");
    assert!(
        (max - 6.0).abs() < f64::EPSILON,
        "expected max=6.0, got {max}"
    );
}

/// Tester B15 BLOCKING: iteration_rate per-phase population via
/// build_phase_buckets_with_stimulus. Synthetic StimulusEvents
/// with total_iterations deltas at known boundaries produce a
/// known per-phase rate.
#[test]
fn build_phase_buckets_with_stimulus_populates_iteration_rate() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    // Snapshot bridge entries fence two Step windows: Step[0]
    // at [100, 1100], Step[1] at [1100, 2100]. Stimulus events
    // carry total_iterations at each boundary. iteration_rate
    // for Step[1] (curr.elapsed_ms=2100, prev.elapsed_ms=1100,
    // iter delta 2000) → 2000 / (1000ms/1000) = 2000.0/s.
    let mk_entry = |tag: &str, step: u16, ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(ms),
        boundary_offset_ms: None,
        step_index: Some(step),
    };
    let drained = vec![
        mk_entry("periodic_000", 1, 100),
        mk_entry("periodic_001", 1, 1100),
        mk_entry("periodic_002", 2, 1100),
        mk_entry("periodic_003", 2, 2100),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let stimulus = vec![
        StimulusEvent {
            elapsed_ms: 100,
            label: "Step[0]".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: Some(0),
            step_index: None,
            is_terminal: false,
            is_step_end: false,
        },
        StimulusEvent {
            elapsed_ms: 1100,
            label: "Step[1]".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: Some(1000),
            step_index: None,
            is_terminal: false,
            is_step_end: false,
        },
        StimulusEvent {
            elapsed_ms: 2100,
            label: "end".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: Some(3000),
            step_index: None,
            is_terminal: false,
            is_step_end: false,
        },
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("Step[1] bucket present");
    let rate = step1
        .metrics
        .get("iteration_rate")
        .copied()
        .expect("iteration_rate populated for Step[1]");
    assert!(
        (rate - 2000.0).abs() < f64::EPSILON,
        "expected iteration_rate=2000.0 iter/s, got {rate}",
    );
}

/// The deferred-fire fix. When the dump-prerequisite gate holds the
/// periodic boundaries until the accessor adopts, they fire in a burst
/// and every capture reads the same late `CURRENT_STEP`, so the stamped
/// step_index collapses to one value (the `phases.len() == 1` bug). The
/// workload-relative `boundary_offset_ms` — computed from the boundary
/// schedule, not the fire time — must instead drive attribution. Four
/// captures all stamped step_index=3 but scheduled across BASELINE +
/// three step windows must land in four distinct buckets.
#[test]
fn build_phase_buckets_with_stimulus_remaps_by_boundary_offset_over_stamped_step() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    // All four stamp step_index=3 (the burst bug) and share a ~uniform
    // run-relative fire time (elapsed_ms), but their SCHEDULED offsets
    // fall before step 1 (BASELINE) and inside steps 1, 2, 3.
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(3),
    };
    let drained = vec![
        mk("periodic_base", 500),  // before step 1 start (1000) -> BASELINE
        mk("periodic_000", 1_500), // step 1 window [1000, 2000)
        mk("periodic_001", 2_500), // step 2 window [2000, 3000)
        mk("periodic_002", 3_500), // step 3 window [3000, ..)
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    // Step-start timeline (scenario-relative): step k starts at k*1000 ms.
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
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let idxs: Vec<u16> = phases.iter().map(|p| p.step_index).collect();
    assert_eq!(
        idxs,
        vec![0, 1, 2, 3],
        "boundary_offset_ms must drive grouping (BASELINE + one capture \
         per step), NOT the uniformly-wrong stamped step_index=3 which \
         would collapse all four into a single bucket; got {idxs:?}",
    );
    for p in &phases {
        assert_eq!(
            p.sample_count, 1,
            "each remapped bucket holds exactly its one scheduled capture; \
             step_index={} count={}",
            p.step_index, p.sample_count,
        );
    }
}

/// Synthesized zero-capture residual: a scenario step whose periodic-boundary window
/// captured ZERO samples (the uniform whole-workload boundary placement
/// skipped it) still gets a PhaseBucket carrying its capture-independent
/// iteration_rate. Steps 1 and 3 capture; step 2 captures nothing. The
/// stimulus carries a StepStart per step with total_iterations, so step
/// 2's rate (StepStart[2]=1000 -> StepStart[3]=2000 over 1000 ms =
/// 1000/s) is measurable from the timeline alone. Before the synthesize
/// seam, step 2 produced no bucket and its rate was silently dropped.
/// Also pins dedup: a captured step is never given a duplicate bucket.
#[test]
fn build_phase_buckets_with_stimulus_synthesizes_zero_capture_step_bucket() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    // Captures: one in step 1's window, one in step 3's window. NONE in
    // step 2's window [2000, 3000).
    let cap = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(1),
    };
    let drained = vec![cap("periodic_000", 1_500), cap("periodic_001", 3_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    // StepStart[k] at k*1000 ms with cumulative iterations; the rate loop
    // pairs consecutive starts (delta iters / delta s).
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
    let stimulus = vec![
        start(1000, 1, 0),
        start(2000, 2, 1000),
        start(3000, 3, 2000),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);

    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("zero-capture step 2 must still produce a synthesized bucket");
    assert_eq!(step2.sample_count, 0, "synthesized bucket is capture-free");
    let rate = step2
        .metrics
        .get("iteration_rate")
        .copied()
        .expect("step 2's capture-independent iteration_rate must be recovered");
    assert!(
        (rate - 1000.0).abs() < f64::EPSILON,
        "step 2 rate = (2000-1000) iters / 1000 ms = 1000/s, got {rate}",
    );
    // Dedup: each captured step keeps a single bucket (no duplicate
    // synthesized bucket for steps 1 and 3) and the vec is step-sorted.
    let idxs: Vec<u16> = phases.iter().map(|p| p.step_index).collect();
    assert_eq!(
        idxs,
        vec![1, 2, 3],
        "captured steps 1/3 keep their single bucket; step 2 synthesized; \
         sorted by step_index; got {idxs:?}",
    );
}

/// Fuller synthesized-bucket variant: a synthesized zero-capture step bucket also
/// recovers its monitor-derived `avg_imbalance_ratio` from the
/// MonitorSamples in its window — not just iteration_rate. The anchor
/// monitor sample at the first stimulus elapsed pins
/// `monitor_clock_offset` to 0 so the in-step-2 sample lands in [2000,
/// 3000).
#[test]
fn build_phase_buckets_with_stimulus_synthesized_bucket_folds_monitor_imbalance() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let cpu = |nr: u32| CpuSnapshot {
        nr_running: nr,
        ..Default::default()
    };
    let mon = MonitorReport {
        samples: vec![
            // Anchor (>500 ms) at the first stimulus elapsed -> offset 0.
            MonitorSample::new(1000, vec![cpu(2), cpu(2)]), // step 1, imbalance 1.0
            MonitorSample::new(2500, vec![cpu(6), cpu(2)]), // step 2, imbalance 3.0
        ],
        ..Default::default()
    };
    // Captures in steps 1 and 3 only; step 2 captures nothing.
    let cap = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(1),
    };
    let drained = vec![cap("periodic_000", 1_500), cap("periodic_001", 3_500)];
    let samples = SampleSeries::from_drained_typed(drained, Some(mon));
    let start = |elapsed_ms: u64, k: u16| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let stimulus = vec![start(1000, 1), start(2000, 2), start(3000, 3)];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("synthesized zero-capture step 2 bucket present");
    assert_eq!(step2.sample_count, 0);
    let avg = step2
        .metrics
        .get("avg_imbalance_ratio")
        .copied()
        .expect("synthesized bucket must recover avg_imbalance_ratio from monitor samples");
    assert!(
        (avg - 3.0).abs() < f64::EPSILON,
        "step 2 in-window monitor imbalance = 6 / max(1, 2) = 3.0, got {avg}",
    );
    // The stimulus carried total_iterations: None, so no rate is fabricated.
    assert!(
        !step2.metrics.contains_key("iteration_rate"),
        "None total_iterations must yield NO iteration_rate (no fabrication); got {:?}",
        step2.metrics,
    );
}

/// A single-step run with a capture in its window produces exactly one
/// Step bucket and synthesizes nothing extra: the synthesize loop only
/// fires for a step that has a StepStart but no bucket.
#[test]
fn build_phase_buckets_with_stimulus_single_captured_step_no_spurious_synthesis() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let cap = DrainedSnapshotEntry {
        tag: "periodic_000".to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(1_500),
        step_index: Some(1),
    };
    let samples = SampleSeries::from_drained_typed(vec![cap], None);
    let stimulus = vec![StimulusEvent {
        elapsed_ms: 1000,
        label: "StepStart[1]".to_string(),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(1),
        is_terminal: false,
        is_step_end: false,
    }];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let idxs: Vec<u16> = phases.iter().map(|p| p.step_index).collect();
    assert_eq!(
        idxs,
        vec![1],
        "exactly the one captured step; the synthesize loop adds no extras",
    );
    assert_eq!(phases[0].sample_count, 1);
}

/// A sched-died last step (a StepStart with no StepEnd, no successor
/// start, and no captures) still produces a present-but-empty bucket:
/// open-ended window (u64::MAX), sample_count 0, and NO iteration_rate
/// (rate_to has no right boundary) — no panic, no phantom rate.
#[test]
fn build_phase_buckets_with_stimulus_sched_died_last_step_yields_empty_present_bucket() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let cap = DrainedSnapshotEntry {
        tag: "periodic_000".to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(1_500),
        step_index: Some(1),
    };
    let samples = SampleSeries::from_drained_typed(vec![cap], None);
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
    // step 1 captured; step 2 started then died (no StepEnd, no successor).
    let stimulus = vec![start(1000, 1, 0), start(2000, 2, 1000)];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("sched-died last step still gets a present bucket");
    assert_eq!(step2.sample_count, 0);
    assert_eq!(
        step2.end_ms,
        u64::MAX,
        "no StepEnd / successor start -> open-ended window",
    );
    assert!(
        !step2.metrics.contains_key("iteration_rate"),
        "no right boundary -> no rate (absent, not a phantom 0); got {:?}",
        step2.metrics,
    );
}

/// A synthesized (sample_count == 0) bucket between two captured phases:
/// its STIMULUS-DERIVED throughput (iteration_rate) is a real measurement
/// that survives the capture gap, so a collapse INTO it and a recovery
/// OUT of it ARE flagged. Its monitor-derived metrics
/// (avg_imbalance) come from a different sampling basis on a zero-sample
/// phase, so they stay SUPPRESSED — a wild synthesized imbalance never
/// flags a phantom change while a real throughput change does. Pins the
/// behavior the prior blanket sample_count gate got wrong.
#[test]
fn synthesized_zero_sample_bucket_flags_throughput_not_phantom_monitor() {
    use crate::timeline::{ChangeDirection, Timeline, TimelineContext};
    let captured = |k: u16, imbalance: f64, rate: f64| crate::assert::PhaseBucket {
        per_cgroup: Default::default(),
        step_index: k,
        label: format!("Step[{}]", k.saturating_sub(1)),
        start_ms: (k as u64) * 1000,
        end_ms: (k as u64) * 1000 + 1000,
        sample_count: 5,
        metrics: std::collections::BTreeMap::from([
            ("avg_imbalance_ratio".to_string(), imbalance),
            ("iteration_rate".to_string(), rate),
        ]),
    };
    // step 1 + step 3 steady & captured; step 2 synthesized with a WILD
    // imbalance (monitor — must stay gated) AND a real collapsed
    // stimulus-derived throughput (must flag).
    let buckets = vec![
        captured(1, 1.0, 1000.0),
        crate::assert::PhaseBucket {
            per_cgroup: Default::default(),
            step_index: 2,
            label: "Step[1]".to_string(),
            start_ms: 2000,
            end_ms: 3000,
            sample_count: 0,
            metrics: std::collections::BTreeMap::from([
                ("avg_imbalance_ratio".to_string(), 100.0),
                ("iteration_rate".to_string(), 10.0),
            ]),
        },
        captured(3, 1.0, 1000.0),
    ];
    let timeline = Timeline::from_phase_buckets(&buckets, &[], &TimelineContext::default());
    // No phase flags a monitor (imbalance) change — the wild synthesized
    // imbalance is gated off on both boundaries it touches.
    assert!(
        timeline
            .phases
            .iter()
            .all(|p| !p.changes.iter().any(|c| c.metric == "imbalance")),
        "a zero-sample bucket's wild imbalance must not flag a phantom change; got {:?}",
        timeline
            .phases
            .iter()
            .map(|p| (p.index, p.changes.clone()))
            .collect::<Vec<_>>(),
    );
    // Phase index 1 = boundary INTO the synthesized step: throughput
    // 1000 -> 10 is a real collapse, flagged Degraded. Pin the phase
    // index so a reorder can't mis-target the boundary while finding a
    // throughput change of the expected direction by coincidence.
    let into_synth = &timeline.phases[1];
    assert_eq!(
        into_synth.index, 2,
        "phases[1] is step_index 2 (the synthesized step)"
    );
    assert!(
        into_synth
            .changes
            .iter()
            .any(|c| c.metric == "throughput" && c.direction == ChangeDirection::Degraded),
        "throughput collapse into the synthesized step must flag Degraded; got {:?}",
        into_synth.changes,
    );
    // Phase index 2 = boundary OUT of the synthesized step: throughput
    // 10 -> 1000 recovery flagged Improved (before == 10 > 0).
    let out_of_synth = &timeline.phases[2];
    assert_eq!(
        out_of_synth.index, 3,
        "phases[2] is step_index 3 (the captured step after)"
    );
    assert!(
        out_of_synth
            .changes
            .iter()
            .any(|c| c.metric == "throughput" && c.direction == ChangeDirection::Improved),
        "throughput recovery out of the synthesized step must flag Improved; got {:?}",
        out_of_synth.changes,
    );
}

/// Event-sort tie-stability: when StepEnd[k] and StepStart[k+1] share
/// the same elapsed_ms (a zero-length inter-step gap at the guest's
/// coarse-ms clock), the rate attributed to step k must be the
/// step-LOCAL StepStart[k]->StepEnd[k] delta, NOT the cross-step
/// StepStart[k]->StepStart[k+1] delta. The total-ordered sort (StepEnd
/// before StepStart on a tie) guarantees the step-local pairing.
#[test]
fn build_phase_buckets_with_stimulus_step_end_tie_attributes_step_local_rate() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let cap = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(1),
    };
    let drained = vec![cap("periodic_000", 1_500), cap("periodic_001", 2_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    // StepEnd[1] and StepStart[2] BOTH at elapsed 2000 (the tie). Step 1's
    // local delta is 500 iters / 1s = 500/s; the cross-step
    // StepStart[1]->StepStart[2] delta would be 9000 iters / 1s = 9000/s.
    let ev = |elapsed_ms: u64, k: u16, iters: u64, is_step_end: bool| StimulusEvent {
        elapsed_ms,
        label: if is_step_end {
            format!("StepEnd[{k}]")
        } else {
            format!("StepStart[{k}]")
        },
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end,
    };
    let stimulus = vec![
        ev(1000, 1, 0, false),    // StepStart[1]
        ev(2000, 1, 500, true),   // StepEnd[1]   (ties with StepStart[2])
        ev(2000, 2, 9000, false), // StepStart[2] (ties with StepEnd[1])
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket present");
    let rate = step1
        .metrics
        .get("iteration_rate")
        .copied()
        .expect("step 1 iteration_rate populated");
    assert!(
        (rate - 500.0).abs() < f64::EPSILON,
        "step 1 must get its LOCAL StepStart[1]->StepEnd[1] rate (500/s), \
         not the cross-step StepStart[1]->StepStart[2] delta (9000/s); got {rate}",
    );
}

/// Hostile input: a synthesized bucket's end_ms clamps to the next
/// step-start, so a non-monotonic (corrupt-wire) StepEnd[k] >=
/// StepStart[k+1] cannot extend the synthesized window past the next
/// step's start (which would over-fold monitor samples into two adjacent
/// synthesized buckets).
#[test]
fn build_phase_buckets_with_stimulus_synthesized_end_ms_clamped_to_next_start() {
    use crate::scenario::snapshot::DrainedSnapshotEntry;
    use crate::timeline::StimulusEvent;
    // No captures -> all steps synthesize.
    let samples = SampleSeries::from_drained_typed(Vec::<DrainedSnapshotEntry>::new(), None);
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
    // StepEnd[1] at 5000 is CORRUPT — it lands AFTER StepStart[2] (2000).
    let stimulus = vec![
        ev(1000, 1, false), // StepStart[1]
        ev(2000, 2, false), // StepStart[2]
        ev(5000, 1, true),  // StepEnd[1] — non-monotonic
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("synthesized step 1 bucket present");
    assert_eq!(
        step1.end_ms, 2000,
        "synthesized step 1 end_ms must clamp to StepStart[2]=2000, NOT the \
         corrupt StepEnd[1]=5000 that would overlap step 2; got {}",
        step1.end_ms,
    );
}

/// Synthesized-bucket regression: a synthesized zero-capture bucket folds the FULL
/// monitor-derived metric set (not just avg_imbalance_ratio), restoring
/// parity with the legacy Timeline::build fallback for a
/// zero-capture-with-monitor run. Before the fix the synthesize path
/// dropped max_imbalance / dsq / fallback / keep_last from the rendered
/// timeline (the path-flip regression pass 3 caught).
#[test]
fn build_phase_buckets_with_stimulus_synthesized_bucket_folds_full_monitor_set() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample, ScxEventCounters};
    use crate::scenario::snapshot::DrainedSnapshotEntry;
    use crate::timeline::StimulusEvent;
    let cpu = |nr: u32, dsq: u32, rq: u64, ev: Option<ScxEventCounters>| CpuSnapshot {
        nr_running: nr,
        local_dsq_depth: dsq,
        rq_clock: rq,
        scx_nr_running: 0,
        scx_flags: 0,
        event_counters: ev,
        schedstat: None,
        vcpu_cpu_time_ns: None,
        vcpu_perf: None,
        avg_irq_util: None,
        sched_domains: None,
    };
    let evc = |fb: i64, kl: i64| {
        Some(ScxEventCounters {
            select_cpu_fallback: fb,
            dispatch_keep_last: kl,
            ..Default::default()
        })
    };
    // Two in-window samples; anchor at the first stimulus elapsed (1000)
    // pins monitor_clock_offset to 0. Per-CPU nr_running 4/2 -> imbalance
    // 2.0; per-CPU dsq (3,1) -> avg 2.0, max 3; event counters
    // 10->110 fallback (delta 100), 5->55 keep_last (delta 50).
    let mon = MonitorReport {
        samples: vec![
            MonitorSample {
                prog_stats: None,
                psi_irq: None,
                elapsed_ms: 1000,
                cpus: vec![cpu(4, 3, 100, evc(10, 5)), cpu(2, 1, 100, None)],
            },
            MonitorSample {
                prog_stats: None,
                psi_irq: None,
                elapsed_ms: 1500,
                cpus: vec![cpu(4, 3, 100, evc(110, 55)), cpu(2, 1, 200, None)],
            },
        ],
        ..Default::default()
    };
    // No captures -> step 1 synthesizes; StepStart[2] bounds its window.
    let samples = SampleSeries::from_drained_typed(Vec::<DrainedSnapshotEntry>::new(), Some(mon));
    let start = |elapsed_ms: u64, k: u16| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let stimulus = vec![start(1000, 1), start(2000, 2)];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("synthesized step 1 bucket present");
    assert_eq!(step1.sample_count, 0, "synthesized bucket is capture-free");
    let g = |k: &str| step1.metrics.get(k).copied();
    // The monitor-derived metrics the pre-fix synthesize path DROPPED,
    // now restored to parity with Timeline::build's compute_metrics:
    assert_eq!(g("max_imbalance_ratio"), Some(2.0), "max imbalance");
    assert_eq!(g("avg_dsq_depth"), Some(2.0), "avg dsq depth");
    assert_eq!(g("max_dsq_depth"), Some(3.0), "max dsq depth");
    assert_eq!(
        g("total_fallback"),
        Some(100.0),
        "fallback counter delta 110-10"
    );
    assert_eq!(
        g("total_keep_last"),
        Some(50.0),
        "keep_last counter delta 55-5"
    );
    // avg_imbalance_ratio was folded pre-fix too; still present.
    assert_eq!(g("avg_imbalance_ratio"), Some(2.0), "avg imbalance");
}
