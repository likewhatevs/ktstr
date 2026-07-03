use super::*;

// -- compare_rows_by per-phase pass tests --------------------------
//
// These tests exercise the per-row-pair phase intersection that
// populates CompareReport.phase_deltas, phase_coverage_diffs, and
// unpaired_phases. They go
// through compare_rows_by directly (rather than the full
// compare_partitions which also does filtering/averaging) because
// the parallel pass lives inside compare_rows_by's row-pair
// iteration and that is the load-bearing surface to pin.
//
// Each test builds 2 GauntletRows via make_row, attaches phase
// buckets explicitly, then asserts the resulting CompareReport
// shape against the expected per-phase + unpaired data flow.

/// Matched phases on both sides populate one
/// PhaseDeltaRow per phase per metric present in both
/// buckets; unpaired_phases stays empty. Pins the matched
/// branch of the parallel pass.
#[test]
fn compare_rows_by_emits_phase_deltas_when_both_sides_have_matched_phases() {
    let mut row_a = make_row("test_a", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("test_a", "tiny-1llc", true, 0.0);
    row_a.phases = vec![
        make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)]),
        make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 8.0)]),
    ];
    row_b.phases = vec![
        make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 6.0)]),
        make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 15.0)]),
    ];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
    assert_eq!(
        report.phase_deltas.len(),
        2,
        "2 phases × 1 metric = 2 deltas"
    );
    assert!(
        report.unpaired_phases.is_empty(),
        "both phases matched, no orphans"
    );
    let baseline = report
        .phase_deltas
        .iter()
        .find(|r| r.step_index == 0)
        .expect("BASELINE delta present");
    assert_eq!(baseline.label, "BASELINE");
    assert_eq!(baseline.a, 5.0);
    assert_eq!(baseline.b, 6.0);
    assert_eq!(baseline.delta, 1.0);
    let step0 = report
        .phase_deltas
        .iter()
        .find(|r| r.step_index == 1)
        .expect("Step[0] delta present");
    assert_eq!(step0.label, "Step[0]");
    assert_eq!(step0.a, 8.0);
    assert_eq!(step0.b, 15.0);
    assert_eq!(step0.delta, 7.0);
}

/// The schbench per-phase metrics drive perf-delta's cross-run per-phase A/B
/// view ([`push_phase_deltas`]) with their REGISTERED polarity, in both
/// directions: `wakeup_p99_latency_us` is LowerBetter (a candidate whose p99
/// rises across a phase is a regression; falling is an improvement) and
/// `schbench_loop_count` is HigherBetter (falling throughput is a regression).
/// The sibling per-phase tests use the generic `max_dsq_depth` metric; this
/// pins that the SCHBENCH-registered names resolve through `metric_def` -- the
/// `.expect()` would fail if one were unregistered, since `push_phase_deltas`
/// skips unknown names -- and that polarity (not raw delta sign) orients the
/// verdict. This is the exact path a `cargo ktstr perf-delta` schbench A/B
/// (baseline-vs-candidate, partitioned per side) renders per phase. Deltas are
/// sized well above each metric's dual gate (p99 abs 50µs / loop_count abs 10)
/// so the classification exercises the regression branch, not the below-gate
/// `unchanged` one.
#[test]
fn schbench_per_phase_metrics_drive_cross_run_verdict_with_registry_polarity() {
    // LowerBetter latency, regression: B's Step[0] p99 rises 1000 -> 2000 µs
    // (delta 1000 >> 50µs abs, rel 1.0 >> 0.25).
    let mut a = make_row("perf_scn", "tiny-1llc", true, 0.0);
    let mut b = make_row("perf_scn", "tiny-1llc", true, 0.0);
    a.phases = vec![make_phase_bucket(
        1,
        "Step[0]",
        &[("wakeup_p99_latency_us", 1000.0)],
    )];
    b.phases = vec![make_phase_bucket(
        1,
        "Step[0]",
        &[("wakeup_p99_latency_us", 2000.0)],
    )];
    let report = compare_rows_by(&[a], &[b], &[], None, &ComparisonPolicy::default());
    let lat = report
        .phase_deltas
        .iter()
        .find(|d| d.metric.name == "wakeup_p99_latency_us")
        .expect("wakeup_p99_latency_us must resolve to a registered metric_def and emit a delta");
    assert_eq!(lat.delta, 1000.0);
    assert!(
        lat.is_regression,
        "a rising p99 on a LowerBetter metric is a regression",
    );

    // The converse direction: B lowers p99 2000 -> 1000 -> improvement, NOT a
    // regression (proves polarity orients the verdict, not the raw delta sign).
    let mut a2 = make_row("perf_scn", "tiny-1llc", true, 0.0);
    let mut b2 = make_row("perf_scn", "tiny-1llc", true, 0.0);
    a2.phases = vec![make_phase_bucket(
        1,
        "Step[0]",
        &[("wakeup_p99_latency_us", 2000.0)],
    )];
    b2.phases = vec![make_phase_bucket(
        1,
        "Step[0]",
        &[("wakeup_p99_latency_us", 1000.0)],
    )];
    let report2 = compare_rows_by(&[a2], &[b2], &[], None, &ComparisonPolicy::default());
    let lat2 = report2
        .phase_deltas
        .iter()
        .find(|d| d.metric.name == "wakeup_p99_latency_us")
        .expect("delta present");
    assert!(
        !lat2.is_regression,
        "a falling p99 on a LowerBetter metric is an improvement",
    );

    // HigherBetter throughput, regression: B's Step[0] loop_count falls
    // 5000 -> 1000 (delta -4000, abs 4000 >> 10, rel 0.8 >> 0.30). The OPPOSITE
    // polarity from the latency metric, so a NEGATIVE delta is the regression.
    let mut a3 = make_row("perf_scn", "tiny-1llc", true, 0.0);
    let mut b3 = make_row("perf_scn", "tiny-1llc", true, 0.0);
    a3.phases = vec![make_phase_bucket(
        1,
        "Step[0]",
        &[("schbench_loop_count", 5000.0)],
    )];
    b3.phases = vec![make_phase_bucket(
        1,
        "Step[0]",
        &[("schbench_loop_count", 1000.0)],
    )];
    let report3 = compare_rows_by(&[a3], &[b3], &[], None, &ComparisonPolicy::default());
    let lc = report3
        .phase_deltas
        .iter()
        .find(|d| d.metric.name == "schbench_loop_count")
        .expect("schbench_loop_count must resolve to a registered metric_def and emit a delta");
    assert_eq!(lc.delta, -4000.0);
    assert!(
        lc.is_regression,
        "falling loop_count on a HigherBetter metric is a regression (less throughput)",
    );
}

/// A-side has phase [0, 1]; B-side has phase [0, 2].
/// Matched phases (step_index = 0) emit PhaseDeltaRow;
/// step_index = 1 emits UnpairedPhaseRow side=A;
/// step_index = 2 emits UnpairedPhaseRow side=B. Pins the
/// cross-cardinality intersection and the surface-don't-drop
/// contract for orphan buckets.
#[test]
fn compare_rows_by_emits_unpaired_phases_when_phase_coverage_differs() {
    let mut row_a = make_row("test_x", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("test_x", "tiny-1llc", true, 0.0);
    row_a.phases = vec![
        make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 4.0)]),
        make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 12.0)]),
    ];
    row_b.phases = vec![
        make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)]),
        make_phase_bucket(2, "Step[1]", &[("max_dsq_depth", 9.0)]),
    ];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
    assert_eq!(
        report.phase_deltas.len(),
        1,
        "only BASELINE matches across sides -> 1 delta"
    );
    assert_eq!(report.phase_deltas[0].step_index, 0);
    assert_eq!(report.unpaired_phases.len(), 2, "step 1 (A) + step 2 (B)");
    let a_orphan = report
        .unpaired_phases
        .iter()
        .find(|u| u.side == ComparePartition::A)
        .expect("A-only orphan present");
    assert_eq!(a_orphan.step_index, 1);
    assert_eq!(a_orphan.label, "Step[0]");
    let b_orphan = report
        .unpaired_phases
        .iter()
        .find(|u| u.side == ComparePartition::B)
        .expect("B-only orphan present");
    assert_eq!(b_orphan.step_index, 2);
    assert_eq!(b_orphan.label, "Step[1]");
}

/// A metric present in only ONE bucket of a MATCHED phase (same step_index) is
/// a PhaseCoverageDiff — the per-phase analog of the scalar CoverageDiff — not
/// silently dropped. Pre-fix, push_phase_deltas iterated only the A-side
/// bucket, so an A-only metric was `continue`d and a B-only one was never
/// iterated; both vanished from the per-phase view (UnpairedPhaseRow covers
/// only WHOLE one-sided phases, not a one-sided metric within a matched phase).
#[test]
fn compare_rows_by_phase_one_sided_metric_in_matched_phase_is_coverage_diff() {
    // BASELINE matches on both sides; max_dsq_depth is present in both (a
    // normal delta), avg_nr_running in only one bucket (the coverage diff).
    let mut row_a = make_row("test_pc", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("test_pc", "tiny-1llc", true, 0.0);
    row_a.phases = vec![make_phase_bucket(
        0,
        "BASELINE",
        &[("max_dsq_depth", 4.0), ("avg_nr_running", 5.0)],
    )];
    row_b.phases = vec![make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 6.0)])];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());

    assert_eq!(
        report.phase_deltas.len(),
        1,
        "only the both-present metric (max_dsq_depth) yields a delta"
    );
    assert_eq!(report.phase_deltas[0].metric.name, "max_dsq_depth");
    assert!(
        report.unpaired_phases.is_empty(),
        "the phase itself matched on both sides — not a whole one-sided phase"
    );
    assert_eq!(
        report.phase_coverage_diffs.len(),
        1,
        "avg_nr_running is one-sided WITHIN the matched phase"
    );
    let cd = &report.phase_coverage_diffs[0];
    assert_eq!(cd.metric.name, "avg_nr_running");
    assert_eq!(cd.step_index, 0);
    assert_eq!(cd.present_side, ComparePartition::A);
    assert_eq!(cd.value, 5.0);

    // Mirror: avg_nr_running present in B's matched bucket only -> present_side B
    // (the case the pre-fix loop never iterated at all).
    let mut row_a2 = make_row("test_pc", "tiny-1llc", true, 0.0);
    let mut row_b2 = make_row("test_pc", "tiny-1llc", true, 0.0);
    row_a2.phases = vec![make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 4.0)])];
    row_b2.phases = vec![make_phase_bucket(
        0,
        "BASELINE",
        &[("max_dsq_depth", 6.0), ("avg_nr_running", 7.0)],
    )];
    let report2 = compare_rows_by(
        &[row_a2],
        &[row_b2],
        &[],
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(report2.phase_coverage_diffs.len(), 1);
    assert_eq!(
        report2.phase_coverage_diffs[0].metric.name,
        "avg_nr_running"
    );
    assert_eq!(
        report2.phase_coverage_diffs[0].present_side,
        ComparePartition::B
    );
    assert_eq!(report2.phase_coverage_diffs[0].value, 7.0);
}

/// `format_phase_block_lines` (the pure renderer behind `print_phase_block`)
/// renders the one-sided-metric coverage table for a report whose ONLY
/// phase data is `phase_coverage_diffs` — no deltas, no unpaired phases. This
/// pins the render gate's `phase_coverage_diffs` disjunct (a coverage-only
/// report still renders) AND the coverage-table content (SIDE/METRIC/VALUE),
/// the exact branch the data-layer `compare_rows_by` test stops short of.
/// A matched BASELINE whose only metrics are one-sided (A: avg_nr_running,
/// B: max_dsq_depth) yields no both-present delta and no whole-phase orphan,
/// so the report carries coverage diffs alone.
#[test]
fn phase_block_lines_render_a_coverage_only_report() {
    let mut row_a = make_row("test_pc", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("test_pc", "tiny-1llc", true, 0.0);
    row_a.phases = vec![make_phase_bucket(0, "BASELINE", &[("avg_nr_running", 5.0)])];
    row_b.phases = vec![make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 6.0)])];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
    assert!(
        report.phase_deltas.is_empty(),
        "no both-present metric -> no phase delta"
    );
    assert!(
        report.unpaired_phases.is_empty(),
        "BASELINE matched on both sides -> no whole-phase orphan"
    );
    assert_eq!(
        report.phase_coverage_diffs.len(),
        2,
        "avg_nr_running (A-only) + max_dsq_depth (B-only) within the matched phase"
    );

    let lines = format_phase_block_lines(&report, &PhaseDisplayOptions::default(), "A", "B");
    let joined = lines.join("\n");
    assert!(
        joined.contains("phase coverage asymmetry (one-sided metrics within a matched phase)"),
        "the render gate fired on coverage diffs alone and built the table: {joined}"
    );
    assert!(
        joined.contains("avg_nr_running") && joined.contains("max_dsq_depth"),
        "both one-sided metric rows render: {joined}"
    );
    assert!(
        !joined.contains("VERDICT"),
        "no phase-delta table renders when phase_deltas is empty: {joined}"
    );
    // The default-options footer hint reports the coverage-diff count, so a
    // coverage-only report no longer reads "0 ... shown" with a populated
    // coverage table above it.
    assert!(
        joined.contains("0 delta row(s)") && joined.contains("2 one-sided-metric coverage diff(s)"),
        "footer hint must report the coverage-diff count, not only deltas: {joined}"
    );
}

/// `format_phase_block_lines` returns no lines when there is no phase data at
/// all — the render gate's false path. Pins that a report with empty
/// phase_deltas/unpaired_phases/phase_coverage_diffs prints nothing.
#[test]
fn phase_block_lines_empty_report_renders_nothing() {
    let report = compare_rows_by(
        &[make_row("t", "tiny-1llc", true, 0.0)],
        &[make_row("t", "tiny-1llc", true, 0.0)],
        &[],
        None,
        &ComparisonPolicy::default(),
    );
    assert!(report.phase_deltas.is_empty());
    assert!(report.unpaired_phases.is_empty());
    assert!(report.phase_coverage_diffs.is_empty());
    assert!(
        format_phase_block_lines(&report, &PhaseDisplayOptions::default(), "A", "B").is_empty(),
        "an all-empty phase report must render no lines"
    );
}

/// `--no-phases` suppresses the whole block even when coverage diffs are
/// present — pins the render gate's `!no_phases` conjunct.
#[test]
fn phase_block_lines_no_phases_flag_suppresses_coverage_only_report() {
    let mut row_a = make_row("test_pc", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("test_pc", "tiny-1llc", true, 0.0);
    row_a.phases = vec![make_phase_bucket(0, "BASELINE", &[("avg_nr_running", 5.0)])];
    row_b.phases = vec![make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 6.0)])];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
    assert!(!report.phase_coverage_diffs.is_empty());
    let opts = PhaseDisplayOptions {
        no_phases: true,
        ..PhaseDisplayOptions::default()
    };
    assert!(
        format_phase_block_lines(&report, &opts, "A", "B").is_empty(),
        "--no-phases must suppress even a coverage-only report"
    );
}

/// The per-phase AND unpaired-phase compare surfaces suppress Rate
/// components too (not just the scalar findings pass). A and B share step 0
/// (matched); A has a step-1 phase B lacks (side=A orphan) and B has a
/// step-2 phase A lacks (side=B orphan) — covering BOTH orphan arms. Every
/// bucket carries `total_phase_iterations` (a suppressed component — the
/// real per-phase producer inserts it into every bucket) and
/// `max_dsq_depth` (not suppressed). The component must appear in NEITHER
/// the PhaseDeltaRows NOR either UnpairedPhaseRow.metrics, while
/// `max_dsq_depth` survives on all. `make_phase_bucket` builds
/// `PhaseBucket.metrics` directly — the exact map the per-phase pass reads.
#[test]
fn compare_rows_per_phase_and_unpaired_suppress_rate_components() {
    let mut row_a = make_row("t", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("t", "tiny-1llc", true, 0.0);
    row_a.phases = vec![
        make_phase_bucket(
            0,
            "BASELINE",
            &[("total_phase_iterations", 1000.0), ("max_dsq_depth", 5.0)],
        ),
        make_phase_bucket(
            1,
            "Step[0]",
            &[("total_phase_iterations", 2000.0), ("max_dsq_depth", 8.0)],
        ),
    ];
    // B lacks step 1 -> A's step-1 phase is an unpaired (side=A) row; B has a
    // step-2 phase A lacks -> an unpaired (side=B) row. Covers BOTH orphan
    // arms (the (Some,None) and (None,Some) metrics_without_suppressed calls).
    row_b.phases = vec![
        make_phase_bucket(
            0,
            "BASELINE",
            &[("total_phase_iterations", 1500.0), ("max_dsq_depth", 6.0)],
        ),
        make_phase_bucket(
            2,
            "Step[1]",
            &[("total_phase_iterations", 3000.0), ("max_dsq_depth", 9.0)],
        ),
    ];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());

    // Matched step 0: component suppressed from phase_deltas; the
    // non-suppressed metric still emits a delta (emitted regardless of gate).
    let delta_names: Vec<&str> = report.phase_deltas.iter().map(|r| r.metric.name).collect();
    assert!(
        !delta_names.contains(&"total_phase_iterations"),
        "Rate component must be suppressed from per-phase deltas; got {delta_names:?}",
    );
    assert!(
        delta_names.contains(&"max_dsq_depth"),
        "non-suppressed per-phase metric must still emit a delta; got {delta_names:?}",
    );

    // Unpaired step 1 (side A): component filtered from the orphan's metrics;
    // the non-suppressed metric survives.
    let orphan = report
        .unpaired_phases
        .iter()
        .find(|r| r.step_index == 1)
        .expect("A's step-1 phase emits an unpaired row");
    assert!(
        !orphan.metrics.contains_key("total_phase_iterations"),
        "Rate component must be filtered from UnpairedPhaseRow.metrics; got {:?}",
        orphan.metrics.keys().collect::<Vec<_>>(),
    );
    assert!(
        orphan.metrics.contains_key("max_dsq_depth"),
        "non-suppressed metric must survive in UnpairedPhaseRow.metrics",
    );

    // Unpaired step 2 (side B): the OTHER orphan arm also filters the
    // component (the two arms are distinct call sites).
    let orphan_b = report
        .unpaired_phases
        .iter()
        .find(|r| r.step_index == 2)
        .expect("B's step-2 phase emits an unpaired row");
    assert!(
        !orphan_b.metrics.contains_key("total_phase_iterations"),
        "Rate component must be filtered from the side-B UnpairedPhaseRow.metrics too; got {:?}",
        orphan_b.metrics.keys().collect::<Vec<_>>(),
    );
    assert!(
        orphan_b.metrics.contains_key("max_dsq_depth"),
        "non-suppressed metric must survive in the side-B UnpairedPhaseRow.metrics",
    );
}

/// A render-suppressed Rate component present on exactly ONE side of a
/// MATCHED phase is dropped, NOT emitted as a PhaseCoverageDiff — the
/// one-sided sibling of the suppression the test above pins for the
/// (Some, Some) delta path and the unpaired arms. push_phase_deltas runs
/// `is_render_suppressed_component` BEFORE the (Some, None) / (None, Some)
/// match, so a one-sided suppressed component is `continue`d ahead of either
/// coverage arm. Pins that ordering: a future refactor moving the suppression
/// check inside only the (Some, Some) arm would leak the component into
/// `phase_coverage_diffs` and the coverage table, and nothing else would catch
/// it.
#[test]
fn compare_rows_by_phase_one_sided_suppressed_component_is_dropped_not_coverage_diff() {
    // Matched BASELINE: A carries only the suppressed component
    // (total_phase_iterations); B carries only a non-suppressed metric
    // (max_dsq_depth). Both are one-sided within the matched phase.
    let mut row_a = make_row("test_sc", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("test_sc", "tiny-1llc", true, 0.0);
    row_a.phases = vec![make_phase_bucket(
        0,
        "BASELINE",
        &[("total_phase_iterations", 1000.0)],
    )];
    row_b.phases = vec![make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 6.0)])];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());

    let cov_names: Vec<&str> = report
        .phase_coverage_diffs
        .iter()
        .map(|c| c.metric.name)
        .collect();
    assert!(
        !cov_names.contains(&"total_phase_iterations"),
        "a one-sided render-suppressed component must NOT become a PhaseCoverageDiff; got {cov_names:?}",
    );
    assert_eq!(
        cov_names,
        vec!["max_dsq_depth"],
        "only the non-suppressed one-sided metric surfaces as a coverage diff",
    );
    assert_eq!(
        report.phase_coverage_diffs[0].present_side,
        ComparePartition::B
    );
    assert_eq!(report.phase_coverage_diffs[0].value, 6.0);
    assert!(
        report.phase_deltas.is_empty(),
        "no metric is present on both sides -> no delta",
    );
    assert!(
        report.unpaired_phases.is_empty(),
        "the phase matched on both sides -> no whole-phase orphan",
    );
}

/// A one-sided Informational metric within a MATCHED phase DOES surface as a
/// PhaseCoverageDiff: the (Some, None) / (None, Some) arms carry no
/// classify_direction / polarity gate (a coverage diff is never a verdict),
/// unlike the (Some, Some) delta arm which skips Informational metrics. Pins
/// the documented intent that a one-sided Informational metric is reported,
/// not dropped.
#[test]
fn compare_rows_by_phase_one_sided_informational_metric_is_coverage_diff() {
    // total_ttwu_count is Polarity::Informational; present only on A's matched
    // BASELINE, with a non-suppressed directional metric only on B.
    let mut row_a = make_row("test_inf_os", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("test_inf_os", "tiny-1llc", true, 0.0);
    row_a.phases = vec![make_phase_bucket(
        0,
        "BASELINE",
        &[("total_ttwu_count", 42.0)],
    )];
    row_b.phases = vec![make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 6.0)])];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());

    let inf = report
        .phase_coverage_diffs
        .iter()
        .find(|c| c.metric.name == "total_ttwu_count")
        .expect("a one-sided Informational metric surfaces as a coverage diff");
    assert_eq!(inf.present_side, ComparePartition::A);
    assert_eq!(inf.value, 42.0);
    assert!(
        report
            .phase_deltas
            .iter()
            .all(|d| d.metric.name != "total_ttwu_count"),
        "an Informational metric is never a phase delta",
    );
}

/// Empty `phases` on either side suppresses the entire
/// per-phase pass (no PhaseDeltaRow, no UnpairedPhaseRow).
/// Pins the single-phase legacy-compat short-circuit: a
/// legacy sidecar with no phase data on EITHER side does
/// not flood the unpaired section with one orphan per
/// populated B-side phase.
#[test]
fn compare_rows_by_skips_phase_pass_when_either_side_phases_empty() {
    let row_a = make_row("test_y", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("test_y", "tiny-1llc", true, 0.0);
    // A has empty phases (legacy sidecar); B has phases.
    row_b.phases = vec![
        make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 6.0)]),
        make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 10.0)]),
    ];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
    assert!(
        report.phase_deltas.is_empty(),
        "empty-on-either-side short-circuit must suppress all per-phase rows"
    );
    assert!(
        report.unpaired_phases.is_empty(),
        "empty-on-either-side must not emit orphan rows for B-side phases"
    );
}

/// Polarity-aware is_regression: a `worst_*`-style
/// LowerBetter metric where B > A flags regression=true;
/// a `*_iterations`/HigherBetter metric where B > A flags
/// regression=false (improvement). Pins the polarity wiring
/// that the existing scalar-finding path uses, applied to
/// the per-phase pass.
///
/// Values chosen to clear the per-phase dual-gate (the same
/// `|delta| < default_abs || rel_delta < default_rel` gate
/// the scalar pass uses inside its per-metric loop in
/// `push_scalar_findings`). max_dsq_depth has
/// default_abs=10.0 / default_rel=0.50; total_iterations has
/// default_abs=100.0 / default_rel=0.10. The 10→25 (delta=15)
/// and 200→400 (delta=200) deltas both clear both gates, so
/// polarity dispatches as documented. Sub-threshold deltas
/// are explicitly exercised by the dual-gate test below.
#[test]
fn compare_rows_by_phase_deltas_respect_metric_polarity() {
    let mut row_a = make_row("test_z", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("test_z", "tiny-1llc", true, 0.0);
    // max_dsq_depth Peak/LowerBetter: delta=15 > abs=10, rel=1.5 > rel=0.5
    // total_iterations Counter/HigherBetter: delta=200 > abs=100, rel=1.0 > rel=0.10
    row_a.phases = vec![make_phase_bucket(
        0,
        "BASELINE",
        &[("max_dsq_depth", 10.0), ("total_iterations", 200.0)],
    )];
    row_b.phases = vec![make_phase_bucket(
        0,
        "BASELINE",
        &[("max_dsq_depth", 25.0), ("total_iterations", 400.0)],
    )];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
    assert_eq!(report.phase_deltas.len(), 2, "2 metrics in 1 bucket");
    let dsq = report
        .phase_deltas
        .iter()
        .find(|r| r.metric.name == "max_dsq_depth")
        .expect("max_dsq_depth delta present");
    assert!(
        dsq.is_regression,
        "max_dsq_depth bigger on B -> regression (LowerBetter polarity)"
    );
    let iters = report
        .phase_deltas
        .iter()
        .find(|r| r.metric.name == "total_iterations")
        .expect("total_iterations delta present");
    assert!(
        !iters.is_regression,
        "total_iterations bigger on B -> improvement (HigherBetter polarity)"
    );
}

/// An `Polarity::Informational` metric placed in a `PhaseBucket` is DROPPED
/// from the scalar per-phase delta table: the per-phase path classifies via
/// `is_regression: bool` and has no informational state, so it skips
/// directionless metrics rather than misclassifying them. Informational
/// phase-bucketed metrics DO exist in production — the IRQ/softirq schedstat
/// counters (`total_hardirqs`, `total_softirq_*`, `total_*_time_ns`,
/// `total_steal_time_ns`) are Informational AND phase-bucketed — and are
/// dropped here by the bool-verdict skip (surfacing them is task-tracked; the
/// noise per-phase path already shows them). This test uses `total_ttwu_count`
/// as one directionless example to pin that the skip fires, never silently
/// flagging a directionless metric a regression.
#[test]
fn compare_rows_by_phase_deltas_skip_informational_metrics() {
    let mut row_a = make_row("test_inf", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("test_inf", "tiny-1llc", true, 0.0);
    // total_ttwu_count is Polarity::Informational; max_dsq_depth is directional.
    row_a.phases = vec![make_phase_bucket(
        0,
        "BASELINE",
        &[("max_dsq_depth", 10.0), ("total_ttwu_count", 1000.0)],
    )];
    row_b.phases = vec![make_phase_bucket(
        0,
        "BASELINE",
        &[("max_dsq_depth", 25.0), ("total_ttwu_count", 5000.0)],
    )];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
    assert_eq!(
        report.phase_deltas.len(),
        1,
        "only the directional metric produces a per-phase delta"
    );
    assert_eq!(report.phase_deltas[0].metric.name, "max_dsq_depth");
    assert!(
        !report
            .phase_deltas
            .iter()
            .any(|r| r.metric.name == "total_ttwu_count"),
        "the Informational metric must be skipped from the per-phase table"
    );
}

/// per-phase pass honors the dual-gate semantic the
/// scalar pass uses inside its per-metric loop in
/// `push_scalar_findings` (`|delta| < default_abs ||
/// rel_delta < default_rel`). Sub-threshold deltas are still
/// emitted into `phase_deltas` (programmatic consumers see
/// every paired comparison) but their `is_regression` flag
/// is `false` — the renderer paints them as "improvement"
/// or unstyled rather than the red REGRESSION verdict.
///
/// Two cases pinned in one test:
/// - sub-abs-gate: `max_dsq_depth` `default_abs=10.0` —
///   delta=5 (5→10) is direction-matching for LowerBetter
///   polarity but `5 < 10` → is_regression=false.
/// - above-both-gates: `max_dsq_depth` delta=15 (10→25)
///   passes both abs and rel gates → is_regression=true.
#[test]
fn compare_rows_by_phase_deltas_dual_gate_suppresses_subthreshold_regressions() {
    let mut row_a = make_row("test_dg", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("test_dg", "tiny-1llc", true, 0.0);
    // BASELINE: delta=5 (under abs=10), Step[0]: delta=15 (over abs=10 + rel=0.5)
    row_a.phases = vec![
        make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)]),
        make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 10.0)]),
    ];
    row_b.phases = vec![
        make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 10.0)]),
        make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 25.0)]),
    ];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
    assert_eq!(
        report.phase_deltas.len(),
        2,
        "both phases emit a delta row regardless of dual-gate"
    );
    let baseline = report
        .phase_deltas
        .iter()
        .find(|r| r.step_index == 0)
        .expect("BASELINE delta present");
    assert_eq!(baseline.delta, 5.0);
    assert!(
        !baseline.is_regression,
        "delta=5 < default_abs=10 → sub-abs-gate; \
             is_regression must clear despite LowerBetter polarity direction"
    );
    let step0 = report
        .phase_deltas
        .iter()
        .find(|r| r.step_index == 1)
        .expect("Step[0] delta present");
    assert_eq!(step0.delta, 15.0);
    assert!(
        step0.is_regression,
        "delta=15 ≥ default_abs=10 AND rel_delta=1.5 ≥ default_rel=0.5 → \
             above both gates; LowerBetter polarity sets is_regression=true"
    );
}

/// Per-phase mirror of the scalar zero-baseline fix: a phase metric
/// rising from a ~zero baseline to a value over its absolute gate must
/// set `is_regression`. Before the fix the per-phase `rel_delta` was
/// forced to `0.0` on a ~zero baseline, which (AND-gated with the
/// relative threshold) cleared `is_regression` for every `0 -> large`
/// phase jump. The fix treats the appearance as an unbounded relative
/// change (`+inf`), so the absolute gate alone decides; a both-~zero
/// phase carries no signal and stays non-regression.
#[test]
fn compare_rows_by_phase_deltas_zero_baseline_jump_is_a_regression() {
    let mut row_a = make_row("test_zphase", "tiny-1llc", true, 0.0);
    let mut row_b = make_row("test_zphase", "tiny-1llc", true, 0.0);
    // step 0: 0 -> 15 (over abs=10) — must flag. step 1: 0 -> 0 — no signal.
    row_a.phases = vec![
        make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 0.0)]),
        make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 0.0)]),
    ];
    row_b.phases = vec![
        make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 15.0)]),
        make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 0.0)]),
    ];
    let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
    let baseline = report
        .phase_deltas
        .iter()
        .find(|r| r.step_index == 0)
        .expect("BASELINE delta present");
    assert_eq!(baseline.delta, 15.0);
    assert!(
        baseline.is_regression,
        "0 -> 15 (>= abs gate 10) from a zero baseline must set is_regression, \
             not be vetoed by the zero-baseline relative gate"
    );
    let step1 = report
        .phase_deltas
        .iter()
        .find(|r| r.step_index == 1)
        .expect("Step[0] delta present");
    assert_eq!(step1.delta, 0.0);
    assert!(
        !step1.is_regression,
        "0 -> 0 carries no signal (rel_delta=0, |delta|=0 < abs gate) → not a regression"
    );
}

// -- PhaseDisplayOptions::rel_threshold --

/// `PhaseDisplayOptions::rel_threshold` returns the
/// `phase_threshold` percent divided by 100 (the override
/// branch) when the flag is set, regardless of what
/// `ComparisonPolicy` says. Confirms `--phase-threshold X`
/// is the sole determinant of per-phase relative-gate
/// resolution at the override branch.
#[test]
fn phase_display_options_rel_threshold_override_branch_takes_precedence() {
    let opts = PhaseDisplayOptions {
        phase_threshold: Some(25.0), // 25%
        ..PhaseDisplayOptions::default()
    };
    let mut policy = ComparisonPolicy::default();
    policy
        .per_metric_percent
        .insert("max_dsq_depth".into(), 99.0);
    let resolved = opts.rel_threshold(&policy, "max_dsq_depth", 0.50);
    assert_eq!(
        resolved, 0.25,
        "--phase-threshold 25 → 0.25 fraction regardless of policy override"
    );
}

/// `PhaseDisplayOptions::rel_threshold` falls through to
/// `ComparisonPolicy::rel_threshold` when `phase_threshold`
/// is absent. The policy's per-metric or default-percent
/// override applies (the operator's CLI surface). Pins the
/// fallback chain so a `--policy file.json` invocation with
/// no `--phase-threshold` continues to feed the per-metric
/// thresholds into the per-phase pass.
#[test]
fn phase_display_options_rel_threshold_falls_through_to_policy_when_unset() {
    let opts = PhaseDisplayOptions::default(); // phase_threshold = None
    let mut policy = ComparisonPolicy::default();
    policy
        .per_metric_percent
        .insert("max_dsq_depth".into(), 30.0); // 30%
    let resolved = opts.rel_threshold(&policy, "max_dsq_depth", 0.50);
    assert_eq!(
        resolved, 0.30,
        "absent --phase-threshold → policy.rel_threshold = 30% / 100 = 0.30"
    );
}

/// When neither `PhaseDisplayOptions::phase_threshold` nor
/// any `ComparisonPolicy` override applies, fall through to
/// the registry default (e.g. `MetricDef.default_rel = 0.50`
/// for `max_dsq_depth`). Pins the "no flag at all" path so
/// the dual-gate semantic at the data layer continues to
/// use the metric registry as the source of truth.
#[test]
fn phase_display_options_rel_threshold_falls_through_to_registry_default() {
    let opts = PhaseDisplayOptions::default();
    let policy = ComparisonPolicy::default(); // no overrides
    let resolved = opts.rel_threshold(&policy, "max_dsq_depth", 0.50);
    assert_eq!(
        resolved, 0.50,
        "no flag, no policy → registry default_rel = 0.50"
    );
}

// -- PhaseDisplayOptions::matches_phase (step-axis filter) --

/// Default-shape opts (`--phase` None, `--steps-only` false)
/// match every step_index. Pins the "no flag, every step
/// renders" default path that the earlier per-phase pass tests
/// implicitly assume but never assert directly on the
/// extracted helper.
#[test]
fn matches_phase_default_opts_pass_all_steps() {
    let opts = PhaseDisplayOptions::default();
    for step in [0u16, 1, 2, 7, 65535] {
        assert!(
            opts.matches_phase(step),
            "default opts must match step_index = {step}"
        );
    }
}

/// `--steps-only` suppresses BASELINE (step_index = 0) and
/// admits every other step. Pins the single-step exclusion
/// behavior and confirms that no other steps are
/// accidentally caught by the gate.
#[test]
fn matches_phase_steps_only_suppresses_only_baseline() {
    let opts = PhaseDisplayOptions {
        steps_only: true,
        ..PhaseDisplayOptions::default()
    };
    assert!(
        !opts.matches_phase(0),
        "--steps-only: step_index = 0 (BASELINE) must be suppressed"
    );
    for step in [1u16, 2, 3, 7, 65535] {
        assert!(
            opts.matches_phase(step),
            "--steps-only: step_index = {step} must NOT be suppressed"
        );
    }
}

/// `--phase N` keeps only step_index == N and rejects every
/// other step (including BASELINE when N != 0). Pins the
/// single-phase filter.
#[test]
fn matches_phase_phase_filter_keeps_only_target_step() {
    let opts = PhaseDisplayOptions {
        phase: Some(2),
        ..PhaseDisplayOptions::default()
    };
    assert!(opts.matches_phase(2), "--phase 2 must keep step_index = 2");
    for step in [0u16, 1, 3, 7, 65535] {
        assert!(
            !opts.matches_phase(step),
            "--phase 2: step_index = {step} must be suppressed"
        );
    }
}

/// `--phase 0` keeps BASELINE only — the
/// `--phase`-via-step-zero arm. Confirms an operator can
/// explicitly request BASELINE through the `--phase` flag
/// rather than only through "no flag at all" default.
/// (Mutually exclusive with `--steps-only` at the CLI parse
/// layer; here we test the method in isolation.)
#[test]
fn matches_phase_phase_zero_keeps_only_baseline() {
    let opts = PhaseDisplayOptions {
        phase: Some(0),
        ..PhaseDisplayOptions::default()
    };
    assert!(opts.matches_phase(0), "--phase 0 must keep step_index = 0");
    for step in [1u16, 2, 7, 65535] {
        assert!(
            !opts.matches_phase(step),
            "--phase 0: step_index = {step} must be suppressed"
        );
    }
}

// -- PhaseDisplayOptions::passes_delta_threshold --

/// `--phase-threshold` unset (default opts) passes every
/// row regardless of delta magnitude — matches the clap
/// doc contract ("absence shows every paired row"). Pins
/// the default render-everything behavior.
#[test]
fn passes_delta_threshold_unset_admits_all_deltas() {
    let opts = PhaseDisplayOptions::default();
    let metric = METRICS
        .iter()
        .find(|m| m.name == "max_dsq_depth")
        .expect("max_dsq_depth in METRICS");
    // delta = 0.0 (no change) and delta = 999.0 (massive change) both pass.
    let zero_delta = PhaseDeltaRow {
        pairing_key: PairingKey(vec!["t".into()]),
        step_index: 0,
        label: "BASELINE".into(),
        metric,
        a: 100.0,
        b: 100.0,
        delta: 0.0,
        is_regression: false,
    };
    let big_delta = PhaseDeltaRow {
        pairing_key: PairingKey(vec!["t".into()]),
        step_index: 0,
        label: "BASELINE".into(),
        metric,
        a: 100.0,
        b: 1099.0,
        delta: 999.0,
        is_regression: true,
    };
    assert!(opts.passes_delta_threshold(&zero_delta));
    assert!(opts.passes_delta_threshold(&big_delta));
}

/// `--phase-threshold 10` (10% gate) suppresses rows whose
/// relative delta is below 10% AND admits rows at or above.
/// Pins the inclusive-at-boundary semantic (`>=`) so a
/// future refactor to `>` silently flips it.
#[test]
fn passes_delta_threshold_inclusive_at_boundary() {
    let opts = PhaseDisplayOptions {
        phase_threshold: Some(10.0),
        ..PhaseDisplayOptions::default()
    };
    let metric = METRICS
        .iter()
        .find(|m| m.name == "max_dsq_depth")
        .expect("max_dsq_depth in METRICS");
    // delta=10, a=100 → rel = 10/100 = 0.10. Exactly at the boundary.
    let at_gate = PhaseDeltaRow {
        pairing_key: PairingKey(vec!["t".into()]),
        step_index: 0,
        label: "BASELINE".into(),
        metric,
        a: 100.0,
        b: 110.0,
        delta: 10.0,
        is_regression: true,
    };
    assert!(
        opts.passes_delta_threshold(&at_gate),
        "at-boundary delta (rel=0.10, gate=0.10) must pass via >= comparison"
    );
    // delta=5, a=100 → rel = 0.05 < 0.10. Below the gate.
    let below_gate = PhaseDeltaRow {
        delta: 5.0,
        b: 105.0,
        ..at_gate
    };
    assert!(
        !opts.passes_delta_threshold(&below_gate),
        "below-boundary delta (rel=0.05, gate=0.10) must be suppressed"
    );
}

/// `--phase-threshold 50` against a ~zero baseline `a = 0.0` with a
/// non-negligible delta: the relative spread is treated as `+INFINITY`
/// (a value from nothing is an unbounded relative change), so the row
/// passes any finite gate. Pins that a `0 -> nonzero` phase delta is
/// never suppressed by `--phase-threshold`.
#[test]
fn passes_delta_threshold_zero_a_nonzero_delta_is_unbounded() {
    let opts = PhaseDisplayOptions {
        phase_threshold: Some(50.0),
        ..PhaseDisplayOptions::default()
    };
    let metric = METRICS
        .iter()
        .find(|m| m.name == "max_dsq_depth")
        .expect("max_dsq_depth in METRICS");
    let zero_a = PhaseDeltaRow {
        pairing_key: PairingKey(vec!["t".into()]),
        step_index: 0,
        label: "BASELINE".into(),
        metric,
        a: 0.0,
        b: 10.0,
        delta: 10.0,
        is_regression: true,
    };
    assert!(
        opts.passes_delta_threshold(&zero_a),
        "zero-baseline nonzero delta → rel = +inf ≥ 0.5 → row passes",
    );
}

/// Distinguishing pin for the `+INFINITY`-on-zero-baseline treatment
/// (vs the old `max(|a|, 1.0)` unit floor): a TINY-but-real value
/// appearing from a ~zero baseline. Under the old floor, rel =
/// `0.001 / max(0, 1) = 0.001`, below the 0.5 gate → SUPPRESSED. Under
/// the fix the appearance is unbounded (`+inf ≥ 0.5`) → RENDERED. A
/// value materializing from nothing is always shown under
/// `--phase-threshold`; reverting to the unit floor flips this to false.
#[test]
fn passes_delta_threshold_zero_a_tiny_delta_renders_unbounded() {
    let opts = PhaseDisplayOptions {
        phase_threshold: Some(50.0),
        ..PhaseDisplayOptions::default()
    };
    let metric = METRICS
        .iter()
        .find(|m| m.name == "max_dsq_depth")
        .expect("max_dsq_depth in METRICS");
    let zero_a_tiny = PhaseDeltaRow {
        pairing_key: PairingKey(vec!["t".into()]),
        step_index: 0,
        label: "BASELINE".into(),
        metric,
        a: 0.0,
        b: 0.001,
        delta: 0.001,
        is_regression: true,
    };
    assert!(
        opts.passes_delta_threshold(&zero_a_tiny),
        "zero-baseline tiny (but > ZERO_MEAN_EPS) delta is an unbounded \
             relative change → must render; the old max(1.0) floor wrongly \
             suppressed it (0.001/1 = 0.001 < 0.5)",
    );
}

/// Both sides ~zero (`|a|` AND `|delta|` below `ZERO_MEAN_EPS`) carry no
/// signal: the relative spread is `0.0`, so any POSITIVE
/// `--phase-threshold` filters the row. Pins the `else => 0.0` branch.
#[test]
fn passes_delta_threshold_both_zero_filtered_by_positive_threshold() {
    let opts = PhaseDisplayOptions {
        phase_threshold: Some(50.0),
        ..PhaseDisplayOptions::default()
    };
    let metric = METRICS
        .iter()
        .find(|m| m.name == "max_dsq_depth")
        .expect("max_dsq_depth in METRICS");
    let both_zero = PhaseDeltaRow {
        pairing_key: PairingKey(vec!["t".into()]),
        step_index: 0,
        label: "BASELINE".into(),
        metric,
        a: 0.0,
        b: 0.0,
        delta: 0.0,
        is_regression: false,
    };
    assert!(
        !opts.passes_delta_threshold(&both_zero),
        "both-~zero row has rel = 0.0 < 0.5 → filtered by a positive threshold",
    );
}

/// `--phase-threshold 0` (PCT = 0) renders a both-~zero row: rel = 0.0
/// and `0.0 >= 0.0` is true (the documented "PCT=0 shows every row"
/// contract). Pins the `else => 0.0` branch against the PCT=0 boundary —
/// the row is admitted by the inclusive `>=`, not by any divisor floor.
#[test]
fn passes_delta_threshold_both_zero_at_pct_zero_renders() {
    let opts = PhaseDisplayOptions {
        phase_threshold: Some(0.0),
        ..PhaseDisplayOptions::default()
    };
    let metric = METRICS
        .iter()
        .find(|m| m.name == "max_dsq_depth")
        .expect("max_dsq_depth in METRICS");
    let both_zero = PhaseDeltaRow {
        pairing_key: PairingKey(vec!["t".into()]),
        step_index: 0,
        label: "BASELINE".into(),
        metric,
        a: 0.0,
        b: 0.0,
        delta: 0.0,
        is_regression: false,
    };
    assert!(
        opts.passes_delta_threshold(&both_zero),
        "both-~zero row at --phase-threshold 0 renders: rel = 0.0, 0.0 >= 0.0",
    );
}

/// `--phase-threshold 0` (PCT = 0) admits every row because
/// `rel >= 0.0` is always true. Pins the documented sentinel
/// "PCT = 0 shows every paired row" — a future refactor that
/// special-cased 0 to mean "no filter" via different code
/// path would still produce the same result, but the
/// straight numeric comparison is the simplest implementation
/// and this test pins it.
#[test]
fn passes_delta_threshold_zero_pct_admits_all_rows() {
    let opts = PhaseDisplayOptions {
        phase_threshold: Some(0.0),
        ..PhaseDisplayOptions::default()
    };
    let metric = METRICS
        .iter()
        .find(|m| m.name == "max_dsq_depth")
        .expect("max_dsq_depth in METRICS");
    let zero_delta = PhaseDeltaRow {
        pairing_key: PairingKey(vec!["t".into()]),
        step_index: 0,
        label: "BASELINE".into(),
        metric,
        a: 100.0,
        b: 100.0,
        delta: 0.0,
        is_regression: false,
    };
    assert!(
        opts.passes_delta_threshold(&zero_delta),
        "--phase-threshold 0 must admit even zero-delta rows (rel = 0 >= 0)"
    );
}

// -- format_average_header / format_per_group_pass_counts --

/// `format_average_header` renders the exact header line that
/// `compare_partitions` prints above the comparison table when
/// `--average` is active. Pins the operator-visible surface
/// (the "averaged across N runs (A) and M runs (B)" string)
/// so a regression that reworded the header without
/// updating downstream parsers / scripts lands here.
#[test]
fn format_average_header_exact_string() {
    let out = format_average_header(5, 3, "kernel-6.14", "kernel-6.15");
    assert_eq!(
        out,
        "averaged across 5 runs (kernel-6.14) and 3 runs (kernel-6.15)",
    );
}

/// Zero-contributor sides are surfaced verbatim — operator
/// will see `0 runs` for an empty side. Pins the empty-side
/// edge case so a regression that special-cased `pre_agg = 0`
/// (e.g. omitted the side, said "no contributors") would
/// fail here. The companion empty-rows path is already
/// guarded upstream by `compare_partitions`' `sidecars_*.is_empty()`
/// bail; this test guards the formatter itself in case it's
/// reused outside the compare path.
#[test]
fn format_average_header_zero_contributor_sides_render_verbatim() {
    assert_eq!(
        format_average_header(0, 0, "a", "b"),
        "averaged across 0 runs (a) and 0 runs (b)",
    );
}

/// Helper for the per-group-block tests: build an
/// `AveragedGroup` with the named identity and pass counters
/// while leaving every metric field at zero. Metrics aren't
/// observed by [`format_per_group_pass_counts`] — only the
/// identity tuple and pass counters drive the output.
fn group(
    scenario: &str,
    topology: &str,
    work_type: &str,
    passes_observed: u32,
    total_observed: u32,
) -> AveragedGroup {
    let mut row = make_row(scenario, topology, true, 0.0);
    row.work_type = work_type.into();
    AveragedGroup {
        row,
        passes_observed,
        skips_observed: 0,
        inconclusives_observed: 0,
        failures_observed: 0,
        total_observed,
    }
}

/// Build an [`AveragedGroup`] with non-zero skip / inc / fail
/// counts for the per-group breakdown test below.
fn group_with_breakdown(
    scenario: &str,
    topology: &str,
    work_type: &str,
    passes: u32,
    skips: u32,
    incs: u32,
    fails: u32,
) -> AveragedGroup {
    let mut row = make_row(scenario, topology, true, 0.0);
    row.work_type = work_type.into();
    AveragedGroup {
        row,
        passes_observed: passes,
        skips_observed: skips,
        inconclusives_observed: incs,
        failures_observed: fails,
        total_observed: passes + skips + incs + fails,
    }
}

/// Regression guard for the per-group 4-state breakdown:
/// `format_per_group_pass_counts` must append `(N skip, N inc,
/// N fail)` when any non-pass bucket is non-zero, mirroring
/// `format_dimension_summary`'s contract. A future refactor that
/// drops the breakdown suffix or stops populating the new
/// AveragedGroup counters would silently regress the operator's
/// per-group view back to the pre-fix `passes_observed /
/// total_observed` shape that hid skip / inconc / fail
/// distinctions in the M-N gap.
#[test]
fn format_per_group_pass_counts_renders_skip_inc_fail_breakdown() {
    let g_a = group_with_breakdown("scn", "topo", "wt", 1, 1, 1, 1);
    let g_b = group_with_breakdown("scn", "topo", "wt", 4, 0, 0, 0);
    let out = format_per_group_pass_counts(&[g_a], &[g_b], "A", "B");
    assert!(
        out.contains("A=1/4 (1 skip, 1 inc, 1 fail)"),
        "non-zero buckets must surface in the per-group breakdown for A: {out}"
    );
    assert!(
        out.contains("B=4/4"),
        "all-pass side must stay terse (no breakdown suffix): {out}"
    );
    assert!(
        !out.contains("B=4/4 ("),
        "B has zero non-pass buckets — must NOT append empty breakdown: {out}"
    );
}

/// Empty input: no groups on either side. The formatter
/// returns an empty string so the caller can suppress the
/// block entirely (no header, no body, no separator).
#[test]
fn format_per_group_pass_counts_empty_returns_empty_string() {
    let out = format_per_group_pass_counts(&[], &[], "a", "b");
    assert!(
        out.is_empty(),
        "empty input must yield empty output, got: {out:?}",
    );
}

/// Both-sides-present: every (scenario, topology, work_type)
/// group renders one line. Healthy 5/5 groups appear
/// alongside unhealthy 3/5 groups — the spec is "show every
/// group", not "show only the broken ones".
#[test]
fn format_per_group_pass_counts_renders_every_group_with_n_over_m() {
    let avg_a = vec![
        group("alpha", "tiny-1llc", "SpinWait", 5, 5),
        group("beta", "tiny-1llc", "SpinWait", 3, 5),
    ];
    let avg_b = vec![
        group("alpha", "tiny-1llc", "SpinWait", 4, 5),
        group("beta", "tiny-1llc", "SpinWait", 5, 5),
    ];
    let out = format_per_group_pass_counts(&avg_a, &avg_b, "a", "b");
    // Header line present.
    assert!(
        out.contains("per-group pass counts"),
        "header line must appear, got: {out:?}",
    );
    // Both groups render with their per-side N/M counters.
    assert!(
        out.contains("alpha/tiny-1llc/SpinWait: a=5/5 b=4/5"),
        "alpha group line missing; got: {out:?}",
    );
    assert!(
        out.contains("beta/tiny-1llc/SpinWait: a=3/5 b=5/5"),
        "beta group line missing; got: {out:?}",
    );
    // Trailing newline so the next section reads cleanly.
    assert!(
        out.ends_with('\n'),
        "block must end with newline, got: {out:?}",
    );
}

/// One-side-only group renders `-` for the missing side.
/// Pins the asymmetric-key path: a B-side row that has no
/// A-side match gets `a=-`; symmetric for A-only / B-side.
/// The block surfaces the asymmetry by name so the operator
/// doesn't have to cross-reference the summary's `new_in_b`
/// / `removed_from_a` counters to know which groups went
/// missing.
#[test]
fn format_per_group_pass_counts_one_side_missing_renders_dash() {
    let avg_a = vec![group("only_a", "tiny-1llc", "SpinWait", 5, 5)];
    let avg_b = vec![group("only_b", "tiny-1llc", "SpinWait", 3, 5)];
    let out = format_per_group_pass_counts(&avg_a, &avg_b, "a", "b");
    assert!(
        out.contains("only_a/tiny-1llc/SpinWait: a=5/5 b=-"),
        "A-only group must render b=-; got: {out:?}",
    );
    assert!(
        out.contains("only_b/tiny-1llc/SpinWait: a=- b=3/5"),
        "B-only group must render a=-; got: {out:?}",
    );
}

// -- ComparePartition::as_str --

/// `ComparePartition::as_str` maps each variant to the one-letter
/// label the scalar table headers use, so the per-phase tables and
/// the scalar table share the same operator-facing side identifier.
/// Both arms pinned: a future swap (`A => "B"`) would silently
/// mislabel every unpaired-phase row's SIDE column.
#[test]
fn compare_partition_as_str_maps_each_variant_to_its_letter() {
    assert_eq!(ComparePartition::A.as_str(), "A");
    assert_eq!(ComparePartition::B.as_str(), "B");
}

// -- compare_partitions render-phase + summary block (e2e through
//    the on-disk sidecar pool) --

/// One phase-bucket spec for [`phase_sidecar`]:
/// `(step_index, label, worst_spread, extra per-bucket metrics)`. The
/// extras let a bucket carry a metric its matched-pair bucket lacks,
/// exercising the one-sided-metric-within-a-matched-phase coverage path.
type PhaseSpec<'a> = (u16, &'a str, f64, &'a [(&'a str, f64)]);

/// Build a sidecar carrying scalar `worst_spread` and the given
/// phase buckets, so a row pair produces both a scalar finding and
/// per-phase deltas when the metric moves across sides. `phases`
/// flows verbatim into `GauntletRow.phases` via `sidecar_to_row`,
/// which the per-phase pass in `compare_rows_by` reads. Phase-coverage
/// asymmetry (a step present on one side only) drives the
/// unpaired-phase render path.
fn phase_sidecar(
    test_name: &str,
    scheduler: &str,
    passed: bool,
    spread: f64,
    phases: &[PhaseSpec],
) -> crate::test_support::SidecarResult {
    let phase_buckets = phases
        .iter()
        .map(|&(idx, label, ps, extras)| {
            let mut metrics: Vec<(&str, f64)> = vec![("worst_spread", ps)];
            metrics.extend_from_slice(extras);
            make_phase_bucket(idx, label, &metrics)
        })
        .collect();
    crate::test_support::SidecarResult {
        test_name: test_name.to_string(),
        scheduler: scheduler.to_string(),
        passed,
        stats: crate::assert::ScenarioStats {
            worst_spread: spread,
            total_iterations: 1000,
            phases: phase_buckets,
            ..crate::assert::ScenarioStats::default()
        },
        ..crate::test_support::SidecarResult::test_fixture()
    }
}

/// End-to-end pin on the `compare_partitions` render path that the
/// unit-level helper tests can't reach: the phase-delta table, the
/// one-sided-phase asymmetry table, the one-sided-metric coverage
/// table, the discovery footer hint, and the scalar summary block
/// (excluded_pairs + new_in_b + removed_from_a) all render inside
/// `compare_partitions`, which pools sidecars off disk. The fixture
/// writes a tempdir pool whose per-side filters slice on `scheduler`,
/// so the comparison joins on the remaining pairing dims.
///
/// Runs with `no_average = true`: per-phase buckets only survive the
/// pool -> row path under `--no-average` — the averaging fold drops
/// them (`into_averaged_group` sets `phases: Vec::new()` because
/// buckets do not aggregate cleanly across an averaged group), so
/// `--no-average` is the only mode that feeds the phase block any
/// data. Each side carries exactly one sidecar per pairing key, so
/// `check_no_duplicate_pairing_keys` does not bail. Per-group pass
/// counts are an averaging-only summary line and therefore do not
/// render here; they have their own `format_per_group_pass_counts`
/// unit tests.
///
/// - `paired_scn` exists on both sides with a `worst_spread`
///   10 -> 30 move (scalar regression -> exit 1). Its BASELINE
///   phase matches on both sides (worst_spread 10 -> 30) -> a
///   per-phase regression row; A's BASELINE additionally carries an
///   `avg_nr_running` metric B's BASELINE lacks -> a one-sided-metric
///   PhaseCoverageDiff (the coverage table); B additionally carries a
///   Step[0] bucket A lacks -> a one-sided UnpairedPhaseRow (the
///   asymmetry table).
/// - `excl_scn` exists on both sides but the B side failed
///   (`passed=false`) -> `excluded_pairs` line.
/// - `new_only_b` exists only on the B side -> `new_in_b` line.
/// - `removed_only_a` exists only on the A side -> `removed_from_a`
///   line.
///
/// Asserts exit 1 (the scalar regression drives the return value)
/// and, by running the full render under several `PhaseDisplayOptions`,
/// that the render-phase block + its render-time `matches_phase` /
/// `passes_delta_threshold` projections execute without panicking. The
/// rendered table CONTENT is asserted by the `format_phase_block_lines`
/// unit tests above; this e2e pins the full pool -> `sidecar_to_row`
/// -> `compare_rows_by` -> render chain reaches the phase block.
#[test]
fn compare_partitions_renders_phase_and_summary_blocks_via_pool() {
    let alt_root = tempfile::TempDir::new().expect("create alt-root tempdir");
    // One subdir per sidecar. (test_name, scheduler, passed, scalar
    // spread, phases). scx_alpha is the A side, scx_beta the B side;
    // scheduler is the slicing dim. The B side of paired_scn carries
    // an extra Step[0] bucket so the matched pair has phase-coverage
    // asymmetry (covers the unpaired-phase render path).
    // `const` (not a `let` binding) so the extra-metrics slices are
    // `'static` and outlive the `sidecars` array that borrows them.
    const NO_EXTRA: &[(&str, f64)] = &[];
    const PAIRED_A_EXTRA: &[(&str, f64)] = &[("avg_nr_running", 5.0)];
    let baseline = |s: f64| vec![(0u16, "BASELINE", s, NO_EXTRA)];
    let baseline_plus_step = |s: f64| {
        vec![
            (0u16, "BASELINE", s, NO_EXTRA),
            (1u16, "Step[0]", s, NO_EXTRA),
        ]
    };
    let sidecars = [
        // paired on both sides: 10 -> 30 scalar + BASELINE phase
        // regression; B's extra Step[0] -> unpaired (side B) phase.
        // A's BASELINE additionally carries an avg_nr_running metric that
        // B's BASELINE lacks -> a one-sided-metric-within-a-matched-phase
        // PhaseCoverageDiff, which drives the new per-phase coverage-diff
        // render branch (no-panic; coverage diffs do not gate, so exit is
        // still 1 from the scalar worst_spread regression).
        (
            "paired_scn",
            "scx_alpha",
            true,
            10.0,
            vec![(0u16, "BASELINE", 10.0, PAIRED_A_EXTRA)],
        ),
        (
            "paired_scn",
            "scx_beta",
            true,
            30.0,
            baseline_plus_step(30.0),
        ),
        // present on both sides but B failed -> excluded_pairs.
        ("excl_scn", "scx_alpha", true, 10.0, baseline(10.0)),
        ("excl_scn", "scx_beta", false, 30.0, baseline(30.0)),
        // B-only -> new_in_b.
        ("new_only_b", "scx_beta", true, 10.0, baseline(10.0)),
        // A-only -> removed_from_a.
        ("removed_only_a", "scx_alpha", true, 10.0, baseline(10.0)),
    ];
    for (i, (name, sched, passed, spread, phases)) in sidecars.iter().enumerate() {
        let run_dir = alt_root.path().join(format!("__phase_render_{i}__"));
        std::fs::create_dir_all(&run_dir).expect("create run dir");
        let sc = phase_sidecar(name, sched, *passed, *spread, phases);
        let json = serde_json::to_string(&sc).expect("serialize sidecar");
        std::fs::write(run_dir.join(format!("{name}_{i}.ktstr.json")), json)
            .expect("write sidecar");
    }

    let filter_a = RowFilter {
        schedulers: vec!["scx_alpha".to_string()],
        ..RowFilter::default()
    };
    let filter_b = RowFilter {
        schedulers: vec!["scx_beta".to_string()],
        ..RowFilter::default()
    };

    // Default phase options: render-everything path (phase-delta table +
    // one-sided-phase asymmetry table + one-sided-metric coverage table +
    // footer hint + scalar summary block).
    let exit = compare_partitions(
        &filter_a,
        &filter_b,
        None,
        &ComparisonPolicy::default(),
        Some(alt_root.path()),
        true, // no_average: phase buckets only survive the pool -> row path under --no-average
        &PhaseDisplayOptions::default(),
    )
    .expect("compare_partitions must pool the fixtures and run");
    assert_eq!(
        exit, 1,
        "paired_scn's 10 -> 30 worst_spread move is a scalar regression, \
         so the return value must be 1",
    );

    // --phase-threshold + --steps-only exercise the render-time
    // `passes_delta_threshold` and `matches_phase` projections inside
    // the phase block. --steps-only suppresses the BASELINE bucket
    // (step 0) so paired_scn's BASELINE delta + coverage rows drop,
    // leaving its Step[0] (step 1) unpaired row; the scalar regression
    // — and thus the exit code — is unchanged.
    let opts_filtered = PhaseDisplayOptions {
        steps_only: true,
        phase_threshold: Some(5.0),
        ..PhaseDisplayOptions::default()
    };
    let exit_filtered = compare_partitions(
        &filter_a,
        &filter_b,
        None,
        &ComparisonPolicy::default(),
        Some(alt_root.path()),
        true, // no_average: keep phases for the render-filter projections
        &opts_filtered,
    )
    .expect("compare_partitions must run under render-filter flags");
    assert_eq!(
        exit_filtered, 1,
        "render-time phase filters are projection-only; the scalar \
         regression still drives exit 1",
    );

    // --phases-only suppresses the scalar table, summary block, and
    // host-context delta but still returns the scalar regression
    // count as the exit code.
    let opts_phases_only = PhaseDisplayOptions {
        phases_only: true,
        ..PhaseDisplayOptions::default()
    };
    let exit_phases_only = compare_partitions(
        &filter_a,
        &filter_b,
        None,
        &ComparisonPolicy::default(),
        Some(alt_root.path()),
        true, // no_average: keep phases so --phases-only has a block to project
        &opts_phases_only,
    )
    .expect("compare_partitions must run under --phases-only");
    assert_eq!(
        exit_phases_only, 1,
        "--phases-only hides the scalar render but the regression \
         count still drives the return value",
    );
}

/// `--no-average` against a pool with two sidecars that share a
/// pairing key on one side bails through
/// `check_no_duplicate_pairing_keys` rather than silently latching
/// onto the first. Drives that bail end-to-end through
/// `compare_partitions` so the on-disk -> duplicate-detection wire
/// is pinned (the sibling unit test pins the helper in isolation).
#[test]
fn compare_partitions_no_average_bails_on_duplicate_pairing_keys() {
    let alt_root = tempfile::TempDir::new().expect("create alt-root tempdir");
    // A side: two sidecars with the SAME scenario+topology+work_type
    // (and every other pairing dim equal) -> identical pairing key.
    // B side: one sidecar so the slicing-dim (scheduler) derivation
    // is non-empty.
    let triples = [
        ("dup_scn", "scx_alpha"),
        ("dup_scn", "scx_alpha"),
        ("dup_scn", "scx_beta"),
    ];
    for (i, (name, sched)) in triples.iter().enumerate() {
        let run_dir = alt_root.path().join(format!("__dup_{i}__"));
        std::fs::create_dir_all(&run_dir).expect("create run dir");
        let sc = phase_sidecar(
            name,
            sched,
            true,
            10.0,
            &[(0, "BASELINE", 10.0, &[] as &[(&str, f64)])],
        );
        let json = serde_json::to_string(&sc).expect("serialize sidecar");
        std::fs::write(run_dir.join(format!("{name}_{i}.ktstr.json")), json)
            .expect("write sidecar");
    }
    let filter_a = RowFilter {
        schedulers: vec!["scx_alpha".to_string()],
        ..RowFilter::default()
    };
    let filter_b = RowFilter {
        schedulers: vec!["scx_beta".to_string()],
        ..RowFilter::default()
    };
    let err = compare_partitions(
        &filter_a,
        &filter_b,
        None,
        &ComparisonPolicy::default(),
        Some(alt_root.path()),
        true, // no_average
        &PhaseDisplayOptions::default(),
    )
    .expect_err("two A-side sidecars sharing a pairing key must bail under --no-average");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("duplicate") || rendered.contains("same pairing key"),
        "the bail must name the duplicate-key condition; got: {rendered}",
    );
    assert!(
        rendered.contains("side A"),
        "the bail must name the offending side; got: {rendered}",
    );
}
