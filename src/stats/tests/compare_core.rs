use super::*;

// -- compare_rows tests --

/// Build a row matching the sidecar-derived schema:
/// `work_type = "SpinWait"`, all metrics zeroed except `spread`
/// and `total_iterations`.
fn cmp_row(scenario: &str, topo: &str, passed: bool, spread: f64, iters: u64) -> GauntletRow {
    let mut r = make_row(scenario, topo, passed, spread);
    r.gap_ms = 0;
    r.migrations = 0;
    r.imbalance_ratio = 0.0;
    r.max_dsq_depth = 0;
    r.total_iterations = iters;
    r
}

#[test]
fn compare_rows_dual_gate_both_must_trigger() {
    // worst_spread default_abs=5.0, default_rel=0.25.
    // 10 -> 12: abs delta 2.0 < 5.0 (abs gate fails); rel 0.20 < 0.25
    // (rel gate also fails). Result: 0 regressions, 0 improvements,
    // unchanged for worst_spread.
    let rows_a = vec![cmp_row("test_a", "tiny-1llc", true, 10.0, 0)];
    let rows_b = vec![cmp_row("test_a", "tiny-1llc", true, 12.0, 0)];
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(res.regressions, 0, "abs gate must block 2.0 < 5.0");
    assert_eq!(res.improvements, 0);
    assert_eq!(
        res.unchanged, 1,
        "worst_spread should be classified unchanged"
    );
    assert!(res.findings.is_empty());

    // Confirm the rel gate alone is not enough: spread 10 -> 14 has
    // rel 0.40 (>= 0.25) but abs delta 4.0 (< 5.0), still unchanged.
    let rows_b2 = vec![cmp_row("test_a", "tiny-1llc", true, 14.0, 0)];
    let res2 = compare_rows_by(
        &rows_a,
        &rows_b2,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(
        res2.regressions, 0,
        "rel-only is insufficient: abs gate must also fire"
    );
    assert_eq!(res2.unchanged, 1);
}

/// A metric jumping from a ~zero baseline to a value ABOVE its absolute
/// gate must surface as a regression — not be hidden as "unchanged".
/// Before the fix the relative delta was forced to `0.0` whenever the
/// baseline was ~zero (a divide-by-zero dodge), which always failed the
/// AND-gated relative threshold and silently classified every `0 ->
/// large` jump as unchanged. The fix treats a value appearing from a
/// ~zero baseline as an unbounded relative change (`rel_delta = +inf`),
/// so the absolute gate alone decides.
#[test]
fn compare_rows_zero_baseline_jump_above_abs_gate_is_a_regression() {
    // worst_spread: LowerBetter (higher_is_worse), default_abs = 5.0.
    // 0.0 -> 10.0: delta 10.0 >= 5.0 clears the absolute gate; the
    // ~zero baseline must NOT let the relative gate veto it.
    let rows_a = vec![cmp_row("zbase", "tiny-1llc", true, 0.0, 0)];
    let rows_b = vec![cmp_row("zbase", "tiny-1llc", true, 10.0, 0)];
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(
        res.regressions, 1,
        "0 -> 10 worst_spread (>= abs gate 5.0) must be a regression, not \
         hidden as unchanged by the zero-baseline relative veto",
    );
    assert_eq!(res.improvements, 0);
    assert_eq!(res.unchanged, 0, "the jump must not be counted unchanged");
    assert!(
        res.findings
            .iter()
            .any(|f| f.metric.name == "worst_spread" && f.kind == FindingKind::Regression),
        "a worst_spread Regression finding must be emitted; got {:?}",
        res.findings
            .iter()
            .map(|f| (f.metric.name, f.delta))
            .collect::<Vec<_>>(),
    );
}

/// Contrast: a ~zero-baseline jump BELOW the absolute gate stays
/// unchanged. The absolute gate is the deciding gate from a zero
/// baseline (the relative gate is satisfied by the `+inf` treatment); a
/// sub-`default_abs` new value is noise, not a regression. Together with
/// the sibling above-gate test this pins that the zero-baseline fix
/// reduces the dual gate to "the absolute gate alone decides" — it does
/// not flag every nonzero appearance.
#[test]
fn compare_rows_zero_baseline_jump_below_abs_gate_is_unchanged() {
    // worst_spread default_abs = 5.0; 0.0 -> 3.0 is below it.
    let rows_a = vec![cmp_row("zbase_small", "tiny-1llc", true, 0.0, 0)];
    let rows_b = vec![cmp_row("zbase_small", "tiny-1llc", true, 3.0, 0)];
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(
        res.regressions, 0,
        "0 -> 3 worst_spread (< abs gate 5.0) must stay unchanged",
    );
    assert_eq!(res.improvements, 0);
    assert!(
        res.findings.iter().all(|f| f.metric.name != "worst_spread"),
        "no worst_spread finding for a sub-abs-gate zero-baseline jump; got {:?}",
        res.findings
            .iter()
            .map(|f| (f.metric.name, f.delta))
            .collect::<Vec<_>>(),
    );
}

/// Improvement direction of the zero-baseline fix: a HigherBetter metric
/// rising from a ~zero baseline to a value above its absolute gate is an
/// IMPROVEMENT, not "unchanged". The prior bug forced `rel_delta` to 0.0
/// on a ~zero baseline, hiding zero-baseline jumps in BOTH directions;
/// the sibling tests pin the regression direction, this pins the
/// improvement direction so a future change that re-vetoed only the
/// improvement path would be caught.
#[test]
fn compare_rows_zero_baseline_jump_above_abs_gate_is_an_improvement() {
    // total_iterations: HigherBetter, Counter, default_abs = 100.0 (the only
    // metric reading r.total_iterations). 0 -> 1000: delta +1000 >= 100
    // clears the absolute gate; HigherBetter + delta > 0 => improvement.
    let rows_a = vec![cmp_row("zbase_imp", "tiny-1llc", true, 0.0, 0)];
    let rows_b = vec![cmp_row("zbase_imp", "tiny-1llc", true, 0.0, 1000)];
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(
        res.improvements, 1,
        "0 -> 1000 total_iterations (>= abs gate 100, HigherBetter) must be \
         an improvement, not hidden as unchanged",
    );
    assert_eq!(res.regressions, 0);
    assert!(
        res.findings
            .iter()
            .any(|f| f.metric.name == "total_iterations" && f.kind == FindingKind::Improvement),
        "a total_iterations Improvement finding must be emitted; got {:?}",
        res.findings
            .iter()
            .map(|f| (f.metric.name, f.delta))
            .collect::<Vec<_>>(),
    );
}

/// compare must NOT flag a sub-integer `stuck_count` difference as a
/// regression. A-side cross-run mean 1.4 vs B-side 1.6 (true delta
/// 0.2, well under `default_abs` = 1.0) classifies UNCHANGED. Before
/// the f64 fix the fold rounded these means to 1 vs 2 (delta 1),
/// which cleared BOTH the abs (1.0, since 1.0 is not < 1.0) and rel
/// (100% >= 50%) gates and fabricated a regression from noise. The
/// f64 `stuck_count` carries the exact mean so compare reads 0.2.
#[test]
fn compare_rows_subinteger_stuck_count_difference_is_unchanged() {
    let mut a = make_row("test_a", "tiny-1llc", true, 10.0);
    a.stuck_count = 1.4;
    let mut b = make_row("test_a", "tiny-1llc", true, 10.0);
    b.stuck_count = 1.6;
    let res = compare_rows_by(
        &[a],
        &[b],
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(
        res.regressions, 0,
        "a 0.2 sub-integer stuck_count delta must NOT be a regression",
    );
    assert_eq!(res.improvements, 0);
    assert!(
        res.findings.iter().all(|f| f.metric.name != "stuck_count"),
        "stuck_count must not be a finding for a sub-abs delta; got {:?}",
        res.findings
            .iter()
            .map(|f| f.metric.name)
            .collect::<Vec<_>>(),
    );
}

/// Contrast: a genuine whole-stall `stuck_count` regression IS still
/// flagged. A-side mean 1.0 vs B-side 2.5 (delta 1.5 >= abs 1.0, rel
/// 150% >= 50%) is a regression — the f64 fix preserves the
/// deliberate single-whole-stall sensitivity (`default_abs` = 1.0),
/// it only stops fabricating regressions from sub-integer noise.
#[test]
fn compare_rows_genuine_stuck_count_regression_is_flagged() {
    let mut a = make_row("test_a", "tiny-1llc", true, 10.0);
    a.stuck_count = 1.0;
    let mut b = make_row("test_a", "tiny-1llc", true, 10.0);
    b.stuck_count = 2.5;
    let res = compare_rows_by(
        &[a],
        &[b],
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(
        res.regressions, 1,
        "a 1.5 stuck_count delta clears both gates and must be a regression",
    );
    assert!(
        res.findings
            .iter()
            .any(|f| f.metric.name == "stuck_count" && f.kind == FindingKind::Regression),
        "stuck_count must be the flagged regression",
    );
}

#[test]
fn compare_rows_synthetic_regression_and_improvement() {
    // spread 10 -> 30: abs delta 20.0 >= 5.0, rel 2.0 >= 0.10 →
    // regression (higher_is_worse).
    // total_iterations 1000 -> 500: abs delta 500 >= 100, rel 0.5
    // >= 0.10, higher_is_worse=false so decrease is a regression.
    // Net: 2 regressions, 0 improvements; one Finding per
    // significant metric.
    let rows_a = vec![cmp_row("test1", "tiny-1llc", true, 10.0, 1000)];
    let rows_b = vec![cmp_row("test1", "tiny-1llc", true, 30.0, 500)];
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::uniform(10.0),
    );
    assert_eq!(
        res.regressions, 2,
        "spread up + iterations down both regress"
    );
    assert_eq!(res.improvements, 0);
    assert_eq!(res.excluded_pairs, 0);
    let metrics: Vec<&str> = res.findings.iter().map(|d| d.metric.name).collect();
    assert!(metrics.contains(&"worst_spread"));
    assert!(metrics.contains(&"total_iterations"));
    for d in &res.findings {
        assert!(
            d.kind == FindingKind::Regression,
            "all reported deltas should be regressions"
        );
        assert_eq!(d.scenario, "test1");
        assert_eq!(d.topology, "tiny-1llc");
    }

    // Reverse direction: improvements should also surface.
    let res_imp = compare_rows_by(
        &rows_b,
        &rows_a,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::uniform(10.0),
    );
    assert_eq!(res_imp.regressions, 0);
    assert_eq!(res_imp.improvements, 2);
    for d in &res_imp.findings {
        assert!(d.kind != FindingKind::Regression);
    }
}

/// Rate-COMPONENT metrics are suppressed from compare findings, but the
/// user-facing rate is not. `total_iterations_pooled` (a suppressed
/// component) differs 1000->2000 — past the default gate, normally a
/// finding — yet emits none; the pooled rate `iterations_per_cpu_sec`
/// differs 500->1000 and DOES emit. Pins the compare-emit suppression while
/// the components stay in `ext_metrics` for the cross-run re-pool.
#[test]
fn compare_rows_suppresses_rate_components_not_the_rate() {
    let mut a = cmp_row("t", "tiny-1llc", true, 0.0, 1000);
    a.ext_metrics
        .insert("total_iterations_pooled".to_string(), 1000.0);
    a.ext_metrics
        .insert("iterations_per_cpu_sec".to_string(), 500.0);
    let mut b = cmp_row("t", "tiny-1llc", true, 0.0, 1000);
    b.ext_metrics
        .insert("total_iterations_pooled".to_string(), 2000.0);
    b.ext_metrics
        .insert("iterations_per_cpu_sec".to_string(), 1000.0);
    let res = compare_rows_by(
        &[a],
        &[b],
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    let names: Vec<&str> = res.findings.iter().map(|d| d.metric.name).collect();
    assert!(
        !names.contains(&"total_iterations_pooled"),
        "the Rate component must be suppressed from compare findings; got {names:?}",
    );
    assert!(
        names.contains(&"iterations_per_cpu_sec"),
        "the user-facing pooled rate must still emit a finding; got {names:?}",
    );
}

#[test]
fn compare_rows_higher_is_worse_inversion() {
    // total_iterations is higher_is_worse=false. A drop of 1000 ->
    // 500 must be reported as a regression, not an improvement.
    let rows_a = vec![cmp_row("t", "tiny-1llc", true, 0.0, 1000)];
    let rows_b = vec![cmp_row("t", "tiny-1llc", true, 0.0, 500)];
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    let iters_delta = res
        .findings
        .iter()
        .find(|d| d.metric.name == "total_iterations")
        .expect("total_iterations should produce a delta");
    assert!(
        iters_delta.kind == FindingKind::Regression,
        "iterations decrease is a regression"
    );
    assert_eq!(iters_delta.delta, -500.0);
    assert_eq!(res.regressions, 1);
    assert_eq!(res.improvements, 0);

    // worst_spread is higher_is_worse=true. An increase must be a
    // regression; a decrease must be an improvement.
    let rows_a2 = vec![cmp_row("t", "tiny-1llc", true, 10.0, 0)];
    let rows_b2 = vec![cmp_row("t", "tiny-1llc", true, 30.0, 0)];
    let res_up = compare_rows_by(
        &rows_a2,
        &rows_b2,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    let spread_up = res_up
        .findings
        .iter()
        .find(|d| d.metric.name == "worst_spread")
        .expect("worst_spread should produce a delta");
    assert!(
        spread_up.kind == FindingKind::Regression,
        "spread increase is a regression"
    );
    assert_eq!(spread_up.delta, 20.0);

    let res_down = compare_rows_by(
        &rows_b2,
        &rows_a2,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    let spread_down = res_down
        .findings
        .iter()
        .find(|d| d.metric.name == "worst_spread")
        .expect("worst_spread should produce a delta");
    assert!(
        spread_down.kind != FindingKind::Regression,
        "spread decrease is an improvement"
    );
    assert_eq!(spread_down.delta, -20.0);
}

#[test]
fn compare_rows_skipped_side_drops_pair_into_excluded_pairs() {
    // A skipped row on either side of the comparison must not
    // contribute to regressions/improvements — a skipped run
    // carries no executed metrics, so the pair must short-circuit
    // via the is_skip() gate before regression math touches the
    // default-zero metric values.
    let mut row_a = cmp_row("t", "tiny-1llc", true, 10.0, 100);
    let mut row_b = cmp_row("t", "tiny-1llc", true, 10.0, 100);
    row_a.skipped = true; // A side was skipped
    let res = compare_rows_by(
        &[row_a.clone()],
        &[row_b.clone()],
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(res.regressions, 0);
    assert_eq!(res.improvements, 0);
    assert_eq!(
        res.excluded_pairs, 1,
        "skipped side must count as excluded_pairs, not produce deltas"
    );

    // Symmetrically on the B side.
    row_a.skipped = false;
    row_b.skipped = true;
    let res = compare_rows_by(
        &[row_a],
        &[row_b],
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(res.regressions, 0);
    assert_eq!(res.improvements, 0);
    assert_eq!(res.excluded_pairs, 1);
}

/// Rows where either side has `passed=false` are dropped from the
/// regression math. A failed scenario's metrics reflect the failure
/// mode (short run, stalled workload, missing samples), not
/// scheduler behavior.
#[test]
fn compare_rows_skips_failed_scenarios() {
    // Three scenarios, all with the same metric movement. Only
    // test_ok (passed on both sides) should be eligible for the
    // regression math; the other two are counted as excluded_pairs.
    let rows_a = vec![
        cmp_row("test_ok", "tiny-1llc", true, 10.0, 1000),
        cmp_row("test_failed_b", "tiny-1llc", true, 10.0, 1000),
        cmp_row("test_failed_a", "tiny-1llc", false, 10.0, 1000),
    ];
    let rows_b = vec![
        cmp_row("test_ok", "tiny-1llc", true, 30.0, 500),
        cmp_row("test_failed_b", "tiny-1llc", false, 30.0, 500),
        cmp_row("test_failed_a", "tiny-1llc", true, 30.0, 500),
    ];
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::uniform(10.0),
    );
    assert_eq!(
        res.excluded_pairs, 2,
        "test_failed_a and test_failed_b skip"
    );
    // test_ok regresses on worst_spread and total_iterations only.
    assert_eq!(res.regressions, 2);
    assert_eq!(res.improvements, 0);
    for d in &res.findings {
        assert_eq!(d.scenario, "test_ok");
    }
}

#[test]
fn compare_rows_filter_substring() {
    // Two scenarios in each run. Filter "alpha" must match the
    // alpha row (substring of the joined "scenario topology
    // scheduler work_type" string) and exclude the beta row.
    let rows_a = vec![
        cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
        cmp_row("beta", "tiny-1llc", true, 10.0, 0),
    ];
    let rows_b = vec![
        cmp_row("alpha", "tiny-1llc", true, 30.0, 0),
        cmp_row("beta", "tiny-1llc", true, 30.0, 0),
    ];
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        Some("alpha"),
        &ComparisonPolicy::default(),
    );
    assert_eq!(res.regressions, 1, "only alpha row should compare");
    assert_eq!(res.findings.len(), 1);
    assert_eq!(res.findings[0].scenario, "alpha");
    // Finding carries work_type so two findings sharing
    // scenario+topology under different workloads stay
    // distinguishable.
    assert_eq!(res.findings[0].work_type, "SpinWait");

    // Filter on topology substring is also honored. Both rows
    // share the "tiny-1llc" topology and only worst_spread crosses
    // both gates (10 -> 30 with default_abs=5.0, default_rel=0.25),
    // so each row contributes exactly one finding.
    let res_topo = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        Some("tiny"),
        &ComparisonPolicy::default(),
    );
    assert_eq!(res_topo.regressions, 2, "both rows match 'tiny' topology");
    assert_eq!(res_topo.findings.len(), 2);

    // Non-matching filter yields no comparisons at all.
    let res_none = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        Some("nomatch"),
        &ComparisonPolicy::default(),
    );
    assert_eq!(res_none.regressions, 0);
    assert_eq!(res_none.improvements, 0);
    assert_eq!(res_none.unchanged, 0);
    assert_eq!(res_none.excluded_pairs, 0);
}

#[test]
fn compare_rows_threshold_override() {
    // worst_spread default_rel=0.25, default_abs=5.0. Move 100 ->
    // 106: abs delta 6.0 >= 5.0 (abs gate passes); rel 0.06 < 0.25
    // (default rel fails) → unchanged with default thresholds.
    let rows_a = vec![cmp_row("t", "tiny-1llc", true, 100.0, 0)];
    let rows_b = vec![cmp_row("t", "tiny-1llc", true, 106.0, 0)];
    let res_default = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    let spread_default = res_default
        .findings
        .iter()
        .find(|d| d.metric.name == "worst_spread");
    assert!(
        spread_default.is_none(),
        "default rel 0.25 must classify 6% change as unchanged"
    );

    // Override threshold to 5% (Some(5.0) → rel_thresh 0.05). Now
    // rel 0.06 >= 0.05, both gates fire → regression.
    let res_override = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::uniform(5.0),
    );
    let spread_override = res_override
        .findings
        .iter()
        .find(|d| d.metric.name == "worst_spread")
        .expect("override 5% must surface 6% spread change");
    assert!(spread_override.kind == FindingKind::Regression);
    assert_eq!(spread_override.delta, 6.0);

    // The override does NOT loosen the abs gate. Move 1.0 -> 1.5:
    // abs delta 0.5 < 5.0; even threshold=1% (rel_thresh 0.01)
    // can't promote it to significant.
    let rows_a_small = vec![cmp_row("t", "tiny-1llc", true, 1.0, 0)];
    let rows_b_small = vec![cmp_row("t", "tiny-1llc", true, 1.5, 0)];
    let res_small = compare_rows_by(
        &rows_a_small,
        &rows_b_small,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::uniform(1.0),
    );
    assert!(
        !res_small
            .findings
            .iter()
            .any(|d| d.metric.name == "worst_spread"),
        "abs gate must still block tiny absolute moves"
    );
}

/// `ComparisonPolicy::rel_threshold` resolution priority pinned
/// by exhaustive enumeration: per-metric override wins over
/// `default_percent`, which wins over the registry fallback.
/// A regression that inverted the priority or shortcut the
/// fallback (e.g. always returning `default_percent` even when
/// a per-metric override exists) surfaces here, not as subtly-
/// wrong thresholds inside `compare_rows`.
#[test]
fn comparison_policy_rel_threshold_resolution_priority() {
    // Empty policy → registry fallback. `default_rel` is
    // passed by the caller (compare_rows supplies it from
    // `m.default_rel`), so we pick an arbitrary fallback here
    // and check it's returned verbatim.
    let empty = ComparisonPolicy::default();
    assert_eq!(
        empty.rel_threshold("worst_spread", 0.25),
        0.25,
        "empty policy must fall through to the registry default_rel",
    );

    // Uniform override → default_percent / 100 wins over
    // the registry default.
    let uniform = ComparisonPolicy::uniform(10.0);
    assert_eq!(
        uniform.rel_threshold("worst_spread", 0.25),
        0.10,
        "uniform(10.0) must override the registry default_rel \
             with 10.0 / 100.0 = 0.10",
    );

    // Per-metric override wins over both `default_percent` and
    // the registry default. Use two metric names so the test
    // also proves other metrics still see `default_percent`
    // when no per-metric entry matches.
    let mut per_metric = ComparisonPolicy::uniform(10.0);
    per_metric
        .per_metric_percent
        .insert("worst_spread".to_string(), 5.0);
    assert_eq!(
        per_metric.rel_threshold("worst_spread", 0.25),
        0.05,
        "per-metric override (5.0) must win over default_percent \
             (10.0) and the registry default (0.25)",
    );
    assert_eq!(
        per_metric.rel_threshold("worst_gap_ms", 0.25),
        0.10,
        "metrics not in the per-metric map must still see the \
             default_percent (10.0 → 0.10), not the registry default",
    );
}

/// `worst_wake_latency_tail_ratio` is ext_metrics-sourced
/// (`MetricKind::WakeLatencyTailRatio`, accessor `|_| None`). The
/// min-iterations noise floor is enforced at the PRODUCER
/// (`populate_run_distribution_metrics` emits no key below the floor —
/// pinned by `wake_latency_tail_ratio_producer_floor_gates_and_maxes` in
/// the assert tests), so on the COMPARE side a sub-threshold (or no-tail)
/// run presents as an ABSENT ext key. This pins the compare-side
/// consequence: an absent key reads as `None` and emits no finding, while a
/// present key with a real delta surfaces as a regression. `MetricDef::read`
/// resolves the value purely from `ext_metrics` (the accessor is `|_| None`).
#[test]
fn wake_latency_tail_ratio_compares_via_ext_metrics() {
    let metric = metric_def("worst_wake_latency_tail_ratio")
        .expect("worst_wake_latency_tail_ratio must be registered in METRICS");
    let key = "worst_wake_latency_tail_ratio";

    // Absent ext key (the producer's sub-threshold / no-tail output): both
    // sides read None, so the `(None, None)` arm skips the pair (no finding,
    // no coverage diff).
    let low_a = make_row("tail_low", "tiny-1llc", true, 0.0);
    let low_b = make_row("tail_low", "tiny-1llc", true, 0.0);
    assert!(
        metric.read(&low_a).is_none(),
        "absent ext key must read as None (accessor is |_| None, no ext entry)",
    );
    let below = compare_rows_by(
        std::slice::from_ref(&low_a),
        std::slice::from_ref(&low_b),
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(
        below.regressions, 0,
        "absent tail-ratio key (identical rows) must surface no regression",
    );
    assert!(
        below.findings.is_empty(),
        "absent tail-ratio key (identical rows) must emit no findings",
    );

    // Present ext key with a 10x delta (the only difference between two
    // otherwise-identical rows): read() returns the ext value and the delta
    // surfaces as a regression.
    let mut hi_a = make_row("tail_hi", "tiny-1llc", true, 0.0);
    let mut hi_b = make_row("tail_hi", "tiny-1llc", true, 0.0);
    hi_a.ext_metrics.insert(key.to_string(), 2.0);
    hi_b.ext_metrics.insert(key.to_string(), 20.0);
    assert_eq!(
        metric.read(&hi_a),
        Some(2.0),
        "present ext key must read via the ext fallback",
    );
    let above = compare_rows_by(
        std::slice::from_ref(&hi_a),
        std::slice::from_ref(&hi_b),
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(
        above.regressions, 1,
        "a present-key 10x tail blow-up must surface as a regression; \
             threshold wiring has a gap otherwise",
    );
}

/// Read-contract pin: a metric absent on BOTH sides reads `None` and the
/// pair is skipped entirely (no verdict, no coverage diff).
///
/// `compare_rows` calls `m.read(row)` for every metric; `read` returns `None`
/// for an absent metric. `worst_wake_latency_tail_ratio` is ext-sourced with a
/// `|_| None` accessor, so an absent ext key (the producer's sub-threshold
/// output) is the `None` condition. When BOTH sides read `None`, the
/// `(None, None) => continue` arm skips the pair before any verdict or
/// coverage-diff bookkeeping. This pins the `read()==None`-on-absent
/// precondition the absent-vs-genuine-zero handling rests on: a regression
/// that synthesized a value for an absent key (the old `unwrap_or(0.0)`) or
/// panicked on `None` would fail here.
///
/// Asserts:
/// 1. `metric.read(&row)` returns `None` on both sides (no ext key).
/// 2. `compare_rows` does NOT panic.
/// 3. No finding AND no coverage diff — both-absent is skipped, distinct from
///    one-sided-absent which IS a coverage diff (see
///    `compare_rows_one_sided_absent_is_coverage_diff_not_verdict`).
#[test]
fn compare_rows_both_absent_ext_key_reads_none_and_is_skipped() {
    let metric =
        metric_def("worst_wake_latency_tail_ratio").expect("tail ratio metric must be registered");

    // Neither row carries the tail-ratio ext key, so read() is None on both
    // sides (accessor |_| None + absent ext entry). make_row no longer
    // paints this key — the producer alone decides its presence.
    let row_a = make_row("none_branch", "tiny-1llc", true, 0.0);
    let row_b = make_row("none_branch", "tiny-1llc", true, 0.0);

    assert!(
        metric.read(&row_a).is_none(),
        "absent ext key must read None on A — otherwise this test is not \
             exercising the None branch of compare_rows",
    );
    assert!(
        metric.read(&row_b).is_none(),
        "absent ext key must read None on B",
    );

    // Both sides read None: the `(None, None) => continue` arm skips the pair
    // before any verdict or coverage-diff bookkeeping — no panic, no finding,
    // and (unlike one-sided-absent) no coverage diff.
    let report = compare_rows_by(
        std::slice::from_ref(&row_a),
        std::slice::from_ref(&row_b),
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(
        report.regressions, 0,
        "both-absent must not be a regression",
    );
    assert_eq!(
        report.improvements, 0,
        "both-absent must not be an improvement",
    );
    assert!(
        report.findings.is_empty(),
        "no findings when the metric is absent on both sides; got: {:?}",
        report.findings,
    );
    assert!(
        report.coverage_diffs.is_empty(),
        "both-absent is skipped, NOT a coverage diff (that is one-sided-absent)",
    );
}

/// `ComparisonPolicy::load_json` round-trips a policy file: a
/// policy constructed in memory, serialized, and reloaded must
/// yield the same thresholds end-to-end. Pins the wire format
/// for the `--policy <path>` CLI flag.
#[test]
fn comparison_policy_load_json_round_trip() {
    let mut original = ComparisonPolicy::uniform(10.0);
    original
        .per_metric_percent
        .insert("worst_spread".to_string(), 5.0);
    original
        .per_metric_percent
        .insert("worst_p99_wake_latency_us".to_string(), 20.0);

    let json = serde_json::to_string(&original).expect("serialize policy");

    let tmp = tempfile::NamedTempFile::new().expect("create tempfile");
    std::fs::write(tmp.path(), json).expect("write policy file");

    let loaded = ComparisonPolicy::load_json(tmp.path()).expect("load policy");

    assert_eq!(
        loaded.default_percent,
        Some(10.0),
        "default_percent must round-trip",
    );
    assert_eq!(
        loaded.per_metric_percent.get("worst_spread"),
        Some(&5.0),
        "per-metric worst_spread override must round-trip",
    );
    assert_eq!(
        loaded.per_metric_percent.get("worst_p99_wake_latency_us"),
        Some(&20.0),
        "per-metric worst_p99 override must round-trip",
    );
    // Resolution-path equivalence: the loaded policy resolves
    // every metric identically to the original.
    for metric_name in ["worst_spread", "worst_p99_wake_latency_us", "worst_gap_ms"] {
        assert_eq!(
            loaded.rel_threshold(metric_name, 0.25),
            original.rel_threshold(metric_name, 0.25),
            "load_json round-trip must preserve threshold \
                 resolution for {metric_name}",
        );
    }
}

/// `ComparisonPolicy::load_json` on a nonexistent path must
/// surface an actionable error naming the path (not a generic
/// "no such file"). Pins the `with_context` chain — a
/// regression that dropped the context would collapse a
/// user-facing `--policy missing.json` invocation into a
/// bare `No such file or directory` with no clue about where
/// the missing file was expected.
#[test]
fn comparison_policy_load_json_nonexistent_path_surfaces_path() {
    let path = std::path::Path::new("/nonexistent/ktstr/policy-DOES-NOT-EXIST.json");
    let err = ComparisonPolicy::load_json(path).expect_err("nonexistent path must fail");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains(&path.display().to_string()),
        "error must name the missing path so a user can see \
             which file was expected; got: {rendered}",
    );
    assert!(
        rendered.to_ascii_lowercase().contains("read")
            || rendered.to_ascii_lowercase().contains("no such"),
        "error must describe the read failure (either the \
             `with_context` \"read comparison policy from ...\" \
             prefix or std's underlying \"No such file...\" \
             reason); got: {rendered}",
    );
}

/// `ComparisonPolicy::load_json` on a malformed JSON body
/// must include both the path (for locating) AND the parse
/// context (for understanding the failure shape). A
/// `serde_json::Error` on its own gives line/column but no
/// file identity; the `with_context` adds the path. Pins
/// both halves.
#[test]
fn comparison_policy_load_json_malformed_json_surfaces_path_and_parse_context() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    // Not JSON — clearly malformed.
    std::fs::write(tmp.path(), "this is not json at all {{{").expect("write");
    let err = ComparisonPolicy::load_json(tmp.path()).expect_err("malformed JSON must fail");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains(&tmp.path().display().to_string()),
        "malformed-JSON error must name the path; got: {rendered}",
    );
    assert!(
        rendered.to_ascii_lowercase().contains("parse")
            || rendered.to_ascii_lowercase().contains("expected"),
        "malformed-JSON error must include a parse-context \
             hint (either the `with_context` \"parse comparison \
             policy from ...\" prefix, or serde_json's \"expected \
             ...\" reason); got: {rendered}",
    );
}

/// `load_json` rejects unknown top-level fields per
/// `deny_unknown_fields`. A misspelled field (e.g.
/// `default_percentage` vs `default_percent`) must surface as
/// a parse error, not silently drop the value and fall back
/// to the default.
#[test]
fn comparison_policy_load_json_rejects_unknown_fields() {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), r#"{"default_percentage": 10.0}"#).expect("write");
    let err = ComparisonPolicy::load_json(tmp.path()).expect_err("unknown field must fail");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("default_percentage")
            || rendered.to_ascii_lowercase().contains("unknown"),
        "unknown-field error must name the typo so a user \
             can fix the policy file; got: {rendered}",
    );
}

/// `validate` rejects negative `default_percent`. A regression
/// that lost the sign check would let `--threshold -10`
/// through to `compare_rows`' dual-gate `.abs()` comparison,
/// where a negative `rel_thresh` makes every delta (including
/// zero) significant — silently inverting the comparison.
#[test]
fn comparison_policy_validate_rejects_negative_default_percent() {
    let policy = ComparisonPolicy::uniform(-10.0);
    let err = policy
        .validate()
        .expect_err("negative default_percent must fail validation");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("default_percent"),
        "validation error must name the field; got: {rendered}",
    );
    assert!(
        rendered.contains("-10"),
        "validation error must echo the rejected value; got: {rendered}",
    );
}

/// `validate` rejects unknown per-metric keys. A typo in the
/// policy file would otherwise silently fall through to
/// `default_percent` — a user debugging a regression with
/// `--policy typo.json` would see the uniform threshold
/// applied instead of the expected override and have no way
/// to know why.
#[test]
fn comparison_policy_validate_rejects_unknown_per_metric_keys() {
    let mut policy = ComparisonPolicy::default();
    policy
        .per_metric_percent
        .insert("wrost_spread".to_string(), 5.0); // typo
    let err = policy
        .validate()
        .expect_err("unknown per-metric key must fail validation");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("wrost_spread"),
        "validation error must echo the unknown key so a user \
             can see the typo; got: {rendered}",
    );
    // Known-metric list should appear so the user can pick the
    // right spelling. Registered metric names include
    // `worst_spread` — a hint toward the correct key.
    assert!(
        rendered.contains("worst_spread"),
        "validation error should include the registered \
             metric list so users can find the right spelling; \
             got: {rendered}",
    );
}

/// `validate` rejects negative per-metric overrides. Covers
/// the sibling case of the default_percent sign check above.
#[test]
fn comparison_policy_validate_rejects_negative_per_metric_value() {
    let mut policy = ComparisonPolicy::default();
    policy
        .per_metric_percent
        .insert("worst_spread".to_string(), -5.0);
    let err = policy
        .validate()
        .expect_err("negative per-metric percent must fail");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("worst_spread") && rendered.contains("-5"),
        "validation error must name both the key and the \
             rejected value; got: {rendered}",
    );
}

/// Defence-in-depth against an on-disk policy missing fields
/// (e.g. older wire format, hand-edited JSON). The struct uses
/// `#[serde(default)]` on every field so a partial JSON
/// (`{}`, `{"default_percent": 5}`) deserializes to a policy
/// with the missing field at its `Default` value. A regression
/// that dropped the `#[serde(default)]` attribute would make
/// `load_json` reject otherwise-valid partial policies.
#[test]
fn comparison_policy_load_json_accepts_partial_fields() {
    let tmp = tempfile::NamedTempFile::new().expect("create tempfile");
    // Empty object → policy with every default.
    std::fs::write(tmp.path(), "{}").expect("write empty policy");
    let loaded = ComparisonPolicy::load_json(tmp.path()).expect("load empty policy");
    assert_eq!(loaded.default_percent, None);
    assert!(loaded.per_metric_percent.is_empty());

    // Only default_percent set → empty per_metric.
    std::fs::write(tmp.path(), r#"{"default_percent": 7.5}"#).expect("write partial policy");
    let loaded = ComparisonPolicy::load_json(tmp.path()).expect("load partial policy");
    assert_eq!(loaded.default_percent, Some(7.5));
    assert!(loaded.per_metric_percent.is_empty());

    // Only per_metric_percent set → default_percent None.
    std::fs::write(
        tmp.path(),
        r#"{"per_metric_percent": {"worst_spread": 3.0}}"#,
    )
    .expect("write per-metric-only policy");
    let loaded = ComparisonPolicy::load_json(tmp.path()).expect("load per-metric-only policy");
    assert_eq!(loaded.default_percent, None);
    assert_eq!(loaded.per_metric_percent.get("worst_spread"), Some(&3.0),);
}

/// `from_cli_flags` resolves the `--threshold` / `--policy` pair
/// the shared way for `stats compare` and `perf-delta`:
/// threshold → uniform (validated), policy → load_json, neither →
/// registry defaults, both → error (the clap-`conflicts_with`
/// backstop). Pin every branch so a future edit can't silently
/// drop the sign check or the mutual-exclusion guard.
#[test]
fn comparison_policy_from_cli_flags_resolves_each_branch() {
    // --threshold N → uniform default_percent = N.
    let p = ComparisonPolicy::from_cli_flags(Some(15.0), None).expect("threshold resolves");
    assert_eq!(p.default_percent, Some(15.0));
    assert!(p.per_metric_percent.is_empty());

    // A negative --threshold is rejected via validate().
    assert!(
        ComparisonPolicy::from_cli_flags(Some(-1.0), None).is_err(),
        "negative --threshold must be rejected before the dual-gate math",
    );

    // --policy PATH → load_json.
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    std::fs::write(tmp.path(), r#"{"default_percent": 8.0}"#).expect("write policy");
    let p = ComparisonPolicy::from_cli_flags(None, Some(tmp.path())).expect("policy file resolves");
    assert_eq!(p.default_percent, Some(8.0));

    // Neither → registry defaults (no uniform override).
    let p = ComparisonPolicy::from_cli_flags(None, None).expect("default resolves");
    assert_eq!(p.default_percent, None);

    // Both set → error: clap `conflicts_with` makes this
    // unreachable at the CLI, but the library entry point must not
    // silently prefer one over the other.
    assert!(
        ComparisonPolicy::from_cli_flags(Some(10.0), Some(tmp.path())).is_err(),
        "--threshold + --policy together must error",
    );
}

/// End-to-end pin: `compare_rows` with a per-metric policy
/// must apply the override for the matching metric AND fall
/// through to `default_percent` for every other metric. The
/// unit-level `comparison_policy_rel_threshold_resolution_priority`
/// test above pins the resolution function in isolation; this
/// test runs it through the actual compare_rows pipeline with
/// rows that trigger distinct deltas on two metrics, proving
/// that `compare_rows` reads `m.name` correctly and hands it
/// to `policy.rel_threshold`. A regression that hard-coded a
/// single metric name, or passed the wrong name to the
/// resolver, would surface here as the wrong regression count.
///
/// Fixture:
/// - A: `worst_spread = 100`, `worst_median_wake_latency_us = 100`
/// - B: `worst_spread = 106` (6% delta, passes the abs gate
///   at 5.0), `worst_median_wake_latency_us = 110` (10%
///   delta).
/// - Policy: `default_percent = 20%`, per_metric
///   `worst_spread = 5%`.
///
/// Expected: `worst_spread`'s 6% delta beats the 5%
/// per-metric override → regression. `worst_median_wake_latency_us`'s
/// 10% delta falls under the 20% default → unchanged. Total
/// regressions = 1.
#[test]
fn compare_rows_per_metric_policy_resolves_each_metric_independently() {
    // Construct rows with both metrics non-default so we can
    // trigger per-metric and default_percent branches in one
    // row pair.
    let mut row_a = cmp_row("t", "tiny-1llc", true, 100.0, 0);
    row_a
        .ext_metrics
        .insert("worst_median_wake_latency_us".to_string(), 100.0);
    let mut row_b = cmp_row("t", "tiny-1llc", true, 106.0, 0);
    row_b
        .ext_metrics
        .insert("worst_median_wake_latency_us".to_string(), 110.0);

    let mut policy = ComparisonPolicy::uniform(20.0);
    policy
        .per_metric_percent
        .insert("worst_spread".to_string(), 5.0);

    let res = compare_rows_by(&[row_a], &[row_b], LEGACY_PAIRING_DIMS, None, &policy);

    let spread_finding = res
        .findings
        .iter()
        .find(|f| f.metric.name == "worst_spread");
    assert!(
        spread_finding.is_some(),
        "worst_spread per-metric override (5%) must fire on 6% \
             delta; got findings: {:?}",
        res.findings
            .iter()
            .map(|f| f.metric.name)
            .collect::<Vec<_>>(),
    );
    let spread_finding = spread_finding.unwrap();
    assert!(
        spread_finding.kind == FindingKind::Regression,
        "6% > 5% → regression"
    );

    // worst_median_wake_latency_us has a 10% delta; under
    // default_percent = 20%, it must be unchanged (not in
    // findings).
    let wake_finding = res
        .findings
        .iter()
        .find(|f| f.metric.name == "worst_median_wake_latency_us");
    assert!(
        wake_finding.is_none(),
        "worst_median_wake_latency_us 10% delta must fall \
             under default_percent 20% and be unchanged. The \
             regression would indicate `compare_rows` ignored \
             default_percent for non-per-metric entries; got \
             finding: {wake_finding:?}",
    );

    assert_eq!(
        res.regressions, 1,
        "exactly one regression expected — the per-metric \
             spread override should win on spread, and the \
             default_percent should suppress wake latency. Got: \
             regressions={}, improvements={}, unchanged={}",
        res.regressions, res.improvements, res.unchanged,
    );
}

/// `compare_rows_by` builds a `HashMap<PairingKey, &GauntletRow>`
/// from `rows_a` via `entry(key).or_insert(row_a)`, then looks up
/// each B-side row with `get`, so when `rows_a` contains two
/// entries with the same `(scenario, topology, work_type)` key
/// the first (earlier-iterated) one wins. Lock that contract in:
/// the second duplicate must be ignored even though it would
/// change the verdict.
#[test]
fn compare_rows_duplicate_key_first_match_wins() {
    // First A-side entry has spread=10 (would yield a regression
    // against B's 30). Second has spread=29 (would be unchanged).
    // The result must reflect the first entry only.
    let rows_a = vec![
        cmp_row("t", "tiny-1llc", true, 10.0, 0),
        cmp_row("t", "tiny-1llc", true, 29.0, 0),
    ];
    let rows_b = vec![cmp_row("t", "tiny-1llc", true, 30.0, 0)];
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(res.regressions, 1, "first match (spread=10) must win");
    let spread = res
        .findings
        .iter()
        .find(|d| d.metric.name == "worst_spread")
        .expect("worst_spread regression should fire");
    assert_eq!(
        spread.val_a, 10.0,
        "val_a must come from the first matching row"
    );
    assert_eq!(spread.delta, 20.0);
}

/// Filtering is applied before the failed-row gate. A failed row
/// that the filter excludes never reaches the `passed` check, so
/// `excluded_pairs` stays at zero -- the failure on the filtered
/// row is invisible by design.
#[test]
fn compare_rows_filter_excludes_failed_from_skip_count() {
    let rows_a = vec![
        cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
        cmp_row("beta", "tiny-1llc", false, 10.0, 0),
    ];
    let rows_b = vec![
        cmp_row("alpha", "tiny-1llc", true, 30.0, 0),
        cmp_row("beta", "tiny-1llc", true, 30.0, 0),
    ];
    // Without a filter, beta's failed row contributes
    // excluded_pairs=1.
    let unfiltered = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(unfiltered.excluded_pairs, 1);
    assert_eq!(unfiltered.regressions, 1, "alpha still regresses");

    // Filtering to "alpha" excludes beta entirely; the failed row
    // is filtered out before the passed gate runs, so
    // excluded_pairs=0.
    let filtered = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        Some("alpha"),
        &ComparisonPolicy::default(),
    );
    assert_eq!(filtered.excluded_pairs, 0);
    assert_eq!(filtered.regressions, 1);
    assert_eq!(filtered.findings.len(), 1);
    assert_eq!(filtered.findings[0].scenario, "alpha");
}

/// The substring filter searches the joined "scenario topology
/// scheduler work_type" string, so a scheduler name uniquely
/// scopes the comparison even when scenarios and topologies
/// overlap. Without scheduler in the join string this would
/// require a less-precise substring (e.g. a scenario name).
#[test]
fn compare_rows_filter_substring_matches_scheduler() {
    let mut a1 = cmp_row("test1", "tiny-1llc", true, 10.0, 0);
    a1.scheduler = "scx_alpha".into();
    let mut a2 = cmp_row("test2", "tiny-1llc", true, 10.0, 0);
    a2.scheduler = "scx_beta".into();
    let mut b1 = cmp_row("test1", "tiny-1llc", true, 30.0, 0);
    b1.scheduler = "scx_alpha".into();
    let mut b2 = cmp_row("test2", "tiny-1llc", true, 30.0, 0);
    b2.scheduler = "scx_beta".into();

    let res = compare_rows_by(
        &[a1, a2],
        &[b1, b2],
        LEGACY_PAIRING_DIMS,
        Some("scx_alpha"),
        &ComparisonPolicy::default(),
    );
    assert_eq!(res.regressions, 1, "only the scx_alpha row compares");
    assert_eq!(res.findings.len(), 1);
    assert_eq!(res.findings[0].scenario, "test1");
    // scx_beta rows are filtered out, not counted as new/removed.
    assert_eq!(res.new_in_b, 0);
    assert_eq!(res.removed_from_a, 0);
}

/// `new_in_b` counts B-side rows whose key has no match on the A
/// side; `removed_from_a` counts the converse. Both are needed so
/// schema drift between two runs (a renamed scenario, an added
/// topology preset, a removed work_type) is visible in the
/// summary instead of silently dropped.
#[test]
fn compare_rows_tracks_new_and_removed_rows() {
    // alpha exists in both -> regression.
    // beta exists only in B -> new_in_b=1.
    // gamma exists only in A -> removed_from_a=1.
    let rows_a = vec![
        cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
        cmp_row("gamma", "tiny-1llc", true, 10.0, 0),
    ];
    let rows_b = vec![
        cmp_row("alpha", "tiny-1llc", true, 30.0, 0),
        cmp_row("beta", "tiny-1llc", true, 30.0, 0),
    ];
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(res.regressions, 1, "alpha regresses on worst_spread");
    assert_eq!(res.new_in_b, 1, "beta is new on B side");
    assert_eq!(res.removed_from_a, 1, "gamma is removed on B side");
    assert_eq!(res.excluded_pairs, 0);
}

/// The filter applies to every counter, including `new_in_b` and
/// `removed_from_a`. An excluded row never reaches matching, so
/// it contributes to no counter at all.
#[test]
fn compare_rows_filter_applies_to_new_and_removed_counters() {
    let rows_a = vec![
        cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
        cmp_row("gamma", "tiny-1llc", true, 10.0, 0),
    ];
    let rows_b = vec![
        cmp_row("alpha", "tiny-1llc", true, 30.0, 0),
        cmp_row("beta", "tiny-1llc", true, 30.0, 0),
    ];

    // Filter to "alpha" -- beta and gamma are excluded by the
    // substring filter on both passes.
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        Some("alpha"),
        &ComparisonPolicy::default(),
    );
    assert_eq!(res.regressions, 1);
    assert_eq!(res.new_in_b, 0, "beta is filtered out, not new");
    assert_eq!(res.removed_from_a, 0, "gamma is filtered out, not removed");
}

// -- format_host_delta: the 5 match arms of the host-delta
//    section emitted under `stats compare --runs a b`. --

/// Builder for a `HostContext` with enough populated fields to
/// exercise `HostContext::diff`. Leaves everything else at its
/// `Default` so each test varies only the field under study.
fn host_ctx(release: &str, kernel_cmdline: Option<&str>) -> crate::host_context::HostContext {
    crate::host_context::HostContext {
        kernel_name: Some("Linux".to_string()),
        kernel_release: Some(release.to_string()),
        kernel_cmdline: kernel_cmdline.map(str::to_string),
        ..Default::default()
    }
}

/// `(Some, Some)` identical: the helper emits a one-line
/// confirmation so users running `stats compare` can distinguish
/// "same host" from "captured but unused" without inspecting
/// individual sidecars.
#[test]
fn format_host_delta_both_present_identical() {
    let ctx = host_ctx("6.14.0", Some("preempt=lazy"));
    let out = format_host_delta(Some(&ctx), Some(&ctx), "a-run", "b-run");
    assert_eq!(out, "\nhost: identical between 'a-run' and 'b-run'\n");
}

/// `(Some, Some)` differing: the helper emits the header line
/// followed by whatever `HostContext::diff` produced. Asserts
/// the structural shape (header present, delta body present)
/// rather than the exact diff formatting so this test stays
/// robust to future tweaks to the diff renderer.
#[test]
fn format_host_delta_both_present_differ() {
    let ha = host_ctx("6.14.0", Some("preempt=lazy"));
    let hb = host_ctx("6.15.1", Some("preempt=lazy"));
    let out = format_host_delta(Some(&ha), Some(&hb), "a", "b");
    assert!(
        out.starts_with("\nhost delta ('a' → 'b'):\n"),
        "got: {out:?}"
    );
    // `kernel_release` differs between the two contexts so the
    // diff body must be non-empty — confirms we routed through
    // the `else` arm and not the `identical` arm.
    let body = &out["\nhost delta ('a' → 'b'):\n".len()..];
    assert!(
        !body.is_empty(),
        "differing contexts must produce a diff body"
    );
    // Pin the trailing-newline contract: the other three arms
    // (`identical`, left-only, right-only) all end with '\n'; the
    // differ arm delegates to `HostContext::diff()` whose output
    // must also terminate with a newline so caller-side
    // concatenation with subsequent sections doesn't butt headers
    // against the last diff line. A regression that trimmed the
    // trailing newline in `HostContext::diff` would produce
    // run-on output only in the differ case — this assertion
    // catches that asymmetry.
    assert!(
        out.ends_with('\n'),
        "differ arm must end with a newline for contiguous-section output: {out:?}",
    );
}

/// `(Some, None)` left-only: one run captured host data, the
/// other did not (mixed tooling version, partial migration
/// window). Surface the asymmetry explicitly so the missing
/// side is diagnosable.
#[test]
fn format_host_delta_left_only() {
    let ctx = host_ctx("6.14.0", Some("preempt=lazy"));
    let out = format_host_delta(Some(&ctx), None, "a-run", "b-run");
    assert_eq!(out, "\nhost: captured in 'a-run' only, delta unavailable\n");
}

/// `(None, Some)` right-only: symmetric complement to
/// `left_only`. The `b`-name must appear (not `a`) — guards
/// against a future copy-paste typo that swaps the names.
#[test]
fn format_host_delta_right_only() {
    let ctx = host_ctx("6.14.0", Some("preempt=lazy"));
    let out = format_host_delta(None, Some(&ctx), "a-run", "b-run");
    assert_eq!(out, "\nhost: captured in 'b-run' only, delta unavailable\n");
}

/// `(None, None)`: neither side carries host data. The section
/// is fully suppressed — no blank line, no header, nothing.
/// Pinning this prevents a regression that introduces a
/// spurious "host: none" footer on legacy runs.
#[test]
fn format_host_delta_both_absent_emits_nothing() {
    assert_eq!(format_host_delta(None, None, "a", "b"), "");
}

/// `(Some, Some)` identical with both sides carrying the SAME
/// arch: the helper appends `(arch: {value})` to the identical
/// confirmation line. Pins the identical-arch surfacing contract
/// so an operator running `stats compare` on two same-arch runs
/// sees that the matching dimension covers arch — distinguishing
/// "both x86_64, identical" from "both aarch64, identical"
/// without inspecting individual sidecars.
#[test]
fn format_host_delta_identical_with_arch_surfaces_arch() {
    let ctx = crate::host_context::HostContext {
        kernel_name: Some("Linux".to_string()),
        arch: Some("x86_64".to_string()),
        ..Default::default()
    };
    let out = format_host_delta(Some(&ctx), Some(&ctx), "a", "b");
    assert_eq!(
        out,
        "\nhost: identical between 'a' and 'b' (arch: x86_64)\n",
    );
}

/// `(Some, Some)` identical with arch on one side only: the
/// helper falls back to the bare identical message. Pins the
/// "partial hint would mislead" arm — emitting
/// `(arch: x86_64)` when only one side has arch could read
/// as if the other side disagreed, so the conservative
/// rendering drops the hint when either side is `None`.
///
/// Both legs of the asymmetry are tested below: arch on `a`
/// only and on `b` only. Each must collapse to the bare
/// message identical to the both-None case.
#[test]
fn format_host_delta_identical_partial_arch_falls_back() {
    // a-side has arch, b-side does not. Note both contexts
    // must compare equal under `HostContext::diff` — arch is
    // hash-participating so populating it on one side would
    // route through the differ arm. Construct two
    // semantically-equal HostContexts (only `arch` differs)
    // — the diff arm DOES emit a row when arch differs, so
    // this branch is unreachable through `format_host_delta`'s
    // identical arm. Verify by asserting it routes through
    // the differ arm instead.
    let ha = crate::host_context::HostContext {
        kernel_name: Some("Linux".to_string()),
        arch: Some("x86_64".to_string()),
        ..Default::default()
    };
    let hb = crate::host_context::HostContext {
        kernel_name: Some("Linux".to_string()),
        arch: None,
        ..Default::default()
    };
    let out = format_host_delta(Some(&ha), Some(&hb), "a", "b");
    // Arch difference routes through the differ arm — pin
    // that the partial-hint case is unreachable from the
    // identical arm by construction.
    assert!(
        out.starts_with("\nhost delta ('a' → 'b'):\n"),
        "asymmetric arch must route through differ arm, not \
             identical arm: {out:?}",
    );
    assert!(
        out.contains("arch:"),
        "differ arm must surface the arch row: {out:?}",
    );
}

/// `(Some, Some)` identical when arch is `None` on both sides:
/// fall back to the bare identical message. Pre-host-context-
/// landing archives or arch-probe failures on both sides hit
/// this arm — the bare message reads correctly without the
/// `(arch: ...)` clause.
#[test]
fn format_host_delta_identical_both_arch_none_falls_back() {
    let ctx = crate::host_context::HostContext {
        kernel_name: Some("Linux".to_string()),
        arch: None,
        ..Default::default()
    };
    let out = format_host_delta(Some(&ctx), Some(&ctx), "a", "b");
    assert_eq!(out, "\nhost: identical between 'a' and 'b'\n");
}

// -- GauntletRow serde round-trip tests --
//
// `ext_metrics: BTreeMap<String, f64>` carries
// `#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]`.
// These tests pin that contract: the key disappears from JSON
// when the map is empty, round-trip through from_str
// reconstructs an equivalent row, and a non-empty payload emits
// its contents verbatim.

/// Empty `ext_metrics` is elided on serialize. Regression guard
/// for the `skip_serializing_if` half — dropping it would make
/// the writer emit `"ext_metrics":{}` noise on every row (the
/// `default` half is guarded by the sibling round-trip test).
#[test]
fn gauntlet_row_empty_ext_metrics_omits_key() {
    let row = make_row("scn", "topo", true, 0.0);
    assert!(row.ext_metrics.is_empty());
    let json = serde_json::to_string(&row).unwrap();
    assert!(
        !json.contains("\"ext_metrics\""),
        "empty ext_metrics must be omitted from JSON: {json}"
    );
}

/// Non-empty `ext_metrics` appears with its full payload. Locks
/// in that `skip_serializing_if` only fires on empty, not on
/// "has content". A false positive here would silently drop
/// extensible metrics from sidecar files.
#[test]
fn gauntlet_row_non_empty_ext_metrics_emits_payload() {
    let mut row = make_row("scn", "topo", true, 0.0);
    row.ext_metrics.insert("custom_metric".into(), 42.5);
    let json = serde_json::to_string(&row).unwrap();
    assert!(
        json.contains("\"custom_metric\":42.5"),
        "ext_metrics payload missing: {json}"
    );
}

/// Round-trip with empty `ext_metrics`: the writer omits the key
/// (via `skip_serializing_if`), so the reader must default it
/// back to empty for the round-trip to close. Regression guard
/// for the `default` half of the symmetric pair — removing it
/// would make deserialize fail on JSON this same process just
/// produced.
#[test]
fn gauntlet_row_round_trip_empty_ext_metrics() {
    let row = make_row("scn", "topo", true, 1.5);
    let json = serde_json::to_string(&row).unwrap();
    let back: GauntletRow = serde_json::from_str(&json).unwrap();
    assert_eq!(back, row);
    assert!(back.ext_metrics.is_empty());
}

/// Round-trip with populated `ext_metrics`: every entry survives
/// the to_string → from_str cycle. Guards against any future
/// field-level serde attribute (e.g. a rename or custom
/// serializer) accidentally shearing content on one side of the
/// cycle.
#[test]
fn gauntlet_row_round_trip_non_empty_ext_metrics() {
    let mut row = make_row("scn", "topo", false, std::f64::consts::PI);
    row.ext_metrics.insert("m1".into(), 1.0);
    row.ext_metrics.insert("m2".into(), 2.5);
    let json = serde_json::to_string(&row).unwrap();
    let back: GauntletRow = serde_json::from_str(&json).unwrap();
    assert_eq!(back, row);
}

/// Round-trip with populated `cpu_budget` / `vcpus`: the
/// `Option<u32>` + `skip_serializing_if` pair emits the numeric
/// keys and reads them back. Distinct from `SidecarResult`'s
/// always-emit u32 round-trip (tests.rs) — this pins the
/// GauntletRow Option serde contract, the compare-pipeline wire
/// shape where the skip_serializing_if subtlety lives.
#[test]
fn gauntlet_row_round_trip_populated_cpu_budget() {
    let mut row = make_row("scn", "topo", true, 1.0);
    row.cpu_budget = Some(4);
    row.vcpus = Some(16);
    let json = serde_json::to_string(&row).unwrap();
    assert!(
        json.contains("\"cpu_budget\":4") && json.contains("\"vcpus\":16"),
        "populated budget/vcpus must emit numeric JSON keys: {json}"
    );
    let back: GauntletRow = serde_json::from_str(&json).unwrap();
    assert_eq!(back, row);
    assert_eq!(back.cpu_budget, Some(4));
    assert_eq!(back.vcpus, Some(16));
}

/// None `cpu_budget` / `vcpus` (skip rows) omit both keys via
/// `skip_serializing_if`; the reader defaults them back to None so
/// the round-trip closes without the keys present.
#[test]
fn gauntlet_row_none_cpu_budget_omits_keys() {
    let row = make_row("scn", "topo", true, 1.0);
    assert!(row.cpu_budget.is_none() && row.vcpus.is_none());
    let json = serde_json::to_string(&row).unwrap();
    assert!(
        !json.contains("\"cpu_budget\"") && !json.contains("\"vcpus\""),
        "None budget/vcpus must be omitted from JSON: {json}"
    );
    let back: GauntletRow = serde_json::from_str(&json).unwrap();
    assert_eq!(back, row);
}

/// `compare_partitions` honours the `--dir` override —
/// pool-collection walks the override path rather than the
/// default [`crate::test_support::runs_root`]. Pool source-of-
/// truth threading regressed silently in earlier versions
/// (`--dir` was parsed but ignored), so this test pins the
/// load-bearing wire from CLI arg through `compare_partitions`
/// down to `collect_pool`.
///
/// Fixture: a tempdir alt-root with two run subdirectories,
/// each holding one sidecar. The two sidecars differ on
/// `scheduler` so the slicing-dim is `scheduler` and
/// `compare_partitions` has a well-defined contrast. Calling
/// `compare_partitions` with `dir = Some(alt_root)` finds the
/// pooled fixtures and returns Ok; calling without `--dir`
/// against runs_root (which doesn't contain these private
/// fixtures) fails with a "no sidecar data" diagnostic.
#[test]
fn compare_partitions_threads_dir_through_to_pool_collection() {
    use crate::test_support::SidecarResult;

    let alt_root = tempfile::TempDir::new().expect("create alt-root tempdir");
    // Two run subdirs; each holds one sidecar. The sidecars
    // differ on scheduler so the slicing-dim derivation has
    // a non-empty result.
    for (run_key, sched) in [
        ("__dir_thread_a__", "scx_alpha"),
        ("__dir_thread_b__", "scx_beta"),
    ] {
        let run_dir = alt_root.path().join(run_key);
        std::fs::create_dir_all(&run_dir).expect("create run dir");
        let sidecar = SidecarResult {
            test_name: "dir_thread_fixture".to_string(),
            scheduler: sched.to_string(),
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sidecar).expect("serialize fixture sidecar");
        let sidecar_path = run_dir.join(format!("{run_key}.ktstr.json"));
        std::fs::write(&sidecar_path, json).expect("write fixture sidecar");
    }

    let filter_a = RowFilter {
        schedulers: vec!["scx_alpha".to_string()],
        ..RowFilter::default()
    };
    let filter_b = RowFilter {
        schedulers: vec!["scx_beta".to_string()],
        ..RowFilter::default()
    };

    // Positive: --dir threads to collect_pool; the two
    // partitions resolve and the comparison runs without
    // bailing. Identical metric values mean exit 0 (no
    // regressions); we only care that the call succeeds.
    let exit = compare_partitions(
        &filter_a,
        &filter_b,
        None,
        &ComparisonPolicy::default(),
        Some(alt_root.path()),
        false,
        &PhaseDisplayOptions::default(),
    )
    .expect("compare_partitions must pool sidecars under --dir override");
    assert_eq!(
        exit, 0,
        "byte-identical metrics across the two scheduler \
             partitions must yield zero regressions (exit 0). \
             A non-zero exit means either the partitions loaded \
             different data than written above or compare_rows \
             regressed on identical inputs.",
    );
}

/// Regression: `perf-delta --noise-adjust` writes N runs per side, all
/// sharing one pairing key, and `compare_partitions_noise` must POOL
/// them (`RowPrep::PerRunPooled`) so `noise_findings` computes each
/// side's spread. The shared prep's duplicate-pairing-key rejection —
/// correct for the pairwise `--no-average` path (`PerRunUnique`) — must
/// NOT fire here. Before the fix `prepare_partitioned_comparison` bailed
/// on the N duplicates, so `--noise-adjust N` (N>1) always failed. The
/// spread math itself is covered by the `noise_findings` tests above;
/// this pins that the N duplicate-key rows REACH it through the pool +
/// prep instead of being rejected.
#[test]
fn compare_partitions_noise_pools_duplicate_pairing_keys() {
    use crate::test_support::SidecarResult;

    let alt_root = tempfile::TempDir::new().expect("create alt-root tempdir");
    // Three runs per side (the `--noise-adjust 3` shape). Each side's
    // three sidecars share one pairing key (identical but for the
    // slicing dim `scheduler`), so each side has 3 duplicate-key rows —
    // exactly the input the old dup-key gate rejected.
    for (sched, tag) in [("scx_base", "base"), ("scx_head", "head")] {
        for i in 0..3 {
            let run_dir = alt_root.path().join(format!("noise_{tag}_{i}"));
            std::fs::create_dir_all(&run_dir).expect("create run dir");
            let sidecar = SidecarResult {
                test_name: "noise_dup_fixture".to_string(),
                scheduler: sched.to_string(),
                ..SidecarResult::test_fixture()
            };
            let json = serde_json::to_string(&sidecar).expect("serialize fixture sidecar");
            std::fs::write(run_dir.join(format!("noise_{tag}_{i}.ktstr.json")), json)
                .expect("write fixture sidecar");
        }
    }

    let filter_a = RowFilter {
        schedulers: vec!["scx_base".to_string()],
        ..RowFilter::default()
    };
    let filter_b = RowFilter {
        schedulers: vec!["scx_head".to_string()],
        ..RowFilter::default()
    };

    // Must POOL the 3-per-side duplicates and return Ok — NOT bail with
    // "N sidecars with the same pairing key". Byte-identical metrics
    // across the two sides mean no confident regression, so exit 0.
    let exit = compare_partitions_noise(
        &filter_a,
        &filter_b,
        Some(alt_root.path()),
        1.0,
        &PhaseDisplayOptions::default(),
    )
        .expect("noise compare must pool N duplicate-key runs per side, not reject them");
    assert_eq!(
        exit, 0,
        "identical metrics across the two sides must yield no confident \
         regression (exit 0); an Err means the old dup-key gate still \
         rejects the pooled per-run rows before noise_findings sees them",
    );
}

// -- render_dirty_warning --

/// No `-dirty` commit values on either side returns `None` so
/// the caller emits no banner. Pins the silent-when-clean
/// contract that lets `warn_on_dirty_builds` be a no-op for
/// release-quality runs.
#[test]
fn render_dirty_warning_silent_when_no_dirty_commits() {
    let mut row = make_row("scn", "topo", true, 1.0);
    row.commit = Some("abcdef1".into());
    row.kernel_commit = Some("0123456".into());
    let other = row.clone();
    assert!(
        super::render_dirty_warning(&[row], &[other]).is_none(),
        "clean rows on both sides must yield no warning"
    );
}

/// Empty input on both sides is silent — `compare_partitions`
/// bails before the call when either side is empty, but the
/// helper itself must still degrade cleanly.
#[test]
fn render_dirty_warning_silent_on_empty_inputs() {
    assert!(
        super::render_dirty_warning(&[], &[]).is_none(),
        "empty inputs must yield no warning"
    );
}

/// Dirty `kernel_commit` values across both sides are deduped
/// into one block under "kernel source", with each distinct
/// value listed once and `commit` (project) absent because
/// none of the rows are dirty on that dimension.
#[test]
fn render_dirty_warning_kernel_only_dedupes_values_across_sides() {
    let mut a = make_row("scn", "topo", true, 1.0);
    a.kernel_commit = Some("aaaaaaa-dirty".into());
    a.commit = Some("clean01".into());
    let mut a2 = make_row("scn2", "topo", true, 1.0);
    a2.kernel_commit = Some("aaaaaaa-dirty".into()); // same as a
    let mut b = make_row("scn", "topo", true, 1.0);
    b.kernel_commit = Some("bbbbbbb-dirty".into());
    let text = super::render_dirty_warning(&[a, a2], &[b])
        .expect("dirty kernel_commit must yield warning");
    assert!(
        text.contains("warning: comparison includes dirty builds:"),
        "missing header in {text:?}"
    );
    assert_eq!(
        text.matches("kernel source: aaaaaaa-dirty").count(),
        1,
        "duplicate kernel_commit must be deduped, got {text:?}"
    );
    assert!(
        text.contains("kernel source: bbbbbbb-dirty"),
        "second distinct dirty kernel_commit must be listed, got {text:?}"
    );
    assert!(
        !text.contains("project:"),
        "no -dirty project commit; the project line must not appear: {text:?}"
    );
    assert!(
        text.contains("Dirty runs overwrite previous results with the same HEAD."),
        "missing trailer line 1 in {text:?}"
    );
    assert!(
        text.contains("Commit changes for reproducible-ish comparisons."),
        "missing trailer line 2 in {text:?}"
    );
}

/// Dirty `commit` (project) values are listed under "project"
/// when no `kernel_commit` is dirty, so each dimension renders
/// only when populated.
#[test]
fn render_dirty_warning_project_only_omits_kernel_section() {
    let mut a = make_row("scn", "topo", true, 1.0);
    a.commit = Some("ccccccc-dirty".into());
    let text = super::render_dirty_warning(&[a], &[]).expect("dirty commit must yield warning");
    assert!(
        text.contains("project: ccccccc-dirty"),
        "expected project line in {text:?}"
    );
    assert!(
        !text.contains("kernel source:"),
        "kernel section must not appear when only project is dirty: {text:?}"
    );
}

/// Both dimensions dirty: the warning lists "kernel source"
/// before "project" in stable order so byte-identical inputs
/// always render byte-identically. BTreeSet ordering of distinct
/// hashes within each dimension is also pinned (lex order).
#[test]
fn render_dirty_warning_both_dimensions_in_stable_order() {
    let mut a = make_row("scn", "topo", true, 1.0);
    a.kernel_commit = Some("kkkkk22-dirty".into());
    a.commit = Some("pppp222-dirty".into());
    let mut b = make_row("scn", "topo", true, 1.0);
    b.kernel_commit = Some("kkkkk11-dirty".into());
    b.commit = Some("pppp111-dirty".into());
    let text =
        super::render_dirty_warning(&[a], &[b]).expect("both dimensions dirty must yield warning");
    let kernel11 = text
        .find("kernel source: kkkkk11-dirty")
        .expect("kernel11 line absent");
    let kernel22 = text
        .find("kernel source: kkkkk22-dirty")
        .expect("kernel22 line absent");
    let project11 = text
        .find("project: pppp111-dirty")
        .expect("project11 line absent");
    let project22 = text
        .find("project: pppp222-dirty")
        .expect("project22 line absent");
    assert!(
        kernel11 < kernel22,
        "kernel section must list values in lex order: {text:?}"
    );
    assert!(
        project11 < project22,
        "project section must list values in lex order: {text:?}"
    );
    assert!(
        kernel22 < project11,
        "kernel section must precede project section: {text:?}"
    );
}

/// `None` commit fields and clean (suffix-free) values on the
/// other rows do not contribute to either set, so the warning
/// only mentions the actually-dirty hash.
#[test]
fn render_dirty_warning_skips_none_and_clean_values() {
    let mut clean_a = make_row("a", "topo", true, 1.0);
    clean_a.commit = Some("clean01".into());
    clean_a.kernel_commit = None;
    let mut dirty_b = make_row("b", "topo", true, 1.0);
    dirty_b.commit = None;
    dirty_b.kernel_commit = Some("dddddd1-dirty".into());
    let text = super::render_dirty_warning(&[clean_a], &[dirty_b])
        .expect("at least one dirty value must yield warning");
    assert!(
        text.contains("kernel source: dddddd1-dirty"),
        "dirty kernel_commit must surface in {text:?}"
    );
    assert!(
        !text.contains("project:"),
        "no dirty project commit; project section must be absent in {text:?}"
    );
    assert!(
        !text.contains("clean01"),
        "clean commit values must not appear in {text:?}"
    );
}

// -- render_overcommit_warning --

fn budget_row(scenario: &str, budget: Option<u32>, vcpus: Option<u32>) -> GauntletRow {
    let mut r = make_row(scenario, "topo", true, 1.0);
    r.cpu_budget = budget;
    r.vcpus = vcpus;
    r
}

/// No hazard: every row's budget meets its vCPU count and no group
/// mixes budgets -> `None`, whether CpuBudget is pairing or sliced.
#[test]
fn render_overcommit_warning_none_when_clean() {
    let pairing: &[Dimension] = &[Dimension::CpuBudget];
    let sliced: &[Dimension] = &[];
    let a = budget_row("a", Some(16), Some(16));
    let b = budget_row("b", Some(32), Some(16)); // roomy, not overcommit
    assert!(
        super::render_overcommit_warning(
            std::slice::from_ref(&a),
            std::slice::from_ref(&b),
            pairing,
        )
        .is_none()
    );
    assert!(super::render_overcommit_warning(&[a], &[b], sliced).is_none());
}

/// Skip rows (budget `None`) carry no budget identity and never
/// trip either check.
#[test]
fn render_overcommit_warning_ignores_skip_rows() {
    let sliced: &[Dimension] = &[];
    let a = budget_row("a", None, None);
    let b = budget_row("b", None, None);
    assert!(super::render_overcommit_warning(&[a], &[b], sliced).is_none());
}

/// An overcommitted run (budget < vcpus) is flagged on its side,
/// names the budget/vcpus pair, and the warning lists run-delay as
/// confounded (pins the kernel-grounded semantics). Fires
/// regardless of pairing.
#[test]
fn render_overcommit_warning_flags_overcommitted_side() {
    let pairing: &[Dimension] = &[Dimension::CpuBudget];
    let a = budget_row("a", Some(4), Some(16));
    let b = budget_row("b", Some(16), Some(16));
    let text = super::render_overcommit_warning(&[a], &[b], pairing)
        .expect("an overcommitted A row must warn");
    assert!(text.contains("side A"), "must name side A: {text}");
    assert!(text.contains("4/16"), "must list budget/vcpus: {text}");
    assert!(
        text.contains("run-delay"),
        "warning must list run-delay as confounded: {text}",
    );
    assert!(
        !text.contains("side B"),
        "the clean B side must not be flagged: {text}",
    );
}

/// The mixed-budget warning fires per pairing GROUP, not side-wide:
/// only rows that share a full PairingKey are averaged together.
/// - CpuBudget pairing: budgets key separate groups -> no fold.
/// - sliced + same scenario: budgets fold into one mean -> warn.
/// - sliced + different scenarios: distinct keys, never folded -> no
///   warning (the precision that distinguishes "side spans budgets"
///   from "a group averages budgets").
#[test]
fn render_overcommit_warning_mixed_budget_per_group() {
    let pairing: &[Dimension] = &[Dimension::CpuBudget];
    // Realistic sliced pairing-dims: production passes
    // Dimension::pairing_dims(&slicing) = ALL minus the sliced dim, so
    // the per-group key includes scheduler/topology/work-type/commits/
    // source — NOT just scenario. Use the real derivation so a
    // from_row key-shape regression on the sliced path is caught.
    let sliced = Dimension::pairing_dims(&[Dimension::CpuBudget]);
    let a = budget_row("a", Some(16), Some(16));
    let b1 = budget_row("b", Some(8), Some(16)); // overcommit + two budgets...
    let b2 = budget_row("b", Some(16), Some(16)); // ...same scenario AND all other dims

    // CpuBudget pairing: budgets key separate groups; the only
    // hazard is the overcommitted 8/16 row, NOT a mixed-budget fold.
    let paired = super::render_overcommit_warning(
        std::slice::from_ref(&a),
        &[b1.clone(), b2.clone()],
        pairing,
    )
    .expect("overcommitted B row still warns");
    assert!(
        paired.contains("8/16") && !paired.contains("share a pairing group"),
        "pairing dim: overcommit flagged, no mixed-fold warning: {paired}",
    );

    // Sliced + b1/b2 share EVERY pairing dim (scenario + scheduler +
    // topology + ... all default-equal): one group, two budgets, so
    // --average folds them -> mixed warning on side B.
    let sliced_same = super::render_overcommit_warning(&[a], &[b1, b2], &sliced)
        .expect("mixed budgets in one group on a sliced side must warn");
    assert!(
        sliced_same.contains("share a pairing group") && sliced_same.contains("side B"),
        "sliced same-key: must warn B's budgets share a pairing group: {sliced_same}",
    );

    // Sliced but the two budgets differ on a NON-budget pairing dim
    // (scheduler): distinct pairing keys -> never folded -> no
    // warning, even though the side has two budgets and shares
    // scenario. Proves the per-group key uses the FULL dim set, not
    // just scenario (the degenerate &[] key would have missed this).
    let mut s1 = budget_row("c", Some(16), Some(16));
    s1.scheduler = "sched_a".to_string();
    let mut s2 = budget_row("c", Some(32), Some(32));
    s2.scheduler = "sched_b".to_string();
    let clean_a = budget_row("d", Some(16), Some(16));
    assert!(
        super::render_overcommit_warning(&[s1, s2], std::slice::from_ref(&clean_a), &sliced)
            .is_none(),
        "two budgets differing on a non-budget pairing dim (scheduler) key \
             separate groups -> no fold -> no warning",
    );

    // Sliced + different scenarios on ONE side: distinct pairing
    // keys, never folded, neither overcommitted -> NO warning.
    let xa = budget_row("x", Some(16), Some(16));
    let ya = budget_row("y", Some(32), Some(32));
    let clean_b = budget_row("z", Some(16), Some(16));
    assert!(
        super::render_overcommit_warning(&[xa, ya], std::slice::from_ref(&clean_b), &sliced)
            .is_none(),
        "one side spanning budgets across distinct scenarios -> no fold -> no warning",
    );
}

/// Mixed budgets in one pairing group with NO overcommit on either
/// side routes through the `else` block of `render_overcommit_warning`
/// — the "mixing two measurement conditions" message, distinct from
/// the host-overcommit message. Each budget meets its own vCPU count
/// (16/16, 32/32) so neither is overcommitted, but the two rows share
/// every pairing dim and CpuBudget is sliced, so `--average` would
/// fold them into one mean. Pins the no-overcommit-but-mixed banner
/// text the existing per-group test never reaches (its mixed case
/// also overcommits, taking the `if` branch).
#[test]
fn render_overcommit_warning_mixed_no_overcommit_uses_else_banner() {
    let sliced = Dimension::pairing_dims(&[Dimension::CpuBudget]);
    // Same scenario + every other pairing dim equal; two distinct
    // budgets, each NOT overcommitted (b == v).
    let b1 = budget_row("m", Some(16), Some(16));
    let b2 = budget_row("m", Some(32), Some(32));
    let clean = budget_row("n", Some(16), Some(16));
    let text = super::render_overcommit_warning(&[b1, b2], std::slice::from_ref(&clean), &sliced)
        .expect("two non-overcommit budgets folding into one group must warn");
    assert!(
        text.contains("mixing two measurement conditions"),
        "no-overcommit mixed-budget case must use the else-branch banner, \
         not the host-overcommit banner; got: {text}",
    );
    assert!(
        !text.contains("host-overcommitted run"),
        "the else branch must NOT claim a host-overcommitted run; got: {text}",
    );
    assert!(
        text.contains("side A") && text.contains("share a pairing group"),
        "the folded side must be named with its mixed budgets; got: {text}",
    );
}

// -- check_no_duplicate_pairing_keys --

/// `check_no_duplicate_pairing_keys` returns `Ok(())` when every row
/// on the side carries a distinct pairing key — the normal
/// `--no-average` path where each sidecar is a unique
/// (scenario, topology, work_type) measurement.
#[test]
fn check_no_duplicate_pairing_keys_ok_when_all_keys_distinct() {
    let rows = vec![
        cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
        cmp_row("beta", "tiny-1llc", true, 10.0, 0),
    ];
    assert!(
        super::check_no_duplicate_pairing_keys(&rows, LEGACY_PAIRING_DIMS, "A").is_ok(),
        "distinct pairing keys must pass the --no-average duplicate gate",
    );
    // Empty input is trivially duplicate-free.
    assert!(
        super::check_no_duplicate_pairing_keys(&[], LEGACY_PAIRING_DIMS, "A").is_ok(),
        "empty side must pass the duplicate gate",
    );
}

/// `check_no_duplicate_pairing_keys` bails when two rows on one side
/// share a pairing key — the `--no-average` guard against
/// `compare_rows_by` silently latching onto the first match. The bail
/// must name the offending side and the duplicate-key condition so the
/// operator can drop `--no-average` or add a disambiguating filter.
#[test]
fn check_no_duplicate_pairing_keys_bails_on_collision_and_names_side() {
    // Two rows with the SAME (scenario, topology, work_type) under
    // LEGACY_PAIRING_DIMS -> identical pairing key.
    let rows = vec![
        cmp_row("dup", "tiny-1llc", true, 10.0, 0),
        cmp_row("dup", "tiny-1llc", true, 20.0, 0),
    ];
    let err = super::check_no_duplicate_pairing_keys(&rows, LEGACY_PAIRING_DIMS, "B")
        .expect_err("two rows sharing a pairing key must bail under --no-average");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("side B"),
        "bail must name the offending side; got: {rendered}",
    );
    assert!(
        rendered.contains("same pairing key"),
        "bail must describe the duplicate-key condition; got: {rendered}",
    );
    assert!(
        rendered.contains("--no-average"),
        "bail must point at the --no-average flag the operator can drop; got: {rendered}",
    );
    // The side label is lowercased into the `--b-X` suggestion.
    assert!(
        rendered.contains("--b-"),
        "bail must suggest a per-side filter to disambiguate; got: {rendered}",
    );
}

// -- noise_findings (perf-delta --noise-adjust row-level core) tests --

/// Three identical runs of one scenario carrying `worst_spread` (LowerBetter, via
/// `spread`) + `total_iterations` (HigherBetter, via `iters`) — the two
/// metric-bearing fields `cmp_row` sets. All-identical so each side's spread is 0
/// (clean), isolating the cross-side direction classification.
fn noise_side(scenario: &str, spread: f64, iters: u64) -> Vec<GauntletRow> {
    vec![
        cmp_row(scenario, "tiny-1llc", true, spread, iters),
        cmp_row(scenario, "tiny-1llc", true, spread, iters),
        cmp_row(scenario, "tiny-1llc", true, spread, iters),
    ]
}

/// Under --noise-adjust a Rate's compared centroid must be the pooled
/// Σnum/Σden (duration-weighted) the registry documents (metric.rs), NOT the
/// mean of per-run ratios — the two differ when run denominators differ. A
/// side with runs (run_delay=100, wall=1s)->100/s and (run_delay=100,
/// wall=10s)->10/s has pooled 200/11 = 18.18/s but mean-of-ratios 55/s. The
/// per-run band ([10,100]) still measures run-to-run spread. This pins that
/// the centroid is pooled (agreeing with the Averaged/--average path).
#[test]
fn noise_findings_rate_centroid_is_pooled_not_mean_of_ratios() {
    let mk = |run_delay: f64, wall: f64| {
        let mut r = cmp_row("rate", "tiny-1llc", true, 10.0, 0);
        r.ext_metrics.insert("total_run_delay".to_string(), run_delay);
        r.ext_metrics
            .insert("total_schedstat_wall_sec".to_string(), wall);
        r
    };
    // B identical to A so the only thing under test is A's reported centroid.
    let a = vec![mk(100.0, 1.0), mk(100.0, 10.0)];
    let b = vec![mk(100.0, 1.0), mk(100.0, 10.0)];
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 1.0, true);
    let f = rep
        .findings
        .iter()
        .find(|f| f.metric.name == "run_delay_per_sec")
        .unwrap_or_else(|| {
            panic!(
                "run_delay_per_sec must appear: {:?}",
                rep.findings.iter().map(|f| f.metric.name).collect::<Vec<_>>(),
            )
        });
    // Pooled Σnum/Σden = 200/11 = 18.18..., NOT mean-of-ratios (100+10)/2 = 55.
    assert!(
        (f.verdict.a.mean - 200.0 / 11.0).abs() < 1e-6,
        "Rate centroid must be pooled Σnum/Σden (18.18), got {} (mean-of-ratios would be 55)",
        f.verdict.a.mean,
    );
}

/// Schedstat Rate metrics (e.g. total_run_delay_ns_per_sched) have a
/// `|_| None` accessor and materialize only via derive_rate_metrics from
/// their ext_metrics components — which sidecar_to_row injects but never
/// derives. The Averaged path derives them (group_and_average_by); the
/// per-run noise path must too, or a real regression in a GATED schedstat
/// rate is silently absent from the verdict. Here run-delay-per-schedule
/// doubles 1000 -> 2000 ns (LowerBetter) with zero per-side spread, so the
/// DERIVED rate must appear and gate as a confident REGRESSION.
#[test]
fn noise_findings_derives_schedstat_rate_from_per_run_components() {
    let mk = |run_delay: f64| {
        let mut r = cmp_row("sched", "tiny-1llc", true, 10.0, 0);
        r.ext_metrics.insert("total_run_delay".to_string(), run_delay);
        r.ext_metrics.insert("total_pcount".to_string(), 1000.0);
        r
    };
    // total_run_delay_ns_per_sched = total_run_delay / total_pcount.
    let a = vec![mk(1_000_000.0), mk(1_000_000.0)]; // 1000 ns/sched
    let b = vec![mk(2_000_000.0), mk(2_000_000.0)]; // 2000 ns/sched
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 1.0, false);
    assert_eq!(rep.paired_scenarios, 1);
    let rate = rep
        .findings
        .iter()
        .find(|f| f.metric.name == "total_run_delay_ns_per_sched")
        .unwrap_or_else(|| {
            panic!(
                "derived schedstat rate must appear in noise findings: {:?}",
                rep.findings.iter().map(|f| f.metric.name).collect::<Vec<_>>(),
            )
        });
    assert_eq!(
        rate.kind,
        NoiseKind::Regression,
        "run-delay-per-schedule doubling (LowerBetter) must gate as a confident regression",
    );
}

/// A per-side run can fail (noise_dual_run logs + continues), writing a
/// `passed=false` sidecar whose failure-mode metric is an outlier. It must
/// be EXCLUDED from the spread pool — mirroring the scalar `compare_rows_by`
/// — or a byte-identical HEAD gates as a false regression. Here B = 2 clean
/// runs at 2000 iters + 1 FAILED run at 990; A = 3 clean runs at 2000.
/// Without exclusion B's mean (1663) < A's band => a false total_iterations
/// regression; with exclusion the failed row is dropped and there is none.
#[test]
fn noise_findings_excludes_failed_run_from_the_spread_pool() {
    let a = noise_side("fx", 10.0, 2000);
    let b = vec![
        cmp_row("fx", "tiny-1llc", true, 10.0, 2000),
        cmp_row("fx", "tiny-1llc", true, 10.0, 2000),
        cmp_row("fx", "tiny-1llc", false, 10.0, 990), // failed run, outlier iters
    ];
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 1.0, false);
    assert_eq!(rep.paired_scenarios, 1);
    assert_eq!(
        rep.regressions(),
        0,
        "a failed per-run row must be excluded from the pool, not gate a false regression: {:?}",
        rep.findings
            .iter()
            .map(|f| (f.metric.name, f.kind))
            .collect::<Vec<_>>(),
    );
}

/// A per-side run failure can leave one side with a single realized
/// sample. Even a large cross-side shift that would otherwise be a
/// confident regression must be reported NOISY (never gated), because a
/// single-point band has no measurable spread. Pins that the `<2`-sample
/// guard in `noise_verdict` flows through `noise_findings` end-to-end.
#[test]
fn noise_findings_degenerate_single_sample_side_is_noisy_not_confident() {
    let a_one = vec![cmp_row("degen", "tiny-1llc", true, 10.0, 2000)];
    let b_three = noise_side("degen", 10.0, 1000); // 3 clean runs, iters dropped 2000->1000
    let rep = noise_findings(&a_one, &b_three, LEGACY_PAIRING_DIMS, 1.0, false);
    assert_eq!(rep.paired_scenarios, 1);
    assert_eq!(
        rep.regressions(),
        0,
        "a single-sample baseline side must NOT yield a confident regression: {:?}",
        rep.findings
            .iter()
            .map(|f| (f.metric.name, f.kind))
            .collect::<Vec<_>>(),
    );
    assert!(
        rep.noisy() >= 1,
        "the shifted metric(s) must be flagged NOISY because side A realized <2 samples",
    );
}

// ---- per-phase noise (perf-delta --noise-adjust, per-phase) ----

/// Build `n` pass rows for one side, each carrying the given phase buckets.
fn phased_rows(
    scenario: &str,
    n: usize,
    buckets: &[crate::assert::PhaseBucket],
) -> Vec<GauntletRow> {
    (0..n)
        .map(|_| {
            let mut r = cmp_row(scenario, "tiny-1llc", true, 10.0, 0);
            r.phases = buckets.to_vec();
            r
        })
        .collect()
}

#[test]
fn noise_phase_findings_emits_per_phase_spread() {
    // Step[1] max_dsq_depth (LowerBetter Peak) rises 8->20 across sides, clean
    // (spread 0 both) -> separated (disjoint bands) and material (delta 12 >= abs
    // 10, 150% >= 50% rel) -> a confident per-phase REGRESSION; BASELINE
    // unchanged.
    let a = phased_rows(
        "scn",
        3,
        &[
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)]),
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 8.0)]),
        ],
    );
    let b = phased_rows(
        "scn",
        3,
        &[
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)]),
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 20.0)]),
        ],
    );
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 5.0, false);
    let s1 = rep
        .phase_findings
        .iter()
        .find(|f| f.step_index == 1 && f.metric.name == "max_dsq_depth")
        .expect("Step[1] max_dsq_depth per-phase finding");
    assert_eq!((s1.verdict.a.mean, s1.verdict.b.mean), (8.0, 20.0));
    assert_eq!(
        s1.kind,
        NoiseKind::Regression,
        "LowerBetter 8->20 rose => per-phase regression",
    );
}

#[test]
fn noise_phase_scoped_regression_is_render_only_not_gated() {
    // A per-phase regression with NO scalar/aggregate move: the row-level
    // metric fields are identical across sides (cmp_row defaults), only the
    // phase bucket shifts. Per-phase is render-only, so the exit basis
    // (aggregate regressions) stays 0 while phase_regressions() counts it.
    let a = phased_rows("scn", 3, &[make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 8.0)])]);
    let b = phased_rows("scn", 3, &[make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 20.0)])]);
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 5.0, false);
    assert_eq!(rep.phase_regressions(), 1, "the per-phase shift is a phase regression");
    assert_eq!(
        rep.regressions(),
        0,
        "no aggregate move -> exit basis unaffected (per-phase is render-only)",
    );
}

#[test]
fn noise_phase_rate_pooled_centroid_within_phase() {
    // iteration_rate = total_phase_iterations / total_phase_duration_sec (Rate,
    // HigherBetter). A side: run1 (100 iters, 1s)=100/s + run2 (100 iters,
    // 10s)=10/s -> pooled 200/11 = 18.18, NOT mean-of-ratios 55; band [10,100].
    let bucket = |iters: f64, sec: f64| {
        make_phase_bucket(
            1,
            "Step[0]",
            &[
                ("total_phase_iterations", iters),
                ("total_phase_duration_sec", sec),
                ("iteration_rate", iters / sec),
            ],
        )
    };
    let side = |scn: &str| {
        vec![
            {
                let mut r = cmp_row(scn, "tiny-1llc", true, 10.0, 0);
                r.phases = vec![bucket(100.0, 1.0)];
                r
            },
            {
                let mut r = cmp_row(scn, "tiny-1llc", true, 10.0, 0);
                r.phases = vec![bucket(100.0, 10.0)];
                r
            },
        ]
    };
    let rep = noise_findings(&side("scn"), &side("scn"), LEGACY_PAIRING_DIMS, 1.0, true);
    let f = rep
        .phase_findings
        .iter()
        .find(|f| f.step_index == 1 && f.metric.name == "iteration_rate")
        .expect("Step[1] iteration_rate per-phase finding");
    assert!(
        (f.verdict.a.mean - 200.0 / 11.0).abs() < 1e-6,
        "per-phase Rate centroid must be pooled (18.18), got {} (mean-of-ratios would be 55)",
        f.verdict.a.mean,
    );
}

#[test]
fn noise_phase_excludes_non_pass_run() {
    // B = 2 clean Step[1]=8 + 1 FAILED Step[1]=40 (outlier). The failed run's
    // phase must be excluded before the per-phase pass, so Step[1] sees only
    // the 2 passing B values (8,8) and there is no false per-phase regression.
    let a = phased_rows("scn", 3, &[make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 8.0)])]);
    let mut b = phased_rows("scn", 2, &[make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 8.0)])]);
    let mut failed = cmp_row("scn", "tiny-1llc", false, 10.0, 0); // is_fail
    failed.phases = vec![make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 40.0)])];
    b.push(failed);
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 1.0, false);
    assert_eq!(
        rep.phase_regressions(),
        0,
        "the failed run's outlier phase must be excluded, no false per-phase regression",
    );
}

#[test]
fn noise_phase_n_lt_2_is_noisy() {
    // Step[1] max_dsq_depth present in only 1 of A's 3 passing runs -> n<2 ->
    // Noisy, never a confident regression, despite a large cross-side shift.
    let mut a = phased_rows("scn", 3, &[make_phase_bucket(1, "Step[0]", &[])]);
    a[0].phases = vec![make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 8.0)])];
    let b = phased_rows("scn", 3, &[make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 40.0)])]);
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 1.0, false);
    let f = rep
        .phase_findings
        .iter()
        .find(|f| f.step_index == 1 && f.metric.name == "max_dsq_depth");
    assert!(
        matches!(f.map(|f| f.kind), Some(NoiseKind::Noisy)),
        "a <2-sample per-phase side must be Noisy, got {:?}",
        f.map(|f| f.kind),
    );
}

#[test]
fn noise_phase_one_sided_metric_is_coverage() {
    // A's matched Step[1] has max_dsq_depth; B's Step[1] does not -> coverage
    // (present_side A), not a finding.
    let a = phased_rows("scn", 3, &[make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 8.0)])]);
    let b = phased_rows("scn", 3, &[make_phase_bucket(1, "Step[0]", &[])]);
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 1.0, false);
    assert!(
        rep.phase_coverage
            .iter()
            .any(|c| c.metric.map(|m| m.name) == Some("max_dsq_depth")
                && c.present_side == ComparePartition::A),
        "A-only max_dsq_depth at a matched step must be a coverage row",
    );
    assert!(
        !rep.phase_findings.iter().any(|f| f.metric.name == "max_dsq_depth"),
        "a one-sided metric is coverage, never a finding",
    );
}

#[test]
fn noise_phase_one_sided_step_is_coverage() {
    // A has BASELINE + Step[1]; B has only BASELINE. The whole one-sided
    // Step[1] must surface as coverage, not be dropped.
    let a = phased_rows(
        "scn",
        3,
        &[
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)]),
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 8.0)]),
        ],
    );
    let b = phased_rows("scn", 3, &[make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)])]);
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 1.0, false);
    assert!(
        rep.phase_coverage
            .iter()
            .any(|c| c.step_index == 1 && c.present_side == ComparePartition::A),
        "the whole one-sided Step[1] must surface as coverage",
    );
    assert_eq!(rep.phase_regressions(), 0, "matched BASELINE unchanged -> no regression");
}

#[test]
fn noise_phase_empty_one_sided_step_surfaces_as_shape_coverage() {
    // A has BASELINE + a Step[1] whose buckets carry NO readable metric (a
    // synthesized capture-free step); B has only BASELINE. The empty one-sided
    // Step[1] must still surface as a metric-less coverage row, not be silently
    // dropped (no-silent-drops; mirrors the scalar empty UnpairedPhaseRow).
    let a = phased_rows(
        "scn",
        3,
        &[
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)]),
            make_phase_bucket(1, "Step[0]", &[]),
        ],
    );
    let b = phased_rows("scn", 3, &[make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)])]);
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 1.0, false);
    assert!(
        rep.phase_coverage.iter().any(|c| c.step_index == 1
            && c.metric.is_none()
            && c.present_side == ComparePartition::A),
        "an empty one-sided Step[1] must surface as a metric-less coverage row: {:?}",
        rep.phase_coverage
            .iter()
            .map(|c| (c.step_index, c.metric.map(|m| m.name)))
            .collect::<Vec<_>>(),
    );
}

#[test]
fn noise_phase_empty_phases_skip() {
    // A carries phases, B has none -> per-phase sub-pass skipped entirely,
    // while the aggregate findings still populate.
    let a = phased_rows("scn", 3, &[make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 8.0)])]);
    let b: Vec<GauntletRow> = (0..3).map(|_| cmp_row("scn", "tiny-1llc", true, 10.0, 0)).collect();
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 1.0, false);
    assert!(
        rep.phase_findings.is_empty() && rep.phase_coverage.is_empty(),
        "no per-phase data when a side has no phases",
    );
}

#[test]
fn format_noise_phase_findings_lines_honors_flags() {
    // BASELINE max_dsq_depth 5->20 and Step[1] 8->20 both shift, each material
    // (delta >= abs 10, rel >= 50%) and separated -> per-phase regressions.
    let a = phased_rows(
        "scn",
        3,
        &[
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)]),
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 8.0)]),
        ],
    );
    let b = phased_rows(
        "scn",
        3,
        &[
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 20.0)]),
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 20.0)]),
        ],
    );
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 5.0, true);
    let render = |opts: &PhaseDisplayOptions| {
        format_noise_phase_findings_lines(&rep.phase_findings, &rep.phase_coverage, opts, "base", "head")
            .join("\n")
    };
    // no_phases -> empty
    assert!(
        format_noise_phase_findings_lines(
            &rep.phase_findings,
            &rep.phase_coverage,
            &PhaseDisplayOptions {
                no_phases: true,
                ..Default::default()
            },
            "base",
            "head",
        )
        .is_empty(),
        "no_phases suppresses the per-phase block",
    );
    // steps_only -> Step[0] present, BASELINE (step 0) absent.
    let s = render(&PhaseDisplayOptions {
        steps_only: true,
        ..Default::default()
    });
    assert!(
        s.contains("1: Step[0]") && !s.contains("0: BASELINE"),
        "steps_only suppresses BASELINE:\n{s}",
    );
    // --phase 0 -> only BASELINE.
    let p = render(&PhaseDisplayOptions {
        phase: Some(0),
        ..Default::default()
    });
    assert!(
        p.contains("0: BASELINE") && !p.contains("1: Step[0]"),
        "phase=0 shows only BASELINE:\n{p}",
    );
    // REGRESSION verdict text present in the default render.
    assert!(
        render(&PhaseDisplayOptions::default()).contains("REGRESSION"),
        "a per-phase regression renders the REGRESSION verdict",
    );
}

#[test]
fn passes_noise_spread_threshold_edges() {
    // Build a NoiseVerdict with known means (2 identical samples/side -> mean).
    let v = |a: f64, b: f64| noise_verdict(&[a, a], &[b, b], 1.0);
    // No --phase-threshold -> every row passes.
    assert!(PhaseDisplayOptions::default().passes_noise_spread_threshold(&v(100.0, 200.0)));
    let o = PhaseDisplayOptions {
        phase_threshold: Some(10.0),
        ..Default::default()
    };
    // |b-a|/|a| under vs over the 10% gate.
    assert!(!o.passes_noise_spread_threshold(&v(100.0, 105.0)), "5% < 10% -> filtered");
    assert!(o.passes_noise_spread_threshold(&v(100.0, 120.0)), "20% >= 10% -> shown");
    // ~zero baseline with a real move -> unbounded relative change -> shown.
    assert!(o.passes_noise_spread_threshold(&v(0.0, 50.0)), "zero baseline + move -> shown");
    // Both ~zero -> no signal -> filtered by any positive threshold.
    assert!(!o.passes_noise_spread_threshold(&v(0.0, 0.0)), "both ~zero -> filtered");
}

#[test]
fn format_noise_phase_findings_lines_renders_coverage() {
    // A one-sided metric (A-only max_dsq_depth at matched Step[1]) + a whole
    // empty one-sided step (Step[2]) -> the coverage table must render both,
    // with `—` for the metric-less shape row.
    let a = phased_rows(
        "scn",
        3,
        &[
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 8.0)]),
            make_phase_bucket(2, "Step[1]", &[]),
        ],
    );
    let b = phased_rows("scn", 3, &[make_phase_bucket(1, "Step[0]", &[])]);
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 1.0, false);
    let out = format_noise_phase_findings_lines(
        &rep.phase_findings,
        &rep.phase_coverage,
        &PhaseDisplayOptions::default(),
        "base",
        "head",
    )
    .join("\n");
    assert!(
        out.contains("per-phase coverage asymmetry"),
        "coverage header rendered:\n{out}",
    );
    assert!(out.contains("max_dsq_depth"), "one-sided metric name rendered:\n{out}");
    assert!(out.contains("—"), "the metric-less empty one-sided step renders `—`:\n{out}");
}

#[test]
fn format_noise_phase_findings_lines_honors_phase_threshold() {
    // Step[0] (step 1) shifts +3%, Step[1] (step 2) shifts +50%. --phase-threshold
    // 10 suppresses the small move, keeps the large one.
    let a = phased_rows(
        "scn",
        3,
        &[
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 100.0)]),
            make_phase_bucket(2, "Step[1]", &[("max_dsq_depth", 100.0)]),
        ],
    );
    let b = phased_rows(
        "scn",
        3,
        &[
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 103.0)]),
            make_phase_bucket(2, "Step[1]", &[("max_dsq_depth", 150.0)]),
        ],
    );
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 1.0, true);
    let out = format_noise_phase_findings_lines(
        &rep.phase_findings,
        &rep.phase_coverage,
        &PhaseDisplayOptions {
            phase_threshold: Some(10.0),
            ..Default::default()
        },
        "base",
        "head",
    )
    .join("\n");
    assert!(
        out.contains("2: Step[1]") && !out.contains("1: Step[0]"),
        "--phase-threshold 10 keeps the +50% step and suppresses the +3% step:\n{out}",
    );
}

#[test]
fn noise_phase_informational_metric_shows_but_never_gates() {
    // DELIBERATE divergence from the scalar per-phase pass (which DROPS
    // Informational metrics — its bool verdict can't represent them): the noise
    // per-phase path SHOWS an Informational metric as NoiseKind::Informational,
    // never gating. total_ttwu_count is a registered directionless Counter; a
    // significant move (1000->5000, clean spread 0) must classify Informational,
    // not Regression, and must NOT count in phase_regressions(). Pins the
    // divergence so a future refactor mirroring the scalar skip can't silently
    // reverse it.
    let a = phased_rows("scn", 3, &[make_phase_bucket(1, "Step[0]", &[("total_ttwu_count", 1000.0)])]);
    let b = phased_rows("scn", 3, &[make_phase_bucket(1, "Step[0]", &[("total_ttwu_count", 5000.0)])]);
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 1.0, false);
    let f = rep
        .phase_findings
        .iter()
        .find(|f| f.step_index == 1 && f.metric.name == "total_ttwu_count")
        .expect("Step[1] total_ttwu_count per-phase finding (noise SHOWS Informational)");
    assert_eq!(
        f.kind,
        NoiseKind::Informational,
        "an Informational per-phase metric shows as Informational, not a regression",
    );
    assert_eq!(rep.phase_regressions(), 0, "Informational never gates");
}

#[test]
fn noise_phase_findings_disambiguate_by_pairing_key_across_topologies() {
    // One scenario name run on TWO topologies forms two distinct pairing-key
    // groups (topology is a pairing dim), each with its own per-phase finding.
    // Their rows must carry DISTINCT pairing_labels (scenario/topology/...),
    // not collapse to identical "scenario"-only labels the operator can't tell
    // apart.
    let phased = |topo: &'static str, depth: f64| {
        (0..3)
            .map(|_| {
                let mut r = cmp_row("scn", topo, true, 10.0, 0);
                r.phases = vec![make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", depth)])];
                r
            })
            .collect::<Vec<_>>()
    };
    let mut a = phased("tiny-1llc", 8.0);
    a.extend(phased("large-4llc", 8.0));
    let mut b = phased("tiny-1llc", 20.0);
    b.extend(phased("large-4llc", 20.0));
    let rep = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 5.0, false);
    let labels: Vec<&str> = rep
        .phase_findings
        .iter()
        .filter(|f| f.metric.name == "max_dsq_depth")
        .map(|f| f.pairing_label.as_str())
        .collect();
    assert_eq!(
        labels.len(),
        2,
        "one per-phase finding per topology group, distinct labels: {labels:?}",
    );
    assert!(
        labels.iter().any(|l| l.contains("tiny-1llc")) && labels.iter().any(|l| l.contains("large-4llc")),
        "pairing labels must include the topology to disambiguate: {labels:?}",
    );
    assert_ne!(labels[0], labels[1], "the two topology groups must render distinct labels");
}

#[test]
fn summarize_side_runs_categorizes_by_exclusion() {
    // A skipped run (is_skip); a failed run (passed=false, not skipped/inc =>
    // is_fail); a comparable run (passed=true). `comparable` must equal the
    // count noise_findings keeps, so a zero explains an empty comparison.
    let mut skip = cmp_row("s", "tiny-1llc", false, 0.0, 0);
    skip.skipped = true;
    let fail = cmp_row("s", "tiny-1llc", false, 0.0, 0);
    let pass = cmp_row("s", "tiny-1llc", true, 0.0, 0);

    // All skipped -> 0 comparable; the breakdown names the skips (the perf-delta
    // "no comparable runs to pair" diagnostic reads this).
    let (ok, desc) = summarize_side_runs(&[skip.clone(), skip.clone(), skip.clone()]);
    assert_eq!(ok, 0, "all skipped -> 0 comparable: {desc}");
    assert!(
        desc.contains("3 run(s)") && desc.contains("0 comparable") && desc.contains("3 skipped"),
        "breakdown names the skipped runs: {desc}"
    );

    // Mixed: 1 pass + 1 skip + 1 fail -> 1 comparable, both exclusions named.
    let (ok2, desc2) = summarize_side_runs(&[pass, skip, fail]);
    assert_eq!(ok2, 1, "one pass is comparable: {desc2}");
    assert!(
        desc2.contains("1 comparable") && desc2.contains("1 skipped") && desc2.contains("1 failed"),
        "mixed breakdown names each excluded category: {desc2}"
    );
}

#[test]
fn noise_findings_classifies_both_polarities() {
    // Both polarities WORSEN: worst_spread (LowerBetter) rises 10->15;
    // total_iterations (HigherBetter) drops 2000->1000. Both sides clean (spread
    // 0), so each is a CONFIDENT regression.
    let rep = noise_findings(
        &noise_side("regress", 10.0, 2000),
        &noise_side("regress", 15.0, 1000),
        LEGACY_PAIRING_DIMS,
        1.0,
        false,
    );
    assert_eq!(rep.paired_scenarios, 1);
    assert_eq!(
        rep.regressions(),
        2,
        "LowerBetter rose + HigherBetter dropped = 2 regressions: {:?}",
        rep.findings
            .iter()
            .map(|f| (f.metric.name, f.kind))
            .collect::<Vec<_>>(),
    );
    assert_eq!(rep.noisy(), 0);
    assert!(rep.findings.iter().all(|f| f.kind == NoiseKind::Regression));

    // Mirror: both polarities IMPROVE (worst_spread drops, total_iterations rises).
    let rep = noise_findings(
        &noise_side("improve", 15.0, 1000),
        &noise_side("improve", 10.0, 2000),
        LEGACY_PAIRING_DIMS,
        1.0,
        false,
    );
    assert_eq!(rep.regressions(), 0);
    assert_eq!(
        rep.findings
            .iter()
            .filter(|f| f.kind == NoiseKind::Improvement)
            .count(),
        2,
        "LowerBetter dropped + HigherBetter rose = 2 improvements",
    );
}

#[test]
fn noise_findings_high_spread_annotates_but_does_not_suppress_regression() {
    // A's worst_spread swings 10..20 (~67% relative spread, over the 5% advisory
    // gate), and B (30) is far higher — a worsening move for LowerBetter. The
    // high per-side spread is ADVISORY: it flags the row (high_spread) but must
    // NOT suppress the confident regression. The old behavior dropped exactly
    // this as NOISY, INVERTING signal and noise (a real regression's degraded
    // side is intrinsically high-variance). The bands are disjoint ([10,20] vs
    // [30,30]) so the move is separated, and the delta is material (15 >= abs 5,
    // 100% >= 25% rel).
    let a = vec![
        cmp_row("noisy", "tiny-1llc", true, 10.0, 2000),
        cmp_row("noisy", "tiny-1llc", true, 20.0, 2000),
        cmp_row("noisy", "tiny-1llc", true, 15.0, 2000),
    ];
    let rep = noise_findings(
        &a,
        &noise_side("noisy", 30.0, 2000),
        LEGACY_PAIRING_DIMS,
        5.0,
        false,
    );
    let ws = rep
        .findings
        .iter()
        .find(|f| f.metric.name == "worst_spread")
        .expect("worst_spread finding present");
    assert_eq!(
        ws.kind,
        NoiseKind::Regression,
        "wide A spread must NOT suppress a separated + material worsening move",
    );
    assert!(
        ws.verdict.high_spread,
        "A's ~67% spread exceeds the 5% advisory gate -> high_spread annotation",
    );
    assert_eq!(
        rep.regressions(),
        1,
        "the separated, material worsening metric gates as a confident regression",
    );
    // total_iterations is unchanged (2000 both sides) and clean -> omitted.
    assert!(
        rep.findings.iter().all(|f| f.metric.name == "worst_spread"),
        "unchanged-and-clean metrics are omitted: {:?}",
        rep.findings
            .iter()
            .map(|f| f.metric.name)
            .collect::<Vec<_>>(),
    );
}

#[test]
fn noise_findings_separated_but_immaterial_stays_stable() {
    // The materiality gate in isolation: worst_spread A [10,10,10] vs B
    // [10.5,10.5,10.5] has fully DISJOINT zero-variance bands -> separated=true,
    // but delta 0.5 < default_abs 5.0 -> material=false. classify_noise must keep
    // it Stable (include_stable) / omit it (gate path), NEVER a regression. Guards
    // the `&& material` clause in classify_noise: dropping it would turn this into
    // a false Regression while every band/separation test still passed.
    let a = noise_side("imm", 10.0, 2000);
    let b = noise_side("imm", 10.5, 2000);
    // Gate path: the separated-but-immaterial move must not gate.
    let gate = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 5.0, false);
    assert_eq!(
        gate.regressions(),
        0,
        "a separated but immaterial (0.5 < abs 5.0) move must not gate: {:?}",
        gate.findings
            .iter()
            .map(|f| (f.metric.name, f.kind))
            .collect::<Vec<_>>(),
    );
    // Render path: it surfaces as Stable, not Regression, and the bands really
    // are separated (so it is the materiality gate — not lack of separation —
    // holding it back).
    let show = noise_findings(&a, &b, LEGACY_PAIRING_DIMS, 5.0, true);
    let ws = show
        .findings
        .iter()
        .find(|f| f.metric.name == "worst_spread")
        .expect("worst_spread row present under include_stable");
    assert_eq!(
        ws.kind,
        NoiseKind::Stable,
        "separated-but-immaterial worst_spread is Stable, not a regression",
    );
    assert!(
        ws.verdict.separated,
        "premise: the [10,10] vs [10.5,10.5] bands are disjoint => separated",
    );
}

#[test]
fn noise_findings_skips_all_zero_and_omits_unchanged() {
    // Both sides exactly 0 on every metric -> no signal -> no findings, but the
    // scenario still counts as paired.
    let rep = noise_findings(
        &noise_side("zero", 0.0, 0),
        &noise_side("zero", 0.0, 0),
        LEGACY_PAIRING_DIMS,
        1.0,
        false,
    );
    assert!(
        rep.findings.is_empty(),
        "all-zero scenario yields no findings"
    );
    assert_eq!(rep.paired_scenarios, 1);

    // Identical non-zero sides -> within-band, clean -> no findings either.
    let rep = noise_findings(
        &noise_side("same", 12.0, 1500),
        &noise_side("same", 12.0, 1500),
        LEGACY_PAIRING_DIMS,
        1.0,
        false,
    );
    assert!(
        rep.findings.is_empty(),
        "unchanged-clean scenario yields no findings"
    );
    assert_eq!((rep.regressions(), rep.noisy()), (0, 0));
}

#[test]
fn noise_findings_include_stable_shows_unchanged_metrics() {
    // include_stable=true (the render path): an unchanged-and-clean metric
    // the gate path omits is instead reported as Stable, so the full
    // comparison table shows every metric. Stable never gates.
    let rep = noise_findings(
        &noise_side("same", 12.0, 1500),
        &noise_side("same", 12.0, 1500),
        LEGACY_PAIRING_DIMS,
        1.0,
        true,
    );
    assert!(
        !rep.findings.is_empty(),
        "include_stable surfaces the unchanged metrics"
    );
    assert!(
        rep.findings.iter().all(|f| f.kind == NoiseKind::Stable),
        "unchanged-clean metrics are Stable: {:?}",
        rep.findings
            .iter()
            .map(|f| (f.metric.name, f.kind))
            .collect::<Vec<_>>(),
    );
    assert_eq!((rep.regressions(), rep.noisy()), (0, 0));

    // Both-zero metrics stay OMITTED even under include_stable=true (the
    // both-zero skip precedes the include_stable branch), so no zero-valued
    // metric leaks into the table as a spurious Stable row.
    let rep_zero = noise_findings(
        &noise_side("zero", 0.0, 0),
        &noise_side("zero", 0.0, 0),
        LEGACY_PAIRING_DIMS,
        1.0,
        true,
    );
    assert!(
        rep_zero.findings.is_empty(),
        "both-zero metrics are omitted (never Stable) even with include_stable: {:?}",
        rep_zero
            .findings
            .iter()
            .map(|f| f.metric.name)
            .collect::<Vec<_>>(),
    );
    assert_eq!(rep_zero.paired_scenarios, 1);
}

#[test]
fn format_noise_findings_table_renders_rows_and_verdicts() {
    // worst_spread rises 10->15 (LowerBetter -> REGRESSION); total_iterations
    // is unchanged (2000 both) -> Stable. The table carries the header, the
    // regressed row + verdict, and the Stable row (full comparison visible).
    let rep = noise_findings(
        &noise_side("mix", 10.0, 2000),
        &noise_side("mix", 15.0, 2000),
        LEGACY_PAIRING_DIMS,
        1.0,
        true,
    );
    let out = format_noise_findings_table(&rep.findings, "base", "head");
    assert!(
        out.contains("TEST / METRIC") && out.contains("VERDICT"),
        "header present: {out}"
    );
    // The TEST column carries the full pairing-key label (scenario + pairing
    // dims: topology/work_type), not scenario alone — matching the scalar path.
    assert!(
        out.contains("mix/tiny-1llc/SpinWait / worst_spread") && out.contains("REGRESSION"),
        "worsened metric row + verdict: {out}"
    );
    assert!(
        out.contains("stable"),
        "unchanged total_iterations renders as a stable row: {out}"
    );
}

#[test]
fn format_noise_findings_table_renders_noisy_improvement_and_advisory_spread() {
    // Pin the remaining verdict-arm strings (a text/color swap in one arm would
    // otherwise slip past the REGRESSION/stable-only table test).

    // Noisy: a side with <2 usable runs (insufficient_samples) -> NOISY row.
    let a = vec![cmp_row("nz", "tiny-1llc", true, 10.0, 2000)];
    let rep = noise_findings(
        &a,
        &noise_side("nz", 30.0, 2000),
        LEGACY_PAIRING_DIMS,
        5.0,
        true,
    );
    let out = format_noise_findings_table(&rep.findings, "base", "head");
    assert!(
        out.contains("NOISY (<2 runs)"),
        "insufficient-samples verdict rendered: {out}"
    );

    // Improvement: worst_spread drops 15->10 (LowerBetter, clean) -> improvement.
    let rep = noise_findings(
        &noise_side("imp", 15.0, 2000),
        &noise_side("imp", 10.0, 2000),
        LEGACY_PAIRING_DIMS,
        5.0,
        true,
    );
    let out = format_noise_findings_table(&rep.findings, "base", "head");
    assert!(
        out.contains("improvement"),
        "improvement verdict rendered: {out}"
    );

    // Advisory spread: a separated + material worsening move whose baseline side
    // is high-variance (worst_spread A [10,20,15] ~67% > 5% gate, B 30) renders
    // as "REGRESSION (noisy spread)" — the advisory flag annotates but does NOT
    // suppress (the signal-inversion fix).
    let a = vec![
        cmp_row("adv", "tiny-1llc", true, 10.0, 2000),
        cmp_row("adv", "tiny-1llc", true, 20.0, 2000),
        cmp_row("adv", "tiny-1llc", true, 15.0, 2000),
    ];
    let rep = noise_findings(
        &a,
        &noise_side("adv", 30.0, 2000),
        LEGACY_PAIRING_DIMS,
        5.0,
        true,
    );
    let out = format_noise_findings_table(&rep.findings, "base", "head");
    assert!(
        out.contains("REGRESSION (noisy spread)"),
        "high_spread annotates the reported regression, never suppresses it: {out}"
    );
}

/// A `Polarity::Informational` metric (the monitor `total_ttwu_count`) that
/// moves significantly is classified `FindingKind::Informational` — it appears
/// in the findings but is NEVER counted as a regression or improvement, so it
/// never affects the exit basis (`report.regressions`). The gate-safety
/// guarantee for the directionless schedstat counters: more wakeups must not
/// read as a regression just because the workload did more work.
#[test]
fn compare_rows_informational_metric_shows_but_never_gates() {
    let mk = |ttwu: f64| {
        // spread (10.0) and total_iterations (100) identical both sides => no
        // directional finding; only the informational ext counter moves.
        let mut r = cmp_row("t", "tiny-1llc", true, 10.0, 100);
        r.ext_metrics.insert("total_ttwu_count".into(), ttwu);
        r
    };
    let res = compare_rows_by(
        &[mk(1000.0)],
        &[mk(5000.0)],
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(
        res.regressions, 0,
        "informational metric must not be a regression"
    );
    assert_eq!(res.improvements, 0, "...nor an improvement");
    assert_eq!(
        res.informational, 1,
        "the 5x total_ttwu_count move is classified informational"
    );
    let f = res
        .findings
        .iter()
        .find(|f| f.metric.name == "total_ttwu_count")
        .expect("total_ttwu_count finding present");
    assert_eq!(f.kind, FindingKind::Informational);
}

/// A metric present on exactly ONE side of a paired row is a coverage
/// difference, NOT a regression/improvement: recorded in `coverage_diffs`,
/// never gated. Pre-fix, `read().unwrap_or(0.0)` coerced the absent side to
/// 0.0 and the rel-gate read it as an unbounded change from a zero baseline —
/// a phantom verdict for a directional metric (avg_nr_running is LowerBetter)
/// that was simply not captured on one side.
#[test]
fn compare_rows_one_sided_absent_is_coverage_diff_not_verdict() {
    // avg_nr_running is LowerBetter + ext-only; present on exactly one side.
    let present = {
        let mut r = cmp_row("t", "tiny-1llc", true, 10.0, 100);
        r.ext_metrics.insert("avg_nr_running".into(), 5.0);
        r
    };
    let absent = cmp_row("t", "tiny-1llc", true, 10.0, 100);

    // A present (5.0), B absent: pre-fix a phantom LowerBetter improvement
    // (5 -> 0); post-fix a coverage diff on side A.
    let res = compare_rows_by(
        std::slice::from_ref(&present),
        std::slice::from_ref(&absent),
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(
        res.regressions, 0,
        "one-sided-absent must not be a regression"
    );
    assert_eq!(res.improvements, 0, "...nor an improvement");
    assert!(
        !res.findings
            .iter()
            .any(|f| f.metric.name == "avg_nr_running"),
        "no phantom avg_nr_running finding for a one-sided-absent metric"
    );
    assert_eq!(res.coverage_diffs.len(), 1, "recorded as a coverage diff");
    assert_eq!(res.coverage_diffs[0].metric.name, "avg_nr_running");
    assert_eq!(res.coverage_diffs[0].present_side, ComparePartition::A);
    assert_eq!(res.coverage_diffs[0].value, 5.0);

    // Mirror: A absent, B present (5.0) — pre-fix a phantom regression
    // (0 -> 5, LowerBetter); post-fix a coverage diff on side B.
    let res2 = compare_rows_by(
        &[absent],
        &[present],
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(res2.regressions, 0, "mirror: not a regression");
    assert_eq!(res2.improvements, 0);
    assert_eq!(res2.coverage_diffs.len(), 1);
    assert_eq!(res2.coverage_diffs[0].present_side, ComparePartition::B);
}

/// A metric absent on BOTH sides contributes nothing (no coverage diff, no
/// finding). A metric present on both sides with a genuine zero on one side is
/// NOT absent — it still goes through the dual-gate verdict (the
/// present-0.0-vs-absent distinction the fix rests on).
#[test]
fn compare_rows_both_absent_skipped_present_zero_still_compared() {
    let res_absent = compare_rows_by(
        &[cmp_row("t", "tiny-1llc", true, 10.0, 100)],
        &[cmp_row("t", "tiny-1llc", true, 10.0, 100)],
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert!(
        res_absent.coverage_diffs.is_empty(),
        "metric absent on both sides is not a coverage diff"
    );
    assert!(
        !res_absent
            .findings
            .iter()
            .any(|f| f.metric.name == "avg_nr_running"),
        "metric absent on both sides produces no finding"
    );

    // Present-zero on A, present-nonzero on B: BOTH present (Some(0.0) vs
    // Some(5.0)), a real comparison, not a coverage diff. 0 -> 5 on a
    // LowerBetter metric is a regression.
    let zero_a = {
        let mut r = cmp_row("t", "tiny-1llc", true, 10.0, 100);
        r.ext_metrics.insert("avg_nr_running".into(), 0.0);
        r
    };
    let nonzero_b = {
        let mut r = cmp_row("t", "tiny-1llc", true, 10.0, 100);
        r.ext_metrics.insert("avg_nr_running".into(), 5.0);
        r
    };
    let res_zero = compare_rows_by(
        &[zero_a],
        &[nonzero_b],
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert!(
        res_zero.coverage_diffs.is_empty(),
        "present-0.0 is NOT absent — no coverage diff"
    );
    assert_eq!(
        res_zero.regressions, 1,
        "0 -> 5 on a LowerBetter metric (both present) is a regression"
    );
}

/// The coverage-diff SURFACING maps `present_side` to the right A/B label — a
/// swapped match arm would mis-report which run has the metric. Tests the
/// extracted `format_coverage_diff_lines` directly (print_summary_block's
/// `println!` is not capturable) for both `ComparePartition::A` and `::B`.
#[test]
fn coverage_diff_lines_map_present_absent_labels_by_side() {
    let present = {
        let mut r = cmp_row("t", "tiny-1llc", true, 10.0, 100);
        r.ext_metrics.insert("avg_nr_running".into(), 5.0);
        r
    };
    let absent = cmp_row("t", "tiny-1llc", true, 10.0, 100);

    // A present, B absent -> present_side A -> "in runA, absent in runB".
    let report = compare_rows_by(
        std::slice::from_ref(&present),
        std::slice::from_ref(&absent),
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    let joined = format_coverage_diff_lines(&report, "runA", "runB").join("\n");
    assert!(
        joined.contains("avg_nr_running"),
        "names the metric: {joined}"
    );
    assert!(
        joined.contains("= 5.00 in 'runA'"),
        "present value + side-A label (runA): {joined}"
    );
    assert!(
        joined.contains("absent in 'runB'"),
        "absent side is the OTHER label (runB): {joined}"
    );

    // Mirror: A absent, B present -> present_side B -> "in runB, absent in runA".
    let report2 = compare_rows_by(
        std::slice::from_ref(&absent),
        std::slice::from_ref(&present),
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    let joined2 = format_coverage_diff_lines(&report2, "runA", "runB").join("\n");
    assert!(
        joined2.contains("= 5.00 in 'runB'"),
        "present on side B maps to runB: {joined2}"
    );
    assert!(
        joined2.contains("absent in 'runA'"),
        "absent maps to runA: {joined2}"
    );
}
