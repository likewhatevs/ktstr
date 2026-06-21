use super::*;

// -- compare_rows_by per-phase pass tests --------------------------
//
// These tests exercise the per-row-pair phase intersection that
// populates CompareReport.phase_deltas + unpaired_phases. They go
// through compare_rows_by directly (rather than the full
// compare_partitions which also does filtering/averaging) because
// the parallel pass lives inside compare_rows_by's row-pair
// iteration and that is the load-bearing surface to pin.
//
// Each test builds 2 GauntletRows via make_row, attaches phase
// buckets explicitly, then asserts the resulting CompareReport
// shape against the expected per-phase + unpaired data flow.

fn make_phase_bucket(
    step_index: u16,
    label: &str,
    metrics: &[(&str, f64)],
) -> crate::assert::PhaseBucket {
    let metrics_map = metrics.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    crate::assert::PhaseBucket {
        per_cgroup: Default::default(),
        step_index,
        label: label.to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        metrics: metrics_map,
    }
}

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
/// `compare_rows_by`). max_dsq_depth has
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

/// per-phase pass honors the dual-gate semantic the
/// scalar pass uses inside its per-metric loop in
/// `compare_rows_by` (`|delta| < default_abs ||
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

/// `--phase-threshold 50` against `a = 0.0` divides by the
/// `max(|a|, 1.0)` floor (NOT zero), so a delta of 10 yields
/// rel = 10/1 = 10.0 (1000%) which passes the 0.5 gate. Pins
/// the NaN-defense in the denominator: a future refactor that
/// drops the `.max(1.0)` would divide by zero and either
/// admit all rows (NaN >= X → false) or panic.
#[test]
fn passes_delta_threshold_zero_a_divides_by_unit_floor() {
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
        "zero-a divisor floor (|a|.max(1.0)) must keep rel finite \
             (rel = |10|/max(0,1) = 10.0); 10.0 ≥ 0.5 → row passes"
    );
}

/// Distinguishing pin for the `.max(1.0)` divisor floor: the
/// a=0/delta=10 case above passes both floored (10/1=10≥0.5) AND
/// unfloored (10/0=+inf≥0.5), so it does NOT guard the floor. The
/// floor actually protects a=0 AND delta=0 at --phase-threshold 0:
/// floored gives 0/1=0, and 0≥0 → the row renders (the documented
/// "PCT=0 shows every row" contract); unfloored gives 0/0=NaN, and
/// NaN≥0 is false → the row is silently dropped. Drop `.max(1.0)`
/// and this assertion flips to false.
#[test]
fn passes_delta_threshold_zero_a_zero_delta_at_pct_zero_renders() {
    let opts = PhaseDisplayOptions {
        phase_threshold: Some(0.0),
        ..PhaseDisplayOptions::default()
    };
    let metric = METRICS
        .iter()
        .find(|m| m.name == "max_dsq_depth")
        .expect("max_dsq_depth in METRICS");
    let zero_a_zero_delta = PhaseDeltaRow {
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
        opts.passes_delta_threshold(&zero_a_zero_delta),
        "zero-a/zero-delta at --phase-threshold 0 must render: floored \
             0/1=0, 0≥0 true. Dropping the .max(1.0) floor yields 0/0=NaN, \
             NaN≥0 false, silently dropping the row",
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
