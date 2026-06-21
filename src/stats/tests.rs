use super::*;
use crate::assert::ScenarioStats;

// -- MetricKind temporal aggregation ----------------------------

/// `Counter` reduces by sum-of-finite-samples. NaN drops, empty
/// returns None.
#[test]
fn aggregate_samples_counter_sums_finite_values() {
    assert_eq!(
        aggregate_samples(&[1.0, 2.0, 3.0], MetricKind::Counter),
        Some(6.0),
    );
    assert_eq!(
        aggregate_samples(&[1.0, f64::NAN, 3.0], MetricKind::Counter),
        Some(4.0),
        "NaN samples drop from the sum",
    );
    assert_eq!(
        aggregate_samples(&[], MetricKind::Counter),
        None,
        "empty input → None",
    );
    assert_eq!(
        aggregate_samples(&[f64::NAN, f64::INFINITY], MetricKind::Counter),
        None,
        "all-non-finite → None",
    );
}

/// `Gauge(Avg)` reduces by arithmetic mean.
#[test]
fn aggregate_samples_gauge_avg_means_finite() {
    let r = aggregate_samples(&[1.0, 2.0, 3.0], MetricKind::Gauge(GaugeAgg::Avg));
    assert_eq!(r, Some(2.0));
}

/// `Gauge(Last)` returns the last finite sample.
#[test]
fn aggregate_samples_gauge_last_returns_last() {
    let r = aggregate_samples(&[1.0, 2.0, 3.0], MetricKind::Gauge(GaugeAgg::Last));
    assert_eq!(r, Some(3.0));
    // NaN at the tail still drops; Last picks the last FINITE.
    let r = aggregate_samples(&[1.0, 2.0, f64::NAN], MetricKind::Gauge(GaugeAgg::Last));
    assert_eq!(r, Some(2.0));
}

/// `Gauge(Max)` and `Peak` both reduce by max.
#[test]
fn aggregate_samples_max_and_peak_pick_largest() {
    let r = aggregate_samples(&[1.0, 5.0, 3.0], MetricKind::Gauge(GaugeAgg::Max));
    assert_eq!(r, Some(5.0));
    let r = aggregate_samples(&[1.0, 5.0, 3.0], MetricKind::Peak);
    assert_eq!(r, Some(5.0));
}

/// `Timestamp` returns the last sample (latest snapshot).
#[test]
fn aggregate_samples_timestamp_returns_last() {
    let r = aggregate_samples(&[100.0, 200.0, 300.0], MetricKind::Timestamp);
    assert_eq!(r, Some(300.0));
}

// -- Weighted Gauge(Avg) ------------------------------------

/// `Gauge(Avg)` with explicit per-sample weights computes the
/// weighted arithmetic mean `sum(v*w) / sum(w)`. (10*5 + 20*15)
/// / 20 = 17.5 — a 50%-larger weight on the higher value
/// pulls the mean above the unweighted midpoint of 15.
#[test]
fn aggregate_samples_weighted_gauge_avg_pulls_toward_heavier_sample() {
    let r = aggregate_samples_weighted(&[(10.0, 5), (20.0, 15)], MetricKind::Gauge(GaugeAgg::Avg));
    assert_eq!(r, Some(17.5));
}

/// `Gauge(Avg)` without weights (unweighted entry point)
/// computes the arithmetic mean (legacy semantic). The
/// previous test's same value vector yields 15.0 here vs
/// 17.5 weighted — the difference is the cross-RUN bias.
#[test]
fn aggregate_samples_gauge_avg_unweighted_is_arithmetic_mean() {
    let r = aggregate_samples(&[10.0, 20.0], MetricKind::Gauge(GaugeAgg::Avg));
    assert_eq!(r, Some(15.0));
}

/// `Gauge(Avg)` with zero total weight falls back to the
/// unweighted arithmetic mean rather than dividing by zero.
/// Mirrors `merge_metric_values` at
/// `crate::assert::merge_matched_phase_buckets` per
/// single-source-of-truth.
#[test]
fn aggregate_samples_weighted_gauge_avg_zero_total_weight_falls_back_to_mean() {
    let r = aggregate_samples_weighted(&[(10.0, 0), (30.0, 0)], MetricKind::Gauge(GaugeAgg::Avg));
    assert_eq!(r, Some(20.0));
}

/// `Counter` ignores weights — sum is weight-independent by
/// construction. Pinned so a future refactor that introduces
/// weight-sensitive Counter semantics breaks here.
#[test]
fn aggregate_samples_weighted_counter_ignores_weights() {
    let r = aggregate_samples_weighted(&[(10.0, 5), (20.0, 15)], MetricKind::Counter);
    assert_eq!(r, Some(30.0));
}

/// `Peak` ignores weights — max is weight-independent.
#[test]
fn aggregate_samples_weighted_peak_ignores_weights() {
    let r = aggregate_samples_weighted(&[(10.0, 5), (20.0, 15)], MetricKind::Peak);
    assert_eq!(r, Some(20.0));
}

/// `Gauge(Max)` ignores weights — max is weight-independent.
#[test]
fn aggregate_samples_weighted_gauge_max_ignores_weights() {
    let r = aggregate_samples_weighted(&[(10.0, 5), (20.0, 15)], MetricKind::Gauge(GaugeAgg::Max));
    assert_eq!(r, Some(20.0));
}

/// Cross-RUN reduction of the derived kinds — the ONLY path that reaches
/// aggregate_finite for Distribution/WorstLowest (WITHIN a run they are
/// `is_derived` and never reduced from a slice). Worst folds by MAX
/// (weight-independent); every other Distribution reduction and every
/// WorstLowest fold by UNWEIGHTED mean — proven with UNEQUAL weights
/// (5 vs 15) so a run_sample_count-weighted mean (which would give 25.0)
/// is distinguishable from the unweighted 20.0.
#[test]
fn aggregate_samples_weighted_distribution_worstlowest_arms() {
    // Worst → MAX, weight-independent: max(10, 20) = 20.
    assert_eq!(
        aggregate_samples_weighted(
            &[(10.0, 5), (20.0, 15)],
            MetricKind::Distribution {
                source: SampleSource::RunDelayNs,
                reduction: SampleReduction::Worst,
            },
        ),
        Some(20.0),
    );
    // Distribution (non-Worst) → UNWEIGHTED mean: (10 + 30)/2 = 20.0,
    // NOT the run_sample_count-weighted (10*5 + 30*15)/20 = 25.0.
    assert_eq!(
        aggregate_samples_weighted(
            &[(10.0, 5), (30.0, 15)],
            MetricKind::Distribution {
                source: SampleSource::WakeLatencyNs,
                reduction: SampleReduction::P99,
            },
        ),
        Some(20.0),
    );
    // WorstLowest → UNWEIGHTED mean: same (10 + 30)/2 = 20.0, not 25.0.
    assert_eq!(
        aggregate_samples_weighted(
            &[(10.0, 5), (30.0, 15)],
            MetricKind::WorstLowest {
                numerator: WorstLowestNumerator::Iterations,
                denominator: WorstLowestDenominator::CpuTimeNs,
            },
        ),
        Some(20.0),
    );
}

/// Cross-RUN weight-0 contributor contract: a monitor-off / no-periodic-
/// capture run (`run_sample_count == 0`) that DID emit a Distribution /
/// WorstLowest key is COUNTED in the unweighted mean, never zero-weighted
/// out. `aggregate_finite`'s Distribution/WorstLowest arm is
/// `sum / finite.len()` (it structurally ignores the weight closure — see
/// the monitor-off rationale at the arm: a weighted fold "would silently
/// zero-weight a monitor-off run"), so a `(value, 0)` pair that survives
/// the `is_finite` filter is included. Guards against a future weight-aware
/// refactor silently dropping a monitor-off run's distributional value.
#[test]
fn aggregate_samples_weighted_distribution_worstlowest_counts_zero_weight_contributor() {
    // (10.0, 0) = a monitor-off run that emitted the key; (30.0, 15) a
    // normal run. The weight-0 run IS counted: (10 + 30)/2 = 20.0 — a
    // weight-aware fold that dropped the 0-weight run would give 30.0.
    assert_eq!(
        aggregate_samples_weighted(
            &[(10.0, 0), (30.0, 15)],
            MetricKind::Distribution {
                source: SampleSource::WakeLatencyNs,
                reduction: SampleReduction::P99,
            },
        ),
        Some(20.0),
    );
    assert_eq!(
        aggregate_samples_weighted(
            &[(10.0, 0), (30.0, 15)],
            MetricKind::WorstLowest {
                numerator: WorstLowestNumerator::Iterations,
                denominator: WorstLowestDenominator::CpuTimeNs,
            },
        ),
        Some(20.0),
    );
}

/// `Gauge(Last)` and `Timestamp` ignore weights — last-finite
/// is weight-independent.
#[test]
fn aggregate_samples_weighted_gauge_last_and_timestamp_ignore_weights() {
    let last =
        aggregate_samples_weighted(&[(10.0, 5), (20.0, 15)], MetricKind::Gauge(GaugeAgg::Last));
    assert_eq!(last, Some(20.0));
    let ts = aggregate_samples_weighted(&[(10.0, 5), (20.0, 15)], MetricKind::Timestamp);
    assert_eq!(ts, Some(20.0));
}

/// NaN-valued pairs drop entirely (value AND weight) — the
/// (f64, usize) pair type keeps the weight bound to its
/// value so a NaN filter can't misalign weights to other
/// samples. (10*5 + 30*20) / (5+20) = 650 / 25 = 26.0.
#[test]
fn aggregate_samples_weighted_gauge_avg_drops_nan_pairs_in_lockstep() {
    let r = aggregate_samples_weighted(
        &[(10.0, 5), (f64::NAN, 10), (30.0, 20)],
        MetricKind::Gauge(GaugeAgg::Avg),
    );
    assert_eq!(r, Some(26.0));
}

// -- Per-phase reductions --------------------------------------

/// `phase_counter_delta` returns `last - first` of finite
/// samples — the right semantic for a per-phase reduction
/// over a cumulative-since-boot counter. Distinct from
/// `aggregate_samples(..., Counter)` which sums the samples
/// (correct for cross-run aggregation, wrong for per-phase
/// deltas).
#[test]
fn phase_counter_delta_returns_last_minus_first() {
    // Cumulative-since-boot counter samples in [100, 150, 175,
    // 200] yield delta 100 across the phase window.
    assert_eq!(
        phase_counter_delta(&[100.0, 150.0, 175.0, 200.0]),
        Some(100.0),
    );
    // NaN samples drop from the finite slice; first/last
    // are computed over the filtered sequence.
    assert_eq!(
        phase_counter_delta(&[f64::NAN, 150.0, 175.0, f64::NAN]),
        Some(25.0),
    );
}

/// `phase_counter_delta` returns `Some(0.0)` for a phase with
/// exactly one finite sample (self-delta — the metric was
/// observed but no per-phase change can be computed), and
/// `None` only when zero samples are finite. The distinction
/// matters for the bucket renderer: `Some(0.0)` paints "phase
/// has data, delta is 0"; `None` paints "no data".
#[test]
fn phase_counter_delta_one_finite_sample_is_self_delta() {
    assert_eq!(phase_counter_delta(&[42.0]), Some(0.0));
    assert_eq!(phase_counter_delta(&[f64::NAN, 42.0, f64::NAN]), Some(0.0));
    assert_eq!(phase_counter_delta(&[]), None);
    assert_eq!(phase_counter_delta(&[f64::NAN, f64::INFINITY]), None);
}

/// A counter that regresses across a phase window
/// (scheduler-restart counter reset, kernel module reload,
/// etc.) clamps to 0 rather than emitting a negative delta a
/// downstream "negative count is impossible" assertion would
/// either misread or trip on. Mirrors the existing
/// `monitor::counter_delta` clamp pattern.
#[test]
fn phase_counter_delta_clamps_negative_to_zero_on_counter_reset() {
    assert_eq!(
        phase_counter_delta(&[500.0, 600.0, 100.0]),
        Some(0.0),
        "last < first clamps to 0 (counter reset detected)",
    );
}

/// `aggregate_samples_for_phase` dispatches Counter through
/// `phase_counter_delta` (per-phase delta) and every other
/// kind through `aggregate_samples` (flat-run semantic). Pins
/// the invariant: a Counter-kind metric must NOT collapse
/// to a sum across the phase window — that's the bug the
/// per-phase aggregator was introduced to fix.
#[test]
fn aggregate_samples_for_phase_returns_none_for_derived_kinds() {
    // Derived kinds (Rate / Distribution / WorstLowest) are `is_derived`,
    // merge as Recompute, and have NO per-phase value: returning None keeps
    // them off the single-slice reducers within a run (their value is
    // produced post-merge by derive_rate_metrics /
    // populate_run_distribution_metrics). Pins the within-run skip-routing.
    let mk = |kind: MetricKind| MetricDef {
        name: "x",
        accessor: |_| None,
        display_unit: "",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 0.0,
        default_rel: 0.0,
        kind,
    };
    for kind in [
        MetricKind::Rate {
            numerator: "a",
            denominator: "b",
        },
        MetricKind::Distribution {
            source: SampleSource::WakeLatencyNs,
            reduction: SampleReduction::P99,
        },
        MetricKind::WorstLowest {
            numerator: WorstLowestNumerator::Iterations,
            denominator: WorstLowestDenominator::NumWorkers,
        },
    ] {
        assert!(kind.is_derived(), "{kind:?} must be is_derived");
        assert_eq!(kind.merge_kind(), MergeKind::Recompute);
        assert_eq!(
            aggregate_samples_for_phase(&mk(kind), &[1.0, 2.0, 3.0]),
            None,
            "derived kind {kind:?} must have no per-phase reduction",
        );
    }
}

#[test]
fn aggregate_samples_for_phase_dispatches_on_kind() {
    let counter = MetricDef {
        name: "total_test_counter",
        accessor: |_| None,
        display_unit: "",
        polarity: crate::test_support::Polarity::HigherBetter,
        default_abs: 0.0,
        default_rel: 0.0,
        kind: MetricKind::Counter,
    };
    // Counter routes through `phase_counter_delta`, NOT
    // `aggregate_samples`'s sum.
    assert_eq!(
        aggregate_samples_for_phase(&counter, &[100.0, 150.0, 175.0]),
        Some(75.0),
        "Counter kind must reduce by last - first, not by sum",
    );
    assert_ne!(
        aggregate_samples_for_phase(&counter, &[100.0, 150.0, 175.0]),
        Some(425.0),
        "Counter kind MUST NOT collapse to flat-run sum across a phase",
    );

    let peak = MetricDef {
        name: "max_test_peak",
        accessor: |_| None,
        display_unit: "",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 0.0,
        default_rel: 0.0,
        kind: MetricKind::Peak,
    };
    // Peak routes through `aggregate_samples` -> max.
    assert_eq!(
        aggregate_samples_for_phase(&peak, &[1.0, 5.0, 3.0]),
        Some(5.0),
        "Peak kind must reduce by max",
    );

    let gauge_avg = MetricDef {
        name: "worst_test_gauge",
        accessor: |_| None,
        display_unit: "",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 0.0,
        default_rel: 0.0,
        kind: MetricKind::Gauge(GaugeAgg::Avg),
    };
    assert_eq!(
        aggregate_samples_for_phase(&gauge_avg, &[2.0, 4.0, 6.0]),
        Some(4.0),
        "Gauge(Avg) kind must reduce by arithmetic mean",
    );

    let delta_sum = MetricDef {
        name: "total_test_delta",
        accessor: |_| None,
        display_unit: "",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 0.0,
        default_rel: 0.0,
        kind: MetricKind::DeltaSum,
    };
    // DeltaSum samples are ALREADY per-read deltas, so the per-phase
    // reduction SUMS them — NOT a last-minus-first that would
    // difference two deltas. [10, 20, 5] -> 35, not 5 - 10 (which a
    // Counter would clamp to 0).
    assert_eq!(
        aggregate_samples_for_phase(&delta_sum, &[10.0, 20.0, 5.0]),
        Some(35.0),
        "DeltaSum kind must reduce by sum of per-read deltas",
    );
    assert_eq!(
        aggregate_samples(&[10.0, 20.0, 5.0], MetricKind::DeltaSum),
        Some(35.0),
        "DeltaSum flat-run reduction is also a sum",
    );
}

/// A [`MetricKind::Rate`] is derived as numerator/denominator at the
/// per-phase level and RE-POOLED (Σnum/Σdenom) across a merge — never
/// averaged as a ready-made ratio. Pins the core of the Rate kind.
#[test]
fn rate_derives_per_phase_and_repools_across_merge() {
    use std::collections::BTreeMap;
    // Per-phase: rate = num / denom.
    let mut phase = BTreeMap::new();
    phase.insert("iters".to_string(), 1000.0);
    phase.insert("secs".to_string(), 4.0);
    derive_rate_metrics_from(&mut phase, std::iter::once(("rate", "iters", "secs")));
    assert_eq!(
        phase.get("rate").copied(),
        Some(250.0),
        "per-phase rate = num/denom",
    );

    // Cross-phase merge re-pools: the components fold by their own
    // kind (Counter -> sum) FIRST, then the rate is re-derived from
    // the pooled components. Phase A = 1000 iters / 1 s = 1000/s;
    // phase B = 10 iters / 9 s ≈ 1.11/s. The correct merged rate is
    // (1000+10)/(1+9) = 101.0, NOT the mean of the two ratios
    // (1000 + 1.11)/2 ≈ 500.6.
    let mut merged = BTreeMap::new();
    merged.insert("iters".to_string(), 1000.0 + 10.0); // Counter sum
    merged.insert("secs".to_string(), 1.0 + 9.0); // Counter sum
    derive_rate_metrics_from(&mut merged, std::iter::once(("rate", "iters", "secs")));
    assert_eq!(
        merged.get("rate").copied(),
        Some(101.0),
        "merged rate must re-pool Σnum/Σdenom",
    );
    let mean_of_ratios = (1000.0 + (10.0 / 9.0)) / 2.0;
    assert!(
        (merged.get("rate").copied().unwrap() - mean_of_ratios).abs() > 100.0,
        "re-pool must differ from mean-of-ratios (got {:?}, mean-of-ratios {mean_of_ratios})",
        merged.get("rate"),
    );
}

/// `derive_rate_metrics` leaves the rate key ABSENT (distinct from a
/// real 0.0) when a component is missing, the denominator is zero, or
/// a component is non-finite.
#[test]
fn rate_absent_on_missing_component_zero_or_nonfinite() {
    use std::collections::BTreeMap;
    // Denominator missing.
    let mut m = BTreeMap::new();
    m.insert("iters".to_string(), 5.0);
    derive_rate_metrics_from(&mut m, std::iter::once(("rate", "iters", "secs")));
    assert!(!m.contains_key("rate"), "absent denom -> no rate key");

    // Denominator zero (must NOT insert inf).
    m.insert("secs".to_string(), 0.0);
    derive_rate_metrics_from(&mut m, std::iter::once(("rate", "iters", "secs")));
    assert!(!m.contains_key("rate"), "zero denom -> no rate key");

    // Non-finite numerator.
    let mut n = BTreeMap::new();
    n.insert("iters".to_string(), f64::NAN);
    n.insert("secs".to_string(), 2.0);
    derive_rate_metrics_from(&mut n, std::iter::once(("rate", "iters", "secs")));
    assert!(!n.contains_key("rate"), "NaN numerator -> no rate key");

    // Finite inputs whose QUOTIENT overflows to inf: absent, not inf.
    let mut o = BTreeMap::new();
    o.insert("iters".to_string(), f64::MAX);
    o.insert("secs".to_string(), f64::MIN_POSITIVE);
    derive_rate_metrics_from(&mut o, std::iter::once(("rate", "iters", "secs")));
    assert!(!o.contains_key("rate"), "inf quotient -> no rate key");
}

/// A Rate has no samples of its own, so the per-phase reducer returns
/// None (the build post-pass derives it from components instead).
#[test]
fn rate_kind_returns_none_from_per_phase_reducer() {
    let rate = MetricDef {
        name: "test_rate",
        accessor: |_| None,
        display_unit: "",
        polarity: crate::test_support::Polarity::HigherBetter,
        default_abs: 0.0,
        default_rel: 0.0,
        kind: MetricKind::Rate {
            numerator: "n",
            denominator: "d",
        },
    };
    assert_eq!(
        aggregate_samples_for_phase(&rate, &[1.0, 2.0, 3.0]),
        None,
        "Rate reduces to None per-phase; derive_rate_metrics owns it",
    );
}

/// Routing a Rate through the single-slice reducer is a bug — it
/// cannot express Σnum/Σdenom — so `aggregate_finite` panics rather
/// than silently producing a meaningless one-slice value.
#[test]
#[should_panic(expected = "must be derived via derive_rate_metrics")]
fn rate_kind_panics_in_single_slice_reducer() {
    let _ = aggregate_samples(
        &[1.0, 2.0],
        MetricKind::Rate {
            numerator: "n",
            denominator: "d",
        },
    );
}

/// All-empty / all-NaN inputs to either entry point return
/// `None`. The phase renderer treats absent values as "no
/// finite samples for this metric in this phase" — distinct
/// from `Some(0.0)` which is a real reduced zero from finite
/// samples — so the `None` shape must round-trip.
#[test]
fn aggregate_samples_for_phase_returns_none_on_empty_or_all_nan() {
    let counter = MetricDef {
        name: "total_x",
        accessor: |_| None,
        display_unit: "",
        polarity: crate::test_support::Polarity::HigherBetter,
        default_abs: 0.0,
        default_rel: 0.0,
        kind: MetricKind::Counter,
    };
    assert_eq!(aggregate_samples_for_phase(&counter, &[]), None);
    assert_eq!(
        aggregate_samples_for_phase(&counter, &[f64::NAN, f64::NAN]),
        None,
    );
    let peak = MetricDef {
        name: "max_x",
        accessor: |_| None,
        display_unit: "",
        polarity: crate::test_support::Polarity::LowerBetter,
        default_abs: 0.0,
        default_rel: 0.0,
        kind: MetricKind::Peak,
    };
    assert_eq!(aggregate_samples_for_phase(&peak, &[]), None);
    assert_eq!(
        aggregate_samples_for_phase(&peak, &[f64::NAN, f64::INFINITY]),
        None,
    );
}

/// Every entry in the `METRICS` registry must have a kind set.
/// Pinned via the registry walk so a future entry that forgot
/// to specify `kind` fails to compile (struct-literal
/// non_exhaustive forces it), and a registry entry whose kind
/// is `Counter` matches one of the well-known total/stuck_count
/// names — drift either direction trips here.
#[test]
fn every_metric_has_kind_consistent_with_naming() {
    for m in METRICS {
        // Counter and DeltaSum metrics are both cumulative totals
        // (Counter = since-boot, DeltaSum = sum of per-read deltas),
        // so both must be named with `total_` / `_count` /
        // `total_iterations` / `stuck_count` per the established
        // convention.
        if matches!(m.kind, MetricKind::Counter | MetricKind::DeltaSum) {
            assert!(
                m.name.starts_with("total_") || m.name.ends_with("_count"),
                "Counter/DeltaSum-kind metric must follow total_*/*_count naming, got {:?}",
                m.name,
            );
        }
        // Peak metrics must be named with `max_` or be the documented
        // worst-case high-water entry `worst_gap_ms`. (worst_run_delay_us
        // is now MetricKind::Distribution{RunDelayNs, Worst}, not Peak.)
        if matches!(m.kind, MetricKind::Peak) {
            assert!(
                m.name.starts_with("max_") || m.name == "worst_gap_ms",
                "Peak-kind metric must use max_* naming OR be a documented worst-* peak, got {:?}",
                m.name,
            );
        }
        // Distribution metrics are re-pooled run-level by
        // `populate_run_distribution_metrics`; for cgroups with no carried
        // samples (backdrop / stripped) it folds the per-cgroup
        // `distribution_cgroup_reduction` worst-wins via `f64::max` (in the
        // `populate_run_distribution_metrics_from` loop), which is the
        // correct worst-wins ONLY for LowerBetter metrics. Enforce that
        // coupling so a future HigherBetter Distribution cannot silently
        // invert the degraded-path regression signal (max would pick the
        // BEST cgroup as the "worst"). A HigherBetter Distribution must
        // first make that worst-wins fold polarity-aware.
        if matches!(m.kind, MetricKind::Distribution { .. }) {
            assert_eq!(
                m.polarity,
                crate::test_support::Polarity::LowerBetter,
                "Distribution-kind metric {:?} must be LowerBetter \
                     (the carrier-less fold maxes); got {:?}",
                m.name,
                m.polarity,
            );
        }
        // WorstLowest metrics are re-pooled by
        // `populate_run_distribution_metrics`'s lowest-wins fold
        // (`worst.is_none_or(|w| v < w)`), which treats the LOWEST per-cgroup
        // value as the worst — correct ONLY for HigherBetter metrics.
        // Enforce the mirror of the Distribution gate so a future
        // LowerBetter WorstLowest cannot silently invert the regression
        // signal (select the least-bad cgroup, mask the starved one); such a
        // metric must first make the lowest-wins fold polarity-aware.
        if matches!(m.kind, MetricKind::WorstLowest { .. }) {
            assert_eq!(
                m.polarity,
                crate::test_support::Polarity::HigherBetter,
                "WorstLowest-kind metric {:?} must be HigherBetter \
                     (the lowest-wins fold treats lowest as worst); got {:?}",
                m.name,
                m.polarity,
            );
        }
        // Rate metrics are derived ratios; name them `*_rate` or
        // `*_per_*` so the registry reads as a rate at a glance.
        if let MetricKind::Rate {
            numerator,
            denominator,
        } = m.kind
        {
            assert!(
                m.name.ends_with("_rate") || m.name.contains("_per_"),
                "Rate-kind metric must use *_rate or *_per_* naming, got {:?}",
                m.name,
            );
            // Components must be registered AND not themselves Rate:
            // derive_rate_metrics is a pure function of non-derived
            // components, which is what keeps the re-pool associative.
            // A rate-of-a-rate would make the post-pass order-dependent
            // on METRICS declaration order (the inner rate's key is
            // skipped in the merge loop, so a stale value could be read).
            for comp in [numerator, denominator] {
                let cd = metric_def(comp).unwrap_or_else(|| {
                    panic!(
                        "Rate metric {:?} component {comp:?} is not registered",
                        m.name
                    )
                });
                assert!(
                    !matches!(cd.kind, MetricKind::Rate { .. }),
                    "Rate metric {:?} component {comp:?} must not itself be Rate \
                         (a rate-of-a-rate breaks the associative re-derive)",
                    m.name,
                );
            }
        }

        // REVERSE gate: a metric NAMED like a per-second rate MUST be a
        // Rate, so a future per-second metric cannot silently ship as a
        // Gauge that averages ready-made ratios (the (r₁+r₂)/2 bug). Scoped
        // to per-SECOND tokens (`_rate` / `_per_sec` / `_per_cpu_sec`) — NOT
        // bare `_per_` — so a count-denominator metric like
        // `worst_iterations_per_worker` (a `WorstLowest` min-selection) is
        // not falsely flagged. `worst_iterations_per_cpu_sec` is the
        // documented exception: it is a `MetricKind::WorstLowest` (the
        // lowest cgroup's iterations / CPU-second, the per-cgroup
        // starvation signal selected lowest-wins), NOT a Σnum/Σdenom pooled
        // rate — so it is correctly NOT a Rate and keeps its accurate
        // `_per_cpu_sec` name. (The pooled cohort rate IS a Rate, under the
        // distinct name `iterations_per_cpu_sec`.)
        let looks_like_rate = m.name.ends_with("_rate")
            || m.name.contains("_per_sec")
            || m.name.contains("_per_cpu_sec");
        if looks_like_rate && m.name != "worst_iterations_per_cpu_sec" {
            assert!(
                matches!(m.kind, MetricKind::Rate { .. }),
                "metric {:?} is named like a per-second rate but is not \
                     MetricKind::Rate (register it as a Rate, or allowlist it \
                     here if it is intentionally a non-re-pooled gauge)",
                m.name,
            );
        }
    }
}

/// Replaces the legacy `col_mean_std_basic` polars-side check
/// with a direct test on the hand-rolled [`mean`] / [`std_dev`]
/// helpers that the new aggregation path uses end-to-end.
/// `1..=5` is symmetric around 3, so the mean lands exactly and
/// the Bessel-corrected std exceeds 1.0 (the population std of
/// `1..=5` is √2 ≈ 1.414; the sample std is √2.5 ≈ 1.581).
#[test]
fn mean_std_basic() {
    let xs = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let m = mean(xs.iter().copied());
    let s = std_dev(xs.iter().copied());
    assert!((m - 3.0).abs() < 0.01);
    assert!(s > 1.0);
}

/// Mirrors the legacy `col_mean_std_missing_column` defense:
/// when no finite values are present, both helpers return 0.0
/// rather than NaN / panic — the same "missing column → (0.0,
/// 0.0)" contract that polars's `col_mean_std` carried, expressed
/// over the iterator surface.
#[test]
fn mean_std_empty_returns_zero() {
    let empty: [f64; 0] = [];
    assert_eq!(mean(empty.iter().copied()), 0.0);
    assert_eq!(std_dev(empty.iter().copied()), 0.0);
    // Single finite value: mean is that value, std is 0.0 (Bessel
    // correction requires count >= 2; matches polars `.std(1)`).
    let single = [7.5_f64];
    assert!((mean(single.iter().copied()) - 7.5).abs() < f64::EPSILON);
    assert_eq!(std_dev(single.iter().copied()), 0.0);
}

/// Non-finite values (NaN, ±inf) must be ignored — a sentinel
/// metric value can't poison the cohort mean. Pins the
/// `.filter(|x| x.is_finite())` arm in both helpers.
#[test]
fn mean_std_skips_non_finite() {
    let xs = [1.0_f64, f64::NAN, 3.0, f64::INFINITY, 5.0];
    // Only 1, 3, 5 contribute. Mean = 3.0; sample std = 2.0.
    assert!((mean(xs.iter().copied()) - 3.0).abs() < 1e-9);
    assert!((std_dev(xs.iter().copied()) - 2.0).abs() < 1e-9);
}

/// Mean and std_dev handle negative values correctly.
/// Negative values are valid for metrics like migration_ratio.
#[test]
fn mean_std_handles_negative_values() {
    let xs = [-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let m = mean(xs.iter().copied());
    let s = std_dev(xs.iter().copied());
    assert!(
        (m - 0.0).abs() < 1e-9,
        "mean of symmetric values should be 0, got {m}"
    );
    // Sample std for [-2, -1, 0, 1, 2]: sqrt( (4+1+0+1+4) / 4 ) = sqrt(10/4) = sqrt(2.5) ≈ 1.581
    assert!((s - 1.58113883).abs() < 1e-6, "std dev mismatch, got {s}");
}

/// Mean and std_dev handle large values without overflow.
/// Uses values near f64::MAX / 10 to avoid actual overflow in sum.
#[test]
fn mean_std_handles_large_values() {
    let large = 1e150_f64;
    let xs = [large, large * 2.0, large * 3.0];
    let m = mean(xs.iter().copied());
    let s = std_dev(xs.iter().copied());
    assert!(
        (m - large * 2.0).abs() / large < 1e-12,
        "mean of large values"
    );
    // Std dev should be large * sqrt( (1^2 + 0^2 + 1^2) / 2 ) = large * 1.0
    assert!((s - large).abs() / large < 1e-12, "std dev of large values");
}

/// Mean and std_dev handle subnormal (very small) values.
/// Subnormals are finite and should be included in calculations.
#[test]
fn mean_std_handles_subnormal_values() {
    let tiny = f64::MIN_POSITIVE / 2.0; // Subnormal
    let xs = [tiny, tiny * 2.0, tiny * 3.0];
    let m = mean(xs.iter().copied());
    // The sum is 6 subnormal ULPs and 6/3 = 2 ULPs exactly, so an
    // exact equality is correct and catches wrong-divisor,
    // partial-sum, and first-element bugs that a finite/positive
    // check would admit (e.g. count-1 → 3*tiny, first sample → tiny).
    assert_eq!(m, 2.0 * tiny, "subnormals must be summed/averaged exactly");
}

/// std_dev with exactly two values uses Bessel's correction (ddof=1).
/// For [a, b], std = sqrt( ((a-m)^2 + (b-m)^2) / 1 ) where m = (a+b)/2.
/// This simplifies to |a-b| / sqrt(2) * sqrt(2) = |a-b|.
#[test]
fn std_dev_two_values_bessel_corrected() {
    let xs = [3.0_f64, 7.0];
    let s = std_dev(xs.iter().copied());
    // Mean = 5.0, deviations = [-2, 2], sum_sq = 8, variance = 8 / 1 = 8, std = sqrt(8) ≈ 2.828
    assert!(
        (s - 2.8284271247461903).abs() < 1e-9,
        "Bessel-corrected std dev for two values"
    );
}

// -- find_outliers and find_worst_topos tests --

/// find_outliers returns empty vec when input is empty.
#[test]
fn find_outliers_empty_input() {
    let rows: Vec<GauntletRow> = vec![];
    let outliers = find_outliers(&rows);
    assert!(outliers.is_empty(), "empty input should yield no outliers");
}

/// find_outliers returns empty vec when no rows pass the filter.
/// Only rows with is_pass() == true are considered for outlier detection.
#[test]
fn find_outliers_no_pass_rows() {
    let r1 = make_row("s1", "t1", false, 10.0); // failed
    let mut r2 = make_row("s2", "t2", true, 20.0);
    r2.skipped = true; // skipped, not a pass
    let rows = vec![r1, r2];
    let outliers = find_outliers(&rows);
    assert!(outliers.is_empty(), "no pass rows should yield no outliers");
}

/// find_outliers returns empty vec when only one scenario exists.
/// With a single scenario, the scenario mean equals the overall mean,
/// so no scenario can exceed the 2-sigma threshold.
#[test]
fn find_outliers_single_scenario_no_outlier() {
    let r1 = make_row("only", "t1", true, 10.0);
    let r2 = make_row("only", "t2", true, 12.0);
    let r3 = make_row("only", "t3", true, 11.0);
    let rows = vec![r1, r2, r3];
    let outliers = find_outliers(&rows);
    assert!(
        outliers.is_empty(),
        "single scenario cannot produce outliers"
    );
}

/// find_outliers detects a clear outlier scenario.
/// Three scenarios: two with low spread (5.0), one with high spread (100.0).
/// The high-spread scenario should be flagged as an outlier on the "spread" metric.
#[test]
fn find_outliers_detects_clear_outlier() {
    // Many "normal" scenarios with low spread (10 rows total)
    let mut rows = Vec::new();
    for i in 0..5 {
        rows.push(make_row("normal1", &format!("t{i}"), true, 5.0));
    }
    for i in 5..10 {
        rows.push(make_row("normal2", &format!("t{i}"), true, 5.0));
    }
    // One outlier scenario with high spread (100+ to ensure > 2-sigma)
    rows.push(make_row("outlier", "t10", true, 100.0));
    rows.push(make_row("outlier", "t11", true, 110.0));

    let outliers = find_outliers(&rows);
    assert!(!outliers.is_empty(), "should detect outlier scenario");

    // Find the spread outlier
    let spread_outlier = outliers.iter().find(|o| o.metric == "spread");
    assert!(
        spread_outlier.is_some(),
        "should have spread metric outlier"
    );
    let outlier = spread_outlier.unwrap();
    assert_eq!(outlier.scenario, "outlier");
    assert!(
        outlier.sigma > 2.0,
        "sigma should exceed 2.0 threshold, got {}",
        outlier.sigma
    );
    // Worst topos should include t10 and t11 (both exceed threshold)
    assert!(outlier.worst_topos.contains(&"t10".to_string()));
    assert!(outlier.worst_topos.contains(&"t11".to_string()));
}

/// find_outliers does NOT flag a scenario at exactly the 2-sigma threshold.
/// The condition is `scenario_mean > threshold`, not `>=`, so exact equality
/// should not produce an outlier. This prevents borderline cases from noise.
#[test]
fn find_outliers_threshold_is_strictly_greater() {
    // Create a scenario where mean is exactly at 2-sigma threshold.
    // For simplicity, use 3 scenarios: two with value 0.0, one with value X.
    // Overall mean = X/3, std = sqrt( ((0-m)^2 + (0-m)^2 + (X-m)^2) / 2 )
    // We want X = mean + 2*std. Solving: X = X/3 + 2*std => 2X/3 = 2*std => X/3 = std
    // std^2 = (m^2 + m^2 + (X-m)^2) / 2 = (2m^2 + (2m)^2) / 2 where m = X/3
    // = (2m^2 + 4m^2) / 2 = 3m^2 => std = m*sqrt(3) = X*sqrt(3)/3
    // Setting X/3 = X*sqrt(3)/3 => 1 = sqrt(3) which is false.
    // Simpler: use many normal rows and one borderline row.
    // Let's construct: 10 rows with value 0.0, 1 row with value V.
    // Mean = V/11, variance = (10*(V/11)^2 + (V - V/11)^2) / 10
    // = (10*V^2/121 + (10V/11)^2) / 10 = (10V^2/121 + 100V^2/121) / 10 = 110V^2 / 1210 = V^2 / 11
    // std = V / sqrt(11), threshold = V/11 + 2V/sqrt(11)
    // For the single row scenario, mean = V. We want V = threshold.
    // V = V/11 + 2V/sqrt(11) => 1 = 1/11 + 2/sqrt(11) => 10/11 = 2/sqrt(11) => sqrt(11) = 22/10 = 2.2 => 11 = 4.84 false.
    // This is getting complex. Instead, test the boundary condition directly:
    // Create a case where scenario_mean is just below threshold (no outlier)
    // and another just above (outlier).

    // Use 5 normal scenarios with spread 10.0, and 1 test scenario with spread 30.0
    // Normal: 5*2 = 10 rows at 10.0. Test: 2 rows at 30.0.
    // Overall mean = (100 + 60) / 12 = 160/12 ≈ 13.33
    // Std: variance = [10*(10-13.33)^2 + 2*(30-13.33)^2] / 11
    // = [10*11.09 + 2*277.78] / 11 = [110.9 + 555.56] / 11 = 666.46 / 11 ≈ 60.59, std ≈ 7.78
    // Threshold = 13.33 + 2*7.78 = 13.33 + 15.56 = 28.89
    // Test scenario mean = 30.0 > 28.89, so it IS an outlier.
    // To test the boundary, we need a value just below threshold.

    // Simpler approach: verify that the > comparison is used, not >=.
    // Create a minimal case where we know the threshold and test both sides.
    // 4 rows: two scenarios with 2 rows each
    // Scenario A: values 0.0, 0.0 (mean 0.0)
    // Scenario B: values 10.0, 10.0 (mean 10.0)
    // Overall mean = 5.0, std = sqrt( ((0-5)^2*2 + (10-5)^2*2) / 3 ) = sqrt( (50 + 50) / 3 ) = sqrt(100/3) ≈ 5.77
    // Threshold = 5.0 + 2*5.77 = 5.0 + 11.55 = 16.55
    // Neither scenario mean (0.0 or 10.0) exceeds 16.55, so no outliers.
    let rows = vec![
        make_row("A", "t1", true, 0.0),
        make_row("A", "t2", true, 0.0),
        make_row("B", "t3", true, 10.0),
        make_row("B", "t4", true, 10.0),
    ];

    let outliers = find_outliers(&rows);
    // No scenario should be an outlier since 10.0 < 16.55
    let spread_outliers: Vec<_> = outliers.iter().filter(|o| o.metric == "spread").collect();
    assert!(
        spread_outliers.is_empty(),
        "no outlier when below threshold"
    );
}

/// Companion to the below-threshold case: a scenario whose mean is
/// far above overall_mean + 2*std MUST be flagged. The strictly-
/// greater test only proves the absence side (10.0 < 16.55); without
/// this, a never-flags or inverted (`<` instead of `>`) comparison
/// passes. 10 scenarios at spread 10 + 1 at 100: overall mean ~18,
/// std ~27, threshold ~72; the 100 scenario clears it decisively.
#[test]
fn find_outliers_flags_scenario_above_threshold() {
    let mut rows: Vec<GauntletRow> = (0..10)
        .map(|i| make_row(&format!("normal{i}"), "t", true, 10.0))
        .collect();
    rows.push(make_row("hot", "t", true, 100.0));
    let outliers = find_outliers(&rows);
    let spread: Vec<_> = outliers.iter().filter(|o| o.metric == "spread").collect();
    assert!(
        spread.iter().any(|o| o.scenario == "hot"),
        "a scenario far above the 2-sigma threshold must be flagged as a spread outlier",
    );
}

/// find_outliers skips metrics with near-zero standard deviation.
/// When std < f64::EPSILON, the metric has no measurable spread,
/// so outlier detection would be noise. The function should skip it.
#[test]
fn find_outliers_skips_zero_std_metrics() {
    // All rows have identical spread value, so std = 0.0
    let r1 = make_row("s1", "t1", true, 10.0);
    let r2 = make_row("s2", "t2", true, 10.0);
    let r3 = make_row("s3", "t3", true, 10.0);
    let rows = vec![r1, r2, r3];

    let outliers = find_outliers(&rows);
    // No spread outliers because std = 0 (below EPSILON)
    let spread_outliers: Vec<_> = outliers.iter().filter(|o| o.metric == "spread").collect();
    assert!(
        spread_outliers.is_empty(),
        "zero std dev should skip metric"
    );
}

/// find_outliers sorts results by sigma descending.
/// Higher sigma (more extreme outliers) should appear first.
#[test]
fn find_outliers_sorts_by_sigma_descending() {
    // Create two outlier scenarios with different sigma values
    // Normal baseline: 30 rows with spread 5.0 (large baseline for stable std)
    let mut rows = Vec::new();
    for i in 0..15 {
        rows.push(make_row("normal1", &format!("t{i}"), true, 5.0));
    }
    for i in 15..30 {
        rows.push(make_row("normal2", &format!("t{i}"), true, 5.0));
    }
    // Outlier 1: (spread 100.0, 110.0)
    rows.push(make_row("outlier1", "t30", true, 100.0));
    rows.push(make_row("outlier1", "t31", true, 110.0));
    // Outlier 2: more extreme (spread 140.0, 150.0)
    // Both should exceed threshold, but outlier2 has higher sigma
    rows.push(make_row("outlier2", "t40", true, 140.0));
    rows.push(make_row("outlier2", "t41", true, 150.0));

    let outliers = find_outliers(&rows);
    let spread_outliers: Vec<_> = outliers.iter().filter(|o| o.metric == "spread").collect();
    assert!(
        spread_outliers.len() >= 2,
        "should have at least 2 spread outliers, got {}",
        spread_outliers.len()
    );

    // Outlier2 should have higher sigma and appear first
    let first = &spread_outliers[0];
    let second = &spread_outliers[1];
    assert!(
        first.sigma >= second.sigma,
        "outliers should be sorted by sigma descending"
    );
    assert_eq!(
        first.scenario, "outlier2",
        "extreme outlier should be first"
    );
}

/// find_worst_topos returns empty vec when no rows match scenario.
#[test]
fn find_worst_topos_empty_when_no_matching_scenario() {
    let r1 = make_row("s1", "t1", true, 10.0);
    let rows = vec![r1];
    let accessor: MetricAccessor = |r| r.spread;
    let worst = find_worst_topos(&rows, "nonexistent", accessor, 5.0);
    assert!(
        worst.is_empty(),
        "no matching scenario should yield empty vec"
    );
}

/// find_worst_topos returns only topologies exceeding threshold.
/// The threshold comparison is strict (>), not >=.
#[test]
fn find_worst_topos_filters_by_threshold() {
    let mut r1 = make_row("s1", "t1", true, 10.0);
    r1.spread = 5.0;
    let mut r2 = make_row("s1", "t2", true, 10.0);
    r2.spread = 15.0;
    let mut r3 = make_row("s1", "t3", true, 10.0);
    r3.spread = 25.0;
    let rows = vec![r1, r2, r3];

    let accessor: MetricAccessor = |r| r.spread;
    // Threshold 15.0: t1 (5.0) no, t2 (15.0) no (strict >), t3 (25.0) yes
    let worst = find_worst_topos(&rows, "s1", accessor, 15.0);
    assert_eq!(worst.len(), 1, "only t3 should exceed threshold");
    assert!(worst.contains(&"t3".to_string()));
    assert!(
        !worst.contains(&"t2".to_string()),
        "t2 at threshold should not be included"
    );
}

/// find_worst_topos includes rows regardless of pass/fail status.
/// The function is used to attribute outliers to topologies, and a
/// failing topology that drives high metric values should still surface.
#[test]
fn find_worst_topos_includes_failed_rows() {
    let mut r1 = make_row("s1", "t1", true, 10.0); // pass
    r1.spread = 30.0;
    let mut r2 = make_row("s1", "t2", false, 10.0); // fail
    r2.spread = 40.0;
    let rows = vec![r1, r2];

    let accessor: MetricAccessor = |r| r.spread;
    let worst = find_worst_topos(&rows, "s1", accessor, 20.0);
    // Both should be included regardless of pass/fail
    assert_eq!(worst.len(), 2);
    assert!(worst.contains(&"t1".to_string()));
    assert!(worst.contains(&"t2".to_string()));
}

/// group_field returns None for unknown column names.
/// Only "scenario", "topology", and "work_type" are valid dimensions.
#[test]
fn group_field_unknown_column_returns_none() {
    let row = make_row("s1", "t1", true, 10.0);
    assert!(group_field(&row, "scenario").is_some());
    assert!(group_field(&row, "topology").is_some());
    assert!(group_field(&row, "work_type").is_some());
    assert!(group_field(&row, "invalid").is_none());
    assert!(group_field(&row, "").is_none());
    assert!(group_field(&row, "Spread").is_none()); // case-sensitive
}

/// group_field extracts the correct field for each valid dimension.
#[test]
fn group_field_extracts_correct_dimension() {
    let row = make_row("my_scenario", "my_topo", true, 10.0);
    assert_eq!(group_field(&row, "scenario"), Some("my_scenario"));
    assert_eq!(group_field(&row, "topology"), Some("my_topo"));
    assert_eq!(group_field(&row, "work_type"), Some("SpinWait"));
}

fn make_row(scenario: &str, topo: &str, passed: bool, spread: f64) -> GauntletRow {
    GauntletRow {
        scenario: scenario.into(),
        topology: topo.into(),
        work_type: "SpinWait".into(),
        scheduler: String::new(),
        kernel_version: None,
        cpu_budget: None,
        vcpus: None,
        commit: None,
        kernel_commit: None,
        run_source: None,
        skipped: false,
        passed,
        inconclusive: false,
        run_sample_count: 0,
        spread,
        gap_ms: 50,
        migrations: 10,
        migration_ratio: 0.0,
        imbalance_ratio: 1.0,
        max_dsq_depth: 2,
        stuck_count: 0,
        fallback_count: 0,
        keep_last_count: 0,
        total_iterations: 0,
        page_locality: 0.0,
        cross_node_migration_ratio: 0.0,
        ext_metrics: BTreeMap::new(),
        phases: Vec::new(),
    }
}

// -- format_dimension_summary tests --

#[test]
fn format_dimension_summary_computed_values() {
    // Two scenarios: "fast" with spread=4.0, gap=40, and "slow" with spread=20.0, gap=200.
    // Each has 1 row. format_dimension_summary sorts by avg_spread descending.
    let mut r1 = make_row("slow", "tiny-1llc", false, 20.0);
    r1.gap_ms = 200;
    r1.imbalance_ratio = 2.5; // > 1.0, should show imbal=2.5
    r1.max_dsq_depth = 8; // > 0, should show dsq=8
    r1.stuck_count = 2; // > 0, should show stuck=2
    r1.fallback_count = 15; // > 0, should show fallback=15
    let r2 = make_row("fast", "tiny-1llc", true, 4.0);
    let rows = vec![r1, r2];
    let out = format_dimension_summary(&rows, "scenario");
    // "slow" has higher spread, should appear first (sorted descending).
    let slow_pos = out.find("slow").unwrap();
    let fast_pos = out.find("fast").unwrap();
    assert!(
        slow_pos < fast_pos,
        "slow should sort before fast, got:\n{out}"
    );
    // Check computed values for "slow"
    assert!(out.contains("0/1 passed"), "slow: 0/1 passed, got:\n{out}");
    assert!(
        out.contains("avg_spread=20.0%"),
        "slow: avg_spread=20.0%, got:\n{out}"
    );
    assert!(
        out.contains("avg_gap=200ms"),
        "slow: avg_gap=200ms, got:\n{out}"
    );
    assert!(out.contains("imbal=2.5"), "slow: imbal=2.5, got:\n{out}");
    assert!(out.contains("dsq=8"), "slow: dsq=8, got:\n{out}");
    assert!(out.contains("stuck=2"), "slow: stuck=2, got:\n{out}");
    assert!(
        out.contains("fallback=15"),
        "slow: fallback=15, got:\n{out}"
    );
    // "fast" should show 1/1 passed
    assert!(out.contains("1/1 passed"), "fast: 1/1 passed, got:\n{out}");
}

/// A row whose `inconclusive` bit is set must render in the
/// "inconclusive" bucket of the dimension-summary line, NOT
/// silently fold into the "failed" bucket as the arithmetic
/// `fail = total - pass - skip` would have done. The current
/// arithmetic `fail = total - pass - skip - inconc` plus the
/// format string `"({} skipped, {} inconclusive, {} failed)"`
/// together produce the correct breakdown.
///
/// A future polars-aggregation refactor that drops the
/// `inconc_count` agg, the `inconclusive` group-by column, or the
/// subtraction at the residual would re-introduce the
/// misclassification — this test fails loudly in that case.
#[test]
fn format_dimension_summary_renders_inconclusive_bucket_distinctly() {
    // Three rows on the same dimension: one Pass, one
    // Inconclusive (passed=false, inconclusive=true), one Fail
    // (all-false). Triple-state coverage in one dimension
    // confirms that the format string surfaces every non-pass
    // bucket honestly.
    let mut r_pass = make_row("group_a", "t1", true, 5.0);
    r_pass.skipped = false;
    r_pass.inconclusive = false;
    let mut r_inc = make_row("group_a", "t1", false, 5.0);
    r_inc.skipped = false;
    r_inc.inconclusive = true;
    let mut r_fail = make_row("group_a", "t1", false, 5.0);
    r_fail.skipped = false;
    r_fail.inconclusive = false;
    let rows = vec![r_pass, r_inc, r_fail];
    let out = format_dimension_summary(&rows, "scenario");
    assert!(
        out.contains("1/3 passed"),
        "expected '1/3 passed' for 1-pass-of-3: got:\n{out}"
    );
    assert!(
        out.contains("1 inconclusive"),
        "inconclusive row must NOT silently fold into the failed \
             bucket; got:\n{out}"
    );
    assert!(
        out.contains("1 failed"),
        "real Fail row must render as 1 failed (not be hidden by \
             the inconclusive subtraction); got:\n{out}"
    );
    assert!(
        out.contains("0 skipped"),
        "no Skip contributor; skipped bucket must be 0: got:\n{out}"
    );
}

// -- analyze_rows tests --

#[test]
fn analyze_rows_empty() {
    assert!(analyze_rows(&[]).is_empty());
}

#[test]
fn analyze_rows_with_work_type_diversity() {
    let mut rows = vec![
        make_row("a", "t1", true, 5.0),
        make_row("a", "t1", true, 6.0),
    ];
    rows[0].work_type = "SpinWait".into();
    rows[1].work_type = "Bursty".into();
    let report = analyze_rows(&rows);
    assert!(
        report.contains("By work_type"),
        "should show work_type section when diverse"
    );
    assert!(report.contains("SpinWait"), "should list SpinWait");
    assert!(report.contains("Bursty"), "should list Bursty");
}

#[test]
fn analyze_rows_no_work_type_section_when_uniform() {
    let rows = vec![
        make_row("a", "t1", true, 5.0),
        make_row("b", "t2", true, 8.0),
    ];
    let report = analyze_rows(&rows);
    assert!(
        !report.contains("By work_type"),
        "should not show work_type when uniform"
    );
}

// -- sidecar_to_row tests --

#[test]
fn sidecar_to_row_basic() {
    use crate::monitor;
    use crate::test_support;
    let sc = test_support::SidecarResult {
        test_name: "my_test".to_string(),
        topology: "1n2l4c2t".to_string(),
        scheduler: "scx_mitosis".to_string(),
        stats: ScenarioStats {
            cgroups: vec![],
            total_workers: 4,
            total_cpus: 8,
            total_migrations: 12,
            worst_spread: 15.0,
            worst_gap_ms: 200,
            worst_gap_cpu: 3,
            ..Default::default()
        },
        monitor: Some(monitor::MonitorSummary {
            total_samples: 10,
            max_imbalance_ratio: 2.5,
            max_local_dsq_depth: 4,
            stuck_detected: true,
            event_deltas: Some(monitor::ScxEventDeltas {
                total_fallback: 7,
                fallback_rate: 0.5,
                max_fallback_burst: 2,
                total_dispatch_offline: 0,
                total_dispatch_keep_last: 3,
                keep_last_rate: 0.2,
                total_enq_skip_exiting: 0,
                total_enq_skip_migration_disabled: 0,
                ..Default::default()
            }),
            schedstat_deltas: None,
            prog_stats_deltas: None,
            ..Default::default()
        }),
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(row.scenario, "my_test");
    assert_eq!(row.topology, "1n2l4c2t");
    assert!(row.is_pass());
    assert_eq!(row.spread, 15.0);
    assert_eq!(row.gap_ms, 200);
    assert_eq!(row.migrations, 12);
    assert_eq!(row.imbalance_ratio, 2.5);
    assert_eq!(row.max_dsq_depth, 4);
    assert_eq!(row.stuck_count, 1);
    assert_eq!(row.fallback_count, 7);
    assert_eq!(row.keep_last_count, 3);
}

/// `worst_iterations_per_cpu_sec` (the overcommit-invariant compare
/// metric the budget warning recommends) flows ScenarioStats ->
/// worst_iterations_per_cpu_sec is now a `MetricKind::WorstLowest` ext
/// metric (re-pooled post-merge by populate_run_distribution_metrics).
/// sidecar_to_row carries it through the ext_metrics copy, and
/// MetricDef::read surfaces it via the `|_| None` accessor's ext fallback;
/// an absent key (no cgroup reported a defined rate) reads None, distinct
/// from a measured 0.0. The metric stays registered so
/// `stats compare --metric worst_iterations_per_cpu_sec` resolves.
#[test]
fn sidecar_to_row_carries_worst_iterations_per_cpu_sec_via_ext() {
    use crate::test_support;
    let mut stats = ScenarioStats::default();
    stats
        .ext_metrics
        .insert("worst_iterations_per_cpu_sec".to_string(), 1234.5);
    let present = test_support::SidecarResult {
        stats,
        ..test_support::SidecarResult::test_fixture()
    };
    let def = metric_def("worst_iterations_per_cpu_sec")
        .expect("metric must be registered so `stats compare --metric` resolves it");
    assert_eq!(def.read(&sidecar_to_row(&present)), Some(1234.5));

    // Absent key (re-pool wrote nothing) → read None, NOT a measured 0.0.
    let absent = test_support::SidecarResult {
        stats: ScenarioStats::default(),
        ..test_support::SidecarResult::test_fixture()
    };
    assert_eq!(def.read(&sidecar_to_row(&absent)), None);
}

#[test]
fn sidecar_to_row_no_monitor() {
    use crate::test_support;
    let sc = test_support::SidecarResult {
        test_name: "eevdf_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        passed: false,
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(row.scenario, "eevdf_test");
    assert!(row.is_fail());
    assert_eq!(row.imbalance_ratio, 0.0);
    assert_eq!(row.max_dsq_depth, 0);
    assert_eq!(row.stuck_count, 0);
    assert_eq!(row.fallback_count, 0);
    assert_eq!(row.keep_last_count, 0);
}

/// `sidecar_to_row` must copy `SidecarResult::project_commit`
/// into `GauntletRow::commit` verbatim so the typed
/// `--project-commit` filter (and any sibling slicers added
/// later) see the value the sidecar writer recorded. A
/// regression that left the field at the
/// `Option::default()` (`None`) would silently drop the
/// commit dimension from every comparison even when the
/// sidecar had a populated value. Pinned for `None`, clean
/// `Some` (no suffix), and dirty `Some` (`-dirty` suffix) to
/// catch a regression that special-cases one shape and not
/// the others — e.g. one that stripped the suffix when copying.
#[test]
fn sidecar_to_row_propagates_project_commit() {
    use crate::test_support;
    let sc_dirty = test_support::SidecarResult {
        test_name: "commit_dirty_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        project_commit: Some("abcdef1-dirty".to_string()),
        ..test_support::SidecarResult::test_fixture()
    };
    let row_dirty = sidecar_to_row(&sc_dirty);
    assert_eq!(
        row_dirty.commit.as_deref(),
        Some("abcdef1-dirty"),
        "populated dirty project_commit must propagate \
             verbatim, including the `-dirty` suffix",
    );

    let sc_clean = test_support::SidecarResult {
        test_name: "commit_clean_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        project_commit: Some("abcdef1".to_string()),
        ..test_support::SidecarResult::test_fixture()
    };
    let row_clean = sidecar_to_row(&sc_clean);
    assert_eq!(
        row_clean.commit.as_deref(),
        Some("abcdef1"),
        "populated clean project_commit (no `-dirty` suffix) \
             must propagate verbatim — a regression that always \
             appended `-dirty` or always stripped a tail would \
             surface here independently of the dirty case above",
    );

    let sc_none = test_support::SidecarResult {
        test_name: "no_commit_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        project_commit: None,
        ..test_support::SidecarResult::test_fixture()
    };
    let row_none = sidecar_to_row(&sc_none);
    assert!(
        row_none.commit.is_none(),
        "absent project_commit must propagate as None — a \
             regression substituting an empty string would dilute \
             every `--project-commit` filter into matching all None rows",
    );
}

/// `sidecar_to_row` must copy `SidecarResult::kernel_commit`
/// into `GauntletRow::kernel_commit` verbatim so the typed
/// `--kernel-commit` filter and per-side
/// `--a-kernel-commit` / `--b-kernel-commit` slicers see the
/// value the sidecar writer recorded. A regression that left
/// the field at the `Option::default()` (`None`) would
/// silently drop the kernel-commit dimension from every
/// comparison even when the sidecar had a populated value.
/// Mirrors `sidecar_to_row_propagates_project_commit` for
/// the kernel_commit field; pinned for `None`, clean `Some`
/// (no suffix), and dirty `Some` (`-dirty` suffix) to catch
/// a regression that special-cases one shape and not the
/// others.
#[test]
fn sidecar_to_row_propagates_kernel_commit() {
    use crate::test_support;
    let sc_dirty = test_support::SidecarResult {
        test_name: "kc_dirty_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        kernel_commit: Some("kabcde7-dirty".to_string()),
        ..test_support::SidecarResult::test_fixture()
    };
    let row_dirty = sidecar_to_row(&sc_dirty);
    assert_eq!(
        row_dirty.kernel_commit.as_deref(),
        Some("kabcde7-dirty"),
        "populated dirty kernel_commit must propagate \
             verbatim, including the `-dirty` suffix",
    );

    let sc_clean = test_support::SidecarResult {
        test_name: "kc_clean_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        kernel_commit: Some("kabcde7".to_string()),
        ..test_support::SidecarResult::test_fixture()
    };
    let row_clean = sidecar_to_row(&sc_clean);
    assert_eq!(
        row_clean.kernel_commit.as_deref(),
        Some("kabcde7"),
        "populated clean kernel_commit (no `-dirty` suffix) \
             must propagate verbatim — a regression that always \
             appended `-dirty` or always stripped a tail would \
             surface here independently of the dirty case above",
    );

    let sc_none = test_support::SidecarResult {
        test_name: "no_kc_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        kernel_commit: None,
        ..test_support::SidecarResult::test_fixture()
    };
    let row_none = sidecar_to_row(&sc_none);
    assert!(
        row_none.kernel_commit.is_none(),
        "absent kernel_commit must propagate as None — a \
             regression substituting an empty string would dilute \
             every `--kernel-commit` filter into matching all \
             None rows",
    );

    // Field non-aliasing pin: kernel_commit and commit must
    // route to distinct row fields. A regression that
    // accidentally cross-wired the two (e.g. `commit:
    // sc.kernel_commit.clone()` instead of
    // `sc.project_commit.clone()`) would hide behind the
    // populated tests above unless the values differ — which
    // they do here. Distinct tokens make the swap obvious.
    let sc_both = test_support::SidecarResult {
        test_name: "both_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        project_commit: Some("project1".to_string()),
        kernel_commit: Some("kernel1".to_string()),
        ..test_support::SidecarResult::test_fixture()
    };
    let row_both = sidecar_to_row(&sc_both);
    assert_eq!(
        row_both.commit.as_deref(),
        Some("project1"),
        "row.commit must come from project_commit, not kernel_commit",
    );
    assert_eq!(
        row_both.kernel_commit.as_deref(),
        Some("kernel1"),
        "row.kernel_commit must come from kernel_commit, not project_commit",
    );
}

/// `sidecar_to_row` must copy `SidecarResult::run_source` into
/// `GauntletRow::run_source` verbatim so the typed `--run-source`
/// filter and per-side `--a-run-source` / `--b-run-source` slicers
/// see the run-environment provenance tag the sidecar writer
/// recorded. A regression that left the field at the
/// `Option::default()` (`None`) would silently drop the
/// run-source dimension from every comparison even when the
/// sidecar had a populated value. Mirrors
/// `sidecar_to_row_propagates_kernel_commit` for the
/// `run_source` field; pinned for `None` and the canonical
/// `Some("local")` / `Some("ci")` / `Some("archive")`
/// values so a regression that special-cased one tag and
/// not the others surfaces here. A non-aliasing pin
/// confirms `run_source` reads from `sc.run_source` rather
/// than being cross-wired to the visually-similar
/// `kernel_commit` / `project_commit` fields.
#[test]
fn sidecar_to_row_propagates_run_source() {
    use crate::test_support;
    for tag in ["local", "ci", "archive"] {
        let sc = test_support::SidecarResult {
            test_name: format!("run_source_{tag}_test"),
            topology: "1n1l2c1t".to_string(),
            run_source: Some(tag.to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(
            row.run_source.as_deref(),
            Some(tag),
            "populated run_source `{tag}` must propagate verbatim",
        );
    }

    let sc_none = test_support::SidecarResult {
        test_name: "no_run_source_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        run_source: None,
        ..test_support::SidecarResult::test_fixture()
    };
    let row_none = sidecar_to_row(&sc_none);
    assert!(
        row_none.run_source.is_none(),
        "absent run_source must propagate as None — a regression \
             substituting an empty string would dilute every \
             `--run-source` filter into matching all None rows",
    );

    // Field non-aliasing pin: `run_source` must route to its
    // own row field. A regression that cross-wired
    // `run_source` to `kernel_commit` (or vice versa) would
    // hide behind the populated tests above unless the values
    // are visibly different. Distinct tokens make the swap
    // obvious.
    let sc_distinct = test_support::SidecarResult {
        test_name: "run_source_distinct_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        run_source: Some("local".to_string()),
        kernel_commit: Some("kabcde7".to_string()),
        project_commit: Some("pabcde7".to_string()),
        ..test_support::SidecarResult::test_fixture()
    };
    let row_distinct = sidecar_to_row(&sc_distinct);
    assert_eq!(
        row_distinct.run_source.as_deref(),
        Some("local"),
        "row.run_source must come from sc.run_source, not from \
             kernel_commit or project_commit",
    );
    assert_eq!(
        row_distinct.kernel_commit.as_deref(),
        Some("kabcde7"),
        "row.kernel_commit must remain sourced from sc.kernel_commit",
    );
    assert_eq!(
        row_distinct.commit.as_deref(),
        Some("pabcde7"),
        "row.commit must remain sourced from sc.project_commit",
    );
}

#[test]
fn sidecar_to_row_no_stall() {
    use crate::monitor;
    use crate::test_support;
    let sc = test_support::SidecarResult {
        monitor: Some(monitor::MonitorSummary {
            prog_stats_deltas: None,
            total_samples: 5,
            max_imbalance_ratio: 1.0,
            max_local_dsq_depth: 0,
            stuck_detected: false,
            event_deltas: None,
            schedstat_deltas: None,
            ..Default::default()
        }),
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(row.stuck_count, 0);
    assert_eq!(row.fallback_count, 0);
    assert_eq!(row.keep_last_count, 0);
}

/// Drive every direct f64 field on [`GauntletRow`] through
/// `finite_or_zero` with `non_finite` planted in the source
/// [`SidecarResult`], then assert each lands as 0.0 on the row.
///
/// Covers the `finite_or_zero` call sites in `sidecar_to_row`: the
/// remaining direct [`ScenarioStats`] f64 fields (worst_spread,
/// worst_migration_ratio, worst_page_locality,
/// worst_cross_node_migration_ratio) plus
/// `imbalance_ratio` from [`MonitorSummary`]. (The wake / run-delay
/// roll-ups are now ext_metrics-sourced — non-finite ext entries are
/// DROPPED, covered by `sidecar_to_row_drops_non_finite_ext_metrics`.) A
/// missed call site would leave one assert comparing the non-finite input
/// to 0.0 (NaN != 0.0, ±Infinity != 0.0) and fail the test.
fn assert_all_direct_f64_fields_sanitized(non_finite: f64) {
    use crate::assert::ScenarioStats;
    use crate::monitor::MonitorSummary;
    use crate::test_support;
    let sc = test_support::SidecarResult {
        stats: ScenarioStats {
            worst_spread: non_finite,
            worst_migration_ratio: non_finite,
            worst_page_locality: non_finite,
            worst_cross_node_migration_ratio: non_finite,
            ..Default::default()
        },
        monitor: Some(MonitorSummary {
            max_imbalance_ratio: non_finite,
            ..Default::default()
        }),
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    for (name, val) in [
        ("spread", row.spread),
        ("migration_ratio", row.migration_ratio),
        ("imbalance_ratio", row.imbalance_ratio),
        ("page_locality", row.page_locality),
        ("cross_node_migration_ratio", row.cross_node_migration_ratio),
    ] {
        assert_eq!(
            val, 0.0,
            "{name} must collapse to 0.0 for non-finite input {non_finite:?}",
        );
    }
    // Motivation check: the sanitized row serializes. Without the
    // `finite_or_zero` wraps, serde_json::to_string would return
    // Err because NaN / Infinity have no JSON representation.
    serde_json::to_string(&row).expect("sanitized row must serialize cleanly");
}

/// `sidecar_to_row` must sanitize NaN in every direct f64 field
/// (both [`ScenarioStats`]-sourced and the
/// [`MonitorSummary`]-sourced `imbalance_ratio`), not just a
/// representative sample — same `serde_json` rejects-NaN
/// motivation. Unlike `ext_metrics`, direct fields can't be
/// dropped (the row schema is fixed), so non-finite collapses to
/// 0.0 with a warn.
#[test]
fn sidecar_to_row_zeros_nan_in_every_direct_f64_field() {
    assert_all_direct_f64_fields_sanitized(f64::NAN);
}

/// Companion to `sidecar_to_row_zeros_nan_in_every_direct_f64_field`
/// pinning the `+Infinity` branch of `finite_or_zero` for every
/// direct f64 field on the row.
#[test]
fn sidecar_to_row_zeros_pos_infinity_in_every_direct_f64_field() {
    assert_all_direct_f64_fields_sanitized(f64::INFINITY);
}

/// Companion to `sidecar_to_row_zeros_nan_in_every_direct_f64_field`
/// pinning the `-Infinity` branch of `finite_or_zero` for every
/// direct f64 field on the row.
#[test]
fn sidecar_to_row_zeros_neg_infinity_in_every_direct_f64_field() {
    assert_all_direct_f64_fields_sanitized(f64::NEG_INFINITY);
}

/// Subnormal f64 values (IEEE 754 denormals) are finite —
/// `is_finite()` returns `true` for them — and must pass through
/// `finite_or_zero` unchanged. Guards against a future refactor
/// that reaches for `is_normal()` instead of `is_finite()`,
/// which would incorrectly collapse subnormals to 0.0 and erase
/// very-small legitimate measurements. `f64::MIN_POSITIVE` is the
/// smallest normal positive; `/ 2.0` lands in the subnormal
/// range.
#[test]
fn sidecar_to_row_preserves_subnormal_f64_in_direct_fields() {
    use crate::assert::ScenarioStats;
    use crate::test_support;
    let subnormal = f64::MIN_POSITIVE / 2.0;
    assert!(subnormal.is_finite(), "subnormal must still be finite");
    assert!(!subnormal.is_normal(), "subnormal must not be normal");
    assert!(subnormal > 0.0, "subnormal is positive");
    let sc = test_support::SidecarResult {
        stats: ScenarioStats {
            worst_spread: subnormal,
            worst_page_locality: -subnormal,
            worst_migration_ratio: subnormal,
            ..Default::default()
        },
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(
        row.spread, subnormal,
        "positive subnormal must pass through finite_or_zero unchanged",
    );
    assert_eq!(
        row.page_locality, -subnormal,
        "negative subnormal must pass through finite_or_zero unchanged",
    );
    assert_eq!(
        row.migration_ratio, subnormal,
        "subnormal on a second direct-f64 field must also pass through",
    );
    // Motivation check: subnormals serialize (unlike NaN / ±Inf,
    // serde_json emits them as standard decimal literals).
    serde_json::to_string(&row).expect("subnormals serialize cleanly");
}

/// Pins that the direct-field NaN sanitization in
/// `sidecar_to_row` does NOT reach into `ext_metrics`. Finite
/// `ext_metrics` entries must survive untouched even when every
/// direct f64 field collapses to 0.0, and the `ext_metrics` map
/// must not grow a sanitization-synthesized entry. Complements
/// [`sidecar_to_row_drops_non_finite_ext_metrics`] (which pins
/// that non-finite `ext_metrics` entries are DROPPED) by pinning
/// the orthogonal claim: direct-field sanitization never writes
/// into `ext_metrics` regardless of the direct values.
#[test]
fn sidecar_to_row_direct_field_nan_does_not_touch_ext_metrics() {
    use crate::assert::ScenarioStats;
    use crate::test_support;
    let mut ext = BTreeMap::new();
    ext.insert("finite_nonzero".to_string(), 2.5);
    ext.insert("finite_zero".to_string(), 0.0);
    ext.insert("finite_negative".to_string(), -7.25);
    let sc = test_support::SidecarResult {
        stats: ScenarioStats {
            // Every remaining direct f64 field non-finite.
            worst_spread: f64::NAN,
            worst_migration_ratio: f64::INFINITY,
            worst_page_locality: f64::INFINITY,
            worst_cross_node_migration_ratio: f64::NEG_INFINITY,
            ext_metrics: ext.clone(),
            ..Default::default()
        },
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);

    // Direct-field collapse still works.
    assert_eq!(row.spread, 0.0);
    assert_eq!(row.migration_ratio, 0.0);
    assert_eq!(row.page_locality, 0.0);

    // ext_metrics survives unchanged — same length, same keys,
    // same values.
    assert_eq!(
        row.ext_metrics.len(),
        ext.len(),
        "direct-field sanitization must not add or drop ext_metrics entries",
    );
    for (k, v) in &ext {
        assert_eq!(
            row.ext_metrics.get(k),
            Some(v),
            "ext_metrics entry {k:?} must pass through unchanged",
        );
    }

    // Motivation check: the full row still serializes.
    serde_json::to_string(&row).expect("sanitized row must serialize cleanly");
}

/// `sidecar_to_row` must drop NaN / +Infinity / -Infinity from
/// `ext_metrics` because `serde_json::to_string` rejects non-finite
/// f64 values — without this guard a single malformed scenario
/// metric would poison every sidecar write on its batch. Finite
/// entries must pass through unchanged. Also checks that the
/// post-filter row serializes cleanly (the motivation for the
/// filter).
#[test]
fn sidecar_to_row_drops_non_finite_ext_metrics() {
    use crate::assert::ScenarioStats;
    use crate::test_support;
    let mut ext = BTreeMap::new();
    ext.insert("good".to_string(), 1.0);
    ext.insert("nan".to_string(), f64::NAN);
    ext.insert("inf".to_string(), f64::INFINITY);
    ext.insert("neg_inf".to_string(), f64::NEG_INFINITY);
    let sc = test_support::SidecarResult {
        stats: ScenarioStats {
            ext_metrics: ext,
            ..Default::default()
        },
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(
        row.ext_metrics.len(),
        1,
        "only the finite entry should survive: {:?}",
        row.ext_metrics
    );
    assert_eq!(row.ext_metrics.get("good"), Some(&1.0));
    assert!(!row.ext_metrics.contains_key("nan"));
    assert!(!row.ext_metrics.contains_key("inf"));
    assert!(!row.ext_metrics.contains_key("neg_inf"));
    // Motivation check: the post-filter row serializes. Without the
    // filter, serde_json::to_string would return Err because NaN /
    // Infinity have no JSON representation.
    serde_json::to_string(&row).expect("filtered row must serialize cleanly");
}

/// `sidecar_to_row` must drop the JSON-walker depth-cap sentinel
/// [`crate::test_support::WALK_TRUNCATION_SENTINEL_NAME`] from
/// `ext_metrics`. The sentinel is diagnostic metadata about the
/// extraction pass (depth cap hit), not a scenario metric, so it
/// must not leak into A/B comparison output where it would be
/// mistaken for a real measurement and skew filter / aggregation
/// logic. Sibling finite entries must survive untouched.
#[test]
fn sidecar_to_row_drops_walk_truncation_sentinel() {
    use crate::assert::ScenarioStats;
    use crate::test_support;
    let mut ext = BTreeMap::new();
    ext.insert("good".to_string(), 1.0);
    ext.insert(
        test_support::WALK_TRUNCATION_SENTINEL_NAME.to_string(),
        72.0,
    );
    let sc = test_support::SidecarResult {
        stats: ScenarioStats {
            ext_metrics: ext,
            ..Default::default()
        },
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(
        row.ext_metrics.len(),
        1,
        "only the real metric should survive: {:?}",
        row.ext_metrics,
    );
    assert_eq!(row.ext_metrics.get("good"), Some(&1.0));
    assert!(
        !row.ext_metrics
            .contains_key(test_support::WALK_TRUNCATION_SENTINEL_NAME),
        "sentinel must not appear in the row's ext_metrics",
    );
}

// -- metric_def tests --

#[test]
fn metric_def_known() {
    let d = metric_def("worst_spread").unwrap();
    assert_eq!(d.name, "worst_spread");
    assert!(d.higher_is_worse());
    assert_eq!(d.display_unit, "%");
}

#[test]
fn metric_def_not_higher_is_worse() {
    let d = metric_def("total_iterations").unwrap();
    assert!(!d.higher_is_worse());
}

#[test]
fn metric_def_unknown() {
    assert!(metric_def("nonexistent").is_none());
}

#[test]
fn infer_higher_is_worse_latency_shaped() {
    // Latency / delay / time-unit names are higher-is-worse.
    // `AssertResult::merge` folds these by `max` (worst case).
    for name in &[
        "p99_wake_latency",
        "wake_latency_us",
        "scheduling_delay",
        "task_run_delay_ns",
        "io_completion_ms",
        "stall_count",
        "stuck_count",
        "schedule_jitter_cv",
        "max_gap_us",
        "spread",
        "page_drop_count",
        "error_rate",
        "fail_count",
        "migration_ratio",
        "imbalance_factor",
    ] {
        assert!(
            infer_higher_is_worse(name),
            "metric `{name}` must infer higher_is_worse=true \
                 (latency/error-shaped); a folded max keeps the \
                 worst case across cgroups"
        );
    }
}

#[test]
fn infer_higher_is_worse_throughput_shaped() {
    // Throughput / rate / iteration-shaped names are
    // higher-is-better. `AssertResult::merge` folds these by
    // `min` (the cgroup that fell behind is the worst case).
    for name in &[
        "read_iops",
        "write_iops",
        "throughput_mbps",
        "bandwidth_kb",
        "total_iterations",
        "iterations_per_worker",
        "ops_per_sec",
        "page_locality",
        "pass_score",
        "goodput",
    ] {
        assert!(
            !infer_higher_is_worse(name),
            "metric `{name}` must infer higher_is_worse=false \
                 (throughput-shaped); a folded min surfaces the \
                 cgroup that fell behind"
        );
    }
}

#[test]
fn infer_higher_is_worse_unknown_falls_back_to_higher_is_worse() {
    // Names that don't match either token list fall back to
    // higher-is-worse. The fold keeps the max — which surfaces
    // an unexpectedly high reading rather than masking it under
    // a min collapse.
    for name in &["unrelated_field", "random_thing", "metric", "x", "a.b.c"] {
        assert!(
            infer_higher_is_worse(name),
            "unknown metric `{name}` must fall back to \
                 higher_is_worse=true (conservative for regression \
                 detection)"
        );
    }
}

#[test]
fn infer_higher_is_worse_compound_names_resolve_to_latency() {
    // Compound names that contain BOTH a latency-shaped token
    // and a throughput-shaped token resolve to higher-is-worse
    // (the latency interpretation wins). Pin the order-of-
    // checks contract so a refactor that swaps the token lists
    // surfaces here.
    assert!(
        infer_higher_is_worse("read_iops_latency_us"),
        "compound name with `latency` and `iops` must resolve \
             to higher-is-worse (latency token checked first)"
    );
}

#[test]
fn metric_def_polarity_inverse_sense() {
    use crate::test_support::Polarity;
    // higher_is_worse=true means growing = regression; the
    // Polarity for "what do we want it to move toward?" is
    // LowerBetter.
    let d = metric_def("worst_spread").unwrap();
    assert!(d.higher_is_worse());
    assert_eq!(d.polarity, Polarity::LowerBetter);
    // higher_is_worse=false means growing = improvement; the
    // Polarity is HigherBetter.
    let d = metric_def("total_iterations").unwrap();
    assert!(!d.higher_is_worse());
    assert_eq!(d.polarity, Polarity::HigherBetter);
}

#[test]
fn metric_def_polarity_covers_all_entries() {
    use crate::test_support::Polarity;
    // Every METRICS entry must map cleanly to HigherBetter or
    // LowerBetter; no entry should produce TargetValue or Unknown
    // from the bool->Polarity adaptor.
    for m in METRICS.iter() {
        assert!(
            matches!(m.polarity, Polarity::HigherBetter | Polarity::LowerBetter),
            "metric {} produced non-binary polarity {:?}",
            m.name,
            m.polarity
        );
    }
}

#[test]
fn metric_def_all_entries_unique() {
    let mut names: Vec<&str> = METRICS.iter().map(|m| m.name).collect();
    let len = names.len();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), len);
}

/// Registry integrity: every `MetricKind::Rate`'s numerator and
/// denominator MUST name a registered `Counter` metric. `derive_rate_metrics`
/// silently skips a Rate whose component key is absent from the map
/// (`derive_rate_metrics_from` `continue`s on a missing key), so a typo'd
/// component name would never derive and never fail a value test — pin the
/// names at the registry level instead. Counter (not Gauge/Peak) because the
/// re-pool needs Σnum/Σdenom (sum-fold), which only the Counter kind gives.
#[test]
fn every_rate_metric_has_registered_counter_components() {
    for m in METRICS.iter() {
        let MetricKind::Rate {
            numerator,
            denominator,
        } = m.kind
        else {
            continue;
        };
        for (role, comp) in [("numerator", numerator), ("denominator", denominator)] {
            let def = metric_def(comp).unwrap_or_else(|| {
                panic!("Rate {} {role} {comp:?} is not a registered metric", m.name)
            });
            assert!(
                matches!(def.kind, MetricKind::Counter),
                "Rate {} {role} {comp:?} must be a Counter for the Σ-fold re-pool",
                m.name,
            );
        }
    }
}

// -- list_metrics tests --

/// Text-mode [`list_metrics`] emits a table that names every
/// registered metric at least once. Uses substring contains
/// rather than column-exact equality so a future comfy-table
/// preset rename (NOTHING → other) that rewraps whitespace
/// does not false-fail — the surface contract is "every metric
/// name appears somewhere in the rendered output", not a
/// column-width pin.
#[test]
fn list_metrics_text_names_every_metric() {
    let out = list_metrics(false).expect("text render must succeed");
    assert!(!out.is_empty(), "text output must be non-empty");
    for m in METRICS {
        assert!(
            out.contains(m.name),
            "list_metrics(false) output missing metric name {}: {out}",
            m.name,
        );
    }
}

/// Text-mode [`list_metrics`] header row names every column. Pins
/// the header contract so a column rename in
/// `list_metrics` lands here instead of silently in downstream CI
/// scripts that grep the output.
#[test]
fn list_metrics_text_header_pins_column_names() {
    let out = list_metrics(false).expect("text render must succeed");
    for header in ["NAME", "POLARITY", "DEFAULT_ABS", "DEFAULT_REL", "UNIT"] {
        assert!(
            out.contains(header),
            "list_metrics(false) output missing column header {header}: {out}",
        );
    }
}

/// JSON-mode [`list_metrics`] parses back to a `Vec<MetricDef>`-
/// shaped structure with one entry per registry member. `MetricDef`
/// itself does not derive `Deserialize` (the `accessor` fn-pointer
/// is unserializable), so we deserialize into a minimal struct
/// that captures the fields the wire contract promises.
#[test]
fn list_metrics_json_round_trips_via_minimal_schema() {
    #[derive(serde::Deserialize)]
    struct MetricEntry {
        name: String,
        default_abs: f64,
        default_rel: f64,
        display_unit: String,
        // polarity is serialized as an enum tag string by serde
        // (Polarity derives Serialize with the default
        // externally-tagged representation). Deserialize into a
        // serde_json::Value to avoid a cross-crate enum
        // dependency in the test-private schema.
        polarity: serde_json::Value,
    }

    let out = list_metrics(true).expect("json render must succeed");
    let parsed: Vec<MetricEntry> = serde_json::from_str(&out).expect("json output must parse");
    assert_eq!(
        parsed.len(),
        METRICS.len(),
        "json entry count must match METRICS.len()",
    );
    for (parsed_m, registry_m) in parsed.iter().zip(METRICS.iter()) {
        assert_eq!(parsed_m.name, registry_m.name);
        assert_eq!(parsed_m.default_abs, registry_m.default_abs);
        assert_eq!(parsed_m.default_rel, registry_m.default_rel);
        assert_eq!(parsed_m.display_unit, registry_m.display_unit);
        assert!(
            !parsed_m.polarity.is_null(),
            "polarity for {} must serialize as a non-null value",
            registry_m.name,
        );
    }
}

/// JSON-mode [`list_metrics`] must NOT expose the `accessor`
/// fn-pointer field. The `#[serde(skip)]` attribute on
/// `MetricDef::accessor` carries that contract; a regression that
/// dropped the attribute would surface here as the emitted JSON
/// gaining an "accessor" key. Pins the wire surface.
#[test]
fn list_metrics_json_omits_accessor_field() {
    let out = list_metrics(true).expect("json render must succeed");
    assert!(
        !out.contains("\"accessor\""),
        "list_metrics(true) must not emit the accessor field — \
             fn-pointers are not serializable and the field carries \
             #[serde(skip)]: {out}",
    );
}

/// The Distribution/WorstLowest MetricKind serde shape is user-facing via
/// `cargo ktstr stats list-metrics --json` (list_metrics(true) serializes
/// the full MetricDef incl. `kind`). Pin the externally-tagged JSON variant
/// + helper-enum strings so a rename of MetricKind / SampleSource /
/// SampleReduction / WorstLowest* trips this test rather than silently
/// changing the CLI output. MetricKind is Serialize-only (output contract;
/// no deserialize symmetry to check).
#[test]
fn distribution_worstlowest_kind_json_shape_pinned() {
    let dist = serde_json::to_string(&MetricKind::Distribution {
        source: SampleSource::WakeLatencyNs,
        reduction: SampleReduction::P99,
    })
    .expect("MetricKind serializes");
    for tok in [
        "\"Distribution\"",
        "\"source\"",
        "\"WakeLatencyNs\"",
        "\"reduction\"",
        "\"P99\"",
    ] {
        assert!(dist.contains(tok), "{tok} missing from {dist}");
    }
    let wl = serde_json::to_string(&MetricKind::WorstLowest {
        numerator: WorstLowestNumerator::Iterations,
        denominator: WorstLowestDenominator::CpuTimeNs,
    })
    .expect("MetricKind serializes");
    for tok in [
        "\"WorstLowest\"",
        "\"numerator\"",
        "\"Iterations\"",
        "\"denominator\"",
        "\"CpuTimeNs\"",
    ] {
        assert!(wl.contains(tok), "{tok} missing from {wl}");
    }
}

/// Iteration order of [`list_metrics`] matches [`METRICS`]
/// declaration order. Registry order is the canonical surface
/// order for sidecar / CI-gate consumers; a renderer that sorted
/// by name or polarity would silently break scripts that key on
/// the first row.
#[test]
fn list_metrics_text_preserves_registry_order() {
    let out = list_metrics(false).expect("text render must succeed");
    let mut last_pos = 0usize;
    for m in METRICS {
        let pos = out
            .find(m.name)
            .unwrap_or_else(|| panic!("metric {} must appear in text output", m.name));
        assert!(
            pos >= last_pos,
            "metric {} appears before a prior metric — text output must \
                 preserve METRICS declaration order",
            m.name,
        );
        last_pos = pos;
    }
}

// -- list_values --

/// Helper that writes N sidecars to `{root}/{run_key}/{run_key}.ktstr.json`.
/// Each sidecar overrides only the fields the test wants to vary;
/// the rest come from `SidecarResult::test_fixture()`. Used by the
/// `list_values_*` tests to build pool fixtures isolated from
/// `runs_root()`.
fn write_listvalues_fixture(
    root: &std::path::Path,
    sidecars: &[crate::test_support::SidecarResult],
) {
    for (i, sc) in sidecars.iter().enumerate() {
        let run_key = format!("__lv_fixture_{i}__");
        let run_dir = root.join(&run_key);
        std::fs::create_dir_all(&run_dir).expect("create run dir");
        let json = serde_json::to_string(sc).expect("serialize fixture sidecar");
        let path = run_dir.join(format!("{run_key}.ktstr.json"));
        std::fs::write(&path, json).expect("write fixture sidecar");
    }
}

/// Empty pool (no run subdirs) must produce a well-formed text
/// shape with the "(no sidecars in pool)" sentinel under each
/// dimension heading. The function does NOT bail — discovery on
/// an empty pool is a valid query that should answer "nothing"
/// rather than fail.
#[test]
fn list_values_empty_pool_text_has_sentinel_per_dim() {
    let alt = tempfile::TempDir::new().expect("tempdir");
    let out = list_values(false, Some(alt.path())).expect("text render must succeed");
    for dim in [
        "kernel:",
        "commit:",
        "kernel_commit:",
        "source:",
        "cpu_budget:",
        "scheduler:",
        "topology:",
        "work_type:",
    ] {
        assert!(
            out.contains(dim),
            "text output must include heading for {dim}: {out}",
        );
    }
    // Each dim should report the empty-pool sentinel exactly eight
    // times — one per dim — so a regression that dropped the
    // sentinel for one dim falls out as a count mismatch.
    let sentinel_count = out.matches("(no sidecars in pool)").count();
    assert_eq!(
        sentinel_count, 8,
        "empty pool must surface the no-sidecars sentinel under every \
             one of the 8 dims (kernel/commit/kernel_commit/source/\
             cpu_budget/scheduler/topology/work_type); got {sentinel_count} \
             occurrences in:\n{out}",
    );
}

/// Empty pool → JSON object with empty arrays for every dim.
/// Pins the JSON shape so a regression that dropped a key (e.g.
/// "scheduler") on the empty-pool branch surfaces here.
#[test]
fn list_values_empty_pool_json_emits_empty_arrays() {
    let alt = tempfile::TempDir::new().expect("tempdir");
    let out = list_values(true, Some(alt.path())).expect("json render must succeed");
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("json output must parse");
    for dim in [
        "kernel",
        "commit",
        "kernel_commit",
        "source",
        "cpu_budget",
        "scheduler",
        "topology",
        "work_type",
    ] {
        let arr = parsed
            .get(dim)
            .unwrap_or_else(|| panic!("missing key {dim}"));
        assert!(arr.is_array(), "key {dim} must serialize as an array");
        assert_eq!(
            arr.as_array().unwrap().len(),
            0,
            "key {dim} must be an empty array on empty pool",
        );
    }
}

/// Populated pool: distinct values per dim are deduplicated and
/// sorted; `kernel_version: None` and `project_commit: None`
/// produce a `null` entry (JSON) and `unknown` line (text).
#[test]
fn list_values_text_dedupes_and_sorts_per_dim() {
    use crate::test_support::SidecarResult;

    let alt = tempfile::TempDir::new().expect("tempdir");
    let sidecars = vec![
        SidecarResult {
            test_name: "t_a".to_string(),
            topology: "1n2l4c1t".to_string(),
            scheduler: "scx_rusty".to_string(),
            work_type: "SpinWait".to_string(),
            kernel_version: Some("6.14.2".to_string()),
            project_commit: Some("abcdef1".to_string()),
            ..SidecarResult::test_fixture()
        },
        SidecarResult {
            test_name: "t_b".to_string(),
            topology: "1n4l2c1t".to_string(),
            scheduler: "eevdf".to_string(),
            work_type: "PageFaultChurn".to_string(),
            kernel_version: None,
            project_commit: None,
            ..SidecarResult::test_fixture()
        },
        // Duplicate of the first sidecar's identity-fields; the
        // BTreeSet must dedupe so each value lands once in the
        // rendered output.
        SidecarResult {
            test_name: "t_c".to_string(),
            topology: "1n2l4c1t".to_string(),
            scheduler: "scx_rusty".to_string(),
            work_type: "SpinWait".to_string(),
            kernel_version: Some("6.14.2".to_string()),
            project_commit: Some("abcdef1".to_string()),
            ..SidecarResult::test_fixture()
        },
    ];
    write_listvalues_fixture(alt.path(), &sidecars);

    let out = list_values(false, Some(alt.path())).expect("text render must succeed");

    // Dedupe: each distinct VALUE appears EXACTLY once per
    // dim (set semantics) even though "scx_rusty" / "1n2l4c1t"
    // / "SpinWait" / "abcdef1" / "6.14.2" come from two of the
    // three fixtures. Each value below is unique to its dim
    // so it should appear once across the rendered text. The
    // `unknown` sentinel is checked separately because both
    // `kernel` and `commit` are optional dims and each emits
    // its own `unknown` line.
    for value in [
        "6.14.2",
        "abcdef1",
        "scx_rusty",
        "eevdf",
        "1n2l4c1t",
        "1n4l2c1t",
        "SpinWait",
        "PageFaultChurn",
    ] {
        let count = out.matches(value).count();
        assert_eq!(
            count, 1,
            "value {value} must appear exactly once in text output (BTreeSet dedup); \
                 got {count} in:\n{out}",
        );
    }
    // `unknown` appears once per Optional dim that has a None
    // entry: kernel, commit, and kernel_commit. The second
    // fixture has `kernel_version: None` and `project_commit:
    // None`; every fixture in this test leaves `kernel_commit`
    // at its `test_fixture` default (None), so the
    // kernel_commit set's None bucket renders one `unknown`
    // line as well. Total: 3 occurrences.
    //
    // `run_source` is the fourth optional dim but does NOT
    // contribute an `unknown` here: `list_values(_, Some(dir))`
    // calls `apply_archive_source_override` on the loaded pool
    // (the `--dir` flag treats the supplied root as an archive),
    // which rewrites every `run_source: None` to
    // `Some("archive")` BEFORE the dimension-set is built. Every
    // fixture above leaves `run_source` at its `test_fixture`
    // default (None), but they all surface as `archive` after
    // the override — the run_source set never holds a None
    // entry on this code path, so no `unknown` line is emitted
    // for it.
    let unknown_count = out.matches("unknown").count();
    assert_eq!(
        unknown_count, 3,
        "`unknown` must render once per optional dim with a None \
             entry (kernel + commit + kernel_commit = 3); got \
             {unknown_count} in:\n{out}",
    );

    // Sort: both schedulers in ascending lex order means
    // "eevdf" appears BEFORE "scx_rusty" in the rendered text.
    let pos_eevdf = out.find("eevdf").expect("eevdf in output");
    let pos_rusty = out.find("scx_rusty").expect("scx_rusty in output");
    assert!(
        pos_eevdf < pos_rusty,
        "values within a dim must render sorted (BTreeSet iter order); \
             expected 'eevdf' before 'scx_rusty' in:\n{out}",
    );
}

/// The populated `cpu_budget` dim renders distinct budgets once
/// each, sorted ascending, in both the text block and as a numeric
/// JSON array (every other dim is a string array; cpu_budget is the
/// sole numeric one). Skip rows (budget 0) are excluded.
#[test]
fn list_values_cpu_budget_renders_distinct_sorted() {
    use crate::test_support::SidecarResult;

    let alt = tempfile::TempDir::new().expect("tempdir");
    let sidecars = vec![
        SidecarResult {
            test_name: "t_a".to_string(),
            cpu_budget: 32,
            vcpus: 32,
            ..SidecarResult::test_fixture()
        },
        SidecarResult {
            test_name: "t_b".to_string(),
            cpu_budget: 4,
            vcpus: 16,
            ..SidecarResult::test_fixture()
        },
        // Duplicate budget 4 — the BTreeSet must dedupe.
        SidecarResult {
            test_name: "t_c".to_string(),
            cpu_budget: 4,
            vcpus: 16,
            ..SidecarResult::test_fixture()
        },
        // Skip row (budget 0) — excluded from the budget set.
        SidecarResult {
            test_name: "t_d".to_string(),
            cpu_budget: 0,
            vcpus: 0,
            ..SidecarResult::test_fixture()
        },
    ];
    write_listvalues_fixture(alt.path(), &sidecars);

    let text = list_values(false, Some(alt.path())).expect("text render must succeed");
    // Distinct budgets appear once each. Scope the search to the
    // cpu_budget block so digits in other values can't false-match:
    // the block is the lines between "cpu_budget:\n" and the next
    // blank line.
    let block_start =
        text.find("cpu_budget:\n").expect("cpu_budget heading") + "cpu_budget:\n".len();
    let block = &text[block_start..];
    let block_end = block.find("\n\n").map(|i| i + 1).unwrap_or(block.len());
    let block = &block[..block_end];
    assert_eq!(
        block.matches("  4\n").count(),
        1,
        "budget 4 once: {block:?}"
    );
    assert_eq!(
        block.matches("  32\n").count(),
        1,
        "budget 32 once: {block:?}"
    );
    let pos_4 = block.find("  4\n").expect("4 present");
    let pos_32 = block.find("  32\n").expect("32 present");
    assert!(pos_4 < pos_32, "budgets must sort ascending: {block:?}");

    let json = list_values(true, Some(alt.path())).expect("json render must succeed");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("json must parse");
    let arr = parsed
        .get("cpu_budget")
        .and_then(|v| v.as_array())
        .expect("cpu_budget must be a JSON array");
    let nums: Vec<u64> = arr
        .iter()
        .map(|v| v.as_u64().expect("numeric budget"))
        .collect();
    assert_eq!(
        nums,
        vec![4, 32],
        "JSON cpu_budget must be a sorted numeric array"
    );
}

/// JSON shape: `kernel` and `commit` arrays carry `null` for
/// absent values, `Value::String` for present values; the other
/// dims are bare `String` arrays.
#[test]
fn list_values_json_carries_null_for_optional_dims() {
    use crate::test_support::SidecarResult;

    let alt = tempfile::TempDir::new().expect("tempdir");
    let sidecars = vec![
        SidecarResult {
            test_name: "t_known".to_string(),
            kernel_version: Some("6.14.2".to_string()),
            project_commit: Some("abcdef1".to_string()),
            ..SidecarResult::test_fixture()
        },
        SidecarResult {
            test_name: "t_unknown".to_string(),
            kernel_version: None,
            project_commit: None,
            ..SidecarResult::test_fixture()
        },
    ];
    write_listvalues_fixture(alt.path(), &sidecars);

    let out = list_values(true, Some(alt.path())).expect("json render must succeed");
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("json output must parse");

    let kernel = parsed
        .get("kernel")
        .expect("kernel key")
        .as_array()
        .unwrap();
    assert!(
        kernel.iter().any(|v| v.is_null()),
        "kernel array must include a literal null for the None entry; got {kernel:?}",
    );
    assert!(
        kernel.iter().any(|v| v.as_str() == Some("6.14.2")),
        "kernel array must include the populated value 6.14.2; got {kernel:?}",
    );

    let commit = parsed
        .get("commit")
        .expect("commit key")
        .as_array()
        .unwrap();
    assert!(
        commit.iter().any(|v| v.is_null()),
        "commit array must include a literal null for the None entry; got {commit:?}",
    );
    assert!(
        commit.iter().any(|v| v.as_str() == Some("abcdef1")),
        "commit array must include the populated value abcdef1; got {commit:?}",
    );
}

/// `dir = None` resolves against `runs_root()`; if `runs_root()`
/// does not exist, the function returns Ok with empty arrays /
/// per-dim sentinel rather than bailing. Pins the no-bail
/// contract on missing-root.
#[test]
fn list_values_none_dir_does_not_bail_on_missing_root() {
    // We cannot reliably wipe `runs_root()` from a unit test, but
    // we can pin the "Some(nonexistent_path)" branch which
    // exercises the same `collect_pool -> empty Vec` codepath
    // (`fs::read_dir` returns Err on a missing root, and
    // `collect_pool` swallows that into an empty pool).
    let alt = tempfile::TempDir::new().expect("tempdir");
    let nonexistent = alt.path().join("definitely_does_not_exist");
    let out = list_values(false, Some(&nonexistent)).expect("must not bail on missing root");
    assert!(
        out.contains("(no sidecars in pool)"),
        "missing root must render the no-sidecars sentinel: {out}",
    );
}

// -- MetricDef::read tests --

fn read_metric(row: &GauntletRow, name: &str) -> Option<f64> {
    metric_def(name).expect("metric name").read(row)
}

#[test]
fn metric_def_read_named_fields() {
    let mut row = make_row("a", "t", true, 42.0);
    row.gap_ms = 100;
    row.migrations = 7;
    row.migration_ratio = 0.3;
    row.imbalance_ratio = 2.0;
    row.max_dsq_depth = 5;
    row.stuck_count = 3;
    row.fallback_count = 11;
    row.keep_last_count = 4;
    row.total_iterations = 1000;
    row.page_locality = 0.8;
    row.cross_node_migration_ratio = 0.1;
    // Distribution roll-ups are ext_metrics-sourced now (accessor |_| None);
    // read_metric resolves them via MetricDef::read's ext fallback.
    for (name, v) in [
        ("worst_p99_wake_latency_us", 99.0),
        ("worst_median_wake_latency_us", 50.0),
        ("worst_wake_latency_cv", 0.5),
        ("worst_mean_run_delay_us", 25.0),
        ("worst_run_delay_us", 200.0),
    ] {
        row.ext_metrics.insert(name.to_string(), v);
    }
    assert_eq!(read_metric(&row, "worst_spread"), Some(42.0));
    assert_eq!(read_metric(&row, "worst_gap_ms"), Some(100.0));
    assert_eq!(read_metric(&row, "total_migrations"), Some(7.0));
    assert_eq!(read_metric(&row, "worst_migration_ratio"), Some(0.3));
    assert_eq!(read_metric(&row, "max_imbalance_ratio"), Some(2.0));
    assert_eq!(read_metric(&row, "max_dsq_depth"), Some(5.0));
    assert_eq!(read_metric(&row, "stuck_count"), Some(3.0));
    assert_eq!(read_metric(&row, "total_fallback"), Some(11.0));
    assert_eq!(read_metric(&row, "total_keep_last"), Some(4.0));
    assert_eq!(read_metric(&row, "worst_p99_wake_latency_us"), Some(99.0));
    assert_eq!(
        read_metric(&row, "worst_median_wake_latency_us"),
        Some(50.0)
    );
    assert_eq!(read_metric(&row, "worst_wake_latency_cv"), Some(0.5));
    assert_eq!(read_metric(&row, "total_iterations"), Some(1000.0));
    assert_eq!(read_metric(&row, "worst_mean_run_delay_us"), Some(25.0));
    assert_eq!(read_metric(&row, "worst_run_delay_us"), Some(200.0));
    assert_eq!(read_metric(&row, "worst_page_locality"), Some(0.8));
    assert_eq!(
        read_metric(&row, "worst_cross_node_migration_ratio"),
        Some(0.1)
    );
}

#[test]
fn metric_def_read_prefers_accessor_over_ext_metrics() {
    // When a name is in METRICS, the built-in accessor wins.
    // Even if ext_metrics carries a colliding entry for the
    // same name, MetricDef::read returns the accessor's value
    // — built-in fields are the authoritative source.
    let mut row = make_row("a", "t", true, 5.0);
    row.ext_metrics.insert("worst_spread".into(), 999.0);
    assert_eq!(read_metric(&row, "worst_spread"), Some(5.0));

    // User ext_metrics with no matching MetricDef are reachable
    // via the direct ext_metrics map; metric_def returns None
    // for unregistered names.
    row.ext_metrics.insert("custom_metric".into(), 77.0);
    assert!(metric_def("custom_metric").is_none());
    assert_eq!(row.ext_metrics.get("custom_metric").copied(), Some(77.0));
}

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
        assert!(d.is_regression, "all reported deltas should be regressions");
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
        assert!(!d.is_regression);
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
        iters_delta.is_regression,
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
    assert!(spread_up.is_regression, "spread increase is a regression");
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
        !spread_down.is_regression,
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
    assert!(spread_override.is_regression);
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
    // sides read None, both collapse to 0.0 via unwrap_or(0.0), and the
    // EPSILON guard classifies the delta as unchanged.
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

/// Explicit None-branch pin on the `compare_rows` ext-fallback contract.
///
/// `compare_rows` calls `m.read(row)` for every metric and falls through
/// `unwrap_or(0.0)` to the EPSILON-guard when the read is `None`. Since
/// `worst_wake_latency_tail_ratio` is now ext-sourced with a `|_| None`
/// accessor, an ABSENT ext key (the producer's sub-threshold output) is the
/// None condition. The sibling `wake_latency_tail_ratio_compares_via_ext_metrics`
/// exercises this embedded in the suppression semantic; this test pins the
/// raw mechanism — a regression that dropped `unwrap_or(0.0)` and panicked
/// on None, or that synthesized a value for an absent key, would fail here.
///
/// Asserts the three observable consequences:
/// 1. `metric.read(&row)` returns `None` on both sides (no ext key).
/// 2. `compare_rows` does NOT panic.
/// 3. The resulting `CompareReport` classifies the pair as `unchanged`.
#[test]
fn compare_rows_handles_none_from_absent_ext_key_as_zero() {
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

    // The call must not panic (a regression that dropped the
    // `unwrap_or` would trip here), and the result must
    // classify the pair as unchanged — both sides collapse to
    // 0.0 via unwrap_or, then the `abs() < EPSILON` guard
    // short-circuits without producing a finding.
    let report = compare_rows_by(
        std::slice::from_ref(&row_a),
        std::slice::from_ref(&row_b),
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    assert_eq!(
        report.regressions, 0,
        "None accessor result must land as unchanged, not a regression",
    );
    assert_eq!(
        report.improvements, 0,
        "None accessor result must land as unchanged, not an improvement",
    );
    assert!(
        report.findings.is_empty(),
        "no findings must be emitted when the accessor returns None; \
             got: {:?}",
        report.findings,
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
    assert!(spread_finding.is_regression, "6% > 5% → regression");

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

/// `compare_rows` uses `Iterator::find` to locate the A-side
/// match for each B-side row, so when `rows_a` contains two
/// entries with the same `(scenario, topology, work_type)` key
/// the first one wins. Lock that contract in: the second
/// duplicate must be ignored even though it would change the
/// verdict.
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
/// "partial hint would mislead" arm in GAP-D — emitting
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

// -- RowFilter / apply_row_filters --

/// Helper that builds a `GauntletRow` with controllable
/// scheduler / topology / work_type / kernel_version for the
/// filter tests. The metric fields default to harmless
/// passing values; tests are interested in identity-field
/// matching, not metrics.
fn make_filter_row(
    scenario: &str,
    scheduler: &str,
    topology: &str,
    work_type: &str,
    kernel_version: Option<&str>,
) -> GauntletRow {
    GauntletRow {
        scenario: scenario.into(),
        topology: topology.into(),
        work_type: work_type.into(),
        scheduler: scheduler.into(),
        kernel_version: kernel_version.map(str::to_owned),
        cpu_budget: None,
        vcpus: None,
        commit: None,
        kernel_commit: None,
        run_source: None,
        passed: true,
        skipped: false,
        inconclusive: false,
        run_sample_count: 0,
        spread: 0.0,
        gap_ms: 0,
        migrations: 0,
        migration_ratio: 0.0,
        imbalance_ratio: 0.0,
        max_dsq_depth: 0,
        stuck_count: 0,
        fallback_count: 0,
        keep_last_count: 0,
        total_iterations: 0,
        page_locality: 0.0,
        cross_node_migration_ratio: 0.0,
        ext_metrics: BTreeMap::new(),
        phases: Vec::new(),
    }
}

/// Default `RowFilter` (every field None/empty) matches every
/// row — it's the identity filter. Pins the no-op contract so a
/// future regression that flipped the default to a "match
/// nothing" semantic lands here.
#[test]
fn row_filter_default_matches_every_row() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.14.2"));
    let filter = RowFilter::default();
    assert!(filter.matches(&row), "empty filter must match every row");
}

/// `--scheduler` is strict equality, NOT substring. A filter of
/// `"scx"` does not match a row with scheduler `"scx_rusty"`.
/// Pins the typed-vs-substring asymmetry: -E stays as the
/// substring knob; typed flags exact-match.
#[test]
fn row_filter_scheduler_strict_equality_rejects_prefix() {
    let row = make_filter_row("t", "scx_rusty", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        schedulers: vec!["scx".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "strict-equality scheduler filter must NOT match a prefix; \
             got match for scheduler=`scx_rusty` against filter=`scx`",
    );
}

/// Exact scheduler match passes; the strict-equality contract's
/// happy path.
#[test]
fn row_filter_scheduler_strict_equality_matches_exact() {
    let row = make_filter_row("t", "scx_rusty", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        schedulers: vec!["scx_rusty".to_string()],
        ..RowFilter::default()
    };
    assert!(filter.matches(&row));
}

/// `--kernel 6.14.2` against a row whose `kernel_version` is
/// `None` must NOT match — the operator opted in to a specific
/// kernel and a None-row would silently dilute the filtered set.
#[test]
fn row_filter_kernel_none_row_never_matches_populated_filter() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        kernels: vec!["6.14.2".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "None-row must not match populated filter; got dilution",
    );
}

/// `--kernel 6.14.2` against a row whose `kernel_version` is
/// `Some("6.14.2")` matches.
#[test]
fn row_filter_kernel_exact_match() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.14.2"));
    let filter = RowFilter {
        kernels: vec!["6.14.2".to_string()],
        ..RowFilter::default()
    };
    assert!(filter.matches(&row));
}

/// `--kernel 6.14.2` against a row whose `kernel_version` is
/// `Some("6.14.3")` rejects.
#[test]
fn row_filter_kernel_mismatch_rejects() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.14.3"));
    let filter = RowFilter {
        kernels: vec!["6.14.2".to_string()],
        ..RowFilter::default()
    };
    assert!(!filter.matches(&row));
}

/// Repeatable `--kernel A --kernel B` is OR-combined: a row
/// matches iff its `kernel_version` equals ANY listed entry.
/// Pins the multi-value semantic.
#[test]
fn row_filter_kernels_or_combined_matches_any_listed() {
    let row_a = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.14.2"));
    let row_b = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.15.0"));
    let row_c = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.16.0"));
    let filter = RowFilter {
        kernels: vec!["6.14.2".to_string(), "6.15.0".to_string()],
        ..RowFilter::default()
    };
    assert!(filter.matches(&row_a), "first listed kernel must match");
    assert!(filter.matches(&row_b), "second listed kernel must match");
    assert!(
        !filter.matches(&row_c),
        "kernel outside the listed set must reject",
    );
}

/// Repeatable `--scheduler A --scheduler B` is OR-combined:
/// a row matches iff its `scheduler` equals ANY listed entry.
/// Pins the multi-value semantic for the
/// post-Vec-promotion `schedulers` field; before promotion
/// `--scheduler` was a single-value `Option<String>` and the
/// OR semantic did not exist.
#[test]
fn row_filter_schedulers_or_combined_matches_any_listed() {
    let row_a = make_filter_row("t", "scx_alpha", "1n2l4c1t", "SpinWait", None);
    let row_b = make_filter_row("t", "scx_beta", "1n2l4c1t", "SpinWait", None);
    let row_c = make_filter_row("t", "scx_gamma", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        schedulers: vec!["scx_alpha".to_string(), "scx_beta".to_string()],
        ..RowFilter::default()
    };
    assert!(filter.matches(&row_a), "first listed scheduler must match",);
    assert!(filter.matches(&row_b), "second listed scheduler must match",);
    assert!(
        !filter.matches(&row_c),
        "scheduler outside the listed set must reject",
    );
}

/// Repeatable `--topology A --topology B` is OR-combined:
/// a row matches iff its `topology` equals ANY listed entry.
/// Mirror of
/// `row_filter_schedulers_or_combined_matches_any_listed`
/// for the topologies field.
#[test]
fn row_filter_topologies_or_combined_matches_any_listed() {
    let row_a = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let row_b = make_filter_row("t", "scx_a", "1n2l4c2t", "SpinWait", None);
    let row_c = make_filter_row("t", "scx_a", "1n4l8c1t", "SpinWait", None);
    let filter = RowFilter {
        topologies: vec!["1n2l4c1t".to_string(), "1n2l4c2t".to_string()],
        ..RowFilter::default()
    };
    assert!(filter.matches(&row_a), "first listed topology must match",);
    assert!(filter.matches(&row_b), "second listed topology must match",);
    assert!(
        !filter.matches(&row_c),
        "topology outside the listed set must reject",
    );
}

/// Repeatable `--work-type A --work-type B` is OR-combined:
/// a row matches iff its `work_type` equals ANY listed
/// entry. Mirror of
/// `row_filter_schedulers_or_combined_matches_any_listed`
/// for the work_types field.
#[test]
fn row_filter_work_types_or_combined_matches_any_listed() {
    let row_a = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let row_b = make_filter_row("t", "scx_a", "1n2l4c1t", "PageFaultChurn", None);
    let row_c = make_filter_row("t", "scx_a", "1n2l4c1t", "MutexContention", None);
    let filter = RowFilter {
        work_types: vec!["SpinWait".to_string(), "PageFaultChurn".to_string()],
        ..RowFilter::default()
    };
    assert!(filter.matches(&row_a), "first listed work_type must match",);
    assert!(filter.matches(&row_b), "second listed work_type must match",);
    assert!(
        !filter.matches(&row_c),
        "work_type outside the listed set must reject",
    );
}

/// `--project-commit abcdef1` against a row whose `commit` is `None`
/// must NOT match — same opt-in policy as `--kernel`. Mirror
/// of `row_filter_kernel_none_row_never_matches_populated_filter`
/// for the project-commit field.
#[test]
fn row_filter_commit_none_row_never_matches_populated_filter() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        project_commits: vec!["abcdef1".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "None-commit row must not match populated filter; \
             got dilution",
    );
}

/// `--project-commit abcdef1` against a row whose `commit` is
/// `Some("abcdef1")` matches; `Some("other")` rejects.
/// Pins the strict-equality contract for commit, including
/// the OR-combined multi-value semantic and the `-dirty`
/// suffix's contribution to identity (a clean and dirty run
/// of the same HEAD bucket separately).
#[test]
fn row_filter_commit_exact_match_and_or_combined() {
    let mut row_clean = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_clean.commit = Some("abcdef1".to_string());
    let mut row_dirty = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_dirty.commit = Some("abcdef1-dirty".to_string());
    let mut row_other = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_other.commit = Some("fedcba2".to_string());

    let filter_single = RowFilter {
        project_commits: vec!["abcdef1".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter_single.matches(&row_clean),
        "exact commit match must succeed",
    );
    assert!(
        !filter_single.matches(&row_dirty),
        "`abcdef1-dirty` must NOT match a filter for `abcdef1` — \
             the suffix is part of identity, so the dirty run buckets \
             separately from the clean run of the same HEAD",
    );
    assert!(
        !filter_single.matches(&row_other),
        "different commit must reject",
    );

    let filter_or = RowFilter {
        project_commits: vec!["abcdef1".to_string(), "fedcba2".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter_or.matches(&row_clean),
        "first listed commit must match in OR-combined filter",
    );
    assert!(
        filter_or.matches(&row_other),
        "second listed commit must match in OR-combined filter",
    );
    assert!(
        !filter_or.matches(&row_dirty),
        "`abcdef1-dirty` must still reject — the suffix-bearing \
             form is its own identity even in OR-combined mode",
    );
}

/// `--kernel-commit kabcde7` against a row whose
/// `kernel_commit` is `None` must NOT match — same opt-in
/// policy as `--project-commit` and `--kernel`. Mirror of
/// `row_filter_commit_none_row_never_matches_populated_filter`
/// for the kernel-commit field.
#[test]
fn row_filter_kernel_commit_none_row_never_matches_populated_filter() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        kernel_commits: vec!["kabcde7".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "None-kernel-commit row must not match populated filter; \
             got dilution",
    );
}

/// `--kernel-commit kabcde7` against a row whose
/// `kernel_commit` is `Some("kabcde7")` matches;
/// `Some("other")` rejects. Pins the strict-equality
/// contract for kernel_commit, including the OR-combined
/// multi-value semantic and the `-dirty` suffix's
/// contribution to identity (a clean and dirty run of the
/// same kernel HEAD bucket separately).
#[test]
fn row_filter_kernel_commit_exact_match_and_or_combined() {
    let mut row_clean = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_clean.kernel_commit = Some("kabcde7".to_string());
    let mut row_dirty = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_dirty.kernel_commit = Some("kabcde7-dirty".to_string());
    let mut row_other = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_other.kernel_commit = Some("fedcba2".to_string());

    let filter_single = RowFilter {
        kernel_commits: vec!["kabcde7".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter_single.matches(&row_clean),
        "exact kernel_commit match must succeed",
    );
    assert!(
        !filter_single.matches(&row_dirty),
        "`kabcde7-dirty` must NOT match a filter for `kabcde7` — \
             the suffix is part of identity, so the dirty run buckets \
             separately from the clean run of the same kernel HEAD",
    );
    assert!(
        !filter_single.matches(&row_other),
        "different kernel_commit must reject",
    );

    let filter_or = RowFilter {
        kernel_commits: vec!["kabcde7".to_string(), "fedcba2".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter_or.matches(&row_clean),
        "first listed kernel_commit must match in OR-combined filter",
    );
    assert!(
        filter_or.matches(&row_other),
        "second listed kernel_commit must match in OR-combined filter",
    );
    assert!(
        !filter_or.matches(&row_dirty),
        "`kabcde7-dirty` must still reject — the suffix-bearing \
             form is its own identity even in OR-combined mode",
    );
}

/// `--kernel-commit` and `--project-commit` filter on DISTINCT row
/// fields. Pins the field non-aliasing: a row whose
/// `kernel_commit` matches but whose `commit` does not (or
/// vice versa) must reject. A regression that cross-wired
/// the `matches()` arms (e.g. `kernel_commits` checked
/// against `row.commit`) would silently dilute filtered
/// sets.
#[test]
fn row_filter_kernel_commit_and_commit_filter_distinct_fields() {
    let mut row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row.commit = Some("project1".to_string());
    row.kernel_commit = Some("kernel1".to_string());

    // Filter on kernel_commit only — commit dimension is unconstrained.
    let kc_only = RowFilter {
        kernel_commits: vec!["kernel1".to_string()],
        ..RowFilter::default()
    };
    assert!(
        kc_only.matches(&row),
        "kernel_commit match with no commit filter must accept",
    );

    let kc_mismatch = RowFilter {
        kernel_commits: vec!["project1".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !kc_mismatch.matches(&row),
        "kernel_commits filter must check `kernel_commit` not `commit` — \
             a regression that cross-wired the fields would accept here",
    );

    // Filter on commit only — kernel_commit dimension is unconstrained.
    let commit_mismatch = RowFilter {
        project_commits: vec!["kernel1".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !commit_mismatch.matches(&row),
        "project_commits filter must check `commit` not `kernel_commit` — \
             a regression that cross-wired the fields would accept here",
    );
}

/// `--run-source local` against a row whose `run_source` is
/// `None` must NOT match — same opt-in policy as `--kernel`,
/// `--project-commit`, and `--kernel-commit`. The operator wrote
/// specific tags and a None-row would silently dilute the
/// filtered set. Mirror of
/// `row_filter_kernel_commit_none_row_never_matches_populated_filter`
/// for the `run_source` field.
#[test]
fn row_filter_run_source_none_row_never_matches_populated_filter() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        run_sources: vec!["local".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "None-run_source row must not match populated filter; \
             got dilution",
    );
}

/// Repeatable `--run-source A --run-source B` is OR-combined: a row
/// matches iff its `run_source` equals ANY listed entry.
/// Mirror of `row_filter_kernels_or_combined_matches_any_listed`
/// for the `run_source` dimension.
#[test]
fn row_filter_run_sources_or_combined_matches_any_listed() {
    let mut row_local = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_local.run_source = Some("local".to_string());
    let mut row_ci = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_ci.run_source = Some("ci".to_string());
    let mut row_archive = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_archive.run_source = Some("archive".to_string());
    let filter = RowFilter {
        run_sources: vec!["local".to_string(), "ci".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter.matches(&row_local),
        "first listed run_source must match",
    );
    assert!(
        filter.matches(&row_ci),
        "second listed run_source must match",
    );
    assert!(
        !filter.matches(&row_archive),
        "run_source outside the listed set must reject",
    );
}

/// `--run-source` and `--kernel-commit` filter on DISTINCT row
/// fields. Pins the field non-aliasing: a row whose
/// `run_source` matches but whose `kernel_commit` does not
/// (or vice versa) must reject. A regression that cross-wired
/// the `matches()` arms (e.g. `run_sources` checked against
/// `row.kernel_commit`) would silently dilute filtered sets.
/// Mirror of
/// `row_filter_kernel_commit_and_commit_filter_distinct_fields`
/// for the `run_source` × `kernel_commit` cross-wire surface.
#[test]
fn row_filter_run_sources_and_kernel_commits_are_distinct_fields() {
    let mut row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row.run_source = Some("local".to_string());
    row.kernel_commit = None;
    let filter = RowFilter {
        run_sources: vec!["local".to_string()],
        kernel_commits: vec!["abc1234".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "AND composition must reject when kernel_commit gate \
             fails (row's kernel_commit is None) even though the \
             run_source gate matches; a regression that cross-wired \
             run_sources against `row.kernel_commit` would accept here",
    );

    // Symmetric arm: run_source mismatches but kernel_commit
    // matches. Whole filter must still reject.
    let mut row2 = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row2.run_source = Some("ci".to_string());
    row2.kernel_commit = Some("abc1234".to_string());
    let filter2 = RowFilter {
        run_sources: vec!["local".to_string()],
        kernel_commits: vec!["abc1234".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter2.matches(&row2),
        "AND composition must reject when run_source gate \
             fails even though kernel_commit gate passes; a \
             regression that cross-wired kernel_commits against \
             `row.run_source` would accept here",
    );
}

/// `--project-commit` and `--kernel` compose with AND semantics: a
/// populated commit filter and a populated kernel filter must
/// BOTH match for the row to survive. Pins the cross-field
/// composition rule for the new commit field, mirroring the
/// existing multi-field test for scheduler+topology+kernel.
#[test]
fn row_filter_commit_and_kernel_compose_and() {
    let mut row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.14.2"));
    row.commit = Some("abcdef1".to_string());
    let filter_both_match = RowFilter {
        kernels: vec!["6.14.2".to_string()],
        project_commits: vec!["abcdef1".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter_both_match.matches(&row),
        "both filters matching must accept the row",
    );
    let filter_kernel_only_match = RowFilter {
        kernels: vec!["6.14.2".to_string()],
        project_commits: vec!["fedcba2".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter_kernel_only_match.matches(&row),
        "AND composition must reject when commit mismatches even \
             though kernel matches",
    );
}

/// `--topology 1n2l4c1t` strict-equal against the row's
/// rendered topology. The filter is the same string the
/// `Topology::Display` impl emits and `cargo ktstr stats list`
/// shows; passing the exact form that appears in the listing
/// is the operator's expected workflow.
#[test]
fn row_filter_topology_strict_equality() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let filter_match = RowFilter {
        topologies: vec!["1n2l4c1t".to_string()],
        ..RowFilter::default()
    };
    assert!(filter_match.matches(&row));
    let filter_miss = RowFilter {
        topologies: vec!["1n2l4c2t".to_string()],
        ..RowFilter::default()
    };
    assert!(!filter_miss.matches(&row));
}

/// Multiple typed filters compose with AND semantics: every
/// populated field must match. A mismatch on any one field
/// rejects the whole match. Pinned via a row that matches 3
/// of 4 filter fields and assertion that it still rejects.
#[test]
fn row_filter_multi_field_and_composes() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.14.2"));
    // 3 of 4 typed fields match (scheduler, topology, kernels);
    // work_type mismatches. Whole filter must reject.
    let filter = RowFilter {
        schedulers: vec!["scx_a".to_string()],
        topologies: vec!["1n2l4c1t".to_string()],
        kernels: vec!["6.14.2".to_string()],
        work_types: vec!["YieldHeavy".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "AND composition must reject when any single field mismatches; \
             got match despite work_type divergence",
    );
}

/// `apply_row_filters` preserves the original row order and
/// drops only non-matching rows. Pinned by feeding a 3-row
/// vec where row 1 of 3 matches; result must be a 1-element
/// vec with the original middle row.
#[test]
fn apply_row_filters_preserves_order_drops_mismatch() {
    let rows = vec![
        make_filter_row("t1", "scx_a", "1n2l4c1t", "SpinWait", None),
        make_filter_row("t2", "scx_b", "1n2l4c1t", "SpinWait", None),
        make_filter_row("t3", "scx_a", "1n2l4c1t", "SpinWait", None),
    ];
    let filter = RowFilter {
        schedulers: vec!["scx_b".to_string()],
        ..RowFilter::default()
    };
    let kept = apply_row_filters(&rows, &filter);
    assert_eq!(kept.len(), 1, "expected 1 surviving row, got {kept:?}");
    assert_eq!(kept[0].scenario, "t2");
}

/// `apply_row_filters` with the default filter is the identity
/// — every row survives in original order.
#[test]
fn apply_row_filters_default_is_identity() {
    let rows = vec![
        make_filter_row("t1", "scx_a", "1n2l4c1t", "SpinWait", None),
        make_filter_row("t2", "scx_b", "1n2l4c2t", "YieldHeavy", Some("6.14.2")),
    ];
    let kept = apply_row_filters(&rows, &RowFilter::default());
    assert_eq!(kept.len(), rows.len());
    for (a, b) in kept.iter().zip(rows.iter()) {
        assert_eq!(a.scenario, b.scenario);
    }
}

// -- group_and_average_by / AveragedGroup --

/// Mutate a row's metric fields away from defaults so
/// aggregation has a non-zero signal to average. Returns the
/// row reference for chaining.
fn paint_metrics(row: &mut GauntletRow, spread: f64, gap_ms: u64, migrations: u64, iters: u64) {
    row.spread = spread;
    row.gap_ms = gap_ms;
    row.migrations = migrations;
    row.migration_ratio = spread / 100.0;
    row.imbalance_ratio = spread / 10.0;
    row.max_dsq_depth = (gap_ms / 10) as u32;
    row.stuck_count = (migrations / 10) as usize;
    row.fallback_count = migrations as i64;
    row.keep_last_count = -(migrations as i64);
    row.total_iterations = iters;
    row.page_locality = 1.0 - spread / 100.0;
    row.cross_node_migration_ratio = spread / 200.0;
    // The wake / run-delay / iteration-efficiency roll-ups are now
    // ext_metrics-sourced (Distribution / WorstLowest); paint them there so
    // the cross-RUN ext fold (group_and_average_by → aggregate_finite)
    // exercises them: the percentile / CV / mean reductions and the
    // WorstLowest selectors MEAN-fold cross-RUN, worst_run_delay_us
    // (Worst) MAX-folds.
    for (name, v) in [
        ("worst_p99_wake_latency_us", spread * 2.0),
        ("worst_median_wake_latency_us", spread),
        ("worst_wake_latency_cv", spread / 50.0),
        ("worst_mean_run_delay_us", gap_ms as f64),
        ("worst_run_delay_us", (gap_ms * 2) as f64),
        ("worst_iterations_per_worker", iters as f64 / 10.0),
        ("worst_iterations_per_cpu_sec", iters as f64 / 5.0),
    ] {
        row.ext_metrics.insert(name.to_string(), v);
    }
}

/// Empty input produces zero aggregated rows. Pins the empty-
/// vec edge case so callers iterating over the result vector
/// don't need to special-case the `--average` path on empty
/// run directories.
#[test]
fn group_and_average_empty_input_yields_empty_output() {
    let out = group_and_average_by(&[], LEGACY_PAIRING_DIMS);
    assert!(out.is_empty());
}

/// Single passing contributor: aggregate is a faithful copy
/// of the input, with `passes_observed = total_observed = 1`.
/// Pins the trivial pass-through path so a regression in the
/// `denom` math (e.g. division by `total_observed` instead of
/// `passes_observed`) lands here.
#[test]
fn group_and_average_single_pass_passes_through_metrics() {
    let mut row = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut row, 12.0, 200, 50, 1000);
    let out = group_and_average_by(std::slice::from_ref(&row), LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert_eq!(ar.passes_observed, 1);
    assert_eq!(ar.total_observed, 1);
    assert!(ar.row.passed);
    assert!(!ar.row.skipped);
    assert_eq!(ar.row.spread, 12.0);
    assert_eq!(ar.row.gap_ms, 200);
    assert_eq!(ar.row.migrations, 50);
    assert_eq!(ar.row.total_iterations, 1000);
    assert_eq!(ar.row.fallback_count, 50);
    assert_eq!(ar.row.keep_last_count, -50);
    // worst_p99_wake_latency_us is now ext_metrics-sourced (Distribution);
    // single-pass pass-through carries it verbatim (spread*2 = 24.0).
    assert_eq!(
        ar.row.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(24.0),
    );
}

/// Three passing contributors with the same key are folded
/// into a single aggregate. Per-MetricKind cross-RUN fold:
/// Counter / Gauge(Last) typed fields take the arithmetic
/// mean (operator-natural cohort comparison); Peak typed
/// fields take the MAX (kind-correct — averaging Peak
/// dilutes the worst-instant signal). f64 means are exact
/// modulo IEEE rounding; u64/i64 means are rounded.
#[test]
fn group_and_average_multi_pass_kind_aware_fold() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut a, 10.0, 100, 30, 900);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut b, 20.0, 200, 60, 1100);
    let mut c = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut c, 30.0, 300, 90, 1000);
    let out = group_and_average_by(&[a, b, c], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert_eq!(ar.passes_observed, 3);
    assert_eq!(ar.total_observed, 3);
    assert!(ar.row.passed);
    assert!(!ar.row.skipped);
    // Gauge(Last) f64 mean: (10 + 20 + 30) / 3 = 20.0.
    assert_eq!(ar.row.spread, 20.0);
    // Peak u64 MAX (NOT mean — kind-correct cross-RUN fold).
    // Was 200 under arithmetic-mean; now 300.
    assert_eq!(ar.row.gap_ms, 300);
    // Counter u64 mean: (30 + 60 + 90) / 3 = 60.
    assert_eq!(ar.row.migrations, 60);
    // Counter u64 mean: (900 + 1100 + 1000) / 3 = 1000.
    assert_eq!(ar.row.total_iterations, 1000);
    // Counter i64 mean for fallback_count: (30 + 60 + 90)/3 = 60.
    assert_eq!(ar.row.fallback_count, 60);
    // Counter i64 mean for keep_last_count: (-30 + -60 + -90)/3 = -60.
    assert_eq!(ar.row.keep_last_count, -60);
    // Distribution worst_p99_wake_latency_us cross-RUN MEAN (unweighted)
    // through the ext fold: spread*2 = 20/40/60; (20 + 40 + 60)/3 = 40.
    assert_eq!(
        ar.row.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(40.0),
    );
    // WorstLowest worst_iterations_per_cpu_sec cross-RUN MEAN through the
    // ext fold: iters/5 = 180/220/200; (180 + 220 + 200)/3 = 200.
    assert_eq!(
        ar.row
            .ext_metrics
            .get("worst_iterations_per_cpu_sec")
            .copied(),
        Some(200.0),
    );
    // worst_run_delay_us is the SOLE Worst Distribution: cross-RUN it
    // folds by MAX (the peak survives), NOT mean — gap_ms*2 = 200/400/600
    // → MAX 600 (a MEAN would give 400). Pins the Worst arm AND its
    // ordering before the general Distribution MEAN arm in aggregate_finite.
    assert_eq!(
        ar.row.ext_metrics.get("worst_run_delay_us").copied(),
        Some(600.0),
    );
    // The remaining Distribution + WorstLowest reductions cross-RUN MEAN
    // (unweighted), locking the full MAX-vs-MEAN split: worst_median =
    // spread 10/20/30 → 20; worst_mean_run_delay_us = gap_ms 100/200/300
    // → 200; worst_iterations_per_worker = iters/10 90/110/100 → 100.
    assert_eq!(
        ar.row
            .ext_metrics
            .get("worst_median_wake_latency_us")
            .copied(),
        Some(20.0),
    );
    assert_eq!(
        ar.row.ext_metrics.get("worst_mean_run_delay_us").copied(),
        Some(200.0),
    );
    assert_eq!(
        ar.row
            .ext_metrics
            .get("worst_iterations_per_worker")
            .copied(),
        Some(100.0),
    );
    // CV mean (spread/50 = 0.2/0.4/0.6 → 0.4) is float-approximate.
    let cv = ar
        .row
        .ext_metrics
        .get("worst_wake_latency_cv")
        .copied()
        .expect("worst_wake_latency_cv present");
    assert!(
        (cv - 0.4).abs() < 1e-9,
        "worst_wake_latency_cv cross-RUN MEAN ~0.4, got {cv}",
    );
}

/// The cross-RUN unweighted mean of a Distribution/WorstLowest metric
/// divides by the count of contributors that EMITTED the key
/// (`finite.len()`), NOT by `passes_observed`: a passing run that omits the
/// key is EXCLUDED from the mean, not folded in as 0.0. Three passing runs,
/// only TWO carry `worst_p99_wake_latency_us` (20, 40) → aggregate is their
/// mean 30.0, NOT the (20+40+0)/3 = 20.0 a passes_observed divisor gives.
#[test]
fn group_and_average_distribution_excludes_key_omitting_run_from_mean() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.ext_metrics
        .insert("worst_p99_wake_latency_us".to_string(), 20.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.ext_metrics
        .insert("worst_p99_wake_latency_us".to_string(), 40.0);
    // Third passing run omits the key entirely.
    let c = make_row("t", "tiny-1llc", true, 0.0);
    let out = group_and_average_by(&[a, b, c], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].passes_observed, 3);
    assert_eq!(
        out[0]
            .row
            .ext_metrics
            .get("worst_p99_wake_latency_us")
            .copied(),
        Some(30.0),
        "unweighted mean over the 2 emitting runs (20+40)/2=30, NOT \
             (20+40+0)/3=20 a passes_observed divisor would give",
    );
}

/// Same cross-RUN exclude-missing MEAN, pinned on the metric #11 relocated:
/// `worst_wake_latency_tail_ratio` (`MetricKind::WakeLatencyTailRatio`), whose
/// deleted typed fold was `sum / passes_observed` (folding sub-threshold runs
/// in as 0.0). Three passing runs, only TWO emit the key (2.0, 8.0) → aggregate
/// is their mean 5.0, NOT (2+8+0)/3 = 3.33 a passes_observed divisor gives. The
/// two emitters carry UNEQUAL `run_sample_count` (1000 vs 1) to prove the fold
/// is UNWEIGHTED — a sample-count-weighted mean would be (2*1000+8*1)/1001 ≈
/// 2.006, far from 5.0.
#[test]
fn group_and_average_tail_ratio_excludes_omitting_run_and_is_unweighted() {
    let key = "worst_wake_latency_tail_ratio";
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.run_sample_count = 1000;
    a.ext_metrics.insert(key.to_string(), 2.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.run_sample_count = 1;
    b.ext_metrics.insert(key.to_string(), 8.0);
    // Third passing run omits the key entirely.
    let c = make_row("t", "tiny-1llc", true, 0.0);
    let out = group_and_average_by(&[a, b, c], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].passes_observed, 3);
    assert_eq!(
        out[0].row.ext_metrics.get(key).copied(),
        Some(5.0),
        "unweighted mean over the 2 emitting runs (2+8)/2=5.0 — NOT \
             (2+8+0)/3=3.33 (passes_observed divisor), and NOT a sample-count \
             weighted mean (the 1000-vs-1 weights would pull it toward 2.0)",
    );
}

/// `group_and_average_by` propagates `run_sample_count` to the
/// aggregated row's `run_sample_count` as the SUM of
/// contributor weights so a downstream consumer that further
/// folds the aggregated rows can apply the same weighted
/// semantic to the next-level cohort.
#[test]
fn group_and_average_run_sample_count_sums_across_contributors() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.run_sample_count = 5;
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.run_sample_count = 15;
    let mut c = make_row("t", "tiny-1llc", true, 0.0);
    c.run_sample_count = 30;
    let out = group_and_average_by(&[a, b, c], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].row.run_sample_count, 50);
}

/// Cross-RUN ext_metrics fold dispatches by registered
/// MetricKind — a registered `Gauge(Avg)` metric with two
/// contributors carrying different `run_sample_count` weights
/// uses the weighted mean rather than arithmetic mean.
/// (10 * 5 + 30 * 15) / (5 + 15) = 25.0 vs unweighted 20.0.
/// Uses `avg_dsq_depth` (registered as `Gauge(Avg)` per the
/// METRICS table) so the dispatch path is exercised against a
/// real registry entry, not a synthetic fixture.
#[test]
fn group_and_average_ext_metrics_gauge_avg_weighted_by_run_sample_count() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.run_sample_count = 5;
    a.ext_metrics.insert("avg_dsq_depth".to_string(), 10.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.run_sample_count = 15;
    b.ext_metrics.insert("avg_dsq_depth".to_string(), 30.0);
    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let mean = out[0]
        .row
        .ext_metrics
        .get("avg_dsq_depth")
        .copied()
        .expect("ext_metrics propagates avg_dsq_depth aggregate");
    // Weighted mean: (10*5 + 30*15) / 20 = 25.0.
    // Unweighted would be (10 + 30) / 2 = 20.0.
    assert!(
        (mean - 25.0).abs() < f64::EPSILON,
        "expected weighted mean 25.0, got {mean}",
    );
}

/// Cross-RUN re-pool of the pooled `iterations_per_cpu_sec` Rate:
/// `group_and_average_by` SKIPS folding the Rate itself and re-derives it
/// from the folded Counter components (`total_iterations_pooled`,
/// `total_cpu_time_sec`). Registered Counters fold cross-RUN as a SUM
/// (`aggregate_finite` Counter arm = `finite.iter().sum()`, weight
/// ignored), so the components sum to Σnum / Σdenom = 1010 / 10.0 = 101.0 —
/// the true pooled rate, count-invariant, NOT the mean-of-per-run-ratios
/// (~500.6). The two components are co-inserted both-or-neither, so they
/// always share a contributor set; the SUM fold makes the rate identical
/// regardless of contributor count. The folded-COMPONENT assertions below
/// discriminate the SUM fold from a (wrong) hypothetical mean fold — the
/// rate value alone cannot, because Σ/Σ equals mean/mean when the
/// contributor count is equal (the N cancels); the component assertions
/// below discriminate. A stale per-run rate value is discarded by the
/// skip-then-derive path.
#[test]
fn group_and_average_repools_iterations_per_cpu_sec_from_components() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.ext_metrics
        .insert("total_iterations_pooled".to_string(), 1000.0);
    a.ext_metrics.insert("total_cpu_time_sec".to_string(), 1.0);
    // A stale per-run rate must be DISCARDED (a Rate is derived, never
    // folded from its own samples).
    a.ext_metrics
        .insert("iterations_per_cpu_sec".to_string(), 999.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.ext_metrics
        .insert("total_iterations_pooled".to_string(), 10.0);
    b.ext_metrics.insert("total_cpu_time_sec".to_string(), 9.0);
    b.ext_metrics
        .insert("iterations_per_cpu_sec".to_string(), 999.0);
    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    // Components fold as SUM (1000+10, 1.0+9.0), NOT mean (505, 5.0) — this
    // is what discriminates the fold mechanism the derived rate cannot.
    assert_eq!(
        out[0]
            .row
            .ext_metrics
            .get("total_iterations_pooled")
            .copied(),
        Some(1010.0),
        "numerator component folds cross-run as SUM (1010), not mean (505)",
    );
    assert_eq!(
        out[0].row.ext_metrics.get("total_cpu_time_sec").copied(),
        Some(10.0),
        "denominator component folds cross-run as SUM (10.0), not mean (5.0)",
    );
    let rate = out[0]
        .row
        .ext_metrics
        .get("iterations_per_cpu_sec")
        .copied()
        .expect("re-derived pooled rate present");
    // Σnum / Σdenom = 1010 / 10.0 = 101.0.
    assert!(
        (rate - 101.0).abs() < 1e-9,
        "cross-run pooled rate must re-derive to Σ/Σ = 101.0, not the \
             mean-of-ratios (~500.6) or the stale 999.0; got {rate}",
    );
}

/// Cross-RUN count-invariance vs a key-ABSENT run: two key-bearing passing
/// runs PLUS a third passing run with NO pooled component keys (all its
/// cgroups unmeasured, so populate_run_pooled_iterations_per_cpu_sec
/// inserted neither). The components SUM over the runs that carry them
/// (aggregate_finite Counter arm folds the present (value, weight) pairs),
/// so the key-absent run contributes NOTHING — the folded components and
/// the derived rate are identical to the two-run cohort. Asserting the
/// components (1010, 10.0 — NOT a mean-over-all-three 336.7, 3.33) guards
/// against a future regression that diluted the fold by treating a
/// key-absent run as a contributor (which a mean-over-all-runs fold would).
#[test]
fn group_and_average_pooled_rate_unaffected_by_key_absent_run() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.ext_metrics
        .insert("total_iterations_pooled".to_string(), 1000.0);
    a.ext_metrics.insert("total_cpu_time_sec".to_string(), 1.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.ext_metrics
        .insert("total_iterations_pooled".to_string(), 10.0);
    b.ext_metrics.insert("total_cpu_time_sec".to_string(), 9.0);
    // Third PASSING run with NO pooled component keys (all cgroups
    // unmeasured — populate_run_pooled inserted neither key).
    let c = make_row("t", "tiny-1llc", true, 0.0);
    let out = group_and_average_by(&[a, b, c], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    // Components SUM over the two key-bearing runs only; the key-absent
    // run contributes nothing — NOT diluted to (1010)/3 or (10.0)/3.
    assert_eq!(
        out[0]
            .row
            .ext_metrics
            .get("total_iterations_pooled")
            .copied(),
        Some(1010.0),
        "key-absent run must not dilute the summed numerator",
    );
    assert_eq!(
        out[0].row.ext_metrics.get("total_cpu_time_sec").copied(),
        Some(10.0),
        "key-absent run must not dilute the summed denominator",
    );
    // Rate identical to the two-run cohort: Σ/Σ = 1010/10.0 = 101.0.
    assert_eq!(
        out[0]
            .row
            .ext_metrics
            .get("iterations_per_cpu_sec")
            .copied(),
        Some(101.0),
        "key-absent run must not change the pooled rate (count-invariant)",
    );
}

/// Unregistered ext_metric keys fall back to arithmetic mean
/// (same legacy semantic the (sum, count) accumulator
/// produced). Pins that the weighted dispatch only fires for
/// METRICS-known keys; unknown keys ignore the weights.
#[test]
fn group_and_average_ext_metrics_unregistered_falls_back_to_arithmetic_mean() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.run_sample_count = 5;
    a.ext_metrics
        .insert("custom.unregistered".to_string(), 10.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.run_sample_count = 15;
    b.ext_metrics
        .insert("custom.unregistered".to_string(), 30.0);
    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    let mean = out[0]
        .row
        .ext_metrics
        .get("custom.unregistered")
        .copied()
        .expect("ext_metrics propagates custom key");
    // Arithmetic mean (legacy semantic): (10 + 30) / 2 = 20.0.
    assert!(
        (mean - 20.0).abs() < f64::EPSILON,
        "expected arithmetic mean 20.0, got {mean}",
    );
}

/// Different (scenario, topology, work_type) groups produce
/// distinct aggregates — the tuple is the join key. Pins the
/// group-key contract so a regression that dropped a key
/// component would land here as a collision.
#[test]
fn group_and_average_distinct_groups_stay_separate() {
    let mut a = make_row("alpha", "tiny-1llc", true, 0.0);
    paint_metrics(&mut a, 10.0, 100, 30, 1000);
    let mut b = make_row("beta", "tiny-1llc", true, 0.0);
    paint_metrics(&mut b, 50.0, 500, 100, 2000);
    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 2);
    // First-seen iteration order preserved (alpha before beta).
    assert_eq!(out[0].row.scenario, "alpha");
    assert_eq!(out[1].row.scenario, "beta");
}

/// Failing contributors are excluded from the metric mean and
/// flip the aggregate's `passed` to false. The aggregate's
/// `total_observed` still counts every contributor;
/// `passes_observed` counts only the clean ones.
#[test]
fn group_and_average_failed_contributors_excluded_from_mean_and_flag_aggregate() {
    let mut pass1 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass1, 10.0, 100, 30, 1000);
    let mut fail = make_row("t", "tiny-1llc", false, 0.0);
    // The failing row's metrics are pathologically large —
    // if they leaked into the mean, the aggregate's `spread`
    // would explode upward.
    paint_metrics(&mut fail, 10000.0, 99999, 99999, 99999);
    let mut pass2 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass2, 30.0, 300, 90, 1000);
    let out = group_and_average_by(&[pass1, fail, pass2], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert_eq!(ar.passes_observed, 2);
    assert_eq!(ar.total_observed, 3);
    // ALL-must-pass: a single failure flips the aggregate.
    assert!(
        !ar.row.passed,
        "any failing contributor must flip the aggregate to passed=false",
    );
    // Mean of only the passing entries' spread (Gauge(Last)):
    // (10 + 30) / 2 = 20.0. If the failing row leaked in,
    // this would be ~3346.
    assert_eq!(ar.row.spread, 20.0);
    // MAX of only the passing entries' gap_ms (Peak): max(100, 300) = 300.
    // If the failing row leaked into the max, it'd be 99999.
    assert_eq!(ar.row.gap_ms, 300);
}

/// Skipped contributors are excluded from the metric mean
/// and flip the aggregate's `skipped` to true (any-skipped
/// OR rule). `passes_observed` does not count them; the
/// passing-only entries still feed the mean cleanly.
#[test]
fn group_and_average_skipped_contributors_excluded_from_mean_and_flag_aggregate() {
    let mut pass1 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass1, 10.0, 100, 30, 1000);
    let mut skip = make_row("t", "tiny-1llc", true, 0.0);
    skip.skipped = true;
    // Pathological metrics on the skipped row to prove the
    // exclusion is real.
    paint_metrics(&mut skip, 9999.0, 99999, 99999, 99999);
    let mut pass2 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass2, 50.0, 500, 70, 2000);
    let out = group_and_average_by(&[pass1, skip, pass2], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert_eq!(ar.passes_observed, 2);
    assert_eq!(ar.total_observed, 3);
    assert!(
        ar.row.skipped,
        "any skipped contributor must flip the aggregate to skipped=true",
    );
    assert!(
        !ar.row.passed,
        "skipped aggregate must collapse `passed` to false so compare_rows \
             routes the pair through the excluded_pairs gate",
    );
    // Gauge(Last) mean of (pass1, pass2): (10 + 50)/2 = 30.0.
    assert_eq!(ar.row.spread, 30.0);
    // Peak MAX of (pass1, pass2) gap_ms: max(100, 500) = 500.
    // Was 300 under arithmetic-mean.
    assert_eq!(ar.row.gap_ms, 500);
}

/// Inconclusive contributors are excluded from the metric
/// mean and flip the aggregate's `inconclusive` to true (per
/// the `Fail > Inconclusive > Pass > Skip` lattice, an
/// Inconclusive contributor in an otherwise-passing cohort
/// dominates the verdict). `passes_observed` does not count
/// them; pathological metrics on the inconclusive row stay
/// out of the cohort means. Pins that the inconclusive bit
/// surfaces on the aggregate so downstream stats tooling can
/// distinguish a cohort that ran-but-couldn't-evaluate from
/// one that truly passed.
#[test]
fn group_and_average_inconclusive_contributors_excluded_from_mean_and_flag_aggregate() {
    let mut pass1 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass1, 10.0, 100, 30, 1000);
    // Inconclusive row: passed=false, skipped=false,
    // inconclusive=true. Pathological metrics on this row
    // must NOT leak into the mean.
    let mut inc = make_row("t", "tiny-1llc", false, 0.0);
    inc.inconclusive = true;
    paint_metrics(&mut inc, 7777.0, 77777, 77777, 77777);
    let mut pass2 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass2, 30.0, 300, 90, 2000);
    let out = group_and_average_by(&[pass1, inc, pass2], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert_eq!(ar.passes_observed, 2);
    assert_eq!(ar.total_observed, 3);
    assert!(
        ar.row.inconclusive,
        "any inconclusive contributor must flip the aggregate to inconclusive=true",
    );
    assert!(
        !ar.row.passed,
        "an inconclusive contributor must flip the aggregate to passed=false \
             (Inconclusive dominates Pass per the lattice)",
    );
    // Mean of only the passing entries' spread (Gauge(Last)):
    // (10 + 30) / 2 = 20.0. If the inconclusive row leaked
    // in, this would be ~2605.
    assert_eq!(ar.row.spread, 20.0);
    // MAX of only the passing entries' gap_ms (Peak):
    // max(100, 300) = 300. Was 77777 under leaked semantics.
    assert_eq!(ar.row.gap_ms, 300);
}

/// Fail dominates Inconclusive: a cohort with both a Fail and
/// an Inconclusive contributor produces `passed=false,
/// inconclusive=false` (Fail wins per the
/// `Fail > Inconclusive > Pass > Skip` lattice). Pins the
/// `inconclusive: acc.any_inconclusive && !acc.any_failed`
/// guard so the aggregate verdict surfaces the dominant Fail
/// signal rather than the lesser Inconclusive one.
#[test]
fn group_and_average_fail_dominates_inconclusive_in_aggregate_verdict() {
    let mut pass = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass, 10.0, 100, 30, 1000);
    let mut inc = make_row("t", "tiny-1llc", false, 0.0);
    inc.inconclusive = true;
    paint_metrics(&mut inc, 7777.0, 77777, 77777, 77777);
    let mut fail = make_row("t", "tiny-1llc", false, 0.0);
    paint_metrics(&mut fail, 9999.0, 99999, 99999, 99999);
    let out = group_and_average_by(&[pass, inc, fail], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert!(!ar.row.passed, "any Fail must flip passed to false");
    assert!(
        !ar.row.inconclusive,
        "Fail dominates Inconclusive: aggregate must surface as Fail \
             (inconclusive=false), not Inconclusive",
    );
}

/// All contributors fail: aggregate has `passes_observed = 0`,
/// `passed = false`, and zero metric values (no contributor
/// fed the running sums). Pins the divide-by-zero guard:
/// `denom` must default to 1.0 when `passes_observed = 0`.
#[test]
fn group_and_average_all_failed_collapses_to_default_zero_metrics_and_failed_flag() {
    let mut fail1 = make_row("t", "tiny-1llc", false, 0.0);
    paint_metrics(&mut fail1, 99.0, 999, 99, 999);
    let mut fail2 = make_row("t", "tiny-1llc", false, 0.0);
    paint_metrics(&mut fail2, 88.0, 888, 88, 888);
    let out = group_and_average_by(&[fail1, fail2], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert_eq!(ar.passes_observed, 0);
    assert_eq!(ar.total_observed, 2);
    assert!(!ar.row.passed);
    // Failed-only group: every metric collapses to its zero
    // default. The aggregate's `passed=false` then routes the
    // pair through compare_rows' excluded_pairs gate.
    assert_eq!(ar.row.spread, 0.0);
    assert_eq!(ar.row.gap_ms, 0);
    assert_eq!(ar.row.migrations, 0);
}

/// `ext_metrics` keys are unioned across passing
/// contributors; each key averages over the contributors
/// that carried it. A key absent on some passing rows is
/// NOT treated as a stored zero — its denominator is the
/// present-only count.
#[test]
fn group_and_average_ext_metrics_average_per_key_present_count() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.ext_metrics.insert("shared".into(), 10.0);
    a.ext_metrics.insert("a_only".into(), 100.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.ext_metrics.insert("shared".into(), 30.0);
    b.ext_metrics.insert("b_only".into(), 200.0);
    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    // shared: (10 + 30) / 2 = 20.
    assert_eq!(ar.row.ext_metrics.get("shared"), Some(&20.0));
    // a_only: present only in a → mean over 1 entry = 100.
    assert_eq!(ar.row.ext_metrics.get("a_only"), Some(&100.0));
    // b_only: present only in b → mean over 1 entry = 200.
    assert_eq!(ar.row.ext_metrics.get("b_only"), Some(&200.0));
}

/// `group_and_average_by` preserves first-seen iteration order so
/// downstream tests against the result remain deterministic
/// even though the internal map uses BTreeMap (key-sorted)
/// for storage. Pinned by feeding keys in z→a order and
/// asserting the output keeps that order.
#[test]
fn group_and_average_preserves_first_seen_order() {
    let zebra = make_row("zebra", "tiny-1llc", true, 0.0);
    let alpha = make_row("alpha", "tiny-1llc", true, 0.0);
    let mango = make_row("mango", "tiny-1llc", true, 0.0);
    let out = group_and_average_by(&[zebra, alpha, mango], LEGACY_PAIRING_DIMS);
    let names: Vec<&str> = out.iter().map(|r| r.row.scenario.as_str()).collect();
    assert_eq!(
        names,
        vec!["zebra", "alpha", "mango"],
        "output must follow first-seen iteration order, not key sort",
    );
}

/// Cohort with mixed clean/dirty `commit` values (same hex)
/// renders with `+mixed` appended to the canonical
/// un-suffixed hex. First contributor is dirty; the second
/// is clean. Pinning the rendered form catches a regression
/// where averaging silently kept first-seen behaviour and
/// hid the WIP-vs-committed disagreement.
#[test]
fn group_and_average_mixed_dirty_project_commit_renders_plus_mixed() {
    let mut dirty = make_row("t", "tiny-1llc", true, 0.0);
    dirty.commit = Some("abc1234-dirty".to_string());
    let mut clean = make_row("t", "tiny-1llc", true, 0.0);
    clean.commit = Some("abc1234".to_string());

    let out = group_and_average_by(&[dirty, clean], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].row.commit.as_deref(),
        Some("abc1234+mixed"),
        "mixed clean+dirty must render as `{{hex}}+mixed`, not first-seen",
    );
}

/// Same shape on `kernel_commit`. Pins the second commit
/// dimension separately because the production code uses
/// two parallel accumulator-state pairs and a regression
/// could miss one.
#[test]
fn group_and_average_mixed_dirty_kernel_commit_renders_plus_mixed() {
    let mut clean = make_row("t", "tiny-1llc", true, 0.0);
    clean.kernel_commit = Some("def5678".to_string());
    let mut dirty = make_row("t", "tiny-1llc", true, 0.0);
    dirty.kernel_commit = Some("def5678-dirty".to_string());

    let out = group_and_average_by(&[clean, dirty], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].row.kernel_commit.as_deref(),
        Some("def5678+mixed"),
        "mixed clean+dirty kernel_commit must render as `{{hex}}+mixed`",
    );
}

/// Homogeneous-dirty cohort (every contributor has `-dirty`)
/// must NOT receive the `+mixed` marker — the cohort agrees
/// on the working-tree state. Pinning this guards against a
/// regression where the marker fires on every dirty value
/// regardless of clean siblings.
#[test]
fn group_and_average_all_dirty_keeps_dirty_suffix_no_mixed() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.commit = Some("abc1234-dirty".to_string());
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.commit = Some("abc1234-dirty".to_string());

    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(
        out[0].row.commit.as_deref(),
        Some("abc1234-dirty"),
        "homogeneous-dirty cohort must keep first-seen `-dirty`, no `+mixed`",
    );
}

/// Homogeneous-clean cohort (every contributor lacks
/// `-dirty`) keeps the un-suffixed first-seen value, no
/// marker.
#[test]
fn group_and_average_all_clean_keeps_value_no_mixed() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.commit = Some("abc1234".to_string());
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.commit = Some("abc1234".to_string());

    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(
        out[0].row.commit.as_deref(),
        Some("abc1234"),
        "homogeneous-clean cohort must keep first-seen value, no `+mixed`",
    );
}

/// Skipped contributors participate in mixed-dirty tracking.
/// The cohort's WIP state is metadata independent of metric
/// outcome — a skipped sidecar from a dirty tree still
/// counts toward the dirty-flag because it records the
/// producer's working-tree state at run time. Pin: one
/// passing-clean + one skipped-dirty contributor renders
/// `+mixed`.
///
/// Tests the SKIPPED arm only (`passed=true, skipped=true`).
/// The failed arm (`passed=false, skipped=false`) is pinned
/// separately by
/// `group_and_average_mixed_dirty_tracking_includes_failed_contributors`
/// — the two arms exit through distinct `continue` statements
/// in `group_and_average_by` and a regression in either is
/// independent of the other.
#[test]
fn group_and_average_mixed_dirty_tracking_includes_skipped() {
    let mut clean_pass = make_row("t", "tiny-1llc", true, 0.0);
    clean_pass.commit = Some("abc1234".to_string());
    let mut dirty_skip = make_row("t", "tiny-1llc", true, 0.0);
    dirty_skip.skipped = true;
    dirty_skip.commit = Some("abc1234-dirty".to_string());

    let out = group_and_average_by(&[clean_pass, dirty_skip], LEGACY_PAIRING_DIMS);
    assert_eq!(
        out[0].row.commit.as_deref(),
        Some("abc1234+mixed"),
        "skipped contributors still flip the dirty flag — \
             cohort metadata is independent of metric outcome",
    );
}

/// Failed contributor pin: a passing-clean row paired with a
/// FAILING-dirty row (`passed=false`, `skipped=false`) must
/// still flip the cohort's mixed-dirty flag and render
/// `+mixed` on the aggregate's commit field. The
/// `update_dirty_tracking` call site executes BEFORE the
/// `if !row.passed { continue; }` short-circuit, which is
/// the load-bearing ordering: dirty-status is per-row
/// metadata about the producer's working tree, NOT a metric
/// outcome, so failed contributors must carry their dirty
/// flag forward even though their metrics are excluded from
/// the mean. A regression that moved `update_dirty_tracking`
/// below the failed-skip continue would silently drop the
/// failed row's dirty status and the cohort would render the
/// clean form — hiding WIP-vs-committed disagreement that
/// the operator needs to see.
///
/// Distinct from
/// `group_and_average_mixed_dirty_tracking_includes_skipped`
/// which exercises the SKIPPED arm only (`passed=true,
/// skipped=true`). The two arms have separate `continue`
/// statements and one could regress without the other; this
/// test pins the FAILED arm specifically.
#[test]
fn group_and_average_mixed_dirty_tracking_includes_failed_contributors() {
    let mut clean_pass = make_row("t", "tiny-1llc", true, 0.0);
    clean_pass.commit = Some("abc1234".to_string());
    let mut dirty_fail = make_row("t", "tiny-1llc", false, 0.0);
    dirty_fail.commit = Some("abc1234-dirty".to_string());

    let out = group_and_average_by(&[clean_pass, dirty_fail], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1, "single cohort key must produce one aggregate");
    assert_eq!(
        out[0].row.commit.as_deref(),
        Some("abc1234+mixed"),
        "failed contributor's `-dirty` flag must still flip the \
             cohort's dirty-tracking — cohort metadata is independent \
             of metric outcome. A regression moving update_dirty_tracking \
             below the `if !row.passed` continue would drop the failed \
             row's dirty status and render `abc1234` instead",
    );
    // Symmetric arm: passing-dirty + failing-clean. The
    // dirty-tracking flip on the failing contributor's clean
    // form must register as well — `any_clean` is the
    // counterpart flag, and the same code path executes for
    // both `Some(hex)` and `Some(hex-dirty)` values.
    let mut dirty_pass = make_row("t", "tiny-1llc", true, 0.0);
    dirty_pass.commit = Some("def5678-dirty".to_string());
    let mut clean_fail = make_row("t", "tiny-1llc", false, 0.0);
    clean_fail.commit = Some("def5678".to_string());

    let out = group_and_average_by(&[dirty_pass, clean_fail], LEGACY_PAIRING_DIMS);
    assert_eq!(
        out[0].row.commit.as_deref(),
        Some("def5678+mixed"),
        "failed contributor's CLEAN form must also flip the \
             cohort's any_clean flag — symmetric to the dirty arm",
    );
    // Failed contributor's `passed=false` still flips the
    // aggregate's `passed` flag (logical-AND across all
    // contributors). This sanity-checks that the new test
    // doesn't accidentally exercise an aggregate-passes path
    // — failed rows are correctly being excluded from the
    // metric mean while contributing to dirty tracking.
    assert!(
        !out[0].row.passed,
        "any failing contributor must flip the aggregate to \
             passed=false, regardless of dirty-tracking semantics",
    );
}

/// Mixed-dirty marker uses canonical un-suffixed hex even
/// when `acc.first` is the dirty form. Pin: first contributor
/// is `abc1234-dirty`, second is `abc1234`; rendered form is
/// `abc1234+mixed`, NOT `abc1234-dirty+mixed`. Guards against
/// a stripping bug in `render_mixed_dirty`.
#[test]
fn group_and_average_mixed_dirty_strips_dirty_from_first_seen() {
    let mut dirty_first = make_row("t", "tiny-1llc", true, 0.0);
    dirty_first.commit = Some("abc1234-dirty".to_string());
    let mut clean_second = make_row("t", "tiny-1llc", true, 0.0);
    clean_second.commit = Some("abc1234".to_string());

    let out = group_and_average_by(&[dirty_first, clean_second], LEGACY_PAIRING_DIMS);
    let rendered = out[0].row.commit.as_deref().expect("commit must render");
    assert_eq!(rendered, "abc1234+mixed");
    assert!(
        !rendered.contains("-dirty"),
        "rendered form must drop `-dirty` even when first contributor was dirty; got: {rendered}",
    );
}

/// `None`-only cohort keeps `None`. Sanity check that the
/// dirty-tracking does not synthesize a marker when no
/// contributor has a commit value.
#[test]
fn group_and_average_all_none_commits_keeps_none_no_mixed() {
    let a = make_row("t", "tiny-1llc", true, 0.0);
    let b = make_row("t", "tiny-1llc", true, 0.0);

    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert!(
        out[0].row.commit.is_none(),
        "None-only cohort must keep None — no synthesized `+mixed`",
    );
}

/// End-to-end: aggregated rows feed `compare_rows` cleanly.
/// Side A has [10, 12, 14] (mean 12); side B has [28, 30, 32]
/// (mean 30). The 18-unit delta on `worst_spread`
/// (default_abs=5.0, default_rel=0.25) clears both gates,
/// producing a regression. Pins the full averaging pipeline.
#[test]
fn group_and_average_then_compare_rows_yields_regression_on_means() {
    let mut a1 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut a1, 10.0, 100, 30, 1000);
    let mut a2 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut a2, 12.0, 120, 35, 1000);
    let mut a3 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut a3, 14.0, 140, 40, 1000);
    let mut b1 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut b1, 28.0, 280, 70, 1000);
    let mut b2 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut b2, 30.0, 300, 75, 1000);
    let mut b3 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut b3, 32.0, 320, 80, 1000);

    let agg_a = group_and_average_by(&[a1, a2, a3], LEGACY_PAIRING_DIMS);
    let agg_b = group_and_average_by(&[b1, b2, b3], LEGACY_PAIRING_DIMS);
    let rows_a: Vec<GauntletRow> = agg_a.iter().map(|r| r.row.clone()).collect();
    let rows_b: Vec<GauntletRow> = agg_b.iter().map(|r| r.row.clone()).collect();
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    let spread = res
        .findings
        .iter()
        .find(|f| f.metric.name == "worst_spread")
        .expect("worst_spread must regress on aggregated means");
    assert!(spread.is_regression);
    assert_eq!(spread.val_a, 12.0, "mean of [10, 12, 14] = 12");
    assert_eq!(spread.val_b, 30.0, "mean of [28, 30, 32] = 30");
    assert_eq!(spread.delta, 18.0);
}

/// `compare_partitions` with the default (averaging-on)
/// path must aggregate every matching sidecar within each
/// side and detect regressions on the aggregated means.
/// End-to-end pin against on-disk fixtures so a regression
/// in the aggregation → compare wiring lands here.
///
/// Fixture: two runs each carrying three sidecars that
/// differ on `scheduler` (the slicing dim). Side A's three
/// trials cluster around `worst_spread = 10` (mean 12);
/// side B's three cluster around `worst_spread = 30` (mean
/// 30). The 18-unit delta clears the default dual gate, so
/// `compare_partitions` returns exit code 1 (regressions
/// detected).
#[test]
fn compare_partitions_with_average_default_produces_regression_on_aggregated_means() {
    use crate::test_support::SidecarResult;

    let alt_root = tempfile::TempDir::new().expect("create alt-root tempdir");
    let run_a = "__avg_thread_a__";
    let run_b = "__avg_thread_b__";

    // Three trials per side, same (scenario, topology,
    // work_type) so they aggregate into a single key. Vary
    // the per-trial spread so the mean is non-degenerate
    // (regression flags would also fire if the values were
    // identical, but the average path is exercised either way).
    let trials_a = [(10.0, 100), (12.0, 120), (14.0, 140)];
    let trials_b = [(28.0, 280), (30.0, 300), (32.0, 320)];

    // Scheduler is the slicing dim: side A's three trials
    // run under "scx_alpha", side B's under "scx_beta". The
    // pairing dims are everything else (kernel/topology/
    // work_type/commit) which match across both runs,
    // so the three trials on each side aggregate into one
    // mean row keyed by `(scenario, topology, work_type)`
    // plus the matching kernel/commit values.
    for (run_key, trials, sched) in [
        (run_a, &trials_a, "scx_alpha"),
        (run_b, &trials_b, "scx_beta"),
    ] {
        let run_dir = alt_root.path().join(run_key);
        std::fs::create_dir_all(&run_dir).expect("create run dir");
        for (i, (spread, gap_ms)) in trials.iter().enumerate() {
            let trial_name = format!("avg_trial_{run_key}_{i}");
            let mut sidecar = SidecarResult {
                test_name: "avg_test".to_string(),
                topology: "1n2l4c1t".to_string(),
                scheduler: sched.to_string(),
                work_type: "SpinWait".to_string(),
                ..SidecarResult::test_fixture()
            };
            sidecar.stats.worst_spread = *spread;
            sidecar.stats.worst_gap_ms = *gap_ms;
            sidecar.passed = true;
            sidecar.skipped = false;
            let json = serde_json::to_string(&sidecar).expect("serialize fixture sidecar");
            let sidecar_path = run_dir.join(format!("{trial_name}.ktstr.json"));
            std::fs::write(&sidecar_path, json).expect("write fixture sidecar");
        }
    }

    let filter_a = RowFilter {
        schedulers: vec!["scx_alpha".to_string()],
        ..RowFilter::default()
    };
    let filter_b = RowFilter {
        schedulers: vec!["scx_beta".to_string()],
        ..RowFilter::default()
    };

    // Default (averaging-on) path: three sidecars per side
    // share one pairing key, so each side aggregates to a
    // single mean row. The 18-unit worst_spread delta on
    // those means (12 vs 30) clears the default dual gate
    // and surfaces exit code 1.
    let exit = compare_partitions(
        &filter_a,
        &filter_b,
        None,
        &ComparisonPolicy::default(),
        Some(alt_root.path()),
        false, // no_average=false → averaging is ON
        &PhaseDisplayOptions::default(),
    )
    .expect("compare_partitions must succeed against valid fixtures");
    assert_eq!(
        exit, 1,
        "an 18-unit worst_spread regression on the aggregated mean \
             (a=12 → b=30) must clear the default dual gate and surface \
             exit code 1; got {exit}",
    );
}

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

// -- PhaseDisplayOptions::rel_threshold (plan_17 step 11c) --

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

// -- Dimension / derive_slicing_dims / pairing dims --

/// `Dimension::ALL` lists all eight dims in canonical order.
/// Order matters for [`PairingKey::from_row`] and for header
/// rendering — a regression that reordered the slice would
/// silently shift every dynamic key, splitting previously-
/// paired rows. Pin the literal order.
#[test]
fn dimension_all_canonical_order() {
    assert_eq!(
        Dimension::ALL,
        &[
            Dimension::Kernel,
            Dimension::Scheduler,
            Dimension::Topology,
            Dimension::WorkType,
            Dimension::ProjectCommit,
            Dimension::KernelCommit,
            Dimension::RunSource,
            Dimension::CpuBudget,
        ],
    );
}

/// `Dimension::pairing_dims` returns every dim NOT in the
/// slicing set, preserving canonical order. Two slicing
/// orderings produce the same pairing-dim list (the function
/// iterates `ALL`, not `slicing`).
#[test]
fn dimension_pairing_dims_complements_slicing() {
    let pair = Dimension::pairing_dims(&[Dimension::Kernel, Dimension::ProjectCommit]);
    assert_eq!(
        pair,
        vec![
            Dimension::Scheduler,
            Dimension::Topology,
            Dimension::WorkType,
            Dimension::KernelCommit,
            Dimension::RunSource,
            Dimension::CpuBudget,
        ],
    );
    // Order of slicing input doesn't change the output —
    // the function iterates ALL and filters.
    let pair_reversed = Dimension::pairing_dims(&[Dimension::ProjectCommit, Dimension::Kernel]);
    assert_eq!(pair, pair_reversed);
}

/// Empty slicing set → every dim is a pairing dim.
#[test]
fn dimension_pairing_dims_empty_slicing_yields_all() {
    let pair = Dimension::pairing_dims(&[]);
    assert_eq!(pair, Dimension::ALL.to_vec());
}

/// `derive_slicing_dims` returns every dimension on which
/// filter_a and filter_b differ. Equal filters → empty
/// slicing.
#[test]
fn derive_slicing_dims_identical_filters_yields_empty() {
    let f = RowFilter {
        schedulers: vec!["scx_alpha".to_string()],
        ..RowFilter::default()
    };
    assert!(derive_slicing_dims(&f, &f).is_empty());
}

/// One-dim diff: only the differing dimension is reported.
#[test]
fn derive_slicing_dims_single_dim_diff() {
    let f_a = RowFilter {
        schedulers: vec!["scx_alpha".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        schedulers: vec!["scx_beta".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(derive_slicing_dims(&f_a, &f_b), vec![Dimension::Scheduler]);
}

/// Vec dims (kernels/commits) compare as sorted-deduped sets —
/// order and duplicates inside the filter don't shift the
/// slicing-dim derivation.
#[test]
fn derive_slicing_dims_vec_compares_as_set() {
    let f_a = RowFilter {
        kernels: vec!["6.14".to_string(), "6.15".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        kernels: vec!["6.15".to_string(), "6.14".to_string(), "6.14".to_string()],
        ..RowFilter::default()
    };
    assert!(
        derive_slicing_dims(&f_a, &f_b).is_empty(),
        "same set in different order/multiplicity must NOT slice",
    );
}

/// Multi-dim diff: every differing dimension is reported, in
/// canonical [`Dimension::ALL`] order.
#[test]
fn derive_slicing_dims_multi_dim_diff_in_canonical_order() {
    let f_a = RowFilter {
        kernels: vec!["6.14".to_string()],
        schedulers: vec!["scx_alpha".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        kernels: vec!["6.15".to_string()],
        schedulers: vec!["scx_beta".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        derive_slicing_dims(&f_a, &f_b),
        vec![Dimension::Kernel, Dimension::Scheduler],
    );
}

/// Source-only diff: filters that disagree on `run_sources`
/// and agree on every other dimension produce a slicing-dim
/// set containing exactly `Dimension::RunSource`. Pins the
/// Source arm of the per-dimension comparison switch in
/// [`derive_slicing_dims`] — a regression that omitted the
/// arm or compared the wrong field would surface here as an
/// empty slicing-dim set (and downstream as a `compare`
/// command that mistakenly bails with "A and B select
/// identical rows" on a legitimate source contrast).
#[test]
fn derive_slicing_dims_source_only_diff() {
    let f_a = RowFilter {
        run_sources: vec!["local".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        run_sources: vec!["ci".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        derive_slicing_dims(&f_a, &f_b),
        vec![Dimension::RunSource],
        "differing `run_sources` must surface Source as a slicing dim",
    );

    // Sorted-deduped Vec semantics also apply on the Source
    // dim — same set in different order/multiplicity must NOT
    // slice. Mirrors the `derive_slicing_dims_vec_compares_as_set`
    // contract for the Source arm.
    let f_c = RowFilter {
        run_sources: vec!["local".to_string(), "ci".to_string()],
        ..RowFilter::default()
    };
    let f_d = RowFilter {
        run_sources: vec!["ci".to_string(), "local".to_string(), "local".to_string()],
        ..RowFilter::default()
    };
    assert!(
        derive_slicing_dims(&f_c, &f_d).is_empty(),
        "same run_source set in different order/multiplicity must NOT slice",
    );
}

/// Topology-only diff: filters that disagree on `topologies`
/// and agree on every other dimension produce a slicing-dim
/// set containing exactly `Dimension::Topology`. Pins the
/// Topology arm of the per-dimension comparison switch in
/// [`derive_slicing_dims`] for the post-Vec-promotion
/// `topologies` field; before promotion `--topology` was a
/// single-value `Option<String>` and the per-arm comparison
/// shape was `Option<String> != Option<String>`. Mirror of
/// `derive_slicing_dims_source_only_diff` for the Topology
/// arm.
#[test]
fn derive_slicing_dims_topology_only_diff() {
    let f_a = RowFilter {
        topologies: vec!["1n2l4c1t".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        topologies: vec!["1n2l4c2t".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        derive_slicing_dims(&f_a, &f_b),
        vec![Dimension::Topology],
        "differing `topologies` must surface Topology as a slicing dim",
    );

    // Sorted-deduped Vec semantics: same set in different
    // order/multiplicity must NOT slice.
    let f_c = RowFilter {
        topologies: vec!["1n2l4c1t".to_string(), "1n2l4c2t".to_string()],
        ..RowFilter::default()
    };
    let f_d = RowFilter {
        topologies: vec![
            "1n2l4c2t".to_string(),
            "1n2l4c1t".to_string(),
            "1n2l4c1t".to_string(),
        ],
        ..RowFilter::default()
    };
    assert!(
        derive_slicing_dims(&f_c, &f_d).is_empty(),
        "same topology set in different order/multiplicity must NOT slice",
    );
}

/// WorkType-only diff: filters that disagree on `work_types`
/// and agree on every other dimension produce a slicing-dim
/// set containing exactly `Dimension::WorkType`. Mirror of
/// `derive_slicing_dims_topology_only_diff` for the WorkType
/// arm.
#[test]
fn derive_slicing_dims_work_type_only_diff() {
    let f_a = RowFilter {
        work_types: vec!["SpinWait".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        work_types: vec!["PageFaultChurn".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        derive_slicing_dims(&f_a, &f_b),
        vec![Dimension::WorkType],
        "differing `work_types` must surface WorkType as a slicing dim",
    );

    // Sorted-deduped Vec semantics: same set in different
    // order/multiplicity must NOT slice.
    let f_c = RowFilter {
        work_types: vec!["SpinWait".to_string(), "PageFaultChurn".to_string()],
        ..RowFilter::default()
    };
    let f_d = RowFilter {
        work_types: vec![
            "PageFaultChurn".to_string(),
            "SpinWait".to_string(),
            "SpinWait".to_string(),
        ],
        ..RowFilter::default()
    };
    assert!(
        derive_slicing_dims(&f_c, &f_d).is_empty(),
        "same work_type set in different order/multiplicity must NOT slice",
    );
}

/// `kernel_filter_matches`: major.minor (`6.12`) prefix
/// matches every patch in the series via the
/// `starts_with("6.12.")` arm, and ALSO matches `6.12`
/// exactly. Three-segment-or-longer filters are strict.
#[test]
fn kernel_filter_matches_major_minor_prefix() {
    // Two-segment filter: prefix matches.
    assert!(kernel_filter_matches("6.12", "6.12"));
    assert!(kernel_filter_matches("6.12", "6.12.0"));
    assert!(kernel_filter_matches("6.12", "6.12.5"));
    assert!(!kernel_filter_matches("6.12", "6.13.0"));
    // Critically: `6.1` must not match `6.10.0` — the
    // trailing-dot in the prefix path prevents the
    // accidental wildcard.
    assert!(!kernel_filter_matches("6.1", "6.10.0"));
}

/// `kernel_filter_matches`: major.minor prefix admits the
/// `MAJOR.MINOR-rcN` pre-release shape via the
/// `starts_with("MAJOR.MINOR-")` arm. The rcN kernel shares
/// the `(major, minor, patch=0)` tuple with the eventual
/// release per `kernel_path::decompose_version_for_compare`,
/// and the operator filtering on `6.14` wants the whole
/// series — release AND pre-releases. This complements the
/// `6.14.0-rc3` (kernel-banner) shape which already matched
/// via the trailing-dot prefix.
#[test]
fn kernel_filter_matches_major_minor_admits_rc_pre_release() {
    // No-patch pre-release shape (kernel_path KernelId::Version
    // doc cites `6.15-rc3` as a valid version string).
    assert!(kernel_filter_matches("6.14", "6.14-rc3"));
    assert!(kernel_filter_matches("6.14", "6.14-rc1"));
    // Patch+rc shape (kernel banner from a kernel.org
    // `v6.14-rc3` tag is `Linux version 6.14.0-rc3+`).
    assert!(kernel_filter_matches("6.14", "6.14.0-rc3"));
    assert!(kernel_filter_matches("6.14", "6.14.0-rc3+"));
    // The dash-prefix arm must NOT wildcard across series:
    // `6.1` filtering must reject `6.14-rc3` for the same
    // reason `6.1` rejects `6.10.0`.
    assert!(!kernel_filter_matches("6.1", "6.14-rc3"));
    // Cross-minor rc rejection.
    assert!(!kernel_filter_matches("6.14", "6.15-rc3"));
}

/// `kernel_filter_matches`: three-segment+ filters are strict
/// equality.
#[test]
fn kernel_filter_matches_strict_for_three_plus_segments() {
    assert!(kernel_filter_matches("6.14.2", "6.14.2"));
    // Critically: `6.14.2` must NOT match `6.14.20` — the
    // strict-equality arm prevents the patch-level prefix
    // wildcarding.
    assert!(!kernel_filter_matches("6.14.2", "6.14.20"));
    assert!(!kernel_filter_matches("6.14.2", "6.14.21"));
    // RC suffixes are also strict.
    assert!(kernel_filter_matches("6.15-rc3", "6.15-rc3"));
    assert!(!kernel_filter_matches("6.15-rc3", "6.15-rc30"));
}

/// `RowFilter::matches` with a major.minor `--kernel` filter
/// admits the row whose `kernel_version` is a patch in that
/// series.
#[test]
fn row_filter_kernel_major_minor_prefix_admits_patch_version() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.12.5"));
    let filter = RowFilter {
        kernels: vec!["6.12".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter.matches(&row),
        "major.minor filter `6.12` must admit row with kernel_version `6.12.5`",
    );
}

// -- PairingKey --

/// `PairingKey::from_row` always puts `scenario` first, then
/// the requested dims in canonical order. Two rows with the
/// same scenario+dims agree; one with a different topology
/// (when topology IS a pairing dim) does not.
#[test]
fn pairing_key_from_row_basic() {
    let row_a = make_filter_row("scenA", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    let row_b = make_filter_row("scenA", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    let row_c = make_filter_row("scenA", "scx_a", "2n2l", "SpinWait", Some("6.14"));
    let dims = &[Dimension::Topology, Dimension::WorkType];
    assert_eq!(
        PairingKey::from_row(&row_a, dims),
        PairingKey::from_row(&row_b, dims),
    );
    assert_ne!(
        PairingKey::from_row(&row_a, dims),
        PairingKey::from_row(&row_c, dims),
        "different topology must distinguish the keys when topology is a pairing dim",
    );
}

/// Slicing on topology means topology is NOT in the pairing
/// dim set — so two rows that differ ONLY on topology pair
/// to the same key, allowing the comparison to contrast
/// them across A/B sides.
#[test]
fn pairing_key_excludes_slicing_dim() {
    let row_a = make_filter_row("scenA", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    let row_b = make_filter_row("scenA", "scx_a", "2n2l", "SpinWait", Some("6.14"));
    // Pairing dims = ALL minus Topology. So these two rows
    // pair iff they agree on everything BUT topology.
    let pair_dims = Dimension::pairing_dims(&[Dimension::Topology]);
    assert_eq!(
        PairingKey::from_row(&row_a, &pair_dims),
        PairingKey::from_row(&row_b, &pair_dims),
        "rows differing only on a slicing dim must produce equal pairing keys",
    );
}

/// `PairingKey::from_row` first slot is always scenario;
/// rendering via `parts.join("/")` reproduces the
/// `scenario/topology/work_type` shape when those dims are
/// pairing dims.
#[test]
fn pairing_key_join_renders_legacy_shape() {
    let row = make_filter_row("test_a", "scx_a", "1n2l", "SpinWait", Some("6.14"));
    let key = PairingKey::from_row(&row, LEGACY_PAIRING_DIMS);
    assert_eq!(
        key.0.join("/"),
        "test_a/1n2l/SpinWait",
        "legacy-shape join must render the three-segment label",
    );
}

/// `PairingKey::from_row` includes the row's `kernel_commit`
/// when `KernelCommit` is in the pairing-dim list, and
/// excludes it when `KernelCommit` is the slicing dim. Pins
/// the [`Dimension::KernelCommit`] arm of the from_row match
/// — a regression that omitted the arm or substituted the
/// wrong row field would surface here as either a missing
/// key slot or a slot carrying the wrong value.
///
/// `None` kernel_commit renders as the empty string slot per
/// the `unwrap_or_default()` policy on Option dims; that
/// shape is shared across every Option-typed dim arm.
#[test]
fn pairing_key_from_row_includes_kernel_commit_when_pairing() {
    let mut row_some = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_some.kernel_commit = Some("kabcde7".to_string());
    let mut row_none = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_none.kernel_commit = None;

    // KernelCommit in pairing dims → key carries the commit
    // value (or the empty slot for None). The two rows
    // therefore produce DIFFERENT keys because their
    // kernel_commit values disagree.
    let pair_dims = &[Dimension::KernelCommit];
    let key_some = PairingKey::from_row(&row_some, pair_dims);
    let key_none = PairingKey::from_row(&row_none, pair_dims);
    assert_eq!(
        key_some.0,
        vec!["scn".to_string(), "kabcde7".to_string()],
        "Some(kernel_commit) must occupy the second slot verbatim",
    );
    assert_eq!(
        key_none.0,
        vec!["scn".to_string(), String::new()],
        "None kernel_commit must collapse to an empty slot per \
             unwrap_or_default policy",
    );
    assert_ne!(
        key_some, key_none,
        "two rows differing on kernel_commit must produce \
             distinct pairing keys when KernelCommit is a pairing dim",
    );

    // KernelCommit excluded (slicing) → the two rows pair to
    // the same key because the dim is dropped. Pins the
    // dimensional-slicing semantic for the new arm.
    let slice_dims = Dimension::pairing_dims(&[Dimension::KernelCommit]);
    assert_eq!(
        PairingKey::from_row(&row_some, &slice_dims),
        PairingKey::from_row(&row_none, &slice_dims),
        "rows differing only on the slicing dim (KernelCommit) \
             must produce equal pairing keys",
    );
}

/// `PairingKey::from_row` includes the row's `run_source`
/// when `RunSource` is in the pairing-dim list, and excludes it
/// when `RunSource` is the slicing dim. Pins the
/// [`Dimension::RunSource`] arm of the from_row match — same
/// shape and motivation as
/// `pairing_key_from_row_includes_kernel_commit_when_pairing`
/// but for the run_source arm. A regression that omitted the
/// arm or substituted `row.kernel_commit` for
/// `row.run_source` would surface here.
#[test]
fn pairing_key_from_row_includes_run_source_when_pairing() {
    let mut row_local = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_local.run_source = Some("local".to_string());
    let mut row_ci = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_ci.run_source = Some("ci".to_string());
    let mut row_none = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_none.run_source = None;

    let pair_dims = &[Dimension::RunSource];
    let key_local = PairingKey::from_row(&row_local, pair_dims);
    let key_ci = PairingKey::from_row(&row_ci, pair_dims);
    let key_none = PairingKey::from_row(&row_none, pair_dims);
    assert_eq!(
        key_local.0,
        vec!["scn".to_string(), "local".to_string()],
        "Some(run_source) must occupy the second slot verbatim",
    );
    assert_eq!(key_ci.0, vec!["scn".to_string(), "ci".to_string()]);
    assert_eq!(
        key_none.0,
        vec!["scn".to_string(), String::new()],
        "None run_source must collapse to an empty slot per \
             unwrap_or_default policy",
    );
    assert_ne!(
        key_local, key_ci,
        "two rows differing on run_source must produce \
             distinct pairing keys when Source is a pairing dim",
    );

    // Source excluded (slicing) → the differing-run_source
    // rows pair to the same key.
    let slice_dims = Dimension::pairing_dims(&[Dimension::RunSource]);
    assert_eq!(
        PairingKey::from_row(&row_local, &slice_dims),
        PairingKey::from_row(&row_ci, &slice_dims),
        "rows differing only on the slicing dim (Source) must \
             produce equal pairing keys",
    );
}

/// `PairingKey::from_row` includes the row's cpu_budget when CpuBudget
/// is a pairing dim — so cross-budget rows NEVER pair — and excludes
/// it when CpuBudget is the slicing dim. This is the whole point of
/// #5: a 4-CPU-budget run and a 32-CPU-budget run measure different
/// things and must not be silently compared.
#[test]
fn pairing_key_from_row_includes_cpu_budget_when_pairing() {
    let mut row_4 = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_4.cpu_budget = Some(4);
    let mut row_32 = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_32.cpu_budget = Some(32);
    let mut row_none = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_none.cpu_budget = None;

    let pair_dims = &[Dimension::CpuBudget];
    let key_4 = PairingKey::from_row(&row_4, pair_dims);
    let key_32 = PairingKey::from_row(&row_32, pair_dims);
    let key_none = PairingKey::from_row(&row_none, pair_dims);
    assert_eq!(key_4.0, vec!["scn".to_string(), "4".to_string()]);
    assert_eq!(key_32.0, vec!["scn".to_string(), "32".to_string()]);
    assert_eq!(
        key_none.0,
        vec!["scn".to_string(), String::new()],
        "None cpu_budget (a skip) collapses to an empty slot",
    );
    assert_ne!(
        key_4, key_32,
        "rows of different cpu_budget must NOT pair when CpuBudget pairs",
    );
    assert_ne!(
        key_4, key_none,
        "a budgeted row must not pair with a skip (None budget)",
    );

    // CpuBudget sliced → the dim is dropped → the two budgets pair.
    let slice_dims = Dimension::pairing_dims(&[Dimension::CpuBudget]);
    assert_eq!(
        PairingKey::from_row(&row_4, &slice_dims),
        PairingKey::from_row(&row_32, &slice_dims),
        "rows differing only on the sliced dim (CpuBudget) must pair",
    );
}

/// Clean and dirty contributors at the same canonical hex
/// must land in the same pairing bucket. Without the
/// `-dirty` strip in `commit_pairing_key_part`, `abc1234`
/// and `abc1234-dirty` shatter into separate groups,
/// defeating `group_and_average_by`'s `+mixed` cohort
/// detection (which can only fire when the two contributors
/// land in ONE group).
#[test]
fn pairing_key_from_row_strips_dirty_suffix_on_commit() {
    let mut row_clean = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_clean.commit = Some("abc1234".to_string());
    let mut row_dirty = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_dirty.commit = Some("abc1234-dirty".to_string());

    let pair_dims = &[Dimension::ProjectCommit];
    let key_clean = PairingKey::from_row(&row_clean, pair_dims);
    let key_dirty = PairingKey::from_row(&row_dirty, pair_dims);

    assert_eq!(
        key_clean, key_dirty,
        "clean `abc1234` and dirty `abc1234-dirty` must produce \
             EQUAL pairing keys so the +mixed cohort machinery in \
             group_and_average_by can surface their disagreement",
    );
    assert_eq!(
        key_clean.0,
        vec!["scn".to_string(), "abc1234".to_string()],
        "key part must be the canonical un-suffixed hex",
    );
}

/// Same shape on the kernel_commit dimension. Pins the
/// second commit dim's strip independently because
/// `from_row` uses two parallel arms; a regression could
/// strip one but not the other.
#[test]
fn pairing_key_from_row_strips_dirty_suffix_on_kernel_commit() {
    let mut row_clean = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_clean.kernel_commit = Some("def5678".to_string());
    let mut row_dirty = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_dirty.kernel_commit = Some("def5678-dirty".to_string());

    let pair_dims = &[Dimension::KernelCommit];
    let key_clean = PairingKey::from_row(&row_clean, pair_dims);
    let key_dirty = PairingKey::from_row(&row_dirty, pair_dims);

    assert_eq!(
        key_clean, key_dirty,
        "clean and dirty kernel_commit at the same canonical \
             hex must pair together",
    );
    assert_eq!(key_clean.0, vec!["scn".to_string(), "def5678".to_string()],);
}

/// Distinct hexes still differentiate even when one carries
/// `-dirty`. Pins that the strip operates ONLY on the
/// suffix, not on the entire value — `aaa1111-dirty` and
/// `bbb2222` remain distinct.
#[test]
fn pairing_key_from_row_distinct_hexes_remain_distinct_under_strip() {
    let mut row_a = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_a.commit = Some("aaa1111-dirty".to_string());
    let mut row_b = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_b.commit = Some("bbb2222".to_string());

    let pair_dims = &[Dimension::ProjectCommit];
    let key_a = PairingKey::from_row(&row_a, pair_dims);
    let key_b = PairingKey::from_row(&row_b, pair_dims);

    assert_ne!(
        key_a, key_b,
        "distinct canonical hexes must remain distinct after the \
             -dirty strip — only the suffix is stripped",
    );
    assert_eq!(key_a.0[1], "aaa1111");
    assert_eq!(key_b.0[1], "bbb2222");
}

/// `None` commit values still collapse to the empty slot
/// (the strip is a no-op on `None`). Pins the absence path
/// against a regression that special-cased the strip and
/// inadvertently changed the unwrap_or_default behavior.
#[test]
fn pairing_key_from_row_none_commit_unchanged_under_strip() {
    let mut row = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row.commit = None;
    row.kernel_commit = None;
    let pair_dims = &[Dimension::ProjectCommit, Dimension::KernelCommit];
    let key = PairingKey::from_row(&row, pair_dims);
    assert_eq!(
        key.0,
        vec!["scn".to_string(), String::new(), String::new()],
        "None commit and None kernel_commit must collapse to empty slots",
    );
}

// -- render_side_label --

/// Empty slicing dims → the bare label is returned.
#[test]
fn render_side_label_empty_dims_yields_bare() {
    let f = RowFilter::default();
    assert_eq!(render_side_label(&f, &[], "A"), "A");
}

/// Single-dim single-value scheduler renders the value
/// verbatim. After the Vec promotion of `--scheduler` the
/// scheduler arm goes through `render_vec_dim` like every
/// other Vec dim; a single entry still surfaces the bare
/// string.
#[test]
fn render_side_label_single_value_dim() {
    let f = RowFilter {
        schedulers: vec!["scx_rusty".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f, &[Dimension::Scheduler], "A"),
        "scx_rusty",
    );
}

/// Vec dim with ≤3 entries joins with `|` (sorted).
#[test]
fn render_side_label_vec_dim_short_joins_with_pipe() {
    let f = RowFilter {
        kernels: vec!["6.15".to_string(), "6.14".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f, &[Dimension::Kernel], "A"),
        "6.14|6.15",
        "≤3 values must join sorted with `|`",
    );
}

/// Vec dim with >3 entries collapses to the bare label.
#[test]
fn render_side_label_vec_dim_long_collapses_to_bare() {
    let f = RowFilter {
        kernels: vec![
            "6.10".to_string(),
            "6.11".to_string(),
            "6.12".to_string(),
            "6.13".to_string(),
            "6.14".to_string(),
        ],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f, &[Dimension::Kernel], "A"),
        "A",
        ">3 values must collapse to the bare letter so the \
             column header stays readable",
    );
}

/// Multi-dim slicing joins per-dim parts with `:`.
#[test]
fn render_side_label_multi_dim_joins_with_colon() {
    let f = RowFilter {
        kernels: vec!["6.14".to_string()],
        schedulers: vec!["scx_rusty".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f, &[Dimension::Kernel, Dimension::Scheduler], "A"),
        "6.14:scx_rusty",
    );
}

/// Empty per-side filter on a slicing dim falls back to the
/// bare label (the slice exists because the OTHER side
/// populated the dim).
#[test]
fn render_side_label_empty_dim_value_uses_bare() {
    let f = RowFilter::default();
    assert_eq!(
        render_side_label(&f, &[Dimension::Kernel], "B"),
        "B",
        "empty Vec dim must fall back to the bare letter",
    );
    assert_eq!(
        render_side_label(&f, &[Dimension::Scheduler], "B"),
        "B",
        "None Option dim must fall back to the bare letter",
    );
}

/// `Dimension::KernelCommit` arm of [`render_side_label`] reads
/// `filter.kernel_commits` (a Vec) and routes through the same
/// `render_vec_dim` path as `Kernel` / `ProjectCommit`. Pins
/// the arm so a regression that omitted it (or substituted the
/// wrong field, e.g. `filter.project_commits`) surfaces here
/// instead of silently rendering the bare label even when the
/// filter is populated.
///
/// Single-value: emits the value verbatim. Two-value: joins
/// sorted with `|` per `render_vec_dim`'s ≤3 rule. >3 values:
/// collapse to bare. Empty Vec: bare. Same shape as the
/// `Kernel` arm pinned above; a regression in the
/// `KernelCommit` arm specifically would NOT be caught by the
/// existing `render_side_label_vec_dim_*` tests because those
/// only exercise the `Kernel` field.
#[test]
fn render_side_label_kernel_commit_arm_renders_filter_value() {
    let f_one = RowFilter {
        kernel_commits: vec!["kabcde7".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_one, &[Dimension::KernelCommit], "A"),
        "kabcde7",
        "single kernel_commit value must render verbatim — \
             a regression that read `filter.project_commits` instead of \
             `filter.kernel_commits` would render `A` here because \
             the project-commit field is empty",
    );

    let f_two = RowFilter {
        kernel_commits: vec!["kbbb222".to_string(), "kaaa111".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_two, &[Dimension::KernelCommit], "A"),
        "kaaa111|kbbb222",
        "≤3 kernel_commit values must join sorted with `|`",
    );

    let f_long = RowFilter {
        kernel_commits: vec![
            "k111".to_string(),
            "k222".to_string(),
            "k333".to_string(),
            "k444".to_string(),
        ],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_long, &[Dimension::KernelCommit], "A"),
        "A",
        ">3 kernel_commit values must collapse to the bare letter",
    );

    let f_empty = RowFilter::default();
    assert_eq!(
        render_side_label(&f_empty, &[Dimension::KernelCommit], "B"),
        "B",
        "empty kernel_commits Vec must fall back to the bare letter",
    );
}

/// `Dimension::RunSource` arm of [`render_side_label`] reads
/// `filter.run_sources` (a Vec) and routes through the same
/// `render_vec_dim` path as the other Vec dims. Mirror of
/// `render_side_label_kernel_commit_arm_renders_filter_value`
/// for the Source arm. A regression that omitted the Source
/// arm or substituted the wrong field would surface here
/// instead of silently rendering the bare label even when
/// the filter is populated.
#[test]
fn render_side_label_source_arm_renders_filter_value() {
    let f_one = RowFilter {
        run_sources: vec!["local".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_one, &[Dimension::RunSource], "A"),
        "local",
        "single run_source value must render verbatim — a \
             regression that read another field would render `A` here",
    );

    let f_two = RowFilter {
        run_sources: vec!["local".to_string(), "ci".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_two, &[Dimension::RunSource], "A"),
        "ci|local",
        "≤3 run_source values must join sorted with `|`",
    );

    let f_long = RowFilter {
        run_sources: vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_long, &[Dimension::RunSource], "A"),
        "A",
        ">3 run_source values must collapse to the bare letter",
    );

    let f_empty = RowFilter::default();
    assert_eq!(
        render_side_label(&f_empty, &[Dimension::RunSource], "B"),
        "B",
        "empty run_sources Vec must fall back to the bare letter",
    );
}

/// `zero_match_diagnostic` flags a `--run-source` value that is
/// not present in the pool, naming the unknown value AND the
/// distinct values actually seen. Guards against the
/// typo-class miss (e.g. `--run-source loca` for `local`,
/// `--run-source CI` for `ci`) that produces a silent
/// zero-match in `compare_partitions`.
#[test]
fn zero_match_diagnostic_unknown_run_source_lists_present_values() {
    let mut row_local = make_row("scn", "1n1l1c1t", true, 1.0);
    row_local.run_source = Some("local".to_string());
    let mut row_ci = make_row("scn", "1n1l1c1t", true, 1.0);
    row_ci.run_source = Some("ci".to_string());
    let rows = vec![row_local, row_ci];
    let filter = RowFilter {
        run_sources: vec!["loca".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        msg.contains("--run-source `loca` not found"),
        "must name the unknown value verbatim; got:\n{msg}",
    );
    assert!(
        msg.contains("`ci`") && msg.contains("`local`"),
        "must list distinct values present in the pool so the \
             operator can correct the typo; got:\n{msg}",
    );
    assert!(
        msg.contains("case-sensitive"),
        "must mention case sensitivity (`ci` ≠ `CI`); got:\n{msg}",
    );
}

/// `zero_match_diagnostic` flags a `--cpu-budget` value not present
/// in the pool, naming the unknown value AND the distinct budgets
/// actually seen — the numeric mirror of the run_source hint. Skip
/// rows (`cpu_budget: None`) contribute no budget to the list.
#[test]
fn zero_match_diagnostic_unknown_cpu_budget_lists_present_values() {
    let mut row4 = make_row("scn", "1n1l1c1t", true, 1.0);
    row4.cpu_budget = Some(4);
    let mut row32 = make_row("scn", "1n1l1c1t", true, 1.0);
    row32.cpu_budget = Some(32);
    let skip = make_row("scn", "1n1l1c1t", true, 1.0); // cpu_budget None
    let rows = vec![row4, row32, skip];
    let filter = RowFilter {
        cpu_budgets: vec!["64".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        msg.contains("--cpu-budget `64` not found"),
        "must name the unknown budget verbatim; got:\n{msg}",
    );
    assert!(
        msg.contains("`4`") && msg.contains("`32`"),
        "must list distinct budgets present in the pool; got:\n{msg}",
    );
}

/// A `--cpu-budget` value that DOES match a row must NOT trigger
/// the unknown-budget hint (guards against the hint firing for
/// every populated `--cpu-budget` regardless of pool membership).
#[test]
fn zero_match_diagnostic_known_cpu_budget_does_not_fire_unknown_hint() {
    let mut row = make_row("scn", "1n1l1c1t", true, 1.0);
    row.cpu_budget = Some(4);
    let rows = vec![row];
    let filter = RowFilter {
        cpu_budgets: vec!["4".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        !msg.contains("--cpu-budget `4` not found"),
        "a present budget must not fire the unknown-budget hint; got:\n{msg}",
    );
}

/// When every row has `run_source: None`, the hint surfaces the
/// "(none — every row has `run_source: null`)" form rather than
/// an empty list. This is the post-`apply_archive_source_override`
/// path with a pool that pre-dates the run_source field, so
/// distinguishing "unknown value, no values present" from
/// "unknown value, here's what's there" is operator-actionable.
#[test]
fn zero_match_diagnostic_unknown_run_source_with_empty_pool_explains_absence() {
    let row = make_row("scn", "1n1l1c1t", true, 1.0);
    let rows = vec![row];
    let filter = RowFilter {
        run_sources: vec!["ci".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        msg.contains("--run-source `ci` not found"),
        "must name the unknown value; got:\n{msg}",
    );
    assert!(
        msg.contains("none — every row has `run_source: null`"),
        "must explain the empty-distinct-values case rather than \
             listing nothing; got:\n{msg}",
    );
}

/// A `--run-source` value that DOES match a row in the pool
/// must NOT trigger the unknown-value hint, even when the
/// filter still matches zero rows due to other dimension
/// mismatches (e.g. scenario filter zeroes the set first).
/// Pinning this guards against a regression where the hint
/// fires for every populated `--run-source` regardless of
/// pool membership.
#[test]
fn zero_match_diagnostic_known_run_source_does_not_fire_unknown_hint() {
    let mut row = make_row("scn", "1n1l1c1t", true, 1.0);
    row.run_source = Some("local".to_string());
    let rows = vec![row];
    let filter = RowFilter {
        run_sources: vec!["local".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        !msg.contains("--run-source") || !msg.contains("not found"),
        "must NOT fire the unknown-source hint when the value is \
             present in the pool; got:\n{msg}",
    );
}

/// `zero_match_diagnostic` fires the dirty-form hint for a
/// `--project-commit X` filter when the pool contains a
/// matching `X-dirty` row — pointing the operator at the
/// dirty form so they don't have to manually scan
/// `stats list-values`. The hint must name the original
/// value, the dirty form, and the suggested replacement
/// flag form.
#[test]
fn zero_match_diagnostic_project_commit_dirty_hint_fires() {
    let mut row = make_row("scn", "1n1l1c1t", true, 1.0);
    row.commit = Some("abcdef1-dirty".to_string());
    let rows = vec![row];
    let filter = RowFilter {
        project_commits: vec!["abcdef1".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        msg.contains("no rows match `--project-commit abcdef1`"),
        "hint must name the unmatched filter value verbatim; \
             got:\n{msg}",
    );
    assert!(
        msg.contains("`abcdef1-dirty` exists in the pool"),
        "hint must surface the dirty form found in the pool; \
             got:\n{msg}",
    );
    assert!(
        msg.contains("did you mean `--project-commit abcdef1-dirty`"),
        "hint must propose the dirty form as the corrected flag; \
             got:\n{msg}",
    );
}

/// Companion to
/// `zero_match_diagnostic_project_commit_dirty_hint_fires`
/// for the `kernel_commits` arm. Same shape: hint names the
/// unmatched value, the matching `-dirty` form found in the
/// pool, and the suggested `--kernel-commit` replacement.
/// A regression that wired the kernel_commits arm to scan
/// `row.commit` (or never wired it at all) would surface
/// here as a missing hint.
#[test]
fn zero_match_diagnostic_kernel_commit_dirty_hint_fires() {
    let mut row = make_row("scn", "1n1l1c1t", true, 1.0);
    row.kernel_commit = Some("kabcde7-dirty".to_string());
    let rows = vec![row];
    let filter = RowFilter {
        kernel_commits: vec!["kabcde7".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        msg.contains("no rows match `--kernel-commit kabcde7`"),
        "hint must name the unmatched kernel_commit value verbatim; \
             got:\n{msg}",
    );
    assert!(
        msg.contains("`kabcde7-dirty` exists in the pool"),
        "hint must surface the dirty form found in the pool; \
             got:\n{msg}",
    );
    assert!(
        msg.contains("did you mean `--kernel-commit kabcde7-dirty`"),
        "hint must propose the dirty form as the corrected flag; \
             got:\n{msg}",
    );
}

/// `zero_match_diagnostic` appends the `stats list-values`
/// redirect when the operator narrowed on a commit
/// dimension (project_commits OR kernel_commits populated)
/// — that redirect points at the per-dimension dump where
/// the commit values can be cross-referenced. Without a
/// commit-dim filter the redirect is suppressed because
/// `list-values` would dump every dimension, which is no
/// more actionable than the existing `stats list` redirect
/// at the top of the message for a kernel / scheduler /
/// topology miss.
#[test]
fn zero_match_diagnostic_list_values_redirect_when_commit_dim_populated() {
    let row = make_row("scn", "1n1l1c1t", true, 1.0);
    let rows = vec![row];
    let filter = RowFilter {
        project_commits: vec!["abcdef1".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        msg.contains("cargo ktstr stats list-values"),
        "must include the list-values redirect when commit \
             dim filter is populated; got:\n{msg}",
    );

    // Same redirect when only kernel_commits is populated.
    let filter_kc = RowFilter {
        kernel_commits: vec!["kabcde7".to_string()],
        ..Default::default()
    };
    let msg_kc = zero_match_diagnostic("A", &filter_kc, &rows, rows.len());
    assert!(
        msg_kc.contains("cargo ktstr stats list-values"),
        "list-values redirect must also fire on the \
             kernel_commits arm; got:\n{msg_kc}",
    );
}

/// Without a commit-dim filter populated, the list-values
/// redirect must NOT fire — generic kernel / scheduler /
/// topology / work-type misses already get the `stats list`
/// redirect, and a list-values dump would be noise rather
/// than signal. Pins the suppression so a regression that
/// always emitted the redirect (or omitted the touched-
/// commit-dim guard) surfaces here.
#[test]
fn zero_match_diagnostic_no_list_values_redirect_when_no_commit_dim() {
    let row = make_row("scn", "1n1l1c1t", true, 1.0);
    let rows = vec![row];
    // Filter narrowed on a non-commit dim only — the
    // redirect must stay quiet.
    let filter = RowFilter {
        schedulers: vec!["scx_alpha".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        !msg.contains("cargo ktstr stats list-values"),
        "list-values redirect must NOT fire when no commit-dim \
             filter is populated; got:\n{msg}",
    );
}

// -- sorted_run_entries (testable extraction of list_runs sort logic) --

/// `sorted_run_entries` orders subdirectories under `root` by
/// directory mtime DESCENDING — newest first. Pins the contract
/// so a regression that flips the sort direction (e.g. drops
/// `Reverse`) or removes the mtime probe (reverting to
/// `file_name`-only sort) surfaces here as the order shift.
///
/// Three subdirs are created with `std::thread::sleep` between
/// `create_dir` calls so each directory's mtime is captured at
/// a strictly later instant than the previous one. 100 ms is
/// generous: ext4/btrfs/xfs nsec resolution + monotonic
/// CLOCK_REALTIME advancement guarantee distinct mtimes per
/// dir at this granularity.
///
/// The OLDEST directory is named `aaa_oldest` and the NEWEST
/// is named `zzz_newest` — paired so the lexical-ascending
/// order (aaa, mmm, zzz) is the OPPOSITE of the mtime-descending
/// order (zzz_newest, mmm_middle, aaa_oldest). Without this
/// pairing, lexical-ascending and mtime-descending would
/// produce the same output and a regression to filename-only
/// sort would not be detectable. With this pairing, any
/// regression that drops `Reverse` (mtime-ASCENDING) OR
/// reverts to filename-only sort (lexical-ASCENDING) yields
/// `aaa, mmm, zzz` — the WRONG order — and the test fails
/// loud.
#[test]
fn sorted_run_entries_orders_by_mtime_descending() {
    use std::thread::sleep;
    use std::time::Duration;

    let root = tempfile::TempDir::new().expect("tempdir");
    let oldest = root.path().join("aaa_oldest");
    let middle = root.path().join("mmm_middle");
    let newest = root.path().join("zzz_newest");
    std::fs::create_dir(&oldest).expect("mkdir oldest");
    sleep(Duration::from_millis(100));
    std::fs::create_dir(&middle).expect("mkdir middle");
    sleep(Duration::from_millis(100));
    std::fs::create_dir(&newest).expect("mkdir newest");

    let rows = super::sorted_run_entries(root.path()).expect("sorted_run_entries must succeed");
    let names: Vec<String> = rows
        .iter()
        .map(|(p, _, _, _)| {
            p.file_name()
                .expect("path must have a file_name")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "zzz_newest".to_string(),
            "mmm_middle".to_string(),
            "aaa_oldest".to_string(),
        ],
        "rows must be sorted by mtime descending: newest dir \
             (`zzz_newest`) first, oldest dir (`aaa_oldest`) last. \
             A regression that drops Reverse (mtime-ascending) or \
             reverts to filename-only sort (lexical-ascending) \
             would yield aaa, mmm, zzz — the OPPOSITE of the \
             expected mtime-descending order — and would fail this \
             assertion.",
    );
}

/// Empty root: `sorted_run_entries` returns an empty vec
/// rather than erroring. Pins the no-runs path that the
/// `list_runs` caller short-circuits with the
/// "no runs found" eprintln.
#[test]
fn sorted_run_entries_empty_root_yields_empty_vec() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let rows = super::sorted_run_entries(root.path()).expect("sorted_run_entries must succeed");
    assert!(
        rows.is_empty(),
        "empty root must yield empty vec; got {rows:?}",
    );
}

/// `sorted_run_entries` skips files (only subdirectories
/// become rows). Pins the `is_dir()` filter — a regression
/// that included file entries would surface here as a row
/// for the file.
#[test]
fn sorted_run_entries_skips_non_directory_entries() {
    let root = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir(root.path().join("a_dir")).expect("mkdir");
    std::fs::write(root.path().join("a_file"), b"not a run dir").expect("write file");

    let rows = super::sorted_run_entries(root.path()).expect("sorted_run_entries must succeed");
    let names: Vec<String> = rows
        .iter()
        .map(|(p, _, _, _)| {
            p.file_name()
                .expect("path must have a file_name")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        names,
        vec!["a_dir".to_string()],
        "only the subdirectory must be returned; file entries are skipped",
    );
}

/// `sorted_run_entries` skips dotfile-prefixed subdirectories.
/// Pins the filter that excludes the flock sentinel
/// subdirectory `.locks/` from `cargo ktstr stats list` —
/// a regression that dropped the dotfile filter would
/// surface here as a `.locks` row in the listing, polluting
/// the operator-facing run table with internal coordination
/// state. Other dotfile directories (`.git`, `.cache`, etc.)
/// are filtered uniformly by the same predicate so the test
/// uses two different dotfile names to pin the rule rather
/// than the specific `.locks` instance.
#[test]
fn sorted_run_entries_skips_dotfile_subdirectories() {
    let root = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir(root.path().join("real-run")).expect("mkdir");
    std::fs::create_dir(root.path().join(".locks")).expect("mkdir .locks");
    std::fs::create_dir(root.path().join(".cache")).expect("mkdir .cache");

    let rows = super::sorted_run_entries(root.path()).expect("sorted_run_entries must succeed");
    let names: Vec<String> = rows
        .iter()
        .map(|(p, _, _, _)| {
            p.file_name()
                .expect("path must have a file_name")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        names,
        vec!["real-run".to_string()],
        "dotfile-prefixed subdirs (.locks, .cache) must be filtered \
             out of the run listing; only `real-run` may surface",
    );
}

/// `sorted_run_entries` extracts the arch from the first
/// sidecar that carries `host.arch`. Pins the arch-extraction
/// contract so a regression that drops the field, scans the
/// wrong option leg, or stops short on a host-None sidecar
/// surfaces here.
#[test]
fn sorted_run_entries_extracts_arch_from_first_sidecar() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let run_dir = root.path().join("run-with-arch");
    std::fs::create_dir(&run_dir).expect("mkdir run dir");
    // First sidecar: host populated → arch surfaces.
    let mut sc = crate::test_support::SidecarResult::test_fixture();
    sc.host = Some(crate::host_context::HostContext::test_fixture());
    std::fs::write(
        run_dir.join("t-0000000000000000.ktstr.json"),
        serde_json::to_string(&sc).expect("serialize fixture"),
    )
    .expect("write sidecar");

    let rows = super::sorted_run_entries(root.path()).expect("sorted_run_entries must succeed");
    assert_eq!(rows.len(), 1, "one run dir must yield one row");
    let (_, _, _, arch) = &rows[0];
    assert_eq!(
        arch.as_deref(),
        Some("x86_64"),
        "arch must come from host.arch on the first sidecar — \
             test_fixture populates `Some(\"x86_64\")`",
    );
}

/// A run with no host-populated sidecars yields `None` for
/// arch. Pins the absent-host fallback so the caller's
/// display-sentinel substitution (the `"-"` cell in
/// `list_runs`) is reached.
#[test]
fn sorted_run_entries_arch_none_when_no_host() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let run_dir = root.path().join("run-no-host");
    std::fs::create_dir(&run_dir).expect("mkdir run dir");
    // SidecarResult::test_fixture defaults host: None.
    let sc = crate::test_support::SidecarResult::test_fixture();
    std::fs::write(
        run_dir.join("t-0000000000000000.ktstr.json"),
        serde_json::to_string(&sc).expect("serialize fixture"),
    )
    .expect("write sidecar");

    let rows = super::sorted_run_entries(root.path()).expect("sorted_run_entries must succeed");
    assert_eq!(rows.len(), 1, "one run dir must yield one row");
    let (_, _, _, arch) = &rows[0];
    assert!(
        arch.is_none(),
        "no host-populated sidecar must yield None arch; got {arch:?}",
    );
}
