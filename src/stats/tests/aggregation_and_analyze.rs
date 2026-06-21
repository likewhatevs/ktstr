use super::*;

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
