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

/// `phase_counter_delta` returns `None` for a phase with FEWER THAN TWO
/// finite samples — a delta is unmeasurable from 0 or 1 points. It returns
/// `Some(0.0)` only for 2+ finite samples that are equal (a REAL measured
/// zero: the counter did not advance). This is the sentinel-free
/// absent-vs-measured-zero contract: a 1-sample phase is "delta
/// unmeasurable" (absent), NOT a phantom 0 that would make a per-phase
/// Counter claim read 0. The renderer's has-data distinction is carried by
/// `PhaseBucket::sample_count` (see `expect_metric`), not by this value.
#[test]
fn phase_counter_delta_under_two_finite_samples_is_unmeasurable() {
    // 1 finite sample (incl. surrounded by NaN) => None: no second point.
    assert_eq!(phase_counter_delta(&[42.0]), None);
    assert_eq!(phase_counter_delta(&[f64::NAN, 42.0, f64::NAN]), None);
    // 0 finite => None (unchanged).
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

/// The IRQ-counter `read_sample` arms fold the per-CPU `PerCpuTimeStats`
/// into a cross-CPU SUM at each freeze. One Sample carrying a two-CPU
/// `per_cpu_time` slice: `total_hardirqs` = 100+200, `total_softirq_net_rx`
/// = 10+20 (the `SOFTIRQ_NET_RX` vector index), `total_irq_time_ns` =
/// 5_000+7_000. Empty `per_cpu_time` -> `None` (loud-absent, distinct from a
/// measured 0). Pins the fold + the softirq index + the empty-slice guard
/// directly, without the bucket machinery.
#[test]
fn irq_read_sample_arms_sum_across_cpus() {
    use crate::monitor::btf_offsets::{NR_SOFTIRQS, SOFTIRQ_NET_RX};
    use crate::monitor::dump::{FailureDumpReport, PerCpuTimeStats};
    use crate::scenario::sample::SampleSeries;
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};

    let net_rx = |n: u64| {
        let mut s = [0u64; NR_SOFTIRQS];
        s[SOFTIRQ_NET_RX] = n;
        s
    };
    let entry = DrainedSnapshotEntry {
        tag: "periodic_000".to_string(),
        report: FailureDumpReport {
            per_cpu_time: vec![
                PerCpuTimeStats {
                    cpu: 0,
                    irqs_sum: 100,
                    softirqs: net_rx(10),
                    cpustat_irq_ns: 5_000,
                    ..Default::default()
                },
                PerCpuTimeStats {
                    cpu: 1,
                    irqs_sum: 200,
                    softirqs: net_rx(20),
                    cpustat_irq_ns: 7_000,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(100),
        boundary_offset_ms: None,
        step_index: Some(1),
    };
    let series = SampleSeries::from_drained_typed(vec![entry], None);
    let sample = series.iter_samples().next().expect("one sample");
    let read = |name: &str| {
        crate::stats::metric_def(name)
            .expect("registered metric")
            .read_sample(&sample)
    };
    assert_eq!(
        read("total_hardirqs"),
        Some(300.0),
        "cross-CPU sum of irqs_sum",
    );
    assert_eq!(
        read("total_softirq_net_rx"),
        Some(30.0),
        "cross-CPU sum of softirqs[SOFTIRQ_NET_RX]",
    );
    assert_eq!(
        read("total_irq_time_ns"),
        Some(12_000.0),
        "cross-CPU sum of cpustat_irq_ns",
    );

    let empty = DrainedSnapshotEntry {
        tag: "periodic_001".to_string(),
        report: FailureDumpReport::default(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(200),
        boundary_offset_ms: None,
        step_index: Some(1),
    };
    let empty_series = SampleSeries::from_drained_typed(vec![empty], None);
    let empty_sample = empty_series.iter_samples().next().expect("one sample");
    assert_eq!(
        crate::stats::metric_def("total_hardirqs")
            .expect("registered metric")
            .read_sample(&empty_sample),
        None,
        "empty per_cpu_time -> None (loud-absent)",
    );
}

/// End-to-end per-phase fold for the IRQ Counter — the CI-runnable pair of
/// the host-gated `irq_metrics_e2e` (no VM). A two-freeze `SampleSeries` with
/// RISING per-CPU `irqs_sum` / NET_RX softirqs, both stamped the same step,
/// builds one bucket carrying the cross-CPU last-minus-first DELTA
/// (`total_hardirqs` = (400+500)-(100+200) = 600; `total_softirq_net_rx` =
/// (40+60)-(10+20) = 70) plus the derived `hardirq_rate` = 600 / 3.0 s = 200
/// (the wall-window co-insertion + `derive_rate_metrics`). The window is
/// [1000, 4000] ms from `elapsed_ms` (no `boundary_offset_ms`), so
/// `total_phase_wall_sec` = 3.0. A SINGLE-freeze phase yields `None` for the
/// counter (a delta is unmeasurable from one point) and therefore no rate —
/// proving the absent-vs-zero contract through the whole `read_sample` ->
/// aggregate -> bucket pipeline.
#[test]
fn irq_counter_folds_per_phase_delta_and_rate() {
    use crate::assert::build_phase_buckets;
    use crate::monitor::btf_offsets::{NR_SOFTIRQS, SOFTIRQ_NET_RX};
    use crate::monitor::dump::{FailureDumpReport, PerCpuTimeStats};
    use crate::scenario::sample::SampleSeries;
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};

    let cpu = |cpu: u32, irqs_sum: u64, net_rx: u64| {
        let mut softirqs = [0u64; NR_SOFTIRQS];
        softirqs[SOFTIRQ_NET_RX] = net_rx;
        PerCpuTimeStats {
            cpu,
            irqs_sum,
            softirqs,
            ..Default::default()
        }
    };
    let freeze =
        |tag: &str, elapsed_ms: u64, c0: PerCpuTimeStats, c1: PerCpuTimeStats| DrainedSnapshotEntry {
            tag: tag.to_string(),
            report: FailureDumpReport {
                per_cpu_time: vec![c0, c1],
                ..Default::default()
            },
            stats: Err(MissingStatsReason::NoSchedulerBinary),
            elapsed_ms: Some(elapsed_ms),
            boundary_offset_ms: None,
            step_index: Some(1),
        };

    // Two freezes 3_000 ms apart in the SAME stamped step -> one bucket.
    let two = SampleSeries::from_drained_typed(
        vec![
            freeze("periodic_000", 1_000, cpu(0, 100, 10), cpu(1, 200, 20)),
            freeze("periodic_001", 4_000, cpu(0, 400, 40), cpu(1, 500, 60)),
        ],
        None,
    );
    let buckets = build_phase_buckets(&two);
    let bucket = buckets
        .iter()
        .find(|b| b.step_index == 1)
        .expect("step-1 bucket present");
    assert_eq!(
        bucket.get("total_hardirqs"),
        Some(600.0),
        "cross-CPU last-minus-first: (400+500)-(100+200)",
    );
    assert_eq!(
        bucket.get("total_softirq_net_rx"),
        Some(70.0),
        "cross-CPU NET_RX delta: (40+60)-(10+20)",
    );
    assert_eq!(
        bucket.get("hardirq_rate"),
        Some(200.0),
        "600 hardirqs / 3.0 s capture window",
    );

    // A single freeze in the phase: the Counter delta is unmeasurable from
    // one point -> None (NOT a phantom 0), so no rate is co-derived either.
    let one = SampleSeries::from_drained_typed(
        vec![freeze("periodic_000", 1_000, cpu(0, 100, 10), cpu(1, 200, 20))],
        None,
    );
    let buckets = build_phase_buckets(&one);
    let bucket = buckets
        .iter()
        .find(|b| b.step_index == 1)
        .expect("step-1 bucket present");
    assert_eq!(
        bucket.get("total_hardirqs"),
        None,
        "single-freeze phase: counter delta unmeasurable -> None",
    );
    assert_eq!(
        bucket.get("hardirq_rate"),
        None,
        "no counter component -> no rate co-insertion",
    );
}

/// Run-level IRQ rate must NOT inflate on a MULTI-phase run (regression guard
/// for the num/den time-base mismatch). The run-level numerator total_hardirqs
/// is the DIRECT whole-run delta (populate_run_ext_metrics: read_sample over
/// ALL freezes); the matching denominator total_phase_wall_sec must be the
/// WHOLE-RUN freeze span (inserted by that same direct path), NOT the
/// Σ-per-phase-capture span (which excludes the cross-capture gap the numerator
/// counts). Two phases capturing [1s,4s] and [6s,9s] with the gap [4s,6s]
/// carrying real (counted) IRQ activity: whole-run Δ = 900-100 = 800; whole-run
/// span = 9-1 = 8 s → hardirq_rate = 100/s. The buggy Σ-phase denominator would
/// be (4-1)+(9-6) = 6 s → 133.3/s (inflated 8/6). Calls BOTH run-level populate
/// fns in eval order to pin the real production path.
#[test]
fn run_level_irq_rate_uses_whole_run_span_not_phase_sum() {
    use crate::assert::{
        build_phase_buckets, populate_run_ext_metrics, populate_run_ext_metrics_from_phases,
    };
    use crate::monitor::dump::{FailureDumpReport, PerCpuTimeStats};
    use crate::scenario::sample::SampleSeries;
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};

    let freeze = |tag: &str, elapsed_ms: u64, step: u16, irqs_sum: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: FailureDumpReport {
            per_cpu_time: vec![PerCpuTimeStats {
                cpu: 0,
                irqs_sum,
                ..Default::default()
            }],
            ..Default::default()
        },
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(elapsed_ms),
        boundary_offset_ms: None,
        step_index: Some(step),
    };
    // Phase 1 captures at 1s,4s; gap 4s..6s (irqs_sum keeps rising — counted in
    // the whole-run delta but in NO phase's within-window delta); phase 2 at
    // 6s,9s. Monotonic irqs_sum across the whole run.
    let series = SampleSeries::from_drained_typed(
        vec![
            freeze("periodic_000", 1_000, 1, 100),
            freeze("periodic_001", 4_000, 1, 400),
            freeze("periodic_002", 6_000, 2, 600),
            freeze("periodic_003", 9_000, 2, 900),
        ],
        None,
    );
    let buckets = build_phase_buckets(&series);
    let mut ext = std::collections::BTreeMap::new();
    // Eval order: direct (whole-run) THEN phase-sum (the latter's wall/count
    // inserts no-op via contains_key once the direct path has filled them).
    populate_run_ext_metrics(&series, &mut ext);
    populate_run_ext_metrics_from_phases(&buckets, &mut ext);

    // Count = whole-run delta (direct), spanning the cross-capture gap.
    assert_eq!(
        ext.get("total_hardirqs").copied(),
        Some(800.0),
        "whole-run delta 900-100, not the Σ-per-phase 300+300=600",
    );
    // Denominator = whole-run freeze span (9-1=8 s), NOT Σ-per-phase (3+3=6 s).
    assert_eq!(
        ext.get("total_phase_wall_sec").copied(),
        Some(8.0),
        "whole-run span 8 s, not the Σ-per-phase 6 s",
    );
    // Rate = 800/8 = 100, NOT the inflated 800/6 = 133.3.
    let rate = ext
        .get("hardirq_rate")
        .copied()
        .expect("hardirq_rate derived at run level");
    assert!(
        (rate - 100.0).abs() < 1e-9,
        "whole-run rate 800/8s=100, not the mismatched 800/6s=133.3; got {rate}",
    );
}

/// The whole-run path SURVIVES the sparse multi-phase case where a
/// Σ-per-phase numerator would VANISH. Two phases with ONE freeze each (2 total,
/// at 1s and 9s): each phase's within-window delta is `None` (a delta needs >=2
/// freezes), so a per-phase-summed numerator would be absent → no run-level
/// rate. The direct whole-run path still measures Δ = 500-100 = 400 over the
/// [1s,9s] = 8 s span → hardirq_rate = 50/s. The rate's floor is >=2 TOTAL
/// freezes (== the count's floor), NOT >=2 per phase.
#[test]
fn run_level_irq_rate_survives_sparse_multi_phase() {
    use crate::assert::{
        build_phase_buckets, populate_run_ext_metrics, populate_run_ext_metrics_from_phases,
    };
    use crate::monitor::dump::{FailureDumpReport, PerCpuTimeStats};
    use crate::scenario::sample::SampleSeries;
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};

    let freeze = |tag: &str, elapsed_ms: u64, step: u16, irqs_sum: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: FailureDumpReport {
            per_cpu_time: vec![PerCpuTimeStats {
                cpu: 0,
                irqs_sum,
                ..Default::default()
            }],
            ..Default::default()
        },
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(elapsed_ms),
        boundary_offset_ms: None,
        step_index: Some(step),
    };
    // One freeze per phase → each per-phase delta is unmeasurable.
    let series = SampleSeries::from_drained_typed(
        vec![
            freeze("periodic_000", 1_000, 1, 100),
            freeze("periodic_001", 9_000, 2, 500),
        ],
        None,
    );
    let buckets = build_phase_buckets(&series);
    for b in &buckets {
        assert_eq!(
            b.get("total_hardirqs"),
            None,
            "single-freeze phase has no measurable per-phase counter delta",
        );
    }
    let mut ext = std::collections::BTreeMap::new();
    populate_run_ext_metrics(&series, &mut ext);
    populate_run_ext_metrics_from_phases(&buckets, &mut ext);
    // The whole-run path still measures Δ=400 over 8 s → 50/s. (A Σ-per-phase
    // numerator would be absent here → no rate at all.)
    assert_eq!(
        ext.get("total_hardirqs").copied(),
        Some(400.0),
        "whole-run delta survives where per-phase deltas vanish",
    );
    assert_eq!(
        ext.get("total_phase_wall_sec").copied(),
        Some(8.0),
        "whole-run span 9-1=8 s",
    );
    let rate = ext
        .get("hardirq_rate")
        .copied()
        .expect("rate derives where a phase-summed numerator would vanish");
    assert!((rate - 50.0).abs() < 1e-9, "400/8s=50; got {rate}");
}

// -- per-CPU IRQ spatial axis (max_cpu_hardirqs + concentration) --

/// Build one periodic freeze with `per_cpu_time` = the given (cpu, irqs_sum)
/// rows, stamped to `step`, anchored at `elapsed_ms`.
fn irq_spatial_freeze(
    tag: &str,
    elapsed_ms: u64,
    step: u16,
    cpus: &[(u32, u64)],
) -> crate::scenario::snapshot::DrainedSnapshotEntry {
    use crate::monitor::dump::{FailureDumpReport, PerCpuTimeStats};
    use crate::scenario::snapshot::MissingStatsReason;
    crate::scenario::snapshot::DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: FailureDumpReport {
            per_cpu_time: cpus
                .iter()
                .map(|&(cpu, irqs_sum)| PerCpuTimeStats {
                    cpu,
                    irqs_sum,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        },
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(elapsed_ms),
        boundary_offset_ms: None,
        step_index: Some(step),
    }
}

/// Build a single (step-1) phase bucket from the freezes and read the per-CPU
/// IRQ spatial metrics: (max_cpu_hardirqs, max_cpu_hardirq_concentration).
fn irq_spatial_bucket(
    freezes: Vec<crate::scenario::snapshot::DrainedSnapshotEntry>,
) -> (Option<f64>, Option<f64>) {
    let series = crate::scenario::sample::SampleSeries::from_drained_typed(freezes, None);
    let buckets = crate::assert::build_phase_buckets(&series);
    let b = buckets
        .into_iter()
        .find(|b| b.step_index == 1)
        .expect("step-1 bucket present");
    (
        b.get("max_cpu_hardirqs"),
        b.get("max_cpu_hardirq_concentration"),
    )
}

/// T1 happy path: two freezes, two CPUs — cpu0 100→200 (Δ100), cpu1 200→500
/// (Δ300). max_cpu_hardirqs = max(100, 300) = 300; concentration = 300 /
/// mean(100, 300) = 300/200 = 1.5.
#[test]
fn max_cpu_hardirqs_happy_path() {
    let (max, conc) = irq_spatial_bucket(vec![
        irq_spatial_freeze("periodic_000", 1_000, 1, &[(0, 100), (1, 200)]),
        irq_spatial_freeze("periodic_001", 4_000, 1, &[(0, 200), (1, 500)]),
    ]);
    assert_eq!(max, Some(300.0), "busiest CPU's delta (cpu1: 500-200)");
    let conc = conc.expect("concentration present with 2 reporting CPUs");
    assert!(
        (conc - 1.5).abs() < 1e-9,
        "300 / mean(100,300)=200 = 1.5; got {conc}",
    );
}

/// T2 (load-bearing): the busiest CPU SHIFTS between freezes — cpu0 hot early,
/// cpu1 hot late. first{cpu0:100,cpu1:50} last{cpu0:150,cpu1:400} → per-CPU
/// deltas d0=50, d1=350 → max=350. This is NOT max(last)=400 − max(first)=100 =
/// 300 (the broken spatial-max-then-delta the design forbids), NOT Σlast−Σfirst.
/// Pins per-CPU-delta-THEN-max.
#[test]
fn max_cpu_hardirqs_uses_per_cpu_delta_not_totals_when_busiest_cpu_shifts() {
    let (max, conc) = irq_spatial_bucket(vec![
        irq_spatial_freeze("periodic_000", 1_000, 1, &[(0, 100), (1, 50)]),
        irq_spatial_freeze("periodic_001", 4_000, 1, &[(0, 150), (1, 400)]),
    ]);
    assert_eq!(
        max,
        Some(350.0),
        "max of per-CPU deltas (d1=350), NOT max(totals)=400-100=300",
    );
    let conc = conc.expect("concentration present");
    assert!(
        (conc - 1.75).abs() < 1e-9,
        "350 / mean(50,350)=200 = 1.75; got {conc}",
    );
}

/// T3: endpoints chosen by win() (boundary_offset/elapsed), NOT positional. The
/// POSITIONAL-first freeze has a LATER win (4000) than the positional-last
/// (1000); by_stamped_phase preserves drain order, so samples_in_phase is
/// [win4000, win1000] (non-win-ordered). The fold must use the win-ordered
/// endpoints (first=win1000, last=win4000) → positive delta. A naive
/// .first()/.last() would compute win1000 − win4000 per CPU → saturating_sub
/// clamps to 0 → max=0.
#[test]
fn max_cpu_hardirqs_uses_win_ordered_endpoints_not_positional() {
    let (max, _) = irq_spatial_bucket(vec![
        irq_spatial_freeze("periodic_000", 4_000, 1, &[(0, 500), (1, 300)]),
        irq_spatial_freeze("periodic_001", 1_000, 1, &[(0, 100), (1, 50)]),
    ]);
    assert_eq!(
        max,
        Some(400.0),
        "win-ordered cpu0 500-100=400 (max); positional .first()/.last() would clamp to 0",
    );
}

/// T4 hotplug-skip: cpu1 present in the first freeze but ABSENT from the last
/// (defensive intersection) → excluded from max+mean; only cpu0 (in both)
/// contributes Δ=300-100=200. With a 1-CPU intersection the concentration is
/// omitted.
#[test]
fn max_cpu_hardirqs_skips_cpu_absent_from_an_endpoint() {
    let (max, conc) = irq_spatial_bucket(vec![
        irq_spatial_freeze("periodic_000", 1_000, 1, &[(0, 100), (1, 9999)]),
        irq_spatial_freeze("periodic_001", 4_000, 1, &[(0, 300)]),
    ]);
    assert_eq!(max, Some(200.0), "only cpu0 (present in both freezes) contributes");
    assert_eq!(conc, None, "1-CPU intersection → concentration omitted");
}

/// T5 <2-freeze → both absent: a single freeze yields no measurable per-CPU
/// delta.
#[test]
fn max_cpu_hardirqs_none_for_single_freeze() {
    let (max, conc) = irq_spatial_bucket(vec![irq_spatial_freeze(
        "periodic_000",
        1_000,
        1,
        &[(0, 100), (1, 200)],
    )]);
    assert_eq!(max, None, "single freeze → no per-CPU delta measurable");
    assert_eq!(conc, None, "single freeze → no concentration");
}

/// T6 <2-CPU → concentration omitted, max kept: a single-CPU intersection makes
/// max/mean == 1 a structural artifact (not a measurement), so the concentration
/// key is omitted — but a per-CPU peak IS meaningful on one CPU, so
/// max_cpu_hardirqs is kept.
#[test]
fn max_cpu_hardirq_concentration_omitted_for_single_cpu() {
    let (max, conc) = irq_spatial_bucket(vec![
        irq_spatial_freeze("periodic_000", 1_000, 1, &[(0, 100)]),
        irq_spatial_freeze("periodic_001", 4_000, 1, &[(0, 700)]),
    ]);
    assert_eq!(max, Some(600.0), "cpu0 delta 700-100");
    assert_eq!(conc, None, "1 CPU → concentration omitted (a trivial 1.0)");
}

/// T7 mean==0 → concentration absent, max a measured zero: two CPUs whose
/// counters did not advance → every per-CPU delta is 0, so max_cpu_hardirqs is a
/// REAL Some(0.0), but the concentration is absent (mean==0 would be a NaN). The
/// max=measured-zero vs concentration=loud-absent distinction.
#[test]
fn max_cpu_hardirq_concentration_absent_when_no_irqs() {
    let (max, conc) = irq_spatial_bucket(vec![
        irq_spatial_freeze("periodic_000", 1_000, 1, &[(0, 500), (1, 500)]),
        irq_spatial_freeze("periodic_001", 4_000, 1, &[(0, 500), (1, 500)]),
    ]);
    assert_eq!(max, Some(0.0), "both CPUs' counters didn't advance → measured zero");
    assert_eq!(conc, None, "mean==0 → concentration absent (no div-by-zero/NaN)");
}

/// Run-level: max_cpu_hardirqs auto-folds across phases as a Peak
/// (max-across-phases) through populate_run_ext_metrics_from_phases — NO custom
/// whole-run block (unlike the Counter metrics, a Peak's cross-phase MAX doesn't
/// undercount the way a Counter SUM does, so the whole-run-direct path isn't
/// needed). is_derived(Peak)==false so it isn't skipped, and
/// aggregate_samples_weighted folds Peak by max. Two phases: phase1 busiest-CPU
/// Δ=300, phase2 busiest-CPU Δ=900 → run-level = max(300, 900) = 900. (The
/// direct read_sample path can't touch it — no read_sample arm — so there's no
/// pre-emption.)
#[test]
fn max_cpu_hardirqs_run_level_auto_folds_max_across_phases() {
    use crate::assert::{
        build_phase_buckets, populate_run_ext_metrics, populate_run_ext_metrics_from_phases,
    };
    use crate::scenario::sample::SampleSeries;

    let series = SampleSeries::from_drained_typed(
        vec![
            irq_spatial_freeze("periodic_000", 1_000, 1, &[(0, 100), (1, 200)]),
            irq_spatial_freeze("periodic_001", 4_000, 1, &[(0, 200), (1, 500)]),
            irq_spatial_freeze("periodic_002", 6_000, 2, &[(0, 1000), (1, 1000)]),
            irq_spatial_freeze("periodic_003", 9_000, 2, &[(0, 1100), (1, 1900)]),
        ],
        None,
    );
    let buckets = build_phase_buckets(&series);
    let mut ext = std::collections::BTreeMap::new();
    populate_run_ext_metrics(&series, &mut ext);
    populate_run_ext_metrics_from_phases(&buckets, &mut ext);
    assert_eq!(
        ext.get("max_cpu_hardirqs").copied(),
        Some(900.0),
        "Peak auto-folds max-across-phases: max(phase1 Δ=300, phase2 Δ=900)",
    );
    // max_cpu_hardirq_concentration (also Peak, no read_sample arm) rides the
    // IDENTICAL auto-fold path: phase1 = 300/mean(100,300) = 1.5, phase2 =
    // 900/mean(100,900) = 1.8 → run-level max(1.5, 1.8) = 1.8.
    let conc = ext
        .get("max_cpu_hardirq_concentration")
        .copied()
        .expect("concentration auto-folds to run-level too (also Peak)");
    assert!(
        (conc - 1.8).abs() < 1e-9,
        "concentration auto-folds max-across-phases: max(1.5, 1.8) = 1.8; got {conc}",
    );
}

// -- per-CPU NET_RX softirq spatial axis (max_cpu_softirq_net_rx +
// concentration) — the softirq sibling of the hardirq axis above. Both run
// through the SAME fold_per_cpu_spatial_max helper, so the shared gates
// (win-ordered endpoints, cpu-field intersection, single-freeze/single-CPU/
// no-activity loud-absent) are pinned by the hardirq tests above; these
// softirq tests pin the softirqs[NET_RX] field accessor + the per-CPU-delta-
// then-max ordering + the run-level Peak auto-fold for the new metric. --

/// Build one periodic freeze with `per_cpu_time` = the given (cpu, net_rx) rows
/// — each CPU's `softirqs[NET_RX]` set to net_rx — stamped to `step`, anchored at
/// `elapsed_ms`. The softirq analog of `irq_spatial_freeze`.
fn softirq_net_rx_freeze(
    tag: &str,
    elapsed_ms: u64,
    step: u16,
    cpus: &[(u32, u64)],
) -> crate::scenario::snapshot::DrainedSnapshotEntry {
    use crate::monitor::btf_offsets::SOFTIRQ_NET_RX;
    use crate::monitor::dump::{FailureDumpReport, PerCpuTimeStats};
    use crate::scenario::snapshot::MissingStatsReason;
    crate::scenario::snapshot::DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: FailureDumpReport {
            per_cpu_time: cpus
                .iter()
                .map(|&(cpu, net_rx)| {
                    let mut c = PerCpuTimeStats {
                        cpu,
                        ..Default::default()
                    };
                    c.softirqs[SOFTIRQ_NET_RX] = net_rx;
                    c
                })
                .collect(),
            ..Default::default()
        },
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(elapsed_ms),
        boundary_offset_ms: None,
        step_index: Some(step),
    }
}

/// Build a single (step-1) phase bucket from the freezes and read the per-CPU
/// NET_RX softirq spatial metrics. The softirq analog of `irq_spatial_bucket`.
fn softirq_net_rx_bucket(
    freezes: Vec<crate::scenario::snapshot::DrainedSnapshotEntry>,
) -> (Option<f64>, Option<f64>) {
    let series = crate::scenario::sample::SampleSeries::from_drained_typed(freezes, None);
    let buckets = crate::assert::build_phase_buckets(&series);
    let b = buckets
        .into_iter()
        .find(|b| b.step_index == 1)
        .expect("step-1 bucket present");
    (
        b.get("max_cpu_softirq_net_rx"),
        b.get("max_cpu_softirq_net_rx_concentration"),
    )
}

/// Softirq happy path (mirrors `max_cpu_hardirqs_happy_path` with the NET_RX
/// field): two freezes, two CPUs — cpu0 100→200 (Δ100), cpu1 200→500 (Δ300).
/// max_cpu_softirq_net_rx = max(100, 300) = 300; concentration = 300 /
/// mean(100, 300) = 1.5. Pins the `softirqs[NET_RX]` field accessor.
#[test]
fn max_cpu_softirq_net_rx_happy_path() {
    let (max, conc) = softirq_net_rx_bucket(vec![
        softirq_net_rx_freeze("periodic_000", 1_000, 1, &[(0, 100), (1, 200)]),
        softirq_net_rx_freeze("periodic_001", 4_000, 1, &[(0, 200), (1, 500)]),
    ]);
    assert_eq!(
        max,
        Some(300.0),
        "busiest CPU's NET_RX softirq delta (cpu1: 500-200)",
    );
    let conc = conc.expect("concentration present with 2 reporting CPUs");
    assert!(
        (conc - 1.5).abs() < 1e-9,
        "300 / mean(100,300)=200 = 1.5; got {conc}",
    );
}

/// The busiest CPU SHIFTS between freezes (the load-bearing per-CPU-delta-then-
/// max case): cpu0 hot early, cpu1 hot late. first{cpu0:100,cpu1:50}
/// last{cpu0:150,cpu1:400} → per-CPU deltas d0=50, d1=350 → max=350, NOT
/// max(last)=400 − max(first)=100 = 300. Pins per-CPU-delta-THEN-max through the
/// softirqs[NET_RX] accessor (the helper the hardirq tests pin).
#[test]
fn max_cpu_softirq_net_rx_uses_per_cpu_delta_when_busiest_cpu_shifts() {
    let (max, conc) = softirq_net_rx_bucket(vec![
        softirq_net_rx_freeze("periodic_000", 1_000, 1, &[(0, 100), (1, 50)]),
        softirq_net_rx_freeze("periodic_001", 4_000, 1, &[(0, 150), (1, 400)]),
    ]);
    assert_eq!(
        max,
        Some(350.0),
        "max of per-CPU NET_RX deltas (d1=350), NOT max(totals)=400-100=300",
    );
    let conc = conc.expect("concentration present");
    assert!(
        (conc - 1.75).abs() < 1e-9,
        "350 / mean(50,350)=200 = 1.75; got {conc}",
    );
}

/// Run-level: max_cpu_softirq_net_rx auto-folds across phases as a Peak
/// (max-across-phases) via populate_run_ext_metrics_from_phases — same path as
/// max_cpu_hardirqs (Peak, no read_sample arm). phase1 busiest Δ=300, phase2
/// busiest Δ=900 → run-level max(300, 900) = 900. Pins the new MetricDef's Peak
/// kind driving the auto-fold (the registry wiring).
#[test]
fn max_cpu_softirq_net_rx_run_level_auto_folds_max_across_phases() {
    use crate::assert::{
        build_phase_buckets, populate_run_ext_metrics, populate_run_ext_metrics_from_phases,
    };
    use crate::scenario::sample::SampleSeries;

    let series = SampleSeries::from_drained_typed(
        vec![
            softirq_net_rx_freeze("periodic_000", 1_000, 1, &[(0, 100), (1, 200)]),
            softirq_net_rx_freeze("periodic_001", 4_000, 1, &[(0, 200), (1, 500)]),
            softirq_net_rx_freeze("periodic_002", 6_000, 2, &[(0, 1000), (1, 1000)]),
            softirq_net_rx_freeze("periodic_003", 9_000, 2, &[(0, 1100), (1, 1900)]),
        ],
        None,
    );
    let buckets = build_phase_buckets(&series);
    let mut ext = std::collections::BTreeMap::new();
    populate_run_ext_metrics(&series, &mut ext);
    populate_run_ext_metrics_from_phases(&buckets, &mut ext);
    assert_eq!(
        ext.get("max_cpu_softirq_net_rx").copied(),
        Some(900.0),
        "Peak auto-folds max-across-phases: max(phase1 Δ=300, phase2 Δ=900)",
    );
    let conc = ext
        .get("max_cpu_softirq_net_rx_concentration")
        .copied()
        .expect("concentration auto-folds to run-level too (also Peak)");
    assert!(
        (conc - 1.8).abs() < 1e-9,
        "concentration auto-folds max-across-phases: max(1.5, 1.8) = 1.8; got {conc}",
    );
}

// -- per-cgroup PSI-irq spatial axis (max_cgroup_irq_pressure + concentration +
// max_cgroup_psi_irq_avg10) — the cgroup sibling of the per-CPU axes above. The
// two COUNTER metrics (irq_pressure + concentration) mirror the
// fold_per_cpu_spatial_max gates (win-ordered endpoints, entity intersection,
// single-freeze/single-entity/no-activity loud-absent) with `cgroup_kva` as the
// per-CPU `cpu`-field analog and a /1000 ns→µs decode on each delta. The GAUGE
// metric (avg10) diverges: an instantaneous reading, so it is present on a SINGLE
// freeze (no delta/≥2 requirement) and folds a per-freeze spatial-max then a
// cross-freeze temporal-max. --

/// Build one periodic freeze with `cgroup_psi` = the given
/// (cgroup_kva, total_ns, avg10_raw) leaf rows, stamped to `step`, anchored at
/// `elapsed_ms`. Each leaf's serial_nr defaults to its cgroup_kva — a unique,
/// stable-across-freezes identity for the no-KVA-reuse case; the KVA-reuse case
/// (a distinct serial at the same KVA) uses `cgroup_psi_freeze_with_serial`. The
/// cgroup analog of `irq_spatial_freeze`.
fn cgroup_psi_freeze(
    tag: &str,
    elapsed_ms: u64,
    step: u16,
    leaves: &[(u64, u64, u64)],
) -> crate::scenario::snapshot::DrainedSnapshotEntry {
    use crate::monitor::cgroup_walk::CgroupPsiStat;
    use crate::monitor::dump::FailureDumpReport;
    use crate::scenario::snapshot::MissingStatsReason;
    crate::scenario::snapshot::DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: FailureDumpReport {
            cgroup_psi: leaves
                .iter()
                .map(|&(cgroup_kva, total_ns, avg10_raw)| CgroupPsiStat {
                    cgroup_kva,
                    total_ns,
                    avg10_raw,
                    // Default serial = the KVA: a unique identity that is stable
                    // across freezes for the same leaf (the no-reuse case). The
                    // reuse case sets a distinct serial via the _with_serial helper.
                    serial_nr: cgroup_kva,
                })
                .collect(),
            ..Default::default()
        },
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(elapsed_ms),
        boundary_offset_ms: None,
        step_index: Some(step),
    }
}

/// Build a freeze with EXPLICIT per-leaf serial_nr — the KVA-reuse case. Leaf
/// rows are (cgroup_kva, serial_nr, total_ns, avg10_raw). Used to model a leaf
/// whose freed slab KVA a new cgroup reused (same kva, different serial across
/// freezes), which `cgroup_psi_freeze`'s serial==kva default cannot express.
fn cgroup_psi_freeze_with_serial(
    tag: &str,
    elapsed_ms: u64,
    step: u16,
    leaves: &[(u64, u64, u64, u64)],
) -> crate::scenario::snapshot::DrainedSnapshotEntry {
    use crate::monitor::cgroup_walk::CgroupPsiStat;
    use crate::monitor::dump::FailureDumpReport;
    use crate::scenario::snapshot::MissingStatsReason;
    crate::scenario::snapshot::DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: FailureDumpReport {
            cgroup_psi: leaves
                .iter()
                .map(|&(cgroup_kva, serial_nr, total_ns, avg10_raw)| CgroupPsiStat {
                    cgroup_kva,
                    total_ns,
                    avg10_raw,
                    serial_nr,
                })
                .collect(),
            ..Default::default()
        },
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(elapsed_ms),
        boundary_offset_ms: None,
        step_index: Some(step),
    }
}

/// Build a single (step-1) phase bucket from the freezes and read the per-cgroup
/// PSI-irq spatial metrics:
/// (max_cgroup_irq_pressure, max_cgroup_irq_pressure_concentration,
/// max_cgroup_psi_irq_avg10). The cgroup analog of `irq_spatial_bucket`.
fn cgroup_psi_bucket(
    freezes: Vec<crate::scenario::snapshot::DrainedSnapshotEntry>,
) -> (Option<f64>, Option<f64>, Option<f64>) {
    let series = crate::scenario::sample::SampleSeries::from_drained_typed(freezes, None);
    let buckets = crate::assert::build_phase_buckets(&series);
    let b = buckets
        .into_iter()
        .find(|b| b.step_index == 1)
        .expect("step-1 bucket present");
    (
        b.get("max_cgroup_irq_pressure"),
        b.get("max_cgroup_irq_pressure_concentration"),
        b.get("max_cgroup_psi_irq_avg10"),
    )
}

/// Happy path: two freezes, two leaf cgroups. A total 100_000→200_000 (Δ100_000
/// ns = 100µs), B 200_000→500_000 (Δ300_000 ns = 300µs). max_cgroup_irq_pressure
/// = max(100, 300) = 300µs; concentration = 300 / mean(100, 300) = 1.5. avg10 =
/// max over freezes of max-across-leaves: freeze1 max(1%,2%)=2, freeze2
/// max(3%,4%)=4 → 4%. Pins the ns→µs decode and the fixed-point avg10 decode.
#[test]
fn max_cgroup_irq_pressure_happy_path() {
    let (max, conc, avg10) = cgroup_psi_bucket(vec![
        cgroup_psi_freeze("periodic_000", 1_000, 1, &[(0xA00, 100_000, 2048), (0xB00, 200_000, 4096)]),
        cgroup_psi_freeze("periodic_001", 4_000, 1, &[(0xA00, 200_000, 6144), (0xB00, 500_000, 8192)]),
    ]);
    assert_eq!(max, Some(300.0), "busiest leaf's delta decoded to µs (B: (500_000-200_000)/1000)");
    let conc = conc.expect("concentration present with 2 reporting leaves");
    assert!((conc - 1.5).abs() < 1e-9, "300 / mean(100,300)=200 = 1.5; got {conc}");
    let avg10 = avg10.expect("avg10 gauge present");
    assert!((avg10 - 4.0).abs() < 1e-9, "max-across-freezes of max-across-leaves: max(2%,4%)=4; got {avg10}");
}

/// Load-bearing: the busiest leaf SHIFTS between freezes — A hot early, B hot
/// late. first{A:100_000,B:50_000} last{A:150_000,B:400_000} → per-leaf deltas
/// dA=50µs, dB=350µs → max=350µs. This is NOT max(last)=400_000 −
/// max(first)=100_000 = 300µs (spatial-max-then-delta the design forbids). Pins
/// per-leaf-delta-THEN-max on the cgroup axis.
#[test]
fn max_cgroup_irq_pressure_uses_per_leaf_delta_when_busiest_shifts() {
    let (max, conc, _) = cgroup_psi_bucket(vec![
        cgroup_psi_freeze("periodic_000", 1_000, 1, &[(0xA00, 100_000, 0), (0xB00, 50_000, 0)]),
        cgroup_psi_freeze("periodic_001", 4_000, 1, &[(0xA00, 150_000, 0), (0xB00, 400_000, 0)]),
    ]);
    assert_eq!(max, Some(350.0), "max of per-leaf deltas (dB=350µs), NOT max(totals)=400-100=300µs");
    let conc = conc.expect("concentration present");
    assert!((conc - 1.75).abs() < 1e-9, "350 / mean(50,350)=200 = 1.75; got {conc}");
}

/// Endpoints chosen by win() (boundary_offset/elapsed), NOT positional. The
/// POSITIONAL-first freeze has a LATER win (4000) than the positional-last
/// (1000); a naive .first()/.last() would compute win1000 − win4000 per leaf →
/// saturating_sub clamps to 0 → max=0. The win-ordered fold yields a positive
/// delta. (Mirrors the per-CPU win-ordering test on the cgroup axis.)
#[test]
fn max_cgroup_irq_pressure_uses_win_ordered_endpoints_not_positional() {
    let (max, _, _) = cgroup_psi_bucket(vec![
        cgroup_psi_freeze("periodic_000", 4_000, 1, &[(0xA00, 500_000, 0), (0xB00, 300_000, 0)]),
        cgroup_psi_freeze("periodic_001", 1_000, 1, &[(0xA00, 100_000, 0), (0xB00, 50_000, 0)]),
    ]);
    assert_eq!(
        max,
        Some(400.0),
        "win-ordered A (500_000-100_000)/1000=400µs (max); positional would clamp to 0",
    );
}

/// Leaf-churn skip: B present in the first freeze but ABSENT from the last
/// (defensive cgroup_kva intersection) → excluded from max+mean; only A (in
/// both) contributes Δ=(300_000-100_000)/1000=200µs. With a 1-leaf intersection
/// the concentration is omitted. A leaf created/destroyed mid-phase.
#[test]
fn max_cgroup_irq_pressure_skips_leaf_absent_from_an_endpoint() {
    let (max, conc, _) = cgroup_psi_bucket(vec![
        cgroup_psi_freeze("periodic_000", 1_000, 1, &[(0xA00, 100_000, 0), (0xB00, 9_999_000, 0)]),
        cgroup_psi_freeze("periodic_001", 4_000, 1, &[(0xA00, 300_000, 0)]),
    ]);
    assert_eq!(max, Some(200.0), "only A (present in both freezes) contributes");
    assert_eq!(conc, None, "1-leaf intersection → concentration omitted");
}

/// The GAUGE/COUNTER divergence: a SINGLE freeze yields no measurable delta, so
/// max_cgroup_irq_pressure + concentration are absent — but max_cgroup_psi_irq_avg10
/// is an instantaneous reading, present on one freeze: max(1%,2%)=2%. This is the
/// per-cgroup gauge's key difference from the per-CPU counter axes (which are all
/// absent on a single freeze).
#[test]
fn max_cgroup_psi_irq_avg10_present_on_single_freeze_while_delta_absent() {
    let (max, conc, avg10) = cgroup_psi_bucket(vec![cgroup_psi_freeze(
        "periodic_000",
        1_000,
        1,
        &[(0xA00, 100_000, 2048), (0xB00, 200_000, 4096)],
    )]);
    assert_eq!(max, None, "single freeze → no per-leaf delta measurable");
    assert_eq!(conc, None, "single freeze → no concentration");
    let avg10 = avg10.expect("avg10 is an instantaneous gauge, present on one freeze");
    assert!((avg10 - 2.0).abs() < 1e-9, "max-across-leaves of the lone freeze: max(1%,2%)=2; got {avg10}");
}

/// <2-leaf intersection → concentration omitted, max kept: a single-leaf
/// intersection makes max/mean == 1 a structural artifact, so the concentration
/// key is omitted — but a per-leaf peak IS meaningful on one leaf, so
/// max_cgroup_irq_pressure is kept (A: (700_000-100_000)/1000=600µs).
#[test]
fn max_cgroup_irq_pressure_concentration_omitted_for_single_leaf() {
    let (max, conc, _) = cgroup_psi_bucket(vec![
        cgroup_psi_freeze("periodic_000", 1_000, 1, &[(0xA00, 100_000, 0)]),
        cgroup_psi_freeze("periodic_001", 4_000, 1, &[(0xA00, 700_000, 0)]),
    ]);
    assert_eq!(max, Some(600.0), "A delta (700_000-100_000)/1000");
    assert_eq!(conc, None, "1 leaf → concentration omitted (a trivial 1.0)");
}

/// mean==0 → concentration absent, max a measured zero: two leaves whose counters
/// did not advance → every per-leaf delta is 0, so max_cgroup_irq_pressure is a
/// REAL Some(0.0), but the concentration is absent (mean==0 would be a NaN). The
/// max=measured-zero vs concentration=loud-absent distinction on the cgroup axis.
#[test]
fn max_cgroup_irq_pressure_concentration_absent_when_no_pressure() {
    let (max, conc, _) = cgroup_psi_bucket(vec![
        cgroup_psi_freeze("periodic_000", 1_000, 1, &[(0xA00, 500_000, 0), (0xB00, 500_000, 0)]),
        cgroup_psi_freeze("periodic_001", 4_000, 1, &[(0xA00, 500_000, 0), (0xB00, 500_000, 0)]),
    ]);
    assert_eq!(max, Some(0.0), "both leaves' counters didn't advance → measured zero");
    assert_eq!(conc, None, "mean==0 → concentration absent (no div-by-zero/NaN)");
}

/// No reporting leaf in any freeze (psi_cgroups off / absent workload root → the
/// walk captured nothing) → all three metrics loud-absent. The cgroup_psi gauge
/// has no leaf to spatial-max, and there is no delta to fold.
#[test]
fn cgroup_psi_metrics_all_absent_when_no_leaf_reported() {
    let (max, conc, avg10) = cgroup_psi_bucket(vec![
        cgroup_psi_freeze("periodic_000", 1_000, 1, &[]),
        cgroup_psi_freeze("periodic_001", 4_000, 1, &[]),
    ]);
    assert_eq!(max, None, "no leaf → no pressure delta");
    assert_eq!(conc, None, "no leaf → no concentration");
    assert_eq!(avg10, None, "no leaf → no avg10 gauge");
}

/// Run-level: all three cgroup metrics are Peak with no read_sample arm, so they
/// auto-fold max-across-phases through populate_run_ext_metrics_from_phases (the
/// max_cpu_hardirqs path). Two phases: phase1 busiest Δ=300µs / conc 1.5 / avg10
/// 2%, phase2 busiest Δ=900µs / conc 1.8 / avg10 6% → run-level max(300,900)=900,
/// max(1.5,1.8)=1.8, max(2,6)=6. Pins the three new MetricDefs' Peak kind driving
/// the registry auto-fold (the gauge included).
#[test]
fn cgroup_psi_metrics_run_level_auto_fold_max_across_phases() {
    use crate::assert::{
        build_phase_buckets, populate_run_ext_metrics, populate_run_ext_metrics_from_phases,
    };
    use crate::scenario::sample::SampleSeries;

    let series = SampleSeries::from_drained_typed(
        vec![
            cgroup_psi_freeze("periodic_000", 1_000, 1, &[(0xA00, 100_000, 2048), (0xB00, 200_000, 4096)]),
            cgroup_psi_freeze("periodic_001", 4_000, 1, &[(0xA00, 200_000, 2048), (0xB00, 500_000, 4096)]),
            cgroup_psi_freeze("periodic_002", 6_000, 2, &[(0xA00, 1_000_000, 10240), (0xB00, 1_000_000, 12288)]),
            cgroup_psi_freeze("periodic_003", 9_000, 2, &[(0xA00, 1_100_000, 10240), (0xB00, 1_900_000, 12288)]),
        ],
        None,
    );
    let buckets = build_phase_buckets(&series);
    let mut ext = std::collections::BTreeMap::new();
    populate_run_ext_metrics(&series, &mut ext);
    populate_run_ext_metrics_from_phases(&buckets, &mut ext);
    assert_eq!(
        ext.get("max_cgroup_irq_pressure").copied(),
        Some(900.0),
        "Peak auto-folds max-across-phases: max(phase1 Δ=300µs, phase2 Δ=900µs)",
    );
    let conc = ext
        .get("max_cgroup_irq_pressure_concentration")
        .copied()
        .expect("concentration auto-folds run-level too (Peak)");
    assert!(
        (conc - 1.8).abs() < 1e-9,
        "max(phase1 1.5, phase2 900/mean(100,900)=1.8) = 1.8; got {conc}",
    );
    let avg10 = ext
        .get("max_cgroup_psi_irq_avg10")
        .copied()
        .expect("avg10 gauge auto-folds run-level too (Peak)");
    assert!(
        (avg10 - 6.0).abs() < 1e-9,
        "max(phase1 2%, phase2 6%) = 6; got {avg10}",
    );
}

/// KVA-reuse disambiguation by serial_nr: a leaf rmdir'd mid-phase whose freed
/// slab KVA a NEW cgroup reused must NOT be correlated across freezes by
/// cgroup_kva alone. Leaf B (kva 0xB00, serial 2, total 500_000) is replaced in
/// the last freeze by a new cgroup C at the SAME kva 0xB00 with serial 3 and
/// total 9_000_000 (cumulative from C's creation, unrelated to B). With
/// cgroup_kva-only correlation the fold would difference C against B and report a
/// bogus (9_000_000-500_000)/1000 = 8500µs delta. The (cgroup_kva, serial_nr)
/// tuple drops the serial-mismatched 0xB00, so only A (kva 0xA00, serial 1 in
/// both freezes) contributes its (200_000-100_000)/1000 = 100µs delta — max =
/// 100µs (NOT 8500µs), concentration absent (1-leaf serial-matched intersection).
/// This pins the cross-phase identity guard against slab-KVA reuse.
#[test]
fn max_cgroup_irq_pressure_disambiguates_reused_kva_by_serial() {
    let (max, conc, _) = cgroup_psi_bucket(vec![
        cgroup_psi_freeze_with_serial(
            "periodic_000",
            1_000,
            1,
            &[(0xA00, 1, 100_000, 0), (0xB00, 2, 500_000, 0)],
        ),
        cgroup_psi_freeze_with_serial(
            "periodic_001",
            4_000,
            1,
            &[(0xA00, 1, 200_000, 0), (0xB00, 3, 9_000_000, 0)],
        ),
    ]);
    assert_eq!(
        max,
        Some(100.0),
        "only A (kva+serial match in both freezes) contributes 100µs; the reused-KVA \
         0xB00 (serial 2→3) is dropped, NOT differenced into a bogus 8500µs",
    );
    assert_eq!(
        conc, None,
        "1-leaf serial-matched intersection → concentration omitted",
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
    // Derived kinds (every `is_derived()`: Rate / Distribution / WorstLowest /
    // WakeLatencyTailRatio / WorstCrossNodeRatio / PerPhase) are `is_derived`,
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
        MetricKind::WakeLatencyTailRatio,
        MetricKind::WorstCrossNodeRatio,
        MetricKind::PerPhase,
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
        // `populate_run_distribution_metrics`'s lowest-wins fold — the
        // iterations-efficiency selectors via `worst.is_none_or(|w| v < w)` over
        // `stats.cgroups`, `worst_page_locality` via `reduce(f64::min)` over the
        // per-phase NUMA carriers — which treats the LOWEST per-cgroup value as
        // the worst — correct ONLY for HigherBetter metrics.
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
        // WorstCrossNodeRatio re-pools by `populate_run_distribution_metrics`'s
        // MAX-wins fold (`reduce(f64::max)` over the per-phase NUMA carriers'
        // per-cgroup churn ratio), which treats the HIGHEST per-cgroup value as
        // the worst — correct ONLY for LowerBetter metrics (the polarity twin of
        // the WorstLowest gate above). Enforce it so a future HigherBetter
        // WorstCrossNodeRatio cannot silently invert the regression signal.
        if matches!(m.kind, MetricKind::WorstCrossNodeRatio) {
            assert_eq!(
                m.polarity,
                crate::test_support::Polarity::LowerBetter,
                "WorstCrossNodeRatio-kind metric {:?} must be LowerBetter \
                     (the highest-wins fold treats highest as worst); got {:?}",
                m.name,
                m.polarity,
            );
        }
        // Rate metrics are derived ratios; name them `*_rate`, `*_per_*`, or —
        // for a dimensionless pooled fraction/ratio of two counters — `*_fraction`
        // / `*_ratio`, so the registry reads as a rate at a glance. (A pooled
        // fraction like `ttwu_local_fraction` = Σlocal/Σcount IS a Rate: the
        // cross-run fold must be ratio-of-sums, not mean-of-ratios.)
        if let MetricKind::Rate {
            numerator,
            denominator,
        } = m.kind
        {
            assert!(
                m.name.ends_with("_rate")
                    || m.name.contains("_per_")
                    || m.name.ends_with("_fraction")
                    || m.name.ends_with("_ratio"),
                "Rate-kind metric must use *_rate / *_per_* / *_fraction / *_ratio naming, got {:?}",
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
        // taobench_hit_rate is the reference taobench server's own field name
        // (1 - misses/cmds, a command-time hit FRACTION, not a per-second rate);
        // it is MetricKind::PerPhase (skipped at the cross-run ext fold) so the
        // mean-of-ratios bug this gate guards cannot apply.
        if looks_like_rate
            && m.name != "worst_iterations_per_cpu_sec"
            && m.name != "taobench_hit_rate"
        {
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
    r1.stuck_count = 2.0; // > 0, should show stuck=2
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
