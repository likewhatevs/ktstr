use super::*;

// -- population-weighted cross-phase wake re-pool --

/// Unit: `weighted_percentile` reduces byte-identically to `percentile` when
/// every weight is 1.0, across the full p range and several pool shapes. This
/// is the parity floor: a run where no phase exceeded the wake cap (every
/// weight 1.0) must reduce exactly as the prior unweighted path did.
#[test]
fn weighted_percentile_matches_percentile_under_unit_weights() {
    let shapes: Vec<Vec<u64>> = vec![
        vec![42_000],
        vec![1_000, 2_000, 3_000, 4_000, 5_000],
        (1..=100u64).map(|v| v * 1000).collect(),
        vec![5_000, 5_000, 5_000],
    ];
    for values in shapes {
        let pairs: Vec<(u64, f64)> = values.iter().map(|&v| (v, 1.0)).collect();
        for &p in &[0.0_f64, 0.25, 0.5, 0.9, 0.99, 1.0] {
            assert_eq!(
                weighted_percentile(&pairs, p),
                percentile(&values, p),
                "unit-weight weighted_percentile must equal percentile at p={p} for {values:?}",
            );
        }
    }
}

/// Unit: empty input is the documented 0 (the caller never reduces an empty
/// pool — the `.then()` guards it — but the fn is total).
#[test]
fn weighted_percentile_empty_is_zero() {
    assert_eq!(weighted_percentile(&[], 0.99), 0);
}

/// Unit: `reduce_weighted_sorted_distribution` reduces byte-identically to
/// `reduce_sorted_distribution` under unit weights for P99 / Median / Mean /
/// Worst, and within the 1e-9 parity bound for Cv (the only arm that cannot
/// keep the u64 mean sum — a weighted variance must sum in f64).
#[test]
fn reduce_weighted_matches_unweighted_under_unit_weights() {
    let values: Vec<u64> = vec![1_000, 2_000, 3_000, 4_000, 5_000, 60_000];
    let pairs: Vec<(u64, f64)> = values.iter().map(|&v| (v, 1.0)).collect();
    for r in [
        SampleReduction::P99,
        SampleReduction::Median,
        SampleReduction::Mean,
        SampleReduction::Worst,
    ] {
        assert_eq!(
            reduce_weighted_sorted_distribution(&pairs, r),
            reduce_sorted_distribution(&values, r),
            "{r:?} must be byte-identical under unit weights",
        );
    }
    let wcv = reduce_weighted_sorted_distribution(&pairs, SampleReduction::Cv);
    let ucv = reduce_sorted_distribution(&values, SampleReduction::Cv);
    assert!(
        (wcv - ucv).abs() < 1e-9,
        "Cv parity within 1e-9: weighted {wcv} vs unweighted {ucv}",
    );
}

/// Unit: weights actually move the reduction. A two-value pool where the HIGH
/// value carries weight 20 and the LOW value weight 1 has weighted
/// median == high (the high value owns 20/21 of the population), while the
/// unit-weight median is the low value (nearest-rank rank 1 of 2). Proves the
/// weight is consumed, not ignored.
#[test]
fn weighted_reduction_shifts_with_population_weight() {
    // total weight 21; median target = ceil(21*0.5) = 11. Cumulative crosses
    // 11 only at the high (weight-20) sample → weighted median = 100 µs.
    let weighted = vec![(10_000u64, 1.0_f64), (100_000u64, 20.0)];
    assert_eq!(
        reduce_weighted_sorted_distribution(&weighted, SampleReduction::Median),
        100.0,
        "population-weighted median is the high value (20/21 of the population)",
    );
    // Unit weights (a length-weighted pool): median target = ceil(2*0.5)
    // = 1 → first (low) value = 10 µs. The two differ — the weight is load-bearing.
    let unit = vec![(10_000u64, 1.0_f64), (100_000u64, 1.0)];
    assert_eq!(
        reduce_weighted_sorted_distribution(&unit, SampleReduction::Median),
        10.0,
        "unit-weight median is the low value — the contrast that proves weighting",
    );
    // Weighted mean = (10_000*1 + 100_000*20)/21/1000 µs.
    let wmean = reduce_weighted_sorted_distribution(&weighted, SampleReduction::Mean);
    assert!((wmean - (10_000.0 + 100_000.0 * 20.0) / 21.0 / 1000.0).abs() < 1e-9);
}

/// THE THESIS (integration through `populate_run_distribution_metrics`): a
/// single cgroup wakes in two phases — phase0 ≤cap (1 sample, true population 1,
/// weight 1.0) and phase1 >cap (reservoir holds 1 sample but true population 20,
/// weight 20.0). The run-level median must reflect the TRUE 20:1 population
/// (high phase dominates → 100 µs), NOT the 1:1 reservoir lengths a length-
/// weighted concat would use (which would give the low phase's 10 µs). The
/// cap-induced cross-phase length skew is removed.
#[test]
fn repool_wake_weights_phases_by_true_population_not_reservoir_length() {
    let phase0 = PhaseCgroupStats {
        wake_latencies_ns: vec![10_000], // 10 µs, ≤cap
        wake_sample_total: 1,            // weight 1.0
        ..PhaseCgroupStats::default()
    };
    let phase1 = PhaseCgroupStats {
        wake_latencies_ns: vec![100_000], // 100 µs, reservoir-capped to 1 sample
        wake_sample_total: 20,            // true population 20 → weight 20.0
        ..PhaseCgroupStats::default()
    };
    let mut b0 = PhaseBucket {
        step_index: 0,
        ..PhaseBucket::default()
    };
    b0.per_cgroup.insert("a".to_string(), phase0);
    let mut b1 = PhaseBucket {
        step_index: 1,
        ..PhaseBucket::default()
    };
    b1.per_cgroup.insert("a".to_string(), phase1);
    let mut stats = ScenarioStats {
        phases: vec![b0, b1],
        ..ScenarioStats::default()
    };
    populate_run_distribution_metrics(&mut stats);

    let median = stats
        .ext_metrics
        .get("worst_median_wake_latency_us")
        .copied();
    assert_eq!(
        median,
        Some(100.0),
        "population-weighted cross-phase median follows the 20:1 true population \
         (high phase), got {median:?}",
    );
    assert_ne!(
        median,
        Some(10.0),
        "must NOT be the reservoir-length-weighted (1:1) median — that is the \
         cross-phase length skew population-weighting removes",
    );
}

/// Boundary: a fully ≤cap multi-phase run (every weight 1.0) reduces IDENTICALLY
/// whether weighted or not — the population-weighting is invisible below the
/// cap. Two phases, 50 samples each, all wake_sample_total == len, must give the
/// exact union percentile the unweighted concat produced.
#[test]
fn repool_wake_under_cap_is_byte_identical_to_unweighted_union() {
    let phase0 = PhaseCgroupStats {
        wake_latencies_ns: (1..=50u64).map(|v| v * 1000).collect(),
        wake_sample_total: 50, // == len → weight 1.0
        ..PhaseCgroupStats::default()
    };
    let phase1 = PhaseCgroupStats {
        wake_latencies_ns: (51..=100u64).map(|v| v * 1000).collect(),
        wake_sample_total: 50, // == len → weight 1.0
        ..PhaseCgroupStats::default()
    };
    let mut b0 = PhaseBucket {
        step_index: 0,
        ..PhaseBucket::default()
    };
    b0.per_cgroup.insert("a".to_string(), phase0);
    let mut b1 = PhaseBucket {
        step_index: 1,
        ..PhaseBucket::default()
    };
    b1.per_cgroup.insert("a".to_string(), phase1);
    let mut stats = ScenarioStats {
        phases: vec![b0, b1],
        ..ScenarioStats::default()
    };
    populate_run_distribution_metrics(&mut stats);

    let union: Vec<u64> = (1..=100u64).map(|v| v * 1000).collect();
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(percentile(&union, 0.99) as f64 / 1000.0),
        "≤cap (unit-weight) p99 must equal the unweighted union percentile",
    );
    assert_eq!(
        stats
            .ext_metrics
            .get("worst_median_wake_latency_us")
            .copied(),
        Some(percentile(&union, 0.5) as f64 / 1000.0),
        "≤cap (unit-weight) median must equal the unweighted union percentile",
    );
}

/// The run-level weighted CV at >cap (through `populate_run_distribution_metrics`)
/// divides variance/mean by Σ weights (the reconstructed true population), NOT
/// pool length. Two phases for one cgroup: phase0 1 sample @ 10 µs (true pop 1,
/// weight 1.0), phase1 1 reservoir-capped sample @ 30 µs (true pop 3, weight 3.0).
/// Pool [(10µs,1),(30µs,3)], Σw=4: weighted mean 25 µs, weighted variance
/// (1·15² + 3·5²)·1e6/4 = 75e6 ns², CV = √75e6/25000 = 0.346410. The
/// length-weighted (pre-change) CV over [10,30] with n=2 would be 0.5 — so this
/// pins the population-weighted denominator, the one wake reduction that visibly
/// shifts at >cap besides the percentiles.
#[test]
fn repool_wake_cv_is_population_weighted_at_over_cap() {
    let phase0 = PhaseCgroupStats {
        wake_latencies_ns: vec![10_000],
        wake_sample_total: 1,
        ..PhaseCgroupStats::default()
    };
    let phase1 = PhaseCgroupStats {
        wake_latencies_ns: vec![30_000], // reservoir-capped to 1 sample
        wake_sample_total: 3,            // true population 3 → weight 3.0
        ..PhaseCgroupStats::default()
    };
    let mut b0 = PhaseBucket {
        step_index: 0,
        ..PhaseBucket::default()
    };
    b0.per_cgroup.insert("a".to_string(), phase0);
    let mut b1 = PhaseBucket {
        step_index: 1,
        ..PhaseBucket::default()
    };
    b1.per_cgroup.insert("a".to_string(), phase1);
    let mut stats = ScenarioStats {
        phases: vec![b0, b1],
        ..ScenarioStats::default()
    };
    populate_run_distribution_metrics(&mut stats);

    let cv = stats
        .ext_metrics
        .get("worst_wake_latency_cv")
        .copied()
        .unwrap();
    assert!(
        (cv - 0.346_410_161_513_775_46).abs() < 1e-9,
        "population-weighted CV = √75e6/25000 = 0.346410, got {cv}",
    );
    assert!(
        (cv - 0.5).abs() > 0.1,
        "must NOT be the unweighted (length n=2) CV 0.5 — pins the weighted denominator",
    );
}

/// Weighted CV guard: an all-equal-value pool (any weights) has zero variance, so
/// CV is 0.0, never NaN (the `mean > 0` path with variance 0). Locks the
/// degenerate arm the >cap integration test cannot reach.
#[test]
fn weighted_cv_all_equal_values_is_zero() {
    assert_eq!(
        reduce_weighted_sorted_distribution(&[(5_000, 3.0), (5_000, 7.0)], SampleReduction::Cv),
        0.0,
    );
}

/// COMPOSITION (merge layer × pool layer, the two-layer de-skew claim): one cgroup
/// in two phases. phase0 is a single ≤cap carrier of 100k wakes @ 10 µs (true pop
/// 100k, weight 1.0). phase1 is the `PhaseCgroupStats::merge` of two 60k @ 30 µs
/// carriers — combined len 120k > MAX_WAKE_SAMPLES, so it routes through the
/// WEIGHTED merge: reservoir re-capped to 100k (all 30 µs, since both sources are
/// all-30 µs) and wake_sample_total SUMMED to 120k. The run-level pool then weights
/// phase1 by 120k/100k = 1.2. EQUAL capped lengths (100k:100k) but UNEQUAL true
/// populations (100k:120k): the population-weighted median is 30 µs (target
/// ceil(220k·0.5)=110k crosses into the 30 µs block), while a length-weighted
/// concat (the deleted path) would land the median at the 100k:100k boundary on
/// the 10 µs side. Proves the merge's summed total drives the pool weight with no
/// double-weighting (the pool weight is recomputed from the merged carrier's
/// fields, not stacked on the merge's internal split).
#[test]
fn repool_wake_merge_and_pool_layers_compose() {
    use crate::workload::MAX_WAKE_SAMPLES as CAP;
    let phase0 = PhaseCgroupStats {
        wake_latencies_ns: vec![10_000; CAP], // 100k @ 10 µs, ≤cap
        wake_sample_total: CAP as u64,        // true pop 100k → weight 1.0
        ..PhaseCgroupStats::default()
    };
    // phase1: merge two 60k @ 30 µs carriers → >cap weighted merge → 100k @ 30 µs,
    // wake_sample_total = 120k. All-equal values make the merged reservoir
    // deterministic regardless of the random source split.
    let sub = || PhaseCgroupStats {
        wake_latencies_ns: vec![30_000; 60_000],
        wake_sample_total: 60_000,
        ..PhaseCgroupStats::default()
    };
    let phase1 = PhaseCgroupStats::merge(sub(), sub());
    assert_eq!(
        phase1.wake_latencies_ns.len(),
        CAP,
        "merge re-caps to MAX_WAKE_SAMPLES"
    );
    assert_eq!(
        phase1.wake_sample_total, 120_000,
        "merge SUMs true population"
    );

    let mut b0 = PhaseBucket {
        step_index: 0,
        ..PhaseBucket::default()
    };
    b0.per_cgroup.insert("a".to_string(), phase0);
    let mut b1 = PhaseBucket {
        step_index: 1,
        ..PhaseBucket::default()
    };
    b1.per_cgroup.insert("a".to_string(), phase1);
    let mut stats = ScenarioStats {
        phases: vec![b0, b1],
        ..ScenarioStats::default()
    };
    populate_run_distribution_metrics(&mut stats);

    let median = stats
        .ext_metrics
        .get("worst_median_wake_latency_us")
        .copied();
    assert_eq!(
        median,
        Some(30.0),
        "merged phase1's summed population (120k > phase0's 100k) wins the weighted \
         median despite EQUAL capped lengths; got {median:?}",
    );
    assert_ne!(
        median,
        Some(10.0),
        "must NOT be the length-weighted (100k:100k) median — that ignores the \
         merge's summed true population",
    );
}
