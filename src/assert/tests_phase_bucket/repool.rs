use super::*;

// -- run-level distributional re-pool (populate_run_distribution_metrics) --

/// THE THESIS: the run-level wake p99 is the percentile over the COMBINED
/// cross-cgroup sample set, NOT the max of per-cgroup p99s. cg_a carries 100
/// low samples (1..100 µs); cg_b carries 3 high samples (1000/2000/3000 µs).
/// The pooled p99 over the 103-sample union is 2000 µs, while the deleted
/// max-of-per-cgroup fold would report cg_b's own p99 (3000 µs).
#[test]
fn repool_distribution_pools_wake_across_cgroups_not_max_of_per_cgroup() {
    let cg_a = PhaseCgroupStats {
        wake_latencies_ns: (1..=100u64).map(|v| v * 1000).collect(),
        wake_sample_total: 100,
        ..PhaseCgroupStats::default()
    };
    let cg_b = PhaseCgroupStats {
        wake_latencies_ns: vec![1_000_000, 2_000_000, 3_000_000],
        wake_sample_total: 3,
        ..PhaseCgroupStats::default()
    };
    let mut stats = repool_stats(vec![("a", cg_a), ("b", cg_b)], vec![]);
    populate_run_distribution_metrics(&mut stats);

    let p99 = stats.ext_metrics.get("worst_p99_wake_latency_us").copied();
    assert_eq!(
        p99,
        Some(2000.0),
        "pooled cross-cgroup p99 over the 103-sample union, got {p99:?}",
    );
    assert_ne!(
        p99,
        Some(3000.0),
        "must NOT be the max of per-cgroup p99s (cg_b's 3000 µs) — the thesis",
    );
}

/// Empty input (no phases, no cgroups) writes NO Distribution or WorstLowest
/// key — absence is preserved, never a synthesized 0.0 (mirrors the
/// derive_rate_metrics both-or-neither contract).
#[test]
fn repool_distribution_empty_inserts_no_keys() {
    let mut stats = ScenarioStats::default();
    populate_run_distribution_metrics(&mut stats);
    for name in [
        "worst_p99_wake_latency_us",
        "worst_median_wake_latency_us",
        "worst_wake_latency_cv",
        "worst_mean_run_delay_us",
        "worst_run_delay_us",
        "worst_iterations_per_worker",
        "worst_iterations_per_cpu_sec",
        "worst_wake_latency_tail_ratio",
    ] {
        assert!(
            !stats.ext_metrics.contains_key(name),
            "{name} must be absent for empty input",
        );
    }
}

/// Single-sample boundary: p99 == median == the sole sample (µs), CV == 0
/// (stddev of one sample is 0).
#[test]
fn repool_distribution_single_sample() {
    let cg = PhaseCgroupStats {
        wake_latencies_ns: vec![42_000],
        wake_sample_total: 1,
        ..PhaseCgroupStats::default()
    };
    let mut stats = repool_stats(vec![("a", cg)], vec![]);
    populate_run_distribution_metrics(&mut stats);
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(42.0),
    );
    assert_eq!(
        stats
            .ext_metrics
            .get("worst_median_wake_latency_us")
            .copied(),
        Some(42.0),
    );
    assert_eq!(
        stats.ext_metrics.get("worst_wake_latency_cv").copied(),
        Some(0.0),
    );
}

/// Value-for-value: build the carrier from the SAME reports cgroup_stats
/// reduces (single cgroup, ≤ cap so the carrier IS the full pool), run the
/// actual re-pool, and assert ext_metrics reproduces cgroup_stats's wake /
/// run-delay reductions exactly.
#[test]
fn repool_distribution_value_for_value_with_cgroup_stats() {
    // Faithful ≤cap reports: wake_sample_total == wake_latencies_ns.len(), the
    // invariant reservoir_push maintains below MAX_WAKE_SAMPLES (count bumped per
    // wakeup, every wakeup also pushed). This makes the run-level weight 1.0, so
    // the weighted re-pool must reproduce cgroup_stats VALUE-FOR-VALUE.
    let reports = vec![
        WorkerReport {
            wake_latencies_ns: vec![1000, 2000, 3000, 4000, 5000],
            wake_sample_total: 5,
            schedstat_run_delay_ns: 7000,
            schedstat_cpu_time_ns: 1_000_000,
            ..rpt(1, 1000, 1_000_000, 0, &[0], 0)
        },
        WorkerReport {
            wake_latencies_ns: vec![6000, 7000, 8000, 9000, 10000],
            wake_sample_total: 5,
            schedstat_run_delay_ns: 3000,
            schedstat_cpu_time_ns: 1_000_000,
            ..rpt(2, 1000, 1_000_000, 0, &[1], 0)
        },
    ];
    let cg = cgroup_stats(&reports);
    let carrier = phase_cgroup_stats(&reports, None);
    let mut stats = repool_stats(vec![("a", carrier)], vec![]);
    populate_run_distribution_metrics(&mut stats);
    let ext = |n: &str| stats.ext_metrics.get(n).copied().unwrap();
    assert!((ext("worst_p99_wake_latency_us") - cg.p99_wake_latency_us).abs() < 1e-9);
    assert!((ext("worst_median_wake_latency_us") - cg.median_wake_latency_us).abs() < 1e-9);
    assert!((ext("worst_wake_latency_cv") - cg.wake_latency_cv).abs() < 1e-9);
    assert!((ext("worst_mean_run_delay_us") - cg.mean_run_delay_us).abs() < 1e-9);
    assert!((ext("worst_run_delay_us") - cg.worst_run_delay_us).abs() < 1e-9);
}

/// `WakeLatencyTailRatio` producer contract: `populate_run_distribution_metrics`
/// emits the `worst_wake_latency_tail_ratio` ext key as the MAX over the
/// per-cgroup `CgroupStats::wake_latency_tail_ratio` values — but ONLY when the
/// run cleared the `WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS` floor AND at least
/// one cgroup carried a measurable tail (median > 0). Below the floor, or with
/// no measurable tail, the key is ABSENT (excluded from the cross-RUN mean,
/// read as None by compare) — never a 0.0 sentinel. This is the floor gate the
/// deleted typed-field accessor used to enforce, now moved to the producer.
#[test]
fn wake_latency_tail_ratio_producer_floor_gates_and_maxes() {
    use crate::stats::WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS as MIN;
    let key = "worst_wake_latency_tail_ratio";
    // Two cgroups with measurable tails: ratios 10.0 (20/2) and 4.0 (8/2).
    let tail_cgroups = || {
        vec![
            CgroupStats {
                cgroup_name: "a".to_string(),
                p99_wake_latency_us: 20.0,
                median_wake_latency_us: 2.0,
                ..CgroupStats::default()
            },
            CgroupStats {
                cgroup_name: "b".to_string(),
                p99_wake_latency_us: 8.0,
                median_wake_latency_us: 2.0,
                ..CgroupStats::default()
            },
        ]
    };

    // Below the floor: no key, even though both cgroups carry a tail.
    let mut below = repool_stats(vec![], tail_cgroups());
    below.total_iterations = MIN - 1;
    populate_run_distribution_metrics(&mut below);
    assert_eq!(
        below.ext_metrics.get(key).copied(),
        None,
        "sub-threshold run must emit no tail-ratio key (floor gate at the producer)",
    );

    // Above the floor: key present, the MAX over per-cgroup ratios (10.0 > 4.0).
    let mut above = repool_stats(vec![], tail_cgroups());
    above.total_iterations = MIN;
    populate_run_distribution_metrics(&mut above);
    assert_eq!(
        above.ext_metrics.get(key).copied(),
        Some(10.0),
        "above-floor key must be the MAX over per-cgroup p99/median ratios",
    );

    // Above the floor but NO measurable tail (median 0 -> per-cgroup ratio 0.0):
    // absent, not a 0.0 sentinel.
    let mut no_tail = repool_stats(
        vec![],
        vec![CgroupStats {
            cgroup_name: "a".to_string(),
            p99_wake_latency_us: 0.0,
            median_wake_latency_us: 0.0,
            ..CgroupStats::default()
        }],
    );
    no_tail.total_iterations = MIN;
    populate_run_distribution_metrics(&mut no_tail);
    assert_eq!(
        no_tail.ext_metrics.get(key).copied(),
        None,
        "a run with no measurable tail (median 0) must emit no key, not Some(0.0)",
    );

    // Above the floor, MIXED: one cgroup with a tail (10/2 = 5.0), one with
    // median 0 (ratio 0.0). The `r > 0.0` guard skips the zero so it is NOT
    // folded into the max; the key is the surviving cgroup's tail, 5.0.
    let mut mixed = repool_stats(
        vec![],
        vec![
            CgroupStats {
                cgroup_name: "a".to_string(),
                p99_wake_latency_us: 10.0,
                median_wake_latency_us: 2.0,
                ..CgroupStats::default()
            },
            CgroupStats {
                cgroup_name: "b".to_string(),
                p99_wake_latency_us: 0.0,
                median_wake_latency_us: 0.0,
                ..CgroupStats::default()
            },
        ],
    );
    mixed.total_iterations = MIN;
    populate_run_distribution_metrics(&mut mixed);
    assert_eq!(
        mixed.ext_metrics.get(key).copied(),
        Some(5.0),
        "a median-0 cgroup (ratio 0.0) is skipped by the r>0.0 guard; the key is \
         the surviving cgroup's tail (10/2=5.0), not folded with the 0.0",
    );
}
