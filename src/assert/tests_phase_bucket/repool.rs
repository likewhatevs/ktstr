use super::*;
use crate::assert::derive_phase_metrics;

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
        "worst_page_locality",
        "worst_cross_node_migration_ratio",
    ] {
        assert!(
            !stats.ext_metrics.contains_key(name),
            "{name} must be absent for empty input",
        );
    }
}

/// `worst_page_locality` (WorstLowest{NumaLocal,NumaTotal}) re-pools into
/// `ext_metrics` from the per-phase NUMA carriers — the producer the cross-run
/// sidecar comparison relies on (the reports-only `CgroupStats` hardcodes
/// page_locality 0.0, so it CANNOT source from `stats.cgroups`; it must come
/// from `stats.phases[].per_cgroup`). The lowest per-cgroup
/// `numa_pages_local / numa_pages_total` over cgroups that measured NUMA wins; a
/// measured 0.0 (all off-node) wins rather than being skipped; an all-unmeasured
/// (`numa_pages_total == 0`) cohort writes NO key (absence preserved, never a
/// 0.0 sentinel).
#[test]
fn repool_worst_page_locality_from_numa_carriers() {
    let pc = |local: u64, total: u64| PhaseCgroupStats {
        numa_pages_local: local,
        numa_pages_total: total,
        ..PhaseCgroupStats::default()
    };
    // Two distinct POSITIVE localities: the lowest (0.5) wins (plain f64::min of
    // the two — distinct from the measured-0.0-wins case below; the fold filters
    // on numa_pages_total>0, NOT on a non-zero value).
    let mut positive = repool_stats(vec![("a", pc(800, 1000)), ("b", pc(500, 1000))], vec![]);
    populate_run_distribution_metrics(&mut positive);
    assert_eq!(
        positive.ext_metrics.get("worst_page_locality").copied(),
        Some(0.5),
        "lowest of two positive per-cgroup localities (0.8 vs 0.5) wins",
    );

    // cg_a 800/1000 = 0.8; cg_b 0/1000 = measured 0.0 (all off-node) → the worst.
    let mut measured = repool_stats(vec![("a", pc(800, 1000)), ("b", pc(0, 1000))], vec![]);
    populate_run_distribution_metrics(&mut measured);
    assert_eq!(
        measured.ext_metrics.get("worst_page_locality").copied(),
        Some(0.0),
        "lowest per-cgroup locality wins; a measured 0.0 is the worst",
    );

    // All cgroups measured no NUMA pages (total == 0) → no key (loud-absent).
    let mut unmeasured = repool_stats(vec![("a", pc(0, 0)), ("b", pc(0, 0))], vec![]);
    populate_run_distribution_metrics(&mut unmeasured);
    assert!(
        !unmeasured.ext_metrics.contains_key("worst_page_locality"),
        "an all-unmeasured cohort must write no worst_page_locality key",
    );
}

/// `worst_cross_node_migration_ratio` (WorstCrossNodeRatio) re-pools into
/// `ext_metrics` from the per-phase NUMA carriers — the polarity twin of
/// worst_page_locality (MAX-wins, LowerBetter). The HIGHEST per-cgroup
/// `cross_node_migrated / numa_pages_total` over cgroups that measured NUMA wins;
/// an all-unmeasured (`numa_pages_total == 0`) cohort writes NO key (absence
/// preserved). A CHURN ratio (cumulative events / residency snapshot) — can
/// exceed 1.0.
#[test]
fn repool_worst_cross_node_migration_ratio_from_numa_carriers() {
    let pc = |total: u64, migr: u64| PhaseCgroupStats {
        numa_pages_total: total,
        cross_node_migrated: migr,
        ..PhaseCgroupStats::default()
    };
    // cg_a 50/1000 = 0.05; cg_b 150/1000 = 0.15 → the worst (highest) wins.
    let mut measured = repool_stats(vec![("a", pc(1000, 50)), ("b", pc(1000, 150))], vec![]);
    populate_run_distribution_metrics(&mut measured);
    let v = measured
        .ext_metrics
        .get("worst_cross_node_migration_ratio")
        .copied()
        .expect("measured cohort yields a value");
    assert!(
        (v - 0.15).abs() < 1e-9,
        "highest per-cgroup churn (0.15) wins; got {v}",
    );

    // CHURN ratio can exceed 1.0 (3000 migrations / 1000 pages = 3.0).
    let mut churn = repool_stats(vec![("a", pc(1000, 3000))], vec![]);
    populate_run_distribution_metrics(&mut churn);
    let c = churn
        .ext_metrics
        .get("worst_cross_node_migration_ratio")
        .copied()
        .expect("measured");
    assert!(
        (c - 3.0).abs() < 1e-9,
        "churn ratio is unbounded (3.0 > 1.0); got {c}",
    );

    // Measured residency (total > 0) but zero cross-node migrations → ratio 0.0,
    // a MEASURED zero (Some(0.0)) — distinct from never-measured None. The max
    // over the sole measured cgroup's [0.0] is a present 0.0.
    let mut measured_zero = repool_stats(vec![("a", pc(1000, 0))], vec![]);
    populate_run_distribution_metrics(&mut measured_zero);
    assert_eq!(
        measured_zero
            .ext_metrics
            .get("worst_cross_node_migration_ratio")
            .copied(),
        Some(0.0),
        "a measured zero (residency present, no migrations) is Some(0.0), not absent",
    );

    // All cgroups measured no NUMA pages (total == 0) → no key (loud-absent).
    let mut unmeasured = repool_stats(vec![("a", pc(0, 0)), ("b", pc(0, 0))], vec![]);
    populate_run_distribution_metrics(&mut unmeasured);
    assert!(
        !unmeasured
            .ext_metrics
            .contains_key("worst_cross_node_migration_ratio"),
        "an all-unmeasured cohort must write no worst_cross_node_migration_ratio key",
    );
}

/// Cross-PHASE producer pin: the ext re-pool (not just `run_metric`) gets the
/// ASYMMETRIC fold — a cgroup measured across TWO phases has its
/// `cross_node_migrated` SUMMED (per-phase deltas) over the LATEST residency
/// total (a gauge, NOT summed), matching `numa_agg_per_cgroup`. This is the
/// multi-phase divergence the migration fixes: the deleted typed `Gauge(Last)`
/// field could not see the per-phase carriers, so its cross-run value diverged
/// from `run_metric` here.
#[test]
fn repool_worst_cross_node_migration_ratio_sums_migrations_over_latest_total() {
    let pc = |total: u64, migr: u64| PhaseCgroupStats {
        numa_pages_total: total,
        cross_node_migrated: migr,
        ..PhaseCgroupStats::default()
    };
    let phase = |step: u16, name: &str, pc: PhaseCgroupStats| {
        let mut b = PhaseBucket {
            step_index: step,
            ..PhaseBucket::default()
        };
        b.per_cgroup.insert(name.to_string(), pc);
        b
    };
    // cgroup A across two phases: migrations 30 + 50 = 80 (per-phase deltas SUM);
    // residency total takes the LATEST measured (2000, NOT summed 3000, NOT
    // phase-1's 1000) -> 80 / 2000 = 0.04. Every wrong fold lands elsewhere
    // (30/1000=0.03, 80/1000=0.08, 80/3000≈0.0267, 50/2000=0.025).
    let mut stats = ScenarioStats {
        phases: vec![phase(1, "a", pc(1000, 30)), phase(2, "a", pc(2000, 50))],
        ..ScenarioStats::default()
    };
    populate_run_distribution_metrics(&mut stats);
    let v = stats
        .ext_metrics
        .get("worst_cross_node_migration_ratio")
        .copied()
        .expect("a multi-phase carrier yields a value");
    assert!(
        (v - 0.04).abs() < 1e-9,
        "summed migrations 80 / LATEST total 2000 = 0.04 (not summed-total, not \
         phase-1 total); got {v}",
    );

    // A later phase that measured NO residency (total == 0) but recorded a
    // nonzero migration delta: the delta STILL sums (e.2 += is unconditional),
    // while the total==0 phase does NOT overwrite the latest measured total.
    // phase1 (total 1000, migr 30) + phase2 (total 0, migr 20) → summed 50 /
    // LATEST measured total 1000 = 0.05.
    let mut zero_total_mid = ScenarioStats {
        phases: vec![phase(1, "a", pc(1000, 30)), phase(2, "a", pc(0, 20))],
        ..ScenarioStats::default()
    };
    populate_run_distribution_metrics(&mut zero_total_mid);
    let z = zero_total_mid
        .ext_metrics
        .get("worst_cross_node_migration_ratio")
        .copied()
        .expect("the earlier measured phase keeps the cgroup in the pool");
    assert!(
        (z - 0.05).abs() < 1e-9,
        "a total==0 phase's migration delta still SUMs (30+20=50) over the LATEST \
         measured total (1000, not 0) = 0.05; got {z}",
    );
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

/// Per-cgroup NON-schbench parity: `derive_phase_metrics` writes each carrier's
/// reduced scalars into `pc.metrics` value-for-value with the single-cgroup
/// `cgroup_stats` reduction (RATIOS / COUNTERS / DISTRIBUTIONS), the pooled
/// RATIOS re-pool deterministically from the concatenated reports, but the
/// DISTRIBUTION families are deliberately NOT pooled==per-cgroup (the pooled
/// value comes from the host sample fold, not a carrier re-pool — see
/// `repool_distribution_pools_wake_across_cgroups_not_max_of_per_cgroup`).
#[test]
fn derive_writes_non_schbench_per_cgroup_value_for_value() {
    let nodes: std::collections::BTreeSet<usize> = std::collections::BTreeSet::from([0usize]);
    // cgroup A: 2 workers, ≤cap wake pools so the carrier IS the full pool and
    // the percentile re-derives value-for-value.
    let reports_a = vec![
        WorkerReport {
            migration_count: 3,
            iterations: 1000,
            schedstat_cpu_time_ns: 2_000_000_000, // 2s on-CPU
            schedstat_run_delay_ns: 4000,
            wake_latencies_ns: vec![1000, 2000, 3000, 4000, 5000],
            wake_sample_total: 5,
            timer_latencies_ns: vec![1500, 2500, 3500, 4500, 5500],
            timer_sample_total: 5,
            numa_pages: [(0usize, 100u64), (1usize, 50u64)].into_iter().collect(),
            vmstat_numa_pages_migrated: 10,
            ..rpt(1, 1000, 1_000_000, 200_000, &[0], 0) // wall 1ms, off-cpu 0.2ms -> 20%
        },
        WorkerReport {
            migration_count: 1,
            iterations: 1000,
            schedstat_cpu_time_ns: 2_000_000_000,
            schedstat_run_delay_ns: 6000,
            wake_latencies_ns: vec![6000, 7000, 8000, 9000, 10000],
            wake_sample_total: 5,
            timer_latencies_ns: vec![6500, 7500, 8500, 9500, 10500],
            timer_sample_total: 5,
            numa_pages: [(0usize, 200u64), (1usize, 50u64)].into_iter().collect(),
            vmstat_numa_pages_migrated: 20,
            ..rpt(2, 1000, 1_000_000, 100_000, &[1], 0) // 10%
        },
    ];
    let cg_a = cgroup_stats(&reports_a);
    let carrier_a = phase_cgroup_stats(&reports_a, Some(&nodes));

    // cgroup B: high-magnitude wake pool so the cross-cgroup union p99 differs
    // from either cgroup's own p99 (the distribution thesis).
    let reports_b = vec![WorkerReport {
        migration_count: 7,
        iterations: 500,
        schedstat_cpu_time_ns: 1_000_000_000,
        schedstat_run_delay_ns: 99000,
        wake_latencies_ns: vec![100_000, 200_000, 300_000],
        wake_sample_total: 3,
        numa_pages: [(0usize, 10u64), (1usize, 90u64)].into_iter().collect(),
        vmstat_numa_pages_migrated: 5,
        ..rpt(3, 500, 1_000_000, 500_000, &[0], 0) // 50%
    }];
    let cg_b = cgroup_stats(&reports_b);
    let carrier_b = phase_cgroup_stats(&reports_b, Some(&nodes));

    let mut bucket = PhaseBucket::default();
    bucket.per_cgroup.insert("a".to_string(), carrier_a);
    bucket.per_cgroup.insert("b".to_string(), carrier_b);
    derive_phase_metrics(std::slice::from_mut(&mut bucket));

    let a = &bucket.per_cgroup["a"];
    let m = |n: &str| -> f64 {
        *a.metrics
            .get(n)
            .unwrap_or_else(|| panic!("missing per-cgroup key {n}"))
    };

    // RATIOS — per-cgroup == single-carrier cgroup_stats value-for-value.
    assert!((m("migration_ratio") - cg_a.migration_ratio).abs() < 1e-12);
    assert!((m("iterations_per_worker") - cg_a.iterations_per_worker().unwrap()).abs() < 1e-9);
    assert!((m("iteration_rate") - cg_a.iterations_per_cpu_sec().unwrap()).abs() < 1e-9);
    assert!((m("cross_node_migration_ratio") - cg_a.cross_node_migration_ratio).abs() < 1e-12);
    // page_locality: cgroup_stats hardcodes 0.0 (no NUMA node context in the
    // reports-only builder), so parity is against the hand-derived value from
    // the carrier (expected nodes {0}: local 100+200=300, total 400).
    assert!((m("page_locality") - (300.0 / 400.0)).abs() < 1e-12);

    // COUNTER — total_cpu_time_ns resolves via the carrier counter (not pc.metrics).
    assert_eq!(
        a.cgroup_counter("total_cpu_time_ns"),
        Some(cg_a.total_cpu_time_ns as f64)
    );

    // DISTRIBUTIONS — per-cgroup == single-carrier reduction (≤cap, value-for-value).
    assert!((m("p99_wake_latency_us") - cg_a.p99_wake_latency_us).abs() < 1e-9);
    assert!((m("median_wake_latency_us") - cg_a.median_wake_latency_us).abs() < 1e-9);
    assert!((m("wake_latency_cv") - cg_a.wake_latency_cv).abs() < 1e-9);
    // Timer-latency per-cgroup derive — the CI-runnable pin for the timer derive
    // (timer_summary -> p99/median/p999_timer_latency_us in pc.metrics), so a SKIP
    // of the host-gated timer_latency_e2e cannot mask a derive regression.
    // Value-for-value with cgroup_stats below the reservoir cap, exactly like wake.
    assert!((m("p99_timer_latency_us") - cg_a.p99_timer_latency_us).abs() < 1e-9);
    assert!((m("median_timer_latency_us") - cg_a.median_timer_latency_us).abs() < 1e-9);
    assert!((m("p999_timer_latency_us") - cg_a.p999_timer_latency_us).abs() < 1e-9);
    assert!((m("mean_run_delay_us") - cg_a.mean_run_delay_us).abs() < 1e-9);
    assert!((m("max_run_delay_us") - cg_a.worst_run_delay_us).abs() < 1e-9);
    assert!((m("avg_off_cpu_pct") - cg_a.avg_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((m("min_off_cpu_pct") - cg_a.min_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((m("max_off_cpu_pct") - cg_a.max_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((m("off_cpu_spread_pct") - cg_a.spread.unwrap()).abs() < 1e-9);

    // POOLED RATIOS re-pool deterministically from the concatenated reports:
    // pooled migration_ratio == migration_ratio_of(Σmig, Σiters).
    let sum_mig = cg_a.total_migrations + cg_b.total_migrations;
    let sum_iters = cg_a.total_iterations + cg_b.total_iterations;
    let mut concat = reports_a.clone();
    concat.extend(reports_b.clone());
    let cg_concat = cgroup_stats(&concat);
    assert!(
        (cg_concat.migration_ratio
            - crate::assert::reductions::migration_ratio_of(sum_mig, sum_iters))
        .abs()
            < 1e-12
    );

    // DISTRIBUTIONS are NOT pooled==per-cgroup: cg_b's high-magnitude wake pool
    // gives it a distinct per-cgroup p99; the per-cgroup keys are present and
    // equal the single carrier (asserted above), NOT a shared pooled value (the
    // host sample fold owns the pooled distribution).
    let b = &bucket.per_cgroup["b"];
    assert_ne!(
        a.metrics.get("p99_wake_latency_us"),
        b.metrics.get("p99_wake_latency_us"),
        "distinct cgroups must have distinct per-cgroup p99 — not a shared pool",
    );
    // And the pooled bucket map is NOT written for the non-schbench families
    // (only pc.metrics is); the pooled value is the host-sample-fold's job.
    assert!(!bucket.metrics.contains_key("p99_wake_latency_us"));
    assert!(!bucket.metrics.contains_key("migration_ratio"));
}

/// ABSENT discipline: a default (no-sample) carrier emits ONLY `migration_ratio`
/// (a measured 0.0 when no iterations ran) — every distribution / ratio / counter
/// key whose source is empty/zero is ABSENT (reads as missing, never a false 0).
#[test]
fn derive_non_schbench_absent_discipline() {
    let mut bucket = PhaseBucket::default();
    bucket
        .per_cgroup
        .insert("cg".to_string(), PhaseCgroupStats::default());
    derive_phase_metrics(std::slice::from_mut(&mut bucket));
    let m = &bucket.per_cgroup["cg"].metrics;
    assert_eq!(
        m.get("migration_ratio"),
        Some(&0.0),
        "migration_ratio is ALWAYS present (measured 0.0 when no iterations)",
    );
    for absent in [
        "p99_wake_latency_us",
        "median_wake_latency_us",
        "wake_latency_cv",
        "mean_run_delay_us",
        "max_run_delay_us",
        "avg_off_cpu_pct",
        "min_off_cpu_pct",
        "max_off_cpu_pct",
        "off_cpu_spread_pct",
        "iterations_per_worker",
        "iteration_rate",
        "page_locality",
        "cross_node_migration_ratio",
    ] {
        assert!(
            !m.contains_key(absent),
            "{absent} must be ABSENT for a default carrier (no false 0)",
        );
    }
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
