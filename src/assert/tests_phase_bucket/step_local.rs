use super::*;

// ---- step-local per-cgroup capture ----
//
// phase_cgroup_stats extracts the RAW components every cgroup_stats reduction
// re-pools from; step_per_cgroup_bucket wraps them in the guest-side carrier;
// fold_guest_per_cgroup_into_host_buckets merges the carrier into the
// host-rebuilt buckets by step_index. These tests pin (1) raw-component parity
// vs cgroup_stats, (2) the math-boundary edges (not-measured off-CPU%, raw-ns
// run delays, the numa partition, MAX-not-SUM cross-node, argmax gap coupling),
// (3) the carrier shape, and (4) the fold (union / orphan / pass-through).

/// Every [`super::CgroupStats`] reduction re-pools EXACTLY from the
/// [`PhaseCgroupStats`] raw components: avg/min/max/spread off-CPU% from
/// `off_cpu_pcts`; p99/median from the pooled `wake_latencies_ns`; mean/worst
/// run-delay from `run_delays_ns` (RAW ns ÷ 1000); migration_ratio /
/// cross_node_migration_ratio from the counters; cpus_used / num_cpus from the
/// union. The one addition is `numa_pages_local` (cgroup_stats has no node
/// context, leaving page_locality 0.0). Built on the SAME two reports so the
/// re-pool MUST reproduce cgroup_stats's reductions value-for-value.
#[test]
fn phase_cgroup_stats_components_repool_to_cgroup_stats() {
    let w1 = WorkerReport {
        schedstat_run_delay_ns: 3000,
        schedstat_cpu_time_ns: 1_000_000,
        migration_count: 2,
        iterations: 100,
        wake_latencies_ns: vec![1000, 2000],
        wake_sample_total: 5,
        vmstat_numa_pages_migrated: 10,
        numa_pages: BTreeMap::from([(0usize, 100u64), (1, 50)]),
        // wall 1_000_000, off 200_000 -> off-CPU% 20.0; gap 40 on cpu 0.
        ..rpt(1, 1000, 1_000_000, 200_000, &[0, 1], 40)
    };
    let w2 = WorkerReport {
        schedstat_run_delay_ns: 5000,
        schedstat_cpu_time_ns: 2_000_000,
        migration_count: 3,
        iterations: 200,
        wake_latencies_ns: vec![3000],
        wake_sample_total: 7,
        vmstat_numa_pages_migrated: 8,
        numa_pages: BTreeMap::from([(1usize, 80u64), (2, 20)]),
        // wall 2_000_000, off 100_000 -> off-CPU% 5.0; gap 60 on cpu 2.
        ..rpt(2, 2000, 2_000_000, 100_000, &[2, 3], 60)
    };
    let reports = vec![w1, w2];
    let nodes: BTreeSet<usize> = [0, 1].into_iter().collect();
    let pcs = phase_cgroup_stats(&reports, Some(&nodes));
    let cg = cgroup_stats(&reports);

    // cpus_used union + worker count.
    assert_eq!(pcs.cpus_used, cg.cpus_used);
    assert_eq!(pcs.cpus_used.len(), cg.num_cpus);
    assert_eq!(pcs.num_workers, cg.num_workers);

    // off-CPU% re-pool to avg/min/max/spread.
    assert_eq!(pcs.off_cpu_pcts.len(), 2);
    let avg = pcs.off_cpu_pcts.iter().sum::<f64>() / pcs.off_cpu_pcts.len() as f64;
    let min = pcs.off_cpu_pcts.iter().cloned().reduce(f64::min).unwrap();
    let max = pcs.off_cpu_pcts.iter().cloned().reduce(f64::max).unwrap();
    assert!((avg - cg.avg_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((min - cg.min_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((max - cg.max_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!(((max - min) - cg.spread.unwrap()).abs() < 1e-9);

    // Pooled wake latencies re-pool to p99/median via the same percentile fn.
    let mut pooled = pcs.wake_latencies_ns.clone();
    pooled.sort_unstable();
    assert_eq!(pooled, vec![1000u64, 2000, 3000]);
    let p99 = percentile(&pooled, 0.99) as f64 / 1000.0;
    let median = percentile(&pooled, 0.5) as f64 / 1000.0;
    assert!((p99 - cg.p99_wake_latency_us).abs() < 1e-9);
    assert!((median - cg.median_wake_latency_us).abs() < 1e-9);
    assert_eq!(pcs.wake_sample_total, 12);
    // CV re-pools over the SAME pooled set with n = wake_latencies_ns.len()
    // (the reservoir-CLAMPED pool size), NOT wake_sample_total — exactly as
    // cgroup_stats computes it (cv = stddev/mean, n = all_latencies.len()).
    // Here len() == 3 but wake_sample_total == 12, so a re-pool that wrongly
    // divided by wake_sample_total would NOT reproduce cg.wake_latency_cv.
    let n = pcs.wake_latencies_ns.len() as f64;
    let mean_ns = pcs.wake_latencies_ns.iter().sum::<u64>() as f64 / n;
    let cv = if mean_ns > 0.0 {
        let variance = pcs
            .wake_latencies_ns
            .iter()
            .map(|&v| (v as f64 - mean_ns).powi(2))
            .sum::<f64>()
            / n;
        variance.sqrt() / mean_ns
    } else {
        0.0
    };
    assert!(
        (cv - cg.wake_latency_cv).abs() < 1e-9,
        "wake_latency_cv re-pools with n = len() ({}), NOT wake_sample_total \
         ({}); got {cv} vs cg {}",
        pcs.wake_latencies_ns.len(),
        pcs.wake_sample_total,
        cg.wake_latency_cv,
    );

    // Run delays stored RAW ns; the re-pool divides by 1000 ONCE to match
    // cgroup_stats's µs mean/worst (pre-dividing here would be a 1000x error).
    assert_eq!(pcs.run_delays_ns, vec![3000u64, 5000]);
    let rd_us: Vec<f64> = pcs
        .run_delays_ns
        .iter()
        .map(|&v| v as f64 / 1000.0)
        .collect();
    let mean_rd = rd_us.iter().sum::<f64>() / rd_us.len() as f64;
    let worst_rd = rd_us.iter().cloned().reduce(f64::max).unwrap();
    assert!((mean_rd - cg.mean_run_delay_us).abs() < 1e-9);
    assert!((worst_rd - cg.worst_run_delay_us).abs() < 1e-9);

    // Counters + the ratios they re-pool.
    assert_eq!(pcs.total_migrations, cg.total_migrations);
    assert_eq!(pcs.total_iterations, cg.total_iterations);
    assert_eq!(pcs.total_cpu_time_ns, cg.total_cpu_time_ns);
    assert!(
        (pcs.total_migrations as f64 / pcs.total_iterations as f64 - cg.migration_ratio).abs()
            < 1e-9
    );

    // NUMA totals: total = sum, cross-node = MAX (system-wide delta observed
    // redundantly), re-pooling to cross_node_migration_ratio.
    assert_eq!(pcs.numa_pages_total, 250);
    assert_eq!(pcs.cross_node_migrated, 10);
    assert!(
        (pcs.cross_node_migrated as f64 / pcs.numa_pages_total as f64
            - cg.cross_node_migration_ratio)
            .abs()
            < 1e-9
    );

    // numa_pages_local: the per-phase addition cgroup_stats lacks. Pages on the
    // expected nodes {0,1}: w1 100+50=150, w2 80 (node 1) -> 230.
    assert_eq!(pcs.numa_pages_local, 230);
    assert_eq!(
        cg.page_locality, 0.0,
        "cgroup_stats has no node context; phase_cgroup_stats captures the locality numerator",
    );
    // numa_pages_local's authority is assert_cgroup (the page_locality numerator),
    // NOT cgroup_stats. Cross-check the re-pool against the real reduction
    // value-for-value so a divergence in the partition node set would be caught.
    let cg_numa =
        crate::assert::Assert::NO_OVERRIDES.assert_cgroup_with_numa(&reports, None, Some(&nodes));
    let page_locality = cg_numa.stats.cgroups[0].page_locality;
    assert!(
        (pcs.numa_pages_local as f64 / pcs.numa_pages_total as f64 - page_locality).abs() < 1e-9,
        "numa_pages_local/total re-pools to assert_cgroup's page_locality ({page_locality})",
    );

    // Worst gap stays coupled to its CPU (argmax of the pair, not two maxes).
    assert_eq!(
        (pcs.max_gap_ms, pcs.max_gap_cpu),
        (cg.max_gap_ms, cg.max_gap_cpu)
    );
    assert_eq!((pcs.max_gap_ms, pcs.max_gap_cpu), (60, 2));
}

/// Boundary: NO worker has measurable wall time -> `off_cpu_pcts` is EMPTY (the
/// not-measured state), so the re-pool yields None for avg/min/max/spread —
/// preserving the not-measured-vs-measured-zero distinction cgroup_stats keeps
/// (`avg_off_cpu_pct` None), NOT a measured-0% / perfectly-fair reading.
#[test]
fn phase_cgroup_stats_off_cpu_pcts_empty_when_no_wall_time() {
    let reports = vec![rpt(1, 1000, 0, 0, &[0], 10), rpt(2, 1000, 0, 0, &[1], 10)];
    let pcs = phase_cgroup_stats(&reports, None);
    assert!(
        pcs.off_cpu_pcts.is_empty(),
        "no measurable wall time -> not-measured (empty), not a measured zero",
    );
    let cg = cgroup_stats(&reports);
    assert!(cg.avg_off_cpu_pct.is_none(), "mirrors cgroup_stats None");
}

/// Boundary (the OTHER half of the not-measured-vs-measured-zero distinction): a
/// worker with measurable wall time (wall_time_ns > 0) and off_cpu_ns == 0 yields
/// a PRESENT off_cpu_pcts sample of 0.0 — a measured zero, NOT an absent/empty
/// vec. cgroup_stats correspondingly reports `Some(0.0)`, not `None`. A future
/// tightening of the wall_time_ns>0 filter to off_cpu_ns>0 would collapse this
/// into the not-measured case; this pins against that silent regression.
#[test]
fn phase_cgroup_stats_off_cpu_pcts_measured_zero_is_present() {
    // wall 5000 > 0, off 0 -> off-CPU% == 0.0 (measured), not filtered out.
    let reports = vec![rpt(1, 1000, 5000, 0, &[0], 10)];
    let pcs = phase_cgroup_stats(&reports, None);
    assert_eq!(
        pcs.off_cpu_pcts,
        vec![0.0],
        "measured zero is a PRESENT 0.0 sample, distinct from the empty not-measured vec",
    );
    let cg = cgroup_stats(&reports);
    assert_eq!(
        cg.avg_off_cpu_pct,
        Some(0.0),
        "mirrors cgroup_stats Some(0.0), not None"
    );
}

/// Boundary (MIXED): a cgroup with SOME workers wall_time_ns==0 and others
/// wall_time_ns>0. The `wall_time_ns > 0` filter must EXCLUDE the zero-wall
/// workers from off_cpu_pcts (no div-by-zero NaN) while still counting them in
/// num_workers. The surviving sample matches the wall>0 worker, and the re-pool
/// equals cgroup_stats over the same mixed reports value-for-value.
#[test]
fn phase_cgroup_stats_off_cpu_pcts_mixed_filters_zero_wall_workers() {
    let reports = vec![
        rpt(1, 1000, 0, 0, &[0], 0), // wall 0 -> excluded from off_cpu_pcts
        rpt(2, 1000, 1_000_000, 200_000, &[1], 0), // wall>0 -> off-CPU% 20.0
    ];
    let pcs = phase_cgroup_stats(&reports, None);
    assert_eq!(
        pcs.off_cpu_pcts,
        vec![20.0],
        "only the wall>0 worker contributes a sample"
    );
    assert_eq!(
        pcs.num_workers, 2,
        "the zero-wall worker still counts toward num_workers"
    );
    // Re-pool matches cgroup_stats over the same mixed reports.
    let cg = cgroup_stats(&reports);
    let avg = pcs.off_cpu_pcts.iter().sum::<f64>() / pcs.off_cpu_pcts.len() as f64;
    assert!(
        (avg - cg.avg_off_cpu_pct.unwrap()).abs() < 1e-9,
        "avg re-pools to cgroup_stats"
    );
    assert!(
        (20.0 - cg.max_off_cpu_pct.unwrap()).abs() < 1e-9,
        "max matches the lone sample"
    );
}

/// Boundary: without an expected-node set `numa_pages_local` is 0 (mirrors
/// cgroup_stats leaving page_locality 0.0 absent NUMA context), while
/// `numa_pages_total` is still the sum so cross_node_migration_ratio re-pools.
#[test]
fn phase_cgroup_stats_numa_local_zero_without_expected_nodes() {
    let reports = vec![WorkerReport {
        numa_pages: BTreeMap::from([(0usize, 100u64), (1, 40)]),
        ..rpt(1, 1, 1000, 0, &[0], 0)
    }];
    let pcs = phase_cgroup_stats(&reports, None);
    assert_eq!(
        pcs.numa_pages_local, 0,
        "no node set -> 0 numerator (page_locality re-pools to 0.0)",
    );
    assert_eq!(
        pcs.numa_pages_total, 140,
        "total still computed without node context"
    );
}

/// Boundary: the partition counts ONLY pages on the expected nodes, exactly as
/// `AssertPlan::assert_cgroup` does (`nodes.contains` gate, summed across
/// workers and numa_maps node entries).
#[test]
fn phase_cgroup_stats_numa_local_partitions_on_expected_nodes() {
    let reports = vec![
        WorkerReport {
            numa_pages: BTreeMap::from([(0usize, 100u64), (3, 25)]),
            ..rpt(1, 1, 1000, 0, &[0], 0)
        },
        WorkerReport {
            numa_pages: BTreeMap::from([(1usize, 60u64), (2, 15)]),
            ..rpt(2, 1, 1000, 0, &[1], 0)
        },
    ];
    let nodes: BTreeSet<usize> = [0, 1].into_iter().collect();
    let pcs = phase_cgroup_stats(&reports, Some(&nodes));
    // local = node0(100) + node1(60) = 160; off-node 3(25)+2(15) excluded.
    assert_eq!(pcs.numa_pages_local, 160);
    assert_eq!(pcs.numa_pages_total, 200);
}

/// Boundary: `vmstat_numa_pages_migrated` is a SYSTEM-WIDE delta each worker
/// observes redundantly, so the fold is MAX, not SUM (summing would inflate by
/// the worker count).
#[test]
fn phase_cgroup_stats_cross_node_migrated_is_max_not_sum() {
    let reports = vec![
        WorkerReport {
            vmstat_numa_pages_migrated: 30,
            ..rpt(1, 1, 1000, 0, &[0], 0)
        },
        WorkerReport {
            vmstat_numa_pages_migrated: 20,
            ..rpt(2, 1, 1000, 0, &[1], 0)
        },
    ];
    let pcs = phase_cgroup_stats(&reports, None);
    assert_eq!(pcs.cross_node_migrated, 30, "MAX (30), not SUM (50)");
}

/// Boundary: the worst scheduling gap and its CPU are taken TOGETHER from the
/// argmax worker — never two independent maxes (which would pair 90 with cpu 7).
#[test]
fn phase_cgroup_stats_gap_argmax_couples_ms_and_cpu() {
    let reports = vec![
        WorkerReport {
            max_gap_ms: 40,
            max_gap_cpu: 7,
            ..rpt(1, 1, 1000, 0, &[0], 0)
        },
        WorkerReport {
            max_gap_ms: 90,
            max_gap_cpu: 3,
            ..rpt(2, 1, 1000, 0, &[1], 0)
        },
        WorkerReport {
            max_gap_ms: 10,
            max_gap_cpu: 5,
            ..rpt(3, 1, 1000, 0, &[2], 0)
        },
    ];
    let pcs = phase_cgroup_stats(&reports, None);
    assert_eq!(
        (pcs.max_gap_ms, pcs.max_gap_cpu),
        (90, 3),
        "argmax keeps the gap bound to its CPU (independent maxes would give (90, 7))",
    );
}

/// num_workers is a Counter (SUM), not a Peak (MAX): a multi-WorkSpec cgroup
/// (`.work(4).work(2)`) emits ONE carrier per handle under the SAME name
/// (`apply_setup` -> `collect_handles`), and `AssertResult::merge` folds those
/// same-name carriers via `PhaseCgroupStats::merge`. The carriers cover DISJOINT
/// worker subsets, so num_workers must SUM (4 + 2 = 6) to match `cgroup_stats`
/// over the pooled reports (`reports.len()` = 6); a MAX (4) would inflate the
/// re-pooled `iterations_per_worker` 1.5x.
#[test]
fn phase_cgroup_stats_num_workers_sums_across_same_name_carriers() {
    // Two handles for one cgroup: 4 workers (100 iters each) + 2 (200 each).
    let reports1: Vec<WorkerReport> = (0..4)
        .map(|i| WorkerReport {
            iterations: 100,
            ..rpt(i, 1, 1000, 0, &[i as usize], 0)
        })
        .collect();
    let reports2: Vec<WorkerReport> = (4..6)
        .map(|i| WorkerReport {
            iterations: 200,
            ..rpt(i, 1, 1000, 0, &[i as usize], 0)
        })
        .collect();
    let pcs1 = phase_cgroup_stats(&reports1, None);
    let pcs2 = phase_cgroup_stats(&reports2, None);
    assert_eq!(pcs1.num_workers, 4);
    assert_eq!(pcs2.num_workers, 2);
    let merged = PhaseCgroupStats::merge(pcs1, pcs2);
    assert_eq!(
        merged.num_workers, 6,
        "disjoint worker subsets SUM, not MAX(4,2)=4"
    );
    assert_eq!(merged.total_iterations, 4 * 100 + 2 * 200);

    // The merged components re-pool to the cgroup_stats reduction over the
    // POOLED reports (the run-level authority): num_workers = reports.len() = 6,
    // and iterations_per_worker = total_iterations / num_workers value-for-value.
    let mut pooled = reports1.clone();
    pooled.extend(reports2.clone());
    let cg = cgroup_stats(&pooled);
    assert_eq!(merged.num_workers, cg.num_workers);
    assert_eq!(merged.total_iterations, cg.total_iterations);
    let repooled = merged.total_iterations as f64 / merged.num_workers as f64;
    let cg_ipw = cg
        .iterations_per_worker()
        .expect("pooled cgroup has workers");
    assert!(
        (repooled - cg_ipw).abs() < 1e-9,
        "re-pooled iterations_per_worker {repooled} must match cgroup_stats {cg_ipw} \
         (800/6); a MAX num_workers would give 800/4",
    );
}

/// The carrier RE-CAPS the pooled wake_latencies_ns at MAX_WAKE_SAMPLES so the
/// serialized AssertResult stays within the guest bulk-port frame even on a
/// many-core cgroup. Without the re-cap the pool would be
/// workers × MAX_WAKE_SAMPLES (here 120k) and could overrun the 16 MiB frame,
/// flipping a PASS to a truncated FAIL. wake_sample_total keeps the true count.
#[test]
fn phase_cgroup_stats_caps_pooled_wake_latencies() {
    use crate::workload::MAX_WAKE_SAMPLES;
    // DISTINCT values so distribution-preservation is testable: worker 1 carries
    // 0..60k, worker 2 carries 60k..120k -> population 0..120_000 (mean 59999.5),
    // pool 120k > MAX_WAKE_SAMPLES (100k). A constant fill would only pin len().
    let mk = |tid: i32, lo: u64, hi: u64| WorkerReport {
        wake_latencies_ns: (lo..hi).collect(),
        wake_sample_total: hi - lo,
        ..rpt(tid, 1, 1000, 0, &[0], 0)
    };
    let reports = vec![mk(1, 0, 60_000), mk(2, 60_000, 120_000)];
    let pcs = phase_cgroup_stats(&reports, None);
    assert_eq!(
        pcs.wake_latencies_ns.len(),
        MAX_WAKE_SAMPLES,
        "pooled wake_latencies re-capped to MAX_WAKE_SAMPLES (not the 120k concat)",
    );
    assert_eq!(
        pcs.wake_sample_total, 120_000,
        "true pre-cap population preserved for the re-pool",
    );
    // PARITY BOUNDARY (>cap): the carrier holds a reservoir SUBSAMPLE (len == cap)
    // of a larger population while cgroup_stats reduces over the FULL concat — the
    // documented distribution-equivalent (not value-for-value) divergence.
    assert!(
        pcs.wake_latencies_ns.len() < pcs.wake_sample_total as usize,
        "carrier is a subsample of the true population — the >cap divergence",
    );
    // DISTRIBUTION PRESERVATION (the load-bearing >cap claim). A
    // distribution-destroying reservoir bug (first-K, constant overwrite, fixed
    // replacement index) would shift the mean or collapse the range — caught
    // here, where the earlier constant-fill version was blind.
    //
    // (1) STATISTICAL tolerance: a uniform 100k-of-120k reservoir has a sample
    // mean tightly tracking the population mean (59999.5); the ±3000 band is ~27×
    // the ~110-wide sample-mean std error, so a valid reservoir never trips it
    // while a shifted distribution does.
    let mean = pcs.wake_latencies_ns.iter().map(|&v| v as f64).sum::<f64>()
        / pcs.wake_latencies_ns.len() as f64;
    assert!(
        (mean - 59_999.5).abs() < 3000.0,
        "reservoir mean {mean} tracks population mean 59999.5 (statistical tolerance)",
    );
    // (2) DETERMINISTIC bound: the reservoir holds 100k of 120k values, excluding
    // only 20k, so by pigeonhole min ≤ 20000 and max ≥ 99999 → range ≥ 99999
    // ALWAYS. Assert > 90_000 (comfortably inside the guaranteed 99999, never a
    // flake) to catch a constant/degenerate fill.
    let min = *pcs.wake_latencies_ns.iter().min().unwrap();
    let max = *pcs.wake_latencies_ns.iter().max().unwrap();
    assert!(
        max < 120_000,
        "every sample drawn from the 0..120_000 population"
    );
    assert!(
        max - min > 90_000,
        "reservoir spans the population (guaranteed range ≥ 99999), not a constant fill",
    );
}

/// PhaseCgroupStats::merge RE-CAPS the merged wake_latencies_ns at
/// MAX_WAKE_SAMPLES. Same-name carriers (a multi-WorkSpec cgroup's per-handle
/// carriers) merge ON THE GUEST before the AssertResult is serialized, so without
/// the re-cap the merged pool would be K × MAX_WAKE_SAMPLES and could overrun the
/// 16 MiB bulk frame (a PASS flipped to a truncated FAIL). A concat already ≤ cap
/// passes through unchanged (value-for-value parity for small pools).
#[test]
fn phase_cgroup_stats_merge_caps_pooled_wake_latencies() {
    use crate::workload::MAX_WAKE_SAMPLES;
    // DISTINCT value ranges so the merged reservoir's distribution is testable.
    let carrier = |lo: u64, hi: u64| PhaseCgroupStats {
        wake_latencies_ns: (lo..hi).collect(),
        wake_sample_total: hi - lo,
        ..Default::default()
    };
    // Two carriers each AT the cap (0..100k and 100k..200k); their summed length
    // 200k > cap, so this routes through the WEIGHTED merge with EQUAL populations
    // (100k each) -> a ~50/50 with-replacement draw over the two ranges -> mean
    // ≈ 99999.5, re-capped to cap.
    let merged = PhaseCgroupStats::merge(carrier(0, 100_000), carrier(100_000, 200_000));
    assert_eq!(
        merged.wake_latencies_ns.len(),
        MAX_WAKE_SAMPLES,
        "merged wake_latencies re-capped to MAX_WAKE_SAMPLES (not 2×cap)",
    );
    assert_eq!(
        merged.wake_sample_total,
        2 * MAX_WAKE_SAMPLES as u64,
        "true pre-cap population SUMs across carriers",
    );
    // Distribution preservation: the merged reservoir's mean tracks the 0..200_000
    // population mean (99999.5). Equal populations make the weighted merge an
    // unbiased ~50/50 draw, so the mean lands at the midpoint (the
    // population-weighting only diverges from 50/50 when the carriers' true
    // populations differ — see weighted_merge_reservoirs and its tests).
    let mean = merged
        .wake_latencies_ns
        .iter()
        .map(|&v| v as f64)
        .sum::<f64>()
        / merged.wake_latencies_ns.len() as f64;
    assert!(
        (mean - 99_999.5).abs() < 3000.0,
        "merged reservoir mean {mean} tracks population mean 99999.5",
    );
    // A merge whose concat is ≤ cap passes through unchanged (no re-sample).
    let small = PhaseCgroupStats::merge(carrier(0, 10), carrier(100, 120));
    assert_eq!(
        small.wake_latencies_ns.len(),
        30,
        "≤cap concat not re-sampled"
    );
}

/// The >cap merge weights by true POPULATION (`wake_sample_total`), not reservoir
/// LENGTH. Two already-capped reservoirs of EQUAL length (cap each) but a 10:1
/// population ratio (A: 1_000_000 wakes, B: 100_000) merge to a sample whose
/// A-fraction tracks `w_a/(w_a+w_b) = 0.909` — NOT the 0.5 a length-weighted
/// concat-and-re-cap (the deleted unweighted reservoir-of-reservoirs) gave. This
/// is the smaller-population skew the weighted merge removes.
#[test]
fn weighted_merge_reservoirs_weights_by_population_not_length() {
    use crate::workload::MAX_WAKE_SAMPLES as CAP;
    // Disjoint value ranges so each merged element's source is identifiable.
    let a: Vec<u64> = (0..CAP as u64).collect(); // range [0, CAP), pre-cap pop 1M
    let b: Vec<u64> = (0..CAP as u64).map(|i| 10_000_000 + i).collect(); // [10M, …), pop 100k
    let merged = PhaseCgroupStats::weighted_merge_reservoirs(&a, 1_000_000, &b, 100_000, CAP);
    assert_eq!(merged.len(), CAP, "merged to exactly cap slots");
    let from_a = merged.iter().filter(|&&v| v < 10_000_000).count() as f64 / CAP as f64;
    assert!(
        (from_a - 0.909).abs() < 0.03,
        "merged A-fraction {from_a} tracks population ratio 1M/1.1M=0.909, NOT a \
         length-weighted 0.5",
    );
}

/// The weighted merge is DETERMINISTIC — a pure function of its inputs (seeded
/// from populations + lengths, NOT gettid like reservoir_push). Two calls with
/// identical inputs produce byte-identical output, so a merge run twice in one
/// process is stable (the property `reservoir_push`'s thread-local stream lacks).
#[test]
fn weighted_merge_reservoirs_is_deterministic() {
    use crate::workload::MAX_WAKE_SAMPLES as CAP;
    let a: Vec<u64> = (0..CAP as u64).collect();
    let b: Vec<u64> = (0..CAP as u64).map(|i| 10_000_000 + i).collect();
    let m1 = PhaseCgroupStats::weighted_merge_reservoirs(&a, 700_000, &b, 300_000, CAP);
    let m2 = PhaseCgroupStats::weighted_merge_reservoirs(&a, 700_000, &b, 300_000, CAP);
    assert_eq!(m1, m2, "weighted merge is a pure function of its inputs");
}

/// Equal populations → ~50/50: the weighted merge degrades to the balanced split
/// exactly when the weights are equal (which is when the old length-weighting was
/// correct). Pins that the weighting does not skew the balanced case.
#[test]
fn weighted_merge_reservoirs_equal_population_is_symmetric() {
    use crate::workload::MAX_WAKE_SAMPLES as CAP;
    let a: Vec<u64> = (0..CAP as u64).collect();
    let b: Vec<u64> = (0..CAP as u64).map(|i| 10_000_000 + i).collect();
    let merged = PhaseCgroupStats::weighted_merge_reservoirs(&a, 500_000, &b, 500_000, CAP);
    let from_a = merged.iter().filter(|&&v| v < 10_000_000).count() as f64 / CAP as f64;
    assert!(
        (from_a - 0.5).abs() < 0.03,
        "equal populations -> ~50/50, got {from_a}"
    );
}

/// Strongly asymmetric weights (≈167:1) drive the merge almost entirely to the
/// larger-population source — the merged A-fraction tracks `w_a/(w_a+w_b)`, not
/// the inputs' equal lengths. The "mixed" case where one carrier's pre-cap
/// population dwarfs the other's.
#[test]
fn weighted_merge_reservoirs_asymmetric_weights_favor_larger_population() {
    use crate::workload::MAX_WAKE_SAMPLES as CAP;
    let a: Vec<u64> = (0..CAP as u64).collect(); // tiny pre-cap population
    let b: Vec<u64> = (0..CAP as u64).map(|i| 10_000_000 + i).collect(); // huge population
    let merged = PhaseCgroupStats::weighted_merge_reservoirs(&a, 60_000, &b, 10_000_000, CAP);
    let from_a = merged.iter().filter(|&&v| v < 10_000_000).count() as f64 / CAP as f64;
    // w_a/(w_a+w_b) = 60_000 / 10_060_000 ≈ 0.006.
    assert!(
        from_a < 0.02,
        "tiny-population A nearly excluded, got A-fraction {from_a}"
    );
}

/// Integration through `PhaseCgroupStats::merge`: two >cap carriers (each at the
/// per-carrier cap) with a 10:1 `wake_sample_total` ratio merge to a
/// population-weighted sample, and `wake_sample_total` SUMs. Pins that merge
/// routes >cap pools through the weighted merge, not the old length-weighted path.
#[test]
fn phase_cgroup_stats_merge_above_cap_is_population_weighted() {
    use crate::workload::MAX_WAKE_SAMPLES as CAP;
    let carrier = |base: u64, pop: u64| PhaseCgroupStats {
        wake_latencies_ns: (0..CAP as u64).map(|i| base + i).collect(),
        wake_sample_total: pop,
        ..Default::default()
    };
    let merged = PhaseCgroupStats::merge(carrier(0, 1_000_000), carrier(10_000_000, 100_000));
    assert_eq!(merged.wake_latencies_ns.len(), CAP, "re-capped to cap");
    assert_eq!(merged.wake_sample_total, 1_100_000, "true populations SUM");
    let from_a = merged
        .wake_latencies_ns
        .iter()
        .filter(|&&v| v < 10_000_000)
        .count() as f64
        / CAP as f64;
    assert!(
        (from_a - 0.909).abs() < 0.03,
        "merge weights >cap pools by population (A-fraction {from_a} ~ 1M/1.1M), \
         not 50/50 by length",
    );
}

/// Sequential 3-way merge preserves the population proportions:
/// `merge(merge(A,B),C)` yields A:B:C fractions tracking `w_x / (w_a+w_b+w_c)`.
/// The first merge sets the combined carrier's `wake_sample_total = w_a+w_b`,
/// which then weights against `w_c` in the second merge, so the recursive
/// weighting telescopes to the true 3-way proportions (the case a cgroup with ≥3
/// >cap WorkSpec handles in one step hits). Tolerance is widened slightly for the
/// compounded resampling variance of two sequential draws; the length-weighted
/// alternative (the deleted path) would give ~0.33 each — far outside it.
#[test]
fn weighted_merge_reservoirs_sequential_three_way_preserves_proportions() {
    use crate::workload::MAX_WAKE_SAMPLES as CAP;
    // Disjoint ranges per source; populations 600k : 300k : 100k -> 0.6 : 0.3 : 0.1.
    let carrier = |base: u64, pop: u64| PhaseCgroupStats {
        wake_latencies_ns: (0..CAP as u64).map(|i| base + i).collect(),
        wake_sample_total: pop,
        ..Default::default()
    };
    let ab = PhaseCgroupStats::merge(carrier(0, 600_000), carrier(10_000_000, 300_000));
    let abc = PhaseCgroupStats::merge(ab, carrier(20_000_000, 100_000));
    assert_eq!(abc.wake_latencies_ns.len(), CAP);
    assert_eq!(
        abc.wake_sample_total, 1_000_000,
        "true populations SUM across all three"
    );
    let frac = |lo: u64, hi: u64| {
        abc.wake_latencies_ns
            .iter()
            .filter(|&&v| v >= lo && v < hi)
            .count() as f64
            / CAP as f64
    };
    let (a, b, c) = (
        frac(0, 10_000_000),
        frac(10_000_000, 20_000_000),
        frac(20_000_000, 30_000_000),
    );
    assert!(
        (a - 0.6).abs() < 0.04,
        "A-fraction {a} ~ 600k/1M = 0.6 (not length-weighted 0.33)"
    );
    assert!((b - 0.3).abs() < 0.04, "B-fraction {b} ~ 300k/1M = 0.3");
    assert!((c - 0.1).abs() < 0.04, "C-fraction {c} ~ 100k/1M = 0.1");
}

/// `AssertResult::strip_phase_cgroup_samples` — the graceful-degradation lever
/// `send_test_result` uses when the serialized result would overrun the bulk
/// frame: it drops every per_cgroup carrier's RAW sample vectors but PRESERVES
/// the counters, `wake_sample_total`, the coupled gap, and the verdict (no
/// PASS→FAIL flip). Pins the drop scope + the dropped-sample count.
#[test]
fn strip_phase_cgroup_samples_drops_only_sample_vecs_preserving_verdict() {
    let mut pc = BTreeMap::new();
    pc.insert(
        "cg".to_string(),
        PhaseCgroupStats {
            num_workers: 3,
            wake_latencies_ns: vec![1, 2, 3],
            wake_sample_total: 99,
            run_delays_ns: vec![10, 20],
            off_cpu_pcts: vec![5.0],
            total_iterations: 42,
            max_gap_ms: 7,
            max_gap_cpu: 2,
            ..Default::default()
        },
    );
    let mut r = crate::assert::AssertResult::pass();
    r.stats.phases = vec![PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        metrics: BTreeMap::new(),
        per_cgroup: pc,
    }];
    let dropped = r.strip_phase_cgroup_samples();
    assert_eq!(
        dropped,
        3 + 2 + 1,
        "wake(3) + run(2) + off(1) samples dropped"
    );
    let cg = &r.stats.phases[0].per_cgroup["cg"];
    assert!(cg.wake_latencies_ns.is_empty(), "wake samples dropped");
    assert!(cg.run_delays_ns.is_empty(), "run-delay samples dropped");
    assert!(cg.off_cpu_pcts.is_empty(), "off-CPU samples dropped");
    // Counters + reduced fields preserved.
    assert_eq!(cg.num_workers, 3);
    assert_eq!(cg.wake_sample_total, 99, "true population preserved");
    assert_eq!(cg.total_iterations, 42);
    assert_eq!((cg.max_gap_ms, cg.max_gap_cpu), (7, 2));
    assert!(r.is_pass(), "verdict preserved (no PASS->FAIL flip)");
}

/// `strip_phase_cgroup_samples` marks a carrier `stripped` ONLY when it actually
/// dropped samples, so the render distinguishes a size-limit drop from a carrier
/// that genuinely measured nothing — a carrier with no sample vectors stays
/// not-stripped through a strip.
#[test]
fn strip_phase_cgroup_samples_marks_stripped_only_on_carriers_with_samples() {
    let mut pc = BTreeMap::new();
    pc.insert(
        "has_samples".to_string(),
        PhaseCgroupStats {
            wake_latencies_ns: vec![1, 2],
            total_iterations: 10,
            ..Default::default()
        },
    );
    pc.insert(
        "empty".to_string(),
        // No sample vectors — genuinely measured nothing.
        PhaseCgroupStats {
            total_iterations: 5,
            ..Default::default()
        },
    );
    let mut r = crate::assert::AssertResult::pass();
    r.stats.phases = vec![PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        metrics: BTreeMap::new(),
        per_cgroup: pc,
    }];
    r.strip_phase_cgroup_samples();
    assert!(
        r.stats.phases[0].per_cgroup["has_samples"].stripped,
        "a carrier that HAD samples is marked stripped after the drop",
    );
    assert!(
        !r.stats.phases[0].per_cgroup["empty"].stripped,
        "a carrier that measured nothing stays not-stripped (distinct from a size-limit drop)",
    );
}

/// `PhaseCgroupStats::merge` ORs `stripped`: a merged carrier is stripped if
/// EITHER input was, so the render flag survives folding a stripped carrier with
/// a fresh one (in any order).
#[test]
fn phase_cgroup_stats_merge_ors_stripped() {
    let stripped = PhaseCgroupStats {
        stripped: true,
        ..Default::default()
    };
    let fresh = PhaseCgroupStats {
        stripped: false,
        ..Default::default()
    };
    assert!(
        PhaseCgroupStats::merge(stripped.clone(), fresh.clone()).stripped,
        "stripped | fresh = stripped",
    );
    assert!(
        PhaseCgroupStats::merge(fresh.clone(), stripped).stripped,
        "fresh | stripped = stripped (order-independent)",
    );
    assert!(
        !PhaseCgroupStats::merge(fresh.clone(), fresh).stripped,
        "fresh | fresh = not stripped",
    );
}

/// PhaseCgroupStats::merge breaks a max_gap_ms TIE toward `b` (last-wins),
/// matching the builders' `max_by_key` (which returns the LAST max on a tie). On
/// equal gaps with different CPUs the later carrier's CPU wins — keeping the
/// cross-carrier merge consistent with a single cgroup_stats over pooled reports.
#[test]
fn phase_cgroup_stats_merge_gap_tie_breaks_to_b() {
    let a = PhaseCgroupStats {
        max_gap_ms: 5,
        max_gap_cpu: 3,
        ..Default::default()
    };
    let b = PhaseCgroupStats {
        max_gap_ms: 5,
        max_gap_cpu: 7,
        ..Default::default()
    };
    let merged = PhaseCgroupStats::merge(a, b);
    assert_eq!(
        (merged.max_gap_ms, merged.max_gap_cpu),
        (5, 7),
        "equal gap -> b (last) wins, matching max_by_key last-wins",
    );
}

/// Cross-carrier gap parity: merging two same-name carriers with EQUAL max_gap_ms
/// (different CPU) yields the SAME (ms, cpu) as cgroup_stats over the concatenated
/// reports IN FOLD ORDER — pinning that the merge's last-wins tie-break stays
/// coupled to the pooled-report order (a reordered fold would desync the CPU).
#[test]
fn phase_cgroup_stats_merge_gap_tie_matches_pooled_cgroup_stats() {
    let r1 = vec![WorkerReport {
        max_gap_ms: 8,
        max_gap_cpu: 1,
        ..rpt(1, 1, 1000, 0, &[1], 0)
    }];
    let r2 = vec![WorkerReport {
        max_gap_ms: 8,
        max_gap_cpu: 9,
        ..rpt(2, 1, 1000, 0, &[9], 0)
    }];
    let merged =
        PhaseCgroupStats::merge(phase_cgroup_stats(&r1, None), phase_cgroup_stats(&r2, None));
    // cgroup_stats over the concatenation (r1 ++ r2, fold order): max_by_key
    // returns the LAST max on the gap-8 tie -> cpu 9.
    let mut pooled = r1.clone();
    pooled.extend(r2.clone());
    let cg = cgroup_stats(&pooled);
    assert_eq!(
        (merged.max_gap_ms, merged.max_gap_cpu),
        (cg.max_gap_ms, cg.max_gap_cpu),
        "cross-carrier gap argmax matches cgroup_stats over pooled reports in fold order",
    );
    assert_eq!((merged.max_gap_ms, merged.max_gap_cpu), (8, 9));
}

/// The guest-side carrier: keyed by cgroup name, stamped with the 1-indexed
/// step_index, labeled via [`Phase`] Display, with the merge-neutral
/// `(u64::MAX, 0)` window and empty metrics — it carries ONLY per_cgroup.
#[test]
fn step_per_cgroup_bucket_keys_by_name_with_sentinel_window() {
    let reports = vec![WorkerReport {
        iterations: 10,
        schedstat_cpu_time_ns: 500,
        ..rpt(1, 1, 1000, 100, &[0], 5)
    }];
    let nodes: BTreeSet<usize> = [0].into_iter().collect();
    let b = step_per_cgroup_bucket("cg_step", &reports, Some(&nodes), 3);
    assert_eq!(b.step_index, 3);
    assert_eq!(
        b.label, "Step[2]",
        "Phase Display: 1-indexed step_index 3 -> Step[2]"
    );
    assert_eq!(
        b.start_ms,
        u64::MAX,
        "merge-neutral: min() against host start is a no-op"
    );
    assert_eq!(
        b.end_ms, 0,
        "merge-neutral: max() against host end is a no-op"
    );
    assert_eq!(b.sample_count, 0);
    assert!(b.metrics.is_empty(), "carrier contributes only per_cgroup");
    assert_eq!(b.per_cgroup.len(), 1);
    let pc = &b.per_cgroup["cg_step"];
    assert_eq!(pc.total_iterations, 10);
    assert_eq!(pc.num_workers, 1);
}

/// The carrier's label for the BASELINE encoding (step_index 0) reads
/// "BASELINE", pinning the 1-indexed Phase encoding at the boundary.
#[test]
fn step_per_cgroup_bucket_baseline_label() {
    let b = step_per_cgroup_bucket("cg", &[], None, 0);
    assert_eq!(b.label, "BASELINE");
    assert_eq!(b.step_index, 0);
    assert_eq!(
        b.per_cgroup["cg"].num_workers, 0,
        "empty reports -> zero-component carrier"
    );
}

/// The fold unions a guest carrier's per_cgroup into the host bucket of the
/// SAME step_index, leaving the host's window / metrics / sample_count intact
/// (the carrier's MAX/0 window and empty metrics are merge-neutral).
#[test]
fn fold_unions_guest_per_cgroup_into_matching_host_bucket() {
    // Host bucket as build_phase_buckets_with_stimulus produces it: the
    // iteration_rate Rate ALONGSIDE its two Counter components (50 = 1000/20),
    // plus a plain non-rate metric. merge_matched_phase_buckets skips Rate keys
    // and re-derives them from the components, so a realistic host bucket keeps
    // its rate; the components must be present (a bare rate would be dropped).
    let host = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 100,
        end_ms: 200,
        sample_count: 3,
        metrics: BTreeMap::from([
            ("iteration_rate".to_string(), 50.0),
            ("total_phase_iterations".to_string(), 1000.0),
            ("total_phase_duration_sec".to_string(), 20.0),
            ("worst_spread".to_string(), 0.42),
        ]),
        per_cgroup: BTreeMap::new(),
    };
    let mut g_pc = BTreeMap::new();
    g_pc.insert(
        "cgA".to_string(),
        PhaseCgroupStats {
            total_iterations: 42,
            ..Default::default()
        },
    );
    let guest = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: BTreeMap::new(),
        per_cgroup: g_pc,
    };
    let out = fold_guest_per_cgroup_into_host_buckets(vec![host], vec![guest]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].step_index, 1);
    assert_eq!(
        out[0].start_ms, 100,
        "host window start preserved (min vs MAX)"
    );
    assert_eq!(out[0].end_ms, 200, "host window end preserved (max vs 0)");
    assert_eq!(out[0].sample_count, 3, "host sample_count preserved (+0)");
    assert_eq!(
        out[0].metrics.get("worst_spread").copied(),
        Some(0.42),
        "host non-rate metric preserved through the carrier merge",
    );
    assert_eq!(
        out[0].metrics.get("iteration_rate").copied(),
        Some(50.0),
        "host Rate re-derived from its carried components (1000/20), not dropped",
    );
    assert_eq!(
        out[0].per_cgroup["cgA"].total_iterations, 42,
        "guest per_cgroup unioned in"
    );
}

/// Negative direction of the Rate-survival contract: a host bucket carrying a
/// Rate metric WITHOUT its Counter components LOSES the rate through the carrier
/// merge — merge_matched_phase_buckets skips Rate keys and re-derives them, and
/// derive_rate_metrics cannot re-derive from absent components. This drop is
/// SAFE in production only because build_phase_buckets_with_stimulus co-inserts
/// total_phase_iterations + total_phase_duration_sec alongside iteration_rate
/// (and iterations_per_cpu_sec lands only in ext_metrics, never PhaseBucket
/// metrics), so no production host bucket carries a component-less Rate. This
/// test pins the failure mode so a future producer violating that contract
/// surfaces here rather than as a silently-dropped metric.
#[test]
fn fold_drops_host_rate_lacking_its_components() {
    let host = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        // iteration_rate WITHOUT its total_phase_iterations /
        // total_phase_duration_sec components — a contract violation no
        // production path produces.
        metrics: BTreeMap::from([("iteration_rate".to_string(), 50.0)]),
        per_cgroup: BTreeMap::new(),
    };
    let mut g_pc = BTreeMap::new();
    g_pc.insert(
        "cg".to_string(),
        PhaseCgroupStats {
            total_iterations: 1,
            ..Default::default()
        },
    );
    let guest = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: BTreeMap::new(),
        per_cgroup: g_pc,
    };
    let out = fold_guest_per_cgroup_into_host_buckets(vec![host], vec![guest]);
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].metrics.get("iteration_rate"),
        None,
        "a component-less Rate is dropped by the carrier merge's re-derive; \
         production never produces this (build_phase_buckets co-inserts the \
         components), so the drop is unreachable in practice",
    );
    // The per_cgroup payload still folds in regardless of the dropped Rate.
    assert_eq!(out[0].per_cgroup["cg"].total_iterations, 1);
}

/// A guest step_index with no host bucket — the DEFENSIVE case where the
/// carrier's step has no StepStart frame in the host stimulus (the host
/// synthesizes a bucket for every StepStart-step, so a captured-but-short step
/// takes the matched arm, not this) — is carried verbatim, NOT dropped, with its
/// window normalized to (0, 0) so duration consumers don't underflow the
/// merge-neutral sentinel. Output stays sorted by step_index.
#[test]
fn fold_carries_orphan_guest_step_index_with_normalized_window() {
    let host = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        metrics: BTreeMap::new(),
        per_cgroup: BTreeMap::new(),
    };
    let mut g_pc = BTreeMap::new();
    g_pc.insert(
        "cgB".to_string(),
        PhaseCgroupStats {
            total_migrations: 7,
            ..Default::default()
        },
    );
    let guest = PhaseBucket {
        step_index: 5,
        label: "Step[4]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: BTreeMap::new(),
        per_cgroup: g_pc,
    };
    let out = fold_guest_per_cgroup_into_host_buckets(vec![host], vec![guest]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].step_index, 1, "sorted by step_index");
    assert_eq!(out[1].step_index, 5);
    let orphan = &out[1];
    assert_eq!(
        orphan.start_ms, 0,
        "orphan window normalized (no underflow)"
    );
    assert_eq!(orphan.end_ms, 0, "orphan window normalized");
    assert_eq!(orphan.label, "Step[4]", "orphan carrier label preserved");
    assert_eq!(
        orphan.per_cgroup["cgB"].total_migrations, 7,
        "orphan per_cgroup not dropped"
    );

    // Saturation boundary: a >65k-step scenario collapses step_index to u16::MAX
    // (the step loop + build_stimulus both saturate). An orphan there is carried
    // identically — (0,0) window, verbatim per_cgroup — since the fold's
    // BTreeMap<u16,_> keys any value, the sentinel included.
    let mut sat_pc = BTreeMap::new();
    sat_pc.insert(
        "cgSat".to_string(),
        PhaseCgroupStats {
            total_iterations: 3,
            ..Default::default()
        },
    );
    let sat_guest = PhaseBucket {
        step_index: u16::MAX,
        label: "Step[65534]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: BTreeMap::new(),
        per_cgroup: sat_pc,
    };
    let sat_out = fold_guest_per_cgroup_into_host_buckets(vec![], vec![sat_guest]);
    assert_eq!(sat_out.len(), 1);
    assert_eq!(sat_out[0].step_index, u16::MAX);
    assert_eq!(
        (sat_out[0].start_ms, sat_out[0].end_ms),
        (0, 0),
        "u16::MAX orphan window normalized",
    );
    assert_eq!(
        sat_out[0].per_cgroup["cgSat"].total_iterations, 3,
        "u16::MAX orphan per_cgroup carried verbatim",
    );
}

/// With NO guest carriers (a run with no step-local cgroups) the fold returns
/// the host buckets unchanged — the prior behavior, which the
/// `phase_buckets_equals_stats_phases*` eval pins depend on.
#[test]
fn fold_empty_guest_passes_host_through_unchanged() {
    let host = vec![
        PhaseBucket {
            step_index: 0,
            label: "BASELINE".to_string(),
            start_ms: 0,
            end_ms: 50,
            sample_count: 2,
            metrics: BTreeMap::from([("k".to_string(), 1.0)]),
            per_cgroup: BTreeMap::new(),
        },
        PhaseBucket {
            step_index: 1,
            label: "Step[0]".to_string(),
            start_ms: 50,
            end_ms: 100,
            sample_count: 3,
            metrics: BTreeMap::new(),
            per_cgroup: BTreeMap::new(),
        },
    ];
    let out = fold_guest_per_cgroup_into_host_buckets(host.clone(), vec![]);
    assert_eq!(out, host, "no guest carriers -> host buckets unchanged");
}

/// Multiple cgroups in one step land in a single carrier bucket (collect_handles
/// merges them); the fold carries all of them into the host bucket.
#[test]
fn fold_multiple_cgroups_in_one_step_all_carried() {
    let mut g_pc = BTreeMap::new();
    g_pc.insert(
        "cgX".to_string(),
        PhaseCgroupStats {
            total_iterations: 10,
            ..Default::default()
        },
    );
    g_pc.insert(
        "cgY".to_string(),
        PhaseCgroupStats {
            total_iterations: 20,
            ..Default::default()
        },
    );
    let guest = PhaseBucket {
        step_index: 2,
        label: "Step[1]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: BTreeMap::new(),
        per_cgroup: g_pc,
    };
    let host = PhaseBucket {
        step_index: 2,
        label: "Step[1]".to_string(),
        start_ms: 0,
        end_ms: 10,
        sample_count: 1,
        metrics: BTreeMap::new(),
        per_cgroup: BTreeMap::new(),
    };
    let out = fold_guest_per_cgroup_into_host_buckets(vec![host], vec![guest]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].per_cgroup.len(), 2);
    assert_eq!(out[0].per_cgroup["cgX"].total_iterations, 10);
    assert_eq!(out[0].per_cgroup["cgY"].total_iterations, 20);
}

/// Defensive: two guest carriers sharing a step_index (the serialized guest
/// result is already per-step-merged, but the fold must not double-count or drop
/// either) merge their same-named per_cgroup components sequentially.
#[test]
fn fold_duplicate_guest_step_index_merges_sequentially() {
    let make = |iters: u64| {
        let mut pc = BTreeMap::new();
        pc.insert(
            "cgZ".to_string(),
            PhaseCgroupStats {
                total_iterations: iters,
                ..Default::default()
            },
        );
        PhaseBucket {
            step_index: 3,
            label: "Step[2]".to_string(),
            start_ms: u64::MAX,
            end_ms: 0,
            sample_count: 0,
            metrics: BTreeMap::new(),
            per_cgroup: pc,
        }
    };
    let out = fold_guest_per_cgroup_into_host_buckets(vec![], vec![make(5), make(8)]);
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].per_cgroup["cgZ"].total_iterations, 13,
        "5 + 8 summed (Counter): neither carrier dropped or double-counted",
    );
    // The orphan window stays normalized (0,0) through the re-merge: the first
    // carrier hits the orphan arm and is normalized to (0,0) BEFORE the second
    // carrier merges via min(0, MAX)=0 / max(0, 0)=0 — so the MAX sentinel never
    // reaches a duration consumer.
    assert_eq!(
        (out[0].start_ms, out[0].end_ms),
        (0, 0),
        "orphan window normalized"
    );
}

/// Defensive boundary: a guest carrier MUST carry the merge-neutral
/// `(u64::MAX, 0)` sentinel window. The fold validates this BEFORE the
/// matched/orphan dispatch, so BOTH arms are guarded — the matched arm relies on
/// the window being merge-neutral (min/max no-ops against the host window) and
/// the orphan arm normalizes it to (0,0). A carrier with a real window matching a
/// host bucket would otherwise silently corrupt the merged window; the
/// `debug_assert!` trips loudly in test builds instead. Host bucket present at the
/// same step_index so the carrier WOULD take the (dangerous) matched arm — the
/// assert fires first.
#[test]
#[should_panic(expected = "guest carrier must carry the merge-neutral")]
fn fold_panics_on_non_sentinel_guest_window() {
    let host = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        metrics: BTreeMap::new(),
        per_cgroup: BTreeMap::new(),
    };
    let mut g_pc = BTreeMap::new();
    g_pc.insert(
        "cg".to_string(),
        PhaseCgroupStats {
            total_iterations: 1,
            ..Default::default()
        },
    );
    // A guest carrier with a REAL window (not the (u64::MAX, 0) sentinel) — the
    // step_per_cgroup_bucket invariant violated.
    let bad = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 100,
        end_ms: 200,
        sample_count: 0,
        metrics: BTreeMap::new(),
        per_cgroup: g_pc,
    };
    let _ = fold_guest_per_cgroup_into_host_buckets(vec![host], vec![bad]);
}

/// Defensive boundary: host buckets MUST have unique step_index. The fold keys
/// them into a `BTreeMap<u16, _>`, so a duplicate would silently MERGE one bucket
/// into the other (a dropped phase, not a panic, in release); the
/// `debug_assert_eq!` on the map size vs the input count trips loudly in test
/// builds instead.
#[test]
#[should_panic(expected = "host buckets must have unique step_index")]
fn fold_panics_on_duplicate_host_step_index() {
    let dup = || PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        metrics: BTreeMap::new(),
        per_cgroup: BTreeMap::new(),
    };
    let _ = fold_guest_per_cgroup_into_host_buckets(vec![dup(), dup()], vec![]);
}
