use super::*;

// -- per-phase per-cgroup display-time reductions (PhaseCgroupStats) --

/// `off_cpu_summary` boundary contract: `None` on empty (NOT measured), `Some`
/// on data — INCLUDING a measured zero, distinct from the `None` state.
/// avg/min/max/spread mirror cgroup_stats's off-CPU reduction (spread=max-min).
#[test]
fn phase_cgroup_off_cpu_summary_boundaries() {
    assert_eq!(PhaseCgroupStats::default().off_cpu_summary(), None);
    let one = PhaseCgroupStats {
        off_cpu_pcts: vec![42.0],
        ..Default::default()
    };
    assert_eq!(one.off_cpu_summary(), Some((42.0, 42.0, 42.0, 0.0)));
    let zeros = PhaseCgroupStats {
        off_cpu_pcts: vec![0.0, 0.0],
        ..Default::default()
    };
    assert_eq!(
        zeros.off_cpu_summary(),
        Some((0.0, 0.0, 0.0, 0.0)),
        "measured zeros are Some((0,..)), distinct from None (not-measured)",
    );
    let multi = PhaseCgroupStats {
        off_cpu_pcts: vec![10.0, 20.0, 30.0],
        ..Default::default()
    };
    assert_eq!(multi.off_cpu_summary(), Some((20.0, 10.0, 30.0, 20.0)));
}

/// `wake_summary` boundary: `None` on empty; single-sample p99 == median == the
/// sole sample (µs); nearest-rank percentile, ns→µs once.
#[test]
fn phase_cgroup_wake_summary_boundaries() {
    assert_eq!(PhaseCgroupStats::default().wake_summary(), None);
    let one = PhaseCgroupStats {
        wake_latencies_ns: vec![5000],
        ..Default::default()
    };
    assert_eq!(one.wake_summary(), Some((5.0, 5.0)));
    // 1000..10000ns: p99 nearest-rank idx ceil(10*0.99)-1=9 -> 10000ns=10µs;
    // median idx ceil(10*0.5)-1=4 -> 5000ns=5µs.
    let ten = PhaseCgroupStats {
        wake_latencies_ns: (1..=10u64).map(|v| v * 1000).collect(),
        ..Default::default()
    };
    assert_eq!(ten.wake_summary(), Some((10.0, 5.0)));
}

/// `run_delay_summary` boundary: `None` on empty; divides raw ns→µs ONCE
/// (mean 200µs / worst 300µs over [100_000, 300_000] ns) — not double-divided
/// (0.2) or un-divided (100_000).
#[test]
fn phase_cgroup_run_delay_summary_boundaries() {
    assert_eq!(PhaseCgroupStats::default().run_delay_summary(), None);
    let two = PhaseCgroupStats {
        run_delays_ns: vec![100_000, 300_000],
        ..Default::default()
    };
    assert_eq!(two.run_delay_summary(), Some((200.0, 300.0)));
}

/// Parity: a carrier built from the SAME reports cgroup_stats reduces yields
/// off-cpu / wake / run-delay summaries equal value-for-value (≤cap) to
/// cgroup_stats's fields — the per-phase render reproduces the whole-run
/// reduction when the phase spans the whole run.
#[test]
fn phase_cgroup_summaries_match_cgroup_stats() {
    let reports = vec![
        WorkerReport {
            wake_latencies_ns: vec![1000, 2000, 3000, 4000, 5000],
            schedstat_run_delay_ns: 7000,
            iterations: 100,
            ..rpt(1, 1000, 10_000_000, 4_000_000, &[0], 0)
        },
        WorkerReport {
            wake_latencies_ns: vec![6000, 7000, 8000, 9000, 10000],
            schedstat_run_delay_ns: 3000,
            iterations: 100,
            ..rpt(2, 1000, 10_000_000, 1_000_000, &[1], 0)
        },
    ];
    let cg = cgroup_stats(&reports);
    let carrier = phase_cgroup_stats(&reports, None);
    let (avg, min, max, spread) = carrier.off_cpu_summary().expect("off-cpu measured");
    assert!((avg - cg.avg_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((min - cg.min_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((max - cg.max_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((spread - cg.spread.unwrap()).abs() < 1e-9);
    let (p99, median) = carrier.wake_summary().expect("wake measured");
    assert!((p99 - cg.p99_wake_latency_us).abs() < 1e-9);
    assert!((median - cg.median_wake_latency_us).abs() < 1e-9);
    let (mean, worst) = carrier.run_delay_summary().expect("run-delay measured");
    assert!((mean - cg.mean_run_delay_us).abs() < 1e-9);
    assert!((worst - cg.worst_run_delay_us).abs() < 1e-9);
}

/// run_delay_summary's mean is f64-ULP-equivalent (not bit-exact) to
/// cgroup_stats — Σns/n/1000 vs Σ(ns/1000)/n reassociate differently. This
/// pins the documented 1e-9 bound with a DIVERGENT input (the value-for-value
/// parity test above uses run-delays that are bit-exact in BOTH reassociations,
/// so its tolerance is dead): these three schedstat_run_delay_ns values make
/// the two means differ at the float level (~1e-12), so the < 1e-9 assert is
/// load-bearing — a reassociation/precision regression would exceed it.
#[test]
fn phase_cgroup_run_delay_mean_within_ulp_of_cgroup_stats() {
    let reports: Vec<WorkerReport> = [8_865_093u64, 9_991_834, 9_627_760]
        .iter()
        .enumerate()
        .map(|(i, &rd)| WorkerReport {
            schedstat_run_delay_ns: rd,
            ..rpt(i as i32 + 1, 1000, 1_000_000, 0, &[i], 0)
        })
        .collect();
    let cg = cgroup_stats(&reports);
    let carrier = phase_cgroup_stats(&reports, None);
    let (mean, _worst) = carrier.run_delay_summary().expect("run-delay measured");
    let delta = (mean - cg.mean_run_delay_us).abs();
    assert!(
        delta < 1e-9,
        "mean within 1e-9 of cgroup_stats; delta={delta:e}"
    );
    assert!(
        delta > 0.0,
        "inputs must actually DIVERGE so the 1e-9 tolerance is load-bearing, not dead; delta={delta:e}",
    );
}

/// Carrier-name dedup across STEPS: a cgroup NAME that carries samples in
/// ANY phase is in `*_carriers`, so EVERY `stats.cgroups` entry with that
/// name — including a separate (handle, step) entry merged in for a later
/// step, as `AssertResult::merge` produces — is skipped from the
/// carrier-less worst-wins fold. The pooled percentile stays the union of
/// the carried samples; bogus typed reductions on the same-named cgroups
/// entries never leak in. Pins the disjointness-holds-within-a-step-not-
/// across-steps contract documented on populate_run_distribution_metrics.
#[test]
fn repool_distribution_carrier_name_dedup_skips_same_name_cgroups_across_steps() {
    let step0 = PhaseCgroupStats {
        wake_latencies_ns: vec![1000, 2000],
        wake_sample_total: 2,
        ..PhaseCgroupStats::default()
    };
    let step1 = PhaseCgroupStats {
        wake_latencies_ns: vec![3000, 4000],
        wake_sample_total: 2,
        ..PhaseCgroupStats::default()
    };
    let mut b0 = PhaseBucket {
        step_index: 1,
        ..PhaseBucket::default()
    };
    b0.per_cgroup.insert("a".to_string(), step0);
    let mut b1 = PhaseBucket {
        step_index: 2,
        ..PhaseBucket::default()
    };
    b1.per_cgroup.insert("a".to_string(), step1);
    // Two same-named cgroups entries (one per step, as AssertResult::merge
    // produces) carrying BOGUS huge p99 reductions that must NOT leak in.
    let cgroups = vec![
        CgroupStats {
            cgroup_name: "a".to_string(),
            p99_wake_latency_us: 9999.0,
            ..CgroupStats::default()
        },
        CgroupStats {
            cgroup_name: "a".to_string(),
            p99_wake_latency_us: 8888.0,
            ..CgroupStats::default()
        },
    ];
    let mut stats = ScenarioStats {
        phases: vec![b0, b1],
        cgroups,
        ..ScenarioStats::default()
    };
    populate_run_distribution_metrics(&mut stats);
    // Pooled p99 over the cross-step union [1,2,3,4] µs (nearest-rank index
    // ceil(4*0.99)-1 = 3) = 4.0 µs; the 9999/8888 bogus per-cgroup reductions
    // are skipped because "a" is in wake_carriers.
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(4.0),
    );
}

/// Registry-impossible misauthoring detector: a Distribution pairing a wake
/// source with the Worst reduction (no `CgroupStats` wake-worst field) hits
/// the cross-source arm of `distribution_cgroup_reduction` via the
/// carrier-less fold (empty pool + a carrier-less cgroup), which
/// `debug_assert!(false, ...)`s in test builds — catching the misauthored
/// registry entry in CI. In release that arm returns `f64::NAN`, which the
/// producer's `is_finite` insert guard (the pass-8 should-fix) drops to
/// ABSENCE rather than writing a NaN that would fail the whole sidecar
/// write. This exercises the `_from` split directly (the testability the
/// doc claims) over a deliberately non-registry (name, kind) pair.
#[test]
#[should_panic(expected = "no CgroupStats wake reduction")]
fn repool_distribution_cross_source_arm_debug_asserts_in_test_build() {
    let mut target = BTreeMap::new();
    let empty_carriers = std::collections::BTreeSet::new();
    // Empty wake pool -> no pooled value -> the carrier-less fold runs over
    // `cgroups`; "a" is absent from (empty) wake_carriers, so it is folded via
    // distribution_cgroup_reduction(cg, WakeLatencyNs, Worst) -> cross-source arm.
    populate_run_distribution_metrics_from(
        &mut target,
        std::iter::once((
            "worst_p99_wake_latency_us",
            MetricKind::Distribution {
                source: SampleSource::WakeLatencyNs,
                reduction: SampleReduction::Worst,
            },
        )),
        &[],
        &empty_carriers,
        &[],
        &empty_carriers,
        &[],
        &empty_carriers,
        &[CgroupStats {
            cgroup_name: "a".to_string(),
            // Measured (false would gate it out of the carrier-less fold before
            // distribution_cgroup_reduction runs, so the cross-source arm —
            // whose debug_assert this test pins — would never be reached).
            wake_measured: true,
            ..CgroupStats::default()
        }],
        0,
    );
}

/// Distribution measured-zero contract: a cohort whose cgroups DID measure the
/// source (the `*_measured` flags are set) but whose per-cgroup reductions are
/// all 0.0 folds to `Some(0.0)` — a real measured zero. Contrast the sibling
/// `repool_distribution_no_measurement_cohort_is_absent_not_zero`, where
/// unmeasured cgroups (no samples) yield absence. Guards against a future
/// refactor collapsing a genuinely-measured quiet run's Distribution to absent.
#[test]
fn repool_distribution_all_zero_reductions_is_measured_zero_not_absent() {
    let cg = CgroupStats {
        cgroup_name: "x".to_string(),
        p99_wake_latency_us: 0.0,
        worst_run_delay_us: 0.0,
        // Measured: this cohort recorded samples that genuinely reduced to 0.0
        // (e.g. instant wakes / no queue delay), distinct from no-measurement.
        wake_measured: true,
        run_delay_measured: true,
        ..CgroupStats::default()
    };
    // Stripped/empty carrier named "x" -> "x" not in *_carriers -> the
    // carrier-less fold reads cg's 0.0 reductions.
    let mut stats = repool_stats(vec![("x", PhaseCgroupStats::default())], vec![cg]);
    populate_run_distribution_metrics(&mut stats);
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(0.0),
        "measured cohort with 0.0 reductions -> Some(0.0), not absent",
    );
    assert_eq!(
        stats.ext_metrics.get("worst_run_delay_us").copied(),
        Some(0.0),
    );
    // WorstLowest contrast: cg has no workers -> iterations_per_worker None ->
    // all-None cohort -> absent (NOT Some(0.0)) — the kind-specific boundary.
    assert!(
        !stats
            .ext_metrics
            .contains_key("worst_iterations_per_worker"),
        "all-None WorstLowest cohort stays absent, distinct from Distribution's 0.0",
    );
}

/// Distribution no-measurement contract (the carrier-less re-pool fix): a
/// cohort with NO measurement for the source — every cgroup unmeasured
/// (`*_measured` false) and empty carriers — yields ABSENCE (None), not a
/// sentinel `Some(0.0)`. A percentile / mean over zero samples is undefined,
/// and folding its 0.0 into the LowerBetter cross-run mean would falsely drag
/// it toward "perfect". This pins the sibling of the measured-zero case above.
#[test]
fn repool_distribution_no_measurement_cohort_is_absent_not_zero() {
    let cg = CgroupStats {
        cgroup_name: "x".to_string(),
        p99_wake_latency_us: 0.0,
        worst_run_delay_us: 0.0,
        // Unmeasured: no wake / run-delay samples (flags default false). The
        // 0.0 reductions are not-measured sentinels, NOT measured zeros.
        ..CgroupStats::default()
    };
    let mut stats = repool_stats(vec![("x", PhaseCgroupStats::default())], vec![cg]);
    populate_run_distribution_metrics(&mut stats);
    assert!(
        !stats.ext_metrics.contains_key("worst_p99_wake_latency_us"),
        "no-wake-measurement cohort must be ABSENT, not a false Some(0.0)",
    );
    assert!(
        !stats.ext_metrics.contains_key("worst_run_delay_us"),
        "no-run-delay-measurement cohort must be ABSENT, not a false Some(0.0)",
    );
}

/// Mixed across-steps dedup (the subtlest documented boundary): a cgroup
/// name that carries samples in ONE phase is in `*_carriers`, so its
/// `stats.cgroups[]` entry is skipped from the carrier-less fold EVEN when
/// another phase's same-name carrier is empty — the bogus per-cgroup
/// reduction never leaks into the pooled value.
#[test]
fn repool_distribution_name_in_carriers_skips_cgroup_even_with_empty_sibling_phase() {
    let with_samples = PhaseCgroupStats {
        wake_latencies_ns: vec![1000, 2000],
        wake_sample_total: 2,
        ..PhaseCgroupStats::default()
    };
    let mut b0 = PhaseBucket {
        step_index: 1,
        ..PhaseBucket::default()
    };
    b0.per_cgroup.insert("a".to_string(), with_samples);
    // phase[1]: same name "a", EMPTY carrier (collected no samples).
    let mut b1 = PhaseBucket {
        step_index: 2,
        ..PhaseBucket::default()
    };
    b1.per_cgroup
        .insert("a".to_string(), PhaseCgroupStats::default());
    // A stats.cgroups "a" entry with a bogus high p99 that must NOT leak in
    // ("a" is in wake_carriers via phase[0], so this entry is skipped).
    let cgroups = vec![CgroupStats {
        cgroup_name: "a".to_string(),
        p99_wake_latency_us: 9999.0,
        ..CgroupStats::default()
    }];
    let mut stats = ScenarioStats {
        phases: vec![b0, b1],
        cgroups,
        ..ScenarioStats::default()
    };
    populate_run_distribution_metrics(&mut stats);
    // Pooled p99 over phase[0]'s [1,2] µs only (nearest-rank index
    // ceil(2*0.99)-1 = 1) = 2.0 µs; the 9999 bogus reduction is skipped.
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(2.0),
    );
}

/// Graceful degradation: when the bulk frame stripped the phase sample pools
/// (carriers empty), the Distribution re-pool falls back to the worst-wins
/// (max — LowerBetter) over the SURVIVING per-cgroup CgroupStats reductions,
/// reproducing the pre-Item-7 cross-cgroup max — never a silent vanish.
#[test]
fn repool_distribution_falls_back_to_cgroup_reductions_when_stripped() {
    // Distinct non-empty cgroup_names mirror production (collect_handles
    // labels every stats.cgroups entry); the carrier-less fold is
    // name-keyed, so production-shaped fixtures keep the dedup realistic.
    let cg0 = CgroupStats {
        cgroup_name: "a".to_string(),
        p99_wake_latency_us: 30.0,
        worst_run_delay_us: 80.0,
        wake_measured: true,
        run_delay_measured: true,
        ..CgroupStats::default()
    };
    let cg1 = CgroupStats {
        cgroup_name: "b".to_string(),
        p99_wake_latency_us: 70.0,
        worst_run_delay_us: 50.0,
        wake_measured: true,
        run_delay_measured: true,
        ..CgroupStats::default()
    };
    // Phase carrier with EMPTY sample vecs (the stripped state). Named "a"
    // but empty, so "a" is NOT in *_carriers — both cgroups fall to the
    // carrier-less fallback fold regardless of name.
    let mut stats = repool_stats(vec![("a", PhaseCgroupStats::default())], vec![cg0, cg1]);
    populate_run_distribution_metrics(&mut stats);
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(70.0),
        "stripped → fallback max over cgroup p99 reductions",
    );
    assert_eq!(
        stats.ext_metrics.get("worst_run_delay_us").copied(),
        Some(80.0),
        "stripped → fallback max over cgroup worst_run_delay reductions",
    );
}

/// Run-delay re-pool divides RAW ns by 1000 ONCE: mean / worst over
/// `run_delays_ns` [100_000, 300_000] ns are 200 µs / 300 µs, not 0.2/0.3
/// (double-divided) or 100_000/300_000 (forgot to divide).
#[test]
fn repool_run_delay_divides_ns_by_1000_once() {
    let cg = PhaseCgroupStats {
        run_delays_ns: vec![100_000, 300_000],
        ..PhaseCgroupStats::default()
    };
    let mut stats = repool_stats(vec![("a", cg)], vec![]);
    populate_run_distribution_metrics(&mut stats);
    assert_eq!(
        stats.ext_metrics.get("worst_mean_run_delay_us").copied(),
        Some(200.0),
    );
    assert_eq!(
        stats.ext_metrics.get("worst_run_delay_us").copied(),
        Some(300.0),
    );
}

/// CROSS-PHASE pooling (the epic thesis on the phase dimension): the run-level
/// re-pool unions a cgroup's raw samples across MULTIPLE phases, so the pooled
/// p99 is the percentile over BOTH phases' samples combined — not phase[0]
/// alone, not a fold of per-phase reductions. populate_run_distribution_metrics
/// iterates every phase bucket, so a regression that pooled only one phase
/// surfaces here (the single-phase repool_* tests cannot catch it).
#[test]
fn repool_distribution_pools_wake_across_phases() {
    let phase0 = PhaseCgroupStats {
        wake_latencies_ns: (1..=50u64).map(|v| v * 1000).collect(), // 1..50 µs
        wake_sample_total: 50,
        ..PhaseCgroupStats::default()
    };
    let phase1 = PhaseCgroupStats {
        wake_latencies_ns: (51..=100u64).map(|v| v * 1000).collect(), // 51..100 µs
        wake_sample_total: 50,
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

    // Union over BOTH phases = [1..100] µs (100 samples); p99 = sorted[98] = 99 µs.
    let union: Vec<u64> = (1..=100u64).map(|v| v * 1000).collect();
    let expected = percentile(&union, 0.99) as f64 / 1000.0;
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(expected),
        "pooled p99 must be over the cross-PHASE union (99 µs), not phase[0] alone",
    );
    // Distinct from phase[0]-only (1..50 µs → p99 = 50 µs) — proves both phases pool.
    let p0: Vec<u64> = (1..=50u64).map(|v| v * 1000).collect();
    let phase0_only = percentile(&p0, 0.99) as f64 / 1000.0;
    assert_ne!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(phase0_only),
        "must NOT pool only phase[0]",
    );
}

/// Mean-reduction thesis — the reduction with the LARGEST divergence from the deleted
/// max-of-per-cgroup-means fold: worst_mean_run_delay_us is the MEAN over the
/// POOLED cross-cgroup run-delay set, NOT the max of per-cgroup means. cg_a: 4
/// workers @ 10 µs; cg_b: 1 worker @ 200 µs. Pooled mean = (10*4 + 200)/5 =
/// 48 µs; max-of-per-cgroup-means = max(10, 200) = 200 µs. They differ, so a
/// regression re-introducing the max-of-means fold for the run-delay source
/// would fail here (the single-cgroup parity tests cannot catch it).
#[test]
fn repool_mean_run_delay_pools_across_cgroups_not_max_of_per_cgroup() {
    let cg_a = PhaseCgroupStats {
        run_delays_ns: vec![10_000, 10_000, 10_000, 10_000],
        ..PhaseCgroupStats::default()
    };
    let cg_b = PhaseCgroupStats {
        run_delays_ns: vec![200_000],
        ..PhaseCgroupStats::default()
    };
    let mut stats = repool_stats(vec![("a", cg_a), ("b", cg_b)], vec![]);
    populate_run_distribution_metrics(&mut stats);
    // Pooled mean over the 5-sample union = (40_000 + 200_000)/5 / 1000 = 48 µs.
    assert_eq!(
        stats.ext_metrics.get("worst_mean_run_delay_us").copied(),
        Some(48.0),
        "pooled cross-cgroup mean run-delay over the union, not max-of-per-cgroup",
    );
    // max-of-per-cgroup-means = max(10 µs, 200 µs) = 200 µs — the deleted fold.
    assert_ne!(
        stats.ext_metrics.get("worst_mean_run_delay_us").copied(),
        Some(200.0),
        "must NOT be max of per-cgroup means (the pre-Item-7 fold)",
    );
}

/// Carrier-less cgroups (a backdrop whose epochs all fell on BASELINE / the
/// inter-step gap — no paired host bucket, so no carrier — or a stripped /
/// empty carrier) are NOT dropped from the run-level Distribution:
/// their surviving per-cgroup CgroupStats reduction folds worst-wins into the
/// pooled value. Here cgroup "a" carries low wake samples (pooled p99 ~50 µs)
/// and is also in stats.cgroups with a bogus p99=9999 that MUST be ignored
/// (it is pooled, not reduction-folded); backdrop "bd" has NO carrier and a
/// p99=500 µs that MUST be folded in. Result = max(pooled 50, bd 500) = 500.
#[test]
fn repool_distribution_folds_carrierless_backdrop_not_dropped() {
    let carrier_a = PhaseCgroupStats {
        wake_latencies_ns: (1..=50u64).map(|v| v * 1000).collect(), // 1..50 µs
        wake_sample_total: 50,
        ..PhaseCgroupStats::default()
    };
    // "a" is carrier-bearing AND in stats.cgroups: its typed p99 (9999) must be
    // IGNORED (pooled, not folded). "bd" has no carrier: its p99 (500) folds in.
    let cg_a = CgroupStats {
        cgroup_name: "a".to_string(),
        p99_wake_latency_us: 9999.0,
        ..CgroupStats::default()
    };
    let cg_bd = CgroupStats {
        cgroup_name: "bd".to_string(),
        p99_wake_latency_us: 500.0,
        wake_measured: true,
        ..CgroupStats::default()
    };
    let mut stats = repool_stats(vec![("a", carrier_a)], vec![cg_a, cg_bd]);
    populate_run_distribution_metrics(&mut stats);
    let p99 = stats.ext_metrics.get("worst_p99_wake_latency_us").copied();
    assert_eq!(
        p99,
        Some(500.0),
        "backdrop p99 (500) folds worst-wins with the pooled carrier p99 (~50); \
         got {p99:?}",
    );
    // The carrier-bearing cgroup "a" is POOLED (its samples), NOT reduction-
    // folded: its bogus typed p99=9999 must not appear.
    assert_ne!(
        p99,
        Some(9999.0),
        "carrier-bearing cgroup must be pooled, not reduction-folded"
    );
}

/// Per-SOURCE carrier independence: the run-delay carrier set is consulted
/// separately from the wake set. carrier "a" carries run_delays (NO wake
/// samples) and is also in stats.cgroups with bogus run-delay reductions that
/// must be IGNORED (pooled, since it IS in the run-delay carrier set); backdrop
/// "bd" has no carrier and folds its run-delay reductions worst-wins. Pins that
/// the run-delay carrier set is checked independently of the (empty) wake set.
#[test]
fn repool_run_delay_folds_carrierless_backdrop_independently_of_wake() {
    let carrier_a = PhaseCgroupStats {
        run_delays_ns: vec![10_000, 10_000], // 2 workers @ 10 µs, pooled mean 10 µs
        // no wake_latencies_ns: "a" is carrier-bearing for run-delay only.
        ..PhaseCgroupStats::default()
    };
    let cg_a = CgroupStats {
        cgroup_name: "a".to_string(),
        mean_run_delay_us: 9999.0, // carrier-bearing for run-delay → IGNORED (pooled)
        worst_run_delay_us: 9999.0,
        ..CgroupStats::default()
    };
    let cg_bd = CgroupStats {
        cgroup_name: "bd".to_string(),
        mean_run_delay_us: 500.0, // no carrier → folds worst-wins
        worst_run_delay_us: 700.0,
        run_delay_measured: true,
        ..CgroupStats::default()
    };
    let mut stats = repool_stats(vec![("a", carrier_a)], vec![cg_a, cg_bd]);
    populate_run_distribution_metrics(&mut stats);
    // Mean: max(pooled carrier mean 10, bd 500) = 500; cg_a's 9999 ignored (pooled).
    assert_eq!(
        stats.ext_metrics.get("worst_mean_run_delay_us").copied(),
        Some(500.0),
    );
    // Worst (max): max(pooled carrier max 10, bd 700) = 700; cg_a's 9999 ignored.
    assert_eq!(
        stats.ext_metrics.get("worst_run_delay_us").copied(),
        Some(700.0),
    );
}
