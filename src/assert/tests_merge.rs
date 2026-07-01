//! `AssertResult::merge` and the per-field worst-wins / sum aggregation
//! rules for `ScenarioStats`, plus the polarity-aware ext-metric min/max
//! fold. Every polarity is exercised in both directions so a sign-flip
//! regression surfaces regardless of which side carries the worse value.

use super::tests_common::rpt;
use super::*;

#[test]
fn merge_cgroups() {
    let r1 = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
        rpt(2, 1000, 5e9 as u64, 6e8 as u64, &[0, 1], 60),
    ]);
    let r2 = assert_not_starved(&[
        rpt(3, 1000, 5e9 as u64, 25e8 as u64, &[2, 3], 50),
        rpt(4, 1000, 5e9 as u64, 26e8 as u64, &[2, 3], 50),
    ]);
    let mut m = r1;
    m.merge(r2);
    assert_eq!(m.stats.cgroups.len(), 2);
    assert_eq!(m.stats.total_workers, 4);
    assert!(m.is_pass(), "diff cgroups diff off_cpu should pass");
}

#[test]
fn merge_takes_worst_gap() {
    let r1 = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 100)]);
    let r2 = assert_not_starved(&[rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 500)]);
    let mut m = r1;
    m.merge(r2);
    assert_eq!(m.stats.worst_gap_ms, 500);
    assert_eq!(m.stats.worst_gap_cpu, 1);
}

/// Reverse direction of [`merge_takes_worst_gap`]: the forward
/// case picks `other`'s larger gap and must re-couple to
/// `other`'s CPU. This test pins the self-retains branch — when
/// `self.worst_gap_ms > other.worst_gap_ms`, `worst_gap_cpu`
/// must stay on `self`'s CPU and NOT leak over to `other`'s.
///
/// Without both directions pinned, a regression that always
/// overwrote `worst_gap_cpu` from `other` (regardless of which
/// gap won) would pass the forward test — the forward case
/// already asks for `other`'s cpu anyway — and land silently.
/// Pairing the two directions is what actually guards the
/// "coupled fields stay coupled" invariant from the merge doc.
#[test]
fn merge_takes_worst_gap_reverse_self_retains() {
    // r1 has the larger gap (700ms on cpu 0); r2 has the smaller
    // gap (200ms on cpu 1). After merge, self must keep both
    // its 700ms AND its cpu 0 — not adopt cpu 1 from the
    // loser's report.
    let r1 = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 700)]);
    let r2 = assert_not_starved(&[rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 200)]);
    let mut m = r1;
    m.merge(r2);
    assert_eq!(
        m.stats.worst_gap_ms, 700,
        "self's larger gap must be retained",
    );
    assert_eq!(
        m.stats.worst_gap_cpu, 0,
        "worst_gap_cpu must stay coupled to self's worst_gap_ms — \
         a regression overwriting cpu from other would set this to 1",
    );
}

#[test]
fn merge_takes_worst_spread() {
    let r1 = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 1e9 as u64, &[0], 50),
        rpt(2, 1000, 5e9 as u64, 12e8 as u64, &[0], 50),
    ]); // spread = 4%
    let r2 = assert_not_starved(&[
        rpt(3, 1000, 5e9 as u64, 1e9 as u64, &[1], 50),
        rpt(4, 1000, 5e9 as u64, 15e8 as u64, &[1], 50),
    ]); // spread = 10%
    let mut m = r1;
    m.merge(r2);
    assert!((m.stats.worst_spread - 10.0).abs() < 0.1);
}

#[test]
fn merge_skip_plus_explicit_pass_demotes_skip() {
    // A bare `AssertResult::pass()` has empty outcomes (the
    // zero-allocation Pass identity). Merging it onto a skip leaves
    // the stream all-Skip, so it does NOT demote. To demote a skip,
    // the passing side must carry an explicit `Outcome::Pass` marker
    // via `record_pass()` — that's the "real Pass beats Skip" semantic.
    let mut a = AssertResult::skip("optional");
    let mut b = AssertResult::pass();
    b.record_pass();
    a.merge(b);
    assert!(
        !a.is_skip(),
        "explicit Pass in the merged stream means not all-Skip"
    );
    assert!(a.is_pass(), "explicit Pass + no Fail → is_pass=true");
}

#[test]
fn merge_skip_plus_empty_pass_stays_skip() {
    // Companion to merge_skip_plus_explicit_pass_demotes_skip: bare
    // `pass()` (empty outcomes) cannot demote a skip; the merged
    // stream is still all-Skip.
    let mut a = AssertResult::skip("optional");
    let b = AssertResult::pass();
    a.merge(b);
    assert!(
        a.is_skip(),
        "empty pass() merges to a no-op; stream stays all-Skip"
    );
    assert!(!a.is_pass(), "all-Skip is not pass");
}

#[test]
fn merge_skip_plus_fail_is_fail_not_skip() {
    let mut a = AssertResult::skip("topo missing");
    let mut b = AssertResult::pass();
    b.record_fail(AssertDetail::new(DetailKind::Other, "synthetic fail"));
    a.merge(b);
    assert!(a.is_fail());
    assert!(!a.is_skip());
}

#[test]
fn merge_accumulates_totals() {
    let r1 = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)]);
    let r2 = assert_not_starved(&[rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 50)]);
    let mut m = r1;
    m.merge(r2);
    assert_eq!(m.stats.total_workers, 2);
    assert_eq!(m.stats.total_cpus, 2);
}

/// Multi-cgroup merge-aggregation contract: merging `N > 2`
/// `AssertResult`s (each carrying one populated `CgroupStats`
/// plus `ScenarioStats` headline fields) must:
///   - append every per-cgroup entry into `stats.cgroups` in
///     merge order, preserving cardinality;
///   - pick the worst value of every merge-folded higher-is-worse
///     `worst_*` field across all merged cgroups;
///   - SUM `total_iterations` across all cgroups, not max it.
///
/// (`worst_page_locality` is NOT merge-folded — it re-pools run-level
/// post-merge from the per-phase NUMA carriers; see
/// `merge_repools_worst_page_locality_across_cgroups_measured_zero_wins`.)
///
/// (The wake / run-delay distributions and the iteration efficiencies are
/// no longer merge-folded — they re-pool post-merge; see the `repool_*`
/// tests.)
///
/// Sibling `merge_scenario_stats_worst_wins_and_iterations_sum`
/// already covers the 2-cgroup case with headline fields only;
/// this test exercises 3 cgroups AND the per-cgroup accumulator
/// (`stats.cgroups.extend`) so a regression that dropped
/// cgroups, clobbered the per-cgroup vector, or flipped one of
/// the polarity folds surfaces in the stronger form.
#[test]
fn merge_three_cgroups_worst_wins_and_iterations_sum() {
    fn mk(
        worst_spread: f64,
        worst_mig: f64,
        page_locality: f64,
        total_iters: u64,
        cg_total_iters: u64,
    ) -> AssertResult {
        let cg = CgroupStats {
            total_iterations: cg_total_iters,
            page_locality,
            ..CgroupStats::default()
        };
        // The wake/run-delay and iteration-efficiency roll-ups are no longer
        // ScenarioStats fields (they are Distribution / WorstLowest, re-pooled
        // post-merge); this test now covers the merge-folded worst-wins
        // (`worst_spread`, `worst_migration_ratio`), `total_iterations`, and the
        // `cgroups.extend` accumulation (including each cgroup's per-cgroup
        // `page_locality` surviving the merge — `worst_page_locality` itself
        // re-pools post-merge from the per-phase carriers, covered elsewhere).
        AssertResult {
            outcomes: vec![],
            passes: vec![],
            stats: ScenarioStats {
                total_iterations: total_iters,
                worst_spread,
                worst_migration_ratio: worst_mig,
                cgroups: vec![cg],
                ..ScenarioStats::default()
            },
            measurements: std::collections::BTreeMap::new(),
            info_notes: vec![],
        }
    }

    // Three cgroups with deliberately heterogeneous values so
    // each `worst_*` aggregation is sourced from a DIFFERENT
    // cgroup — a regression that folded only within-cgroup
    // would still produce a plausible-looking aggregate on a
    // 2-cgroup test but would fail here.
    let mut acc = mk(10.0, 0.1, 0.8, 100, 100);
    acc.merge(mk(5.0, 0.3, 0.5, 200, 200));
    acc.merge(mk(20.0, 0.2, 0.9, 400, 400));

    let s = &acc.stats;
    assert_eq!(
        s.cgroups.len(),
        3,
        "3 cgroups must accumulate; a missing entry means stats.cgroups.extend dropped a merge",
    );
    // Per-cgroup order is preserved (merge calls, in order):
    assert_eq!(s.cgroups[0].total_iterations, 100);
    assert_eq!(s.cgroups[1].total_iterations, 200);
    assert_eq!(s.cgroups[2].total_iterations, 400);
    // Each cgroup's per-cgroup `page_locality` telemetry survives the merge into
    // stats.cgroups (the run-level worst re-pools from these post-merge):
    assert_eq!(s.cgroups[0].page_locality, 0.8);
    assert_eq!(s.cgroups[1].page_locality, 0.5);
    assert_eq!(s.cgroups[2].page_locality, 0.9);

    // Worst-wins across 3 cgroups (higher-is-worse):
    assert_eq!(s.worst_spread, 20.0, "third cgroup's 20.0 is worst");
    assert_eq!(s.worst_migration_ratio, 0.3, "second cgroup's 0.3 is worst");
    // total_iterations SUMS across cgroups, not maxes:
    assert_eq!(
        s.total_iterations,
        100 + 200 + 400,
        "total_iterations must sum (not max) across all merged cgroups",
    );
}

#[test]
fn iterations_per_worker_distinguishes_no_workers_from_ran_zero() {
    // num_workers == 0: no per-worker throughput is defined → None.
    let no_workers = CgroupStats {
        num_workers: 0,
        total_iterations: 0,
        ..CgroupStats::default()
    };
    assert_eq!(no_workers.iterations_per_worker(), None);

    // Workers ran but completed zero iterations → measured Some(0.0),
    // NOT None: this is a real throughput collapse, not missing data.
    let ran_zero = CgroupStats {
        num_workers: 4,
        total_iterations: 0,
        ..CgroupStats::default()
    };
    assert_eq!(ran_zero.iterations_per_worker(), Some(0.0));

    // Workers ran with iterations → the throughput value.
    let ran = CgroupStats {
        num_workers: 4,
        total_iterations: 400,
        ..CgroupStats::default()
    };
    assert_eq!(ran.iterations_per_worker(), Some(100.0));
}

#[test]
fn repool_worst_iterations_per_worker_lets_measured_zero_win() {
    // A cgroup that ran zero iterations (per-cgroup Some(0.0)) is the worst
    // per-worker throughput and MUST win the lowest bucket; a later healthy
    // reading does not displace it. `populate_run_distribution_metrics`
    // selects lowest-wins None-aware over the per-cgroup
    // `iterations_per_worker()` — the semantic the deleted cross-cgroup
    // `fold_lowest_some` carried.
    fn cg(num_workers: usize, total_iterations: u64) -> AssertResult {
        let mut r = AssertResult::pass();
        r.stats.cgroups = vec![CgroupStats {
            num_workers,
            total_iterations,
            ..CgroupStats::default()
        }];
        r
    }
    let mut acc = AssertResult::pass();
    acc.merge(cg(1, 100)); // iterations_per_worker == 100.0
    acc.merge(cg(4, 0)); // iterations_per_worker == Some(0.0) (ran zero)
    acc.merge(cg(1, 250)); // iterations_per_worker == 250.0
    populate_run_distribution_metrics(&mut acc.stats);
    assert_eq!(
        acc.stats
            .ext_metrics
            .get("worst_iterations_per_worker")
            .copied(),
        Some(0.0),
        "a cgroup that ran zero iterations must win the worst bucket",
    );
}

#[test]
fn repool_worst_iterations_per_worker_skips_no_data() {
    // No-worker cgroups (num_workers == 0 → iterations_per_worker() None) are
    // skipped, never treated as zero: an all-None cohort writes NO key
    // (absence preserved, distinct from a measured 0.0), and a None never
    // displaces a real reading.
    fn cg(num_workers: usize, total_iterations: u64) -> AssertResult {
        let mut r = AssertResult::pass();
        r.stats.cgroups = vec![CgroupStats {
            num_workers,
            total_iterations,
            ..CgroupStats::default()
        }];
        r
    }
    let mut acc = AssertResult::pass();
    acc.merge(cg(0, 0));
    acc.merge(cg(0, 0));
    populate_run_distribution_metrics(&mut acc.stats);
    assert_eq!(
        acc.stats.ext_metrics.get("worst_iterations_per_worker"),
        None,
        "all-None cohort must write no key (absence != measured 0.0)",
    );

    let mut acc2 = AssertResult::pass();
    acc2.merge(cg(1, 75)); // Some(75.0)
    acc2.merge(cg(0, 0)); // None, skipped
    populate_run_distribution_metrics(&mut acc2.stats);
    assert_eq!(
        acc2.stats
            .ext_metrics
            .get("worst_iterations_per_worker")
            .copied(),
        Some(75.0),
        "a None contributor must not displace a real reading",
    );
}

#[test]
fn repool_worst_iterations_per_cpu_sec_lowest_wins_none_aware() {
    // worst_iterations_per_cpu_sec (overcommit-invariant efficiency) uses the
    // same lowest-wins None-aware re-pool: the least-efficient cgroup wins, a
    // measured Some(0.0) beats a healthy reading, and None (no workers or no
    // on-CPU time) is skipped, never fabricated as zero.
    fn cg(num_workers: usize, total_iterations: u64, cpu_ns: u64) -> AssertResult {
        let mut r = AssertResult::pass();
        r.stats.cgroups = vec![CgroupStats {
            num_workers,
            total_iterations,
            total_cpu_time_ns: cpu_ns,
            ..CgroupStats::default()
        }];
        r
    }
    let mut acc = AssertResult::pass();
    acc.merge(cg(0, 0, 0)); // None (no workers / no on-CPU time), skipped
    acc.merge(cg(1, 900, 1_000_000_000)); // 900 / 1.0s == 900.0
    acc.merge(cg(1, 0, 1_000_000_000)); // 0 / 1.0s == Some(0.0), worst
    acc.merge(cg(1, 1500, 1_000_000_000)); // 1500.0, does not displace 0.0
    populate_run_distribution_metrics(&mut acc.stats);
    assert_eq!(
        acc.stats
            .ext_metrics
            .get("worst_iterations_per_cpu_sec")
            .copied(),
        Some(0.0),
        "least-efficient cgroup (measured 0.0) wins; None skipped",
    );
}

/// The POOLED `iterations_per_cpu_sec` Rate (the cross-cgroup re-pool) is
/// Σiterations / Σcpu-seconds across cgroups — NOT a mean of per-cgroup
/// ratios, NOT the worst single cgroup. Unequal per-cgroup cpu-time makes
/// the three distinct: re-pool 101.0 vs mean-of-ratios ~500.6 vs worst ~1.11
/// (the value the rejected merge-fold route would wrongly produce). The new
/// pooled metric must NOT mutate the existing worst_iterations_per_cpu_sec
/// (the min-fold starvation selector).
#[test]
fn populate_run_pooled_iterations_per_cpu_sec_repools_across_cgroups() {
    // cg1: 1000 iters over 1.0 cpu-s -> 1000/cpu-s.
    let cg1 = CgroupStats {
        total_iterations: 1000,
        total_cpu_time_ns: 1_000_000_000,
        num_workers: 1,
        ..CgroupStats::default()
    };
    // cg2: 10 iters over 9.0 cpu-s -> ~1.11/cpu-s.
    let cg2 = CgroupStats {
        total_iterations: 10,
        total_cpu_time_ns: 9_000_000_000,
        num_workers: 1,
        ..CgroupStats::default()
    };
    let stats_for = |cg: &CgroupStats| ScenarioStats {
        total_iterations: cg.total_iterations,
        cgroups: vec![cg.clone()],
        ..ScenarioStats::default()
    };
    let mk = |cg: &CgroupStats| AssertResult {
        outcomes: vec![],
        passes: vec![],
        stats: stats_for(cg),
        measurements: std::collections::BTreeMap::new(),
        info_notes: vec![],
    };
    let mut acc = mk(&cg1);
    acc.merge(mk(&cg2));
    populate_run_pooled_iterations_per_cpu_sec(&mut acc.stats);

    // Σiters / Σcpu-s = (1000 + 10) / ((1e9 + 9e9)/1e9) = 1010 / 10.0 = 101.0.
    assert_eq!(
        acc.stats.ext_metrics.get("iterations_per_cpu_sec").copied(),
        Some(101.0),
        "pooled rate must be Σiters/Σcpu-s = 101.0, NOT mean-of-ratios \
         (~500.6) or the worst cgroup (~1.11); got {:?}",
        acc.stats.ext_metrics.get("iterations_per_cpu_sec"),
    );
    // BOTH cgroups have measured cpu-time, so the ext-only pooled numerator
    // equals the merge-summed typed total_iterations (both Σ over all
    // cgroups). They diverge only when a zero-cpu-time cgroup is excluded from
    // the pooled sum — see populate_run_pooled_..._excludes_zero_cpu_cgroup.
    assert_eq!(acc.stats.total_iterations, 1010);
    assert_eq!(
        acc.stats
            .ext_metrics
            .get("total_iterations_pooled")
            .copied(),
        Some(acc.stats.total_iterations as f64),
        "total_iterations_pooled must equal the merge-summed typed total_iterations \
         when every cgroup is measured",
    );
    // The WorstLowest worst_iterations_per_cpu_sec (lowest-wins starvation
    // selector) is DISTINCT from the pooled rate: re-pooled separately by
    // populate_run_distribution_metrics, the lower per-cgroup rate (cg2's
    // 10/9) wins. The pooled iterations_per_cpu_sec Rate above is unaffected.
    populate_run_distribution_metrics(&mut acc.stats);
    let worst = acc
        .stats
        .ext_metrics
        .get("worst_iterations_per_cpu_sec")
        .copied()
        .expect("worst_iterations_per_cpu_sec present in ext_metrics");
    assert!(
        (worst - 10.0 / 9.0).abs() < 1e-9,
        "worst_iterations_per_cpu_sec stays the lowest-wins selector (~1.11), \
         distinct from the pooled rate; got {worst}",
    );
}

/// Host-only / no-schedstat run: every cgroup reports zero on-CPU time, so
/// the pooled rate is undefined. The helper inserts NEITHER component
/// (both-or-neither) so no rate derives — matching
/// `CgroupStats::iterations_per_cpu_sec`'s None-on-zero.
#[test]
fn populate_run_pooled_iterations_per_cpu_sec_absent_on_zero_cpu_time() {
    let cg = CgroupStats {
        total_iterations: 500,
        total_cpu_time_ns: 0,
        num_workers: 1,
        ..CgroupStats::default()
    };
    let mut acc = AssertResult {
        outcomes: vec![],
        passes: vec![],
        stats: ScenarioStats {
            total_iterations: cg.total_iterations,
            cgroups: vec![cg],
            ..ScenarioStats::default()
        },
        measurements: std::collections::BTreeMap::new(),
        info_notes: vec![],
    };
    populate_run_pooled_iterations_per_cpu_sec(&mut acc.stats);
    assert!(
        !acc.stats.ext_metrics.contains_key("iterations_per_cpu_sec"),
        "no pooled rate when Σcpu-time is 0",
    );
    assert!(
        !acc.stats.ext_metrics.contains_key("total_cpu_time_sec")
            && !acc
                .stats
                .ext_metrics
                .contains_key("total_iterations_pooled"),
        "both-or-neither: neither component inserted when Σcpu-time is 0",
    );
}

/// Mixed run: one cgroup has iterations but ZERO measured cpu-time (schedstat
/// gap), the other has both. The zero-cpu cgroup is EXCLUDED from BOTH pooled
/// sums (mirroring the per-cgroup None-on-zero) — its iterations are NOT
/// credited against the measured cgroup's cpu-seconds, which would inflate the
/// cohort efficiency. So the pooled rate is the measured cgroup's rate, and
/// total_iterations_pooled (measured only) is strictly LESS than the
/// merge-summed typed total_iterations (which includes both).
#[test]
fn populate_run_pooled_iterations_per_cpu_sec_excludes_zero_cpu_cgroup() {
    // Unmeasured: 500 iters, 0 cpu-time (schedstat unavailable).
    let unmeasured = CgroupStats {
        total_iterations: 500,
        total_cpu_time_ns: 0,
        num_workers: 1,
        ..CgroupStats::default()
    };
    // Measured: 1000 iters over 1.0 cpu-s -> 1000/cpu-s.
    let measured = CgroupStats {
        total_iterations: 1000,
        total_cpu_time_ns: 1_000_000_000,
        num_workers: 1,
        ..CgroupStats::default()
    };
    let mk = |cg: &CgroupStats| AssertResult {
        outcomes: vec![],
        passes: vec![],
        stats: ScenarioStats {
            total_iterations: cg.total_iterations,
            cgroups: vec![cg.clone()],
            ..ScenarioStats::default()
        },
        measurements: std::collections::BTreeMap::new(),
        info_notes: vec![],
    };
    let mut acc = mk(&unmeasured);
    acc.merge(mk(&measured));
    populate_run_pooled_iterations_per_cpu_sec(&mut acc.stats);

    // Pooled rate excludes the zero-cpu cgroup: 1000 / 1.0 == 1000.0, NOT
    // (500 + 1000) / 1.0 == 1500.0 (which would credit un-costed iters).
    assert_eq!(
        acc.stats.ext_metrics.get("iterations_per_cpu_sec").copied(),
        Some(1000.0),
        "zero-cpu cgroup's iters must NOT inflate the pooled rate; got {:?}",
        acc.stats.ext_metrics.get("iterations_per_cpu_sec"),
    );
    // The pooled numerator counts only the measured cgroup (1000) and is
    // strictly LESS than the merge-summed typed total_iterations (1500).
    assert_eq!(
        acc.stats
            .ext_metrics
            .get("total_iterations_pooled")
            .copied(),
        Some(1000.0),
    );
    assert_eq!(acc.stats.total_iterations, 1500);
}

/// Single measured cgroup: the pooled rate is exactly that cgroup's per-cgroup
/// rate (degenerate Σ over one element).
#[test]
fn populate_run_pooled_iterations_per_cpu_sec_single_cgroup() {
    let cg = CgroupStats {
        total_iterations: 750,
        total_cpu_time_ns: 3_000_000_000,
        num_workers: 1,
        ..CgroupStats::default()
    };
    let mut acc = AssertResult {
        outcomes: vec![],
        passes: vec![],
        stats: ScenarioStats {
            total_iterations: cg.total_iterations,
            cgroups: vec![cg.clone()],
            ..ScenarioStats::default()
        },
        measurements: std::collections::BTreeMap::new(),
        info_notes: vec![],
    };
    populate_run_pooled_iterations_per_cpu_sec(&mut acc.stats);
    // 750 / 3.0 == 250.0 == the per-cgroup rate.
    assert_eq!(
        acc.stats.ext_metrics.get("iterations_per_cpu_sec").copied(),
        cg.iterations_per_cpu_sec(),
    );
    assert_eq!(
        acc.stats.ext_metrics.get("iterations_per_cpu_sec").copied(),
        Some(250.0),
    );
}

/// Empty cgroups vec: nothing to pool, no keys inserted (both-or-neither).
#[test]
fn populate_run_pooled_iterations_per_cpu_sec_empty_cgroups() {
    let mut stats = ScenarioStats::default();
    populate_run_pooled_iterations_per_cpu_sec(&mut stats);
    assert!(
        stats.ext_metrics.is_empty(),
        "no components inserted for an empty cgroups vec",
    );
}

/// Costed-yet-idle cgroup INCLUDED in both sums (the symmetric counterpart of
/// excludes_zero_cpu_cgroup): a cgroup with measured cpu-time but ZERO
/// iterations (a stalled/spinning worker that burned CPU doing no work). The
/// filter gates on total_cpu_time_ns > 0 (NOT on iterations), so this cgroup IS
/// included — its CPU adds to the denominator and its 0 iters add nothing to
/// the numerator, correctly diluting the cohort rate downward (burning CPU with
/// no work IS less efficient).
#[test]
fn populate_run_pooled_iterations_per_cpu_sec_includes_costed_idle_cgroup() {
    // Costed but idle: 0 iters over 2.0 cpu-s.
    let idle = CgroupStats {
        total_iterations: 0,
        total_cpu_time_ns: 2_000_000_000,
        num_workers: 1,
        ..CgroupStats::default()
    };
    // Productive: 1000 iters over 1.0 cpu-s.
    let busy = CgroupStats {
        total_iterations: 1000,
        total_cpu_time_ns: 1_000_000_000,
        num_workers: 1,
        ..CgroupStats::default()
    };
    let mk = |cg: &CgroupStats| AssertResult {
        outcomes: vec![],
        passes: vec![],
        stats: ScenarioStats {
            total_iterations: cg.total_iterations,
            cgroups: vec![cg.clone()],
            ..ScenarioStats::default()
        },
        measurements: std::collections::BTreeMap::new(),
        info_notes: vec![],
    };
    let mut acc = mk(&idle);
    acc.merge(mk(&busy));
    populate_run_pooled_iterations_per_cpu_sec(&mut acc.stats);

    // The idle cgroup's CPU MUST count: rate = 1000 / ((2e9+1e9)/1e9) = 1000/3.0
    // == ~333.33, NOT 1000/1.0 == 1000 (which would ignore the wasted CPU).
    let rate = acc
        .stats
        .ext_metrics
        .get("iterations_per_cpu_sec")
        .copied()
        .expect("pooled rate present");
    assert!(
        (rate - 1000.0 / 3.0).abs() < 1e-9,
        "costed-idle cgroup's CPU must dilute the rate to ~333.33, not 1000; got {rate}",
    );
    // Numerator = 0 + 1000; denominator counts the idle cgroup's 2.0s too.
    assert_eq!(
        acc.stats
            .ext_metrics
            .get("total_iterations_pooled")
            .copied(),
        Some(1000.0),
    );
    assert_eq!(
        acc.stats.ext_metrics.get("total_cpu_time_sec").copied(),
        Some(3.0),
    );
}

/// Tiny-denominator finite-quotient guard: a cgroup with total_cpu_time_ns=1
/// (total_cpu_time_sec = 1e-9) and a large iteration count yields a
/// finite-but-enormous rate (~1e12). derive_rate_metrics_from's finite guard
/// KEEPS it — an absent rate is reserved for a zero or non-finite denominator,
/// not a tiny one — and the pooled wrapper feeds that same guard. (u64-summed
/// ns cannot overflow within centuries, so no overflow case is reachable.)
#[test]
fn populate_run_pooled_iterations_per_cpu_sec_tiny_denominator_stays_finite() {
    let cg = CgroupStats {
        total_iterations: 1000,
        total_cpu_time_ns: 1,
        num_workers: 1,
        ..CgroupStats::default()
    };
    let mut acc = AssertResult {
        outcomes: vec![],
        passes: vec![],
        stats: ScenarioStats {
            total_iterations: cg.total_iterations,
            cgroups: vec![cg],
            ..ScenarioStats::default()
        },
        measurements: std::collections::BTreeMap::new(),
        info_notes: vec![],
    };
    populate_run_pooled_iterations_per_cpu_sec(&mut acc.stats);
    let rate = acc
        .stats
        .ext_metrics
        .get("iterations_per_cpu_sec")
        .copied()
        .expect("tiny-denom rate present (finite, not dropped)");
    assert!(
        rate.is_finite() && rate > 0.0,
        "tiny-denom rate must be finite-but-enormous (~1e12), not inf/absent; got {rate}",
    );
}

/// Whole-run taobench qps + hit Rates re-pool across the run's Taobench
/// cgroups via sum-of-ops and max-of-wall (the window is shared by the
/// concurrent cohorts, so it is taken as MAX, never summed — a summed window
/// would deflate every qps). With cg1 (fast 800, slow 200, 10 s) and cg2 (fast
/// 300, slow 700, 8 s), the pool is fast 1100, slow 900, ops 2000, wall
/// MAX(10,8) = 10 s, giving total 200/s, fast 110/s, slow 90/s, and hit
/// 1100/2000 = 0.55. A summed-wall denominator (18 s) would give total roughly
/// 111/s, so the MAX window is observable.
#[test]
fn populate_run_pooled_taobench_repools_across_cgroups() {
    let tb = |fast: u64, slow: u64, secs: u64| {
        Some(crate::workload::taobench::run::TaobenchStats {
            get_cmds: fast + slow,
            get_misses: slow,
            fast_ops: fast,
            slow_ops: slow,
            elapsed_ns: secs * 1_000_000_000,
        })
    };
    let cg1 = CgroupStats {
        taobench_whole: tb(800, 200, 10),
        num_workers: 1,
        ..CgroupStats::default()
    };
    let cg2 = CgroupStats {
        taobench_whole: tb(300, 700, 8),
        num_workers: 1,
        ..CgroupStats::default()
    };
    let mk = |cg: &CgroupStats| AssertResult {
        outcomes: vec![],
        passes: vec![],
        stats: ScenarioStats {
            cgroups: vec![cg.clone()],
            ..ScenarioStats::default()
        },
        measurements: std::collections::BTreeMap::new(),
        info_notes: vec![],
    };
    let mut acc = mk(&cg1);
    acc.merge(mk(&cg2));
    populate_run_pooled_taobench(&mut acc.stats);
    let e = &acc.stats.ext_metrics;
    // Counter components: Σ ops, MAX wall.
    assert_eq!(e.get("total_taobench_ops").copied(), Some(2000.0));
    assert_eq!(e.get("total_taobench_fast_ops").copied(), Some(1100.0));
    assert_eq!(e.get("total_taobench_slow_ops").copied(), Some(900.0));
    assert_eq!(
        e.get("total_taobench_wall_sec").copied(),
        Some(10.0),
        "wall is MAX(10,8) = 10, not Σ = 18",
    );
    // Derived Rates = Σnum / Σden over the cohort.
    assert_eq!(e.get("taobench_total_ops_per_sec").copied(), Some(200.0));
    assert_eq!(e.get("taobench_fast_ops_per_sec").copied(), Some(110.0));
    assert_eq!(e.get("taobench_slow_ops_per_sec").copied(), Some(90.0));
    let hit = e
        .get("taobench_hit_fraction")
        .copied()
        .expect("hit_fraction derived");
    assert!(
        (hit - 0.55).abs() < 1e-9,
        "Σfast/Σops = 1100/2000 = 0.55, got {hit}",
    );
}

/// No Taobench cgroup (every `taobench_whole` is `None`) → no keys written, so a
/// non-taobench run stays distinct from a measured zero. Also covers the
/// empty-cgroups case (the same `pooled == None` early return).
#[test]
fn populate_run_pooled_taobench_absent_without_taobench_cgroup() {
    let cg = CgroupStats {
        num_workers: 1,
        ..CgroupStats::default()
    };
    let mut stats = ScenarioStats {
        cgroups: vec![cg],
        ..ScenarioStats::default()
    };
    populate_run_pooled_taobench(&mut stats);
    assert!(
        stats.ext_metrics.is_empty(),
        "no taobench keys when no cgroup ran taobench",
    );
    // Empty cgroups vec: same early return, no keys.
    let mut empty = ScenarioStats::default();
    populate_run_pooled_taobench(&mut empty);
    assert!(empty.ext_metrics.is_empty(), "no keys for empty cgroups");
}

/// `taobench_whole` present but `elapsed_ns == 0` (degenerate window): qps is
/// undefined, so NEITHER components NOR rates are written (both-or-neither gate
/// on the measured wall window).
#[test]
fn populate_run_pooled_taobench_absent_on_zero_wall() {
    let cg = CgroupStats {
        taobench_whole: Some(crate::workload::taobench::run::TaobenchStats {
            get_cmds: 100,
            get_misses: 10,
            fast_ops: 90,
            slow_ops: 10,
            elapsed_ns: 0,
        }),
        num_workers: 1,
        ..CgroupStats::default()
    };
    let mut stats = ScenarioStats {
        cgroups: vec![cg],
        ..ScenarioStats::default()
    };
    populate_run_pooled_taobench(&mut stats);
    assert!(
        stats.ext_metrics.is_empty(),
        "no keys when the wall window is 0 (qps undefined)",
    );
}

/// Wall window measured but ZERO ops completed (a cohort that issued no
/// completions): the qps components/rates land at 0, but `taobench_hit_fraction`
/// stays ABSENT — `derive_rate_metrics` skips its zero denominator (total ops
/// 0), keeping a no-ops run distinct from a real 0.0 hit fraction. This is the
/// per-metric gate: qps gated on the wall window, hit_fraction gated on
/// completed ops via the derive's denominator guard.
#[test]
fn populate_run_pooled_taobench_hit_fraction_absent_when_no_ops() {
    let cg = CgroupStats {
        taobench_whole: Some(crate::workload::taobench::run::TaobenchStats {
            get_cmds: 0,
            get_misses: 0,
            fast_ops: 0,
            slow_ops: 0,
            elapsed_ns: 5_000_000_000,
        }),
        num_workers: 1,
        ..CgroupStats::default()
    };
    let mut stats = ScenarioStats {
        cgroups: vec![cg],
        ..ScenarioStats::default()
    };
    populate_run_pooled_taobench(&mut stats);
    let e = &stats.ext_metrics;
    // Components present (wall measured); qps = 0.
    assert_eq!(e.get("total_taobench_ops").copied(), Some(0.0));
    assert_eq!(e.get("total_taobench_wall_sec").copied(), Some(5.0));
    assert_eq!(e.get("taobench_total_ops_per_sec").copied(), Some(0.0));
    // hit_fraction ABSENT (0/0 skipped), not a false 0.0.
    assert!(
        !e.contains_key("taobench_hit_fraction"),
        "hit_fraction absent when no ops completed",
    );
    // command_hit_rate likewise ABSENT — Σget_cmds == 0, so derive_rate_metrics
    // skips its zero denominator (a no-lookups run is distinct from a real 0.0 hit
    // rate). The components are present as measured zeros.
    assert_eq!(e.get("total_taobench_get_cmds").copied(), Some(0.0));
    assert_eq!(e.get("total_taobench_get_hits").copied(), Some(0.0));
    assert!(
        !e.contains_key("taobench_command_hit_rate"),
        "command_hit_rate absent when no lookups issued",
    );
}

/// The whole-run COMMAND-time hit (`taobench_command_hit_rate` = Σhits/Σcmds,
/// hits = cmds − misses) is a DISTINCT axis from the RESPONSE-time
/// `taobench_hit_fraction` (Σfast/Σcompleted): a request is counted at lookup
/// (cmd/miss) and again at completion (fast/slow), and under open-loop arrival a
/// lookup can be in flight (counted) before it completes, so `get_misses` need not
/// equal `slow_ops`. Two cgroups whose two hit measurements diverge pin the new
/// command-time components (Σ pool) and the derived rate, plus the divergence the
/// feature exists to expose.
#[test]
fn populate_run_pooled_taobench_command_hit_diverges_from_response() {
    let mk = |cmds: u64, misses: u64, fast: u64, slow: u64, secs: u64| {
        Some(crate::workload::taobench::run::TaobenchStats {
            get_cmds: cmds,
            get_misses: misses,
            fast_ops: fast,
            slow_ops: slow,
            elapsed_ns: secs * 1_000_000_000,
        })
    };
    let cg1 = CgroupStats {
        taobench_whole: mk(1000, 100, 850, 50, 10),
        num_workers: 1,
        ..CgroupStats::default()
    };
    let cg2 = CgroupStats {
        taobench_whole: mk(2000, 400, 1500, 100, 10),
        num_workers: 1,
        ..CgroupStats::default()
    };
    let mut stats = ScenarioStats {
        cgroups: vec![cg1, cg2],
        ..ScenarioStats::default()
    };
    populate_run_pooled_taobench(&mut stats);
    let e = &stats.ext_metrics;
    // Command-time components pool by Σ; hits = Σcmds − Σmisses = 3000 − 500 = 2500.
    assert_eq!(e.get("total_taobench_get_cmds").copied(), Some(3000.0));
    assert_eq!(e.get("total_taobench_get_hits").copied(), Some(2500.0));
    let cmd_hit = e
        .get("taobench_command_hit_rate")
        .copied()
        .expect("command_hit_rate derived");
    assert!(
        (cmd_hit - 2500.0 / 3000.0).abs() < 1e-9,
        "Σhits/Σcmds = 2500/3000 = 0.8333, got {cmd_hit}",
    );
    let resp_hit = e
        .get("taobench_hit_fraction")
        .copied()
        .expect("hit_fraction derived");
    assert!(
        (resp_hit - 2350.0 / 2500.0).abs() < 1e-9,
        "Σfast/Σcompleted = 2350/2500 = 0.94, got {resp_hit}",
    );
    assert!(
        (cmd_hit - resp_hit).abs() > 0.05,
        "command-time ({cmd_hit}) and response-time ({resp_hit}) hit DIVERGE — \
         the reason both are registered whole-run keys",
    );
}

/// A taobench per-phase per-cgroup carrier whose serve-latency `PlatStats` holds
/// `lows` samples at `low_us` µs + `highs` samples at `high_us` µs (counters left
/// zero). The taobench analog of [`schbench_wakeup_pc`].
fn taobench_serve_pc(lows: u32, low_us: u32, highs: u32, high_us: u32) -> PhaseCgroupStats {
    let mut serve = crate::workload::schbench::plat::PlatStats::default();
    for _ in 0..lows {
        serve.add_lat(low_us);
    }
    for _ in 0..highs {
        serve.add_lat(high_us);
    }
    PhaseCgroupStats {
        taobench: Some(crate::workload::taobench::run::TaobenchPhaseStats {
            serve_lat: serve,
            ..Default::default()
        }),
        ..PhaseCgroupStats::default()
    }
}

/// populate_run_pooled_taobench_distribution re-pools the whole-run serve-latency
/// percentile by UNIONING the per-cgroup serve `PlatStats` histograms and
/// re-deriving — the percentile-OF-the-union, NOT a mean of per-cgroup
/// percentiles. cg_a has 99 low serve samples (10 µs), cg_b has 1 high (10000 µs):
/// the union p99 (rank 99 of 100 pooled) is a LOW bucket (~10 µs), whereas
/// mean-of-per-cgroup-p99 would be ≈ (10 + 10000)/2 = 5005. The union MAX is the
/// high sample (combine's max-of-max). Pins the cross-cgroup union + non-linearity
/// (`schbench_phase` is a generic `PhaseBucket` builder, used here for taobench).
#[test]
fn populate_run_pooled_taobench_distribution_unions_cross_cgroup_percentile() {
    let mut stats = ScenarioStats {
        phases: vec![schbench_phase(
            1,
            vec![
                ("cg_a", taobench_serve_pc(99, 10, 0, 0)),
                ("cg_b", taobench_serve_pc(0, 0, 1, 10000)),
            ],
        )],
        ..ScenarioStats::default()
    };
    populate_run_pooled_taobench_distribution(&mut stats);
    let e = &stats.ext_metrics;
    let p99 = e
        .get("taobench_serve_p99_us_whole")
        .copied()
        .expect("union p99 present");
    assert!(
        p99 < 1000.0,
        "union p99 = percentile-of-pooled (~10 µs bucket), got {p99} \
         (mean-of-per-cgroup-p99 would be ~5005)",
    );
    let max = e
        .get("taobench_serve_max_us_whole")
        .copied()
        .expect("union max present");
    assert!(
        max > 5000.0,
        "union max = max-of-max (the high sample's ~10000 µs bucket), got {max}",
    );
}

/// The serve union spans PHASES, not only cgroups: the same cgroup contributes 99
/// low serve samples in phase 1 and 1 high in phase 2. The whole-run p99 (rank 99
/// of 100 pooled) is the LOW bucket — proving phase-1 samples are in the union —
/// while the MAX is phase-2's high sample.
#[test]
fn populate_run_pooled_taobench_distribution_unions_cross_phase_percentile() {
    let mut stats = ScenarioStats {
        phases: vec![
            schbench_phase(1, vec![("cg", taobench_serve_pc(99, 10, 0, 0))]),
            schbench_phase(2, vec![("cg", taobench_serve_pc(0, 0, 1, 10000))]),
        ],
        ..ScenarioStats::default()
    };
    populate_run_pooled_taobench_distribution(&mut stats);
    let e = &stats.ext_metrics;
    let p99 = e
        .get("taobench_serve_p99_us_whole")
        .copied()
        .expect("union p99 present");
    assert!(
        p99 < 1000.0,
        "cross-phase union p99 = percentile-of-pooled (~10 µs); phase-1 samples \
         must be in the union, got {p99}",
    );
    let max = e
        .get("taobench_serve_max_us_whole")
        .copied()
        .expect("union max present");
    assert!(
        max > 5000.0,
        "cross-phase union max = phase-2's high sample (~10000 µs), got {max}",
    );
}

/// Closed loop (or any run with no serve samples) writes NO serve `*_us_whole`
/// keys — they read ABSENT, never a false 0. Covers both an empty-phases run and a
/// phase whose taobench carrier has an empty serve histogram.
#[test]
fn populate_run_pooled_taobench_distribution_absent_without_serve_samples() {
    // Empty phases: no carriers at all.
    let mut empty = ScenarioStats::default();
    populate_run_pooled_taobench_distribution(&mut empty);
    assert!(
        !empty
            .ext_metrics
            .contains_key("taobench_serve_p99_us_whole"),
        "no serve keys for an empty run",
    );
    // A taobench carrier present but with an empty serve histogram (closed loop).
    let mut stats = ScenarioStats {
        phases: vec![schbench_phase(
            1,
            vec![("cg", taobench_serve_pc(0, 0, 0, 0))],
        )],
        ..ScenarioStats::default()
    };
    populate_run_pooled_taobench_distribution(&mut stats);
    assert!(
        !stats
            .ext_metrics
            .contains_key("taobench_serve_p99_us_whole"),
        "no serve keys when the histogram is empty (closed loop)",
    );
    assert!(
        !stats
            .ext_metrics
            .contains_key("taobench_serve_max_us_whole"),
        "no serve max key when the histogram is empty",
    );
}

/// A schbench per-phase per-cgroup carrier with the given Class-3 raw pairs.
fn schbench_pc(msg_rd: u64, msg_pc: u64, wkr_rd: u64, wkr_pc: u64, loops: u64) -> PhaseCgroupStats {
    PhaseCgroupStats {
        schbench: Some(crate::workload::schbench::run::SchbenchPhaseStats {
            msg_run_delay_ns: msg_rd,
            msg_pcount: msg_pc,
            worker_run_delay_ns: wkr_rd,
            worker_pcount: wkr_pc,
            loop_count: loops,
            ..Default::default()
        }),
        ..PhaseCgroupStats::default()
    }
}

/// A schbench per-phase per-cgroup carrier whose wakeup PlatStats holds `lows`
/// samples at `low_us` µs + `highs` samples at `high_us` µs (other streams empty).
fn schbench_wakeup_pc(lows: u32, low_us: u32, highs: u32, high_us: u32) -> PhaseCgroupStats {
    let mut p = crate::workload::schbench::plat::PlatStats::default();
    for _ in 0..lows {
        p.add_lat(low_us);
    }
    for _ in 0..highs {
        p.add_lat(high_us);
    }
    PhaseCgroupStats {
        schbench: Some(crate::workload::schbench::run::SchbenchPhaseStats {
            wakeup: p,
            ..Default::default()
        }),
        ..PhaseCgroupStats::default()
    }
}

/// A PhaseBucket carrying the given per-cgroup schbench carriers.
fn schbench_phase(step: u16, cgs: Vec<(&str, PhaseCgroupStats)>) -> PhaseBucket {
    PhaseBucket {
        step_index: step,
        label: String::new(),
        start_ms: 0,
        end_ms: 0,
        sample_count: 0,
        metrics: std::collections::BTreeMap::new(),
        per_cgroup: cgs.into_iter().map(|(n, c)| (n.to_string(), c)).collect(),
    }
}

/// populate_run_pooled_schbench sums the Class-3 raw pairs across ALL phases AND
/// ALL cgroups (the message / worker ROLES kept separate) and derives the two
/// per-schedule run-delay Rates + the loop Counter. phase1: cg_a {msg 300/3,
/// worker 500/1, 400 loops} + cg_b {worker 300/1, 600 loops}; phase2: cg_a {msg
/// 100/97, worker 200/8}. Σ: msg 400/100, worker 1000/10, loops 1000 → msg 4
/// ns/sched, worker 100 ns/sched (mean-of-ratios would give msg ~50.5).
#[test]
fn populate_run_pooled_schbench_repools_across_phases_and_cgroups() {
    let mut stats = ScenarioStats {
        phases: vec![
            schbench_phase(
                1,
                vec![
                    ("cg_a", schbench_pc(300, 3, 500, 1, 400)),
                    ("cg_b", schbench_pc(0, 0, 300, 1, 600)),
                ],
            ),
            schbench_phase(2, vec![("cg_a", schbench_pc(100, 97, 200, 8, 0))]),
        ],
        ..ScenarioStats::default()
    };
    populate_run_pooled_schbench(&mut stats);
    let e = &stats.ext_metrics;
    // Counter components Σ across phases+cgroups, roles separate.
    assert_eq!(
        e.get("total_schbench_msg_run_delay_ns").copied(),
        Some(400.0),
    );
    assert_eq!(e.get("total_schbench_msg_pcount").copied(), Some(100.0));
    assert_eq!(
        e.get("total_schbench_worker_run_delay_ns").copied(),
        Some(1000.0),
    );
    assert_eq!(e.get("total_schbench_worker_pcount").copied(), Some(10.0));
    assert_eq!(e.get("total_schbench_loops").copied(), Some(1000.0));
    // Gate-Rates = Σrun_delay / Σpcount per role.
    assert_eq!(
        e.get("schbench_msg_run_delay_ns_per_sched").copied(),
        Some(4.0),
        "Σ400/Σ100 = 4 ns/sched",
    );
    assert_eq!(
        e.get("schbench_worker_run_delay_ns_per_sched").copied(),
        Some(100.0),
        "Σ1000/Σ10 = 100 ns/sched",
    );
}

/// Role gating: a carrier with only the worker role scheduled (msg_pcount == 0)
/// emits the worker components + Rate and the loop Counter, but the message
/// components + Rate stay ABSENT (never a 0/0 rate or a false-zero component).
#[test]
fn populate_run_pooled_schbench_role_gating_omits_unscheduled_role() {
    let mut stats = ScenarioStats {
        phases: vec![schbench_phase(
            1,
            vec![("cg", schbench_pc(0, 0, 200, 4, 50))],
        )],
        ..ScenarioStats::default()
    };
    populate_run_pooled_schbench(&mut stats);
    let e = &stats.ext_metrics;
    assert_eq!(
        e.get("schbench_worker_run_delay_ns_per_sched").copied(),
        Some(50.0),
        "200/4 = 50 ns/sched",
    );
    assert_eq!(e.get("total_schbench_loops").copied(), Some(50.0));
    assert!(
        !e.contains_key("total_schbench_msg_pcount"),
        "message components absent when that role never scheduled",
    );
    assert!(
        !e.contains_key("schbench_msg_run_delay_ns_per_sched"),
        "message Rate absent (no 0/0)",
    );
}

/// No schbench carrier anywhere → no keys (a non-schbench run stays distinct from
/// a measured zero); empty phases likewise.
#[test]
fn populate_run_pooled_schbench_absent_without_schbench() {
    let mut stats = ScenarioStats {
        phases: vec![schbench_phase(1, vec![("cg", PhaseCgroupStats::default())])],
        ..ScenarioStats::default()
    };
    populate_run_pooled_schbench(&mut stats);
    assert!(
        stats.ext_metrics.is_empty(),
        "no schbench keys when no carrier ran",
    );
    let mut empty = ScenarioStats::default();
    populate_run_pooled_schbench(&mut empty);
    assert!(empty.ext_metrics.is_empty(), "no keys for empty phases");
}

/// populate_run_pooled_schbench_distribution re-pools the whole-run percentile by
/// UNIONING the per-cgroup PlatStats histograms and re-deriving — the
/// percentile-OF-the-union, NOT a mean of per-cgroup percentiles. cg_a has 99 low
/// wakeup samples (10 µs), cg_b has 1 high (10000 µs): the union p99 (rank 99 of
/// 100 pooled samples) is a LOW bucket (~10 µs), whereas mean-of-per-cgroup-p99
/// would be ≈ (10 + 10000)/2 = 5005. The union MAX is the high sample
/// (combine's max-of-max). Pins the cross-cgroup union + the non-linearity.
#[test]
fn populate_run_pooled_schbench_distribution_unions_cross_cgroup_percentile() {
    let mut stats = ScenarioStats {
        phases: vec![schbench_phase(
            1,
            vec![
                ("cg_a", schbench_wakeup_pc(99, 10, 0, 0)),
                ("cg_b", schbench_wakeup_pc(0, 0, 1, 10000)),
            ],
        )],
        ..ScenarioStats::default()
    };
    populate_run_pooled_schbench_distribution(&mut stats);
    let e = &stats.ext_metrics;
    let p99 = e
        .get("wakeup_p99_latency_us_whole")
        .copied()
        .expect("union p99 present");
    assert!(
        p99 < 1000.0,
        "union p99 = percentile-of-pooled (~10 µs bucket), got {p99} \
         (mean-of-per-cgroup-p99 would be ~5005)",
    );
    let max = e
        .get("wakeup_max_latency_us_whole")
        .copied()
        .expect("union max present");
    assert!(
        max > 5000.0,
        "union max = max-of-max (the high sample's ~10000 µs bucket), got {max}",
    );
    // request / rps streams had no samples → their keys are ABSENT (per-stream
    // gate), never a false 0.
    assert!(
        !e.contains_key("request_p99_latency_us_whole"),
        "request_*_whole absent when no request samples",
    );
    assert!(
        !e.contains_key("rps_p50_whole"),
        "rps_*_whole absent when no rps samples",
    );
}

/// The union spans PHASES, not only cgroups: the same cgroup contributes 99 low
/// wakeup samples in phase 1 and 1 high sample in phase 2. The whole-run p99 (rank
/// 99 of the 100 pooled samples) is the LOW bucket — proving phase 1's samples are
/// in the union — while the MAX is phase 2's high sample. Used alone, one phase
/// would make p99 the high (≥ 1000) OR max the low (≤ 1000); both together pin the
/// cross-phase union.
#[test]
fn populate_run_pooled_schbench_distribution_unions_cross_phase_percentile() {
    let mut stats = ScenarioStats {
        phases: vec![
            schbench_phase(1, vec![("cg", schbench_wakeup_pc(99, 10, 0, 0))]),
            schbench_phase(2, vec![("cg", schbench_wakeup_pc(0, 0, 1, 10000))]),
        ],
        ..ScenarioStats::default()
    };
    populate_run_pooled_schbench_distribution(&mut stats);
    let e = &stats.ext_metrics;
    let p99 = e
        .get("wakeup_p99_latency_us_whole")
        .copied()
        .expect("union p99 present");
    assert!(
        p99 < 1000.0,
        "cross-phase union p99 = percentile-of-pooled (~10 µs); phase-1 samples \
         must be in the union, got {p99}",
    );
    let max = e
        .get("wakeup_max_latency_us_whole")
        .copied()
        .expect("union max present");
    assert!(
        max > 5000.0,
        "cross-phase union max = phase-2's high sample (~10000 µs), got {max}",
    );
}

/// Overflow-safety: the cross-cgroup PhaseCgroupStats::merge pools the
/// guest-runtime monotonic counters with saturating arithmetic — a corrupt
/// u64::MAX component saturates (an absurd-but-finite value) instead of
/// debug-panicking / release-wrapping a derived metric to a small wrong number.
#[test]
fn phase_cgroup_merge_saturates_counter_overflow() {
    let a = PhaseCgroupStats {
        total_iterations: u64::MAX,
        total_cpu_time_ns: u64::MAX,
        total_migrations: u64::MAX,
        numa_pages_local: u64::MAX,
        numa_pages_total: u64::MAX,
        wake_sample_total: u64::MAX,
        timer_sample_total: u64::MAX,
        ..PhaseCgroupStats::default()
    };
    let b = PhaseCgroupStats {
        total_iterations: 5,
        total_cpu_time_ns: 5,
        total_migrations: 5,
        numa_pages_local: 5,
        numa_pages_total: 5,
        wake_sample_total: 5,
        timer_sample_total: 5,
        ..PhaseCgroupStats::default()
    };
    let m = PhaseCgroupStats::merge(a, b);
    assert_eq!(m.total_iterations, u64::MAX, "saturates, not wraps to 4");
    assert_eq!(m.total_cpu_time_ns, u64::MAX);
    assert_eq!(m.total_migrations, u64::MAX);
    assert_eq!(m.numa_pages_local, u64::MAX);
    assert_eq!(m.numa_pages_total, u64::MAX);
    assert_eq!(m.wake_sample_total, u64::MAX);
    assert_eq!(m.timer_sample_total, u64::MAX);
}

/// AssertResult::merge saturates the pooled run-level guest counters
/// (total_migrations / total_iterations) rather than wrapping on a corrupt
/// u64::MAX component. (total_workers / total_cpus are bounded topology counts,
/// kept as plain `+=`.)
#[test]
fn assert_result_merge_saturates_run_level_counters() {
    let mut a = AssertResult::pass();
    a.stats.total_iterations = u64::MAX;
    a.stats.total_migrations = u64::MAX;
    let mut b = AssertResult::pass();
    b.stats.total_iterations = 7;
    b.stats.total_migrations = 7;
    a.merge(b);
    assert_eq!(a.stats.total_iterations, u64::MAX, "saturates, not wraps");
    assert_eq!(a.stats.total_migrations, u64::MAX);
}

/// populate_run_pooled_schbench saturates its run-delay / pcount / loop pools on
/// a corrupt u64::MAX component (matching the already-saturating
/// SchbenchPhaseStats::merge); a wrapped sum would corrupt the gate-Rates and the
/// loop Counter.
#[test]
fn populate_run_pooled_schbench_saturates_counter_overflow() {
    let mut stats = ScenarioStats {
        phases: vec![
            schbench_phase(
                1,
                vec![(
                    "cg",
                    schbench_pc(u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX),
                )],
            ),
            schbench_phase(2, vec![("cg", schbench_pc(5, 5, 5, 5, 5))]),
        ],
        ..ScenarioStats::default()
    };
    populate_run_pooled_schbench(&mut stats);
    let e = &stats.ext_metrics;
    assert_eq!(
        e.get("total_schbench_msg_run_delay_ns").copied(),
        Some(u64::MAX as f64),
        "msg run-delay pool saturates, not wraps",
    );
    assert_eq!(
        e.get("total_schbench_loops").copied(),
        Some(u64::MAX as f64),
        "loop Counter pool saturates",
    );
}

/// populate_run_pooled_iterations_per_cpu_sec folds with saturating arithmetic:
/// two cgroups whose total_cpu_time_ns / total_iterations each approach u64::MAX
/// sum to a saturated u64::MAX (a huge-but-finite total) rather than wrapping.
#[test]
fn populate_run_pooled_iterations_per_cpu_sec_saturates_counter_overflow() {
    let mut stats = ScenarioStats {
        cgroups: vec![
            CgroupStats {
                total_cpu_time_ns: u64::MAX,
                total_iterations: u64::MAX,
                ..CgroupStats::default()
            },
            CgroupStats {
                total_cpu_time_ns: u64::MAX,
                total_iterations: 10,
                ..CgroupStats::default()
            },
        ],
        ..ScenarioStats::default()
    };
    super::run_metrics::populate_run_pooled_iterations_per_cpu_sec(&mut stats);
    let e = &stats.ext_metrics;
    assert_eq!(
        e.get("total_iterations_pooled").copied(),
        Some(u64::MAX as f64),
        "summed_iters saturates, not wraps to 9",
    );
    assert_eq!(
        e.get("total_cpu_time_sec").copied(),
        Some(u64::MAX as f64 / 1e9),
        "summed_ns saturates to a finite huge cpu-time-sec",
    );
}

/// Build-path overflow-safety: cgroup_stats pools the per-worker WorkerReport
/// counters — the FIRST cross-source fold, the one most exposed to a corrupt
/// guest report — with saturating arithmetic. A u64::MAX component saturates
/// instead of wrapping the per-cgroup total / migration-ratio to a small wrong
/// number.
#[test]
fn cgroup_stats_saturates_per_worker_counter_overflow() {
    let report = |iters: u64, mig: u64, cpu_ns: u64| crate::workload::WorkerReport {
        iterations: iters,
        migration_count: mig,
        schedstat_cpu_time_ns: cpu_ns,
        ..crate::workload::WorkerReport::default()
    };
    let reports = vec![report(u64::MAX, u64::MAX, u64::MAX), report(5, 5, 5)];
    let cg = super::reductions::cgroup_stats(&reports);
    assert_eq!(
        cg.total_iterations,
        u64::MAX,
        "iterations pool saturates, not wraps to 4",
    );
    assert_eq!(cg.total_migrations, u64::MAX, "migration pool saturates");
    assert_eq!(cg.total_cpu_time_ns, u64::MAX, "cpu-time pool saturates");
}

/// numa_agg_per_cgroup's cross-PHASE cross_node_migrated pool saturates: a
/// corrupt u64::MAX migration count summed across phases clamps to u64::MAX (→ a
/// huge-but-finite worst_cross_node_migration_ratio) rather than wrapping
/// (u64::MAX + 5 → 4) to a tiny, silently-"good" ratio.
#[test]
fn numa_agg_cross_node_migrated_pool_saturates_counter_overflow() {
    let pc = |migrated: u64| PhaseCgroupStats {
        numa_pages_local: 50,
        numa_pages_total: 100,
        cross_node_migrated: migrated,
        ..PhaseCgroupStats::default()
    };
    let mut stats = ScenarioStats {
        phases: vec![
            schbench_phase(1, vec![("cg", pc(u64::MAX))]),
            schbench_phase(2, vec![("cg", pc(5))]),
        ],
        ..ScenarioStats::default()
    };
    populate_run_distribution_metrics(&mut stats);
    let ratio = stats
        .ext_metrics
        .get("worst_cross_node_migration_ratio")
        .copied()
        .expect("worst_cross_node_migration_ratio present");
    // saturated migrated (u64::MAX) / latest total (100) ≈ 1.8e17; a wrapped sum
    // (u64::MAX + 5 → 4) would give 4/100 = 0.04.
    assert!(
        ratio > 1e10,
        "cross-phase migrated pool saturates (huge ratio), got {ratio} \
         (a wrapped sum would read ~0.04)",
    );
}

/// assert_thresholds' max_migrations absolute-count gate saturates: a corrupt
/// per-worker migration_count summing past u64::MAX clamps to u64::MAX so the
/// gate FIRES, instead of wrapping to a small value that would silently PASS a
/// limit it should fail.
#[test]
fn assert_thresholds_max_migrations_gate_saturates_not_wraps() {
    let report = |mig: u64| crate::workload::WorkerReport {
        migration_count: mig,
        iterations: 1,
        ..crate::workload::WorkerReport::default()
    };
    let reports = vec![report(u64::MAX), report(5)];
    let thresholds = super::reductions::AbsoluteThresholds {
        max_migrations: Some(1000),
        ..Default::default()
    };
    let r = super::reductions::assert_thresholds(&reports, &thresholds);
    assert!(
        r.is_fail(),
        "saturated total_mig (u64::MAX) > 1000 must FAIL the gate; a wrapped sum \
         (u64::MAX + 5 → 4) would falsely pass",
    );
}

/// run_metric(TotalMigrations / TotalIterations) saturates its cross-cgroup sum:
/// two cgroups at u64::MAX read back u64::MAX, not a wrapped-small typed value.
#[test]
fn run_metric_total_counters_saturate_cross_cgroup() {
    let cg = |mig: u64, iters: u64| CgroupStats {
        total_migrations: mig,
        total_iterations: iters,
        ..CgroupStats::default()
    };
    let stats = ScenarioStats {
        cgroups: vec![cg(u64::MAX, u64::MAX), cg(5, 5)],
        ..ScenarioStats::default()
    };
    assert_eq!(
        stats.run_metric(crate::stats::BuiltinMetric::TotalMigrations),
        Some(u64::MAX as f64),
        "total_migrations cross-cgroup typed read saturates",
    );
    assert_eq!(
        stats.run_metric(crate::stats::BuiltinMetric::TotalIterations),
        Some(u64::MAX as f64),
        "total_iterations cross-cgroup typed read saturates",
    );
}

/// PhaseBucket::cgroup_counter_total (the post_vm by-name read) saturates its
/// cross-cgroup counter pool: two per-cgroup carriers at u64::MAX read back
/// u64::MAX, not a wrapped-small value.
#[test]
fn cgroup_counter_total_saturates_cross_cgroup() {
    let pc = |mig: u64| PhaseCgroupStats {
        total_migrations: mig,
        ..PhaseCgroupStats::default()
    };
    let bucket = schbench_phase(1, vec![("a", pc(u64::MAX)), ("b", pc(5))]);
    assert_eq!(
        bucket.cgroup_counter_total("total_migrations"),
        Some(u64::MAX as f64),
        "cross-cgroup counter pool saturates, not wraps",
    );
}

#[test]
fn merge_scenario_stats_worst_wins_and_iterations_sum() {
    // Aggregates-across-cgroups contract for the MERGE-FOLDED worst-wins
    // fields: each takes the larger value (higher-is-worse max) and
    // `total_iterations` sums. The wake-latency / run-delay roll-ups and the
    // wake-latency tail ratio are no longer merge-folded (they are derived
    // `MetricKind`s re-pooled post-merge — see the `repool_*` tests, and
    // `worst_cross_node_migration_ratio` likewise moved to the post-merge re-pool;
    // this covers `worst_spread` and `worst_migration_ratio`.
    let mut a = AssertResult::pass();
    a.stats.total_iterations = 100;
    a.stats.worst_spread = 5.0;
    a.stats.worst_migration_ratio = 0.1;

    let mut b = AssertResult::pass();
    b.stats.total_iterations = 400;
    b.stats.worst_spread = 15.0;
    b.stats.worst_migration_ratio = 0.4;

    a.merge(b);

    assert_eq!(a.stats.total_iterations, 500);
    assert_eq!(a.stats.worst_spread, 15.0);
    assert_eq!(a.stats.worst_migration_ratio, 0.4);
}

#[test]
fn merge_scenario_stats_worst_wins_when_other_is_smaller() {
    // Symmetric case: when `other` reports smaller values, `self`
    // retains its larger worst. Covers the "self wins" branch of the
    // merge-folded scalar worst-comparisons: worst_spread,
    // worst_migration_ratio (both `.max()`) and the coupled worst_gap_ms/cpu
    // guard. (Wake-latency / run-delay roll-ups, the wake-latency tail ratio, and
    // both NUMA roll-ups moved to the post-merge re-pool — see the repool_* tests.)
    let mut a = AssertResult::pass();
    a.stats.worst_spread = 30.0;
    a.stats.worst_gap_ms = 500;
    a.stats.worst_gap_cpu = 7;
    a.stats.worst_migration_ratio = 0.9;
    a.stats.total_iterations = 500;

    let mut b = AssertResult::pass();
    b.stats.worst_spread = 5.0;
    b.stats.worst_gap_ms = 100;
    b.stats.worst_gap_cpu = 3;
    b.stats.worst_migration_ratio = 0.1;
    b.stats.total_iterations = 50;

    a.merge(b);

    assert_eq!(a.stats.worst_spread, 30.0);
    assert_eq!(a.stats.worst_gap_ms, 500);
    // `worst_gap_cpu` stays 7: coupling means it retains `self`'s
    // index when `self` wins on `worst_gap_ms`.
    assert_eq!(a.stats.worst_gap_cpu, 7);
    assert_eq!(a.stats.worst_migration_ratio, 0.9);
    // Totals always sum, independent of worst-wins direction.
    assert_eq!(a.stats.total_iterations, 550);
}

// `worst_page_locality` no longer merge-folds as a typed field (the
// `fold_lowest_nonzero` 0.0-sentinel path is removed). Its None-aware
// re-pool over the per-phase NUMA carriers — where a MEASURED 0.0 wins the
// lowest (the bug the sentinel masked) and an unmeasured cgroup is skipped —
// is pinned by `merge_repools_worst_page_locality_across_cgroups_measured_zero_wins`
// (in `tests_numa.rs`) and the read-side
// `run_metric_numa_fields_latest_residency_summed_migrations_none_aware`.

#[test]
fn merge_ext_metrics_higher_is_worse_takes_max() {
    // "worst_spread" is registered with higher_is_worse=true → merge max.
    let mut a = AssertResult::pass();
    a.stats.ext_metrics.insert("worst_spread".into(), 10.0);
    let mut b = AssertResult::pass();
    b.stats.ext_metrics.insert("worst_spread".into(), 42.0);
    a.merge(b);
    assert_eq!(a.stats.ext_metrics["worst_spread"], 42.0);
}

#[test]
fn merge_ext_metrics_higher_is_better_takes_min() {
    // Regression: "total_iterations" is registered with
    // higher_is_worse=false. Merge must take min (worst case)
    // rather than max (best case). Previously returned 42.0.
    let mut a = AssertResult::pass();
    a.stats.ext_metrics.insert("total_iterations".into(), 10.0);
    let mut b = AssertResult::pass();
    b.stats.ext_metrics.insert("total_iterations".into(), 42.0);
    a.merge(b);
    assert_eq!(
        a.stats.ext_metrics["total_iterations"], 10.0,
        "higher_is_worse=false must take min on merge"
    );
}

#[test]
fn merge_ext_metrics_unknown_metric_defaults_to_max() {
    // Unregistered metric names fall back to max (conservative —
    // treat as higher-is-worse until a MetricDef is registered).
    let mut a = AssertResult::pass();
    a.stats.ext_metrics.insert("unknown_metric".into(), 10.0);
    let mut b = AssertResult::pass();
    b.stats.ext_metrics.insert("unknown_metric".into(), 42.0);
    a.merge(b);
    assert_eq!(a.stats.ext_metrics["unknown_metric"], 42.0);
}

#[test]
fn merge_ext_metrics_first_insert_uses_other_value() {
    // When the key is absent on self, insert other's value verbatim
    // regardless of polarity (no prior value to compare against).
    let mut a = AssertResult::pass();
    let mut b = AssertResult::pass();
    b.stats.ext_metrics.insert("total_iterations".into(), 77.0);
    a.merge(b);
    assert_eq!(a.stats.ext_metrics["total_iterations"], 77.0);
}

#[test]
fn merge_pass_and_fail() {
    let pass = AssertResult::pass();
    let mut fail = AssertResult::pass();
    fail.record_fail(AssertDetail::new(DetailKind::Other, "something failed"));

    let mut merged = pass;
    merged.merge(fail);
    assert!(merged.is_fail(), "merging pass+fail must produce fail");
    assert!(
        merged
            .failure_details()
            .any(|d| d.message.contains("something failed"))
    );
}

#[test]
fn merge_fail_and_pass() {
    let mut fail = AssertResult::pass();
    fail.record_fail(AssertDetail::new(DetailKind::Other, "first failed"));
    let pass = AssertResult::pass();

    let mut merged = fail;
    merged.merge(pass);
    assert!(merged.is_fail(), "merging fail+pass must produce fail");
}

/// `merge` must preserve TWO independent invariants in lock-step:
/// (1) outcomes vec extends (both sides' outcomes concatenate);
/// (2) ScenarioStats fields SUM. A
/// regression that conflated the two (e.g. clamped totals to
/// outcomes.len()) would trip here. Pins the dual invariant
/// cleanly: one Fail + one Skip on distinct sides, distinct
/// stats, observe both extension AND sum.
#[test]
fn merge_outcomes_extend_and_stats_sum_coexist() {
    let mut a = AssertResult::pass();
    a.record_fail(AssertDetail::new(DetailKind::Other, "fail_a"));
    a.stats.total_iterations = 100;
    a.stats.total_workers = 2;
    let mut b = AssertResult::pass();
    b.record_skip("skip_b");
    b.stats.total_iterations = 50;
    b.stats.total_workers = 3;
    a.merge(b);
    assert_eq!(a.outcomes.len(), 2, "Fail + Skip both extend");
    assert!(a.is_fail(), "Fail dominates the verdict");
    assert_eq!(a.stats.total_iterations, 150, "stats SUM (not max)");
    assert_eq!(a.stats.total_workers, 5);
    assert_eq!(a.failure_details().count(), 1);
    assert_eq!(a.skip_details().count(), 1);
}

/// `AssertResult::merge` Inconclusive precedence: the lattice
/// is `Fail > Inconclusive > Pass > Skip`. Pin every cell of the
/// merge lattice involving Inconclusive so a regression that
/// inverts the ordering surfaces immediately.
///
/// Each sub-case constructs two AssertResults, merges them
/// commutatively (lhs+rhs AND rhs+lhs), and asserts the verdict.
/// The commutative half catches any non-symmetric short-circuit
/// (e.g. an early `if self.is_fail() return` that would mask
/// regressions when Inconclusive appears on the right).
#[test]
fn merge_inconclusive_precedence() {
    fn merged(lhs: AssertResult, rhs: AssertResult) -> AssertResult {
        let mut a = lhs;
        a.merge(rhs);
        a
    }
    fn mk_pass() -> AssertResult {
        AssertResult::pass()
    }
    fn mk_skip() -> AssertResult {
        let mut r = AssertResult::pass();
        r.record_skip("s");
        r
    }
    fn mk_inconc() -> AssertResult {
        let mut r = AssertResult::pass();
        r.record_inconclusive(AssertDetail::new(DetailKind::Other, "i"));
        r
    }
    fn mk_fail() -> AssertResult {
        let mut r = AssertResult::pass();
        r.record_fail(AssertDetail::new(DetailKind::Other, "f"));
        r
    }

    // Pass + Inconclusive => Inconclusive (both orders).
    let pi = merged(mk_pass(), mk_inconc());
    assert!(pi.is_inconclusive() && !pi.is_fail() && !pi.is_pass());
    let ip = merged(mk_inconc(), mk_pass());
    assert!(ip.is_inconclusive() && !ip.is_fail() && !ip.is_pass());

    // Skip + Inconclusive => Inconclusive (Inconclusive > Skip).
    let si = merged(mk_skip(), mk_inconc());
    assert!(si.is_inconclusive() && !si.is_skip() && !si.is_fail());
    let is_ = merged(mk_inconc(), mk_skip());
    assert!(is_.is_inconclusive() && !is_.is_skip() && !is_.is_fail());

    // Fail + Inconclusive => Fail (Fail > Inconclusive).
    let fi = merged(mk_fail(), mk_inconc());
    assert!(fi.is_fail() && !fi.is_inconclusive() && !fi.is_pass());
    let if_ = merged(mk_inconc(), mk_fail());
    assert!(if_.is_fail() && !if_.is_inconclusive() && !if_.is_pass());

    // Inconclusive + Inconclusive => Inconclusive, both extend.
    let ii = merged(mk_inconc(), mk_inconc());
    assert!(ii.is_inconclusive() && !ii.is_fail() && !ii.is_pass());
    assert_eq!(
        ii.outcomes.len(),
        2,
        "both Inconclusive outcomes extend the merged vec"
    );
}

#[test]
fn assert_result_merge_combines_stats() {
    let mut a = AssertResult {
        outcomes: vec![Outcome::Fail(AssertDetail::new(DetailKind::Other, "a"))],
        passes: vec![],
        stats: ScenarioStats {
            cgroups: vec![],
            total_workers: 2,
            total_cpus: 4,
            total_migrations: 10,
            worst_spread: 5.0,
            worst_gap_ms: 100,
            worst_gap_cpu: 0,
            ..Default::default()
        },
        measurements: std::collections::BTreeMap::new(),
        info_notes: vec![],
    };
    let b = AssertResult {
        outcomes: vec![Outcome::Fail(AssertDetail::new(DetailKind::Other, "b"))],
        passes: vec![],
        stats: ScenarioStats {
            cgroups: vec![],
            total_workers: 3,
            total_cpus: 6,
            total_migrations: 20,
            worst_spread: 15.0,
            worst_gap_ms: 500,
            worst_gap_cpu: 2,
            ..Default::default()
        },
        measurements: std::collections::BTreeMap::new(),
        info_notes: vec![],
    };
    a.merge(b);
    assert!(a.is_fail());
    assert_eq!(
        a.failure_details()
            .map(|d| d.message.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(a.stats.total_workers, 5);
    assert_eq!(a.stats.total_cpus, 10);
    assert_eq!(a.stats.total_migrations, 30);
    assert_eq!(a.stats.worst_spread, 15.0);
    assert_eq!(a.stats.worst_gap_ms, 500);
    assert_eq!(a.stats.worst_gap_cpu, 2);
}

// -- AssertResult::merge ext_metrics --

#[test]
fn assert_result_merge_ext_metrics_max_value() {
    let mut a = AssertResult::pass();
    a.stats.ext_metrics.insert("latency".into(), 10.0);
    a.stats.ext_metrics.insert("throughput".into(), 100.0);

    let mut b = AssertResult::pass();
    b.stats.ext_metrics.insert("latency".into(), 20.0);
    b.stats.ext_metrics.insert("jitter".into(), 5.0);

    a.merge(b);
    assert_eq!(a.stats.ext_metrics["latency"], 20.0);
    assert_eq!(a.stats.ext_metrics["throughput"], 100.0);
    assert_eq!(a.stats.ext_metrics["jitter"], 5.0);
}

#[test]
fn assert_result_merge_ext_metrics_keeps_larger() {
    let mut a = AssertResult::pass();
    a.stats.ext_metrics.insert("x".into(), 50.0);

    let mut b = AssertResult::pass();
    b.stats.ext_metrics.insert("x".into(), 30.0);

    a.merge(b);
    assert_eq!(a.stats.ext_metrics["x"], 50.0);
}

// -- AssertResult::merge per-phase --
//
// Pins the per-step-index phase merge dispatch through
// `MetricKind::merge_kind`. Counter / Peak / Gauge(Max) /
// Gauge(Avg) follow the commutative paths; Gauge(Last) /
// Timestamp use the `end_ms` tiebreak. Unpaired phases (one
// side only) carry through verbatim per the no-silent-drops
// contract.

fn phase_bucket(
    step_index: u16,
    label: &str,
    start_ms: u64,
    end_ms: u64,
    sample_count: usize,
    metrics: &[(&str, f64)],
) -> PhaseBucket {
    PhaseBucket {
        per_cgroup: Default::default(),
        step_index,
        label: label.to_string(),
        start_ms,
        end_ms,
        sample_count,
        metrics: metrics
            .iter()
            .map(|(k, v)| ((*k).to_string(), *v))
            .collect(),
    }
}

#[test]
fn assert_result_merge_per_phase_counter_sums() {
    // `total_migrations` is `MetricKind::Counter`; the per-phase
    // merge sums the two reduced values so multiple cgroups'
    // per-phase deltas accumulate.
    let mut a = AssertResult::pass();
    a.stats.phases = vec![phase_bucket(
        1,
        "Step[0]",
        0,
        100,
        5,
        &[("total_migrations", 25.0)],
    )];
    let mut b = AssertResult::pass();
    b.stats.phases = vec![phase_bucket(
        1,
        "Step[0]",
        0,
        100,
        5,
        &[("total_migrations", 75.0)],
    )];
    a.merge(b);
    assert_eq!(a.stats.phases.len(), 1);
    assert_eq!(a.stats.phases[0].metrics["total_migrations"], 100.0);
}

#[test]
fn assert_result_merge_per_phase_peak_takes_max() {
    // `worst_gap_ms` is `MetricKind::Peak`; the per-phase merge
    // takes the max so a worse peak on either side wins.
    let mut a = AssertResult::pass();
    a.stats.phases = vec![phase_bucket(
        2,
        "Step[1]",
        0,
        100,
        5,
        &[("worst_gap_ms", 12.0)],
    )];
    let mut b = AssertResult::pass();
    b.stats.phases = vec![phase_bucket(
        2,
        "Step[1]",
        0,
        100,
        5,
        &[("worst_gap_ms", 7.0)],
    )];
    a.merge(b);
    assert_eq!(a.stats.phases[0].metrics["worst_gap_ms"], 12.0);
}

#[test]
fn assert_result_merge_per_phase_per_cgroup_unions_and_folds() {
    use crate::assert::PhaseCgroupStats;
    use std::collections::BTreeSet;
    // Two same-step buckets. cg_0 is shared — its RAW components FOLD by class:
    // sample Vecs (latencies/run_delays/off_cpu_pcts) CONCAT, cpus_used UNIONs,
    // genuine counters (num_workers, migrations, iterations, cpu time, numa
    // pages, wake_sample_total) SUM — num_workers included, because two carriers
    // for one cgroup name are per-handle subsets covering DISJOINT workers — and
    // the one Peak cross_node_migrated [system-wide vmstat delta] takes MAX while
    // the coupled gap takes ARGMAX. cg_x (shared) pins not-measured (empty
    // off_cpu_pcts) UNION measured. cg_a/cg_b are one-sided and carried verbatim
    // BY VALUE. Pins the per_cgroup union in merge_matched_phase_buckets.
    let mut a = AssertResult::pass();
    let mut a_bucket = phase_bucket(1, "Step[0]", 0, 100, 5, &[]);
    a_bucket.per_cgroup.insert(
        "cg_0".to_string(),
        PhaseCgroupStats {
            num_workers: 4,
            cpus_used: BTreeSet::from([0, 1]),
            wake_latencies_ns: vec![10, 20],
            wake_sample_total: 2,
            timer_latencies_ns: vec![],
            timer_sample_total: 0,
            run_delays_ns: vec![1_000],
            off_cpu_pcts: vec![5.0, 20.0],
            total_migrations: 3,
            total_iterations: 100,
            total_cpu_time_ns: 1_000,
            numa_pages_local: 90,
            numa_pages_total: 100,
            cross_node_migrated: 100,
            max_gap_ms: 7,
            max_gap_cpu: 3,
            stripped: false,
            metrics: std::collections::BTreeMap::new(),
            schbench: None,
            taobench: None,
        },
    );
    a_bucket.per_cgroup.insert(
        "cg_x".to_string(),
        PhaseCgroupStats {
            off_cpu_pcts: vec![], // not measured on a's side
            ..Default::default()
        },
    );
    a_bucket.per_cgroup.insert(
        "cg_a".to_string(),
        PhaseCgroupStats {
            num_workers: 1,
            ..Default::default()
        },
    );
    a.stats.phases = vec![a_bucket];

    let mut b = AssertResult::pass();
    let mut b_bucket = phase_bucket(1, "Step[0]", 0, 100, 5, &[]);
    b_bucket.per_cgroup.insert(
        "cg_0".to_string(),
        PhaseCgroupStats {
            num_workers: 4,
            cpus_used: BTreeSet::from([1, 2]),
            wake_latencies_ns: vec![30],
            wake_sample_total: 1,
            timer_latencies_ns: vec![],
            timer_sample_total: 0,
            run_delays_ns: vec![2_000, 3_000],
            off_cpu_pcts: vec![3.0, 15.0],
            total_migrations: 2,
            total_iterations: 50,
            total_cpu_time_ns: 500,
            numa_pages_local: 40,
            numa_pages_total: 50,
            cross_node_migrated: 50,
            max_gap_ms: 9,
            max_gap_cpu: 5,
            stripped: false,
            metrics: std::collections::BTreeMap::new(),
            schbench: None,
            taobench: None,
        },
    );
    b_bucket.per_cgroup.insert(
        "cg_x".to_string(),
        PhaseCgroupStats {
            off_cpu_pcts: vec![7.0], // measured on b's side
            ..Default::default()
        },
    );
    b_bucket.per_cgroup.insert(
        "cg_b".to_string(),
        PhaseCgroupStats {
            num_workers: 2,
            ..Default::default()
        },
    );
    b.stats.phases = vec![b_bucket];

    a.merge(b);
    let pc = &a.stats.phases[0].per_cgroup;
    // One-sided cgroups carried verbatim BY VALUE (not just key presence).
    assert_eq!(pc["cg_a"].num_workers, 1, "cg_a (a-only) carried by value");
    assert_eq!(pc["cg_b"].num_workers, 2, "cg_b (b-only) carried by value");
    // Shared cg_0 folded component-wise.
    let cg0 = &pc["cg_0"];
    assert_eq!(cg0.wake_latencies_ns, vec![10, 20, 30], "latencies concat");
    assert_eq!(cg0.wake_sample_total, 3, "wake_sample_total sums");
    assert_eq!(
        cg0.run_delays_ns,
        vec![1_000, 2_000, 3_000],
        "run_delays concat (raw ns)"
    );
    assert_eq!(
        cg0.off_cpu_pcts,
        vec![5.0, 20.0, 3.0, 15.0],
        "off_cpu_pcts concat (mean + spread re-pool from these raw samples)",
    );
    assert_eq!(cg0.cpus_used, BTreeSet::from([0, 1, 2]), "cpus_used union");
    assert_eq!(cg0.total_migrations, 5, "migrations sum (3+2)");
    assert_eq!(cg0.total_iterations, 150, "iterations sum (100+50)");
    assert_eq!(cg0.total_cpu_time_ns, 1_500, "cpu time sum (1000+500)");
    assert_eq!(cg0.numa_pages_local, 130, "numa_pages_local sum (90+40)");
    assert_eq!(cg0.numa_pages_total, 150, "numa_pages_total sum (100+50)");
    assert_eq!(
        cg0.cross_node_migrated, 100,
        "cross_node_migrated MAX(100,50)=100 NOT 150 — system-wide vmstat delta",
    );
    // Coupled worst gap folds as an ARGMAX: b has the larger gap (9 > 7) so
    // BOTH ms and cpu come from b — never desynced into a's cpu.
    assert_eq!(
        cg0.max_gap_ms, 9,
        "gap ms = argmax-by-ms(7@cpu3, 9@cpu5) = 9"
    );
    assert_eq!(
        cg0.max_gap_cpu, 5,
        "gap cpu coupled to the winning gap (b's 5, NOT a's 3)",
    );
    assert_eq!(
        cg0.num_workers, 8,
        "num_workers SUMs (4+4) — a Counter over disjoint per-handle worker \
         subsets, not a Peak",
    );
    // Not-measured (empty) UNION measured = measured (empty concat is a no-op).
    assert_eq!(
        pc["cg_x"].off_cpu_pcts,
        vec![7.0],
        "empty off_cpu_pcts (not measured) ∪ measured = measured",
    );
}

/// Cross-STEP per_cgroup survival through `AssertResult::merge`: two DIFFERENT
/// step_index carriers (step 1 cgA, step 2 cgB) BOTH reach the merged output via
/// the unpaired-step-index arm. This is the cross-phase core invariant — each
/// step's per_cgroup survives the guest-side merge, not just a single matched
/// step. The per_cgroup union test above only exercises a SINGLE matched
/// step_index; this pins the unpaired (different-step) path carries per_cgroup.
#[test]
fn assert_result_merge_keeps_per_cgroup_across_distinct_steps() {
    use crate::assert::{PhaseBucket, PhaseCgroupStats};
    let bucket = |idx: u16, name: &str, iters: u64| {
        let mut pc = std::collections::BTreeMap::new();
        pc.insert(
            name.to_string(),
            PhaseCgroupStats {
                total_iterations: iters,
                ..Default::default()
            },
        );
        PhaseBucket {
            step_index: idx,
            label: format!("Step[{}]", idx - 1),
            start_ms: 0,
            end_ms: 100,
            sample_count: 0,
            metrics: std::collections::BTreeMap::new(),
            per_cgroup: pc,
        }
    };
    let mut a = AssertResult::pass();
    a.stats.phases = vec![bucket(1, "cgA", 11)];
    let mut b = AssertResult::pass();
    b.stats.phases = vec![bucket(2, "cgB", 22)];
    a.merge(b);
    assert_eq!(
        a.stats.phases.len(),
        2,
        "both distinct-step buckets survive"
    );
    let s1 = a
        .stats
        .phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 survives");
    let s2 = a
        .stats
        .phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("step 2 survives");
    assert_eq!(
        s1.per_cgroup["cgA"].total_iterations, 11,
        "step 1 per_cgroup carried"
    );
    assert_eq!(
        s2.per_cgroup["cgB"].total_iterations, 22,
        "step 2 per_cgroup carried"
    );
}

#[test]
fn assert_result_merge_per_phase_gauge_last_takes_later_end_ms() {
    // `worst_spread` is `MetricKind::Gauge(GaugeAgg::Last)`. The
    // per-phase merge resolves to the value from the bucket with
    // the later `end_ms` per `MergeKind::NonCommutative`. The
    // arrival order doesn't decide the winner — the timestamp does.
    let mut a = AssertResult::pass();
    a.stats.phases = vec![phase_bucket(
        1,
        "Step[0]",
        0,
        200,
        5,
        &[("worst_spread", 0.42)],
    )];
    let mut b = AssertResult::pass();
    b.stats.phases = vec![phase_bucket(
        1,
        "Step[0]",
        0,
        100,
        5,
        &[("worst_spread", 0.11)],
    )];
    a.merge(b);
    // a.end_ms = 200 > b.end_ms = 100 → a's value wins.
    assert_eq!(a.stats.phases[0].metrics["worst_spread"], 0.42);
    // Merged window covers both: start_ms = min, end_ms = max.
    assert_eq!(a.stats.phases[0].start_ms, 0);
    assert_eq!(a.stats.phases[0].end_ms, 200);
}

#[test]
fn assert_result_merge_per_phase_gauge_last_reverse_picks_later_end_ms() {
    // Same metric, opposite end_ms ordering — verifies the
    // NonCommutative tiebreak follows the timestamp, not the
    // operand order. `b` has the later `end_ms` so b's value wins
    // even though it's on the right side of the merge.
    let mut a = AssertResult::pass();
    a.stats.phases = vec![phase_bucket(
        1,
        "Step[0]",
        0,
        100,
        5,
        &[("worst_spread", 0.42)],
    )];
    let mut b = AssertResult::pass();
    b.stats.phases = vec![phase_bucket(
        1,
        "Step[0]",
        0,
        200,
        5,
        &[("worst_spread", 0.11)],
    )];
    a.merge(b);
    // b.end_ms = 200 > a.end_ms = 100 → b's value wins.
    assert_eq!(a.stats.phases[0].metrics["worst_spread"], 0.11);
    assert_eq!(a.stats.phases[0].end_ms, 200);
}

#[test]
fn assert_result_merge_per_phase_unpaired_step_indices_keep_both() {
    // One side has step_index 1, the other has step_index 2. The
    // merge keeps both — no-silent-drops contract.
    let mut a = AssertResult::pass();
    a.stats.phases = vec![phase_bucket(
        1,
        "Step[0]",
        0,
        100,
        3,
        &[("total_migrations", 5.0)],
    )];
    let mut b = AssertResult::pass();
    b.stats.phases = vec![phase_bucket(
        2,
        "Step[1]",
        100,
        200,
        3,
        &[("total_migrations", 8.0)],
    )];
    a.merge(b);
    assert_eq!(a.stats.phases.len(), 2);
    // Sorted by step_index for deterministic output.
    assert_eq!(a.stats.phases[0].step_index, 1);
    assert_eq!(a.stats.phases[1].step_index, 2);
    assert_eq!(a.stats.phases[0].metrics["total_migrations"], 5.0);
    assert_eq!(a.stats.phases[1].metrics["total_migrations"], 8.0);
}

#[test]
fn assert_result_merge_per_phase_unknown_metric_takes_mean() {
    // Unregistered metric name → fallback to arithmetic mean. The
    // safest commutative default when the merge can't query
    // `MetricKind`.
    let mut a = AssertResult::pass();
    a.stats.phases = vec![phase_bucket(
        0,
        "BASELINE",
        0,
        100,
        5,
        &[("custom.metric", 10.0)],
    )];
    let mut b = AssertResult::pass();
    b.stats.phases = vec![phase_bucket(
        0,
        "BASELINE",
        0,
        100,
        5,
        &[("custom.metric", 30.0)],
    )];
    a.merge(b);
    assert_eq!(a.stats.phases[0].metrics["custom.metric"], 20.0);
}

#[test]
fn assert_result_merge_per_phase_one_side_only_keeps_value() {
    // Metric present on one side only inside an otherwise-paired
    // step_index. The merge takes the available value (no fold
    // against a missing operand).
    let mut a = AssertResult::pass();
    a.stats.phases = vec![phase_bucket(
        1,
        "Step[0]",
        0,
        100,
        5,
        &[("total_migrations", 7.0), ("worst_gap_ms", 12.0)],
    )];
    let mut b = AssertResult::pass();
    b.stats.phases = vec![phase_bucket(
        1,
        "Step[0]",
        0,
        100,
        5,
        &[("total_migrations", 3.0)],
    )];
    a.merge(b);
    assert_eq!(a.stats.phases[0].metrics["total_migrations"], 10.0);
    assert_eq!(a.stats.phases[0].metrics["worst_gap_ms"], 12.0);
}

#[test]
fn assert_result_merge_per_phase_window_invariants() {
    // start_ms = min, end_ms = max, sample_count = sum across
    // both sides. The merged window spans every sample reported
    // by either side.
    let mut a = AssertResult::pass();
    a.stats.phases = vec![phase_bucket(1, "Step[0]", 50, 150, 4, &[])];
    let mut b = AssertResult::pass();
    b.stats.phases = vec![phase_bucket(1, "Step[0]", 10, 200, 6, &[])];
    a.merge(b);
    assert_eq!(a.stats.phases[0].start_ms, 10);
    assert_eq!(a.stats.phases[0].end_ms, 200);
    assert_eq!(a.stats.phases[0].sample_count, 10);
}

#[test]
fn merge_kind_enum_exhaustively_covers_metric_kind_variants() {
    // Every `MetricKind` must map to a `MergeKind` via
    // `MetricKind::merge_kind`. Exercising every variant here
    // means a new `MetricKind` addition either compiles (variant
    // listed in `merge_kind`'s exhaustive match) or fails the
    // build at that match site — never silently falls through to
    // a wrong default.
    use crate::stats::{GaugeAgg, MergeKind, MetricKind};
    assert_eq!(MetricKind::Counter.merge_kind(), MergeKind::Commutative);
    assert_eq!(MetricKind::Peak.merge_kind(), MergeKind::Commutative);
    assert_eq!(
        MetricKind::Gauge(GaugeAgg::Avg).merge_kind(),
        MergeKind::Commutative,
    );
    assert_eq!(
        MetricKind::Gauge(GaugeAgg::Max).merge_kind(),
        MergeKind::Commutative,
    );
    assert_eq!(
        MetricKind::Gauge(GaugeAgg::Last).merge_kind(),
        MergeKind::NonCommutative,
    );
    assert_eq!(
        MetricKind::Timestamp.merge_kind(),
        MergeKind::NonCommutative,
    );
    assert_eq!(MetricKind::DeltaSum.merge_kind(), MergeKind::Commutative);
    // PerPhaseDeltaSum's cross-cgroup same-step merge sums the per-cgroup
    // CPU-time deltas (Commutative, like Counter); the SUM-cross-phase /
    // MEAN-cross-run split lives at the fold sites, not in merge_kind.
    assert_eq!(
        MetricKind::PerPhaseDeltaSum.merge_kind(),
        MergeKind::Commutative,
    );
    assert_eq!(
        MetricKind::Rate {
            numerator: "n",
            denominator: "d",
        }
        .merge_kind(),
        MergeKind::Recompute,
    );
    // The derived kinds are all Recompute (re-derived post-merge, never folded
    // from two ready-made values): the per-phase merge loop skips them
    // (is_derived) and `populate_run_distribution_metrics` / `derive_phase_metrics`
    // re-pool the value. Pin each so a new derived kind that forgets the Recompute
    // arm fails here, not silently.
    assert_eq!(
        MetricKind::Distribution {
            source: crate::stats::SampleSource::WakeLatencyNs,
            reduction: crate::stats::SampleReduction::P99,
        }
        .merge_kind(),
        MergeKind::Recompute,
    );
    assert_eq!(
        MetricKind::WorstLowest {
            numerator: crate::stats::WorstLowestNumerator::NumaLocal,
            denominator: crate::stats::WorstLowestDenominator::NumaTotal,
        }
        .merge_kind(),
        MergeKind::Recompute,
    );
    assert_eq!(
        MetricKind::WakeLatencyTailRatio.merge_kind(),
        MergeKind::Recompute,
    );
    assert_eq!(
        MetricKind::WorstCrossNodeRatio.merge_kind(),
        MergeKind::Recompute,
    );
    assert_eq!(MetricKind::PerPhase.merge_kind(), MergeKind::Recompute);
    assert_eq!(
        MetricKind::PerRunDistribution.merge_kind(),
        MergeKind::Recompute,
    );
}

/// merge_matched_phase_buckets must INCLUDE a synthesized
/// (sample_count==0) bucket's capture-independent iteration_rate when
/// merging it against a captured bucket at the same step_index. Since
/// iteration_rate is a MetricKind::Rate, the merge sums each side's
/// Counter components (total_phase_iterations / total_phase_duration_sec)
/// and re-derives the rate as Σiters/Σseconds — so the synthesized side's
/// iterations are pooled in, never dropped. Guards the no-silent-drops
/// invariant for any cross-result phase merge (e.g. the per-cgroup
/// phase-bucket fold). Unequal durations make the re-pool (450) distinct
/// from a mean-of-ratios (500) and a dropped-synthesized result (400).
#[test]
fn merge_matched_phase_buckets_repools_synthesized_zero_count() {
    use std::collections::BTreeMap;
    let synth = PhaseBucket {
        per_cgroup: Default::default(),
        step_index: 2,
        label: "Step[1]".to_string(),
        start_ms: 2000,
        end_ms: 3000,
        sample_count: 0, // synthesized zero-capture bucket: 600 iters / 1s
        metrics: BTreeMap::from([
            ("total_phase_iterations".to_string(), 600.0),
            ("total_phase_duration_sec".to_string(), 1.0),
        ]),
    };
    let captured = PhaseBucket {
        per_cgroup: Default::default(),
        step_index: 2,
        label: "Step[1]".to_string(),
        start_ms: 2000,
        end_ms: 3000,
        sample_count: 5, // 1200 iters / 3s
        metrics: BTreeMap::from([
            ("total_phase_iterations".to_string(), 1200.0),
            ("total_phase_duration_sec".to_string(), 3.0),
        ]),
    };
    let merged = merge_matched_phase_buckets(synth, captured);
    // Re-pool: Σiters / Σseconds = (600 + 1200) / (1 + 3) = 1800/4 = 450/s.
    // The synthesized side's 600 iters are SUMMED in (Counter merge), so
    // 450 — NOT 400 (synthesized dropped, captured 1200/3) and NOT 500
    // (mean of the two ready ratios 600 and 400).
    let r = merged
        .metrics
        .get("iteration_rate")
        .copied()
        .expect("merged bucket carries the re-derived iteration_rate");
    assert!(
        (r - 450.0).abs() < f64::EPSILON,
        "synthesized rate's iterations must pool into Σiters/Σseconds = 450, \
         not be dropped (400) or averaged as ratios (500); got {r}",
    );
    assert_eq!(
        merged.metrics.get("total_phase_iterations").copied(),
        Some(1800.0),
        "iteration components sum across the merged buckets",
    );
}
