//! `AssertResult::merge` and the per-field worst-wins / lowest-non-zero
//! / sum aggregation rules for `ScenarioStats`. Every polarity is
//! exercised in both directions so a sign-flip regression surfaces
//! regardless of which side carries the worse value.

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
///   - pick the worst value of every higher-is-worse
///     `worst_*` field across all merged cgroups;
///   - pick the lowest-non-zero value of `worst_page_locality`
///     and `worst_iterations_per_worker` (0.0 is the unreported
///     sentinel for both fields, matching the accumulator-pass
///     convention in `AssertResult::pass().merge(real)`);
///   - SUM `total_iterations` across all cgroups, not max it.
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
        worst_p99_us: f64,
        total_iters: u64,
        page_locality: f64,
        iters_per_worker: f64,
        cg_total_iters: u64,
    ) -> AssertResult {
        let cg = CgroupStats {
            total_iterations: cg_total_iters,
            page_locality,
            ..CgroupStats::default()
        };
        // `iters_per_worker` flows into the ScenarioStats roll-up
        // below; the per-cgroup [`CgroupStats::iterations_per_worker`]
        // is now method-only and recomputed on read from
        // `total_iterations / num_workers`.
        AssertResult {
            outcomes: vec![],
            passes: vec![],
            stats: ScenarioStats {
                total_iterations: total_iters,
                worst_spread,
                worst_migration_ratio: worst_mig,
                worst_p99_wake_latency_us: worst_p99_us,
                worst_page_locality: page_locality,
                worst_iterations_per_worker: Some(iters_per_worker),
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
    let mut acc = mk(10.0, 0.1, 50.0, 100, 0.8, 300.0, 100);
    acc.merge(mk(5.0, 0.3, 20.0, 200, 0.5, 150.0, 200));
    acc.merge(mk(20.0, 0.2, 70.0, 400, 0.9, 500.0, 400));

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

    // Worst-wins across 3 cgroups (higher-is-worse):
    assert_eq!(s.worst_spread, 20.0, "third cgroup's 20.0 is worst");
    assert_eq!(s.worst_migration_ratio, 0.3, "second cgroup's 0.3 is worst");
    assert_eq!(
        s.worst_p99_wake_latency_us, 70.0,
        "third cgroup's 70.0us p99 is worst",
    );
    // Lower-is-worse rollups across 3 cgroups (every value is
    // strictly positive so the sentinel/None branches are never
    // taken). `worst_page_locality` uses `fold_lowest_nonzero`;
    // `worst_iterations_per_worker` uses the None-aware
    // `fold_lowest_some`:
    assert_eq!(
        s.worst_page_locality, 0.5,
        "second cgroup's 0.5 is the lowest-non-zero — 0 sentinel never wins",
    );
    assert_eq!(
        s.worst_iterations_per_worker,
        Some(150.0),
        "second cgroup's 150 is the lowest per-worker throughput",
    );
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
fn merge_worst_iterations_per_worker_lets_measured_zero_win() {
    // The regression this pins: a cgroup that ran zero iterations
    // (Some(0.0)) is the worst possible per-worker throughput and MUST
    // win the "lowest" bucket. The old `fold_lowest_nonzero` treated
    // 0.0 as an unreported sentinel and discarded it, hiding the
    // starved cgroup behind a healthier one's reading.
    fn with_worst(v: Option<f64>) -> AssertResult {
        AssertResult {
            outcomes: vec![],
            passes: vec![],
            stats: ScenarioStats {
                worst_iterations_per_worker: v,
                ..ScenarioStats::default()
            },
            measurements: std::collections::BTreeMap::new(),
            info_notes: vec![],
        }
    }

    // Accumulator starts None (Default).
    let mut acc = AssertResult::pass();
    assert_eq!(acc.stats.worst_iterations_per_worker, None);

    // Fold a healthy reading, then a measured zero: the zero wins.
    acc.merge(with_worst(Some(100.0)));
    acc.merge(with_worst(Some(0.0)));
    assert_eq!(
        acc.stats.worst_iterations_per_worker,
        Some(0.0),
        "a cgroup that ran zero iterations must win the worst bucket",
    );

    // A later healthy reading does not displace the worst zero.
    acc.merge(with_worst(Some(250.0)));
    assert_eq!(acc.stats.worst_iterations_per_worker, Some(0.0));
}

#[test]
fn merge_worst_iterations_per_worker_skips_no_data() {
    // None contributors (cgroups with no workers) are skipped, never
    // treated as zero — they must not displace a real reading nor
    // fabricate a Some(0.0).
    fn with_worst(v: Option<f64>) -> AssertResult {
        AssertResult {
            outcomes: vec![],
            passes: vec![],
            stats: ScenarioStats {
                worst_iterations_per_worker: v,
                ..ScenarioStats::default()
            },
            measurements: std::collections::BTreeMap::new(),
            info_notes: vec![],
        }
    }

    // Only None folded in → stays None (no data, not a zero).
    let mut acc = AssertResult::pass();
    acc.merge(with_worst(None));
    acc.merge(with_worst(None));
    assert_eq!(acc.stats.worst_iterations_per_worker, None);

    // None does not displace a real reading.
    acc.merge(with_worst(Some(75.0)));
    acc.merge(with_worst(None));
    assert_eq!(acc.stats.worst_iterations_per_worker, Some(75.0));
}

#[test]
fn merge_worst_iterations_per_cpu_sec_lowest_wins_none_aware() {
    // worst_iterations_per_cpu_sec (overcommit-invariant efficiency)
    // uses the same fold_lowest_some merge as worst_iterations_per_worker:
    // the least-efficient cgroup wins (lowest Some), a measured Some(0.0)
    // beats a healthy reading, and None (no defined rate) is skipped,
    // never fabricated as zero.
    fn with(v: Option<f64>) -> AssertResult {
        AssertResult {
            outcomes: vec![],
            passes: vec![],
            stats: ScenarioStats {
                worst_iterations_per_cpu_sec: v,
                ..ScenarioStats::default()
            },
            measurements: std::collections::BTreeMap::new(),
            info_notes: vec![],
        }
    }
    let mut acc = AssertResult::pass();
    assert_eq!(acc.stats.worst_iterations_per_cpu_sec, None);
    // None skipped; the real reading survives.
    acc.merge(with(None));
    acc.merge(with(Some(900.0)));
    assert_eq!(acc.stats.worst_iterations_per_cpu_sec, Some(900.0));
    // Lower efficiency wins; a measured zero is the worst and is not
    // displaced by a later healthy reading.
    acc.merge(with(Some(0.0)));
    acc.merge(with(Some(1500.0)));
    assert_eq!(acc.stats.worst_iterations_per_cpu_sec, Some(0.0));
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
        worst_iterations_per_cpu_sec: cg.iterations_per_cpu_sec(),
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
    // The existing worst_iterations_per_cpu_sec (min-fold starvation
    // selector) is UNCHANGED by the new pooled metric: the lower per-cgroup
    // rate (cg2's 10/9) wins.
    let worst = acc
        .stats
        .worst_iterations_per_cpu_sec
        .expect("worst_iterations_per_cpu_sec present");
    assert!(
        (worst - 10.0 / 9.0).abs() < 1e-9,
        "worst_iterations_per_cpu_sec stays the min-fold selector (~1.11), \
         not mutated by the pooled metric; got {worst}",
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

#[test]
fn merge_scenario_stats_worst_wins_and_iterations_sum() {
    // Aggregates-across-cgroups contract: every `worst_*` field on
    // ScenarioStats takes the larger value between the two cgroups,
    // and `total_iterations` sums. Exercises fields that are not
    // covered by the narrower merge_takes_worst_* tests: the wake-
    // latency trio, the run-delay pair, the migration ratio, and
    // the cross-node migration ratio.
    let mut a = AssertResult::pass();
    a.stats.total_iterations = 100;
    a.stats.worst_spread = 5.0;
    a.stats.worst_migration_ratio = 0.1;
    a.stats.worst_p99_wake_latency_us = 20.0;
    a.stats.worst_median_wake_latency_us = 10.0;
    a.stats.worst_wake_latency_cv = 0.2;
    a.stats.worst_run_delay_us = 50.0;
    a.stats.worst_mean_run_delay_us = 30.0;
    a.stats.worst_cross_node_migration_ratio = 0.05;

    let mut b = AssertResult::pass();
    b.stats.total_iterations = 400;
    b.stats.worst_spread = 15.0;
    b.stats.worst_migration_ratio = 0.4;
    b.stats.worst_p99_wake_latency_us = 80.0;
    b.stats.worst_median_wake_latency_us = 40.0;
    b.stats.worst_wake_latency_cv = 0.5;
    b.stats.worst_run_delay_us = 120.0;
    b.stats.worst_mean_run_delay_us = 90.0;
    b.stats.worst_cross_node_migration_ratio = 0.25;

    a.merge(b);

    assert_eq!(a.stats.total_iterations, 500);
    assert_eq!(a.stats.worst_spread, 15.0);
    assert_eq!(a.stats.worst_migration_ratio, 0.4);
    assert_eq!(a.stats.worst_p99_wake_latency_us, 80.0);
    assert_eq!(a.stats.worst_median_wake_latency_us, 40.0);
    assert_eq!(a.stats.worst_wake_latency_cv, 0.5);
    assert_eq!(a.stats.worst_run_delay_us, 120.0);
    assert_eq!(a.stats.worst_mean_run_delay_us, 90.0);
    assert_eq!(a.stats.worst_cross_node_migration_ratio, 0.25);
}

/// `ScenarioStats::merge` rolls up the new derived-ratio fields
/// across cgroups with opposite polarities: `worst_wake_latency_tail_ratio`
/// is higher-is-worse (max), `worst_iterations_per_worker` is
/// lower-is-worse (`fold_lowest_nonzero` — 0.0 is the unreported
/// sentinel matching the accumulator-pass convention; the
/// `AssertResult::pass().merge(real)` pattern relies on a
/// positive `other` overriding `self`'s default-zero rather
/// than being masked by it).  A regression that merged either
/// with the wrong polarity would surface a regression as an
/// improvement or vice versa — exactly the kind of sign-flip
/// that would silently break `stats compare`.
#[test]
fn merge_derived_ratios_use_correct_polarities() {
    let mut a = AssertResult::pass();
    a.stats.worst_wake_latency_tail_ratio = 2.0;
    a.stats.worst_iterations_per_worker = Some(500.0);

    let mut b = AssertResult::pass();
    b.stats.worst_wake_latency_tail_ratio = 8.0;
    b.stats.worst_iterations_per_worker = Some(100.0);

    a.merge(b);

    assert_eq!(
        a.stats.worst_wake_latency_tail_ratio, 8.0,
        "tail ratio uses max — 8.0 is worse than 2.0 (more \
         amplification); got {}",
        a.stats.worst_wake_latency_tail_ratio,
    );
    assert_eq!(
        a.stats.worst_iterations_per_worker,
        Some(100.0),
        "iterations_per_worker uses lowest — 100.0 is worse than \
         500.0 (less throughput per worker); got {:?}",
        a.stats.worst_iterations_per_worker,
    );

    // No-data convention, direction 1: a `None` reading on `other`
    // (no cgroup reported a worker) MUST NOT clobber self's real
    // measurement. `fold_lowest_some` skips None contributors.
    let mut c = AssertResult::pass();
    c.stats.worst_iterations_per_worker = Some(300.0);
    let mut empty = AssertResult::pass();
    empty.stats.worst_iterations_per_worker = None;
    c.merge(empty);
    assert_eq!(
        c.stats.worst_iterations_per_worker,
        Some(300.0),
        "self=Some(300) must be retained when other=None (no data); \
         got {:?}",
        c.stats.worst_iterations_per_worker,
    );

    // No-data convention, direction 2: `self` starts at None (the
    // accumulator default from `AssertResult::pass()`) and `other`
    // reports a real reading. self must adopt other's measurement;
    // this is the load-bearing case for
    // `AssertResult::pass().merge(real)`.
    let mut d = AssertResult::pass();
    assert_eq!(
        d.stats.worst_iterations_per_worker, None,
        "accumulator default must be None",
    );
    let mut real = AssertResult::pass();
    real.stats.worst_iterations_per_worker = Some(300.0);
    d.merge(real);
    assert_eq!(
        d.stats.worst_iterations_per_worker,
        Some(300.0),
        "self=None (accumulator default) must adopt other=Some(300) \
         — the `AssertResult::pass().merge(real)` pattern depends \
         on this; got {:?}",
        d.stats.worst_iterations_per_worker,
    );

    // Both-None: no data on either side stays None. A measured
    // Some(0.0) winning the bucket is covered separately by
    // `merge_worst_iterations_per_worker_lets_measured_zero_win`.
    let mut e = AssertResult::pass();
    e.stats.worst_iterations_per_worker = None;
    let mut f = AssertResult::pass();
    f.stats.worst_iterations_per_worker = None;
    e.merge(f);
    assert_eq!(
        e.stats.worst_iterations_per_worker, None,
        "both-None must stay None; got {:?}",
        e.stats.worst_iterations_per_worker,
    );

    // Tail-ratio polarity, reverse direction: when `self`
    // starts at the higher value and `other` is smaller,
    // `self` must retain its larger worst. Pair with the
    // forward direction above (self=2, other=8 → 8) so both
    // branches of the `.max()` are pinned — otherwise a
    // regression that silently flipped to `.min()` would
    // pass the forward-direction assertion and surface
    // only here.
    let mut g = AssertResult::pass();
    g.stats.worst_wake_latency_tail_ratio = 8.0;
    let mut h = AssertResult::pass();
    h.stats.worst_wake_latency_tail_ratio = 2.0;
    g.merge(h);
    assert_eq!(
        g.stats.worst_wake_latency_tail_ratio, 8.0,
        "tail_ratio uses max: self=8.0, other=2.0 → self \
         retains 8.0 (higher is worse); got {}",
        g.stats.worst_wake_latency_tail_ratio,
    );
}

#[test]
fn merge_scenario_stats_worst_wins_when_other_is_smaller() {
    // Symmetric case: when `other` reports smaller values, `self`
    // retains its larger worst. Covers the "self wins" branch of
    // every scalar worst-comparison in merge (9 fields total:
    // 8 `.max()` calls + the coupled `worst_gap_ms` guard).
    let mut a = AssertResult::pass();
    a.stats.worst_spread = 30.0;
    a.stats.worst_gap_ms = 500;
    a.stats.worst_gap_cpu = 7;
    a.stats.worst_migration_ratio = 0.9;
    a.stats.worst_p99_wake_latency_us = 100.0;
    a.stats.worst_median_wake_latency_us = 60.0;
    a.stats.worst_wake_latency_cv = 0.7;
    a.stats.worst_run_delay_us = 300.0;
    a.stats.worst_mean_run_delay_us = 200.0;
    a.stats.worst_cross_node_migration_ratio = 0.35;
    a.stats.total_iterations = 500;

    let mut b = AssertResult::pass();
    b.stats.worst_spread = 5.0;
    b.stats.worst_gap_ms = 100;
    b.stats.worst_gap_cpu = 3;
    b.stats.worst_migration_ratio = 0.1;
    b.stats.worst_p99_wake_latency_us = 10.0;
    b.stats.worst_median_wake_latency_us = 5.0;
    b.stats.worst_wake_latency_cv = 0.1;
    b.stats.worst_run_delay_us = 40.0;
    b.stats.worst_mean_run_delay_us = 20.0;
    b.stats.worst_cross_node_migration_ratio = 0.05;
    b.stats.total_iterations = 50;

    a.merge(b);

    assert_eq!(a.stats.worst_spread, 30.0);
    assert_eq!(a.stats.worst_gap_ms, 500);
    // `worst_gap_cpu` stays 7: coupling means it retains `self`'s
    // index when `self` wins on `worst_gap_ms`.
    assert_eq!(a.stats.worst_gap_cpu, 7);
    assert_eq!(a.stats.worst_migration_ratio, 0.9);
    assert_eq!(a.stats.worst_p99_wake_latency_us, 100.0);
    assert_eq!(a.stats.worst_median_wake_latency_us, 60.0);
    assert_eq!(a.stats.worst_wake_latency_cv, 0.7);
    assert_eq!(a.stats.worst_run_delay_us, 300.0);
    assert_eq!(a.stats.worst_mean_run_delay_us, 200.0);
    assert_eq!(a.stats.worst_cross_node_migration_ratio, 0.35);
    // Totals always sum, independent of worst-wins direction.
    assert_eq!(a.stats.total_iterations, 550);
}

#[test]
fn merge_worst_page_locality_lowest_non_zero() {
    // `worst_page_locality` can't use plain `.min()` because 0.0
    // is the "unreported" sentinel — a fresh cgroup with no NUMA
    // readings would otherwise clobber a real reading from a
    // reporting cgroup. The merge instead takes the lowest
    // non-zero value.

    // (a) self=0.0 (unreported) + other=0.8 (reported) → 0.8.
    let mut a = AssertResult::pass();
    a.stats.worst_page_locality = 0.0;
    let mut b = AssertResult::pass();
    b.stats.worst_page_locality = 0.8;
    a.merge(b);
    assert_eq!(
        a.stats.worst_page_locality, 0.8,
        "unreported self must adopt other's reading"
    );

    // (b) self=0.6 + other=0.8 → 0.6 (self's lower reading wins).
    let mut a = AssertResult::pass();
    a.stats.worst_page_locality = 0.6;
    let mut b = AssertResult::pass();
    b.stats.worst_page_locality = 0.8;
    a.merge(b);
    assert_eq!(
        a.stats.worst_page_locality, 0.6,
        "lower non-zero reading wins across cgroups"
    );

    // (c) self=0.8 (reported) + other=0.0 (unreported) → 0.8.
    // Plain `.min()` would select 0.0 here — the guard rejects
    // other's sentinel instead of overwriting self.
    let mut a = AssertResult::pass();
    a.stats.worst_page_locality = 0.8;
    let mut b = AssertResult::pass();
    b.stats.worst_page_locality = 0.0;
    a.merge(b);
    assert_eq!(
        a.stats.worst_page_locality, 0.8,
        "unreported other must not clobber self's reading"
    );
}

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
    assert_eq!(
        MetricKind::Rate {
            numerator: "n",
            denominator: "d",
        }
        .merge_kind(),
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
