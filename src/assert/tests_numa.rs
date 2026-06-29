//! NUMA-flavored assertions: `parse_numa_maps`, `page_locality`,
//! `parse_vmstat_numa_pages_migrated`, `assert_page_locality`,
//! `assert_slow_tier_ratio`, `assert_cross_node_migration`, plus
//! the `Assert` builder/merge plumbing for the NUMA-related
//! threshold fields and the `ScenarioStats` cross-node merge.

use super::tests_common::rpt;
use super::*;

// -- numa_maps parsing tests --

#[test]
fn parse_numa_maps_basic() {
    let content = "\
00400000 default file=/bin/cat mapped=10 N0=8 N1=2
00600000 default anon=5 N0=3 N1=2";
    let entries = parse_numa_maps(content);
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].addr, 0x00400000);
    assert_eq!(entries[0].node_pages[&0], 8);
    assert_eq!(entries[0].node_pages[&1], 2);
    assert_eq!(entries[1].addr, 0x00600000);
    assert_eq!(entries[1].node_pages[&0], 3);
    assert_eq!(entries[1].node_pages[&1], 2);
}

#[test]
fn parse_numa_maps_empty() {
    assert!(parse_numa_maps("").is_empty());
}

#[test]
fn parse_numa_maps_no_node_fields() {
    let content = "00400000 default file=/bin/cat mapped=10";
    let entries = parse_numa_maps(content);
    assert!(entries.is_empty());
}

#[test]
fn parse_numa_maps_single_node() {
    let content = "7f000000 default anon=100 N0=100";
    let entries = parse_numa_maps(content);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].node_pages[&0], 100);
    assert_eq!(entries[0].node_pages.len(), 1);
}

#[test]
fn parse_numa_maps_high_node_ids() {
    let content = "7f000000 default N0=10 N3=20 N7=5";
    let entries = parse_numa_maps(content);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].node_pages[&0], 10);
    assert_eq!(entries[0].node_pages[&3], 20);
    assert_eq!(entries[0].node_pages[&7], 5);
}

#[test]
fn parse_numa_maps_malformed_lines() {
    let content = "\
not_hex default N0=10
00400000 default N0=10
 default N0=5";
    let entries = parse_numa_maps(content);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].addr, 0x00400000);
}

// -- page_locality tests --

#[test]
fn page_locality_all_local() {
    let entries = vec![NumaMapsEntry {
        addr: 0x1000,
        node_pages: [(0, 100)].into_iter().collect(),
    }];
    let expected: BTreeSet<usize> = [0].into_iter().collect();
    let loc = page_locality(&entries, &expected);
    assert!((loc - 1.0).abs() < f64::EPSILON);
}

#[test]
fn page_locality_mixed_nodes() {
    let entries = vec![NumaMapsEntry {
        addr: 0x1000,
        node_pages: [(0, 80), (1, 20)].into_iter().collect(),
    }];
    let expected: BTreeSet<usize> = [0].into_iter().collect();
    let loc = page_locality(&entries, &expected);
    assert!((loc - 0.8).abs() < f64::EPSILON);
}

#[test]
fn page_locality_multi_expected_nodes() {
    let entries = vec![NumaMapsEntry {
        addr: 0x1000,
        node_pages: [(0, 40), (1, 40), (2, 20)].into_iter().collect(),
    }];
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    let loc = page_locality(&entries, &expected);
    assert!((loc - 0.8).abs() < f64::EPSILON);
}

#[test]
fn page_locality_empty_entries() {
    // Zero-allocation workloads are NOT vacuously local — a 1.0
    // return would let `min_page_locality` thresholds silently pass
    // on broken runs that produced no NUMA signal. Returning 0.0
    // forces the threshold to surface the missing data as a
    // failure.
    let expected: BTreeSet<usize> = [0].into_iter().collect();
    let loc = page_locality(&[], &expected);
    assert!((loc - 0.0).abs() < f64::EPSILON);
}

#[test]
fn page_locality_no_local_pages() {
    let entries = vec![NumaMapsEntry {
        addr: 0x1000,
        node_pages: [(1, 50)].into_iter().collect(),
    }];
    let expected: BTreeSet<usize> = [0].into_iter().collect();
    let loc = page_locality(&entries, &expected);
    assert!((loc - 0.0).abs() < f64::EPSILON);
}

#[test]
fn page_locality_empty_expected_set() {
    let entries = vec![NumaMapsEntry {
        addr: 0x1000,
        node_pages: [(0, 50)].into_iter().collect(),
    }];
    let loc = page_locality(&entries, &BTreeSet::new());
    assert!((loc - 0.0).abs() < f64::EPSILON);
}

// -- assert_page_locality tests --

#[test]
fn assert_page_locality_pass() {
    let r = assert_page_locality(0.9, Some(0.8), 100, 90);
    assert!(r.is_pass(), "{:?}", r.outcomes);
}

#[test]
fn assert_page_locality_fail() {
    let r = assert_page_locality(0.5, Some(0.8), 100, 50);
    assert!(r.is_fail());
    let detail = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::PageLocality))
        .unwrap();
    // Percentage form must accompany the fraction so an operator
    // reading the diagnostic doesn't mentally translate 0.5000 → 50%.
    assert!(
        detail.message.contains("50.00%"),
        "must include observed %: {detail}"
    );
    assert!(
        detail.message.contains("80.00%"),
        "must include threshold %: {detail}"
    );
}

#[test]
fn assert_page_locality_no_threshold() {
    let r = assert_page_locality(0.1, None, 100, 10);
    assert!(r.is_pass());
}

#[test]
fn assert_page_locality_exact_threshold() {
    let r = assert_page_locality(0.8, Some(0.8), 100, 80);
    assert!(r.is_pass(), "{:?}", r.outcomes);
}

/// Page-locality is the explicit COUNTER-EXAMPLE to the
/// zero-denominator-becomes-Inconclusive convention: a workload
/// that produced no NUMA pages must FAIL the locality gate, not
/// flip to Inconclusive.
///
/// Rationale (`page_locality` doc, mod.rs L170-178): reporting
/// `1.0` for zero observed pages would silently green broken runs
/// that produced no NUMA signal; the function deliberately
/// returns `0.0` instead, and the gate compares it against the
/// threshold like any other observation. This test pins that the
/// stays-Fail policy holds for the zero case so a well-meaning
/// "convert all zero-denominator ratios to Inconclusive" refactor
/// would trip the assertion and force a deliberate review.
#[test]
fn assert_page_locality_zero_observed_stays_fail_not_inconclusive() {
    // page_locality returned 0.0 (zero total pages observed).
    // assert_page_locality must FAIL against any positive threshold,
    // NOT downgrade to Inconclusive.
    let r = assert_page_locality(0.0, Some(0.8), 0, 0);
    assert!(
        r.is_fail(),
        "zero-page workload must Fail, not Pass/Inconclusive: {:?}",
        r.outcomes
    );
    assert!(
        !r.is_inconclusive(),
        "zero-page workload must NOT be Inconclusive (page_locality is the deliberate stays-Fail counter-example): {:?}",
        r.outcomes
    );
    let detail = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::PageLocality))
        .expect("PageLocality detail required");
    assert!(
        detail.message.contains("0.00%"),
        "must surface observed 0% so operator sees the zero-signal cause: {detail}"
    );
}

// -- assert_slow_tier_ratio tests --

#[test]
fn assert_slow_tier_ratio_pass() {
    let mut pages = BTreeMap::new();
    pages.insert(0, 90);
    pages.insert(1, 10);
    let nodes: BTreeSet<usize> = [0, 1].into_iter().collect();
    let r = assert_slow_tier_ratio(&pages, 0.5, 100, Some(&nodes));
    assert!(r.is_pass(), "{:?}", r.outcomes);
}

#[test]
fn assert_slow_tier_ratio_fail() {
    let mut pages = BTreeMap::new();
    pages.insert(0, 40);
    pages.insert(2, 60);
    let nodes: BTreeSet<usize> = [0].into_iter().collect();
    let r = assert_slow_tier_ratio(&pages, 0.5, 100, Some(&nodes));
    assert!(r.is_fail());
    let detail = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::SlowTier))
        .unwrap();
    // 60% slow-tier (node 2 has 60 pages) vs 50% threshold; both
    // surfaces appear so the operator sees raw ratio AND human %.
    assert!(
        detail.message.contains("60.00%"),
        "must include observed %: {detail}"
    );
    assert!(
        detail.message.contains("50.00%"),
        "must include threshold %: {detail}"
    );
}

#[test]
fn assert_slow_tier_ratio_none_numa_nodes() {
    let mut pages = BTreeMap::new();
    pages.insert(0, 100);
    let r = assert_slow_tier_ratio(&pages, 0.1, 100, None);
    assert!(r.is_pass());
}

#[test]
fn assert_slow_tier_ratio_zero_pages() {
    let pages = BTreeMap::new();
    let nodes: BTreeSet<usize> = [0].into_iter().collect();
    let r = assert_slow_tier_ratio(&pages, 0.5, 0, Some(&nodes));
    assert!(r.is_pass());
}

#[test]
fn assert_slow_tier_ratio_all_local() {
    let mut pages = BTreeMap::new();
    pages.insert(0, 100);
    let nodes: BTreeSet<usize> = [0].into_iter().collect();
    let r = assert_slow_tier_ratio(&pages, 0.0, 100, Some(&nodes));
    assert!(r.is_pass(), "{:?}", r.outcomes);
}

/// A cgroup where every worker reported empty (or all-zero) NUMA
/// pages drops out of the per-worker slow-tier loop. Previously
/// the gate silently returned Pass; now it records Inconclusive
/// so a workload that produced no NUMA allocations at all doesn't
/// masquerade as meeting the slow-tier ratio. Pin the
/// `evaluated == 0` carve-out in the AssertPlan dispatch.
#[test]
fn plan_slow_tier_ratio_no_numa_signal_is_inconclusive() {
    use super::AssertPlan;
    use std::collections::BTreeSet;
    let a = rpt(1, 1000, 1_000_000_000, 0, &[0], 0);
    let b = rpt(2, 1000, 1_000_000_000, 0, &[0], 0);
    // Both workers have empty numa_pages — no NUMA signal at all.
    let plan = AssertPlan {
        not_starved: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: None,
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: Some(0.1),
    };
    let nodes: BTreeSet<usize> = [0].into_iter().collect();
    let r = plan.assert_cgroup(&[a, b], None, Some(&nodes));
    assert!(
        r.is_inconclusive(),
        "no-NUMA-signal cgroup must be Inconclusive, not Pass: {:?}",
        r.outcomes,
    );
    assert!(!r.is_pass(), "must not silently pass on zero denominator");
    assert!(!r.is_fail(), "no actual ratio violation to report");
    let reason = r
        .inconclusive_details()
        .find(|d| d.kind == DetailKind::SlowTier)
        .unwrap_or_else(|| panic!("expected SlowTier Inconclusive, got {:?}", r.outcomes));
    assert!(
        reason.message.contains("no worker reported any NUMA pages"),
        "diagnostic must name the root cause: {reason}"
    );
    assert!(
        reason.message.contains("denominator is zero"),
        "diagnostic must surface the operator-actionable hint: {reason}"
    );
}

// -- Assert NUMA builder and merge tests --

#[test]
fn assert_min_page_locality_setter() {
    let v = Assert::NO_OVERRIDES.min_page_locality(0.9);
    assert_eq!(v.min_page_locality, Some(0.9));
}

#[test]
fn assert_merge_numa_fields() {
    let base = Assert::NO_OVERRIDES.min_page_locality(0.9);
    let merged = base.merge(&Assert::NO_OVERRIDES);
    assert_eq!(merged.min_page_locality, Some(0.9));
}

#[test]
fn assert_merge_numa_override() {
    let base = Assert::NO_OVERRIDES.min_page_locality(0.9);
    let other = Assert::NO_OVERRIDES.min_page_locality(0.5);
    assert_eq!(base.merge(&other).min_page_locality, Some(0.5));
}

#[test]
fn assert_numa_has_worker_checks() {
    assert!(
        Assert::NO_OVERRIDES
            .min_page_locality(0.8)
            .has_worker_checks()
    );
}

#[test]
fn assert_page_locality_method_pass() {
    let a = Assert::NO_OVERRIDES.min_page_locality(0.8);
    let r = a.assert_page_locality(0.9, 100, 90);
    assert!(r.is_pass(), "{:?}", r.outcomes);
}

#[test]
fn assert_page_locality_method_fail() {
    let a = Assert::NO_OVERRIDES.min_page_locality(0.95);
    let r = a.assert_page_locality(0.8, 100, 80);
    assert!(r.is_fail());
}

// -- ScenarioStats NUMA merge tests --

/// `worst_page_locality` is no longer a typed field folded in `merge`; it
/// re-pools None-aware over the per-phase NUMA carriers POST-merge. This pins
/// the cross-cgroup-merge property: two handles (separate `AssertResult`s)
/// each carrying one cgroup's per-phase NUMA in the same step merge their
/// `per_cgroup` carriers into one phase, and the run-level worst re-pools over
/// the COMBINED cohort — with a MEASURED 0.0 (all pages off-node, the worst
/// cell) winning the lowest rather than being skipped as an unreported
/// sentinel (the `fold_lowest_nonzero` read-side bug this migration removes).
#[test]
fn merge_repools_worst_page_locality_across_cgroups_measured_zero_wins() {
    use crate::stats::BuiltinMetric as B;
    let pc = |local: u64, total: u64| PhaseCgroupStats {
        numa_pages_local: local,
        numa_pages_total: total,
        ..Default::default()
    };
    let with_cgroup = |name: &str, pcs: PhaseCgroupStats| {
        let mut r = AssertResult::pass();
        let mut bucket = PhaseBucket {
            step_index: 1,
            ..Default::default()
        };
        bucket.per_cgroup.insert(name.to_string(), pcs);
        r.stats.phases = vec![bucket];
        r
    };
    // Handle A: cgroup "A" healthy 0.8 (800/1000). Handle B: cgroup "B" all
    // off-node — a MEASURED 0.0 (0/1000). The merge unions {A} and {B} into one
    // step's `per_cgroup`, so the post-merge re-pool sees both and B's 0.0 wins.
    let mut a = with_cgroup("A", pc(800, 1000));
    a.merge(with_cgroup("B", pc(0, 1000)));
    assert_eq!(
        a.stats.run_metric(B::WorstPageLocality),
        Some(0.0),
        "post-merge re-pool sees BOTH cgroups; B's measured 0.0 wins the lowest",
    );

    // None-aware across the merge: a merged cgroup that never measured NUMA
    // (numa_pages_total == 0) is skipped, so the other handle's measured value
    // is the run-level worst — not a 0.0 sentinel clobbering it.
    let mut a = with_cgroup("A", pc(700, 1000));
    a.merge(with_cgroup("B", pc(0, 0)));
    assert_eq!(
        a.stats.run_metric(B::WorstPageLocality),
        Some(0.7),
        "an unmeasured (total == 0) merged cgroup is skipped, not a 0.0 sentinel",
    );
}

#[test]
fn cgroup_stats_numa_defaults() {
    let c = CgroupStats::default();
    assert_eq!(c.page_locality, 0.0);
    assert_eq!(c.cross_node_migration_ratio, 0.0);
}

#[test]
fn scenario_stats_numa_defaults() {
    let s = ScenarioStats::default();
    // Neither NUMA roll-up is a typed field — a default (phase-less)
    // ScenarioStats has no NUMA carriers, so both re-pool to None.
    assert_eq!(
        s.run_metric(crate::stats::BuiltinMetric::WorstPageLocality),
        None
    );
    assert_eq!(
        s.run_metric(crate::stats::BuiltinMetric::WorstCrossNodeMigrationRatio),
        None
    );
}

// -- parse_vmstat_numa_pages_migrated tests --

#[test]
fn parse_vmstat_present() {
    let content = "\
nr_free_pages 12345
numa_hit 100
numa_pages_migrated 42
numa_miss 5";
    assert_eq!(parse_vmstat_numa_pages_migrated(content), Some(42));
}

#[test]
fn parse_vmstat_absent() {
    let content = "nr_free_pages 12345\nnuma_hit 100";
    assert_eq!(parse_vmstat_numa_pages_migrated(content), None);
}

#[test]
fn parse_vmstat_zero() {
    let content = "numa_pages_migrated 0";
    assert_eq!(parse_vmstat_numa_pages_migrated(content), Some(0));
}

#[test]
fn parse_vmstat_large_value() {
    let content = "numa_pages_migrated 9999999999";
    assert_eq!(parse_vmstat_numa_pages_migrated(content), Some(9999999999));
}

#[test]
fn parse_vmstat_empty() {
    assert_eq!(parse_vmstat_numa_pages_migrated(""), None);
}

#[test]
fn parse_vmstat_malformed_value() {
    let content = "numa_pages_migrated abc";
    assert_eq!(parse_vmstat_numa_pages_migrated(content), None);
}

// -- assert_cross_node_migration tests --

#[test]
fn assert_cross_node_migration_pass() {
    let r = assert_cross_node_migration(5, 100, Some(0.1));
    assert!(r.is_pass(), "{:?}", r.outcomes);
}

#[test]
fn assert_cross_node_migration_fail() {
    let r = assert_cross_node_migration(20, 100, Some(0.1));
    assert!(r.is_fail());
    let detail = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::CrossNodeMigration))
        .unwrap();
    // 20% migrated vs 10% threshold; pin both percentage tokens so
    // dropping either form regresses here.
    assert!(
        detail.message.contains("20.00%"),
        "must include observed %: {detail}"
    );
    assert!(
        detail.message.contains("10.00%"),
        "must include threshold %: {detail}"
    );
}

#[test]
fn assert_cross_node_migration_no_threshold() {
    let r = assert_cross_node_migration(50, 100, None);
    assert!(r.is_pass());
}

#[test]
fn assert_cross_node_migration_exact_threshold() {
    let r = assert_cross_node_migration(10, 100, Some(0.1));
    assert!(r.is_pass(), "{:?}", r.outcomes);
}

#[test]
fn assert_cross_node_migration_zero_pages_is_inconclusive() {
    // Both inputs zero = zero-denominator with no signal — neither
    // Pass (would silently green a workload that allocated nothing)
    // nor Fail (no actual ratio violation observed) is truthful.
    // Pin the Inconclusive outcome and the diagnostic that surfaces
    // the operator-actionable hint ("did the workload allocate any
    // memory?").
    let r = assert_cross_node_migration(0, 0, Some(0.1));
    assert!(
        r.is_inconclusive(),
        "zero total + zero migrated must be Inconclusive, not Pass or Fail: {:?}",
        r.outcomes,
    );
    assert!(!r.is_pass(), "must not silently pass on zero denominator");
    assert!(!r.is_fail(), "no ratio violation to report");
    let reason = r
        .inconclusive_details()
        .find(|d| d.kind == DetailKind::CrossNodeMigration)
        .unwrap_or_else(|| panic!("expected Inconclusive reason, got {:?}", r.outcomes));
    assert!(
        reason.message.contains("denominator is zero"),
        "diagnostic must name the root cause: {reason}"
    );
    assert!(
        reason.message.contains("allocate any memory"),
        "diagnostic must surface the operator-actionable hint: {reason}"
    );
}

#[test]
fn assert_cross_node_migration_inconsistent_zero_total_nonzero_migrated() {
    // vmstat reported migrations but numa_maps shows zero pages —
    // inconsistent measurement that must surface as a failure
    // rather than silently coercing to ratio=0.0.
    let r = assert_cross_node_migration(5, 0, Some(0.1));
    assert!(!r.is_pass(), "inconsistent input must fail loudly");
    let detail = r
        .failure_details()
        .find(|d| d.message.contains("inconsistent"))
        .unwrap_or_else(|| panic!("expected inconsistent diagnostic, got {:?}", r.outcomes));
    assert!(
        detail.message.contains("5 pages migrated"),
        "must surface migrated count: {detail}"
    );
    assert!(
        detail.message.contains("0 pages observed"),
        "must surface total=0: {detail}"
    );
}

// -- Assert cross-node migration builder/merge --

#[test]
fn assert_max_cross_node_migration_ratio_setter() {
    let v = Assert::NO_OVERRIDES.max_cross_node_migration_ratio(0.05);
    assert_eq!(v.max_cross_node_migration_ratio, Some(0.05));
}

#[test]
fn assert_merge_cross_node_migration() {
    let base = Assert::NO_OVERRIDES.max_cross_node_migration_ratio(0.1);
    let other = Assert::NO_OVERRIDES.max_cross_node_migration_ratio(0.05);
    assert_eq!(
        base.merge(&other).max_cross_node_migration_ratio,
        Some(0.05)
    );
}

#[test]
fn assert_merge_cross_node_migration_preserves() {
    let base = Assert::NO_OVERRIDES.max_cross_node_migration_ratio(0.1);
    assert_eq!(
        base.merge(&Assert::NO_OVERRIDES)
            .max_cross_node_migration_ratio,
        Some(0.1)
    );
}

#[test]
fn assert_cross_node_migration_has_worker_checks() {
    assert!(
        Assert::NO_OVERRIDES
            .max_cross_node_migration_ratio(0.1)
            .has_worker_checks()
    );
}

#[test]
fn assert_cross_node_migration_method_pass() {
    let a = Assert::NO_OVERRIDES.max_cross_node_migration_ratio(0.1);
    let r = a.assert_cross_node_migration(5, 100);
    assert!(r.is_pass(), "{:?}", r.outcomes);
}

#[test]
fn assert_cross_node_migration_method_fail() {
    let a = Assert::NO_OVERRIDES.max_cross_node_migration_ratio(0.05);
    let r = a.assert_cross_node_migration(20, 100);
    assert!(r.is_fail());
}

// -- ScenarioStats cross-node migration merge re-pool --

/// worst_cross_node_migration_ratio is no longer a typed field merge-folded in
/// `merge`; it re-pools MAX-wins over the per-phase NUMA carriers POST-merge (the
/// polarity twin of worst_page_locality's lowest-wins). Two handles, each carrying
/// one cgroup's per-phase NUMA in the same step, merge their `per_cgroup`
/// carriers; the run-level worst re-pools the highest per-cgroup churn ratio over
/// the COMBINED cohort. None-aware: a cgroup that measured no NUMA pages
/// (`numa_pages_total == 0`) is skipped.
#[test]
fn merge_repools_worst_cross_node_migration_ratio_across_cgroups_max_wins() {
    use crate::stats::BuiltinMetric as B;
    let pc = |total: u64, migr: u64| PhaseCgroupStats {
        numa_pages_total: total,
        cross_node_migrated: migr,
        ..Default::default()
    };
    let with_cgroup = |name: &str, pcs: PhaseCgroupStats| {
        let mut r = AssertResult::pass();
        let mut bucket = PhaseBucket {
            step_index: 1,
            ..Default::default()
        };
        bucket.per_cgroup.insert(name.to_string(), pcs);
        r.stats.phases = vec![bucket];
        r
    };
    // Handle A: cgroup "A" 50/1000 = 0.05. Handle B: cgroup "B" 150/1000 = 0.15 —
    // the worst (highest). The merge unions {A} and {B} into one step's per_cgroup,
    // so the post-merge re-pool sees both and B's 0.15 wins (max).
    let mut a = with_cgroup("A", pc(1000, 50));
    a.merge(with_cgroup("B", pc(1000, 150)));
    let r = a
        .stats
        .run_metric(B::WorstCrossNodeMigrationRatio)
        .expect("measured cohort yields a value");
    assert!(
        (r - 0.15).abs() < 1e-9,
        "post-merge re-pool sees BOTH cgroups; B's 0.15 wins the max; got {r}",
    );

    // None-aware: a merged cgroup that never measured NUMA (numa_pages_total == 0)
    // is skipped, so the other handle's measured value is the run-level worst.
    let mut a = with_cgroup("A", pc(1000, 70));
    a.merge(with_cgroup("B", pc(0, 0)));
    let r = a
        .stats
        .run_metric(B::WorstCrossNodeMigrationRatio)
        .expect("one measured cgroup yields a value");
    assert!(
        (r - 0.07).abs() < 1e-9,
        "an unmeasured (total == 0) merged cgroup is skipped; A's 0.07 is the worst; got {r}",
    );
}

// -- AssertPlan: cross-node migration aggregation --

#[test]
fn plan_cross_node_migration_aggregates_cgroup_total() {
    // `vmstat_numa_pages_migrated` is system-wide and per-worker
    // observations of the same time window overlap heavily. The
    // per-worker loop previously divided each worker's vmstat
    // delta by that worker's own page total, producing N inflated
    // ratios. Verify the new behavior: max of per-worker vmstat
    // deltas (closest to total system migrations during the run)
    // divided by the SUM of per-worker numa_pages totals.
    let mut a = rpt(1, 1000, 1_000_000_000, 0, &[0], 0);
    let mut b = rpt(2, 1000, 1_000_000_000, 0, &[1], 0);
    a.numa_pages = [(0, 50), (1, 50)].into_iter().collect();
    b.numa_pages = [(0, 50), (1, 50)].into_iter().collect();
    // Each worker observed the same system-wide delta of 5
    // (overlapping windows). Per-worker calc would compute
    // 5/100 = 0.05 twice. Aggregated calc: max(5, 5) / (100+100)
    // = 5/200 = 0.025.
    a.vmstat_numa_pages_migrated = 5;
    b.vmstat_numa_pages_migrated = 5;
    let plan = AssertPlan {
        not_starved: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: None,
        min_page_locality: None,
        max_cross_node_migration_ratio: Some(0.03),
        max_slow_tier_ratio: None,
    };
    let r = plan.assert_cgroup(&[a, b], None, None);
    assert!(
        r.is_pass(),
        "0.025 < 0.03 must pass under aggregated calc; per-worker would have failed at 0.05: {:?}",
        r.outcomes
    );
}

#[test]
fn plan_cross_node_migration_emits_one_failure_not_per_worker() {
    // Failure surfaces once for the cgroup, not N times per
    // worker. The per-worker loop emitted one failure per worker;
    // the aggregated calc emits at most one.
    let mut a = rpt(1, 1000, 1_000_000_000, 0, &[0], 0);
    let mut b = rpt(2, 1000, 1_000_000_000, 0, &[1], 0);
    a.numa_pages = [(0, 50)].into_iter().collect();
    b.numa_pages = [(0, 50)].into_iter().collect();
    a.vmstat_numa_pages_migrated = 50;
    b.vmstat_numa_pages_migrated = 50;
    let plan = AssertPlan {
        not_starved: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: None,
        min_page_locality: None,
        max_cross_node_migration_ratio: Some(0.1),
        max_slow_tier_ratio: None,
    };
    let r = plan.assert_cgroup(&[a, b], None, None);
    assert!(r.is_fail());
    let cross_node_failures = r
        .failure_details()
        .filter(|d| matches!(d.kind, DetailKind::CrossNodeMigration))
        .count();
    assert_eq!(
        cross_node_failures, 1,
        "exactly one cross-node migration failure for the cgroup (not per-worker): {:?}",
        r.outcomes
    );
}

// -- AssertPlan: min_page_locality on zero-allocation workloads --

#[test]
fn plan_min_page_locality_fails_on_zero_allocation_cgroup() {
    // Workers that produced no NUMA signal (empty numa_pages)
    // previously got skipped, letting `min_page_locality` look
    // green on broken runs. The aggregated calc treats zero
    // observed pages as zero locality, surfacing the missing
    // signal as a failure.
    let a = rpt(1, 1000, 1_000_000_000, 0, &[0], 0);
    let b = rpt(2, 1000, 1_000_000_000, 0, &[0], 0);
    let plan = AssertPlan {
        not_starved: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: None,
        min_page_locality: Some(0.8),
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };
    let nodes: BTreeSet<usize> = [0].into_iter().collect();
    let r = plan.assert_cgroup(&[a, b], None, Some(&nodes));
    assert!(
        !r.is_pass(),
        "zero-allocation cgroup must fail min_page_locality, not silently pass: {:?}",
        r.outcomes
    );
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::PageLocality)),
        "must surface a PageLocality detail: {:?}",
        r.outcomes
    );
}

#[test]
fn plan_min_page_locality_aggregates_across_cgroup() {
    // Aggregated calc: (sum of local pages) / (sum of total
    // pages) across the cgroup. Two workers each with 100 pages
    // (one all-local, one all-remote) yield 100/200 = 0.5
    // locality, below a 0.8 floor.
    let mut a = rpt(1, 1000, 1_000_000_000, 0, &[0], 0);
    let mut b = rpt(2, 1000, 1_000_000_000, 0, &[1], 0);
    a.numa_pages = [(0, 100)].into_iter().collect();
    b.numa_pages = [(1, 100)].into_iter().collect();
    let plan = AssertPlan {
        not_starved: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: None,
        min_page_locality: Some(0.8),
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };
    let nodes: BTreeSet<usize> = [0].into_iter().collect();
    let r = plan.assert_cgroup(&[a, b], None, Some(&nodes));
    assert!(
        !r.is_pass(),
        "cgroup-aggregate locality 0.5 < 0.8 must fail"
    );
}

#[test]
fn cgroup_numa_telemetry_populates_without_a_check() {
    // page_locality + cross_node_migration_ratio are pure measurement and
    // must populate in stats.cgroups even when NO NUMA check is configured
    // (pre-fix they were computed only inside the check arms and otherwise
    // hardcoded 0.0, so a NUMA test reading the telemetry got nothing).
    let mut w = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50);
    w.numa_pages = [(0usize, 80u64), (1usize, 20u64)].into_iter().collect();
    w.vmstat_numa_pages_migrated = 10;
    let nodes: BTreeSet<usize> = [0].into_iter().collect();
    // NO_OVERRIDES: no worker check at all -> telemetry-only path.
    let r = Assert::NO_OVERRIDES.assert_cgroup_with_numa(&[w], None, Some(&nodes));
    let cg = &r.stats.cgroups[0];
    // locality = pages on node 0 (80) / total (100) = 0.8.
    assert!(
        (cg.page_locality - 0.8).abs() < 1e-9,
        "page_locality must populate as telemetry; got {}",
        cg.page_locality,
    );
    // cross_node = max migrated (10) / total pages (100) = 0.1.
    assert!(
        (cg.cross_node_migration_ratio - 0.1).abs() < 1e-9,
        "cross_node_migration_ratio must populate; got {}",
        cg.cross_node_migration_ratio,
    );
    // Neither NUMA roll-up is a typed field: both re-pool from the per-phase NUMA
    // carriers, which the direct-assert (`assert_cgroup_with_numa`) path does not
    // populate (it builds stats.cgroups, not stats.phases), so the run-level worst
    // is None here — the per-cgroup `cg.page_locality` / `cg.cross_node_migration_ratio`
    // above are the telemetry this path surfaces.
    assert_eq!(
        r.stats
            .run_metric(crate::stats::BuiltinMetric::WorstPageLocality),
        None,
    );
    assert_eq!(
        r.stats
            .run_metric(crate::stats::BuiltinMetric::WorstCrossNodeMigrationRatio),
        None,
    );
}
