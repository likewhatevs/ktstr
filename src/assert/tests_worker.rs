//! Worker-level checks: `assert_not_stuck`, `assert_isolation`,
//! gap / spread / stuck classification, single-worker passthroughs,
//! and negative-diagnostic-message tests that pin the
//! human-readable strings every consumer greps for.

use super::tests_common::rpt;
use super::*;

#[test]
fn healthy_pass() {
    let r = assert_not_stuck(&[
        rpt(1, 1000, 5_000_000_000, 500_000_000, &[0, 1], 50),
        rpt(2, 1000, 5_000_000_000, 600_000_000, &[0, 1], 60),
        rpt(3, 1000, 5_000_000_000, 550_000_000, &[0, 1], 45),
    ]);
    assert!(r.is_pass(), "{:?}", r.outcomes);
}

#[test]
fn no_progress_fails() {
    let r = assert_not_stuck(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 0, 5e9 as u64, 5e9 as u64, &[0], 50),
    ]);
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::NoProgress))
    );
}

/// work_units-vs-iterations split: a report with `work_units > 0` but
/// `iterations == 0` (the naive-Custom-author footgun; `rpt` always sets
/// `iterations: 0`) reports ZERO throughput — `CgroupStats::total_iterations`
/// reads `iterations`, not `work_units` — yet PASSES the zero-work-units gate,
/// which reads `work_units`. Pins both halves of the split and would catch a
/// forbidden silent `iterations == 0 ? work_units` fallback in the reducer.
#[test]
fn work_units_without_iterations_is_zero_throughput_but_not_stuck() {
    let reports = [
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
        rpt(2, 1000, 5e9 as u64, 6e8 as u64, &[0, 1], 50),
    ];
    assert_eq!(
        cgroup_stats(&reports).total_iterations,
        0,
        "total_iterations reads WorkerReport::iterations (0), NOT work_units (1000)"
    );
    assert!(
        assert_not_stuck(&reports).is_pass(),
        "the zero-work-units gate reads work_units (1000 > 0), so it must pass"
    );
}

/// Inverse: `work_units == 0` but `iterations > 0` reports throughput
/// (`total_iterations` sums `iterations`) yet trips the zero-work-units gate
/// (which reads `work_units == 0`). Pins that the two counters are not
/// interchangeable in either direction.
#[test]
fn iterations_without_work_units_reports_throughput_but_is_not_stuck_failure() {
    let reports = [
        WorkerReport {
            iterations: 1000,
            ..rpt(1, 0, 5e9 as u64, 5e8 as u64, &[0, 1], 50)
        },
        WorkerReport {
            iterations: 1000,
            ..rpt(2, 0, 5e9 as u64, 6e8 as u64, &[0, 1], 50)
        },
    ];
    assert_eq!(
        cgroup_stats(&reports).total_iterations,
        2000,
        "total_iterations sums WorkerReport::iterations (1000+1000), independent of work_units (0)"
    );
    let r = assert_not_stuck(&reports);
    assert!(
        r.is_fail(),
        "work_units == 0 must trip the zero-work-units gate"
    );
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::NoProgress)),
        "the zero-work-units gate reads work_units == 0 → NoProgress"
    );
}

#[test]
fn unfair_spread_fail() {
    let r = assert_not_stuck(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50), // 10%
        rpt(2, 500, 5e9 as u64, 4e9 as u64, &[0, 1], 50),  // 80%
        rpt(3, 800, 5e9 as u64, 2e9 as u64, &[0, 1], 50),  // 40%
    ]);
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::Unfair))
    );
}

#[test]
fn fair_oversubscribed_pass() {
    let r = assert_not_stuck(&[
        rpt(1, 100, 5e9 as u64, (3.75e9) as u64, &[0], 50),
        rpt(2, 100, 5e9 as u64, (3.70e9) as u64, &[0], 50),
        rpt(3, 100, 5e9 as u64, (3.80e9) as u64, &[0], 50),
        rpt(4, 100, 5e9 as u64, (3.75e9) as u64, &[0], 50),
    ]);
    assert!(r.is_pass(), "{:?}", r.outcomes);
}

#[test]
fn stuck_fail() {
    let threshold = gap_threshold_ms();
    let r = assert_not_stuck(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold + 500),
    ]);
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::Stuck))
    );
}

#[test]
fn isolation_pass() {
    let expected: BTreeSet<usize> = [0, 1, 2, 3].into_iter().collect();
    let r = assert_isolation(
        &[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[2, 3], 50),
        ],
        &expected,
    );
    assert!(r.is_pass());
}

#[test]
fn isolation_fail() {
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    let r = assert_isolation(
        &[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 4], 50)],
        &expected,
    );
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::Isolation))
    );
}

#[test]
fn spread_boundary() {
    let threshold = spread_threshold_pct();
    // At threshold exactly - pass
    // Worker 1: 10% off-CPU, Worker 2: 10%+threshold off-CPU
    let at_threshold_ns = ((10.0 + threshold) / 100.0 * 5e9) as u64;
    let r = assert_not_stuck(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50), // 10%
        rpt(2, 1000, 5e9 as u64, at_threshold_ns, &[0], 50), // 10% + threshold
    ]);
    assert!(
        r.is_pass(),
        "{threshold}% spread at threshold: {:?}",
        r.outcomes
    );
    // Above threshold - fail
    let above_ns = ((15.0 + threshold) / 100.0 * 5e9) as u64;
    let r = assert_not_stuck(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50), // 10%
        rpt(2, 1000, 5e9 as u64, above_ns, &[0], 50),   // 10% + threshold + 5%
    ]);
    assert!(!r.is_pass(), "spread above {threshold}% should fail");
}

#[test]
fn empty_pass() {
    assert!(assert_not_stuck(&[]).is_pass());
}

#[test]
fn zero_wall_time() {
    let r = assert_not_stuck(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 0, 0, 0, &[], 0),
    ]);
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::NoProgress))
    );
}

#[test]
fn single_worker_always_pass() {
    let r = assert_not_stuck(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50)]);
    assert!(r.is_pass());
    assert_eq!(r.stats.total_workers, 1);
    assert_eq!(r.stats.cgroups.len(), 1);
}

#[test]
fn stats_accuracy() {
    let r = assert_not_stuck(&[
        rpt(1, 1000, 5e9 as u64, 1e9 as u64, &[0], 50),  // 20%
        rpt(2, 1000, 5e9 as u64, 15e8 as u64, &[1], 60), // 30%
    ]);
    assert!(r.is_pass()); // spread = 10% < 15%
    let c = &r.stats.cgroups[0];
    assert_eq!(c.num_workers, 2);
    assert_eq!(c.num_cpus, 2);
    assert!((c.min_off_cpu_pct.unwrap() - 20.0).abs() < 0.1);
    assert!((c.max_off_cpu_pct.unwrap() - 30.0).abs() < 0.1);
    assert!((c.spread.unwrap() - 10.0).abs() < 0.1);
    assert!((c.avg_off_cpu_pct.unwrap() - 25.0).abs() < 0.1);
}

#[test]
fn isolation_empty_reports() {
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    assert!(assert_isolation(&[], &expected).is_pass());
}

#[test]
fn gap_boundary_at_threshold_pass() {
    let threshold = gap_threshold_ms();
    let r = assert_not_stuck(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold)]);
    assert!(
        r.is_pass(),
        "gap at threshold should pass: {:?}",
        r.outcomes
    );
}

#[test]
fn gap_boundary_above_threshold_fail() {
    let threshold = gap_threshold_ms();
    let r = assert_not_stuck(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold + 1)]);
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::Stuck))
    );
}

#[test]
fn multiple_stuck_workers() {
    let threshold = gap_threshold_ms();
    let r = assert_not_stuck(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold + 500),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], threshold + 1500),
    ]);
    assert!(r.is_fail());
    let stuck_count = r
        .failure_details()
        .filter(|d| matches!(d.kind, DetailKind::Stuck))
        .count();
    assert_eq!(stuck_count, 2, "both workers should be flagged stuck");
}

#[test]
fn migration_tracking() {
    let mut report = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 2], 50);
    report.migration_count = 5;
    let r = assert_not_stuck(&[report]);
    assert_eq!(r.stats.total_migrations, 5);
}

#[test]
fn single_worker_spread_zero() {
    let r = assert_not_stuck(&[rpt(1, 500, 5e9 as u64, 25e8 as u64, &[0, 1], 50)]);
    assert!(r.is_pass());
    let c = &r.stats.cgroups[0];
    // Single worker with measurable wall time: a real measured zero
    // spread (Some(0.0)), not None.
    assert_eq!(c.spread, Some(0.0));
}

#[test]
fn zero_wall_time_nonzero_work() {
    // wall_time=0 but work_units>0: the worker did work but the timer
    // didn't advance. Should not produce a no-progress failure since work was done.
    // The off_cpu_pct computation skips this worker (no pcts entry).
    let r = assert_not_stuck(&[rpt(1, 100, 0, 0, &[0], 0)]);
    assert!(
        r.is_pass(),
        "nonzero work with zero wall_time: {:?}",
        r.outcomes
    );
    // No worker had measurable wall time, so off-CPU% is not defined.
    // The fields must be None (not measured), NOT 0.0 — a not-measured
    // cgroup must not read as a perfectly-on-CPU / perfectly-fair one.
    let c = &r.stats.cgroups[0];
    assert_eq!(c.avg_off_cpu_pct, None, "off-cpu% must be None, not 0.0");
    assert_eq!(c.min_off_cpu_pct, None);
    assert_eq!(c.max_off_cpu_pct, None);
    assert_eq!(
        c.spread, None,
        "spread must be None (inconclusive), not 0.0"
    );
}

#[test]
fn isolation_empty_expected_set() {
    // Empty expected set means no CPUs are "expected", so any CPU
    // used by the worker is unexpected. difference(empty) == worker's set.
    let expected: BTreeSet<usize> = BTreeSet::new();
    let r = assert_isolation(
        &[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50)],
        &expected,
    );
    // Worker used CPUs {0,1}, expected is empty, so all are unexpected.
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::Isolation))
    );
}

#[test]
fn isolation_worker_used_no_cpus() {
    // Worker used no CPUs -- difference with expected is empty, so passes.
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    let r = assert_isolation(&[rpt(1, 0, 0, 0, &[], 0)], &expected);
    assert!(r.is_pass());
}

#[test]
fn isolation_all_unexpected_cpus() {
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    let r = assert_isolation(
        &[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[4, 5, 6], 50)],
        &expected,
    );
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::Isolation))
    );
}

// ---------------------------------------------------------------
// Negative tests: check that diagnostics catch controlled failures
// ---------------------------------------------------------------

#[test]
fn neg_no_progress_zero_work_detected() {
    let r = assert_not_stuck(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
        rpt(2, 0, 5e9 as u64, 0, &[0], 0), // no progress
        rpt(3, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
    ]);
    assert!(!r.is_pass(), "zero-work worker must be caught");
    let no_progress = r
        .failure_details()
        .filter(|d| matches!(d.kind, DetailKind::NoProgress))
        .count();
    assert_eq!(no_progress, 1, "exactly one no-progress worker expected");
    // Format: "tid 2 made no progress (0 work units)"
    let detail = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::NoProgress))
        .unwrap();
    assert!(
        detail.message.contains("tid 2"),
        "must name the no-progress tid: {detail}"
    );
    assert!(
        detail.message.contains("0 work units"),
        "must state zero work: {detail}"
    );
}

#[test]
fn neg_isolation_violation_outside_cpuset() {
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    let reports = [
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 2, 3], 50),
    ];
    let r = assert_isolation(&reports, &expected);
    assert!(!r.is_pass(), "isolation violation must be caught");
    // Format: "tid 2 ran on unexpected CPUs {2, 3}"
    let detail = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::Isolation))
        .unwrap();
    assert!(
        detail.message.contains("tid 2"),
        "must name violating tid: {detail}"
    );
    assert!(
        detail.message.contains("2"),
        "must list out-of-set CPU 2: {detail}"
    );
    assert!(
        detail.message.contains("3"),
        "must list out-of-set CPU 3: {detail}"
    );
    // Worker 1 ran only on {0,1} which is within expected — no violation.
    assert_eq!(r.outcomes.len(), 1, "only tid 2 should violate");
}

#[test]
fn neg_unfairness_extreme_spread_detected() {
    let r = assert_not_stuck(&[
        rpt(1, 100, 5e9 as u64, 25e7 as u64, &[0, 1], 50), // 5%
        rpt(2, 5000, 5e9 as u64, 475e7 as u64, &[0, 1], 50), // 95%
    ]);
    assert!(!r.is_pass(), "extreme unfairness must be caught");
    // Format: "unfair cgroup: spread=90% (5-95%) 2 workers on 2 cpus (threshold 15%)"
    let detail = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::Unfair))
        .unwrap();
    assert!(
        detail.message.contains("spread="),
        "must include spread value: {detail}"
    );
    assert!(
        detail.message.contains("workers"),
        "must include worker count: {detail}"
    );
    assert!(
        detail.message.contains("cpus"),
        "must include cpu count: {detail}"
    );
    // Threshold must appear so a regression dropping the bound surfaces here.
    // The literal value comes from `spread_threshold_pct()` which differs
    // between debug and release builds; pin only the textual prefix.
    assert!(
        detail.message.contains("threshold "),
        "must include threshold bound: {detail}"
    );
    let c = &r.stats.cgroups[0];
    let spread = c.spread.expect("measured workers => Some");
    assert!(spread > 80.0, "spread should be >80%, got {spread:.1}");
    assert_eq!(c.num_workers, 2);
    assert_eq!(c.num_cpus, 2);
    let min_pct = c.min_off_cpu_pct.expect("measured workers => Some");
    assert!(min_pct < 10.0, "min pct should be ~5%: {min_pct:.1}");
    let max_pct = c.max_off_cpu_pct.expect("measured workers => Some");
    assert!(max_pct > 90.0, "max pct should be ~95%: {max_pct:.1}");
}

#[test]
fn neg_scheduling_gap_exceeds_threshold() {
    let threshold = gap_threshold_ms();
    let gap = threshold + 2000;
    let r = assert_not_stuck(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], gap),
    ]);
    assert!(!r.is_pass(), "scheduling gap must be caught");
    // Format: "tid 2 stuck {gap}ms on cpu1 at +1000ms (threshold 2000ms)"
    let detail = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::Stuck))
        .unwrap();
    assert!(
        detail.message.contains(&format!("{}ms", gap)),
        "must include gap duration: {detail}"
    );
    assert!(
        detail.message.contains("on cpu"),
        "must include CPU number: {detail}"
    );
    assert!(
        detail.message.contains("at +"),
        "must include timing offset: {detail}"
    );
    assert!(detail.message.contains("cpu1"), "gap is on cpu1: {detail}");
    // tid must be named so an operator triaging a multi-worker cgroup can
    // identify the offender without reverse-mapping CPU placement.
    assert!(
        detail.message.contains("tid 2"),
        "must name violating tid (2): {detail}"
    );
    // Threshold appears for parity with the AssertPlan custom-threshold path
    // and so a regression dropping the bound from the default-path message
    // surfaces here.
    assert!(
        detail
            .message
            .contains(&format!("threshold {}ms", threshold)),
        "must include default-path threshold: {detail}"
    );
    // Stats must reflect the gap.
    assert_eq!(r.stats.worst_gap_ms, gap);
    assert_eq!(r.stats.worst_gap_cpu, 1);
}

#[test]
fn neg_plan_custom_gap_catches_lower_threshold() {
    let plan = AssertPlan::new().check_not_stuck().max_gap_ms(500);
    let reports = [
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 1000),
    ];
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(!r.is_pass(), "custom 500ms threshold must catch 1000ms gap");
    // Format: "tid 2 stuck 1000ms on cpu1 at +1000ms (threshold 500ms)"
    let detail = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::Stuck))
        .unwrap();
    assert!(
        detail.message.contains("1000ms"),
        "must include gap duration: {detail}"
    );
    assert!(
        detail.message.contains("cpu1"),
        "must include CPU: {detail}"
    );
    assert!(
        detail.message.contains("threshold 500ms"),
        "must include custom threshold: {detail}"
    );
    // tid must be named; pins parity with the bare-path message.
    assert!(
        detail.message.contains("tid 2"),
        "must name violating tid (2): {detail}"
    );
}

#[test]
fn neg_isolation_plus_no_progress_both_reported() {
    let plan = AssertPlan::new().check_not_stuck().check_isolation();
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    let reports = [
        rpt(1, 0, 5e9 as u64, 0, &[0], 0),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[4, 5], 50),
    ];
    let r = plan.assert_cgroup(&reports, Some(&expected), None);
    assert!(r.is_fail());
    // No-progress detail must name tid 1 with "0 work units".
    let no_progress_detail = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::NoProgress))
        .unwrap();
    assert!(
        no_progress_detail.message.contains("tid 1"),
        "no-progress tid: {no_progress_detail}"
    );
    assert!(
        no_progress_detail.message.contains("0 work units"),
        "format: {no_progress_detail}"
    );
    // Isolation detail must name tid 2 with CPUs {4, 5}.
    let iso_detail = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::Isolation))
        .unwrap();
    assert!(
        iso_detail.message.contains("tid 2"),
        "isolation tid: {iso_detail}"
    );
    assert!(
        iso_detail.message.contains("4"),
        "must list CPU 4: {iso_detail}"
    );
    assert!(
        iso_detail.message.contains("5"),
        "must list CPU 5: {iso_detail}"
    );
}

#[test]
fn neg_assert_cgroup_via_assert_struct() {
    let v = Assert::NO_OVERRIDES.check_not_stuck().check_isolation();
    let expected: BTreeSet<usize> = [0].into_iter().collect();
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 2], 50)];
    let r = v.assert_cgroup(&reports, Some(&expected));
    assert!(
        !r.is_pass(),
        "Assert.assert_cgroup must catch isolation failure"
    );
    let detail = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::Isolation))
        .unwrap();
    assert!(detail.message.contains("tid 1"), "must name tid: {detail}");
    assert!(detail.message.contains("1"), "must list CPU 1: {detail}");
    assert!(detail.message.contains("2"), "must list CPU 2: {detail}");
}

#[test]
fn neg_plan_custom_gap_passes_below_threshold() {
    let plan = AssertPlan::new().check_not_stuck().max_gap_ms(5000);
    let reports = [
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 1000),
    ];
    let r = plan.assert_cgroup(&reports, None, None);
    // 1000ms gap < 5000ms threshold, so it passes.
    let has_stuck = r
        .failure_details()
        .any(|d| matches!(d.kind, DetailKind::Stuck));
    assert!(!has_stuck, "1000ms gap should pass 5000ms threshold");
}

#[test]
fn not_stuck_empty_reports_surface_zero_worker_cgroup() {
    // no-silent-drop: a declared cgroup that collected zero reports
    // must surface as a num_workers=0 telemetry entry, not vanish. Pre-fix
    // assert_not_stuck early-returned empty stats on reports.is_empty().
    let r = assert_not_stuck(&[]);
    assert!(r.is_pass(), "no workers -> no fairness fail");
    assert_eq!(
        r.stats.cgroups.len(),
        1,
        "empty-reports cgroup must surface a zero-worker entry, not be dropped",
    );
    assert_eq!(r.stats.cgroups[0].num_workers, 0);
    assert_eq!(r.stats.total_workers, 0);
}

#[test]
fn not_stuck_no_double_count() {
    // assert_not_stuck builds telemetry via the shared `cgroup_stats`
    // builder exactly once — a regression re-introducing a second build
    // site would yield two cgroup entries for one call.
    let r = assert_not_stuck(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 1000, 5e9 as u64, 6e8 as u64, &[1], 60),
    ]);
    assert_eq!(
        r.stats.cgroups.len(),
        1,
        "exactly one cgroup entry per call"
    );
    assert_eq!(r.stats.total_workers, 2);
}

/// `cgroup_stats` retains the UNION of every worker's `cpus_used` as the
/// per-cgroup `cpus_used` set (not just the `num_cpus` count), and leaves
/// `cgroup_name` empty (the collection layer labels it). Pins the
/// capture: which CPUs a cgroup ran on is available on every run, not
/// only the failure-path scheduler dump.
#[test]
fn cgroup_stats_retains_cpus_used_union_and_leaves_name_empty() {
    let reports = vec![
        rpt(1, 100, 1_000_000, 0, &[0, 1], 10),
        rpt(2, 100, 1_000_000, 0, &[1, 2, 3], 10),
    ];
    let cg = cgroup_stats(&reports);
    assert_eq!(
        cg.cpus_used,
        [0usize, 1, 2, 3]
            .into_iter()
            .collect::<std::collections::BTreeSet<usize>>(),
        "cpus_used must be the union of every worker's cpus_used",
    );
    assert_eq!(cg.num_cpus, 4, "num_cpus stays the union length");
    assert!(
        cg.cgroup_name.is_empty(),
        "cgroup_stats leaves the name empty; collect_handles labels it",
    );
}
