//! `assert_benchmarks` and `AssertPlan` benchmarking-path tests:
//! p99 / CV / iteration-rate thresholds, the ns-vs-µs unit
//! invariant, wake-latency populate paths in `assert_not_stuck`,
//! schedstat run-delay aggregation, and the `assert_cgroup`
//! migration-ratio gate.

use super::tests_common::{rpt, rpt_with_latencies};
use super::*;

#[test]
fn assert_benchmarks_empty_reports() {
    // Empty reports → skip terminal verdict. Skip is
    // is_skip()=true, is_fail()=false, is_pass()=false (a skipped
    // scenario didn't run, so it's not a pass). The thresholds
    // supplied here cannot be evaluated against zero signal — a
    // silent pass would mask a broken run.
    let r = assert_benchmarks(&[], Some(1000), Some(0.5), Some(100.0));
    assert!(
        !r.is_fail(),
        "empty-reports skip must not surface as failure"
    );
    assert!(r.is_skip(), "no reports must surface as skipped");
    assert!(!r.is_pass(), "skip is not pass");
    let skip_details: Vec<&AssertDetail> = r.skip_details().collect();
    assert!(
        skip_details
            .iter()
            .any(|d| matches!(d.kind, DetailKind::Skip) && d.message.contains("no worker reports")),
        "skip detail must carry the 'no worker reports' reason: {:?}",
        skip_details,
    );
}

#[test]
fn assert_benchmarks_no_thresholds() {
    let reports = [rpt_with_latencies(
        1,
        vec![1000, 2000, 3000],
        10,
        5_000_000_000,
    )];
    let r = assert_benchmarks(&reports, None, None, None);
    assert!(r.is_pass());
}

#[test]
fn assert_benchmarks_p99_pass() {
    let reports = [rpt_with_latencies(
        1,
        vec![100, 200, 300, 400, 500],
        10,
        5_000_000_000,
    )];
    let r = assert_benchmarks(&reports, Some(1000), None, None);
    assert!(r.is_pass(), "p99 500ns < 1000ns limit: {:?}", r.outcomes);
}

#[test]
fn assert_benchmarks_p99_n100_at_limit_passes() {
    // With samples [0..100], the nearest-rank p99 is 98
    // (sorted[ceil(100*0.99) - 1] = sorted[98]). Setting the
    // limit to 99 must pass (98 <= 99). An off-by-one that
    // returns sorted[99] = 99 would pass the same limit for
    // the wrong reason — the paired _fail test below pins
    // down the correct index.
    let latencies: Vec<u64> = (0..100).collect();
    let reports = [rpt_with_latencies(1, latencies, 100, 5_000_000_000)];
    let r = assert_benchmarks(&reports, Some(99), None, None);
    assert!(
        r.is_pass(),
        "p99 should be 98, under limit 99: {:?}",
        r.outcomes
    );
}

#[test]
fn assert_benchmarks_p99_n100_below_old_p100_passes() {
    // Tighter regression: with samples [0..100], set the limit to
    // 98. Correct p99 (98) equals the limit and passes (strict
    // `p99 > p99_limit` comparison). The old off-by-one returned
    // 99, which would have FAILED (99 > 98). This test therefore
    // only passes with the corrected index.
    let latencies: Vec<u64> = (0..100).collect();
    let reports = [rpt_with_latencies(1, latencies, 100, 5_000_000_000)];
    let r = assert_benchmarks(&reports, Some(98), None, None);
    assert!(
        r.is_pass(),
        "corrected p99 (98) must equal limit 98 and pass: {:?}",
        r.outcomes
    );
}

#[test]
fn assert_not_stuck_p99_n100_is_99_microseconds() {
    // assert_not_stuck computes p99 as microseconds on the per-cgroup
    // CgroupStats. Samples = [1000, 2000, ..., 100_000] ns
    // (100 values at kilo-ns spacing) so the reported p99 is
    // exactly 99.0us with the correct index
    // (sorted[ceil(100*0.99) - 1] = sorted[98] = 99_000ns = 99us).
    // An off-by-one that returns sorted[99] would yield 100us.
    let latencies: Vec<u64> = (1..=100).map(|v: u64| v * 1000).collect();
    let reports = [rpt_with_latencies(1, latencies, 100, 5_000_000_000)];
    let r = assert_not_stuck(&reports);
    assert_eq!(
        r.stats.cgroups[0].p99_wake_latency_us, 99.0,
        "p99 must equal 99.0us (sorted[98] = 99_000ns), got {}us",
        r.stats.cgroups[0].p99_wake_latency_us
    );
}

#[test]
fn assert_benchmarks_p99_fail() {
    let reports = [rpt_with_latencies(
        1,
        vec![100, 200, 300, 400, 2000],
        10,
        5_000_000_000,
    )];
    let r = assert_benchmarks(&reports, Some(1000), None, None);
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::Benchmark)
                && d.message.contains("p99 wake latency"))
    );
}

/// Unit-boundary pin: the `max_p99_wake_latency_ns` threshold
/// MUST be compared against `WorkerReport::wake_latencies_ns`
/// (nanoseconds) — never against the microsecond-valued
/// `CgroupStats::p99_wake_latency_us` field. A regression that
/// divided either side by 1000 (or multiplied by 1000) would
/// make the threshold fire 1000× too often or 1000× too rarely,
/// silently corrupting every regression gate that uses this
/// field.
///
/// Construction: plant `wake_latencies_ns` values that are
/// clearly in the NS scale (e.g. 5000 ns = 5 µs) and set a
/// threshold of 4999 ns. The assertion must FAIL at 4999 ns and
/// PASS at 5001 ns. If the comparison were accidentally
/// converting the threshold to µs (dividing by 1000), 4999
/// would behave like "4.999 µs threshold against a 5 µs p99"
/// — technically still a fail but for the wrong reason. The
/// bracket here (5000-1 vs 5000+1) sits inside the 1000× slop
/// so a unit-swap regression would flip the verdict on one of
/// the two cases.
#[test]
fn assert_p99_ns_threshold_compares_against_ns_latencies() {
    // Single-sample latency set: p99 == the sample value.
    let reports = [rpt_with_latencies(1, vec![5000], 10, 5_000_000_000)];

    // Threshold just below the 5000 ns sample -> FAIL.
    let fail = assert_benchmarks(&reports, Some(4999), None, None);
    assert!(
        fail.is_fail(),
        "threshold 4999 ns against 5000 ns p99 must fail — if this \
         passes, the comparison may be converting to µs and eating \
         3 digits of resolution",
    );

    // Threshold just above the 5000 ns sample -> PASS.
    let pass = assert_benchmarks(&reports, Some(5001), None, None);
    assert!(
        pass.is_pass(),
        "threshold 5001 ns against 5000 ns p99 must pass — if this \
         fails, the comparison may be multiplying the threshold by \
         1000 (treating it as µs)",
    );

    // Cross-check the reporting path: `assert_not_stuck` computes
    // `CgroupStats::p99_wake_latency_us` in MICROSECONDS (ns / 1000). A
    // regression that conflated the reporting field with the threshold input
    // would surface as either `us == ns` (forgot to divide) or
    // `us == ns/1_000_000` (double-converted).
    let stats = assert_not_stuck(&reports);
    assert_eq!(
        stats.stats.cgroups[0].p99_wake_latency_us, 5.0,
        "5000 ns / 1000 = 5.0 µs — if this renders as 5000 (forgot /1000) \
         or 0.005 (extra /1000), the reporting-path unit conversion drifted",
    );
}

#[test]
fn assert_benchmarks_cv_pass() {
    // All same latency -> CV = 0.
    let reports = [rpt_with_latencies(
        1,
        vec![1000, 1000, 1000, 1000],
        10,
        5_000_000_000,
    )];
    let r = assert_benchmarks(&reports, None, Some(0.5), None);
    assert!(r.is_pass(), "uniform latencies CV=0: {:?}", r.outcomes);
}

#[test]
fn assert_benchmarks_cv_fail() {
    // High variance latencies.
    let reports = [rpt_with_latencies(
        1,
        vec![100, 100, 100, 100000],
        10,
        5_000_000_000,
    )];
    let r = assert_benchmarks(&reports, None, Some(0.5), None);
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::Benchmark)
                && d.message.contains("wake latency CV"))
    );
}

#[test]
fn assert_benchmarks_iteration_rate_pass() {
    // 1000 iterations in 5 seconds = 200/s, above 100/s floor.
    let reports = [rpt_with_latencies(1, vec![], 1000, 5_000_000_000)];
    let r = assert_benchmarks(&reports, None, None, Some(100.0));
    assert!(r.is_pass(), "200/s > 100/s floor: {:?}", r.outcomes);
}

#[test]
fn assert_benchmarks_iteration_rate_fail() {
    // 10 iterations in 5 seconds = 2/s, below 100/s floor.
    let reports = [rpt_with_latencies(1, vec![], 10, 5_000_000_000)];
    let r = assert_benchmarks(&reports, None, None, Some(100.0));
    assert!(r.is_fail());
    assert!(
        r.failure_details().any(
            |d| matches!(d.kind, DetailKind::Benchmark) && d.message.contains("iteration rate")
        )
    );
}

#[test]
fn assert_benchmarks_zero_cpu_time_fails_floor() {
    // A worker that spent zero CPU time (`cpu_time_ns == 0`, here via a
    // zero wall_time report) has NO proven throughput. The
    // CPU-denominated gate rates it 0 and fails any positive floor —
    // no CPU spent means no work demonstrated. There is no Inconclusive
    // arm: this is a real, actionable failure, not an unknowable one.
    let reports = [rpt_with_latencies(1, vec![], 10, 0)];
    let r = assert_benchmarks(&reports, None, None, Some(100.0));
    assert!(
        r.is_fail(),
        "zero cpu_time must fail the floor, not pass or go inconclusive: {:?}",
        r.outcomes,
    );
    assert!(
        !r.is_inconclusive(),
        "zero CPU is a hard failure, not unknowable"
    );
    let reason = r
        .failure_details()
        .find(|d| d.kind == DetailKind::Benchmark)
        .unwrap_or_else(|| panic!("expected Benchmark failure, got {:?}", r.outcomes));
    assert!(
        reason.message.contains("0.0/cpu-s") && reason.message.contains("floor"),
        "diagnostic must show the zero CPU rate against the floor: {reason}"
    );
}

#[test]
fn assert_benchmarks_mixed_zero_and_nonzero_cpu_both_fail() {
    // With CPU-denomination there is no zero-worker skip: a zero-CPU
    // worker rates 0 (fails) and a below-floor worker fails on its own
    // rate. Both surface — the gate never short-circuits on one
    // worker's zero CPU time.
    let reports = [
        rpt_with_latencies(1, vec![], 10, 0), // cpu 0 -> rate 0 -> fail
        rpt_with_latencies(2, vec![], 1, 5_000_000_000), // cpu 2.5s, 1 iter -> 0.4/cpu-s < 100
    ];
    let r = assert_benchmarks(&reports, None, None, Some(100.0));
    assert!(
        r.is_fail(),
        "both workers below floor must fail: {:?}",
        r.outcomes,
    );
    assert!(!r.is_inconclusive());
    assert!(
        r.failure_details()
            .any(|d| d.message.contains("worker 2") && d.message.contains("iteration rate")),
        "expected worker-2 rate failure: {:?}",
        r.outcomes,
    );
}

#[test]
fn assert_benchmarks_iteration_rate_is_dilation_safe() {
    // The whole point of CPU-denomination: a worker starved of CPU by a
    // busy host (wall ≫ CPU) is judged on the CPU it actually got, not
    // wall-clock. iterations=100, cpu_time=50s, wall=200s:
    //   CPU rate  = 100 / 50  = 2.0/cpu-s  -> PASSES floor 1.5
    //   wall rate = 100 / 200 = 0.5/s      -> would FALSE-FAIL floor 1.5
    // The old wall-denominated gate false-failed this honest workload
    // under host dilation; the CPU gate passes it.
    let mut w = rpt_with_latencies(1, vec![], 100, 200_000_000_000);
    w.cpu_time_ns = 50_000_000_000;
    let r = assert_benchmarks(&[w], None, None, Some(1.5));
    assert!(
        r.is_pass(),
        "CPU rate 2.0/cpu-s must pass floor 1.5 despite wall rate 0.5/s: {:?}",
        r.outcomes,
    );
}

#[test]
fn assert_benchmarks_iteration_rate_detects_below_floor_regardless_of_wall() {
    // Detection is preserved: a worker whose CPU rate is below the floor
    // fails even when its wall rate looks healthy. iterations=10,
    // cpu_time=10s (CPU rate 1.0/cpu-s), wall=1ms (wall rate 10000/s):
    // the CPU verdict fails floor 100 while the wall rate would sail
    // past it — the CPU gate is not fooled by a tiny wall window.
    let mut w = rpt_with_latencies(1, vec![], 10, 1_000_000);
    w.cpu_time_ns = 10_000_000_000;
    let r = assert_benchmarks(&[w], None, None, Some(100.0));
    assert!(
        r.is_fail(),
        "CPU rate 1.0/cpu-s must fail floor 100 despite high wall rate: {:?}",
        r.outcomes,
    );
    let reason = r
        .failure_details()
        .find(|d| d.kind == DetailKind::Benchmark)
        .unwrap_or_else(|| panic!("expected Benchmark failure, got {:?}", r.outcomes));
    assert!(
        reason.message.contains("1.0/cpu-s") && reason.message.contains("100.0/cpu-s"),
        "message must show one CPU-denominated value and floor: {reason}"
    );
}

#[test]
fn dilation_annotation_fires_on_latency_failure_under_dilation() {
    // Synthetic D=3.0 with a wall-denominated latency failure present:
    // the annotation renders the measured dilation, the CPU-share
    // percent (1/3 ≈ 33%), and the wall-latency caveat.
    let mut r = AssertResult::pass();
    r.record_fail(AssertDetail::new(
        DetailKind::Benchmark,
        "p99 wake latency 9000ns exceeds limit 1ns".to_string(),
    ));
    let note = dilation_annotation(Some(3.0), &r);
    assert!(
        note.contains("--- host dilation ---"),
        "section header: {note}"
    );
    assert!(note.contains("D=3.00x"), "measured D: {note}");
    assert!(note.contains("33%"), "1/D CPU-share percent: {note}");
    assert!(
        note.contains("include host preemption"),
        "wall-latency caveat: {note}"
    );
}

#[test]
fn dilation_annotation_silent_in_performance_mode() {
    // D at/below 1.05 is effectively undilated (perf mode) — nothing to
    // explain even with a latency failure present.
    let mut r = AssertResult::pass();
    r.record_fail(AssertDetail::new(
        DetailKind::Benchmark,
        "p99 wake latency over limit".to_string(),
    ));
    assert!(dilation_annotation(Some(1.02), &r).is_empty());
    // None (schedstats-off host) is treated as D=1.0 -> no annotation.
    assert!(dilation_annotation(None, &r).is_empty());
}

#[test]
fn dilation_annotation_requires_a_confounded_failure() {
    // A failure kind for which D carries no context (an isolation
    // violation) must NOT draw the note, even at high dilation.
    let mut r = AssertResult::pass();
    r.record_fail(AssertDetail::new(
        DetailKind::Isolation,
        "tid 1 used cpu 7 outside cpuset".to_string(),
    ));
    assert!(
        dilation_annotation(Some(3.0), &r).is_empty(),
        "a non-dilation-confounded kind must not trigger the note",
    );
}

#[test]
fn dilation_annotation_fires_on_stuck_for_wall_gap_context() {
    // A Stuck failure DOES draw the note: the verdict is the CPU gap
    // (dilation-safe), but the message carries wall-gap evidence and D is
    // what relates the two (wall ≈ cpu × D under pure host preemption; a
    // wall gap far beyond that points at guest-side starvation).
    let mut r = AssertResult::pass();
    r.record_fail(AssertDetail::new(
        DetailKind::Stuck,
        "tid 1 stuck 9000ms cpu-gap on cpu0 at +1000ms (threshold 50ms, wall gap 9100ms)"
            .to_string(),
    ));
    let note = dilation_annotation(Some(3.0), &r);
    assert!(
        note.contains("D=3.00x") && note.contains("wall-gap evidence"),
        "Stuck must draw the D note as wall-gap context: {note}",
    );
}

/// A p99 threshold over ZERO wake samples records Inconclusive, not a
/// silent Pass — the same fail-closed shape as the zero-mean CV arm
/// below. The silent skip let a wake-latency gate look "green" on a
/// workload that never blocks (the sched_perf_negative vacuity).
#[test]
fn assert_benchmarks_no_latencies_inconclusive_p99() {
    let reports = [rpt_with_latencies(1, vec![], 10, 5_000_000_000)];
    let r = assert_benchmarks(&reports, Some(1000), None, None);
    assert!(
        r.is_inconclusive(),
        "empty latencies with a p99 threshold must be inconclusive, got: {r:?}"
    );
    assert!(
        format!("{r:?}").contains("0 wake samples"),
        "inconclusive detail must name the zero-sample cause: {r:?}"
    );
}

#[test]
fn assert_benchmarks_single_latency_cv_skipped() {
    // Single sample -> len < 2, CV check skipped.
    let reports = [rpt_with_latencies(1, vec![1000], 10, 5_000_000_000)];
    let r = assert_benchmarks(&reports, None, Some(0.1), None);
    assert!(r.is_pass(), "single sample should skip CV check");
}

/// Wake-latency CV gate with N>=2 samples but every sample is
/// zero (mean==0) → Inconclusive, not Pass. CV is dispersion /
/// mean, so a zero mean makes the denominator zero and the
/// metric undefined. Previously slid past the gate as a silent
/// Pass (the `if mean > 0` arm was skipped without recording
/// anything); the Inconclusive arm at mod.rs records a
/// `DetailKind::Benchmark` carrying the operator hint instead.
/// Pins the zero-mean CV path explicitly so a regression that
/// reverts to "skip silently when mean == 0" is caught.
#[test]
fn assert_benchmarks_wake_latency_cv_zero_mean_yields_inconclusive() {
    // Multi-sample worker but every wake-latency sample is zero
    // (mean == 0, len >= 2 → hits the zero-mean Inconclusive arm).
    let reports = [rpt_with_latencies(1, vec![0, 0, 0, 0], 10, 5_000_000_000)];
    let r = assert_benchmarks(&reports, None, Some(0.5), None);
    assert!(
        r.is_inconclusive(),
        "zero-mean wake-latency CV must be Inconclusive, not Pass: {:?}",
        r.outcomes,
    );
    assert!(!r.is_pass(), "must not silently pass on zero denominator");
    assert!(!r.is_fail(), "no actual CV violation to report");
    let reason = r
        .inconclusive_details()
        .find(|d| d.kind == DetailKind::Benchmark)
        .unwrap_or_else(|| {
            panic!(
                "expected Benchmark Inconclusive reason, got {:?}",
                r.outcomes
            )
        });
    assert!(
        reason.message.contains("wake latency CV inconclusive"),
        "diagnostic must label the gate: {reason}",
    );
    assert!(
        reason.message.contains("zero mean wake"),
        "diagnostic must name the root cause: {reason}",
    );
    assert!(
        reason.message.contains("non-zero latency"),
        "diagnostic must surface the operator-actionable hint: {reason}",
    );
}

// -- wake latency stats in assert_not_stuck --
//
// These assert the PER-CGROUP reductions (`r.stats.cgroups[0].*`), the
// source CgroupStats fields. The run-level surface that re-pools them
// (the deleted `r.stats.worst_*` roll-ups, now ext-sourced Distribution
// metrics) is covered separately: `repool_distribution_value_for_value_
// with_cgroup_stats` (tests_phase_bucket/repool.rs) pins that a single-cgroup
// run's run-level pooled value reproduces these cgroup_stats reductions.

#[test]
fn not_stuck_wake_latency_stats() {
    let reports = [
        rpt_with_latencies(1, vec![1000, 2000, 3000, 4000, 5000], 100, 5_000_000_000),
        rpt_with_latencies(2, vec![6000, 7000, 8000, 9000, 10000], 200, 5_000_000_000),
    ];
    let r = assert_not_stuck(&reports);
    assert!(r.is_pass(), "{:?}", r.outcomes);
    let cg = &r.stats.cgroups[0];
    // p99 of [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000] in us:
    // sorted, percentile index = ceil(10*0.99) - 1 = 9 -> sorted[9] = 10000ns = 10.0us
    assert!(
        cg.p99_wake_latency_us > 9.0,
        "p99: {}",
        cg.p99_wake_latency_us
    );
    // median of 10 samples via `percentile(sorted, 0.5)`:
    // nearest-rank index = ceil(10 * 0.5) - 1 = 4 →
    // sorted[4] = 5000ns = 5.0us. The lower-of-two-middles
    // bound matches the convention documented on
    // `CgroupStats::median_wake_latency_us`.
    assert!(
        (cg.median_wake_latency_us - 5.0).abs() < 0.1,
        "median: {}",
        cg.median_wake_latency_us
    );
    assert!(cg.wake_latency_cv > 0.0, "cv: {}", cg.wake_latency_cv);
    assert_eq!(r.stats.total_iterations, 300);
}

#[test]
fn not_stuck_empty_latencies_zero_stats() {
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
    let r = assert_not_stuck(&reports);
    assert!(r.is_pass());
    assert_eq!(r.stats.cgroups[0].p99_wake_latency_us, 0.0);
    assert_eq!(r.stats.cgroups[0].median_wake_latency_us, 0.0);
    assert_eq!(r.stats.cgroups[0].wake_latency_cv, 0.0);
}

#[test]
fn not_stuck_run_delay_stats() {
    let mut w1 = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50);
    w1.schedstat_run_delay_ns = 100_000; // 100us
    let mut w2 = rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 50);
    w2.schedstat_run_delay_ns = 300_000; // 300us
    let r = assert_not_stuck(&[w1, w2]);
    assert!(r.is_pass(), "{:?}", r.outcomes);
    // mean_run_delay = (100 + 300) / 2 = 200us
    assert!(
        (r.stats.cgroups[0].mean_run_delay_us - 200.0).abs() < 0.1,
        "mean: {}",
        r.stats.cgroups[0].mean_run_delay_us
    );
    // worst_run_delay = 300us
    assert!(
        (r.stats.cgroups[0].worst_run_delay_us - 300.0).abs() < 0.1,
        "worst: {}",
        r.stats.cgroups[0].worst_run_delay_us
    );
}

// -- AssertPlan benchmarking integration --

#[test]
fn plan_benchmarks_p99_via_assert_cgroup() {
    let plan = AssertPlan {
        not_stuck: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: Some(500),
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: None,
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };
    let reports = [rpt_with_latencies(
        1,
        vec![100, 200, 300, 400, 1000],
        10,
        5_000_000_000,
    )];
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(!r.is_pass(), "p99 1000ns > 500ns limit");
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::Benchmark)
                && d.message.contains("p99 wake latency"))
    );
}

#[test]
fn plan_migration_ratio_gate() {
    let mut w = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50);
    w.migration_count = 10;
    w.iterations = 100;
    // ratio = 10/100 = 0.10, threshold 0.05 → fail
    let plan = AssertPlan {
        not_stuck: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: Some(0.05),
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };
    let r = plan.assert_cgroup(&[w], None, None);
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::Migration)
                && d.message.contains("migration ratio"))
    );
}

#[test]
fn plan_migration_ratio_gate_pass() {
    let mut w = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50);
    w.migration_count = 2;
    w.iterations = 100;
    // ratio = 2/100 = 0.02, threshold 0.05 → pass
    let plan = AssertPlan {
        not_stuck: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: Some(0.05),
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };
    let r = plan.assert_cgroup(&[w], None, None);
    assert!(r.is_pass(), "{:?}", r.outcomes);
}

/// A workload that produced zero iterations across every worker
/// gives the migration-ratio check a zero denominator. Without the
/// Inconclusive carve-out, `total_mig as f64 / 0` would collapse
/// to 0.0 (via the prior `if total_iters > 0 ... else { 0.0 }`
/// fallback) and 0.0 ≤ threshold would trivially pass — a false
/// "no migrations under the bar" verdict on a workload that
/// never ran. Pin that the gate now records an Inconclusive
/// outcome carrying a `DetailKind::Migration` reason that names
/// the zero-denominator condition.
#[test]
fn plan_migration_ratio_zero_iterations_is_inconclusive_not_pass() {
    let mut w = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50);
    w.migration_count = 5;
    w.iterations = 0; // zero denominator — workload did not iterate
    let plan = AssertPlan {
        not_stuck: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: Some(0.05),
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };
    let r = plan.assert_cgroup(&[w], None, None);
    assert!(
        !r.is_pass(),
        "zero-iteration workload must NOT trivially pass migration-ratio gate; got: {:?}",
        r.outcomes
    );
    assert!(
        !r.is_fail(),
        "zero-iteration workload is Inconclusive, not Fail; got: {:?}",
        r.outcomes
    );
    assert!(
        r.is_inconclusive(),
        "expected Inconclusive verdict on zero denominator; got: {:?}",
        r.outcomes
    );
    let reasons: Vec<_> = r.inconclusive_details().collect();
    assert_eq!(reasons.len(), 1, "exactly one Inconclusive reason expected");
    assert_eq!(reasons[0].kind, DetailKind::Migration);
    assert!(
        reasons[0].message.contains("0 iterations") && reasons[0].message.contains("inconclusive"),
        "Inconclusive reason must name zero-iteration condition; got: {}",
        reasons[0].message
    );
}

#[test]
fn plan_benchmarks_iteration_rate_via_assert_cgroup() {
    let plan = AssertPlan {
        not_stuck: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: Some(1000.0),
        max_migration_ratio: None,
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };
    let reports = [rpt_with_latencies(1, vec![], 10, 5_000_000_000)];
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(!r.is_pass(), "2/s < 1000/s floor");
    assert!(
        r.failure_details().any(
            |d| matches!(d.kind, DetailKind::Benchmark) && d.message.contains("iteration rate")
        )
    );
}

#[test]
fn assert_throughput_parity_all_zero_cpu_time_inconclusive_when_cv_set() {
    // When every worker recorded zero cpu_time the per-worker rate
    // is zero, the mean is zero, and CV is mathematically
    // undefined. The previous gate (`mean > 0.0`) silently skipped
    // the check and reported a pass — masking a workload that
    // never accumulated any CPU time. The fix surfaces it as
    // Inconclusive: the check ran but had no signal to evaluate;
    // neither Pass (would mask the broken state) nor Fail (no
    // actual CV violation observed) is truthful.
    let mut a = rpt(1, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    let mut b = rpt(2, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    a.cpu_time_ns = 0;
    b.cpu_time_ns = 0;
    let r = assert_throughput_parity(&[a, b], Some(0.5), None);
    assert!(
        r.is_inconclusive(),
        "all-zero cpu_time must be Inconclusive when max_cv set: {:?}",
        r.outcomes,
    );
    assert!(!r.is_pass(), "must not silently pass on zero denominator");
    assert!(!r.is_fail(), "no actual CV violation to report");
    let reason = r
        .inconclusive_details()
        .find(|d| d.kind == DetailKind::Benchmark)
        .unwrap_or_else(|| panic!("expected Inconclusive reason, got {:?}", r.outcomes));
    assert!(
        reason.message.contains("zero cpu_time_ns"),
        "diagnostic must name the root cause: {reason}"
    );
    assert!(
        reason.message.contains("able to run"),
        "diagnostic must surface the operator-actionable hint: {reason}"
    );
}

#[test]
fn assert_throughput_parity_all_zero_cpu_time_inconclusive_when_min_rate_set() {
    // Symmetric case for the min_rate floor: per-worker zero-cpu
    // rates would synthesize N duplicate Fails on data that
    // couldn't be evaluated. The fix detects all-zero-cpu once
    // and records a single Inconclusive naming the unevaluated
    // limit (here `min_rate 100`). Pins the all-zero path under
    // the min_rate-only branch separately from the CV branch.
    let mut a = rpt(1, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    let mut b = rpt(2, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    a.cpu_time_ns = 0;
    b.cpu_time_ns = 0;
    let r = assert_throughput_parity(&[a, b], None, Some(100.0));
    assert!(
        r.is_inconclusive(),
        "all-zero cpu_time must be Inconclusive when min_rate set: {:?}",
        r.outcomes,
    );
    assert!(
        !r.is_fail(),
        "no per-worker Fail when every worker is zero-cpu"
    );
    let reason = r
        .inconclusive_details()
        .find(|d| d.kind == DetailKind::Benchmark)
        .unwrap_or_else(|| panic!("expected Benchmark Inconclusive, got {:?}", r.outcomes));
    assert!(
        reason.message.contains("zero cpu_time_ns"),
        "diagnostic must name the root cause: {reason}"
    );
    assert!(
        reason.message.contains("min_rate 100"),
        "diagnostic must name the unevaluated limit: {reason}"
    );
}

#[test]
fn assert_throughput_parity_all_zero_cpu_time_emits_single_inconclusive_when_both_limits_set() {
    // When both `max_cv` and `min_rate` are set AND every worker
    // recorded zero cpu_time, the same root cause (denominator is
    // zero) blocks both gates. Emit ONE Inconclusive listing both
    // unevaluated limits — not two separate records with stuttering
    // "denominator is zero" diagnostics. Pins the dedup behavior.
    let mut a = rpt(1, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    let mut b = rpt(2, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    a.cpu_time_ns = 0;
    b.cpu_time_ns = 0;
    let r = assert_throughput_parity(&[a, b], Some(0.5), Some(100.0));
    assert!(r.is_inconclusive());
    let inconclusives: Vec<_> = r.inconclusive_details().collect();
    assert_eq!(
        inconclusives.len(),
        1,
        "all-zero-cpu with both limits set must produce a single Inconclusive, got {:?}",
        r.outcomes,
    );
    let msg = &inconclusives[0].message;
    assert!(
        msg.contains("max_cv 0.500"),
        "must list max_cv limit: {msg}"
    );
    assert!(
        msg.contains("min_rate 100"),
        "must list min_rate limit: {msg}"
    );
    assert!(
        msg.contains("zero cpu_time_ns"),
        "must name root cause: {msg}"
    );
}

#[test]
fn assert_throughput_parity_mixed_zero_and_nonzero_cpu_does_not_short_circuit() {
    // One worker has zero cpu_time (skipped) but another has valid
    // cpu_time below floor = the gate evaluates the non-zero worker
    // and records Fail (NOT Inconclusive). Pins the
    // zero_cpu_count == reports.len() guard — a regression that
    // triggered on any zero-cpu worker would hide real rate
    // failures on the workers that did run.
    let mut a = rpt(1, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    let mut b = rpt(2, 1, 5_000_000_000, 5_000_000_000, &[0], 0);
    a.cpu_time_ns = 0;
    b.cpu_time_ns = 5_000_000_000;
    // b: work_units=1, cpu_time_s=5 → rate = 0.2 work/cpu_s
    let r = assert_throughput_parity(&[a, b], None, Some(100.0));
    assert!(
        r.is_fail(),
        "non-zero-cpu worker below floor must fail: {:?}",
        r.outcomes,
    );
    assert!(!r.is_inconclusive(), "only all-zero is Inconclusive");
    assert!(
        r.failure_details()
            .any(|d| d.message.contains("worker 2") && d.message.contains("below floor")),
        "expected worker-2 below-floor failure: {:?}",
        r.outcomes,
    );
}

#[test]
fn assert_throughput_parity_cv_excludes_zero_cpu_workers() {
    // A zero-cpu worker's rate is unknowable, NOT 0.0. The CV must be computed
    // over MEASURED workers only — matching the min_rate gate's documented
    // zero-cpu exclusion. Two uniform real workers (rate 1000 each) plus one
    // zero-cpu worker: CV over the measured pair is 0.0 -> PASS. Folding the
    // zero-cpu worker's forced 0.0 would give mean 666.7, cv 0.707 > 0.5 and
    // FAIL a genuinely uniform workload.
    let mut a = rpt(1, 0, 0, 1_000_000_000, &[0], 0);
    let mut b = rpt(2, 0, 0, 1_000_000_000, &[0], 0);
    let mut c = rpt(3, 0, 0, 1_000_000_000, &[0], 0);
    a.work_units = 1000;
    a.cpu_time_ns = 1_000_000_000; // rate 1000 work/cpu_s
    b.work_units = 1000;
    b.cpu_time_ns = 1_000_000_000; // rate 1000 work/cpu_s
    c.cpu_time_ns = 0; // zero-cpu -> excluded from the CV
    let r = assert_throughput_parity(&[a, b, c], Some(0.5), None);
    assert!(
        r.is_pass(),
        "uniform measured workers (cv 0.0) must PASS; a zero-cpu worker's \
         unknowable rate must not inflate the CV: {:?}",
        r.outcomes,
    );
}

#[test]
fn assert_throughput_parity_all_zero_cpu_time_passes_without_cv() {
    // No CV check requested → no failure. The min_rate floor is
    // also unset, so the function has nothing to evaluate and
    // passes. This pins the gate scope: the new failure is
    // specific to the configured-CV-with-zero-mean case.
    let mut a = rpt(1, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    let mut b = rpt(2, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    a.cpu_time_ns = 0;
    b.cpu_time_ns = 0;
    let r = assert_throughput_parity(&[a, b], None, None);
    assert!(
        r.is_pass(),
        "no CV configured → no failure: {:?}",
        r.outcomes
    );
}
