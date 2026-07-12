//! Fixed-work CPU-time measurement sanity — the permanent half of the
//! steal-exclusion verification.
//!
//! PROPERTY UNDER TEST (the full claim): a guest worker's
//! `CLOCK_THREAD_CPUTIME_ID` reading excludes host-stolen time — the guest
//! task clock is driven by `rq->clock_task`, which the kernel steal-adjusts
//! (`update_rq_clock_task` subtracts `paravirt_steal_rq_enabled` steal), so
//! the CPU-second denominators behind `min_iteration_rate`, `max_gap_ms`,
//! and the CPU-denominated throughput metrics measure the workload, not the
//! host.
//!
//! The full A/B (idle host vs loaded host, fixed-iteration SpinWait,
//! `cpu_time_ns` invariant within ±20% while wall time inflates) is a
//! HOST-CONDITION experiment — it needs a deliberately loaded host, which a
//! CI runner can neither guarantee nor safely create. It is run MANUALLY
//! (the recorded numbers live in the change description); this file lands
//! the idle-only sanity that keeps the measurement machinery honest on
//! every run:
//!
//! - the fixed work completes exactly (`iterations == N`, `park_after`);
//! - `cpu_time_ns` is nonzero and consistent with the kernel's OWN on-CPU
//!   accounting for the same thread (`/proc/<tid>/schedstat` field 1,
//!   `sum_exec_runtime`) within a generous band — the two clocks are
//!   independent code paths that must agree on steal-adjusted on-CPU time;
//! - the measured `cpu_time_ns` is surfaced as an info note so a manual
//!   A/B run can read the number straight from the test output/sidecar.

use anyhow::Result;
use ktstr::assert::{Assert, AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::workload::{WorkType, WorkloadConfig, WorkloadHandle};
use std::time::{Duration, Instant};

/// Fixed work quota: SpinWait outer iterations (1024 spin work units
/// each). ~0.5M iterations is a few seconds of busy spin on one idle
/// vCPU — long enough that scheduler noise is amortized, short enough
/// to finish well inside the duration even at several-fold host load.
const FIXED_ITERATIONS: u64 = 500_000;

#[ktstr_test(
    llcs = 1,
    cores = 1,
    threads = 2,
    duration_s = 30,
    watchdog_timeout_s = 60
)]
fn steal_time_fixed_work_cpu_sanity(_ctx: &Ctx) -> Result<AssertResult> {
    let config = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::SpinWait,
        // Fixed WORK, not fixed time: the worker parks after exactly this
        // many iterations, so `cpu_time_ns` is the cost of a constant job —
        // the quantity that must be host-load-invariant.
        park_after_iterations: Some(FIXED_ITERATIONS),
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    // Wait for the quota (the worker parks at N and publishes its terminal
    // count), bounded well under the entry duration.
    let deadline = Instant::now() + Duration::from_secs(25);
    while handle.snapshot_iterations().first().copied().unwrap_or(0) < FIXED_ITERATIONS
        && Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(50));
    }
    let reports = handle.stop_and_collect();

    let mut r = AssertResult::pass();
    let Some(w) = reports.first() else {
        r.record_fail(AssertDetail::new(
            DetailKind::NoProgress,
            "no worker report collected".to_string(),
        ));
        return Ok(r);
    };
    if !w.completed || w.iterations != FIXED_ITERATIONS {
        r.record_fail(AssertDetail::new(
            DetailKind::NoProgress,
            format!(
                "fixed work did not complete: iterations {} of {FIXED_ITERATIONS}, \
                 completed={}",
                w.iterations, w.completed,
            ),
        ));
        return Ok(r);
    }
    if w.cpu_time_ns == 0 {
        r.record_fail(AssertDetail::new(
            DetailKind::Benchmark,
            "cpu_time_ns is 0 for a completed spin quota — CLOCK_THREAD_CPUTIME_ID \
             measurement is broken"
                .to_string(),
        ));
        return Ok(r);
    }
    // Cross-clock agreement: CLOCK_THREAD_CPUTIME_ID vs the kernel's
    // schedstat on-CPU accounting (`sum_exec_runtime` delta over the work
    // loop). Independent code paths, both steal-adjusted; ±20% absorbs the
    // small coverage difference (the thread clock spans slightly more than
    // the work loop) plus sub-ms accounting granularity. Skipped when
    // schedstats are unavailable in the guest (field reads 0).
    if w.schedstat_cpu_time_ns > 0 {
        let a = w.cpu_time_ns as f64;
        let b = w.schedstat_cpu_time_ns as f64;
        let rel = (a - b).abs() / a.max(b);
        if rel > 0.20 {
            r.record_fail(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "thread-CPU clock disagrees with schedstat on-CPU accounting by \
                     {:.0}%: CLOCK_THREAD_CPUTIME_ID={}ns vs sum_exec_runtime \
                     delta={}ns (both must report steal-adjusted on-CPU time)",
                    rel * 100.0,
                    w.cpu_time_ns,
                    w.schedstat_cpu_time_ns,
                ),
            ));
        }
    }
    // Surface the fixed-work CPU cost for the manual idle-vs-loaded A/B:
    // this number must be ~invariant (±20%) across host load while wall
    // time inflates with dilation.
    r.note(format!(
        "fixed work {FIXED_ITERATIONS} iterations: cpu_time_ns={} wall_time_ns={} \
         schedstat_cpu_time_ns={}",
        w.cpu_time_ns, w.wall_time_ns, w.schedstat_cpu_time_ns,
    ));
    // Fold the per-cgroup telemetry (default_checks = NO_OVERRIDES: pure
    // telemetry, no extra gates) so the sidecar carries total_cpu_time_ns /
    // total_iterations — the manual idle-vs-loaded A/B reads its numbers
    // from the sidecar of each run.
    r.merge(Assert::default_checks().assert_cgroup(&reports, None));
    Ok(r)
}

/// The B-side of the steal A/B, self-contained (no host-wide load): the
/// SAME fixed work, but the VM's 2 vCPU threads share a 1-host-CPU budget
/// (`cpu_budget = 1` + `no_perf_mode`) while a second, free-running spin
/// worker keeps the other vCPU busy — so the host time-slices the two vCPU
/// threads against each other (D ≈ 2), stealing CPU from the fixed-work
/// vCPU exactly like an overcommitted host would, without loading anything
/// outside the VM's own budget.
///
/// The steal-exclusion claim under test: the fixed work's `cpu_time_ns`
/// stays ≈ the idle run's value (guest thread CPU-time excludes the stolen
/// slices) while its wall time roughly doubles. The cross-run ±20%
/// comparison against the idle sibling is done by the operator over the two
/// runs' sidecars/info notes (the note carries the fixed-work worker's own
/// numbers). In-test the assertions are budget-agnostic (fixed work
/// completes; cross-clock agreement holds under preemption too): whether
/// contention actually materializes depends on the resolved cpu_budget —
/// under a runner that grants the full budget the run degrades to a second
/// idle-sanity pass, and the measured `host_dilation` in the sidecar tells
/// which case a given run exercised (D≈2 when the 1-CPU budget applied).
#[ktstr_test(
    llcs = 1,
    cores = 1,
    threads = 2,
    duration_s = 60,
    watchdog_timeout_s = 120,
    no_perf_mode,
    cpu_budget = 1
)]
fn steal_time_fixed_work_cpu_invariant_under_overcommit(_ctx: &Ctx) -> Result<AssertResult> {
    // Free-running burner on the sibling vCPU: keeps the second vCPU
    // runnable for the whole window so the 1-CPU host budget is contended
    // and the fixed-work vCPU is continuously preempted.
    let mut burner = WorkloadHandle::spawn(&WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::SpinWait,
        ..Default::default()
    })?;
    burner.start();
    let config = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::SpinWait,
        park_after_iterations: Some(FIXED_ITERATIONS),
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    // ~2x the idle busy phase at a 50% CPU share, bounded under duration.
    let deadline = Instant::now() + Duration::from_secs(50);
    while handle.snapshot_iterations().first().copied().unwrap_or(0) < FIXED_ITERATIONS
        && Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(50));
    }
    let reports = handle.stop_and_collect();
    let burner_reports = burner.stop_and_collect();

    let mut r = AssertResult::pass();
    let Some(w) = reports.first() else {
        r.record_fail(AssertDetail::new(
            DetailKind::NoProgress,
            "no worker report collected".to_string(),
        ));
        return Ok(r);
    };
    if !w.completed || w.iterations != FIXED_ITERATIONS {
        r.record_fail(AssertDetail::new(
            DetailKind::NoProgress,
            format!(
                "fixed work did not complete under overcommit: iterations {} of \
                 {FIXED_ITERATIONS}, completed={}",
                w.iterations, w.completed,
            ),
        ));
        return Ok(r);
    }
    if w.cpu_time_ns == 0 {
        r.record_fail(AssertDetail::new(
            DetailKind::Benchmark,
            "cpu_time_ns is 0 for a completed spin quota".to_string(),
        ));
        return Ok(r);
    }
    // Same cross-clock agreement as the idle sibling — under active host
    // preemption BOTH clocks must still agree on steal-adjusted on-CPU time.
    if w.schedstat_cpu_time_ns > 0 {
        let a = w.cpu_time_ns as f64;
        let b = w.schedstat_cpu_time_ns as f64;
        let rel = (a - b).abs() / a.max(b);
        if rel > 0.20 {
            r.record_fail(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "thread-CPU clock disagrees with schedstat on-CPU accounting under \
                     overcommit by {:.0}%: CLOCK_THREAD_CPUTIME_ID={}ns vs \
                     sum_exec_runtime delta={}ns",
                    rel * 100.0,
                    w.cpu_time_ns,
                    w.schedstat_cpu_time_ns,
                ),
            ));
        }
    }
    r.note(format!(
        "overcommit fixed work {FIXED_ITERATIONS} iterations: cpu_time_ns={} \
         wall_time_ns={} schedstat_cpu_time_ns={} (burner: cpu_time_ns={})",
        w.cpu_time_ns,
        w.wall_time_ns,
        w.schedstat_cpu_time_ns,
        burner_reports.first().map(|b| b.cpu_time_ns).unwrap_or(0),
    ));
    r.merge(Assert::default_checks().assert_cgroup(&reports, None));
    Ok(r)
}
