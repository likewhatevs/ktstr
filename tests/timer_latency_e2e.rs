//! End-to-end proof that the [`WorkType::TimerLatency`] cyclictest probe runs
//! inside a real VM under scx-ktstr and emits its per-phase timer-latency
//! metric.
//!
//! Each worker sleeps to an ABSOLUTE deadline via
//! `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, next)` and records the wake
//! latency = observed wake − deadline (floored at 0), accumulating
//! `next += interval` so a late wake surfaces AS latency rather than shifting the
//! next period (the coordinated-omission-free measurement cyclictest makes). The
//! per-cycle latency feeds the distinct `timer_latencies_ns` reservoir, which the
//! host reduces to `p99_timer_latency_us` per phase.
//!
//! The gate is the FRAMEWORK INVARIANT: the TimerLatency workload boots, runs its
//! cyclictest loop in the guest, and emits its per-phase `p99_timer_latency_us` --
//! proof the worker dispatch arm ran, the clock_nanosleep loop produced samples,
//! and the timer reservoir reduced through the metric API into a `PhaseBucket`. It
//! does NOT gate on an absolute latency (scx-ktstr is a fixture, not an RT
//! scheduler). A deadlock, panic, or stats-plumbing break would produce no metric
//! and fail the gate.
//!
//! The VM-free CI-runnable companion is `build_phase_slice_full_delta_math`
//! (a plain `#[test]`), which pins the timer reservoir flowing through the
//! per-phase carrier -- so the carrier math stays covered even when this
//! host-gated e2e is skipped. `worktype_coverage_fork_gauntlet_e2e` also drives
//! the TimerLatency dispatch arm, but it is itself a VM-booting host-gated e2e
//! (skipped under the same host gate as this test), so it does NOT preserve
//! dispatch-arm coverage when skipped.

use anyhow::Result;
use ktstr::assert::{AssertResult, Phase};
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::test_support::{Scheduler, SchedulerSpec};

const TIMER_SCX: Scheduler =
    Scheduler::named("timer_scx").binary(SchedulerSpec::Discover("scx-ktstr"));

/// The per-cgroup per-phase metric the TimerLatency run must emit: the
/// 99th-percentile timer wake latency. Read via `phase_cgroup_metric` (it is a
/// generic per-cgroup PerPhase metric on the PhaseCgroupStats carrier, not a
/// pooled engine metric). Its presence proves the cyclictest loop ran end-to-end
/// in the VM and the per-cycle samples reduced through the metric API.
const TIMER_METRIC: &str = "p99_timer_latency_us";

/// post_vm: the framework-invariant gate. The TimerLatency workload produced its
/// per-phase timer latency -- proves the cyclictest loop ran to completion in the
/// guest (no deadlock, no panic, no stats-plumbing break).
fn assert_timer_latency_ran(result: &VmResult) -> Result<()> {
    let phase = Phase::step(0); // the lone hold step (step_index 1)
    // p99_timer_latency_us is a GENERIC per-cgroup PerPhase metric (written to the
    // PhaseCgroupStats carrier by write_carrier_scalars), so it surfaces via
    // phase_cgroup_metric (per-cgroup), NOT phase_metric (which reads the pooled
    // PhaseBucket + a counter fallback — that path is for engine metrics like
    // schbench/taobench that pool to bucket.metrics).
    anyhow::ensure!(
        result
            .phase_cgroup_metric(phase, "timer_probe", TIMER_METRIC)
            .is_some(),
        "TimerLatency phase produced no {TIMER_METRIC} for the timer_probe cgroup -- the cyclictest workload did not run"
    );
    Ok(())
}

// Single 6s hold step; duration_s covers it with margin and watchdog_timeout_s
// lifts the VM hard deadline to cover boot + the hold.
#[ktstr_test(
    scheduler = TIMER_SCX,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 8,
    watchdog_timeout_s = 30,
    cleanup_budget_ms = 5000,
    num_snapshots = 3,
    post_vm = assert_timer_latency_ran,
)]
fn timer_latency_runs_in_vm(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::backdrop::Backdrop;
    use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_scenario};
    use ktstr::workload::WorkType;
    use std::time::Duration;

    // Two cyclictest workers at 1 kHz (cyclictest's default), so a 6 s hold yields
    // ~6000 samples per worker -- well within the reservoir cap, so the emitted
    // p99 is the full-population percentile, not a subsample.
    let backdrop = Backdrop::new().push_cgroup(
        CgroupDef::named("timer_probe")
            .work_type(WorkType::TimerLatency { interval_us: 1000 })
            .workers(2),
    );

    let steps = vec![Step::new(vec![], HoldSpec::fixed(Duration::from_secs(6)))];

    execute_scenario(ctx, backdrop, steps)
}
