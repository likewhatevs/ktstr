//! End-to-end coverage for the observer-minimal per-phase system/user
//! CPU-time metric: `system_time_ns` / `user_time_ns`.
//!
//! These are read HOST-SIDE from frozen guest `task_struct.utime`/`stime`
//! (+ the thread-group `signal_struct` accumulator for exited threads)
//! at each periodic freeze boundary — ZERO guest work, no taskstats
//! genetlink query. The per-phase value is a per-thread-GROUP delta
//! (each tgid's `thread_group_cputime` at its first vs last appearance
//! in the phase, summed) — see `assert::phase_group_cpu_delta`. This e2e
//! is the offset-and-capture validation the host-side unit tests (which
//! use synthetic enrichments) cannot give: it boots a real scheduler,
//! reads real `task_struct` fields, and confirms the values flow through
//! capture → enrichment → per-phase fold with plausible magnitudes (a
//! wrong BTF offset would read garbage and blow the plausibility ceiling
//! or the > 0 floor).
//!
//! Workload: `YieldHeavy` (a tight `sched_yield(2)` loop) — every
//! iteration is a syscall + a reschedule, so it generates substantial
//! in-kernel (SYSTEM) time, the exact signal the metric targets (a
//! DSQ-spinlock regression shows up as rising system time).
//!
//! Assertions (robust to scheduler placement / host load):
//!   * some phase reports `system_time_ns` > 0 and another (or the
//!     same) reports `user_time_ns` > 0 — the metric flows
//!     end-to-end for both the system and user reads;
//!   * every reported value is finite and below a generous plausibility
//!     ceiling (`wall × cores × 2`) — a wrong offset reading a pointer
//!     or adjacent huge field fails here;
//!   * peak per-phase `system_time_ns` clears a 1 ms floor — proves
//!     real in-kernel time was captured, not a near-zero garbage read
//!     (YieldHeavy on multiple workers for seconds generates far more).

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};
use ktstr::workload::WorkType;

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// VM core count (matches the `cores` attr below) — the plausibility
/// ceiling for per-phase CPU time is `wall × CORES`.
const CORES: u128 = 2;

fn assert_per_phase_cpu_time(result: &VmResult) -> Result<()> {
    anyhow::ensure!(
        result.periodic_fired >= 2,
        "periodic_fired = {} of {} — need >= 2 captures in a phase for a \
         per-thread-group first-vs-last delta; per-phase CPU time cannot \
         be exercised otherwise",
        result.periodic_fired,
        result.periodic_target,
    );
    let buckets = result.phase_buckets();
    anyhow::ensure!(
        !buckets.is_empty(),
        "phase_buckets() empty despite periodic_fired = {}",
        result.periodic_fired,
    );

    // Generous plausibility ceiling: a per-phase CPU-time delta cannot
    // exceed wall-clock * core-count; ×2 slack covers measurement edges.
    // A wrong BTF offset reading a pointer / huge adjacent field blows
    // this immediately.
    let ceiling_ns = (result.duration.as_nanos() * CORES * 2) as f64;

    let mut saw_system = false;
    let mut saw_user = false;
    let mut peak_system = 0.0_f64;
    for b in &buckets {
        for key in ["system_time_ns", "user_time_ns"] {
            if let Some(&v) = b.metrics.get(key) {
                anyhow::ensure!(
                    v.is_finite() && v >= 0.0,
                    "{key} in phase {} (step_index {}) is not a finite \
                     non-negative ns value: {v}",
                    b.label,
                    b.step_index,
                );
                anyhow::ensure!(
                    v <= ceiling_ns,
                    "{key} = {v} ns in phase {} exceeds the plausibility \
                     ceiling {ceiling_ns} ns (wall {} ns × {CORES} cores × 2) \
                     — a wrong task_struct BTF offset would read garbage like \
                     this",
                    b.label,
                    result.duration.as_nanos(),
                );
            }
        }
        if let Some(&sys) = b.metrics.get("system_time_ns") {
            if sys > 0.0 {
                saw_system = true;
            }
            peak_system = peak_system.max(sys);
        }
        if b.metrics.get("user_time_ns").is_some_and(|&u| u > 0.0) {
            saw_user = true;
        }
    }

    anyhow::ensure!(
        saw_system,
        "no phase reported system_time_ns > 0 — per-phase SYSTEM CPU \
         time is not flowing end-to-end (frozen task_struct.stime capture / \
         signal fold / per-phase aggregation broken). buckets = {:?}",
        buckets
            .iter()
            .map(|b| (b.step_index, b.metrics.get("system_time_ns").copied()))
            .collect::<Vec<_>>(),
    );
    anyhow::ensure!(
        saw_user,
        "no phase reported user_time_ns > 0 — task_struct.utime \
         capture not flowing",
    );
    anyhow::ensure!(
        peak_system > 1_000_000.0,
        "peak per-phase system_time_ns = {peak_system} ns is below the \
         1 ms floor — a YieldHeavy (sched_yield loop) workload on multiple \
         workers must generate substantial in-kernel time; a near-zero peak \
         means the stime read is wrong or capturing nothing",
    );
    Ok(())
}

/// Boots scx-ktstr on a 2-core VM running a YieldHeavy workload (heavy
/// SYSTEM time) across periodic captures, then asserts per-phase
/// system/user CPU time flows host-side with plausible magnitudes.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 15,
    watchdog_timeout_s = 25,
    num_snapshots = 6,
    auto_repro = false,
    post_vm = assert_per_phase_cpu_time,
)]
fn per_phase_cpu_time_reads_frozen_task_struct_e2e(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_cputime")
                .workers(4)
                .work_type(WorkType::YieldHeavy),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}
