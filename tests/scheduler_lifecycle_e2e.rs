//! End-to-end exercise of the scheduler-lifecycle Ops
//! ([`Op::AttachScheduler`](ktstr::scenario::ops::Op::AttachScheduler),
//! [`Op::DetachScheduler`](ktstr::scenario::ops::Op::DetachScheduler),
//! [`Op::RestartScheduler`](ktstr::scenario::ops::Op::RestartScheduler),
//! [`Op::ReplaceScheduler`](ktstr::scenario::ops::Op::ReplaceScheduler))
//! against a real VM. Boots with scx-ktstr as the boot scheduler,
//! stages a second scx-ktstr-backed entry under a distinct name,
//! and runs Op::ReplaceScheduler mid-experiment to validate the
//! swap path end-to-end.
//!
//! The two `Scheduler` constants both resolve to the same scx-ktstr
//! binary via `SchedulerSpec::Discover("scx-ktstr")` — the staging
//! pipeline keys on `Scheduler.name` not on the binary path, so
//! distinct names land at distinct
//! `/staging/schedulers/<name>/scheduler` archive entries even when
//! the resolved binaries match. This keeps the test focused on the
//! lifecycle wire-up (kill prev / spawn new / SCHED_PID swap)
//! rather than requiring a second in-tree scheduler binary.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::test_support::{Scheduler, SchedulerSpec};

const PRIMARY_SCHED: Scheduler =
    Scheduler::named("lifecycle_primary").binary(SchedulerSpec::Discover("scx-ktstr"));

const STAGED_ALT_SCHED: Scheduler =
    Scheduler::named("lifecycle_alt").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Boots with `lifecycle_primary` as the boot scheduler, stages
/// `lifecycle_alt` into `/staging/schedulers/lifecycle_alt/`, and
/// dispatches Op::ReplaceScheduler mid-scenario. A successful
/// dispatch:
///
/// 1. Reads SCHED_PID (boot scheduler's pid set by start_scheduler)
/// 2. SIGTERM → SIGKILL escalation via kill_scheduler_process
/// 3. Clears SCHED_PID via set_sched_pid(0)
/// 4. Spawns the staged binary at the per-name archive path
/// 5. spawn_scheduler_from_paths re-populates SCHED_PID with the
///    new child's pid via the internal Release store
///
/// Any failure mode (missing staged binary, kill failure, spawn
/// failure, attach failure) surfaces as an actionable error
/// through the apply_ops error path, fails the step, and bubbles
/// up to the test verdict.
///
/// The test body succeeds when the Op chain completes without
/// error — the framework's scheduler-attached watchdog +
/// scx-ktstr's `sched_ext_dump` tracepoint confirm the
/// post-replace scheduler bound to sched_ext correctly (same
/// path the boot scheduler exercises).
// NOTE: `ignore = true` because the framework's host-side scheduler
// liveness monitor flags Op::ReplaceScheduler's brief no-scheduler
// window (between the SIGTERM-driven detach and the next scheduler's
// attach) as "scheduler process died unexpectedly during workload".
// The dispatch wire-up itself is verified end-to-end inside the VM —
// the kill path successfully drives the scx state machine to
// `disabled` (per `wait_for_scx_disabled`) and the spawn path
// re-publishes SCHED_PID — but the monitor's interpretation needs a
// lifecycle-aware hook to suppress the false-positive death signal
// during a known swap. That hook is a separate follow-up (depends on
// guest→host swap-pending wire format that doesn't exist today). The
// 5 `scenario::ops::tests::apply_ops_*_scheduler_*` unit tests + the
// `staged_scheduler_log_path_is_per_name_keyed` pin cover the
// dispatch correctness; this VM e2e validates the full path once the
// monitor hook lands. Toggle to `ignore = false` after the
// monitor-lifecycle wire-up commit.
#[ktstr_test(
    scheduler = PRIMARY_SCHED,
    staged_schedulers = [STAGED_ALT_SCHED],
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 5,
    cleanup_budget_ms = 5000,
    ignore = true,
)]
fn scheduler_replace_mid_experiment_swaps_via_staged_pack(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::ops::{HoldSpec, Op, Step, execute_steps};
    let steps = vec![
        // Pre-swap settle window — boot scheduler runs alone so the
        // post-swap PID change is unambiguous.
        Step::new(
            vec![],
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
        // The swap. ReplaceScheduler kills the boot scheduler, spawns
        // STAGED_ALT_SCHED's binary from the staged archive path,
        // re-publishes SCHED_PID to the new child. Failure here
        // (missing binary, kill failure, attach failure) bubbles up
        // through the apply_ops error path.
        Step::new(
            vec![Op::replace_scheduler(&STAGED_ALT_SCHED)],
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
        // Post-swap settle window. The staged scheduler's bind to
        // sched_ext gets verified by the spawn_scheduler_from_paths
        // attach poll; this hold simply gives downstream metric
        // capture a window to confirm the post-swap scheduler ran
        // workload-free without panicking.
        Step::new(
            vec![],
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
    ];
    execute_steps(ctx, steps)
}
