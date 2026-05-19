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
#[ktstr_test(
    scheduler = PRIMARY_SCHED,
    staged_schedulers = [STAGED_ALT_SCHED],
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 5,
    cleanup_budget_ms = 5000,
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

/// Validates the [`Op::RestartScheduler`](ktstr::scenario::ops::Op::RestartScheduler)
/// hot-restart path: kills the currently-attached scheduler and
/// re-spawns the BOOT scheduler at `/scheduler` + `/sched_args` +
/// `/tmp/sched.log` (the canonical boot paths the wrapper passes
/// to `spawn_scheduler_from_paths`). Successful restart means:
///
/// 1. SCHED_PID atomic reflects a NEW pid post-restart (different
///    from the boot pid). The Op handler reads SCHED_PID before
///    kill, SIGTERMs, waits for sched_ext state to reach
///    `disabled`, spawns, and the spawn helper re-publishes
///    SCHED_PID via [`set_sched_pid`](ktstr::vmm::rust_init::set_sched_pid).
/// 2. The post-restart scheduler successfully binds to sched_ext —
///    verified inside `spawn_scheduler_from_paths` via
///    `poll_scx_attached` against `/sys/kernel/sched_ext/root/ops`.
/// 3. The framework's host-side scheduler liveness monitor does
///    NOT flag the kill as "scheduler died unexpectedly" —
///    `SCHED_EXIT_SUPPRESS` gates the guest's sched_exit_monitor
///    from sending the SchedExit message that would otherwise
///    promote into the run-wide kill flag.
///
/// State-preservation note: scheduler BPF state is intentionally
/// RESET across an Op::RestartScheduler (the kernel teardown +
/// fresh prog load drops per-CPU + arena state). The test
/// validates that the restart MECHANICS work — that the scheduler
/// can be torn down + re-attached cleanly without leaving the
/// guest in a stuck state. Per-state continuity is a separate
/// concern outside Op::RestartScheduler's contract.
#[ktstr_test(
    scheduler = PRIMARY_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 5,
    cleanup_budget_ms = 5000,
)]
fn scheduler_restart_mid_experiment_reattaches_cleanly(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::ops::{HoldSpec, Op, Step, execute_steps};
    let steps = vec![
        // Pre-restart settle window — boot scheduler runs alone.
        Step::new(
            vec![],
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
        // The restart. RestartScheduler kills the boot scheduler
        // via the same SIGTERM + sysrq-S + wait_for_scx_disabled
        // path as ReplaceScheduler, then re-spawns the BOOT
        // scheduler binary at /scheduler + /sched_args. Failure
        // here (kill timeout, scx state stuck, attach failure)
        // bubbles up through the apply_ops error path.
        Step::new(
            vec![Op::restart_scheduler()],
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
        // Post-restart settle window. The freshly-spawned boot
        // scheduler's bind to sched_ext gets verified by the
        // spawn helper's attach poll; this hold gives the live
        // SCHED_PID monitor a window to confirm the post-restart
        // scheduler runs workload-free without panicking.
        Step::new(
            vec![],
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
    ];
    execute_steps(ctx, steps)
}
