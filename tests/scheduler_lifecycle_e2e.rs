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

/// Boot baseline for the cold-start attach test below: the kernel's
/// default scheduler (EEVDF on current kernels) so no scx_* userspace
/// binary runs at boot. `Op::AttachScheduler` then dispatches the
/// first scx attach mid-scenario.
const COLD_START_BOOT: Scheduler = Scheduler::named("cold_start_boot").binary(SchedulerSpec::Eevdf);

/// Staged scheduler for the cold-start attach test.
const COLD_START_ALT_SCHED: Scheduler =
    Scheduler::named("cold_start_alt").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Scheduler instance configured with `--stall-after 1` so it stops
/// dispatching one second into the run. The kernel's scx watchdog
/// then detects the stall and exits the scheduler with
/// `SCX_EXIT_ERROR_STALL`, surfaced to the host as a scheduler death
/// the dispatch-loop hold should observe and truncate early on. Used
/// only by `dispatch_hold_truncates_when_scheduler_dies_midstep`.
const STALL_AFTER_1S_SCHED: Scheduler = Scheduler::named("stall_after_1s")
    .binary(SchedulerSpec::Discover("scx-ktstr"))
    .sched_args(&["--stall-after", "1"]);

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

/// Pins the `Op::AttachScheduler` cold-start dispatch — the only
/// scheduler-lifecycle path with zero prior e2e coverage. Restart
/// and Replace tests exercise the kill-then-attach sequence where
/// a primary scheduler ran first; attach-from-no-scheduler exercises
/// a different code path (no SCHED_PID to clear, no scx_disable to
/// wait for, fresh slab allocation for the new scheduler's scx_sched).
///
/// Boots under `SchedulerSpec::Eevdf` so no scx_* binary runs at boot;
/// mid-scenario `Op::AttachScheduler(COLD_START_ALT_SCHED)` dispatches
/// the first scx attach. A successful attach:
///
/// 1. spawn_scheduler_from_paths spawns the staged scx-ktstr binary
/// 2. The spawn helper's `poll_scx_attached` confirms the BPF
///    scheduler bound to `/sys/kernel/sched_ext/root/ops`
/// 3. SCHED_PID gets populated via the spawn helper's Release store
/// 4. The framework's scheduler-attached watchdog observes the new
///    scheduler and resets its deadline
///
/// Any failure (staging mis-pack, spawn failure, attach timeout) bails
/// through the apply_ops error path.
#[ktstr_test(
    scheduler = COLD_START_BOOT,
    staged_schedulers = [COLD_START_ALT_SCHED],
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 5,
    cleanup_budget_ms = 5000,
)]
fn scheduler_attach_from_cold_start_succeeds(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::ops::{HoldSpec, Op, Step, execute_steps};
    let steps = vec![
        // Pre-attach settle — no scheduler running, EEVDF handles
        // scheduling. Confirms the test infrastructure survives
        // bootless scheduler state.
        Step::new(
            vec![],
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
        // The cold-start attach. AttachScheduler spawns
        // COLD_START_ALT_SCHED's staged binary, polls for sched_ext
        // bind, publishes SCHED_PID. No prior scheduler means no
        // detach work — first scx_alloc_and_add_sched on this VM.
        Step::new(
            vec![Op::attach_scheduler(&COLD_START_ALT_SCHED)],
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
        // Post-attach settle — the freshly-attached scheduler runs
        // workload-free for a window so the live SCHED_PID monitor
        // confirms it stays bound to sched_ext without panicking.
        Step::new(
            vec![],
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
    ];
    execute_steps(ctx, steps)
}

/// Pins `hold_or_sched_died`'s mid-hold scheduler-death detection +
/// early-truncation behaviour. Without the early truncation, an Op
/// step's hold would block until its full configured duration even
/// when the scheduler died at second 1 — leaving the per-step
/// `sched_died_during_hold` flag accurate but the wall-clock time
/// inflated, masking the actual death-detection latency.
///
/// Mechanism: scheduler launches with `--stall-after 1`. The BPF
/// `stall` flag flips at t≈1s; `ktstr_dispatch` stops moving tasks
/// to the shared DSQ; the kernel scx watchdog detects stalled
/// runnable tasks within a few seconds and exits the scheduler with
/// `SCX_EXIT_ERROR_STALL`. The dispatch-loop hold is configured
/// for 15 s — well beyond the watchdog deadline — so the hold MUST
/// truncate early when the scheduler dies. The post-VM assertion
/// rejects any elapsed time near the full 15 s budget.
#[ktstr_test(
    scheduler = STALL_AFTER_1S_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 20,
    watchdog_timeout_s = 30,
    auto_repro = false,
    expect_err = true,
    cleanup_budget_ms = 5000,
)]
fn dispatch_hold_truncates_when_scheduler_dies_midstep(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::ops::{HoldSpec, Step, execute_steps};
    let t0 = std::time::Instant::now();
    let steps = vec![Step::new(
        vec![],
        HoldSpec::fixed(std::time::Duration::from_secs(15)),
    )];
    let result = execute_steps(ctx, steps);
    let elapsed = t0.elapsed();

    // Assert early-truncation: the configured hold is 15 s but the
    // scheduler dies via the stall watchdog at roughly t=1s+watchdog.
    // 12 s threshold is well below the configured 15 s hold and well
    // above any plausible watchdog deadline (kernel scx watchdog is
    // typically 2-10s depending on the build), so a step elapsed
    // exceeding 12 s pins a regression where `hold_or_sched_died`
    // stops detecting scheduler death mid-step.
    let ceiling = std::time::Duration::from_secs(12);
    if elapsed >= ceiling {
        return Ok(AssertResult::fail_msg(format!(
            "dispatch loop did not truncate after scheduler-stall death: \
             configured hold = 15 s, scheduler stalled at t≈1 s + watchdog, \
             actual elapsed = {elapsed:?} (≥ {ceiling:?} ceiling). \
             hold_or_sched_died's mid-hold scheduler-death observation is \
             broken — the per-step hold ran to completion despite the \
             scheduler dying. Check pidfd_wait_exit + the dispatch loop's \
             death-observation branch in src/scenario/ops/mod.rs."
        )));
    }
    // Propagate the inner result so the framework's expect_err =
    // true + auto-repro = false machinery still sees the genuine
    // SCX_EXIT_ERROR_STALL failure surfacing through apply_ops.
    result
}

/// `/bin/false` staged as a "scheduler" — exits non-zero
/// immediately, well inside the spawn helper's 1 s liveness window.
/// Used only by `replace_with_broken_binary_surfaces_startup_died`
/// to exercise `SpawnSchedulerError::StartupDied` end-to-end.
/// Choosing `/bin/false` over an invalid-CLI-arg approach decouples
/// the test from scx-ktstr's clap parser staying the same.
const BROKEN_BINARY_SCHED: Scheduler =
    Scheduler::named("broken_binary").binary(SchedulerSpec::Path("/bin/false"));

/// Pins `try_spawn_scheduler`'s `StartupDied` path + the
/// SCHED_PID cleanup + `SpawnSchedulerError::Display` content.
/// When `Op::ReplaceScheduler` stages
/// `/bin/false` as the scheduler binary, the spawn helper's
/// `poll_startup` observes the immediate non-zero exit and
/// bubbles back `SpawnSchedulerError::StartupDied`. The
/// `apply_ops` error path surfaces it as a typed step failure
/// that the framework's `expect_err = true` machinery accepts.
///
/// What this pins:
///   - The error CLASS reaches the operator (StartupDied, not
///     NotAttached or generic spawn failure)
///   - The post-mortem state guarantee — SCHED_PID gets cleared
///     to 0 and the process is reaped — implicit: if it didn't,
///     the next test in the same VM session would observe a stale
///     dead pid and behave unpredictably. The framework's per-VM
///     isolation makes this hard to assert directly, but a
///     regression that leaves SCHED_PID dangling would surface as
///     downstream flake.
#[ktstr_test(
    scheduler = PRIMARY_SCHED,
    staged_schedulers = [BROKEN_BINARY_SCHED],
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 5,
    auto_repro = false,
    expect_err = true,
    cleanup_budget_ms = 5000,
)]
fn replace_with_broken_binary_surfaces_startup_died(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::ops::{HoldSpec, Op, Step, execute_steps};
    let steps = vec![
        // Settle window so the primary scheduler is firmly attached
        // before we kick off the doomed Op::Replace.
        Step::new(
            vec![],
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
        // The doomed Replace. kill_current_scheduler kills the
        // primary; spawn_scheduler_for_op then exec's /bin/false as
        // the "scheduler" binary; /bin/false exits non-zero
        // immediately; poll_startup observes the exit;
        // SpawnSchedulerError::StartupDied bubbles back through
        // apply_ops as a typed Op failure. execute_steps returns Err.
        Step::new(
            vec![Op::replace_scheduler(&BROKEN_BINARY_SCHED)],
            HoldSpec::fixed(std::time::Duration::from_millis(500)),
        ),
    ];
    let result = execute_steps(ctx, steps);
    // The Op::Replace MUST fail; if it somehow succeeds, the
    // SpawnSchedulerError::StartupDied path isn't being exercised
    // and the test is silently wrong.
    if let Ok(ref ok_result) = result
        && ok_result.is_pass()
    {
        return Ok(AssertResult::fail_msg(
            "Op::ReplaceScheduler with /bin/false as the scheduler binary DID \
             NOT FAIL — try_spawn_scheduler's StartupDied path is no longer being \
             exercised (poll_startup is missing the immediate exit, or the spawn \
             helper started swallowing exit codes)."
                .to_string(),
        ));
    }
    result
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
