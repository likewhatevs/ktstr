//! Scheduler-process lifecycle: reboot, sched-pid/monitor state, SIGCHLD handling, kill-with-grace.
//!
//! Split from rust_init.rs; the shared consts/statics/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;

/// Reboot immediately. Used for fatal init errors and normal shutdown.
pub(crate) fn force_reboot() -> ! {
    let _ = reboot(RebootMode::RB_AUTOBOOT);
    // The kernel is rebooting — no event will ever fire. Park the
    // thread forever; this is cheaper than a sleep loop because
    // `park` blocks in the kernel without a wake-up timer attached.
    // No `unpark` call exists in this path; the process dies when
    // the reboot syscall completes.
    loop {
        std::thread::park();
    }
}

/// Live identity of the currently-attached scheduler, parallel to
/// [`SCHED_PID`]'s pid side-channel. `null` means "no scheduler
/// attached" — the initial value at process start and the post-
/// `Op::DetachScheduler` state. Non-null points at a
/// `&'static SchedulerSpec` (the `binary` field of the
/// `&'static Scheduler` the Op carries), so consumers can read
/// `has_bpf_scheduler()` / `has_active_scheduling()` against the
/// LIVE identity rather than the boot-time `entry.scheduler`
/// descriptor that goes stale after `Op::ReplaceScheduler` swaps
/// the attached binary mid-scenario.
///
/// Storage: `AtomicPtr<SchedulerSpec>` because the value is a
/// reference to immutable static data (every `Scheduler` const
/// declared via `declare_scheduler!` lives in `.rodata` for the
/// lifetime of the process); the producer stores the `&'static
/// SchedulerSpec` re-cast to `*mut`, the consumer reads back as
/// `*const` and dereferences under the SAFETY argument that the
/// pointer either originated from a `&'static SchedulerSpec` (so
/// the `'static` lifetime is the entire process) or is `null`
/// (filtered by the wrapper). `*mut` storage is the only Atomic*
/// type the standard library exposes for raw pointer values — the
/// `*mut` vs `*const` is a Rust-level type distinction, not a
/// kernel-level mutability claim; the pointed-to data is never
/// mutated through this pointer.
///
/// `Acquire`/`Release` ordering pairs with [`SCHED_PID`]'s — the
/// two side channels co-publish a single logical scheduler-attach
/// event, and a reader that observes the new pid via
/// [`sched_pid`] also observes the new scheduler identity via
/// [`current_scheduler`].
static CURRENT_SCHEDULER: std::sync::atomic::AtomicPtr<crate::test_support::SchedulerSpec> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Active [`SchedExitStop`] handle for the currently-running
/// scheduler's exit monitor. The boot path installs the initial
/// handle here via [`install_initial_sched_exit_monitor`]; the
/// scheduler-lifecycle Op dispatcher swaps it out via
/// [`stop_sched_exit_monitor`] + [`restart_sched_exit_monitor_with_log`]
/// so each post-Op scheduler PID gets its own monitor watching it.
///
/// Mutex (not Atomic) because [`SchedExitStop`] is move-only —
/// `stop_and_join` consumes it. `Option` because Op::DetachScheduler
/// leaves no scheduler attached, so the slot is empty between
/// detach and the next attach.
static SCHED_EXIT_MONITOR_SLOT: OnceLock<std::sync::Mutex<Option<SchedExitStop>>> = OnceLock::new();

/// Boot-captured context that
/// [`restart_sched_exit_monitor_with_log`] needs to re-supply when
/// it spawns a fresh monitor against the post-Op scheduler PID.
/// `suppress_sched_log` + `probe_output_done` are determined at boot
/// (based on whether the probe stack is active) and don't change
/// across Op dispatches — capturing once at install time keeps
/// the restart helper signature minimal.
struct SchedExitMonitorBootCtx {
    suppress_sched_log: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
}

static SCHED_EXIT_MONITOR_BOOT_CTX: OnceLock<SchedExitMonitorBootCtx> = OnceLock::new();

/// Install the boot-time scheduler-exit monitor handle and capture
/// the dispatch context [`restart_sched_exit_monitor_with_log`]
/// needs to spawn replacement monitors. Called once at boot
/// after [`start_sched_exit_monitor`] returns.
///
/// `boot_stop` is `None` when no scheduler is configured at boot; the
/// slot stays empty and the first Op::AttachScheduler dispatch populates
/// it via [`restart_sched_exit_monitor_with_log`].
pub(crate) fn install_initial_sched_exit_monitor(
    boot_stop: Option<SchedExitStop>,
    suppress_sched_log: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
) {
    let slot = SCHED_EXIT_MONITOR_SLOT.get_or_init(|| std::sync::Mutex::new(None));
    *slot.lock().unwrap() = boot_stop;
    let _ = SCHED_EXIT_MONITOR_BOOT_CTX.set(SchedExitMonitorBootCtx {
        suppress_sched_log,
        probe_output_done,
    });
}

/// Stop the currently-installed scheduler-exit monitor (if any).
/// The scheduler-lifecycle Op handler calls this BEFORE SIGTERM-ing
/// the scheduler so the monitor thread exits cleanly without
/// sending the `MSG_TYPE_SCHED_EXIT` message that the host's
/// freeze coordinator would otherwise promote into the run-wide
/// kill flag (per `src/vmm/freeze_coord/dispatch.rs` SchedExit
/// arm). Returns true when the monitor observed scheduler exit (including
/// the race where stop suppressed the frame) or the monitor failed, so
/// verifier cleanup can fail closed.
/// Idempotent — returns false when the slot is already empty.
pub(crate) fn stop_sched_exit_monitor() -> bool {
    let Some(slot) = SCHED_EXIT_MONITOR_SLOT.get() else {
        return false;
    };
    let prev = slot.lock().unwrap().take();
    if let Some(stop) = prev {
        stop.stop_and_join()
    } else {
        false
    }
}

/// Returns true iff no scheduler-exit monitor is currently installed.
/// Used by the scenario-Op dispatch layer in `kill_current_scheduler`
/// to `debug_assert!` that `stop_sched_exit_monitor` properly cleared
/// the slot before the subsequent spawn restarts the monitor. The
/// `Op::AttachScheduler` path legitimately bypasses the kill helper
/// (no prior scheduler to stop) and the defensive `take()` in
/// [`restart_sched_exit_monitor_with_log`] handles that path's
/// possibly-non-empty entry — so the invariant is "after kill, slot
/// is empty," not "always empty before restart." Briefly locks the
/// slot mutex; release builds where the assertion is a no-op still
/// pay the lock cost, which is negligible vs the surrounding
/// procfs writes + signal delivery + polling the dispatch site is
/// already doing.
pub(crate) fn sched_exit_monitor_slot_is_empty() -> bool {
    let Some(slot) = SCHED_EXIT_MONITOR_SLOT.get() else {
        return true;
    };
    slot.lock().unwrap().is_none()
}

/// Spawn a fresh scheduler-exit monitor for the live SCHED_PID and install it
/// into the slot. The Op handler hands in the exact pidfd opened for the
/// scheduler at spawn, closing both the post-attach `pidfd_open` gap and
/// numeric-pid reuse ambiguity. `log_path` is the per-spawn log file path —
/// all three lifecycle Ops (Attach, Replace, Restart) pass the seq-suffixed
/// path from `staged_scheduler_log_path`.
///
/// Uses the boot-captured `suppress_sched_log` + `probe_output_done`
/// so the new monitor behaves identically to the boot monitor. Missing boot
/// context or SCHED_PID is a synchronous error: lifecycle-Op success can
/// never be reported while the replacement scheduler is unmonitored.
pub(crate) fn restart_sched_exit_monitor_with_log(
    pidfd: OwnedFd,
    log_path: Option<&str>,
) -> std::io::Result<()> {
    let ctx = SCHED_EXIT_MONITOR_BOOT_CTX.get().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "scheduler-exit monitor boot context is not installed",
        )
    })?;
    let pid = sched_pid().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "cannot install scheduler-exit monitor: SCHED_PID is unset",
        )
    })?;
    let pid = u32::try_from(pid).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("cannot install scheduler-exit monitor for invalid pid {pid}"),
        )
    })?;
    let slot = SCHED_EXIT_MONITOR_SLOT.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = slot.lock().unwrap();
    // Defensive: if the Op handler skipped stop_sched_exit_monitor
    // for any reason, stop_and_join the stale handle before
    // installing the new one. The take() leaves the slot empty
    // for the duration of start_sched_exit_monitor — readers in
    // that window observe "no monitor", which is correct since
    // the new monitor hasn't been spawned yet.
    if let Some(prev) = guard.take() {
        let _ = prev.stop_and_join();
    }
    let stop = start_sched_exit_monitor(
        pid,
        pidfd,
        log_path,
        ctx.suppress_sched_log.clone(),
        ctx.probe_output_done.clone(),
    )?;
    *guard = Some(stop);
    Ok(())
}

/// Read the scheduler PID published by [`start_scheduler`]. Returns
/// `None` when the scheduler has not been spawned yet (the atomic
/// reads as `0`, the sentinel for "unset"). `Acquire` synchronises
/// against the producer's `Release` store so any side effects
/// `start_scheduler` performed before the publish are visible to the
/// reader.
pub(crate) fn sched_pid() -> Option<libc::pid_t> {
    let v = SCHED_PID.load(Ordering::Acquire);
    if v == 0 { None } else { Some(v) }
}

/// Publish `pid` to the [`SCHED_PID`] side channel. Used by the
/// scheduler-lifecycle Op dispatch on the guest to swap the live PID
/// across Detach (`pid = 0`) / Attach (`pid = new child`) /
/// Replace (`pid = swap`) transitions. The boot path
/// ([`spawn_scheduler_from_paths`], via `try_spawn_scheduler`)
/// bypasses this helper and stores the freshly-spawned
/// `child.id()` into [`SCHED_PID`] directly.
///
/// `Release` ordering pairs with the `Acquire` load in
/// [`sched_pid`]; the writer's side effects (Op log emit, prior
/// kill) are visible to the next reader.
pub(crate) fn set_sched_pid(pid: libc::pid_t) {
    SCHED_PID.store(pid, Ordering::Release);
}

/// Read the live scheduler identity published by the dispatch
/// arms of `Op::AttachScheduler` / `Op::ReplaceScheduler` (the
/// matching `set_current_scheduler` call site lives in
/// `src/scenario/ops/dispatch.rs`). Returns `None` when no scheduler
/// is currently attached — the pre-attach state at process start
/// and the post-`Op::DetachScheduler` state.
///
/// `Acquire` ordering synchronises against the producer's
/// `Release` store so any side effects the dispatch path
/// performed before the publish are visible to the reader.
///
/// The returned reference inherits the `'static` lifetime of the
/// stored `&'static SchedulerSpec` — every `Scheduler` declared
/// via `declare_scheduler!` lives in `.rodata` for the process
/// lifetime, and the producer always stores a reference into that
/// region.
pub fn current_scheduler() -> Option<&'static crate::test_support::SchedulerSpec> {
    let ptr = CURRENT_SCHEDULER.load(Ordering::Acquire);
    if ptr.is_null() {
        None
    } else {
        // SAFETY: every non-null value stored in CURRENT_SCHEDULER
        // came from a `&'static SchedulerSpec` re-cast to `*mut`
        // via `set_current_scheduler`; the pointee is in `.rodata`
        // and outlives the process, so the `&'static` lifetime is
        // sound. The `*mut` → `*const` conversion is a no-op at
        // runtime — only required because `AtomicPtr<T>` exposes
        // `*mut T` storage even when the program never mutates
        // through the pointer.
        Some(unsafe { &*(ptr as *const _) })
    }
}

/// Publish `scheduler` as the currently-attached scheduler, or
/// clear the slot when `None`. Called by the
/// `Op::AttachScheduler` / `Op::ReplaceScheduler` /
/// `Op::DetachScheduler` dispatch arms in
/// `src/scenario/ops/dispatch.rs` immediately after the corresponding
/// pid change so the two side channels (pid + identity) stay
/// co-published.
///
/// `Release` ordering pairs with the `Acquire` load in
/// [`current_scheduler`].
pub(crate) fn set_current_scheduler(
    scheduler: Option<&'static crate::test_support::SchedulerSpec>,
) {
    let ptr = match scheduler {
        Some(r) => r as *const _ as *mut _,
        None => std::ptr::null_mut(),
    };
    CURRENT_SCHEDULER.store(ptr, Ordering::Release);
}

/// RAII guard that flips SIGCHLD to a target disposition on
/// construction and restores the previous handler on drop. Used by
/// [`with_sigchld_default`] so a panic inside the closure cannot
/// leak `SIG_DFL` into the rest of the guest's lifetime — Drop
/// runs even on unwind.
///
/// `libc::signal` returns the previous handler on every call, so
/// the snapshot we capture in `install` is the authoritative value
/// to restore in `Drop`. Re-installing the snapshot makes the
/// guard idempotent across nested calls (an outer guard's restore
/// observes the inner guard's restore as a no-op rebind to the
/// same handler).
struct SigchldDispositionGuard {
    prev: libc::sighandler_t,
}

impl SigchldDispositionGuard {
    /// Install `handler` as the SIGCHLD disposition and capture
    /// the previous handler for restoration on drop.
    ///
    /// SAFETY: signal disposition is a process-wide property. PID
    /// 1 owns the disposition for the whole guest, so no other
    /// thread can race the signal install. `libc::signal` is
    /// async-signal-safe per POSIX.1-2008 TC2.
    ///
    /// # Panics
    ///
    /// Panics if `libc::signal` returns `SIG_ERR` — the libc
    /// failure indicator (`!0 as sighandler_t`) for an invalid
    /// signal number or other install failure. Without the check,
    /// `SIG_ERR` would be captured into `prev` as if it were a
    /// valid handler, and Drop would then attempt to install
    /// `SIG_ERR` (which the kernel rejects with `EINVAL`,
    /// surfacing as a separate `SIG_ERR` return that the no-check
    /// Drop also drops on the floor — silently leaking the
    /// install error). For SIGCHLD the failure path is
    /// implausible in practice (the signal number is valid and
    /// `SIG_DFL`/`SIG_IGN` are always-installable handlers), but
    /// the library invariant is general — `signal(2)` returning
    /// `SIG_ERR` is a programming error, not a runtime condition,
    /// so panicking is the right discipline.
    fn install(handler: libc::sighandler_t) -> Self {
        let prev = unsafe { libc::signal(libc::SIGCHLD, handler) };
        assert_ne!(
            prev,
            libc::SIG_ERR,
            "failed to install SIGCHLD handler — libc::signal returned SIG_ERR; \
             check signum / handler validity",
        );
        Self { prev }
    }
}

impl Drop for SigchldDispositionGuard {
    fn drop(&mut self) {
        // SAFETY: `self.prev` was returned by an earlier
        // `libc::signal` call on the same signal number, so
        // re-installing it is the documented restore pattern. The
        // `Drop` runs on both the normal-return and panic-unwind
        // paths, so a panic inside the protected closure cannot
        // leak the temporary disposition into the rest of the
        // process.
        unsafe {
            libc::signal(libc::SIGCHLD, self.prev);
        }
    }
}

/// Run `f` with SIGCHLD temporarily restored to `SIG_DFL` so the
/// kernel does not auto-reap any child spawned inside the closure.
/// `Command::status()` calls `waitpid(2)`, which returns `ECHILD`
/// when SIGCHLD is `SIG_IGN` (the default installed by
/// [`ktstr_guest_init`] for zombie prevention) — losing the real
/// exit status. Restoring `SIG_DFL` for the closure's lifetime
/// re-enables `waitpid` reaping; the post-closure restore puts
/// the previous disposition back so subsequent guest children
/// continue to be auto-reaped without leaking zombies.
///
/// Mirrors the inline save/restore pattern formerly open-coded at
/// the [`ktstr_guest_init`] shell `--exec` site (now also routed
/// through this helper). Both call sites share the same
/// SIGCHLD-vs-`waitpid` hazard; centralising the helper prevents
/// drift between the two implementations.
///
/// Restore is panic-safe via [`SigchldDispositionGuard`]: a panic
/// in `f` runs the guard's `Drop`, which re-installs the previous
/// SIGCHLD handler before unwinding past the helper boundary.
/// Without the guard, a panicking child-spawn site would leak
/// `SIG_DFL` into the rest of the guest, breaking PID 1's zombie
/// reaping for every subsequent fork.
///
/// The closure must reap every child it spawns before returning.
/// Leaving an unreaped child at the boundary where `SIG_IGN` is
/// restored would orphan the zombie until the next reaper cycle.
/// `Command::status()` waits synchronously, so the typical caller
/// satisfies this invariant by construction.
pub(crate) fn with_sigchld_default<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = SigchldDispositionGuard::install(libc::SIG_DFL);
    f()
}

/// Whether `/proc/{pid}` exists. Used as a `waitpid`-free liveness
/// probe: under SIGCHLD `SIG_IGN` the kernel auto-reaps children, so
/// `waitpid` returns `ECHILD` even when the child exited cleanly.
/// `/proc/{pid}` removal is signal-disposition-independent — the
/// directory disappears the moment the kernel finishes
/// `release_task` for the pid (see kernel/exit.c
/// `release_task` → `proc_flush_pid`), regardless of whether
/// `waitpid` ever ran.
///
/// Returns `true` when `/proc/{pid}` exists (process alive or
/// pre-reap), `false` when it does not (process exited and the
/// kernel has dropped the procfs entry).
/// SIGCHLD = SIG_IGN-safe liveness probe via procfs. The guest init
/// installs `SIGCHLD = SIG_IGN` process-wide (see
/// [`with_sigchld_default`] doc) so the kernel auto-reaps children
/// without explicit `waitpid`. Under that disposition `waitpid`
/// returns `ECHILD` even on a clean exit, so a `Command::status` /
/// `Child::wait` is the wrong tool for "is this pid still running".
///
/// `/proc/{pid}` removal is signal-disposition-independent: the
/// directory disappears the moment the kernel finishes `release_task`
/// for the pid (see kernel/exit.c `release_task` →
/// `proc_flush_pid`), regardless of how SIGCHLD is handled. Polling
/// `/proc/{pid}` therefore observes the real exit on every code path
/// where SIGCHLD might be ignored. Returns `true` when `/proc/{pid}`
/// exists (process alive or pre-reap), `false` when it does not
/// (process exited and the kernel has dropped the procfs entry).
pub(crate) fn proc_pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// Outcome reported by a successful [`kill_scheduler_process`] call.
/// Three variants because the operator-visible signal (caller-side
/// logging, sidecar event) differs by how the child responded:
/// already-gone callers know there was nothing to do; sigterm-graceful
/// exit is the scx-convention happy path; sigkill-escalation is the
/// notable case (the scheduler binary either ignored SIGTERM or its
/// userspace signal handler ran too slow against the grace window).
//
// `#[allow(dead_code)]` because the helper has no production caller.
// The Op::DetachScheduler / Op::RestartScheduler /
// Op::ReplaceScheduler dispatchers deliberately bypass it: they
// route kills through `kill_current_scheduler`, which uses direct
// `libc::kill(SIGTERM)` + `wait_for_scx_disabled` because this
// helper's strict /proc-absence verification can fire
// `StillAliveAfterSigkill` when the scheduler's exit blocks on BPF
// detach (see src/scenario/ops/dispatch.rs). Tests in this module
// exercise every variant + the InvalidPid error path, so the
// helper is verified-correct.
#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum KillSchedulerOutcome {
    /// `pid` was not alive when the call started — `/proc/{pid}`
    /// already absent. Treated as success because lifecycle ops
    /// (Op::DetachScheduler) are idempotent: detaching when nothing
    /// is running is a no-op, not an error.
    AlreadyExited,
    /// SIGTERM landed and the scheduler exited cleanly within the
    /// grace window. The scx convention (per scx_simple.c
    /// `sigint_handler` at L37-39 of the upstream
    /// tools/sched_ext/scx_simple.c) is to catch SIGTERM, drop the
    /// BPF skeleton, run scx_disable_workfn via the destructor path,
    /// and exit. This is the operator-visible happy path.
    ExitedAfterSigterm,
    /// SIGTERM did not produce an exit within the grace window;
    /// SIGKILL was sent and the process reaped. The scheduler
    /// either failed to install its SIGTERM handler, was stuck in
    /// uninterruptible kernel state, or its handler took longer
    /// than the grace allowed. Operators may want to inspect the
    /// scheduler binary's signal-handler implementation when this
    /// fires.
    EscalatedToSigkill,
}

/// Failure modes for [`kill_scheduler_process`]. Both indicate the
/// caller-supplied invariant (a kill-able pid) was violated or the
/// kernel refused to honor a SIGKILL — neither is recoverable at the
/// call site, but both carry distinct operator diagnostics.
#[allow(dead_code)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum KillSchedulerError {
    /// `pid` was not a positive pid_t value. POSIX `kill(2)` reserves
    /// 0 (the caller's process group) and negative values (signal a
    /// process group) for special semantics — the scheduler-lifecycle
    /// call site only ever wants to signal a specific known pid, so a
    /// non-positive value is a programming error in the caller.
    InvalidPid,
    /// SIGKILL was sent but `/proc/{pid}` was still present after the
    /// post-SIGKILL grace window. POSIX guarantees SIGKILL cannot be
    /// caught or ignored, so this indicates either kernel-side stall
    /// (rare uninterruptible D-state) or a process that re-spawned a
    /// new pid before procfs cleaned up — neither plausible in the
    /// scheduler-binary case but reported distinctly so the caller
    /// can surface a "scheduler refused to die" diagnostic rather
    /// than silently believing the detach succeeded.
    StillAliveAfterSigkill,
}

/// Send SIGTERM to `pid`, wait up to `sigterm_grace` for the process
/// to exit (observed via `/proc/{pid}` removal), then escalate to
/// SIGKILL if the polite shutdown did not land. Returns the variant
/// that describes how the kill resolved.
///
/// # Why procfs polling instead of `waitpid`
///
/// The guest init installs SIGCHLD = SIG_IGN globally so PID 1 does
/// not have to reap every zombie (see [`with_sigchld_default`] and
/// the doc on [`proc_pid_alive`]). Under that disposition the kernel
/// auto-reaps children before `waitpid` runs, so `waitpid` returns
/// `ECHILD` even on a clean exit. `/proc/{pid}` removal is
/// signal-disposition-independent: the directory disappears the
/// moment the kernel runs `release_task` for the pid, regardless of
/// how SIGCHLD is handled. Polling `/proc/{pid}` therefore observes
/// the real exit on every code path where SIGCHLD might be ignored.
///
/// # Why SIGTERM first, SIGKILL fallback
///
/// scx schedulers (per the upstream
/// `tools/sched_ext/scx_simple.c:71-72` convention) install one
/// shared signal handler for SIGINT + SIGTERM: setting an exit-
/// request flag that the scheduler's main loop polls, then dropping
/// the BPF skeleton which triggers the kernel's `scx_disable_workfn`
/// path. SIGTERM is the safe shutdown signal — every well-behaved
/// scx scheduler honors it. SIGKILL bypasses the userspace handler
/// (final-log-flush, graceful destructor) but the kernel still
/// observes the BPF program refcount drop and runs the disable path,
/// so the kernel-side scheduler state cleans up regardless. SIGKILL
/// after a bounded SIGTERM grace is the strict-correctness fallback
/// for a scheduler binary that has no SIGTERM handler installed or
/// took longer than `sigterm_grace` to exit.
///
/// # Pid lifecycle semantic
///
/// This function does NOT mutate [`SCHED_PID`]. The
/// scheduler-lifecycle dispatcher owns that side channel and is
/// responsible for storing 0 after a successful detach so subsequent
/// liveness checks (`sched_pid()` readers) short-circuit. Keeping
/// the kill helper generic (no implicit singleton-pid assumption)
/// lets unit tests exercise it against any spawned child pid.
///
/// # Wait mechanism
///
/// Absence is detected via the kernel-evented pidfd path in
/// [`poll_proc_pid_absent`] (`pidfd_wait_exit`), not a sleep-poll
/// loop; the `interval` argument passed there is currently unused.
/// The post-SIGKILL grace is the module-level [`POST_SIGKILL_GRACE`]
/// const (see that const's doc for the 200ms-vs-magic-number
/// rationale).
#[allow(dead_code)] // production callers (Op::*Scheduler dispatch) wire up in follow-up work
pub(crate) fn kill_scheduler_process(
    pid: libc::pid_t,
    sigterm_grace: std::time::Duration,
) -> Result<KillSchedulerOutcome, KillSchedulerError> {
    if pid <= 0 {
        return Err(KillSchedulerError::InvalidPid);
    }
    let pid_u32 = pid as u32;

    // Already-absent short-circuit: lifecycle ops are idempotent, so a
    // detach against a non-running scheduler is a no-op success.
    if !proc_pid_alive(pid_u32) {
        return Ok(KillSchedulerOutcome::AlreadyExited);
    }

    // SAFETY: libc::kill is async-signal-safe per POSIX and the
    // pid was validated above. EPERM (signal denied) or ESRCH
    // (process exited between the alive check and the kill) are
    // both observable via the subsequent procfs poll — EPERM means
    // the process keeps running and we'll escalate to SIGKILL;
    // ESRCH means the process is already gone and the poll will
    // immediately observe procfs absence.
    let _ = unsafe { libc::kill(pid, libc::SIGTERM) };

    let interval = std::time::Duration::from_millis(50);
    if poll_proc_pid_absent(pid_u32, interval, sigterm_grace) {
        return Ok(KillSchedulerOutcome::ExitedAfterSigterm);
    }

    // SIGTERM grace elapsed — escalate. SAFETY identical to the
    // SIGTERM call above; SIGKILL cannot be caught or ignored per
    // POSIX so the kernel will run the exit path even if the
    // scheduler binary was actively ignoring SIGTERM.
    let _ = unsafe { libc::kill(pid, libc::SIGKILL) };

    if poll_proc_pid_absent(pid_u32, interval, POST_SIGKILL_GRACE) {
        Ok(KillSchedulerOutcome::EscalatedToSigkill)
    } else {
        Err(KillSchedulerError::StillAliveAfterSigkill)
    }
}

/// Post-SIGKILL grace inside [`kill_scheduler_process`]. SIGKILL
/// triggers the kernel's `exit_notify` → `release_task` cascade
/// (kernel/exit.c) which removes `/proc/{pid}`; the wait here covers
/// both the routine reap path (sub-100ms for a simple userspace
/// process) AND the scheduler-lifecycle Op kill path where an scx
/// scheduler's exit blocks on `scx_disable_workfn`
/// (`kernel/sched/ext.c:6101`) tearing down BPF programs from a
/// workqueue. BPF tear-down dominates the SIGKILL→/proc removal
/// latency for scx_* binaries and routinely exceeds 1s on
/// loaded kernels; 2s leaves comfortable headroom while keeping
/// the unit-test fast for the simple-process case (the test
/// closure exits immediately on SIGKILL so the post-SIGKILL poll
/// returns in <50ms).
///
/// A `StillAliveAfterSigkill` firing AFTER this budget indicates a
/// structurally wrong target — D-state hang, kernel UB, BPF cleanup
/// deadlock — and operators should treat the variant as a debug
/// signal, not a transient retry case. Carried as a module-level
/// const so the value is greppable + paired with a single doc
/// explaining the choice rather than left as a magic number at the
/// call site.
const POST_SIGKILL_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// Wait (kernel-evented via `pidfd_wait_exit`) for `/proc/{pid}`
/// absence up to `timeout`. Returns `true` if the pid's procfs
/// entry disappears within the budget, `false` otherwise. The
/// `interval` parameter is currently unused (the evented path
/// carries no sleep cadence).
///
/// Single source of truth for "wait until the kernel runs
/// release_task for this pid": [`kill_scheduler_process`] uses it to
/// observe SIGTERM / SIGKILL aftermath, and [`poll_startup`]'s
/// pidfd-unavailable fallback uses it to observe early-death during
/// scheduler launch. Both call sites need the same SIG_IGN-safe
/// latency profile, so folding the loop here keeps a future EINTR
/// or signal-pause refinement applied uniformly.
pub(crate) fn poll_proc_pid_absent(
    pid: u32,
    _interval: std::time::Duration,
    timeout: std::time::Duration,
) -> bool {
    // Evented via `pidfd_wait_exit` in the shared
    // `freeze_coord::evented_wait` module. The kernel fires POLLIN
    // on the pidfd when the task enters EXIT_ZOMBIE
    // (do_notify_pidfd from exit_notify in kernel/exit.c). The
    // closure passes `proc_pid_alive` as the source of truth so
    // races between SIGTERM/SIGKILL and `pidfd_open` resolve to
    // the /proc-observable answer.
    let start = std::time::Instant::now();
    let deadline = start + timeout;
    let exited = crate::vmm::freeze_coord::evented_wait::pidfd_wait_exit(pid, deadline, || {
        proc_pid_alive(pid)
    });
    if !exited {
        // Log on timeout so the caller chain — which may swallow
        // the bool into a non-error path — leaves a visible
        // breadcrumb in /tmp/ktstr*.log per the "log on timeout
        // when no error surfaces" rule.
        tracing::warn!(
            pid,
            elapsed_s = start.elapsed().as_secs_f64(),
            timeout_s = timeout.as_secs_f64(),
            "poll_proc_pid_absent: timeout — pid still alive after deadline; \
             pidfd POLLIN never fired and /proc entry persists. Common causes: \
             scheduler not honoring SIGTERM (check its signal handler), scheduler \
             stuck in D-state on a kernel mutex, or the caller's grace window is \
             too tight for the scheduler's exit path (post-libbpf-detach can take \
             seconds on cold caches)"
        );
    }
    exited
}
