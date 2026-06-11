/// Rust init (PID 1) for the VM guest.
///
/// When the test binary is
/// packed as `/init` in the initramfs, `ktstr_guest_init()` is called
/// from the ctor when PID 1 is detected.
/// It never returns — it mounts filesystems, then either dispatches
/// a test (start scheduler, run test, reboot) or drops into an
/// interactive shell (when `KTSTR_MODE=shell` is on the kernel
/// cmdline).
use std::fs;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use crate::sync::Latch;

use nix::mount::{MsFlags, mount};
use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
use nix::pty::openpty;
use nix::sys::reboot::{RebootMode, reboot};
use nix::sys::termios::{SetArg, cfmakeraw, tcgetattr, tcsetattr};

/// COM2 device path for sentinel and diagnostic output.
const COM2: &str = "/dev/ttyS1";
/// COM1 device path for kernel console / trace output.
const COM1: &str = "/dev/ttyS0";
/// Virtio-console device path. Used for shell I/O when available.
const HVC0: &str = "/dev/hvc0";

/// tracefs enable gate for the `sched_ext_dump` tracepoint. Writing
/// `"1"` activates the event, `"0"` deactivates it.
const TRACE_SCHED_EXT_DUMP_ENABLE: &str =
    "/sys/kernel/tracing/events/sched_ext/sched_ext_dump/enable";
/// Global tracefs on/off switch. Writing `"0"` stops new events from
/// being recorded into the ring buffer (`ring_buffer_record_off`); the
/// userspace trace_pipe reader still has to drain whatever is already
/// buffered before reboot. Disabling the producer side first is what
/// makes the reader's drain window terminate — once no new events
/// arrive, poll eventually returns 0 and the drain_deadline elapses.
const TRACE_TRACING_ON: &str = "/sys/kernel/tracing/tracing_on";
/// tracefs streaming endpoint for the active trace. The trace_pipe
/// reader opens this once per boot and forwards every line to COM1.
const TRACE_PIPE: &str = "/sys/kernel/tracing/trace_pipe";

/// sysfs attribute exposing the active sched_ext root scheduler's
/// name. Empty / absent when no scheduler is registered; populated
/// (with a trailing newline) when registration has completed.
/// Kernel-side owner: `kernel/sched/ext.c` creates this via
/// `kobject_init_and_add` under the `sched_ext` kset after
/// `sch->ops.name` is set.
const SYSFS_SCHED_EXT_ROOT_OPS: &str = "/sys/kernel/sched_ext/root/ops";

/// Reboot immediately. Used for fatal init errors and normal shutdown.
fn force_reboot() -> ! {
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

/// Side channel for the scheduler PID published by [`start_scheduler`]
/// once `Command::spawn` returns. The guest test-dispatch path
/// (e.g. [`crate::test_support`] consumers that need the scheduler's
/// pid for cgroup attach / kill / probe) reads it via [`sched_pid`].
///
/// Replaces a previous `std::env::set_var("SCHED_PID", ...)` write.
/// Mutating glibc's global `__environ` array while another thread is
/// live (the Phase A probe thread spawned in `start_probe_phase_a`
/// runs concurrently with `start_scheduler`) is documented UB on
/// Linux — see
/// [`crate::test_support::propagate_rust_env_from_cmdline`] for the
/// mirroring rationale. An atomic side channel is the
/// data-race-free alternative.
///
/// Sentinel: `0` means "no scheduler started". `pid_t` is a signed
/// integer in glibc; the kernel never returns `0` from `fork(2)` to
/// the parent, so `0` is a safe "unset" marker for the producer to
/// initialise with and the consumer to filter on.
static SCHED_PID: AtomicI32 = AtomicI32::new(0);

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
/// `suppress_com2` + `probe_output_done` are determined at boot
/// (based on whether the probe stack is active) and don't change
/// across Op dispatches — capturing once at install time keeps
/// the restart helper signature minimal.
struct SchedExitMonitorBootCtx {
    suppress_com2: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
}

static SCHED_EXIT_MONITOR_BOOT_CTX: OnceLock<SchedExitMonitorBootCtx> = OnceLock::new();

/// Install the boot-time scheduler-exit monitor handle and capture
/// the dispatch context [`restart_sched_exit_monitor_with_log`]
/// needs to spawn replacement monitors. Called once at boot
/// after [`start_sched_exit_monitor`] returns.
///
/// `boot_stop` may be `None` when [`start_sched_exit_monitor`]
/// returned None (no scheduler configured at boot); the slot
/// stays empty and the first Op::AttachScheduler dispatch
/// populates it via [`restart_sched_exit_monitor_with_log`].
pub(crate) fn install_initial_sched_exit_monitor(
    boot_stop: Option<SchedExitStop>,
    suppress_com2: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
) {
    let slot = SCHED_EXIT_MONITOR_SLOT.get_or_init(|| std::sync::Mutex::new(None));
    *slot.lock().unwrap() = boot_stop;
    let _ = SCHED_EXIT_MONITOR_BOOT_CTX.set(SchedExitMonitorBootCtx {
        suppress_com2,
        probe_output_done,
    });
}

/// Stop the currently-installed scheduler-exit monitor (if any).
/// The scheduler-lifecycle Op handler calls this BEFORE SIGTERM-ing
/// the scheduler so the monitor thread exits cleanly without
/// sending the `MSG_TYPE_SCHED_EXIT` message that the host's
/// freeze coordinator would otherwise promote into the run-wide
/// kill flag (per `src/vmm/freeze_coord/dispatch.rs` SchedExit
/// arm). Idempotent — a no-op when the slot is already empty.
pub(crate) fn stop_sched_exit_monitor() {
    let Some(slot) = SCHED_EXIT_MONITOR_SLOT.get() else {
        return;
    };
    let prev = slot.lock().unwrap().take();
    if let Some(stop) = prev {
        stop.stop_and_join();
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

/// Spawn a fresh scheduler-exit monitor for the live SCHED_PID
/// and install it into the slot. Op handler calls this AFTER the
/// new scheduler is spawned and SCHED_PID is published, so the
/// monitor watches the post-Op PID. `log_path` is the per-spawn
/// log file path — all three lifecycle Ops (Attach, Replace,
/// Restart) pass the seq-suffixed path from
/// `staged_scheduler_log_path`.
///
/// Uses the boot-captured `suppress_com2` + `probe_output_done`
/// so the new monitor behaves identically to the boot monitor. If
/// the boot ctx was never installed (degenerate test environment
/// where `install_initial_sched_exit_monitor` never ran) the
/// helper is a no-op and the new scheduler stays unmonitored —
/// the boot path is the only legitimate context that installs
/// the ctx.
pub(crate) fn restart_sched_exit_monitor_with_log(log_path: Option<&str>) {
    let Some(ctx) = SCHED_EXIT_MONITOR_BOOT_CTX.get() else {
        return;
    };
    let slot = SCHED_EXIT_MONITOR_SLOT.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = slot.lock().unwrap();
    // Defensive: if the Op handler skipped stop_sched_exit_monitor
    // for any reason, stop_and_join the stale handle before
    // installing the new one. The take() leaves the slot empty
    // for the duration of start_sched_exit_monitor — readers in
    // that window observe "no monitor", which is correct since
    // the new monitor hasn't been spawned yet.
    if let Some(prev) = guard.take() {
        prev.stop_and_join();
    }
    *guard = start_sched_exit_monitor(
        sched_pid().map(|p| p as u32),
        log_path,
        ctx.suppress_com2.clone(),
        ctx.probe_output_done.clone(),
    );
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
/// ([`spawn_scheduler_from_paths`]) calls this directly with the
/// freshly-spawned `child.id()`.
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
/// `src/scenario/ops/mod.rs`). Returns `None` when no scheduler
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
/// `src/scenario/ops/mod.rs` immediately after the corresponding
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
fn with_sigchld_default<F, R>(f: F) -> R
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
fn proc_pid_alive(pid: u32) -> bool {
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
// `#[allow(dead_code)]` because the helper has no production caller
// in this commit — the Op::DetachScheduler / Op::RestartScheduler /
// Op::ReplaceScheduler dispatchers that will consume it land in
// follow-up work. Tests in this module exercise every variant + the
// InvalidPid error path, so the helper is verified-correct as it
// lands; the allow becomes a no-op the moment the first production
// caller wires up.
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
/// # Poll cadence
///
/// 50ms polling interval — matches the existing
/// [`poll_startup`] cadence so the latency-vs-CPU tradeoff is
/// consistent across the scheduler-lifecycle helpers. The
/// post-SIGKILL grace is the module-level [`POST_SIGKILL_GRACE`]
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
/// (`kernel/sched/ext.c:5923`) tearing down BPF programs from a
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

/// Poll `/proc/{pid}` for absence up to `timeout`, sleeping at the
/// caller's `interval` cadence between checks. Returns `true` if the
/// pid's procfs entry disappears within the budget, `false`
/// otherwise.
///
/// Single source of truth for "wait until the kernel runs
/// release_task for this pid": [`kill_scheduler_process`] uses it to
/// observe SIGTERM / SIGKILL aftermath, and [`poll_startup`]'s
/// pidfd-unavailable fallback uses it to observe early-death during
/// scheduler launch. Both call sites need the same SIG_IGN-safe
/// latency profile, so folding the loop here keeps a future EINTR
/// or signal-pause refinement applied uniformly.
fn poll_proc_pid_absent(
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

/// Async-signal-safe rendering of `value` as lowercase hex (no `0x`
/// prefix, no leading-zero trim) into the tail of `buf`. Returns the
/// byte slice covering the rendered digits.
///
/// Used by [`fatal_signal_handler`], where every libc allocator
/// boundary is forbidden — `format!`, `write!`, and even
/// `core::fmt::Display` formatters can pull in heap or thread-local
/// state. A hand-rolled nibble walk over a stack buffer is the only
/// AS-safe way to surface the faulting address.
///
/// 16 hex digits cover the full `u64` range. The caller passes a
/// `[u8; 16]` and uses the returned subslice (always exactly 16
/// bytes) directly.
fn u64_to_hex_asm(value: u64, buf: &mut [u8; 16]) -> &[u8] {
    static HEX: &[u8; 16] = b"0123456789abcdef";
    for (i, slot) in buf.iter_mut().enumerate() {
        let nibble = (value >> ((15 - i) * 4)) & 0xf;
        *slot = HEX[nibble as usize];
    }
    &buf[..]
}

/// AS-safe write of every byte in `bytes` to fd `fd`. Loops on partial
/// writes; bails on the first error or zero-byte return so a closed/
/// faulted fd cannot wedge the handler.
fn write_all_asm(fd: libc::c_int, bytes: &[u8]) {
    let mut off = 0;
    while off < bytes.len() {
        // SAFETY: `write(2)` is async-signal-safe per signal-safety(7)
        // on Linux. `bytes.as_ptr().add(off)` is in-bounds because
        // `off < bytes.len()`. The write is best-effort — any
        // failure short-circuits the loop and the handler proceeds
        // to `reboot(2)`.
        let n = unsafe {
            libc::write(
                fd,
                bytes.as_ptr().add(off) as *const libc::c_void,
                bytes.len() - off,
            )
        };
        if n <= 0 {
            return;
        }
        off += n as usize;
    }
}

/// Async-signal-safe handler for SIGSEGV / SIGBUS / SIGILL.
///
/// The Rust panic hook installed in [`ktstr_guest_init`] does NOT
/// fire for native CPU faults: the kernel raises these signals with
/// `SIG_DFL` disposition, which calls `do_coredump` and terminates
/// the process. Inside guest init that means PID 1 dies, the kernel
/// observes "init exited", and the host sees the VM force-reboot
/// without any guest-side diagnostic on COM2.
///
/// This handler closes the gap by emitting a `PANIC:`-prefixed line
/// — matching the prefix `extract_panic_message` anchors on — that
/// names the signal and the faulting address before driving
/// [`force_reboot`]. The host crash-classification pipeline then
/// surfaces native faults through the same code path as Rust panics.
///
/// Constraints, all enforced inside the handler:
///
/// - Async-signal-safety per `signal-safety(7)`. No `fs::write`, no
///   `format!`, no `Backtrace::force_capture` — all of those touch
///   the heap, locks, or per-thread formatter state. Only `open(2)`,
///   `write(2)`, `tcdrain(2)`, and `reboot(2)` (all in the AS-safe
///   list) are invoked, plus pure stack arithmetic.
/// - No thread-local state. Worker threads spawned later
///   (`hvc0_poll_loop`, `start_trace_pipe`) inherit the parent's
///   sigaction disposition because Linux signal dispositions are
///   process-wide; this handler runs on whichever thread faulted.
/// - Bounded recursion. `SA_RESETHAND` is set so a fault inside this
///   handler reverts to `SIG_DFL`, which terminates immediately
///   instead of looping.
unsafe extern "C" fn fatal_signal_handler(
    sig: libc::c_int,
    info: *mut libc::siginfo_t,
    _ctx: *mut libc::c_void,
) {
    // Static prefixes per signal. Hard-coded because signal-name
    // formatting via `strsignal(3)` allocates / touches locale
    // state and is not AS-safe.
    let prefix: &[u8] = match sig {
        libc::SIGSEGV => b"PANIC: fatal signal SIGSEGV at addr 0x",
        libc::SIGBUS => b"PANIC: fatal signal SIGBUS at addr 0x",
        libc::SIGILL => b"PANIC: fatal signal SIGILL at addr 0x",
        _ => b"PANIC: fatal signal (unknown) at addr 0x",
    };

    // Faulting address from `siginfo_t.si_addr`. `siginfo_t` field
    // access in Rust requires going through the libc bindings;
    // `si_addr()` is the canonical accessor that handles the union
    // layout differences between glibc and musl. Falls back to 0
    // when `info` is null. Defensive null check; Linux always
    // populates info for SA_SIGINFO handlers (see kernel/signal.c
    // `force_sig_fault_to_task` → `force_sig_info_to_task` and the
    // arch `setup_rt_frame` paths, which unconditionally pass
    // `&frame->info` to the handler).
    let addr: u64 = if info.is_null() {
        0
    } else {
        // SAFETY: `info` is non-null here; `si_addr()` reads the
        // address-fault arm of the siginfo union, which is the
        // valid arm for SIGSEGV / SIGBUS / SIGILL per the kernel's
        // `force_sig_fault` path (`kernel/signal.c`).
        let p = unsafe { (*info).si_addr() };
        p as u64
    };

    let mut hex_buf = [0u8; 16];
    let hex = u64_to_hex_asm(addr, &mut hex_buf);

    // Open COM2 first (canonical destination), then COM1. Both with
    // `O_WRONLY | O_NONBLOCK` so the open and the `write_all_asm`
    // loop never block on guest-side flow control. `tcdrain(2)`
    // does NOT honor `O_NONBLOCK` — it is a separate ioctl that
    // waits for the kernel tty layer's write queue to drain — but
    // the wait is bounded by UART FIFO drain time (microseconds at
    // worst) because PIO commits each byte inside `KVM_RUN` before
    // userspace returns; the kernel sees its own output queue empty
    // almost immediately after the final `write(2)` returns.
    //
    // SAFETY: `open(2)`, `write(2)`, `tcdrain(2)`, and `close(2)`
    // are all in the signal-safety(7) AS-safe set. The path
    // strings are static C strings with explicit NUL terminators.
    for path in [c"/dev/ttyS1", c"/dev/ttyS0"] {
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_WRONLY | libc::O_NONBLOCK) };
        if fd < 0 {
            continue;
        }
        write_all_asm(fd, prefix);
        write_all_asm(fd, hex);
        write_all_asm(fd, b"\n");
        // Seal the contract: tcdrain waits for the kernel's output
        // queue to drain before we issue `reboot(2)`. PIO commits
        // per byte so the wait is effectively immediate; tcdrain
        // ignores `O_NONBLOCK` but the drain time is bounded by
        // UART FIFO depth, not by host-side back-pressure.
        unsafe {
            libc::tcdrain(fd);
            libc::close(fd);
        }
    }

    // `reboot(LINUX_REBOOT_CMD_RESTART)` is the AS-safe analogue of
    // `force_reboot()`'s nix wrapper. The syscall does not return
    // on success; if it somehow does (CAP_SYS_BOOT missing,
    // already rebooting), `_exit(1)` ensures the handler does
    // NOT fall through to user code with a corrupt stack /
    // mid-fault state.
    unsafe {
        libc::reboot(libc::LINUX_REBOOT_CMD_RESTART);
        libc::_exit(1);
    }
}

/// Install [`fatal_signal_handler`] for SIGSEGV, SIGBUS, and SIGILL.
///
/// `SA_SIGINFO` makes the handler receive the `siginfo_t *` whose
/// `si_addr` carries the faulting address. `SA_RESETHAND` reverts
/// the disposition to `SIG_DFL` after the first delivery so a fault
/// inside the handler terminates cleanly instead of looping. `SA_ONSTACK`
/// directs the kernel to run the handler on the alternate stack
/// registered via `sigaltstack(2)` below — without it a stack-overflow
/// SIGSEGV faults again on the overflowed stack and the kernel
/// terminates the process before any diagnostic reaches the host.
///
/// `sa_mask` adds SIGSEGV / SIGBUS / SIGILL so that while one fatal-
/// signal handler is executing, the other two cannot interrupt it.
/// Cross-signal nesting (e.g. SIGBUS arriving while the SIGSEGV
/// handler is mid-write to COM2) would scribble interleaved bytes
/// onto the serial output and lose the diagnostic. The signal being
/// delivered is also masked by default; combined with `SA_RESETHAND`
/// a re-fault of the same signal terminates under `SIG_DFL` instead
/// of looping back into this handler.
///
/// Failures are silently ignored: if `sigaction(2)` rejects the
/// install (returns -1), the previous disposition (typically
/// `SIG_DFL`) remains in place — which is exactly the pre-fix
/// behavior. There's no user-visible regression on failure, just
/// the unchanged gap the panic hook also doesn't cover. `mmap(2)` /
/// `sigaltstack(2)` failures are similarly tolerated: the handler
/// stays installed without `SA_ONSTACK`, which only loses the
/// stack-overflow diagnostic — every other fatal-signal path keeps
/// working.
fn install_fatal_signal_handlers() {
    // SAFETY: `std::mem::zeroed::<libc::sigaction>()` produces a
    // valid all-zero `sigaction` (all libc fields are integer or
    // pointer-typed, zero is valid for all of them). The
    // `sa_sigaction` field is then set to a function pointer with
    // the correct `extern "C"` signature, and `sa_flags` is set
    // to a valid combination of POSIX `SA_*` constants.
    let mut act: libc::sigaction = unsafe { std::mem::zeroed() };
    act.sa_sigaction = fatal_signal_handler as *const () as usize;
    act.sa_flags = libc::SA_SIGINFO | libc::SA_RESETHAND;
    // Initialize the mask, then add every fatal signal so that one
    // handler in flight cannot be interrupted by another fatal
    // signal — see fn doc for why interleaved handlers corrupt the
    // diagnostic.
    unsafe {
        libc::sigemptyset(&mut act.sa_mask);
        libc::sigaddset(&mut act.sa_mask, libc::SIGSEGV);
        libc::sigaddset(&mut act.sa_mask, libc::SIGBUS);
        libc::sigaddset(&mut act.sa_mask, libc::SIGILL);
    }

    // Allocate and register a signal alternate stack so a stack-
    // overflow SIGSEGV runs the handler on a separate stack instead
    // of faulting again on the overflowed one. `SIGSTKSZ` is the
    // platform's recommended minimum; clamp to 64 KiB so older libc
    // headers (where SIGSTKSZ is 8 KiB) still leave headroom for the
    // backtrace-free handler frame plus `write(2)` / `tcdrain(2)` /
    // `reboot(2)` syscall trampolines.
    //
    // `mmap(MAP_PRIVATE | MAP_ANONYMOUS)` is the AS-safe-allocation
    // analogue to a heap allocation: pages are zero-initialised on
    // first touch, so no separate clear is needed. The mapping is
    // intentionally leaked — `sigaltstack` keeps the kernel pointing
    // at it for the lifetime of the process, and PID 1 never returns
    // from `ktstr_guest_init`.
    //
    // SAFETY: `mmap(2)` and `sigaltstack(2)` are both POSIX-defined
    // syscalls. The pointers / lengths supplied are well-formed
    // (NULL hint, fd=-1 for anonymous mappings, offset=0). Failure
    // returns `MAP_FAILED`; on that path we skip `sigaltstack` and
    // leave `SA_ONSTACK` unset on `sa_flags` — see fn doc for the
    // failure-mode rationale.
    let stack_size = libc::SIGSTKSZ.max(65536);
    let stack = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            stack_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if stack != libc::MAP_FAILED {
        let ss = libc::stack_t {
            ss_sp: stack,
            ss_flags: 0,
            ss_size: stack_size,
        };
        // SAFETY: `ss` is a fully-initialised `stack_t` with a
        // valid mmap'd buffer and matching size. Passing
        // `null_mut()` for `oss` discards the previous alternate
        // stack — PID 1 has no prior alternate stack at this
        // call site (signal handling has not been touched yet).
        unsafe {
            libc::sigaltstack(&ss, std::ptr::null_mut());
        }
        act.sa_flags |= libc::SA_ONSTACK;
    }

    for sig in [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL] {
        // SAFETY: `sigaction(2)` with a valid `struct sigaction`
        // and a NULL old-action pointer is well-defined.
        // Failures are silently swallowed (see fn doc).
        let _ = unsafe { libc::sigaction(sig, &act, std::ptr::null_mut()) };
    }
}

/// Full guest init lifecycle. Called from the ctor when PID 1 is
/// detected. Mounts filesystems, then either runs the test lifecycle
/// (scheduler + dispatch + reboot) or drops into an interactive
/// shell. Never returns.
pub(crate) fn ktstr_guest_init() -> ! {
    let t0 = std::time::Instant::now();

    // Crash diagnostic capture has two arms because they have
    // disjoint trigger surfaces:
    //
    // 1. Native fatal signals (`install_fatal_signal_handlers`,
    //    installed first): SIGSEGV / SIGBUS / SIGILL invoke the
    //    kernel's `do_coredump` under SIG_DFL — they bypass the
    //    panic hook entirely. Without a sigaction handler the
    //    kernel terminates init, which the parent kernel observes
    //    as "init exited" and force-reboots without any guest-side
    //    diagnostic reaching the host. Installing this arm before
    //    the panic hook minimises the window where an early fault
    //    (heap setup, mount syscalls, anything before the hook
    //    registers) escapes capture.
    // 2. Rust panic hook (below): fires on `panic!`, `unwrap`,
    //    assertion failures, and any other invocation of the Rust
    //    panic machinery (both `panic = "unwind"` and
    //    `panic = "abort"` runtimes invoke the hook before
    //    unwinding/aborting).
    //
    // Both arms write a `PANIC:`-prefixed line to COM2 (and COM1)
    // so the host-side `extract_panic_message` picks them up
    // through the same code path. COM2 is the canonical crash-
    // diagnostic transport, surviving a wedged virtio port: the
    // bulk-virtio path is intentionally NOT used here because the
    // kernel `virtio_console` TX can block on host backpressure
    // and blocking inside a fault handler would deadlock the
    // guest before the diagnostic reached the host. COM2 (16550
    // UART) PIO writes commit synchronously inside `KVM_RUN`
    // before userspace returns, so the host's serial capture
    // sees every byte even on a wedged guest.
    install_fatal_signal_handlers();
    std::panic::set_hook(Box::new(|info| {
        let bt = std::backtrace::Backtrace::force_capture();
        let msg = format!("PANIC: {info}\n{bt}\n");
        // COM2 / COM1 serial. COM2 is the canonical crash log
        // destination for the host's serial-capture path; the
        // host parses the `PANIC:` prefix via
        // `extract_panic_message` to reconstruct the crash
        // diagnostic.
        let _ = fs::write(COM2, &msg);
        let _ = fs::write(COM1, &msg);
        // Push any buffered Rust-side bytes into the underlying pipe
        // before reboot. After stdio redirect, fd 1 / fd 2 are
        // pipe write ends drained by `redirect_stdio_to_bulk_port`'s
        // forwarder threads — `tcdrain` is unavailable here (the
        // pipe is not a tty, the syscall returns ENOTTY silently).
        // `flush()` is the equivalent: it commits any
        // BufWriter-buffered bytes into the pipe's kernel buffer
        // where the forwarder thread can pick them up. The
        // forwarder threads are not joined before `force_reboot`;
        // bytes that have not yet been read out of the pipe and
        // shipped over the bulk port at the moment of reboot are
        // lost — see the queue task on joining the forwarders for
        // the residual gap. The COM1/COM2 `fs::write` above remains
        // the synchronous-PIO path that guarantees the panic
        // diagnostic itself reaches the host before reboot.
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        force_reboot();
    }));

    // Ignore SIGCHLD so child processes don't become zombies.
    // PID 1 is the reaper — without this, zombie processes accumulate.
    unsafe {
        libc::signal(libc::SIGCHLD, libc::SIG_IGN);
    }

    // Phase 1: Mounts.
    mount_filesystems();
    let t_mounts = t0.elapsed();

    // Install the tracing subscriber as early as possible — right after
    // `mount_filesystems()` so /proc is available for the RUST_LOG
    // cmdline extraction below, and BEFORE the rest of guest init runs
    // so every subsequent `tracing::*` call is captured. Earlier
    // versions installed the subscriber after `redirect_stdio_to_bulk_port`,
    // which silently dropped every tracing event before the redirect.
    //
    // EnvFilter respects RUST_LOG when set; default is `warn` so
    // teardown diagnostics (`tracing::warn!`, `tracing::error!`)
    // surface without requiring RUST_LOG to be plumbed through the
    // guest cmdline. `from_default_env()` alone would collapse to
    // the implicit `error` level and swallow warn-level output —
    // exactly the diagnostics needed to debug teardown failures.
    if let Ok(cmdline) = fs::read_to_string("/proc/cmdline")
        && let Some(val) = cmdline
            .split_whitespace()
            .find(|s| s.starts_with("RUST_LOG="))
            .and_then(|s| s.strip_prefix("RUST_LOG="))
    {
        // SAFETY: single-threaded PID 1 context.
        unsafe { std::env::set_var("RUST_LOG", val) };
    }
    let t_pre_subscriber = t0.elapsed();
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    let t_subscriber = t0.elapsed();

    // Verify initramfs extraction completed. The sentinel file is the
    // last entry written by build_initramfs_base — its absence means
    // the kernel ran out of memory during cpio extraction. The memory
    // formula should prevent this; hitting it indicates an estimation bug.
    if !Path::new("/.ktstr_init_ok").exists() {
        // Dump dmesg to serial so the host sees the kernel OOM messages.
        if let Ok(raw) = rmesg::logs_raw(rmesg::Backend::Default, false) {
            let _ = fs::write(COM2, &raw);
            let _ = fs::write(COM1, &raw);
        }
        let msg = "FATAL: initramfs extraction incomplete — kernel ran out of \
                   memory during cpio extraction. This indicates a bug in ktstr's \
                   memory estimation. Please report this issue. As a workaround, \
                   try `--memory N` with a larger value.";
        let _ = fs::write(COM2, msg);
        let _ = fs::write(COM1, msg);
        tracing::error!("{msg}");
        force_reboot();
    }

    // Boot-complete signal. The host monitor's pre-sample
    // `epoll_wait` blocks on a sys_rdy eventfd; the freeze
    // coordinator's bulk-drain dispatch promotes a CRC-valid
    // `MSG_TYPE_SYS_RDY` frame into that eventfd. Sending here —
    // after `mount_filesystems()` brought up devtmpfs and the
    // initramfs-extraction sentinel confirms userspace is sound —
    // guarantees the host's first sample observes a fully-booted
    // guest with `setup_per_cpu_areas` populated and KASLR
    // randomization already complete (both kernel-boot
    // prerequisites for the monitor's `__per_cpu_offset[]` /
    // `page_offset_base` reads). Replaces the earlier trigger that
    // fired on the first port-0 TX byte (kernel printk via
    // `/dev/hvc0`), which depended on incidental console traffic
    // rather than an explicit readiness signal.
    //
    // `/dev/vport0p1` may not yet exist at this point: the kernel
    // virtio_console driver's multiport handshake (DEVICE_READY →
    // PORT_ADD → PORT_READY → PORT_OPEN, see
    // `drivers/char/virtio_console.c`) completes asynchronously
    // and is independent of devtmpfs being mounted. The retry
    // protocol, wall-clock deadline, and failure diagnostics live
    // in [`send_sys_rdy_with_retry`].
    let kern_phys_base = crate::vmm::guest_comms::read_phys_base_from_iomem().unwrap_or(0);
    // Runtime KVA of `_text`, the kernel image start symbol.
    // Powers the host-side virt-KASLR derive at
    // `src/vmm/freeze_coord/dispatch.rs::dispatch_bulk_message`'s
    // KERN_ADDRS arm: subtracting the link-time KVA (from the
    // host's vmlinux parse) yields the virt-KASLR slide so the
    // monitor and dump pipelines can resolve per-CPU `rq` /
    // `kernel_cpustat` / `kstat` KVAs under
    // `CONFIG_RANDOMIZE_BASE=y`. `_text` is defined in
    // `vmlinux.lds.S` on every Linux build, so this works on
    // both x86_64 and aarch64. `None` only when the symbol is
    // masked (kptr_restrict + non-CAP_SYSLOG, which we are not —
    // PID 1 has all caps) or `/proc/kallsyms` is unreadable; the
    // host's KERN_ADDRS handler treats `None` as "guest could
    // not derive" and leaves the slot at its prior value (the
    // BSP MSR_LSTAR path may still publish on x86_64).
    let kern_text_kva = crate::vmm::guest_comms::read_kernel_text_from_kallsyms();
    // `page_offset_base` slot — derive the runtime KVA of the
    // `page_offset_base` global from /proc/kallsyms (kernel-PhD-
    // confirmed it lives in `.data..ro_after_init`, declared at
    // `arch/x86/kernel/head64.c:63`). The KVA here is the symbol's
    // ADDRESS — the host reads the runtime VALUE (the direct-map
    // base) by translating this KVA to PA via
    // `monitor::symbols::text_kva_to_pa_with_base` (using
    // `kern_phys_base`) and `read_u64`-ing at that PA. Returns
    // `None` on arm64 (no `page_offset_base` global — `PAGE_OFFSET`
    // is compile-time per `arch/arm64/include/asm/memory.h:43-45`)
    // and when CONFIG_RANDOMIZE_MEMORY=n (symbol absent). The
    // wire field is `u64`, so `None` collapses to 0 — host treats
    // 0 as "use DEFAULT_PAGE_OFFSET fallback" (matching the
    // historical pre-derivation behavior).
    let kern_page_offset_base_kva =
        crate::vmm::guest_comms::read_kernel_page_offset_base_from_kallsyms().unwrap_or(0);
    let kern_addrs =
        crate::vmm::wire::KernAddrs::new(kern_phys_base, kern_page_offset_base_kva, kern_text_kva);
    // `count_online_cpus()` reads /sys/devices/system/cpu/online which
    // `mount_filesystems()` mounted earlier. Fallback to 1 yields the
    // single-vCPU budget (base + 1×per-vCPU) if the read fails —
    // preserves the original single-CPU default rather than panicking
    // on a procfs hiccup.
    let vcpus = count_online_cpus().unwrap_or(1);
    let budget = std::time::Duration::from_millis(crate::test_support::sys_rdy_budget_ms(vcpus));
    send_sys_rdy_with_retry(
        budget,
        vcpus,
        &kern_addrs,
        std::path::Path::new(crate::vmm::guest_comms::BULK_PORT_DEV),
    );

    // Phase 1.5: Auto-mount the user data disk at /mnt/disk0 if the
    // host pre-formatted it (KTSTR_DISK0_FS=<tag> on the cmdline).
    // Runs BEFORE `disk_template_mode_requested()` is checked below
    // — but the template-build cmdline never carries
    // `KTSTR_DISK0_FS` (the host emits it only for non-Raw disks
    // and the template-build VM attaches a Raw disk because the
    // whole point is to format it), so this call is a no-op
    // during template-build and the build path is unaffected.
    auto_mount_data_disks();
    // Enable per-program BPF runtime stats (cnt, nsecs). The kernel
    // only populates bpf_prog_stats when bpf_stats_enabled_key is set.
    let _ = fs::write("/proc/sys/kernel/bpf_stats_enabled", "1");

    // Phase 2: Lifecycle event + stdio redirect. The lifecycle frame
    // is for the test harness on the host; shell mode doesn't need it
    // and would route the InitStarted phase into the operator's
    // bulk-port-backed transcript otherwise.
    if !shell_mode_requested() {
        crate::vmm::guest_comms::send_lifecycle(crate::vmm::wire::LifecyclePhase::InitStarted, "");
    }
    redirect_stdio_to_bulk_port();
    let t_stdio = t0.elapsed();

    // Phase 2c: spawn the scheduler-stats relay UNCONDITIONALLY.
    // Event-driven: the relay uses inotify to wait for the
    // scheduler's `/var/run/scx/root/stats` socket to appear, and
    // poll(2) to multiplex between the port fd, the socket fd, and
    // a stop eventfd. No timeouts, no retry sleeps — the only
    // wakeups are real I/O events or the stop edge written by
    // phase-6 cleanup.
    //
    // By this point `redirect_stdio_to_bulk_port` has run (line
    // above) and the bulk port has been opened, which proves the
    // multiport handshake completed; `/dev/vport0p2` is already
    // present, so the relay's first port-2 open succeeds without
    // retry.
    let stats_relay_stop = start_sched_stats_relay();

    tracing::debug!(
        mount_ms = t_mounts.as_millis() as u64,
        stdio_ms = t_stdio.as_millis() as u64,
        pre_subscriber_ms = t_pre_subscriber.as_millis() as u64,
        subscriber_ms = t_subscriber.as_millis() as u64,
        "guest_init_timing",
    );

    // Set environment variables.
    // SAFETY: single-threaded context — PID 1 before any threads spawn.
    unsafe {
        std::env::set_var("PATH", build_include_path());
        // Mark this process tree as running under guest init (PID 1).
        // Workers forked inside the guest legitimately have
        // `getppid() == 1` because init IS their parent, so the
        // host-side orphan-detection fast-path in `workload.rs` must
        // skip the `_exit(0)` branch when this variable is present.
        // The variable is inherited across fork/exec, so every
        // descendant of guest init (including workloads that re-exec
        // /init to run scenarios) observes it.
        std::env::set_var(crate::KTSTR_GUEST_INIT_ENV, "1");
    }

    // Disk-template build mode: format /dev/vda with the embedded
    // mkfs binary, then reboot. No scheduler load, no test dispatch,
    // no shell. Must run before shell_mode_requested() so a future
    // operator-facing shell command cannot accidentally trip the
    // template path. See [`crate::vmm::disk_template`] for the host
    // side that drives this mode.
    if disk_template_mode_requested() {
        let _span = tracing::debug_span!("disk_template_mode").entered();
        let code = run_disk_template_mode();
        // Match the post-test exit semantics: push buffered stdio
        // bytes into the pipe (the forwarder threads then ship them
        // over the bulk port), emit the binary exit code over the
        // bulk data port so the host knows we're done, reboot.
        // `flush()` replaces the broken `tcdrain(1/2)`
        // which returned ENOTTY against the pipe write ends; the
        // forwarder threads aren't joined here, so bytes still in
        // the pipe at reboot time are lost — see the queue task
        // for forwarder-join plumbing.
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        crate::vmm::guest_comms::send_exit(code);
        // The bulk-port write inside `send_exit` commits via MMIO
        // before userspace returns from KVM_RUN — the EXIT frame is
        // in the host's port-1 RX buffer the moment `send_exit`
        // returns. No additional wait needed before reboot.
        force_reboot();
    }

    // Shell mode: interactive busybox shell instead of test dispatch.
    if shell_mode_requested() {
        let _shell_span = tracing::debug_span!("shell_mode").entered();
        let console_dev = shell_console_device();
        redirect_all_stdio_to(console_dev);

        // Create busybox applet symlinks.
        {
            let _s = tracing::debug_span!("busybox_install").entered();
            let _ = Command::new("/bin/busybox")
                .args(["--install", "-s", "/bin"])
                .status();
        }

        // Mount devpts so PTY allocation works.
        mount_devpts();

        // Run scheduler enable cmds (from `--ktstr-shell-test=NAME`'s
        // ShellTestDescriptor.scheduler_enable_cmds — Phase B of the
        // KernelBuiltin lifecycle, packed into /sched_enable by the
        // VM builder). Idempotent / safe when the file doesn't exist
        // (returns Ok(())). Mirrors the test-mode wire-up at L1329 so
        // the shell-mode operator drops into the SAME scheduler-loaded
        // environment a test would see — without this, the shell falls
        // through to whatever scheduler the kernel boots with and the
        // banner's "running N enable cmd(s)" claim would be a lie.
        exec_shell_script("/sched_enable");

        // --exec mode: run a command non-interactively instead of
        // dropping into an interactive shell. Inherits stdio from init
        // which redirect_all_stdio_to() already pointed at the console
        // device (virtio-console /dev/hvc0 when available, COM2
        // otherwise). The host stdout writer thread drains virtio TX.
        // Checked before MOTD so exec output is not polluted.
        if let Some(cmd) = shell_exec_cmd() {
            tracing::debug!(cmd = %cmd, "shell exec mode");
            // Disable OPOST on stdout so the tty layer does not
            // convert \n to \r\n. Without this, every newline in
            // command output gains a spurious \r visible to the host.
            let stdout_fd = unsafe { BorrowedFd::borrow_raw(1) };
            if let Ok(mut termios) = tcgetattr(stdout_fd) {
                termios
                    .output_flags
                    .remove(nix::sys::termios::OutputFlags::OPOST);
                let _ = tcsetattr(stdout_fd, SetArg::TCSANOW, &termios);
            }
            // [`with_sigchld_default`] flips SIGCHLD to SIG_DFL
            // for the closure body so `Command::status()` (which
            // calls `waitpid(2)`) reaps the child and reports the
            // real exit code. The `SIG_IGN` disposition installed
            // earlier in [`ktstr_guest_init`] for zombie
            // prevention is restored on closure return — and on
            // panic unwind, via the helper's RAII guard.
            let status = with_sigchld_default(|| {
                Command::new("/bin/busybox")
                    .args(["sh", "-c", &cmd])
                    .status()
            });
            let code = match status {
                Ok(s) => s.code().unwrap_or(1),
                Err(e) => {
                    tracing::error!(err = %e, "ktstr-init: exec failed");
                    1
                }
            };
            // Exit code travels via the bulk data port so it does
            // not pollute captured command output on stdout.
            crate::vmm::guest_comms::send_exec_exit(code as i32);
            let _ = std::io::stdout().flush();
            let _ = std::io::stderr().flush();
            // tcdrain is synchronous on the vCPU exit: when these
            // syscalls return, every byte is already in the host's
            // serial writer Vec (or virtio-console TX path). No
            // additional wait needed before reboot.
            unsafe {
                libc::tcdrain(1);
            }
            unsafe {
                libc::tcdrain(2);
            }
            // Run scheduler disable cmds before reboot — symmetric
            // bracket with /sched_enable above; idempotent when the
            // file doesn't exist.
            exec_shell_script("/sched_disable");
            // Drain stdout/stderr after /sched_disable so any
            // stdout/stderr writes from the disable script (e.g.
            // the `echo > /proc/1/fd/1` marker pattern used by
            // the shell-mode lifecycle e2e fixture) reach host
            // capture before force_reboot triggers
            // device_shutdown. tcdrain bounds the TTY FIFO drain
            // in userspace; the virtio TX ring drain itself
            // happens during the kernel's hvc_close path in
            // device_shutdown. Sysfs-only disable scripts (e.g.
            // `echo 0 > /sys/...`) don't write to the TTY FIFO;
            // tcdrain is a harmless no-op for them. No prior
            // Rust stdout/stderr flush is needed because
            // exec_shell_line writes via fs::write, bypassing
            // Rust's BufWriter. Symmetric with the post-payload
            // drain above that protects the /sched_enable +
            // --exec output bracket.
            unsafe {
                libc::tcdrain(1);
            }
            unsafe {
                libc::tcdrain(2);
            }
            force_reboot();
        }

        // MOTD (printed to console before PTY proxy takes over).
        // Skipped in exec mode (handled above).
        let kernel_version = fs::read_to_string("/proc/version")
            .ok()
            .and_then(|v| v.split_whitespace().nth(2).map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".to_string());
        let mem_mib = fs::read_to_string("/proc/meminfo").ok().and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kib| kib.parse::<u64>().ok())
                .map(|kib| kib / 1024)
        });
        println!("ktstr shell");
        println!("  kernel:    {kernel_version}");
        if let Some(mib) = mem_mib {
            println!("  memory:    {mib} MiB");
        }
        print_topology_line();
        print_includes_line();
        println!("  tools:     busybox (ls, ps, top, dmesg, ip, vi, ...)");
        println!("  mounts:    /proc /sys /dev /sys/fs/cgroup /sys/fs/bpf /tmp");
        println!("             /sys/kernel/debug /sys/kernel/tracing /dev/pts");
        println!("  type `exit` for clean shutdown, Ctrl+A X to force-kill");
        let _ = std::io::stdout().flush();

        // Allocate a PTY pair so busybox sh gets a controlling terminal
        // (required for job control: Ctrl+Z, bg, fg).
        tracing::debug!("spawning interactive shell with PTY");
        spawn_shell_with_pty();

        // Run scheduler disable cmds before reboot — symmetric
        // bracket with /sched_enable. Runs after the operator types
        // `exit` (spawn_shell_with_pty returns when the shell exits).
        exec_shell_script("/sched_disable");
        // Drain stdout/stderr after /sched_disable so any
        // stdout/stderr writes from the disable script reach
        // host capture before force_reboot triggers
        // device_shutdown. The interactive-shell path shares
        // the same race + drain semantics as the exec-mode
        // path above (see that comment for the TTY FIFO vs
        // virtio TX ring + fs::write bypass rationale).
        unsafe {
            libc::tcdrain(1);
        }
        unsafe {
            libc::tcdrain(2);
        }
        force_reboot();
    }

    // Read test args from /args early so Phase 2b can parse
    // --ktstr-probe-stack for probe setup before the scheduler starts.
    let args: Vec<String> = {
        let content = fs::read_to_string("/args").unwrap_or_default();
        let mut a = vec!["/init".to_string()];
        a.extend(content.lines().map(|s| s.to_string()));
        a
    };
    tracing::debug!(args = ?args, "parsed /args");

    // Propagate RUST_BACKTRACE and RUST_LOG from the kernel cmdline to
    // the process environment BEFORE Phase A spawns its probe thread.
    // `std::env::set_var` mutates glibc's `__environ` without locking;
    // calling it while the probe thread is live is UB on Linux.
    crate::test_support::propagate_rust_env_from_cmdline();

    // Phase 2b: Probe Phase A (before scheduler starts).
    // Attaches kprobes + trigger + kernel fexit so the one-shot
    // sched_ext_exit tracepoint is captured even if the scheduler
    // crashes immediately on startup.
    let _s_phase2b = tracing::debug_span!("phase2b_probe_phase_a").entered();
    let probe_phase_a = crate::test_support::start_probe_phase_a(&args);
    let probes_active = probe_phase_a.is_some();
    drop(_s_phase2b);

    // Phase 3: Cgroup parent + Scheduler.
    // Create the cgroup parent directory before starting the scheduler
    // so it exists when the scheduler looks for it.
    let _s_phase3 = tracing::debug_span!("phase3_scheduler_start").entered();
    // Per-test workload-cgroup root. Sourced from
    // `KtstrTestEntry::workload_root_cgroup`. The framework owns
    // this slot; the scheduler never sees it.
    create_workload_root_cgroup_from_file();
    // Per-scheduler cgroup the scheduler process is placed in.
    // Sourced from `Scheduler::cgroup_parent`. mkdir + enable
    // controllers here so the tree is ready when `start_scheduler`
    // spawns the child. Distinct from
    // `create_cgroup_parent_from_sched_args` (which fires only
    // when `--cell-parent-cgroup` is present in `/sched_args` for
    // cell-aware schedulers).
    create_scheduler_cgroup_parent_from_file();
    create_cgroup_parent_from_sched_args();
    exec_shell_script("/sched_enable");
    // Plumb the probe pipeline's `stop` + `output_done` into
    // `start_scheduler` so the early-bail paths (Died / not
    // attached / spawn error) can drain probe JSON to COM2 before
    // calling `force_reboot()`. Without the drain, every path that
    // crashes the scheduler before the test dispatches loses its
    // probe payload to the reboot — exactly the diagnostic the
    // probes were attached to capture.
    let probe_drain = probe_phase_a.as_ref().map(|pa| ProbeDrain {
        stop: pa.pipeline.stop.clone(),
        output_done: pa.pipeline.output_done.clone(),
    });
    let (mut sched_child, sched_log_path) = start_scheduler(probe_drain);
    drop(_s_phase3);

    // Phase 4: hvc0 polling + trace pipe (background threads).
    let _s_phase4 = tracing::debug_span!("phase4_vc_poll").entered();
    let (trace_stop, trace_handle) = start_trace_pipe();
    let vc_poll_stop = start_hvc0_poll(trace_stop.clone());
    drop(_s_phase4);

    // Phase 4b: Scheduler death monitor.
    // Spawn a thread that polls /proc/{pid}. If the scheduler exits during
    // the test, the thread writes MSG_TYPE_SCHED_EXIT via bulk port so the host
    // can detect early death without waiting for the watchdog.
    //
    // When probes are active, suppress COM2 log dump to avoid
    // interleaving with probe JSON output on the same serial port.
    let suppress_com2 = Arc::new(AtomicBool::new(probes_active));
    let probe_output_done = probe_phase_a
        .as_ref()
        .map(|pa| pa.pipeline.output_done.clone());
    // Install the boot-time scheduler-exit monitor handle into
    // the module-level slot via `install_initial_sched_exit_monitor`
    // so the scheduler-lifecycle Op dispatcher in
    // `src/scenario/ops/mod.rs` can swap the monitor across
    // Op::AttachScheduler / DetachScheduler / RestartScheduler /
    // ReplaceScheduler. The earlier local-binding pattern held
    // the SchedExitStop in this stack frame, which made it
    // unreachable from the Op dispatch path. The shutdown cascade
    // below calls `stop_sched_exit_monitor` instead of the
    // pre-refactor local `stop_and_join`. Cloning the Arcs is
    // cheap and the boot start_sched_exit_monitor call retains
    // its original semantics — the only difference is the
    // ownership chain after spawn.
    let boot_stop = start_sched_exit_monitor(
        sched_child.as_ref().map(|c| c.id()),
        sched_log_path.as_deref(),
        suppress_com2.clone(),
        probe_output_done.clone(),
    );
    install_initial_sched_exit_monitor(boot_stop, suppress_com2, probe_output_done);

    // Phase 5: Dispatch.
    let _s_phase5 = tracing::debug_span!("phase5_dispatch").entered();
    tracing::debug!("dispatching test");
    crate::vmm::guest_comms::send_lifecycle(crate::vmm::wire::LifecyclePhase::PayloadStarting, "");
    crate::vmm::guest_comms::send_scenario_start();

    #[cfg(feature = "wprof")]
    let wprof_handle = spawn_wprof_if_configured();

    unsafe { libc::signal(libc::SIGCHLD, libc::SIG_DFL) };
    let code = if let Some(pa) = probe_phase_a {
        crate::test_support::maybe_dispatch_vm_test_with_phase_a(&args, pa).unwrap_or(1)
    } else {
        crate::test_support::maybe_dispatch_vm_test_with_args(&args).unwrap_or(1)
    };
    unsafe { libc::signal(libc::SIGCHLD, libc::SIG_IGN) };
    crate::vmm::guest_comms::send_scenario_pause();

    #[cfg(feature = "wprof")]
    if let Some(handle) = wprof_handle
        && let Ok(Some(pb_bytes)) = handle.join()
    {
        crate::vmm::guest_comms::send_wprof_trace(&pb_bytes);
    }

    drop(_s_phase5);

    // Flush test output before teardown. Rust's BufWriter on stdout
    // holds data until flushed; without this the host may not see the
    // test result before reboot.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    crate::test_support::try_flush_profraw();

    // Phase 6: Scheduler cleanup.
    let _s_phase6 = tracing::debug_span!("phase6_cleanup").entered();

    // Stop the sched-exit monitor BEFORE killing the scheduler.
    // Without this ordering, child.kill() makes the scheduler
    // exit, the monitor's pidfd poll wakes, it sees /proc/{pid}
    // gone and emits MSG_TYPE_SCHED_EXIT on the bulk port, the
    // host promotes kill=true, and the BSP exits with ExternalKill
    // before the guest reaches send_exit — producing exit_code=-1
    // on an otherwise clean run.
    //
    // `stop_and_join` sets stop=true (Release), writes the wake
    // eventfd to drop poll wake latency from 250 ms to
    // microseconds, then joins the monitor thread. Joining is
    // event-driven: the monitor's loop checks stop at the top,
    // exits cleanly after `poll(2)` returns, and the join
    // returns. After this call the monitor is guaranteed to have
    // exited without sending MSG_TYPE_SCHED_EXIT, so the
    // subsequent child.kill() cannot trigger the race.
    // Stop the live sched_exit_monitor (whichever scheduler PID it
    // was last installed for — boot or post-Op::Replace) before
    // tearing down the scheduler child below. The slot may be
    // empty if the test ran Op::DetachScheduler without a
    // re-attach; the helper handles that case as a no-op.
    stop_sched_exit_monitor();

    if let Some(ref mut child) = sched_child {
        // On a crash the scheduler is shutting down and flushing its
        // userspace diagnostics to its stderr log. Give it a brief
        // BOUNDED grace to finish writing and exit on its own BEFORE the
        // hard kill, so SIGKILL doesn't truncate that output
        // (`dump_sched_output` below reads the stderr log). Gated on
        // dump_started (the `sched_ext_dump:` tracepoint fires only on an
        // error exit) so clean runs pay nothing; the grace returns early
        // the moment the scheduler exits, and is bounded
        // (`SCHED_KILL_GRACE`) so a userspace hang can't wedge teardown.
        let exited_in_grace =
            scx_dump_started_latch().is_set() && reap_child_bounded(child, SCHED_KILL_GRACE);
        if !exited_in_grace {
            let _ = child.kill();
            // Bounded, evented reap. A SIGKILL'd scheduler normally exits
            // <<1s — post-crash bypass keeps it CFS-schedulable and it is
            // not held in the kernel disable (see `SCHED_REAP_TIMEOUT`).
            // The bound caps the rare case where the process can't take its
            // pending SIGKILL promptly; the VM reboot below reaps any
            // straggler, so cap the wait rather than risk blocking teardown.
            if !reap_child_bounded(child, SCHED_REAP_TIMEOUT) {
                tracing::warn!(
                    ?SCHED_REAP_TIMEOUT,
                    "scheduler did not exit within the reap bound after SIGKILL \
                     (still uninterruptible — unexpected); leaving it for VM reboot to reap"
                );
            }
        }
        if let Some(ref log_path) = sched_log_path {
            dump_sched_output(log_path);
        }
    }
    dump_staged_scheduler_logs();
    exec_shell_script("/sched_disable");

    // Phase 6b: probe finalisation. Now that the scheduler is
    // killed and `/sched_disable` has run, the kernel's
    // `scx_disable_irq_workfn` path runs `scx_claim_exit` which
    // fires `trace_sched_ext_exit`. The probe's tp_btf listener is
    // STILL attached at this point because
    // [`crate::test_support::probe::publish_result_and_collect`]
    // stashed the probe stop+handle into a deferred slot rather
    // than detaching at end-of-dispatch. Draining now means the
    // trigger event lands in the ring buffer, the BSS latch flips,
    // the probe poll loop sees `ktstr_err_exit_detected != 0`, and
    // the readout phase stitches the kprobe events that fired
    // during the actual stall window.
    //
    // The drain is bounded internally (5 s wait for
    // `/sys/kernel/sched_ext/state == disabled`, plus a one-shot
    // `rb.poll(100 ms)` final ringbuf drain inside the probe loop
    // when `bss_triggered` is observed); a non-responding kernel
    // cannot stall teardown. When no probes were stashed
    // (single-phase ctor path or EEVDF runs), the call is a no-op.
    crate::test_support::finalize_probe_after_unwind();

    // Stop remaining background threads.
    if let Some(ref stop) = vc_poll_stop {
        stop.store(true, Ordering::Release);
    }
    stats_relay_stop.signal_stop();

    // Flush COM1 trace data before reboot. The reader thread runs on
    // a poll(POLLIN, 200ms) cadence over a non-blocking trace_pipe fd
    // (see start_trace_pipe), so setting `stop` is what bounds
    // `handle.join()` — the thread observes the flag at the next poll
    // wake and enters its 5s drain window. Effective shutdown latency
    // is up to ~5.2s in the worst case: the 200ms poll cadence elapses
    // before the thread notices the stop flag, then the 5s drain
    // deadline begins. Disabling the tracepoint and writing 0 to
    // `tracing_on` first quiesces the producer side so the drain
    // window terminates promptly: no new events are recorded into the
    // ring buffer, the reader sees POLLIN until the buffer is empty,
    // then poll returns 0 each cycle and the drain_deadline elapses
    // cleanly. Trace events arriving after the 5s deadline are dropped
    // by design — bounded drain is the explicit tradeoff that
    // guarantees cleanup completes (a faulty producer that never
    // pauses cannot wedge teardown).
    //
    // tracing_on=0 alone does NOT wake a trace_pipe reader stuck at
    // `iter->pos == 0` — the kernel wake fires `ring_buffer_wake_waiters`
    // but the trace_pipe wait uses `wait_pipe_cond` (not
    // `rb_wait_once`), and that condition only flips when `iter->closed`
    // or `iter->wait_index` change. The non-blocking + poll design
    // sidesteps this by never blocking in the kernel wait at all.
    // Tier-2 (best-effort ftrace dump): if a sched_ext exit dump started
    // streaming this run, hold the dump tracepoint open until the reader
    // has forwarded its end-marker to COM1 (or the bound elapses) BEFORE
    // disabling it below, so a fast teardown does not disable the
    // tracepoint mid-emit. Only paid when a dump is in flight — clean runs
    // never start one. Returns immediately for a small dump; the bound
    // (`SCX_DUMP_CAPTURE_TIMEOUT`) caps the wait for a LARGE dump, whose
    // per-task content can take tens of seconds to forward over the slow
    // PIO COM1 UART. On that bound this ftrace copy is truncated — but the
    // full dump content is captured independently via the scheduler's
    // stderr log (`dump_sched_output`; scx_utils reads the same kernel
    // `ei->dump`) over the fast bulk port, which is the authoritative
    // copy. Best-effort, not lossless.
    if scx_dump_started_latch().is_set()
        && !scx_dump_complete_latch().wait_timeout(SCX_DUMP_CAPTURE_TIMEOUT)
    {
        tracing::warn!(
            ?SCX_DUMP_CAPTURE_TIMEOUT,
            "sched_ext exit dump did not reach its end-marker within the capture bound \
             before tracepoint teardown; the rendered dump may be truncated"
        );
    }
    let _ = fs::write(TRACE_SCHED_EXT_DUMP_ENABLE, "0");
    if let Some(ref stop) = trace_stop {
        stop.store(true, Ordering::Release);
    }
    let _ = fs::write(TRACE_TRACING_ON, "0");
    if let Some(handle) = trace_handle {
        let _ = handle.join();
    }
    if let Ok(com1) = fs::OpenOptions::new().write(true).open(COM1) {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::tcdrain(com1.as_raw_fd());
        }
    }

    // Phase 7: Exit.
    // Push buffered stdout/stderr bytes into the pipe write ends so
    // the bulk-port forwarder threads can ship them before reboot.
    // After stdio redirect, fd 1 / fd 2 are pipe write ends
    // (not the COM2 UART) so `tcdrain(1)` would return ENOTTY
    // silently — `flush()` is the equivalent for pipes. The
    // forwarder threads are not joined before `force_reboot`; bytes
    // still resident in the pipe buffer at reboot time are lost
    // (see the queue task for forwarder-join plumbing).
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    // Write exit code via the typed guest API on the bulk data
    // port. The legacy COM2 `SENTINEL_EXIT_PREFIX` fallback is gone
    // — bulk-port backpressure guarantees delivery and the host's
    // `collect_results` walks `guest_messages` for a binary
    // `MSG_TYPE_EXIT` frame as the sole authoritative source.
    crate::vmm::guest_comms::send_exit(code as i32);

    // Drain COM2 UART for any panic-hook bytes that may still be
    // in flight (the panic hook is the one remaining COM2 writer).
    // tcdrain is synchronous on the vCPU exit: when it returns,
    // every byte is already in the host's COM2 writer Vec.
    if let Ok(com2) = fs::OpenOptions::new().write(true).open(COM2) {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::tcdrain(com2.as_raw_fd());
        }
    }

    force_reboot()
}

/// Maximum bytes per `MsgType::Stdout` / `MsgType::Stderr` TLV
/// chunk emitted by the pipe forwarder threads. 4 KiB matches a
/// page-size pipe read; well under the host-side per-frame cap
/// [`crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD`] so a chunk fits
/// comfortably in one frame even with the 16-byte header.
const STDIO_CHUNK_BYTES: usize = 4 * 1024;

/// Redirect stdout and stderr through bulk-port forwarder threads.
///
/// Pre-bulk-port-migration: dup2'd `/dev/ttyS1` over fd 1 and fd 2 so
/// every `println!` / `eprintln!` reached the host as a stream of
/// COM2 bytes.  The bulk-port migration replaces COM2 with one
/// `MsgType::Stdout` / `MsgType::Stderr` TLV frame per chunk:
///
///   1. Open a pair of `pipe(2)` pipes (one for stdout, one for
///      stderr).
///   2. `dup2` each pipe's write end over fd 1 / fd 2 so every
///      `println!` / `eprintln!` lands in the pipe.
///   3. Spawn one reader thread per pipe.  Each thread reads up to
///      [`STDIO_CHUNK_BYTES`] at a time from the pipe's read end and
///      ships the chunk via
///      [`crate::vmm::guest_comms::send_stdout_chunk`] /
///      [`crate::vmm::guest_comms::send_stderr_chunk`].
///
/// The threads are detached: they exit cleanly when fd 1 / fd 2 are
/// closed (process exit / `force_reboot`) because the read end then
/// returns EOF.
///
/// Panic diagnostics still go to COM2 — the panic hook in
/// [`ktstr_guest_init`] writes directly to `/dev/ttyS1` because the
/// hook cannot block on virtio backpressure.  Every other guest
/// stream now travels over the bulk port.
///
/// On any pipe / dup2 / thread-spawn failure the function logs via
/// the tracing subscriber (which writes to stderr; fd 2 is still
/// attached to the kernel console at the failure point, so the
/// operator sees the misroute) and returns — stdout/stderr stay
/// attached to whatever fd they pointed at on entry.
fn redirect_stdio_to_bulk_port() {
    use std::io::Read;
    use std::os::unix::io::{AsRawFd, FromRawFd};

    fn make_pipe() -> Option<(std::fs::File, std::fs::File)> {
        let mut fds = [0i32; 2];
        // SAFETY: `fds` is a valid `&mut [i32; 2]`; `pipe(2)` writes
        // exactly two file descriptors on success.  Passing `O_CLOEXEC`
        // would belong on `pipe2`, but we deliberately want the pipe
        // ends to survive across any forks the test may perform — the
        // dup2'd write end carries fd 1 / fd 2 across exec/fork, which
        // is the entire point.
        let r = unsafe { libc::pipe(fds.as_mut_ptr()) };
        if r < 0 {
            return None;
        }
        // SAFETY: `pipe(2)` just returned with the two fds populated.
        // `from_raw_fd` takes ownership of each side; both close on
        // drop.  Held by `File` for the natural Read/Write impls.
        let read_end = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let write_end = unsafe { std::fs::File::from_raw_fd(fds[1]) };
        Some((read_end, write_end))
    }

    fn spawn_forwarder(mut read_end: std::fs::File, name: &'static str, sender: fn(&[u8]) -> bool) {
        let _ = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let mut buf = [0u8; STDIO_CHUNK_BYTES];
                loop {
                    match read_end.read(&mut buf) {
                        Ok(0) => break, // EOF — fd 1/2 closed.
                        Ok(n) => {
                            // Fire-and-forget.  `send_*_chunk`
                            // returns false when the bulk port is
                            // not yet ready; bytes emitted before
                            // the multiport handshake completes are
                            // dropped.  Same caveat as the prior
                            // COM2 path's pre-handshake byte loss.
                            let _ = sender(&buf[..n]);
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(_) => break,
                    }
                }
            });
    }

    let Some((stdout_r, stdout_w)) = make_pipe() else {
        tracing::error!("ktstr-init: redirect_stdio_to_bulk_port: pipe(stdout) failed");
        return;
    };
    let Some((stderr_r, stderr_w)) = make_pipe() else {
        tracing::error!("ktstr-init: redirect_stdio_to_bulk_port: pipe(stderr) failed");
        return;
    };

    // Capture errno via `last_os_error` BEFORE any subsequent libc
    // call: errno is per-thread but every libc call may clobber it.
    let (rc1, err1, rc2, err2) = unsafe {
        let r1 = libc::dup2(stdout_w.as_raw_fd(), 1);
        let e1 = std::io::Error::last_os_error();
        let r2 = libc::dup2(stderr_w.as_raw_fd(), 2);
        let e2 = std::io::Error::last_os_error();
        (r1, e1, r2, e2)
    };
    // The dup2 above duplicated each pipe's write end onto fd 1 / fd 2;
    // the originals (`stdout_w` / `stderr_w`) close on this scope's
    // exit.  Without that close, the read end of each pipe would see
    // EOF only after the test process holding fd 1 / fd 2 also dropped
    // those file descriptors — but we want the EOF condition to fire
    // when fd 1 / fd 2 reach their natural close-on-exit, not when
    // some other holder of `stdout_w` closes too.  Letting the
    // originals drop here is correct because `dup2` increments the
    // file's refcount.
    if rc1 < 0 {
        tracing::error!(err = %err1, "ktstr-init: redirect_stdio_to_bulk_port: dup2(stdout) failed");
    }
    if rc2 < 0 {
        tracing::error!(err = %err2, "ktstr-init: redirect_stdio_to_bulk_port: dup2(stderr) failed");
    }

    spawn_forwarder(stdout_r, "ktstr-stdout-fwd", |b| {
        crate::vmm::guest_comms::send_stdout_chunk(b)
    });
    spawn_forwarder(stderr_r, "ktstr-stderr-fwd", |b| {
        crate::vmm::guest_comms::send_stderr_chunk(b)
    });
}

/// Check kernel cmdline for KTSTR_MODE=shell.
fn shell_mode_requested() -> bool {
    fs::read_to_string("/proc/cmdline")
        .map(|c| cmdline_contains_token(&c, "KTSTR_MODE=shell"))
        .unwrap_or(false)
}

/// Check kernel cmdline for `KTSTR_MODE=disk_template`. The host
/// asserts this when booting a one-shot template-build VM (see
/// [`crate::vmm::disk_template`]).
fn disk_template_mode_requested() -> bool {
    fs::read_to_string("/proc/cmdline")
        .map(|c| cmdline_contains_token(&c, "KTSTR_MODE=disk_template"))
        .unwrap_or(false)
}

/// Pure-function cmdline-token check, factored out of
/// [`shell_mode_requested`] / [`disk_template_mode_requested`] so
/// the precedence-and-multiplicity behavior can be tested without
/// mocking `/proc/cmdline`. Whitespace-separated, exact match (the
/// kernel passes cmdline tokens verbatim — no quoting, no escapes).
fn cmdline_contains_token(cmdline: &str, token: &str) -> bool {
    cmdline.split_whitespace().any(|s| s == token)
}

/// Disk-template build dispatch: exec `/bin/mkfs.btrfs /dev/vda`
/// (the host packed `mkfs.btrfs` into the initramfs at this path),
/// wait for it, return its exit code so the caller emits the exit
/// sentinel on COM2 before rebooting. Returns `0` on success and
/// the binary's exit code (or `1` on spawn failure) otherwise.
///
/// The disk image at `/dev/vda` is the host-side staging file
/// (sparse, sized to the requested capacity); after this function
/// returns and the VM reboots, the host's [`crate::vmm::disk_template::store_atomic`]
/// publishes the now-formatted image into the cache.
///
/// The host never execs `mkfs.btrfs` against a real backing file —
/// driving the format through this guest-side dispatch keeps the
/// kernel under test as the on-disk-format authority, so any btrfs
/// feature regression in that kernel surfaces as a guest format
/// failure here instead of as a host/guest mkfs disagreement that
/// would slip past testing.
fn run_disk_template_mode() -> i32 {
    redirect_stdio_to_bulk_port();
    // The mkfs.btrfs binary is packed at `bin/mkfs.btrfs` by
    // [`crate::vmm::disk_template::build_template_via_vm`] via
    // `include_files`; that function — not `ensure_template` — is
    // the host-side site that assembles the template-VM
    // initramfs.
    const MKFS: &str = "/bin/mkfs.btrfs";
    // `-f` forces overwrite of any existing signature so a leftover
    // ext4 magic from a host that recycled the staging file does
    // not block formatting. mkfs runs verbose (no `--quiet`) so its
    // own diagnostics land in the guest stderr, which the host's
    // publish gate surfaces on a magic-check failure. `/dev/vda` is
    // the singleton virtio-blk device the host attached.
    //
    // No `--metadata DUP` override: btrfs picks DUP metadata by
    // default on a single-device fs, which is the desired
    // production format. The 256 MiB minimum capacity (see
    // VIRTIO_BLK_DEFAULT_CAPACITY_BYTES doc) accommodates DUP.
    tracing::info!(mkfs = MKFS, target = "/dev/vda", "running mkfs.btrfs");
    // SIGCHLD is `SIG_IGN` for the rest of this process (installed by
    // [`ktstr_guest_init`] for zombie prevention). `Command::output()`
    // calls `waitpid(2)` internally; under `SIG_IGN` the kernel
    // auto-reaps the child before `waitpid` runs, so the syscall
    // returns `ECHILD`, the std-lib maps it to
    // `Err(io::Error::ECHILD)`, and the original `match` branch fell
    // into the `Err(_) => 1` arm — surfacing a fixed `1` exit code for
    // every successful `mkfs.btrfs` run. The host would then see
    // "template build failed" for a perfectly formatted image. Restore
    // `SIG_DFL` for the closure's lifetime so `waitpid` reaps and
    // reports the real status; the post-closure restore re-installs
    // `SIG_IGN` for any future child this process spawns.
    //
    // The build VM's stdio forwarders are NOT joined before
    // `force_reboot`'s `RB_AUTOBOOT`, so bytes still in the
    // stdout/stderr pipes at reboot are lost (see the dispatch site),
    // and the host does not drain the bulk-port stdout/stderr frames
    // for a fast-exiting build VM either — only the EXIT frame and COM2
    // survive. So capture mkfs's output in-process via
    // `Command::output()` and surface it on the two channels that DO
    // survive: the exit code (a pass/no-op verdict) and COM2
    // (`/dev/ttyS1`, the full text) — so the publish gate can show WHY
    // a build produced no filesystem.
    let mut diag = format!(
        "MKFS_DIAG /dev/vda exists={} /sys/class/block/vda exists={}\n",
        std::path::Path::new("/dev/vda").exists(),
        std::path::Path::new("/sys/class/block/vda").exists(),
    );
    match std::fs::metadata(MKFS) {
        Ok(m) => diag.push_str(&format!("MKFS_DIAG {MKFS} size={}\n", m.len())),
        Err(e) => diag.push_str(&format!("MKFS_DIAG {MKFS} metadata failed: {e}\n")),
    }
    let output = with_sigchld_default(|| Command::new(MKFS).args(["-f", "/dev/vda"]).output());
    let mut code = match output {
        Ok(o) => {
            diag.push_str(&format!(
                "MKFS_DIAG exit={:?}\nMKFS_STDOUT:\n{}\nMKFS_STDERR:\n{}\n",
                o.status.code(),
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr),
            ));
            o.status.code().unwrap_or(1)
        }
        Err(e) => {
            diag.push_str(&format!("MKFS_DIAG mkfs spawn failed: {e}\n"));
            1
        }
    };
    const BTRFS_DEV_MAGIC: u64 = 0x4D5F_5366_5248_425F;
    let magic = read_dev_vda_magic();
    let magic_present = magic.as_ref().is_ok_and(|&m| m == BTRFS_DEV_MAGIC);
    diag.push_str(&format!(
        "MKFS_DIAG in-guest /dev/vda magic@0x10040 = {}\n",
        match &magic {
            Ok(m) => format!("0x{m:016x}"),
            Err(e) => format!("read failed: {e}"),
        },
    ));
    // Only sync + run the no-op verdict on a clean mkfs exit; a non-zero
    // mkfs exit propagates through the host's clean-exit gate
    // (build_template_via_vm), which surfaces mkfs's stderr tail.
    if code == 0 {
        if let Some(flush_err) = flush_template_disk() {
            diag.push_str(&format!("MKFS_DIAG flush: {flush_err}\n"));
        }
        // A clean mkfs exit that leaves /dev/vda WITHOUT the btrfs magic
        // — in the guest's own page-cache view — is a silent no-op.
        // Override to a distinct exit code so the host's clean-exit check
        // reports it (the exit code is the only build-VM channel proven
        // to survive force_reboot, so the verdict lands even if the COM2
        // diag below is lost). Distinguishes a guest-side no-op from a
        // host-side write loss (which Layer B's host-magic check catches).
        if !magic_present {
            diag.push_str(
                "MKFS_DIAG VERDICT: mkfs exited 0 but /dev/vda has no btrfs magic \
                 in-guest — silent no-op (exit overridden to 64)\n",
            );
            code = 64;
        }
    }
    // Mirror the diag to COM2 (`/dev/ttyS1`) — the host's
    // fault-diagnostic serial channel (the panic hook uses it),
    // captured into `result.output` via `com2_bytes` in
    // `collect_results`. COM2 is synchronous and survives the immediate
    // `force_reboot`, unlike the bulk-port stdio forwarders.
    if let Ok(mut com2) = std::fs::OpenOptions::new().write(true).open("/dev/ttyS1") {
        use std::io::Write;
        let _ = com2.write_all(diag.as_bytes());
        let _ = com2.flush();
    }
    code
}

/// Read the 8-byte little-endian btrfs superblock magic at offset
/// `0x10040` from `/dev/vda` as the guest sees it through the
/// block-device page cache. Returns the raw `u64` (the caller compares
/// it against the expected magic for the no-op verdict — a value
/// compare, not a fragile string compare) or the io::Error if the
/// device can't be opened/read. The build-VM diagnostic formats it to
/// distinguish "mkfs wrote the magic but the host backing lost it" from
/// "mkfs reported success without writing anything".
fn read_dev_vda_magic() -> std::io::Result<u64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open("/dev/vda")?;
    f.seek(SeekFrom::Start(0x1_0040))?;
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Flush `/dev/vda` to the host backing before `force_reboot`'s
/// `RB_AUTOBOOT` (which performs no sync). Belt-and-braces:
/// `bdev_release` already `sync_blockdev`s the page cache when the
/// last opener closes (block/bdev.c), so a normal buffered write is
/// already durable on the host by the time mkfs exits — but a guest
/// path that bypasses the bdev page cache (mmap, an O_DIRECT skip)
/// would otherwise leave writes unflushed at reboot, and `RB_AUTOBOOT`
/// abandons them (virtio_blk has no `.shutdown` op that drains IO).
///
/// Returns `None` on success or `Some(msg)` describing the failure.
/// The caller folds the message into the COM2-mirrored diag rather
/// than logging it: build-VM stderr is dropped at `force_reboot` (the
/// bulk-port forwarders are not joined before the reboot), so a
/// `tracing::error!` here would never reach the host.
fn flush_template_disk() -> Option<String> {
    match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/vda")
    {
        Ok(f) => f
            .sync_all()
            .err()
            .map(|e| format!("sync_all(/dev/vda) failed: {e}")),
        Err(e) => Some(format!("open(/dev/vda) for flush failed: {e}")),
    }
}

/// Read /exec_cmd from the initramfs if present.
/// The host writes this file via build_suffix when --exec is used.
fn shell_exec_cmd() -> Option<String> {
    fs::read_to_string("/exec_cmd")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Extract a KEY=value pair from the kernel cmdline.
fn cmdline_val(key: &str) -> Option<String> {
    let cmdline = fs::read_to_string("/proc/cmdline").ok()?;
    let prefix = format!("{key}=");
    cmdline
        .split_whitespace()
        .find_map(|s| s.strip_prefix(&prefix))
        .map(|s| s.to_string())
}

/// Build PATH with /include-files directories containing executables.
///
/// Walks /include-files recursively, collects directories that contain
/// at least one executable file, prepends them all to PATH. This makes
/// included binaries runnable by name regardless of subdirectory depth
/// (e.g. `-i ../scx/target/release` → `scx_cake` works directly).
fn build_include_path() -> String {
    use std::collections::BTreeSet;
    use std::os::unix::fs::PermissionsExt;
    let include_dir = std::path::Path::new("/include-files");
    let mut dirs = BTreeSet::new();

    if include_dir.is_dir() {
        for entry in walkdir::WalkDir::new(include_dir).follow_links(true) {
            let Ok(entry) = entry else { continue };
            if entry.file_type().is_file()
                && entry
                    .metadata()
                    .is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
                && let Some(parent) = entry.path().parent()
            {
                dirs.insert(parent.to_string_lossy().to_string());
            }
        }
    }

    let mut path_parts: Vec<String> = dirs.into_iter().collect();
    path_parts.push("/bin".to_string());
    path_parts.join(":")
}

/// Redirect stdin, stdout, and stderr to the given device with O_RDWR.
///
/// Shell mode needs all three fds on the console device: stdin for
/// reading input, stdout/stderr for writing output.
///
/// `dup2` failures are logged via `tracing::error!`. A failing `dup2`
/// leaves the target fd unchanged, so the diagnostic still reaches
/// the pre-redirect stderr (kernel console / COM1) through the
/// tracing subscriber and the operator sees the misroute rather than
/// the failing path silently writing to a wrong device.
fn redirect_all_stdio_to(path: &str) {
    use std::os::unix::io::AsRawFd;

    let Ok(dev) = fs::OpenOptions::new().read(true).write(true).open(path) else {
        return;
    };
    let fd = dev.as_raw_fd();
    // Capture errno per call before the next libc call clobbers
    // it. Run all three syscalls sequentially without aborting on
    // a partial failure — fd 0 redirect failing should not stop us
    // from at least getting stdout/stderr onto the console.
    let (rc0, err0, rc1, err1, rc2, err2) = unsafe {
        let r0 = libc::dup2(fd, 0);
        let e0 = std::io::Error::last_os_error();
        let r1 = libc::dup2(fd, 1);
        let e1 = std::io::Error::last_os_error();
        let r2 = libc::dup2(fd, 2);
        let e2 = std::io::Error::last_os_error();
        (r0, e0, r1, e1, r2, e2)
    };
    if rc0 < 0 {
        tracing::error!(path, err = %err0, "ktstr-init: redirect_all_stdio_to: dup2(stdin) failed");
    }
    if rc1 < 0 {
        tracing::error!(path, err = %err1, "ktstr-init: redirect_all_stdio_to: dup2(stdout) failed");
    }
    if rc2 < 0 {
        tracing::error!(path, err = %err2, "ktstr-init: redirect_all_stdio_to: dup2(stderr) failed");
    }
}

/// Select the console device for shell mode.
/// Prefers /dev/hvc0 (virtio-console) when available, falls back to COM2.
fn shell_console_device() -> &'static str {
    if Path::new(HVC0).exists() { HVC0 } else { COM2 }
}

/// Mount devpts at /dev/pts for PTY allocation.
///
/// Required before `openpty()` — the C library opens `/dev/ptmx` and
/// the slave device lives under `/dev/pts/N`.
fn mount_devpts() {
    mkdir_p("/dev/pts");
    let result = mount(
        Some("devpts"),
        "/dev/pts",
        Some("devpts"),
        MsFlags::empty(),
        None::<&str>,
    );
    if let Err(e) = result {
        tracing::error!(err = %e, "ktstr-init: mount devpts on /dev/pts failed");
    }
}

/// Spawn busybox sh with a PTY as its controlling terminal.
///
/// Allocates a PTY pair via `openpty()`, spawns sh with the slave as
/// stdin/stdout/stderr and `setsid` + `TIOCSCTTY` in `pre_exec` so sh
/// gets a controlling terminal (job control). The parent proxies data
/// between COM2 (fd 0/1) and the PTY master until the child exits.
///
/// SIGCHLD remains SIG_IGN (set earlier for zombie prevention), so
/// waitpid returns ECHILD after the kernel auto-reaps the child.
/// This is expected and suppressed.
fn spawn_shell_with_pty() {
    let pty = match openpty(None, None) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(err = %e, "ktstr-init: openpty failed");
            return;
        }
    };

    let slave_fd = pty.slave.as_raw_fd();

    // Set PTY size from host terminal dimensions passed via cmdline.
    if let (Some(cols), Some(rows)) = (cmdline_val("KTSTR_COLS"), cmdline_val("KTSTR_ROWS"))
        && let (Ok(cols), Ok(rows)) = (cols.parse::<u16>(), rows.parse::<u16>())
    {
        let ws = libc::winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe {
            libc::ioctl(slave_fd, libc::TIOCSWINSZ, &ws);
        }
    }

    // Set terminal type from host. Default to "linux" if not passed.
    let term = cmdline_val("KTSTR_TERM").unwrap_or_else(|| "linux".to_string());
    let colorterm = cmdline_val("KTSTR_COLORTERM");

    let child = unsafe {
        let mut cmd = Command::new("/bin/busybox");
        cmd.arg("sh")
            .env("TERM", &term)
            .env("PS1", "\x1b[2m^Ax=quit\x1b[0m \\w # ");
        if let Some(ref ct) = colorterm {
            cmd.env("COLORTERM", ct);
        }
        cmd.stdin(Stdio::from(OwnedFd::from_raw_fd(libc::dup(slave_fd))))
            .stdout(Stdio::from(OwnedFd::from_raw_fd(libc::dup(slave_fd))))
            .stderr(Stdio::from(OwnedFd::from_raw_fd(libc::dup(slave_fd))))
            .pre_exec(move || {
                // Create a new session so sh becomes session leader.
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                // Acquire a controlling terminal.
                if libc::ioctl(slave_fd, libc::TIOCSCTTY, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            })
            .spawn()
    };

    // Close slave in parent — the child has its own copies.
    drop(pty.slave);

    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(err = %e, "ktstr-init: spawn shell failed");
            return;
        }
    };

    let child_pid = child.id();

    // Set COM2 serial (fd 0) to raw mode so the kernel line discipline
    // passes bytes through without processing. Without this, special
    // characters like tab (0x09) are consumed by the line discipline
    // instead of being forwarded through the proxy to the PTY.
    let stdin_fd = unsafe { BorrowedFd::borrow_raw(0) };
    if let Ok(mut termios) = tcgetattr(stdin_fd) {
        cfmakeraw(&mut termios);
        let _ = tcsetattr(stdin_fd, SetArg::TCSANOW, &termios);
    }

    // Proxy between COM2 (fd 0 for input, fd 1 for output) and PTY master.
    proxy_serial_pty(&pty.master, child_pid);

    // SIGCHLD is SIG_IGN so the kernel auto-reaps the child. waitpid
    // returns ECHILD — expected, not an error.
    match child.wait() {
        Ok(status) => {
            tracing::debug!(?status, "shell exited");
        }
        Err(e) if e.raw_os_error() == Some(libc::ECHILD) => {}
        Err(e) => {
            tracing::warn!(err = %e, "ktstr-init: wait for shell failed");
        }
    }

    // No guest-side exit message — the host prints "Connection to VM
    // closed." after the VM shuts down. Printing here too would
    // duplicate it, and writing to COM2 in raw mode after PTY teardown
    // leaks garbage bytes.
}

/// Proxy data between COM2 serial (fd 0/1) and a PTY master fd.
///
/// Uses poll(2) to multiplex reads from both fds. Exits when the PTY
/// master returns EOF (child closed the slave side) or the child process
/// no longer exists.
fn proxy_serial_pty(master: &OwnedFd, child_pid: u32) {
    let stdin_fd = unsafe { BorrowedFd::borrow_raw(0) };
    let stdout_fd = unsafe { BorrowedFd::borrow_raw(1) };
    let master_fd = master.as_fd();

    let mut buf = [0u8; 4096];

    loop {
        let mut pollfds = [
            PollFd::new(stdin_fd, PollFlags::POLLIN),
            PollFd::new(master_fd, PollFlags::POLLIN),
        ];

        match poll(&mut pollfds, PollTimeout::from(200u16)) {
            Ok(0) => {
                // Timeout — check if child is still alive.
                if !Path::new(&format!("/proc/{child_pid}")).exists() {
                    break;
                }
                continue;
            }
            Ok(_) => {}
            Err(nix::errno::Errno::EINTR) => continue,
            Err(_) => break,
        }

        // Serial input -> PTY master (user typing).
        if let Some(revents) = pollfds[0].revents() {
            if revents.contains(PollFlags::POLLIN) {
                match nix::unistd::read(stdin_fd, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let _ = nix::unistd::write(master_fd, &buf[..n]);
                    }
                    Err(nix::errno::Errno::EINTR) => {}
                    Err(_) => break,
                }
            }
            if revents.intersects(PollFlags::POLLERR | PollFlags::POLLHUP) {
                break;
            }
        }

        // PTY master -> serial output (shell output).
        // Check POLLHUP/POLLERR before POLLIN: when the shell exits,
        // both flags can arrive in the same poll iteration. Reading
        // after the slave closes produces partial/garbage bytes from
        // the PTY teardown (manifests as a raw U+FFFD on the terminal).
        if let Some(revents) = pollfds[1].revents() {
            if revents.intersects(PollFlags::POLLERR | PollFlags::POLLHUP) {
                break;
            }
            if revents.contains(PollFlags::POLLIN) {
                match nix::unistd::read(master_fd, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let _ = nix::unistd::write(stdout_fd, &buf[..n]);
                    }
                    Err(nix::errno::Errno::EINTR) => {}
                    Err(_) => break,
                }
            }
        }
    }
}

/// Print the topology line for the shell MOTD.
///
/// Parses KTSTR_TOPO=N,L,C,T from /proc/cmdline (passed by the host).
/// Falls back to counting online CPUs via /sys/devices/system/cpu/online.
fn print_topology_line() {
    if let Some((n, l, c, t)) = parse_topo_from_cmdline() {
        let total = l * c * t;
        if n > 1 {
            println!(
                "  topology:  {n} NUMA nodes, {l} LLC{}, {c} core{}, {t} thread{} ({total} vCPU{})",
                if l == 1 { "" } else { "s" },
                if c == 1 { "" } else { "s" },
                if t == 1 { "" } else { "s" },
                if total == 1 { "" } else { "s" },
            );
        } else {
            println!(
                "  topology:  {l} LLC{}, {c} core{}, {t} thread{} ({total} vCPU{})",
                if l == 1 { "" } else { "s" },
                if c == 1 { "" } else { "s" },
                if t == 1 { "" } else { "s" },
                if total == 1 { "" } else { "s" },
            );
        }
    } else if let Some(count) = count_online_cpus() {
        println!(
            "  topology:  {count} vCPU{}",
            if count == 1 { "" } else { "s" }
        );
    }
}

/// Parse KTSTR_TOPO=N,L,C,T from /proc/cmdline.
fn parse_topo_from_cmdline() -> Option<(u32, u32, u32, u32)> {
    let val = cmdline_val("KTSTR_TOPO")?;
    let parts: Vec<&str> = val.split(',').collect();
    if parts.len() != 4 {
        return None;
    }
    let n: u32 = parts[0].parse().ok()?;
    let l: u32 = parts[1].parse().ok()?;
    let c: u32 = parts[2].parse().ok()?;
    let t: u32 = parts[3].parse().ok()?;
    Some((n, l, c, t))
}

#[cfg(feature = "wprof")]
/// Spawn `/bin/wprof` in a background thread if the host set
/// `KTSTR_WPROF_ARGS` on the kernel cmdline. Returns a join handle
/// whose `.join()` yields `Some(Vec<u8>)` (the `.pb` trace bytes)
/// on success, or `None` on failure / no-op.
///
/// The spawned thread:
/// 1. Parses `KTSTR_WPROF_ARGS` from `/proc/cmdline`
/// 2. Runs `/bin/wprof <args> -T /tmp/wprof.pb -D /tmp/wprof.data`
/// 3. Waits for the process to exit
/// 4. Reads `/tmp/wprof.pb` and returns the bytes
///
/// If `KTSTR_WPROF_ARGS` is absent or `/bin/wprof` doesn't exist,
/// returns `None` (no thread spawned, no-op). The caller joins the
/// handle after the test workload dispatch returns and ships the
/// bytes via [`crate::vmm::guest_comms::send_wprof_trace`].
fn spawn_wprof_if_configured() -> Option<std::thread::JoinHandle<Option<Vec<u8>>>> {
    let args_str = cmdline_val("KTSTR_WPROF_ARGS")?;
    let wprof_bin = std::path::Path::new("/bin/wprof");
    if !wprof_bin.exists() {
        tracing::warn!("KTSTR_WPROF_ARGS set but /bin/wprof missing from initramfs");
        return None;
    }
    Some(
        std::thread::Builder::new()
            .name("wprof-capture".into())
            .spawn(move || {
                // Host encodes args with ASCII Unit Separator (\x1F)
                // via `WprofConfig::args_cmdline` because kernel
                // cmdline tokenization would truncate a space-joined
                // value at the first space. Split on the same
                // delimiter here to recover the per-arg vec.
                let mut cmd_args: Vec<String> = args_str.split('\x1f').map(String::from).collect();
                cmd_args.extend([
                    "-T".to_string(),
                    "/tmp/wprof.pb".to_string(),
                    "-D".to_string(),
                    "/tmp/wprof.data".to_string(),
                ]);
                tracing::debug!(args = ?cmd_args, "spawning /bin/wprof");
                let status = std::process::Command::new("/bin/wprof")
                    .args(&cmd_args)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::inherit())
                    .status();
                match status {
                    Ok(s) if s.success() => match std::fs::read("/tmp/wprof.pb") {
                        Ok(bytes) if !bytes.is_empty() => {
                            tracing::debug!(pb_bytes = bytes.len(), "wprof trace captured");
                            Some(bytes)
                        }
                        Ok(_) => {
                            tracing::warn!("wprof exited OK but /tmp/wprof.pb is empty");
                            None
                        }
                        Err(e) => {
                            tracing::warn!(%e, "read /tmp/wprof.pb after successful run");
                            None
                        }
                    },
                    Ok(s) => {
                        tracing::warn!(exit = s.code(), "wprof exited non-zero");
                        None
                    }
                    Err(e) => {
                        tracing::warn!(%e, "spawn /bin/wprof failed");
                        None
                    }
                }
            })
            .expect("spawn wprof-capture thread"),
    )
}

/// Poll for the virtio-console bulk port and deliver the
/// KERN_ADDRS + SYS_RDY frames to the host.
///
/// The kernel virtio_console driver's multiport handshake
/// (DEVICE_READY → PORT_ADD → PORT_READY → PORT_OPEN, see
/// `drivers/char/virtio_console.c`) completes asynchronously.
/// `/dev/vport0p1` is created by `add_port` when PORT_ADD arrives
/// (via `device_create` + devtmpfs `wait_for_completion`);
/// `host_connected` flips true only when PORT_OPEN arrives later in
/// the same `control_work_handler` batch. So a writev between port
/// creation and host_connected blocks inside `wait_port_writable`
/// (see `wait_event_freezable` at `include/linux/wait.h` — no
/// timeout argument). The loop polls the device-node existence at
/// 100 ms cadence with a wall-clock deadline so blocking writev
/// time counts against the budget. The deadline is captured INSIDE
/// the function so guest-init setup (mounts, kallsyms reads) does
/// not eat the handshake's budget.
///
/// Both KERN_ADDRS and SYS_RDY are required for the host. KERN_ADDRS
/// is latched: once a `send_kern_addrs` call returns true the loop
/// skips it on subsequent iterations. The early-return condition is
/// `kern_addrs_sent && send_sys_rdy()` — we never exit until BOTH
/// have been delivered (a successful sys_rdy on the re-opened FD
/// after a kern_addrs failure must not leave kern_addrs unsent,
/// because the host's KERN_ADDRS arm is the only virt-KASLR
/// publisher on aarch64; see `src/vmm/freeze_coord/dispatch.rs`'s
/// KERN_ADDRS handler).
///
/// Host idempotency for KERN_ADDRS retries (matters when the latch
/// is reset by a failed write that cleared the cached FD):
/// `kern_phys_base` uses `.store(Release)` (overwrites every
/// CRC-valid frame) and `kern_virt_kaslr` uses CAS-once. The
/// payload bytes are identical across retries (built once from
/// `KernAddrs::new`), so repeated stores and a CAS-success-then-
/// no-op-on-equal-existing both produce the same final state.
///
/// On budget exhaustion the function emits a structured WARN with
/// fields `budget_ms`, `vcpus`, `elapsed_ms` (loop wall time),
/// `port_exists` (sampled once before WARN), and `kern_addrs_sent`.
/// The guest then continues — the host monitor's `data_valid` gate
/// keeps reads safe without SYS_RDY, and the freeze coordinator's
/// `Option::take` makes a late SYS_RDY harmless (fire-once). See
/// `doc/guide/src/troubleshooting.md#send_sys_rdy-timeout` for the
/// operator-facing diagnosis flow.
fn send_sys_rdy_with_retry(
    budget: std::time::Duration,
    vcpus: u32,
    kern_addrs: &crate::vmm::wire::KernAddrs,
    port_path: &std::path::Path,
) {
    let loop_t0 = std::time::Instant::now();
    let deadline = loop_t0 + budget;
    let mut kern_addrs_sent = false;
    loop {
        if port_path.exists() {
            if !kern_addrs_sent {
                kern_addrs_sent = crate::vmm::guest_comms::send_kern_addrs(kern_addrs);
            }
            if kern_addrs_sent && crate::vmm::guest_comms::send_sys_rdy() {
                return;
            }
        }
        if std::time::Instant::now() >= deadline {
            // Snapshot before WARN so the field reports the
            // last-attempt state, not a fresh stat that could
            // observe a port appearing in the gap between the
            // final loop iteration and the WARN call.
            let port_exists_snapshot = port_path.exists();
            tracing::warn!(
                budget_ms = budget.as_millis() as u64,
                vcpus,
                elapsed_ms = loop_t0.elapsed().as_millis() as u64,
                port_exists = port_exists_snapshot,
                kern_addrs_sent,
                "ktstr-init: send_sys_rdy failed within boot budget; \
                 see https://likewhatevs.github.io/ktstr/guide/troubleshooting.html#send_sys_rdy-timeout",
            );
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Count online CPUs from /sys/devices/system/cpu/online.
///
/// The file contains a range list like "0-3" or "0-1,3". Parse and
/// count individual CPUs.
fn count_online_cpus() -> Option<u32> {
    let content = fs::read_to_string("/sys/devices/system/cpu/online").ok()?;
    parse_online_cpus(&content)
}

/// Parse a cpulist string (kernel `/sys/.../online` format) and
/// return the total count of CPUs it covers. Comma-separated tokens,
/// each either a single index or a `start-end` inclusive range.
/// Returns `None` on any unparseable token, inverted range, or
/// completely empty content. The `sys_rdy` budget caller at
/// [`count_online_cpus`]'s primary use defaults to 1 vCPU on `None`
/// (safe degradation to the single-vCPU budget); the topology-print
/// caller skips the MOTD line instead of substituting a default.
fn parse_online_cpus(content: &str) -> Option<u32> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut count = 0u32;
    for range in trimmed.split(',') {
        if let Some((start, end)) = range.split_once('-') {
            let s: u32 = start.parse().ok()?;
            let e: u32 = end.parse().ok()?;
            count = count.checked_add(e.checked_sub(s)?.checked_add(1)?)?;
        } else {
            let _: u32 = range.parse().ok()?;
            count = count.checked_add(1)?;
        }
    }
    Some(count)
}

/// Print the include-files line for the shell MOTD.
///
/// Scans /include-files/ and lists each entry. Executable files
/// are marked with "(executable)".
fn print_includes_line() {
    let include_dir = Path::new("/include-files");
    if !include_dir.is_dir() {
        return;
    }
    let mut files: Vec<(String, bool)> = Vec::new();
    // Walk recursively to discover files in nested directories.
    for entry in walkdir::WalkDir::new(include_dir)
        .min_depth(1)
        .sort_by_file_name()
    {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(include_dir)
            .unwrap_or(entry.path());
        let name = rel.to_string_lossy().to_string();
        let executable = entry
            .metadata()
            .map(|m| {
                use std::os::unix::fs::PermissionsExt;
                m.permissions().mode() & 0o111 != 0
            })
            .unwrap_or(false);
        files.push((name, executable));
    }
    if files.is_empty() {
        return;
    }
    for (i, (name, executable)) in files.iter().enumerate() {
        let marker = if *executable { " (executable)" } else { "" };
        let path = format!("/include-files/{name}{marker}");
        if i == 0 {
            println!("  includes:  {path}");
        } else {
            println!("             {path}");
        }
    }
}

/// Mount essential filesystems.
fn mount_filesystems() {
    let mounts: &[(&str, &str, &str, bool)] = &[
        ("/proc", "proc", "proc", true),
        ("/sys", "sys", "sysfs", true),
        ("/dev", "dev", "devtmpfs", true),
        ("/sys/kernel/debug", "debugfs", "debugfs", false),
        ("/sys/kernel/tracing", "tracefs", "tracefs", false),
        ("/sys/fs/bpf", "bpffs", "bpf", false),
        ("/sys/fs/cgroup", "none", "cgroup2", false),
        ("/tmp", "tmpfs", "tmpfs", true),
        ("/dev/shm", "tmpfs", "tmpfs", false),
        ("/run", "tmpfs", "tmpfs", false),
    ];

    for &(target, source, fstype, required) in mounts {
        mkdir_p(target);
        let result = mount(
            Some(source),
            target,
            Some(fstype),
            MsFlags::empty(),
            None::<&str>,
        );
        if let Err(e) = result
            && required
        {
            // mount_filesystems() runs BEFORE the tracing subscriber
            // is installed (the subscriber needs /proc mounted to read
            // RUST_LOG from /proc/cmdline, so subscriber init follows
            // this call). Until that point fd 2 still routes to the
            // kernel console, but a `tracing::error!` event is dropped
            // because no subscriber is installed yet — this is the
            // tradeoff for installing the subscriber as early as
            // possible. A failed required-mount this early is itself
            // diagnosed downstream when /proc, /sys, or /dev are
            // missing for subsequent guest init steps.
            tracing::error!(fstype, target, err = %e, "ktstr-init: mount failed");
        }
    }

    // Standard /dev/fd symlinks. Needed by bpftrace and shell
    // process substitution (e.g. <(cmd)).
    let _ = std::os::unix::fs::symlink("/proc/self/fd", "/dev/fd");
    let _ = std::os::unix::fs::symlink("/proc/self/fd/0", "/dev/stdin");
    let _ = std::os::unix::fs::symlink("/proc/self/fd/1", "/dev/stdout");
    let _ = std::os::unix::fs::symlink("/proc/self/fd/2", "/dev/stderr");
}

/// Auto-mount the user-configured data disk at `/mnt/disk0` if the
/// host pre-formatted it. Driven by two kernel cmdline tokens
/// emitted by the host's
/// [`crate::vmm::KtstrVmBuilder::build`] cmdline assembly:
///
/// * `KTSTR_DISK0_FS=<tag>` — selects the on-disk filesystem to
///   pass to `mount(2)` (`btrfs` for the only non-Raw variant
///   today). Absence short-circuits this whole function: a `Raw`
///   disk has nothing to mount, and a config with no disk attached
///   never sees a `KTSTR_DISK0_FS` token at all.
/// * `KTSTR_DISK0_RO=1` — set when the host configured the disk
///   `read_only`. The virtio_blk device advertises
///   `VIRTIO_BLK_F_RO` for that case so the guest's gendisk is
///   read-only at the block layer; mounting RW would fail with
///   `-EROFS` (kernel `do_mount` sets the superblock RO from the
///   bdev). Setting `MS_RDONLY` proactively avoids that error path
///   entirely.
///
/// Failure modes are non-fatal: if the mount syscall returns an
/// error (unrecognized fstype tag, kernel `CONFIG_BTRFS_FS=n`,
/// device probe race, ENOMEM), the function logs to COM2 and
/// returns. The test still gets a usable VM; a subsequent test
/// step that depends on `/mnt/disk0` surfaces as a clean
/// userspace filesystem error rather than a confusing init abort.
///
/// Skips entirely when `KTSTR_DISK0_FS` is absent. The cmdline
/// emission on the host side is gated on
/// `disks[0].filesystem != Filesystem::Raw`, so this branch
/// matches the host-side opt-in: every config that requests an
/// on-disk filesystem gets the auto-mount, and every config that
/// doesn't is unaffected.
fn auto_mount_data_disks() {
    let Some(fstype) = cmdline_val("KTSTR_DISK0_FS") else {
        return;
    };
    // Validate the fstype against the known set. Today only
    // `btrfs` is wired (mirroring `Filesystem::Btrfs::cache_tag`);
    // unknown values warn-and-skip rather than handing arbitrary
    // strings to `mount(2)`. A future `Filesystem` variant must
    // add its tag here AND in the disk_config.rs `cache_tag`
    // match — keeping both lists in lockstep is the on-disk-format
    // / cmdline contract.
    let recognized = matches!(fstype.as_str(), "btrfs");
    if !recognized {
        let msg = format!(
            "ktstr-init: KTSTR_DISK0_FS={fstype} not recognized; \
             skipping auto-mount of /dev/vda"
        );
        let _ = fs::write(COM2, &msg);
        tracing::warn!("{msg}");
        return;
    }
    // RO bit. Absent or any value other than "1" means RW.
    // Strict-`==` rather than truthy-string parsing keeps the
    // contract simple and aligned with the host-side emission
    // (`KTSTR_DISK0_RO=1`).
    let ro = cmdline_val("KTSTR_DISK0_RO").as_deref() == Some("1");
    // Mount path. The host emits `KTSTR_DISK0_MOUNT=<path>` based
    // on `DiskConfig.name` — `/mnt/<name>` when set, `/mnt/disk0`
    // otherwise. Fall back to the default if the host-side value
    // is absent so a future host that emits FS but not MOUNT
    // (e.g. an older binary against a newer kernel) still mounts
    // somewhere sane rather than failing.
    let mount_point_owned =
        cmdline_val("KTSTR_DISK0_MOUNT").unwrap_or_else(|| "/mnt/disk0".to_string());
    let mount_point = mount_point_owned.as_str();
    mkdir_p(mount_point);
    let flags = if ro {
        MsFlags::MS_RDONLY
    } else {
        MsFlags::empty()
    };
    let result = mount(
        Some("/dev/vda"),
        mount_point,
        Some(fstype.as_str()),
        flags,
        None::<&str>,
    );
    if let Err(e) = result {
        let msg = format!(
            "ktstr-init: mount {fstype} on {mount_point} \
             (ro={ro}): {e}"
        );
        let _ = fs::write(COM2, &msg);
        tracing::warn!("{msg}");
    }
}

/// Recursive mkdir -p equivalent. `DirBuilder::recursive(true)` is
/// idempotent (returns Ok when the path already exists as a
/// directory) and walks parents internally, so the hand-rolled
/// recursion this replaced was redundant. Errors are swallowed to
/// match the previous behavior — the early guest init best-effort
/// creates each mount point and continues regardless, since any
/// real failure surfaces downstream when `mount()` itself fails.
///
/// Directory mode is pinned explicitly at 0o755 via
/// `DirBuilder::mode`. Relying on the default (0o777 & !umask) is
/// fragile: the guest init's umask is process state inherited from
/// the kernel/caller, and a caller that sets umask=0 before exec
/// would produce world-writable mount points. Pinning the mode in
/// the mkdir syscall itself keeps the traversal bit stable
/// regardless of umask.
fn mkdir_p(path: &str) {
    use std::os::unix::fs::DirBuilderExt;
    let _ = fs::DirBuilder::new()
        .recursive(true)
        .mode(0o755)
        .create(path);
}

/// Write a line to COM2 (the application serial port).
/// Falls back to the tracing subscriber (writing to stderr) if COM2
/// is not available.
fn write_com2(msg: &str) {
    if let Ok(mut f) = fs::OpenOptions::new().write(true).open(COM2) {
        let _ = writeln!(f, "{msg}");
    } else {
        // COM2 unavailable (devtmpfs mount failed or device missing).
        // Surface via the tracing subscriber so the host sees
        // something on the COM1 fallback path.
        tracing::warn!(target: "com1_fallback", "ktstr-init: {msg}");
    }
}

/// Create the cgroup parent directory specified by `--cell-parent-cgroup`
/// (two-token or `=`-combined form) in `/sched_args`. The directory must
/// exist before the scheduler starts because the scheduler expects it at
/// startup.
///
/// In cgroup v2, a controller is only visible inside a cgroup when its
/// parent's `cgroup.subtree_control` enables it. The kernel enforces
/// this in `cgroup_subtree_control_write` via `cgroup_control(cgrp)`,
/// which returns `parent->subtree_control` for non-root cgroups. To
/// make `cpuset` and `cpu` available in the leaf, every ancestor from
/// the cgroup root down to (and including) the leaf's immediate parent
/// must enable both controllers. Writes are applied root-to-leaf so
/// each level's prerequisite is already in place when its child is
/// written.
/// Materialise the per-test workload-cgroup root declared via
/// `#[ktstr_test(workload_root_cgroup = "/path")]`. Reads
/// `/workload_root_cgroup` (written by
/// [`crate::vmm::initramfs::build_suffix`] when
/// [`crate::vmm::initramfs::SuffixParams::workload_root_cgroup`] is
/// `Some`), validates the absolute-path shape, mkdir's
/// `/sys/fs/cgroup{path}`, and enables `+cpuset +cpu` controllers
/// along every ancestor so the workload cgroups the test author
/// creates beneath this root inherit the controllers they need.
///
/// Distinct from [`create_cgroup_parent_from_sched_args`]: that one
/// services the `--cell-parent-cgroup` scheduler-argv knob (only
/// present when the scheduler declaration explicitly carries the
/// flag); this one services the framework's per-test workload root
/// (created unconditionally when the test sets the field). Both
/// run in Phase 3 before `start_scheduler`; ordering between the
/// two is `workload_root_cgroup` first so it's visible when a
/// scheduler that does carry `--cell-parent-cgroup` walks the
/// cgroup tree at startup.
#[tracing::instrument]
fn create_workload_root_cgroup_from_file() {
    create_cgroup_from_file("/workload_root_cgroup");
}

/// Materialise the per-scheduler cgroup the scheduler process is
/// placed in. Reads `/scheduler_cgroup_parent` (written by
/// [`crate::vmm::initramfs::build_suffix`] when
/// [`crate::vmm::initramfs::SuffixParams::scheduler_cgroup_parent`]
/// is `Some` — sourced from
/// [`crate::test_support::Scheduler::cgroup_parent`]), validates
/// the absolute-path shape, mkdir's `/sys/fs/cgroup{path}`, and
/// enables `+cpuset +cpu` controllers along every ancestor so the
/// scheduler's later cgroup operations find the controllers
/// already available.
///
/// Distinct from [`create_workload_root_cgroup_from_file`] (per-
/// test workload tree) and from
/// [`create_cgroup_parent_from_sched_args`] (which fires only
/// when `--cell-parent-cgroup` is present in `/sched_args` for
/// cell-aware schedulers).
#[tracing::instrument]
fn create_scheduler_cgroup_parent_from_file() {
    create_cgroup_from_file("/scheduler_cgroup_parent");
}

/// Shared mkdir + subtree-controller setup for any
/// framework-stamped cgroup-path file. Centralises the file-read,
/// path-validation, mkdir, and `enable_subtree_controllers_to`
/// sequence so future cgroup-path slots reuse the same flow
/// without duplicating the guard logic.
fn create_cgroup_from_file(file: &str) {
    let raw = match fs::read_to_string(file) {
        Ok(s) => s,
        Err(_) => return,
    };
    let path = raw.trim();
    if !crate::test_support::cell_parent_path_is_valid(path) {
        if !path.is_empty() {
            write_com2(&format!(
                "ktstr-init: ignoring malformed `{file}` value {path:?}; \
                 skipping cgroup creation (host-side `CgroupPath::new` \
                 gate normally rejects this at compile time)",
            ));
        }
        return;
    }
    let cgroup_dir = format!("/sys/fs/cgroup{path}");
    mkdir_p(&cgroup_dir);
    enable_subtree_controllers_to(&cgroup_dir);
}

#[tracing::instrument]
fn create_cgroup_parent_from_sched_args() {
    let sched_args = match fs::read_to_string("/sched_args") {
        Ok(s) => s,
        Err(_) => return,
    };
    // Defense-in-depth filter: the host-side gate in
    // `runtime::append_base_sched_args` panics on malformed values
    // (non-absolute, bare `/`, missing) before `/sched_args` is
    // written. Reaching this path with a bad value means the gate
    // was bypassed (operator hand-edited an exported `.run` script,
    // ad-hoc argv injection); log to COM2 and skip the cgroup-tree
    // setup rather than mkdir on the host cgroup root.
    let path = match crate::test_support::parse_cell_parent_cgroup(sched_args.split_whitespace()) {
        crate::test_support::CellParentCgroupArg::Value(p)
            if crate::test_support::cell_parent_path_is_valid(p) =>
        {
            p
        }
        crate::test_support::CellParentCgroupArg::Value(bad) => {
            write_com2(&format!(
                "ktstr-init: ignoring malformed `--cell-parent-cgroup` value \
                 {bad:?} in /sched_args; skipping per-test cgroup creation \
                 (host-side gate normally panics on this)",
            ));
            return;
        }
        crate::test_support::CellParentCgroupArg::MissingValue => {
            write_com2(
                "ktstr-init: ignoring bare `--cell-parent-cgroup` (no value) \
                 in /sched_args; skipping per-test cgroup creation",
            );
            return;
        }
        crate::test_support::CellParentCgroupArg::Absent => return,
    };
    let cgroup_dir = format!("/sys/fs/cgroup{path}");
    mkdir_p(&cgroup_dir);
    enable_subtree_controllers_to(&cgroup_dir);
}

/// Enable `+cpuset +cpu` in `cgroup.subtree_control` at every ancestor
/// from `/sys/fs/cgroup` (inclusive) down to (and including) the
/// immediate parent of `leaf`. Writes are ordered root-first so each
/// level's parent already advertises the controllers when its child is
/// written — without that ordering the kernel rejects the write with
/// `-ENOENT` (see `cgroup_subtree_control_write` /
/// `cgroup_control` in `kernel/cgroup/cgroup.c`).
///
/// `leaf` is expected to live under `/sys/fs/cgroup/...` (the format
/// emitted at the call site). The leaf itself is NOT written: enabling
/// controllers in a cgroup means they are visible inside that cgroup's
/// CHILDREN, so the leaf's own `subtree_control` only matters if the
/// scheduler ever creates sub-cgroups under it. The scheduler attaches
/// tasks to the leaf, so what it needs is `cpuset`/`cpu` enabled IN
/// the leaf — which is achieved by writing to the leaf's parent.
///
/// Failures on individual writes are logged via [`write_com2`] and do
/// not abort the walk: a single intermediate level that already has
/// both controllers enabled returns `0` from kernel side, so most
/// failures observed here will surface a real misconfiguration that
/// the scheduler's own `cgroup_attach` will then re-report with
/// scheduler-specific context.
fn enable_subtree_controllers_to(leaf: &str) {
    let cgroup_root = Path::new("/sys/fs/cgroup");
    let leaf_path = Path::new(leaf);
    // Verify leaf is under the cgroup root before touching anything.
    // A malformed `--cell-parent-cgroup` argument that produces a path
    // outside `/sys/fs/cgroup` (e.g. an empty or missing-leading-slash
    // value) would otherwise walk into `/sys/fs`, `/sys`, or `/`.
    if !leaf_path.starts_with(cgroup_root) || leaf_path == cgroup_root {
        return;
    }
    // `Path::ancestors` yields leaf-first; collect the strict ancestors
    // (skip the leaf itself) up to and including the cgroup root.
    let mut ancestors: Vec<&Path> = leaf_path
        .ancestors()
        .skip(1)
        .take_while(|p| p.starts_with(cgroup_root))
        .collect();
    // Apply root-to-leaf-parent: each level's parent must already
    // enable the controller before the child write is accepted.
    ancestors.reverse();
    for level in ancestors {
        let control = level.join("cgroup.subtree_control");
        if let Err(e) = fs::write(&control, "+cpuset +cpu") {
            write_com2(&format!(
                "ktstr-init: write {} +cpuset +cpu: {}",
                control.display(),
                e
            ));
        }
    }
}

/// Outcome of [`poll_startup`].
#[derive(Debug)]
enum StartupStatus {
    /// Child exited before the poll window closed.
    Died,
    /// Child was still running when the poll window closed.
    Alive,
}

/// Outcome of [`poll_scx_attached`].
#[derive(Debug, PartialEq, Eq)]
enum ScxAttachStatus {
    /// sched_ext root kobject exposes a non-empty `ops` attribute —
    /// scheduler registered and its ops name is populated.
    Attached,
    /// Poll window closed. At least one read of `root/ops` succeeded
    /// (the kernel supports sched_ext and the kset exists), but the
    /// file never became non-empty before the timeout. Typically
    /// means the scheduler process is alive but has not finished
    /// `scx_alloc_and_add_sched` — often a BPF verifier reject, an
    /// ops-mismatch, or a slow userspace init path.
    Timeout,
    /// Every read of `root/ops` returned `Err`. Either the kernel
    /// lacks sched_ext support entirely or the sysfs tree has not
    /// been created for the current kernel — distinct from
    /// [`Timeout`](Self::Timeout), where reads succeed but the file
    /// is empty.
    SysfsAbsent,
}

impl ScxAttachStatus {
    /// True when the scheduler registered successfully. Equivalent to
    /// the pre-enum `bool` return value.
    fn is_attached(&self) -> bool {
        matches!(self, ScxAttachStatus::Attached)
    }
}

/// Poll `/sys/kernel/sched_ext/root/ops` at `interval` cadence for up
/// to `timeout`.
///
/// Returns [`ScxAttachStatus::Attached`] as soon as the file is
/// non-empty (a scheduler is registered and its ops struct has a
/// populated name). When the window closes without a successful
/// attachment, distinguishes [`Timeout`](ScxAttachStatus::Timeout)
/// (reads succeeded but the file never became non-empty — the
/// scheduler did not finish registering) from
/// [`SysfsAbsent`](ScxAttachStatus::SysfsAbsent) (every read
/// errored — the kernel lacks sched_ext sysfs entirely).
///
/// The sysfs path is built in two steps by the kernel:
/// - `kernel/sched/ext.c` creates the `sched_ext` kset under
///   `kernel_kobj` via `kset_create_and_add("sched_ext", ...)` in
///   the scx init path, giving `/sys/kernel/sched_ext/`.
/// - Each `struct scx_sched` allocation assigns `sch->kobj.kset =
///   scx_kset` then calls `kobject_init_and_add(..., NULL, "root")`
///   (or `"sub-%llu"` when `CONFIG_EXT_SUB_SCHED` and a parent is
///   present), yielding `/sys/kernel/sched_ext/root/`. The `ops`
///   attribute is registered on `scx_ktype` via `scx_sched_groups`;
///   `scx_attr_ops_show` emits `sch->ops.name` through `sysfs_emit`.
///
/// Semantics we can claim based on the kernel flow above: a non-empty
/// `root/ops` proves the scheduler completed `scx_alloc_and_add_sched`
/// — the scx_sched struct is allocated, `sch->ops = *ops` has copied
/// the userspace-provided ops (including `name`), and the kobject is
/// registered with the kset. The kobject add happens BEFORE any BPF
/// callback (`ops.init`, `ops.enable`, `ops.runnable`, etc.) runs, so
/// a non-empty read does NOT prove those callbacks validated. Use
/// this poll only to confirm "scheduler registered and name
/// populated"; verify BPF callback success via monitor telemetry or
/// the scheduler's own exit kind.
///
/// Separate from [`poll_startup`] (which watches the child process
/// state): a scheduler can be `Alive` from the process-waitpid
/// perspective and still have zero progress on scx registration.
fn poll_scx_attached(
    interval: std::time::Duration,
    timeout: std::time::Duration,
) -> ScxAttachStatus {
    use crate::vmm::freeze_coord::evented_wait::{KernfsWaitOutcome, kernfs_evented_wait};
    use nix::sys::inotify::AddWatchFlags;

    let start = std::time::Instant::now();
    // Reusable read buffer for the attribute file. Keeping the
    // allocation across the predicate's iterations is the
    // steady-state fast path.
    let mut buf = String::with_capacity(64);
    let mut ever_read_ok = false;
    // Track whether read ever succeeded so the Timeout vs SysfsAbsent
    // distinction stays correct after the helper returns.
    let check_done = || -> Option<()> {
        buf.clear();
        let read_outcome = std::fs::File::open(SYSFS_SCHED_EXT_ROOT_OPS).and_then(|mut f| {
            use std::io::Read;
            f.read_to_string(&mut buf)
        });
        if read_outcome.is_ok() {
            ever_read_ok = true;
            if !buf.trim().is_empty() {
                return Some(());
            }
        }
        None
    };

    // Evented wake sources are managed inside kernfs_evented_wait:
    //   - POLLPRI on `/sys/kernel/sched_ext/root/ops` (future-proofed
    //     for kernels that add `sysfs_notify` on the attribute)
    //   - inotify on `/sys/kernel/sched_ext/` for IN_CREATE /
    //     IN_MOVED_TO (fires when scx_alloc_and_add_sched calls
    //     kobject_init_and_add(..., "root"))
    //
    // BELT-AND-BRACES CADENCE: the helper's `cadence` parameter caps
    // each poll(2) at `interval`. Verified at kernel/sched/ext.c:6380
    // scx_alloc_and_add_sched — `sch->ops = *ops` runs BEFORE
    // `kobject_init_and_add(..., "root")`, so by IN_CREATE wake time
    // the attribute reads non-empty. The cadence is defense-in-depth
    // against (a) future kernel reordering, (b) inotify event loss
    // under pressure, (c) out-of-band kobject creation without
    // ops.name pre-population.
    let outcome = kernfs_evented_wait(
        "/sys/kernel/sched_ext/",
        AddWatchFlags::IN_CREATE | AddWatchFlags::IN_MOVED_TO,
        Some("/sys/kernel/sched_ext/root/ops"),
        interval,
        start + timeout,
        check_done,
    );

    match outcome {
        KernfsWaitOutcome::Done(()) => ScxAttachStatus::Attached,
        KernfsWaitOutcome::NoEventedSource => {
            // Both attr fd open and inotify_add_watch failed. We
            // target kernel 6.12+ where kernfs + inotify are
            // universally present, so /sys/kernel/sched_ext/ is
            // fundamentally missing or broken. Surface as
            // SysfsAbsent; the log makes the operator-actionable
            // path-existence-but-fd-unopenable case visible.
            tracing::warn!(
                "poll_scx_attached: both attr-fd open (/sys/kernel/sched_ext/root/ops) \
                 AND inotify_add_watch (/sys/kernel/sched_ext/) failed; surfacing \
                 SysfsAbsent. Diagnose: zcat /proc/config.gz | grep -E \
                 'CONFIG_SCHED_CLASS_EXT|CONFIG_INOTIFY_USER' — both must be =y"
            );
            ScxAttachStatus::SysfsAbsent
        }
        KernfsWaitOutcome::Timeout => {
            let status = if ever_read_ok {
                ScxAttachStatus::Timeout
            } else {
                ScxAttachStatus::SysfsAbsent
            };
            // Per "log on timeout when no error surfaces": callers
            // may swallow this into a non-error path (boot-time);
            // log here for a visible breadcrumb in /tmp/ktstr*.log
            // even when the typed return is later consumed silently.
            tracing::warn!(
                elapsed_s = start.elapsed().as_secs_f64(),
                timeout_s = timeout.as_secs_f64(),
                ever_read_ok,
                status = ?status,
                "poll_scx_attached: timeout — sched_ext attach not observed \
                 within deadline"
            );
            status
        }
    }
}

/// Block on `pidfd` becoming readable for up to `timeout`. Returns
/// as soon as the child exits (pidfd POLLIN edge fires
/// microseconds after the kernel reaps), or when the deadline
/// elapses with the child still alive.
///
/// `pidfd_open` has been available since kernel 5.3 (2019); ktstr
/// targets 6.16+ where it is unconditionally present. The interval
/// parameter is unused here because `poll(2)` blocks until the fd
/// becomes readable or the absolute deadline elapses — there is
/// nothing to "poll faster" inside the wait. The deadline is
/// enforced via `Instant::now()` re-checks across loop iterations
/// because `poll(2)` may return EINTR (e.g. SIGCHLD coalescing); the
/// outer re-check rebuilds the remaining timeout against the
/// absolute deadline.
///
/// Liveness is observed via [`proc_pid_alive`] / pidfd POLLIN, never
/// `Child::try_wait`. PID 1 has SIGCHLD set to `SIG_IGN` for zombie
/// prevention (see [`ktstr_guest_init`]), so the kernel auto-reaps
/// the scheduler child the moment it exits. `try_wait` (which calls
/// `waitpid(pid, ..., WNOHANG)`) then returns `ECHILD`, which the
/// previous implementation mapped to `WaitError` and the caller
/// treated as still-alive — leaving a crashed scheduler undetected.
/// pidfd POLLIN and `/proc/{pid}` removal are signal-disposition
/// independent (the pidfd is readable on exit regardless of who
/// reaps; the procfs entry disappears on `release_task`), so they
/// observe the real state.
fn poll_startup(
    child: &mut Child,
    interval: std::time::Duration,
    timeout: std::time::Duration,
) -> StartupStatus {
    let pid = child.id();
    // SAFETY: `pidfd_open(2)` accepts any process the caller can
    // signal. We just spawned `child`; its pid is owned by this
    // process, so the syscall is safe to issue with no other
    // synchronisation. Failure (rare — e.g. very tight pid reuse,
    // sandbox restriction) falls back to a `proc_pid_alive` loop
    // below.
    let pidfd =
        unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::c_int, 0u32) as libc::c_int };
    if pidfd < 0 {
        // pidfd_open unsupported on this kernel. Procfs polling is
        // the SIG_IGN-safe fallback: the procfs entry vanishes when
        // the kernel runs `release_task` on the child, regardless
        // of how SIGCHLD is handled. The shared
        // [`poll_proc_pid_absent`] helper carries the loop body so
        // any future EINTR / signal-pause refinement applies
        // uniformly here and in [`kill_scheduler_process`]'s
        // SIGTERM/SIGKILL aftermath polls.
        return if poll_proc_pid_absent(pid, interval, timeout) {
            StartupStatus::Died
        } else {
            StartupStatus::Alive
        };
    }
    let start = std::time::Instant::now();
    let result = loop {
        let now = std::time::Instant::now();
        if now >= start + timeout {
            // Deadline elapsed. pidfd POLLIN never fired across
            // the entire window, so the kernel hasn't signalled
            // exit on the pidfd. Re-confirm via /proc to cover
            // the rare race where the child died between the
            // last poll and now (poll cadence is bounded by
            // EINTR-driven loops; a ~microsecond-wide window
            // exists where the child could have exited
            // post-poll-pre-now).
            break if proc_pid_alive(pid) {
                StartupStatus::Alive
            } else {
                StartupStatus::Died
            };
        }
        let remaining_ms = (start + timeout - now).as_millis().min(i32::MAX as u128) as i32;
        let mut pfd = libc::pollfd {
            fd: pidfd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: `pfd` is a single-element pollfd; nfds is 1.
        // Every poll outcome (ready, timeout, EINTR, error) loops
        // back to the deadline check above, which rebuilds
        // `remaining_ms` against the absolute start+timeout so
        // EINTR cannot extend the wait past the requested
        // duration.
        let rc = unsafe { libc::poll(&mut pfd, 1, remaining_ms) };
        if rc > 0 && pfd.revents & libc::POLLIN != 0 {
            // pidfd POLLIN fires precisely at child exit (kernel
            // `pidfd_poll` in `fs/pidfs.c` checks `exit_state`,
            // woken via `do_notify_pidfd` from `exit_notify`).
            // No `try_wait` follow-up needed — POLLIN itself is
            // the proof.
            break StartupStatus::Died;
        }
        // rc == 0 (timeout) or rc < 0 (EINTR/error) re-checks the
        // deadline at the top of the loop. EINTR with remaining
        // budget loops once more; deadline-exhausted falls into
        // the elapsed branch above.
    };
    // SAFETY: pidfd is owned by this function and not used after
    // close.
    unsafe {
        libc::close(pidfd);
    }
    result
}

/// Probe-pipeline drain handles passed to [`start_scheduler`] so the
/// early-bail paths (scheduler Died, not Attached, spawn Err) can
/// flush probe output to COM2 before calling `force_reboot()`. The
/// success path's drain runs in [`start_sched_exit_monitor`]
/// instead — it sees the scheduler exit notification and waits on
/// `output_done` there.
pub(crate) struct ProbeDrain {
    /// Probe-thread stop request. Setting this wakes the probe
    /// thread out of its ring-buffer poll loop; the thread then
    /// emits its payload and sets `output_done`.
    stop: Arc<AtomicBool>,
    /// One-shot signal: set by the probe thread after writing
    /// `PROBE_PAYLOAD_END` to COM2. Waited on event-driven; the
    /// outer VM wall-clock timeout is the only safety net for a
    /// hung probe (per the queue-management policy: don't add
    /// arbitrary local timeouts when an event source exists).
    output_done: Arc<crate::sync::Latch>,
}

/// Drain the probe pipeline: signal stop, then block on
/// `output_done`. Called from each early-bail path in
/// [`start_scheduler`] before `force_reboot()` so the probe
/// payload (or the diagnostic-only payload the probe thread emits
/// on a forced stop) reaches COM2's host-side capture buffer.
///
/// `drain` is `None` when no probe stack was supplied — every
/// caller is a no-op in that case.
fn drain_probe_pipeline(drain: Option<&ProbeDrain>) {
    let Some(d) = drain else { return };
    d.stop.store(true, Ordering::Release);
    d.output_done.wait();
}

/// Bound on [`reap_child_bounded`]: how long teardown waits for a
/// SIGKILL'd scheduler to exit before giving up and letting the VM reboot
/// reap it. A SIGKILL'd scheduler normally exits <<1s — post-crash bypass
/// keeps it CFS-schedulable, and it is NOT held in the kernel scx disable:
/// its `struct_ops` detach (`bpf_scx_unreg`) only `kthread_flush_work`s
/// the `scx_root_disable` the crash irq_work already kicked, which is
/// ms-scale (bypass + per-task reclass + one `synchronize_rcu` + the BPF
/// `ops.exit`, all fast for these schedulers). The bound is a defensive
/// cap — only a pathological multi-second `ops.exit` or RCU stall could
/// approach it — so teardown caps the wait rather than risk adding such a
/// stall to every crashed-scheduler teardown.
const SCHED_REAP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// Grace given to a CRASHED scheduler to finish flushing its userspace
/// diagnostics to stderr and exit on its own BEFORE the hard SIGKILL, so
/// the kill doesn't truncate that output (`dump_sched_output` reads it).
/// Bounded so a userspace hang can't wedge teardown; returns early the
/// moment the scheduler exits. Only applied on a crash (dump_started).
/// Sized for the USERSPACE flush, not the kernel dump: the kernel's scx
/// exit dump is bounded and truncated in-kernel, but the scheduler's
/// userspace flush of it to stderr (plus libbpf teardown) can run past a
/// shorter window, and the SIGKILL then truncates the tail of THAT output.
const SCHED_KILL_GRACE: std::time::Duration = std::time::Duration::from_millis(3000);

/// Wait up to `timeout` for `child` to exit (evented via `pidfd_open` +
/// `poll`), then reap it. Does NOT send a signal — callers drive the
/// exit: the crash-grace caller calls this BEFORE `child.kill()` (giving
/// a self-unregistering scheduler a chance to exit on its own within the
/// grace), and the post-grace caller calls it AFTER `child.kill()` (to
/// reap the pending SIGKILL). Returns `true` iff reaped within the
/// window; on timeout the child is left for the VM reboot to reap —
/// teardown must not block unboundedly on a wedged process (see
/// [`SCHED_REAP_TIMEOUT`]).
fn reap_child_bounded(child: &mut std::process::Child, timeout: std::time::Duration) -> bool {
    // Fast path: already exited (e.g. a clean scheduler that took the
    // SIGKILL immediately).
    if let Ok(Some(_)) = child.try_wait() {
        return true;
    }
    match crate::sync::pidfd_poll_exited(child.id() as libc::pid_t, timeout) {
        // Readable => zombie => the reap is now non-blocking.
        crate::sync::PidfdWait::Exited => {
            let _ = child.wait();
            true
        }
        // Timed out: still alive — leave it for the VM reboot.
        crate::sync::PidfdWait::TimedOut => false,
        // pidfd_open failed (ESRCH/gone or env defect): one non-blocking
        // reap attempt, then give up to the reboot.
        crate::sync::PidfdWait::NoPidfd => matches!(child.try_wait(), Ok(Some(_))),
    }
}

/// Start the boot scheduler binary if it exists. Thin wrapper around
/// [`spawn_scheduler_from_paths`] supplying the boot-time paths
/// (`/scheduler` + `/sched_args` + `/tmp/sched.log`). Returns the
/// child process and the path to its log file.
///
/// Mid-experiment scheduler-lifecycle Op dispatch
/// ([`Op::AttachScheduler`](crate::scenario::ops::Op::AttachScheduler) /
/// [`Op::ReplaceScheduler`](crate::scenario::ops::Op::ReplaceScheduler))
/// calls [`spawn_scheduler_from_paths`] directly with paths under
/// `/staging/schedulers/<name>/` so swap binaries don't shadow the
/// boot slot.
#[tracing::instrument(skip(probe_drain))]
pub(crate) fn start_scheduler(probe_drain: Option<ProbeDrain>) -> (Option<Child>, Option<String>) {
    spawn_scheduler_from_paths("/scheduler", "/sched_args", "/tmp/sched.log", probe_drain)
}

/// Failure modes for [`try_spawn_scheduler`]. Distinct variants
/// per the three observable failure points in the spawn pipeline
/// so callers can branch on the specific outcome — the boot path
/// uniformly responds with dump + lifecycle + force_reboot, while
/// the scheduler-lifecycle Op dispatch path surfaces each variant
/// as an actionable test-failure diagnostic via the per-variant
/// `Display` text.
#[derive(Debug)]
pub(crate) enum SpawnSchedulerError {
    /// `Command::spawn` returned `Err` — fork/exec failed at the
    /// kernel boundary (ENOMEM, EACCES on the binary, EAGAIN from
    /// rlimit). Carries the underlying `io::Error` so the boot
    /// path can synthesize a `SCHED_OUTPUT_START / END`-framed
    /// log payload via `send_sched_log_text`.
    SpawnFailed(std::io::Error),

    /// `poll_startup` observed the process exit within the
    /// liveness window — typical for a scheduler that crashes in
    /// BPF prog load (verifier reject) or argv validation before
    /// the bind to `/sys/kernel/sched_ext/root/ops` lands.
    /// `log_path` is the file the spawn helper wrote
    /// stdout+stderr into; callers use it for `dump_sched_output`.
    ///
    /// **Post-mortem state guarantee.** [`try_spawn_scheduler`]
    /// returns this variant only AFTER clearing [`SCHED_PID`] to 0
    /// (the dead pid was published optimistically at spawn so the
    /// sched_exit_monitor caller path could install against a known
    /// id; the StartupDied branch never gets that far so the spawn
    /// helper owns the rollback). The process is already reaped via
    /// `poll_startup`'s internal `try_wait`. No manual cleanup
    /// required by the caller.
    StartupDied { log_path: String },

    /// Process is alive past the liveness window but
    /// `poll_scx_attached` did NOT observe the bind marker.
    /// `reason` is one of `"timeout"` (attach poll exhausted) or
    /// `"sched_ext sysfs absent"` (kernel lacks sched_ext). The
    /// caller (boot path → `force_reboot`, Op path → bail) uses
    /// `log_path` to surface the scheduler's own diagnostic
    /// output.
    ///
    /// **Post-mortem state guarantee.** [`try_spawn_scheduler`]
    /// returns this variant only AFTER SIGKILLing the orphan
    /// process (which is alive but not bound to scx, so it would
    /// otherwise keep running and could late-bind on the next
    /// scheduler attempt) and waiting on it via `child.wait()` to
    /// reap the zombie, plus clearing [`SCHED_PID`] to 0. No manual
    /// cleanup required by the caller.
    NotAttached {
        reason: &'static str,
        log_path: String,
    },
}

impl std::fmt::Display for SpawnSchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SpawnFailed(e) => {
                write!(f, "Command::spawn failed: {e}")
            }
            Self::StartupDied { log_path } => {
                write!(
                    f,
                    "scheduler exited before passing the 1-second liveness gate \
                     (framework waits for the scheduler binary to remain alive at \
                     least 1 s before checking for sched_ext bind via /sys/kernel/\
                     sched_ext/root/ops). Common causes: BPF verifier rejection \
                     (look for 'libbpf' / 'verifier' lines in the log), missing \
                     CONFIG_SCHED_CLASS_EXT, scheduler binary segfault at init, \
                     argv validation failure. Log content rendered below as part \
                     of the failure dump (log captured at {log_path}); the process \
                     was reaped and SCHED_PID cleared before this error surfaced."
                )
            }
            Self::NotAttached { reason, log_path } => {
                write!(
                    f,
                    "scheduler alive but did not bind to sched_ext within the \
                     attach window: {reason} (framework polls /sys/kernel/sched_ext/\
                     root/ops for the BPF scheduler attach marker after the \
                     scheduler binary's liveness gate; this variant surfaces when \
                     the binary stayed alive but never wrote the bind marker). \
                     Common causes for 'timeout': BPF program load stalled on a \
                     slow CI runner past the 10s window, verifier ran long but \
                     succeeded eventually (bump the window or warm the BPF cache). \
                     Common causes for 'sched_ext sysfs absent': kernel built \
                     without CONFIG_SCHED_CLASS_EXT (rebuild with that config). \
                     Log content rendered below as part of the failure dump (log \
                     captured at {log_path}); the framework SIGKILLed and reaped \
                     the orphan + cleared SCHED_PID before this error surfaced."
                )
            }
        }
    }
}

impl std::error::Error for SpawnSchedulerError {}

/// Pure spawn helper — runs the spawn → poll-startup → poll-attached
/// pipeline and returns a `Result` so callers can choose how to
/// handle each failure mode. The boot path uniformly responds with
/// `dump_sched_output` + `send_lifecycle` + `force_reboot`; the
/// scheduler-lifecycle Op dispatch surfaces each `Err` variant as
/// a typed test-failure rather than rebooting the VM.
///
/// `Ok(None)` means the binary file is missing — the caller decides
/// whether that is a degenerate-but-acceptable state (boot path:
/// no scheduler configured) or a hard error (Op dispatch:
/// staging pipeline mis-packed). Per the [`SpawnSchedulerError`]
/// doc, the three failure variants each carry the context the
/// boot path needs (log_path for dump, io::Error for spawn
/// failure) so the wrapper can preserve the prior boot-failure
/// semantics without changes to host-side diagnostics.
pub(crate) fn try_spawn_scheduler(
    binary_path: &str,
    args_path: &str,
    log_path: &str,
) -> Result<Option<(Child, String)>, SpawnSchedulerError> {
    if !Path::new(binary_path).exists() {
        return Ok(None);
    }

    let sched_args = fs::read_to_string(args_path)
        .unwrap_or_default()
        .trim()
        .to_string();
    let args: Vec<&str> = if sched_args.is_empty() {
        vec![]
    } else {
        sched_args.split_whitespace().collect()
    };

    let log_file = fs::File::create(log_path).ok();
    let stdout = match log_file.as_ref().and_then(|f| f.try_clone().ok()) {
        Some(f) => Stdio::from(f),
        None => Stdio::null(),
    };
    let stderr = match log_file {
        Some(f) => Stdio::from(f),
        None => Stdio::null(),
    };

    // Build RUST_LOG for the scheduler: append libbpf noise
    // suppression to whatever the guest already has. libbpf
    // emits debug/info messages through the `log` crate via
    // scx_utils::libbpf_logger; raising its threshold to warn
    // keeps scheduler output readable.
    let sched_rust_log = match std::env::var("RUST_LOG") {
        Ok(existing) => format!("{existing},scx_utils::libbpf_logger=warn"),
        Err(_) => "info,scx_utils::libbpf_logger=warn".to_string(),
    };

    let mut child = Command::new(binary_path)
        .args(&args)
        .env("RUST_LOG", &sched_rust_log)
        .stdout(stdout)
        .stderr(stderr)
        .spawn()
        .map_err(SpawnSchedulerError::SpawnFailed)?;

    // Publish the scheduler PID via the [`SCHED_PID`] atomic side
    // channel — readers retrieve it through [`sched_pid`]. The
    // previous implementation called `std::env::set_var("SCHED_PID",
    // ...)` here, but the Phase A probe thread spawned earlier in
    // `ktstr_guest_init` (`start_probe_phase_a`) is alive at this
    // point, so mutating glibc's global `__environ` array races
    // with the probe thread's potential `getenv`/`execve` traffic
    // — documented UB on Linux. The atomic store is data-race-free
    // and the published value reaches readers via the same
    // `Acquire`/`Release` synchronisation the [`sched_pid`] reader
    // uses.
    //
    // The `child.id()` value fits in `i32` because Linux pids are
    // `pid_t` (signed 32-bit on every supported arch).
    // `kernel.pid_max` is a 22-bit limit by default and the kernel
    // never returns negative pids from `fork(2)`, so the cast is
    // exact.
    SCHED_PID.store(child.id() as i32, Ordering::Release);

    match poll_startup(
        &mut child,
        std::time::Duration::from_millis(50),
        std::time::Duration::from_secs(1),
    ) {
        StartupStatus::Died => {
            // Process already exited — SIGCHLD reaped via poll_startup's
            // try_wait. SCHED_PID still points at the dead pid; clear so a
            // subsequent Op dispatch's sched_pid() returns None instead of
            // the stale dead/recycled id. The pid was published optimistically
            // at spawn so the sched_exit_monitor caller path can install
            // against a known id, but the StartupDied branch never gets that
            // far so we own the rollback.
            SCHED_PID.store(0, Ordering::Release);
            Err(SpawnSchedulerError::StartupDied {
                log_path: log_path.to_string(),
            })
        }
        StartupStatus::Alive => {
            // Verify the scheduler actually BOUND to sched_ext —
            // a scheduler process can be alive but stuck in its
            // BPF init (verifier reject, ops mismatch), which
            // would leave the test running against the default
            // kernel scheduler without the host ever noticing.
            // `root/ops` is the post-attach marker.
            //
            // 10s budget aligns with SCHED_LIFECYCLE_KILL_GRACE on
            // the kill side. A cold-cache BPF verifier + cgroup_init
            // walking all tasks can plausibly run 5s+ on a slow CI
            // runner; the prior 3s budget produced sporadic
            // NotAttached(Timeout) returns under load even when the
            // scheduler eventually bound seconds later. The 10s
            // ceiling still surfaces real verifier-reject /
            // ops-mismatch failures fast enough for an operator to
            // act, while giving headroom for warm-boot timing.
            let status = poll_scx_attached(
                std::time::Duration::from_millis(50),
                std::time::Duration::from_secs(10),
            );
            if !status.is_attached() {
                let reason = match status {
                    ScxAttachStatus::Timeout => "timeout",
                    ScxAttachStatus::SysfsAbsent => "sched_ext sysfs absent",
                    ScxAttachStatus::Attached => unreachable!(),
                };
                // The process is ALIVE (poll_startup said so) but never
                // bound to sched_ext. If we just return Err, the orphaned
                // process keeps running and may bind LATE — polluting kernel
                // state for the next Op dispatch (next AttachScheduler would
                // see root/ops populated by an unknown owner; next Replace
                // would race against the stale scheduler's eventual death).
                // SIGKILL + waitpid here removes the orphan deterministically.
                // SIGKILL not SIGTERM: the process never bound to scx so there's
                // no in-kernel scheduler state to tear down via the libbpf path.
                let pid = child.id() as libc::pid_t;
                unsafe {
                    let _ = libc::kill(pid, libc::SIGKILL);
                }
                let _ = child.wait();
                SCHED_PID.store(0, Ordering::Release);
                return Err(SpawnSchedulerError::NotAttached {
                    reason,
                    log_path: log_path.to_string(),
                });
            }
            Ok(Some((child, log_path.to_string())))
        }
    }
}

/// Spawn a scheduler binary with BOOT failure semantics: on any
/// failure mode reported by [`try_spawn_scheduler`], dump the
/// scheduler log via the bulk data port, signal
/// `LifecyclePhase::SchedulerDied` / `SchedulerNotAttached` over
/// guest_comms, send an exit code, drain the probe pipeline, and
/// call [`force_reboot`]. Used by the boot wrapper
/// [`start_scheduler`] where a missing or broken scheduler is a
/// terminal condition.
///
/// Mid-experiment scheduler-lifecycle Op dispatch should call
/// [`try_spawn_scheduler`] directly and surface failures as
/// typed test-failure diagnostics instead of rebooting the VM.
///
/// `Ok(None)` from `try_spawn_scheduler` (binary missing) returns
/// `(None, None)` — preserves the prior contract where an absent
/// `/scheduler` is "no scheduler configured" rather than a
/// failure.
///
/// `probe_drain` is consumed only on the force_reboot paths; the
/// Ok-success path leaves it for the caller to drop normally.
///
/// # Path contract
///
/// `binary_path` is checked for existence inline; an absent
/// binary returns `(None, None)`. `args_path` may be absent (an
/// empty file produces empty argv). `log_path` is created or
/// truncated at function entry.
#[tracing::instrument(skip(probe_drain), fields(binary = %binary_path))]
pub(crate) fn spawn_scheduler_from_paths(
    binary_path: &str,
    args_path: &str,
    log_path: &str,
    probe_drain: Option<ProbeDrain>,
) -> (Option<Child>, Option<String>) {
    match try_spawn_scheduler(binary_path, args_path, log_path) {
        Ok(None) => (None, None),
        Ok(Some((child, log))) => (Some(child), Some(log)),
        Err(SpawnSchedulerError::SpawnFailed(e)) => {
            tracing::error!(err = %e, "ktstr-init: spawn scheduler failed");
            // Synthesize a minimal sched-log payload framed by
            // the existing SCHED_OUTPUT_START/END markers so the
            // host's `parse_sched_output` returns the spawn-
            // failure diagnostic exactly as the prior COM2 path
            // did.
            crate::vmm::guest_comms::send_sched_log(crate::verifier::SCHED_OUTPUT_START.as_bytes());
            send_sched_log_text(&format!("failed to spawn: {e}"));
            crate::vmm::guest_comms::send_sched_log(crate::verifier::SCHED_OUTPUT_END.as_bytes());
            crate::vmm::guest_comms::send_lifecycle(
                crate::vmm::wire::LifecyclePhase::SchedulerDied,
                "",
            );
            crate::vmm::guest_comms::send_exit(1);
            // Drain the probe pipeline before reboot so
            // PROBE_OUTPUT_END hits COM2 ahead of force_reboot.
            // No-op when no probe stack was supplied.
            drain_probe_pipeline(probe_drain.as_ref());
            force_reboot();
        }
        Err(SpawnSchedulerError::StartupDied { log_path }) => {
            // Scheduler died during startup. Dump the scheduler
            // log via the bulk data port — the
            // SCHED_OUTPUT_START / SCHED_OUTPUT_END markers
            // travel verbatim inside the chunk bytes so the
            // host's `parse_sched_output` walker keeps working
            // unchanged.
            dump_sched_output(&log_path);
            crate::vmm::guest_comms::send_lifecycle(
                crate::vmm::wire::LifecyclePhase::SchedulerDied,
                "",
            );
            crate::vmm::guest_comms::send_exit(1);
            drain_probe_pipeline(probe_drain.as_ref());
            force_reboot();
        }
        Err(SpawnSchedulerError::NotAttached { reason, log_path }) => {
            dump_sched_output(&log_path);
            crate::vmm::guest_comms::send_lifecycle(
                crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
                reason,
            );
            crate::vmm::guest_comms::send_exit(1);
            drain_probe_pipeline(probe_drain.as_ref());
            force_reboot();
        }
    }
}

/// Path of the scheduler-stats Unix socket inside the guest. Owned
/// by the running scx_* scheduler binary (created via
/// `scx_utils::stats::ScxStatsServer`). Empty when no scheduler is
/// running.
const SCHED_STATS_SOCKET: &str = "/var/run/scx/root/stats";

/// Path of the guest-side stats relay's port-2 device node. The
/// kernel virtio-console driver creates this when the multiport
/// PORT_NAME control message lands ahead of PORT_OPEN; see
/// [`crate::vmm::wire::PORT2_NAME`].
const SCHED_STATS_PORT_DEV: &str = "/dev/vport0p2";

/// Per-iteration scratch buffer size. Matches
/// [`crate::vmm::sched_stats::MAX_REQUEST_BYTES`] (256 KiB) so a
/// single legitimate request or response fits in one read. Larger
/// payloads span multiple loop iterations.
const RELAY_BUFFER_BYTES: usize = 256 * 1024;

/// Parent directory of the scheduler-stats Unix socket. The relay
/// creates this directory if it doesn't exist (the scheduler
/// userspace creates it before bind, but we may race) and watches
/// it via inotify for the `stats` socket file's `IN_CREATE` event.
const SCHED_STATS_SOCKET_DIR: &str = "/var/run/scx/root";

/// File name (final component) of the scheduler-stats Unix socket
/// inside [`SCHED_STATS_SOCKET_DIR`]. Matched against
/// [`nix::sys::inotify::InotifyEvent::name`] entries to detect
/// the scheduler's bind without polling.
const SCHED_STATS_SOCKET_NAME: &str = "stats";

/// Inline JSON error response the relay writes back to the host
/// when it has not yet connected to (or has lost connection to)
/// the scheduler's Unix socket. The host's
/// [`crate::vmm::sched_stats::SchedStatsClient`] parses the
/// `ktstr_relay_error` field into a typed
/// [`crate::vmm::sched_stats::SchedStatsError::NoScheduler`]. The
/// trailing `\n` matches scx_stats's line-delimited wire format.
const SCHED_STATS_RELAY_NO_SCHEDULER_REPLY: &[u8] =
    b"{\"ktstr_relay_error\":\"no scheduler available\"}\n";

/// Stop signal for the scheduler-stats relay thread. Carries an
/// `AtomicBool` source-of-truth flag plus an `EventFd` wake fd so
/// callers in phase-6 cleanup can interrupt a relay that is parked
/// in `poll(2)` without waiting for any timeout. The relay
/// registers the eventfd in its poll set and re-checks the
/// AtomicBool at every wake.
pub(crate) struct RelayStopSignal {
    flag: Arc<AtomicBool>,
    evt: Arc<vmm_sys_util::eventfd::EventFd>,
}

impl RelayStopSignal {
    /// Flip the source-of-truth flag and write the eventfd. The
    /// flag is set with `Release` before the fd write so a relay
    /// that wakes on the eventfd edge observes `true` on its
    /// `Acquire` load. Errors from the eventfd write are silently
    /// ignored — the AtomicBool is authoritative and a saturated
    /// counter (or torn fd) just means the relay's next natural
    /// wake re-checks the flag.
    fn signal_stop(&self) {
        self.flag.store(true, Ordering::Release);
        let _ = self.evt.write(1);
    }
}

/// Spawn the scheduler-stats relay thread.
///
/// Event-driven design: the relay opens [`SCHED_STATS_PORT_DEV`]
/// once (no retry — `redirect_stdio_to_bulk_port` already proved
/// the multiport handshake completed by the time this is called),
/// then runs an outer loop that:
///
/// 1. Waits for the scheduler's Unix socket to appear via inotify
///    (no sleep loop).
/// 2. Connects, then poll(2)s on the port fd, the socket fd, and
///    the stop eventfd. On port→socket data arriving, forwards
///    the bytes; on socket→host data, forwards back; on socket
///    EOF/error, writes the inline error envelope to the port and
///    falls back to inotify wait; on stop, returns.
///
/// Returns a [`RelayStopSignal`] the caller flips on teardown.
/// The thread is detached; the kernel reboot path tears down both
/// device nodes synchronously, so the relay exits when its
/// blocking I/O returns EBADF/EOF.
fn start_sched_stats_relay() -> RelayStopSignal {
    use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};
    let flag = Arc::new(AtomicBool::new(false));
    let evt = match EventFd::new(EFD_NONBLOCK) {
        Ok(e) => Arc::new(e),
        Err(err) => {
            tracing::error!(
                error = %err,
                "stats relay: eventfd create failed; relay disabled \
                 (host SchedStatsClient calls will hang on shutdown)"
            );
            // Return a flag-only signal; the relay never spawns.
            return RelayStopSignal {
                flag,
                evt: Arc::new(EventFd::new(0).unwrap_or_else(|_| {
                    // Last-resort: try without EFD_NONBLOCK. If
                    // even this fails, the host is in a degraded
                    // state where no relay can run anyway.
                    panic!("stats relay: cannot create any eventfd")
                })),
            };
        }
    };
    let flag_for_thread = flag.clone();
    let evt_for_thread = evt.clone();
    let _ = std::thread::Builder::new()
        .name("ktstr-sched-stats-relay".into())
        .spawn(move || {
            sched_stats_relay_loop(flag_for_thread, evt_for_thread);
        });
    RelayStopSignal { flag, evt }
}

/// Inner loop for the stats relay thread. Opens the port-2 device
/// node once (single open — the multiport handshake completed
/// before this function was called) and drives the outer
/// inotify-wait → connect → poll-relay-session cycle until `stop`
/// flips.
/// Maximum consecutive `PortEof` returns from the inner functions
/// (`wait_for_stats_socket` and `run_relay_session`) we tolerate
/// before declaring the virtio-console port dead and exiting the
/// relay thread. Any non-`PortEof` exit (`RelaySessionExit::Other`,
/// `WaitSocketResult::Connected`, `WaitSocketResult::Stopped`)
/// resets the counter.
///
/// B14: the stats-port reader can return Ok(0) when the host
/// hasn't connected its end of `/dev/vport0p2` yet, when the host
/// closes its console connection, or when the kernel virtio-console
/// driver hits a transient disconnect. A single Ok(0) is recoverable
/// (the inner functions exit cleanly, the outer loop re-arms via
/// inotify and the scheduler/host can re-establish the link).
/// But a port that's permanently closed produces back-to-back Ok(0)
/// returns indefinitely — re-arming inotify, getting woken by the
/// race-free initial probe (which can succeed against a still-bound
/// socket file even though the port itself is dead), running a
/// micro-session that immediately exits on Ok(0), and looping. This
/// busy-loop wastes CPU and produces a log flood. After three
/// consecutive Ok(0) returns the relay thread exits — the host
/// loses scheduler-stats relay (no automatic recovery) but the
/// guest's CPU bill stops.
const SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF: u32 = 3;

/// Return value of [`run_relay_session`] / [`wait_for_stats_socket`]
/// signalling why the inner function exited so the outer
/// [`sched_stats_relay_loop`] can count consecutive port EOFs and
/// bail when the virtio-console port is persistently dead. See
/// [`SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF`] for the policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelaySessionExit {
    /// `port.read` returned Ok(0). Counts toward the consecutive
    /// EOF budget the outer loop tracks.
    PortEof,
    /// Any other clean exit (socket EOF, scheduler error, stop_evt
    /// fired). Resets the consecutive EOF counter.
    Other,
}

fn sched_stats_relay_loop(stop: Arc<AtomicBool>, stop_evt: Arc<vmm_sys_util::eventfd::EventFd>) {
    let mut port = match fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(SCHED_STATS_PORT_DEV)
    {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = SCHED_STATS_PORT_DEV,
                "stats relay: open vport0p2 failed; relay disabled"
            );
            return;
        }
    };

    // B14: count consecutive `port.read` Ok(0) outcomes from the
    // inner functions. A single Ok(0) is recoverable; after
    // `SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF` in a row we
    // assume the virtio-console port is permanently dead and exit.
    let mut consecutive_port_eof: u32 = 0;

    // Outer loop: wait for socket via inotify, connect, run
    // session, fall back to inotify on socket failure. Stops only
    // when stop_evt fires (signal_stop flipped the flag and woke
    // every blocked syscall).
    while !stop.load(Ordering::Acquire) {
        let wait_exit = wait_for_stats_socket(&mut port, &stop, &stop_evt);
        match wait_exit {
            WaitSocketResult::Connected(socket) => {
                // A successful connect refreshes the
                // consecutive-EOF budget. Without this reset, a
                // run of inotify-wait Ok(0)s could leave the
                // counter near the cap; if the next session
                // happens to return PortEof once it would push
                // past the cap and exit even though the port
                // proved live enough to deliver a connect-edge
                // and run a session.
                consecutive_port_eof = 0;
                let exit = run_relay_session(&mut port, socket, &stop, &stop_evt);
                match exit {
                    RelaySessionExit::PortEof => {
                        consecutive_port_eof += 1;
                    }
                    RelaySessionExit::Other => {
                        consecutive_port_eof = 0;
                    }
                }
                if consecutive_port_eof >= SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF {
                    tracing::warn!(
                        consecutive_port_eof,
                        "stats relay: vport0p2 returned Ok(0) on \
                         {SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF} consecutive \
                         relay sessions — assuming the port is permanently dead and \
                         exiting the relay thread to avoid a busy-loop"
                    );
                    return;
                }
            }
            WaitSocketResult::PortEof => {
                consecutive_port_eof += 1;
                if consecutive_port_eof >= SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF {
                    tracing::warn!(
                        consecutive_port_eof,
                        "stats relay: vport0p2 returned Ok(0) on \
                         {SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF} consecutive \
                         inotify-wait drains — assuming the port is permanently \
                         dead and exiting the relay thread to avoid a busy-loop"
                    );
                    return;
                }
                // Continue the outer loop to re-arm inotify; the
                // count keeps climbing until it hits the cap or a
                // non-EOF event resets it.
            }
            WaitSocketResult::Stopped => {
                // wait_for_stats_socket returned None only when
                // stop flipped or inotify itself errored. Either
                // way, exit.
                return;
            }
        }
    }
}

/// Result of [`wait_for_stats_socket`]: distinguishes a successful
/// connect from the two clean-exit paths so the outer loop can
/// classify them correctly. B14: `PortEof` (port read returned
/// Ok(0)) feeds the consecutive-EOF counter; `Stopped` (stop_evt
/// fired or inotify errored) terminates the loop unconditionally.
enum WaitSocketResult {
    /// Scheduler socket connected; the relay can run a session.
    Connected(std::os::unix::net::UnixStream),
    /// `port.read` returned Ok(0) while waiting for the scheduler
    /// to bind. Counts toward the outer loop's consecutive-EOF
    /// budget — see
    /// [`SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF`].
    PortEof,
    /// `stop_evt` fired or inotify itself errored — exit
    /// unconditionally.
    Stopped,
}

/// Block (event-driven) until the scheduler's Unix socket exists,
/// then connect and return the stream. Uses inotify on the parent
/// directory to receive a `IN_CREATE` event when the scheduler
/// binds. Returns `Stopped` when `stop_evt` fires or inotify
/// itself errors out, `PortEof` when the host-side port read
/// reports Ok(0).
///
/// Race-free initial check: after setting up the watch, attempt
/// to connect once. If the socket already exists (scheduler
/// finished binding before we created the watch) the connect
/// succeeds and we return without ever reading from inotify.
fn wait_for_stats_socket(
    port: &mut std::fs::File,
    stop: &Arc<AtomicBool>,
    stop_evt: &Arc<vmm_sys_util::eventfd::EventFd>,
) -> WaitSocketResult {
    use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
    use nix::sys::inotify::{AddWatchFlags, InitFlags, Inotify};
    use std::ffi::OsStr;
    use std::os::unix::io::AsFd;

    // Best-effort: ensure the parent directory exists so the
    // inotify watch can attach. The scheduler creates this
    // directory before bind, but we may race; pre-creating is
    // idempotent.
    let _ = fs::create_dir_all(SCHED_STATS_SOCKET_DIR);

    let inotify = match Inotify::init(InitFlags::IN_CLOEXEC | InitFlags::IN_NONBLOCK) {
        Ok(i) => i,
        Err(e) => {
            tracing::warn!(error = %e, "stats relay: inotify_init failed");
            return WaitSocketResult::Stopped;
        }
    };
    // B2 fix: include IN_ATTRIB so a chmod-on-listen (some
    // schedulers tighten perms after listen()) wakes us; include
    // IN_OPEN so any client that successfully connects (including
    // ourselves on a retry) re-fires the watch even if the
    // initial CREATE-then-connect race already lost. The broader
    // mask catches more edges than IN_CREATE alone, so a connect
    // that fails with ECONNREFUSED post-CREATE has additional
    // events to wake on rather than wedging.
    if let Err(e) = inotify.add_watch(
        SCHED_STATS_SOCKET_DIR,
        AddWatchFlags::IN_CREATE
            | AddWatchFlags::IN_MOVED_TO
            | AddWatchFlags::IN_ATTRIB
            | AddWatchFlags::IN_OPEN,
    ) {
        tracing::warn!(
            error = %e,
            dir = SCHED_STATS_SOCKET_DIR,
            "stats relay: inotify add_watch failed"
        );
        return WaitSocketResult::Stopped;
    }

    // Race-free initial probe: socket may already exist before the
    // watch was added. Try connect; on success skip the loop.
    if stop.load(Ordering::Acquire) {
        return WaitSocketResult::Stopped;
    }
    if let Ok(s) = std::os::unix::net::UnixStream::connect(SCHED_STATS_SOCKET) {
        tracing::debug!("stats relay: connected to scheduler socket (race-free initial probe)");
        return WaitSocketResult::Connected(s);
    }

    // Park on poll(inotify_fd, port_fd, stop_evt). Each wake:
    //   - inotify edge: re-read events; on any event in the
    //     watched dir, retry connect (the B2 expanded mask plus
    //     this any-event retry policy guards against the
    //     IN_CREATE-then-listen() race that left the prior code
    //     waiting on a CREATE-only edge that never came again).
    //   - port edge: B3 fix — host pushed a request before the
    //     scheduler came up. Drain it and reply with the inline
    //     error envelope so the host's request_raw wakes
    //     immediately with NoScheduler instead of waiting for the
    //     scheduler to appear.
    //   - stop_evt edge: shutdown.
    let target = OsStr::new(SCHED_STATS_SOCKET_NAME);
    let mut buf = vec![0u8; RELAY_BUFFER_BYTES];
    loop {
        if stop.load(Ordering::Acquire) {
            return WaitSocketResult::Stopped;
        }
        let inotify_fd = inotify.as_fd();
        let port_fd = port.as_fd();
        // SAFETY: `stop_evt` is held by the surrounding `Arc`, so
        // the raw fd is valid for the whole loop body.
        let stop_evt_fd =
            unsafe { std::os::unix::io::BorrowedFd::borrow_raw(stop_evt.as_raw_fd()) };
        let mut fds = [
            PollFd::new(inotify_fd, PollFlags::POLLIN),
            PollFd::new(port_fd, PollFlags::POLLIN),
            PollFd::new(stop_evt_fd, PollFlags::POLLIN),
        ];
        match poll(&mut fds, PollTimeout::NONE) {
            Ok(_) => {}
            Err(nix::errno::Errno::EINTR) => continue,
            Err(e) => {
                tracing::warn!(error = %e, "stats relay: poll on inotify failed");
                return WaitSocketResult::Stopped;
            }
        };
        let inotify_ready = fds[0]
            .revents()
            .is_some_and(|r| r.contains(PollFlags::POLLIN));
        let port_ready = fds[1]
            .revents()
            .is_some_and(|r| r.contains(PollFlags::POLLIN));
        let stop_ready = fds[2]
            .revents()
            .is_some_and(|r| r.contains(PollFlags::POLLIN));

        // Stop-fd ready? Drain and exit. The only writer is
        // `RelayStopSignal::signal_stop`.
        if stop_ready {
            let _ = stop_evt.read();
            return WaitSocketResult::Stopped;
        }

        // B3: host pushed a request while we're still waiting for
        // the scheduler. Drain whatever bytes are available and
        // reply with the inline error envelope so the request
        // surfaces NoScheduler immediately. A burst that exceeds
        // RELAY_BUFFER_BYTES gets one error reply per drain; the
        // host's request_raw will see the first envelope and
        // return — subsequent envelopes are harmless because
        // they sit in the response_buf as stale bytes that the
        // next request clears.
        if port_ready {
            match port.read(&mut buf) {
                Ok(0) => {
                    // B14: this Ok(0) feeds the outer loop's
                    // consecutive-EOF counter via the PortEof
                    // return. A single Ok(0) is recoverable
                    // (the outer re-arms inotify); after the
                    // configured cap the relay thread exits.
                    tracing::debug!(
                        "stats relay: port read EOF in inotify wait; \
                         returning to outer loop for EOF accounting"
                    );
                    return WaitSocketResult::PortEof;
                }
                Ok(n) => {
                    tracing::debug!(
                        bytes = n,
                        "stats relay: host pushed request while waiting for scheduler; \
                         emitting no-scheduler error envelope"
                    );
                    if let Err(e) = port.write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY) {
                        tracing::warn!(
                            error = %e,
                            "stats relay: port write failed in inotify wait; exiting"
                        );
                        return WaitSocketResult::Stopped;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "stats relay: port read error in inotify wait; exiting"
                    );
                    return WaitSocketResult::Stopped;
                }
            }
        }

        // inotify fd ready: drain events and try connect. B2:
        // try connect on ANY event in the watched directory, not
        // just IN_CREATE for our target — the bind-without-listen
        // window means a CREATE-only check would miss the
        // listen-edge that follows.
        if inotify_ready {
            let events = match inotify.read_events() {
                Ok(e) => e,
                Err(nix::errno::Errno::EINTR) => continue,
                Err(nix::errno::Errno::EAGAIN) => continue,
                Err(e) => {
                    tracing::warn!(error = %e, "stats relay: inotify read_events failed");
                    return WaitSocketResult::Stopped;
                }
            };
            // If any event names our target — or if any event
            // names anything (the dir's only legitimate occupant
            // is our socket plus possible peer scheduler files) —
            // attempt connect. The connect itself is the
            // synchronisation primitive: ECONNREFUSED means we'll
            // wait for the next inotify edge.
            let saw_target_or_any = events
                .iter()
                .any(|ev| ev.name.as_deref() == Some(target) || ev.name.is_some());
            if !saw_target_or_any {
                continue;
            }
            match std::os::unix::net::UnixStream::connect(SCHED_STATS_SOCKET) {
                Ok(s) => {
                    tracing::debug!("stats relay: connected to scheduler socket via inotify edge");
                    return WaitSocketResult::Connected(s);
                }
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "stats relay: socket appeared but connect failed (likely \
                         bind-without-listen race); will retry on next inotify edge"
                    );
                }
            }
        }
    }
}

/// On socket loss, drain whatever request bytes the host has
/// already pushed onto port 2 and answer each readable batch with
/// the inline error envelope. Without this, B12: a request that
/// the host wrote AFTER we forwarded the prior request to the
/// (now-dead) socket would otherwise be carried over into the
/// next relay session, where it would be forwarded to a fresh
/// scheduler — meaning the host's old request gets answered by
/// the new scheduler's stats, not by an error.
///
/// Uses non-blocking poll-with-zero-timeout via PollTimeout::ZERO
/// to drain only what's already queued, then returns. Each drained
/// batch gets one error envelope; the host's request_raw observes
/// the first envelope and surfaces NoScheduler — the rest sit as
/// stale bytes in response_buf that the next request clears.
fn drain_port_emit_errors(port: &mut std::fs::File) {
    use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
    use std::io::ErrorKind;
    use std::os::unix::io::AsFd;

    let mut buf = vec![0u8; RELAY_BUFFER_BYTES];
    loop {
        let port_ready = {
            let port_fd = port.as_fd();
            let mut fds = [PollFd::new(port_fd, PollFlags::POLLIN)];
            match poll(&mut fds, PollTimeout::ZERO) {
                Ok(_) => fds[0]
                    .revents()
                    .is_some_and(|r| r.contains(PollFlags::POLLIN)),
                Err(_) => false,
            }
        };
        if !port_ready {
            break;
        }
        match port.read(&mut buf) {
            Ok(0) => break,
            Ok(_) => {
                if port
                    .write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY)
                    .is_err()
                {
                    break;
                }
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
}

/// Run a single port-↔-socket relay session. Returns
/// [`RelaySessionExit::PortEof`] when `port.read` returned Ok(0)
/// (the outer loop counts these toward the busy-loop budget — see
/// [`SCHED_STATS_RELAY_MAX_CONSECUTIVE_PORT_EOF`]) and
/// [`RelaySessionExit::Other`] for every other clean exit (socket
/// EOF, scheduler error, stop_evt fired, port write error). Uses
/// poll(2) on (port_fd, socket_fd, stop_evt) so the thread blocks
/// in the kernel until exactly one of those fds is readable — no
/// spinning, no timeouts, and `stop_evt` interrupts any blocked
/// I/O within microseconds.
///
/// Single-thread serialization: the relay is the only writer and
/// the only reader of `/dev/vport0p2` inside the guest, so no
/// userspace mutex around the port fd is required. scx_stats
/// requests are strictly request/response on a single socket
/// connection — no req-id multiplexing — so the natural ordering
/// of the relay's per-iteration loop (read host → write socket
/// → read socket → write host) preserves the protocol semantics.
fn run_relay_session(
    port: &mut std::fs::File,
    mut socket: std::os::unix::net::UnixStream,
    stop: &Arc<AtomicBool>,
    stop_evt: &Arc<vmm_sys_util::eventfd::EventFd>,
) -> RelaySessionExit {
    use nix::poll::{PollFd, PollFlags, PollTimeout, poll};
    use std::io::ErrorKind;
    use std::os::unix::io::AsFd;

    let mut buf = vec![0u8; RELAY_BUFFER_BYTES];
    // B6: track socket health across poll iterations. Set false the
    // moment POLLHUP/POLLERR is observed on the socket fd; gate
    // every `socket.write_all` on this flag so we never write into
    // a HUP'd socket (which fails with EPIPE/SIGPIPE and surfaces
    // as a noisy error path). When POLLHUP and POLLIN both arrive
    // in the same poll, drain the buffered POLLIN data first so the
    // host sees the scheduler's last response before we declare
    // the session dead — the kernel keeps already-queued data
    // readable across the half-close, so reading after POLLHUP
    // is well-defined.
    let mut socket_healthy = true;

    while !stop.load(Ordering::Acquire) {
        // Wait for one of: host pushed bytes (port readable),
        // scheduler emitted bytes (socket readable), or shutdown
        // (stop_evt readable). Wrap the poll call in an inner
        // scope so the `fds` array (and the immutable borrows on
        // port + socket it holds) drops before we try to read or
        // write either of them.
        let (port_ready, socket_in, socket_hup_seen, stop_ready) = {
            let port_fd = port.as_fd();
            let socket_fd = socket.as_fd();
            // SAFETY: `stop_evt` is held by the surrounding `Arc`,
            // so the raw fd is valid for the whole inner scope.
            let stop_evt_fd =
                unsafe { std::os::unix::io::BorrowedFd::borrow_raw(stop_evt.as_raw_fd()) };
            let mut fds = [
                PollFd::new(port_fd, PollFlags::POLLIN),
                PollFd::new(socket_fd, PollFlags::POLLIN),
                PollFd::new(stop_evt_fd, PollFlags::POLLIN),
            ];
            match poll(&mut fds, PollTimeout::NONE) {
                Ok(_) => {}
                Err(nix::errno::Errno::EINTR) => continue,
                Err(e) => {
                    tracing::warn!(error = %e, "stats relay: poll failed; exiting session");
                    return RelaySessionExit::Other;
                }
            }
            // Snapshot the revents and drop the borrows.
            let port_rev = fds[0].revents();
            let socket_rev = fds[1].revents();
            let stop_rev = fds[2].revents();
            let port_ready = port_rev.is_some_and(|r| r.contains(PollFlags::POLLIN));
            let socket_in = socket_rev.is_some_and(|r| r.contains(PollFlags::POLLIN));
            // B6: any POLLHUP or POLLERR on the socket — with or
            // without POLLIN — is a permanent transition to
            // unhealthy. POLLHUP+POLLIN means buffered data is
            // still drainable; the same-iteration drain of the
            // socket POLLIN happens below after `socket_healthy`
            // is flipped, because reading from a HUP'd socket with
            // buffered data is well-defined — only WRITES need the
            // gate.
            let socket_hup_seen = socket_rev
                .is_some_and(|r| r.contains(PollFlags::POLLHUP) || r.contains(PollFlags::POLLERR));
            let stop_ready = stop_rev.is_some_and(|r| r.contains(PollFlags::POLLIN));
            (port_ready, socket_in, socket_hup_seen, stop_ready)
        };

        // Flip `socket_healthy` to false IMMEDIATELY when
        // POLLHUP/POLLERR is observed in the current iteration —
        // before any port-read processing. Earlier code flipped
        // the flag at the END of the loop body, which raced when
        // POLLHUP and port-POLLIN arrived in the same revents:
        // the port arm at `if !socket_healthy` saw stale `true`
        // and forwarded the host's request into the HUP'd socket
        // (EPIPE / SIGPIPE). The socket-POLLIN drain below still
        // runs because reading buffered scheduler responses
        // across a half-close remains well-defined; only the
        // WRITE side needs gating.
        if socket_hup_seen {
            socket_healthy = false;
        }

        // Stop edge: drain and exit.
        if stop_ready {
            let _ = stop_evt.read();
            return RelaySessionExit::Other;
        }

        // Host→guest port readable: read bytes and forward to
        // socket. The socket forward is a blocking write — bounded
        // by the kernel's Unix-socket buffer, not by any user
        // timeout. B6: skip the write_all entirely when the socket
        // is already known unhealthy (POLLHUP seen on a prior
        // iteration). Reading the port still drains the host's
        // queued request bytes so they don't pile up in the kernel
        // buffer; we answer with the inline error envelope and
        // exit so the host's pending request_raw wakes with
        // NoScheduler instead of timing out.
        if port_ready {
            let n = match port.read(&mut buf) {
                Ok(0) => {
                    // B14: this Ok(0) is the busy-loop trigger —
                    // surface it through the typed return so the
                    // outer `sched_stats_relay_loop` can count
                    // consecutive port-EOF exits and bail when the
                    // budget is exhausted.
                    tracing::debug!(
                        "stats relay: port read EOF; returning to outer loop \
                         for EOF accounting"
                    );
                    return RelaySessionExit::PortEof;
                }
                Ok(n) => n,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::warn!(error = %e, "stats relay: port read error; exiting session");
                    return RelaySessionExit::Other;
                }
            };
            if !socket_healthy {
                tracing::debug!(
                    bytes = n,
                    "stats relay: port→socket forward skipped (socket already \
                     unhealthy); emitting error envelopes and reconnecting"
                );
                let _ = port.write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY);
                drain_port_emit_errors(port);
                return RelaySessionExit::Other;
            }
            if let Err(e) = socket.write_all(&buf[..n]) {
                tracing::debug!(
                    error = %e,
                    "stats relay: socket write failed; emitting error envelopes and reconnecting"
                );
                // B12: the host may have additional queued
                // requests on the port that we haven't read yet —
                // the failed write_all means we're abandoning the
                // socket without forwarding them. Answer the
                // request that triggered this write_all PLUS any
                // already-queued follow-up requests with error
                // envelopes so they don't survive into the next
                // session and get forwarded to a fresh scheduler.
                // No need to mutate `socket_healthy` here — every
                // arm that reaches a write-failure path returns
                // immediately and the local goes out of scope.
                let _ = port.write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY);
                drain_port_emit_errors(port);
                return RelaySessionExit::Other;
            }
        }

        // Scheduler→host socket readable: read response bytes and
        // forward to port. B6: read POLLIN data even when POLLHUP
        // arrived in the same poll — buffered scheduler responses
        // remain readable across the half-close until the kernel
        // socket buffer drains. `socket_healthy` was already
        // flipped at the top of the loop body if POLLHUP/POLLERR
        // appeared in the same revents; the `!socket_healthy`
        // reconnect block below catches that case after this drain.
        if socket_in {
            let m = match socket.read(&mut buf) {
                Ok(0) => {
                    tracing::debug!(
                        "stats relay: socket EOF; emitting error envelopes and reconnecting"
                    );
                    let _ = port.write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY);
                    drain_port_emit_errors(port);
                    return RelaySessionExit::Other;
                }
                Ok(m) => m,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::debug!(
                        error = %e,
                        "stats relay: socket read error; emitting error envelopes and reconnecting"
                    );
                    let _ = port.write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY);
                    drain_port_emit_errors(port);
                    return RelaySessionExit::Other;
                }
            };
            if let Err(e) = port.write_all(&buf[..m]) {
                tracing::warn!(error = %e, "stats relay: port write failed; exiting session");
                return RelaySessionExit::Other;
            }
        }

        // With `socket_healthy` already flipped at the top
        // of the loop body when POLLHUP/POLLERR arrived, the
        // post-drain reconnect just checks the flag. Reaching
        // this point with `!socket_healthy` means either (a)
        // POLLHUP arrived alone — we exit immediately so we don't
        // spin re-arming poll on a dead fd; or (b) POLLHUP+POLLIN
        // arrived together — we drained the buffered scheduler
        // responses above and now exit. Both cases share the
        // reconnect path: emit the inline error envelope plus
        // drain any queued port requests.
        if !socket_healthy {
            tracing::debug!(
                drained_in = socket_in,
                "stats relay: socket POLLHUP/POLLERR; reconnecting after draining"
            );
            let _ = port.write_all(SCHED_STATS_RELAY_NO_SCHEDULER_REPLY);
            drain_port_emit_errors(port);
            return RelaySessionExit::Other;
        }
    }
    // Reached only when the outer `stop` flag is observed at the
    // top of the loop — an ordinary clean shutdown.
    RelaySessionExit::Other
}

/// Maximum scheduler-log chunk emitted in a single
/// [`crate::vmm::guest_comms::send_sched_log`] frame. Sub-cap of
/// [`crate::vmm::bulk::MAX_BULK_FRAME_PAYLOAD`] so a chunk fits
/// comfortably inside one TLV frame; chunks above this size are
/// split before emission.
const SCHED_LOG_CHUNK_BYTES: usize = 64 * 1024;

/// Send the scheduler log to the host bracketed by
/// [`crate::verifier::SCHED_OUTPUT_START`] /
/// [`crate::verifier::SCHED_OUTPUT_END`] markers. Replaces the
/// prior COM2 dump path: the markers travel verbatim inside the
/// chunk bytes so the host's `parse_sched_output` walker (which
/// scans for the start/end pair after concatenating chunks) keeps
/// working unchanged. The BPF verifier section embedded in the
/// scheduler's stderr / stdout passes through byte-for-byte so a
/// scheduler author still sees the kernel's verifier rejection
/// text in the host-side failure render.
fn dump_sched_output(log_path: &str) {
    crate::vmm::guest_comms::send_sched_log(crate::verifier::SCHED_OUTPUT_START.as_bytes());
    send_sched_log_file(log_path);
    crate::vmm::guest_comms::send_sched_log(crate::verifier::SCHED_OUTPUT_END.as_bytes());
}

/// Walk `/tmp/sched_*.log` and emit each non-empty file as a
/// separate `SCHED_OUTPUT_START` / `SCHED_OUTPUT_END` frame.
/// Captures logs from Op-spawned schedulers (Attach, Replace,
/// Restart) that the boot path's single `dump_sched_output` call
/// cannot reach. Sorted by filename so the host sees logs in
/// spawn order (the monotonic seq suffix from
/// `staged_scheduler_log_path` guarantees lexicographic = temporal).
fn dump_staged_scheduler_logs() {
    let Ok(entries) = fs::read_dir("/tmp") else {
        return;
    };
    let mut paths: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("sched_") && n.ends_with(".log"))
        })
        .collect();
    paths.sort();
    for p in paths {
        if let Some(s) = p.to_str() {
            dump_sched_output(s);
        }
    }
}

/// Read the scheduler log file and emit it to the host as one or
/// more [`crate::vmm::wire::MsgType::SchedLog`] TLV chunks bounded
/// by [`SCHED_LOG_CHUNK_BYTES`]. Empty / missing file is a silent
/// no-op (mirrors the prior `dump_file_to_com2` behaviour where an
/// `Err` from `read_to_string` skipped the dump rather than
/// emitting a partial marker pair).
fn send_sched_log_file(path: &str) {
    let Ok(content) = fs::read_to_string(path) else {
        return;
    };
    let bytes = content.as_bytes();
    let mut start = 0usize;
    while start < bytes.len() {
        let end = (start + SCHED_LOG_CHUNK_BYTES).min(bytes.len());
        crate::vmm::guest_comms::send_sched_log(&bytes[start..end]);
        start = end;
    }
}

/// Send a fixed text snippet (e.g. a "failed to spawn" diagnostic)
/// to the host as a single [`crate::vmm::wire::MsgType::SchedLog`]
/// TLV chunk. The snippet is bounded by `SCHED_LOG_CHUNK_BYTES`
/// like every other chunk; oversized snippets would be rejected
/// by the host-side per-frame cap and are guarded here by
/// truncating the input before the call.
fn send_sched_log_text(s: &str) {
    let bytes = s.as_bytes();
    let cap = SCHED_LOG_CHUNK_BYTES.min(bytes.len());
    crate::vmm::guest_comms::send_sched_log(&bytes[..cap]);
}

/// Enable sched_ext_dump trace event and pipe trace_pipe to COM1 in a
/// background thread. Returns the stop flag and thread join handle.
///
/// The reader opens trace_pipe with `O_NONBLOCK` and uses `poll()` on
/// a 200ms cadence so the loop is responsive to `stop` even when the
/// kernel never emits a sched_ext_dump event. A blocking `read(2)` on
/// trace_pipe parks the task in `tracing_wait_pipe` (kernel/trace/trace.c);
/// once that wait is entered with `iter->pos == 0` (no event ever
/// dispatched into the iterator), the kernel re-enters `wait_on_pipe`
/// after every wake because the inner loop in `tracing_wait_pipe` only
/// breaks when `!tracer_tracing_is_on(tr) && iter->pos`. Writing 0 to
/// `tracing_on` does fire `ring_buffer_wake_waiters`, but the
/// trace_pipe path supplies `wait_pipe_cond` (not the default
/// `rb_wait_once`) and that condition only flips when `iter->closed`
/// or `iter->wait_index` change — neither is touched by the trace_pipe
/// fops, so the wake produces a spurious return into `tracing_wait_pipe`
/// which immediately re-sleeps. Going non-blocking sidesteps the kernel
/// wait entirely: every iteration the userspace thread checks the stop
/// flag, polls for data, and drains any pending events without ever
/// parking in the kernel.
fn start_trace_pipe() -> (Option<Arc<AtomicBool>>, Option<std::thread::JoinHandle<()>>) {
    if Path::new(TRACE_SCHED_EXT_DUMP_ENABLE).exists() {
        let _ = fs::write(TRACE_SCHED_EXT_DUMP_ENABLE, "1");

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        let handle = std::thread::Builder::new()
            .name("trace-pipe".into())
            .spawn(move || {
                use std::os::unix::fs::OpenOptionsExt;
                let Ok(mut trace) = fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_NONBLOCK)
                    .open(TRACE_PIPE)
                else {
                    return;
                };
                let Ok(mut com1) = fs::OpenOptions::new().write(true).open(COM1) else {
                    return;
                };
                let mut buf = [0u8; 4096];
                // Tier-2 (lossless dump): rolling tail so an exit-dump
                // marker split across two reads is still matched by
                // `scan_dump_markers`.
                let mut scan_tail: Vec<u8> = Vec::new();
                loop {
                    // Break promptly once teardown signals stop. The only
                    // ftrace event ktstr enables, `sched_ext_dump`, fires
                    // ONLY as the one-shot crash dump (kernel scx_dump_state,
                    // emitted contiguously before the disable workfn); the
                    // disable itself emits no further trace. So trace_pipe
                    // at stop holds only the residual TAIL of that one dump
                    // still draining — a per-task dump (scx_dump_state with
                    // dump_all_tasks) whose size scales with runnable-task
                    // count, forwarded byte-by-byte over the slow PIO COM1
                    // UART, so its drain time scales with task count — that
                    // byte-by-byte COM1 forwarding is the tens-of-seconds
                    // cost (the kernel disable itself is ms-scale), and the
                    // dump is NOT disable-emitted.
                    // Forwarding the whole tail pinned `trace_handle.join`
                    // (the prior `drain_deadline` was checked only between
                    // polls, never inside the inner drain). Dropping it on
                    // stop is safe: teardown sets `stop` only AFTER the
                    // dump-complete latch (end-marker already on COM1) or
                    // the `SCX_DUMP_CAPTURE_TIMEOUT` bound; on the bound
                    // this ftrace copy truncates, but the full dump is
                    // captured via the scheduler's stderr log
                    // (`dump_sched_output`, scx_utils' `ei->dump`) over the
                    // fast bulk port — the authoritative copy.
                    if stop_clone.load(Ordering::Acquire) {
                        break;
                    }

                    let mut pollfds = [PollFd::new(trace.as_fd(), PollFlags::POLLIN)];
                    match poll(&mut pollfds, PollTimeout::from(200u16)) {
                        Ok(0) => continue,
                        Ok(_) => {}
                        Err(nix::errno::Errno::EINTR) => continue,
                        Err(_) => break,
                    }
                    if let Some(revents) = pollfds[0].revents() {
                        if revents.intersects(PollFlags::POLLERR | PollFlags::POLLNVAL) {
                            break;
                        }
                        if !revents.contains(PollFlags::POLLIN) {
                            // POLLHUP without POLLIN means no buffered
                            // data to drain; with POLLIN, fall through
                            // to read first so events that arrived
                            // before hangup still reach COM1.
                            if revents.contains(PollFlags::POLLHUP) {
                                break;
                            }
                            continue;
                        }
                    }

                    // Drain the bytes poll reported ready, re-checking
                    // `stop` after each chunk (below) so a continuous read
                    // cannot pin the reader here. Inner-loop exits use
                    // `break` (not `return`) so the outer loop's
                    // prompt-stop check + poll fd-state handling
                    // (POLLHUP/POLLERR) run on the next iteration.
                    loop {
                        match trace.read(&mut buf) {
                            Ok(0) => break,
                            Ok(n) => {
                                let _ = com1.write_all(&buf[..n]);
                                scan_dump_markers(&buf[..n], &mut scan_tail);
                                // Re-check stop mid-batch so a continuous
                                // stream cannot pin the reader here; break
                                // to the prompt-stop check at the outer top.
                                if stop_clone.load(Ordering::Acquire) {
                                    break;
                                }
                            }
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                            Err(_) => break,
                        }
                    }
                }
            })
            .ok();
        (Some(stop), handle)
    } else {
        (None, None)
    }
}

/// Trailing bytes of the previous trace_pipe chunk retained by
/// [`scan_dump_markers`] so an exit-dump marker split across a read
/// boundary is still matched. The longest marker
/// (`SCX_EV_SUB_BYPASS_DISPATCH`, 26 bytes) fits with margin.
const SCAN_TAIL_KEEP: usize = 32;

/// Bound on how long teardown waits for the exit dump's end-marker to be
/// forwarded to COM1 before disabling the `sched_ext_dump` tracepoint.
/// The kernel builds+emits the whole dump synchronously at crash time
/// (the `scx_disable_irq_workfn` irq path), so a SMALL dump's marker is
/// forwarded well before teardown and the wait returns at once. A LARGE
/// dump (many runnable tasks → scx_dump_state(dump_all_tasks) builds a
/// per-task dump) can take tens of seconds to forward byte-by-byte over
/// the slow PIO COM1 UART; this bound caps that so a big crash dump
/// cannot wedge teardown. On the bound the ftrace copy is truncated; the
/// authoritative full dump is the scheduler stderr log
/// (`dump_sched_output`) over the fast bulk port.
const SCX_DUMP_CAPTURE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Fired by the trace_pipe reader when the `sched_ext_dump` tracepoint
/// emits its FIRST line this run — i.e. an exit dump started streaming.
/// Read by teardown to decide whether to wait for completion: clean runs
/// never start a dump, so they never pay the [`SCX_DUMP_CAPTURE_TIMEOUT`]
/// wait.
static SCX_DUMP_STARTED_LATCH: OnceLock<Arc<Latch>> = OnceLock::new();

/// Fired by the trace_pipe reader when the exit dump's end-marker
/// (`SCX_EV_SUB_BYPASS_DISPATCH`, the last event-counter line, or
/// `~~~~ TRUNCATED ~~~~`) reaches it — the full dump is captured.
/// Awaited by teardown before disabling the dump tracepoint so a fast
/// crash teardown does not truncate the dump mid-emit.
static SCX_DUMP_COMPLETE_LATCH: OnceLock<Arc<Latch>> = OnceLock::new();

fn scx_dump_started_latch() -> Arc<Latch> {
    SCX_DUMP_STARTED_LATCH
        .get_or_init(|| Arc::new(Latch::new()))
        .clone()
}

fn scx_dump_complete_latch() -> Arc<Latch> {
    SCX_DUMP_COMPLETE_LATCH
        .get_or_init(|| Arc::new(Latch::new()))
        .clone()
}

/// True if `needle` occurs in `haystack`.
fn slice_find(haystack: &[u8], needle: &[u8]) -> bool {
    needle.len() <= haystack.len() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// Scan a freshly-read trace_pipe chunk for the sched_ext exit-dump
/// start + end markers, firing [`scx_dump_started_latch`] /
/// [`scx_dump_complete_latch`]. `tail` carries the last
/// [`SCAN_TAIL_KEEP`] bytes of the previous chunk so a marker split
/// across a read boundary is still matched. No-op once the dump is
/// complete (the common case after one full dump).
fn scan_dump_markers(chunk: &[u8], tail: &mut Vec<u8>) {
    if scx_dump_complete_latch().is_set() {
        return;
    }
    tail.extend_from_slice(chunk);
    if !scx_dump_started_latch().is_set() && slice_find(tail, b"sched_ext_dump:") {
        scx_dump_started_latch().set();
    }
    if slice_find(tail, b"SCX_EV_SUB_BYPASS_DISPATCH") || slice_find(tail, b"~~~~ TRUNCATED ~~~~") {
        scx_dump_complete_latch().set();
        tail.clear();
        return;
    }
    let excess = tail.len().saturating_sub(SCAN_TAIL_KEEP);
    if excess > 0 {
        tail.drain(..excess);
    }
}

/// Process-wide latch fired by the guest's `hvc0_poll_loop` when the
/// host's `bpf-map-write` thread pushes `SIGNAL_BPF_WRITE_DONE` through
/// virtio-console RX.
///
/// Producer: [`hvc0_poll_loop`] (this file). Consumer: the scenario
/// executor's [`crate::scenario::Ctx::wait_for_map_write`] gate
/// (in `scenario::ops`). A test that declares `bpf_map_write` on
/// its `KtstrTestEntry` flips `wait_for_map_write=true`; the
/// scenario runner then blocks on this latch's
/// [`Latch::wait_timeout`] before starting the workload phase, so
/// the workload never observes a stale BPF map value.
///
/// `OnceLock` so the first caller materialises the [`Latch`] and
/// every subsequent caller (producer or consumer) shares the same
/// instance. `Arc` so callers can hold the latch across
/// thread-spawn boundaries without re-resolving the static.
static BPF_MAP_WRITE_DONE_LATCH: OnceLock<Arc<Latch>> = OnceLock::new();

/// Shared `accessor_ready` latch — fired by `hvc0_poll_loop` on
/// `SIGNAL_ACCESSOR_READY`, awaited by
/// `scenario::ops::await_accessor_ready`. Mirrors
/// [`BPF_MAP_WRITE_DONE_LATCH`].
static ACCESSOR_READY_LATCH: OnceLock<Arc<Latch>> = OnceLock::new();

/// Lazily materialise and return the shared `bpf_map_write_done`
/// latch. Both the producer (`hvc0_poll_loop`) and consumer (scenario
/// `wait_for_map_write` gate) reach for this — the first caller
/// installs the [`Latch`] into [`BPF_MAP_WRITE_DONE_LATCH`], every
/// subsequent caller observes the same instance.
pub(crate) fn bpf_map_write_done_latch() -> Arc<Latch> {
    BPF_MAP_WRITE_DONE_LATCH
        .get_or_init(|| Arc::new(Latch::new()))
        .clone()
}

/// Lazily materialise and return the shared `accessor_ready` latch.
/// The producer (`hvc0_poll_loop`, on `SIGNAL_ACCESSOR_READY`) and the
/// consumer (`scenario::ops::await_accessor_ready`) both reach for this;
/// the first caller installs the [`Latch`], every subsequent caller
/// observes the same instance. Mirrors [`bpf_map_write_done_latch`].
pub(crate) fn accessor_ready_latch() -> Arc<Latch> {
    ACCESSOR_READY_LATCH
        .get_or_init(|| Arc::new(Latch::new()))
        .clone()
}

/// Start the hvc0 wake-byte poll loop.
///
/// Spawns a background thread that polls `/dev/hvc0` for host→guest
/// wake bytes and dispatches SysRq-D / shutdown / bpf-map-write-done
/// based on the wake byte. Returns the thread's stop flag so callers
/// can request termination on teardown.
///
/// `trace_stop` is the trace_pipe reader's stop flag. The graceful
/// shutdown handler sets it so the reader enters drain mode.
fn start_hvc0_poll(trace_stop: Option<Arc<AtomicBool>>) -> Option<Arc<AtomicBool>> {
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    std::thread::Builder::new()
        .name("hvc0-poll".into())
        .spawn(move || {
            hvc0_poll_loop(&stop_clone, trace_stop.as_deref());
        })
        .ok();

    Some(stop)
}

/// Poll `/dev/hvc0` for host→guest wake bytes and dispatch SysRq-D /
/// shutdown / bpf-map-write-done based on the wake byte alone.
///
/// Wake source: opens `/dev/hvc0` non-blocking (`O_NONBLOCK`) and
/// `poll()`s the fd with `POLLIN` at a 1000 ms safety timeout. The
/// host pushes a byte via `VirtioConsole::queue_input` whenever it
/// requests a dump (`SIGNAL_VC_DUMP`), a graceful shutdown
/// (`SIGNAL_VC_SHUTDOWN`), or a `bpf-map-write`-complete notification
/// (`SIGNAL_BPF_WRITE_DONE`). The poll wakes within microseconds of
/// the push.
///
/// On any wake the loop:
///   1. scans every drained hvc0 byte for `SIGNAL_VC_DUMP`; on
///      observing one, triggers SysRq-D via `/proc/sysrq-trigger`.
///   2. scans every drained hvc0 byte for `SIGNAL_BPF_WRITE_DONE`;
///      on observing one, fires [`bpf_map_write_done_latch`] so the
///      scenario's `wait_for_map_write` gate resumes.
///   3. scans every drained hvc0 byte for `SIGNAL_VC_SHUTDOWN`; on
///      observing one, drives graceful shutdown (set `trace_stop`,
///      disable tracing, flush stdio + serial) and breaks.
fn hvc0_poll_loop(stop: &AtomicBool, trace_stop: Option<&AtomicBool>) {
    use std::os::unix::io::AsRawFd;

    // Open the virtio-console wake fd. Failure here used to be
    // `.expect()`d, which panicked the worker thread; the
    // process-wide panic hook installed at PID-1 entry calls
    // `force_reboot()`, so a transient open failure (e.g. devtmpfs
    // not yet populated when the thread spawns) tore the VM down
    // before any test could dispatch. Log + return instead so the
    // poll loop simply doesn't deliver wake bytes for this boot —
    // tests that rely on `bpf_map_write` notification will time out
    // on their `wait_for_map_write` latch with a recoverable error
    // instead of a forced reboot.
    let hvc0 = match fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(HVC0)
    {
        Ok(f) => f,
        Err(e) => {
            write_com2(&format!(
                "ktstr-init: hvc0 poll loop disabled — open {HVC0}: {e}"
            ));
            return;
        }
    };
    let poll_timeout_ms: PollTimeout = 1000u16.into();

    while !stop.load(Ordering::Acquire) {
        let borrowed = unsafe { BorrowedFd::borrow_raw(hvc0.as_raw_fd()) };
        let mut fds = [PollFd::new(borrowed, PollFlags::POLLIN)];
        match poll(&mut fds, poll_timeout_ms) {
            Ok(0) => continue,
            Ok(_) => {}
            Err(nix::errno::Errno::EINTR) => continue,
            Err(_) => break,
        }
        // Inspect revents before reading: a host-side virtio-console
        // disconnect raises POLLHUP/POLLERR permanently, and without
        // this guard the bare `read().unwrap_or(0)` below returns
        // Ok(0) every iteration, the next `poll()` returns
        // immediately because the hangup is still latched, and the
        // loop spins burning CPU until `stop` is set. Mirrors the
        // pattern in `start_trace_pipe` (above): break on
        // POLLERR/POLLNVAL, break on POLLHUP-without-POLLIN, and
        // skip the read on a wake without POLLIN.
        if let Some(revents) = fds[0].revents() {
            if revents.intersects(PollFlags::POLLERR | PollFlags::POLLNVAL) {
                break;
            }
            if !revents.contains(PollFlags::POLLIN) {
                if revents.contains(PollFlags::POLLHUP) {
                    break;
                }
                continue;
            }
        }
        let mut buf = [0u8; 16];
        let mut hvc_ref: &fs::File = &hvc0;
        // Retry on EINTR (the read was interrupted by a signal before
        // returning data). The previous `unwrap_or(0)` collapsed both
        // EINTR and EIO into 0 bytes, masking transient signal races
        // (drops a real wake byte) and permanent device errors (silent
        // hang in the next poll iteration). Treat:
        //   - Ok(n): consume n bytes and dispatch signals below. An
        //     `Ok(0)` here is rare (poll already confirmed POLLIN)
        //     but harmless — the byte-contains checks no-op and the
        //     outer loop iterates normally, same as the original
        //     `unwrap_or(0)` behaviour for that case.
        //   - EINTR: retry the read inline; poll already confirmed
        //     POLLIN, so the wake byte is still in the device's RX
        //     queue waiting to be drained.
        //   - other Err: log via tracing::warn and break the outer
        //     poll loop. A non-EINTR read error after POLLIN means
        //     the device is in an unrecoverable state (host-side
        //     disconnect that didn't surface as POLLHUP, kernel-side
        //     I/O error, fd revoked) and continuing would either
        //     spin on the same error or silently miss every wake
        //     byte for the rest of the run.
        let n = 'read_retry: loop {
            match hvc_ref.read(&mut buf) {
                Ok(n) => break 'read_retry Some(n),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::warn!(
                        err = %e,
                        "ktstr-init: hvc0 read failed; aborting poll loop"
                    );
                    break 'read_retry None;
                }
            }
        };
        let Some(n) = n else { break };
        if buf[..n].contains(&crate::vmm::virtio_console::SIGNAL_VC_DUMP) {
            let _ = fs::write("/proc/sysrq-trigger", "D");
        }
        if buf[..n].contains(&crate::vmm::virtio_console::SIGNAL_BPF_WRITE_DONE) {
            bpf_map_write_done_latch().set();
        }
        if buf[..n].contains(&crate::vmm::virtio_console::SIGNAL_ACCESSOR_READY) {
            accessor_ready_latch().set();
        }
        if buf[..n].contains(&crate::vmm::virtio_console::SIGNAL_VC_SHUTDOWN) {
            tracing::info!("ktstr-init: shutdown request received, draining");
            if let Some(ts) = trace_stop {
                ts.store(true, Ordering::Release);
            }
            let _ = fs::write(TRACE_TRACING_ON, "0");
            let _ = std::io::stdout().flush();
            let _ = std::io::stderr().flush();
            if let Ok(f) = fs::OpenOptions::new().write(true).open(COM1) {
                unsafe {
                    libc::tcdrain(std::os::unix::io::AsRawFd::as_raw_fd(&f));
                }
            }
            if let Ok(f) = fs::OpenOptions::new().write(true).open(COM2) {
                unsafe {
                    libc::tcdrain(std::os::unix::io::AsRawFd::as_raw_fd(&f));
                }
            }
            break;
        }
    }
}

/// Stop handle for the sched-exit monitor. Carries the
/// `Arc<AtomicBool>` source-of-truth flag, a writable eventfd handle
/// the cleanup site uses to wake the monitor thread out of `poll(2)`
/// without waiting for the legacy 250 ms cadence, and the monitor
/// thread's `JoinHandle` so the cleanup site can wait for the
/// thread to actually exit before proceeding.
///
/// Cleanup contract: before any action that could be misinterpreted
/// by the monitor as an unexpected scheduler exit (e.g. `child.kill()`
/// on the scheduler), the cleanup site MUST call
/// [`SchedExitStop::stop_and_join`] (or its equivalent of
/// `store(true, Release)` + [`SchedExitStop::wake`] + joining the
/// thread). Otherwise the monitor races: it sees `/proc/{pid}` gone
/// after the kill, takes the `if exited` branch, and emits
/// `MSG_TYPE_SCHED_EXIT` to the host, which terminates the VM
/// before the orderly `MSG_TYPE_EXIT` frame can be sent.
///
/// The bool is the source of truth; the eventfd write delivers the
/// edge that pulls the thread out of an indefinite `poll`. The
/// eventfd is owned by this struct on the writer side and by the
/// monitor thread on the reader side; both sides drop their fds when
/// the run ends, so the kernel-side counter is reclaimed cleanly.
pub(crate) struct SchedExitStop {
    /// Stop flag the monitor thread polls under `Acquire` ordering at
    /// every loop iteration. Setting `true` is the only way to make
    /// the thread exit through its top-of-loop early-return arm; the
    /// eventfd below is the wake-edge that pairs with this store.
    pub(crate) stop: Arc<AtomicBool>,
    /// Owned eventfd write side. `wake()` writes `1` here; the
    /// monitor's `poll(2)` returns within microseconds. `None` when
    /// `eventfd(2)` failed at monitor spawn (legacy 250 ms timeout
    /// still bounds wake latency in that degraded path).
    wake_fd: Option<OwnedFd>,
    /// Monitor thread join handle. `None` when
    /// `std::thread::Builder::spawn` failed (the monitor never
    /// started; nothing to join). Consumed by
    /// [`SchedExitStop::stop_and_join`].
    join_handle: Option<std::thread::JoinHandle<()>>,
}

impl SchedExitStop {
    /// Wake the monitor thread out of its `poll(2)` wait. Idempotent
    /// — eventfd in counter mode coalesces multiple writes into a
    /// single wake. EAGAIN under `EFD_NONBLOCK` (counter saturation —
    /// physically impossible with a single writer + 64-bit counter)
    /// is silently absorbed; the `Acquire`-loaded `stop` bool above
    /// remains the source of truth.
    pub(crate) fn wake(&self) {
        if let Some(ref fd) = self.wake_fd {
            // SAFETY: `fd` is the owned write side of an eventfd
            // created with `EFD_NONBLOCK`; a single 8-byte write of
            // a non-zero u64 advances the counter and edge-fires
            // every reader's `poll(POLLIN)`. The bytes pointer is a
            // 64-bit aligned local; `count` is exactly 8 as
            // eventfd(2) requires.
            let val: u64 = 1;
            let bytes = val.to_ne_bytes();
            let _ = unsafe {
                libc::write(
                    fd.as_raw_fd(),
                    bytes.as_ptr() as *const libc::c_void,
                    bytes.len(),
                )
            };
        }
    }

    /// Atomically request stop and wait for the monitor thread to
    /// exit. Sets `stop=true` (Release) and writes the wake eventfd
    /// so the monitor's `poll(2)` returns within microseconds, then
    /// joins the thread. After this returns, the monitor has
    /// observed `stop=true` at the top of its loop and exited
    /// without sending `MSG_TYPE_SCHED_EXIT` — making it safe for
    /// the caller to proceed with actions (like killing the
    /// scheduler child) that the monitor would otherwise interpret
    /// as an unexpected scheduler exit.
    ///
    /// `JoinHandle::join` propagates a panic from the monitor closure
    /// as `Err`; it is consumed and ignored — a panicked monitor is
    /// already dead and there is no recovery path during teardown.
    pub(crate) fn stop_and_join(self) {
        self.stop.store(true, Ordering::Release);
        self.wake();
        if let Some(handle) = self.join_handle {
            let _ = handle.join();
        }
    }
}

/// Monitor the scheduler child process for unexpected exit.
///
/// Blocks the monitor thread in `poll(2)` against the scheduler's
/// pidfd plus a stop-eventfd; the wait returns when either the
/// child exits (pidfd POLLIN edge from the kernel's `do_notify_pidfd`)
/// or the cleanup site fires the stop-eventfd. `/proc/{pid}` is
/// re-checked post-wake to catch the rare "pidfd opened after kernel
/// reaped" race. When `suppress_com2` is false (normal mode), writes
/// MSG_TYPE_SCHED_EXIT to the bulk port and dumps the scheduler log
/// to COM2. The host detects the bulk message and can terminate the
/// VM early. When `suppress_com2` is true (probes active), both the
/// SCHED_EXIT signal and COM2 dump are suppressed — the probe
/// pipeline handles crash detection via tp_btf/sched_ext_exit
/// instead, and the VM must stay alive for the probe thread to emit
/// output.
///
/// Uses procfs instead of waitpid because SIGCHLD is SIG_IGN (the kernel
/// auto-reaps children, making waitpid return ECHILD).
///
/// The returned [`SchedExitStop`] carries the `Arc<AtomicBool>` the
/// monitor reads, an eventfd the cleanup site writes via
/// [`SchedExitStop::wake`] to drop wake latency from 250 ms (legacy
/// poll timeout) to microseconds, and the monitor thread's
/// `JoinHandle` so [`SchedExitStop::stop_and_join`] can confirm the
/// thread has exited before the caller proceeds with actions
/// (e.g. `child.kill()`) the monitor would otherwise interpret as
/// an unexpected scheduler exit.
///
/// Returns None when no scheduler is running.
fn start_sched_exit_monitor(
    sched_pid: Option<u32>,
    log_path: Option<&str>,
    suppress_com2: Arc<AtomicBool>,
    probe_output_done: Option<Arc<crate::sync::Latch>>,
) -> Option<SchedExitStop> {
    let pid = sched_pid?;
    let proc_path = format!("/proc/{pid}");
    let log_path = log_path.map(|s| s.to_string());
    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = stop.clone();

    // Allocate a stop-eventfd. Two fds are needed: one owned by the
    // monitor thread (read + close on exit), one owned by the
    // [`SchedExitStop`] writer (`wake` writes here). `dup(2)` shares
    // the underlying counter so a write on either fd advances both
    // sides' visibility. EFD_NONBLOCK so a doubled cleanup path can't
    // stall behind a saturated counter; EFD_CLOEXEC so a future
    // `Command::new` from this thread doesn't leak the fd into a
    // child.
    //
    // `eventfd(2)` failure (extremely unlikely on KVM hosts — the
    // syscall is unconditionally available since kernel 2.6.22) falls
    // back to the legacy 250 ms `poll(2)` timeout: stop still works
    // via the `Acquire`-loaded bool, just with a worst-case 250 ms
    // wake latency instead of microseconds.
    let (monitor_fd, writer_fd): (Option<OwnedFd>, Option<OwnedFd>) = {
        let raw = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if raw < 0 {
            let err = std::io::Error::last_os_error();
            tracing::warn!(
                err = %err,
                "ktstr-init: sched-exit-mon eventfd allocation failed; \
                 falling back to 250 ms stop poll cadence"
            );
            (None, None)
        } else {
            // SAFETY: `eventfd(2)` returned a fresh non-negative fd
            // owned by this caller. Wrapping in `OwnedFd` transfers
            // close-on-drop responsibility; `try_clone` issues a
            // `dup` so writer and monitor each carry an independent
            // fd that addresses the same kernel-side counter. A
            // dup failure leaves the monitor fd alive and disables
            // the wake path (degrades to the no-eventfd branch).
            let monitor_fd = unsafe { OwnedFd::from_raw_fd(raw) };
            match monitor_fd.try_clone() {
                Ok(writer_fd) => (Some(monitor_fd), Some(writer_fd)),
                Err(e) => {
                    tracing::warn!(
                        err = %e,
                        "ktstr-init: sched-exit-mon eventfd dup failed; \
                         falling back to 250 ms stop poll cadence"
                    );
                    (Some(monitor_fd), None)
                }
            }
        }
    };

    let join_handle = std::thread::Builder::new()
        .name("sched-exit-mon".into())
        .spawn(move || {
            // pidfd_open lets us block on SIGCHLD-equivalent
            // notification for the scheduler process exit instead
            // of polling /proc/{pid} on a sleep cadence.
            // SAFETY: pid is the scheduler's stable pid for the
            // run; pidfd_open(2) accepts any process the caller
            // can signal (we are pid 1). pidfd_open has been
            // available since kernel 5.3 (2019); ktstr targets
            // 6.16+ where it is unconditionally present, so the
            // procfs fallback is dead code. A failure here means
            // the kernel rejected the syscall entirely (sandbox /
            // seccomp filter); abort the monitor rather than
            // fabricate a polling fallback that hides the
            // configuration error.
            let pidfd = unsafe {
                libc::syscall(libc::SYS_pidfd_open, pid as libc::c_int, 0u32) as libc::c_int
            };
            if pidfd < 0 {
                tracing::error!(
                    pid,
                    err = %std::io::Error::last_os_error(),
                    "ktstr-init: pidfd_open failed for sched — sched exit monitor disabled",
                );
                return;
            }
            // The monitor-side stop fd's raw value, or `-1` when the
            // caller's eventfd allocation or dup failed. `-1` in a
            // pollfd entry is valid: the kernel ignores the slot
            // (returns revents=0), so the same `poll(2)` call works
            // on the degraded path with a finite timeout that
            // re-checks `stop` periodically.
            let stop_fd = monitor_fd.as_ref().map(|f| f.as_raw_fd()).unwrap_or(-1);
            // Poll timeout policy: when the stop eventfd is live
            // (`stop_fd >= 0`), a stop request fires the eventfd
            // edge and the wait returns within microseconds — so an
            // indefinite `-1` timeout is correct; the loop never has
            // to wake just to re-check `stop`. When the eventfd
            // allocation degraded to `None`, the legacy 250 ms
            // cadence is the only path that pulls the thread out
            // of the wait, so we fall back to that timeout.
            let poll_timeout: i32 = if stop_fd >= 0 { -1 } else { 250 };
            while !stop_clone.load(Ordering::Acquire) {
                let exited = {
                    // pidfd POLLIN fires at child exit (kernel
                    // `pidfd_poll` in `fs/pidfs.c` checks
                    // `exit_state`, woken via `do_notify_pidfd`
                    // from `exit_notify`). Adding the stop eventfd
                    // alongside makes a stop request also wake the
                    // poll, so cleanup latency drops from the
                    // legacy 250 ms (re-checking `stop` after each
                    // `poll` timeout) to the kernel's eventfd
                    // wakeup latency (microseconds).
                    //
                    // Re-checking proc_path post-`poll` is a
                    // belt-and-suspenders against the rare
                    // "pidfd was opened but the kernel reaped
                    // before we entered poll" race — an exited
                    // child's pidfd POLLIN may already be latched
                    // by the time we add it to the poll set;
                    // checking proc_path independently catches
                    // that case.
                    let mut pfds = [
                        libc::pollfd {
                            fd: pidfd,
                            events: libc::POLLIN,
                            revents: 0,
                        },
                        libc::pollfd {
                            fd: stop_fd,
                            events: libc::POLLIN,
                            revents: 0,
                        },
                    ];
                    // SAFETY: pfds is a 2-element pollfd array on
                    // the local stack; nfds matches. A `stop_fd`
                    // value of `-1` is valid per poll(2) — the
                    // kernel skips that slot. Return value not
                    // consulted — the loop re-checks the stop
                    // flag and the proc path each iteration
                    // regardless.
                    let _ = unsafe {
                        libc::poll(pfds.as_mut_ptr(), pfds.len() as libc::nfds_t, poll_timeout)
                    };
                    !Path::new(&proc_path).exists()
                };
                if exited {
                    if suppress_com2.load(Ordering::Acquire) {
                        // Probes active: wait event-driven on the
                        // probe thread's `output_done` latch.
                        // Outer wall-clock VM timeout is the
                        // safety net for a hung probe — adding a
                        // local timer would cap teardown latency
                        // but also truncate slow-but-progressing
                        // probe drains, which is the exact bug
                        // we're avoiding here.
                        if let Some(ref done) = probe_output_done {
                            done.wait();
                        }
                    } else if let Some(ref path) = log_path {
                        dump_sched_output(path);
                    }
                    // Suppress SchedExit when the host cleanup
                    // initiated the kill (stop flag set before
                    // child.kill). Without this gate, Phase 6
                    // child.kill → pidfd POLLIN → monitor enters
                    // this branch → sends SchedExit → host sets
                    // kill=true → BSP exits with ExternalKill
                    // before the guest reaches send_exit,
                    // producing exit_code=-1 on a clean run.
                    if stop_clone.load(Ordering::Acquire) {
                        unsafe {
                            libc::close(pidfd);
                        }
                        return;
                    }
                    let exit_code: i32 = 1;
                    crate::vmm::guest_comms::send_sched_exit(exit_code);
                    // SAFETY: pidfd is owned by this thread
                    // and is no longer used after close.
                    unsafe {
                        libc::close(pidfd);
                    }
                    // `monitor_fd` (Option<OwnedFd>) drops here on
                    // function return — the OwnedFd's Drop closes
                    // the read side of the stop eventfd. The
                    // writer-side `OwnedFd` lives on the
                    // SchedExitStop returned to the caller.
                    return;
                }
                // Drain any pending stop-eventfd reads so the next
                // `poll` doesn't immediately re-fire on the same
                // edge. The `stop` AtomicBool is the source of
                // truth (re-checked at the top of the loop); the
                // eventfd is purely a wake-edge, so a missed read
                // is benign — the next iteration's poll wakes
                // either way. EAGAIN under EFD_NONBLOCK (counter
                // already 0 from a racing reader, or no edge
                // arrived) is the steady-state non-stop case.
                if stop_fd >= 0 {
                    let mut buf = [0u8; 8];
                    // SAFETY: `stop_fd` is the borrowed read side
                    // of an eventfd, valid for the lifetime of
                    // this thread (the OwnedFd is owned by the
                    // closure's `monitor_fd` and not dropped
                    // until the closure returns). `buf` is an
                    // 8-byte stack slot matching eventfd(2)'s
                    // 8-byte read requirement.
                    let _ = unsafe {
                        libc::read(stop_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len())
                    };
                }
            }
            // SAFETY: same as above — close on exit path.
            unsafe {
                libc::close(pidfd);
            }
            // `monitor_fd` drops here as the closure returns.
        })
        .ok();

    Some(SchedExitStop {
        stop,
        wake_fd: writer_fd,
        join_handle,
    })
}

/// Execute shell-script-like commands from a file.
///
/// Handles the patterns used by sched_enable/sched_disable scripts:
/// - `echo VALUE > /path` (write VALUE to a file)
/// - Lines starting with `#` are comments
/// - Empty lines are ignored
///
/// # Failure surface
///
/// File-not-found is a legitimate "no script" condition (the
/// sched_enable/sched_disable hooks are optional per
/// `ShellTestDescriptor`). Logged at debug level and returns
/// silently. All other read errors are logged at error level —
/// the file exists but couldn't be read (permission denied,
/// I/O error, etc.) is a real defect.
///
/// Per-line failures (file-write failures, unsupported commands)
/// are counted and reported via a single error-level summary at
/// the end. The script is not aborted on first failure —
/// sched_enable/sched_disable hooks are typically independent
/// settings (cpufreq governor, scheduler sysctl, tracing knobs),
/// so the operator gets partial-apply behavior with a loud
/// summary instead of silent partial-apply. Catches
/// silent-drop violations where a typo'd `/sys/`
/// path silently dropped before this rewrite.
#[tracing::instrument]
fn exec_shell_script(path: &str) {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::debug!(path, "ktstr-init: exec_shell_script: no script (skipping)");
            return;
        }
        Err(e) => {
            tracing::error!(path, err = %e, "ktstr-init: exec_shell_script: read failed");
            return;
        }
    };

    let mut ok_count = 0u32;
    let mut fail_count = 0u32;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if exec_shell_line(line).is_ok() {
            ok_count += 1;
        } else {
            fail_count += 1;
        }
    }
    if fail_count > 0 {
        tracing::error!(
            path,
            ok_count,
            fail_count,
            "ktstr-init: exec_shell_script partial-apply: {fail_count} line(s) failed, {ok_count} line(s) ok"
        );
    }
}

/// Execute a single shell-like command line.
///
/// Supports:
/// - `echo VALUE > /path` — write VALUE followed by newline to /path
///
/// Returns `Err(())` on file-write failure or unsupported command
/// so the caller can count partial-apply failures and emit a
/// summary. The per-line error is logged here; the unit-typed
/// `Err` is only a counter signal.
fn exec_shell_line(line: &str) -> Result<(), ()> {
    if let Some(rest) = line.strip_prefix("echo ")
        && let Some((value, path)) = rest.split_once(" > ")
    {
        let value = value.trim();
        let path = path.trim();
        if let Err(e) = fs::write(path, format!("{value}\n")) {
            tracing::error!(value, path, err = %e, "ktstr-init: echo redirect failed");
            return Err(());
        }
        return Ok(());
    }
    tracing::error!(line, "ktstr-init: unsupported command");
    Err(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::MutexExt;

    #[test]
    fn mkdir_p_creates_nested() {
        let _tempdir_keep_alive = tempfile::Builder::new()
            .prefix("ktstr-rust-init-test-mkdir-")
            .tempdir()
            .unwrap();
        let base = _tempdir_keep_alive.path();
        let nested = base.join("a/b/c");
        mkdir_p(nested.to_str().unwrap());
        assert!(nested.exists());
    }

    /// Uses raw `std::env::temp_dir()` (not `tempfile::TempDir`)
    /// because the test's premise is "mkdir_p is a no-op when the
    /// dir already exists" — pointing at an existing dir is the
    /// whole point. `tempfile::TempDir` would also work, but raw
    /// `temp_dir()` is closer to the production input: `mkdir_p`
    /// is called against arbitrary already-existing system paths.
    #[test]
    fn mkdir_p_existing_is_noop() {
        let tmp = std::env::temp_dir();
        mkdir_p(tmp.to_str().unwrap());
    }

    #[test]
    fn exec_shell_line_echo_redirect() {
        let _tempfile_keep_alive = tempfile::Builder::new()
            .prefix("ktstr-rust-init-echo-test-")
            .tempfile()
            .unwrap();
        let path = _tempfile_keep_alive.path().to_str().unwrap();
        assert!(exec_shell_line(&format!("echo 42 > {path}")).is_ok());
        let content = fs::read_to_string(_tempfile_keep_alive.path()).unwrap();
        assert_eq!(content, "42\n");
    }

    #[test]
    fn exec_shell_line_unsupported_input_returns_err() {
        // Comments are filtered upstream in exec_shell_script;
        // a bare "# comment" reaching exec_shell_line is an
        // unsupported command. Pinning the Err signal so the
        // partial-apply counter in exec_shell_script catches
        // typo'd lines instead of silently skipping them.
        assert!(exec_shell_line("# this is a comment").is_err());
    }

    /// `exec_shell_script` emits an error-level summary instead of
    /// silently partial-applying. A script with mixed-success lines
    /// must surface the failure count to the operator — the
    /// prior implementation only logged per-line errors with no
    /// roll-up, so an operator scanning init-log for the
    /// sched_enable result couldn't easily count failures.
    #[test]
    fn exec_shell_script_counts_per_line_failures() {
        // Build a script with one valid echo + one unsupported
        // command. The valid line writes a sentinel value to a
        // tempfile so the test asserts the partial-apply did
        // produce the expected side effect — proving the function
        // didn't short-circuit on first failure.
        let _payload_keep_alive = tempfile::Builder::new()
            .prefix("ktstr-tax-payload-")
            .tempfile()
            .unwrap();
        let payload_path = _payload_keep_alive.path().to_str().unwrap();
        let mut script = tempfile::Builder::new()
            .prefix("ktstr-tax-script-")
            .tempfile()
            .unwrap();
        use std::io::Write;
        writeln!(script, "echo 7 > {payload_path}").unwrap();
        writeln!(script, "not_a_supported_command").unwrap();
        script.flush().unwrap();
        exec_shell_script(script.path().to_str().unwrap());
        let payload = fs::read_to_string(payload_path).unwrap();
        assert_eq!(payload, "7\n", "valid line must still apply");
    }

    /// File-not-found returns silently (legitimate "no script"
    /// case for the optional sched_enable/sched_disable hooks).
    /// Pins the debug-level skip so a future refactor that flipped
    /// the missing-file path to error-level would surface here.
    #[test]
    fn exec_shell_script_missing_file_returns_silently() {
        exec_shell_script("/tmp/ktstr-tax-nonexistent-script-path");
    }

    #[test]
    fn shell_mode_not_requested_in_test() {
        // /proc/cmdline exists on the host but won't contain KTSTR_MODE=shell.
        assert!(!shell_mode_requested());
    }

    #[test]
    fn disk_template_mode_not_requested_in_test() {
        // /proc/cmdline on the host won't contain KTSTR_MODE=disk_template.
        assert!(!disk_template_mode_requested());
    }

    #[test]
    fn disk_template_dispatch_precedes_shell_when_both_present() {
        // The dispatch order in `ktstr_guest_init` is:
        //   1. disk_template_mode_requested → run mkfs + reboot, never returns
        //   2. shell_mode_requested → drop into busybox shell
        //   3. test dispatch
        //
        // If both KTSTR_MODE entries appear in /proc/cmdline (e.g.
        // operator typo, host-side cmdline-construction bug), the
        // disk_template path MUST win — running shell mode against
        // a disk that the operator intended to format would skip
        // the formatting step silently. Pin the token-parser
        // semantics so a future refactor that changes the matching
        // logic (regex, prefix-only, or per-token last-wins) does
        // not silently invert the precedence.
        let cmdline = "ro KTSTR_MODE=disk_template KTSTR_MODE=shell console=ttyS0";
        // Both checks see their token in the cmdline.
        assert!(cmdline_contains_token(cmdline, "KTSTR_MODE=disk_template"));
        assert!(cmdline_contains_token(cmdline, "KTSTR_MODE=shell"));
        // The dispatch order in ktstr_guest_init runs the
        // disk_template check FIRST, so the disk_template path is
        // taken and the shell branch is never reached. This test
        // pins the token-parser invariant; the dispatch-order
        // invariant lives in the code at ktstr_guest_init's
        // disk-template-mode block.
        //
        // Reverse-token order produces the same result — the
        // checks are commutative and dispatch-order is the only
        // disambiguator.
        let cmdline_reversed = "ro KTSTR_MODE=shell KTSTR_MODE=disk_template console=ttyS0";
        assert!(cmdline_contains_token(
            cmdline_reversed,
            "KTSTR_MODE=disk_template"
        ));
        assert!(cmdline_contains_token(cmdline_reversed, "KTSTR_MODE=shell"));
    }

    #[test]
    fn cmdline_contains_token_exact_match_not_prefix() {
        // Matching is whole-token, not prefix. A future kernel
        // cmdline that introduces e.g. `KTSTR_MODE=shell_extended`
        // must not accidentally trip the shell-mode dispatch.
        assert!(cmdline_contains_token(
            "KTSTR_MODE=shell",
            "KTSTR_MODE=shell"
        ));
        assert!(!cmdline_contains_token(
            "KTSTR_MODE=shell_extended",
            "KTSTR_MODE=shell"
        ));
        assert!(!cmdline_contains_token(
            "prefix_KTSTR_MODE=shell",
            "KTSTR_MODE=shell"
        ));
        assert!(!cmdline_contains_token("", "KTSTR_MODE=shell"));
    }

    #[test]
    fn count_online_cpus_returns_some() {
        // On any Linux host, /sys/devices/system/cpu/online exists.
        let count = count_online_cpus();
        assert!(count.is_some());
        assert!(count.unwrap() >= 1);
    }

    #[test]
    fn parse_online_cpus_single_index() {
        assert_eq!(parse_online_cpus("0"), Some(1));
        assert_eq!(parse_online_cpus("7"), Some(1));
    }

    #[test]
    fn parse_online_cpus_simple_range() {
        assert_eq!(parse_online_cpus("0-3"), Some(4));
        assert_eq!(parse_online_cpus("4-7"), Some(4));
    }

    #[test]
    fn parse_online_cpus_mixed_ranges_and_singles() {
        assert_eq!(parse_online_cpus("0,2,4"), Some(3));
        assert_eq!(parse_online_cpus("0-1,4-7"), Some(6));
        assert_eq!(parse_online_cpus("0-2,4,6-7"), Some(6));
    }

    #[test]
    fn parse_online_cpus_strips_trailing_newline() {
        // /sys/devices/system/cpu/online emits a trailing '\n'.
        assert_eq!(parse_online_cpus("0-3\n"), Some(4));
    }

    #[test]
    fn parse_online_cpus_single_cpu_zero() {
        assert_eq!(parse_online_cpus("0-0"), Some(1));
    }

    #[test]
    fn parse_online_cpus_empty_content_is_none() {
        assert_eq!(parse_online_cpus(""), None);
        assert_eq!(parse_online_cpus("   "), None);
        assert_eq!(parse_online_cpus("\n"), None);
    }

    #[test]
    fn parse_online_cpus_non_numeric_is_none() {
        assert_eq!(parse_online_cpus("abc"), None);
        assert_eq!(parse_online_cpus("0-abc"), None);
        assert_eq!(parse_online_cpus("a-3"), None);
        assert_eq!(parse_online_cpus("0,abc,3"), None);
        // Empty tokens from malformed list shapes — the kernel never
        // produces these but the parser must reject loudly rather
        // than silently skip.
        assert_eq!(parse_online_cpus("0,"), None); // trailing comma
        assert_eq!(parse_online_cpus(",0"), None); // leading comma
        assert_eq!(parse_online_cpus("-3"), None); // leading dash → empty range start
    }

    #[test]
    fn parse_online_cpus_inverted_range_is_none() {
        // Defensive: an inverted range "10-3" would previously
        // panic in debug (overflow) or wrap in release. checked_sub
        // returns None instead.
        assert_eq!(parse_online_cpus("10-3"), None);
    }

    #[test]
    fn parse_online_cpus_extreme_range_does_not_overflow() {
        // u32::MAX - 0 + 1 overflows u32; checked_add returns None.
        assert_eq!(parse_online_cpus(&format!("0-{}", u32::MAX)), None);
    }

    #[test]
    fn parse_online_cpus_large_topology() {
        // 256 vCPUs as a single range.
        assert_eq!(parse_online_cpus("0-255"), Some(256));
    }

    /// Zero budget: loop exits within one sleep step and emits the
    /// WARN with the expected diagnostic fields. The `traced_test`
    /// attribute installs a capturing subscriber so `logs_contain`
    /// can verify the message body and each structured field
    /// rendered into the log line.
    ///
    /// Pins both the time bound AND the WARN content — a regression
    /// that silently dropped the structured fields, or moved the
    /// emit to a lower log level, would trip the logs_contain
    /// assertions.
    #[test]
    #[tracing_test::traced_test]
    fn send_sys_rdy_retry_exits_when_budget_exhausted() {
        let budget = std::time::Duration::from_millis(0);
        let addrs = crate::vmm::wire::KernAddrs::new(0, 0, None);
        // Use a path that won't exist on the host so the loop
        // takes the port_exists=false branch — no real device
        // interaction in unit tests.
        let port_path =
            std::path::Path::new("/tmp/ktstr-test-nonexistent-port-please-do-not-create");
        let t0 = std::time::Instant::now();
        send_sys_rdy_with_retry(budget, 1, &addrs, port_path);
        let elapsed = t0.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "send_sys_rdy_with_retry with zero budget took {elapsed:?}; \
             must exit within one sleep step (with slack for CI load)",
        );
        // Verify the WARN content. tracing-test's logs_contain does
        // a substring match against captured log lines.
        assert!(
            logs_contain("send_sys_rdy failed within boot budget"),
            "WARN message must be emitted on budget exhaustion",
        );
        for field in [
            "budget_ms=0",
            "vcpus=1",
            "elapsed_ms=",
            "port_exists=false",
            "kern_addrs_sent=false",
        ] {
            assert!(
                logs_contain(field),
                "WARN must include structured field `{field}`",
            );
        }
        assert!(
            logs_contain("send_sys_rdy-timeout"),
            "WARN must include the docs anchor pointer",
        );
    }

    /// Wall-time floor invariant: the loop must wait at least
    /// `budget` wall-clock time before emitting the WARN (with the
    /// port absent). Parameterized over several (budget, vcpus)
    /// combinations to pin the invariant across the production
    /// budget range — replaces the deleted count-based formula
    /// coupling test.
    #[test]
    fn send_sys_rdy_retry_respects_budget_across_sizes() {
        let port_path =
            std::path::Path::new("/tmp/ktstr-test-nonexistent-port-please-do-not-create");
        let addrs = crate::vmm::wire::KernAddrs::new(0, 0, None);
        for &(budget_ms, vcpus) in &[(50u64, 1u32), (150, 2), (250, 8), (500, 32)] {
            let budget = std::time::Duration::from_millis(budget_ms);
            let t0 = std::time::Instant::now();
            send_sys_rdy_with_retry(budget, vcpus, &addrs, port_path);
            let elapsed = t0.elapsed();
            assert!(
                elapsed >= budget,
                "(budget={budget_ms}ms, vcpus={vcpus}): elapsed {elapsed:?} \
                 < budget; the loop must wait at least the budget before \
                 the WARN fires",
            );
            // Generous upper bound — CI runners under load can
            // stretch std::thread::sleep significantly. Keep wide
            // enough to never flake while still catching a runaway
            // (e.g. count-based loop ignoring the budget).
            let cap = budget + std::time::Duration::from_secs(2);
            assert!(
                elapsed < cap,
                "(budget={budget_ms}ms, vcpus={vcpus}): elapsed {elapsed:?} \
                 exceeded {cap:?}; the loop should not overshoot by more \
                 than ~2s of slack",
            );
        }
    }

    /// Port-exists branch coverage: when the device-node path
    /// resolves, the loop takes the `if port_path.exists()` arm
    /// and calls `send_kern_addrs` / `send_sys_rdy`. In host
    /// context those calls no-op via `assert_guest_context` (and
    /// `write_to_bulk_port`'s hardcoded `/dev/vport0p1` open will
    /// fail too), so kern_addrs_sent stays false and the loop
    /// exhausts the budget. The WARN must report
    /// `port_exists=true, kern_addrs_sent=false` — the
    /// diagnostic combination the troubleshooting doc explains as
    /// "the port device exists but writes failed".
    #[test]
    #[tracing_test::traced_test]
    fn send_sys_rdy_retry_reports_port_exists_when_path_resolves() {
        let tmpfile =
            tempfile::NamedTempFile::new().expect("create tempfile to stand in for /dev/vport0p1");
        let budget = std::time::Duration::from_millis(150);
        let addrs = crate::vmm::wire::KernAddrs::new(0, 0, None);
        send_sys_rdy_with_retry(budget, 4, &addrs, tmpfile.path());
        assert!(
            logs_contain("port_exists=true"),
            "WARN must report port_exists=true when the path resolves",
        );
        assert!(
            logs_contain("kern_addrs_sent=false"),
            "WARN must report kern_addrs_sent=false when host-context \
             writes no-op via assert_guest_context",
        );
        assert!(logs_contain("vcpus=4"), "WARN must include the vcpus value",);
    }

    #[test]
    fn parse_topo_from_cmdline_not_present_on_host() {
        // Host /proc/cmdline won't contain KTSTR_TOPO.
        assert!(parse_topo_from_cmdline().is_none());
    }

    /// A child that exits immediately must be observed as `Died`
    /// well before the poll timeout. This is the regression gate
    /// for the old unconditional `sleep(1s)` — we don't want to
    /// wait a full second to notice an instant crash.
    #[test]
    fn poll_startup_detects_early_death_quickly() {
        let mut child = std::process::Command::new("/bin/true")
            .spawn()
            .expect("spawn /bin/true");
        let start = std::time::Instant::now();
        let status = poll_startup(
            &mut child,
            std::time::Duration::from_millis(10),
            std::time::Duration::from_secs(1),
        );
        let elapsed = start.elapsed();
        assert!(
            matches!(status, StartupStatus::Died),
            "expected Died, got {status:?}"
        );
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "early death must be detected fast, took {elapsed:?}"
        );
    }

    /// A child that stays alive past the poll window must be
    /// observed as `Alive` within ~timeout — the caller accepts
    /// this as "scheduler ready" without any longer wait.
    #[test]
    fn poll_startup_reports_alive_after_timeout() {
        let mut child = std::process::Command::new("/bin/sleep")
            .arg("5")
            .spawn()
            .expect("spawn /bin/sleep");
        let start = std::time::Instant::now();
        let status = poll_startup(
            &mut child,
            std::time::Duration::from_millis(20),
            std::time::Duration::from_millis(100),
        );
        let elapsed = start.elapsed();
        let _ = child.kill();
        let _ = child.wait();
        assert!(
            matches!(status, StartupStatus::Alive),
            "expected Alive, got {status:?}"
        );
        assert!(
            elapsed >= std::time::Duration::from_millis(100),
            "Alive must wait the full timeout, took only {elapsed:?}"
        );
        // Poll is allowed one extra interval of slack.
        assert!(
            elapsed < std::time::Duration::from_millis(300),
            "Alive should not overshoot timeout significantly, took {elapsed:?}"
        );
    }

    // -- kill_scheduler_process tests --
    //
    // The kill helper is the building block for Op::DetachScheduler /
    // Op::RestartScheduler / Op::ReplaceScheduler dispatch (follow-up
    // work). Tests pin the three outcome variants
    // (AlreadyExited / ExitedAfterSigterm / EscalatedToSigkill) plus
    // the InvalidPid error path. The escalation test deliberately
    // installs SIGTERM-ignoring trap to force the SIGKILL branch —
    // matches the scx-scheduler-without-handler scenario the
    // EscalatedToSigkill variant is named for.

    /// `pid` <= 0 must surface InvalidPid immediately without
    /// touching the kernel. POSIX kill(2) reserves 0 (caller's pgrp)
    /// and negative values (signal pgrp), neither of which the
    /// scheduler-lifecycle call site ever wants. The check is a
    /// programming-error guard for callers that fail to validate
    /// SCHED_PID readouts.
    #[test]
    fn kill_scheduler_process_invalid_pid_returns_err() {
        assert_eq!(
            kill_scheduler_process(0, std::time::Duration::from_millis(50)),
            Err(KillSchedulerError::InvalidPid),
        );
        assert_eq!(
            kill_scheduler_process(-1, std::time::Duration::from_millis(50)),
            Err(KillSchedulerError::InvalidPid),
        );
    }

    /// A pid that was never alive (or was reaped before the call)
    /// surfaces as AlreadyExited — the idempotent-detach case that
    /// lifecycle Op semantics rely on (detaching nothing is success,
    /// not error).
    #[test]
    fn kill_scheduler_process_already_exited_pid_yields_already_exited() {
        // Spawn /bin/true and let it exit + reap before kill_scheduler_process
        // is called. /bin/true exits ~immediately.
        let mut child = std::process::Command::new("/bin/true")
            .spawn()
            .expect("spawn /bin/true");
        let pid = child.id() as libc::pid_t;
        let _ = child.wait();
        // After wait, /proc/{pid} has been released. Poll briefly
        // to ensure procfs cleanup has propagated.
        let mut waits = 0u32;
        while proc_pid_alive(pid as u32) && waits < 50 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            waits += 1;
        }
        assert!(
            !proc_pid_alive(pid as u32),
            "procfs should have released the pid after wait"
        );
        assert_eq!(
            kill_scheduler_process(pid, std::time::Duration::from_millis(50)),
            Ok(KillSchedulerOutcome::AlreadyExited),
        );
    }

    /// A responsive child (one that catches SIGTERM and exits)
    /// produces ExitedAfterSigterm. /bin/sleep installs the default
    /// SIGTERM handler (terminate-on-signal — kernel-side action,
    /// no userspace handler, but the kernel exit completes well
    /// inside the grace window).
    ///
    /// Installs SIGCHLD=SIG_IGN for the test duration — matches the
    /// production guest-init disposition, where the kernel
    /// auto-reaps children so `/proc/{pid}` disappears at exit
    /// without an explicit `waitpid`. Without this the test would
    /// race with the standard SIGCHLD=SIG_DFL test environment that
    /// keeps the exited child as a zombie (procfs entry persists)
    /// until the explicit Child::wait, breaking the poll_pid_gone
    /// observation that kill_scheduler_process relies on.
    #[test]
    fn kill_scheduler_process_responsive_child_yields_exited_after_sigterm() {
        let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
        let _restore = SigchldGuard::install(libc::SIG_IGN);

        let mut child = std::process::Command::new("/bin/sleep")
            .arg("60")
            .spawn()
            .expect("spawn /bin/sleep");
        let pid = child.id() as libc::pid_t;
        let outcome = kill_scheduler_process(pid, std::time::Duration::from_millis(500));
        // Best-effort reap. Under SIG_IGN the kernel auto-reaps so
        // Child::wait returns ECHILD; the call is harmless either
        // way. SigchldGuard's Drop restores the previous disposition
        // before the test exits so subsequent tests aren't poisoned.
        let _ = child.wait();
        assert_eq!(outcome, Ok(KillSchedulerOutcome::ExitedAfterSigterm));
    }

    /// A child that ignores SIGTERM must produce
    /// EscalatedToSigkill. /bin/sh -c 'trap "" TERM; sleep 30'
    /// installs an empty SIGTERM trap, so SIGTERM is no-op'd and
    /// the SIGKILL fallback is the only way to terminate. Pins the
    /// escalation branch against a regression that drops the
    /// SIGKILL step or treats SIGTERM-grace-exhausted as success.
    ///
    /// SIGCHLD=SIG_IGN for the same reason as the
    /// `_responsive_child_` sibling test — see that test's docs.
    ///
    /// Synchronizes via filesystem marker rather than a timing-based
    /// settle delay so the test is immune to CI scheduling jitter.
    /// The shell does `trap '' TERM; touch <marker>; sleep 30`, the
    /// test polls for marker existence with a generous 5s deadline,
    /// THEN sends SIGTERM. This eliminates the race where the kill
    /// can land before the shell has installed its trap — the marker
    /// existence is a kernel-observable HAPPENS-AFTER signal proving
    /// the trap installation already returned. Marker filename uses
    /// a fixed path because SIGCHLD_TEST_LOCK serializes the tests
    /// that write SIGCHLD disposition, so concurrent writers cannot
    /// collide.
    #[test]
    fn kill_scheduler_process_ignoring_sigterm_child_escalates_to_sigkill() {
        let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
        let _restore = SigchldGuard::install(libc::SIG_IGN);

        let marker = "/tmp/ktstr_kill_test_trap_ready";
        // Clear any stale marker from a prior aborted run.
        let _ = std::fs::remove_file(marker);

        let mut child = std::process::Command::new("/bin/sh")
            .arg("-c")
            // `exec sleep 30` forces sleep to replace sh in-place
            // so SIGKILL on the sh pid kills the sleep too. Without
            // `exec`, sh runs `touch` first (which forces sh to stay
            // around as a process), then forks for `sleep` — and
            // SIGKILL on the sh pid leaves the orphaned sleep
            // re-parented to init, surfaced by nextest's leak
            // detector as a "leaky" test.
            .arg(format!("trap '' TERM; touch {marker}; exec sleep 30"))
            .spawn()
            .expect("spawn /bin/sh");
        let pid = child.id() as libc::pid_t;

        // Wait for the marker — proves the trap is installed.
        let marker_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !std::path::Path::new(marker).exists() {
            if std::time::Instant::now() >= marker_deadline {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_file(marker);
                panic!(
                    "shell did not create trap-ready marker within 5s — \
                     /bin/sh failed to start or filesystem is too slow"
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Tight SIGTERM grace (200ms) so the test doesn't burn a
        // full second on the polite-shutdown timeout. The trap is
        // confirmed installed via the marker so the shell will
        // ignore SIGTERM and force the SIGKILL escalation.
        let outcome = kill_scheduler_process(pid, std::time::Duration::from_millis(200));
        let _ = child.wait();
        let _ = std::fs::remove_file(marker);
        assert_eq!(outcome, Ok(KillSchedulerOutcome::EscalatedToSigkill));
    }

    /// kill_scheduler_process MUST NOT mutate SCHED_PID — the design
    /// at L320-327 of rust_init.rs explicitly keeps the helper
    /// generic-pid (no implicit singleton-pid assumption) and defers
    /// SCHED_PID ownership to the dispatcher (the future
    /// Op::DetachScheduler arm). This test pins that contract against
    /// a future "improvement" that adds an implicit SCHED_PID reset
    /// for symmetry with the dispatcher path — silent decoupling
    /// breakage that would couple kill-pid choice to the singleton
    /// scheduler pid in unintended ways.
    ///
    /// Seeds SCHED_PID with a sentinel distinct from any spawnable
    /// pid (99_999_999 > Linux's default kernel.pid_max), exercises
    /// kill_scheduler_process against an unrelated /bin/sleep pid,
    /// and asserts the sentinel survives. Restores SCHED_PID to 0
    /// at end so subsequent tests see a clean baseline.
    #[test]
    fn kill_scheduler_process_does_not_mutate_sched_pid() {
        let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
        let _restore = SigchldGuard::install(libc::SIG_IGN);

        let original = SCHED_PID.load(Ordering::Acquire);
        let sentinel: i32 = 99_999_999;
        SCHED_PID.store(sentinel, Ordering::Release);

        let mut child = std::process::Command::new("/bin/sleep")
            .arg("60")
            .spawn()
            .expect("spawn /bin/sleep");
        let pid = child.id() as libc::pid_t;
        let _ = kill_scheduler_process(pid, std::time::Duration::from_millis(500));
        let _ = child.wait();

        let observed = SCHED_PID.load(Ordering::Acquire);
        // Restore BEFORE the assert so a failure does not leak
        // sentinel state to subsequent tests.
        SCHED_PID.store(original, Ordering::Release);

        assert_eq!(
            observed, sentinel,
            "kill_scheduler_process(pid={pid}) mutated SCHED_PID \
             (sentinel={sentinel}, observed={observed}); the helper \
             must NOT touch SCHED_PID — that side channel is the \
             dispatcher's responsibility per the helper's design \
             decoupling. A future commit that adds an implicit reset \
             couples the helper to singleton-pid semantics that the \
             design explicitly avoids."
        );
    }

    /// SIGCHLD signal disposition is process-wide, so the
    /// `with_sigchld_default_*` and `poll_startup_under_sigign_*`
    /// regression tests must serialize. Without this lock, two
    /// concurrent `libc::signal(SIGCHLD, ...)` calls from different
    /// test threads could leave SIGCHLD in an unexpected state when
    /// either test inspects or restores it. Acquired via
    /// [`crate::sync::MutexExt::lock_unpoisoned`] so a panic in one
    /// signal-aware test does not poison every other one.
    static SIGCHLD_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard that snapshots the current SIGCHLD disposition on
    /// construction and restores it on drop. Tests that flip
    /// `SIGCHLD` to `SIG_IGN` to reproduce the PID-1 environment
    /// must not bleed that disposition into the rest of the test
    /// run — the cargo nextest binary runs every test in a single
    /// process under threads, so a leaked `SIG_IGN` would make
    /// every subsequent `Child::wait` (in unrelated tests) return
    /// ECHILD. `signal(2)` returns the previous handler; we restore
    /// it verbatim via a second `signal` call.
    struct SigchldGuard {
        prev: libc::sighandler_t,
    }

    impl SigchldGuard {
        fn install(handler: libc::sighandler_t) -> Self {
            // SAFETY: `libc::signal` accepts any process-wide signal
            // disposition; the returned value is the previous
            // handler, captured here for restoration in `Drop`.
            let prev = unsafe { libc::signal(libc::SIGCHLD, handler) };
            Self { prev }
        }
    }

    impl Drop for SigchldGuard {
        fn drop(&mut self) {
            // SAFETY: `self.prev` was returned by an earlier
            // `libc::signal` call on the same signal number;
            // re-installing it is the documented restore pattern.
            unsafe {
                libc::signal(libc::SIGCHLD, self.prev);
            }
        }
    }

    /// Regression: with SIGCHLD set to `SIG_IGN`, a bare
    /// `Command::status()` returns `Err(ECHILD)` because the kernel
    /// auto-reaps the child before `waitpid` can observe it.
    /// `with_sigchld_default` must restore `SIG_DFL` for the
    /// closure's lifetime so `waitpid` reaps and reports a real
    /// status. After the closure returns, `SIG_IGN` must be
    /// restored.
    #[test]
    fn with_sigchld_default_captures_real_exit_status() {
        let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
        let _restore = SigchldGuard::install(libc::SIG_IGN);

        // Sanity: under SIG_IGN, plain Command::status() returns
        // Err(ECHILD) — proves the ambient state matches PID 1.
        let bare = Command::new("/bin/true").status();
        assert!(
            bare.is_err(),
            "under SIG_IGN, Command::status must fail with ECHILD; got {bare:?}",
        );

        // Helper restores SIG_DFL for the closure body, so the same
        // Command::status() succeeds and reports exit code 0.
        let wrapped = with_sigchld_default(|| Command::new("/bin/true").status());
        let status = wrapped.expect("with_sigchld_default must capture status");
        assert_eq!(
            status.code(),
            Some(0),
            "/bin/true must exit 0 under helper; got {status:?}",
        );

        // After the closure returns, SIG_IGN must be back in place
        // so subsequent guest children continue to be auto-reaped.
        // SAFETY: signal(SIG_IGN) reads the previous disposition
        // and re-installs SIG_IGN; we compare the previous value to
        // SIG_IGN to assert nothing changed it underneath us.
        let after = unsafe { libc::signal(libc::SIGCHLD, libc::SIG_IGN) };
        assert_eq!(
            after,
            libc::SIG_IGN,
            "with_sigchld_default must restore SIG_IGN after closure returns",
        );
    }

    /// Regression (non-zero exit propagation): the helper
    /// must surface the child's real non-zero exit code, not the
    /// previous-implementation `Err(_) => 1` mapping that swallowed
    /// every status under SIG_IGN.
    #[test]
    fn with_sigchld_default_captures_nonzero_exit_status() {
        let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
        let _restore = SigchldGuard::install(libc::SIG_IGN);

        let wrapped = with_sigchld_default(|| Command::new("/bin/false").status());
        let status = wrapped.expect("with_sigchld_default must capture status");
        // /bin/false on every supported Unix exits with code 1.
        assert_eq!(
            status.code(),
            Some(1),
            "/bin/false must surface non-zero code under helper; got {status:?}",
        );
    }

    /// Regression: under `SIGCHLD = SIG_IGN`, a child that
    /// exits before the poll window closes MUST be observed as
    /// `Died`. The previous implementation called `Child::try_wait`
    /// which internally calls `waitpid(pid, ..., WNOHANG)`; under
    /// SIG_IGN that returns `ECHILD` and the old code mapped it to
    /// `WaitError`, which the caller in `start_scheduler` then
    /// treated as alive — leaving a crashed scheduler undetected.
    /// The fix uses `proc_pid_alive` and pidfd POLLIN, both of
    /// which are signal-disposition independent.
    #[test]
    fn poll_startup_detects_death_under_sigchld_ignore() {
        let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
        let _restore = SigchldGuard::install(libc::SIG_IGN);

        let mut child = std::process::Command::new("/bin/true")
            .spawn()
            .expect("spawn /bin/true");
        let status = poll_startup(
            &mut child,
            std::time::Duration::from_millis(10),
            std::time::Duration::from_secs(1),
        );
        assert!(
            matches!(status, StartupStatus::Died),
            "under SIG_IGN, an exited child must be observed as Died (was {status:?})",
        );
    }

    /// Regression (Alive arm under SIG_IGN): a child that
    /// is still running when the timeout elapses must be observed
    /// as `Alive` even when SIGCHLD is `SIG_IGN`. This guards the
    /// post-timeout `proc_pid_alive` re-check that replaced the
    /// old `try_wait` call (which would have returned ECHILD-as-
    /// `WaitError` and the caller would have reported alive
    /// anyway, but the new path must not regress that branch).
    #[test]
    fn poll_startup_reports_alive_under_sigchld_ignore() {
        let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();
        let _restore = SigchldGuard::install(libc::SIG_IGN);

        let mut child = std::process::Command::new("/bin/sleep")
            .arg("5")
            .spawn()
            .expect("spawn /bin/sleep");
        let status = poll_startup(
            &mut child,
            std::time::Duration::from_millis(20),
            std::time::Duration::from_millis(100),
        );
        // Reap the still-running child via SIGKILL + waitpid. We
        // need to drop SIG_IGN before waiting or `child.wait()`
        // would itself return ECHILD; the SigchldGuard's Drop
        // restores at the end of the test, so flip to SIG_DFL for
        // the cleanup. SAFETY: signal disposition is process-wide
        // but this test holds SIGCHLD_TEST_LOCK, so no other
        // signal-aware test runs concurrently.
        let _ = child.kill();
        unsafe {
            libc::signal(libc::SIGCHLD, libc::SIG_DFL);
        }
        let _ = child.wait();
        assert!(
            matches!(status, StartupStatus::Alive),
            "under SIG_IGN, a running child must be observed as Alive (was {status:?})",
        );
    }

    /// Regression: the [`SCHED_PID`] side channel must
    /// publish the writer's value and `sched_pid()` must return
    /// `Some(pid)` when set, `None` when the sentinel `0` is in
    /// place. Since `SCHED_PID` is a process-wide static, the test
    /// snapshots the current value, exercises both store paths,
    /// and restores the snapshot — so concurrent tests (and the
    /// real producer in `start_scheduler` if some other test ever
    /// drives it) do not see ambient corruption.
    #[test]
    fn sched_pid_side_channel_roundtrips() {
        // Snapshot and restore with `Acquire`/`Release` to mirror
        // the production load/store ordering. The test must hold
        // exclusive access to the static for its lifetime; serial
        // execution under the same process means concurrent
        // `sched_pid()` readers in other tests would race, so this
        // test is annotated to acquire `SIGCHLD_TEST_LOCK` even
        // though it has no signal interaction — the existing lock
        // is already the chokepoint for "tests that touch
        // process-wide state" and serializing through it is
        // cheaper than introducing a second mutex for one test.
        let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();

        let snapshot = SCHED_PID.load(Ordering::Acquire);

        // Sentinel 0 must read as None.
        SCHED_PID.store(0, Ordering::Release);
        assert_eq!(sched_pid(), None, "0 must read as None (sentinel)");

        // Non-zero writer publishes, reader observes.
        SCHED_PID.store(12345, Ordering::Release);
        assert_eq!(
            sched_pid(),
            Some(12345),
            "writer must publish via the atomic side channel",
        );

        // Restore so the test does not leak state into peers.
        SCHED_PID.store(snapshot, Ordering::Release);
    }

    /// Regression (no env-var write): the new fix must NOT
    /// touch `std::env::set_var("SCHED_PID", ...)` because
    /// mutating glibc's `__environ` while the probe thread is live
    /// is documented UB. Asserting that the env var is absent
    /// after a fresh atomic store is a proxy for "no rogue
    /// env-mutation snuck back in." If a future refactor brings
    /// `set_var` back, this test fails immediately.
    #[test]
    fn sched_pid_does_not_publish_via_env_var() {
        let _guard = SIGCHLD_TEST_LOCK.lock_unpoisoned();

        // Clear any ambient env var — some test harnesses inherit
        // `SCHED_PID` from a parent shell. SAFETY: holding the
        // mutex guarantees no concurrent env reader/writer in this
        // test binary.
        unsafe { std::env::remove_var("SCHED_PID") };

        let snapshot = SCHED_PID.load(Ordering::Acquire);
        SCHED_PID.store(99999, Ordering::Release);
        assert_eq!(sched_pid(), Some(99999));
        assert!(
            std::env::var("SCHED_PID").is_err(),
            "atomic side channel must not publish via env var",
        );
        SCHED_PID.store(snapshot, Ordering::Release);
    }

    /// T2 regression: the trace_pipe→COM1 reader's dump-marker scanner
    /// fires the started + complete latches, and matches the end-marker
    /// even when it is split across a read boundary (the rolling-tail
    /// seam — `SCAN_TAIL_KEEP` must exceed the longest marker so the split
    /// prefix survives into the next chunk).
    #[test]
    fn scan_dump_markers_fires_latches_across_chunk_seam() {
        let mut tail: Vec<u8> = Vec::new();
        assert!(!scx_dump_started_latch().is_set());
        assert!(!scx_dump_complete_latch().is_set());

        // First dump line fires the started latch.
        scan_dump_markers(
            b"  init-1 [000] d.h1. 1.0: sched_ext_dump: init[1] triggered exit kind 1024:\n",
            &mut tail,
        );
        assert!(
            scx_dump_started_latch().is_set(),
            "started latch fires on the first `sched_ext_dump:` line"
        );
        assert!(
            !scx_dump_complete_latch().is_set(),
            "complete latch unset before the end-marker"
        );

        // End-marker split across two reads — the rolling tail must match.
        scan_dump_markers(b"  ...event counters... SCX_EV_SUB_BYPASS", &mut tail);
        assert!(
            !scx_dump_complete_latch().is_set(),
            "a partial end-marker must not fire the complete latch"
        );
        scan_dump_markers(b"_DISPATCH: 0\n", &mut tail);
        assert!(
            scx_dump_complete_latch().is_set(),
            "the seam-split end-marker matches via the rolling tail"
        );
    }

    /// T3 regression: `reap_child_bounded` reaps a child that exits within
    /// the bound, and gives up (false) on a still-live child once the
    /// bound elapses — so a process that can't take its pending SIGKILL
    /// promptly (the defensive case `SCHED_REAP_TIMEOUT` caps) cannot stall
    /// teardown.
    #[test]
    fn reap_child_bounded_reaps_quick_and_times_out_on_live() {
        let mut quick = std::process::Command::new("sleep")
            .arg("0.1")
            .spawn()
            .expect("spawn sleep 0.1");
        assert!(
            reap_child_bounded(&mut quick, std::time::Duration::from_secs(10)),
            "a child that exits within the bound is reaped"
        );

        let mut live = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep 30");
        assert!(
            !reap_child_bounded(&mut live, std::time::Duration::from_millis(200)),
            "a still-live child is not reaped within the bound"
        );
        live.kill().unwrap();
        live.wait().unwrap();
    }
}
