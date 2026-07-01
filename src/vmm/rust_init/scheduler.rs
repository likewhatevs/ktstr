//! sched_ext scheduler spawn, attach polling, and bounded child reap.
//!
//! Split from rust_init.rs; the shared consts/statics/imports live in the
//! parent module (`super`), reached via the glob below.
use super::*;
use crate::scenario::ops::{ScxState, scx_state};

/// Outcome of [`poll_startup`].
#[derive(Debug)]
pub(crate) enum StartupStatus {
    /// Child exited before the poll window closed.
    Died,
    /// Child was still running when the poll window closed.
    Alive,
}

/// Outcome of [`poll_scx_attached`].
#[derive(Debug, PartialEq, Eq)]
enum ScxAttachStatus {
    /// sched_ext is registered AND fully enabled: `root/ops` is non-empty
    /// AND `/sys/kernel/sched_ext/state` reads `enabled` — the scheduler
    /// finished `scx_enable` (ops.init ran, every task was initialized, the
    /// kernel set `SCX_ENABLED`), not merely registered. See
    /// `scx_attach_ready`.
    Attached,
    /// Poll window closed. At least one read of `root/ops` succeeded
    /// (the kernel supports sched_ext and the kset exists), but the
    /// attach never completed before the timeout: either `root/ops`
    /// stayed empty (scheduler never registered) or it registered but
    /// `/sys/kernel/sched_ext/state` never reached `enabled` — stuck in
    /// `enabling`, or the enable FAILED and the state moved on to
    /// `disabling`/`disabled`. Typically a BPF verifier reject, an
    /// ops-mismatch, a slow init, or an ops.init/enable failure.
    Timeout,
    /// Every read of `root/ops` returned `Err`. Either the kernel
    /// lacks sched_ext support entirely or the sysfs tree has not
    /// been created for the current kernel — distinct from
    /// [`Timeout`](Self::Timeout), where reads succeed but the file
    /// is empty.
    SysfsAbsent,
}

impl ScxAttachStatus {
    /// True when the scheduler is registered AND fully enabled (see
    /// `ScxAttachStatus::Attached`).
    fn is_attached(&self) -> bool {
        matches!(self, ScxAttachStatus::Attached)
    }
}

/// True when sched_ext is both REGISTERED (`root/ops` content non-empty) and
/// fully ENABLED (`/sys/kernel/sched_ext/state` reads "enabled", i.e. `state`
/// is `Some(ScxState::Enabled)`).
///
/// The kernel adds the `root` kobject — making `root/ops` non-empty — EARLY in
/// `scx_root_enable_workfn` (kernel/sched/ext.c), BEFORE it runs `ops.init`,
/// initializes every task, and performs the final `SCX_ENABLING` ->
/// `SCX_ENABLED` transition (`scx_tryset_enable_state`). Gating attach on
/// `root/ops` alone therefore reports "attached" while the scheduler is still
/// ENABLING, letting the workload (and the host monitor's first reads) run
/// against a half-up scheduler — a silent wrong result. Requiring
/// `state == Some(ScxState::Enabled)` closes that race; it also rejects a
/// registered-but-FAILED enable, where the state goes `ENABLING` ->
/// `DISABLING` -> `DISABLED` and the ops-only check would mis-report attached.
///
/// Pure over its inputs so the predicate is unit-testable without a live scx
/// scheduler; `poll_scx_attached` supplies the live `root/ops` contents and the
/// `scx_state` reading.
pub(crate) fn scx_attach_ready(root_ops_contents: &str, state: Option<ScxState>) -> bool {
    !root_ops_contents.trim().is_empty() && state == Some(ScxState::Enabled)
}

/// Poll `/sys/kernel/sched_ext/root/ops` at `interval` cadence for up
/// to `timeout`.
///
/// Returns [`ScxAttachStatus::Attached`] once `root/ops` is non-empty
/// AND `/sys/kernel/sched_ext/state` reads `enabled` — registered AND
/// fully enabled (see `scx_attach_ready`). When the window closes
/// without that, distinguishes [`Timeout`](ScxAttachStatus::Timeout)
/// (`root/ops` reads succeeded but attach never completed — never
/// registered, or registered but never reached `enabled`) from
/// [`SysfsAbsent`](ScxAttachStatus::SysfsAbsent) (every `root/ops` read
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
/// Semantics: a non-empty `root/ops` proves the scheduler completed
/// `scx_alloc_and_add_sched` (scx_sched allocated, `sch->ops = *ops`
/// copied the ops including `name`, kobject registered with the kset),
/// but the kobject add happens BEFORE `ops.init`, per-task init, and the
/// `SCX_ENABLED` transition — so a non-empty `root/ops` does NOT prove
/// the scheduler is live. This poll therefore ALSO requires
/// `/sys/kernel/sched_ext/state` == `enabled` (via `scx_state`), so a
/// returned `Attached` means the BPF enable completed and the workload
/// can run against a fully-up scheduler. (Crash/exit detection AFTER
/// enable is still the caller's job — monitor telemetry or the
/// scheduler's own exit kind.)
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
            // Attached only when REGISTERED (non-empty root/ops) AND fully
            // ENABLED (state == "enabled"): the kernel adds root/ops early in
            // scx_root_enable_workfn, before ops.init / per-task init / the
            // SCX_ENABLED transition, so root/ops alone races a still-ENABLING
            // scheduler. scx_state() re-reads /sys/kernel/sched_ext/state each
            // call (cheap; the cadence below drives the ENABLING->ENABLED
            // re-check on kernels that do not sysfs_notify the transition).
            if scx_attach_ready(&buf, scx_state()) {
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
    // each poll(2) at `interval`. In scx_alloc_and_add_sched
    // `sch->ops = *ops` runs BEFORE `kobject_init_and_add(..., "root")`,
    // so by IN_CREATE wake time `root/ops` reads non-empty. The cadence
    // is ALSO load-bearing for the state==`enabled` half of the
    // predicate: the `enabling`->`enabled` transition (later in
    // scx_root_enable_workfn) is not `sysfs_notify`'d and fires no
    // inotify event on this dir, so the periodic cadence re-check is what
    // observes it. Plus defense-in-depth against (a) future kernel
    // reordering, (b) inotify event loss under pressure, (c) out-of-band
    // kobject creation without ops.name pre-population.
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
            // target kernel 6.16+ where kernfs + inotify are
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
pub(crate) fn poll_startup(
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
/// flush probe output before calling `force_reboot()`, and to the hvc0
/// graceful-shutdown handler so it can drain on the watchdog
/// soft-shutdown path. The success path's drain runs in
/// [`start_sched_exit_monitor`] instead — it sees the scheduler exit
/// notification and waits on `output_done` there. The payload travels
/// the virtio bulk port (the probe thread `println!`s it to stdout,
/// which `redirect_stdio_to_bulk_port` ships over the bulk port), NOT
/// COM2; the host recovers it when it drains that port.
pub(crate) struct ProbeDrain {
    /// Probe-thread stop request. Setting this wakes the probe
    /// thread out of its ring-buffer poll loop; the thread then
    /// emits its payload and sets `output_done`.
    pub(crate) stop: Arc<AtomicBool>,
    /// One-shot signal: set by the probe thread after `println!`ing
    /// `PROBE_OUTPUT_END` (the payload goes to stdout → the bulk-port
    /// forwarder, not COM2). Waited on with a BOUNDED cap on both drain
    /// paths: the early-bail paths cap at `PROBE_DRAIN_GRACE` (30s, the
    /// host's VM-deadline grace — they have no watchdog window of their own);
    /// the hvc0 shutdown drain uses a tighter bound fitting the 3s watchdog
    /// window. On the cap the host recovers the partial payload
    /// (`extract_probe_output`).
    pub(crate) output_done: Arc<crate::sync::Latch>,
}

/// Drain the probe pipeline: signal stop, then wait (bounded by `timeout`) for
/// `output_done`. Called from each early-bail path in [`start_scheduler`] before
/// `force_reboot()` so the probe payload (or the diagnostic-only payload the
/// probe thread emits on a forced stop) reaches the host — emitted over the
/// virtio bulk port by the stdout forwarder, recovered when the host drains it.
/// Returns `true` if `output_done` was observed within `timeout` (or no probe
/// stack was supplied — a no-op), `false` on the cap. Bounded so a hung probe
/// thread cannot wedge the bail until the coarse outer VM wall-clock — the prior
/// unbounded `output_done.wait()` relied on that wall-clock as its only net.
///
/// The early-bail callers pass the auto-repro probe-drain budget
/// ([`crate::test_support::PROBE_DRAIN_GRACE`], 30s) — the same grace the
/// host already adds to the VM deadline for the probe drain, so the guest cap
/// stays within the host's window (over-sized on success: the guest force-reboots
/// the moment the drain completes). The hvc0 soft-shutdown drain
/// (`dump::drain_probe_for_shutdown`) uses its own tighter bound that
/// fits the 3 s watchdog window — a different outer budget, so NOT shared. On the
/// cap the payload may be truncated, but it is loudly absent: the host's
/// `extract_probe_output` recovers the complete events captured before the
/// terminator and persists the partial.
///
/// `drain` is `None` when no probe stack was supplied — every caller is a
/// no-op in that case.
fn drain_probe_pipeline(drain: Option<&ProbeDrain>, timeout: std::time::Duration) -> bool {
    let Some(d) = drain else { return true };
    d.stop.store(true, Ordering::Release);
    let drained = d.output_done.wait_timeout(timeout);
    if !drained {
        tracing::warn!(
            ?timeout,
            "probe drain hit the cap before PROBE_OUTPUT_END; the probe payload may be \
             truncated (the host recovers the partial)"
        );
    }
    drained
}

/// Wait up to `timeout` for `child` to exit (evented via `pidfd_open` +
/// `poll`), then reap it. Does NOT send a signal — callers drive the
/// exit: the crash-grace caller calls this BEFORE `child.kill()` (giving
/// a self-unregistering scheduler a chance to exit on its own within the
/// grace), and the post-grace caller calls it AFTER `child.kill()` (to
/// reap the pending SIGKILL). Returns `true` iff reaped within the
/// window; on timeout the child is left for the VM reboot to reap —
/// teardown must not block unboundedly on a wedged process (see
/// [`SCHED_REAP_TIMEOUT`]).
pub(crate) fn reap_child_bounded(
    child: &mut std::process::Child,
    timeout: std::time::Duration,
) -> bool {
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
    /// helper owns the rollback). The dead child is auto-reaped by
    /// the kernel because PID 1 runs SIGCHLD=`SIG_IGN` (see
    /// [`poll_startup`]'s doc and [`ktstr_guest_init`]);
    /// `poll_startup` only observes the exit via pidfd POLLIN, it
    /// does not reap. No manual `wait`/`try_wait` required by the
    /// caller.
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
            // Process already exited — the dead child is auto-reaped by
            // the kernel under SIGCHLD=SIG_IGN (poll_startup only observes
            // the exit via pidfd POLLIN, it does not reap). SCHED_PID still
            // points at the dead pid; clear so a
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
            // Drain the probe pipeline before reboot so PROBE_OUTPUT_END
            // is emitted over the bulk port ahead of force_reboot.
            // No-op when no probe stack was supplied.
            drain_probe_pipeline(probe_drain.as_ref(), crate::test_support::PROBE_DRAIN_GRACE);
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
            drain_probe_pipeline(probe_drain.as_ref(), crate::test_support::PROBE_DRAIN_GRACE);
            force_reboot();
        }
        Err(SpawnSchedulerError::NotAttached { reason, log_path }) => {
            dump_sched_output(&log_path);
            crate::vmm::guest_comms::send_lifecycle(
                crate::vmm::wire::LifecyclePhase::SchedulerNotAttached,
                reason,
            );
            crate::vmm::guest_comms::send_exit(1);
            drain_probe_pipeline(probe_drain.as_ref(), crate::test_support::PROBE_DRAIN_GRACE);
            force_reboot();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `scx_attach_ready` (the attach predicate) is true ONLY when the
    /// scheduler is both registered (non-empty `root/ops`) AND fully enabled
    /// (`state == Enabled`). A registered-but-still-ENABLING scheduler — the
    /// race window where the kernel added `root/ops` but has not yet set
    /// `SCX_ENABLED` — is NOT reported attached; nor is a failed enable
    /// (`Disabling`/`Disabled`) or a missing state read. Exhaustive over the
    /// state axis and the empty/non-empty `root/ops` axis.
    #[test]
    fn scx_attach_ready_requires_registered_and_enabled() {
        // Registered + enabled -> the one true case.
        assert!(scx_attach_ready("ktstr_ops\n", Some(ScxState::Enabled)));
        // Registered but mid-enable / failed-enable / absent state -> NOT ready
        // (the race + failed-enable cases the old ops-only predicate mis-reported
        // as attached).
        assert!(!scx_attach_ready("ktstr_ops\n", Some(ScxState::Enabling)));
        assert!(!scx_attach_ready("ktstr_ops\n", Some(ScxState::Disabling)));
        assert!(!scx_attach_ready("ktstr_ops\n", Some(ScxState::Disabled)));
        assert!(!scx_attach_ready("ktstr_ops\n", None));
        // Not registered (empty / whitespace-only root/ops) -> NOT ready even
        // when state reads enabled (a stale/foreign enable without our ops).
        assert!(!scx_attach_ready("", Some(ScxState::Enabled)));
        assert!(!scx_attach_ready("   \n", Some(ScxState::Enabled)));
        assert!(!scx_attach_ready("", None));
    }

    fn probe_drain() -> ProbeDrain {
        ProbeDrain {
            stop: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            output_done: std::sync::Arc::new(crate::sync::Latch::new()),
        }
    }

    /// The early-bail drain is BOUNDED: an unset `output_done` (a hung probe
    /// thread) returns `false` at the cap instead of blocking forever (the prior
    /// `output_done.wait()` was unbounded). Asserts `stop` was
    /// signaled and the call returned well inside the cap (a short test cap so
    /// the test is fast).
    #[test]
    fn drain_probe_pipeline_caps_on_unset_output_done() {
        let pd = probe_drain();
        let t0 = std::time::Instant::now();
        let drained = drain_probe_pipeline(Some(&pd), std::time::Duration::from_millis(20));
        assert!(
            !drained,
            "unset output_done -> cap-hit -> false (not a hang)"
        );
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(2),
            "returned at the cap, did not wait unbounded"
        );
        assert!(
            pd.stop.load(std::sync::atomic::Ordering::Acquire),
            "stop signaled to wake the probe thread"
        );
    }

    /// `output_done` already set (PROBE_OUTPUT_END emitted) -> returns `true`
    /// immediately, well inside the cap: the drain-until-marker (early-return)
    /// half is preserved. A 60s cap we never approach pins "returns on the
    /// marker, not the cap".
    #[test]
    fn drain_probe_pipeline_returns_early_on_output_done() {
        let pd = probe_drain();
        pd.output_done.set();
        let t0 = std::time::Instant::now();
        let drained = drain_probe_pipeline(Some(&pd), std::time::Duration::from_secs(60));
        assert!(drained, "set output_done -> true");
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(1),
            "returned on the marker, not after the 60s cap"
        );
    }

    /// No probe stack -> no-op, returns `true` (every caller is a no-op).
    #[test]
    fn drain_probe_pipeline_none_is_noop() {
        assert!(drain_probe_pipeline(
            None,
            std::time::Duration::from_millis(1)
        ));
    }
}
