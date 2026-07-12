//! Per-frame dispatch for the virtio-console port-1 bulk TLV stream.
//!
//! The freeze coordinator's TOKEN_TX epoll branch drives this module:
//! after `bulk_assembler.feed(...)` returns a `BulkMessages` vec, the
//! coordinator iterates each `BulkMessage` through
//! [`dispatch_bulk_message`], which returns a verdict-bearing
//! [`crate::vmm::wire::ShmEntry`] for the run-wide bucket or `None`
//! for a frame whose only dispatch effect is on `sinks`. The
//! `None`-returning arms (no bucket entry) are `SysRdy` (fire-once
//! sys-rdy eventfd), `SchedSwapNotify` (periodic-capture +
//! watchpoint-invalidation latch), `KERN_ADDRS` (phys-base / KASLR
//! stores + eventfds), and `SnapshotRequest` / `KernelOpRequest`
//! (decode-and-stash of a request payload). Several arms return
//! `Some` AND stamp `sinks`: `SchedExit` (`kill` flag + eventfd,
//! bucketed only when `crc_ok`), `ScenarioStart` (scenario anchor +
//! watchdog reset), `ScenarioPause` / `ScenarioResume` (watchdog
//! pause / cumulative-pause), `ScenarioEnd` (watchdog reset), and
//! `Stimulus` (host-side step-index mirror). For a variant without
//! an explicit arm the catch-all consults
//! [`crate::vmm::wire::MsgType::is_coordinator_internal`] — verbatim
//! bucket push when it is false, silent drop when true. See the
//! per-arm bodies for each arm's exact return and side effect.
//!
//! Splitting the dispatch out of the run-loop closure body lets test
//! code drive arbitrary CRC-mangled frame sequences against a pure
//! function — no VM boot required, no Arc plumbing beyond the sinks
//! a test wants to observe. Production behaviour is byte-for-byte
//! preserved; the only logic change relative to the inline code is
//! the function boundary.
//!
//! Hostile-guest discipline is identical to the inline arms: every
//! CRC-bearing promotion gates on `msg.crc_ok`, the SysRdy promotion
//! is fire-once via [`Option::take`], unknown msg_type entries warn-
//! and-drop without polluting the verdict, and SnapshotReply / other
//! coordinator-internal variants are filtered via
//! [`crate::vmm::wire::MsgType::is_coordinator_internal`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use vmm_sys_util::eventfd::EventFd;

use super::snapshot::decode_snapshot_request;
use super::state::SnapshotRequest;
use crate::vmm::KERNEL_HALF_CANONICAL as KERNEL_HALF_CANONICAL_4LEVEL;

/// Aggregate of the coordinator-side sinks the TOKEN_TX dispatch can
/// touch. Bundling them keeps the [`dispatch_bulk_message`] signature
/// readable (one `&mut` arg instead of four) and makes the test
/// fixture explicit: a test sets up exactly the sinks it wants to
/// observe, runs the dispatch, then asserts the post-state.
///
/// `sys_rdy_evt` is `&mut Option<...>` so [`Option::take`] can fire
/// the SysRdy eventfd exactly once per coordinator lifetime — the
/// inline production code uses the same `Option::take` to drop the
/// host-side handle after the first promotion, and the function-
/// extracted form preserves that behaviour by mutating through the
/// passed reference.
pub(super) struct BulkDispatchSinks<'a> {
    /// Run-wide kill flag flipped on a CRC-valid `MSG_TYPE_SCHED_EXIT`.
    /// Loaded by the BSP run loop, the watchdog, and the freeze
    /// coordinator's outer `while` predicate.
    pub kill: &'a Arc<AtomicBool>,
    /// Wake fd paired with [`Self::kill`]. Written `1` immediately
    /// after the AtomicBool flip so any consumer blocked in
    /// `epoll_wait` returns within microseconds rather than waiting
    /// up to one full poll interval.
    pub kill_evt: &'a Arc<EventFd>,
    /// True when THIS run is a wprof run (`freeze_coord_wprof`). While
    /// set, the SchedExit arm SKIPS the kill promotion: on a self-crash
    /// the still-live guest sched-exit monitor sends SCHED_EXIT during
    /// Phase 5 (before Phase 6 stops it), and an unconditional kill here
    /// would tear the coordinator down before the guest's Phase-5 wprof
    /// ship. Gating on the RUN being wprof (not on the grace deadline
    /// already being armed) closes an arming-order race: the SchedExit
    /// can be dispatched in the SAME coord iteration as — or before — the
    /// error-exit arm sets `wprof_ship_deadline`, so a deadline snapshot
    /// would be stale and promote the kill anyway. Instead the coord arms
    /// the grace on the drained SchedExit (or the error-exit dump), and
    /// terminates on the WprofTrace frame or the `WPROF_SHIP_GRACE`
    /// backstop. The SchedExit verdict entry is still bucketed; only the
    /// kill promotion holds off.
    pub run_is_wprof: bool,
    /// Boot-complete signal. Promoted exactly once on the first
    /// CRC-valid empty-payload `MSG_TYPE_SYS_RDY` frame; the
    /// `Option::take` retains the host-side handle until that point
    /// and drops it after firing so subsequent SYS_RDY frames (a
    /// hostile guest could in principle resend) skip the eventfd
    /// write.
    pub sys_rdy_evt: &'a mut Option<Arc<EventFd>>,
    /// Scheduler-swap notification latch. Set `true` on a CRC-valid
    /// `MSG_TYPE_SCHED_SWAP_NOTIFY` frame; the freeze coordinator's
    /// run-loop reads-and-clears it each iteration and synchronously
    /// invalidates the stale periodic-capture accessor (waking the
    /// accessor-init worker to rebuild against the next scheduler)
    /// rather than waiting up to one SCAN_INTERVAL for the scx_root
    /// watchpoint poll to notice the rebind. A torn frame does NOT set
    /// it (the CRC gate in the dispatch arm) so a garbled or hostile
    /// frame cannot force a spurious mid-capture accessor rebuild.
    pub sched_swap_notify: &'a std::sync::atomic::AtomicBool,
    /// Per-iteration accumulator for decoded
    /// `MSG_TYPE_SNAPSHOT_REQUEST` frames. Drained later in the run-
    /// loop body where `freeze_and_dispatch` /
    /// `arm_user_watchpoint` are in scope. CRC-bad frames and
    /// malformed payloads (size mismatch, KIND_NONE, request_id == 0)
    /// never reach this Vec — [`decode_snapshot_request`] returns
    /// `None` and the entry is dropped without observable side effect.
    pub snapshot_requests_pending: &'a mut Vec<SnapshotRequest>,
    /// Per-iteration accumulator for decoded
    /// `MSG_TYPE_KERNEL_OP_REQUEST` frames. Drained later in the run-
    /// loop body where `freeze_and_dispatch` is in scope — each
    /// pending kernel-op request triggers its own freeze rendezvous,
    /// runs `gmem.write_obj` / `gmem.read_obj` while every vCPU is
    /// parked, and ships the
    /// [`crate::vmm::wire::KernelOpReplyPayload`] back to the guest
    /// over port-1 RX. CRC-bad frames are silently dropped (a torn
    /// frame would otherwise let a hostile guest force a freeze or
    /// inject a write target the host never validated); malformed
    /// postcard payloads decode to `None` and drop. Bounded by the
    /// host's port-1 RX queue capacity — guest publishers backpressure
    /// against that gate rather than this Vec.
    pub kernel_op_requests_pending: &'a mut Vec<crate::vmm::wire::KernelOpRequestPayload>,
    /// Guest-reported `phys_base + 1`. Stored by the KERN_ADDRS arm
    /// so the monitor thread can pick it up via Acquire load.
    pub kern_phys_base: &'a Arc<std::sync::atomic::AtomicU64>,
    /// Fires when `kern_phys_base` transitions from 0 to non-zero.
    pub kern_phys_base_evt: &'a EventFd,
    /// Derived virt-KASLR offset (biased `+1` so 0 = "not yet
    /// derived"). The KERN_ADDRS arm extracts the guest-reported
    /// runtime `_text` KVA from the third payload slot and stores
    /// `(runtime - link) + 1` here when the link KVA from
    /// [`Self::kernel_text_link_kva`] is non-zero AND the runtime
    /// KVA is above the kernel-half threshold (rejects torn
    /// payloads from a hostile guest). Idempotent CAS: the
    /// BSP-side MSR_LSTAR path
    /// (`src/vmm/x86_64/msr_kaslr::read_and_derive`) publishes the
    /// same derived value to this slot from the BSP thread, so the
    /// first writer wins and the second observes the existing
    /// non-zero value via the CAS-fail branch (KASLR shifts both
    /// `_text` and `entry_SYSCALL_64` by the same `kaslr_offset`,
    /// so the two paths produce identical offsets even though
    /// they read different runtime symbols). Consumers (monitor +
    /// dump) `.load()` and subtract 1 to recover the offset; 0
    /// means "no path has succeeded yet — use literal 0 for
    /// KASLR-off semantics".
    pub kern_virt_kaslr: &'a Arc<std::sync::atomic::AtomicU64>,
    /// Fires when `kern_virt_kaslr` transitions from 0 to non-zero.
    /// Mirrors [`Self::kern_phys_base_evt`].
    pub kern_virt_kaslr_evt: &'a EventFd,
    /// Link-time KVA of `_text` (the kernel image start symbol)
    /// from the host's vmlinux parse
    /// (`KernelSymbols::kernel_text_kva`). The KERN_ADDRS arm
    /// subtracts this from the guest-reported runtime KVA to
    /// derive the virt-KASLR offset. `0` when the symbol is
    /// absent (extremely stripped vmlinux) — the arm
    /// short-circuits and leaves [`Self::kern_virt_kaslr`] at 0
    /// in that case (matches KASLR-off semantics). `_text` is
    /// defined in `vmlinux.lds.S` on every architecture so the
    /// host-side extraction is cross-arch.
    pub kernel_text_link_kva: u64,
    /// Watchdog reset atomic + workload duration + `run_start` anchor +
    /// the parallel provenance tag. SCENARIO_START stores
    /// `(now - run_start + duration).as_nanos()` so the watchdog starts
    /// the workload clock from scenario start, not from boot or SYS_RDY;
    /// SCENARIO_RESUME / SCENARIO_END re-arm it too. Every arm that
    /// stores `reset_ns` also stamps the tag with
    /// `WatchdogResetTag::ScenarioStart` (one tag for the whole
    /// scenario-dispatch subsystem) so the watchdog dump does not read a
    /// stale earlier writer's tag.
    pub watchdog_reset: Option<(
        &'a std::sync::atomic::AtomicU64,
        std::time::Duration,
        std::time::Instant,
        &'a std::sync::atomic::AtomicU8,
    )>,
    /// Pause timestamp (nanos since run_start). 0 = not paused.
    /// ScenarioPause stores current elapsed; ScenarioStart clears
    /// it and extends the deadline by the pause duration.
    pub watchdog_pause_ns: &'a std::sync::atomic::AtomicU64,
    /// First-`ScenarioStart` timestamp (nanos since `run_start`),
    /// clamped to a `1` floor so `0` means "not yet observed" (a
    /// boot-unreachable elapsed==0 maps to 1; every real stamp is
    /// exact, not a `+1` shift). The first
    /// CRC-valid `MSG_TYPE_SCENARIO_START` frame stamps
    /// `u64::try_from(run_start.elapsed().as_nanos()).unwrap_or(u64::MAX).max(1)`
    /// here via
    /// a one-shot `compare_exchange(0, ..)`; subsequent ScenarioStart
    /// frames (the guest may publish multiple if the workload
    /// re-runs) leave the prior stamp untouched. Consumed by the
    /// freeze coordinator's periodic-capture loop as the window's END
    /// anchor (`scenario_start + workload_duration`, clamped) and the
    /// scenario-relative offset frame for `KtstrTestEntry::num_snapshots`
    /// boundaries; the window START floats to the later of this stamp
    /// and the prereq-ready moment. Boot + verifier time before the
    /// first ScenarioStart does not eat the budget.
    pub scenario_start_ns: &'a std::sync::atomic::AtomicU64,
    /// Cumulative wall-clock pause time observed between matched
    /// `MSG_TYPE_SCENARIO_PAUSE` / `MSG_TYPE_SCENARIO_RESUME` pairs
    /// (nanoseconds). Bumped on every `ScenarioResume` by
    /// `(now - paused_at)`. Periodic-capture boundaries in the
    /// coord run-loop are anchored to workload time, not wall-clock
    /// time — they subtract this cumulative pause from
    /// `run_start.elapsed()` so a guest that pauses for `P` ns
    /// shifts every un-fired boundary by `P` ns, matching the
    /// guest's logical clock.
    pub scenario_pause_cumulative_ns: &'a std::sync::atomic::AtomicU64,
    /// Run-start anchor for elapsed-time computations. Available
    /// unconditionally (no `Option` wrapper) so the
    /// `MSG_TYPE_SCENARIO_START` arm can stamp
    /// [`Self::scenario_start_ns`] regardless of whether the
    /// caller wired up a watchdog reset budget — periodic capture
    /// (which consumes the stamp) only requires
    /// `workload_duration` at the run-loop level, not at the
    /// dispatch level.
    pub run_start: std::time::Instant,
    /// Host-side mirror of the guest's scenario phase index.
    /// Updated on every CRC-valid `MSG_TYPE_STIMULUS` frame in
    /// the dispatch loop from
    /// [`crate::vmm::wire::StimulusEvent::step_index`]; consumed
    /// by the freeze-coordinator's periodic-capture path so
    /// each periodic sample carries the phase it fired in.
    /// Encoded per the framework's 1-indexed phase convention
    /// (`0` = BASELINE settle window, `1..=N` = Step ordinals)
    /// to match the encoding the guest publishes (see
    /// [`crate::scenario::Ctx::current_step`] and
    /// [`crate::assert::PhaseBucket::step_index`]).
    /// `Release` on the writer here pairs with `Acquire` on the
    /// periodic-fire reader so the published step is visible
    /// once the dispatch loop returns from the frame that
    /// promoted it.
    pub current_step: &'a std::sync::Arc<std::sync::atomic::AtomicU16>,
    /// Live lifecycle-stage ledger shared with the monitor and (a LATER
    /// commit) the VM watchdog. The dispatch arms advance it
    /// FORWARD-ONLY as they observe the guest's lifecycle / scenario
    /// frames — `SysRdy`→Attach, `PayloadStarting`→Dispatch,
    /// `WorkloadDispatched`/`ScenarioStart`→Body,
    /// `ScenarioEnd`/`Exit`/`TestResult`→Teardown — and an `InitStarted`
    /// lifecycle frame records a boot-progress epoch without moving the
    /// stage. Each advance also folds a progress epoch (a stage change is
    /// new-work evidence). `None` for callers that do not wire a ledger
    /// (the CRC-defense / tx-dispatch unit fixtures); every advance site
    /// guards on the `Option`, so an unwired caller is a silent no-op.
    /// Only advanced on `msg.crc_ok` frames — a torn or hostile-guest
    /// frame must not forge stage progress.
    pub progress_ledger: Option<&'a crate::monitor::ProgressLedger>,
}

/// Advance the shared lifecycle ledger to `stage`, stamping the frame's
/// `run_start`-relative wall time. Forward-only and idempotent per
/// [`crate::monitor::ProgressLedger::advance_phase`], so a duplicate or
/// out-of-order frame is a no-op. Centralises the `Option`-guard + wall-
/// time sample shared by every advancing arm. No-op without a wired
/// ledger (unit fixtures).
fn advance_stage(sinks: &BulkDispatchSinks<'_>, stage: crate::monitor::LifecycleStage) {
    if let Some(ledger) = sinks.progress_ledger {
        let wall_ns = u64::try_from(sinks.run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        ledger.advance_phase(stage as u8, wall_ns);
    }
}

/// Record a progress epoch WITHOUT a stage change — a boot heartbeat
/// (`Lifecycle` `InitStarted`) that proves the guest is live during
/// `Boot` but does not move the coarse stage forward. Folds into
/// `progress_epoch` exactly like a monitor-observed new-work tick. No-op
/// without a wired ledger.
fn record_boot_progress(sinks: &BulkDispatchSinks<'_>) {
    if let Some(ledger) = sinks.progress_ledger {
        let wall_ns = u64::try_from(sinks.run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let cpu_ns_now = ledger.cpu_ns_now.load(std::sync::atomic::Ordering::Relaxed);
        ledger.record_progress(cpu_ns_now, wall_ns);
    }
}

/// Arm the watchdog reset deadline to `elapsed + workload_duration` and
/// stamp `tag` as its provenance, mirroring the [`crate::vmm::wire::MsgType::ScenarioStart`]
/// arm's arithmetic. `elapsed` is `run_start.elapsed()` sampled by the
/// caller (so a single clock read serves both the deadline and any
/// sibling stamp). Plain `Release` store of `reset_ns` (not a max-style
/// compare) — a fresh confirmed-progress frame re-anchors the workload
/// clock unconditionally; the tag store is Relaxed (diagnostic only).
/// No-op without a wired `watchdog_reset` budget (the CRC-defense /
/// tx-dispatch unit fixtures leave it `None`).
fn arm_watchdog_reset(
    sinks: &BulkDispatchSinks<'_>,
    elapsed: std::time::Duration,
    tag: crate::vmm::freeze_coord::WatchdogResetTag,
) {
    if let Some((reset_ns, duration, _, reset_tag)) = sinks.watchdog_reset.as_ref() {
        let target_ns = elapsed.as_nanos().saturating_add(duration.as_nanos());
        let encoded = u64::try_from(target_ns).unwrap_or(u64::MAX).max(1);
        reset_ns.store(encoded, std::sync::atomic::Ordering::Release);
        reset_tag.store(tag as u8, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Apply the live-stage side effect of a `Lifecycle` frame: decode the
/// 1-byte [`crate::vmm::wire::LifecyclePhase`] discriminant (`payload[0]`)
/// and advance the ledger / record boot progress accordingly. Crc-gated
/// — a torn frame must not forge stage progress. Pure ledger side
/// effect; the caller still buckets the frame verbatim. Extracted from
/// [`dispatch_bulk_message`] so the arm stays a one-liner (and the
/// per-fn size guard is not perturbed by the phase match).
fn lifecycle_stage_advance(sinks: &BulkDispatchSinks<'_>, msg: &crate::vmm::bulk::BulkMessage) {
    if !msg.crc_ok {
        return;
    }
    let Some(phase) = msg
        .payload
        .first()
        .and_then(|b| crate::vmm::wire::LifecyclePhase::from_wire(*b))
    else {
        return;
    };
    match phase {
        // InitStarted: boot heartbeat — reached init but still Boot.
        // Counts as progress (liveness during a slow boot) WITHOUT a
        // stage advance.
        crate::vmm::wire::LifecyclePhase::InitStarted => record_boot_progress(sinks),
        // →Dispatch: guest dispatch is entering the workload preamble.
        crate::vmm::wire::LifecyclePhase::PayloadStarting => {
            advance_stage(sinks, crate::monitor::LifecycleStage::Dispatch)
        }
        // →Body: a verifier cell proved a worker made forward progress
        // on-CPU (a test cell reaches Body via ScenarioStart instead —
        // either advances).
        crate::vmm::wire::LifecyclePhase::WorkloadDispatched => {
            advance_stage(sinks, crate::monitor::LifecycleStage::Body)
        }
        // SchedulerAttached: a REAL sched_ext scheduler bound with a live
        // child (see `spawn_scheduler_from_paths`). This lands DURING the
        // Attach stage — `SysRdy` already moved Boot→Attach and the
        // →Dispatch/→Body advances come from `PayloadStarting` /
        // `ScenarioStart` / `WorkloadDispatched` — so it records PROGRESS
        // (confirmed attach is new-work evidence) WITHOUT advancing the
        // coarse stage. THE MAIN ACT: arm the watchdog reset to
        // now + workload_duration, tagged `GuestAttachConfirm`, so the
        // progress watchdog measures the workload budget from the
        // guest-confirmed attach moment rather than from VM boot. This is
        // the authoritative, evented counterpart to the monitor's polled
        // `*scx_root` latch (now demoted to a CAS fallback in
        // `monitor::reader`): the guest frame's plain store wins over a
        // later latch store, and the latch's CAS(0,..) fails once this
        // frame has armed the deadline.
        crate::vmm::wire::LifecyclePhase::SchedulerAttached => {
            record_boot_progress(sinks);
            arm_watchdog_reset(
                sinks,
                sinks.run_start.elapsed(),
                crate::vmm::freeze_coord::WatchdogResetTag::GuestAttachConfirm,
            );
        }
        // SchedulerDied / SchedulerNotAttached: NO stage change and NO
        // progress. The guest continues to Phase 5 regardless of an
        // attach failure, and a failure is not forward progress.
        crate::vmm::wire::LifecyclePhase::SchedulerDied
        | crate::vmm::wire::LifecyclePhase::SchedulerNotAttached => {}
    }
}

/// Classify and dispatch a single `BulkMessage` from the port-1
/// TLV stream. Returns the verdict-bearing [`crate::vmm::wire::ShmEntry`]
/// to push into the run-wide bucket, or `None` for coordinator-
/// internal frames whose only effect was on `sinks`.
///
/// # Promotion gates (hostile-guest defence)
///
/// * `MSG_TYPE_SCHED_EXIT` flips `kill` and writes `kill_evt` ONLY
///   when `msg.crc_ok`. A torn frame would otherwise let a hostile
///   guest force a false early exit. CRC-bad SchedExit also does NOT
///   land in the verdict bucket — the per-type contract requires
///   `crc_ok` for SchedExit to be observable.
/// * `MSG_TYPE_SYS_RDY` fires its eventfd ONLY when `msg.crc_ok &&
///   msg.payload.is_empty()`. The empty-payload gate is the safety
///   net against a hostile guest tacking smuggle bytes onto a SysRdy
///   frame past the [`crate::vmm::wire::MsgType::is_coordinator_internal`]
///   filter. Promotion is fire-once via [`Option::take`].
/// * `MSG_TYPE_SCHED_SWAP_NOTIFY` latches the run-loop's synchronous
///   periodic-capture + watchpoint-invalidation step ONLY when
///   `msg.crc_ok && msg.payload.is_empty()` — the same empty-payload
///   safety net as `SysRdy`, so a hostile guest cannot smuggle bytes
///   past the [`crate::vmm::wire::MsgType::is_coordinator_internal`]
///   filter under this tag. A torn frame must NOT latch: it would
///   otherwise force a spurious accessor rebuild mid-capture. Unlike
///   `SysRdy` the store is not fire-once — the coordinator run-loop
///   reads-and-clears the latch (`swap(false)`) each iteration.
/// * `MSG_TYPE_SNAPSHOT_REQUEST` decodes via [`decode_snapshot_request`]
///   ONLY when `msg.crc_ok`. The decoder additionally rejects
///   `request_id == 0`, `kind == SNAPSHOT_KIND_NONE`, and any
///   payload whose size does not match the typed wire layout.
/// * Every other variant: pushes verbatim if not coordinator-
///   internal, drops silently if it is. Unknown msg_type values
///   warn-and-drop so a future guest variant cannot synthesise a
///   phantom verdict entry on the host.
///
/// # CRC handling on verdict-bearing arms
///
/// Non-SchedExit verdict frames (Stimulus, ScenarioStart,
/// ScenarioEnd, Exit, TestResult, Crash, PayloadMetrics,
/// Profraw, Stdout, Stderr, SchedLog, Lifecycle,
/// ExecExit, Dmesg, ProbeOutput) accumulate even when `crc_ok` is
/// false — the host-side consumers filter on per-type contract.
/// SchedExit is the lone exception: its kill-flag promotion makes a
/// torn-frame leak load-bearing for a hostile guest, so we gate the
/// bucket push on the same `crc_ok` flag the promotion checks.
pub(super) fn dispatch_bulk_message(
    msg: &crate::vmm::bulk::BulkMessage,
    sinks: &mut BulkDispatchSinks<'_>,
) -> Option<crate::vmm::wire::ShmEntry> {
    let kind = crate::vmm::wire::MsgType::from_wire(msg.msg_type);
    match kind {
        Some(crate::vmm::wire::MsgType::SchedExit) => {
            // Promote a guest-side SCHED_EXIT into the run-wide kill
            // flag so the BSP loop and the watchdog exit promptly
            // instead of running until the watchdog deadline. CRC
            // failures DO NOT promote — a torn frame would otherwise
            // let a hostile guest force a false early exit.
            // Skip the kill promotion on a wprof run (run_is_wprof): on a
            // self-crash the still-live guest sched-exit monitor sends
            // SCHED_EXIT during Phase 5. This frame can arrive in the SAME
            // coord iteration as — or before — the error-exit arm sets the
            // grace deadline, so gating on "grace already armed" would race
            // and promote the kill anyway. Gate on the RUN being wprof
            // instead; the coord arms the grace on this drained SchedExit
            // (see the coord loop) or the error-exit dump, and terminates
            // on the WprofTrace frame or the WPROF_SHIP_GRACE backstop.
            // Promoting the kill here would tear the coordinator down
            // before the Phase-5 wprof ship. The verdict entry is still
            // bucketed below.
            if msg.crc_ok {
                if sinks.run_is_wprof {
                    eprintln!(
                        "freeze_coord: SchedExit received; kill SKIPPED (wprof run; grace armed on drain)"
                    );
                } else {
                    eprintln!("freeze_coord: SchedExit received; kill PROMOTED");
                    sinks.kill.store(true, Ordering::Release);
                    // EFD_NONBLOCK on a freshly-created eventfd never
                    // legitimately fails; log unconditionally so a future
                    // regression (e.g. the eventfd was closed by another
                    // owner) surfaces in the host log instead of silently
                    // swallowing the kill edge.
                    if let Err(e) = sinks.kill_evt.write(1) {
                        tracing::warn!(
                            err = %e,
                            "freeze_coord: kill_evt write on SCHED_EXIT \
                             promotion failed; the kill AtomicBool above is \
                             still authoritative"
                        );
                    }
                }
            }
            // SchedExit is verdict data — bucket only on CRC-valid
            // frames so a torn or hostile-guest tag never surfaces as
            // a phantom verdict entry in `BulkDrainResult`. The
            // promotion gate above already short-circuits on
            // crc_ok=false; mirror the same discipline here so the
            // verdict-side filter is not deferred to a downstream
            // consumer that does not exist.
            if msg.crc_ok {
                Some(crate::vmm::wire::ShmEntry {
                    msg_type: msg.msg_type,
                    payload: msg.payload.to_vec(),
                    crc_ok: msg.crc_ok,
                })
            } else {
                None
            }
        }
        Some(crate::vmm::wire::MsgType::SysRdy) => {
            // Promote a CRC-valid, empty-payload SysRdy into the
            // monitor's boot-complete eventfd so the monitor thread's
            // pre-sample `epoll_wait` returns within microseconds
            // rather than waiting for the 5 s fallback. CRC failures
            // DO NOT promote — a torn frame would let a hostile guest
            // forge a fake boot signal that races ahead of
            // `setup_per_cpu_areas` / KASLR. The `Option::take` makes
            // promotion fire-once: a resent SysRdy skips the eventfd
            // write so the counter does not pump. SysRdy must carry
            // no payload — a hostile guest tacking bytes on would
            // otherwise smuggle data past the
            // is_coordinator_internal filter; this strict shape gate
            // is the safety net.
            if msg.crc_ok
                && msg.payload.is_empty()
                && let Some(evt) = sinks.sys_rdy_evt.take()
                && let Err(e) = evt.write(1)
            {
                tracing::warn!(
                    err = %e,
                    "freeze_coord: sys_rdy write failed; monitor will \
                     rely on kill_evt or 5 s timeout to leave its \
                     pre-sample wait"
                );
            }
            // Boot→Attach: SYS_RDY means the guest reached boot-complete
            // and the scheduler is attaching to sched_ext. Gate on crc_ok
            // only — the empty-payload gate above is the anti-smuggle net
            // for the eventfd, but a CRC-valid boot signal advances the
            // stage regardless of a (hostile) trailing byte. Forward-only,
            // so a resent SysRdy is a no-op.
            if msg.crc_ok {
                advance_stage(sinks, crate::monitor::LifecycleStage::Attach);
            }
            // SysRdy is coordinator-internal — do NOT bucket.
            None
        }
        Some(crate::vmm::wire::MsgType::SchedSwapNotify) => {
            // CRC-valid, EMPTY-payload swap-notify: latch the
            // synchronous periodic-capture + watchpoint invalidation the
            // coordinator's run-loop performs at the top of its next
            // iteration. A torn frame must NOT latch — a garbled or
            // hostile frame would otherwise force a spurious accessor
            // rebuild mid-capture. The empty-payload gate mirrors the
            // SysRdy safety net: the latch ignores payload bytes, so a
            // hostile guest must not be able to smuggle bytes past the
            // is_coordinator_internal filter under this tag.
            // Coordinator-internal — do NOT bucket (mirrors SysRdy).
            if msg.crc_ok && msg.payload.is_empty() {
                sinks
                    .sched_swap_notify
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            None
        }
        _ if msg.msg_type == crate::vmm::wire::MSG_TYPE_KERN_ADDRS => {
            // Payload carries (via [`crate::vmm::wire::KernAddrs`]):
            //   [0..8]   phys_base + 1                (biased)
            //   [8..16]  page_offset_base             (not read by
            //                                          this arm; guest
            //                                          sends the symbol
            //                                          KVA from
            //                                          kallsyms, 0 only
            //                                          on arm64 /
            //                                          RANDOMIZE_MEMORY=n
            //                                          / kallsyms
            //                                          unreadable)
            //   [16..24] kernel_text_runtime_kva + 1  (biased; `_text`
            //                                          symbol from
            //                                          guest's
            //                                          /proc/kallsyms)
            //
            // All three slots are u64 LE. The biased slots use the
            // +1 trick so the 0 sentinel distinguishes "not yet
            // received / could not derive" from a legitimately-zero
            // value (phys_base = 0 with KASLR off; kernel_text = 0
            // means symbol masked or absent).
            //
            // CRC failures DO NOT promote — a torn frame would
            // otherwise let a hostile guest forge a wrong KVA and
            // poison the monitor's per-CPU resolution. The decoder
            // gates on `payload.len() == KernAddrs::WIRE_LEN`
            // (exact 24 bytes); shorter or longer payloads never
            // publish either slot. Exact-length match (vs `>=`)
            // makes a future protocol extension that appends bytes
            // trip loudly at this arm rather than silently dropping
            // the new bytes.
            if msg.crc_ok
                && let Some(addrs) = crate::vmm::wire::KernAddrs::from_payload(&msg.payload)
            {
                if addrs.has_phys_present_bit() {
                    let biased_phys = addrs.phys_base.wrapping_add(1);
                    sinks
                        .kern_phys_base
                        .store(biased_phys, std::sync::atomic::Ordering::Release);
                    // Mirror the SchedExit promotion's wake-failure logging:
                    // the store above is authoritative, and the monitor also
                    // polls, so a missed wake self-heals within a poll
                    // interval — but surface the failure rather than dropping
                    // the Result silently.
                    if let Err(e) = sinks.kern_phys_base_evt.write(1) {
                        tracing::warn!(
                            err = %e,
                            "freeze_coord: kern_phys_base_evt wake write failed; \
                             the kern_phys_base store above is authoritative and \
                             the monitor poll self-heals within a poll interval"
                        );
                    }
                }
                // Derive virt-KASLR from the guest-reported runtime
                // `_text` KVA + the host's link-time KVA. Skip if
                // either input is unavailable:
                //   * link KVA == 0    — vmlinux missing the symbol
                //                         (extremely stripped build)
                //   * runtime_kva == 0 — guest could not read
                //                         /proc/kallsyms
                // In either case the BSP MSR_LSTAR path may still
                // publish a non-zero value on x86_64; leaving the
                // slot at 0 matches that fallback's "not yet
                // derived" sentinel. `_text` is the kernel image
                // start symbol — defined on every Linux build, so
                // this derivation works on both x86_64 and aarch64.
                // Hostile-input gate stack (defense in depth against
                // a torn / compromised guest payload AND a corrupted
                // host vmlinux ELF):
                //
                //   (l) `link >= KERNEL_HALF_CANONICAL_4LEVEL` —
                //       host-side: if the vmlinux ELF parse
                //       returned a low / garbage `_text` symbol
                //       (e.g. 0x1000 from a corrupted symbol
                //       table), refuse to derive against it.
                //       Catches the broken-ELF case before gates
                //       (a)/(b)/(c) below have to.
                //
                //   (a) `runtime >= KERNEL_HALF_CANONICAL_4LEVEL` —
                //       runtime KVA must be in the kernel-half
                //       canonical range (bits 63..47 all set).
                //       Rejects non-canonical and userspace
                //       addresses.
                //
                //   (b) `runtime >= link` — KASLR shifts text
                //       symbols by a non-negative slot per
                //       `find_random_virt_addr` in
                //       arch/x86/boot/compressed/kaslr.c. A
                //       runtime KVA below the link KVA would
                //       wrap into a huge u64 offset.
                //
                //   (c) `offset <= RANDOMIZE_BASE_MAX_OFFSET` —
                //       per-arch upper bound on the legitimate
                //       KASLR slot range.
                //       - **x86_64**: `1 GiB` per
                //         `arch/x86/include/asm/page_64_types.h`
                //         (`KERNEL_IMAGE_SIZE`).
                //         `find_random_virt_addr` in
                //         `arch/x86/boot/compressed/kaslr.c`
                //         picks within
                //         `[0, KERNEL_IMAGE_SIZE - KERNEL_BASE)`.
                //       - **aarch64**: bounded above by the kernel's
                //         own picker range
                //         `(VMALLOC_END - KIMAGE_VADDR) / 2` in
                //         `arch/arm64/kernel/pi/kaslr_early.c:60-61`,
                //         which under any supported `VA_BITS`
                //         (48 or 52) is comfortably less than
                //         `1 << 48` (256 TiB).  The previous
                //         `1 GiB` bound was the x86_64 invariant
                //         applied unconditionally and silently
                //         rejected every legitimate aarch64 KASLR
                //         offset (kernel picks
                //         `range/2 + uniform([0, range))` =
                //         `[range/2, 3*range/2)` where `range` is
                //         tens of TiB on VA_BITS=48), leaving
                //         `kern_virt_kaslr` at the unpublished
                //         sentinel and indefinitely deferring every
                //         freeze coordinator path that gates on
                //         `kern_virt_kaslr_published()` (notably
                //         the periodic-snapshot capture loop and
                //         the cleanup-time accessor resolution).
                //       The forged / torn-payload defense is
                //       preserved at the per-arch bound; anything
                //       above the kernel's plausible picker range
                //       still rejects.
                #[cfg(target_arch = "x86_64")]
                const RANDOMIZE_BASE_MAX_OFFSET: u64 = 1 << 30; // 1 GiB
                #[cfg(target_arch = "aarch64")]
                const RANDOMIZE_BASE_MAX_OFFSET: u64 = 1 << 48; // 256 TiB
                if let Some(runtime) = addrs.kernel_text_runtime_kva
                    && sinks.kernel_text_link_kva >= KERNEL_HALF_CANONICAL_4LEVEL
                {
                    let link = sinks.kernel_text_link_kva;
                    if runtime >= KERNEL_HALF_CANONICAL_4LEVEL
                        && runtime >= link
                        && (runtime - link) <= RANDOMIZE_BASE_MAX_OFFSET
                    {
                        let offset = runtime - link;
                        let biased_offset = offset.wrapping_add(1);
                        // CAS-once: idempotent vs the BSP-side
                        // MSR_LSTAR publisher. Both paths derive
                        // the SAME virt-KASLR (KASLR is a single
                        // boot-time slot pick stored in
                        // `kaslr_offset`). Release pairs with the
                        // consumer `.load(Acquire)` in the
                        // periodic-capture gate
                        // (`kern_virt_kaslr_published`) and the
                        // cleanup-time accessor resolution in
                        // freeze_coord/mod.rs.
                        //
                        // CAS-fail cross-check: an EQUAL value is
                        // the expected no-op (MSR_LSTAR publisher
                        // won the race). CAS-fail with a DIFFERENT
                        // value means divergence — either a stale
                        // vmlinux template, a hostile guest payload
                        // that forged `_text`, or a kernel mid-boot
                        // KASLR re-roll bug. Fail loud rather than
                        // dropping silently; the tracing::error!
                        // surfaces the delta for operator diagnosis
                        // without aborting the run (next consumer
                        // .load() still reads the first-writer-wins
                        // value).
                        match sinks.kern_virt_kaslr.compare_exchange(
                            0,
                            biased_offset,
                            std::sync::atomic::Ordering::Release,
                            std::sync::atomic::Ordering::Acquire,
                        ) {
                            Ok(_) => {
                                // Same wake-failure discipline as the phys_base
                                // arm above: the CAS is authoritative and the
                                // monitor poll self-heals a missed wake.
                                if let Err(e) = sinks.kern_virt_kaslr_evt.write(1) {
                                    tracing::warn!(
                                        err = %e,
                                        "freeze_coord: kern_virt_kaslr_evt wake write \
                                         failed; the kern_virt_kaslr CAS above is \
                                         authoritative and the monitor poll self-heals \
                                         within a poll interval"
                                    );
                                }
                            }
                            Err(existing) if existing != biased_offset => {
                                let lstar_derived = existing.saturating_sub(1);
                                tracing::error!(
                                    kern_addrs_derived = format_args!("{offset:#x}"),
                                    lstar_derived = format_args!("{lstar_derived:#x}"),
                                    delta = format_args!("{:#x}", offset ^ lstar_derived),
                                    "VirtKaslrDivergence: KERN_ADDRS-derived virt-KASLR \
                                     offset disagrees with the previously-published \
                                     MSR_LSTAR-derived value. Both should equal the \
                                     boot-time slot pick. Possible causes: stale vmlinux \
                                     template (rebuild + retest), kernel mid-boot KASLR \
                                     re-roll, hostile guest payload (KERN_ADDRS _text \
                                     was forged)."
                                );
                            }
                            Err(_) => {
                                // CAS-fail with EQUAL value:
                                // both publishers agree.
                            }
                        }
                    }
                }
            }
            None
        }
        Some(crate::vmm::wire::MsgType::SnapshotRequest) => {
            // Decode and stash a CRC-valid SnapshotRequest for
            // dispatch later in this iteration's body.
            // `freeze_and_dispatch` / `thaw_and_barrier` /
            // `arm_user_watchpoint` are not in scope here. CRC-bad
            // frames are ignored (a torn frame would otherwise let a
            // hostile guest force a capture). Malformed payloads
            // (size mismatch, KIND_NONE, request_id == 0) decode to
            // `None` and drop.
            if msg.crc_ok
                && let Some(req) = decode_snapshot_request(&msg.payload[..])
            {
                sinks.snapshot_requests_pending.push(req);
            }
            // SnapshotRequest is coordinator-internal — its matching
            // reply ships over port-1 RX. Do NOT bucket.
            None
        }
        Some(crate::vmm::wire::MsgType::KernelOpRequest) => {
            // Decode and stash a CRC-valid KernelOpRequest for
            // dispatch later in this iteration's body where
            // `freeze_and_dispatch` + `gmem.write_obj` are in scope.
            //
            // CRC-bad frames drop without a reply: a CRC-failed
            // frame is BY DEFINITION corrupted — the embedded
            // `request_id` and `tag` cannot be trusted, so
            // addressing a reply to "the right" request is
            // impossible. Replying with `request_id=0` or a guessed
            // id would risk landing the reply against a DIFFERENT
            // in-flight request and confusing the guest's blocking
            // reader. Silent drop is the only defensible option for
            // a corrupted frame; the guest's per-op transport
            // deadline (30 s) surfaces the failure prominently
            // enough that an operator catching a timeout in CI can
            // localize to the wire layer. The same discipline
            // governs torn `MSG_TYPE_SNAPSHOT_REQUEST` frames at
            // [`MsgType::SnapshotRequest`] above. Malformed postcard
            // payloads (valid CRC, invalid serialization) decode to
            // `None` via `postcard::from_bytes` and drop the same
            // way for the same reason.
            //
            // Tag bound: `req.tag` is a `String` and downstream
            // formatters (the reply `reason` field, tracing) embed
            // it inline. A framework bug or test-author misuse
            // (e.g. a tag accidentally concatenated with a debug
            // dump) producing a multi-megabyte tag would inflate
            // the postcard-encoded reply past `KERNEL_OP_REPLY_MAX`
            // and the reply would silently drop at the guest's RX
            // cap, surfacing only as a 30 s transport timeout.
            // Bounding the tag length at decode time keeps reply
            // sizes predictable for any input the framework
            // produces. The host-side cold handler (see
            // `freeze_coord/mod.rs` pending-request processing)
            // validates the resolved PA against the kernel-half +
            // guest-memory-size + width-alignment invariants before
            // invoking `gmem.write_obj` / `gmem.read_obj`.
            if msg.crc_ok
                && let Ok(mut req) = postcard::from_bytes::<crate::vmm::wire::KernelOpRequestPayload>(
                    &msg.payload[..],
                )
            {
                // String::truncate panics if the cut lands inside
                // a multi-byte UTF-8 sequence. Any input string
                // (operator-typed test name, framework-generated
                // phase label) could happen to land its 256th byte
                // mid-codepoint, and the bare truncate would crash
                // the coordinator thread. Walk down from
                // KERNEL_OP_TAG_MAX until is_char_boundary returns
                // true; index 0 is guaranteed-valid so the loop
                // terminates without panic.
                if req.tag.len() > crate::vmm::wire::KERNEL_OP_TAG_MAX {
                    let mut idx = crate::vmm::wire::KERNEL_OP_TAG_MAX;
                    while !req.tag.is_char_boundary(idx) {
                        idx -= 1;
                    }
                    req.tag.truncate(idx);
                }
                sinks.kernel_op_requests_pending.push(req);
            }
            // KernelOpRequest is coordinator-internal — its matching
            // reply ships over port-1 RX. Do NOT bucket.
            None
        }
        Some(crate::vmm::wire::MsgType::ScenarioStart) => {
            if msg.crc_ok {
                // One-shot stamp of scenario_start_ns at the FIRST
                // observation, hoisted OUTSIDE the watchdog_reset
                // gate so it fires even when the caller did not
                // wire a workload-duration budget. The `.max(1)`
                // clamps a (boot-unreachable) elapsed==0 to 1 so 0
                // stays the "unset" sentinel — letting the periodic-
                // capture loop distinguish "no scenario started yet"
                // from "scenario started exactly at run_start"; every
                // real stamp passes through unchanged (not a `+1` shift).
                // `compare_exchange` (rather than `store`) makes
                // the stamp idempotent — a guest that publishes
                // ScenarioStart more than once (workload re-runs,
                // multi-phase tests) leaves the first anchor in
                // place. Relaxed ordering is enough: the periodic
                // loop runs in the same coordinator thread, so
                // happens-before is local; no other thread
                // observes this slot.
                let elapsed = sinks.run_start.elapsed();
                let elapsed_ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX).max(1);
                let _ = sinks.scenario_start_ns.compare_exchange(
                    0,
                    elapsed_ns,
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                );
                if let Some((reset_ns, duration, _, reset_tag)) = sinks.watchdog_reset.as_ref() {
                    let target_ns = elapsed.as_nanos().saturating_add(duration.as_nanos());
                    let encoded = u64::try_from(target_ns).unwrap_or(u64::MAX).max(1);
                    reset_ns.store(encoded, std::sync::atomic::Ordering::Release);
                    // Stamp provenance alongside the ns store (Relaxed —
                    // diagnostic only) so the watchdog dump attributes
                    // the deadline to scenario dispatch, not a stale tag.
                    reset_tag.store(
                        crate::vmm::freeze_coord::WatchdogResetTag::ScenarioStart as u8,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                // →Body: a test cell enters its body at ScenarioStart
                // (a verifier cell reaches Body via WorkloadDispatched in
                // the Lifecycle arm instead — either advances). Forward-
                // only, so a workload re-run's later ScenarioStart is a
                // no-op once Body (or Teardown) has been reached.
                advance_stage(sinks, crate::monitor::LifecycleStage::Body);
            }
            Some(crate::vmm::wire::ShmEntry {
                msg_type: msg.msg_type,
                payload: msg.payload.to_vec(),
                crc_ok: msg.crc_ok,
            })
        }
        Some(crate::vmm::wire::MsgType::ScenarioPause) => {
            if msg.crc_ok {
                let elapsed = sinks
                    .watchdog_reset
                    .as_ref()
                    .map(|(_, _, run_start, _)| run_start.elapsed().as_nanos())
                    .unwrap_or(0);
                let encoded = u64::try_from(elapsed).unwrap_or(u64::MAX).max(1);
                sinks
                    .watchdog_pause_ns
                    .store(encoded, std::sync::atomic::Ordering::Release);
            }
            Some(crate::vmm::wire::ShmEntry {
                msg_type: msg.msg_type,
                payload: msg.payload.to_vec(),
                crc_ok: msg.crc_ok,
            })
        }
        Some(crate::vmm::wire::MsgType::ScenarioResume) => {
            if msg.crc_ok
                && let Some((reset_ns, _, run_start, reset_tag)) = sinks.watchdog_reset.as_ref()
            {
                let paused_at = sinks
                    .watchdog_pause_ns
                    .swap(0, std::sync::atomic::Ordering::AcqRel);
                if paused_at > 0 {
                    let elapsed = run_start.elapsed();
                    let pause_duration = elapsed.as_nanos().saturating_sub(paused_at as u128);
                    let prior = reset_ns.load(std::sync::atomic::Ordering::Acquire);
                    let extended = (prior as u128).saturating_add(pause_duration);
                    let encoded = u64::try_from(extended).unwrap_or(u64::MAX).max(1);
                    reset_ns.store(encoded, std::sync::atomic::Ordering::Release);
                    // Same scenario-dispatch subsystem as ScenarioStart —
                    // stamp the tag so the dump never shows a stale
                    // writer after a resume re-arms the deadline.
                    reset_tag.store(
                        crate::vmm::freeze_coord::WatchdogResetTag::ScenarioStart as u8,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    // Bump the periodic-capture cumulative pause
                    // counter by the same `pause_duration`. Periodic
                    // boundaries are anchored to workload time, so a
                    // guest that paused for `pause_duration` ns
                    // shifts every un-fired boundary by that amount
                    // — the run-loop subtracts this cumulative pause
                    // from `run_start.elapsed()` to compute effective
                    // workload-time. Saturating add keeps the bump
                    // honest under the (essentially-unreachable) case
                    // where total pause time exceeds u64::MAX ns.
                    let prior_cumulative = sinks
                        .scenario_pause_cumulative_ns
                        .load(std::sync::atomic::Ordering::Acquire);
                    let new_cumulative = (prior_cumulative as u128).saturating_add(pause_duration);
                    let encoded_cumulative = u64::try_from(new_cumulative).unwrap_or(u64::MAX);
                    sinks
                        .scenario_pause_cumulative_ns
                        .store(encoded_cumulative, std::sync::atomic::Ordering::Release);
                }
            }
            Some(crate::vmm::wire::ShmEntry {
                msg_type: msg.msg_type,
                payload: msg.payload.to_vec(),
                crc_ok: msg.crc_ok,
            })
        }
        Some(crate::vmm::wire::MsgType::ScenarioEnd) => {
            if msg.crc_ok
                && let Some((reset_ns, duration, run_start, reset_tag)) =
                    sinks.watchdog_reset.as_ref()
            {
                let elapsed = run_start.elapsed();
                let target_ns = elapsed.as_nanos().saturating_add(duration.as_nanos());
                let encoded = u64::try_from(target_ns).unwrap_or(u64::MAX).max(1);
                reset_ns.store(encoded, std::sync::atomic::Ordering::Release);
                // Same scenario-dispatch subsystem — stamp the tag so a
                // scenario-end re-arm is not attributed to a stale writer.
                reset_tag.store(
                    crate::vmm::freeze_coord::WatchdogResetTag::ScenarioStart as u8,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            // →Teardown: the scenario ended; the guest is winding down.
            // Hoisted OUT of the watchdog_reset gate above so it fires
            // even when no workload-duration budget was wired. Forward-
            // only, so a later Exit/TestResult Teardown advance is a
            // no-op.
            if msg.crc_ok {
                advance_stage(sinks, crate::monitor::LifecycleStage::Teardown);
            }
            Some(crate::vmm::wire::ShmEntry {
                msg_type: msg.msg_type,
                payload: msg.payload.to_vec(),
                crc_ok: msg.crc_ok,
            })
        }
        Some(crate::vmm::wire::MsgType::Stimulus) => {
            // Decode the published step_index into the host-side
            // mirror so the freeze-coordinator periodic-capture
            // path can stamp samples with the scenario phase the
            // guest was in when the periodic boundary fired.
            //
            // CRC-bad frames do NOT promote — a torn frame would
            // otherwise let a hostile guest forge a phase that
            // mislabels a periodic sample. The decoder also gates
            // on the exact 24-byte payload size in
            // [`crate::vmm::wire::StimulusEvent::from_payload`];
            // an oversized / undersized payload returns None and
            // the publish is skipped. The frame still buckets
            // verbatim below so the post-run drain in
            // `collect_results` recovers the full stimulus log
            // unchanged.
            //
            // Release pairs with the periodic-fire reader's
            // `Acquire` load on the same atomic so the published
            // step is visible immediately on the next iteration's
            // wake. The encoded 1-indexed convention (`0` =
            // BASELINE, `1..=N` = Step ordinals) matches the
            // encoding the guest publishes via
            // [`crate::scenario::ops::build_stimulus`].
            if msg.crc_ok
                && let Some(event) = crate::vmm::wire::StimulusEvent::from_payload(&msg.payload)
            {
                sinks
                    .current_step
                    .store(event.step_index, std::sync::atomic::Ordering::Release);
            }
            // Stimulus is verdict-bearing — bucket verbatim so the
            // post-run drain recovers the full TLV log into
            // `VmResult::guest_messages`, from which
            // `VmResult::stimulus_timeline()` derives the per-phase
            // timeline (step frames + scenario-end terminal) on demand.
            Some(crate::vmm::wire::ShmEntry {
                msg_type: msg.msg_type,
                payload: msg.payload.to_vec(),
                crc_ok: msg.crc_ok,
            })
        }
        Some(crate::vmm::wire::MsgType::Lifecycle) => {
            // Live stage tracking from the guest's 1-byte lifecycle
            // discriminant (see [`lifecycle_stage_advance`]). The frame is
            // STILL bucketed verbatim below: Lifecycle is verdict data the
            // post-hoc verdict scan reads (attach / dispatch failure
            // classification), so the stage side effect must NOT consume
            // it. Mirrors the ScenarioStart arm — act, then return the
            // verbatim ShmEntry.
            lifecycle_stage_advance(sinks, msg);
            Some(crate::vmm::wire::ShmEntry {
                msg_type: msg.msg_type,
                payload: msg.payload.to_vec(),
                crc_ok: msg.crc_ok,
            })
        }
        Some(crate::vmm::wire::MsgType::Exit) => {
            // →Teardown: the guest payload is exiting. Bucket verbatim
            // (Exit is verdict data the post-hoc scan reads) AND advance
            // the stage, forward-only + crc-gated.
            if msg.crc_ok {
                advance_stage(sinks, crate::monitor::LifecycleStage::Teardown);
            }
            Some(crate::vmm::wire::ShmEntry {
                msg_type: msg.msg_type,
                payload: msg.payload.to_vec(),
                crc_ok: msg.crc_ok,
            })
        }
        Some(crate::vmm::wire::MsgType::TestResult) => {
            // →Teardown: the guest reported a test verdict; it is winding
            // down. Same bucket-and-advance shape as Exit.
            if msg.crc_ok {
                advance_stage(sinks, crate::monitor::LifecycleStage::Teardown);
            }
            Some(crate::vmm::wire::ShmEntry {
                msg_type: msg.msg_type,
                payload: msg.payload.to_vec(),
                crc_ok: msg.crc_ok,
            })
        }
        Some(other) if !other.is_coordinator_internal() => {
            // Every other typed verdict-bearing variant
            // (StepEnd, Crash, PayloadMetrics,
            // Profraw, WprofTrace, WprofTraceChunk,
            // Stdout, Stderr, SchedLog, ExecExit, Dmesg,
            // ProbeOutput) accumulates into the bucket verbatim. (ExecExit is listed for
            // completeness but is shell-mode-only -- sent only by
            // `cargo ktstr shell --exec` and consumed host-side by
            // `KtstrVm::run_interactive`, not the freeze coordinator;
            // the scheduler-test path never receives one, so it is
            // never actually bucketed here.) Stimulus, ScenarioEnd,
            // Lifecycle, Exit, and TestResult have their own typed arms
            // above (Stimulus decodes step_index into the host-side
            // mirror; ScenarioEnd/Exit/TestResult advance the lifecycle
            // ledger to Teardown; Lifecycle advances the ledger per its
            // 1-byte phase; all then bucket). StepEnd has NO dedicated arm
            // and buckets here: unlike Stimulus it publishes no new
            // step_index for the host mirror — its step_index equals the
            // StepStart frame that already set `current_step` for the
            // step, so re-storing it would be a no-op. SnapshotReply is host→guest
            // only and is filtered out by the
            // `is_coordinator_internal` guard above; a guest TX
            // frame stamped with that tag falls through to the
            // `Some(_)` arm below and is dropped silently. CRC-bad
            // entries still land here — the host-side consumers
            // filter on `crc_ok` per their own per-type contract.
            Some(crate::vmm::wire::ShmEntry {
                msg_type: msg.msg_type,
                payload: msg.payload.to_vec(),
                crc_ok: msg.crc_ok,
            })
        }
        Some(_) => {
            // Coordinator-internal variant with no inline side-effect
            // arm above (e.g. a future is_coordinator_internal entry).
            // Drop silently — by definition this variant should not
            // surface as a verdict entry, and any side effect must be
            // added here explicitly.
            None
        }
        None => {
            // Unknown msg_type — log once and drop. A future guest
            // variant the host does not know about would otherwise
            // produce a phantom verdict entry.
            tracing::warn!(
                msg_type = msg.msg_type,
                len = msg.payload.len(),
                crc_ok = msg.crc_ok,
                "freeze_coord: unknown MSG_TYPE_* on bulk port; dropping"
            );
            None
        }
    }
}

#[cfg(test)]
mod stage_tests {
    //! Live lifecycle-stage tracking: drive the production
    //! [`dispatch_bulk_message`] against synthetic guest frames with a
    //! wired [`crate::monitor::ProgressLedger`] and assert the coarse
    //! stage walks `Boot`→`Attach`→`Dispatch`→`Body`→`Teardown`, that
    //! epochs bump and CPU anchors snapshot on each advance, that the
    //! advance is monotone (no regress from a late/duplicate frame), that
    //! an attach-failure lifecycle frame does NOT advance, and that the
    //! Lifecycle arm still returns the verbatim bucket entry.
    use super::{BulkDispatchSinks, dispatch_bulk_message};
    use crate::monitor::{LifecycleStage, ProgressLedger};
    use crate::vmm::freeze_coord::state::SnapshotRequest;
    use crate::vmm::wire::{LifecyclePhase, MsgType, ShmEntry};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
    use std::time::Instant;
    use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};

    /// Owns every backing value a [`BulkDispatchSinks`] borrows PLUS a
    /// real [`ProgressLedger`], so a test can drive the production
    /// dispatch and then read the ledger's post-state.
    struct SinkState {
        ledger: ProgressLedger,
        kill: Arc<AtomicBool>,
        kill_evt: Arc<EventFd>,
        sys_rdy_evt: Option<Arc<EventFd>>,
        snapshot_requests_pending: Vec<SnapshotRequest>,
        kernel_op_requests_pending: Vec<crate::vmm::wire::KernelOpRequestPayload>,
        kern_phys_base: Arc<AtomicU64>,
        kern_phys_base_evt: EventFd,
        kern_virt_kaslr: Arc<AtomicU64>,
        kern_virt_kaslr_evt: EventFd,
        watchdog_pause_ns: AtomicU64,
        scenario_start_ns: AtomicU64,
        sched_swap_notify: AtomicBool,
        scenario_pause_cumulative_ns: AtomicU64,
        run_start: Instant,
        current_step: Arc<AtomicU16>,
        /// Watchdog-reset deadline atomic + provenance tag. Wired into
        /// `watchdog_reset` only when `wire_reset` is set (the default is
        /// `None`, matching the pre-C6 fixture so every stage test that
        /// does not opt in observes the original no-budget behavior).
        reset_ns: AtomicU64,
        reset_tag: std::sync::atomic::AtomicU8,
        workload_duration: std::time::Duration,
        wire_reset: bool,
    }

    impl SinkState {
        fn new() -> Self {
            Self {
                ledger: ProgressLedger::default(),
                kill: Arc::new(AtomicBool::new(false)),
                kill_evt: Arc::new(EventFd::new(EFD_NONBLOCK).expect("kill eventfd")),
                sys_rdy_evt: Some(Arc::new(
                    EventFd::new(EFD_NONBLOCK).expect("sys_rdy eventfd"),
                )),
                snapshot_requests_pending: Vec::new(),
                kernel_op_requests_pending: Vec::new(),
                kern_phys_base: Arc::new(AtomicU64::new(0)),
                kern_phys_base_evt: EventFd::new(EFD_NONBLOCK).expect("phys_base eventfd"),
                kern_virt_kaslr: Arc::new(AtomicU64::new(0)),
                kern_virt_kaslr_evt: EventFd::new(EFD_NONBLOCK).expect("virt_kaslr eventfd"),
                watchdog_pause_ns: AtomicU64::new(0),
                scenario_start_ns: AtomicU64::new(0),
                sched_swap_notify: AtomicBool::new(false),
                scenario_pause_cumulative_ns: AtomicU64::new(0),
                run_start: Instant::now(),
                current_step: Arc::new(AtomicU16::new(0)),
                reset_ns: AtomicU64::new(0),
                reset_tag: std::sync::atomic::AtomicU8::new(0),
                workload_duration: std::time::Duration::from_secs(60),
                wire_reset: false,
            }
        }

        /// Opt this fixture into a wired `watchdog_reset` budget so the
        /// dispatch arms that arm the deadline (ScenarioStart, ScenarioEnd,
        /// the SchedulerAttached Lifecycle arm) exercise their reset stores.
        fn arm_reset(&mut self) {
            self.wire_reset = true;
        }

        fn sinks(&mut self) -> BulkDispatchSinks<'_> {
            BulkDispatchSinks {
                kill: &self.kill,
                kill_evt: &self.kill_evt,
                run_is_wprof: false,
                sys_rdy_evt: &mut self.sys_rdy_evt,
                snapshot_requests_pending: &mut self.snapshot_requests_pending,
                kernel_op_requests_pending: &mut self.kernel_op_requests_pending,
                kern_phys_base: &self.kern_phys_base,
                kern_phys_base_evt: &self.kern_phys_base_evt,
                kern_virt_kaslr: &self.kern_virt_kaslr,
                kern_virt_kaslr_evt: &self.kern_virt_kaslr_evt,
                kernel_text_link_kva: 0,
                watchdog_reset: if self.wire_reset {
                    Some((
                        &self.reset_ns,
                        self.workload_duration,
                        self.run_start,
                        &self.reset_tag,
                    ))
                } else {
                    None
                },
                watchdog_pause_ns: &self.watchdog_pause_ns,
                scenario_start_ns: &self.scenario_start_ns,
                sched_swap_notify: &self.sched_swap_notify,
                scenario_pause_cumulative_ns: &self.scenario_pause_cumulative_ns,
                run_start: self.run_start,
                current_step: &self.current_step,
                progress_ledger: Some(&self.ledger),
            }
        }

        fn dispatch(&mut self, msg: &crate::vmm::bulk::BulkMessage) -> Option<ShmEntry> {
            let mut sinks = self.sinks();
            dispatch_bulk_message(msg, &mut sinks)
        }

        fn phase(&self) -> u8 {
            self.ledger.phase.load(Ordering::Relaxed)
        }
        fn phase_epoch(&self) -> u32 {
            self.ledger.phase_epoch.load(Ordering::Acquire)
        }
        fn progress_epoch(&self) -> u64 {
            self.ledger.progress_epoch.load(Ordering::Acquire)
        }
        fn reset_ns(&self) -> u64 {
            self.reset_ns.load(Ordering::Acquire)
        }
        fn reset_tag(&self) -> u8 {
            self.reset_tag.load(Ordering::Relaxed)
        }
    }

    /// Build a synthetic assembled `BulkMessage` — the shape
    /// `dispatch_bulk_message` receives after `HostAssembler::feed`.
    fn frame(msg_type: u32, payload: Vec<u8>, crc_ok: bool) -> crate::vmm::bulk::BulkMessage {
        crate::vmm::bulk::BulkMessage {
            msg_type,
            payload: payload.into(),
            crc_ok,
        }
    }

    fn lifecycle(phase: LifecyclePhase, crc_ok: bool) -> crate::vmm::bulk::BulkMessage {
        frame(
            MsgType::Lifecycle.wire_value(),
            vec![phase.wire_value()],
            crc_ok,
        )
    }

    /// Test cell path: Boot →(SysRdy)→ Attach →(PayloadStarting)→
    /// Dispatch →(ScenarioStart)→ Body →(Exit)→ Teardown. Each forward
    /// step bumps both epochs by exactly one. (The Tier-1 CPU anchor is
    /// monitor-side now, so no `cpu_ns_at_phase` snapshot is asserted here.)
    #[test]
    fn test_cell_stage_walk() {
        let mut s = SinkState::new();
        assert_eq!(s.phase(), LifecycleStage::Boot as u8);
        assert_eq!(s.phase_epoch(), 0);
        assert_eq!(s.progress_epoch(), 0);

        s.dispatch(&frame(MsgType::SysRdy.wire_value(), Vec::new(), true));
        assert_eq!(s.phase(), LifecycleStage::Attach as u8);
        assert_eq!(s.phase_epoch(), 1);
        assert_eq!(s.progress_epoch(), 1);

        s.dispatch(&lifecycle(LifecyclePhase::PayloadStarting, true));
        assert_eq!(s.phase(), LifecycleStage::Dispatch as u8);
        assert_eq!(s.phase_epoch(), 2);
        assert_eq!(s.progress_epoch(), 2);

        s.dispatch(&frame(
            MsgType::ScenarioStart.wire_value(),
            Vec::new(),
            true,
        ));
        assert_eq!(s.phase(), LifecycleStage::Body as u8);
        assert_eq!(s.phase_epoch(), 3);
        assert_eq!(s.progress_epoch(), 3);

        s.dispatch(&frame(MsgType::Exit.wire_value(), Vec::new(), true));
        assert_eq!(s.phase(), LifecycleStage::Teardown as u8);
        assert_eq!(s.phase_epoch(), 4);
        assert_eq!(s.progress_epoch(), 4);
    }

    /// Verifier cell path reaches Body via `WorkloadDispatched` (not
    /// ScenarioStart), and ScenarioEnd drives Teardown.
    #[test]
    fn verifier_cell_reaches_body_via_workload_dispatched() {
        let mut s = SinkState::new();
        s.dispatch(&frame(MsgType::SysRdy.wire_value(), Vec::new(), true));
        s.dispatch(&lifecycle(LifecyclePhase::PayloadStarting, true));
        assert_eq!(s.phase(), LifecycleStage::Dispatch as u8);
        s.dispatch(&lifecycle(LifecyclePhase::WorkloadDispatched, true));
        assert_eq!(s.phase(), LifecycleStage::Body as u8);
        s.dispatch(&frame(MsgType::ScenarioEnd.wire_value(), Vec::new(), true));
        assert_eq!(s.phase(), LifecycleStage::Teardown as u8);
    }

    /// Monotone: after Exit drove Teardown, a stray (late/duplicate)
    /// ScenarioStart must NOT regress the stage back to Body, and must
    /// bump neither epoch — a duplicate is not fresh progress.
    #[test]
    fn no_regress_from_stray_scenario_start_after_exit() {
        let mut s = SinkState::new();
        s.dispatch(&frame(MsgType::Exit.wire_value(), Vec::new(), true));
        assert_eq!(s.phase(), LifecycleStage::Teardown as u8);
        let pe = s.phase_epoch();
        let pr = s.progress_epoch();

        s.dispatch(&frame(
            MsgType::ScenarioStart.wire_value(),
            Vec::new(),
            true,
        ));
        assert_eq!(
            s.phase(),
            LifecycleStage::Teardown as u8,
            "stray ScenarioStart regressed the stage"
        );
        assert_eq!(s.phase_epoch(), pe, "no-op advance bumped phase_epoch");
        assert_eq!(
            s.progress_epoch(),
            pr,
            "no-op advance bumped progress_epoch"
        );
    }

    /// InitStarted is a boot heartbeat: it records PROGRESS but does NOT
    /// advance the stage off Boot or bump `phase_epoch`.
    #[test]
    fn init_started_is_progress_without_stage_advance() {
        let mut s = SinkState::new();
        s.dispatch(&lifecycle(LifecyclePhase::InitStarted, true));
        assert_eq!(s.phase(), LifecycleStage::Boot as u8);
        assert_eq!(s.phase_epoch(), 0, "InitStarted advanced the stage");
        assert_eq!(
            s.progress_epoch(),
            1,
            "InitStarted was not counted as progress"
        );
    }

    /// An attach-failure lifecycle frame (`SchedulerDied` /
    /// `SchedulerNotAttached`) must NOT advance the stage or record
    /// progress — the guest proceeds to Phase 5 regardless, and a failure
    /// is not forward progress.
    #[test]
    fn scheduler_failure_does_not_advance() {
        let mut s = SinkState::new();
        // Get to Dispatch first so a regression would be observable.
        s.dispatch(&frame(MsgType::SysRdy.wire_value(), Vec::new(), true));
        s.dispatch(&lifecycle(LifecyclePhase::PayloadStarting, true));
        let pe = s.phase_epoch();
        let pr = s.progress_epoch();

        s.dispatch(&lifecycle(LifecyclePhase::SchedulerDied, true));
        assert_eq!(s.phase(), LifecycleStage::Dispatch as u8);
        assert_eq!(s.phase_epoch(), pe);
        assert_eq!(s.progress_epoch(), pr);

        s.dispatch(&lifecycle(LifecyclePhase::SchedulerNotAttached, true));
        assert_eq!(s.phase(), LifecycleStage::Dispatch as u8);
        assert_eq!(s.phase_epoch(), pe);
        assert_eq!(s.progress_epoch(), pr);
    }

    /// `SchedulerAttached` records PROGRESS but does NOT advance the coarse
    /// stage (it lands during Attach; the stage advances come from
    /// PayloadStarting / ScenarioStart / WorkloadDispatched), and it does
    /// NOT bump `phase_epoch`.
    #[test]
    fn scheduler_attached_is_progress_without_stage_advance() {
        let mut s = SinkState::new();
        // SysRdy first so the stage is Attach when the confirm lands.
        s.dispatch(&frame(MsgType::SysRdy.wire_value(), Vec::new(), true));
        assert_eq!(s.phase(), LifecycleStage::Attach as u8);
        let pe = s.phase_epoch();
        let pr = s.progress_epoch();

        s.dispatch(&lifecycle(LifecyclePhase::SchedulerAttached, true));
        assert_eq!(
            s.phase(),
            LifecycleStage::Attach as u8,
            "SchedulerAttached advanced the stage"
        );
        assert_eq!(s.phase_epoch(), pe, "SchedulerAttached bumped phase_epoch");
        assert_eq!(
            s.progress_epoch(),
            pr + 1,
            "SchedulerAttached was not counted as progress"
        );
    }

    /// `SchedulerAttached` arms the watchdog reset deadline to
    /// now + workload_duration and stamps `GuestAttachConfirm` provenance —
    /// the evented counterpart to the monitor's polled scx_root latch. A
    /// CRC-bad confirm frame must NOT arm it (a torn frame must not forge a
    /// deadline).
    #[test]
    fn scheduler_attached_arms_watchdog_reset_with_confirm_tag() {
        let mut s = SinkState::new();
        s.arm_reset();
        assert_eq!(s.reset_ns(), 0, "precondition: reset not yet armed");

        // CRC-bad confirm: no arm.
        s.dispatch(&lifecycle(LifecyclePhase::SchedulerAttached, false));
        assert_eq!(
            s.reset_ns(),
            0,
            "a torn SchedulerAttached forged a deadline"
        );

        // CRC-ok confirm: arm + stamp.
        s.dispatch(&lifecycle(LifecyclePhase::SchedulerAttached, true));
        assert_ne!(
            s.reset_ns(),
            0,
            "confirmed attach must arm the reset deadline"
        );
        assert_eq!(
            s.reset_tag(),
            crate::vmm::freeze_coord::WatchdogResetTag::GuestAttachConfirm as u8,
            "the confirm arm must stamp GuestAttachConfirm provenance",
        );
    }

    /// A CRC-bad lifecycle frame must NOT forge stage progress, but STILL
    /// buckets verbatim (the host-side verdict scan filters on crc_ok).
    #[test]
    fn crc_bad_lifecycle_does_not_advance_but_buckets() {
        let mut s = SinkState::new();
        let entry = s.dispatch(&lifecycle(LifecyclePhase::PayloadStarting, false));
        assert_eq!(
            s.phase(),
            LifecycleStage::Boot as u8,
            "torn frame forged a stage advance"
        );
        assert_eq!(s.phase_epoch(), 0);
        let entry = entry.expect("Lifecycle must still bucket even CRC-bad");
        assert_eq!(entry.msg_type, MsgType::Lifecycle.wire_value());
        assert!(!entry.crc_ok);
    }

    /// The Lifecycle arm both acts AND returns the verbatim ShmEntry so
    /// the post-hoc verdict scan sees the frame (bucketing preserved).
    #[test]
    fn lifecycle_arm_returns_verbatim_entry() {
        let mut s = SinkState::new();
        let payload = vec![LifecyclePhase::WorkloadDispatched.wire_value()];
        let entry = s
            .dispatch(&frame(
                MsgType::Lifecycle.wire_value(),
                payload.clone(),
                true,
            ))
            .expect("Lifecycle must bucket");
        assert_eq!(entry.msg_type, MsgType::Lifecycle.wire_value());
        assert_eq!(entry.payload, payload, "bucketed payload not verbatim");
        assert!(entry.crc_ok);
        // And the act half still fired.
        assert_eq!(s.phase(), LifecycleStage::Body as u8);
    }

    /// Exit and TestResult both drive Teardown and bucket verbatim.
    #[test]
    fn exit_and_test_result_drive_teardown_and_bucket() {
        for wire in [MsgType::Exit.wire_value(), MsgType::TestResult.wire_value()] {
            let mut s = SinkState::new();
            let entry = s
                .dispatch(&frame(wire, vec![1, 2, 3], true))
                .expect("must bucket");
            assert_eq!(entry.msg_type, wire);
            assert_eq!(entry.payload, vec![1, 2, 3]);
            assert_eq!(s.phase(), LifecycleStage::Teardown as u8);
        }
    }
}
