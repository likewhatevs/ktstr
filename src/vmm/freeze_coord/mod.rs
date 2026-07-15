//! Run-loop orchestration for `KtstrVm`: spawning AP vCPU threads,
//! the freeze coordinator, the BPF map writer, the BSP loop, and
//! result collection. This is the kernel-boundary heart of the VMM
//! runtime — every method here runs after [`super::setup`] hands the
//! configured [`KtstrKvm`](super::kvm::KtstrKvm) over and before the
//! VM exits.
//!
//! Reopens [`impl KtstrVm`](super::KtstrVm) so the canonical struct
//! definition stays in [`super`].

pub(crate) mod evented_wait;
mod gate;
mod kernel_op_dispatch;

use anyhow::{Context, Result};
use kvm_ioctls::VcpuExit;
use std::io::{Read, Seek};
use std::os::fd::AsRawFd;
use std::os::unix::thread::JoinHandleExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU16, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use vm_memory::{GuestAddress, GuestMemory};
use vmm_sys_util::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};
use vmm_sys_util::timerfd::TimerFd;

use crate::monitor;
use crate::sync::MutexExt;

use super::exit_dispatch::{self, ExitAction, classify_exit, vcpu_run_loop_unified};
use super::host_comms::BulkDrainResult;
use super::pi_mutex::PiMutex;
use super::result::{HostVcpuSchedstat, VmResult, VmRunState};
use super::vcpu::{
    ApFreezeHandles, BpfMapWriteParams, ImmediateExitHandle, VcpuThread, WatchpointArm,
    duration_to_jiffies, load_probe_bss_offset, open_vcpu_perf_capture, pin_current_thread,
    register_vcpu_signal_handler, self_arm_watchpoint, set_rt_priority, set_thread_cpumask,
    vcpu_signal,
};
use super::vmlinux::{cached_vmlinux_bytes, find_vmlinux};
use super::{
    KtstrVm, console, host_comms, vcpu_panic, virtio_blk, virtio_console, virtio_net, wire,
};
// MSI-X routing is x86_64-only (aarch64 uses GICv3 with no PCI MSI-X); its only
// users are the `#[cfg(target_arch = "x86_64")]` PCI run-loop paths below.
#[cfg(target_arch = "x86_64")]
use super::virtio_msix;

#[cfg(target_arch = "aarch64")]
use super::aarch64::kvm;
#[cfg(target_arch = "x86_64")]
use super::x86_64::kvm;

// `DRAM_BASE` is defined in `super` and used here for guest-memory
// host-address resolution. The const is arch-gated; the import
// follows the same gating implicitly via where it is consumed.
use super::DRAM_BASE;

mod dispatch;
pub(crate) mod latency_verdict;
mod lazy_init;
mod snapshot;
mod state;
// `pub(crate)` so the monitor can drive `CpuTrickleTracker` /
// `trickle_floor_for_currency`: the width-sound trickle-stall verdict is
// computed monitor-side (it owns the per-vCPU CPU data the busiest-vCPU
// windowed currency needs) and published to the ledger for the watchdog.
pub(crate) mod watchdog_step;
mod watchpoint;

/// Guest scx `watchdog_timeout` multiplier for COVERAGE builds only
/// (cargo-llvm-cov compiles with `--cfg coverage`; plain builds get 1 =
/// no-op, byte-identical). The in-guest test binary and fixture
/// schedulers are llvm-cov-instrumented (~2x slower per measured host
/// cost), and the coverage CI legs' deepened oversubscription on
/// colocated runners starves single vCPUs long enough that the tight 5 s
/// default evicts churn-heavy fixture schedulers the uninstrumented legs
/// run green on the same hosts. 3x (5 s → 15 s) matches the margin the
/// heaviest fixtures already declare explicitly and stays a fraction of
/// the kernel's SCX_WATCHDOG_MAX_TIMEOUT (30 s) for the default.
/// Deliberately compile-time: the tight default keeps its full
/// stall-detection value in every non-coverage build, so a slow-stall
/// bug can never hide behind this margin outside coverage runs.
#[cfg(coverage)]
const GUEST_SCX_WATCHDOG_COVERAGE_SCALE: u64 = 3;
#[cfg(not(coverage))]
const GUEST_SCX_WATCHDOG_COVERAGE_SCALE: u64 = 1;

#[cfg(test)]
mod bss_tests;

#[cfg(test)]
mod early_snapshot_guard_tests;

use self::dispatch::{BulkDispatchSinks, dispatch_bulk_message};
#[allow(unused_imports)]
use self::lazy_init::{
    try_init_owned_accessor_with_hint, try_init_owned_prog_accessor_with_hint,
    try_init_prog_per_cpu_offsets,
};
#[allow(unused_imports)]
use self::snapshot::{
    VmlinuxSymbolCache, arm_user_watchpoint, decode_snapshot_request, frame_kernel_op_reply,
    frame_snapshot_reply, poll_eventfd_until_ready_or_timeout, snapshot_tagged_path,
};
use self::state::{
    BspExitReason, FREEZE_RENDEZVOUS_TIMEOUT, FreezeState, SnapshotRequest,
    compute_periodic_boundaries_ns, periodic_accessor_current, periodic_tag,
    resolve_periodic_window,
};
use self::watchpoint::{WatchpointPublishResult, republish_watchpoint_on_rebind};

#[cfg(target_arch = "x86_64")]
fn warn_kvm_clock_failure(ioctl_phase: &'static str, e: &std::io::Error) {
    // KVM_GET_CLOCK / KVM_SET_CLOCK do not gate on SEV-ES / SEV-SNP /
    // TDX in arch/x86/kvm/x86.c (kvm_vm_ioctl_get_clock /
    // kvm_vm_ioctl_set_clock take no confidential-compute side
    // path). Realistic failures are EFAULT (bad userspace pointer)
    // and EINVAL (invalid flags on SET); EBADF would indicate the
    // VM fd was closed underneath us (real bug). On any of these,
    // log + fall through — the dump completes without the
    // save/restore and the guest sees freeze duration as elapsed
    // kvm_clock until the next pvclock update. If this fires
    // repeatedly in practice, it is a real bug worth filing.
    let when = if ioctl_phase == "GET_CLOCK" {
        "entry"
    } else {
        "exit"
    };
    tracing::warn!(
        error = %e,
        ioctl = ioctl_phase,
        when = when,
        "freeze-coord: KVM_{ioctl_phase} failed at freeze {when}; guest \
         will see freeze duration as elapsed kvm_clock until the next \
         pvclock update. Likely causes: EFAULT (bad pointer), EINVAL \
         (invalid flags on SET), or EBADF (fd closed). File an issue \
         with the error + kernel version if this fires repeatedly."
    );
}

/// Bounded grace a wprof run's error-exit teardown waits for the guest's
/// LATE trace ship (guest Phase 5, `guest_comms::send_wprof_trace` over the
/// bulk port) before killing the VM. The coordinator's error-exit arms set
/// `wprof_ship_deadline = Instant::now() + WPROF_SHIP_GRACE`; it is also
/// added to the primary VM's host watchdog budget in
/// [`crate::test_support::runtime::vm_timeout_from_entry`] so a late crash
/// still fits its full ship window inside the deadline (rather than the
/// watchdog pre-empting the grace and synthesizing a spurious timeout).
pub(crate) const WPROF_SHIP_GRACE: Duration = Duration::from_secs(30);

/// Force run teardown from the coordinator: set the run-level kill flag
/// and kick its eventfd so the BSP run loop (`kill.load`) and the
/// `'coord:` loop guard (`freeze_coord_kill.load`) both observe the edge
/// on their next wake.
///
/// The `write` Result is intentionally dropped: the only failure modes
/// for an eventfd write are EAGAIN (counter saturated at `u64::MAX-1`, so
/// the add would overflow — the wake is already signaled past saturation)
/// and EBADF (fd torn down — the coord is already exiting). Both mean
/// "kill is already observed or about to be"; no recovery is meaningful.
fn trigger_freeze_coord_kill(kill: &AtomicBool, kill_evt: &EventFd) {
    kill.store(true, Ordering::Release);
    let _ = kill_evt.write(1);
}

/// True iff `msg` is the guest's wprof trace ship — a crc-valid,
/// non-empty `MsgType::WprofTrace` frame. The coordinator's error-exit
/// grace promotes the teardown the instant this returns true for a
/// drained frame. A torn (`!crc_ok`), empty, or wrong-type frame is NOT
/// the ship and must not terminate the grace early (notably a SchedExit,
/// which the SchedExit dispatch arm declines to promote on a wprof run —
/// see the `run_is_wprof` sink field — and which instead arms the grace
/// via `is_sched_exit_frame` in the coord loop).
fn is_wprof_ship_frame(msg: &crate::vmm::bulk::BulkMessage) -> bool {
    msg.crc_ok && !msg.payload.is_empty() && msg.msg_type == crate::vmm::wire::MSG_TYPE_WPROF_TRACE
}

/// True iff an armed wprof-ship grace deadline has expired at `now`. The
/// coordinator's per-iteration backstop kills the VM when this returns
/// true — the bounded fallback if the guest wedges before shipping its
/// trace. `None` (no grace armed) never kills; the boundary is inclusive
/// (`now >= deadline`).
fn wprof_grace_should_kill(deadline: Option<Instant>, now: Instant) -> bool {
    deadline.is_some_and(|dl| now >= dl)
}

/// True iff `msg` is a crc-valid guest SCHED_EXIT frame. On a wprof run
/// the coordinator arms the ship grace when it drains one of these
/// (rather than promoting the kill in `dispatch_bulk_message`): a
/// self-crash makes the guest sched-exit monitor send SCHED_EXIT during
/// Phase 5, and treating it as a grace-arm (not a kill) holds the VM open
/// for the guest's late wprof ship even if the error-exit watchpoint dump
/// has not armed the grace yet (e.g. watchpoint unavailable). CRC-bad
/// frames do not arm — a torn/hostile frame must not force the grace.
fn is_sched_exit_frame(msg: &crate::vmm::bulk::BulkMessage) -> bool {
    msg.crc_ok && msg.msg_type == crate::vmm::wire::MSG_TYPE_SCHED_EXIT
}

/// True iff `msg` is a crc-valid guest stdout frame carrying the
/// probe-payload end delimiter. On an auto-repro (dual_snapshot) wprof
/// run the guest ships its probe report (Phase 6b,
/// `collect_and_print_probe_data` -> `println!(PROBE_OUTPUT_END)` over
/// the bulk-port stdout forwarder) AFTER the Phase-5 wprof trace, so the
/// coordinator gates the ship-grace kill on this frame instead of the
/// bare `WprofTrace` — killing on the trace would preempt Phase 6b and
/// drop the `=== AUTO-PROBE: ... ===` report. A frame torn mid-marker
/// (delimiter split across two chunks) simply does not match; the
/// `WPROF_SHIP_GRACE` backstop then bounds teardown, so a missed match
/// degrades to the deadline rather than a hang.
fn is_probe_output_end_frame(msg: &crate::vmm::bulk::BulkMessage) -> bool {
    let needle = crate::test_support::PROBE_OUTPUT_END.as_bytes();
    msg.crc_ok
        && msg.msg_type == crate::vmm::wire::MSG_TYPE_STDOUT
        && msg.payload.len() >= needle.len()
        && msg.payload.windows(needle.len()).any(|w| w == needle)
}

/// Provenance tag for the shared `watchdog_reset_ns` deadline atomic.
///
/// FOUR independent subsystems store into that single `AtomicU64`
/// (the monitor's scx_root attach latch, the guest scenario-dispatch
/// arms, freeze-cycle extension, and wprof-ship grace), so a watchdog
/// dump keyed only on "was the deadline reset at all" cannot say WHICH
/// one armed it. Worse, the scenario-start arm fires even when the
/// scheduler already died, so the old `reset_by_scheduler_attach=`
/// key mislabeled dead schedulers as attached during triage. This
/// parallel `AtomicU8` records the last writer so the dump renders the
/// true provenance.
///
/// A SEPARATE atomic (not bit-packed into the u64) because the freeze
/// and grace writers do load/compare arithmetic on the raw ns value —
/// a packed tag would corrupt that. Each writer stores its ns and its
/// tag back-to-back under Relaxed ordering, so a reader can transiently
/// observe a fresh ns with a stale tag (or vice-versa). That race is
/// benign: the tag feeds human-facing diagnostics only, never a control
/// decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum WatchdogResetTag {
    /// No writer has armed the reset deadline yet (the initial value).
    Unset = 0,
    /// The host monitor observed `*scx_root` transition null → non-null
    /// (a scheduler attached) and reset the deadline to attach-moment +
    /// workload duration. See `monitor::reader::monitor_loop`'s
    /// watchdog-reset arm.
    ScxRootLatch = 1,
    /// The guest's scenario-dispatch machinery armed/extended the reset:
    /// a `ScenarioStart` (start the workload clock at scenario start),
    /// `ScenarioResume` (add the paused span back), or `ScenarioEnd`
    /// (re-anchor to now + duration) frame — all three carry this one
    /// tag since they are the same subsystem. NOTE: the `ScenarioStart`
    /// arm fires even when the scheduler already DIED, so this tag does
    /// NOT imply a live scheduler — the exact ambiguity that misled the
    /// old `reset_by_scheduler_attach` key.
    ScenarioStart = 2,
    /// A freeze cycle extended the reset by the freeze duration so the
    /// frozen span does not eat the workload budget
    /// (`extend_watchdog_for_freeze`).
    FreezeExtend = 3,
    /// A wprof-ship grace extended the reset to cover the ship window
    /// (`extend_watchdog_reset_for_grace`).
    WprofGrace = 4,
    /// The guest emitted a `LifecyclePhase::SchedulerAttached` frame — a
    /// REAL sched_ext scheduler bound with a live child — and the dispatch
    /// thread re-armed the reset to attach-moment + workload duration. The
    /// evented, authoritative counterpart to [`Self::ScxRootLatch`]: unlike
    /// the polled `*scx_root` latch, this frame is only sent on a confirmed
    /// live-scheduler attach (never for an EEVDF/no-scheduler run or a dead
    /// scheduler), so — unlike `ScenarioStart` — it DOES imply a live
    /// scheduler. See `freeze_coord::dispatch`'s Lifecycle arm.
    GuestAttachConfirm = 5,
}

impl WatchdogResetTag {
    /// Decode a raw atomic byte. Any unrecognized value (only reachable
    /// via a never-written slot) maps to [`WatchdogResetTag::Unset`].
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::ScxRootLatch,
            2 => Self::ScenarioStart,
            3 => Self::FreezeExtend,
            4 => Self::WprofGrace,
            5 => Self::GuestAttachConfirm,
            _ => Self::Unset,
        }
    }

    /// Human-facing token for the watchdog dump's `reset_armed_by=` key.
    fn render(self) -> &'static str {
        match self {
            Self::Unset => "none",
            Self::ScxRootLatch => "scx-root-attach-latch",
            Self::ScenarioStart => "scenario-start",
            Self::FreezeExtend => "freeze-extend",
            Self::WprofGrace => "wprof-grace",
            Self::GuestAttachConfirm => "guest-attach-confirm",
        }
    }
}

/// Which rule fired the watchdog, recorded so the deadline-expired dump
/// names the true cause. Mirrors [`WatchdogResetTag`]'s shape (a `#[repr(u8)]`
/// tag with `from_u8`/`render`). The watchdog thread owns the sole
/// [`std::sync::atomic::AtomicU8`] holding this: it is stored on the fire
/// path and loaded back for the dump within the same thread, so no
/// cross-thread ordering is required — the atomic keeps the store/render
/// idiom uniform with the reset tag and leaves room for a future shared
/// reader (e.g. the verdict) without reshaping the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum KillReasonTag {
    /// No kill recorded yet (the initial value).
    Unset = 0,
    /// Tier-1: guest CPU burned past the phase budget with no progress —
    /// a spinning wedge (`watchdog_step::KillDecision::Tier1CpuBudget`).
    Tier1Cpu = 1,
    /// Tier-2: no progress past the phase wall backstop with no runnable
    /// demand — a silent idle wedge (`KillDecision::Tier2IdleWedge`).
    Tier2Idle = 2,
    /// Tier-3: the guest-derived hard deadline expired AND the deadman was
    /// not deferred — the monitor was dead (`!monitor_live`) or the cell
    /// was inert (CPU trickle-stalled AND no milestone within
    /// [`watchdog_step::TIER3_PROGRESS_GRACE_NS`]). NOT an unconditional
    /// wall clock: a starved-but-alive cell (still accruing CPU) outlives
    /// the wall deadline by design and is bounded by the harness/operator
    /// instead; the deadman only kills when the machinery is dead or the
    /// cell is wedged.
    Tier3Deadman = 3,
    /// An AP set the kill flag (a panic-driven kill), not a watchdog
    /// timeout. Distinct so the dump does not mislabel it as an expiry.
    ApKill = 4,
}

impl KillReasonTag {
    /// Decode a raw atomic byte. Any unrecognized value maps to
    /// [`KillReasonTag::Unset`].
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Tier1Cpu,
            2 => Self::Tier2Idle,
            3 => Self::Tier3Deadman,
            4 => Self::ApKill,
            _ => Self::Unset,
        }
    }

    /// Human-facing token for the watchdog dump's `kill_reason=` key.
    fn render(self) -> &'static str {
        match self {
            Self::Unset => "none",
            Self::Tier1Cpu => "tier1-cpu-budget",
            Self::Tier2Idle => "tier2-idle-wedge",
            Self::Tier3Deadman => "tier3-deadman-deadline",
            Self::ApKill => "ap-kill",
        }
    }
}

/// Decode the watchdog's raw kill-reason byte into the public
/// [`crate::vmm::WatchdogKillReason`] mirror for `VmResult`. Kept here
/// (not in `result.rs`) so the byte layout stays owned by
/// [`KillReasonTag`]; `Unset` (and any unknown byte, via
/// `KillReasonTag::from_u8`'s conservative mapping) decodes to `None` —
/// "no watchdog kill recorded".
fn decode_watchdog_kill_reason(raw: u8) -> Option<crate::vmm::WatchdogKillReason> {
    use crate::vmm::WatchdogKillReason as Pub;
    match KillReasonTag::from_u8(raw) {
        KillReasonTag::Unset => None,
        KillReasonTag::Tier1Cpu => Some(Pub::Tier1CpuBudget),
        KillReasonTag::Tier2Idle => Some(Pub::Tier2IdleWedge),
        KillReasonTag::Tier3Deadman => Some(Pub::Tier3Deadman),
        KillReasonTag::ApKill => Some(Pub::ApKill),
    }
}

/// Decode the ledger's raw lifecycle-stage byte into the public
/// [`crate::vmm::GuestLifecyclePhase`] mirror for `VmResult`. Routes
/// through `LifecycleStage::from_u8` so the conservative unknown→Boot
/// mapping stays single-sourced.
fn decode_guest_phase(raw: u8) -> crate::vmm::GuestLifecyclePhase {
    use crate::vmm::GuestLifecyclePhase as Pub;
    match monitor::LifecycleStage::from_u8(raw) {
        monitor::LifecycleStage::Boot => Pub::Boot,
        monitor::LifecycleStage::Attach => Pub::Attach,
        monitor::LifecycleStage::Dispatch => Pub::Dispatch,
        monitor::LifecycleStage::Body => Pub::Body,
        monitor::LifecycleStage::Teardown => Pub::Teardown,
    }
}

/// Decode the BPF-map-write injection delivery byte for
/// [`crate::vmm::VmResult::bpf_map_writes_delivered`]: 0 = no writes
/// configured (`None`), 2 = delivered-and-guest-signalled
/// (`Some(true)`), anything else = configured but never delivered
/// (`Some(false)` — conservative for unknown bytes: an undelivered
/// injection must never read as delivered).
fn decode_bpf_map_write_delivery(raw: u8) -> Option<bool> {
    match raw {
        0 => None,
        2 => Some(true),
        _ => Some(false),
    }
}

/// Extend the watchdog's reset deadline so it accommodates an armed
/// wprof-ship grace. The watchdog fires at
/// `effective_deadline = reset_deadline.max(hard_deadline)` where
/// `reset_deadline = run_start + reset_ns`. `hard_deadline` already
/// includes `WPROF_SHIP_GRACE` (via `vm_timeout_from_entry`), but a
/// scheduler-attach / freeze reset can push `reset_deadline` PAST
/// `hard_deadline`, masking the grace. Bumping `reset_ns` to at least
/// `now - run_start + grace` guarantees the watchdog does not fire before
/// the grace's own deadline, so a late crash's ship window is not
/// pre-empted (and no spurious `timed_out` is synthesized). `max`-style:
/// a larger existing reset is never shrunk.
fn extend_watchdog_reset_for_grace(
    reset_ns: &std::sync::atomic::AtomicU64,
    reset_tag: &std::sync::atomic::AtomicU8,
    run_start: Instant,
    grace: Duration,
) {
    let target = run_start.elapsed().saturating_add(grace);
    let encoded = u64::try_from(target.as_nanos()).unwrap_or(u64::MAX).max(1);
    if encoded > reset_ns.load(Ordering::Acquire) {
        reset_ns.store(encoded, Ordering::Release);
        // Stamp provenance whenever we win the max-style store, so the
        // watchdog dump attributes this deadline to the grace and not
        // to a stale earlier writer. Relaxed is fine — diagnostic only.
        reset_tag.store(WatchdogResetTag::WprofGrace as u8, Ordering::Relaxed);
    }
}

/// Arm the wprof-ship grace: extend the watchdog reset deadline to cover
/// it (so the internal watchdog cannot pre-empt the grace — see
/// [`extend_watchdog_reset_for_grace`]) and return the grace deadline.
/// Called at every grace-arm site so the deadline and the watchdog
/// extension always move together.
fn arm_wprof_grace(
    run_start: Instant,
    reset_ns: &std::sync::atomic::AtomicU64,
    reset_tag: &std::sync::atomic::AtomicU8,
    grace: Duration,
) -> Instant {
    extend_watchdog_reset_for_grace(reset_ns, reset_tag, run_start, grace);
    Instant::now() + grace
}

#[cfg(test)]
mod watchdog_reset_tag_tests {
    use super::{KillReasonTag, WatchdogResetTag, decode_guest_phase, decode_watchdog_kill_reason};
    use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    /// The VmResult plumbing decodes each fired tier's raw byte into the
    /// public mirror, and the Unset/unknown bytes into `None` — so a clean
    /// run (watchdog never fired, latch still 0) surfaces
    /// `watchdog_kill_reason: None` and a fired tier surfaces its variant.
    #[test]
    fn watchdog_kill_reason_decode_per_tier_and_none_on_clean() {
        use crate::vmm::WatchdogKillReason as Pub;
        // Clean run: the latch stays at the initial Unset byte.
        assert_eq!(
            decode_watchdog_kill_reason(KillReasonTag::Unset as u8),
            None
        );
        // Unknown byte (future variant / torn write): conservative None.
        assert_eq!(decode_watchdog_kill_reason(200), None);
        for (tag, expected) in [
            (KillReasonTag::Tier1Cpu, Pub::Tier1CpuBudget),
            (KillReasonTag::Tier2Idle, Pub::Tier2IdleWedge),
            (KillReasonTag::Tier3Deadman, Pub::Tier3Deadman),
            (KillReasonTag::ApKill, Pub::ApKill),
        ] {
            assert_eq!(decode_watchdog_kill_reason(tag as u8), Some(expected));
        }
    }

    /// Phase decode: each ledger stage byte maps to its public mirror;
    /// unknown bytes fall back to `Boot` (the ledger's own conservative
    /// mapping). The public enum's `Ord` follows boot progress so
    /// fixtures can ask "did the guest reach the wedge phase".
    /// Injection-delivery decode: 0 (no writes) → None; 2 (delivered)
    /// → Some(true); 1 and any unknown byte → Some(false) — an
    /// undelivered injection must never decode as delivered.
    #[test]
    fn bpf_map_write_delivery_decode() {
        assert_eq!(super::decode_bpf_map_write_delivery(0), None);
        assert_eq!(super::decode_bpf_map_write_delivery(1), Some(false));
        assert_eq!(super::decode_bpf_map_write_delivery(2), Some(true));
        assert_eq!(super::decode_bpf_map_write_delivery(99), Some(false));
    }

    #[test]
    fn guest_phase_decode_and_progress_order() {
        use crate::vmm::GuestLifecyclePhase as Pub;
        for (raw, expected) in [
            (0u8, Pub::Boot),
            (1, Pub::Attach),
            (2, Pub::Dispatch),
            (3, Pub::Body),
            (4, Pub::Teardown),
            (250, Pub::Boot), // unknown → conservative Boot
        ] {
            assert_eq!(decode_guest_phase(raw), expected);
        }
        assert!(Pub::Boot < Pub::Attach);
        assert!(Pub::Body < Pub::Teardown);
    }

    /// The dump-facing tokens and the decode round-trip are stable — the
    /// watchdog dump's `kill_reason=` key relies on both, and the fire
    /// path stores each variant into an `AtomicU8` the dump decodes.
    #[test]
    fn kill_reason_render_and_decode() {
        for (tag, token) in [
            (KillReasonTag::Unset, "none"),
            (KillReasonTag::Tier1Cpu, "tier1-cpu-budget"),
            (KillReasonTag::Tier2Idle, "tier2-idle-wedge"),
            (KillReasonTag::Tier3Deadman, "tier3-deadman-deadline"),
            (KillReasonTag::ApKill, "ap-kill"),
        ] {
            assert_eq!(tag.render(), token);
            assert_eq!(KillReasonTag::from_u8(tag as u8), tag);
            // Round-trip through the atomic slot the watchdog stores into.
            let slot = AtomicU8::new(KillReasonTag::Unset as u8);
            slot.store(tag as u8, Ordering::Relaxed);
            assert_eq!(KillReasonTag::from_u8(slot.load(Ordering::Relaxed)), tag);
        }
        // Any unknown byte decodes to Unset (a never-written slot).
        assert_eq!(KillReasonTag::from_u8(200), KillReasonTag::Unset);
    }

    /// The dump-facing tokens and the decode round-trip are stable — the
    /// watchdog dump's `reset_armed_by=` key relies on both.
    #[test]
    fn tag_render_and_decode() {
        for (tag, token) in [
            (WatchdogResetTag::Unset, "none"),
            (WatchdogResetTag::ScxRootLatch, "scx-root-attach-latch"),
            (WatchdogResetTag::ScenarioStart, "scenario-start"),
            (WatchdogResetTag::FreezeExtend, "freeze-extend"),
            (WatchdogResetTag::WprofGrace, "wprof-grace"),
            (WatchdogResetTag::GuestAttachConfirm, "guest-attach-confirm"),
        ] {
            assert_eq!(tag.render(), token);
            assert_eq!(WatchdogResetTag::from_u8(tag as u8), tag);
        }
        // Any unknown byte decodes to Unset (a never-written slot).
        assert_eq!(WatchdogResetTag::from_u8(200), WatchdogResetTag::Unset);
    }

    /// The wprof-grace writer stamps its tag alongside the ns store when
    /// (and only when) it wins the max-style store.
    #[test]
    fn grace_writer_stamps_wprof_tag_on_win() {
        let reset_ns = AtomicU64::new(0);
        let reset_tag = AtomicU8::new(WatchdogResetTag::ScxRootLatch as u8);
        // First arm from a zero deadline: the grace store wins, so both
        // the ns and the tag move.
        super::extend_watchdog_reset_for_grace(
            &reset_ns,
            &reset_tag,
            Instant::now(),
            Duration::from_secs(5),
        );
        assert_ne!(reset_ns.load(Ordering::Acquire), 0);
        assert_eq!(
            WatchdogResetTag::from_u8(reset_tag.load(Ordering::Relaxed)),
            WatchdogResetTag::WprofGrace,
        );

        // A far-future existing deadline is NOT shrunk, so the grace does
        // not win and must not overwrite the prior tag.
        let reset_ns = AtomicU64::new(u64::MAX);
        let reset_tag = AtomicU8::new(WatchdogResetTag::ScxRootLatch as u8);
        super::extend_watchdog_reset_for_grace(
            &reset_ns,
            &reset_tag,
            Instant::now(),
            Duration::from_secs(5),
        );
        assert_eq!(reset_ns.load(Ordering::Acquire), u64::MAX);
        assert_eq!(
            WatchdogResetTag::from_u8(reset_tag.load(Ordering::Relaxed)),
            WatchdogResetTag::ScxRootLatch,
        );
    }
}

/// Three-way result of polling the BPF probe's `.bss` latch via the
/// cached guest-physical-address path used by [`bss_read_state`].
///
/// `read_u32` returns `0` for two semantically distinct reasons: the
/// probe has not latched yet (genuine "no fire") AND the cached PA no
/// longer resolves to a live DRAM region (out-of-bounds, hole between
/// regions). Conflating the two masks a stale-cache regression as
/// "still waiting for the trigger" and lets the freeze coordinator
/// drift past a real fire when the probe has been torn down or its
/// vmalloc page recycled. Each consumer decides how to react —
/// production gates the err_triggered flag on `Triggered` only and
/// surfaces `OutOfBounds` as a diagnostic so an operator can correlate
/// late-run BSS misses with map-idr churn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BssReadState {
    /// Cache is unset (probe not yet discovered) or `mem` is None
    /// (no NUMA layout published yet — pre-boot window). The read
    /// path short-circuits without touching guest memory.
    NotResolved,
    /// Cache is set and the PA is in-bounds, but the latched u32 is
    /// still `0`. The probe has not flipped its sticky 0→1 latch yet.
    NotTriggered,
    /// Cache is set and the PA is in-bounds; the read returned a
    /// non-zero value. The probe has latched its
    /// `ktstr_err_exit_detected` flag.
    Triggered,
    /// Cache is set but the PA falls outside every live DRAM region.
    /// Distinct from `NotTriggered` so callers can warn on a stale
    /// cache without conflating it with "no fire yet". A bare
    /// `read_u32` on the same PA returns `0` per
    /// `monitor::reader::GuestMem::read_scalar`'s OOB-zero path,
    /// which would hide the regression.
    OutOfBounds,
}

/// Resolve the BPF `.bss` latch read into the three-way
/// [`BssReadState`].
///
/// Pure function so the freeze coordinator's poll loop can be tested
/// in isolation — drives the same `mem.read_u32` and `region_avail`
/// calls the production loop performs at the `bss_state` binding,
/// but without booting a VM. The `OutOfBounds` branch uses
/// `region_avail(pa) >= 4` to confirm the cached PA still resolves
/// to a 4-byte-readable mapping; without that check, an OOB PA would
/// silently report `NotTriggered` because
/// [`monitor::reader::GuestMem::read_u32`] returns zeroes for
/// out-of-bounds PAs.
pub(super) fn bss_read_state(
    mem: Option<&monitor::reader::GuestMem>,
    cached_pa: Option<u64>,
) -> BssReadState {
    match (mem, cached_pa) {
        (Some(m), Some(pa)) => {
            if m.region_avail(pa) < 4 {
                BssReadState::OutOfBounds
            } else if m.read_u32(pa, 0) != 0 {
                BssReadState::Triggered
            } else {
                BssReadState::NotTriggered
            }
        }
        _ => BssReadState::NotResolved,
    }
}

/// Combine the watchpoint hit latch and the bss-latch state into the
/// run-loop's "fire this iteration" verdict. The hardware watchpoint
/// is the primary path (synchronous KVM_EXIT_DEBUG delivery); the
/// bss-latch read is the fallback for kernels where the watchpoint
/// could not be armed (no `scx_root` symbol, BTF stripped of
/// `scx_sched`, KVM_SET_GUEST_DEBUG ioctl rejected). Either signal
/// alone is sufficient to start the late-trigger freeze.
///
/// Only [`BssReadState::Triggered`] counts as a fire on the bss
/// path — `OutOfBounds`, `NotResolved`, and `NotTriggered` all
/// resolve to "no observable fire this iteration" so a stale
/// cached PA after probe unload cannot
/// synthesise a phantom fire from arbitrary DRAM bytes.
pub(super) fn compute_err_triggered(watchpoint_hit: bool, bss_state: BssReadState) -> bool {
    watchpoint_hit || matches!(bss_state, BssReadState::Triggered)
}

/// Predicate that distinguishes a watchpoint-only trigger from a
/// bss-confirmed trigger. Returns `true` when the hardware watchpoint
/// fired but the BPF `.bss` latch did NOT — the trigger is
/// "watchpoint sees a write to `*scx_root->exit_kind`, probe hasn't
/// latched the error class (yet)".
///
/// Used at the `freeze_and_dispatch` call site (a closure inside
/// the run-loop, not a free fn) to compute the `gate_on_exit_kind`
/// argument: a watchpoint-only trigger sets
/// the gate because the watchpoint catches every write (including the
/// init/teardown clean values `SCX_EXIT_NONE` / `SCX_EXIT_DONE` that
/// would synthesise a bogus dump without the gate); a bss-confirmed
/// trigger skips the gate because `probe.bpf.c:687`'s
/// `__sync_val_compare_and_swap(&ktstr_err_exit_detected, 0, 1)`
/// already proved the kernel observed `kind >= SCX_EXIT_ERROR` at the
/// tp_btf hook firing instant.
pub(super) fn compute_watchpoint_only_trigger(
    watchpoint_hit: bool,
    bss_state: BssReadState,
) -> bool {
    watchpoint_hit && !matches!(bss_state, BssReadState::Triggered)
}

/// Pure exit_kind gate decision: does the live scheduler exit warrant a
/// failure dump? `Some(kind)` = `*scx_root->exit_kind` translated and
/// read as `kind`; `None` = the KVA no longer translates (the slab page
/// holding `*scx_root` was freed mid-teardown). Returns true only for an
/// error-class exit (`kind >= SCX_EXIT_ERROR`). `SCX_EXIT_ERROR = 1024`
/// is the first error-class value in the kernel's `enum scx_exit_kind`;
/// values below are clean (NONE/DONE) or normal unregister classes
/// (UNREG/SYSRQ/PARENT) that do not warrant a dump. A non-translating
/// KVA (`None`) suppresses — there is no scheduler state to capture.
///
/// Pure so the gate decision is unit-tested without booting a VM: the
/// memory translate + read belong to the caller; this is the decision
/// over their result.
pub(super) fn exit_kind_warrants_dump(kind: Option<u32>) -> bool {
    /// First error-class value in the kernel's `enum scx_exit_kind`.
    const SCX_EXIT_ERROR: u32 = 1024;
    matches!(kind, Some(k) if k >= SCX_EXIT_ERROR)
}

/// Decide whether the BPF `.bss` latch rescues a dump the exit_kind
/// gate decided to suppress. Called only when the gate suppressed
/// (`gate_decision == false` at the call site): the live
/// `*scx_root->exit_kind` read as a clean value, or its KVA no longer
/// translated. The probe's sticky `ktstr_err_exit_detected` latch is
/// the historical authority — if it independently observed an
/// error-class exit ([`BssReadState::Triggered`]) AND the cached `.bss`
/// PA has not gone [`BssReadState::OutOfBounds`] this run
/// (`bss_oob_warn_logged == false`, so a recycled vmalloc page cannot
/// synthesise a phantom fire from unrelated bytes), the suppressed dump
/// is rescued and emitted anyway. Any other state honors the gate's
/// suppression.
///
/// Pure so this F15-class "gate would suppress but latch rescues"
/// defense is unit-tested without booting a VM: the rescue is a
/// host-side coordinator decision, and the host/guest process boundary
/// makes an e2e seam impossible (a static set by the guest payload
/// never reaches the host coordinator's copy).
pub(super) fn bss_latch_rescues_suppressed_dump(
    bss_state: BssReadState,
    bss_oob_warn_logged: bool,
) -> bool {
    matches!(bss_state, BssReadState::Triggered) && !bss_oob_warn_logged
}

/// Snake-case label for a [`BssReadState`] suitable for embedding in
/// the [`crate::monitor::dump::DegradedFailureDumpReport::bss_latch_state`]
/// wire field. Stable wire format: callers (auto-repro tail renderer,
/// operator `jq` inspection) match against the exact strings returned
/// here — adding a new state requires a new label rather than mutating
/// an existing one.
pub(super) fn bss_state_label(state: BssReadState) -> &'static str {
    match state {
        BssReadState::Triggered => "triggered",
        BssReadState::NotTriggered => "not_triggered",
        BssReadState::OutOfBounds => "out_of_bounds",
        BssReadState::NotResolved => "not_resolved",
    }
}

/// Walk back from `cap` to the largest valid UTF-8 char boundary
/// `<= cap` (and `<= s.len()`). Returns the safe slice length: the
/// caller may then do `&s[..returned]` without panicking.
///
/// Used by the stderr last-ditch preservation paths (`emit_json` /
/// `emit_degraded_json` closure fallbacks + end-of-coord drain) where
/// the naive `&s[..cap]` would panic if `cap` lands inside a multi-
/// byte UTF-8 sequence. The serde-serialized FailureDumpReport carries
/// non-ASCII bytes in any `String` field — task `comm` via
/// `String::from_utf8_lossy` yields U+FFFD (3 bytes) for invalid input,
/// and operator-supplied tags or kernel symbol strings can include
/// Unicode. Without this walk, an ENOSPC/EROFS-triggered stderr emit
/// can panic instead of preserving the payload — exactly the silent-
/// drop failure mode the fallback exists to prevent.
///
/// Worst-case retreat is 3 bytes (UTF-8 sequences are 1-4 bytes; the
/// boundary at byte 0 of a valid sequence is always a char boundary,
/// and 0 itself is always a boundary). Returns 0 for an empty input
/// or when `cap == 0`.
pub(super) fn utf8_safe_truncate_len(s: &str, cap: usize) -> usize {
    let bounded = cap.min(s.len());
    (0..=bounded)
        .rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0)
}

/// Indices of vCPUs whose register snapshot is `None` — the vCPUs
/// that either failed to park during the freeze rendezvous or whose
/// `KVM_GET_REGS` ioctl errored mid-shutdown. Surfaced alongside the
/// structured summary in every stderr last-ditch preservation path
/// (`emit_json` fallback, end-of-coord drain) so an operator sees
/// WHICH vCPUs failed — not just the count.
///
/// Pre-computed once at each caller (Captured arm, drain) and passed
/// by `&[usize]` reference into the emit closure. Centralised here
/// so the filter_map pattern lives at one location — a future change
/// to "what counts as a stalled vCPU" updates this one fn rather than
/// every emit site.
pub(super) fn vcpu_none_indices(
    vcpu_regs: &[Option<crate::monitor::dump::VcpuRegSnapshot>],
) -> Vec<usize> {
    vcpu_regs
        .iter()
        .enumerate()
        .filter_map(|(i, r)| if r.is_none() { Some(i) } else { None })
        .collect()
}

/// Format the trailing `path_part` clause for the emit closures'
/// tracing summary line. Disambiguates 3 states an operator needs
/// to distinguish at-a-glance:
///
/// - `(Some(p), _)` — sink configured AND write succeeded; renders
///   ` -> {p}` so the operator sees the on-disk dump path inline.
/// - `(None, true)` — sink configured but the atomic write failed;
///   renders ` (atomic write failed; see preceding warn)` so the
///   operator knows to scroll up for the warn carrying the underlying
///   `io::Error`. The preceding warn fires at the and_then Err arm
///   in both `emit_json`-class closures.
/// - `(None, false)` — operator never wired `failure_dump_path`;
///   renders ` (no file sink)` — the quiet expected-behavior case
///   for verifier/shell/template builders.
///
/// Centralised here so emit_json and emit_degraded_json don't each
/// carry the same 3-arm match inline; future emit-site additions reuse
/// the disambiguation rule for free.
pub(super) fn format_path_part(path_str: Option<&str>, write_failed: bool) -> String {
    match (path_str, write_failed) {
        (Some(p), _) => format!(" -> {p}"),
        (None, true) => " (atomic write failed; see preceding warn)".to_string(),
        (None, false) => " (no file sink)".to_string(),
    }
}

/// Format the trailing "(truncated to N bytes ...)" marker appended to
/// stderr last-ditch preservation payload-head lines when the
/// UTF-8-safe truncation actually dropped bytes. Returns empty when
/// the truncated head reaches the full payload (no truncation
/// happened) so the caller's `eprintln!` reads cleanly without a
/// no-op clause.
///
/// Mirrors the `STDERR_DUMP_CAP` cascade pattern used by every
/// stderr-fallback site (emit_json / emit_degraded_json / end-of-coord
/// drain). The cap value is supplied by the caller because the drain
/// site uses a distinct `DRAIN_STDERR_DUMP_CAP` constant for its own
/// scope, but the truncation-marker format string is the same.
///
/// Centralised so the marker text lives in one place rather than three
/// byte-identical inline copies; also exposes the truncation contract
/// for unit testing.
pub(super) fn format_truncation_marker(head_end: usize, total_len: usize) -> String {
    if head_end < total_len {
        let dropped = total_len - head_end;
        format!(
            " (truncated to {head_end} bytes at UTF-8 boundary; {dropped} of {total_len} payload bytes dropped — summary above)"
        )
    } else {
        String::new()
    }
}

/// Shared prefix for every stderr last-ditch preservation `summary`
/// line emitted on a failed atomic-publish OR serialize-failure
/// path. 13 emit sites carry the same
/// `"freeze-coord: STDERR-PRESERVED summary"` literal at the head of
/// their `eprintln!` / `stderr_summary` closure return:
/// - run_vm emit_json Captured (write_failed)
/// - run_vm Captured serialize-failure (serde_json::to_string error)
/// - run_vm emit_degraded_json (write_failed)
/// - TLV CAPTURE Captured + Degraded
/// - on-demand periodic Captured + Degraded
/// - on-demand user-watchpoint Captured + Degraded
/// - on-demand early-snapshot Degraded
/// - dual-snapshot early-pre-late-degraded
/// - dual-snapshot early-only-late-suppressed
/// - end-of-coord drain
///
/// Suffix after this prefix varies per site:
/// `" ({variant}, write failed): {body}"` for most;
/// `" ({variant}, serialize failed): {body}"` for the serde-error path;
/// `": {body}"` for the end-of-coord drain (no per-variant context
/// because `early_retain_tag` carries it separately).
///
/// Operator-grep contract: every stderr-fallback site emits a line
/// matchable by this prefix so a log scraper can pin the entire
/// stderr-preservation surface with one regex. The const exists so
/// adding a new emit site (e.g. a future watchpoint class) reuses
/// the prefix automatically + so a future prefix change updates one
/// site instead of 13.
pub(super) const SNAPSHOT_SUMMARY_PREFIX: &str = "freeze-coord: STDERR-PRESERVED summary";

/// Atomically write a JSON dump payload to a sibling of `dump_path`
/// keyed by `tag`, surfacing operator-visible stderr preservation
/// when the configured sink fails. Used by every on-demand dispatch
/// site (TLV CAPTURE / periodic / user-watchpoint / early-snapshot
/// Degraded) so the 7 sites share one atomic-publish + stderr
/// fallback contract.
///
/// Atomic-publish mirrors the [`KtstrVm`]-scope emit_json /
/// emit_degraded_json cascade in steps 1-4 but DIVERGES from them on
/// step-4 (parent-fsync) failure handling: this tagged-sibling helper
/// rolls back the rename and returns `Err`, whereas the canonical-path
/// closures log the durability gap and return `Ok` per user direction
/// ("operator-has-the-data" discipline on the operator-facing main
/// dump). The split is intentional and documented at both call sites
/// (search "operator-has-the-data discipline" in this file).
///
/// 1. Write payload to a sibling `.json.tmp`.
/// 2. `sync_all()` flushes file data + metadata to disk.
/// 3. `rename()` atomically swaps `.json.tmp` into the tagged path
///    (POSIX rename atomicity).
/// 4. `sync_all()` on the parent-directory fd flushes the rename so a
///    host crash post-rename does not lose the dump on ext4/xfs/btrfs.
///    Per CF3 — POSIX rename(2) atomicity is about ordering visible
///    to other processes, not durability across crashes; directory-
///    entry durability requires fsync on the parent dir, matching
///    the pattern database engines (SQLite, RocksDB) use for journal
///    commits.
///
/// On failure (ENOSPC / EROFS / EACCES / EIO) the helper:
///
/// - Best-effort removes the leftover `.json.tmp` so a future
///   operator does not see a stale sibling next to a stale dump.
/// - If the failure was specifically at the post-rename parent-dir
///   fsync (rename succeeded but durability could not be verified),
///   the helper rolls back by removing the tagged file so the
///   `Err` return remains consistent with "no dump visible to
///   consumers" — keeping the caller's bridge bookkeeping aligned
///   with the filesystem. If rollback itself fails, a warn fires
///   carrying both errors plus the cue
///   "tagged file may be visible on disk despite Err return —
///   operator may need to reconcile bridge state with filesystem"
///   so the operator can spot the rare asymmetric state and
///   reconcile manually.
/// - Fires `tracing::warn!` with `warn_msg` plus the atomic-publish
///   path metadata (`path`, `tmp_path`) and the underlying
///   `io::Error`.
/// - Emits the caller-supplied `stderr_summary` line carrying the
///   structured signals an operator needs for triage (per-site
///   field set; the closure owns its own field choices since each
///   site carries different fields — TLV CAPTURE has request_id +
///   map_count + vcpu_regs_count, periodic has idx, Degraded has
///   reason, etc.).
/// - Emits a payload-head line with [`format_truncation_marker`]
///   appended so the operator sees the structured summary AND a
///   UTF-8-boundary-safe head of the JSON for forensic context.
///
/// Returns:
/// - `Ok(None)` when `dump_path` is `None` — verifier / shell /
///   template iteration that never wired a sink, per ADV V11-1's
///   "no capture requested" case. `stderr_summary` is NOT invoked.
/// - `Ok(Some(tagged))` on successful publish with the published
///   path for the caller's bookkeeping.
/// - `Err(io::Error)` after the stderr fallback has fired. Caller
///   decides on bridge placeholder / reply status / consume-on-
///   success bookkeeping — the helper only owns the file write
///   and the stderr preservation. Today's 10 callers split:
///   7 on-demand sites (TLV CAPTURE / periodic / user-watchpoint /
///   early-snapshot Degraded) `let _ =` the Result because the
///   stderr fallback is their entire recovery contract; the 3
///   early-preservation sites (early-pre-late-degraded,
///   early-only-late-suppressed, end-of-coord drain) destructure
///   `Ok(Some(tagged))` to log a success info trace with the
///   published path and the early-snapshot bookkeeping (consume
///   on success, retain for drain retry on failure).
///
/// # Concurrency
///
/// Single-threaded caller per (dump_path, tag). The freeze
/// coordinator's run-loop closure invokes the 7 on-demand sites
/// sequentially, so the assumption holds today. Concurrent
/// callers with the same tag would race on the `.json.tmp`
/// sibling — `File::create(tmp_path)` with O_TRUNC would clobber
/// the in-flight tmp write of the other caller, silently losing
/// one thread's payload. A future caller dispatched from a
/// different thread must either uniquify `tmp_path` (e.g. via
/// `tempfile::NamedTempFile::new_in(parent)`) or serialize on
/// `tag` before invoking this helper.
/// Same 16 KiB cap as the run_vm-scope `STDERR_DUMP_CAP` and the
/// drain-scope `DRAIN_STDERR_DUMP_CAP`. Operator-grep stability:
/// every stderr-fallback site emits a payload head bounded by the
/// same cap, so an operator-side log parser observing the
/// `truncated to {N} bytes` marker can pin the upper bound at
/// 16 KiB across the whole stderr-fallback surface. Module-scoped
/// (not nested inside [`write_to_tagged_path`]) so unit tests in
/// the sibling test module can import the same value rather than
/// duplicating the literal — a future bump to e.g. 32 KiB updates
/// the production cap AND the test's boundary construction in
/// lockstep.
pub(super) const ONDEMAND_STDERR_DUMP_CAP: usize = 16 * 1024;

pub(super) fn write_to_tagged_path(
    dump_path: Option<&std::path::Path>,
    tag: &str,
    json: &str,
    stderr_summary: impl FnOnce() -> String,
    warn_msg: &'static str,
) -> std::io::Result<Option<std::path::PathBuf>> {
    let Some(base_path) = dump_path else {
        // Ok(None) semantic: dump_path was unwired by the caller
        // (verifier / shell / template iteration; ADV V11-1 "no
        // capture requested" case). Stable contract — the
        // `EarlySnapshotGuard::drain_to_disk` Ok(None) arm carries
        // a `debug_assert!(false, ...)` that depends on this
        // semantic NOT widening to other no-write conditions
        // (e.g. unwritable parent), since the guard's own
        // pre-filter guarantees dump_path is Some when reaching
        // the helper. A widened Ok(None) would fire the
        // debug_assert in test/debug spuriously.
        return Ok(None);
    };

    let tagged = snapshot_tagged_path(base_path, tag);
    if let Some(parent) = tagged.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp_path = tagged.with_extension("json.tmp");
    let write_atomic = || -> std::io::Result<()> {
        use std::io::Write as _;
        let mut f = std::fs::File::create(&tmp_path)?;
        f.write_all(json.as_bytes())?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp_path, &tagged)?;
        // After rename succeeds, parent-dir fsync failure means the
        // file IS visible on disk but its directory entry is not
        // durable across a host crash. To preserve the contract that
        // `Err` means "no dump visible to consumers" (so the caller's
        // bridge bookkeeping stays consistent with the filesystem
        // state), roll back the rename by removing the tagged file.
        // Best-effort: if the rollback itself fails, the operator-
        // visible asymmetric state is logged and the original Err
        // propagates.
        if let Some(dir) = tagged.parent()
            && let Err(parent_fsync_err) = std::fs::File::open(dir).and_then(|d| d.sync_all())
        {
            if let Err(rollback_err) = std::fs::remove_file(&tagged) {
                tracing::warn!(
                    path = %tagged.display(),
                    parent_fsync_error = %parent_fsync_err,
                    rollback_error = %rollback_err,
                    "freeze-coord: parent-dir fsync failed AND rollback of tagged file also failed; tagged file may be visible on disk despite Err return — operator may need to reconcile bridge state with filesystem"
                );
            }
            return Err(parent_fsync_err);
        }
        Ok(())
    };
    match write_atomic() {
        Ok(()) => Ok(Some(tagged)),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp_path);
            tracing::warn!(
                path = %tagged.display(),
                tmp_path = %tmp_path.display(),
                error = %e,
                "{warn_msg}"
            );
            eprintln!("{}", stderr_summary());
            let head_end = utf8_safe_truncate_len(json, ONDEMAND_STDERR_DUMP_CAP);
            let truncated_marker = format_truncation_marker(head_end, json.len());
            // Per-site context (which on-demand path, which retain-tag,
            // why it fired) lives in the structured summary line above —
            // the caller supplied it via `stderr_summary`. The payload-
            // head line stays terse to match the run_vm-scope drain
            // discipline at end-of-coord; an operator-grep on
            // "STDERR-PRESERVED payload head" matches every stderr
            // fallback site uniformly.
            eprintln!(
                "freeze-coord: STDERR-PRESERVED payload head{}: {}",
                truncated_marker,
                &json[..head_end]
            );
            Err(e)
        }
    }
}

/// Outcome of one freeze-and-capture cycle. Replaces the prior
/// `Option<(FailureDumpReport, Instant)>` return so the late-trigger
/// dispatch can distinguish three semantically distinct cases instead
/// of conflating "aborted with reason worth recording" and
/// "legit no-dump suppression" into a single `None`:
///
/// - `Self::Captured` — full dump assembled; emit as
///   [`crate::monitor::dump::SCHEMA_SINGLE`] (or wrapped into
///   [`crate::monitor::dump::SCHEMA_DUAL`] by the dual-snapshot path).
/// - `Self::Degraded` — trigger fired but capture aborted in a way
///   the operator needs to see (rendezvous timed out, etc.). Carries a
///   pre-built [`crate::monitor::dump::DegradedFailureDumpReport`] for
///   immediate emission via the
///   [`crate::monitor::dump::SCHEMA_DEGRADED`] schema. Per the
///   no-silent-drops policy: the trigger fired, an operator must
///   see something — the degraded JSON IS that something.
/// - `Self::Suppressed` — exit_kind gate decided the trigger was a
///   benign init / teardown write (kind &lt; SCX_EXIT_ERROR) and the
///   BPF `.bss` latch agreed it was not an error. Legit "no dump
///   needed" — no JSON emitted. The watchpoint hit flag is reset and
///   the coordinator transitions to [`FreezeState::Done`] so the
///   late-trigger machine stops watching for further error-class
///   triggers on this rebind cycle. On-demand captures (TLV,
///   periodic, user-watchpoint) remain serviceable since they don't
///   gate on `freeze_state`.
// `Captured` carries a full [`crate::monitor::dump::FailureDumpReport`]
// inline (~680 bytes of map / vcpu_regs / walker data); the other
// variants are small. Boxing the dominant variant would double-
// allocate on the success path — the closure returns the outcome to
// a single immediate consumer (the late-trigger dispatch at one
// site, the on-demand sites at three sites) which destructures and
// drops the box right away. Stack-passing the unboxed
// FailureDumpReport reuses the closure's already-allocated frame.
// Macro for the 5 Capture-mode call sites' defensive
// `FreezeOutcome::KernelOp(_)` arms (dedups 5 identical 3-line
// blocks). The macro body is `unreachable!`
// (returns `!`) so each match arm stays exhaustive without
// extra return-type plumbing. Search for `capture_only_unreachable!`
// to find all five sites.
macro_rules! capture_only_unreachable {
    () => {
        unreachable!("FreezeMode::Capture dispatch cannot return KernelOp")
    };
}

#[allow(clippy::large_enum_variant)]
pub(super) enum FreezeOutcome {
    /// Successfully captured a full failure dump.
    /// Produced ONLY by [`FreezeMode::Capture`] dispatch paths.
    Captured(crate::monitor::dump::FailureDumpReport, Instant),
    /// Aborted in a way that warrants surfacing a degraded JSON.
    /// Boxed to keep the enum's discriminant size bounded — the
    /// degraded report carries per-vCPU register data inline.
    /// Produced ONLY by [`FreezeMode::Capture`] dispatch paths.
    Degraded(Box<crate::monitor::dump::DegradedFailureDumpReport>),
    /// Legit suppression: gate decided clean exit. The watchpoint
    /// hit is reset and the coordinator transitions to
    /// [`FreezeState::Done`]. No late dump emitted. In dual-snapshot
    /// mode, if an early Captured snapshot was previously held, it
    /// is preserved to the
    /// [`crate::monitor::dump::SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED`]
    /// tagged sibling path so the operator can read the early
    /// observation; see the late-trigger `Suppressed` arm in
    /// `run_bsp_loop` for the emit detail.
    /// Produced ONLY by [`FreezeMode::Capture`] dispatch paths.
    Suppressed,
    /// Cold-path kernel-memory operation reply.
    /// Produced ONLY by [`FreezeMode::ColdOp`] dispatch paths.
    /// The five Capture-mode call sites in this file each carry a
    /// defensive `FreezeOutcome::KernelOp(_) => unreachable!(…)` arm
    /// (search the file for `freeze_and_dispatch(FreezeMode::Capture`
    /// to find them); the ColdOp call site at the pending-cold-op
    /// drain cannot observe the Capture variants symmetrically.
    KernelOp(crate::vmm::wire::KernelOpReplyPayload),
}

/// Dispatch mode for the freeze rendezvous closure
/// (`freeze_and_dispatch`). The closure body branches on this enum
/// post-park-ack to select between snapshot capture (`Capture`) and
/// cold-path kernel-memory op dispatch (`ColdOp`).
///
/// # Variants
///
/// - [`FreezeMode::Capture`] runs the snapshot CAPTURE pipeline
///   (`dump_state` + numa-stats + serialise + JSON file emit) and
///   returns [`FreezeOutcome::Captured`] / [`FreezeOutcome::Degraded`]
///   / [`FreezeOutcome::Suppressed`].
/// - [`FreezeMode::ColdOp`] invokes
///   [`crate::vmm::freeze_coord::kernel_op_dispatch::dispatch_kernel_op_batch`]
///   against the borrowed request payload and returns
///   [`FreezeOutcome::KernelOp`] wrapping the reply.
///
/// # Lifetime
///
/// The lifetime parameter `'a` binds the [`FreezeMode::ColdOp`]
/// variant's `&'a KernelOpRequestPayload` borrow; the
/// [`FreezeMode::Capture`] variant ignores it. Use elided syntax
/// `FreezeMode<'_>` at function signatures unless the call site
/// needs to name the lifetime.
pub(super) enum FreezeMode<'a> {
    /// Snapshot capture. `gate_on_exit_kind` is the existing
    /// dual-snapshot gate: when true, the closure suppresses
    /// non-late-trigger captures so the watchpoint hit is the
    /// snapshot's only signal.
    Capture { gate_on_exit_kind: bool },
    /// Cold-path kernel-memory op dispatch. Borrows the request
    /// payload from the coordinator's pending-cold-op Vec.
    ColdOp(&'a crate::vmm::wire::KernelOpRequestPayload),
}

/// Owns the dual-snapshot early-trigger `FailureDumpReport` plus the
/// drain bookkeeping (retain tag from a failed late-trigger arm
/// write, dump path, dual-snapshot gate). The guard exists to close
/// a silent-drop window: the freeze coordinator's closure runs on
/// a spawned thread, and a panic anywhere in the closure body would
/// unwind past the end-of-coord drain that flushes a held
/// `early_snapshot` to disk. Per the no-silent-drops policy,
/// every Captured early MUST reach disk regardless of how the
/// closure exits.
///
/// The guard's [`Self::drain_to_disk`] runs the same emit logic the
/// end-of-coord drain used to inline (atomic publish via
/// [`write_to_tagged_path`] + stderr fallback on write failure).
/// The [`Drop`] impl invokes the same drain so an unwinding panic
/// flushes the snapshot before the thread tears down. Both paths
/// are idempotent: `self.snapshot.take()` returns `None` on second
/// call, making a normal-path drain followed by Drop a no-op.
///
/// Cross-cutting state captured at construction:
/// - `dump_path`: cloned from the closure's `freeze_coord_dump_path`
///   so the Drop body has its own owned PathBuf without lifetime
///   dependencies on the closure's locals.
/// - `dual_snapshot`: copy of `freeze_coord_dual_snapshot` so the
///   drain honors the same gate the end-of-coord drain used to.
/// - `retain_tag`: set by the late-trigger Degraded / Suppressed
///   arms when their tagged-sibling write fails, so the drain
///   lands the recovered file at the operator-correct path rather
///   than the default NEVER_FIRED tag.
///
/// Excluded from the guard (verified to NOT need panic-safe
/// preservation):
/// - `early_max_age_jiffies` / `early_threshold_jiffies` — consumed
///   only at the late-Captured `DualFailureDumpReport` assembly
///   site; if lost on panic the on-disk drained file just lacks
///   these annotations and the snapshot itself survives.
/// - `early_degraded_reason` — consumed only at the late-trigger
///   `early_skipped_reason` calculator; the underlying degraded
///   JSON already landed on disk at the
///   `SNAPSHOT_TAG_EARLY_DEGRADED` sibling regardless of panic.
///
/// Drop body must not panic (would abort the process via double-
/// panic on the unwinding thread). The drain body uses only
/// `serde_json::to_string` (panic-free for well-typed input),
/// `write_to_tagged_path` (returns `io::Result`, never panics), and
/// `vcpu_none_indices` (pure filter_map, panic-free). No additional
/// catch_unwind wrap needed at first land.
pub(super) struct EarlySnapshotGuard {
    pub(super) snapshot: Option<crate::monitor::dump::FailureDumpReport>,
    pub(super) retain_tag: Option<&'static str>,
    pub(super) dump_path: Option<std::path::PathBuf>,
    pub(super) dual_snapshot: bool,
}

impl EarlySnapshotGuard {
    /// Flush the held snapshot to its tagged sibling via the shared
    /// atomic-publish helper. Idempotent: a second call after the
    /// first taken the snapshot returns immediately. No-op when
    /// `dual_snapshot` is false, when `dump_path` is unset, or when
    /// `snapshot` is already None.
    ///
    /// Invoked at two sites: the end-of-coord drain (terminal
    /// normal-path flush) and the [`Drop`] impl (panic-unwind
    /// flush). Both produce the same operator-visible artifact.
    pub(super) fn drain_to_disk(&mut self) {
        if !self.dual_snapshot {
            return;
        }
        let Some(dump_path) = self.dump_path.as_deref() else {
            return;
        };
        let Some(early) = self.snapshot.take() else {
            return;
        };
        let drain_tag = self
            .retain_tag
            .unwrap_or(crate::monitor::dump::SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED);
        match serde_json::to_string(&early) {
            Ok(json) => {
                match write_to_tagged_path(
                    Some(dump_path),
                    drain_tag,
                    &json,
                    || {
                        let vcpu_none = vcpu_none_indices(&early.vcpu_regs);
                        format!(
                            "{SNAPSHOT_SUMMARY_PREFIX}: schema={} vcpu_regs_count={} vcpu_none_indices={:?} maps_count={} tasks_enriched={} json_bytes={}",
                            early.schema,
                            early.vcpu_regs.len(),
                            vcpu_none,
                            early.maps.len(),
                            early.task_enrichments.len(),
                            json.len(),
                        )
                    },
                    "freeze-coord: early-only (coord exit drain) write failed; emitting structured summary + payload head to stderr as last-ditch preservation",
                ) {
                    Ok(Some(tagged)) => {
                        tracing::info!(
                            path = %tagged.display(),
                            retain_tag_used = self.retain_tag.is_some(),
                            "freeze-coord: early snapshot preserved at coord exit"
                        );
                    }
                    Ok(None) => {
                        // Unreachable: write_to_tagged_path returns
                        // Ok(None) only when its dump_path argument
                        // is None — but drain_to_disk's own
                        // pre-filter at the `let Some(dump_path) =
                        // self.dump_path.as_deref()` arm above
                        // guarantees `Some(dump_path)` reaches the
                        // helper. A None here means a regression
                        // removed the pre-filter; debug builds
                        // fail-loud, release builds stay silent
                        // (matches the prior empty-arm semantics).
                        debug_assert!(
                            false,
                            "Ok(None) unreachable: drain_to_disk pre-filters \
                             dump_path before calling write_to_tagged_path; \
                             a None reaching here means the pre-filter was \
                             removed"
                        );
                    }
                    Err(_) => {}
                }
            }
            Err(e) => tracing::error!(
                error = %e,
                "freeze-coord: early-only (coord exit drain) JSON serialization \
                 failed (FailureDumpReport serialization is infallible for the \
                 concrete shape — see test \
                 failure_dump_report_serialization_is_infallible_for_max_synthetic_input \
                 in src/monitor/dump/tests.rs; if this error ever fires a new \
                 field with a fallible Serialize impl landed)"
            ),
        }
    }
}

impl Drop for EarlySnapshotGuard {
    fn drop(&mut self) {
        // Fires on closure exit (normal OR panic-unwind). Normal path
        // calls drain_to_disk() explicitly before exit, which takes()
        // the snapshot — so this Drop is a no-op in the normal case.
        // On panic-unwind, the normal drain didn't run; this Drop
        // ensures the captured early reaches disk.
        self.drain_to_disk();
    }
}

/// Free-function inner of [`KtstrVm::has_bpf_scheduler_attached`]
/// — extracted so the gating predicate can be unit-tested
/// without constructing a full [`KtstrVm`].
///
/// Returns true IFF `scheduler_binary` is `Some`. The
/// `sched_enable_cmds` parameter is intentionally accepted but
/// IGNORED: the prior gate OR'd `!sched_enable_cmds.is_empty()`
/// into the predicate, which falsely signalled "BPF scheduler
/// attached" for KernelBuiltin sysctl-toggle configs. The
/// argument stays in the signature so a future caller that
/// wants to combine both signals (e.g. for a stats-collection
/// gate that genuinely cares about whether sysctl writes ran)
/// has a single canonical helper to pass through, and so the
/// unit tests at the bottom of this file pin both axes of the
/// truth table — the `None + non-empty cmds` regression-pin
/// case is the one that demonstrates the fix.
///
/// Generic over `P: AsRef<std::path::Path>` so callers can pass
/// `Option<&PathBuf>` or `Option<&Path>` without an extra ref
/// dance.
fn has_bpf_scheduler_attached_inner<P: AsRef<std::path::Path>>(
    scheduler_binary: Option<&P>,
    _sched_enable_cmds: &[String],
) -> bool {
    scheduler_binary.is_some()
}

/// Kick every still-running vCPU thread out of `KVM_RUN` and join them all,
/// draining the guest-memory readers before the backing mmap unmaps. The
/// caller MUST set `kill`, signal `kill_evt`, and clear `freeze` first so a
/// parked AP observes the shutdown and exits promptly. Kicks + unparks each
/// non-exited AP, epoll-waits on their `exit_evt`s with a 2 s deadline
/// (re-kicking each cycle), then joins every handle. Shared by
/// [`KtstrVm::collect_results`] and [`RunVmThreadGuard`]'s `Drop` so both
/// drain the vCPU threads through the identical, deadlock-safe sequence.
fn kick_and_join_ap_threads(ap_threads: Vec<VcpuThread>) {
    for vt in &ap_threads {
        if !vt.exited.load(Ordering::Acquire) {
            vt.kick();
        }
        vt.handle.thread().unpark();
    }
    let mut remaining = ap_threads.len();
    if remaining > 0
        && let Ok(epoll) = Epoll::new()
    {
        for (i, vt) in ap_threads.iter().enumerate() {
            if vt.exited.load(Ordering::Acquire) {
                remaining -= 1;
                continue;
            }
            let _ = epoll.ctl(
                ControlOperation::Add,
                vt.exit_evt.as_raw_fd(),
                EpollEvent::new(EventSet::IN, i as u64),
            );
        }
        if remaining > 0 {
            let mut events = vec![EpollEvent::default(); remaining];
            let deadline = Instant::now() + Duration::from_secs(2);
            while remaining > 0 {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    break;
                }
                let ms = left.as_millis().min(i32::MAX as u128) as i32;
                match epoll.wait(ms, &mut events) {
                    Ok(0) => break,
                    Ok(n) => remaining = remaining.saturating_sub(n),
                    Err(_) => break,
                }
                for vt in &ap_threads {
                    if !vt.exited.load(Ordering::Acquire) {
                        vt.kick();
                    }
                }
            }
        }
    }
    for vt in ap_threads {
        let _ = vt.handle.join();
    }
}

/// Handles reclaimed from [`RunVmThreadGuard::disarm`] on the Ok path so the
/// normal teardown owns them again (join the watchdog + coordinator before
/// `bsp` drops; move the vCPU / monitor / bpf-write handles onto `VmRunState`).
struct RunVmHandles {
    ap_threads: Vec<VcpuThread>,
    monitor: Option<JoinHandle<monitor::reader::MonitorLoopResult>>,
    bpf_write: Option<JoinHandle<()>>,
    freeze_coord: Option<JoinHandle<()>>,
    watchdog: Option<JoinHandle<()>>,
}

/// The AP-ready boot gate in [`KtstrVm::run_vm`] timed out (or was cut short by
/// `kill`) with one or more AP host threads not yet in `KVM_RUN`. Carries the
/// facts observed at the trip and NOTHING inferred: the prior fixed
/// "host CPU starvation" message was empirically refuted (fair-scheduler
/// contention cannot trip this gate even at 25x oversubscription), so the real
/// cause — pure starvation, D-state blocking, a wedged AP — is left to the
/// per-thread kernel evidence in `evidence` rather than asserted.
///
/// `pub(crate)` (and a named type, not an anyhow message) so
/// `test_support::boot_retry::run_vm_with_ap_gap_retry` can
/// `downcast_ref` it out of the (context-wrapped) error chain and retry the
/// cold boot, the same recovery the guest-side AP-bring-up-gap marker gets.
#[derive(Debug, thiserror::Error)]
#[error(
    "vCPU bring-up gate tripped: vCPU(s) {not_ready:?} did not reach KVM_RUN \
     within {elapsed:?} of waiting (kill flag: {killed}; when set the wait was \
     cut short by a panicking/exiting vCPU rather than running out the full \
     bring-up cap, so `elapsed` is the real wait, not the cap). No cause is \
     asserted — per-thread kernel evidence follows:\n{evidence}"
)]
pub(crate) struct ApGateTimeout {
    /// Guest CPU ids (BSP is vCPU 0; AP index `i` is vCPU `i + 1`) whose
    /// boot latch never fired.
    pub(crate) not_ready: Vec<usize>,
    /// Real time spent in the gate wait, from the gate-start `Instant`.
    pub(crate) elapsed: Duration,
    /// Whether `kill` was set when the wait ended — a set flag means the
    /// wait broke early, so `elapsed` is below the cap by design.
    pub(crate) killed: bool,
    /// Per-not-ready-vCPU kernel evidence, one line each (see the gate's
    /// dump code for the fields and their meaning).
    pub(crate) evidence: String,
}

/// RAII guard that joins the vCPU / monitor / bpf-write / freeze-coordinator /
/// watchdog threads [`KtstrVm::run_vm`] spawns BEFORE the `vm` local (and its
/// `guest_mem` mmap) drops, on EVERY exit path. The normal Ok teardown
/// [`disarm`](Self::disarm)s it and joins the handles itself; any `?`
/// early-return or panic-unwind between the first spawn and that teardown
/// triggers `Drop` instead. Without it, an early exit DETACHES those threads —
/// they hold BARE raw pointers into `vm.guest_mem` (no `Arc`) and run `KVM_RUN`,
/// so touching that memory after `KtstrKvm::Drop` munmaps it is a host-side
/// use-after-free (kernel fd teardown is refcounted, so the danger is purely the
/// dangling host pointer). Accumulates handles as `run_vm` spawns them; mirrors
/// the fixed two-handle `CrossThreadKickGuard` on the interactive path (a
/// function-local struct in `run_interactive`, so it has no linkable path).
///
/// A partial spawn INSIDE `spawn_ap_threads` is covered by its own
/// [`PartialApSpawnGuard`] (before this guard is armed), closing the
/// guest_mem-reader detach window for the vCPUs spawned there. (The sched-stats
/// drainer is NOT a guest_mem reader: its only device call, `drain_port2_bulk`,
/// swaps a host-side `tx_buf` VecDeque the vCPU thread already copied into via
/// `process_tx` — it never dereferences the `virtio_con` guest-memory handle —
/// so it needs no join here and its `ClientShared::drop` stays signal-only.)
///
/// Declared AFTER `bsp` so it drops — and joins the watchdog + coordinator that
/// hold `bsp`'s `ImmediateExitHandle` — BEFORE `bsp` drops; and, as a function
/// local while `vm` is a parameter, BEFORE `vm` drops (parameters drop last).
struct RunVmThreadGuard {
    ap_threads: Vec<VcpuThread>,
    monitor: Option<JoinHandle<monitor::reader::MonitorLoopResult>>,
    bpf_write: Option<JoinHandle<()>>,
    freeze_coord: Option<JoinHandle<()>>,
    watchdog: Option<JoinHandle<()>>,
    kill: Arc<AtomicBool>,
    kill_evt: Arc<EventFd>,
    freeze: Arc<AtomicBool>,
    bsp_done: Arc<AtomicBool>,
    bsp_done_evt: Arc<EventFd>,
}

impl RunVmThreadGuard {
    /// Reclaim every handle on the Ok path, leaving the guard empty so its
    /// `Drop` joins nothing. The caller then runs the normal teardown.
    fn disarm(&mut self) -> RunVmHandles {
        RunVmHandles {
            ap_threads: std::mem::take(&mut self.ap_threads),
            monitor: self.monitor.take(),
            bpf_write: self.bpf_write.take(),
            freeze_coord: self.freeze_coord.take(),
            watchdog: self.watchdog.take(),
        }
    }
}

impl Drop for RunVmThreadGuard {
    fn drop(&mut self) {
        if self.ap_threads.is_empty()
            && self.monitor.is_none()
            && self.bpf_write.is_none()
            && self.freeze_coord.is_none()
            && self.watchdog.is_none()
        {
            return; // disarmed on the Ok path — nothing to join
        }
        // Early-return / panic-unwind cleanup. Replicate the Ok teardown's EXACT
        // stop sequence: set `bsp_done` (+ its evt) so the freeze-coordinator
        // takes its clean bsp-done exit rather than the sched-exit crash-final
        // pass (its outer loop DOES terminate on `kill` alone via that pass, but
        // bsp_done is the correct teardown signal the Ok path also uses), then
        // set `kill` (+ its evt) and clear `freeze`. Join the WATCHDOG and
        // COORDINATOR first: they hold the vCPUs' `ImmediateExitHandle`s, so
        // joining a vCPU (which drops its `kvm_run` mmap) while a coordinator
        // still kicks it would be a use-after-free — the kickers must be
        // quiesced before their targets.
        self.bsp_done.store(true, Ordering::Release);
        let _ = self.bsp_done_evt.write(1);
        self.kill.store(true, Ordering::Release);
        let _ = self.kill_evt.write(1);
        self.freeze.store(false, Ordering::Release);
        if let Some(h) = self.watchdog.take() {
            let _ = h.join();
        }
        if let Some(h) = self.freeze_coord.take() {
            let _ = h.join();
        }
        kick_and_join_ap_threads(std::mem::take(&mut self.ap_threads));
        if let Some(h) = self.monitor.take() {
            let _ = h.join();
        }
        if let Some(h) = self.bpf_write.take() {
            let _ = h.join();
        }
    }
}

/// Restores the calling (BSP) thread's CPU affinity on every exit path from
/// [`KtstrVm::run_vm`]. `run_vm` narrows this thread to the BSP host mask (via
/// [`pin_current_thread`] / [`set_thread_cpumask`]) and never widens it back,
/// which leaked the narrowed mask two ways when the same process ran a second
/// VM (a boot retry, or a subsequent cell in the same process):
///   (a) AP host threads `clone(2)` from this thread and inherit its affinity
///       at spawn, so a leaked narrow mask would confine the next VM's APs to
///       the previous VM's BSP CPU(s); and
///   (b) [`crate::vmm::host_topology::host_allowed_cpus`] seeds the CPU budget from THIS
///       thread's `sched_getaffinity`, so a leaked mask would make replanning
///       compute against the previous VM's cpuset instead of the host's.
/// Captured BEFORE the narrowing and restored on `Drop`, keeping `run_vm`
/// affinity-neutral for its caller. Only `run_vm` needs this; the interactive
/// shell path is one-shot-then-exit.
struct BspAffinityGuard {
    /// The pre-narrowing affinity, or `None` if `sched_getaffinity` failed —
    /// then `Drop` is a no-op (nothing trustworthy to restore).
    saved: Option<nix::sched::CpuSet>,
}

impl BspAffinityGuard {
    /// Snapshot the calling thread's affinity. Call this BEFORE applying the
    /// BSP mask. Mirrors the module's `sched_setaffinity(Pid::from_raw(0), ..)`
    /// idiom (pid 0 = calling thread) in the reverse direction.
    fn capture() -> Self {
        Self {
            saved: nix::sched::sched_getaffinity(nix::unistd::Pid::from_raw(0)).ok(),
        }
    }
}

impl Drop for BspAffinityGuard {
    fn drop(&mut self) {
        if let Some(saved) = &self.saved {
            let _ = nix::sched::sched_setaffinity(nix::unistd::Pid::from_raw(0), saved);
        }
    }
}

/// RAII guard over `spawn_ap_threads`' partially-built vCPU thread set. Each AP
/// thread begins `KVM_RUN` against `vm.guest_mem` the instant it spawns, so if a
/// LATER per-AP `?` fails mid-loop (the exit-eventfd alloc or the thread spawn),
/// the already-spawned APs are live. Without this guard the local
/// `Vec<VcpuThread>` would drop and DETACH them (`VcpuThread` has no `Drop`), and
/// the caller's `vm` would then munmap guest_mem under them — the same host-side
/// use-after-free [`RunVmThreadGuard`] covers, but for the window INSIDE
/// `spawn_ap_threads`, before the outer guard is armed. `Drop` signals kill +
/// clears freeze and joins the partial set via [`kick_and_join_ap_threads`]; the
/// success path [`disarm`](Self::disarm)s it so `Drop` joins nothing.
struct PartialApSpawnGuard {
    ap_threads: Vec<VcpuThread>,
    kill: Arc<AtomicBool>,
    kill_evt: Arc<EventFd>,
    freeze: Arc<AtomicBool>,
}

impl PartialApSpawnGuard {
    /// Reclaim the fully-spawned set on success, leaving the guard empty.
    fn disarm(&mut self) -> Vec<VcpuThread> {
        std::mem::take(&mut self.ap_threads)
    }
}

impl Drop for PartialApSpawnGuard {
    fn drop(&mut self) {
        if self.ap_threads.is_empty() {
            return; // disarmed on success (or a zero-AP / first-AP failure)
        }
        self.kill.store(true, Ordering::Release);
        let _ = self.kill_evt.write(1);
        self.freeze.store(false, Ordering::Release);
        kick_and_join_ap_threads(std::mem::take(&mut self.ap_threads));
    }
}

/// Parse the on-CPU / run-delay pair from a `/proc/<pid>/task/<tid>/schedstat`
/// line. The file holds three space-separated numbers:
///   field 1 = time spent on the CPU (ns),
///   field 2 = time spent runnable-but-not-running (ns),
///   field 3 = number of timeslices run on this CPU.
/// Returns `(on_cpu_ns, run_delay_ns)`, or `None` when either of the first
/// two fields is missing or non-numeric (a short/malformed line). A
/// CONFIG_SCHEDSTATS-off host renders this as `"0 0 0"`, which parses to
/// `Some((0, 0))` — the caller's on-cpu==0 folds that into a `None`
/// dilation, keeping "schedstats unavailable" distinct from a real 1.0.
///
/// `pub(crate)`: shared with the monitor loop's per-tick per-phase
/// contention witness ([`crate::monitor::reader`]), which reads the same
/// schedstat file every tick to attribute host run-delay to lifecycle
/// phases — one parser, one format contract.
pub(crate) fn parse_schedstat_line(line: &str) -> Option<(u64, u64)> {
    let mut it = line.split_whitespace();
    let on_cpu = it.next()?.parse::<u64>().ok()?;
    let run_delay = it.next()?.parse::<u64>().ok()?;
    Some((on_cpu, run_delay))
}

/// Sum the host-side schedstat totals over the given vCPU thread TIDs.
///
/// Reads `/proc/self/task/<tid>/schedstat` for each TID and accumulates
/// fields 1 and 2 (see [`parse_schedstat_line`]). A `tid == 0` (an AP that
/// never stamped its TID — it never scheduled) and any TID whose schedstat
/// is unreadable or malformed are skipped. Returns `None` when NOTHING was
/// sampled (every TID skipped); otherwise `Some(HostVcpuSchedstat)` with
/// the raw sums and the sampled-thread count.
///
/// MUST be called while the vCPU threads are still alive (before they are
/// joined and their `/proc/self/task/<tid>` dirs vanish). Each vCPU thread
/// lives exactly one VM run, so these whole-thread-life totals ARE this
/// run's totals — no baseline subtraction is needed.
///
/// `pub(crate)`: [`ContentionWitnessRecorder`] calls this at lifecycle
/// boundaries to derive per-phase deltas and the task-specific Body cap; it is
/// deliberately absent from the monitor's 100 ms hot path. The whole-run
/// [`VmResult::host_vcpu_schedstat`] read at teardown uses the same function.
pub(crate) fn read_host_vcpu_schedstat(tids: &[i32]) -> Option<HostVcpuSchedstat> {
    read_host_vcpu_schedstat_with_exit_snapshots(tids, &[])
}

/// Sum live vCPU schedstat, falling back to AP self-snapshots after exit.
///
/// A vCPU's proc task directory disappears at thread return. APs publish one
/// final cumulative self-snapshot before that return; an aligned entry here is
/// consulted only when the live proc read failed. This closes teardown's
/// read-vs-exit race without adding any periodic per-vCPU work.
fn read_host_vcpu_schedstat_with_exit_snapshots(
    tids: &[i32],
    exit_snapshots: &[Arc<std::sync::Mutex<Option<HostVcpuSchedstat>>>],
) -> Option<HostVcpuSchedstat> {
    let mut acc = HostVcpuSchedstat::default();
    for (index, &tid) in tids.iter().enumerate() {
        if tid == 0 {
            continue;
        }
        let live = std::fs::read_to_string(format!("/proc/self/task/{tid}/schedstat"))
            .ok()
            .and_then(|line| parse_schedstat_line(line.trim()))
            .map(|(on_cpu, run_delay)| HostVcpuSchedstat {
                total_on_cpu_ns: on_cpu,
                total_run_delay_ns: run_delay,
                sampled_vcpus: 1,
            });
        let sample = live.or_else(|| {
            exit_snapshots
                .get(index)
                .and_then(|slot| *slot.lock_unpoisoned())
        });
        let Some(sample) = sample else {
            continue;
        };
        acc.total_on_cpu_ns = acc.total_on_cpu_ns.saturating_add(sample.total_on_cpu_ns);
        acc.total_run_delay_ns = acc
            .total_run_delay_ns
            .saturating_add(sample.total_run_delay_ns);
        acc.sampled_vcpus = acc.sampled_vcpus.saturating_add(sample.sampled_vcpus);
    }
    (acc.sampled_vcpus > 0).then_some(acc)
}

/// Maximum number of host-pressure intervals retained for one Body phase.
/// At the monitor's nominal 100 ms cadence this is one hour. The cap bounds a
/// wedged cell without affecting normal minute-scale scenarios.
const BODY_CONTENTION_INTERVAL_CAP: usize = 36_000;

/// Persistent reader for the current cgroup's `cpu.pressure` cumulative
/// `some total`.
///
/// CPU PSI `some` is wall time during which at least one runnable host task is
/// stalled for CPU. Every ktstr vCPU is in the process's cgroup, so a delayed
/// vCPU is included even when the competing task is outside that cgroup.
/// Scoping PSI to ktstr's cgroup avoids charging arbitrary pressure elsewhere
/// on a wide host; other tasks deliberately sharing the runner cgroup can only
/// increase the counter. The fd is opened once and rewound for each sample;
/// the hot path performs no allocation.
struct HostCpuPressureReader {
    file: std::fs::File,
}

impl HostCpuPressureReader {
    fn open() -> std::io::Result<Self> {
        let cgroup = std::fs::read_to_string("/proc/self/cgroup")?;
        let relative = parse_unified_cgroup_path(&cgroup).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "missing unified `0::` entry in /proc/self/cgroup",
            )
        })?;
        // On an un-namespaced host, `/` is the machine-wide cgroup and has
        // the same precision failure as `/proc/pressure/cpu`: unrelated work
        // anywhere can make W approach wall time. Prefer the complete
        // lifecycle schedstat fallback in that case. A cgroup namespace can
        // also render its delegated root as `/`; rejecting it only loses
        // localization, never witness soundness.
        if relative == "/" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "current process is in the cgroup-v2 root",
            ));
        }
        let pressure_path = std::path::Path::new("/sys/fs/cgroup")
            .join(relative.trim_start_matches('/'))
            .join("cpu.pressure");
        Ok(Self {
            file: std::fs::File::open(pressure_path)?,
        })
    }

    fn read_some_total_ns(&mut self) -> std::io::Result<u64> {
        self.file.rewind()?;
        let mut buf = [0u8; 512];
        let n = self.file.read(&mut buf)?;
        let text = std::str::from_utf8(&buf[..n]).map_err(std::io::Error::other)?;
        parse_cpu_pressure_some_total_us(text)
            .map(|us| us.saturating_mul(1_000))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "missing `some ... total=` in cgroup cpu.pressure",
                )
            })
    }
}

fn parse_unified_cgroup_path(text: &str) -> Option<&str> {
    text.lines()
        .find_map(|line| line.strip_prefix("0::"))
        .filter(|path| path.starts_with('/'))
}

fn parse_cpu_pressure_some_total_us(text: &str) -> Option<u64> {
    let line = text.lines().find(|line| line.starts_with("some "))?;
    line.split_ascii_whitespace()
        .find_map(|field| field.strip_prefix("total="))?
        .parse()
        .ok()
}

/// `/sys/fs/cgroup/.../cpu.pressure` exposes a cumulative integer-microsecond
/// counter. A difference of two truncated endpoints can be almost 1 µs below
/// the real delta, so charge one full microsecond to every closed interval.
/// A regressing counter means the source reset or changed and cannot anchor a
/// complete series.
fn conservative_pressure_delta_ns(cur_ns: u64, prev_ns: u64) -> Option<u64> {
    cur_ns
        .checked_sub(prev_ns)
        .map(|delta| delta.saturating_add(1_000))
}

fn complete_schedstat_delay_cap_ns(
    expected_vcpus: usize,
    anchor_tids: &[i32],
    anchor: Option<HostVcpuSchedstat>,
    current_tids: &[i32],
    current: Option<HostVcpuSchedstat>,
) -> Option<u64> {
    let expected = u32::try_from(expected_vcpus).ok()?;
    let current = current?;
    if expected == 0
        || current_tids.len() != expected_vcpus
        || current_tids.iter().any(|tid| *tid <= 0)
        || current.sampled_vcpus != expected
        // CONFIG_SCHEDSTATS-off lines are readable as `0 0 0`.
        || current.total_on_cpu_ns == 0
    {
        return None;
    }

    // Prefer the tight delta over the conservatively widened Body span when
    // the preceding lifecycle snapshot saw the same complete live vCPU set.
    if anchor_tids == current_tids
        && let Some(anchor) = anchor
        && anchor.sampled_vcpus == expected
        && current.total_on_cpu_ns > anchor.total_on_cpu_ns
    {
        return Some(
            current
                .total_run_delay_ns
                .saturating_sub(anchor.total_run_delay_ns),
        );
    }

    // The first usable vCPU snapshot can race Body entry (notably when the
    // process is in cgroup-v2 root and PSI is deliberately unavailable).
    // Every vCPU thread is born for this VM run and lives through Body, so its
    // complete cumulative run-delay at Body exit encloses the Body interval.
    // This is a looser upper bound than a boundary delta, but it cannot turn
    // host contention into a false confirmed scheduler failure.
    Some(current.total_run_delay_ns)
}

/// Event-anchored host-contention witness recorder.
///
/// Expensive vCPU schedstat sweeps happen only at real lifecycle transitions
/// (Boot→Attach→Dispatch→Body→Teardown) and once at finalization.
/// The monitor's already-existing tick calls [`Self::sample_pressure`], which
/// reads one width-independent runner-cgroup PSI counter to retain a localized
/// `W(L)` series. Dispatch also samples that counter immediately at Body
/// entry/exit, making the series complete even when the monitor is starved.
/// A complete task-specific schedstat delta over the widened Body span caps
/// PSI so concurrent cells sharing the runner cgroup cannot inflate `W`
/// beyond the target VM's own accumulated run delay. If the preceding
/// boundary snapshot raced vCPU startup, the complete whole-thread-life total
/// is a conservative enclosing fallback.
pub(crate) struct ContentionWitnessRecorder {
    inner: std::sync::Mutex<ContentionWitnessRecorderInner>,
}

struct ContentionWitnessRecorderInner {
    tids: Vec<Arc<AtomicI32>>,
    exit_schedstats: Vec<Arc<std::sync::Mutex<Option<HostVcpuSchedstat>>>>,
    run_start: Instant,
    stage: crate::monitor::LifecycleStage,
    schedstat_anchor: Option<HostVcpuSchedstat>,
    schedstat_anchor_tids: Vec<i32>,
    saw_schedstat: bool,
    per_phase: super::result::PerPhaseSchedstat,
    pressure: Option<HostCpuPressureReader>,
    pressure_prev_ns: Option<u64>,
    pressure_prev_wall_ns: u64,
    body_deltas: Vec<u64>,
    body_widths_ns: Vec<u64>,
    body_saturated: bool,
    body_enter_wall_ns: Option<u64>,
    body_exit_wall_ns: Option<u64>,
    body_pressure_anchored: bool,
    body_pressure_complete: bool,
    body_schedstat_anchor: Option<HostVcpuSchedstat>,
    body_schedstat_anchor_tids: Vec<i32>,
    body_schedstat_cap_ns: u64,
    body_schedstat_cap_complete: bool,
    finalized: bool,
}

impl ContentionWitnessRecorder {
    pub(crate) fn new(
        tids: Vec<Arc<AtomicI32>>,
        exit_schedstats: Vec<Arc<std::sync::Mutex<Option<HostVcpuSchedstat>>>>,
        run_start: Instant,
    ) -> Self {
        let live_tids = tids
            .iter()
            .map(|tid| tid.load(Ordering::Acquire))
            .collect::<Vec<_>>();
        let schedstat_anchor = read_host_vcpu_schedstat(&live_tids);
        let schedstat_anchor_tids = live_tids;
        let saw_schedstat = schedstat_anchor.is_some();
        let mut pressure = HostCpuPressureReader::open().ok();
        let pressure_prev_ns = pressure
            .as_mut()
            .and_then(|reader| reader.read_some_total_ns().ok());
        let now_ns = u64::try_from(run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Self {
            inner: std::sync::Mutex::new(ContentionWitnessRecorderInner {
                tids,
                exit_schedstats,
                run_start,
                stage: crate::monitor::LifecycleStage::Boot,
                schedstat_anchor,
                schedstat_anchor_tids,
                saw_schedstat,
                per_phase: super::result::PerPhaseSchedstat::default(),
                pressure,
                pressure_prev_ns,
                pressure_prev_wall_ns: now_ns,
                body_deltas: Vec::new(),
                body_widths_ns: Vec::new(),
                body_saturated: false,
                body_enter_wall_ns: None,
                body_exit_wall_ns: None,
                body_pressure_anchored: false,
                body_pressure_complete: false,
                body_schedstat_anchor: None,
                body_schedstat_anchor_tids: Vec::new(),
                body_schedstat_cap_ns: 0,
                body_schedstat_cap_complete: false,
                finalized: false,
            }),
        }
    }

    /// One constant-cost sample from the monitor's existing timer wake.
    pub(crate) fn sample_pressure(&self) {
        let mut inner = self.inner.lock_unpoisoned();
        if inner.finalized || inner.stage != crate::monitor::LifecycleStage::Body {
            return;
        }
        let now_ns = u64::try_from(inner.run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        inner.sample_pressure_at(now_ns);
    }

    /// Close the old lifecycle stage and enter `new_stage`. The caller invokes
    /// this only after `ProgressLedger::advance_phase` accepted the forward
    /// transition, so duplicates and late frames cost nothing.
    pub(crate) fn advance(&self, new_stage: crate::monitor::LifecycleStage) {
        let mut inner = self.inner.lock_unpoisoned();
        if inner.finalized || new_stage as u8 <= inner.stage as u8 {
            return;
        }
        let now_ns = u64::try_from(inner.run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let pressure_anchor_ns = inner.pressure_prev_ns;
        let pressure_anchor_wall_ns = inner.pressure_prev_wall_ns;
        let schedstat_cap_anchor = (new_stage == crate::monitor::LifecycleStage::Body)
            .then_some(inner.schedstat_anchor)
            .flatten();
        let schedstat_cap_anchor_tids = if new_stage == crate::monitor::LifecycleStage::Body {
            inner.schedstat_anchor_tids.clone()
        } else {
            Vec::new()
        };
        let pressure_ok = inner.sample_pressure_at(now_ns);
        inner.close_schedstat_stage();

        if inner.stage == crate::monitor::LifecycleStage::Body {
            inner.body_exit_wall_ns = Some(now_ns);
            inner.body_pressure_complete = inner.body_pressure_anchored && pressure_ok;
        }
        inner.stage = new_stage;
        if new_stage == crate::monitor::LifecycleStage::Body {
            // The lifecycle frame can wait briefly in the virtio TX queue
            // before dispatch accepts it. Start the W(L) witness at the
            // PREVIOUS lifecycle boundary and charge that whole prelude to
            // Body. This may include Dispatch contention (larger W →
            // indeterminate), but cannot omit early Body contention that
            // accrued before the host processed ScenarioStart.
            inner.body_enter_wall_ns = Some(pressure_anchor_wall_ns);
            inner.body_pressure_anchored = pressure_ok && pressure_anchor_ns.is_some();
            // Match the PSI witness's deliberately widened start: retain the
            // schedstat snapshot from the lifecycle boundary PRECEDING Body,
            // not the just-taken accepted-Body snapshot. The resulting cap
            // includes the same virtio-dispatch prelude and therefore cannot
            // undercut early Body delay hidden behind frame delivery.
            inner.body_schedstat_anchor = schedstat_cap_anchor;
            inner.body_schedstat_anchor_tids = schedstat_cap_anchor_tids;
            if let (Some(anchor), Some(cur)) = (pressure_anchor_ns, inner.pressure_prev_ns) {
                if let Some(delta) = conservative_pressure_delta_ns(cur, anchor) {
                    inner.push_body_interval(delta, now_ns.saturating_sub(pressure_anchor_wall_ns));
                } else {
                    inner.body_pressure_anchored = false;
                }
            }
        }
    }

    /// Close the final stage and assemble the immutable result passed through
    /// `VmRunState`. A still-live vCPU is read from proc; an AP that already
    /// returned contributes its one-shot exit snapshot.
    pub(crate) fn finish(&self) -> Option<super::result::ContentionWitness> {
        let mut inner = self.inner.lock_unpoisoned();
        if inner.finalized {
            return None;
        }
        let now_ns = u64::try_from(inner.run_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let pressure_ok = inner.sample_pressure_at(now_ns);
        inner.close_schedstat_stage();
        if inner.stage == crate::monitor::LifecycleStage::Body {
            inner.body_exit_wall_ns = Some(now_ns);
            inner.body_pressure_complete = inner.body_pressure_anchored && pressure_ok;
        }
        inner.finalized = true;

        let body_wall_ns = inner
            .body_enter_wall_ns
            .zip(inner.body_exit_wall_ns)
            .map(|(enter, exit)| exit.saturating_sub(enter))
            .unwrap_or(0);
        let mut deltas = std::mem::take(&mut inner.body_deltas);
        let mut widths = std::mem::take(&mut inner.body_widths_ns);
        let mut saturated = inner.body_saturated;
        let mut complete = inner.body_pressure_complete
            && !saturated
            && !deltas.is_empty()
            && deltas.len() == widths.len();

        // PSI unavailable or failed mid-Body: use the event-anchored summed
        // vCPU run-delay over the WHOLE widened Body witness span as one
        // coarse interval. It loses localization but remains a complete upper
        // bound; never convert a missing PSI counter into W=0.
        if !complete && body_wall_ns > 0 && inner.body_schedstat_cap_complete {
            deltas = vec![inner.body_schedstat_cap_ns];
            widths = vec![body_wall_ns];
            complete = true;
            // The replacement is a complete event-boundary delta, not
            // the truncated PSI prefix that hit the cap.
            saturated = false;
        }

        if !inner.saw_schedstat && deltas.is_empty() {
            return None;
        }
        Some(super::result::ContentionWitness {
            per_phase: inner.per_phase,
            body_window: super::result::BodyContentionWindow {
                tick_deltas: deltas,
                tick_widths_ns: widths,
                // Retained for legacy readers; new window math consumes the
                // real widths above.
                tick_ns: Duration::from_millis(100).as_nanos() as u64,
                saturated,
                complete,
                schedstat_cap_ns: inner.body_schedstat_cap_ns,
                schedstat_cap_complete: inner.body_schedstat_cap_complete,
                body_wall_ns,
                body_covered_ns: if complete { body_wall_ns } else { 0 },
            },
        })
    }
}

impl ContentionWitnessRecorderInner {
    /// Sample PSI at an explicit run-relative wall. Returns true when a valid
    /// counter value was read and installed as the new anchor.
    fn sample_pressure_at(&mut self, now_ns: u64) -> bool {
        let Some(reader) = self.pressure.as_mut() else {
            return false;
        };
        let cur = match reader.read_some_total_ns() {
            Ok(cur) => cur,
            Err(e) => {
                tracing::warn!(err = %e, "host CPU PSI read failed; contention witness will use Body schedstat fallback");
                self.pressure = None;
                self.pressure_prev_ns = None;
                return false;
            }
        };
        if let Some(prev) = self.pressure_prev_ns {
            let Some(delta) = conservative_pressure_delta_ns(cur, prev) else {
                tracing::warn!(
                    previous_ns = prev,
                    current_ns = cur,
                    "host CPU PSI counter regressed; contention witness will use Body schedstat fallback"
                );
                self.pressure = None;
                self.pressure_prev_ns = None;
                return false;
            };
            if self.stage == crate::monitor::LifecycleStage::Body {
                self.push_body_interval(delta, now_ns.saturating_sub(self.pressure_prev_wall_ns));
            }
        }
        self.pressure_prev_ns = Some(cur);
        self.pressure_prev_wall_ns = now_ns;
        true
    }

    fn push_body_interval(&mut self, delta_ns: u64, width_ns: u64) {
        if self.body_deltas.len() < BODY_CONTENTION_INTERVAL_CAP {
            self.body_deltas.push(delta_ns);
            self.body_widths_ns.push(width_ns);
        } else {
            self.body_saturated = true;
        }
    }

    fn close_schedstat_stage(&mut self) {
        let tids = self
            .tids
            .iter()
            .map(|tid| tid.load(Ordering::Acquire))
            .collect::<Vec<_>>();
        let cur = read_host_vcpu_schedstat_with_exit_snapshots(&tids, &self.exit_schedstats);
        if self.stage == crate::monitor::LifecycleStage::Body {
            if let Some(cap) = complete_schedstat_delay_cap_ns(
                self.tids.len(),
                &self.body_schedstat_anchor_tids,
                self.body_schedstat_anchor,
                &tids,
                cur,
            ) {
                self.body_schedstat_cap_ns = cap;
                self.body_schedstat_cap_complete = true;
            } else {
                self.body_schedstat_cap_complete = false;
            }
        }
        match (self.schedstat_anchor, cur) {
            (Some(anchor), Some(cur)) => {
                self.per_phase.phases[self.stage as usize] = cur.delta_from(&anchor);
                self.schedstat_anchor = Some(cur);
                self.saw_schedstat = true;
            }
            (_, Some(cur)) => {
                self.schedstat_anchor = Some(cur);
                self.saw_schedstat = true;
            }
            (_, None) => {
                // Do not carry an old anchor across an unattested boundary;
                // a later recovery must seed afresh rather than mix phases.
                self.schedstat_anchor = None;
            }
        }
        self.schedstat_anchor_tids = tids;
    }
}

/// Snapshot the CALLING thread's own cumulative schedstat (on-CPU,
/// run-delay) — the instrument for the cleanup-window dilation
/// evidence. `read_host_vcpu_schedstat(&[gettid()])` with the same
/// parser/format contract; `None` when the read fails or
/// CONFIG_SCHEDSTATS is off (an all-zero line still parses; the
/// upstream delta then reads 0 on-CPU and consumers treat the window
/// as unattested).
pub(crate) fn read_self_thread_schedstat() -> Option<HostVcpuSchedstat> {
    // SAFETY: gettid(2) takes no arguments and cannot fail.
    let tid = unsafe { libc::gettid() } as i32;
    read_host_vcpu_schedstat(&[tid])
}

#[cfg(test)]
mod schedstat_tests {
    use super::{
        HostCpuPressureReader, complete_schedstat_delay_cap_ns, conservative_pressure_delta_ns,
        parse_cpu_pressure_some_total_us, parse_schedstat_line, parse_unified_cgroup_path,
        read_host_vcpu_schedstat, read_host_vcpu_schedstat_with_exit_snapshots,
    };
    use crate::vmm::HostVcpuSchedstat;

    /// A valid three-field line yields `(on_cpu, run_delay)`; a
    /// two-field line still parses (field 3 is unused); a one-field or
    /// non-numeric line is `None`. The CONFIG_SCHEDSTATS-off `"0 0 0"`
    /// parses to `Some((0, 0))` (folded to a `None` dilation upstream).
    #[test]
    fn parse_schedstat_line_valid_malformed_short() {
        assert_eq!(parse_schedstat_line("12345 678 9"), Some((12345, 678)));
        assert_eq!(parse_schedstat_line("100 200"), Some((100, 200)));
        assert_eq!(parse_schedstat_line("0 0 0"), Some((0, 0)));
        assert_eq!(parse_schedstat_line(""), None, "empty line");
        assert_eq!(parse_schedstat_line("42"), None, "short: one field");
        assert_eq!(parse_schedstat_line("abc def"), None, "non-numeric");
        assert_eq!(parse_schedstat_line("100 x"), None, "second field bad");
    }

    #[test]
    fn parse_cpu_pressure_some_total_ignores_averages_and_full() {
        let text = "some avg10=1.25 avg60=2.50 avg300=3.75 total=987654\n\
                    full avg10=0.00 avg60=0.00 avg300=0.00 total=42\n";
        assert_eq!(parse_cpu_pressure_some_total_us(text), Some(987_654));
        assert_eq!(parse_cpu_pressure_some_total_us("full total=42\n"), None);
        assert_eq!(
            parse_cpu_pressure_some_total_us("some avg10=0 total=wat\n"),
            None
        );
    }

    #[test]
    fn pressure_delta_rounds_up_and_rejects_counter_regression() {
        assert_eq!(conservative_pressure_delta_ns(12_000, 10_000), Some(3_000));
        assert_eq!(conservative_pressure_delta_ns(10_000, 10_000), Some(1_000));
        assert_eq!(conservative_pressure_delta_ns(9_999, 10_000), None);
        assert_eq!(conservative_pressure_delta_ns(u64::MAX, 0), Some(u64::MAX));
    }

    #[test]
    fn schedstat_cap_uses_tight_delta_or_complete_lifetime_fallback() {
        let anchor = HostVcpuSchedstat {
            total_on_cpu_ns: 1_000,
            total_run_delay_ns: 100,
            sampled_vcpus: 2,
        };
        let current = HostVcpuSchedstat {
            total_on_cpu_ns: 2_000,
            total_run_delay_ns: 350,
            sampled_vcpus: 2,
        };
        assert_eq!(
            complete_schedstat_delay_cap_ns(
                2,
                &[101, 102],
                Some(anchor),
                &[101, 102],
                Some(current),
            ),
            Some(250)
        );
        assert_eq!(
            complete_schedstat_delay_cap_ns(
                2,
                &[101, 102],
                Some(anchor),
                &[101, 103],
                Some(current),
            ),
            Some(350),
            "a mismatched anchor falls back to the complete current lifetime"
        );
        assert_eq!(
            complete_schedstat_delay_cap_ns(2, &[], None, &[101, 102], Some(current)),
            Some(350),
            "a missing pre-Body anchor still has a sound whole-lifetime bound"
        );
        assert_eq!(
            complete_schedstat_delay_cap_ns(2, &[101, 102], Some(anchor), &[101, 0], Some(current),),
            None,
            "an unstamped vCPU makes the cap partial"
        );
        assert_eq!(
            complete_schedstat_delay_cap_ns(
                2,
                &[101, 102],
                Some(anchor),
                &[101, 102],
                Some(HostVcpuSchedstat {
                    sampled_vcpus: 1,
                    ..current
                }),
            ),
            None,
            "a failed schedstat read makes the cap partial"
        );
        assert_eq!(
            complete_schedstat_delay_cap_ns(
                2,
                &[101, 102],
                Some(HostVcpuSchedstat {
                    total_on_cpu_ns: 0,
                    total_run_delay_ns: 0,
                    sampled_vcpus: 2,
                }),
                &[101, 102],
                Some(HostVcpuSchedstat {
                    total_on_cpu_ns: 0,
                    total_run_delay_ns: 0,
                    sampled_vcpus: 2,
                }),
            ),
            None,
            "CONFIG_SCHEDSTATS-off zeros are not a complete cap"
        );
    }

    #[test]
    fn schedstat_sum_uses_exit_snapshot_after_proc_task_disappears() {
        let missing_tid = i32::MAX;
        let saved = HostVcpuSchedstat {
            total_on_cpu_ns: 4_000,
            total_run_delay_ns: 9_000,
            sampled_vcpus: 1,
        };
        let slots = vec![std::sync::Arc::new(std::sync::Mutex::new(Some(saved)))];
        assert_eq!(
            read_host_vcpu_schedstat_with_exit_snapshots(&[missing_tid], &slots),
            Some(saved),
        );
        assert_eq!(
            read_host_vcpu_schedstat_with_exit_snapshots(&[missing_tid], &[]),
            None,
            "without the aligned exit slot a vanished proc task is unavailable",
        );
    }

    #[test]
    fn unified_cgroup_path_requires_the_v2_entry_and_absolute_path() {
        assert_eq!(
            parse_unified_cgroup_path("11:cpu:/old\n0::/runner/job\n"),
            Some("/runner/job")
        );
        assert_eq!(parse_unified_cgroup_path("0::/\n"), Some("/"));
        assert_eq!(parse_unified_cgroup_path("0::relative\n"), None);
        assert_eq!(parse_unified_cgroup_path("11:cpu:/old\n"), None);
    }

    #[test]
    fn cpu_pressure_reader_rewinds_between_samples_when_available() {
        let Ok(mut reader) = HostCpuPressureReader::open() else {
            // CPU PSI is optional at runtime; parsing is covered separately.
            return;
        };
        let first = reader.read_some_total_ns().expect("first PSI sample");
        let second = reader.read_some_total_ns().expect("second PSI sample");
        assert!(second >= first, "the cumulative PSI clock cannot decrease");
    }

    /// An all-zero (or empty) TID list samples nothing -> `None`. tid==0
    /// is skipped without a read attempt.
    #[test]
    fn read_host_vcpu_schedstat_none_when_nothing_sampled() {
        assert!(read_host_vcpu_schedstat(&[]).is_none(), "empty -> None");
        assert!(
            read_host_vcpu_schedstat(&[0, 0]).is_none(),
            "all tid==0 skipped -> None"
        );
    }
}

impl KtstrVm {
    /// Spawn threads and run the BSP. Returns all state needed for
    /// `collect_results`.
    ///
    /// # Failure-dump freeze
    ///
    /// When the BPF probe latches a sched_ext error-class exit
    /// (SCX_EXIT_ERROR / _BPF / _STALL), a host-side coordinator
    /// thread freezes every vCPU long enough to read BPF map state
    /// for post-mortem analysis. The freeze is transparent to test
    /// authors — the test still observes the same failure verdict
    /// and exit path — but adds up to ~10 ms of thaw latency to the
    /// failure path (the parked-vCPU poll cadence). Healthy runs
    /// never enter the freeze path; the latch only fires on an
    /// error-class scheduler exit.
    pub(super) fn run_vm(
        &self,
        run_start: Instant,
        mut vm: kvm::KtstrKvm,
        default_cpu_mask: Option<&[usize]>,
        effective_pinning_plan: Option<&super::host_topology::PinningPlan>,
        // Refreshed no-perf CPU list from `acquire_run_locks`' run-time
        // replan. `Some` only on the no-perf path; used in preference to
        // the stale build-time `self.no_perf_plan.cpus` for every affinity
        // mask below (vCPU-thread mask, BSP mask, virtio-blk worker
        // placement) so those masks match the LLCs the run-scoped flocks
        // hold. `None` keeps the pre-replan `no_perf_plan` / default-mask
        // fallback (the interactive path and every non-no-perf run).
        effective_no_perf_cpus: Option<&[usize]>,
    ) -> Result<VmRunState> {
        let com1 = Arc::new(PiMutex::new(console::Serial::new(console::COM1_BASE)));
        let com2 = Arc::new(PiMutex::new(console::Serial::new(console::COM2_BASE)));
        // Userspace IOAPIC handle for the split-irqchip path: the device + the
        // (Copy) raw VM fd, so each run loop reprograms MSI routes on a guest
        // RTE write without borrowing `&vm.vm_fd`. Created here, in the same
        // scope as the device handles, so it is visible to both the BSP loop
        // and the per-AP spawn. The raw fd is alive for `run_vm`; the APs only
        // touch the IOAPIC while the VM runs (at shutdown `kill` stops them
        // before vm drops), so the post-drop window is held-but-unused. `None`
        // for <=254-vCPU guests (in-kernel IOAPIC).
        // x86-only: `vm.ioapic` (the device) and `IoapicHandle::new` exist
        // only on the split-irqchip path. On aarch64 `IoapicHandle` is the
        // uninhabited placeholder and KtstrKvm has no `ioapic` field (the GIC
        // routes device IRQs), so the handle is always `None` there.
        #[cfg(target_arch = "x86_64")]
        let ioapic_handle: Option<Arc<crate::vmm::IoapicHandle>> = vm.ioapic.as_ref().map(|io| {
            Arc::new(crate::vmm::IoapicHandle::new(
                io.clone(),
                vm.vm_fd.as_raw_fd(),
            ))
        });
        #[cfg(not(target_arch = "x86_64"))]
        let ioapic_handle: Option<Arc<crate::vmm::IoapicHandle>> = None;

        // Full-irqchip shared GSI route owner — the inverse of `ioapic_handle`:
        // `Some` only on the full in-kernel irqchip path (`!split_irqchip`) with
        // PCI enabled, where it owns the device MSI(-X) routes. `install_defaults`
        // installs an EXPLICIT default routing table (IOAPIC pins 0..24 + PIC)
        // route-identical to KVM's implicit default, so the legacy INTx routes
        // survive the first MSI-X `KVM_SET_GSI_ROUTING` (a whole-table replace
        // that would otherwise wipe them). `None` on split-irqchip (routes go via
        // the userspace IOAPIC / `ioapic_handle`) and on non-PCI guests.
        #[cfg(target_arch = "x86_64")]
        let full_route_owner: Option<Arc<kvm::FullIrqchipRouteOwner>> = (!vm.split_irqchip
            && vm.pci_enabled)
            .then(|| Arc::new(kvm::FullIrqchipRouteOwner::new(vm.vm_fd.as_raw_fd())));
        #[cfg(target_arch = "x86_64")]
        if let Some(owner) = full_route_owner.as_ref() {
            owner
                .install_defaults()
                .context("install full-irqchip default GSI routing")?;
        }

        // PCI host bridge handle for the virtio-PCI transport: the single-bus
        // PCIe segment with ECAM/CAM config access, constructed only when this
        // VM enables PCI. `None` keeps non-PCI guests byte-identical (no
        // ECAM/CAM dispatch). Created here in the same scope as the other
        // device handles so it threads into both the BSP loop and the per-AP
        // spawn. x86-only for now (aarch64 PCI is design-only).
        #[cfg(target_arch = "x86_64")]
        let pci_bus_handle: Option<Arc<PiMutex<crate::vmm::pci::PciBus>>> = if vm.pci_enabled {
            Some(Arc::new(PiMutex::new(crate::vmm::pci::PciBus::new(
                kvm::PCI_ECAM_BASE,
                kvm::PCI_ECAM_SIZE,
            ))))
        } else {
            None
        };
        #[cfg(not(target_arch = "x86_64"))]
        let pci_bus_handle: Option<Arc<PiMutex<crate::vmm::pci::PciBus>>> = None;

        // Register serial EventFds with KVM's irqfd for interrupt-driven TX.
        // On x86 split-irqchip (>254 APIC IDs) the serial IRQ routes through
        // the userspace IOAPIC (ioapic_handle above is threaded into the run
        // loops; the guest's RTE write installs the MSI route); on the
        // in-kernel-irqchip (x86 <=254) and aarch64 (GIC) paths the kernel
        // routes the GSI directly.
        #[cfg(target_arch = "x86_64")]
        {
            vm.vm_fd
                .register_irqfd(com1.lock().irq_evt(), console::COM1_IRQ)
                .context("register COM1 irqfd")?;
            vm.vm_fd
                .register_irqfd(com2.lock().irq_evt(), console::COM2_IRQ)
                .context("register COM2 irqfd")?;
        }
        #[cfg(target_arch = "aarch64")]
        {
            vm.vm_fd
                .register_irqfd(com1.lock().irq_evt(), kvm::SERIAL_IRQ)
                .context("register serial irqfd")?;
            vm.vm_fd
                .register_irqfd(com2.lock().irq_evt(), kvm::SERIAL2_IRQ)
                .context("register serial2 irqfd")?;
        }

        // Serialises on-demand captures against themselves: the
        // coordinator sets this Acquire-bool while a TLV-driven
        // snapshot dispatch runs and clears it on completion, so a
        // user-watchpoint hit firing during a CAPTURE-class request
        // does not open a second concurrent capture window. The TX
        // handler is single-threaded on the freeze coord, so the
        // gate's primary defence is against the user-watchpoint
        // dispatcher (which runs in the same iteration body after
        // pending TLV requests drain). Independent of `freeze_state`,
        // which governs only the error-class trigger machine —
        // on-demand captures must service even when
        // `freeze_state == Done` so post-failure `Op::CaptureSnapshot` calls
        // still work.
        let on_demand_in_flight = Arc::new(AtomicBool::new(false));

        // Host-side snapshot bridge. Owned by the freeze coordinator
        // and exposed back through `VmRunState` so test code can
        // drain captured reports after the VM exits. The bridge's
        // capture callback returns `None` — the coordinator never
        // calls `bridge.capture()`; instead it runs
        // `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })` directly and stores the
        // resulting report via `bridge.store(name, report)` so the
        // host owns the entire capture pipeline.
        // Allocate the accessor-init worker's publish-seqno + worker-
        // state atomics here so the snapshot bridge below adopts the
        // same Arc the worker (the `vmm-accessor-init` worker,
        // spawned much later) bumps on
        // every successful publish / state transition. The dispatch
        // for `Op::ReplaceScheduler` / `Op::AttachScheduler` reads the
        // seqno via the thread-local bridge and waits for it to
        // advance, gating the op on the new scheduler's BPF accessor
        // pair landing — see
        // [`crate::scenario::snapshot::SnapshotBridge::wait_for_accessor_publish_advance`].
        let accessor_publish_seqno: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let accessor_worker_state: Arc<std::sync::atomic::AtomicU8> =
            Arc::new(std::sync::atomic::AtomicU8::new(
                crate::scenario::snapshot::bridge::accessor_worker_state::TRYING,
            ));
        // Dispatcher wake EventFd — pulsed by the worker on every
        // seqno bump AND on FAILED_PERMANENTLY exit so the bridge's
        // wait paths react at kernel-scheduling-tick latency instead
        // of the 50 ms sleep tail an atomic-only loop would carry.
        // EFD_NONBLOCK so a saturating writer never blocks; the
        // atomic is the source of truth, the fd is just a wake.
        // Distinct from `accessor_ready_evt` to avoid the dispatcher
        // racing the coord's epoll loop for the same wake count.
        let accessor_dispatcher_wake_evt: Arc<EventFd> =
            Arc::new(EventFd::new(EFD_NONBLOCK).expect("eventfd for accessor_dispatcher_wake"));
        let snapshot_bridge = {
            let cb: crate::scenario::snapshot::CaptureCallback = Arc::new(|_| None);
            crate::scenario::snapshot::SnapshotBridge::new(cb).with_accessor_state(
                accessor_publish_seqno.clone(),
                accessor_worker_state.clone(),
                accessor_dispatcher_wake_evt.clone(),
            )
        };

        // Probes-ready EventFd. Used solely by the bpf-map-write thread's
        // phase 1 as a 200 ms `poll(POLLIN)` backoff while it polls guest
        // kernel state (.bss latch reads); both bpf phases write it on
        // success (phase 2 polls `kill_evt` instead — the fd goes
        // level-high after the first write, so re-polling it would spin).
        // `EFD_NONBLOCK` keeps `write()` from stalling at saturation, and
        // readers `poll` (never `read`) so the level stays high once any
        // writer fires. Moved directly into `start_bpf_map_write` (its only
        // consumer), so no clone is needed.
        let probes_ready_evt = EventFd::new(EFD_NONBLOCK).context("create probes-ready EventFd")?;

        // Shared parked_evt: every vCPU thread + the virtio-blk
        // worker writes 1 to this counter-mode EventFd immediately
        // after its respective `parked.store(true, Release)` /
        // `paused.store(true, Release)`. The freeze coordinator's
        // rendezvous loop polls this fd alongside kill_evt and
        // bsp_done_evt instead of spin-sleeping on a 100µs cadence.
        // EFD_NONBLOCK so a writer never stalls; counter mode (no
        // EFD_SEMAPHORE) so a single drain consumes any number of
        // coalesced parked signals — the coordinator drains once
        // and re-checks every parked flag.
        //
        // Allocated BEFORE init_virtio_blk so we can plumb the fd
        // into the device's `set_parked_evt` setter immediately
        // after construction, before the worker spawns and observes
        // its first pause.
        let parked_evt = Arc::new(EventFd::new(EFD_NONBLOCK).context("create parked EventFd")?);
        // Shared thaw_evt: written by the freeze coordinator after
        // `freeze.store(false, Release)` so every parked vCPU
        // observes the thaw within microseconds rather than waiting
        // up to 10ms on `park_timeout`. Same EFD_NONBLOCK + counter
        // semantics as parked_evt.
        let thaw_evt = Arc::new(EventFd::new(EFD_NONBLOCK).context("create thaw EventFd")?);

        // virtio-blk is set up AFTER the PCI bus + MSI-X route sink (below),
        // because its x86_64 transport is a virtio-pci function that needs both.
        // See the arch-split block following the virtio-net setup. (Like
        // virtio-net, it is constructed before vm.vcpus is torn down so the
        // helper can still read `vm.guest_mem` and register irqfds on
        // `vm.vm_fd`.)

        // Optional virtio-net. The transport is arch-split:
        //
        // - x86_64: the NIC is a virtio-pci function installed on
        //   `pci_bus_handle` (slot 1). There is no MMIO handle for the
        //   dispatch loops to drive — guest BAR accesses route through
        //   the PCI bus — so `virtio_net` stays `None` here (the MMIO
        //   net arm in the dispatch loops goes inert), and the counters
        //   Arc + INTx resample eventfd come back in `NetDeviceHandles`.
        //   The resample fd MUST outlive the run (KVM holds its raw fd
        //   to de-assert the level GSI on guest EOI); `_net_resample_evt`
        //   keeps it alive through `run_bsp_loop`.
        // - aarch64: the NIC stays on virtio-MMIO (aarch64 PCI is a
        //   later increment); the handle drives the dispatch loops and
        //   supplies the counters Arc.
        //
        // Same construction-before-vcpu-takedown rule as virtio-blk.
        #[cfg(target_arch = "x86_64")]
        let virtio_net: Option<Arc<PiMutex<virtio_net::VirtioNet>>> = None;
        // The active GSI-route owner as the MSI-X route sink: `IoapicHandle` on
        // split-irqchip, `FullIrqchipRouteOwner` on full (both impl
        // `MsixRouteSink`). Threaded into `init_virtio_net_pci` so MSI-X is
        // offered on both irqchip paths; cloned (Arc) so each owner stays
        // available for the run loops + teardown routing diagnostics.
        #[cfg(target_arch = "x86_64")]
        let msix_sink: Option<Arc<dyn virtio_msix::MsixRouteSink>> = if vm.split_irqchip {
            ioapic_handle
                .clone()
                .map(|h| h as Arc<dyn virtio_msix::MsixRouteSink>)
        } else {
            full_route_owner
                .clone()
                .map(|o| o as Arc<dyn virtio_msix::MsixRouteSink>)
        };
        // One (counters, resample_evt) pair per installed NIC. The resample fds
        // (Some only on the full-irqchip path) are held alive in `_net_resample_evts`
        // for the run (KVM holds each raw fd to de-assert its level GSI on EOI);
        // the counters are snapshotted+aggregated into VmResult at collect time.
        #[cfg(target_arch = "x86_64")]
        let (virtio_net_counters, _net_resample_evts): (Vec<_>, Vec<_>) =
            match pci_bus_handle.as_ref() {
                Some(bus) => self
                    .init_virtio_net_pci(&vm, bus, msix_sink.clone())?
                    .into_iter()
                    .map(|h| (h.counters, h.resample_evt))
                    .unzip(),
                None => (Vec::new(), Vec::new()),
            };
        #[cfg(not(target_arch = "x86_64"))]
        let virtio_net = self.init_virtio_net(&vm)?;
        #[cfg(not(target_arch = "x86_64"))]
        let virtio_net_counters: Vec<_> = virtio_net
            .as_ref()
            .map(|d| d.lock().counters())
            .into_iter()
            .collect();

        // Optional virtio-blk. Transport is arch-split (same pattern as
        // virtio-net above), but virtio-blk runs a request worker the freeze
        // coordinator pauses, so the run loop needs the device itself — not just
        // counters. Two handles result:
        //  - `virtio_blk`: the MMIO-dispatch routing handle. `None` on x86 (the
        //    device is a PCI function; guest BAR accesses route through the PCI
        //    bus, so the dispatch loops' MMIO blk arm goes inert), `Some` on
        //    aarch64 (the MMIO transport).
        //  - `blk_device`: the canonical device Arc for the freeze
        //    pause/resume, counters, paused-handle, and parked-eventfd — set on
        //    BOTH arches (on x86 it is the `BlkDeviceHandles::device` clone whose
        //    sibling backs the PciBus function; on aarch64 it is the same Arc as
        //    `virtio_blk`).
        // On x86 the INTx resample eventfd is held alive for the run (KVM holds
        // its raw fd to de-assert the level GSI on guest EOI).
        // (mmio_routing_handle, canonical_device, intx_resample_evt). The inner
        // types are `Option<Arc<PiMutex<VirtioBlk>>>` / `Option<EventFd>`,
        // inferred from the arms + later use (a written-out tuple trips
        // clippy::type_complexity, matching the `(Vec<_>, Vec<_>)` style the NIC
        // counters use above).
        #[cfg(target_arch = "x86_64")]
        let (virtio_blk, blk_device, _blk_resample_evt): (Option<_>, Option<_>, Option<_>) =
            match pci_bus_handle.as_ref() {
                Some(bus) => {
                    match self.init_virtio_blk_pci(&vm, bus, msix_sink, effective_no_perf_cpus)? {
                        Some(h) => (None, Some(h.device), h.resample_evt),
                        None => (None, None, None),
                    }
                }
                None => (None, None, None),
            };
        #[cfg(not(target_arch = "x86_64"))]
        let (virtio_blk, blk_device): (Option<_>, Option<_>) = {
            let dev = self.init_virtio_blk(&vm, effective_no_perf_cpus)?;
            (dev.clone(), dev)
        };
        // Plumb the shared parked_evt into the device (both transports) so its
        // worker wakes the freeze coordinator's rendezvous on park. Lands before
        // the deferred initial worker spawn (DRIVER_OK, after vCPUs start), so
        // the first worker observes it.
        if let Some(ref blk) = blk_device {
            blk.lock().set_parked_evt(parked_evt.clone());
        }

        // Virtio-console for host→guest wake delivery. The setup_memory
        // path always emits the device's MMIO node on the kernel
        // cmdline (x86_64) / FDT (aarch64), so the kernel's
        // `virtio_mmio` driver probes for the device unconditionally.
        // The guest's `hvc0_poll_loop` blocks on `/dev/hvc0` and wakes
        // within microseconds when the host pushes a byte. The
        // coordinator and watchdog use this as the host→guest signal
        // channel: the monitor pushes `SIGNAL_VC_DUMP` for SysRq-D
        // dump requests (the dispatch is wake-byte-only — no SHM
        // control byte), the watchdog pushes `SIGNAL_VC_SHUTDOWN` for
        // graceful shutdown, and the bpf-map-write thread pushes
        // `SIGNAL_BPF_WRITE_DONE` to release `wait_for_map_write`.
        let mut vc = virtio_console::VirtioConsole::new();
        vc.set_mem((*vm.guest_mem).clone());
        let virtio_con = Arc::new(PiMutex::new(vc));
        // Register the virtio-console irqfd. On x86 split-irqchip (>254 APIC
        // IDs) the route is installed by the userspace IOAPIC when the guest
        // programs the RTE — the irqfd is inert until then, and
        // kvm_irq_routing_update rebinds it on the KVM_SET_GSI_ROUTING that
        // the RTE write triggers. On the in-kernel-irqchip (x86 <=254) and
        // aarch64 (GICv3) paths the kernel routes the GSI directly.
        // register_irqfd works in all three modes (irqchip_in_kernel).
        vm.vm_fd
            .register_irqfd(virtio_con.lock().irq_evt(), kvm::VIRTIO_CONSOLE_IRQ)
            .context("register virtio-console irqfd")?;

        let kill = Arc::new(AtomicBool::new(false));
        // Watchdog-set timeout flag. Distinct from `kill` because
        // `kill` flips on every shutdown path (BSP shutdown, AP
        // panic, watchdog hard timeout) and the consumer
        // (`VmResult::timed_out`) only wants to know when the
        // watchdog fired its hard-deadline branch. The watchdog
        // thread sets this to `true` ONLY on the
        // `Instant::now() >= effective_deadline` arm; the BSP
        // reads it post-loop and the resulting `timed_out` element
        // of `run_bsp_loop`'s return tuple flows through
        // `VmRunState::timed_out` → `VmResult::timed_out`.
        let timed_out_flag = Arc::new(AtomicBool::new(false));
        // Wake fd paired with the `kill` AtomicBool. Setters that
        // flip `kill` (run_vm post-BSP-exit, vCPU shutdown classifier,
        // panic hook) ALSO write to this EventFd so any consumer
        // sleeping on `epoll_wait` returns within microseconds of
        // the flip rather than waiting up to one full poll
        // interval. Production consumers: the monitor loop and the
        // watchdog thread, both spawned below. `EFD_NONBLOCK` keeps
        // the writer's `write()` from stalling if the counter is
        // already saturated; the AtomicBool remains the source of
        // truth — the EventFd is purely a wake signal.
        let kill_evt = Arc::new(EventFd::new(EFD_NONBLOCK).context("create kill EventFd")?);
        // Boot-complete eventfd. Fired by the freeze coordinator
        // when the guest publishes a CRC-valid
        // [`crate::vmm::wire::MSG_TYPE_SYS_RDY`] TLV frame on the
        // virtio-console bulk port. The monitor thread's pre-sample
        // `epoll_wait` registers this fd alongside `kill_evt` and
        // a 5 s timeout — the SYS_RDY frame is the explicit
        // boot-complete signal from the guest's userspace init,
        // sent after `mount_filesystems()` so by the time the
        // monitor wakes the kernel-side prerequisites
        // (`__per_cpu_offset[]` populated by `setup_per_cpu_areas`,
        // `page_offset_base` populated by KASLR randomization) are
        // already met. Replaces an earlier port-0-TX trigger that
        // depended on incidental console traffic. `EFD_NONBLOCK`
        // because the only writer is the coordinator's TLV dispatch
        // and the only reader is the monitor's `epoll_wait`; a
        // stuck or saturated counter is harmless because the wake
        // semantics are level-triggered. Surfaced as a `warn`
        // rather than a hard failure so a kernel without eventfd
        // support (extremely unlikely for KVM-capable hosts) still
        // boots — the monitor will fall through its 5 s timeout
        // without a guest signal.
        let sys_rdy_evt: Option<Arc<EventFd>> = match EventFd::new(EFD_NONBLOCK) {
            Ok(evt) => Some(Arc::new(evt)),
            Err(e) => {
                tracing::warn!(
                    err = %e,
                    "failed to create sys_rdy EventFd; \
                     monitor will not gate on guest-boot signal"
                );
                None
            }
        };
        // Failure-dump freeze rendezvous: broadcast `freeze` flag plus a
        // per-vCPU `parked` ACK, parallel to the existing `kill` +
        // `exited` shutdown rendezvous. The freeze coordinator
        // (spawned below alongside the watchdog) polls the BPF probe's
        // `ktstr_err_exit_detected` .bss flag via `BpfMapAccessor`;
        // when the flag flips it sets `freeze`, kicks every vCPU,
        // awaits N-of-N parked confirmations, runs the dump (placeholder
        // in this batch), and then clears `freeze` to thaw.
        let freeze = Arc::new(AtomicBool::new(false));
        // Scheduler-stats client. Constructed only when the run
        // has a scheduler attached — without a scheduler there is
        // nothing to query, and spawning a drainer thread plus
        // plumbing a client onto `VmResult` would force every test
        // that does `stats_client.unwrap().stats(...)` to wait for
        // its full timeout before discovering "no scheduler". When
        // `scheduler_binary` is `None`, the field on
        // [`VmResult::stats_client`] stays `None` and callers can
        // branch on `.is_none()` to skip the stats path entirely.
        let stats_client = if self.scheduler_binary.is_some() {
            Some(
                crate::vmm::sched_stats::SchedStatsClient::new(
                    virtio_con.clone(),
                    Some(freeze.clone()),
                    // Run-wide kill flag plumbed as the cancel
                    // signal: when the BSP / watchdog flips
                    // `kill`, blocked `request_raw` calls wake and
                    // return `Cancelled` instead of hanging forever.
                    // The host watchdog is the only "timeout" in
                    // the stats path.
                    Some(kill.clone()),
                    // Paired wake fd: the drainer's epoll watches
                    // `kill_evt` so the cancel edge propagates to
                    // a blocked cvar wait within microseconds.
                    Some(kill_evt.clone()),
                )
                .context("construct scheduler-stats client")?,
            )
        } else {
            None
        };
        // Hardware data-write watchpoint state shared between the
        // freeze coordinator (publishes the resolved
        // `*scx_root->exit_kind` KVA into `request_kva`) and every
        // vCPU thread (self-arms when `request_kva` changes; sets
        // `hit` on `KVM_EXIT_DEBUG`). See [`WatchpointArm`] for the
        // full protocol; this Arc is the only carrier and outlives
        // every consumer (the coordinator joins before the vCPU
        // teardown drops the kvm_run mmaps).
        let watchpoint =
            Arc::new(WatchpointArm::new().context("create WatchpointArm.hit_evt EventFd")?);
        let bsp_parked = Arc::new(AtomicBool::new(false));
        let bsp_regs: Arc<std::sync::Mutex<Option<exit_dispatch::VcpuRegSnapshot>>> =
            Arc::new(std::sync::Mutex::new(None));

        let has_immediate_exit = vm.has_immediate_exit;
        let mut vcpus = std::mem::take(&mut vm.vcpus);
        let mut bsp = vcpus.remove(0);

        // Build per-vCPU pin targets from the stored pinning plan.
        // Index i holds the host CPU for vCPU i. BSP is index 0.
        let pin_targets: Vec<Option<usize>> = if let Some(plan) = effective_pinning_plan {
            let total = self.topology.total_cpus() as usize;
            let mut targets = vec![None; total];
            for &(vcpu_id, host_cpu) in &plan.assignments {
                if (vcpu_id as usize) < total {
                    targets[vcpu_id as usize] = Some(host_cpu);
                }
            }
            targets
        } else {
            Vec::new()
        };

        // AP pin targets: indices 1..N.
        let ap_pins: Vec<Option<usize>> = if pin_targets.len() > 1 {
            pin_targets[1..].to_vec()
        } else {
            vec![None; vcpus.len()]
        };

        // No-perf + --cpu-cap: flat CPU list from the LLC plan gets
        // sched_setaffinity'd on every vCPU thread as a mask (not a
        // hard pin). Mutually exclusive with perf-mode's pin_targets.
        // The run-time replan's `effective_no_perf_cpus` wins over the
        // build-time `no_perf_plan.cpus`: they name the LLCs the
        // run-scoped flocks actually hold, whereas the build-time plan
        // may have Spread-planned against then-truthful-now-stale holder
        // counts. Falls through to `no_perf_plan` (interactive path,
        // where no replan runs) and finally `default_cpu_mask`
        // (overcommit).
        let no_perf_mask: Option<&[usize]> = effective_no_perf_cpus
            .or(self.no_perf_plan.as_ref().map(|p| p.cpus.as_slice()))
            .or(default_cpu_mask);

        // Per-AP TID slots — each AP thread stamps gettid() into its
        // `AtomicI32` and fires the paired `Latch` at startup so the
        // monitor can open per-vCPU `perf_event_open` counters bound
        // to the right thread. Index = AP index (0-based among APs);
        // the BSP TID is stamped into a separate slot below since it
        // runs on the current thread. The latch lets the
        // perf-capture path block in `Latch::wait_timeout` instead
        // of sleep-polling the atomic — see
        // [`open_vcpu_perf_capture`].
        let ap_tid_slots: Vec<(Arc<AtomicI32>, Arc<crate::sync::Latch>)> = (0..vcpus.len())
            .map(|_| {
                (
                    Arc::new(AtomicI32::new(0)),
                    Arc::new(crate::sync::Latch::new()),
                )
            })
            .collect();

        // Per-AP boot-ordering latches — each AP fires its latch at the
        // instant it is about to enter `vcpu_run_loop_unified` (after
        // signal-handler registration, affinity, and RT-prio setup,
        // immediately before its first KVM_RUN). The BSP blocks on all of
        // them below before it starts executing guest code.
        //
        // Why this gate exists: the guest kernel's `do_boot_cpu` brings APs
        // up strictly sequentially, INIT-SIPI'ing each one and then waiting a
        // bounded ~10s for it to check in before moving on. KVM buffers the
        // INIT/SIPI for a vCPU already blocked in KVM_RUN with
        // MP_STATE_UNINITIALIZED, so an AP that is inside its run loop cannot
        // miss its wakeup. But `spawn_ap_threads` only *creates* the host
        // threads and returns — on an oversubscribed host an AP host-thread
        // that the scheduler hasn't yet run into KVM_RUN when its INIT-SIPI
        // arrives misses the window, and the guest marks that CPU
        // present-but-offline (observed in CI as 128-vCPU guests
        // intermittently losing 1-2 mid-range CPUs). Gating guest boot on
        // every AP being in KVM_RUN closes that race. One-shot `Latch` per AP
        // (many producers, single BSP waiter) — reusing the module's existing
        // primitive rather than a new counter type.
        let ap_boot_latches: Vec<Arc<crate::sync::Latch>> = (0..vcpus.len())
            .map(|_| Arc::new(crate::sync::Latch::new()))
            .collect();

        // BSP-done signal pair, hoisted above the first thread spawn so the
        // `RunVmThreadGuard` below can carry it (its Drop signals `bsp_done` so
        // the freeze coordinator takes its clean teardown exit — see the guard's
        // `Drop`) and so its fallible `EventFd` alloc is no longer a post-spawn `?`
        // early-return window. `bsp_done` flips true when the BSP exits its run
        // loop; `bsp_done_evt` (EFD_NONBLOCK, so a doubled write from the panic
        // hook + post-loop store cannot stall) wakes the coordinator's epoll.
        let bsp_done = Arc::new(AtomicBool::new(false));
        let bsp_done_evt = Arc::new(EventFd::new(EFD_NONBLOCK).context("create bsp_done EventFd")?);

        let (ap_threads, ap_freeze_handles) = self.spawn_ap_threads(
            vcpus,
            has_immediate_exit,
            &com1,
            &com2,
            Some(&virtio_con),
            virtio_blk.as_ref(),
            virtio_net.as_ref(),
            ioapic_handle.as_ref(),
            pci_bus_handle.as_ref(),
            &kill,
            &kill_evt,
            &freeze,
            &watchpoint,
            &ap_pins,
            no_perf_mask,
            &ap_tid_slots,
            &ap_boot_latches,
            Some(&parked_evt),
            Some(&thaw_evt),
        )?;

        // UAF guard: from the first spawn onward, any `?` early-return or panic
        // between here and the Ok teardown must join every spawned thread before
        // `vm`/`guest_mem` drops (the threads hold bare raw pointers into
        // guest_mem). The guard owns `ap_threads` (borrowed below via
        // `guard.ap_threads`) and accumulates the monitor / bpf-write /
        // coordinator / watchdog handles as they spawn; the Ok path `disarm`s it.
        // Declared after `bsp` (above) so it joins the watchdog + coordinator
        // (which hold bsp's ImmediateExitHandle) before `bsp` drops.
        //
        // Snapshot the BSP thread's affinity BEFORE the mask narrowing below,
        // and declare the restore guard BEFORE `RunVmThreadGuard` so — Rust
        // dropping locals in reverse declaration order — the thread-join guard
        // drops FIRST (joining every AP) and this guard restores affinity
        // AFTER, leaving `run_vm` affinity-neutral for any retry/next VM in the
        // same process. See [`BspAffinityGuard`] for why the leak matters.
        let _bsp_affinity_guard = BspAffinityGuard::capture();
        let mut guard = RunVmThreadGuard {
            ap_threads,
            monitor: None,
            bpf_write: None,
            freeze_coord: None,
            watchdog: None,
            kill: kill.clone(),
            kill_evt: kill_evt.clone(),
            freeze: freeze.clone(),
            bsp_done: bsp_done.clone(),
            bsp_done_evt: bsp_done_evt.clone(),
        };

        // Pin / mask BSP (runs on current thread, pid=0 means calling thread).
        if let Some(Some(host_cpu)) = pin_targets.first() {
            pin_current_thread(*host_cpu, "BSP (vCPU 0)");
        } else if let Some(mask) = no_perf_mask {
            set_thread_cpumask(mask, "BSP (vCPU 0)");
        }
        if self.performance_mode {
            set_rt_priority(1, "BSP (vCPU 0)");
        }

        // Collect vCPU pthread_t handles for monitor stuck detection.
        // BSP runs on the current thread; APs have spawned threads.
        let vcpu_pthreads = {
            let mut pts = Vec::with_capacity(1 + guard.ap_threads.len());
            pts.push(unsafe { libc::pthread_self() } as libc::pthread_t);
            for vt in &guard.ap_threads {
                pts.push(vt.handle.as_pthread_t() as libc::pthread_t);
            }
            pts
        };

        // Build the per-vCPU TID vec the monitor needs for
        // `perf_event_open(2)`. Index 0 is the BSP — running on this
        // thread, so SYS_gettid here returns the current thread's
        // TID. Indexes 1..n are AP slots stamped by each AP thread at
        // startup. Slots may still be 0 here if an AP hasn't reached
        // its tid_slot.store; the monitor polls them with a deadline
        // before opening counters and skips per-vCPU perf for any
        // slot still 0 at the deadline.
        let bsp_tid_slot = Arc::new(AtomicI32::new(unsafe {
            libc::syscall(libc::SYS_gettid) as i32
        }));
        // BSP latch is pre-set so `open_vcpu_perf_capture` returns
        // immediately for index 0 — the BSP TID is stamped
        // synchronously above on this very thread.
        let bsp_latch = Arc::new(crate::sync::Latch::new());
        bsp_latch.set();
        let vcpu_tid_slots: Vec<(Arc<AtomicI32>, Arc<crate::sync::Latch>)> =
            std::iter::once((bsp_tid_slot, bsp_latch))
                .chain(ap_tid_slots.iter().cloned())
                .collect();
        // AP proc task directories disappear at thread return. Each AP keeps
        // one final self-schedstat snapshot in its VcpuThread handle; align
        // those slots with `vcpu_tid_slots`. The BSP remains live through
        // witness finalization, so its fallback slot is intentionally empty.
        let vcpu_exit_schedstats = std::iter::once(Arc::new(std::sync::Mutex::new(None)))
            .chain(
                guard
                    .ap_threads
                    .iter()
                    .map(|thread| Arc::clone(&thread.schedstat_at_exit)),
            )
            .collect::<Vec<_>>();

        // Open per-vCPU `perf_event_open` counters once at run-vm
        // scope so both the monitor thread (per-tick timeline) and
        // the freeze coordinator (freeze-instant snapshot) can read
        // through a shared `Arc`. Polling vCPU TIDs here (rather than
        // inside the monitor closure) lets the freeze coord see a
        // consistent capture immediately when the latch fires —
        // before the monitor has even taken its first sample. AP
        // threads stamp their TID into the slots before they enter
        // KVM_RUN; BSP slot is stamped synchronously above.
        // `Arc<Option<...>>` lets a host that lacks
        // `perf_event_open` permission still run the rest of the
        // dump pipeline; the inner Option is None and every
        // consumer's `as_ref()` chain produces None for that field.
        let perf_capture = Arc::new(open_vcpu_perf_capture(&vcpu_tid_slots));

        // aarch64 TCR_EL1 cache. Populated by the BSP loop on first
        // successful read post-MMU-bringup. `None` on x86_64 (the
        // register does not exist there). Threads that build a
        // `GuestKernel` for page-table walks (monitor, BPF map
        // writer, freeze coordinator's scan_tick path,
        // collect_verifier_stats) load this atomic.
        #[cfg(target_arch = "aarch64")]
        let tcr_el1_cache: Option<Arc<std::sync::atomic::AtomicU64>> =
            Some(Arc::new(std::sync::atomic::AtomicU64::new(0)));
        #[cfg(target_arch = "x86_64")]
        let tcr_el1_cache: Option<Arc<std::sync::atomic::AtomicU64>> = None;

        // CR3 (x86_64) / TTBR1_EL1 (aarch64) cache. Populated lazily
        // by the BSP loop after the kernel has established its
        // initial page tables. Used by host-side `GuestKernel`
        // constructions to walk the page tables for `phys_base`
        // resolution — see [`crate::monitor::symbols::resolve_phys_base`].
        // `0` is the bootstrap value; readers tolerate it (the walk
        // fails and `phys_base` falls back to `0`, which is correct
        // on non-KASLR boots).
        let cr3_cache: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        // Scheduler-attach watchdog reset. Shared `AtomicU64`
        // written once by the host monitor when it observes
        // `*scx_root` transition from null to non-null in guest
        // memory (a scheduler attached); read each tick by the
        // watchdog so the hard deadline resets to attach moment +
        // `self.workload_duration` instead of being counted from
        // VM boot. `0` (the default) is the "no reset requested"
        // sentinel — the watchdog ignores it and keeps using the
        // original `timeout`-derived deadline. The reset CAN extend
        // past the original deadline (no min clamp) so boot-time
        // delays do not eat into the workload budget. Defined ahead of
        // `start_monitor` so the monitor closure can capture a
        // clone; the watchdog clone is taken below at the
        // watchdog setup site.
        let watchdog_reset_ns: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        // Parallel provenance tag for `watchdog_reset_ns` — records
        // which of the four writers last armed the reset deadline so the
        // watchdog dump can name it (see [`WatchdogResetTag`]). Cloned to
        // each writer alongside the ns atomic; the two are always stored
        // back-to-back.
        let watchdog_reset_tag: Arc<std::sync::atomic::AtomicU8> = Arc::new(
            std::sync::atomic::AtomicU8::new(WatchdogResetTag::Unset as u8),
        );
        // Lock-free progress/liveness ledger. The monitor thread writes
        // it every tick (heartbeat, cpu_ns, progress epochs); the VM
        // watchdog and freeze coordinator will read it in LATER commits
        // to distinguish a genuinely wedged guest from one that is idle
        // or still making scheduling progress. Declared here beside
        // `watchdog_reset_tag` and cloned into `start_monitor` so the
        // monitor closure owns a producer handle; consumers are wired
        // later (this commit is observability only, so the Arc has a
        // single clone — no unused-binding warnings).
        let progress_ledger: Arc<monitor::ProgressLedger> =
            Arc::new(monitor::ProgressLedger::default());
        let kern_phys_base: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let kern_phys_base_evt = Arc::new(EventFd::new(0).expect("eventfd for kern_phys_base"));
        // Derived virt-KASLR offset (biased `+1`). Populated by
        // either the BSP-side MSR_LSTAR readback
        // (`src/vmm/x86_64/msr_kaslr::read_and_derive` invoked from
        // `run_bsp_loop`) or the guest-channel KERN_ADDRS
        // dispatcher (`src/vmm/freeze_coord/dispatch.rs`'s
        // KERN_ADDRS arm). Both paths CAS-publish the same offset
        // so first-writer wins; the dual sourcing is defense in
        // depth — KASLR-on is the default state and both publishers
        // produce non-zero values on the default path. Tests that
        // explicitly opt out (`#[ktstr_test(kaslr = false)]` adding
        // `nokaslr` through `runtime::build_cmdline_extra`, OR
        // `Scheduler::kargs(&["nokaslr"])` operator escape) keep the
        // offset at 0 for the duration of that test run. Consumers
        // `.load(Acquire)` and subtract 1; the 0 sentinel means "no
        // path has published yet" OR "published as 0 (nokaslr)" —
        // both produce the same observable behaviour (per-CPU
        // template arithmetic in `monitor::symbols::per_cpu_kva`
        // lands on the compile-time base).
        let kern_virt_kaslr: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let kern_virt_kaslr_evt = Arc::new(EventFd::new(0).expect("eventfd for kern_virt_kaslr"));
        // Link-time KVAs from vmlinux for the two virt-KASLR
        // derivation paths. Resolved once here so both the dispatch
        // sinks (the KERN_ADDRS arm subtracts `_text` to derive)
        // and `run_bsp_loop` (the BSP MSR_LSTAR path subtracts
        // `entry_SYSCALL_64`) share a single source of truth.
        // `find_vmlinux` is cheap path discovery; the
        // `cached_vmlinux_artifacts` hit is shared with the monitor's
        // later `start_monitor` lookup so the host pays the ELF read
        // AND the symbol/BTF parse at most once per (path, mtime) per
        // process. Both fields land 0 when the symbol is
        // absent — `_text` only on extremely stripped vmlinux,
        // `entry_SYSCALL_64` on aarch64 and non-x86_64 builds.
        // Both paths short-circuit on a 0 link KVA, leaving the
        // shared Arc at 0 (matches the KASLR-off semantics the
        // nokaslr karg also produces). KASLR shifts every text
        // symbol by the same `kaslr_offset`, so the two link
        // KVAs back distinct derivations that produce identical
        // offsets — guaranteeing the two writers don't race on
        // different values.
        let host_vmlinux_path: Option<std::path::PathBuf> = find_vmlinux(&self.kernel);
        let host_vmlinux_artifacts = host_vmlinux_path
            .as_ref()
            .and_then(|p| super::vmlinux::cached_vmlinux_artifacts(p));
        let host_kernel_symbols: Option<crate::monitor::symbols::KernelSymbols> =
            host_vmlinux_artifacts.as_ref().map(|a| a.symbols.clone());
        // This vmlinux's GNU build-id — compared in the KERN_BUILD_ID
        // dispatch arm against the booted kernel's build-id to catch a
        // stale/mismatched cache entry before its symbols/offsets
        // silently mis-read guest memory. `None` (no note) disables the
        // check.
        let host_vmlinux_build_id: Option<Vec<u8>> = host_vmlinux_artifacts
            .as_ref()
            .and_then(|a| a.build_id.clone());
        let kernel_text_link_kva: u64 = host_kernel_symbols
            .as_ref()
            .and_then(|s| s.kernel_text_kva)
            .unwrap_or(0);
        let entry_syscall_64_link_kva: u64 = host_kernel_symbols
            .as_ref()
            .and_then(|s| s.entry_syscall_64_kva)
            .unwrap_or(0);
        let accessor_ready_evt = Arc::new(EventFd::new(0).expect("eventfd for accessor_ready"));

        // TID handles for the monitor's per-phase schedstat witness — one
        // per `vcpu_tid_slots` entry (index 0 BSP, 1.. APs), read live each
        // tick. Cloned (not moved) so the teardown whole-run schedstat read
        // still has `vcpu_tid_slots`.
        let vcpu_tid_atomics: Vec<Arc<AtomicI32>> = vcpu_tid_slots
            .iter()
            .map(|(slot, _)| slot.clone())
            .collect();
        // Per-phase host-contention recorder. It owns the live TID handles for
        // rare lifecycle-boundary schedstat sweeps and a persistent O(1) CPU
        // PSI reader sampled from the monitor's existing timer wake.
        let contention_recorder = Arc::new(ContentionWitnessRecorder::new(
            vcpu_tid_atomics,
            vcpu_exit_schedstats.clone(),
            run_start,
        ));
        // KERN_ADDRS observability counters (frames consumed / CRC-failed),
        // incremented by the coordinator's dispatch arm and read by the
        // monitor's pre-latch diagnostic (see `RqRefresh::kern_addrs_frames`):
        // a dead-channel run self-reports "publish never arrived" (0) vs
        // "arrived but a derive gate rejected it" (non-zero). Created here,
        // before the monitor spawn, so both consumers share one instance.
        let kern_addrs_frames: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let kern_addrs_crc_bad: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let kern_addrs_frames_for_coord = kern_addrs_frames.clone();
        let kern_addrs_crc_bad_for_coord = kern_addrs_crc_bad.clone();
        // Expected (vmlinux) build-id moved into the coord dispatch
        // closure for the KERN_BUILD_ID consistency check.
        let host_vmlinux_build_id_for_coord = host_vmlinux_build_id.clone();
        // Unknown-msg_type frame counter (the header-outside-CRC probe —
        // see `BulkDispatchSinks::unknown_type_frames`).
        let unknown_type_frames: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let unknown_type_frames_for_coord = unknown_type_frames.clone();
        // Readiness-vs-window evidence for the capture-starvation gates
        // (run-relative ns; 0 = never): WHEN the periodic prereqs (kaslr
        // publish + both accessors) became ready, and WHERE the capture
        // window ends. Readiness at-or-after the window end makes zero
        // captures STRUCTURALLY inevitable at any host dilation — the
        // honest environmental-skip signal the D-threshold missed (a
        // ~7 s cold accessor build outruns a 4-5 s window at D ≈ 1.2).
        let periodic_prereqs_ready_at: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let periodic_window_end_at: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let periodic_prereqs_ready_for_coord = periodic_prereqs_ready_at.clone();
        let periodic_window_end_for_coord = periodic_window_end_at.clone();
        let monitor_handle = self.start_monitor(
            &vm,
            &kill,
            &kill_evt,
            run_start,
            vcpu_pthreads,
            contention_recorder.clone(),
            perf_capture.clone(),
            Some(virtio_con.clone()),
            sys_rdy_evt.clone(),
            tcr_el1_cache.clone(),
            cr3_cache.clone(),
            watchdog_reset_ns.clone(),
            watchdog_reset_tag.clone(),
            progress_ledger.clone(),
            kern_phys_base.clone(),
            kern_phys_base_evt.clone(),
            kern_virt_kaslr.clone(),
            kern_addrs_frames,
            kern_addrs_crc_bad,
        )?;
        // Hand the monitor handle to the guard so an early-return past here
        // joins it before guest_mem drops (the monitor holds a bare raw pointer
        // into guest_mem). Reclaimed by `guard.disarm()` on the Ok path.
        guard.monitor = monitor_handle;
        let watchdog_reset_for_coord = watchdog_reset_ns.clone();
        let watchdog_reset_tag_for_coord = watchdog_reset_tag.clone();
        // Second clone of the progress ledger for the coordinator's
        // dispatch sinks: the first (above) went to `start_monitor` as
        // the monitor's producer handle; this one lets the scenario-
        // dispatch arms advance the live lifecycle stage as guest
        // lifecycle / scenario frames arrive. Both hold the same Arc, so
        // dispatch's forward-only stage stores and the monitor's per-tick
        // liveness stores land in one ledger.
        let progress_ledger_for_coord = progress_ledger.clone();
        let contention_recorder_for_coord = contention_recorder.clone();
        let watchdog_pause_ns: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let watchdog_pause_for_coord = watchdog_pause_ns.clone();
        let workload_duration_for_coord = self.workload_duration;
        // First-ScenarioStart timestamp (nanos since `run_start`),
        // clamped to a `1` floor (via `.max(1)`) so `0` stays the
        // "no ScenarioStart frame observed yet" sentinel — a
        // boot-unreachable elapsed==0 maps to 1, every real stamp is
        // exact (not a `+1` shift). The dispatch.rs ScenarioStart arm
        // CAS-stamps this on the first frame; the periodic-snapshot
        // loop in the coord run-loop uses it as the window END anchor
        // (`scenario_start + workload_duration`, clamped) and the
        // scenario-relative offset frame, while the window START
        // floats to the later of this stamp and the prereq-ready
        // moment — so neither boot nor `run_start` eats the budget.
        // `Arc<AtomicU64>` gives the coord thread shared ownership
        // with the dispatch sinks; both run in the same thread so
        // Relaxed ordering suffices, but the AtomicU64 keeps the
        // type story uniform with `watchdog_reset_for_coord` /
        // `watchdog_pause_for_coord`.
        let scenario_start_ns: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let scenario_start_ns_for_coord = scenario_start_ns.clone();
        // Scheduler-swap notification latch. The dispatch.rs
        // SchedSwapNotify arm sets it on a CRC-valid frame; the coord
        // run-loop reads-and-clears it each iteration to synchronously
        // invalidate the stale periodic-capture accessor on a swap Op.
        // Same coord thread as the dispatch sinks, so Relaxed suffices
        // — uniform with `scenario_start_ns`.
        let sched_swap_notify_for_coord: Arc<std::sync::atomic::AtomicBool> =
            Arc::new(std::sync::atomic::AtomicBool::new(false));
        // Cumulative wall-clock pause time observed between
        // matched `MSG_TYPE_SCENARIO_PAUSE` / `MSG_TYPE_SCENARIO_RESUME`
        // pairs (nanoseconds). Periodic-snapshot boundaries are
        // anchored to workload time, NOT wall-clock time — when the
        // guest workload pauses, the host's `run_start.elapsed()`
        // ticks during the pause but the workload's logical clock
        // does not. The dispatch.rs `ScenarioResume` arm bumps this
        // atomic by `pause_duration` so the run-loop can subtract it
        // from `run_start.elapsed()` to get effective workload-time
        // for the boundary-crossing check. The matching
        // `watchdog_pause_ns` atomic continues to track only the
        // current pause's start (used by the watchdog deadline-
        // extension path); cumulative pause is a periodic-only
        // concern.
        let scenario_pause_cumulative_ns: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let scenario_pause_cumulative_for_coord = scenario_pause_cumulative_ns.clone();
        // Periodic-snapshot count plumbed through KtstrVm for the
        // coord run-loop's periodic-capture cadence. `0` (the
        // default) skips the loop entirely — no boundary
        // computation, no per-iteration check.
        let freeze_coord_num_snapshots = self.num_snapshots;
        // Live periodic-fire count published by the run-loop after
        // each successful capture / placeholder store. Threaded
        // out to `VmResult::periodic_fired` so test code can
        // assert coverage. Written by the coordinator thread,
        // read by run_vm AFTER the coordinator joins, so Relaxed
        // ordering paired with the join's happens-before suffices.
        let periodic_fired_slot: Arc<std::sync::atomic::AtomicU32> =
            Arc::new(std::sync::atomic::AtomicU32::new(0));
        let periodic_fired_for_coord = periodic_fired_slot.clone();

        // BPF map write thread: sleeps, discovers a BPF map, writes a value.
        // Delivery evidence for host-triggered BPF-map-write injections
        // (0 = no writes configured, 1 = configured but not yet
        // delivered, 2 = delivered — see `VmResult::bpf_map_writes_delivered`).
        // The injection thread stamps 2 after signalling the guest; the
        // neg_* fixtures' starvation gates read the decoded value to
        // distinguish "expected crash absent because the injection never
        // landed" from a real detection failure.
        let bpf_map_write_delivery: Arc<std::sync::atomic::AtomicU8> = Arc::new(
            std::sync::atomic::AtomicU8::new(if self.bpf_map_writes.is_empty() { 0 } else { 1 }),
        );
        let bpf_write_handle = self.start_bpf_map_write(
            &vm,
            &kill,
            &kill_evt,
            probes_ready_evt,
            tcr_el1_cache.clone(),
            cr3_cache.clone(),
            virtio_con.clone(),
            kern_phys_base.clone(),
            bpf_map_write_delivery.clone(),
        )?;
        // Same as the monitor: the bpf-map-write thread holds a raw pointer into
        // guest_mem, so the guard must join it on an early-return past here.
        guard.bpf_write = bpf_write_handle;

        // Run BSP on this thread.
        register_vcpu_signal_handler();
        let timeout = self.timeout;

        // Watchdog thread.
        let bsp_ie = if has_immediate_exit {
            Some(ImmediateExitHandle::from_vcpu(&mut bsp))
        } else {
            None
        };
        let bsp_tid = unsafe { libc::pthread_self() };
        // `bsp_done` is created above (hoisted before the first spawn).
        let bsp_done_for_wd = bsp_done.clone();
        // BSP-IE-handle liveness gate. The freeze coordinator's
        // captured `ImmediateExitHandle` for the BSP addresses the
        // BSP `VcpuFd`'s kvm_run mmap; that mapping disappears the
        // moment `bsp` (a local in run_vm) falls out of scope. The
        // primary defense against UAF is the `freeze_coord_handle`
        // join inside run_vm BEFORE bsp drops, but this flag is a
        // cheap secondary check the closure consults before any
        // `bsp_ie_handle.set(1)` call so a future restructure that
        // moves the join doesn't silently reintroduce the UAF. Set
        // to `false` by run_vm right before bsp drops; gate every
        // BSP-side immediate_exit write on `bsp_alive.load(Acquire)`.
        let bsp_alive = Arc::new(AtomicBool::new(true));
        let bsp_alive_for_coord = bsp_alive.clone();
        // `bsp_done_evt` is created above (hoisted before the first spawn).
        let kill_for_watchdog = kill.clone();
        let timed_out_for_watchdog = timed_out_flag.clone();
        // Wake fds the watchdog blocks on via epoll, paired with the
        // `kill_for_watchdog` and `bsp_done_for_wd` AtomicBools above.
        // The watchdog wakes within microseconds of either flip
        // instead of polling on a 100 ms thread::sleep cadence.
        let kill_evt_for_watchdog = kill_evt.clone();
        let bsp_done_evt_for_wd = bsp_done_evt.clone();
        let wd_service_cpu = effective_pinning_plan.and_then(|p| p.service_cpu);
        // Clone the virtio-console Arc into the watchdog so the
        // soft-deadline path can push `SIGNAL_VC_SHUTDOWN` to
        // `/dev/hvc0` for graceful shutdown. The guest's
        // `hvc0_poll_loop` blocks on the device read and recognises
        // the byte directly — no SHM signal slot involved.
        let wd_virtio_con = virtio_con.clone();
        // Watchdog-side clones for the scheduler-attach reset
        // signal. The shared `AtomicU64` and the policy decision
        // ("reset is meaningful only when a distinct workload
        // duration was set") are bound here for the watchdog's
        // `move` closure; the matching monitor clone was taken
        // above at `start_monitor` invocation. Skipped at decode
        // time when `workload_duration_for_wd` is `None` — see
        // the watchdog's per-tick reset block.
        let watchdog_reset_for_wd = watchdog_reset_ns.clone();
        let watchdog_reset_tag_for_wd = watchdog_reset_tag.clone();
        let workload_duration_for_wd = self.workload_duration;
        // Progress-ledger consumer handle for the watchdog. The monitor
        // (producer) took its clone at the `start_monitor` site above and
        // the coordinator's dispatch sinks another; this third clone lets
        // the watchdog READ the ledger each tick to fold the Tier-1/2
        // progress verdicts alongside the Tier-3 deadline. All three hold
        // the same Arc, so the watchdog sees the monitor's liveness/CPU
        // stores and dispatch's phase advances in one ledger.
        let progress_ledger_for_wd = progress_ledger.clone();
        // Fourth ledger clone, read once at `VmRunState` build: the final
        // phase / progress_epoch land on `VmResult` so mechanism-asserting
        // fixtures can distinguish "guest reached the phase under test"
        // from "guest never booted" (see `VmResult::final_guest_phase`).
        let progress_ledger_for_result = progress_ledger.clone();
        // Kill-reason latch, shared between the watchdog thread (sole
        // writer, on its fire path) and the `VmRunState` build (read after
        // the watchdog join, so the value is final). Surfaces the dump's
        // `cause=` verdict on `VmResult::watchdog_kill_reason` — the
        // mechanism signal wedge fixtures assert instead of wall bounds.
        let watchdog_kill_reason: Arc<std::sync::atomic::AtomicU8> =
            Arc::new(std::sync::atomic::AtomicU8::new(KillReasonTag::Unset as u8));
        let watchdog_kill_reason_for_wd = watchdog_kill_reason.clone();
        // Kill-time publish-state evidence for the watchdog dump (see the
        // `kern_addrs_frames= (kill-time)` line): the pre-latch diag's
        // one-shot ~10 s snapshot cannot distinguish a publish that never
        // arrived from one that arrived late, so the dump re-reads the
        // live counters at the kill.
        let kern_addrs_frames_for_wd = kern_addrs_frames_for_coord.clone();
        let unknown_type_frames_for_wd = unknown_type_frames.clone();
        let kern_virt_kaslr_for_wd = kern_virt_kaslr.clone();
        let kern_phys_base_for_wd = kern_phys_base.clone();

        // Freeze coordinator thread: triggers a failure-dump freeze when
        // the BPF probe's `ktstr_err_exit_detected` .bss latch fires
        // (sched_ext error-class exit observed by tp_btf inside
        // probe.bpf.c). The flag lives in the probe BPF program's
        // .bss map — the coordinator polls it via host-side guest
        // physical memory access, NOT via SHM TLV. Discovery is
        // lazy: each iteration tries
        // `BpfMapAccessor::find_map("probe_bp.bss")` (suffix-matched
        // to avoid colliding with a scheduler-under-test's own .bss
        // map) until the probe is loaded into map_idr, then caches
        // the field PA — the .bss value-region PA plus the
        // BTF-resolved byte offset of `ktstr_err_exit_detected`
        // within the section (see `cached_bss_offset`). Subsequent
        // polls run through [`bss_read_state`], which returns a
        // typed Triggered / NotTriggered / OutOfBounds /
        // NotResolved result so a stale PA after a probe unload
        // surfaces as an explicit diagnostic rather than
        // masquerading as "no fire".
        //
        // Sequencing combines Cloud Hypervisor's pause/snapshot
        // pattern (drain dance + N-of-N rendezvous on parked acks)
        // with Firecracker's SIGRTMIN+immediate_exit kick:
        //   1. observe `ktstr_err_exit_detected != 0` via .bss read
        //   2. set `freeze=true`
        //   3. set every vCPU's immediate_exit=1 (two-pass kick: all
        //      flags first, then signal all)
        //   4. signal every vCPU thread (pthread_kill SIGRTMIN)
        //   5. wait for N-of-N parked acks (Acquire-load on each
        //      `parked` flag — synchronizes-with the vCPU's Release
        //      store after the drain dance, providing the happens-
        //      before edge that makes guest-memory reads correct on
        //      weakly-ordered architectures)
        //   6. call dump_state to read BPF map state, vCPU regs,
        //      and per-CPU prog/cputime captures into a
        //      FailureDumpReport, then emit the report as JSON
        //      via tracing::error and the optional file sink
        //   7. clear freeze=false; each parked vCPU polling on
        //      park_timeout(10ms) observes the clear within 10 ms
        //      and resumes — no explicit unpark needed
        //
        // DMA quiescence: virtio-blk's independent worker thread
        // is paused before the vCPU SIGRTMIN kick (see
        // `blk.lock().pause()` in freeze_and_dispatch below); the
        // rendezvous waits for the worker's paused ack alongside
        // the vCPU parked acks. virtio-net (v0) and virtio-console
        // run synchronously on the vCPU thread, so they freeze
        // automatically once the vCPU rendezvous completes. A
        // future device with its own worker thread would need to
        // be added to the pause sequence.
        let freeze_coord_freeze = freeze.clone();
        let freeze_coord_kill = kill.clone();
        // Per-VM scenario phase index, host-side mirror of the
        // guest's `Ctx::current_step` publisher. Updated whenever
        // the freeze-coord dispatch loop decodes a CRC-valid
        // `MSG_TYPE_STIMULUS` frame on the bulk port (see the
        // `MsgType::Stimulus` arm in `dispatch.rs`); consumed by
        // the periodic-fire path below to stamp each captured
        // sample with the scenario phase the guest was in when
        // the periodic boundary fired. Encoded per the framework's
        // 1-indexed phase convention (`0` = BASELINE settle
        // window, `1..=N` = Step ordinals) to match every other
        // step_index slot in the pipeline. Per-VM (not
        // process-global) so parallel in-process gauntlet
        // variants get independent atomics.
        let host_current_step: Arc<AtomicU16> = Arc::new(AtomicU16::new(0));
        let freeze_coord_current_step = Arc::clone(&host_current_step);
        // Guest scenario-elapsed clock (ns) published by the workload-progress
        // heartbeat; the run-loop reads it to clock periodic captures off
        // guest progress. 0 until the first heartbeat.
        let host_periodic_guest_elapsed: Arc<std::sync::atomic::AtomicU64> =
            Arc::new(std::sync::atomic::AtomicU64::new(0));
        let freeze_coord_periodic_guest_elapsed = Arc::clone(&host_periodic_guest_elapsed);
        // Optional virtio-blk handle for the failure-dump
        // worker-pause rendezvous. None when no disk is attached.
        // Cloned into the closure so the dump path can call
        // `dev.lock().pause()` BEFORE kicking the vCPUs and
        // `dev.lock().resume()` after the dump completes — without
        // this, the worker thread would continue mutating the
        // backing file (and the avail/used rings) while the host
        // reads guest memory for the dump. Only virtio-blk has an
        // independent worker thread; virtio-net (v0) and
        // virtio-console run synchronously on the vCPU thread and
        // are automatically frozen when the vCPU rendezvous
        // completes (their `mmio_write` handlers must have already
        // returned for the vCPU to reach the parked state).
        // Use `blk_device` (set on both transports), NOT `virtio_blk` (the
        // MMIO-routing handle, `None` on x86): the coordinator must pause/resume
        // the request worker regardless of how the guest reaches the device.
        let freeze_coord_virtio_blk = blk_device.clone();
        // Lock-free `paused` flag handle. The freeze coordinator
        // polls the worker's parked-state in two paths (the
        // rendezvous timeout-diagnostic snapshot and the post-thaw
        // barrier predicate). Both previously read via
        // `d.lock().is_paused()`, which contends with every
        // concurrent device operation that holds the device mutex
        // — `mmio_read`/`mmio_write` from the vCPU thread and any
        // other freeze-coord call site holding the lock. The
        // underlying field is already `Arc<AtomicBool>`, so
        // exposing a clone here lets the rendezvous read it
        // lock-free. The Acquire/Release ordering on the worker's
        // `paused` writes provides the same happens-before edges
        // with the worker's parked-state stores that
        // `is_paused()` does.
        let freeze_coord_virtio_blk_paused: Option<Arc<AtomicBool>> =
            blk_device.as_ref().map(|d| d.lock().paused_handle());
        // Clone the virtio-console Arc into the coordinator so it
        // can drain port-1 bulk TLV bytes as the guest writes them
        // (event-driven via the tx_evt eventfd registered into the
        // coord's epoll set below). Bytes are accumulated into
        // `coord_bulk_buf` and parsed at the end of the run; an
        // early SCHED_EXIT TLV flips `kill` so the watchdog and
        // BSP loop exit promptly.
        let freeze_coord_virtio_con = virtio_con.clone();
        // Clone the virtio-console tx_evt so the coord epoll wakes
        // immediately whenever the guest publishes a TX descriptor
        // chain on port 0 or port 1. The tx_evt is shared between
        // those two ports — a spurious wake on port-0 traffic is
        // harmless: the coord just calls `drain_bulk()` and finds
        // an empty buffer. Port 2 TX (scheduler stats) is owned
        // entirely by [`crate::vmm::sched_stats::SchedStatsClient`]
        // and never reaches this coordinator's epoll set.
        let freeze_coord_tx_evt = virtio_con
            .lock()
            .tx_evt()
            .try_clone()
            .context("clone virtio-console tx_evt for coordinator")?;
        let freeze_coord_bsp_parked = bsp_parked.clone();
        let freeze_coord_bsp_regs = bsp_regs.clone();
        let freeze_coord_bsp_done = bsp_done.clone();
        // Watchpoint-arming state shared with every vCPU thread (BSP
        // + APs). The coordinator publishes the resolved
        // `*scx_root->exit_kind` KVA into `request_kva` and polls
        // `hit` instead of the prior BPF .bss latch read. See
        // [`WatchpointArm`] for the full protocol; the Arc outlives
        // every vCPU thread because `collect_results` joins the
        // coordinator BEFORE the AP thread joins drop the VcpuFds.
        let freeze_coord_watchpoint = watchpoint.clone();
        // Shared per-vCPU perf-counter capture. The Arc lets the
        // monitor sampling loop (per-tick timeline) and the freeze
        // coordinator (freeze-instant snapshot) read through the same
        // fds. Inner `Option` is `None` when `perf_event_open` was
        // unavailable on the host; both consumers gracefully degrade
        // to "no perf data" without aborting the run.
        let freeze_coord_perf_capture = perf_capture.clone();
        let freeze_coord_vmlinux = find_vmlinux(&self.kernel);
        // Read vmlinux bytes once at run_vm scope. Shared via Arc
        // with the coordinator closure (for accessor init, dump_btf,
        // dump_cpu_time_symbols) and VmRunState (for
        // collect_verifier_stats). Eliminates the 14-28s cold-cache
        // re-read that caused cleanup hangs.
        let vmlinux_data_shared: Option<Arc<Vec<u8>>> = freeze_coord_vmlinux
            .as_ref()
            .and_then(|p| super::vmlinux::cached_vmlinux_bytes(p));
        // Cached `name -> KVA` map for `Op::WatchSnapshot` arming.
        // Build once here at run_vm scope so every TLV-driven
        // WATCH request is an O(1) HashMap lookup instead of a
        // 50MB+ vmlinux read + ELF parse. None when vmlinux can't
        // be found or the parse failed — `arm_user_watchpoint`
        // will report a clean diagnostic on lookup. Hoisted out of
        // the closure so the spawn-time parse cost is paid once
        // even when the run ends without any WATCH requests.
        let vmlinux_data_for_result = vmlinux_data_shared.clone();
        let prog_accessor_slot: Arc<
            std::sync::Mutex<Option<crate::monitor::bpf_prog::GuestMemProgAccessorOwned>>,
        > = Arc::new(std::sync::Mutex::new(None));
        let prog_accessor_slot_for_coord = prog_accessor_slot.clone();
        let freeze_coord_symbol_cache: Option<Arc<VmlinuxSymbolCache>> = freeze_coord_vmlinux
            .as_deref()
            .and_then(|p| match VmlinuxSymbolCache::from_path(p) {
                Ok(c) => Some(Arc::new(c)),
                Err(e) => {
                    tracing::warn!(
                        path = %p.display(),
                        error = %e,
                        "freeze-coord: vmlinux symbol cache build failed; \
                         Op::WatchSnapshot WATCH requests will return errors"
                    );
                    None
                }
            });
        // Optional file sink for the failure-dump JSON. Cloned out
        // of the builder field so the closure owns a copy and the
        // freeze coord can write the file without touching the env
        // or the parent `KtstrVm`.
        let freeze_coord_dump_path = self.failure_dump_path.clone();
        // Dual-snapshot mode: when true, the freeze coordinator
        // additionally polls per-CPU `rq->scx.runnable_list` for any
        // task whose `jiffies - p->scx.runnable_at` crosses
        // `watchdog_timeout/2`, takes a snapshot at that point, and
        // wraps both early + late snapshots into a
        // [`monitor::dump::DualFailureDumpReport`]. Set by
        // `attempt_auto_repro` for the repro VM only.
        let freeze_coord_dual_snapshot = self.dual_snapshot;
        // Whether this run captures a wprof trace. On an error-class scx
        // exit the guest ships its wprof `.pb` LATE — the crash unwinds
        // dispatch, then Phase 5 joins the wprof thread and sends the trace
        // over the bulk port, then Phase 6 reboots (src/vmm/rust_init) — so
        // the error-exit dump path below must NOT tear the VM down at the
        // dump, or the `.wprof.pb` / `.repro.wprof.pb` artifact is dropped.
        // Split to a plain bool so it crosses into the always-compiled
        // coord closure regardless of the feature.
        #[cfg(feature = "wprof")]
        let freeze_coord_wprof = self.wprof.is_some();
        #[cfg(not(feature = "wprof"))]
        let freeze_coord_wprof = false;
        // The test's workload-root cgroup path (host-held), for the
        // per-cgroup PSI-irq walk's subtree filter. Defaults to the guest's
        // resolve_cgroup_root default (/sys/fs/cgroup/ktstr) when unset, so the
        // walk targets the same subtree the scenario created its cgroups under.
        let freeze_coord_workload_root = self
            .workload_root_cgroup
            .clone()
            .unwrap_or_else(|| "/sys/fs/cgroup/ktstr".to_string());
        // Phase A precondition (loud check): the per-cgroup PSI walk filters
        // to the workload-root subtree, so a scheduler cgroup nested UNDER the
        // workload root would be mis-collected as a spurious workload leaf. The
        // default is safe (no scheduler cgroup_parent → the scheduler inherits
        // the root cgroup, outside the workload subtree). Warn if a config nests
        // them — the confound is otherwise silent in the cross-cgroup fold.
        if let Some(sched_parent) = self.scheduler_cgroup_parent.as_deref() {
            let cgroup_rel = |p: &str| {
                p.strip_prefix("/sys/fs/cgroup")
                    .unwrap_or(p)
                    .trim_matches('/')
                    .to_string()
            };
            let sched_rel = cgroup_rel(sched_parent);
            let wl_rel = cgroup_rel(&freeze_coord_workload_root);
            if !wl_rel.is_empty()
                && (sched_rel == wl_rel || sched_rel.starts_with(&format!("{wl_rel}/")))
            {
                tracing::warn!(
                    scheduler_cgroup_parent = %sched_parent,
                    workload_root = %freeze_coord_workload_root,
                    "per-cgroup PSI-irq: scheduler cgroup is nested under the \
                     workload root — its IRQ-servicing pressure will be mis-collected \
                     as a workload leaf; place the scheduler cgroup outside the \
                     workload root to keep the per-cgroup axis clean",
                );
            }
        }
        // Half of the configured watchdog timeout, in nanoseconds.
        // Used by the dual-snapshot scanner to compare against each
        // task's runnable-age in jiffies (converted via the guest's
        // CONFIG_HZ at scan time). The fallback default
        // (`Duration::from_secs(4)` per the builder default) means
        // a coord that never received an explicit
        // `watchdog_timeout()` call still has a coherent half-way
        // mark — 2 s of stall before the early snapshot fires.
        let freeze_coord_watchdog_half = self
            .watchdog_timeout
            .unwrap_or(Duration::from_secs(4))
            .checked_div(2)
            .unwrap_or(Duration::ZERO);
        // Guest CONFIG_HZ resolved from the kernel image. Used to
        // convert the watchdog_half Duration into a jiffies-domain
        // threshold the runnable_at scan can compare against.
        let freeze_coord_hz = monitor::guest_kernel_hz(Some(&self.kernel));
        // GuestMem for the coordinator's .bss-poll path. Built from
        // the same guest_mem the monitor uses; lifetime tied to the
        // VM run.
        // GuestMem owns its host pointer for the duration of the run.
        // Wrapped in `Arc` so the worker thread that lazy-builds the
        // `GuestMemMapAccessorOwned` and the coordinator's own
        // accessor-borrow paths share the same backing mapping.
        // `Arc<GuestMem>` is `Send` because `GuestMem` is `Send + Sync`
        // (see `unsafe impl Send for GuestMem` in `monitor::reader`).
        let freeze_coord_mem: Option<Arc<monitor::reader::GuestMem>> = match vm.numa_layout.as_ref()
        {
            Some(layout) => Some(Arc::new(monitor::reader::GuestMem::from_layout(
                layout,
                &vm.guest_mem,
            ))),
            None => {
                use vm_memory::GuestMemoryRegion;
                if let Ok(host_base) = vm.guest_mem.get_host_address(GuestAddress(DRAM_BASE))
                    && let Some(r) = vm.guest_mem.iter().next()
                {
                    let mem_size = r.len();
                    // SAFETY: host_base came from GuestMemoryMmap's
                    // get_host_address; mapping outlives this GuestMem
                    // (vm.guest_mem outlives the coordinator thread —
                    // collect_results joins the coordinator before vm
                    // is dropped).
                    Some(Arc::new(unsafe {
                        monitor::reader::GuestMem::new(host_base, mem_size)
                    }))
                } else {
                    None
                }
            }
        };
        // Extract a fresh ImmediateExitHandle for the freeze coord —
        // the watchdog grabs another one below for its own kick path.
        // Both views address the same kvm_run.immediate_exit byte
        // (single-byte volatile writes), distinct from the BSP's own
        // owned handle inside its run loop.
        let freeze_coord_bsp_ie_handle = if has_immediate_exit {
            Some(ImmediateExitHandle::from_vcpu(&mut bsp))
        } else {
            None
        };
        let freeze_coord_bsp_tid = unsafe { libc::pthread_self() };
        // Snapshot the AP-side freeze handles. `parked` flags and
        // register-snapshot slots come from `ap_freeze_handles` —
        // populated alongside the threads inside `spawn_ap_threads`,
        // kept out of `VcpuThread` so that struct stays minimal
        // (only `handle` + `exited` + `immediate_exit` are needed
        // for teardown). The freeze coordinator owns these Vecs
        // for the rest of run_vm. `pthread_t`s and immediate-exit
        // handles still come from `ap_threads` because those are
        // teardown-relevant too.
        let ApFreezeHandles {
            parked: freeze_coord_ap_parked,
            regs: freeze_coord_ap_regs,
        } = ap_freeze_handles;
        let freeze_coord_ap_pthreads: Vec<libc::pthread_t> = guard
            .ap_threads
            .iter()
            .map(|vt| vt.handle.as_pthread_t() as libc::pthread_t)
            .collect();
        // ImmediateExitHandle is Copy+Send+Sync, so the coordinator
        // captures a Vec of them by move. The kvm_run mmap is shared
        // between the spawned vCPU thread (which owns its handle
        // inside VcpuThread) and the coordinator's copy — single-byte
        // volatile writes through `set` from either side address the
        // same MAP_SHARED page.
        let freeze_coord_ap_ies: Vec<Option<ImmediateExitHandle>> = guard
            .ap_threads
            .iter()
            .map(|vt| vt.immediate_exit)
            .collect();
        // Per-AP `alive` flags paired with the IE handles above. The
        // coordinator's pass-1 kick (in `freeze_and_dispatch`) and
        // `arm_user_watchpoint` gate each `ie.set` on a fresh
        // Acquire load of the corresponding entry, mirroring the
        // BSP-side `bsp_alive` TOCTOU-tightened gate. Without this,
        // an AP panic-unwind under `panic = "unwind"` (test profile)
        // can drop `vcpu` mid-cycle and the coordinator's
        // `Vec<ImmediateExitHandle>` would issue a `write_volatile`
        // through a freed `kvm_run` mapping. The Vec lives the
        // entire coordinator lifetime; index alignment with
        // `freeze_coord_ap_ies` and `freeze_coord_ap_pthreads` is
        // load-bearing — every AP-loop site uses `iter().enumerate()`
        // (or `zip`) so a future change that drops or reorders any
        // one Vec is loud about the regression.
        let freeze_coord_ap_alive: Vec<Arc<AtomicBool>> =
            guard.ap_threads.iter().map(|vt| vt.alive.clone()).collect();
        // Total vCPU count (BSP + APs). Forwarded into dump_state so
        // PERCPU_ARRAY map rendering knows how many per-CPU slots to
        // read — `bpf_array.pptrs[k]` is a `void __percpu *` whose
        // per-CPU expansion needs `__per_cpu_offset[0..nr_cpu_ids]`.
        let freeze_coord_num_cpus = (guard.ap_threads.len() + 1) as u32;
        // NUMA node count from the configured topology. Forwarded
        // into the scx walker (per-node global DSQ pass) and the
        // per-node NUMA event walker. Defaults to 1 on UMA topologies.
        let freeze_coord_num_nodes = self.topology.num_numa_nodes();
        // Lazy BPF cast-analysis handle produced at builder time.
        // The handle is `Arc<LazyCastMap>` and holds only the
        // scheduler binary path plus a `OnceLock` slot; the
        // analyzer runs only when `.get_full()` is first called
        // at dump time on the freeze-coordinator host thread (NOT
        // a vCPU thread — the freeze rendezvous has already
        // paused vCPUs by the time `dump_state` runs). The clone
        // shares the `OnceLock`, so a periodic-capture dump and
        // the final freeze in the same VM both resolve to the
        // same analyzed `Arc<CastAnalysisOutput>` after the first
        // `.get_full()`.
        let freeze_coord_cast_map: Arc<crate::vmm::cast_analysis_load::LazyCastMap> =
            self.cast_map.clone();
        let freeze_coord_on_demand_in_flight = on_demand_in_flight.clone();
        // Snapshot bridge clone for the coord run-loop. The bridge
        // stores `FailureDumpReport` entries under per-tag keys (TLV
        // CAPTURE, periodic, user-watchpoint paths). On a Degraded
        // outcome the bridge entry is flattened to a SCHEMA_SINGLE
        // placeholder with `is_placeholder=true` so existing in-process
        // `Snapshot` API consumers don't need to branch on the
        // dispatcher's 3-arm match; the canonical SCHEMA_DEGRADED form
        // (with `watchpoint_hit`, `bss_latch_state`, `exit_kind`,
        // `elapsed_ms` fields) lives in the on-disk failure-dump JSON
        // written via [`write_to_tagged_path`]. Consumers wanting the
        // structured trigger-state fields should read the file via
        // [`crate::monitor::dump::FailureDumpReportAny::from_json`]
        // rather than the bridge. The early-snapshot Degraded path
        // writes a tagged sibling via [`write_to_tagged_path`] but
        // does NOT publish to the bridge — late-trigger arms consume
        // held early snapshots from a closure-local `early_snapshot`
        // variable, not the bridge.
        let freeze_coord_snapshot_bridge = snapshot_bridge.clone();
        // Stats-client clone for the periodic-capture path. The
        // periodic-fire branch issues a `stats(&[])` request BEFORE
        // calling `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })` so the JSON it returns
        // reflects the running scheduler — once the freeze rendezvous
        // begins the scheduler's userspace thread is paused and the
        // request would either time out or wedge until thaw. The
        // resulting `serde_json::Value` is bundled with the
        // FailureDumpReport via `SnapshotBridge::store_with_stats` so
        // a later `Sample` view exposes both axes from the same
        // boundary. `None` when no scheduler is configured (the
        // outer `stats_client` builder above returns `None` when
        // `scheduler_binary.is_none()`); in that case periodic
        // captures store `None` in the parallel stats slot and the
        // temporal-stats projection surfaces a per-sample missing-
        // stats failure that the test author can opt to ignore.
        let freeze_coord_stats_client = stats_client.clone();
        // Wake-fd handles for the coord epoll loop. `kill_evt` and
        // `bsp_done_evt` are written by every thread that flips the
        // matching AtomicBool (run_vm post-BSP-exit, vCPU shutdown
        // classifier, BSP panic hook, AP panic hook); the epoll wait
        // fires immediately on either edge instead of polling on a
        // 500 µs sleep cadence. The watchpoint hit_evt clone lets
        // the coord wake on a hardware-watchpoint fire (vCPU thread
        // calls `WatchpointArm::latch_hit`, which writes the
        // eventfd alongside the AtomicBool flip). All three live
        // for the lifetime of the run — `run_vm` joins the coord
        // BEFORE the eventfds drop.
        let freeze_coord_kill_evt = kill_evt.clone();
        // aarch64 TCR_EL1 cache populated by the BSP loop. Threaded
        // into `GuestKernel::new` constructions inside the
        // freeze-coord scan_tick closure (BPF map accessor and
        // prog accessor) so vmalloc-backed kernel reads succeed
        // post-MMU-bringup. None on x86_64.
        let freeze_coord_tcr_el1 = tcr_el1_cache.clone();
        // CR3 (x86_64) / TTBR1_EL1 (aarch64) cache populated by
        // the BSP loop. Threaded into `GuestKernel::new` so the
        // boot-time `phys_base` resolution can walk the live
        // kernel page tables.
        let freeze_coord_cr3 = cr3_cache.clone();
        let freeze_coord_bsp_done_evt = bsp_done_evt.clone();
        // Clone the WatchpointArm.hit_evt for the epoll set. EventFd
        // clones share the underlying counter via dup(2), so the
        // vCPU's `latch_hit` write delivers an edge to every clone.
        let freeze_coord_hit_evt = watchpoint
            .hit_evt
            .try_clone()
            .context("clone WatchpointArm.hit_evt for coordinator")?;
        // Shared parked_evt for the rendezvous wait. Every vCPU
        // thread + the virtio-blk worker writes to this fd
        // immediately after their respective parked/paused Release
        // store; the rendezvous loop polls on this fd alongside
        // kill_evt and bsp_done_evt instead of spin-sleeping.
        let freeze_coord_parked_evt = parked_evt.clone();
        // Shared thaw_evt: the coordinator writes 1 here AFTER the
        // `freeze.store(false, Release)` so every parked vCPU's
        // poll wakes within microseconds rather than waiting on the
        // legacy 10ms park_timeout cadence.
        let freeze_coord_thaw_evt = thaw_evt.clone();
        // Shared bulk-message buffer: the TOKEN_TX handler in the
        // coordinator parses port-1 TLV bytes via `HostAssembler`
        // and drains the per-frame `BulkMessage` values. Without
        // this buffer those messages would be discarded after the
        // SCHED_EXIT scan, leaving `collect_results` blind to every
        // EXIT / TEST / PAYLOAD_METRICS /
        // PROFRAW frame the guest already published mid-run. The
        // post-exit `drain_bulk()` only catches what arrived AFTER
        // the coordinator stopped draining — not the bulk of a
        // typical run. The Mutex serialises the coord's pushes
        // against `collect_results`'s drain; both occur strictly
        // after the closure spawns and strictly before the
        // coordinator joins, so contention is rare.
        let freeze_coord_bulk_messages: Arc<std::sync::Mutex<Vec<crate::vmm::wire::ShmEntry>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let freeze_coord_bulk_messages_for_closure = freeze_coord_bulk_messages.clone();

        // Captured sys_rdy eventfd for the coordinator's TLV
        // dispatch loop. The TOKEN_TX handler promotes a CRC-valid
        // `MSG_TYPE_SYS_RDY` frame into a single
        // [`EventFd::write`] on this fd, releasing the monitor
        // thread's pre-sample `epoll_wait`. The `Option<Arc<...>>`
        // is replaced with `None` after the first promotion via
        // [`Option::take`] so subsequent SYS_RDY frames (a hostile
        // guest could in principle resend) skip the eventfd write
        // and do not pump the counter. `None` initially when the
        // sys_rdy machinery was not constructed (`EventFd::new`
        // failed at boot — already logged); in that case the
        // monitor will fall through its 5 s boot-wait timeout
        // without a guest signal. `move` semantics on the closure
        // mean the moved-in `Option` is dropped at coordinator
        // shutdown, releasing the host-side reference.
        let mut freeze_coord_sys_rdy_evt = sys_rdy_evt.clone();

        // One-time probe of the host's hardware-watchpoint slot
        // count via `KVM_CHECK_EXTENSION(KVM_CAP_GUEST_DEBUG_HW_WPS)`.
        // The slot 0 reservation for `*scx_root->exit_kind` plus
        // [`crate::scenario::snapshot::MAX_WATCH_SNAPSHOTS`] user
        // slots means the framework needs at least 4 hardware
        // watchpoint slots to arm every requested
        // [`crate::scenario::ops::Op::WatchSnapshot`]. KVM returns
        // the count via `check_extension_int`; `<= 0` means the
        // capability is unavailable. Log only — do not block VM
        // creation: a kernel without the capability still runs
        // tests, just without the watch-driven snapshots, and a
        // probe failure surfacing here is more actionable than a
        // silent `KVM_SET_GUEST_DEBUG` rejection later.
        let hw_wps = vm.vm_fd.check_extension_int(kvm_ioctls::Cap::DebugHwWps);
        if hw_wps <= 0 {
            tracing::warn!(
                "KVM_CAP_GUEST_DEBUG_HW_WPS unavailable on this host \
                 (returned {hw_wps}); Op::WatchSnapshot triggers may \
                 not arm — falling back to BPF .bss poll for the \
                 error-class freeze trigger"
            );
        } else {
            tracing::info!(
                "KVM host advertises {hw_wps} hardware watchpoint \
                 slots via KVM_CAP_GUEST_DEBUG_HW_WPS"
            );
            if hw_wps < 4 {
                tracing::warn!(
                    "KVM host advertises only {hw_wps} hardware \
                     watchpoint slots; the framework reserves slot 0 \
                     for the *scx_root->exit_kind error-class trigger \
                     plus up to {} user slots for Op::WatchSnapshot \
                     — some watch_snapshot arms may fail",
                    crate::scenario::snapshot::MAX_WATCH_SNAPSHOTS,
                );
            }
        }

        let kern_phys_base_for_result = kern_phys_base.clone();
        // Mirror of kern_phys_base_for_result for the kern_virt_kaslr
        // Arc — snapshot at run-end into VmResult.kern_kaslr_offset.
        let kern_virt_kaslr_for_result = kern_virt_kaslr.clone();
        // Sibling clone for the BSP loop: the freeze-coord closure
        // below captures `kern_virt_kaslr` by move (the dispatch
        // sink uses it for the guest-channel KERN_ADDRS derive), so
        // the BSP MSR_LSTAR path (which runs on this thread AFTER
        // the coord spawn at `run_bsp_loop`) needs its own clone to
        // CAS-publish into the same Arc. KASLR offset is a single
        // boot-time slot pick so both writers always derive the
        // same value — the CAS-fail branch is benign (first writer
        // wins; second observes its own derivation match the
        // existing slot value).
        let kern_virt_kaslr_for_bsp = kern_virt_kaslr.clone();
        // Clone the publish-EventFd for the BSP MSR_LSTAR-derive path
        // (mirrors the KERN_ADDRS-side write at dispatch.rs:547) so
        // both publishers signal the same eventfd on successful CAS.
        // No production consumer epolls this fd today, but the
        // asymmetry would foot-gun a future change that does.
        let kern_virt_kaslr_evt_for_bsp = kern_virt_kaslr_evt.clone();
        // Effective rendezvous-wait deadline shared by the worker-park
        // and post-thaw barriers downstream. Downstream comments
        // reference `FREEZE_RENDEZVOUS_TIMEOUT` (the 30 s const) as
        // the canonical bound; the captured local resolves
        // to that const unless `KtstrVmBuilder::rendezvous_timeout` was
        // set — the override path is test-fixture only. Captured here
        // (outside the spawn) so the closure copies the `Duration`
        // (`Copy`) by value rather than borrowing `self`, which the
        // `'static` thread::spawn bound forbids.
        let rendezvous_timeout = self.rendezvous_timeout.unwrap_or(FREEZE_RENDEZVOUS_TIMEOUT);
        // Cache vm.vm_fd's raw fd (Copy = i32) before the
        // freeze-coord spawn. The spawned closure is `move ||` —
        // capturing vm directly would consume it and break L10885
        // construction of VmRunState. The raw fd is owned by
        // `vm.vm_fd` (in the outer scope) and stays valid for the
        // entire `run_vm` lifetime; coord-thread joins via
        // `freeze_coord_handle.join()` at L10832 before vm drops.
        // Used by the kvm_clock save/restore path inside
        // freeze_and_dispatch + thaw_and_barrier closures via
        // [`kvm_get_clock_via_raw_fd`] +
        // [`kvm_set_clock_via_raw_fd`].
        let vm_fd_raw_for_coord: i32 = vm.vm_fd.as_raw_fd();
        let freeze_coord_handle = std::thread::Builder::new()
            .name("vmm-freeze-coord".into())
            .spawn(move || {
                // The hang detector's own sensing must not dilate with the
                // load it measures. This thread runs the dispatch sinks whose
                // `advance_stage` -> `ProgressLedger::record_progress` bumps
                // `progress_epoch` — the MILESTONE anchor the Tier-3 deadman
                // reads as `wall_since_milestone_ns`. If it starves under host
                // dilation, a milestone the guest already earned lands late,
                // `wall_since_milestone` grows, and it feeds an inert misread.
                // So it gets the same UNCONDITIONAL FIFO-2 as the watchdog and
                // monitor. It is NOT pinned, deliberately: the watchdog and
                // monitor pin to the single reserved `service_cpu`, and this
                // thread's freeze-time guest-memory scans would head-of-line
                // block those same-priority FIFO threads if colocated there —
                // the exact monitor starvation being fixed. Left unpinned it
                // may briefly preempt a FIFO-1 vCPU on wake in perf mode
                // (intended, sensing wins) but is epoll-idle between events,
                // so it does not erode perf-mode vCPU isolation during a body
                // run. Best-effort — warns once/process without CAP_SYS_NICE.
                set_rt_priority(2, "freeze-coord");
                // Per-CPU runnable_at scanner context. Holds every
                // input the scanner needs, all resolved once and
                // cached for the rest of the run. Only built when
                // dual_snapshot is enabled AND every prerequisite
                // resolves (vmlinux ELF parses, BTF resolves the
                // four runnable_scan offsets, jiffies_64 symbol is
                // present, the GuestKernel handshake completes so
                // we have a cr3_pa / page_offset / l5 view).
                struct RunnableScanCtx {
                    /// KVA of the kernel's global `scx_tasks` LIST_HEAD
                    /// (`kernel/sched/ext.c:48`). The walker reads
                    /// `scx_tasks.next` via the runtime kernel image
                    /// base ([`Self::start_kernel_map`]) and
                    /// container_of's each list entry back to its
                    /// `task_struct`.
                    scx_tasks_kva: u64,
                    /// Per-CPU `struct rq` PAs (one per logical CPU).
                    /// Built by `compute_rq_pas(runqueues_kva,
                    /// __per_cpu_offset[*], page_offset)`. Each entry
                    /// addresses the rq whose `scx.runnable_list`
                    /// the per-rq walker walks; vec index = CPU index.
                    /// Empty when the per-CPU offset array can't be
                    /// resolved (per-rq walk silently falls back to
                    /// the global walk).
                    rq_pas: Vec<u64>,
                    offsets: crate::monitor::btf_offsets::RunnableScanOffsets,
                    jiffies_64_pa: u64,
                    /// PA of `scx_watchdog_timestamp`
                    /// (`kernel/sched/ext.c:93`). The kernel's
                    /// `scx_tick` (`kernel/sched/ext.c:3492`) compares
                    /// `jiffies - scx_watchdog_timestamp` against the
                    /// scheduler's `watchdog_timeout` and fires
                    /// `SCX_EXIT_ERROR_STALL` when the workqueue
                    /// stopped running. Reading the same value here
                    /// gives the dual-snapshot path the global stall
                    /// signal regardless of whether any individual
                    /// task is stuck on a per-rq runnable_list. None
                    /// when the symbol is absent (kernel without
                    /// sched_ext or stripped vmlinux); per-rq /
                    /// global walks still cover the per-task case.
                    watchdog_timestamp_pa: Option<u64>,
                    /// Paging context (cr3_pa / page_offset / l5 /
                    /// tcr_el1) threaded into the runnable_scan helpers.
                    walk: crate::monitor::reader::WalkContext,
                    /// Runtime kernel image base
                    /// (`__START_KERNEL_map` on x86_64,
                    /// `KIMAGE_VADDR` on aarch64). Threaded into the
                    /// runnable_scan helpers so `scx_tasks` and other
                    /// kernel-text-mapped symbols translate via the
                    /// VA-bits-aware base resolved from `TCR_EL1` —
                    /// matches the [`super::super::monitor::guest::GuestKernel`]
                    /// the surrounding accessors share.
                    start_kernel_map: u64,
                    /// Runtime KASLR offset (`phys_base` on x86_64;
                    /// `0` on aarch64 / non-KASLR boots). Required by
                    /// `text_kva_to_pa_with_base` so KASLR kernels
                    /// resolve `scx_tasks` / `jiffies_64` /
                    /// `scx_watchdog_timestamp` correctly.
                    phys_base: u64,
                    /// Runtime VIRTUAL KASLR slide — distinct from
                    /// `phys_base`, the PHYSICAL slide. `scx_tasks` is a
                    /// link-time symbol but the global list's `.next`
                    /// pointers are runtime addresses, so the global
                    /// walk's terminator must compare against the runtime
                    /// head `slid_kernel_kva(scx_tasks_kva, kaslr_offset)`.
                    /// Sourced from `coord_kaslr_offset()`; `0` when KASLR
                    /// is off / not yet derived.
                    kaslr_offset: u64,
                }
                // Lazy-construct BpfMapAccessorOwned. The constructor
                // parses vmlinux ELF (goblin) and BTF (~MB-scale
                // work) and reads guest-memory bootstrap symbols
                // (`page_offset_base`, `pgtable_l5_enabled`,
                // `init_top_pgt`); the latter aren't readable until
                // the guest kernel has populated them, so a
                // construction attempt at coord-start can fail with
                // a still-booting guest. The fix is the same lazy-
                // discovery pattern that `cached_bss_pa` uses below:
                // try each iteration until success, then cache —
                // gated on `owned_accessor.is_none()` so the heavy
                // parse runs at most once per coordinator (only the
                // failed attempts re-pay it, and only until the
                // first success). A single one-shot construct at
                // coord-start would have left the accessor None
                // permanently if the guest hadn't booted yet,
                // disabling freeze detection AND the dump for the
                // entire run.
                // Cached vmlinux bytes shared across every retry of
                // `try_init_owned_accessor` and
                // `try_init_owned_prog_accessor`. The previous code
                // re-ran `std::fs::read(vmlinux)` inside both helpers
                // on every scan tick — at 50-340 MB per call on cold
                // disk cache the pair could exceed the 12 s post-
                // BSP-done kill timer before the coord ever reached
                // its epoll wait. Reading once at coord scope cuts
                // the per-iteration cost to a few-millisecond
                // `goblin::elf::Elf::parse` against the cached bytes.
                //
                // The borrow lifetime constraint blocks caching the
                // parsed `Elf<'static>` (it borrows from the Vec); the
                // helpers re-parse the cached bytes per call instead.
                // Parsing is microseconds — only the file read was
                // slow.
                let _tvmr = std::time::Instant::now();
                let vmlinux_data: Option<Arc<Vec<u8>>> = vmlinux_data_shared.clone();
                // Worker-populated accessor pair. Built off the freeze
                // coordinator thread so the slow ELF + BTF parse +
                // symbol HashMap (~4 s on debug vmlinux) does not
                // block the coordinator from servicing TOKEN_TX
                // events on its epoll loop. The worker writes both
                // accessors atomically via `OnceLock::set` once the
                // GuestKernel handshake succeeds and both BTF parses
                // land. Subsequent reads from the coordinator are
                // nanosecond-scale `OnceLock::get` calls.
                //
                // `Arc<OnceLock<(...)>>` shape: `Arc` so the worker
                // and coordinator share ownership; `OnceLock` so the
                // publish is one-shot and lock-free on read; the
                // tuple shape so both accessors land atomically — a
                // failure-dump path that builds a `ScxWalkerCapture`
                // must also have the matching `prog_runtime_stats`
                // accessor, so partial pairs would skew the dump.
                //
                // `GuestMemMapAccessorOwned` and
                // `GuestMemProgAccessorOwned` are `Send` because they
                // own `GuestKernel`, which holds `Arc<GuestMem>`
                // (was `&'a GuestMem`). The Arc shape lets the worker
                // own the kernel handle independently of the
                // coordinator's stack.
                // Re-acquirable accessor slot. Replaces the prior
                // `OnceLock<...>` shape so the slot can be re-populated
                // after a scheduler-detach / scheduler-replace event:
                // the BPF maps the old scheduler used are RCU-
                // freed when sched_ext unregisters, and the new
                // scheduler's maps land at fresh kernel-allocated
                // addresses. Without re-acquisition, post-swap periodic
                // captures would silently read stale memory (the prior
                // scheduler's BSS, now recycled) for the lifetime of
                // the run.
                //
                // The worker (spawned below) writes into the slot via
                // `lock().replace(...)` on every successful init; it
                // loops on `accessor_reinit_evt` so a reset triggered
                // by the coordinator's `WatchpointPublishResult::
                // Detached` / `RebindDisarmed` arms wakes the worker,
                // re-runs init, and re-publishes a
                // fresh pair. The coordinator's adopt site
                // (`owned_accessor.is_none()` check below) takes the
                // pair out of the slot via `lock().take()` so the
                // slot returns to `None` and the worker can publish
                // a future re-acquisition into the same slot.
                type AccessorPair = (
                    crate::monitor::bpf_map::GuestMemMapAccessorOwned,
                    Option<crate::monitor::bpf_prog::GuestMemProgAccessorOwned>,
                );
                let accessors_slot: Arc<std::sync::Mutex<Option<AccessorPair>>> =
                    Arc::new(std::sync::Mutex::new(None));
                // Re-init trigger eventfd. Coordinator writes 1 to
                // signal a scheduler-detach/-replace; worker's loop
                // wakes from `poll()` and re-enters the init retry
                // loop. The fd is non-blocking (EFD_NONBLOCK) so
                // spurious writes are bounded — multiple writes
                // before a single read coalesce to one wake.
                let accessor_reinit_evt: Arc<EventFd> = Arc::new(
                    EventFd::new(EFD_NONBLOCK)
                        .expect("eventfd for accessor_reinit"),
                );
                // Owned views into the slot. Switched from `Option<&T>`
                // (borrowed from OnceLock) to `Option<T>` (owned by the
                // coordinator) so the slot can be re-populated under
                // the coordinator's feet without invalidating outstanding
                // borrows. `Option::as_ref()` still yields `Option<&T>`
                // at every call site, so down-stream code that takes a
                // reference compiles unchanged.
                let mut owned_accessor:
                    Option<crate::monitor::bpf_map::GuestMemMapAccessorOwned> = None;
                let mut owned_prog_accessor:
                    Option<crate::monitor::bpf_prog::GuestMemProgAccessorOwned> = None;
                // Virt-KASLR offset accessor — re-reads the shared
                // `kern_virt_kaslr` Arc on every call so consumers
                // observe the freshest value the BSP MSR_LSTAR path
                // or the guest-channel KERN_ADDRS path has published.
                // `saturating_sub(1)` folds the `+1` bias (0 sentinel
                // → 0, 1 → 0, N>1 → N-1). The closure capture keeps
                // call sites read like a plain value (`coord_kaslr_offset()`)
                // and avoids a stale local snapshot that could
                // produce different results in different iterations
                // when the publisher fires mid-loop.
                let coord_kaslr_offset = || -> u64 {
                    kern_virt_kaslr
                        .load(std::sync::atomic::Ordering::Acquire)
                        .saturating_sub(1)
                };
                // True once the publisher (MSR_LSTAR readback or
                // KERN_ADDRS chain) has written a `+1`-biased value
                // — including the biased-0 (raw = 1) write under
                // `nokaslr` where the actual offset is 0. The raw
                // atomic distinguishes "unpublished" (load == 0)
                // from "published as 0" (load == 1, decoded as 0);
                // [`coord_kaslr_offset`] cannot since both decode
                // to 0. Defer/drain gates that ask "has the publish
                // chain landed?" — independent of slide value —
                // must use this predicate, not the value getter,
                // otherwise `kaslr = false` test runs defer forever
                // (the publish DOES land with value 0; the gate keeps
                // seeing offset == 0 and re-defers).
                let kern_virt_kaslr_published = || -> bool {
                    kern_virt_kaslr.load(std::sync::atomic::Ordering::Acquire) != 0
                };
                // Spawn the accessor-init worker before entering the
                // coordinator's epoll loop. The worker:
                //   1. Loops `try_init_owned_accessor` +
                //      `try_init_owned_prog_accessor` against the
                //      shared `Arc<GuestMem>` until both succeed.
                //   2. On success: stores `phys_base + 1` (biased) in
                //      `kern_phys_base` via `compare_exchange(0, ..)`
                //      so the monitor thread observes the value
                //      regardless of whether the guest's port-2
                //      publish landed first.
                //   3. Publishes the pair by storing `Some(pair)` into
                //      the `accessors_slot` `Mutex<Option<AccessorPair>>`.
                //   4. Exits. The coordinator adopts via
                //      `accessors_slot.lock().take()`; the slot is
                //      re-populatable — the swap-notify teardown and the
                //      watchpoint Detached / RebindDisarmed arms reset it
                //      to `None` to force a fresh worker rebuild on a swap.
                //
                // The worker honors `freeze_coord_kill` between
                // retries and bails immediately on shutdown so a
                // still-booting VM that's killed mid-init does not
                // delay coord teardown. The 60s budget is the same
                // order as `start_bpf_map_write`'s phase-1 deadline;
                // a boot that hasn't published the bootstrap symbols
                // by then is genuinely stuck and the dump path is
                // unavailable for the rest of the run regardless.
                //
                // After a successful publish the worker parks on
                // `poll(kill_evt | accessor_reinit_evt, -1)` and
                // re-enters the init-retry phase whenever the
                // coordinator pulses `accessor_reinit_evt` (see the
                // synchronous sched-swap-notify teardown and the
                // `WatchpointPublishResult::Detached` / `RebindDisarmed`
                // arms in the coord-loop body).
                // Re-init drops the per-iteration 60 s budget — by
                // the time the first re-init fires the guest has
                // already booted to the point of having a live
                // scheduler attached, so `from_elf_with_hint` should
                // succeed on the first attempt; the inner retry
                // still tolerates the brief slab-reuse window between
                // an old scheduler's BPF map free and the new
                // scheduler's BPF map load.
                let accessor_init_handle: Option<std::thread::JoinHandle<()>> = match (
                    freeze_coord_mem.as_ref(),
                    freeze_coord_vmlinux.as_ref(),
                    vmlinux_data.as_deref(),
                ) {
                    (Some(mem), Some(vmlinux), Some(data)) => {
                        let mem_for_worker = mem.clone();
                        let vmlinux_for_worker = vmlinux.clone();
                        let data_for_worker = data.clone();
                        let tcr_for_worker = freeze_coord_tcr_el1.clone();
                        let cr3_for_worker = freeze_coord_cr3.clone();
                        let kern_phys_base_for_worker = kern_phys_base.clone();
                        let kern_phys_base_evt_for_worker = kern_phys_base_evt.clone();
                        let accessor_ready_evt_for_worker = accessor_ready_evt.clone();
                        let kill_for_worker = freeze_coord_kill.clone();
                        let kill_evt_for_worker = freeze_coord_kill_evt.clone();
                        let slot_for_worker = accessors_slot.clone();
                        let reinit_for_worker = accessor_reinit_evt.clone();
                        let publish_seqno_for_worker = accessor_publish_seqno.clone();
                        let worker_state_for_worker = accessor_worker_state.clone();
                        let dispatcher_wake_for_worker =
                            accessor_dispatcher_wake_evt.clone();
                        std::thread::Builder::new()
                            .name("vmm-accessor-init".into())
                            .spawn(move || {
                                let _init_t0 = Instant::now();
                                // poll() on kill_evt so the worker
                                // wakes instantly on shutdown instead
                                // of sleeping through a 100ms window.
                                let kill_fd = {
                                    use std::os::unix::io::AsRawFd;
                                    kill_evt_for_worker.as_raw_fd()
                                };
                                let reinit_fd = {
                                    use std::os::unix::io::AsRawFd;
                                    reinit_for_worker.as_raw_fd()
                                };
                                let elf = match goblin::elf::Elf::parse(&data_for_worker) {
                                    Ok(e) => e,
                                    Err(e) => {
                                        tracing::warn!(
                                            error = %e,
                                            "accessor-init: vmlinux ELF parse failed"
                                        );
                                        // Mark FAILED_PERMANENTLY so
                                        // a swap-op wait surfaces the
                                        // terminal-worker diagnostic
                                        // rather than the generic
                                        // deadline-exceeded message.
                                        worker_state_for_worker.store(
                                            crate::scenario::snapshot::bridge::accessor_worker_state::FAILED_PERMANENTLY,
                                            Ordering::Release,
                                        );
                                        // Wake any pending dispatcher
                                        // wait so it surfaces the
                                        // FAILED_PERMANENTLY bail
                                        // immediately instead of
                                        // waiting for its deadline.
                                        let _ = dispatcher_wake_for_worker.write(1);
                                        return;
                                    }
                                };
                                let mut first_init = true;
                                'reinit: loop {
                                    // First init honours the 60 s boot
                                    // budget. Re-inits (after a scheduler
                                    // swap) get a shorter 30 s budget: the
                                    // guest is already booted, so a
                                    // transient slab-reuse race resolves
                                    // within a handful of retries; a wedged
                                    // swap that never rebuilds must still
                                    // stop retrying (the prior deadline=None
                                    // busy-looped every ~100-200 ms until
                                    // kill).
                                    let is_first_init = first_init;
                                    let deadline = if is_first_init {
                                        Instant::now() + Duration::from_secs(60)
                                    } else {
                                        Instant::now() + Duration::from_secs(30)
                                    };
                                    first_init = false;
                                    let publish: Option<AccessorPair> = loop {
                                        if kill_for_worker.load(Ordering::Acquire) {
                                            return;
                                        }
                                        if Instant::now() >= deadline {
                                            if is_first_init {
                                                tracing::warn!(
                                                    "freeze-coord accessor-init worker: \
                                                     60s first-init deadline exceeded; \
                                                     coordinator will run without an \
                                                     owned-accessor pair (freeze dump \
                                                     path unavailable)"
                                                );
                                                // First-init budget burned
                                                // without ever publishing —
                                                // no accessor exists. Mark
                                                // FAILED_PERMANENTLY so the
                                                // dispatch wait sees the
                                                // terminal state instead of
                                                // blocking the full deadline
                                                // on a worker that gave up.
                                                worker_state_for_worker.store(
                                                    crate::scenario::snapshot::bridge::accessor_worker_state::FAILED_PERMANENTLY,
                                                    Ordering::Release,
                                                );
                                                let _ = dispatcher_wake_for_worker.write(1);
                                                break None;
                                            }
                                            // Re-init budget exceeded — a
                                            // wedged scheduler swap that did
                                            // not settle. Do NOT mark
                                            // FAILED_PERMANENTLY: a prior
                                            // accessor is still in the slot
                                            // and a later swap may rebuild
                                            // cleanly. Stop busy-spinning —
                                            // break to the park loop below,
                                            // wait for the next reinit_evt,
                                            // and keep the prior accessor
                                            // usable.
                                            tracing::warn!(
                                                "freeze-coord accessor-init worker: \
                                                 30s re-init deadline exceeded \
                                                 (scheduler swap did not settle); \
                                                 keeping the prior accessor and \
                                                 parking for the next reset"
                                            );
                                            break None;
                                        }
                                        // Use the guest-reported phys_base
                                        // (includes kaslr_offset) as a hint
                                        // so GuestKernel gets the correct
                                        // value instead of 0 from the
                                        // failing page-table walk.
                                        let biased = kern_phys_base_for_worker
                                            .load(Ordering::Acquire);
                                        let tcr_val = tcr_for_worker
                                            .as_ref()
                                            .map(|c| c.load(Ordering::Acquire))
                                            .unwrap_or(0);
                                        let cr3_val = cr3_for_worker.load(Ordering::Acquire);
                                        let pb_hint = if biased != 0 {
                                            biased.wrapping_sub(1)
                                        } else if cfg!(target_arch = "aarch64")
                                            && tcr_val != 0
                                            && cr3_val != 0
                                        {
                                            // aarch64 accessor decouple: phys_base =
                                            // code_start - ram_start = text_offset = 0 for a
                                            // relocatable kernel (CONFIG_RANDOMIZE_BASE pinned
                                            // in ktstr.kconfig; KERNEL_LOAD_ADDR == DRAM_START,
                                            // aarch64/kvm.rs:22/90), so from_elf_with_hint with
                                            // hint=0 yields the IDENTICAL accessor as the
                                            // eventual KERN_ADDRS hint (which also resolves
                                            // phys_base to 0 via wrapping_sub(1)). Build the
                                            // accessor the moment the MMU is up (tcr_el1 + cr3
                                            // programmed, early boot) instead of blocking on
                                            // KERN_ADDRS, whose delivery waits on the guest's
                                            // slow bulk-port-1 virtio_console handshake (~9s
                                            // post-attach) and otherwise defers every periodic
                                            // capture into the final scenario step (collapsing
                                            // the per-phase pipeline to one bucket). The KASLR
                                            // text offset still arrives via KERN_ADDRS for the
                                            // kaslr publish gate (~L8204); only the map-accessor
                                            // build is unblocked here. The host runs the guest
                                            // same-arch under KVM, so cfg!(target_arch) is the
                                            // guest arch. x86_64 keeps waiting for the real
                                            // phys_base (no fixed-zero guarantee there).
                                            //
                                            // cr3 stability: the cached cr3 (TTBR1, read by the
                                            // BSP loop) is swapper_pg_dir here, NOT the transient
                                            // init_pg_dir. map_kernel installs init_pg_dir then
                                            // switches to swapper_pg_dir inside the PI stub
                                            // (arch/arm64/kernel/pi/map_kernel.c) while the MMU
                                            // comes online — no MMIO/interrupt occurs, so the vCPU
                                            // never exits to userspace in that window and the BSP
                                            // loop's first non-zero TTBR1 read is already
                                            // swapper_pg_dir (a stable .bss symbol, never freed).
                                            // A future change adding any very-early guest exit
                                            // could latch init_pg_dir (reclaimed by free_initmem)
                                            // → the cached cr3 would then walk garbage; validating
                                            // cr3 == swapper_pg_dir's PA before accepting this
                                            // early build would catch that (deferred — the PA
                                            // derivation is VA_BITS-sensitive).
                                            0
                                        } else {
                                            let pb_evt_fd = {
                                                use std::os::unix::io::AsRawFd;
                                                kern_phys_base_evt_for_worker.as_raw_fd()
                                            };
                                            let mut pfds = [
                                                libc::pollfd { fd: kill_fd, events: libc::POLLIN, revents: 0 },
                                                libc::pollfd { fd: pb_evt_fd, events: libc::POLLIN, revents: 0 },
                                            ];
                                            unsafe { libc::poll(pfds.as_mut_ptr(), 2, 200) };
                                            continue;
                                        };
                                        let map_res = crate::monitor::bpf_map::GuestMemMapAccessorOwned
                                            ::from_elf_with_hint(
                                                mem_for_worker.clone(),
                                                &elf,
                                                &data_for_worker,
                                                &vmlinux_for_worker,
                                                tcr_val,
                                                cr3_val,
                                                pb_hint,
                                            );
                                        if kill_for_worker.load(Ordering::Acquire) {
                                            return;
                                        }
                                        if let Ok(map) = map_res {
                                            let po = map.guest_kernel()
                                                .walk_context().page_offset;
                                            if po & (1u64 << 63) == 0
                                                || po & 0xFFF != 0
                                            {
                                                let mut pfd = libc::pollfd {
                                                    fd: kill_fd,
                                                    events: libc::POLLIN,
                                                    revents: 0,
                                                };
                                                unsafe {
                                                    libc::poll(&mut pfd, 1, 100);
                                                }
                                                continue;
                                            }
                                            let phys_base =
                                                map.guest_kernel().phys_base();
                                            let _ = kern_phys_base_for_worker
                                                .compare_exchange(
                                                    0,
                                                    phys_base.wrapping_add(1),
                                                    Ordering::Release,
                                                    Ordering::Relaxed,
                                                );
                                            // Reuse the map accessor's already-built
                                            // GuestKernel for the prog accessor instead of
                                            // rebuilding it. The kernel image (symbols,
                                            // TTBR1/TCR/phys_base, page_offset) is identical
                                            // for both accessors — only prog_idr_kva +
                                            // prog offsets differ. Rebuilding via
                                            // from_elf_with_symbols re-parsed the full
                                            // vmlinux symtab and re-walked the page tables
                                            // (~7 s on aarch64), which delayed the accessor
                                            // publish past the periodic-capture window so
                                            // no snapshot ever fired (periodic_fired=0).
                                            // Cloning shares the Arc-backed `mem` +
                                            // `symbols` — cheap — and `finish` only adds the
                                            // prog IDR symbol + BTF offsets.
                                            let prog_res =
                                                crate::monitor::bpf_prog::GuestMemProgAccessorOwned::finish(
                                                    map.guest_kernel().clone(),
                                                    &elf,
                                                    &data_for_worker,
                                                    &vmlinux_for_worker,
                                                );
                                            break Some((map, prog_res.ok()));
                                        }
                                        // Wait on kill_evt with 200ms
                                        // timeout. Wakes instantly on
                                        // kill; retries on timeout.
                                        let mut pfd = libc::pollfd {
                                            fd: kill_fd,
                                            events: libc::POLLIN,
                                            revents: 0,
                                        };
                                        unsafe {
                                            libc::poll(&mut pfd, 1, 200);
                                        }
                                    };
                                    if let Some(pair) = publish {
                                        // Replace any leftover pair (from
                                        // a re-init race where the coord
                                        // hasn't yet adopted the previous
                                        // publish) so the slot always
                                        // holds the freshest values.
                                        if let Ok(mut guard) = slot_for_worker.lock() {
                                            *guard = Some(pair);
                                        }
                                        // Bump publish seqno + flip
                                        // worker_state to SUCCEEDED
                                        // BEFORE the eventfd writes so
                                        // any dispatcher poll that
                                        // wakes on the fd and re-reads
                                        // the atomic sees the advance.
                                        // Release ordering pairs with
                                        // the dispatcher's Acquire
                                        // loads in
                                        // wait_for_accessor_publish_advance
                                        // / wait_for_worker_state_not_trying.
                                        publish_seqno_for_worker
                                            .fetch_add(1, Ordering::Release);
                                        worker_state_for_worker.store(
                                            crate::scenario::snapshot::bridge::accessor_worker_state::SUCCEEDED,
                                            Ordering::Release,
                                        );
                                        // Two distinct wake fds: the
                                        // coord drains accessor_ready_evt
                                        // via its epoll loop, and the
                                        // dispatcher (Op::AttachScheduler /
                                        // Op::ReplaceScheduler) drains
                                        // accessor_dispatcher_wake_evt via
                                        // its bridge-side poll. Separate
                                        // fds prevent the two consumers
                                        // from racing for the same wake
                                        // count.
                                        let _ = accessor_ready_evt_for_worker.write(1);
                                        let _ = dispatcher_wake_for_worker.write(1);
                                    }
                                    // Park until kill or reinit. 100 ms
                                    // poll timeout so a kill_evt missed
                                    // because of a teardown race (e.g.
                                    // the coord's own epoll loop drains
                                    // kill_evt before the worker sees
                                    // POLLIN) is bounded to ~100 ms of
                                    // shutdown lag — without the
                                    // timeout, an indefinite poll on a
                                    // drained eventfd would never wake
                                    // and the test would hang at
                                    // `handle.join()`. Inner loop reads
                                    // reinit_evt with EAGAIN-tolerant
                                    // `is_ok()` (eventfd constructed
                                    // with EFD_NONBLOCK) so the 100 ms
                                    // timeout wakes are pure kill polls
                                    // and don't trigger spurious re-init.
                                    loop {
                                        let mut pfds = [
                                            libc::pollfd { fd: kill_fd, events: libc::POLLIN, revents: 0 },
                                            libc::pollfd { fd: reinit_fd, events: libc::POLLIN, revents: 0 },
                                        ];
                                        unsafe { libc::poll(pfds.as_mut_ptr(), 2, 100) };
                                        if kill_for_worker.load(Ordering::Acquire) {
                                            return;
                                        }
                                        if reinit_for_worker.read().is_ok() {
                                            break;
                                        }
                                    }
                                    continue 'reinit;
                                }
                            })
                            .ok()
                    }
                    _ => None,
                };
                // Per-CPU offset array used by `runtime_stats` to
                // locate each CPU's `bpf_prog_stats` slot. Resolved
                // once after `owned_prog_accessor` lands by reading
                // `__per_cpu_offset` from guest memory; cached so
                // every dump iteration reuses it. None until either
                // the prog accessor isn't ready yet or the
                // `__per_cpu_offset` symbol couldn't be located in
                // the kernel's symbol table.
                let mut prog_per_cpu_offsets: Option<Vec<u64>> = None;
                // BTF + arena offsets resolved once at coordinator
                // start. Used by `dump_state` after the rendezvous
                // succeeds to render every BPF map's contents. None
                // values disable rendering for the relevant code path
                // (no BTF → no BTF-driven rendering at all; no arena
                // offsets → arena maps fall back to an explanatory
                // error string in the report).
                //
                // Arena offsets derive from the same parsed `Btf`
                // handle (`from_btf`, not `from_vmlinux`) so the
                // ELF-to-BTF parse runs exactly once per coordinator
                // — a second `from_vmlinux` would re-read and
                // re-parse the same file.
                let dump_btf = vmlinux_data.as_deref().zip(freeze_coord_vmlinux.as_ref())
                    .and_then(|(data, path)| crate::monitor::btf_offsets::load_btf_from_bytes(data, path).ok());
                let dump_arena_offsets = dump_btf
                    .as_ref()
                    .and_then(|btf| crate::monitor::arena::BpfArenaOffsets::from_btf(btf).ok());
                // Per-CPU CPU-time / softirq / IRQ / iowait offsets
                // and the matching `.data..percpu` symbol KVAs.
                // Resolved once at coordinator start, mirroring
                // `dump_arena_offsets`. Both Option-typed: a stripped
                // vmlinux without any of `kernel_cpustat` / `kstat` /
                // `tick_cpu_sched` symbols still resolves the BTF
                // offsets fine, but the dump path checks both sides
                // before constructing a `CpuTimeCapture` so the
                // capture site only fires when the data is actually
                // readable.
                let dump_cpu_time_offsets = dump_btf
                    .as_ref()
                    .and_then(|btf| crate::monitor::btf_offsets::CpuTimeOffsets::from_btf(btf).ok());
                // Per-cgroup PSI-irq walk offsets (Phase A): the cgroup
                // hierarchy field offsets + the shared psi_group offsets (a
                // per-cgroup psi_group is the same struct as the system-wide
                // psi_system). Either None → the per-cgroup capture is skipped
                // (loud-absent).
                let dump_cgroup_offsets = dump_btf.as_ref().and_then(|btf| {
                    crate::monitor::btf_offsets::CgroupWalkOffsets::from_btf(btf).ok()
                });
                let dump_cgroup_psi_offsets = dump_btf.as_ref().and_then(|btf| {
                    crate::monitor::btf_offsets::PsiGroupOffsets::from_btf(btf).ok()
                });
                let dump_cpu_time_symbols = vmlinux_data.as_deref()
                    .and_then(|data| crate::monitor::symbols::KernelSymbols::from_vmlinux_bytes(data).ok());
                // SCX walker BTF sub-group offsets. Resolved once at
                // coord start; per-sub-group resolution failures land
                // inside the composite as None so the walker's
                // `missing_groups()` can report which passes are blind
                // (a kernel built without CONFIG_NUMA loses
                // `scx_sched_pnode`, etc.).
                let dump_scx_walker_offsets = dump_btf
                    .as_ref()
                    .and_then(|btf| {
                        crate::monitor::btf_offsets::ScxWalkerOffsets::from_btf(btf).ok()
                    });
                // Per-task enrichment BTF offsets. All-or-nothing —
                // any missing sub-group leaves the composite Err and
                // the enrichment capture is skipped. The walker
                // never runs partially: every Tier-1 field must be
                // resolvable, otherwise the dump path falls back to
                // `REASON_NO_TASK_WALKER`.
                let dump_task_enrichment_offsets = dump_btf
                    .as_ref()
                    .and_then(|btf| {
                        crate::monitor::btf_offsets::TaskEnrichmentOffsets::from_btf(btf).ok()
                    });
                // Per-node NUMA event BTF offsets. Required for the
                // per-node `vm_numa_event[]` walker. Resolved once at
                // coord start; absent on stripped vmlinux or kernels
                // built without `CONFIG_NUMA + CONFIG_VM_EVENT_COUNTERS`.
                let dump_numa_offsets = dump_btf
                    .as_ref()
                    .and_then(|btf| {
                        crate::monitor::btf_offsets::NumaStatsOffsets::from_btf(btf).ok()
                    });
                // Hoisted scan_ctx prerequisites. These are pure
                // functions of the host inputs (vmlinux ELF and the
                // already-loaded BTF), so they succeed or fail
                // deterministically at coord-start — no boot-race
                // window to retry through. Computing once here avoids
                // re-parsing the BTF on every scan_ctx try_resolve
                // iteration. The previous per-iteration retry pattern
                // was harmless functionally (idempotent) but burned
                // ~MB-scale ELF reparse work every SCAN_INTERVAL until
                // owned_accessor caught up. These two values plus
                // `dump_cpu_time_symbols.scx_tasks` and `runqueues`
                // feed RunnableScanCtx construction below — the
                // global walker reads `scx_tasks` directly via
                // `text_kva_to_pa_with_base` (or
                // `GuestKernel::text_kva_to_pa`), the per-rq walker uses
                // `runqueues` + `__per_cpu_offset` to address each
                // CPU's `rq`.
                let scan_offsets = dump_btf.as_ref().and_then(|btf| {
                    crate::monitor::btf_offsets::RunnableScanOffsets::from_btf(btf).ok()
                });
                // jiffies_64 lives on the KernelSymbols instance
                // computed above for the dump capture. Reusing it
                // pays a single from_vmlinux cost per coordinator.
                let scan_jiffies_64_kva =
                    dump_cpu_time_symbols.as_ref().and_then(|s| s.jiffies_64);
                // Lazy-discovered cached PA of `ktstr_err_exit_detected`
                // within the probe BPF program's .bss map. None until
                // the probe loads into map_idr (rust_init phase 2b);
                // discovery retries each iteration until success.
                //
                // Invalidated each scan tick when the source `.bss`
                // map disappears from `map_idr` or rebinds to a
                // different `value_kva` — see the rediscovery guard
                // below. Without that, a probe BPF program that
                // unloads mid-run leaves the freed vmalloc page's
                // PA cached here; the kernel can re-allocate that
                // page for unrelated guest memory, and the next
                // `read_u32(pa, 0)` returns whatever bytes that
                // page now holds (any non-zero value latches a
                // phantom `err_triggered` and synthesizes a bogus
                // failure dump).
                let mut cached_bss_pa: Option<u64> = None;
                // Companion to `cached_bss_pa`: the `value_kva` of
                // the `.bss` map that produced it. Used as a stale-
                // probe canary — if the next scan tick finds the
                // same-named map with a different `value_kva` (the
                // bpf_array slab moved across an unload+reload) the
                // PA is invalidated and re-resolved. Stays in sync
                // with `cached_bss_pa`: both Some or both None.
                let mut cached_bss_value_kva: Option<u64> = None;
                // Cache the BTF-resolved offset of the field within
                // the .bss section. The Datasec walk parses the
                // probe's BTF (a few-KB blob copy + parse) every
                // call — caching keeps that work to once-per-coord-
                // lifetime instead of once-per-discovery-iteration.
                // Resolution can fail two ways:
                //   - guest still booting → retry (offset stays None)
                //   - BTF parse / Datasec walk broken → fall back
                //     to offset 0 once, log a warn, and stop retrying
                //     (warn_logged is the latch).
                let mut cached_bss_offset: Option<u32> = None;
                let mut bss_offset_warn_logged = false;
                // One-shot latch for the cached_bss_pa-points-OOB
                // diagnostic. The OOB read state can occur if the
                // cached PA was resolved against a probe `.bss` map
                // that has since been freed (probe unload mid-run,
                // vmalloc page recycled). The first observation
                // surfaces a warn so an operator inspecting the run
                // knows the .bss path has gone silent; subsequent
                // observations stay debug-level so the logs do not
                // fill up across the remaining run lifetime.
                let mut bss_oob_warn_logged = false;
                // Cached `*scx_root` value (the vmalloc/slab KVA of
                // the live `struct scx_sched`). Tracked across scan
                // ticks so we can detect a sched_ext detach + reattach
                // cycle: when the kernel tears down the scheduler the
                // pointer goes 0 (the slab page is freed); when a new
                // scheduler attaches it points at a fresh slab. Each
                // change re-publishes `request_kva` AND
                // `kind_host_ptr` so vCPU threads re-arm on the new
                // KVA and post-fire `read_volatile` reads land on the
                // current slab — the previous one-shot publish gate
                // pinned `kind_host_ptr` at the original slab page
                // forever, and a stale deref after rebind would touch
                // freed (or repurposed) host memory.
                //
                // Resolution sequence per scan tick:
                //   1. read scx_root_kva from KernelSymbols (resolved
                //      once at coord-start via vmlinux);
                //   2. translate scx_root_kva → root_pa via
                //      `GuestKernel::text_kva_to_pa` (it lives in the
                //      kernel text mapping, not vmalloc);
                //   3. read u64 at root_pa to get sched_kva (the
                //      vmalloc-allocated `struct scx_sched`);
                //   4. compare against `last_sched_kva` — bail on no
                //      change (fast path on every scan tick post-
                //      attach);
                //   5. on change to non-zero: publish
                //      `sched_kva + exit_kind_offset` into
                //      `request_kva` (and the matching host pointer
                //      into `kind_host_ptr`); each vCPU thread polls
                //      that slot before its next KVM_RUN and re-arms;
                //   6. on change to zero (detach): publish 0 / null so
                //      vCPUs disarm via `KVM_SET_GUEST_DEBUG` without
                //      this slot's enable bits and stop tripping on
                //      the now-freed slab address.
                //
                // `*scx_root` only becomes non-NULL once a sched_ext
                // scheduler attaches; before that we silently retry
                // — the BPF .bss fallback (still wired up below)
                // covers the gap.
                let mut last_sched_kva: u64 = 0;
                let mut cached_exit_kind_pa: Option<u64> = None;
                let mut freeze_state = FreezeState::Idle;
                // Cached early snapshot from a midway-trigger freeze.
                // Held until the late freeze fires; then both early
                // and late are wrapped into a DualFailureDumpReport
                // and emitted as one file. The late-trigger Captured
                // / Degraded / Suppressed arms each consume the early
                // via `.take()` and emit it alongside the late as a
                // tagged sibling per the
                // [`crate::monitor::dump::SNAPSHOT_TAG_EARLY_*`]
                // constants; if the late trigger never fires for the
                // run, the end-of-coord drain at the bottom of this
                // closure body emits the early to
                // [`crate::monitor::dump::SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED`].
                // Every captured early reaches disk via one of the
                // four paths.
                //
                // EarlySnapshotGuard wraps `early_snapshot` and
                // `early_retain_tag` so a panic-unwind anywhere in the
                // coord closure body still flushes a held Captured
                // early to disk via the Drop impl. The normal-path
                // end-of-coord drain (below) calls
                // `early_guard.drain_to_disk()` explicitly; Drop's
                // call is a no-op in the normal case because the
                // snapshot.take() in the normal drain leaves the
                // guard's `snapshot` field None.
                //
                // `retain_tag` field: when the late-trigger Degraded /
                // Suppressed arm fails to write its tagged sibling
                // and retains the snapshot for the end-of-coord drain
                // to retry, the failed arm sets
                // `early_guard.retain_tag = Some(...)` so the drain
                // reads it on retry and lands the recovered file at
                // the operator-correct path (e.g.
                // `early-pre-late-degraded` for a failed Degraded-arm
                // write) rather than the default
                // `early-only-late-never-fired` which would
                // misrepresent the case. `None` means "no retry
                // pending — the late never fired" (drain uses its
                // default tag).
                let mut early_guard = EarlySnapshotGuard {
                    snapshot: None,
                    retain_tag: None,
                    dump_path: freeze_coord_dump_path.clone(),
                    dual_snapshot: freeze_coord_dual_snapshot,
                };
                // Stash the early-trigger Degraded reason so the
                // late-trigger emit can surface it via
                // `DualFailureDumpReport::early_skipped_reason`.
                // Without this, an early-Degraded outcome leaves
                // `early_snapshot` None AND `early_peak_max_age_jiffies`
                // already populated (the max_age was read before the
                // Degraded outcome was matched), so the late-arm's
                // `early_skipped_reason` calculator falls through to
                // the existing "max_age never crossed threshold"
                // branch and surfaces an internally contradictory
                // reason (peak >= threshold) — operator reads a
                // misleading "max_age never crossed" when the actual
                // cause was a rendezvous timeout in the early capture.
                // Stashing the reason here makes the early-Degraded
                // tagged sibling discoverable from the dual wrapper.
                let mut early_degraded_reason: Option<String> = None;
                // Per-snapshot scanner metadata, captured at the
                // early-trigger site and threaded into the
                // DualFailureDumpReport wrapper alongside the
                // snapshots themselves. Both fields stay 0 when no
                // early snapshot fires, mirroring the report's
                // `skip_serializing_if = is_zero` behaviour.
                let mut early_max_age_jiffies: u64 = 0;
                let mut early_threshold_jiffies: u64 = 0;
                // Lazy-resolved runnable_at scanner context. Built
                // from the same vmlinux ELF the .bss-poll path uses.
                // None disables the early-trigger path on this
                // iteration but does not block the late path. Becomes
                // Some once every prerequisite resolves; cached for
                // the rest of the run.
                let mut scan_ctx: Option<RunnableScanCtx> = None;
                // Latest skip reason from try_resolve. Captures the
                // specific prerequisite that prevented resolution on
                // the most recent attempt (most useful when scan_ctx
                // is still None at the late-trigger point) so the
                // late-trigger emission can stamp it into
                // `DualFailureDumpReport::early_skipped_reason`. Set
                // back to None on a successful resolve so a once-
                // failed-then-recovered run does not carry stale
                // breadcrumbs forward.
                let mut scan_ctx_skip_reason: Option<&'static str> = None;
                // Retry counter and one-shot warn latch for the
                // scan_ctx resolve. The resolve runs once per
                // SCAN_INTERVAL (250 ms) poll iteration until it
                // succeeds; without a diagnostic an operator who
                // built ktstr against a kernel lacking
                // sched_ext_entity (or stripped of jiffies_64)
                // gets a silent dual-snapshot disable.
                // Wait `SCAN_CTX_WARN_AFTER_ITERS` iterations
                // (~3 s at 250 ms cadence) before warning so legit
                // boot-time delays (owned_accessor not yet ready,
                // GuestKernel handshake mid-flight) don't trigger
                // false alarms. The latch ensures the warn fires at
                // most once per VM run.
                let mut scan_ctx_retries: u32 = 0;
                let mut scan_ctx_warned: bool = false;
                const SCAN_CTX_WARN_AFTER_ITERS: u32 = 12;
                // The accessor-init worker spawned above owns the
                // retry/warn discipline for its two `try_init_*`
                // helpers; the coordinator no longer tracks
                // `accessor_retries` / `accessor_warned` /
                // `accessor_last_err` fields here. The constant below
                // is reused by the `prog_per_cpu_offsets` /
                // `scan_ctx` retry blocks further down.
                const LAZY_ACCESSOR_WARN_AFTER_ITERS: u32 = 10;
                // Sibling state for `try_init_prog_per_cpu_offsets`.
                // Two distinct failure modes warrant different
                // diagnostics: a missing `__per_cpu_offset` symbol
                // (`per_cpu_offset_kva == 0`) is a PERMANENT failure
                // that warns immediately on the first observation —
                // the symbol won't materialise mid-run, so retrying
                // silently masks a stripped vmlinux. Conversely, a
                // present symbol whose live array still has zero
                // slots (`offsets.contains(&0)`) is a TRANSIENT
                // boot-progress condition that resolves once the
                // guest's `setup_per_cpu_areas` populates each
                // CPU's slot; warn after `LAZY_ACCESSOR_WARN_AFTER_ITERS`
                // retries so a guest that genuinely fails to bring
                // up its per-CPU areas surfaces a diagnostic
                // instead of permanently-disabled
                // prog_runtime_stats. Each warn latches via its
                // own `_warned` bool to fire at most once per VM run.
                let mut per_cpu_offsets_retries: u32 = 0;
                let mut per_cpu_offsets_warned: bool = false;
                let mut per_cpu_offsets_kva_warned: bool = false;
                // Half of the configured watchdog timeout, expressed
                // in guest jiffies. Computed once from
                // freeze_coord_watchdog_half + freeze_coord_hz so each
                // poll's comparison is a cheap u64 compare against
                // the scan's max age.
                //
                // ms-precision conversion lives in [`duration_to_jiffies`];
                // see its doc for why the seconds-based form is wrong.
                let half_threshold_jiffies =
                    duration_to_jiffies(freeze_coord_watchdog_half, freeze_coord_hz);
                // Trajectory tracking for the early-trigger diagnostic.
                // Records the max `max_age` observed across the run
                // and how many scan iterations have run. Surfaced in a
                // warn when err_triggered fires while
                // freeze_state == Idle (i.e. the early path never
                // captured) so an operator can distinguish three
                // failure modes from a single log line:
                //
                //   - early_scan_iters == 0   → scan_ctx never resolved
                //                                (scan_ctx_warn already
                //                                fires earlier; this
                //                                cross-checks).
                //   - peak_max_age == 0       → scan ran but never
                //                                observed a live task
                //                                (likely empty
                //                                runnable_list, wrong
                //                                offsets, or the scan
                //                                was reading unmapped
                //                                memory).
                //   - peak_max_age > 0 but    → scan was working but
                //     < half_threshold          the kernel watchdog
                //                                fired before any task
                //                                aged past the
                //                                half-way mark (very
                //                                short stalls or an
                //                                err-class exit that
                //                                isn't a stall, e.g.
                //                                scx_bpf_error()).
                //
                // The `FailureDumpReport` Display impl in
                // `crate::monitor::dump::display` already points
                // operators at `RUST_LOG=ktstr=debug` for scan
                // resolution; this trajectory snapshot is the more
                // actionable signal because it's emitted at the
                // moment of failure with structured fields rather
                // than as a per-iteration debug stream.
                let mut early_peak_max_age_jiffies: u64 = 0;
                let mut early_scan_iters: u64 = 0;
                // Cadence policy. The loop blocks in `epoll_wait`
                // until one of the registered fds fires (kill,
                // bsp_done, virtio-console TX, watchpoint hit,
                // scanner tick) OR `POLL_TIMEOUT_MS` elapses. The
                // previous
                // implementation drove this by `thread::sleep(500
                // µs)` and a `poll_iter % 200 == 0` decimator. The
                // event-driven design wakes the coordinator within
                // microseconds of any trigger source — including
                // the watchpoint hit and the kill / bsp_done flips —
                // and only does heavy work (boot-race accessor
                // construction, BPF .bss-PA lookup, runnable_at
                // scan) when the periodic scanner timerfd fires.
                const POLL_TIMEOUT_MS: i32 = 500;
                // 250 ms gives enough resolution for typical
                // half-watchdog thresholds (e.g. 4000 jiffies on
                // a 1 kHz HZ kernel = 4 s, so a 250 ms scan
                // cadence catches the half-way crossing within
                // 6.25% of the threshold) while halving the
                // freeze coord's scan-tick CPU draw vs the
                // legacy 100 ms cadence. The early-trigger path
                // walks both the global `scx_tasks` list and
                // every per-CPU `rq->scx.runnable_list` per
                // tick; on a many-vCPU host the larger interval
                // matters.
                const SCAN_INTERVAL: Duration = Duration::from_millis(250);
                // Per-fd epoll tokens. Match-on tokens dispatches
                // events without re-reading fd numbers.
                const TOKEN_KILL: u64 = 0;
                const TOKEN_BSP_DONE: u64 = 1;
                const TOKEN_WATCHPOINT: u64 = 3;
                const TOKEN_SCANNER: u64 = 4;
                /// virtio-console tx_evt — wakes whenever the guest
                /// publishes a TX descriptor chain on port 0 or port 1.
                /// The coordinator drains port-1 bulk TLV bytes and
                /// promotes a SCHED_EXIT entry into the run-wide
                /// `kill` flag, and intercepts
                /// [`crate::vmm::wire::MSG_TYPE_SNAPSHOT_REQUEST`]
                /// frames so the matching dispatch (CAPTURE / WATCH)
                /// runs in the same iteration body and the reply
                /// is pushed back to the guest via
                /// [`crate::vmm::virtio_console::VirtioConsole::queue_input_port1`].
                /// Port-0 (console) TX wakes are harmless: the coord
                /// drain returns an empty buffer and the byte stays
                /// in the host stdout thread's `drain_output` slot.
                /// Port 2 TX (scheduler stats) does not reach this
                /// epoll set — the
                /// [`crate::vmm::sched_stats::SchedStatsClient`]
                /// owns its own drainer thread and stats_tx_evt
                /// epoll, leaving this coordinator unaffected by
                /// stats traffic.
                const TOKEN_TX: u64 = 5;
                const TOKEN_ACCESSOR_READY: u64 = 6;
                let epoll = match Epoll::new() {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            "freeze-coord: epoll_create1 failed; aborting coordinator"
                        );
                        return;
                    }
                };
                use std::os::unix::io::AsRawFd;
                let mut scanner_tfd = match TimerFd::new() {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            "freeze-coord: timerfd_create failed; aborting coordinator"
                        );
                        return;
                    }
                };
                if let Err(e) = scanner_tfd.reset(SCAN_INTERVAL, Some(SCAN_INTERVAL)) {
                    tracing::error!(
                        error = %e,
                        "freeze-coord: timerfd_settime failed; aborting coordinator"
                    );
                    return;
                }
                // Register every fd. Failure to register any one of
                // these would cause the coordinator to silently miss
                // a wake source, so abort instead of degrading.
                for (fd, token, name) in [
                    (freeze_coord_kill_evt.as_raw_fd(), TOKEN_KILL, "kill_evt"),
                    (
                        freeze_coord_bsp_done_evt.as_raw_fd(),
                        TOKEN_BSP_DONE,
                        "bsp_done_evt",
                    ),
                    (
                        freeze_coord_hit_evt.as_raw_fd(),
                        TOKEN_WATCHPOINT,
                        "watchpoint_hit_evt",
                    ),
                    (scanner_tfd.as_raw_fd(), TOKEN_SCANNER, "scanner_tfd"),
                    (freeze_coord_tx_evt.as_raw_fd(), TOKEN_TX, "virtio_console_tx_evt"),
                    (accessor_ready_evt.as_raw_fd(), TOKEN_ACCESSOR_READY, "accessor_ready_evt"),
                ] {
                    if let Err(e) = epoll.ctl(
                        ControlOperation::Add,
                        fd,
                        EpollEvent::new(EventSet::IN, token),
                    ) {
                        tracing::error!(
                            error = %e,
                            fd_name = name,
                            "freeze-coord: epoll_ctl ADD failed; aborting coordinator"
                        );
                        return;
                    }
                }
                let mut events_buf = [EpollEvent::default(); 6];
                // Accumulator for partially-received TLV bulk frames.
                // The kernel's virtio_console TX path issues
                // descriptor chains as the guest writes; a single
                // logical TLV frame can span multiple wakes if the
                // guest's `write_all` was split across pages or
                // descriptor sizes. The streaming
                // [`crate::vmm::bulk::HostAssembler`] retains partial
                // bytes across `feed` calls so a frame split across
                // multiple TX wakes is recovered without loss.
                //
                // SCHED_EXIT promotion: every drained message is
                // inspected for [`wire::MSG_TYPE_SCHED_EXIT`]; when
                // observed, the run-wide `kill` flag flips so the
                // BSP run loop and the watchdog exit promptly
                // instead of waiting for the watchdog deadline.
                let mut bulk_assembler = crate::vmm::bulk::HostAssembler::new();
                // Per-iteration accumulator for guest-side
                // [`crate::vmm::wire::MSG_TYPE_SNAPSHOT_REQUEST`]
                // frames the TOKEN_TX handler decoded. Drained later
                // in the iteration body where `freeze_and_dispatch` /
                // `thaw_and_barrier` / `arm_user_watchpoint` are in
                // scope; the dispatch frames a
                // `MSG_TYPE_SNAPSHOT_REPLY` TLV and pushes it back
                // through `queue_input_port1`. CRC-failed frames are
                // never appended — a torn frame would otherwise let
                // a hostile guest force a spurious capture, mirroring
                // the SCHED_EXIT promotion gate.
                let mut snapshot_requests_pending: Vec<SnapshotRequest> = Vec::new();
                // Per-iteration pending KernelOpRequest queue.
                // [`crate::vmm::freeze_coord::dispatch::dispatch_bulk_message`]'s
                // `MsgType::KernelOpRequest` arm decodes incoming
                // postcard frames into
                // [`crate::vmm::wire::KernelOpRequestPayload`] and
                // pushes them here; the coord-loop pending-request
                // processing block (mirrors the
                // `snapshot_requests_pending` drain shape) takes the
                // vec, triggers `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })`, runs
                // `gmem.write_obj` / `gmem.read_obj` per entry while
                // all vCPUs are parked, and ships a
                // [`crate::vmm::wire::KernelOpReplyPayload`] back to
                // the guest over `port1_tx_buf` per request. CRC-bad
                // frames + malformed postcard payloads never reach
                // this vec — they drop silently at the dispatch arm
                // so a hostile guest cannot force a freeze or inject
                // an unvalidated host-side write target.
                let mut kernel_op_requests_pending:
                    Vec<crate::vmm::wire::KernelOpRequestPayload> = Vec::new();
                // CAPTURE requests received before `owned_accessor`
                // adoption are queued here instead of being serviced
                // immediately. Servicing pre-adoption produces a
                // partial-dump report (0 maps, vcpu_regs only — see
                // the `// Partial dump:` branch in
                // `freeze_and_dispatch`) which is useless to the test
                // author who asked for `Op::capture_snapshot("...")`.
                //
                // The queue is drained at the accessor-adoption site
                // by appending its contents back onto
                // `snapshot_requests_pending`, so the same iteration's
                // CAPTURE drain dispatches them through the normal
                // `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })` flow with the accessor
                // present.
                //
                // If the accessor never adopts (worker permanently
                // failed past its 60 s deadline), the queue is
                // dropped at coord exit and the guest's blocking
                // reader on `/dev/vport0p1` times out at the per-Op
                // 30 s deadline — same observable behaviour as a
                // late-boot rendezvous timeout. WATCH requests are
                // NOT deferred: WATCH only needs the symbol cache,
                // which is independent of `owned_accessor`.
                let mut capture_requests_deferred: Vec<SnapshotRequest> = Vec::new();
                // Cold-op requests received before `owned_accessor`
                // adoption are queued here instead of being serviced
                // immediately. Servicing pre-adoption hits the
                // `Some(owned) = owned_accessor` else-branch at the
                // ColdOp arm in `freeze_and_dispatch`, which returns a
                // synthetic `success = false` reply with
                // "owned_accessor not yet initialised; ColdOp dispatch
                // dropped". The guest's executor converts that into an
                // `anyhow::bail!` via `check_kernel_op_reply`, failing
                // every cold-op test whose first op fires before the
                // accessor-init worker publishes — a race that hits
                // boot-fast scenarios (no Backdrop, single Op step)
                // reliably because the guest reaches `apply_ops` while
                // KERN_ADDRS / accessor-init is still in flight on the
                // host. Mirrors `capture_requests_deferred` shape.
                //
                // The queue is drained at the accessor-adoption site
                // by appending its contents back onto
                // `kernel_op_requests_pending`, so the same iteration's
                // ColdOp drain dispatches them through the normal
                // `freeze_and_dispatch(FreezeMode::ColdOp(...))` flow
                // with the accessor present.
                //
                // If the accessor never adopts (worker permanently
                // failed past its 60 s deadline), the queue is
                // dropped at coord exit and the guest's blocking
                // reader on `/dev/vport0p1` times out at the per-Op
                // 30 s deadline — same observable behaviour as a
                // late-boot rendezvous timeout. Symmetric with
                // `capture_requests_deferred`.
                let mut kernel_op_requests_deferred:
                    Vec<crate::vmm::wire::KernelOpRequestPayload> = Vec::new();
                // Periodic-capture state. `periodic_boundaries_ns`
                // is the precomputed list of `Instant` deadlines
                // (encoded as nanos-since-`run_start`) at which the
                // run-loop fires `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })`. Lazily
                // built on the first iteration AFTER ALL:
                //   1. `KtstrVm::num_snapshots > 0` (periodic capture
                //      is requested), AND
                //   2. `workload_duration_for_coord` is `Some(d)`
                //      (the workload has a duration to slice), AND
                //   3. `periodic_prereqs_ready_ns != 0` (kaslr publish +
                //      map accessor + prog accessor all first held,
                //      stamped `.max(1)`), AND
                //   4. a non-zero anchor is resolvable — `scenario_start_ns`
                //      (the ScenarioStart frame stamped by the dispatch
                //      arm) OR, if that is still 0, the `watchdog_reset`
                //      fallback (the observed scheduler-attach time). While
                //      the anchor reads 0 the build DEFERS (a 0 anchor
                //      would invert the clamped window and 0-fire),
                //      retrying until one source latches.
                //
                // Boundaries divide the 10%–90% slice of the
                // capturable window into `N + 1` equal intervals,
                // producing `N` interior boundaries — `N == 1` lands
                // a single sample at the span midpoint; `N == 3`
                // lands at 0.3 / 0.5 / 0.7 of the span. On a warm
                // boot window_start == anchor, so the span equals the
                // workload duration `d` and the landings are
                // `0.3 d / 0.5 d / 0.7 d`; on a cold boot window_start
                // floats later and the fractions are of the shorter
                // clamped span. The 10% pre-boundary buffer and 10%
                // post-boundary buffer give the workload ramp-up /
                // ramp-down room without periodic samples landing on
                // transient state.
                //
                // `next_periodic_idx` tracks how many boundaries
                // have already fired. When the gate
                // (`freeze_coord_on_demand_in_flight`) is held by a
                // concurrent on-demand or watchpoint capture, the
                // periodic boundary is deferred (NOT skipped) until
                // a subsequent iteration finds the gate clear — the
                // 10% buffer is the slack budget for this wait.
                let mut periodic_boundaries_ns: Option<Vec<u64>> = None;
                // Anchor (`scenario_start_ns` or the derived fallback) the
                // boundaries were computed against; retained at loop scope
                // so each periodic-fire store can derive its capture's
                // workload-relative `boundary_offset_ms` = boundary_ns -
                // anchor_ns, which `build_phase_buckets_with_stimulus` maps
                // onto the guest step timeline (the run_start-relative fire
                // time is ~uniform across the deferred burst and useless
                // for per-phase attribution).
                let mut periodic_anchor_ns: u64 = 0;
                // run_start-relative ns at which ALL periodic-capture prereqs
                // (kaslr publish + map accessor + prog accessor) first held.
                // On a cold boot resolve_periodic_window floats window_start to
                // this moment so no boundary lands in the already-elapsed
                // pre-ready span (the cold-boot page-table walk + scx attach
                // transient); the separate end-clamp (window_end =
                // scenario_anchor + duration) is what keeps the window off
                // post-workload idle. 0 = not yet ready.
                let mut periodic_prereqs_ready_ns: u64 = 0;
                let mut next_periodic_idx: u32 = 0;
                // Periodic-capture clock, driven off GUEST progress. The
                // boundaries below are sliced over the workload in scenario-
                // relative time; comparing them against host wall-clock makes
                // every boundary fire while a host-oversubscribed (dilated)
                // guest is still in its FIRST phase, starving later phases. So
                // fire each boundary when the GUEST'S OWN scenario-elapsed clock
                // (published by the workload-progress heartbeat) crosses it —
                // the same fixed num_snapshots samples then spread across the
                // guest's real phases regardless of host dilation, WITHOUT any
                // extra captures/freezes. `periodic_prev_guest_elapsed_ns` is
                // the last-seen guest clock; `periodic_last_hb_wall_ns` is the
                // run-relative wall time it last ADVANCED, used to tell a wedge
                // (heartbeats STOP) from load (they merely SLOW): past the
                // staleness bound the host resumes on wall-clock so a degraded
                // guest's stuck state is still captured. Both 0 until the first
                // heartbeat, when the clock falls back to wall-clock exactly as
                // before (old guest / pre-heartbeat window).
                let mut periodic_prev_guest_elapsed_ns: u64 = 0;
                let mut periodic_last_hb_wall_ns: u64 = 0;
                // Wall-clock gap with no heartbeat advance that flips the clock
                // from guest-progress back to wall-clock (wedge backstop). Must
                // exceed the heartbeat cadence times the largest dilation a
                // healthy-but-slow guest reaches, so load never looks like a
                // wedge; 4 s covers the 100 ms cadence out past ~40x.
                const PERIODIC_HB_STALE_NS: u64 = 4_000_000_000;
                // Consecutive parked-vCPU rendezvous failures during
                // periodic capture. Reset to 0 on every successful
                // `freeze_and_dispatch(..)`. After 2 consecutive
                // timeouts the run-loop abandons the remaining
                // periodic boundaries and logs once — repeated
                // 30 s rendezvous waits on a wedged guest would
                // otherwise eat the entire wall-clock budget without
                // producing useful captures, and a single abandoned
                // boundary keeps periodic noise off a guest the
                // operator already knows is degraded.
                let mut periodic_consecutive_timeouts: u32 = 0;
                let mut periodic_abandoned: bool = false;
                const PERIODIC_TIMEOUT_ABANDON_THRESHOLD: u32 = 2;
                // First iteration always runs scan-tick work so
                // boot-race lazy resolution attempts fire
                // immediately rather than waiting up to 100 ms for
                // the timerfd's first edge. Subsequent iterations
                // gate scan-tick on the SCANNER token (or on a
                // POLL_TIMEOUT-driven wake) — the watchpoint event
                // itself never sets scan_tick, which is correct:
                // that trigger is a fast path that should not block
                // the next wake on heavy bss-PA / scan_ctx work.
                let mut scan_tick: bool;
                let mut first_iter = true;
                let mut bsp_done_final_pass = false;
                // Wall-clock cap on the post-BSP_DONE stay-alive
                // window. Bounds the wait for the
                // `owned_accessor`-gated drain of
                // [`capture_requests_deferred`] /
                // [`kernel_op_requests_deferred`]: if the accessor
                // hasn't adopted by the cap, surface the empty
                // reply path rather than blocking the test forever.
                // 30 s leaves headroom over the accessor-init
                // worker's own 60 s first-init deadline only in the
                // genuinely-stuck scenario (worker bailed
                // FAILED_PERMANENTLY); under healthy boots the
                // adoption fires within seconds and the stay-alive
                // exits immediately on the `owned_accessor.is_some()`
                // path. The cap exists exclusively to keep a
                // catastrophic prerequisite failure from converting
                // a test fault into an unbounded test run. The
                // `scenario::ops::await_accessor_ready` pre-stall gate
                // front-runs this cap for stall-driven dumps (the stall
                // fires only after adoption), so post-gate it backs only
                // the never-adopt cases: pre-adoption `Op::CaptureSnapshot`
                // requests and scenarios that don't await the gate.
                let mut bsp_done_final_pass_start: Option<Instant> = None;
                const DEFERRED_DRAIN_GRACE: Duration = Duration::from_secs(30);
                // Bounded grace for a wprof run's LATE trace ship. On an
                // error-class exit (Captured or Degraded) the coordinator
                // captures the dump, thaws, and — for a non-wprof run —
                // kills immediately. A wprof run ships its Perfetto `.pb`
                // over the bulk port in guest Phase 5 AFTER the crash
                // (`guest_comms::send_wprof_trace` -> `MsgType::WprofTrace`),
                // then Phase 6 reboots — and that reboot can hang on some
                // kernels, so waiting on `TOKEN_BSP_DONE` is not safe.
                // Killing at capture time would tear the VM down before the
                // ship. Instead the error-exit arms arm this deadline (they
                // do NOT kill); `freeze_state == Done` blocks re-freeze, the
                // `'coord:` guard keeps looping (kill unset => clause A of
                // the guard holds), the TOKEN_TX drain promotes the kill the
                // instant a crc-valid `WprofTrace` frame lands, and this
                // deadline is the backstop if the guest wedges before
                // shipping. `None` = no ship pending (non-wprof runs never
                // arm it).
                let mut wprof_ship_deadline: Option<Instant> = None;
                // Mirror of `bsp_done_final_pass` for the SCHED_EXIT
                // kill-promotion-without-bsp-done case: kill can be
                // promoted by sources other than a clean BSP_DONE —
                // e.g. the guest's
                // sched-exit-monitor fires `send_sched_exit` on
                // scheduler-binary pidfd POLLIN, the host bulk-port
                // dispatch flips `freeze_coord_kill` — but the BPF
                // tp_btf/sched_ext_exit handler that latches
                // `ktstr_err_exit_detected` in `.bss` fires from
                // inside the kernel before the userspace pidfd
                // POLLIN. Without a sched-exit final pass the coord
                // loop exits on the kill edge BEFORE the next BSS
                // read at the iteration top, so the sticky kernel-
                // side latch never gets observed and the failure
                // dump is silently dropped. Granting one more
                // iteration when kill is set (any source) closes
                // that race — the BSS read at the iteration top
                // catches the already-flipped latch and the
                // late-trigger path emits a normal dump. The BSS
                // write happens before the SCHED_EXIT msg arrives
                // on the host (kernel tp_btf precedes userspace
                // pidfd POLLIN), so one final-pass iteration is
                // sufficient.
                let mut sched_exit_final_pass = false;
                'coord: while !freeze_coord_kill.load(Ordering::Acquire)
                    || (freeze_coord_bsp_done.load(Ordering::Acquire)
                        && !bsp_done_final_pass)
                    || (!freeze_coord_bsp_done.load(Ordering::Acquire)
                        && !sched_exit_final_pass)
                    // wprof-ship grace stay-alive: while a grace deadline is
                    // armed, keep looping regardless of a set kill /
                    // bsp_done / sched_exit_final_pass, so a SCHED_EXIT- or
                    // watchdog-set kill cannot break the loop before the
                    // guest ships its Phase-5 wprof trace. edit-c (WprofTrace)
                    // or the `wprof_grace_should_kill` backstop clears the
                    // deadline and sets the kill; the NEXT top-of-loop check
                    // then finds this clause false and exits via the normal
                    // sched_exit_final_pass path. (The per-iteration backstop
                    // bounds this at `WPROF_SHIP_GRACE`; an EINTR-continue
                    // only defers it one wake, and no SIGRTMIN kicks occur
                    // once the vCPUs are thawed for the grace.)
                    || wprof_ship_deadline.is_some()
                {
                    if freeze_coord_bsp_done.load(Ordering::Acquire) {
                        if bsp_done_final_pass {
                            // Cover both deferral queues — ColdOp
                            // requests (`kernel_op_requests_deferred`)
                            // share the same `owned_accessor`
                            // adoption gate as capture requests, so
                            // a test that fires
                            // `Op::ReadKernelCold` before the
                            // accessor publishes would otherwise see
                            // no reply at all (workload ends, deferred
                            // queue never drains). The wall-clock
                            // cap at `DEFERRED_DRAIN_GRACE` keeps a
                            // genuinely-stuck prerequisite from
                            // hanging the test indefinitely.
                            let any_deferred = !capture_requests_deferred.is_empty()
                                || !kernel_op_requests_deferred.is_empty();
                            let grace_expired = bsp_done_final_pass_start
                                .map(|t| t.elapsed() >= DEFERRED_DRAIN_GRACE)
                                .unwrap_or(false);
                            if !any_deferred
                                || owned_accessor.is_some()
                                || grace_expired
                            {
                                if grace_expired && any_deferred {
                                    eprintln!(
                                        "freeze-coord: deferred drain grace ({:?}) expired with queue still non-empty: captures={} kernel_ops={}",
                                        DEFERRED_DRAIN_GRACE,
                                        capture_requests_deferred.len(),
                                        kernel_op_requests_deferred.len(),
                                    );
                                }
                                break 'coord;
                            }
                            eprintln!(
                                "freeze-coord: staying alive for deferred requests: captures={} kernel_ops={} accessor={}",
                                capture_requests_deferred.len(),
                                kernel_op_requests_deferred.len(),
                                owned_accessor.is_some()
                            );
                        } else {
                            bsp_done_final_pass_start = Some(Instant::now());
                        }
                        bsp_done_final_pass = true;
                    } else if freeze_coord_kill.load(Ordering::Acquire) {
                        // Kill is set but BSP_DONE is not — SCHED_EXIT
                        // or other external kill source. Grant one
                        // more iteration so the BSS-latch read at
                        // the iteration top observes any sticky
                        // kernel-side flip that landed in the same
                        // window the kill promotion did. Second
                        // iteration through this branch (sched_exit_
                        // final_pass already true) exits immediately
                        // — no deferred-capture-stay-alive semantics
                        // here because the scheduler has crashed,
                        // not exited cleanly.
                        if sched_exit_final_pass {
                            break 'coord;
                        }
                        sched_exit_final_pass = true;
                    }
                    // Unified event dispatch: epoll.wait on EVERY
                    // iteration. iter1 uses timeout=0 (non-blocking)
                    // so scan_tick fires immediately; subsequent
                    // iterations block up to POLL_TIMEOUT_MS. This
                    // ensures TOKEN_TX (KERN_ADDRS, SYS_RDY) is
                    // dispatched event-driven on every iteration —
                    // no manual drain calls needed.
                    let poll_ms = if first_iter { 0 } else { POLL_TIMEOUT_MS };
                    if first_iter {
                        scan_tick = true;
                        first_iter = false;
                    } else {
                        scan_tick = false;
                    }
                    if bsp_done_final_pass || sched_exit_final_pass {
                        scan_tick = true;
                    }
                    {
                        let event_count = match epoll.wait(poll_ms, &mut events_buf) {
                            Ok(n) => n,
                            // EINTR: retry the wait. This `continue` skips
                            // this iteration's wprof-grace backstop
                            // (`wprof_grace_should_kill`) and edit-c, but the
                            // guard's `wprof_ship_deadline.is_some()`
                            // stay-alive keeps the loop live so the next
                            // successful epoll wake re-evaluates both. No
                            // SIGRTMIN kicks occur once the vCPUs are thawed
                            // for the grace, so an EINTR storm that could
                            // indefinitely defer the backstop is not expected.
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(e) => {
                                tracing::error!(
                                    error = %e,
                                    "freeze-coord: epoll_wait failed; exiting coordinator"
                                );
                                break 'coord;
                            }
                        };
                        // Drain every fd that fired. Tokens map
                        // 1:1 to source fds; KILL / BSP_DONE both
                        // exit the loop, the others either set
                        // scan_tick (SCANNER) or surface state via
                        // the existing latch reads later in the
                        // body (WATCHPOINT).
                        for ev in &events_buf[..event_count] {
                            match ev.data() {
                                TOKEN_KILL | TOKEN_BSP_DONE => {
                                    let _ = freeze_coord_kill_evt.read();
                                    let _ = freeze_coord_bsp_done_evt.read();
                                }
                                TOKEN_SCANNER => {
                                    // Drain the timerfd's expiry
                                    // counter — re-arming is
                                    // automatic for periodic
                                    // timers, but the counter
                                    // accumulates and would re-
                                    // wake on the next epoll_wait
                                    // if not drained.
                                    let _ = scanner_tfd.wait();
                                    scan_tick = true;
                                }
                                TOKEN_WATCHPOINT => {
                                    // Drain the eventfd counter so
                                    // a subsequent epoll_wait
                                    // doesn't immediately re-fire
                                    // on the same edge. The
                                    // watchpoint.hit AtomicBool is
                                    // the source of truth — its
                                    // state survives the eventfd
                                    // drain and the late-trigger
                                    // detection later in the loop
                                    // re-loads it with Acquire.
                                    let _ = freeze_coord_hit_evt.read();
                                }
                                TOKEN_ACCESSOR_READY => {
                                    let _ = accessor_ready_evt.read();
                                    scan_tick = true;
                                }
                                TOKEN_TX => {
                                    // Drain the tx_evt counter so
                                    // a subsequent epoll_wait
                                    // doesn't immediately re-fire
                                    // on the same edge. The drain
                                    // below uses the device's TX
                                    // buffer (port1_tx_buf) as the
                                    // source of truth — bytes the
                                    // device accumulated since the
                                    // last wake are returned by
                                    // `drain_bulk` and threaded
                                    // through `bulk_assembler`. A
                                    // counter overflow under
                                    // EFD_NONBLOCK is benign
                                    // because the buffer state is
                                    // authoritative.
                                    //
                                    // Critical-section discipline:
                                    // `tx_evt.read()` is a syscall
                                    // and `bulk_assembler.feed()`
                                    // does TLV parsing (memcpy +
                                    // CRC + per-frame cap check).
                                    // Both are kept STRICTLY
                                    // outside the device mutex so
                                    // the vCPU thread emitting
                                    // bytes via virtio-console TX
                                    // never blocks behind the
                                    // coord. The explicit
                                    // `let bytes = { ... };`
                                    // block bounds the lock to the
                                    // single `drain_bulk` call —
                                    // a future refactor that
                                    // moves work into the block
                                    // is loud about the regression.
                                    let _ = freeze_coord_tx_evt.read();
                                    let bytes = {
                                        let mut g =
                                            freeze_coord_virtio_con.lock();
                                        g.drain_bulk()
                                    };
                                    let drained = bulk_assembler.feed(&bytes);
                                    // Per-frame typed dispatch.
                                    // Exhaustive `match
                                    // MsgType::from_wire(...)` so a
                                    // future MsgType variant addition
                                    // is a compile error here — the
                                    // arms call out exactly which
                                    // frames have coordinator-side
                                    // side effects (SchedExit / SysRdy
                                    // / SnapshotRequest), and every
                                    // other variant falls through to a
                                    // single "test-verdict-bearing"
                                    // arm whose only action is to
                                    // accumulate the entry into the
                                    // shared bucket. The prior if-
                                    // ladder of `msg.msg_type ==
                                    // MSG_TYPE_*` checks let a new
                                    // variant slip past the host
                                    // without an explicit decision.
                                    //
                                    // Every CRC-bearing arm gates on
                                    // `msg.crc_ok` so a torn frame
                                    // cannot promote into kill_evt /
                                    // sys_rdy_evt or trigger a
                                    // capture — same hostile-guest
                                    // discipline as the prior code.
                                    let mut bucket: Vec<crate::vmm::wire::ShmEntry> =
                                        Vec::new();
                                    let mut sinks = BulkDispatchSinks {
                                        kill: &freeze_coord_kill,
                                        kill_evt: &freeze_coord_kill_evt,
                                        run_is_wprof: freeze_coord_wprof,
                                        sys_rdy_evt: &mut freeze_coord_sys_rdy_evt,
                                        snapshot_requests_pending:
                                            &mut snapshot_requests_pending,
                                        kernel_op_requests_pending:
                                            &mut kernel_op_requests_pending,
                                        kern_phys_base: &kern_phys_base,
                                        kern_phys_base_evt: &kern_phys_base_evt,
                                        kern_virt_kaslr: &kern_virt_kaslr,
                                        kern_virt_kaslr_evt: &kern_virt_kaslr_evt,
                                        kernel_text_link_kva,
                                        kern_addrs_frames: &kern_addrs_frames_for_coord,
                                        kern_addrs_crc_bad: &kern_addrs_crc_bad_for_coord,
                                        unknown_type_frames: &unknown_type_frames_for_coord,
                                        watchdog_reset: workload_duration_for_coord.map(|d| {
                                            (
                                                watchdog_reset_for_coord.as_ref(),
                                                d,
                                                run_start,
                                                watchdog_reset_tag_for_coord.as_ref(),
                                            )
                                        }),
                                        watchdog_pause_ns: watchdog_pause_for_coord.as_ref(),
                                        scenario_start_ns: scenario_start_ns_for_coord.as_ref(),
                                        sched_swap_notify:
                                            sched_swap_notify_for_coord.as_ref(),
                                        scenario_pause_cumulative_ns:
                                            scenario_pause_cumulative_for_coord.as_ref(),
                                        run_start,
                                        current_step: &freeze_coord_current_step,
                                        periodic_guest_elapsed_ns:
                                            &freeze_coord_periodic_guest_elapsed,
                                        progress_ledger: Some(
                                            progress_ledger_for_coord.as_ref(),
                                        ),
                                        contention_recorder: Some(
                                            contention_recorder_for_coord.as_ref(),
                                        ),
                                        expected_kernel_build_id:
                                            host_vmlinux_build_id_for_coord.as_deref(),
                                    };
                                    for msg in &drained.messages {
                                        if let Some(entry) =
                                            dispatch_bulk_message(msg, &mut sinks)
                                        {
                                            bucket.push(entry);
                                        }
                                    }
                                    // wprof-ship grace arm on SCHED_EXIT: on a
                                    // wprof run a drained crc-valid SCHED_EXIT
                                    // (the guest self-crash monitor's frame,
                                    // which `dispatch_bulk_message` no longer
                                    // promotes to a kill) arms the grace if it
                                    // is not already armed by the error-exit
                                    // dump — holding the VM open for the
                                    // guest's late Phase-5 wprof ship even
                                    // when the exit_kind watchpoint dump has
                                    // not (yet) fired (e.g. watchpoint
                                    // unavailable). edit-c below then kills on
                                    // the WprofTrace, or the WPROF_SHIP_GRACE
                                    // backstop does.
                                    if freeze_coord_wprof
                                        && wprof_ship_deadline.is_none()
                                        && drained.messages.iter().any(is_sched_exit_frame)
                                    {
                                        eprintln!(
                                            "freeze-coord: SCHED_EXIT drained on wprof run; \
                                             arming wprof ship grace ({WPROF_SHIP_GRACE:?})"
                                        );
                                        wprof_ship_deadline =
                                            Some(arm_wprof_grace(run_start, &watchdog_reset_for_coord, &watchdog_reset_tag_for_coord, WPROF_SHIP_GRACE));
                                    }
                                    // wprof-ship grace: if an error-exit arm
                                    // (or the SCHED_EXIT arm above) armed
                                    // `wprof_ship_deadline`, promote the
                                    // kill the instant the guest's Phase-5
                                    // wprof trace lands — a crc-valid,
                                    // non-empty `MsgType::WprofTrace` frame —
                                    // rather than waiting out the deadline or
                                    // the guest reboot (which can hang). The
                                    // frame is not coordinator-internal, so it
                                    // also flowed into `bucket` above for
                                    // `collect_results`; this block is only
                                    // the teardown trigger. Clearing the
                                    // deadline makes the per-iteration
                                    // backstop below a no-op afterward.
                                    //
                                    // EXCEPTION — auto-repro (dual_snapshot)
                                    // runs with an active probe stack: the
                                    // guest ships its probe payload (Phase 6b,
                                    // `PROBE_OUTPUT_END` over stdout,
                                    // init.rs `finalize_probe_after_unwind`)
                                    // AFTER the Phase-5 wprof trace. Killing on
                                    // the WprofTrace frame here would preempt
                                    // Phase 6b, so the `=== AUTO-PROBE ... ===`
                                    // report never ships. On those runs, hold
                                    // for the probe payload: skip the kill on
                                    // the trace and instead promote it when the
                                    // `PROBE_OUTPUT_END` stdout frame drains
                                    // (arm below). The `WPROF_SHIP_GRACE`
                                    // backstop still bounds teardown if the
                                    // guest wedges before shipping the payload
                                    // (e.g. a stall-exit repro that skipped
                                    // probe attachment ships no payload) —
                                    // keeping 7fe5cc7e's anti-hang property.
                                    if freeze_coord_dual_snapshot {
                                        if wprof_ship_deadline.is_some()
                                            && drained
                                                .messages
                                                .iter()
                                                .any(is_probe_output_end_frame)
                                        {
                                            eprintln!(
                                                "freeze-coord: probe payload shipped \
                                                 (PROBE_OUTPUT_END); kill triggered after ship"
                                            );
                                            trigger_freeze_coord_kill(
                                                &freeze_coord_kill,
                                                &freeze_coord_kill_evt,
                                            );
                                            wprof_ship_deadline = None;
                                        }
                                    } else if wprof_ship_deadline.is_some()
                                        && drained.messages.iter().any(is_wprof_ship_frame)
                                    {
                                        eprintln!(
                                            "freeze-coord: wprof trace received; \
                                             kill triggered after ship"
                                        );
                                        trigger_freeze_coord_kill(
                                            &freeze_coord_kill,
                                            &freeze_coord_kill_evt,
                                        );
                                        wprof_ship_deadline = None;
                                    }
                                    // Append the verdict-bearing entries
                                    // to the shared bucket so
                                    // `collect_results` can merge them
                                    // into the final `BulkDrainResult`.
                                    // Coordinator-internal control
                                    // frames are filtered inside
                                    // `dispatch_bulk_message` (the
                                    // SysRdy / SnapshotRequest arms
                                    // return None) — keying on
                                    // [`crate::vmm::wire::MsgType::is_coordinator_internal`]
                                    // keeps the filter set in lockstep
                                    // with `collect_results`'s post-run
                                    // drain. Without this stash, every
                                    // TLV frame the guest published
                                    // mid-run is silently dropped —
                                    // only late-arriving bytes that
                                    // landed in `port1_tx_buf` after
                                    // the coord stopped polling reach
                                    // the verdict.
                                    if !bucket.is_empty() {
                                        let mut buf = freeze_coord_bulk_messages_for_closure
                                            .lock_unpoisoned();
                                        buf.extend(bucket);
                                    }
                                }
                                _ => {}
                            }
                        }
                        // No break on kill/bsp_done here — the
                        // iteration body must run so the
                        // late-trigger err_triggered check and
                        // freeze_and_dispatch can fire. The while
                        // condition + inner bsp_done check handle
                        // loop exit after the body completes.
                    }
                    // wprof-ship grace backstop. Reached every iteration
                    // after the epoll drain, before any conditional
                    // `continue` in the loop body. The TOKEN_TX drain above
                    // clears the deadline (and promotes the kill) the moment
                    // the WprofTrace lands; this branch fires only if the
                    // guest wedges before shipping, bounding the wait at
                    // `WPROF_SHIP_GRACE`. Once the kill is set, the `'coord:`
                    // guard runs one more pass (the
                    // `!bsp_done && !sched_exit_final_pass` clause), sets
                    // `sched_exit_final_pass`, then breaks — bounded
                    // teardown, no hang.
                    if wprof_grace_should_kill(wprof_ship_deadline, Instant::now()) {
                        eprintln!(
                            "freeze-coord: wprof-ship grace ({WPROF_SHIP_GRACE:?}) expired \
                             without a trace; killing"
                        );
                        trigger_freeze_coord_kill(
                            &freeze_coord_kill,
                            &freeze_coord_kill_evt,
                        );
                        wprof_ship_deadline = None;
                    }
                    // Synchronous periodic-capture + watchpoint
                    // invalidation on an explicit guest swap-notify
                    // (Op::Detach / Restart / ReplaceScheduler). The
                    // guest's `kill_current_scheduler` sends
                    // `MSG_TYPE_SCHED_SWAP_NOTIFY` once `*scx_root` is NULL
                    // (after `wait_for_scx_disabled`;
                    // `RCU_INIT_POINTER(scx_root, NULL)` precedes
                    // `scx_set_enable_state(SCX_DISABLED)` in
                    // kernel/sched/ext.c scx_root_disable), so the prior
                    // `scx_sched` object is unlinked (`*scx_root` NULLed)
                    // and its slab is subject to RCU-grace-period reuse —
                    // BOTH the owned accessor AND the slot-0 DR0 watchpoint
                    // (armed on that object's `exit_kind` KVA) are stale.
                    // Perform the FULL watchpoint-Detached teardown
                    // synchronously, not just the accessor half:
                    //   * disarm DR0 (`WatchpointArm::disarm`) + reset
                    //     `last_sched_kva = 0`, so a stale DR0 cannot fire
                    //     into the recycled slab (a `>= SCX_EXIT_ERROR`
                    //     read would latch a phantom failure dump on a
                    //     deliberate swap), and the next
                    //     owned-accessor-present scan tick republishes as a
                    //     fresh `0 -> B` attach;
                    //   * clear `cached_exit_kind_pa`, so the scan-tick
                    //     err-trigger probe and a rendezvous-timeout
                    //     Degraded dump never read `exit_kind` from the
                    //     recycled page;
                    //   * drop the owned accessor + pulse
                    //     `accessor_reinit_evt` so the accessor-init worker
                    //     rebuilds against the next scheduler NOW (its
                    //     `accessor_ready_evt` -> TOKEN_ACCESSOR_READY ->
                    //     adopt path re-publishes it).
                    // Doing only the accessor half is UNSOUND: this
                    // teardown sets `owned_accessor = None`, which gates
                    // OFF the watchpoint tick (the only other code that
                    // disarms DR0 / clears `cached_exit_kind_pa`), so the
                    // post-notify state MUST already be the Detached state.
                    // Net effect: collapses the up-to-one-SCAN_INTERVAL
                    // watchpoint-poll detection window to ~0. Read-and-
                    // clear so it fires once per notify; ungated by
                    // `scan_tick` (skipping the scan cadence is the point)
                    // and idempotent against the poll's own later Detached
                    // arm. A notify that lands after the poll already
                    // detached (owned_accessor already None) re-runs this
                    // body harmlessly; the only cost is an extra
                    // accessor_reinit_evt pulse — the worker coalesces
                    // pulses that arrive while it is parked into one
                    // redundant rebuild, while a pulse landing mid-rebuild
                    // costs one extra idempotent (live-state) rebuild.
                    if sched_swap_notify_for_coord.swap(false, Ordering::Relaxed) {
                        freeze_coord_watchpoint.disarm();
                        last_sched_kva = 0;
                        cached_exit_kind_pa = None;
                        owned_accessor = None;
                        owned_prog_accessor = None;
                        *accessors_slot.lock().unwrap_or_else(|e| e.into_inner()) = None;
                        let _ = accessor_reinit_evt.write(1);
                        tracing::info!(
                            target: "ktstr::failure_dump",
                            "freeze-coord: sched-swap notify — synchronous \
                             periodic-capture + watchpoint invalidation \
                             (full Detached teardown; skipping the \
                             watchpoint-poll detection window)"
                        );
                    }
                    // Adopt the worker-published accessor pair as
                    // soon as it lands. The worker stores `Some(pair)`
                    // into the `accessors_slot` mutex in one shot, so a
                    // single `lock().take()` returns either both halves
                    // or neither — no partial-Some shape to handle. The
                    // worker logs
                    // its own warn/eprintln on a permanent failure
                    // (60 s deadline exceeded), so the coordinator
                    // doesn't track separate retry counters here:
                    // a None pair after the worker has exited just
                    // means the dump path is unavailable for this
                    // run, which the existing call-site `is_some()`
                    // gates already handle gracefully.
                    if scan_tick && owned_accessor.is_none()
                        && let Some((map, prog)) = accessors_slot
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .take()
                    {
                        owned_accessor = Some(map);
                        if let Some(prog) = prog {
                            owned_prog_accessor = Some(prog);
                        }
                        // Signal the guest that the accessor is adopted
                        // (`owned_accessor` is now `Some`), so a scenario
                        // gated on `await_accessor_ready` resumes and triggers
                        // its stall only now — guaranteeing the freeze dump
                        // renders real BPF map values, not placeholders.
                        // Pushed at the adoption point (not the worker's
                        // SUCCEEDED publish) so the dump path is fully armed
                        // before the guest can stall.
                        super::host_comms::request_accessor_ready(&freeze_coord_virtio_con);
                        // (virt-KASLR derivation lives in
                        // `coord_kaslr_offset` above — sourced from
                        // the shared `kern_virt_kaslr` Arc, populated
                        // by BSP MSR_LSTAR + guest-channel KERN_ADDRS.
                        // The prior in-line derivation here computed
                        // `phys_base - real_phys_base` where both
                        // reads target the same kernel global, which
                        // reduced to a structural zero. Deleted in
                        // favour of the cross-thread Arc-based path.)
                        // Drain CAPTURE requests deferred during the
                        // pre-adoption window. Append onto
                        // `snapshot_requests_pending` so the existing
                        // CAPTURE drain (further down this iteration
                        // body) dispatches them through the normal
                        // `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })` flow with the
                        // accessor present — no flow duplication.
                        if !capture_requests_deferred.is_empty() {
                            let n = capture_requests_deferred.len();
                            tracing::info!(
                                deferred_count = n,
                                "freeze-coord: draining deferred CAPTURE \
                                 requests after owned_accessor adoption"
                            );
                            snapshot_requests_pending.append(
                                &mut capture_requests_deferred,
                            );
                        }
                        // Symmetric drain of cold-op requests queued
                        // before adoption. Appending onto
                        // `kernel_op_requests_pending` routes them
                        // through this iteration's existing cold-op
                        // drain (further down the loop body), which
                        // now has `owned_accessor = Some(...)` so
                        // `freeze_and_dispatch(FreezeMode::ColdOp)` can
                        // service the request instead of returning the
                        // "owned_accessor not yet initialised" error.
                        if !kernel_op_requests_deferred.is_empty() {
                            let n = kernel_op_requests_deferred.len();
                            tracing::info!(
                                deferred_count = n,
                                "freeze-coord: draining deferred ColdOp \
                                 kernel-op requests after owned_accessor adoption"
                            );
                            kernel_op_requests_pending.append(
                                &mut kernel_op_requests_deferred,
                            );
                        }
                    }
                    // Resolve the per-CPU offset array once the prog
                    // accessor lands. Reads `__per_cpu_offset` from
                    // the kernel's static symbol table and uses it
                    // to read each CPU's offset slot. Cached for the
                    // rest of the run — the array is fixed at boot
                    // (per-CPU areas are allocated at kernel init,
                    // see `setup_per_cpu_areas`) and the freeze
                    // coordinator never sees a CPU hot-plug event,
                    // so a single read is enough.
                    //
                    // The `__per_cpu_offset` KVA is sourced from the
                    // already-cached `dump_cpu_time_symbols` —
                    // re-parsing vmlinux every scan tick (~100 ms)
                    // while waiting for the per-CPU areas to come up
                    // would re-read 50 MB+ of ELF and rebuild the
                    // symbol table on every iteration. The KVA is
                    // fixed at kernel link time so a single resolution
                    // suffices for the rest of the run; if
                    // `dump_cpu_time_symbols` is None (vmlinux
                    // unparseable at coord start) or its
                    // `per_cpu_offset` is 0 (symbol stripped),
                    // `try_init_prog_per_cpu_offsets` returns None
                    // and the cache stays unset — same behaviour as
                    // the prior in-helper parse path.
                    if scan_tick
                        && prog_per_cpu_offsets.is_none()
                        && let Some(mem) = freeze_coord_mem.as_deref()
                    {
                        let per_cpu_offset_kva = dump_cpu_time_symbols
                            .as_ref()
                            .map(|s| s.per_cpu_offset)
                            .unwrap_or(0);
                        if per_cpu_offset_kva == 0 {
                            // Permanent failure: the symbol is absent
                            // from `dump_cpu_time_symbols` (vmlinux
                            // unparseable at coord start, or
                            // `__per_cpu_offset` stripped from the
                            // image). Warn immediately on the first
                            // observation — no amount of retrying
                            // will materialise a missing symbol — and
                            // latch via `per_cpu_offsets_kva_warned`
                            // so the warn fires at most once per VM
                            // run. The `prog_per_cpu_offsets` cache
                            // stays None and downstream
                            // prog_runtime_stats capture is
                            // permanently degraded for this run.
                            if !per_cpu_offsets_kva_warned {
                                tracing::warn!(
                                    "freeze-coord: __per_cpu_offset symbol absent from \
                                     dump_cpu_time_symbols (vmlinux unparseable at coord \
                                     start, or symbol stripped) — prog_runtime_stats \
                                     capture is permanently degraded for this run; \
                                     will not retry"
                                );
                                per_cpu_offsets_kva_warned = true;
                            }
                        } else if let Some(owned) = owned_accessor.as_ref() {
                            // Gate the init on `owned_accessor` being
                            // adopted so `phys_base` is the resolved
                            // KASLR physical displacement, not the
                            // bootstrap 0 default. Under a KASLR-on
                            // guest, `phys_base == 0` makes
                            // `text_kva_to_pa_with_base` resolve
                            // `__per_cpu_offset` to a wrong PA — the
                            // resulting offsets read are garbage from
                            // an unrelated guest page. The cache then
                            // holds that garbage permanently (the
                            // outer `prog_per_cpu_offsets.is_none()`
                            // gate never retries once Some is set),
                            // and every downstream per-CPU walker
                            // (cpu_time, prog_runtime_stats, scx)
                            // resolves to wrong KVAs and reads zero
                            // values from random pages. Sibling
                            // `compute_rq_pas` at L4827 uses
                            // `kernel.text_kva_to_pa` which threads
                            // the live `phys_base` from the accessor
                            // — same source-of-truth.
                            let phys_base = owned.guest_kernel().phys_base();
                            prog_per_cpu_offsets = try_init_prog_per_cpu_offsets(
                                mem,
                                per_cpu_offset_kva,
                                freeze_coord_tcr_el1.as_ref(),
                                phys_base,
                                freeze_coord_num_cpus,
                            );
                            if prog_per_cpu_offsets.is_none() {
                                // Transient boot-progress condition:
                                // the symbol is present (kva != 0)
                                // but at least one CPU's offset slot
                                // is still zero. The guest's
                                // `setup_per_cpu_areas` populates
                                // every slot before SMP bringup, so
                                // a non-zero retry count after
                                // `LAZY_ACCESSOR_WARN_AFTER_ITERS`
                                // iterations indicates the guest
                                // genuinely failed to bring up its
                                // per-CPU areas (or
                                // `freeze_coord_num_cpus` exceeds
                                // the configured `nr_cpu_ids` so
                                // slots beyond the live count
                                // legitimately read 0).
                                per_cpu_offsets_retries += 1;
                                if !per_cpu_offsets_warned
                                    && per_cpu_offsets_retries
                                        >= LAZY_ACCESSOR_WARN_AFTER_ITERS
                                {
                                    tracing::warn!(
                                        retries = per_cpu_offsets_retries,
                                        num_cpus = freeze_coord_num_cpus,
                                        "freeze-coord: __per_cpu_offset array still has \
                                         zero slots after retries — most commonly a \
                                         still-booting guest (per-CPU areas not yet \
                                         allocated); a permanent failure (num_cpus \
                                         exceeds nr_cpu_ids, partial SMP bringup) \
                                         leaves prog_runtime_stats degraded. Will \
                                         continue retrying."
                                    );
                                    per_cpu_offsets_warned = true;
                                }
                            }
                        }
                    }
                    // Try to discover the probe .bss map and cache the
                    // PA of ktstr_err_exit_detected. Match by suffix
                    // "probe_bp.bss" rather than ".bss" so we don't
                    // race a scheduler-under-test's own .bss map when
                    // multiple BPF programs are loaded — libbpf names
                    // BPF program .bss maps as "<obj_short_name>.bss",
                    // and the probe object's name is "probe_bp" (per
                    // build.rs probe-skel generation, see the
                    // generated probe_skel.rs match arm
                    // `"probe_bp.bss" => bss = Some(map)`).
                    //
                    // Resolve the byte offset of
                    // `ktstr_err_exit_detected` within the probe's
                    // `.bss` section via BTF Datasec rather than
                    // hardcoding 0. The probe BPF program ships its
                    // own split BTF; its Datasec for `.bss` carries
                    // a VarSecinfo per writable global with the
                    // exact byte offset the BPF JIT places it at. A
                    // hardcoded 0 worked while the field was the
                    // sole writable global in `probe.bpf.c`, but a
                    // future addition that reorders globals (or that
                    // adds another writable global before this one)
                    // would silently shift the offset and break the
                    // freeze trigger. The BTF lookup keeps the
                    // detection robust across declaration changes.
                    //
                    // Falls back to offset 0 when the program BTF
                    // can't be loaded yet (guest still booting) or
                    // the Datasec walk fails — same recovery
                    // behaviour as the previous always-zero path.
                    //
                    // Invalidation pass first: a previously-cached
                    // PA is only as valid as the underlying map. If
                    // the probe BPF program unloads (test teardown,
                    // userspace explicit unload, parent process
                    // panicking before Drop) the kernel frees the
                    // bpf_array vmalloc page; any subsequent
                    // `read_u32(cached_bss_pa, 0)` reads whatever
                    // the page allocator hands out next — typically
                    // non-zero for slab pages reused by an unrelated
                    // subsystem. The result latches a phantom
                    // `err_triggered` and synthesizes a bogus
                    // failure dump on a healthy run. Re-walk
                    // `map_idr` and require the same-named map's
                    // `value_kva` to match the one we resolved
                    // against; on mismatch (map gone OR rebound to
                    // a fresh slab) clear the PA + companion
                    // value_kva cache so the discovery block below
                    // re-resolves from scratch. The walk uses a
                    // fresh `as_accessor()` instance — its
                    // `maps_cache` re-fills from a current map_idr
                    // traversal, so a stale entry from a prior dump
                    // cannot keep an unloaded map visible.
                    if scan_tick
                        && cached_bss_pa.is_some()
                        && let Some(owned) = owned_accessor.as_ref()
                    {
                        let accessor = owned.as_accessor();
                        let still_valid = match accessor.find_array_map("probe_bp.bss") {
                            Some(m) => m.value_kva == cached_bss_value_kva,
                            None => false,
                        };
                        if !still_valid {
                            tracing::warn!(
                                stale_value_kva = cached_bss_value_kva
                                    .map(|k| format!("{k:#x}"))
                                    .unwrap_or_else(|| "None".to_string()),
                                "freeze-coord: probe_bp.bss map gone or \
                                 rebound — invalidating cached_bss_pa to \
                                 prevent reads of a freed vmalloc page \
                                 (probe unload mid-run)"
                            );
                            cached_bss_pa = None;
                            cached_bss_value_kva = None;
                            // bss_field_offset is BTF-derived from
                            // probe.bpf.c globals; the layout
                            // cannot change across an unload+reload
                            // of the same probe object so the
                            // offset cache stays valid. Re-resolving
                            // it would re-pay the BTF parse for no
                            // semantic gain.
                        }
                    }
                    if scan_tick
                        && cached_bss_pa.is_none()
                        && let Some(owned) = owned_accessor.as_ref()
                        && let Some(ref mem) = freeze_coord_mem
                    {
                        let accessor = owned.as_accessor();
                        // Single map_idr walk per discovery attempt.
                        // value_kva is Some for ARRAY maps (the .bss
                        // map is a single-key ARRAY whose flex array
                        // holds the section's bytes); translate it
                        // (vmalloc-backed) to PA via the existing
                        // GuestMem page-walk and cache the result so
                        // subsequent polls are pure DRAM reads.
                        if let Some(map) = accessor.find_array_map("probe_bp.bss")
                            && let Some(value_kva) = map.value_kva
                        {
                            // Bind kernel once and reuse — pre-fix
                            // owned.guest_kernel() ran three times here
                            // and once again at the BTF Datasec walk
                            // below. The accessor is cheap but the
                            // repetition was noisy at the freeze hot
                            // path's read site.
                            let kernel = owned.guest_kernel();
                            let walk = kernel.walk_context();
                            // BTF-driven offset: load the probe's
                            // program BTF and walk its `.bss`
                            // Datasec for the named global. The
                            // result is cached in `cached_bss_offset`
                            // — only the first successful resolution
                            // pays the BTF parse cost. A None here
                            // before the cache is populated means
                            // either the program BTF isn't loaded
                            // yet (still-booting guest, retry
                            // silently) or the BTF walk is broken
                            // (warn once, fall back to offset 0).
                            if cached_bss_offset.is_none()
                                && map.btf_kva != 0
                                && let Some(ref base) = dump_btf
                            {
                                match load_probe_bss_offset(
                                    kernel,
                                    map.btf_kva,
                                    base,
                                    accessor.offsets(),
                                ) {
                                    Some(off) => {
                                        cached_bss_offset = Some(off);
                                    }
                                    None => {
                                        // map.btf_kva is non-zero
                                        // and dump_btf is loaded,
                                        // so the probe IS loaded
                                        // — a None now means the
                                        // BTF parse / Datasec
                                        // walk failed. Fall back
                                        // to 0 and stop retrying.
                                        if !bss_offset_warn_logged {
                                            tracing::warn!(
                                                "freeze-coord: BTF Datasec resolution \
                                                     failed, falling back to offset 0"
                                            );
                                            bss_offset_warn_logged = true;
                                        }
                                        cached_bss_offset = Some(0);
                                    }
                                }
                                // else: probe not loaded yet
                                // (map.btf_kva == 0 or dump_btf
                                // missing). Leave cached_bss_offset
                                // None so the next iteration retries
                                // without the warn fallback.
                            }
                            let bss_field_offset = cached_bss_offset.unwrap_or(0);
                            // Bound the BTF-derived offset against
                            // the map's declared `value_size`. The
                            // probe's BTF Datasec walk parses
                            // guest-supplied bytes — a corrupted
                            // (or hostile) BTF can return a u32
                            // offset that extends past the ARRAY's
                            // flex-array storage, so the
                            // `wrapping_add(bss_field_offset)`
                            // below would wrap into an unrelated
                            // guest page. Reading the resulting PA
                            // latches a phantom `err_triggered`
                            // and synthesizes a bogus failure
                            // dump. Reject any offset whose 4-byte
                            // read would walk past the map's
                            // value bytes; treat the failure
                            // exactly like a broken BTF walk —
                            // warn once via the existing latch and
                            // fall back to offset 0 for this and
                            // every subsequent iteration so the
                            // detection survives in degraded form
                            // instead of going silent. Saturating
                            // subtract guards `value_size < 4`
                            // (the map could not legitimately
                            // hold a u32 in that case, so
                            // `bss_field_offset > 0` rejects every
                            // non-zero offset, matching the
                            // "value_size too small" intent
                            // without a separate branch).
                            let max_offset = map.value_size.saturating_sub(4);
                            let bss_field_offset = if bss_field_offset > max_offset {
                                if !bss_offset_warn_logged {
                                    tracing::warn!(
                                        bss_field_offset,
                                        value_size = map.value_size,
                                        "freeze-coord: BTF-resolved bss field \
                                         offset exceeds value_size - 4 — \
                                         refusing to cache PA that would \
                                         read past the .bss flex array; \
                                         falling back to offset 0"
                                    );
                                    bss_offset_warn_logged = true;
                                }
                                cached_bss_offset = Some(0);
                                0
                            } else {
                                bss_field_offset
                            };
                            if let Some(translated) = crate::monitor::idr::translate_any_kva(
                                mem,
                                walk.cr3_pa,
                                walk.page_offset,
                                value_kva,
                                walk.l5,
                                walk.tcr_el1,
                            ) {
                                cached_bss_pa =
                                    Some(translated.wrapping_add(bss_field_offset as u64));
                                cached_bss_value_kva = Some(value_kva);
                            }
                        }
                    }
                    // Resolve the watchpoint target KVA
                    // (`*scx_root + exit_kind_offset`) and (re-)
                    // publish it whenever `*scx_root` changes. Runs
                    // every scan tick — the `last_sched_kva == new`
                    // fast path keeps the steady-state cost a single
                    // u64 read.
                    //
                    // Resolution requires:
                    //   - dump_cpu_time_symbols (KernelSymbols) for
                    //     `scx_root` symbol KVA — present whenever
                    //     vmlinux parsed at coord-start;
                    //   - dump_scx_walker_offsets.sched.exit_kind for
                    //     the field offset within `struct scx_sched`
                    //     — present whenever BTF carries the type;
                    //   - owned_accessor's GuestKernel for cr3_pa /
                    //     page_offset / l5 — needed for the same
                    //     direct-mapping translation `cached_bss_pa`
                    //     uses.
                    //
                    // The BPF .bss fallback below continues to update
                    // `cached_bss_pa`; both signals can fire and the
                    // late-trigger arm (a few iterations down the
                    // loop) treats either as ground truth. The
                    // watchpoint's advantages are synchronous
                    // delivery (no 100 ms polling window) AND
                    // independence from the probe BPF program loading
                    // correctly.
                    if scan_tick
                        && owned_accessor.is_some()
                        && let Some(ref syms) = dump_cpu_time_symbols
                        && let Some(scx_root_kva) = syms.scx_root
                        && let Some(ref scx_offsets) = dump_scx_walker_offsets
                        && let Some(ref sched_offs) = scx_offsets.sched
                        && let Some(ref mem) = freeze_coord_mem
                    {
                        // scx_root is a kernel-text-mapped pointer.
                        // The owned_accessor's GuestKernel carries the
                        // VA-bits-aware kernel image base resolved from
                        // TCR_EL1 (mirrors `read_scx_sched_state` in
                        // `monitor/scx_walker.rs`).
                        let kernel_for_root = owned_accessor
                            .as_ref()
                            .expect("owned_accessor.is_some() gate above")
                            .guest_kernel();
                        let root_pa = kernel_for_root.text_kva_to_pa(scx_root_kva);
                        let sched_kva = mem.read_u64(root_pa, 0);
                        // Drive the watchpoint state machine via the
                        // pure helper so unit tests can exercise the
                        // full `(last_sched_kva, sched_kva)` transition
                        // matrix (Unchanged / Detached / RebindDisarmed
                        // / Published / PublishDeferred) without
                        // booting a VM. The helper performs all
                        // ordered atomic stores per the contract on
                        // [`super::vcpu::WatchpointArm`]; the caller
                        // owns `last_sched_kva` and the result-driven
                        // logging.
                        match republish_watchpoint_on_rebind(
                            sched_kva,
                            last_sched_kva,
                            sched_offs.exit_kind as u32,
                            &freeze_coord_watchpoint,
                            kernel_for_root,
                            mem,
                        ) {
                            WatchpointPublishResult::Unchanged => {}
                            WatchpointPublishResult::Detached => {
                                tracing::info!(
                                    "freeze-coord: scx_root cleared (scheduler \
                                     detached); watchpoint disarmed pending next \
                                     attach"
                                );
                                last_sched_kva = 0;
                                // Clear cached_exit_kind_pa: the kernel
                                // RCU-freed the prior scx_sched (per
                                // kernel/sched/ext.c::scx_root_disable
                                // → RCU_INIT_POINTER(scx_root, NULL)),
                                // so the cached PA now points at slab
                                // memory subject to allocator reuse. A
                                // subsequent rendezvous-timeout
                                // Degraded report would read garbage
                                // from the recycled page. Invalidate
                                // here; the next Published arm will
                                // republish a fresh PA once the
                                // scheduler re-attaches.
                                cached_exit_kind_pa = None;
                                // Drop the owned BPF map/prog accessors
                                // and wake the accessor-init worker so
                                // it re-builds against the kernel state
                                // the next scheduler will leave behind.
                                // The owned pair's `GuestKernel` /
                                // `map_idr_kva` / `prog_idr_kva` /
                                // `BpfMapOffsets` / `BpfProgOffsets` are
                                // kernel-boot constants and survive a
                                // scheduler swap structurally, but the
                                // per-owner caches (`per_cpu_offsets_
                                // cache` etc.) and any future scheduler-
                                // touching state belong with the
                                // scheduler that was alive when they
                                // populated. Refreshing here keeps the
                                // accessor strictly aligned with the
                                // live scheduler and prevents future
                                // additions to the accessor surface
                                // from silently caching across swaps.
                                owned_accessor = None;
                                owned_prog_accessor = None;
                                *accessors_slot
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner()) = None;
                                let _ = accessor_reinit_evt.write(1);
                            }
                            WatchpointPublishResult::RebindDisarmed {
                                previous,
                                next,
                            } => {
                                tracing::info!(
                                    last_sched_kva = format_args!("{:#x}", previous),
                                    new_sched_kva = format_args!("{:#x}", next),
                                    "freeze-coord: scx_root rebind detected \
                                     (A → B); watchpoint disarmed this tick, \
                                     B will be republished next tick after \
                                     vCPUs clear DR0"
                                );
                                last_sched_kva = 0;
                                // Same rationale as the Detached arm:
                                // the prior scx_sched object is detached
                                // and pending RCU free, and the cached PA
                                // can begin reading recycled slab data.
                                // The next iteration's Published arm
                                // republishes for the new scheduler.
                                cached_exit_kind_pa = None;
                                // Refresh the owned BPF accessors for
                                // the new scheduler (see the
                                // `Detached` arm for the full
                                // rationale).
                                owned_accessor = None;
                                owned_prog_accessor = None;
                                *accessors_slot
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner()) = None;
                                let _ = accessor_reinit_evt.write(1);
                            }
                            WatchpointPublishResult::Published {
                                exit_kind_kva,
                                kind_pa,
                            } => {
                                last_sched_kva = sched_kva;
                                cached_exit_kind_pa = Some(kind_pa);
                                tracing::info!(
                                    exit_kind_kva =
                                        format_args!("{:#x}", exit_kind_kva),
                                    sched_kva = format_args!("{:#x}", sched_kva),
                                    kind_pa = format_args!("{:#x}", kind_pa),
                                    "freeze-coord: watchpoint target \
                                     published; vCPU threads will self-arm \
                                     KVM_SET_GUEST_DEBUG on next iteration"
                                );
                            }
                            WatchpointPublishResult::PublishDeferred {
                                exit_kind_kva,
                            } => {
                                tracing::debug!(
                                    exit_kind_kva =
                                        format_args!("{:#x}", exit_kind_kva),
                                    "freeze-coord: exit_kind translate or \
                                     host-ptr lookup failed; deferring \
                                     watchpoint publish"
                                );
                            }
                        }
                    }
                    // Lazy-resolve the per-CPU runnable_at scan
                    // context once `owned_accessor` lands and the
                    // bootstrap symbols are readable. Skipped entirely
                    // when dual_snapshot is off; failed prerequisites
                    // (missing jiffies_64 symbol, BTF without
                    // sched_ext_entity, etc.) leave `scan_ctx` None
                    // and the early-trigger path stays dormant for
                    // the rest of the run — the late path still works.
                    //
                    // Each failed prerequisite emits a per-iteration
                    // `tracing::debug!` line under the
                    // `RUST_LOG=ktstr=debug` filter — the
                    // DualFailureDumpReport's absent-early Display
                    // message points operators here. Per-iteration
                    // (not single-shot) is the right cadence for
                    // debug output: an operator who asked for verbose
                    // logging wants to see the full retry pattern,
                    // not just one snapshot. The aggregate "something
                    // is wrong" signal stays at the warn level (see
                    // `scan_ctx_warned` below) so default-visible
                    // output still surfaces a single line per run.
                    if scan_tick && freeze_coord_dual_snapshot && scan_ctx.is_none() {
                        // try_resolve consumes the hoisted prereqs
                        // (scan_offsets, scan_jiffies_64_kva,
                        // dump_cpu_time_symbols).
                        // The only field that can flip Some after
                        // coord-start is owned_accessor (boot-race);
                        // every other input is a deterministic function
                        // of the host inputs and was already attempted
                        // at coord-start. A None among them means the
                        // dependency is permanently absent — the
                        // diagnostic warn already names which leg
                        // failed. The closure returns the reason
                        // string alongside None so the late-trigger
                        // skip-reason path can quote it directly into
                        // DualFailureDumpReport::early_skipped_reason.
                        let try_resolve = || -> Result<RunnableScanCtx, &'static str> {
                            let owned = owned_accessor
                                .as_ref()
                                .ok_or("owned_accessor not ready (guest still booting)")?;
                            let scan_offsets = scan_offsets
                                .ok_or("RunnableScanOffsets unavailable (BTF lacks sched_ext_entity)")?;
                            let jiffies_64_kva = scan_jiffies_64_kva
                                .ok_or("jiffies_64 symbol absent from vmlinux")?;
                            let syms = dump_cpu_time_symbols
                                .as_ref()
                                .ok_or("KernelSymbols unavailable (vmlinux parse failed)")?;
                            // The global `scx_tasks` LIST_HEAD is the
                            // walker's only memory anchor. Absent on a
                            // stripped vmlinux or a kernel without
                            // sched_ext — fail the resolve so the
                            // late-trigger skip-reason path quotes the
                            // missing symbol.
                            let scx_tasks_kva = syms.scx_tasks.ok_or(
                                "scx_tasks symbol absent from vmlinux \
                                 (kernel without sched_ext or stripped vmlinux)",
                            )?;
                            let mem = freeze_coord_mem
                                .as_ref()
                                .ok_or("GuestMem unavailable")?;
                            let kernel = owned.guest_kernel();
                            let walk = kernel.walk_context();
                            // Translate jiffies_64's KVA to a PA.
                            // Lives in the kernel text/data mapping —
                            // same as scx_root et al. Use the
                            // GuestKernel-resident base so VA_BITS=47
                            // hosts translate correctly.
                            let jiffies_64_pa = kernel.text_kva_to_pa(jiffies_64_kva);
                            // Compute per-CPU rq PAs for the per-rq
                            // runnable_list walker. The KernelOffsets
                            // schema guarantees `runqueues != 0` (see
                            // `monitor/symbols.rs` — its absence is a
                            // construction-time error), so the only
                            // failure path here is reading
                            // `__per_cpu_offset` early during boot:
                            // the per-CPU offset table reads as zero
                            // for not-yet-online CPUs. A zero offset
                            // does NOT yield a zero PA — `compute_rq_pas`
                            // wraps via `wrapping_sub` into the
                            // upper-half KVA region (see
                            // `compute_rq_pas` doc comment in
                            // `monitor/symbols.rs`), so the resulting
                            // PA is bogus, not zero, and there is no
                            // downstream `rq_pa == 0` short-circuit
                            // to suppress it. Caching such a vec is
                            // permanent for the run and would have
                            // every subsequent walk read garbage for
                            // the not-yet-online slots. Mirror the
                            // `prog_per_cpu_offsets` gate above:
                            // defer scan_ctx construction until every
                            // offset slot is non-zero. A retry is
                            // cheap; a cached miss is permanent.
                            let pco_pa = kernel.text_kva_to_pa(syms.per_cpu_offset);
                            let pco_offsets = crate::monitor::symbols::read_per_cpu_offsets(
                                mem,
                                pco_pa,
                                freeze_coord_num_cpus,
                            );
                            if pco_offsets.contains(&0) {
                                return Err(
                                    "not all per_cpu_offsets resolved \
                                     (some CPUs still booting)",
                                );
                            }
                            // Virt-KASLR slide from the shared
                            // `kern_virt_kaslr` Arc (populated by
                            // BSP MSR_LSTAR derive on x86_64 +
                            // guest-channel KERN_ADDRS `_text`
                            // subtraction on both arches). 0 fallback
                            // matches KASLR-off semantics (compute_rq_pas
                            // collapses to the no-slide formula); the
                            // `coord_kaslr_offset` closure handles the
                            // `+1` bias strip + 0-sentinel folding.
                            let rq_pas = crate::monitor::symbols::compute_rq_pas(
                                syms.runqueues,
                                &pco_offsets,
                                walk.page_offset,
                                coord_kaslr_offset(),
                            );
                            // scx_watchdog_timestamp is a `.data`
                            // file-scope static — same text-mapping
                            // translation as scx_watchdog_timeout
                            // (which lives a few lines below the
                            // timestamp in kernel/sched/ext.c).
                            // Optional because the symbol is absent
                            // on kernels without sched_ext or
                            // stripped vmlinux; max_runnable_age
                            // skips the contribution when None.
                            let watchdog_timestamp_pa =
                                syms.scx_watchdog_timestamp.map(|kva| kernel.text_kva_to_pa(kva));
                            Ok(RunnableScanCtx {
                                scx_tasks_kva,
                                rq_pas,
                                offsets: scan_offsets,
                                jiffies_64_pa,
                                watchdog_timestamp_pa,
                                walk,
                                start_kernel_map: kernel.start_kernel_map(),
                                phys_base: kernel.phys_base(),
                                kaslr_offset: coord_kaslr_offset(),
                            })
                        };
                        match try_resolve() {
                            Ok(ctx) => {
                                scan_ctx = Some(ctx);
                                scan_ctx_skip_reason = None;
                            }
                            Err(reason) => {
                                scan_ctx_skip_reason = Some(reason);
                            }
                        }
                    }
                    // Single-shot warn when the resolve has been
                    // failing long enough that "still booting" is no
                    // longer a plausible explanation. Without this
                    // an operator running ktstr against a kernel that
                    // lacks `sched_ext_entity` BTF (sched_ext disabled)
                    // or `jiffies_64` (stripped vmlinux) gets the
                    // dual-snapshot path silently disabled; the late
                    // dump still works, but the early snapshot would
                    // never fire and the missing wrapper could be
                    // mistaken for "stall fired before half-way
                    // threshold". Counting iterations under the
                    // dual-snapshot gate ensures the message only
                    // surfaces in runs where the path was requested.
                    if scan_tick && freeze_coord_dual_snapshot && scan_ctx.is_none() {
                        scan_ctx_retries += 1;
                        if !scan_ctx_warned && scan_ctx_retries >= SCAN_CTX_WARN_AFTER_ITERS {
                            tracing::warn!(
                                "freeze-coord: runnable_at scan prerequisites unavailable \
                                 (most commonly: guest still booting; or BTF lacks \
                                 sched_ext_entity, jiffies_64 symbol missing) — \
                                 early-trigger path delayed — will continue retrying"
                            );
                            scan_ctx_warned = true;
                        }
                    }
                    // Poll for the late-trigger condition. The
                    // hardware watchpoint on `*scx_root->exit_kind`
                    // is the primary path: every vCPU thread sets
                    // `freeze_coord_watchpoint.hit` (Release) on
                    // `KVM_EXIT_DEBUG`, which the Acquire load here
                    // observes synchronously — no 100 ms polling
                    // window. The BPF .bss `cached_bss_pa` read
                    // (gated through [`bss_read_state`] for
                    // PA-validity vs not-fired distinction) is
                    // checked alongside the watchpoint every
                    // iteration: it remains a useful redundancy on
                    // kernels where the watchpoint armed (the
                    // typed three-way result also catches a stale
                    // cached PA that bare `read_u32` would mask as
                    // "no fire") AND a fallback for kernels where
                    // the watchpoint never armed (no `scx_root`
                    // symbol, BTF stripped of `scx_sched`, or
                    // `KVM_SET_GUEST_DEBUG` rejected by the host).
                    //
                    // Once `freeze_state == Done` the late-trigger
                    // dispatch has already taken its terminal
                    // transition — re-evaluating
                    // `compute_err_triggered(...)` is wasted work
                    // for the rest of the run (sticky bss latch
                    // keeps reporting Triggered, sticky watchpoint
                    // hit keeps reporting true). Skip the read
                    // entirely once the state machine has closed.
                    let (watchpoint_hit, bss_state) =
                        if freeze_state == FreezeState::Done {
                            (false, BssReadState::NotResolved)
                        } else {
                            let wp = freeze_coord_watchpoint
                                .hit
                                .load(Ordering::Acquire);
                            let st = bss_read_state(
                                freeze_coord_mem.as_deref(),
                                cached_bss_pa,
                            );
                            // OnlyTriggered counts as "fire";
                            // OutOfBounds and NotResolved /
                            // NotTriggered all mean "no
                            // observable fire this iteration".
                            // Surfacing OOB once with a warn lets
                            // an operator notice when the .bss
                            // path has gone stale without
                            // changing the trigger arithmetic.
                            if matches!(st, BssReadState::OutOfBounds)
                                && !bss_oob_warn_logged
                            {
                                tracing::warn!(
                                    cached_bss_pa =
                                        cached_bss_pa
                                            .map(|p| format!("{p:#x}"))
                                            .unwrap_or_else(|| "None".to_string()),
                                    "freeze-coord: cached BPF .bss PA no \
                                     longer resolves to a 4-byte-readable \
                                     DRAM region — probe map likely freed \
                                     mid-run; .bss late-trigger fallback is \
                                     now silent for the rest of the run \
                                     (watchpoint path, if armed, remains \
                                     active)"
                                );
                                bss_oob_warn_logged = true;
                            }
                            (wp, st)
                        };
                    let mut err_triggered =
                        compute_err_triggered(watchpoint_hit, bss_state);
                    if !err_triggered
                        && scan_tick
                        && freeze_state != FreezeState::Done
                        && let Some(ek_pa) = cached_exit_kind_pa
                        && let Some(ref mem) = freeze_coord_mem
                    {
                        let kind = mem.read_u32(ek_pa, 0);
                        const SCX_EXIT_ERROR: u32 = 1024;
                        if kind >= SCX_EXIT_ERROR {
                            err_triggered = true;
                        }
                    }
                    if !err_triggered
                        && (bsp_done_final_pass || sched_exit_final_pass)
                        && freeze_state != FreezeState::Done
                        && let (Some(owned), Some(mem)) =
                            (owned_accessor.as_ref(), freeze_coord_mem.as_deref())
                        {
                            let kernel = owned.guest_kernel();
                            let walk = kernel.walk_context();
                            let mut ek_kva = freeze_coord_watchpoint
                                .request_kva
                                .load(Ordering::Acquire);
                            if ek_kva == 0
                                && let Some(syms) = dump_cpu_time_symbols.as_ref()
                                    && let Some(root_kva) = syms.scx_root
                                    && let Some(ref offs) = dump_scx_walker_offsets
                                    && let Some(ref so) = offs.sched
                                {
                                    let root_pa = kernel.text_kva_to_pa(root_kva);
                                    let sched_kva = mem.read_u64(root_pa, 0);
                                    if sched_kva != 0 {
                                        ek_kva = sched_kva + so.exit_kind as u64;
                                    }
                                }
                            if ek_kva != 0
                                && let Some(pa) = crate::monitor::idr::translate_any_kva(
                                    mem,
                                    walk.cr3_pa,
                                    walk.page_offset,
                                    ek_kva,
                                    walk.l5,
                                    walk.tcr_el1,
                                ) {
                                    let kind = mem.read_u32(pa, 0);
                                    const SCX_EXIT_ERROR: u32 = 1024;
                                    if kind >= SCX_EXIT_ERROR {
                                        err_triggered = true;
                                    }
                                }
                        }
                    // Closures capture by reference. Building the
                    // full freeze-rendezvous-dump cycle once and
                    // calling it for either the early or late
                    // snapshot keeps the drain-dance contract
                    // (immediate_exit pass 1 → release fence →
                    // signal pass 2 → N-of-N rendezvous) defined in
                    // exactly one place. Returns
                    // `Some(FailureDumpReport)` when the rendezvous
                    // succeeded; None on timeout (the surrounding
                    // logic still thaws). The thaw is the caller's
                    // responsibility so the same closure works for
                    // a state-resetting late freeze (thaw to allow
                    // teardown to run) and a transient early freeze
                    // (thaw to let the test continue).
                    // `gate_on_exit_kind` filters out spurious watchpoint
                    // fires on a non-error `exit_kind` value. The
                    // hardware watchpoint catches every write to
                    // `*scx_root->exit_kind` regardless of value —
                    // including transient writes during init/teardown
                    // that the kernel sets to `SCX_EXIT_NONE` (0) or
                    // `SCX_EXIT_DONE` (1). Without the gate, every
                    // clean scheduler shutdown would synthesize a
                    // bogus failure dump. The gate runs AFTER the
                    // rendezvous succeeds (vCPUs parked → guest
                    // memory consistent) and BEFORE building the
                    // dump: read the 4-byte `exit_kind` value at the
                    // already-resolved KVA, compare against the
                    // error-class boundary `SCX_EXIT_ERROR = 1024`
                    // (per `kernel/sched/ext_internal.h::scx_exit_kind`).
                    // Gate failures return None — the late-trigger
                    // call site treats this as "spurious watchpoint
                    // fire, reset hit and keep watching" rather than
                    // the normal "rendezvous timed out, give up"
                    // semantics. The early (runnable_at) trigger and
                    // BPF-bss late trigger pass `false`: those paths
                    // are already gated on their own conditions
                    // (half-way age threshold; tp_btf handler latch
                    // on error-class kinds), so an extra exit_kind
                    // read would be redundant overhead.
                    // kvm_clock save/restore around the freeze
                    // rendezvous. Captured AFTER all vCPUs ack
                    // park (inside the polling loop, gated on
                    // `parked_count >= expected_parks`) + restored
                    // BEFORE the thaw gate flip so the guest's
                    // first post-resume kvm_clock read returns the
                    // parked-state value rather than the current
                    // (advanced-by-freeze-duration) host monotonic.
                    // Without this, every snapshot dump (~100ms)
                    // leaves the guest seeing the freeze window
                    // as elapsed time, which accumulates phantom
                    // run_delay in schedstat for the currently-
                    // running task and can trip ktstr's own
                    // monitor assertions on run_delay thresholds.
                    //
                    // Post-park-ack capture avoids a backward-jump
                    // race on stable-TSC guests: those bypass the
                    // `__pvclock_clocksource_read` cmpxchg ratchet
                    // (arch/x86/kernel/pvclock.c), so if any vCPU
                    // reads pvclock between a pre-freeze GET and
                    // park-ack, the post-thaw restored read would
                    // appear lower than the last pre-park read.
                    //
                    // Kernel ground truth: KVM_GET_CLOCK is a pure
                    // seqcount read (arch/x86/kvm/x86.c
                    // get_kvmclock — no lock, no SEV-ES gating);
                    // KVM_SET_CLOCK queues a per-vCPU
                    // KVM_REQ_CLOCK_UPDATE that the vCPU processes
                    // on its next entry into guest mode (the
                    // ioctl syscall itself provides the
                    // memory-ordering barrier — no explicit fence
                    // needed between the kvm_set_clock call and
                    // the subsequent freeze.store(false, Release)
                    // because the syscall return happens-before
                    // any subsequent atomic write). SET MUST
                    // therefore complete before the thaw signal
                    // flips freeze=false (otherwise a vCPU that
                    // wakes and re-enters KVM_RUN before the
                    // request is queued reads the OLD pvclock
                    // page on its first guest instruction).
                    // `flags = 0` on SET is mandatory — leaving
                    // `KVM_CLOCK_REALTIME` in flags causes
                    // set_clock to apply a realtime adjustment
                    // that double-counts elapsed time (matches the
                    // existing boot-time precedent in this crate).
                    //
                    // Differs from firecracker
                    // (`arch/x86_64/vm.rs` set_clock) which only
                    // strips `KVM_CLOCK_TSC_STABLE` and leaves the
                    // `KVM_CLOCK_REALTIME` bit set — fc accepts
                    // the realtime re-base because its
                    // snapshot/restore lifecycle crosses host
                    // boundaries where the realtime base may
                    // legitimately have shifted; ktstr's
                    // same-host freeze mask wants ZERO
                    // adjustment so the guest sees the freeze
                    // duration as instant. Follows the project's
                    // reference-impl-divergence documentation
                    // convention.
                    //
                    // Skip-freeze fast path (BSP exited before
                    // gate flip): no clock save needed — no vCPU
                    // will read kvm_clock again before VM
                    // teardown. None sentinel signals to
                    // thaw_and_barrier that there's nothing to
                    // restore.
                    //
                    // RefCell because freeze_and_dispatch (writer)
                    // and thaw_and_barrier (reader+consumer) are
                    // sibling closures sharing the state. Both
                    // run on the coordinator thread; never
                    // concurrent. The borrow_mut() pattern
                    // matches the snapshot-bridge state
                    // management at L831-842.
                    let kvm_clock_save_for_freeze: std::cell::RefCell<
                        Option<kvm_bindings::kvm_clock_data>,
                    > = std::cell::RefCell::new(None);
                    // Use the raw fd (Copy) rather than a borrow of
                    // vm.vm_fd — `vm` is moved into a downstream
                    // closure later in this scope, so a borrow of
                    // vm.vm_fd here would extend the borrow past
                    // that move via the freeze_and_dispatch +
                    // thaw_and_barrier closures and the compiler
                    // rejects. The raw fd is closed when vm.vm_fd
                    // drops at run_vm scope exit; freeze_and_dispatch
                    // + thaw_and_barrier always run before that
                    // drop so the fd is live for every ioctl call.
                    let vm_fd_raw_for_freeze: i32 = vm_fd_raw_for_coord;
                    let freeze_and_dispatch =
                        |mode: FreezeMode<'_>|
                            -> FreezeOutcome {
                            // Destructure mode into the per-mode locals used
                            // by the rest of the closure body. The Capture
                            // arm's gate_on_exit_kind threads through the
                            // existing snapshot pipeline unchanged; the
                            // ColdOp branch fires at the V4 split (post
                            // park-ack rendezvous) and returns
                            // FreezeOutcome::KernelOp(reply) directly.
                            let gate_on_exit_kind = match &mode {
                                FreezeMode::Capture { gate_on_exit_kind } => *gate_on_exit_kind,
                                FreezeMode::ColdOp(_) => false,
                            };
                            let skip_freeze =
                                freeze_coord_bsp_done.load(Ordering::Acquire);
                            // The kvm_clock GET capture is
                            // deferred until after all vCPUs are
                            // confirmed parked (see the
                            // `all_parked = true; break;` site in
                            // the polling loop below). Capturing
                            // before the SIGRTMIN kick + park-ack
                            // rendezvous would race with vCPU
                            // pvclock reads in the window between
                            // GET and the last park-ack; stable-
                            // TSC guests bypass the
                            // `__pvclock_clocksource_read` cmpxchg
                            // ratchet (arch/x86/kernel/pvclock.c)
                            // and would observe a backward jump
                            // on resume (last pre-park read at
                            // pvclock(t_kick + ε) → post-thaw
                            // restored read at pvclock(t_get) <
                            // t_kick). Post-park-ack capture
                            // closes the window: no vCPU reads
                            // pvclock between the captured
                            // moment and the restore + thaw.
                            if skip_freeze {
                                tracing::info!(
                                    gate_on_exit_kind,
                                    "freeze-coord: BSP exited, capturing \
                                     quiesced guest memory without freeze"
                                );
                            } else {
                                tracing::info!(
                                    gate_on_exit_kind,
                                    "freeze-coord: freezing vCPUs for snapshot"
                                );
                            }
                            // Capture wall-clock start for the
                            // post-dump timing summary one-liner.
                            // Returned alongside the report so the
                            // call site can reuse it across the
                            // post-thaw JSON emit (covers freeze
                            // rendezvous → dump_state → numa-stats →
                            // serialise → file-write window with one
                            // anchor).
                            let capture_start = Instant::now();
                            // Soft deadline for the whole capture path
                            // (rendezvous + dump_state + numa stats).
                            // Set to half the configured watchdog so a
                            // slow dump can't keep vCPUs parked past
                            // the kernel's own SCX_EXIT_ERROR_STALL
                            // emission line. `freeze_coord_watchdog_half`
                            // already encodes the divide-by-2 (see
                            // its definition above) and falls back to
                            // 2s when the builder didn't set
                            // watchdog_timeout. Using it here couples
                            // the dump bailout to the same horizon
                            // the per-CPU runnable_at scanner uses
                            // for the dual-snapshot half-way trigger.
                            let capture_deadline = if freeze_coord_watchdog_half
                                > Duration::ZERO
                            {
                                Some(capture_start + freeze_coord_watchdog_half)
                            } else {
                                None
                            };
                            // 'capture labeled block: every exit
                            // from the freeze→park→dump phases
                            // (rendezvous timeout, gate-suppressed,
                            // full dump, partial dump) `break
                            // 'capture <result>` so all paths
                            // converge on the labeled block's value
                            // — which is the closure's return.
                            // The caller is responsible for invoking
                            // `thaw_and_barrier` AFTER it has done
                            // any while-frozen work it needs (the
                            // late-trigger backstop reads guest
                            // memory while quiesced, so the thaw
                            // cannot be unconditional inside the
                            // closure).
                            'capture: {
                            // Cycle-entry snapshot of BSP liveness
                            // used for non-UAF-sensitive bookkeeping:
                            // parked_evt pre-seed gating
                            // (`bsp_parked` lookup), `expected_parks`
                            // accounting (+1 for BSP), pass-2
                            // `pthread_kill`, and the rendezvous-wait
                            // diagnostics. None of those callsites
                            // dereference the BSP's `kvm_run` mmap, so
                            // a stale `true` is benign:
                            // `pthread_kill` against an exited tid
                            // returns ESRCH, an over-counted
                            // `expected_parks` heals on the next
                            // SIGRTMIN/park-ack overshoot path, and
                            // pre-seed reads only the AtomicBool
                            // `bsp_parked` flag.
                            //
                            // The TOCTOU-sensitive
                            // `ImmediateExitHandle::set(1)` against
                            // the BSP's `kvm_run` mmap is gated by
                            // its own fresh Acquire load further
                            // below — see the "Re-load `bsp_alive`
                            // immediately before the BSP `ie.set()`"
                            // comment for the full rationale (a stale
                            // snapshot there would write through a
                            // pointer into freed `kvm_run` pages
                            // after the BSP drops its `VcpuFd`).
                            //
                            // The primary line of defense remains
                            // `freeze_coord_handle.join()` in run_vm
                            // BEFORE the BSP `VcpuFd` falls out of
                            // scope; the in-closure loads are
                            // defense-in-depth.
                            let bsp_alive_at_start =
                                bsp_alive_for_coord.load(Ordering::Acquire);
                            // Drain `parked_evt` BEFORE flipping
                            // `freeze=true` and BEFORE issuing
                            // pass-0 (worker pause), pass-1
                            // (immediate_exit), or pass-2 (SIGRTMIN).
                            // From this point forward every
                            // increment to the parked_evt counter
                            // is unambiguously a park-ack for THIS
                            // cycle. Draining AFTER the kicks is a
                            // race: a fast vCPU or worker may park
                            // and bump the counter between the kick
                            // and the drain — that ack is then
                            // absorbed by the drain instead of
                            // counted toward `parked_count`, and
                            // the rendezvous waits 30 s for an ack
                            // that already fired.
                            //
                            // EAGAIN under EFD_NONBLOCK (counter
                            // already 0 from the prior cycle's
                            // post-thaw barrier drain) is benign.
                            //
                            // The Acquire ordering synchronizes-with
                            // the parker's Release store after its
                            // drain dance — this rendezvous IS the
                            // memory barrier that makes the future
                            // host-side guest-memory reads correct.
                            // The eventfd write ordering is
                            // load-bearing: the AtomicBool Release
                            // happens-before the eventfd write, so
                            // every counter increment we observe in
                            // the loop below implies every
                            // guest-side queue mutation the parker
                            // performed pre-park is visible to the
                            // dump.
                            //
                            // Also drain `thaw_evt` here. The
                            // coordinator writes thaw_evt ONCE per
                            // thaw, and every parked vCPU polls the
                            // SAME fd in `handle_freeze` without
                            // draining (the multi-reader fan-out
                            // wake design pinned at
                            // `vmm/exit_dispatch.rs::handle_freeze`).
                            // Without a per-cycle drain by the
                            // coordinator the counter is monotonic
                            // — every successive freeze cycle sees
                            // a level-high thaw_evt left over from
                            // the previous cycle, which makes
                            // `handle_freeze`'s poll return
                            // immediately on every iteration and
                            // burns CPU spinning on
                            // `freeze.load(Acquire)` until the
                            // coordinator clears `freeze`. Draining
                            // before pass-1 / pass-2 means the next
                            // poll inside `handle_freeze` blocks on
                            // an empty counter and wakes only when
                            // (a) the coordinator's post-rendezvous
                            // `thaw_evt.write(1)` lands or (b) the
                            // 100 ms poll backstop fires.
                            use std::os::fd::AsRawFd;
                            let _ = freeze_coord_parked_evt.read();
                            let _ = freeze_coord_thaw_evt.read();
                            // Reset every per-vCPU regs slot to None
                            // before the freeze kicks. Without this
                            // clear, regs slots written during a
                            // PRIOR successful capture (periodic /
                            // user-watchpoint / late-trigger) stay
                            // `Some(stale_snapshot)` across cycles —
                            // and a vCPU that DOES NOT park this
                            // cycle (the rendezvous-timeout case the
                            // Degraded report is meant to surface)
                            // shows up as `Some(stale)` instead of
                            // the contracted `None`. That violates
                            // the
                            // [`crate::monitor::dump::DegradedFailureDumpReport::vcpu_regs`]
                            // doc invariant ("None identifies vCPUs
                            // that never parked") and silently
                            // misleads the operator about which
                            // vCPUs stalled. The Mutex acquisitions
                            // are cheap (single-digit microseconds);
                            // the writer is the vCPU's SIGRTMIN
                            // handler which only runs AFTER the
                            // freeze kicks below, so this clear
                            // races nothing.
                            *freeze_coord_bsp_regs
                                .lock_unpoisoned() = None;
                            for ap in &freeze_coord_ap_regs {
                                *ap.lock_unpoisoned() = None;
                            }
                            // Snapshot virtio-blk worker liveness
                            // BEFORE pause(). When the device exists
                            // but the worker thread is not yet
                            // spawned (pre-DRIVER_OK) or has been
                            // joined (post-stop / failed-respawn),
                            // pause() short-circuits with the
                            // "no-live-worker" fast path and writes
                            // no parked_evt ack — counting +1 in
                            // that case makes the rendezvous wait
                            // 30 s for a worker that does not
                            // exist. The pre-pause `paused` flag
                            // is the cheapest available proxy: the
                            // worker spawn flips it to false on
                            // entry to the run loop and the
                            // post-thaw barrier guarantees a live
                            // worker has cleared it before this
                            // cycle starts. `paused == true` at
                            // cycle entry therefore means "no live
                            // worker" (the construction sentinel
                            // or post-stop re-armed sentinel from
                            // `resume()`). Gate the +1 below on
                            // `worker_was_running` instead of bare
                            // `is_some()`.
                            let worker_was_running = freeze_coord_virtio_blk_paused
                                .as_ref()
                                .is_some_and(|p| !p.load(Ordering::Acquire));
                            // Pre-seed the parked_evt counter for any
                            // parker whose flag is STILL `true` at
                            // cycle entry. The post-thaw barrier at
                            // the end of every prior cycle SHOULD
                            // have observed every parker clear its
                            // flag before returning, but the barrier
                            // can hit its FREEZE_RENDEZVOUS_TIMEOUT
                            // and break early (logged as
                            // "post-thaw barrier timed out — a parker
                            // did not clear within
                            // FREEZE_RENDEZVOUS_TIMEOUT" further
                            // below). When that happens the parker is
                            // still inside `handle_freeze`'s park
                            // loop with `parked=true`. The next cycle
                            // sets `freeze=true` and SIGRTMINs every
                            // vCPU; the kicked vCPU's poll wakes
                            // (EINTR), re-checks `freeze` (now true),
                            // and stays in the SAME `handle_freeze`
                            // invocation — `parked.store(true)` plus
                            // `parked_evt.write(1)` only run on
                            // ENTRY to `handle_freeze` (the
                            // freeze-park ack path in
                            // `crate::vmm::exit_dispatch`), NOT
                            // per `freeze=true` flip while parked.
                            // Without pre-seeding, the rendezvous
                            // countdown latch never receives an ack
                            // from that parker and waits the full
                            // 30 s for an event that already happened
                            // a cycle ago.
                            //
                            // Pre-seeding +1 to `parked_evt` per
                            // still-parked parker compensates: the
                            // rendezvous loop drains the counter
                            // and credits each as a park-ack for
                            // THIS cycle. This is equivalent to the
                            // historical force-clear of `parked`
                            // flags but targeted — only fires for
                            // the timed-out subset, leaving a
                            // healthy mid-thaw parker (which still
                            // has its `parked=true` from the prior
                            // cycle and is about to clear it within
                            // a few ms) untouched. The worst case
                            // is a healthy parker that races the
                            // pre-seed: its own
                            // `parked_evt.write(1)` on the next
                            // entry to `handle_freeze` adds another
                            // count, which is harmless — the
                            // rendezvous loop only checks
                            // `parked_count >= expected_parks` and
                            // overshoot is fine.
                            //
                            // The bsp/ap loads here are Acquire to
                            // synchronise-with the prior cycle's
                            // post-thaw barrier reads — the post-
                            // thaw barrier already loaded these
                            // with Acquire, but a healthy parker may
                            // have flipped its flag back to false
                            // between the barrier's last load and
                            // this point. The seed only fires when
                            // we still observe `true`, so a healthy
                            // late-clear is a no-op.
                            let mut still_parked: u32 = 0;
                            for ap in freeze_coord_ap_parked.iter() {
                                if ap.load(Ordering::Acquire) {
                                    still_parked = still_parked.saturating_add(1);
                                }
                            }
                            if bsp_alive_at_start
                                && freeze_coord_bsp_parked.load(Ordering::Acquire)
                            {
                                still_parked = still_parked.saturating_add(1);
                            }
                            // virtio-blk worker: only pre-seed when
                            // the worker was running (otherwise
                            // pause() short-circuits and we won't
                            // count +1 anyway). If the worker is
                            // running AND `paused == true`, the
                            // worker is mid-park from the prior
                            // cycle and won't re-write its ack on
                            // the next pause()-driven epoll wake,
                            // mirroring the vCPU case. Pause-fd
                            // writes happen-before `paused.store
                            // (true)`, which happens-before the
                            // worker's `parked_evt.write(1)` —
                            // matching the vCPU sequence in
                            // `handle_freeze`.
                            if worker_was_running
                                && freeze_coord_virtio_blk_paused
                                    .as_ref()
                                    .is_some_and(|p| p.load(Ordering::Acquire))
                            {
                                still_parked = still_parked.saturating_add(1);
                            }
                            if still_parked > 0 {
                                tracing::warn!(
                                    still_parked,
                                    ?rendezvous_timeout,
                                    "freeze-coord: detected stale parked=true \
                                     parker(s) at cycle entry — prior post-thaw \
                                     barrier likely timed out. Pre-seeding \
                                     parked_evt to credit them as acks for this \
                                     cycle so the rendezvous does not wait the \
                                     full rendezvous timeout for events that \
                                     already fired."
                                );
                                if let Err(e) =
                                    freeze_coord_parked_evt.write(still_parked as u64)
                                {
                                    tracing::warn!(
                                        err = %e,
                                        still_parked,
                                        ?rendezvous_timeout,
                                        "freeze-coord: parked_evt pre-seed write \
                                         failed; rendezvous may wait the full \
                                         rendezvous timeout for stale parker(s)"
                                    );
                                }
                            }
                            freeze_coord_freeze.store(true, Ordering::Release);
                            // No force-clear of `parked` flags here.
                            // The post-thaw barrier at the END of
                            // every prior freeze_and_dispatch cycle
                            // (see `// Post-thaw barrier` below)
                            // is the primary guarantee that every
                            // vCPU has run its trailing
                            // `parked.store(false)` before this
                            // cycle starts. Force-clearing
                            // mid-cycle would erase the legitimate
                            // `parked=true` of a vCPU still in cycle
                            // N's park loop and deadlock the
                            // rendezvous (vCPU never re-stores
                            // parked=true; coord waits 30 s). The
                            // pre-seed above handles the residual
                            // case where the post-thaw barrier
                            // itself timed out.
                            // Pass 0: signal every device worker to
                            // pause. virtio-blk has an independent
                            // worker thread that must be parked
                            // before we read guest memory — otherwise
                            // it can race-mutate the avail/used rings
                            // and the backing file mid-dump,
                            // producing a torn view of in-flight
                            // requests. Other devices (virtio-net,
                            // virtio-console) run on the vCPU thread
                            // and freeze automatically at the vCPU
                            // rendezvous below.
                            //
                            // The worker may be in `pread`/`pwrite`
                            // when this lands; the eventfd write
                            // returns immediately (counter mode +
                            // EFD_NONBLOCK) and the syscall completes
                            // before the worker reaches the next
                            // `epoll_wait` and observes PAUSE_TOKEN.
                            // The rendezvous loop below polls each
                            // worker's `paused` flag with the same
                            // FREEZE_RENDEZVOUS_TIMEOUT budget that
                            // bounds the vCPU wait — workers ack
                            // within ~1 ms in healthy state and the
                            // 30 s ceiling absorbs sick-system stalls.
                            if let Some(ref blk) = freeze_coord_virtio_blk {
                                blk.lock().pause();
                            }
                            // Pass 1: set every immediate_exit=1.
                            // Each ImmediateExitHandle::set is a
                            // single-byte write_volatile into the
                            // corresponding kvm_run mmap (MAP_SHARED,
                            // lifetime tied to the running VcpuFd
                            // that owns it).
                            //
                            // Primary defense for the AP path: the
                            // AP threads are joined in
                            // `collect_results` AFTER the coord
                            // joins, so in the normal lifecycle the
                            // coord cannot outlive an AP's `VcpuFd`.
                            // The exception is panic-unwind under
                            // `panic = "unwind"` (test profile),
                            // where the AP's panic hook fires
                            // synchronously on the panicking thread
                            // and the subsequent stack drop unmaps
                            // the AP's `kvm_run` page mid-cycle —
                            // before any join. Without a per-AP
                            // gate the unguarded `ie.set(1)` above
                            // would `write_volatile` through a
                            // pointer into freed memory.
                            //
                            // Secondary defense: each AP carries an
                            // `Arc<AtomicBool>` (`VcpuThread::alive`)
                            // that the AP's panic hook flips to
                            // `false` BEFORE unwinding starts.
                            // The Acquire load below
                            // synchronizes-with that Release store
                            // (panic hook runs synchronously on the
                            // panicking AP thread before unwind),
                            // so a `true` reading observed here
                            // happens-before any subsequent unwind
                            // drop of `vcpu`. Mirrors the
                            // BSP-side `bsp_alive` TOCTOU-tightened
                            // gate: load fresh at the actual
                            // `ie.set` site, not at cycle entry.
                            // `iter().enumerate()` walks index
                            // alongside the handle so the
                            // `freeze_coord_ap_alive[i]` lookup
                            // stays index-aligned.
                            //
                            // The BSP IE write is gated on
                            // `bsp_alive` because run_vm drops the
                            // BSP before collect_results runs; see
                            // the gate's doc above.
                            for (i, ie) in freeze_coord_ap_ies.iter().enumerate() {
                                if let Some(ie) = ie
                                    && freeze_coord_ap_alive[i]
                                        .load(Ordering::Acquire)
                                {
                                    ie.set(1);
                                }
                            }
                            // Re-load `bsp_alive` immediately before the
                            // BSP `ie.set()` instead of reusing the
                            // cycle-entry snapshot (`bsp_alive_at_start`).
                            // The snapshot is captured at the top of
                            // 'capture and is many milliseconds stale by
                            // the time pass-1 runs (worker pause()+ack,
                            // parked_evt pre-seed, the freeze=true
                            // Release store, and the virtio-blk
                            // pause()-rendezvous all happen in between).
                            // The BSP run-loop can transition
                            // `bsp_alive=false` and drop its `VcpuFd` at
                            // any point in that window. Without a
                            // fresh load, `ImmediateExitHandle::set(1)`
                            // would issue a `write_volatile` through a
                            // pointer into a `kvm_run` mmap whose
                            // backing pages were unmapped when the BSP
                            // `VcpuFd` was dropped (the kernel's
                            // `kvm_vcpu_release` path tears down the
                            // `kvm_run` MAP_SHARED region; subsequent
                            // userspace writes against the stale
                            // pointer are use-after-free into freed
                            // pages). The Acquire load pairs with the
                            // BSP run-loop's Release store of `false`
                            // on its way out: a `bsp_alive_now == true`
                            // observed here happens-before any
                            // `false` the BSP could subsequently
                            // store, which means the BSP `VcpuFd` is
                            // still alive AT the moment of `ie.set()`
                            // and cannot be dropped until the next
                            // load reads false. Pass-2's pthread_kill
                            // and the rendezvous-wait below issue
                            // their own fresh Acquire loads for the
                            // same TOCTOU reason.
                            let bsp_alive_for_ie =
                                bsp_alive_for_coord.load(Ordering::Acquire);
                            if bsp_alive_for_ie
                                && let Some(ref ie) = freeze_coord_bsp_ie_handle
                            {
                                ie.set(1);
                            }
                            // Release fence between pass 1 and pass 2
                            // so all immediate_exit writes are
                            // observable before any vCPU thread
                            // receives the kick signal — without
                            // this, a thread could process its signal,
                            // enter KVM_RUN, and miss the
                            // immediate_exit byte that is supposed to
                            // short-circuit guest entry.
                            std::sync::atomic::fence(Ordering::Release);
                            // Pass 2: signal every vCPU. AP signals
                            // are always safe; the BSP signal is
                            // gated on `bsp_alive_at_start` — the
                            // cycle-entry snapshot — rather than a
                            // fresh load. `pthread_kill` against an
                            // exited tid returns ESRCH and is
                            // harmless either way: a stale `true`
                            // here just adds one ESRCH-suppressing
                            // log line; a stale `false` is fine
                            // because the BSP transitioned dead
                            // between entry and now, so it neither
                            // needs nor can receive the kick. Unlike
                            // `ImmediateExitHandle::set(1)` above,
                            // `pthread_kill` does not dereference
                            // any per-`VcpuFd`-owned mmap, so there
                            // is no use-after-free hazard requiring
                            // a re-load.
                            for &tid in &freeze_coord_ap_pthreads {
                                unsafe {
                                    libc::pthread_kill(tid, vcpu_signal());
                                }
                            }
                            if bsp_alive_at_start {
                                unsafe {
                                    libc::pthread_kill(freeze_coord_bsp_tid, vcpu_signal());
                                }
                            }
                            // Wait for N-of-N parked acks via a
                            // countdown latch over `parked_evt`. The
                            // counter-mode eventfd accumulates one
                            // write per parker (every vCPU + the
                            // virtio-blk worker writes 1 AFTER its
                            // own Release store on parked/paused).
                            // Each `read()` drains the accumulated
                            // count atomically and resets it; the
                            // closure tallies these drains until the
                            // total reaches `expected`. Replaces the
                            // per-iteration O(N) AtomicBool scan with
                            // an O(1) counter add — the AtomicBool
                            // flags remain the synchronizes-with
                            // anchor for the diagnostic timeout-log
                            // path below, but they are no longer the
                            // hot-path readiness check.
                            //
                            // The pre-pass drain above (before the
                            // `freeze=true` flip and the kicks)
                            // ensures every increment we observe
                            // from here on is a park-ack for THIS
                            // cycle, not an ack from cycle N-1
                            // that arrived after the post-thaw
                            // barrier's drain.
                            //
                            // The +1 for virtio-blk is gated on
                            // `worker_was_running` — when the
                            // worker thread is not alive, pause()
                            // is a no-op and writes no parked_evt
                            // ack, so counting +1 would make the
                            // rendezvous wait 30 s for an ack that
                            // never comes.
                            let mut expected_parks: u64 =
                                freeze_coord_ap_parked.len() as u64
                                    + if bsp_alive_at_start { 1 } else { 0 }
                                    + if worker_was_running { 1 } else { 0 };
                            let deadline = Instant::now() + rendezvous_timeout;
                            // Sub-deadline for the virtio-blk worker
                            // ack. The `Drop` shutdown path
                            // (`lifecycle.rs`) and its reset sibling
                            // (`control.rs::stop_worker_and_reclaim_state`)
                            // write `paused.store(false, Release)` BEFORE
                            // signalling stop_fd and joining the
                            // worker — see `lifecycle.rs:427`
                            // (`self.paused.store(false,
                            // Ordering::Release)`) + `lifecycle.rs:436`
                            // (`signal_worker_stop`).
                            // Between that store and the worker
                            // exiting (with no further `paused=true`
                            // store on the shutdown path), the
                            // freeze-coord pre-pause snapshot here
                            // observes `paused == false` and counts
                            // `worker_was_running = true → +1` —
                            // but no live thread will write
                            // `parked_evt` for this cycle. Without a
                            // sub-deadline the rendezvous waits the
                            // full 30 s for an ack the worker
                            // physically cannot send.
                            //
                            // 1 s budget covers a healthy worker's
                            // `pread`/`pwrite` drain on warm page
                            // cache (the same envelope
                            // `DROP_JOIN_TIMEOUT` (1 s) commits to
                            // for the worker join in
                            // `lifecycle.rs`). If the worker hasn't
                            // parked within 1 s, it's likely
                            // mid-shutdown (signal_worker_stop
                            // pre-clears paused=false). Dropping
                            // the +1 avoids a 30 s timeout. A
                            // slow-but-alive worker mid-drain
                            // could still mutate ring state
                            // concurrently; this is accepted
                            // because tmpfs backing bounds drain
                            // time below the sub-timeout.
                            const WORKER_PARK_SUB_TIMEOUT: Duration =
                                Duration::from_secs(1);
                            let worker_sub_deadline =
                                Instant::now() + WORKER_PARK_SUB_TIMEOUT;
                            let mut worker_dropped: bool = false;
                            let mut parked_count: u64 = 0;
                            let mut all_parked = false;
                            // Snapshots WHICH loop-break condition fired
                            // so the post-loop Degraded reason can label
                            // accurately. A bare post-loop
                            // `freeze_coord_kill.load(Acquire)` reads the
                            // CURRENT state, not the cause of the break,
                            // and races a kill setter that lands in the
                            // ns window between the deadline-break and
                            // the post-loop read — mislabelling a true
                            // 30s timeout as a kill. Setting the flag
                            // inside the kill-break arm (and only there)
                            // captures the actual break cause.
                            let mut killed_during_rendezvous = false;
                            loop {
                                if freeze_coord_bsp_done.load(Ordering::Acquire) {
                                    break;
                                }
                                // Short-circuit on kill (e.g. SCHED_EXIT
                                // promoted kill mid-rendezvous). Without
                                // this, the loop burns the full
                                // FREEZE_RENDEZVOUS_TIMEOUT (~30s) before
                                // exiting via the deadline branch, then
                                // the Degraded path reports a misleading
                                // ~30s elapsed_ms when the actual cause
                                // was an external kill landing during the
                                // wait. Breaking here routes through the
                                // existing !all_parked branch — which
                                // re-checks bsp_done (the kill may have
                                // been promoted by the BSP exit) and
                                // emits Degraded with a meaningful
                                // elapsed_ms reflecting how long the
                                // rendezvous actually ran. The bss/
                                // watchpoint state in the Degraded
                                // report still surfaces whatever the
                                // trigger source observed before kill
                                // landed.
                                // Check parked-count BEFORE kill so a
                                // coalesced wake (kill_evt + the final
                                // parked_evt arriving in the same epoll
                                // batch) doesn't misattribute a
                                // successful rendezvous as a Degraded
                                // "rendezvous timed out" — the drain
                                // above can bump parked_count to
                                // expected_parks before the kill check
                                // runs, and we want to record the
                                // ground-truth Captured outcome rather
                                // than emit a misleading Degraded
                                // reason that says vCPUs didn't park
                                // when they did.
                                if parked_count >= expected_parks {
                                    all_parked = true;
                                    // All vCPUs parked → no vCPU
                                    // is reading pvclock. Capture
                                    // kvm_clock NOW so the
                                    // restored value in
                                    // thaw_and_barrier reflects
                                    // the parked-state moment;
                                    // post-thaw guest reads
                                    // resume from this exact
                                    // value with no backward
                                    // jump (see the "deferred"
                                    // comment at freeze_and_dispatch
                                    // entry for the timing
                                    // rationale).
                                    // Skip the GET on the skip-freeze
                                    // fast path: no freeze gate was
                                    // flipped, no SIGRTMIN kick was
                                    // issued, no vCPUs are actually
                                    // parked at a save-relevant
                                    // boundary — and thaw_and_barrier
                                    // will not SET because the
                                    // RefCell stays None.
                                    if !skip_freeze {
                                        // kvm_clock save/restore is x86_64
                                        // only — aarch64 KVM does not
                                        // expose KVM_GET_CLOCK / KVM_SET_CLOCK
                                        // (pvclock is x86-specific). On
                                        // aarch64 the guest timer uses
                                        // CNTVCT_EL0 which the host KVM
                                        // does not freeze, so the
                                        // freeze-rendezvous time-skew
                                        // mitigation doesn't apply.
                                        #[cfg(target_arch = "x86_64")]
                                        {
                                            let captured = kvm::kvm_get_clock_via_raw_fd(
                                                vm_fd_raw_for_freeze,
                                            );
                                            if let Err(ref e) = captured {
                                                warn_kvm_clock_failure("GET_CLOCK", e);
                                            }
                                            *kvm_clock_save_for_freeze.borrow_mut() =
                                                captured.ok();
                                        }
                                        #[cfg(not(target_arch = "x86_64"))]
                                        {
                                            let _ = vm_fd_raw_for_freeze;
                                            let _ = &kvm_clock_save_for_freeze;
                                        }
                                    }
                                    break;
                                }
                                // Kill check follows parked-count by
                                // design — see the "Check parked-count
                                // BEFORE kill" comment above for the
                                // coalesced-wake misattribution that
                                // motivates this ordering.
                                if freeze_coord_kill.load(Ordering::Acquire) {
                                    killed_during_rendezvous = true;
                                    break;
                                }
                                // Worker sub-timeout. Only fires
                                // when the worker was counted in
                                // `expected_parks` (i.e.
                                // `worker_was_running` was true at
                                // pre-pause snapshot) and we have
                                // not yet decremented for it. The
                                // condition `parked_count <
                                // expected_parks` plus the wall-
                                // clock check (`now >=
                                // worker_sub_deadline`) localises
                                // the bookkeeping change to the
                                // path where the worker really did
                                // not ack. The Acquire load on
                                // `paused` synchronises-with any
                                // worker `Release` it might still
                                // perform on a slow path; if the
                                // worker DID park we observe
                                // `paused == true` and DO NOT
                                // decrement (the matching ack will
                                // arrive imminently or has already
                                // arrived in `parked_count`).
                                if !worker_dropped
                                    && worker_was_running
                                    && Instant::now() >= worker_sub_deadline
                                    && freeze_coord_virtio_blk_paused
                                        .as_ref()
                                        .is_some_and(|p| !p.load(Ordering::Acquire))
                                {
                                    // Final paused re-check before
                                    // decrementing. The condition
                                    // above sampled paused==false,
                                    // but a slow-but-alive worker
                                    // could have transitioned
                                    // paused=true between that
                                    // sample and here. Re-loading
                                    // with Acquire pairs with the
                                    // worker's Release store on
                                    // pause(). If the worker DID
                                    // park, skip the drop and let
                                    // the next loop iteration
                                    // observe the matching
                                    // parked_evt ack — never
                                    // double-count by both dropping
                                    // expected_parks and absorbing
                                    // the eventfd write.
                                    if freeze_coord_virtio_blk_paused
                                        .as_ref()
                                        .is_some_and(|p| p.load(Ordering::Acquire))
                                    {
                                        continue;
                                    }
                                    tracing::warn!(
                                        worker_park_sub_timeout_ms =
                                            WORKER_PARK_SUB_TIMEOUT.as_millis() as u64,
                                        parked_count,
                                        expected_parks,
                                        ?rendezvous_timeout,
                                        "freeze-coord: virtio-blk worker did \
                                         not ack park within sub-timeout AND \
                                         `paused` is still false — most \
                                         likely the worker is mid-shutdown \
                                         (signal_worker_stop already cleared \
                                         paused=false on its way out), so no \
                                         live thread will write parked_evt \
                                         for this cycle. Dropping the +1 \
                                         from expected_parks so the \
                                         rendezvous proceeds without waiting \
                                         the full rendezvous timeout for an \
                                         ack that physically cannot arrive."
                                    );
                                    expected_parks =
                                        expected_parks.saturating_sub(1);
                                    worker_dropped = true;
                                    // Re-check the `all_parked`
                                    // predicate immediately so a
                                    // concurrent vCPU ack that just
                                    // pushed parked_count to the
                                    // (now lower) expected value is
                                    // recognised in this iteration
                                    // rather than after another
                                    // poll cycle.
                                    if parked_count >= expected_parks {
                                        all_parked = true;
                                        break;
                                    }
                                }
                                let now = Instant::now();
                                if now > deadline {
                                    // Diagnostic snapshot of every
                                    // parker's flag, computed once on
                                    // timeout for the error log. Hot
                                    // path no longer reads these
                                    // bools per iteration.
                                    let ap_states: Vec<bool> = freeze_coord_ap_parked
                                        .iter()
                                        .map(|p| p.load(Ordering::Acquire))
                                        .collect();
                                    let bsp_p = freeze_coord_bsp_parked.load(Ordering::Acquire);
                                    // Lock-free read via the
                                    // pre-acquired `paused_handle()`
                                    // Arc — avoids taking the device
                                    // mutex on the timeout-diagnostic
                                    // path. Acquire ordering pairs
                                    // with the worker's Release on
                                    // `paused.store(true)` so the
                                    // diagnostic sees a coherent
                                    // worker state.
                                    let blk_parked = freeze_coord_virtio_blk_paused
                                        .as_ref()
                                        .is_none_or(|p| p.load(Ordering::Acquire));
                                    tracing::error!(
                                        ?ap_states,
                                        bsp_parked = bsp_p,
                                        blk_parked,
                                        parked_count,
                                        expected_parks,
                                        "freeze-coord: timed out waiting for vCPUs / worker to park. \
                                         If blk_parked=false, the worker is most likely stuck in a \
                                         slow pread/pwrite against the backing file — verify the \
                                         backing is fast (tmpfs / warm page cache); the vCPU \
                                         thread's blocking budget is bounded by the freeze \
                                         rendezvous timeout, so a backing slow enough to push \
                                         per-request IO past that bound prevents the rendezvous \
                                         from completing. The worker observes PAUSE_TOKEN only \
                                         between blocking syscalls, so a long pread/pwrite delays \
                                         the park-ack until the syscall returns."
                                    );
                                    break;
                                }
                                let remaining_ms = (deadline - now)
                                    .as_millis()
                                    .min(i32::MAX as u128) as i32;
                                let mut pfds = [
                                    libc::pollfd {
                                        fd: freeze_coord_parked_evt.as_raw_fd(),
                                        events: libc::POLLIN,
                                        revents: 0,
                                    },
                                    libc::pollfd {
                                        fd: freeze_coord_kill_evt.as_raw_fd(),
                                        events: libc::POLLIN,
                                        revents: 0,
                                    },
                                    libc::pollfd {
                                        fd: freeze_coord_bsp_done_evt.as_raw_fd(),
                                        events: libc::POLLIN,
                                        revents: 0,
                                    },
                                ];
                                // SAFETY: pfds is a 3-element pollfd
                                // array; nfds matches. Every poll
                                // outcome (ready, timeout, EINTR,
                                // error) loops back to the
                                // countdown predicate at the top.
                                // EINTR from SIGRTMIN is harmless:
                                // the wait simply restarts.
                                unsafe {
                                    libc::poll(
                                        pfds.as_mut_ptr(),
                                        pfds.len() as libc::nfds_t,
                                        remaining_ms,
                                    );
                                }
                                // Drain parked_evt counter once per
                                // wake. Counter mode: a single read
                                // returns the accumulated count and
                                // resets to 0; multiple coalesced
                                // parker writes are absorbed in one
                                // drain. EAGAIN (counter already 0)
                                // is benign — the poll wake may have
                                // come from kill_evt or
                                // bsp_done_evt (those are NOT
                                // drained here; the outer epoll
                                // loop owns them). Saturating add
                                // is defensive — counter mode
                                // eventfd values cap at 2^64 - 2
                                // and physically cannot overflow
                                // a u64 in any realistic VM run.
                                if let Ok(n) = freeze_coord_parked_evt.read() {
                                    parked_count = parked_count.saturating_add(n);
                                }
                            }
                            // Collect per-vCPU register snapshots.
                            // Reads happens-after the rendezvous
                            // Acquire on each vCPU's `parked` flag,
                            // which synchronizes-with the vCPU
                            // thread's Release store after its
                            // capture_vcpu_regs / regs_slot write —
                            // so these Mutex reads see the captured
                            // values even on weakly-ordered
                            // architectures. Index 0 = BSP, 1..N =
                            // APs.
                            let collect_vcpu_regs = ||
                                -> Vec<Option<exit_dispatch::VcpuRegSnapshot>> {
                                let mut regs:
                                    Vec<Option<exit_dispatch::VcpuRegSnapshot>> =
                                    Vec::with_capacity(1 + freeze_coord_ap_regs.len());
                                regs.push(
                                    *freeze_coord_bsp_regs
                                        .lock_unpoisoned(),
                                );
                                for ap in &freeze_coord_ap_regs {
                                    regs.push(
                                        *ap.lock_unpoisoned(),
                                    );
                                }
                                regs
                            };
                            if !all_parked {
                                // V5 ColdOp early-bail: rendezvous-timeout
                                // under ColdOp mode returns
                                // FreezeOutcome::KernelOp(error_reply)
                                // rather than Degraded — Capture-mode's
                                // Degraded carries vcpu_regs +
                                // DegradedFailureDumpReport which the
                                // L8119 ColdOp caller cannot consume
                                // (it expects KernelOp). Build a typed
                                // error reply naming the rendezvous
                                // failure so the wire reply correlates
                                // back to the request via request_id.
                                // Capture mode falls through to the
                                // existing Degraded construction below.
                                if let FreezeMode::ColdOp(req) = &mode {
                                    let reply = crate::vmm::wire::KernelOpReplyPayload {
                                        request_id: req.request_id,
                                        success: false,
                                        reason: format!(
                                            "freeze-coord: rendezvous-timeout under ColdOp mode \
                                             (parked_count={parked_count} expected_parks={expected_parks} \
                                             killed={killed_during_rendezvous} \
                                             elapsed_ms={})",
                                            capture_start.elapsed().as_millis() as u64,
                                        ),
                                        read_values: Vec::new(),
                                    };
                                    break 'capture FreezeOutcome::KernelOp(reply);
                                }
                                // Re-check bsp_done AFTER the rendezvous
                                // loop exits. The loop has a
                                // `if freeze_coord_bsp_done.load(...) { break; }`
                                // guard near its top, so a clean BSP
                                // exit mid-rendezvous breaks out with
                                // `all_parked = false` even though the
                                // cause was NOT a rendezvous timeout.
                                // Without this re-check, the Degraded
                                // path below would misattribute the
                                // clean shutdown as
                                // "rendezvous timed out at <Nms>;
                                // 0/N vCPUs parked" — confusing to the
                                // operator because the run is
                                // terminating normally. The pre-cycle
                                // `skip_freeze` snapshot (taken at
                                // cycle entry) is stale for this case:
                                // bsp_done was false at entry but
                                // flipped during the wait.
                                let bsp_done_late =
                                    freeze_coord_bsp_done.load(Ordering::Acquire);
                                if skip_freeze || bsp_done_late {
                                    tracing::info!(
                                        skip_freeze,
                                        bsp_done_late,
                                        "freeze-coord: skip_freeze — vCPUs \
                                         exited, proceeding with dump on \
                                         quiesced memory"
                                    );
                                } else {
                                    // Per the no-silent-drops policy:
                                    // the trigger fired (the watchpoint hit
                                    // and/or the BPF `.bss` latch flipped),
                                    // so the operator must see something —
                                    // a debug! log alone hides the
                                    // attempted-but-aborted state. Emit a
                                    // [`crate::monitor::dump::SCHEMA_DEGRADED`]
                                    // report carrying the partial
                                    // `vcpu_regs` (the regs slots for vCPUs
                                    // that DID park before the timeout,
                                    // `None` for the ones that stalled),
                                    // the live watchpoint + bss-latch
                                    // state, the `exit_kind` value when
                                    // the cached PA still resolves, and
                                    // the elapsed milliseconds the
                                    // coordinator spent trying.
                                    let vcpu_regs = collect_vcpu_regs();
                                    let watchpoint_hit = freeze_coord_watchpoint
                                        .hit
                                        .load(Ordering::Acquire);
                                    let bss_state = bss_read_state(
                                        freeze_coord_mem.as_deref(),
                                        cached_bss_pa,
                                    );
                                    let exit_kind = match (
                                        cached_exit_kind_pa,
                                        freeze_coord_mem.as_deref(),
                                    ) {
                                        (Some(pa), Some(mem)) => {
                                            Some(mem.read_u32(pa, 0))
                                        }
                                        _ => None,
                                    };
                                    let elapsed_ms =
                                        capture_start.elapsed().as_millis() as u64;
                                    let total = expected_parks;
                                    let stuck = total.saturating_sub(parked_count);
                                    // Distinguish the kill-during-rendezvous
                                    // case from a true 30s timeout via the
                                    // in-loop `killed_during_rendezvous`
                                    // snapshot declared before the loop.
                                    // A bare post-loop `freeze_coord_kill.
                                    // load(Acquire)` reads CURRENT state,
                                    // not the cause of the break — a kill
                                    // setter (SCHED_EXIT propagation,
                                    // watchdog hard-deadline expiry,
                                    // panic-hook) racing into the ns
                                    // window between the deadline-break
                                    // and the read mislabels a true 30s
                                    // timeout as a kill. The in-loop flag
                                    // captures the actual break cause:
                                    // set inside the kill-check arm
                                    // before the break, false otherwise.
                                    // The kill-aware path lands here when
                                    // kill flipped during the wait, with
                                    // elapsed_ms typically in single-
                                    // digit ms, where "timed out" would
                                    // be contradictory.
                                    let killed = killed_during_rendezvous;
                                    let reason_prefix = if killed {
                                        crate::monitor::dump::REASON_DEGRADED_KILL_DURING_RENDEZVOUS
                                    } else {
                                        crate::monitor::dump::REASON_DEGRADED_RENDEZVOUS_TIMEOUT
                                    };
                                    let reason = format!(
                                        "{reason_prefix}: {elapsed_ms}ms; {parked_count}/{total} vCPUs parked",
                                    );
                                    // Branch the warn body too so the
                                    // tracing-stream consumer reads the
                                    // same kill-vs-timeout disambiguation
                                    // as the on-disk Degraded JSON.
                                    // Without this, an operator scrolling
                                    // traces sees "rendezvous timed out"
                                    // with elapsed_ms=5ms on every kill
                                    // while the JSON correctly carries
                                    // the kill-aware reason — two
                                    // channels disagreeing on the same
                                    // event.
                                    let warn_summary = if killed {
                                        "freeze-coord: dump degraded — vCPU rendezvous aborted by external kill"
                                    } else {
                                        "freeze-coord: dump degraded — vCPU rendezvous timed out"
                                    };
                                    tracing::warn!(
                                        elapsed_ms,
                                        parked_count,
                                        expected_parks = total,
                                        stuck,
                                        killed,
                                        watchpoint_hit,
                                        bss_latch_state = bss_state_label(bss_state),
                                        exit_kind,
                                        "{warn_summary}"
                                    );
                                    let degraded = crate::monitor::dump::DegradedFailureDumpReport {
                                        schema: crate::monitor::dump::SCHEMA_DEGRADED.to_string(),
                                        reason,
                                        vcpu_regs,
                                        watchpoint_hit,
                                        bss_latch_state:
                                            bss_state_label(bss_state).to_string(),
                                        exit_kind,
                                        elapsed_ms,
                                    };
                                    break 'capture FreezeOutcome::Degraded(
                                        Box::new(degraded),
                                    );
                                }
                            }
                            // V4 split: ColdOp dispatch fires here, after
                            // the park-ack rendezvous + kvm_clock save +
                            // V5 !all_parked Degraded handling, and BEFORE
                            // the Capture-mode exit-kind gate + dump
                            // pipeline below. Returns
                            // FreezeOutcome::KernelOp(reply) directly so
                            // the L8119 stub-replacement call site can
                            // frame the wire reply without traversing the
                            // dump path. Capture mode falls through.
                            if let FreezeMode::ColdOp(req) = &mode {
                                let reply = if let Some(owned) = owned_accessor.as_ref() {
                                    kernel_op_dispatch::dispatch_kernel_op_batch(
                                        owned.guest_kernel(),
                                        dump_btf.as_ref(),
                                        coord_kaslr_offset(),
                                        req,
                                    )
                                } else {
                                    // Caller at L8119 should gate on
                                    // owned_accessor.is_some() before
                                    // invoking; defensive error frame if
                                    // it slipped through unset. No panic
                                    // — the request_id round-trip lets
                                    // the caller correlate the failure.
                                    crate::vmm::wire::KernelOpReplyPayload {
                                        request_id: req.request_id,
                                        success: false,
                                        reason:
                                            "freeze-coord: owned_accessor not yet initialised; \
                                             ColdOp dispatch dropped"
                                                .to_string(),
                                        read_values: Vec::new(),
                                    }
                                };
                                break 'capture FreezeOutcome::KernelOp(reply);
                            }
                            // Exit-kind gate. The hardware watchpoint
                            // catches every write to
                            // `*scx_root->exit_kind`, including
                            // transient writes during init/teardown
                            // that the kernel sets to
                            // `SCX_EXIT_NONE` (0) or `SCX_EXIT_DONE`
                            // (1). Without this gate every clean
                            // scheduler shutdown produces a bogus
                            // failure dump. Read the live `exit_kind`
                            // value through the same direct-mapping +
                            // page-walk translation
                            // `read_scx_sched_state` uses; gate on
                            // `kind >= SCX_EXIT_ERROR (= 1024)` per
                            // `kernel/sched/ext_internal.h::scx_exit_kind`.
                            //
                            // Required prerequisites all flow from
                            // the same set the watchpoint resolution
                            // earlier in the loop validated when it
                            // published the `request_kva`:
                            //   - request_kva non-zero (resolved
                            //     `*scx_root + exit_kind_offset`)
                            //   - owned_accessor for cr3_pa /
                            //     page_offset / l5
                            //   - freeze_coord_mem for read_u32
                            // Any prereq absence means the watchpoint
                            // could not have armed (publish path
                            // requires the same handles), so a
                            // gated call without prerequisites is a
                            // logic bug — log + dump anyway rather
                            // than silently swallow the trigger.
                            if gate_on_exit_kind {
                                let exit_kind_kva = freeze_coord_watchpoint
                                    .request_kva
                                    .load(Ordering::Acquire);
                                let gate_decision = match (
                                    exit_kind_kva,
                                    owned_accessor.as_ref(),
                                    freeze_coord_mem.as_deref(),
                                ) {
                                    (0, _, _) | (_, None, _) | (_, _, None) => {
                                        tracing::warn!(
                                            exit_kind_kva = format_args!(
                                                "{:#x}",
                                                exit_kind_kva
                                            ),
                                            owned_accessor_present =
                                                owned_accessor.is_some(),
                                            mem_present = freeze_coord_mem.is_some(),
                                            "freeze-coord: exit_kind gate \
                                             prerequisites missing — proceeding \
                                             with dump (watchpoint should not \
                                             have armed without these)"
                                        );
                                        // Treat missing prereqs as
                                        // "do not gate" so a bona fide
                                        // dump still emits.
                                        true
                                    }
                                    (kva, Some(owned), Some(mem)) => {
                                        let kernel = owned.guest_kernel();
                                        let walk = kernel.walk_context();
                                        let translate_result =
                                            crate::monitor::idr::translate_any_kva(
                                                mem,
                                                walk.cr3_pa,
                                                walk.page_offset,
                                                kva,
                                                walk.l5,
                                                walk.tcr_el1,
                                            );
                                        // The gate decision is the pure
                                        // `exit_kind_warrants_dump` (unit-tested
                                        // truth table over Some(error)/Some(clean)
                                        // /None); the per-branch tracing below
                                        // explains the verdict. `None` =
                                        // translate failed (slab page freed
                                        // mid-teardown) → does not warrant a
                                        // dump.
                                        match translate_result {
                                            Some(pa) => {
                                                let kind = mem.read_u32(pa, 0);
                                                let warrants =
                                                    exit_kind_warrants_dump(Some(kind));
                                                if warrants {
                                                    tracing::debug!(
                                                        kind,
                                                        "freeze-coord: \
                                                         exit_kind gate passed \
                                                         (kind >= 1024)"
                                                    );
                                                } else {
                                                    tracing::info!(
                                                        kind,
                                                        exit_kind_kva =
                                                            format_args!("{:#x}", kva),
                                                        "freeze-coord: \
                                                         exit_kind gate \
                                                         suppressed dump \
                                                         (kind < 1024 = clean \
                                                         shutdown / non-error \
                                                         transition)"
                                                    );
                                                }
                                                warrants
                                            }
                                            None => {
                                                // KVA was published but no
                                                // longer translates: most
                                                // likely the slab page that
                                                // held `*scx_root` was
                                                // freed during teardown.
                                                // Suppress the dump — there
                                                // is no scheduler state to
                                                // capture anyway.
                                                tracing::info!(
                                                    exit_kind_kva =
                                                        format_args!("{:#x}", kva),
                                                    "freeze-coord: exit_kind \
                                                     gate translate failed \
                                                     (scheduler likely torn \
                                                     down) — suppressing dump"
                                                );
                                                exit_kind_warrants_dump(None)
                                            }
                                        }
                                    }
                                };
                                if !gate_decision {
                                    // Cross-reference with the BPF probe
                                    // `.bss` latch before suppressing.
                                    // probe.bpf.c sets
                                    // `ktstr_err_exit_detected = 1` via
                                    // a `__sync_val_compare_and_swap`
                                    // 0→1 with release semantics over
                                    // the preceding `ktstr_exit_kind_snap`
                                    // and `ktstr_exit_*` stores — once
                                    // observed Triggered, the kernel
                                    // recorded an error-class exit even
                                    // if the live `*scx_root->exit_kind`
                                    // currently reads as a transient
                                    // clean value (the slab page was
                                    // freed mid-teardown, or the
                                    // kernel overwrote with NONE/DONE
                                    // during scx_ops_disable_workfn
                                    // teardown after firing
                                    // tp_btf/sched_ext_exit). In that
                                    // race window the gate would
                                    // silently drop a real failure
                                    // dump; the BPF latch is the
                                    // historical authority and wins.
                                    let bss_state = bss_read_state(
                                        freeze_coord_mem.as_deref(),
                                        cached_bss_pa,
                                    );
                                    // The cross-reference Triggered
                                    // override is only safe when the
                                    // cached `.bss` PA has not been
                                    // observed OOB in this run. Once
                                    // OOB fires (probe map freed mid-
                                    // run, vmalloc page recycled),
                                    // any subsequent non-zero read on
                                    // the cached PA could come from
                                    // page-allocator reuse with
                                    // unrelated data — synthesising
                                    // a phantom dump. Trust the gate
                                    // suppression in that case rather
                                    // than overriding on potentially
                                    // recycled bytes.
                                    //
                                    // Probe-lifecycle assumption: this
                                    // defense assumes probe BPF map
                                    // unload is bracketed by an
                                    // `OutOfBounds` observation that
                                    // latches `bss_oob_warn_logged`
                                    // BEFORE the next gate check fires.
                                    // Concurrent unload + slab recycle
                                    // within a single iteration body
                                    // (probe unloads AND the page is
                                    // re-allocated AND the gate fires
                                    // between two `bss_read_state(...)`
                                    // calls) is unreachable under the
                                    // current probe lifecycle: the
                                    // probe runs in the guest kernel
                                    // and lives for the guest VM's
                                    // lifetime; the freeze coordinator
                                    // joins inside `run_vm` BEFORE the
                                    // guest VM drops (see the
                                    // watchpoint-invalidation comment
                                    // near `run.vm` teardown), so the
                                    // probe's `.bss` page remains
                                    // valid for the entire coord
                                    // observation window. Revisit if
                                    // probe ownership moves to a
                                    // separate lifecycle (e.g. per-
                                    // test hot-reload that unloads +
                                    // reloads the probe mid-VM-run) —
                                    // that would require the latch
                                    // to fire per (unload, reload)
                                    // cycle rather than once per VM.
                                    if bss_latch_rescues_suppressed_dump(bss_state, bss_oob_warn_logged) {
                                        tracing::warn!(
                                            "freeze-coord: exit_kind gate \
                                             would suppress dump but BPF \
                                             `.bss` latch reports error \
                                             exit — proceeding with dump \
                                             (latch is historical authority)"
                                        );
                                    } else {
                                        break 'capture FreezeOutcome::Suppressed;
                                    }
                                }
                            }
                            if let Some(owned) = owned_accessor.as_ref()
                                && let Some(ref btf) = dump_btf
                            {
                                // Build the prog-runtime capture
                                // when both prerequisites are ready.
                                // Each is independent of the other
                                // (the accessor needs prog_idr +
                                // BTF; the offsets need __per_cpu_offset),
                                // so a partial setup yields no
                                // capture rather than a half-correct
                                // one — `dump_state` then writes an
                                // empty `prog_runtime_stats` vec
                                // alongside the full map render.
                                let prog_acc_borrow =
                                    owned_prog_accessor.as_ref().map(|o| o.as_accessor());
                                let prog_capture = match (
                                    prog_acc_borrow.as_ref(),
                                    prog_per_cpu_offsets.as_deref(),
                                ) {
                                    (Some(acc), Some(offsets)) => {
                                        Some(crate::monitor::dump::ProgRuntimeCapture {
                                            accessor: acc,
                                            per_cpu_offsets: offsets,
                                        })
                                    }
                                    _ => None,
                                };
                                let map_accessor = owned.as_accessor();
                                // Bind kernel once for the whole dump
                                // block. Pre-fix this called
                                // owned.guest_kernel() three times
                                // (scx_walker_capture, task_enrichment_capture,
                                // cpu_time_capture). The accessor is a
                                // trivial &-return but the repetition
                                // obscured ownership — every consumer
                                // wants the same kernel handle.
                                let dump_kernel = owned.guest_kernel();
                                // Pre-collect register snapshots: needed
                                // for both the report's vcpu_regs field
                                // AND the per-task enrichment running_pc
                                // mapping (walking rq->scx.curr to the
                                // corresponding vCPU's IP). Capturing
                                // here before the dump means the same
                                // snapshot drives every consumer.
                                let vcpu_regs = collect_vcpu_regs();
                                // SCX walker owned data — backs the
                                // borrow-only `ScxWalkerCapture`. The
                                // capture runs while every vCPU is
                                // paused at the freeze rendezvous, so
                                // each phase emits a tracing::debug
                                // duration line so operators can
                                // budget against the watchdog timeout.
                                let scx_build_t0 = std::time::Instant::now();
                                // Pass the virt-KASLR offset from
                                // the shared Arc (populated by BSP
                                // MSR_LSTAR + guest-channel KERN_ADDRS
                                // publishers). 0 sentinel means
                                // KASLR-off-or-not-yet-published; the
                                // `per_cpu_kva` helper inside
                                // `build` handles both as a no-op
                                // slide. See `coord_kaslr_offset`
                                // closure above for the load + bias
                                // unwrap.
                                let scx_owned = crate::vmm::capture_scx::build(
                                    owned,
                                    dump_scx_walker_offsets.as_ref(),
                                    dump_cpu_time_symbols.as_ref(),
                                    prog_per_cpu_offsets.as_deref(),
                                    coord_kaslr_offset(),
                                );
                                tracing::debug!(
                                    elapsed_us = scx_build_t0.elapsed().as_micros() as u64,
                                    populated = scx_owned.is_some(),
                                    "freeze-coord: capture_scx::build"
                                );
                                let scx_walker_capture = scx_owned.as_ref().and_then(|so| {
                                    let offsets = dump_scx_walker_offsets.as_ref()?;
                                    Some(crate::monitor::dump::ScxWalkerCapture {
                                        kernel: dump_kernel,
                                        offsets,
                                        scx_root_kva: so.scx_root_kva,
                                        rq_kvas: &so.rq_kvas,
                                        rq_pas: &so.rq_pas,
                                        per_cpu_offsets: prog_per_cpu_offsets
                                            .as_deref()
                                            .unwrap_or(&[]),
                                        nr_nodes: freeze_coord_num_nodes,
                                    })
                                });
                                // Task-enrichment owned data — backs the
                                // borrow-only `TaskEnrichmentCapture`.
                                let task_build_t0 = std::time::Instant::now();
                                let task_owned = crate::vmm::capture_tasks::build(
                                    owned,
                                    scx_owned.as_ref(),
                                    dump_scx_walker_offsets.as_ref(),
                                    dump_task_enrichment_offsets.as_ref(),
                                    &vcpu_regs,
                                );
                                tracing::debug!(
                                    elapsed_us = task_build_t0.elapsed().as_micros() as u64,
                                    populated = task_owned.is_some(),
                                    tasks = task_owned.as_ref().map(|t| t.tasks.len()).unwrap_or(0),
                                    "freeze-coord: capture_tasks::build"
                                );
                                let task_enrichment_capture = task_owned.as_ref().and_then(|to| {
                                    let te_offsets = dump_task_enrichment_offsets.as_ref()?;
                                    Some(crate::monitor::dump::TaskEnrichmentCapture {
                                        kernel: dump_kernel,
                                        offsets: te_offsets,
                                        sched_classes: &to.sched_classes,
                                        lock_slowpaths: &to.lock_slowpaths,
                                        tasks: &to.tasks,
                                    })
                                });
                                // Per-CPU CPU-time / softirq / IRQ
                                // capture context. All four prereqs
                                // must be present to fire: BTF
                                // offsets (resolved at coord start),
                                // KernelSymbols carrying the
                                // `kernel_cpustat`/`kstat`
                                // per-CPU symbol KVAs (also at coord
                                // start), the per-CPU offset array
                                // (lazy-resolved alongside the prog
                                // accessor), and the freeze-coord
                                // GuestMem. Either of `kernel_cpustat`
                                // or `kstat` symbol absent makes the
                                // capture useless — both backing
                                // structs are needed for the dump's
                                // narrative (`tick_cpu_sched` is
                                // optional and feeds only the
                                // iowait_sleeptime field). The
                                // `tick_cpu_sched_kva` is forwarded
                                // to dump/mod.rs as Option so the per-CPU
                                // walker can skip iowait_sleeptime
                                // independently per CPU.
                                let cpu_time_capture = match (
                                    freeze_coord_mem.as_deref(),
                                    dump_cpu_time_offsets.as_ref(),
                                    dump_cpu_time_symbols.as_ref(),
                                    prog_per_cpu_offsets.as_deref(),
                                ) {
                                    (Some(mem), Some(offsets), Some(syms), Some(pcpu)) => {
                                        match (syms.kernel_cpustat, syms.kstat) {
                                            (Some(kcpustat_kva), Some(kstat_kva)) => {
                                                let page_offset = dump_kernel.page_offset();
                                                // Virt-KASLR offset published
                                                // by the BSP MSR_LSTAR path
                                                // (`run_bsp_loop` →
                                                // `msr_kaslr::read_and_derive`,
                                                // x86_64) or the guest-channel
                                                // KERN_ADDRS path
                                                // (`dispatch.rs`, both arches).
                                                // `+1` bias on the shared Arc;
                                                // `saturating_sub(1)` folds the
                                                // "no publisher yet" sentinel
                                                // and the "KASLR off, real
                                                // offset = 0" case to the same
                                                // observable 0 — both produce
                                                // correct per-CPU template
                                                // arithmetic in
                                                // `monitor::symbols::per_cpu_kva`.
                                                let kaslr_offset = kern_virt_kaslr
                                                    .load(
                                                        std::sync::atomic::Ordering::Acquire,
                                                    )
                                                    .saturating_sub(1);
                                                Some(crate::monitor::dump::CpuTimeCapture {
                                                    mem,
                                                    offsets,
                                                    kernel_cpustat_kva: kcpustat_kva,
                                                    kstat_kva,
                                                    tick_cpu_sched_kva: syms.tick_cpu_sched,
                                                    per_cpu_offsets: pcpu,
                                                    page_offset,
                                                    kaslr_offset,
                                                })
                                            }
                                            _ => None,
                                        }
                                    }
                                    _ => None,
                                };
                                // Per-cgroup PSI-irq capture (Phase A).
                                // Some only when guest memory, the cgroup +
                                // psi_group offsets, and the cgrp_dfl_root
                                // symbol all resolve. Pre-translate the
                                // hierarchy root cgroup (cgrp_dfl_root +
                                // offsetof(cgroup_root, cgrp)) here: text PA for
                                // the entry read, KASLR-slid runtime KVA for the
                                // children-list anchor compare (the scx_walker
                                // runtime-head discipline). Every descendant is
                                // direct-mapped, translated inside the walk.
                                let cgroup_psi_capture = match (
                                    freeze_coord_mem.as_deref(),
                                    dump_cgroup_offsets.as_ref(),
                                    dump_cgroup_psi_offsets.as_ref(),
                                    dump_cpu_time_symbols.as_ref().and_then(|s| s.cgrp_dfl_root),
                                ) {
                                    (
                                        Some(mem),
                                        Some(cgroup_offsets),
                                        Some(psi_offsets),
                                        Some(cgrp_dfl_root_kva),
                                    ) => {
                                        let page_offset = dump_kernel.page_offset();
                                        let kaslr_offset = kern_virt_kaslr
                                            .load(std::sync::atomic::Ordering::Acquire)
                                            .saturating_sub(1);
                                        let link_root_kva = cgrp_dfl_root_kva
                                            .wrapping_add(cgroup_offsets.cgroup_root_cgrp as u64);
                                        let root_cgroup_pa =
                                            dump_kernel.text_kva_to_pa(link_root_kva);
                                        let root_cgroup_kva =
                                            crate::monitor::symbols::slid_kernel_kva(
                                                link_root_kva,
                                                kaslr_offset,
                                            );
                                        Some(crate::monitor::dump::CgroupPsiCapture {
                                            mem,
                                            cgroup_offsets,
                                            psi_offsets,
                                            root_cgroup_kva,
                                            root_cgroup_pa,
                                            workload_root_path: &freeze_coord_workload_root,
                                            page_offset,
                                        })
                                    }
                                    _ => None,
                                };
                                // Force the lazy cast-analysis on this
                                // dump's host coordinator thread (NOT a
                                // vCPU thread — vCPUs are paused at the
                                // freeze rendezvous). Bind the resulting
                                // `Option<Arc<CastAnalysisOutput>>`
                                // BEFORE the `DumpContext` literal so
                                // the inner `Arc` outlives
                                // `dump_state`'s borrow. First dump does
                                // the work; subsequent dumps in the same
                                // VM hit the `OnceLock` and return
                                // immediately.
                                //
                                // The full output carries both the cast
                                // map AND the cross-BTF Fwd resolution
                                // index (every parsed embedded BPF
                                // object's BTF + a name-keyed lookup).
                                // Both halves are threaded into
                                // `DumpContext` so the renderer's chase
                                // paths can resolve `BTF_KIND_FWD`
                                // pointees that live in a sibling
                                // object's BTF — the typical multi-
                                // `.bpf.objs` shape where one object
                                // declares `struct foo;` (forward) and
                                // another defines the body.
                                let cast_analysis = freeze_coord_cast_map.get_full();
                                // Single-object only: cast_lookup consults this one
                                // map by (parent_type_id, offset). Multi-object
                                // schedulers would need per-btf_kva selection here;
                                // build_cast_analysis_from_bytes logs a loud error!
                                // when >1 object carries casts (per-object BTF
                                // id-spaces collide). See its "Single-object only" note.
                                let cast_map_ref = cast_analysis
                                    .as_ref()
                                    .and_then(|out| out.cast_maps.first().map(|m| m.as_ref()));
                                let cross_btf_fwd_index_owned = cast_analysis.as_ref().map(|out| {
                                    crate::monitor::dump::CrossBtfFwdIndex {
                                        btfs: &out.btfs,
                                        fwd_index: &out.fwd_index,
                                    }
                                });
                                let dump_state_t0 = std::time::Instant::now();
                                let mut report = crate::monitor::dump::dump_state(
                                    crate::monitor::dump::DumpContext {
                                        accessor: &map_accessor,
                                        btf,
                                        num_cpus: freeze_coord_num_cpus,
                                        arena_offsets: dump_arena_offsets.as_ref(),
                                        prog_capture: prog_capture.as_ref(),
                                        cpu_time_capture: cpu_time_capture.as_ref(),
                                        cgroup_psi_capture: cgroup_psi_capture.as_ref(),
                                        task_enrichment_capture: task_enrichment_capture
                                            .as_ref(),
                                        // Per-sample SCX_EV_* event counter
                                        // timeline. Today's freeze coordinator
                                        // does not share the monitor sampler's
                                        // accumulated samples vec — that
                                        // would require an Arc<Mutex<...>>
                                        // hand-off plumbed through
                                        // `start_monitor` / `monitor_loop`.
                                        // Leaving None preserves current
                                        // behavior (event_counter_timeline
                                        // stays empty in the failure dump
                                        // JSON); the timeline is still
                                        // recorded on `VmResult.monitor.samples`
                                        // for the post-run sidecar consumer.
                                        // Wiring the share would populate
                                        // this with
                                        // `Some(EventCounterCapture { samples })`.
                                        event_counter_capture: None,
                                        scx_walker_capture: scx_walker_capture
                                            .as_ref(),
                                        // Per-vCPU PMU capture is shared
                                        // with the monitor sampler via the
                                        // `freeze_coord_perf_capture` Arc;
                                        // dump_state reads it once at the
                                        // freeze instant into
                                        // `vcpu_perf_at_freeze`. None when
                                        // perf was unavailable on this host
                                        // (paranoid > 2 / no CAP_PERFMON /
                                        // hardware lacks counters).
                                        perf_capture: (*freeze_coord_perf_capture).as_ref(),
                                        deadline: capture_deadline,
                                        // The bound `cast_map_ref` is
                                        // `Option<&CastMap>` derived from
                                        // the full output's inner `Arc`.
                                        // The full output keeps the
                                        // `CastMap` alive for the
                                        // duration of this `dump_state`
                                        // call.
                                        cast_map: cast_map_ref,
                                        // Cross-BTF Fwd resolution
                                        // context — see
                                        // [`DumpContext::cross_btf_fwd_index`].
                                        cross_btf_fwd_index: cross_btf_fwd_index_owned,
                                        alloc_size_types: cast_analysis
                                            .as_ref()
                                            .map(|o| o.alloc_size_types.as_slice())
                                            .unwrap_or(&[]),
                                    },
                                );
                                tracing::debug!(
                                    elapsed_us = dump_state_t0.elapsed().as_micros() as u64,
                                    maps = report.maps.len(),
                                    "freeze-coord: dump_state"
                                );
                                report.vcpu_regs = vcpu_regs;
                                // Per-node NUMA stats — overwrite the
                                // empty default `dump_state` writes when
                                // the producer lands a non-empty Vec.
                                let numa_build_t0 = std::time::Instant::now();
                                let numa_stats = crate::vmm::capture_numa::build(
                                    owned,
                                    dump_numa_offsets.as_ref(),
                                    dump_cpu_time_symbols.as_ref(),
                                    freeze_coord_num_nodes,
                                );
                                tracing::debug!(
                                    elapsed_us = numa_build_t0.elapsed().as_micros() as u64,
                                    nodes = numa_stats.as_ref().map(|s| s.len()).unwrap_or(0),
                                    "freeze-coord: capture_numa::build"
                                );
                                if let Some(stats) = numa_stats
                                    && !stats.is_empty()
                                {
                                    report.per_node_numa = stats;
                                    report.per_node_numa_unavailable = None;
                                }
                                FreezeOutcome::Captured(report, capture_start)
                            } else {
                                // Partial dump: vcpu_regs only.
                                //
                                // `per_cpu_time` stays `Vec::new()` because
                                // the cpu_time walker's prereqs are
                                // strictly tied to BOTH of the partial-
                                // path triggers (cf. cpu_time_capture
                                // construction at L6615):
                                //   - if we're here because
                                //     `owned_accessor.is_none()`,
                                //     `prog_per_cpu_offsets` is also None
                                //     (its lazy-init at L4229 gates on
                                //     owned_accessor.is_some() so
                                //     `phys_base` is the real KASLR
                                //     displacement, not the bootstrap 0
                                //     default — without that gate the
                                //     init would cache garbage permanently
                                //     under KASLR-on guests).
                                //   - if we're here because `dump_btf.is_none()`,
                                //     `dump_cpu_time_offsets` (BTF-derived
                                //     at L2989) is also None.
                                // Either way the cpu_time_capture 4-prereq
                                // match at L6615 falls to `None` and the
                                // walker can't run. Hardcoded empty here
                                // matches that result without re-walking
                                // the prereqs at this site.
                                let report = crate::monitor::dump::FailureDumpReport {
                                    schema: crate::monitor::dump::SCHEMA_SINGLE.to_string(),
                                    active_map_kvas: Vec::new(),
                                    maps: Vec::new(),
                                    vcpu_regs: collect_vcpu_regs(),
                                    sdt_allocations: Vec::new(),
                                    prog_runtime_stats: Vec::new(),
                                    prog_runtime_stats_unavailable: Some(
                                        "dump prerequisites unavailable".to_string(),
                                    ),
                                    per_cpu_time: Vec::new(),
                                    cgroup_psi: Vec::new(),
                                    task_enrichments: Vec::new(),
                                    task_enrichments_unavailable: Some(
                                        "dump prerequisites unavailable".to_string(),
                                    ),
                                    event_counter_timeline: Vec::new(),
                                    rq_scx_states: Vec::new(),
                                    dsq_states: Vec::new(),
                                    scx_sched_state: None,
                                    scx_walker_unavailable: Some(
                                        "dump prerequisites unavailable".to_string(),
                                    ),
                                    vcpu_perf_at_freeze: Vec::new(),
                                    per_node_numa: Vec::new(),
                                    per_node_numa_unavailable: Some(
                                        "dump prerequisites unavailable".to_string(),
                                    ),
                                    dump_truncated_at_us: None,
                                    // Partial dump never enters the per-map
                                    // render loop, so no map was dropped by
                                    // deadline truncation.
                                    maps_truncated: 0,
                                    probe_counters: None,
                                    scx_static_ranges: Default::default(),
                                    is_placeholder: false,
                                    sdt_alloc_unavailable: Some(
                                        "dump prerequisites unavailable".to_string(),
                                    ),
                                    // Partial dump path: no maps walked,
                                    // so the struct_ops/scx_root matcher
                                    // cannot identify an active obj.
                                    active_obj_name: None,
                                };
                                tracing::warn!(
                                    owned_accessor = owned_accessor.is_some(),
                                    dump_btf = dump_btf.is_some(),
                                    "freeze-coord: dump prerequisites unavailable; \
                                     emitting partial report with vcpu_regs only"
                                );
                                FreezeOutcome::Captured(report, capture_start)
                            }
                        } // end 'capture labeled block (the closure
                          // returns this block's value; the caller
                          // is responsible for invoking
                          // `thaw_and_barrier` AFTER any
                          // while-frozen work it needs to perform
                          // — the late-trigger backstop reads
                          // guest memory while quiesced, so the
                          // thaw cannot be unconditional inside
                          // the closure).
                        };
                    // Unified thaw + post-thaw barrier. Called by
                    // every site after `freeze_and_dispatch` returns
                    // (and after any while-frozen work the site
                    // needs). Replaces the per-site thaw block that
                    // previously diverged on which ordering rules
                    // fired. Resumes the virtio-blk worker FIRST so
                    // its `paused.load(Acquire)` poll exits before
                    // the freeze flag clears (worker polls `paused`,
                    // vCPUs poll `freeze`; resume-then-freeze=false
                    // means both wake paths land cleanly), then
                    // clears `freeze` and writes `thaw_evt` so every
                    // parked vCPU's poll wakes within microseconds.
                    //
                    // Post-thaw barrier — wait for every parker to
                    // clear its flag (vCPUs run their trailing
                    // `parked.store(false)` in handle_freeze AFTER
                    // observing freeze=false; the worker clears
                    // `paused` on resume()). Cycle N+1's
                    // rendezvous loop assumes all parked flags are
                    // false at entry; without this barrier a
                    // still-mid-thaw vCPU's `parked=true` would
                    // either be cleared by a force-clear and
                    // deadlock the cycle (legitimate parked=true
                    // for cycle N+1 never re-stored), OR be
                    // race-observed as a false positive (vCPU never
                    // parked for cycle N+1).
                    //
                    // No dedicated unparked_evt fd exists
                    // (handle_freeze does not write any eventfd on
                    // its trailing `parked.store(false)`); the
                    // barrier polls the AtomicBools at a 10 ms
                    // cadence — the same backstop handle_freeze
                    // uses for its `freeze.load(Acquire)` re-check
                    // when the thaw_evt poll's level fans across
                    // multiple parkers. EINTR / partial wakes are
                    // harmless; the predicate re-evaluates each
                    // iteration.
                    //
                    // Finally drain `parked_evt` so cycle N+1's
                    // countdown latch starts at 0.
                    let thaw_and_barrier = || {
                        // Always unfreeze + thaw even on teardown
                        // so vCPUs don't stay parked.
                        if let Some(ref blk) = freeze_coord_virtio_blk {
                            blk.lock().resume();
                        }
                        // Restore the pre-freeze kvm_clock BEFORE
                        // flipping freeze=false. Kernel ground
                        // truth: KVM_SET_CLOCK queues per-vCPU
                        // KVM_REQ_CLOCK_UPDATE; if the thaw
                        // signal fires first, a parked vCPU may
                        // re-enter guest mode before the request
                        // is queued and read the OLD pvclock page
                        // on its first guest instruction. The
                        // take() + flags=0 pattern matches the
                        // cloud-hypervisor precedent
                        // (`src/vmm/src/vm.rs` reset_flags) —
                        // leaving KVM_CLOCK_REALTIME in flags
                        // causes a realtime adjustment that
                        // double-counts elapsed time per the
                        // existing boot-time pattern in this
                        // crate. SET_CLOCK failure is observable
                        // as a single warn log; guest sees
                        // freeze duration as elapsed kvm_clock
                        // if the restore fails.
                        // kvm_clock SET — x86_64-only, matching the
                        // GET site above. Aarch64 path's
                        // `kvm_clock_save_for_freeze` stays empty
                        // (the GET-arm is cfg-skipped); the take()
                        // would be None anyway.
                        #[cfg(target_arch = "x86_64")]
                        if let Some(mut clock) =
                            kvm_clock_save_for_freeze.borrow_mut().take()
                        {
                            clock.flags = 0;
                            if let Err(e) = kvm::kvm_set_clock_via_raw_fd(
                                vm_fd_raw_for_freeze,
                                &clock,
                            ) {
                                warn_kvm_clock_failure("SET_CLOCK", &e);
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            let _ = vm_fd_raw_for_freeze;
                            let _ = &kvm_clock_save_for_freeze;
                        }
                        freeze_coord_freeze.store(false, Ordering::Release);
                        let _ = freeze_coord_thaw_evt.write(1);
                        if freeze_coord_bsp_done.load(Ordering::Acquire) {
                            return;
                        }

                        let post_thaw_deadline = Instant::now() + rendezvous_timeout;
                        loop {
                            if freeze_coord_kill.load(Ordering::Acquire)
                                || freeze_coord_bsp_done.load(Ordering::Acquire)
                            {
                                break;
                            }
                            let aps_unparked = freeze_coord_ap_parked
                                .iter()
                                .all(|p| !p.load(Ordering::Acquire));
                            let bsp_unparked = !freeze_coord_bsp_parked
                                .load(Ordering::Acquire);
                            // Lock-free read via the pre-acquired
                            // `paused_handle()` Arc — avoids
                            // taking the device mutex inside the
                            // post-thaw barrier hot loop. Acquire
                            // ordering pairs with the worker's
                            // Release on `paused.store(false)`
                            // (resume path) so the predicate sees
                            // a coherent worker state.
                            let blk_unpaused = freeze_coord_virtio_blk_paused
                                .as_ref()
                                .is_none_or(|p| !p.load(Ordering::Acquire));
                            if aps_unparked && bsp_unparked && blk_unpaused {
                                break;
                            }
                            let now = Instant::now();
                            if now > post_thaw_deadline {
                                let ap_states: Vec<bool> =
                                    freeze_coord_ap_parked
                                        .iter()
                                        .map(|p| p.load(Ordering::Acquire))
                                        .collect();
                                tracing::warn!(
                                    ?ap_states,
                                    bsp_parked = !bsp_unparked,
                                    blk_paused = !blk_unpaused,
                                    ?rendezvous_timeout,
                                    "freeze-coord: post-thaw barrier timed out — \
                                     a parker did not clear within the rendezvous \
                                     timeout; subsequent freeze cycles may see \
                                     stale parked=true and timeout the rendezvous"
                                );
                                break;
                            }
                            let remaining_ms = (post_thaw_deadline - now)
                                .as_millis()
                                .min(i32::MAX as u128) as i32;
                            let mut pfds = [
                                libc::pollfd {
                                    fd: freeze_coord_kill_evt.as_raw_fd(),
                                    events: libc::POLLIN,
                                    revents: 0,
                                },
                                libc::pollfd {
                                    fd: freeze_coord_bsp_done_evt.as_raw_fd(),
                                    events: libc::POLLIN,
                                    revents: 0,
                                },
                            ];
                            // SAFETY: pfds is a 2-element pollfd
                            // array; nfds matches. Bounded 10 ms
                            // wait is the cadence at which the
                            // AtomicBool predicate re-runs.
                            let wait_ms = 10.min(remaining_ms);
                            unsafe {
                                libc::poll(
                                    pfds.as_mut_ptr(),
                                    pfds.len() as libc::nfds_t,
                                    wait_ms,
                                );
                            }
                        }
                        // Drain parked_evt so cycle N+1's countdown
                        // latch starts at 0. EAGAIN (counter already
                        // 0) is benign.
                        let _ = freeze_coord_parked_evt.read();
                    };
                    // Helper: extend the watchdog deadline by the
                    // wall-clock duration of a single
                    // `freeze_and_dispatch(..)` cycle. Captures eat
                    // host wall-clock that would otherwise count
                    // against the workload's `workload_duration`
                    // budget; without this push, a 5 s test that
                    // fires a 2 s freeze gets only 3 s of guest
                    // execution before the watchdog kicks. Reads the
                    // current encoded reset target (or falls back to
                    // `workload_duration` counted from now) and
                    // writes back the sum + freeze_duration so the
                    // watchdog observes the extended deadline on its
                    // next tick. The watchdog only consults this
                    // atomic when `workload_duration` is set; runs
                    // without a workload budget remain on the
                    // boot-relative `hard_deadline` and this push is
                    // a no-op.
                    //
                    // Shared with the TLV CAPTURE handler, the
                    // user-watchpoint dispatcher, and the periodic-
                    // capture drain so the same arithmetic and
                    // ordering discipline apply at every fire site.
                    let extend_watchdog_for_freeze = |freeze_start: Instant| {
                        if let Some(d) = workload_duration_for_coord {
                            let freeze_duration = freeze_start.elapsed();
                            let prior = watchdog_reset_for_coord.load(Ordering::Acquire);
                            let prior_ns = if prior == 0 {
                                run_start
                                    .elapsed()
                                    .as_nanos()
                                    .saturating_add(d.as_nanos())
                            } else {
                                prior as u128
                            };
                            let new_target_ns =
                                prior_ns.saturating_add(freeze_duration.as_nanos());
                            let encoded = u64::try_from(new_target_ns).unwrap_or(u64::MAX).max(1);
                            watchdog_reset_for_coord.store(encoded, Ordering::Release);
                            // Stamp provenance so the watchdog dump
                            // attributes the deadline to the freeze
                            // extension. Relaxed — diagnostic only.
                            watchdog_reset_tag_for_coord
                                .store(WatchdogResetTag::FreezeExtend as u8, Ordering::Relaxed);
                        }
                    };
                    // Helper: persist the JSON to the optional file
                    // sink, then log a single info-level summary line
                    // referencing the file path + byte count +
                    // capture timing. The JSON is NOT inlined into
                    // the trace log — a 50-map dump runs hundreds of
                    // KB and floods every downstream sink (file
                    // logger, journald, stderr) with a payload that
                    // is already on disk at the dump path.
                    //
                    // On atomic-write failure, emit a structured
                    // summary line + truncated JSON head to stderr
                    // as last-ditch preservation. stderr is a
                    // separate channel from the filesystem and
                    // typically survives the kinds of failures
                    // (ENOSPC, EROFS, EACCES) that fail file writes.
                    // The summary uses the operator-friendly stats
                    // this closure already receives (map_count,
                    // vcpu_regs_count, tasks_enriched, elapsed_ms,
                    // json_bytes) so the structural signal survives
                    // independently of the 16 KiB head truncation
                    // that may drop late fields in the serde-ordered
                    // payload. Silent loss of a Captured
                    // failure-dump on disk-full is the exact
                    // silent-drop class to prevent.
                    //
                    // STDERR last-ditch preservation cap is shared
                    // by the 3 stderr-fallback sites (emit_json,
                    // emit_degraded_json, end-of-coord drain).
                    // UTF-8-safe slicing delegated to the module-
                    // level `utf8_safe_truncate_len` helper above so
                    // the 3 sites share one implementation rather
                    // than each carrying an inline copy that could
                    // drift.
                    //
                    // Why stderr fallback, not Result propagation:
                    // the freeze coordinator is an autonomous thread —
                    // its closure has no caller to receive an `Err`
                    // from a failed emit. Propagating Err to the
                    // thread join site in `run_vm` would block teardown
                    // on a serde-/tracing-/FS-failure path the operator
                    // may want to see immediately, not at join time.
                    // `tracing::warn!` dispatch can itself fail
                    // (subscriber disconnected, journald socket closed,
                    // file-appender out of space) with no visible
                    // signal when it does. stderr is the OS-mediated
                    // last channel: writes survive ENOSPC/EROFS on the
                    // dump filesystem (stderr is typically a separate
                    // pipe to the test harness), survive subscriber
                    // teardown (`eprintln!` bypasses the tracing
                    // pipeline), and surface in the same captured
                    // stream the harness already monitors for panic
                    // messages. Per the no-silent-drops policy: the
                    // dump fires loudly even when the configured sink
                    // AND the structured log channels are both
                    // unavailable.
                    const STDERR_DUMP_CAP: usize = 16 * 1024;
                    #[allow(clippy::too_many_arguments)]
                    let emit_json = |json: &str,
                                     map_count: usize,
                                     vcpu_regs_count: usize,
                                     vcpu_none_indices: &[usize],
                                     tasks_enriched: usize,
                                     elapsed_ms: u64,
                                     truncated_at_us: Option<u64>| {
                        // Stderr fallback gates on `write_failed`
                        // ONLY — sink configured AND write_atomic
                        // failed (ENOSPC / EROFS / EACCES).
                        // Production callers that don't wire
                        // `failure_dump_path` (verifier, shell,
                        // template build per builder default = None)
                        // never asked for failure-dump preservation,
                        // so an err-class scheduler exit there is
                        // not a silent drop — it's "no capture was
                        // ever requested." Only the test-dispatch
                        // path in
                        // `test_support::eval::run_ktstr_test_inner`
                        // (search `failure_dump_path(primary_dump_path)`)
                        // and the auto-repro path in
                        // `test_support::probe::attempt_auto_repro`
                        // (search `failure_dump_path(&repro_dump_path)`)
                        // wire the sink; those are the call sites
                        // where silent-drop avoidance applies. The
                        // tracing::info! summary line at the end of
                        // this closure (path = None branch) remains
                        // the operator-visible signal in the no-sink
                        // case, with no stderr noise for verifier /
                        // shell / template iteration.
                        let mut write_failed = false;
                        let path_str: Option<String> =
                            freeze_coord_dump_path.as_ref().and_then(|p| {
                                if let Some(parent) = p.parent() {
                                    let _ = std::fs::create_dir_all(parent);
                                }
                                // Atomic-publish pattern: write to a sibling
                                // .tmp file, fsync the file, then rename into
                                // place, then fsync the parent directory.
                                // Guarantees a reader of `p` either sees the
                                // previous file (if any) or the complete new
                                // dump — never a truncated / mid-write JSON
                                // file. The fsync on the tmp file holds the
                                // bytes against a host crash between rename()
                                // and writeback; the parent-dir fsync (CF3)
                                // holds the directory-entry update against a
                                // host crash post-rename so the operator
                                // post-reboot sees the new file at `p`
                                // rather than an empty parent. POSIX rename(2)
                                // atomicity covers ordering visible to other
                                // processes, NOT durability — directory-entry
                                // durability requires fsync on the parent dir,
                                // matching what database engines (SQLite,
                                // RocksDB) do for journal commits.
                                let tmp_path = p.with_extension("json.tmp");
                                let write_atomic = || -> std::io::Result<()> {
                                    use std::io::Write as _;
                                    let mut f =
                                        std::fs::File::create(&tmp_path)?;
                                    f.write_all(json.as_bytes())?;
                                    f.sync_all()?;
                                    drop(f);
                                    std::fs::rename(&tmp_path, p)?;
                                    // Canonical-path parent-dir fsync (CF3):
                                    // best-effort durability. If the parent
                                    // fsync fails (ENOSPC / EROFS / EIO), do
                                    // NOT roll back the rename — per user
                                    // direction, the operator-facing
                                    // failure-dump.json on disk is more
                                    // valuable than bridge/FS contract
                                    // symmetry. The file IS visible (rename
                                    // succeeded); only the directory-entry
                                    // durability across a host crash is in
                                    // doubt. Log the durability gap and
                                    // continue — operators inspecting the
                                    // dump dir post-test see the file, with
                                    // a warn flagging the missed fsync. The
                                    // tagged-sibling helper write_to_tagged_path
                                    // takes the opposite stance (it rolls
                                    // back) because per-trigger sibling
                                    // dumps don't have the same operator-
                                    // signal preservation concern.
                                    if let Some(dir) = p.parent()
                                        && let Err(parent_fsync_err) =
                                            std::fs::File::open(dir).and_then(|d| d.sync_all())
                                    {
                                        tracing::warn!(
                                            path = %p.display(),
                                            parent_fsync_error = %parent_fsync_err,
                                            "freeze-coord: failure-dump parent-dir fsync failed after rename; file IS visible on disk but directory-entry durability across a host crash is not guaranteed — operator-facing dump preserved per operator-has-the-data discipline"
                                        );
                                    }
                                    Ok(())
                                };
                                match write_atomic() {
                                    Ok(()) => Some(p.display().to_string()),
                                    Err(e) => {
                                        write_failed = true;
                                        // Best-effort cleanup of the leftover
                                        // tmp file so a future operator
                                        // doesn't see a stale `.json.tmp`
                                        // alongside the previous dump.
                                        let _ = std::fs::remove_file(&tmp_path);
                                        tracing::warn!(
                                            path = %p.display(),
                                            tmp_path = %tmp_path.display(),
                                            error = %e,
                                            "freeze-coord: failure-dump atomic write failed; stderr fallback below"
                                        );
                                        None
                                    }
                                }
                            });
                        // Stderr last-ditch preservation fires ONLY
                        // when a configured sink failed
                        // (`write_failed`). The structured summary
                        // line surfaces the operator-critical
                        // signals (vcpu count + none-indices, map
                        // count, payload bytes) regardless of
                        // payload size, then the truncated JSON head
                        // provides forensic context.
                        // `vcpu_none_indices` shows the operator
                        // WHICH vCPUs were never parked — same
                        // signal the drain summary surfaces.
                        // Callers that never wired
                        // `failure_dump_path` (verifier / shell /
                        // template) get the closure's tail
                        // tracing::info! "no file sink" line and no
                        // stderr noise — no capture was ever
                        // requested.
                        if write_failed {
                            eprintln!(
                                "{SNAPSHOT_SUMMARY_PREFIX} (Captured, write failed): map_count={} vcpu_regs_count={} vcpu_none_indices={:?} tasks_enriched={} elapsed_ms={} json_bytes={}",
                                map_count,
                                vcpu_regs_count,
                                vcpu_none_indices,
                                tasks_enriched,
                                elapsed_ms,
                                json.len(),
                            );
                            let head_end = utf8_safe_truncate_len(
                                json,
                                STDERR_DUMP_CAP,
                            );
                            let truncated_marker =
                                format_truncation_marker(head_end, json.len());
                            eprintln!(
                                "freeze-coord: STDERR-PRESERVED payload head (Captured, write failed){}: {}",
                                truncated_marker,
                                &json[..head_end]
                            );
                        }
                        let json_bytes = json.len();
                        // Disambiguation and consolidation. See
                        // [`format_path_part`] for the 3-state rule.
                        let path_part = format_path_part(path_str.as_deref(), write_failed);
                        let trunc_part = truncated_at_us
                            .map(|us| format!(" (truncated at {us}us)"))
                            .unwrap_or_default();
                        tracing::info!(
                            target: "ktstr::failure_dump",
                            map_count,
                            vcpu_regs_count,
                            tasks_enriched,
                            json_bytes,
                            elapsed_ms,
                            truncated_at_us,
                            path = path_str.as_deref(),
                            "freeze-coord: dump complete{trunc_part}, {map_count} maps, {tasks_enriched} tasks enriched, {elapsed_ms}ms freeze, {json_bytes} bytes{path_part}"
                        );
                    };
                    // Persist a SCHEMA_DEGRADED JSON via the same
                    // atomic-publish dance `emit_json` uses. Separate
                    // closure so the log target / message identifies
                    // the degraded shape distinctly from a full dump
                    // — an operator reading the log sees "dump
                    // degraded" rather than the "dump complete"
                    // banner the full-dump path emits. The reason +
                    // elapsed_ms surface in the log line so a quick
                    // scan of trace output answers the operator's
                    // first question ("what went wrong?") without
                    // opening the JSON.
                    let emit_degraded_json = |json: &str,
                                              reason: &str,
                                              elapsed_ms: u64| {
                        // Symmetric with emit_json: stderr fallback
                        // hoisted OUT of the and_then closure so a
                        // Degraded-path write_atomic failure
                        // surfaces to stderr regardless of the
                        // closure short-circuit, AND gated on
                        // `write_failed` so callers that never
                        // wired `failure_dump_path` (verifier /
                        // shell / template) don't get a stderr
                        // blast for "no capture was requested."
                        // See emit_json above for the full
                        // rationale.
                        let mut write_failed = false;
                        let path_str: Option<String> =
                            freeze_coord_dump_path.as_ref().and_then(|p| {
                                if let Some(parent) = p.parent() {
                                    let _ = std::fs::create_dir_all(parent);
                                }
                                let tmp_path = p.with_extension("json.tmp");
                                let write_atomic = || -> std::io::Result<()> {
                                    use std::io::Write as _;
                                    let mut f =
                                        std::fs::File::create(&tmp_path)?;
                                    f.write_all(json.as_bytes())?;
                                    f.sync_all()?;
                                    drop(f);
                                    std::fs::rename(&tmp_path, p)?;
                                    // Canonical-path parent-dir fsync (CF3):
                                    // best-effort durability — symmetric with
                                    // emit_json above. On parent-fsync failure
                                    // after a successful rename, do NOT roll
                                    // back the renamed file: the operator-
                                    // facing degraded dump on disk is more
                                    // valuable than bridge/FS contract
                                    // symmetry. Log the durability gap and
                                    // continue; operators reading the dump
                                    // dir post-test see the file with a warn
                                    // flagging the missed fsync. The tagged-
                                    // sibling helper write_to_tagged_path
                                    // takes the opposite stance (rollback)
                                    // because per-trigger siblings don't
                                    // carry the same operator-signal
                                    // preservation concern.
                                    if let Some(dir) = p.parent()
                                        && let Err(parent_fsync_err) =
                                            std::fs::File::open(dir).and_then(|d| d.sync_all())
                                    {
                                        tracing::warn!(
                                            path = %p.display(),
                                            parent_fsync_error = %parent_fsync_err,
                                            "freeze-coord: degraded-dump parent-dir fsync failed after rename; file IS visible on disk but directory-entry durability across a host crash is not guaranteed — operator-facing dump preserved per operator-has-the-data discipline"
                                        );
                                    }
                                    Ok(())
                                };
                                match write_atomic() {
                                    Ok(()) => Some(p.display().to_string()),
                                    Err(e) => {
                                        write_failed = true;
                                        let _ = std::fs::remove_file(&tmp_path);
                                        tracing::warn!(
                                            path = %p.display(),
                                            tmp_path = %tmp_path.display(),
                                            error = %e,
                                            "freeze-coord: degraded-dump atomic write failed; stderr fallback below"
                                        );
                                        None
                                    }
                                }
                            });
                        if write_failed {
                            eprintln!(
                                "{SNAPSHOT_SUMMARY_PREFIX} (Degraded, write failed): reason={:?} elapsed_ms={} json_bytes={}",
                                reason,
                                elapsed_ms,
                                json.len(),
                            );
                            let head_end = utf8_safe_truncate_len(json, STDERR_DUMP_CAP);
                            let truncated_marker =
                                format_truncation_marker(head_end, json.len());
                            eprintln!(
                                "freeze-coord: STDERR-PRESERVED payload head (Degraded, write failed){}: {}",
                                truncated_marker,
                                &json[..head_end]
                            );
                        }
                        let json_bytes = json.len();
                        // Same as emit_json. See [`format_path_part`]
                        // for the rule.
                        let path_part = format_path_part(path_str.as_deref(), write_failed);
                        tracing::warn!(
                            target: "ktstr::failure_dump",
                            reason,
                            json_bytes,
                            elapsed_ms,
                            path = path_str.as_deref(),
                            "freeze-coord: degraded dump emitted, {reason}, {elapsed_ms}ms, {json_bytes} bytes{path_part}"
                        );
                    };
                    // On-demand snapshot handler. Drains every
                    // [`crate::vmm::wire::MSG_TYPE_SNAPSHOT_REQUEST`]
                    // frame the TOKEN_TX handler accumulated this
                    // iteration, regardless of `freeze_state`. The
                    // `on_demand_in_flight` AcqRel-bool serialises
                    // CAPTURE/WATCH against the user-watchpoint
                    // dispatcher below — a snapshot capture in
                    // progress here makes the watchpoint loop re-arm
                    // its `hit` flag for the next iteration instead
                    // of opening a second concurrent capture window.
                    //
                    // CAPTURE runs `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })` and
                    // stores the report on the bridge under the
                    // tag, then frames a `MSG_TYPE_SNAPSHOT_REPLY`
                    // TLV (header + 72-byte payload) and pushes it
                    // through `queue_input_port1` so the guest's
                    // blocking reader on `/dev/vport0p1` wakes
                    // within microseconds and observes
                    // `reply.request_id == request.request_id`.
                    // WATCH resolves the symbol via the cached
                    // vmlinux ELF symbol table, allocates a free
                    // user watchpoint slot, publishes the resolved
                    // KVA + tag into `WatchpointArm`, kicks every
                    // vCPU so `self_arm_watchpoint` picks up the
                    // new arm before the next `KVM_RUN`, and
                    // replies OK over the same TLV channel. A
                    // future guest write to the resolved KVA fires
                    // the corresponding `KVM_EXIT_DEBUG` and the
                    // user-watchpoint dispatcher (further down the
                    // iteration) drives the matching capture.
                    // Drain WATCH requests deferred during the
                    // pre-KASLR / pre-accessor window once BOTH
                    // inputs are ready (kern_virt_kaslr Arc has
                    // PUBLISHED — including the biased-zero CAS
                    // under nokaslr where the actual slide is 0
                    // but the raw atomic is 1 — AND owned_accessor
                    // has been adopted). Without
                    // this drain, a WATCH request deferred for
                    // either reason would sit in
                    // `capture_requests_deferred` indefinitely
                    // (the owned_accessor-adoption drain at the
                    // accessor-arrival site fires once at
                    // adoption — if kaslr wasn't ready then, it
                    // re-defers and needs this loop to catch it
                    // on the kaslr-ready iteration; symmetrically,
                    // a WATCH that arrived after adoption but
                    // before kaslr publication needs this drain).
                    // Filter to WATCH-kind — CAPTURE entries only
                    // care about accessor adoption and continue
                    // through the dedicated accessor-adoption
                    // drain path.
                    if kern_virt_kaslr_published()
                        && owned_accessor.is_some()
                        && !capture_requests_deferred.is_empty()
                    {
                        let drained = std::mem::take(&mut capture_requests_deferred);
                        for req in drained {
                            if req.kind == crate::vmm::wire::SNAPSHOT_KIND_WATCH {
                                snapshot_requests_pending.push(req);
                            } else {
                                capture_requests_deferred.push(req);
                            }
                        }
                    }
                    let pending = std::mem::take(&mut snapshot_requests_pending);
                    for SnapshotRequest {
                        request_id,
                        kind,
                        tag,
                    } in pending
                    {
                        if kind == crate::vmm::wire::SNAPSHOT_KIND_CAPTURE
                            && owned_accessor.is_none()
                        {
                            tracing::info!(
                                request_id,
                                %tag,
                                "freeze-coord: TLV CAPTURE deferred \
                                 (owned_accessor not yet adopted)"
                            );
                            capture_requests_deferred.push(SnapshotRequest {
                                request_id,
                                kind,
                                tag,
                            });
                            continue;
                        }
                        let _gate_guard = match gate::OnDemandGateGuard::try_acquire(
                            &freeze_coord_on_demand_in_flight,
                        ) {
                            Some(g) => g,
                            None => {
                                // A user-watchpoint capture is already
                                // in flight (or a prior iteration
                                // somehow left the gate set). Reply
                                // ERR rather than let the guest block
                                // its full 30 s deadline; the test
                                // can retry once the in-flight
                                // capture completes.
                                let reply = frame_snapshot_reply(
                                    request_id,
                                    crate::vmm::wire::SNAPSHOT_STATUS_ERR,
                                    "another snapshot capture is in flight; retry",
                                );
                                freeze_coord_virtio_con
                                    .lock()
                                    .queue_input_port1(&reply);
                                tracing::warn!(
                                    request_id,
                                    %tag,
                                    kind,
                                    "freeze-coord: snapshot request rejected (in-flight gate held)"
                                );
                                continue;
                            }
                        };
                        match kind {
                            crate::vmm::wire::SNAPSHOT_KIND_CAPTURE => {
                                tracing::info!(
                                    request_id,
                                    %tag,
                                    "freeze-coord: TLV CAPTURE request"
                                );
                                // CAPTURE has no while-frozen work,
                                // so thaw immediately after the
                                // dump returns. Then extend the
                                // watchdog deadline by the freeze
                                // duration via the shared closure
                                // (TLV CAPTURE / user watchpoint /
                                // periodic-capture all use the same
                                // arithmetic — see
                                // `extend_watchdog_for_freeze` for
                                // the full rationale).
                                let freeze_start = Instant::now();
                                let on_demand = freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false });
                                thaw_and_barrier();
                                extend_watchdog_for_freeze(freeze_start);
                                let mut reply_status =
                                    crate::vmm::wire::SNAPSHOT_STATUS_OK;
                                let mut reply_reason = String::new();
                                match on_demand {
                                    FreezeOutcome::Captured(report, capture_start) => {
                                    let map_count = report.maps.len();
                                    let vcpu_regs_count =
                                        report.vcpu_regs.len();
                                    let tasks_enriched =
                                        report.task_enrichments.len();
                                    // File mirror first via `&report`
                                    // (no clone). Bridge `store`
                                    // consumes the report, so any
                                    // additional reader needs to run
                                    // BEFORE the move. `to_string`
                                    // (compact) replaces
                                    // `to_string_pretty` to halve
                                    // serialization cost — the JSON
                                    // is consumed by tests and tools,
                                    // not by humans, and `jq` /
                                    // `serde_json::from_str` parse
                                    // both forms identically. Avoids
                                    // the prior `report.clone()` deep
                                    // copy of hundreds-of-KB-scale
                                    // dump data.
                                    if freeze_coord_dump_path.is_some() {
                                        match serde_json::to_string(&report) {
                                            Ok(json) => {
                                                let _ = write_to_tagged_path(
                                                    freeze_coord_dump_path.as_deref(),
                                                    &tag,
                                                    &json,
                                                    || {
                                                        format!(
                                                            "{SNAPSHOT_SUMMARY_PREFIX} (on-demand TLV CAPTURE, write failed): request_id={request_id} tag={tag} map_count={map_count} vcpu_regs_count={vcpu_regs_count} tasks_enriched={tasks_enriched} elapsed_ms={} json_bytes={}",
                                                            capture_start.elapsed().as_millis() as u64,
                                                            json.len(),
                                                        )
                                                    },
                                                    "freeze-coord: on-demand TLV CAPTURE dump file write failed",
                                                );
                                            }
                                            Err(e) => tracing::error!(
                                                error = %e,
                                                map_count,
                                                vcpu_regs_count,
                                                "freeze-coord: on-demand dump (JSON serialization failed)"
                                            ),
                                        }
                                    }
                                    let elapsed_ms = capture_start
                                        .elapsed()
                                        .as_millis() as u64;
                                    tracing::info!(
                                        target: "ktstr::failure_dump",
                                        kind = "on_demand_capture",
                                        request_id,
                                        %tag,
                                        map_count,
                                        vcpu_regs_count,
                                        tasks_enriched,
                                        elapsed_ms,
                                        "freeze-coord: snapshot captured and stored on bridge"
                                    );
                                    // Persist on the bridge LAST —
                                    // store moves the report. Test
                                    // code drains the bridge after
                                    // VM exit and walks the reports
                                    // via the public `Snapshot`
                                    // accessor.
                                    freeze_coord_snapshot_bridge.store(&tag, report);
                                    }
                                    FreezeOutcome::Degraded(degraded) => {
                                        // Rendezvous timed out building
                                        // the on-demand capture. Write
                                        // the degraded JSON to the same
                                        // tagged dump path and store a
                                        // placeholder on the bridge
                                        // carrying the partial vcpu_regs
                                        // + the degraded reason —
                                        // operator's structured
                                        // diagnostics survive even when
                                        // the bridge entry is a
                                        // legacy-shape placeholder.
                                        if freeze_coord_dump_path.is_some() {
                                            match serde_json::to_string(degraded.as_ref()) {
                                                Ok(json) => {
                                                    let _ = write_to_tagged_path(
                                                        freeze_coord_dump_path.as_deref(),
                                                        &tag,
                                                        &json,
                                                        || {
                                                            format!(
                                                                "{SNAPSHOT_SUMMARY_PREFIX} (on-demand TLV CAPTURE Degraded, write failed): request_id={request_id} tag={tag} reason={:?} json_bytes={}",
                                                                degraded.reason,
                                                                json.len(),
                                                            )
                                                        },
                                                        "freeze-coord: on-demand TLV CAPTURE degraded dump file write failed",
                                                    );
                                                }
                                                Err(e) => tracing::error!(
                                                    error = %e,
                                                    "freeze-coord: on-demand degraded dump (JSON serialization failed)"
                                                ),
                                            }
                                        }
                                        let placeholder = crate::monitor::dump::FailureDumpReport {
                                            vcpu_regs: degraded.vcpu_regs.clone(),
                                            ..crate::monitor::dump::FailureDumpReport::placeholder(
                                                degraded.reason.clone(),
                                            )
                                        };
                                        freeze_coord_snapshot_bridge.store(&tag, placeholder);
                                        reply_status =
                                            crate::vmm::wire::SNAPSHOT_STATUS_ERR;
                                        reply_reason = degraded.reason.clone();
                                        tracing::warn!(
                                            request_id,
                                            %tag,
                                            reason = %degraded.reason,
                                            "freeze-coord: on-demand capture degraded (rendezvous timeout)"
                                        );
                                    }
                                    FreezeOutcome::Suppressed => {
                                        // Unreachable in practice: the gate
                                        // only runs when gate_on_exit_kind=true
                                        // and the on-demand path always
                                        // passes false. Defensive: reply
                                        // ERR so a future code path that
                                        // changes the gate flag can't
                                        // silently drop the request.
                                        reply_status =
                                            crate::vmm::wire::SNAPSHOT_STATUS_ERR;
                                        reply_reason =
                                            "on-demand capture suppressed (gate decision)".to_string();
                                        tracing::warn!(
                                            request_id,
                                            %tag,
                                            "freeze-coord: on-demand capture unexpectedly suppressed by gate"
                                        );
                                    }
                                    FreezeOutcome::KernelOp(_) => capture_only_unreachable!(),
                                }
                                let reply = frame_snapshot_reply(
                                    request_id,
                                    reply_status,
                                    &reply_reason,
                                );
                                freeze_coord_virtio_con
                                    .lock()
                                    .queue_input_port1(&reply);
                            }
                            crate::vmm::wire::SNAPSHOT_KIND_WATCH => {
                                if !kern_virt_kaslr_published()
                                    || owned_accessor.is_none()
                                    || owned_prog_accessor.is_none()
                                {
                                    // WATCH arms a DR against
                                    // `link_kva + kaslr_offset` AND
                                    // every subsequent DR fire drives
                                    // a `freeze_and_dispatch(Capture)`
                                    // pass that walks the prog
                                    // accessor's BPF maps. All three
                                    // inputs must be ready before the
                                    // arm:
                                    //   - kaslr_offset == 0 against a
                                    //     kernel-high-half symbol
                                    //     under KASLR-on (the default)
                                    //     would arm at the link-time
                                    //     KVA (DR never fires under
                                    //     the runtime slide) OR hit
                                    //     the fail-loud Err in
                                    //     `arm_user_watchpoint`.
                                    //   - owned_accessor == None means
                                    //     the per-fire capture walks
                                    //     an absent accessor and
                                    //     produces a placeholder /
                                    //     0-map report — the test
                                    //     surface sees the slot's tag
                                    //     with no real data.
                                    //   - owned_prog_accessor == None
                                    //     means the dump-side
                                    //     `identify_active_obj_from_struct_ops`
                                    //     walker is unavailable; the
                                    //     report's `active_map_kvas`
                                    //     stays empty and same-binary
                                    //     scheduler-swap consumers
                                    //     (the pinned-bss
                                    //     disambiguation surface) lose
                                    //     their KVA whitelist.
                                    // Defer on any of the three; the
                                    // per-iteration drain above (which
                                    // gates on kern_virt_kaslr_published()
                                    // — true post-publish even under
                                    // nokaslr) AND the accessor-adoption
                                    // drain at the accessor-arrival
                                    // site both feed the deferred
                                    // queue back into pending.
                                    tracing::info!(
                                        request_id,
                                        %tag,
                                        kaslr_published = kern_virt_kaslr_published(),
                                        accessor_adopted = owned_accessor.is_some(),
                                        prog_accessor_adopted = owned_prog_accessor.is_some(),
                                        "freeze-coord: TLV WATCH deferred \
                                         (kern_virt_kaslr, owned_accessor, or \
                                         owned_prog_accessor not yet published)"
                                    );
                                    capture_requests_deferred.push(SnapshotRequest {
                                        request_id,
                                        kind,
                                        tag,
                                    });
                                    continue;
                                }
                                tracing::info!(
                                    request_id,
                                    %tag,
                                    "freeze-coord: TLV WATCH request"
                                );
                                // Reply path branches on whether the
                                // cached vmlinux symbol map is available.
                                // The fall-through (no `continue`) lets
                                // the user-watchpoint loop and the
                                // late-trigger handler later in this
                                // iteration still run, so a WATCH that
                                // cannot resolve does not stall an
                                // already-pending err_triggered dump
                                // for a full poll interval.
                                let (status, reason) = match freeze_coord_symbol_cache.as_ref() {
                                    None => (
                                        crate::vmm::wire::SNAPSHOT_STATUS_ERR,
                                        "vmlinux symbol cache unavailable \
                                         (vmlinux not found or parse failed at \
                                         coord init)"
                                            .to_string(),
                                    ),
                                    Some(symbol_cache) => {
                                        // Pass the bsp_alive Arc by
                                        // reference so each BSP-touching
                                        // site inside `arm_user_watchpoint`
                                        // (the BSP `ie.set` and the BSP
                                        // `pthread_kill`) issues its own
                                        // fresh Acquire load immediately
                                        // before the syscall. A bool
                                        // snapshot taken here would be
                                        // stale by the time the kick
                                        // pass reaches the BSP — long
                                        // enough for the BSP run-loop to
                                        // publish `false` (Release) and
                                        // drop its `VcpuFd`, leaving a
                                        // `true`-snapshot writing through
                                        // freed kvm_run mmap pages.
                                        // `run_vm` flips bsp_alive to
                                        // false only AFTER joining the
                                        // coordinator (see `bsp_alive`
                                        // in run_vm), so a `true`
                                        // reading inside the helper is
                                        // load-bearing for the BSP
                                        // kvm_run mmap's liveness.
                                        match arm_user_watchpoint(
                                            &freeze_coord_watchpoint,
                                            symbol_cache,
                                            &tag,
                                            coord_kaslr_offset(),
                                            &freeze_coord_ap_pthreads,
                                            &freeze_coord_ap_ies,
                                            &freeze_coord_ap_alive,
                                            freeze_coord_bsp_tid,
                                            freeze_coord_bsp_ie_handle.as_ref(),
                                            &bsp_alive_for_coord,
                                        ) {
                                            Ok(slot_idx) => {
                                                tracing::info!(
                                                    request_id,
                                                    %tag,
                                                    slot_idx,
                                                    "freeze-coord: hardware watchpoint armed"
                                                );
                                                (
                                                    crate::vmm::wire::SNAPSHOT_STATUS_OK,
                                                    String::new(),
                                                )
                                            }
                                            Err(reason) => {
                                                tracing::warn!(
                                                    request_id,
                                                    %tag,
                                                    %reason,
                                                    "freeze-coord: WATCH register failed"
                                                );
                                                (
                                                    crate::vmm::wire::SNAPSHOT_STATUS_ERR,
                                                    reason,
                                                )
                                            }
                                        }
                                    }
                                };
                                let reply = frame_snapshot_reply(
                                    request_id,
                                    status,
                                    &reason,
                                );
                                freeze_coord_virtio_con
                                    .lock()
                                    .queue_input_port1(&reply);
                            }
                            unknown => {
                                tracing::warn!(
                                    request_id,
                                    %tag,
                                    kind = unknown,
                                    "freeze-coord: TLV snapshot request with unknown kind"
                                );
                                let reply = frame_snapshot_reply(
                                    request_id,
                                    crate::vmm::wire::SNAPSHOT_STATUS_ERR,
                                    &format!("unknown snapshot kind {unknown}"),
                                );
                                freeze_coord_virtio_con
                                    .lock()
                                    .queue_input_port1(&reply);
                            }
                        }
                    }

                    // Cold-path Op handler — wire-up landed; freeze +
                    // gmem.write_obj/read_obj handler internals are a
                    // follow-up. For now every decoded
                    // KernelOpRequest replies with a typed `success =
                    // false` carrying the in-progress diagnostic so
                    // the guest's executor surfaces the error
                    // promptly (well under the 30 s per-op transport
                    // deadline) rather than hanging until timeout.
                    // Without this drain, the per-iteration pending
                    // queue would accumulate without bound and every
                    // guest-side `Op::WriteKernelCold` /
                    // `Op::ReadKernelCold` would silently hang — a
                    // silent-drop violation. The error text names
                    // the in-progress work so an operator catching
                    // the error in CI can find the live work.
                    //
                    // Drain ordering: this block runs AFTER the
                    // snapshot drain above. Snapshot ops carry a
                    // 30 s guest-side per-op deadline; running the
                    // snapshot drain first preserves that headroom
                    // when a cold-op batch later consumes
                    // freeze-rendezvous budget from the same
                    // iteration. Snapshot-priority is a behavior
                    // choice — Op::CaptureSnapshot / Op::WatchSnapshot
                    // call sites assume the captured state reflects
                    // the moment the op was issued, not the moment
                    // after an arbitrary-duration kernel-op batch
                    // finished modifying that same kernel state.
                    // Drain ColdOp requests deferred during the
                    // pre-KASLR / pre-accessor window once BOTH
                    // inputs are ready. Mirror of the WATCH-drain
                    // at the snapshot-request site. Without this,
                    // a ColdOp deferred when kaslr=0 (e.g. accessor
                    // adopted but kaslr publish hasn't landed yet)
                    // would sit in `kernel_op_requests_deferred`
                    // until the next accessor-adoption fire — which
                    // is "never" because accessor adopts once.
                    // Re-queue the deferred list onto pending so
                    // this iteration's normal dispatch path runs
                    // with both prerequisites in place.
                    if kern_virt_kaslr_published()
                        && owned_accessor.is_some()
                        && !kernel_op_requests_deferred.is_empty()
                    {
                        kernel_op_requests_pending.append(
                            &mut kernel_op_requests_deferred,
                        );
                    }
                    let pending_kernel_ops = std::mem::take(&mut kernel_op_requests_pending);
                    for req in pending_kernel_ops {
                        // Defer-on-no-accessor gate. Mirrors the
                        // `owned_accessor.is_none()` check the TLV
                        // CAPTURE drain runs above. Without this,
                        // requests landing before the
                        // accessor-init worker publishes hit the
                        // ColdOp arm in `freeze_and_dispatch` where
                        // the `else { defensive error }` branch
                        // synthesises a `success = false` reply with
                        // "owned_accessor not yet initialised" — the
                        // guest's executor converts that into an
                        // `anyhow::bail!`, failing the test. Boot-
                        // fast scenarios (single-Op step with no
                        // Backdrop) hit this race reliably because
                        // the guest reaches `apply_ops` while
                        // KERN_ADDRS + accessor-init are still in
                        // flight on the host. Queue here and the
                        // adoption site (where the OnceLock-published
                        // pair lands on this coord) appends the queue
                        // back onto `kernel_op_requests_pending` so
                        // the same iteration body's drain dispatches
                        // them through the normal flow with the
                        // accessor present.
                        if owned_accessor.is_none() || !kern_virt_kaslr_published() {
                            // ColdOp dispatch resolves per-CPU /
                            // task-field / symbol KVAs via
                            // `per_cpu_kva = template + kaslr_offset`
                            // (kernel_op_dispatch's resolve_per_cpu_
                            // field_pa) and depends on the prog
                            // accessor for any field reads that walk
                            // BPF maps. Both inputs must be ready
                            // before dispatch:
                            //   - owned_accessor == None: the ColdOp
                            //     arm in freeze_and_dispatch hits the
                            //     defensive "owned_accessor not yet
                            //     initialised" reply branch.
                            //   - !kern_virt_kaslr_published(): the
                            //     publish chain hasn't landed yet
                            //     (raw atomic still 0). Under a
                            //     KASLR-on guest dispatching now
                            //     would resolve per_cpu_kva on the
                            //     link-time KVA and trip the
                            //     kernel-half-floor check in
                            //     resolve_per_cpu_field_pa. Under
                            //     nokaslr the publisher still fires
                            //     (with biased-zero) so the gate
                            //     unblocks quickly; the check
                            //     correctly distinguishes "publish
                            //     incomplete" from "publish complete
                            //     with offset 0".
                            // Same pattern as the WATCH-deferral at
                            // L7767; defer + drain when both
                            // prerequisites land.
                            tracing::info!(
                                request_id = req.request_id,
                                tag = %req.tag,
                                kaslr_published = kern_virt_kaslr_published(),
                                accessor_adopted = owned_accessor.is_some(),
                                "freeze-coord: ColdOp deferred \
                                 (owned_accessor or kern_virt_kaslr not yet published)"
                            );
                            kernel_op_requests_deferred.push(req);
                            continue;
                        }
                        // Per-iteration `on_demand_in_flight` gate
                        // acquire: ColdOp dispatch must NOT race a
                        // concurrent CAPTURE / periodic /
                        // user-watchpoint rendezvous. Same RAII guard
                        // pattern as the snapshot capture sites at
                        // the TLV CAPTURE and periodic call sites.
                        // If the gate is already held, reply ERR
                        // rather than block — caller can re-queue
                        // via the host TLV layer.
                        let _gate_guard = match gate::OnDemandGateGuard::try_acquire(
                            &freeze_coord_on_demand_in_flight,
                        ) {
                            Some(g) => g,
                            None => {
                                let reply = crate::vmm::wire::KernelOpReplyPayload {
                                    request_id: req.request_id,
                                    success: false,
                                    reason:
                                        "freeze-coord: on_demand_in_flight gate held; ColdOp deferred; retry"
                                            .to_string(),
                                    read_values: Vec::new(),
                                };
                                freeze_coord_snapshot_bridge
                                    .record_kernel_op_reply(req.tag.clone(), reply.clone());
                                match frame_kernel_op_reply(&reply) {
                                    Ok(frame) => {
                                        freeze_coord_virtio_con
                                            .lock()
                                            .queue_input_port1(&frame);
                                        tracing::warn!(
                                            request_id = req.request_id,
                                            tag = %req.tag,
                                            "freeze-coord: KernelOpRequest rejected \
                                             (in-flight gate held)"
                                        );
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            request_id = req.request_id,
                                            tag = %req.tag,
                                            error = %e,
                                            "freeze-coord: KernelOpReply postcard serialize failed \
                                             (gate-held branch); guest will see transport timeout"
                                        );
                                    }
                                }
                                continue;
                            }
                        };
                        // Gate held (RAII `_gate_guard` above). Invoke
                        // the freeze rendezvous + dispatcher; thaw
                        // after; the guard's `Drop` releases the gate
                        // when this for-loop iteration's scope exits.
                        // Panic-safe by construction: a panic in any of
                        // the framing / serialize calls below unwinds
                        // through `Drop` so the gate clears even on
                        // unwind. The closure body's V4 split returns
                        // FreezeOutcome::KernelOp(reply) for ColdOp
                        // mode; V5 !all_parked ColdOp arm builds an
                        // error reply with rendezvous-timeout context.
                        // Non-KernelOp outcomes are unreachable per the
                        // closure body's structure but defensively
                        // typed to surface a coordinator-side bug if
                        // future closure surgery breaks the invariant.
                        let outcome = freeze_and_dispatch(
                            FreezeMode::ColdOp(&req),
                        );
                        thaw_and_barrier();
                        let reply = match outcome {
                            FreezeOutcome::KernelOp(reply) => reply,
                            FreezeOutcome::Captured(_, _)
                            | FreezeOutcome::Degraded(_)
                            | FreezeOutcome::Suppressed => {
                                tracing::error!(
                                    request_id = req.request_id,
                                    tag = %req.tag,
                                    "freeze-coord: invariant violated — FreezeMode::ColdOp dispatch returned non-KernelOp variant; framing synthetic error reply"
                                );
                                crate::vmm::wire::KernelOpReplyPayload {
                                    request_id: req.request_id,
                                    success: false,
                                    reason:
                                        "freeze-coord: ColdOp dispatch returned wrong FreezeOutcome variant (coordinator bug)"
                                            .to_string(),
                                    read_values: Vec::new(),
                                }
                            }
                        };
                        freeze_coord_snapshot_bridge
                            .record_kernel_op_reply(req.tag.clone(), reply.clone());
                        match frame_kernel_op_reply(&reply) {
                            Ok(frame) => {
                                freeze_coord_virtio_con
                                    .lock()
                                    .queue_input_port1(&frame);
                                tracing::info!(
                                    request_id = req.request_id,
                                    tag = %req.tag,
                                    mode = ?req.mode,
                                    direction = ?req.direction,
                                    entries = req.entries.len(),
                                    success = reply.success,
                                    "freeze-coord: KernelOpRequest dispatched + replied"
                                );
                            }
                            Err(e) => {
                                tracing::error!(
                                    request_id = req.request_id,
                                    tag = %req.tag,
                                    error = %e,
                                    "freeze-coord: KernelOpReply postcard serialize failed; \
                                     guest will see transport timeout"
                                );
                            }
                        }
                    }
                    // Periodic-capture cadence runs BEFORE the
                    // user-watchpoint dispatch below so periodic
                    // boundaries get priority over Op::CaptureSnapshot /
                    // Op::WatchSnapshot fires when both contend for
                    // the same `freeze_coord_on_demand_in_flight`
                    // gate. Iteration ordering within the body:
                    // TLV CAPTURE runs first (request-reply,
                    // self-throttling); periodic runs second with
                    // priority over user-watchpoint hits.
                    // Lazily compute the boundary list once
                    // `num_snapshots > 0`, the workload duration is
                    // known, and the first ScenarioStart has been
                    // stamped — then on every iteration check
                    // whether `now` has crossed the next un-fired
                    // boundary, and fire a host-side
                    // `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })` for each crossed
                    // boundary. Reuses the same gate
                    // (`freeze_coord_on_demand_in_flight`) the
                    // TLV CAPTURE / user-watchpoint paths use —
                    // when the gate is held the boundary is
                    // deferred to the next iteration rather than
                    // skipped, so a burst of on-demand captures
                    // cannot cause a missed periodic sample. The
                    // 10% / 10% pre/post buffers in the boundary
                    // formula are the budget that absorbs this
                    // deferral lag.
                    // Stamp the prereq-ready moment once (run_start-relative):
                    // the moment all periodic-capture prereqs first hold, which
                    // the boundary window anchors at on a cold boot so boundaries
                    // fire in the capturable window instead of being stranded in
                    // the already-past pre-ready region. Same three predicates the
                    // fire-loop defers on. `.max(1)` keeps a (boot-unreachable)
                    // elapsed==0 from colliding with the not-ready 0 sentinel.
                    if periodic_prereqs_ready_ns == 0
                        && kern_virt_kaslr_published()
                        && owned_accessor.is_some()
                        && owned_prog_accessor.is_some()
                    {
                        periodic_prereqs_ready_ns = u64::try_from(run_start.elapsed().as_nanos())
                            .unwrap_or(u64::MAX)
                            .max(1);
                        // Export for the post-run readiness-vs-window gate.
                        periodic_prereqs_ready_for_coord
                            .store(periodic_prereqs_ready_ns, Ordering::Release);
                        // Signal a periodic-capture guest that ALL prereqs now
                        // hold, so it can open the capture window (send its
                        // ScenarioStart / start the workload) with the full
                        // duration ahead of it instead of racing the KASLR
                        // publish. Only meaningful when the run declares
                        // periodic captures; the guest gates its wait on the
                        // matching `KTSTR_AWAIT_PERIODIC_READY` karg, so the
                        // byte is inert (sticky RX, consumed by the poll loop,
                        // no waiter) on non-periodic runs — but skip the queue
                        // push entirely on those to avoid needless RX traffic.
                        // Fires once (the `== 0` stamp guard is one-shot).
                        if freeze_coord_num_snapshots > 0 {
                            super::host_comms::request_periodic_prereqs_ready(
                                &freeze_coord_virtio_con,
                            );
                        }
                    }
                    if freeze_coord_num_snapshots > 0 && !periodic_abandoned {
                        if periodic_boundaries_ns.is_none()
                            && periodic_prereqs_ready_ns != 0
                            && let Some(workload_d) = workload_duration_for_coord
                        {
                            // Anchor selection — preferring guest-stamped
                            // `MSG_TYPE_SCENARIO_START` over the host's
                            // own scheduler-attach observation.
                            //
                            // Primary: the guest's
                            // `send_scenario_start` (Phase 5 of
                            // `vmm::rust_init`) writes
                            // `MSG_TYPE_SCENARIO_START` over the bulk
                            // port; the host's `dispatch_bulk_message`
                            // stamps `scenario_start_ns` on first
                            // CRC-valid arrival. This is the canonical
                            // anchor and accounts for any guest-side
                            // delay between scheduler attach and
                            // workload-start.
                            //
                            // Fallback: a scheduler that breaks the
                            // bulk port (the scx_mitosis cell-mode
                            // case where the
                            // guest never gets to / completes
                            // send_scenario_start) leaves
                            // scenario_start_ns at 0 forever and the
                            // periodic-capture pipeline silently
                            // 0-fires. Synthesise the anchor from
                            // the host-monitor-observed scheduler-
                            // attach elapsed time (encoded in
                            // `watchdog_reset_ns` as
                            // `attach_elapsed_ns + workload_duration_ns`
                            // — see the `reset_ns.store` on the first non-NULL
                            // `*scx_root` transition in `src/monitor/reader.rs`
                            // (~2865)).
                            // The derived anchor is the moment the
                            // host saw `*scx_root` transition to
                            // non-NULL; that's the earliest meaningful
                            // workload start we can name without a
                            // guest signal.
                            let mut scenario_anchor =
                                scenario_start_ns_for_coord.load(Ordering::Relaxed);
                            if scenario_anchor == 0 {
                                let reset_encoded = watchdog_reset_for_coord
                                    .load(Ordering::Acquire);
                                if reset_encoded != 0 {
                                    let workload_ns = workload_d.as_nanos() as u64;
                                    scenario_anchor =
                                        reset_encoded.saturating_sub(workload_ns).max(1);
                                    tracing::warn!(
                                        target: "ktstr::failure_dump",
                                        reset_encoded_ns = reset_encoded,
                                        workload_ns,
                                        derived_anchor_ns = scenario_anchor,
                                        "freeze-coord: scenario_start_ns never \
                                         stamped (guest's send_scenario_start may \
                                         have been lost — see periodic-capture \
                                         pipeline docs); deriving anchor from \
                                         observed scheduler-attach time"
                                    );
                                }
                            }
                            // resolve_periodic_window floats window_start to
                            // max(scenario_anchor, prereqs_ready), CLAMPS
                            // window_end to scenario_anchor + duration, and
                            // returns None to DEFER while scenario_anchor is
                            // still 0 — the inverted-window 0-fire guard (a 0
                            // anchor would make window_end = 0 + d <
                            // window_start, re-introducing the cold-boot
                            // 0-sample flake, and stamp anchor_ns = 0,
                            // mis-bucketing offsets). See the helper doc for
                            // the full rationale. On defer we leave
                            // periodic_boundaries_ns None and retry next
                            // iteration once scenario_start_ns or the
                            // watchdog_reset fallback supplies a non-zero
                            // anchor; prereqs_ready stays stamped so no rework
                            // is lost. anchor_ns is scenario_anchor (NOT
                            // window_start) so boundary offsets stay
                            // scenario-relative for build_phase_buckets. The
                            // slicer self-guards the degenerate/tiny window
                            // (returns empty).
                            if let Some(window) = resolve_periodic_window(
                                scenario_anchor,
                                periodic_prereqs_ready_ns,
                                workload_d.as_nanos() as u64,
                            ) {
                                let boundaries = compute_periodic_boundaries_ns(
                                    window.window_start_ns,
                                    window.window_end_ns,
                                    freeze_coord_num_snapshots,
                                );
                                tracing::info!(
                                    target: "ktstr::failure_dump",
                                    num_snapshots = freeze_coord_num_snapshots,
                                    scenario_anchor_ns = scenario_anchor,
                                    window_start_ns = window.window_start_ns,
                                    window_end_ns = window.window_end_ns,
                                    prereqs_ready_ns = periodic_prereqs_ready_ns,
                                    workload_duration_ns = workload_d.as_nanos() as u64,
                                    "freeze-coord: periodic snapshot boundaries computed"
                                );
                                periodic_anchor_ns = window.anchor_ns;
                                // Export for the post-run readiness-vs-window
                                // gate. `window_end_ns` is in the SAME
                                // run-relative frame as the prereq stamp,
                                // `.max(1)` vs the 0 sentinel.
                                periodic_window_end_for_coord
                                    .store(window.window_end_ns.max(1), Ordering::Release);
                                periodic_boundaries_ns = Some(boundaries);
                            }
                        }
                        // Advance the guest-progress clock once per tick: when
                        // the workload-progress heartbeat has published a newer
                        // scenario-elapsed than last seen, adopt it and stamp
                        // the wall time it advanced (the wedge-vs-load
                        // discriminator the fire loop reads).
                        {
                            let g = freeze_coord_periodic_guest_elapsed
                                .load(Ordering::Acquire);
                            if g > periodic_prev_guest_elapsed_ns {
                                periodic_prev_guest_elapsed_ns = g;
                                periodic_last_hb_wall_ns =
                                    u64::try_from(run_start.elapsed().as_nanos())
                                        .unwrap_or(u64::MAX);
                            }
                        }
                        if let Some(ref boundaries) = periodic_boundaries_ns {
                            // Drain every crossed boundary in this
                            // iteration. `now_ns` is recomputed at
                            // the top of every inner-loop iteration
                            // so a mid-drain ScenarioPause /
                            // ScenarioResume pair (a single periodic
                            // capture can run for several seconds
                            // through the parked-vCPU rendezvous)
                            // shifts un-fired boundaries forward as
                            // soon as the cumulative pause atomic
                            // updates.
                            loop {
                                if (next_periodic_idx as usize) >= boundaries.len() {
                                    break;
                                }
                                let raw_now_ns =
                                    u64::try_from(run_start.elapsed().as_nanos())
                                        .unwrap_or(u64::MAX);
                                let cumulative_pause = scenario_pause_cumulative_for_coord
                                    .load(Ordering::Acquire);
                                let in_flight_pause_at = watchdog_pause_for_coord
                                    .load(Ordering::Acquire);
                                let in_flight_pause = if in_flight_pause_at > 0 {
                                    raw_now_ns.saturating_sub(in_flight_pause_at)
                                } else {
                                    0
                                };
                                let now_ns = raw_now_ns
                                    .saturating_sub(cumulative_pause)
                                    .saturating_sub(in_flight_pause);
                                // Fire this boundary when the GUEST'S scenario
                                // clock reaches it (see the periodic-clock state
                                // above), not host wall-clock, so the fixed
                                // num_snapshots budget spreads across the guest's
                                // real phases under dilation. No heartbeat yet →
                                // pause-adjusted wall-clock, byte-identical to
                                // the prior behavior. Stale heartbeat (a wedge:
                                // heartbeats stopped) → resume on wall-clock so
                                // the stuck state is still captured. `boundaries`
                                // and `periodic_anchor_ns` are run-relative, so
                                // the boundary's scenario-relative offset shares
                                // the guest clock's frame (both 0 at scenario
                                // start).
                                let boundary_offset = boundaries[next_periodic_idx as usize]
                                    .saturating_sub(periodic_anchor_ns);
                                let effective_scenario_elapsed = if periodic_last_hb_wall_ns == 0 {
                                    now_ns.saturating_sub(periodic_anchor_ns)
                                } else {
                                    let stale =
                                        raw_now_ns.saturating_sub(periodic_last_hb_wall_ns);
                                    if stale > PERIODIC_HB_STALE_NS {
                                        periodic_prev_guest_elapsed_ns.saturating_add(stale)
                                    } else {
                                        periodic_prev_guest_elapsed_ns
                                    }
                                };
                                if boundary_offset > effective_scenario_elapsed {
                                    break;
                                }
                                if freeze_coord_kill.load(Ordering::Acquire) {
                                    break;
                                }
                                // Defer the boundary if the dump
                                // prerequisites haven't landed yet.
                                // The `dump_state` path inside the
                                // 'capture block resolves per-CPU
                                // KVAs (rq + cpustat + kstat +
                                // tick_sched) via `per_cpu_kva`
                                // (`template + kaslr_offset +
                                // per_cpu_off`); an unpublished
                                // `kern_virt_kaslr` reads as raw 0,
                                // collapsing the formula to the
                                // link-time path. Under a KASLR-on
                                // guest that resolves to wrong
                                // pages and reads zero values into
                                // `per_cpu_time` (a silent-data-
                                // loss class bug — the field is
                                // populated but every counter is
                                // 0). Same rationale as the ColdOp
                                // gate at L8013 and the WATCH gate
                                // at L7767: defer until the
                                // publisher (BSP MSR_LSTAR on
                                // x86_64 OR guest-channel KERN_ADDRS
                                // on both arches) lands. Under
                                // `nokaslr` the publisher still
                                // fires (with biased-zero) so the
                                // gate unblocks quickly; the
                                // predicate distinguishes "publish
                                // incomplete" from "publish complete
                                // with offset 0" by reading the raw
                                // atomic directly.
                                //
                                // `owned_accessor` is the second
                                // half of the prereq set: the FULL
                                // dump path at L6479 only enters
                                // when the accessor is adopted;
                                // otherwise the dump falls through
                                // to the PARTIAL path at L6838
                                // which hard-codes `per_cpu_time:
                                // Vec::new()` and skips every other
                                // walker output (maps, prog
                                // runtime, task enrichment, scx
                                // walker). Defer the boundary so
                                // the next iteration retries once
                                // the accessor-init worker
                                // publishes.
                                // The dump-side struct_ops walker
                                // (`identify_active_obj_from_struct_ops`)
                                // is only invoked when
                                // `owned_prog_accessor` is present —
                                // it powers the primary
                                // disambiguation path for same-binary
                                // scheduler swaps (multi-bss windows
                                // held alive by `Op::PinBpfMap`). If
                                // a periodic boundary fires before
                                // `owned_prog_accessor` adopts
                                // (worker still publishing OR
                                // coordinator hasn't reached the next
                                // scan_tick after a swap-induced
                                // reset cleared it), the report
                                // carries an empty `active_map_kvas`
                                // and the consumer surfaces
                                // `NoActiveScheduler` with the
                                // multi-bss diagnostic. The gate
                                // makes periodic captures wait for
                                // the prog accessor instead of
                                // emitting reports the walker cannot
                                // enrich.
                                if !kern_virt_kaslr_published()
                                    || owned_accessor.is_none()
                                    || owned_prog_accessor.is_none()
                                {
                                    tracing::info!(
                                        target: "ktstr::failure_dump",
                                        idx = next_periodic_idx,
                                        tag = %periodic_tag(next_periodic_idx),
                                        kaslr_published = kern_virt_kaslr_published(),
                                        accessor_present = owned_accessor.is_some(),
                                        prog_accessor_present = owned_prog_accessor.is_some(),
                                        "freeze-coord: periodic snapshot deferred \
                                         (kaslr_offset, owned_accessor, or \
                                         owned_prog_accessor not yet published); \
                                         retrying next iteration"
                                    );
                                    break;
                                }
                                // Window-B guard: the gate above ensures
                                // the accessor EXISTS, but a scheduler swap
                                // (Op::ReplaceScheduler / a same-binary rebind)
                                // the <= SCAN_INTERVAL scx_root watchpoint poll
                                // has not yet processed leaves owned_accessor
                                // bound to the PRIOR scheduler while *scx_root
                                // already points at the new obj. Reading the
                                // prior obj's .bss through the stale accessor
                                // returns the PRIOR scheduler's counters, not
                                // the live one's; and once the prior BPF
                                // object's maps are released (map refcount -> 0
                                // on scheduler-process exit / fd close — NOT
                                // synchronously by scx_root_disable
                                // (kernel/sched/ext.c), which only unlinks the
                                // scx_sched and NULLs *scx_root), that .bss page
                                // is recycled/zeroed, a silent nr_dispatched=0.
                                // Read *scx_root NOW
                                // and defer if it moved since the last watchpoint
                                // republish (last_sched_kva); the next iteration
                                // retries once the poll re-resolves the accessor
                                // for the live scheduler. Mirrors the watchpoint
                                // rebind detection but at the periodic boundary
                                // so a sub-SCAN_INTERVAL swap cannot emit a stale
                                // sample. Gated on scx_root being resolvable:
                                // when the symbol/offsets/mem are absent the dump
                                // degrades to a loud NoActiveScheduler, not a
                                // silent 0, so no guard is needed there.
                                if let Some(ref syms) = dump_cpu_time_symbols
                                    && let Some(scx_root_kva) = syms.scx_root
                                    && let Some(ref mem) = freeze_coord_mem
                                    && let Some(ref acc) = owned_accessor
                                {
                                    let root_pa =
                                        acc.guest_kernel().text_kva_to_pa(scx_root_kva);
                                    let live_sched_kva = mem.read_u64(root_pa, 0);
                                    if !periodic_accessor_current(last_sched_kva, live_sched_kva)
                                    {
                                        tracing::info!(
                                            target: "ktstr::failure_dump",
                                            idx = next_periodic_idx,
                                            tag = %periodic_tag(next_periodic_idx),
                                            last_sched_kva =
                                                format_args!("{last_sched_kva:#x}"),
                                            live_sched_kva =
                                                format_args!("{live_sched_kva:#x}"),
                                            "freeze-coord: periodic snapshot deferred \
                                             (scheduler swap not yet re-resolved by the \
                                             scx_root watchpoint poll; owned_accessor is \
                                             stale for the live scx_root); retrying next \
                                             iteration"
                                        );
                                        break;
                                    }
                                }
                                let _gate_guard = match gate::OnDemandGateGuard::try_acquire(
                                    &freeze_coord_on_demand_in_flight,
                                ) {
                                    Some(g) => g,
                                    None => {
                                        // Gate held — defer (do NOT
                                        // skip): leave next_periodic_idx
                                        // as-is so the next iteration
                                        // retries this same boundary
                                        // once the gate clears.
                                        tracing::info!(
                                            target: "ktstr::failure_dump",
                                            idx = next_periodic_idx,
                                            tag = %periodic_tag(next_periodic_idx),
                                            "freeze-coord: periodic snapshot deferred \
                                             (in-flight gate held by another capture); \
                                             retrying next iteration"
                                        );
                                        break;
                                    }
                                };
                                let tag = periodic_tag(next_periodic_idx);
                                tracing::info!(
                                    target: "ktstr::failure_dump",
                                    idx = next_periodic_idx,
                                    %tag,
                                    "freeze-coord: periodic snapshot boundary crossed"
                                );
                                // Request scx_stats BEFORE the freeze
                                // rendezvous so the scheduler's
                                // userspace thread is still alive to
                                // service the request. Failure modes
                                // (no scheduler, relay error, non-
                                // zero envelope errno) all collapse
                                // to `None` — the parallel stats
                                // slot stays absent and the
                                // temporal-stats projection surfaces
                                // a per-sample missing-stats failure
                                // the test author can opt to ignore.
                                let stats_value: Option<Result<
                                    serde_json::Value,
                                    crate::scenario::snapshot::MissingStatsReason,
                                >> = Some(if let Some(ref client) = freeze_coord_stats_client {
                                    match client.stats(&[]) {
                                        Ok(v) => Ok(v),
                                        Err(e) => {
                                            tracing::debug!(
                                                target: "ktstr::failure_dump",
                                                %tag,
                                                error = %e,
                                                "freeze-coord: periodic stats request \
                                                 failed; bundling typed reason into Sample"
                                            );
                                            Err((&e).into())
                                        }
                                    }
                                } else {
                                    Err(crate::scenario::snapshot::MissingStatsReason::NoSchedulerBinary)
                                });
                                // Sample timestamp anchor = the moment
                                // the stats request COMPLETED (or
                                // failed). Captured AFTER the stats
                                // client returns so the value
                                // reflects when the running
                                // scheduler's stats were observed,
                                // NOT when we entered the
                                // periodic-fire branch. Stats and
                                // BPF freeze can be ~50 ms apart;
                                // the stats-completion timestamp is
                                // the authoritative anchor for the
                                // sample because the JSON content
                                // was observed at this instant. The
                                // BPF state captured by the freeze
                                // that follows is observed up to
                                // FREEZE_RENDEZVOUS_TIMEOUT later.
                                //
                                // Pause-adjusted: subtract cumulative
                                // ScenarioPause/Resume pause time and
                                // any in-flight pause currently
                                // running, mirroring the boundary
                                // check above. Without this, a
                                // scenario that pauses (e.g. for a
                                // multi-second on-demand capture)
                                // would advance the elapsed_ms
                                // anchor through the pause window
                                // and the temporal patterns would
                                // see false-positive rate drops as
                                // the workload appears to "skip" a
                                // window of progress.
                                let anchor_raw_now_ns =
                                    u64::try_from(run_start.elapsed().as_nanos())
                                        .unwrap_or(u64::MAX);
                                let anchor_cumulative_pause =
                                    scenario_pause_cumulative_for_coord
                                        .load(Ordering::Acquire);
                                let anchor_in_flight_pause_at =
                                    watchdog_pause_for_coord.load(Ordering::Acquire);
                                let anchor_in_flight_pause =
                                    if anchor_in_flight_pause_at > 0 {
                                        anchor_raw_now_ns
                                            .saturating_sub(anchor_in_flight_pause_at)
                                    } else {
                                        0
                                    };
                                let sample_elapsed_ms_anchor = anchor_raw_now_ns
                                    .saturating_sub(anchor_cumulative_pause)
                                    .saturating_sub(anchor_in_flight_pause)
                                    / 1_000_000;
                                let freeze_start = Instant::now();
                                let on_demand = freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false });
                                thaw_and_barrier();
                                extend_watchdog_for_freeze(freeze_start);
                                match on_demand {
                                    FreezeOutcome::Captured(report, capture_start) => {
                                    let map_count = report.maps.len();
                                    let vcpu_regs_count = report.vcpu_regs.len();
                                    let tasks_enriched = report.task_enrichments.len();
                                    if freeze_coord_dump_path.is_some() {
                                        match serde_json::to_string(&report) {
                                            Ok(json) => {
                                                let _ = write_to_tagged_path(
                                                    freeze_coord_dump_path.as_deref(),
                                                    &tag,
                                                    &json,
                                                    || {
                                                        format!(
                                                            "{SNAPSHOT_SUMMARY_PREFIX} (on-demand periodic, write failed): idx={next_periodic_idx} tag={tag} map_count={map_count} vcpu_regs_count={vcpu_regs_count} tasks_enriched={tasks_enriched} elapsed_ms={} json_bytes={}",
                                                            capture_start.elapsed().as_millis() as u64,
                                                            json.len(),
                                                        )
                                                    },
                                                    "freeze-coord: on-demand periodic dump file write failed",
                                                );
                                            }
                                            Err(e) => tracing::error!(
                                                error = %e,
                                                map_count,
                                                vcpu_regs_count,
                                                "freeze-coord: on-demand periodic dump (JSON serialization failed)"
                                            ),
                                        }
                                    }
                                    let elapsed_ms =
                                        capture_start.elapsed().as_millis() as u64;
                                    tracing::info!(
                                        target: "ktstr::failure_dump",
                                        kind = "periodic",
                                        idx = next_periodic_idx,
                                        %tag,
                                        map_count,
                                        vcpu_regs_count,
                                        tasks_enriched,
                                        elapsed_ms,
                                        stats_present = matches!(stats_value, Some(Ok(_))),
                                        sample_elapsed_ms = sample_elapsed_ms_anchor,
                                        "freeze-coord: periodic snapshot captured"
                                    );
                                    // Stamp the periodic capture with the
                                    // current scenario phase (1-indexed:
                                    // 0 = BASELINE, 1..=N = Step ordinals)
                                    // read from the host-side mirror the
                                    // dispatch loop populates on every
                                    // CRC-valid MSG_TYPE_STIMULUS frame.
                                    // `Acquire` pairs with the dispatch
                                    // loop's `Release` store so the
                                    // most-recently-published step is
                                    // visible here. A periodic boundary
                                    // that fires before any stimulus
                                    // event reads `0` (BASELINE) -- the
                                    // settle-window initial value --
                                    // which is the correct bucket for a
                                    // pre-first-Step sample.
                                    let phase_step_index = freeze_coord_current_step
                                        .load(Ordering::Acquire);
                                    freeze_coord_snapshot_bridge.store_with_stats_and_step(
                                        &tag,
                                        report,
                                        stats_value,
                                        Some(sample_elapsed_ms_anchor),
                                        Some(
                                            boundaries[next_periodic_idx as usize]
                                                .saturating_sub(periodic_anchor_ns)
                                                / 1_000_000,
                                        ),
                                        phase_step_index,
                                    );
                                    // Successful capture resets the
                                    // consecutive-timeout counter so
                                    // a transient rendezvous miss
                                    // does not arm the abandon
                                    // threshold for unrelated future
                                    // boundaries.
                                    periodic_consecutive_timeouts = 0;
                                    }
                                    FreezeOutcome::Degraded(degraded) => {
                                        // Periodic rendezvous timed out.
                                        // Preserve the structured trigger
                                        // state by writing the degraded JSON
                                        // to the periodic tagged path AND
                                        // storing a placeholder on the
                                        // bridge that carries the partial
                                        // vcpu_regs + the degraded reason so
                                        // a temporal-pattern consumer sees
                                        // why the periodic sample is missing
                                        // BPF data.
                                        if freeze_coord_dump_path.is_some() {
                                            match serde_json::to_string(degraded.as_ref()) {
                                                Ok(json) => {
                                                    let _ = write_to_tagged_path(
                                                        freeze_coord_dump_path.as_deref(),
                                                        &tag,
                                                        &json,
                                                        || {
                                                            format!(
                                                                "{SNAPSHOT_SUMMARY_PREFIX} (on-demand periodic Degraded, write failed): idx={next_periodic_idx} tag={tag} reason={:?} json_bytes={}",
                                                                degraded.reason,
                                                                json.len(),
                                                            )
                                                        },
                                                        "freeze-coord: on-demand periodic degraded dump file write failed",
                                                    );
                                                }
                                                Err(e) => tracing::error!(
                                                    error = %e,
                                                    "freeze-coord: on-demand periodic degraded dump (JSON serialization failed)"
                                                ),
                                            }
                                        }
                                        tracing::warn!(
                                            idx = next_periodic_idx,
                                            %tag,
                                            reason = %degraded.reason,
                                            "freeze-coord: periodic capture degraded (rendezvous timeout); storing placeholder + degraded file"
                                        );
                                        let placeholder = crate::monitor::dump::FailureDumpReport {
                                            vcpu_regs: degraded.vcpu_regs.clone(),
                                            ..crate::monitor::dump::FailureDumpReport::placeholder(
                                                degraded.reason.clone(),
                                            )
                                        };
                                        // Stamp the degraded placeholder
                                        // with the current scenario phase
                                        // for the same reason the Captured
                                        // path does: a degraded sample
                                        // belongs to the phase the
                                        // periodic boundary fired in, NOT
                                        // BASELINE (which is what the
                                        // bucket renderer's "no stamped
                                        // index" fallback would land it
                                        // in). Without this stamp,
                                        // rendezvous timeouts would
                                        // silently shift their placeholders
                                        // into the wrong phase bucket.
                                        let degraded_phase_step_index =
                                            freeze_coord_current_step.load(Ordering::Acquire);
                                        freeze_coord_snapshot_bridge.store_with_stats_and_step(
                                            &tag,
                                            placeholder,
                                            stats_value,
                                            Some(sample_elapsed_ms_anchor),
                                            Some(
                                                boundaries[next_periodic_idx as usize]
                                                    .saturating_sub(periodic_anchor_ns)
                                                    / 1_000_000,
                                            ),
                                            degraded_phase_step_index,
                                        );
                                        periodic_consecutive_timeouts =
                                            periodic_consecutive_timeouts.saturating_add(1);
                                    }
                                    FreezeOutcome::Suppressed => {
                                        // Unreachable: gate only runs with
                                        // gate_on_exit_kind=true and the
                                        // periodic path always passes false.
                                        tracing::warn!(
                                            idx = next_periodic_idx,
                                            %tag,
                                            "freeze-coord: periodic capture unexpectedly suppressed by gate"
                                        );
                                        let placeholder =
                                            crate::monitor::dump::FailureDumpReport::placeholder(
                                                "periodic capture suppressed (gate decision)",
                                            );
                                        // Stamp the suppressed placeholder
                                        // with the current scenario phase
                                        // for the same reason as the
                                        // Degraded path above. Even on the
                                        // unreachable arm, a placeholder
                                        // that ever lands in the bridge
                                        // must carry the phase so the
                                        // bucket renderer's fallback does
                                        // not silently mislabel it as
                                        // BASELINE.
                                        let suppressed_phase_step_index =
                                            freeze_coord_current_step.load(Ordering::Acquire);
                                        freeze_coord_snapshot_bridge.store_with_stats_and_step(
                                            &tag,
                                            placeholder,
                                            stats_value,
                                            Some(sample_elapsed_ms_anchor),
                                            Some(
                                                boundaries[next_periodic_idx as usize]
                                                    .saturating_sub(periodic_anchor_ns)
                                                    / 1_000_000,
                                            ),
                                            suppressed_phase_step_index,
                                        );
                                    }
                                    FreezeOutcome::KernelOp(_) => capture_only_unreachable!(),
                                }
                                next_periodic_idx =
                                    next_periodic_idx.saturating_add(1);
                                // Publish the live fire count so
                                // run_vm can read it after the
                                // coordinator joins and forward
                                // onto VmResult::periodic_fired.
                                periodic_fired_for_coord.store(
                                    next_periodic_idx,
                                    Ordering::Relaxed,
                                );
                                // After PERIODIC_TIMEOUT_ABANDON_THRESHOLD
                                // consecutive rendezvous timeouts the
                                // remaining boundaries are unlikely
                                // to produce useful captures — every
                                // fire costs up to
                                // FREEZE_RENDEZVOUS_TIMEOUT (30 s)
                                // of wall-clock waiting on a wedged
                                // guest. Set the abandon flag and
                                // break the inner drain; the outer
                                // periodic guard short-circuits on
                                // the next iteration.
                                if periodic_consecutive_timeouts
                                    >= PERIODIC_TIMEOUT_ABANDON_THRESHOLD
                                    && !periodic_abandoned
                                {
                                    let remaining = boundaries
                                        .len()
                                        .saturating_sub(next_periodic_idx as usize);
                                    tracing::warn!(
                                        target: "ktstr::failure_dump",
                                        consecutive_timeouts =
                                            periodic_consecutive_timeouts,
                                        threshold =
                                            PERIODIC_TIMEOUT_ABANDON_THRESHOLD,
                                        remaining_boundaries = remaining,
                                        "freeze-coord: periodic capture abandoned \
                                         after {} consecutive rendezvous timeouts \
                                         ({} boundaries skipped)",
                                        periodic_consecutive_timeouts,
                                        remaining,
                                    );
                                    periodic_abandoned = true;
                                    break;
                                }
                            }
                        }
                    }
                    // After every TLV-driven snapshot dispatch path
                    // runs, also service any user-watchpoint hits on
                    // slots 1..=3.
                    // The vCPU's KVM_EXIT_DEBUG handler latches the
                    // matching slot's `hit` flag and writes hit_evt;
                    // the coordinator's epoll fires WATCHPOINT, the
                    // hit_evt drain at the top of the loop already
                    // ran. Walk every slot and dispatch a capture
                    // for each hit.
                    for slot_idx in 0..3 {
                        if !freeze_coord_watchpoint.user[slot_idx]
                            .hit
                            .swap(false, Ordering::AcqRel)
                        {
                            continue;
                        }
                        let tag = freeze_coord_watchpoint.user[slot_idx]
                            .tag
                            .lock_unpoisoned()
                            .clone();
                        let _gate_guard = match gate::OnDemandGateGuard::try_acquire(
                            &freeze_coord_on_demand_in_flight,
                        ) {
                            Some(g) => g,
                            None => {
                                // A capture is already in flight (e.g.
                                // a CAPTURE-class TLV request still
                                // holds the gate). Re-arm the slot's
                                // hit flag so a subsequent iteration
                                // services it, and write a fresh
                                // `hit_evt` edge so the outer
                                // `epoll.wait` wakes promptly — the
                                // hit_evt drain at the top of this
                                // iteration consumed the original wake,
                                // and without a new edge the re-armed
                                // hit could sit for the full
                                // POLL_TIMEOUT_MS before re-inspection.
                                // `continue` (rather than `break`) so
                                // OTHER slots in the same iteration
                                // still get checked — each slot's
                                // `hit` is independent (per-slot
                                // hardware watchpoint dispatch), so a
                                // gate-blocked slot N must not strand
                                // an unrelated fire on slot N+1
                                // waiting for the next iteration's
                                // wake. The outer loop's next iteration
                                // re-evaluates the gate and either
                                // services the re-armed slot or hits
                                // the same in-flight branch and
                                // re-arms again — bounded by the
                                // single-threaded freeze coordinator's
                                // serial dispatch of CAPTURE/WATCH,
                                // which always clears the gate before
                                // returning here.
                                freeze_coord_watchpoint.user[slot_idx]
                                    .hit
                                    .store(true, Ordering::Release);
                                let _ = freeze_coord_watchpoint.hit_evt.write(1);
                                continue;
                            }
                        };
                        tracing::info!(
                            slot_idx,
                            %tag,
                            "freeze-coord: user watchpoint fire; capturing"
                        );
                        // Snapshot the current scenario phase NOW —
                        // BEFORE the freeze rendezvous + file write +
                        // tracing emit that follow. The trip moment
                        // is when the vCPU observed KVM_EXIT_DEBUG (it
                        // set the hit flag and woke this loop). The
                        // closest the coord thread can get to that
                        // moment is here, immediately after we observed
                        // the swap on .hit and acquired the gate guard.
                        // Capturing later (e.g. just before the bridge
                        // store at the Captured/Degraded/Suppressed
                        // arms) would let the scenario driver's
                        // `current_step.store(Release)` race ahead
                        // during the freeze+file-write window — under
                        // slow IO that drift can stamp step-k trip
                        // data into a step-(k+1) bucket, exactly the
                        // silent-mis-attribution this stamping was
                        // meant to prevent. Acquire pairs with the
                        // scenario writer's Release at
                        // `src/scenario/ops/mod.rs:1194`.
                        let trip_phase_step_index =
                            freeze_coord_current_step.load(Ordering::Acquire);
                        // User watchpoint has no while-frozen work,
                        // so thaw immediately.
                        let on_demand = freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false });
                        thaw_and_barrier();
                        match on_demand {
                            FreezeOutcome::Captured(report, capture_start) => {
                            let map_count = report.maps.len();
                            // File mirror via `&report` (no clone),
                            // then move the report into the bridge.
                            // See the CAPTURE-class TLV handler
                            // above for the full rationale on the
                            // serialize-then-store ordering and the
                            // `to_string` vs `to_string_pretty`
                            // tradeoff.
                            if freeze_coord_dump_path.is_some() {
                                match serde_json::to_string(&report) {
                                    Ok(json) => {
                                        let _ = write_to_tagged_path(
                                            freeze_coord_dump_path.as_deref(),
                                            &tag,
                                            &json,
                                            || {
                                                format!(
                                                    "{SNAPSHOT_SUMMARY_PREFIX} (on-demand user-watchpoint, write failed): slot_idx={slot_idx} tag={tag} map_count={map_count} elapsed_ms={} json_bytes={}",
                                                    capture_start.elapsed().as_millis() as u64,
                                                    json.len(),
                                                )
                                            },
                                            "freeze-coord: on-demand user-watchpoint dump file write failed",
                                        );
                                    }
                                    Err(e) => tracing::error!(
                                        error = %e,
                                        map_count,
                                        "freeze-coord: on-demand user-watchpoint dump (JSON serialization failed)"
                                    ),
                                }
                            }
                            let elapsed_ms =
                                capture_start.elapsed().as_millis() as u64;
                            tracing::info!(
                                target: "ktstr::failure_dump",
                                kind = "user_watchpoint",
                                slot_idx,
                                %tag,
                                map_count,
                                elapsed_ms,
                                "freeze-coord: user-watchpoint snapshot captured"
                            );
                            // Use `trip_phase_step_index` captured at
                            // the top of this iteration (immediately
                            // after observing the hit swap, before the
                            // freeze+file-write window). Stamp from
                            // CURRENT_STEP atomic at trip moment — NOT
                            // at bridge-store moment, which can drift
                            // by the freeze+IO latency.
                            freeze_coord_snapshot_bridge.store_with_stats_and_step(
                                &tag,
                                report,
                                None,
                                None,
                                // On-demand watchpoint: no periodic boundary,
                                // so no workload-relative offset; the trip
                                // phase below is authoritative.
                                None,
                                trip_phase_step_index,
                            );
                            }
                            FreezeOutcome::Degraded(degraded) => {
                                // User-watchpoint rendezvous timeout.
                                // Preserve the trigger state by writing
                                // the degraded JSON to the tagged path
                                // AND storing a placeholder carrying
                                // partial vcpu_regs + the degraded
                                // reason so a test that registered
                                // `Op::WatchSnapshot` reads a
                                // structured entry from the bridge
                                // instead of silently missing data.
                                if freeze_coord_dump_path.is_some() {
                                    match serde_json::to_string(degraded.as_ref()) {
                                        Ok(json) => {
                                            let _ = write_to_tagged_path(
                                                freeze_coord_dump_path.as_deref(),
                                                &tag,
                                                &json,
                                                || {
                                                    format!(
                                                        "{SNAPSHOT_SUMMARY_PREFIX} (on-demand user-watchpoint Degraded, write failed): slot_idx={slot_idx} tag={tag} reason={:?} json_bytes={}",
                                                        degraded.reason,
                                                        json.len(),
                                                    )
                                                },
                                                "freeze-coord: on-demand user-watchpoint degraded dump file write failed",
                                            );
                                        }
                                        Err(e) => tracing::error!(
                                            error = %e,
                                            "freeze-coord: on-demand user-watchpoint degraded dump (JSON serialization failed)"
                                        ),
                                    }
                                }
                                tracing::warn!(
                                    slot_idx,
                                    %tag,
                                    reason = %degraded.reason,
                                    "freeze-coord: user-watchpoint capture degraded (rendezvous timeout); storing placeholder + degraded file"
                                );
                                let placeholder = crate::monitor::dump::FailureDumpReport {
                                    vcpu_regs: degraded.vcpu_regs.clone(),
                                    ..crate::monitor::dump::FailureDumpReport::placeholder(
                                        degraded.reason.clone(),
                                    )
                                };
                                // Reuse trip_phase_step_index captured
                                // at the top of this iteration. Even a
                                // degraded placeholder must carry the
                                // TRIP phase, not a post-freeze-rendezvous
                                // re-read of current_step which can drift
                                // ahead during the freeze+placeholder
                                // assembly window.
                                freeze_coord_snapshot_bridge.store_with_stats_and_step(
                                    &tag,
                                    placeholder,
                                    None,
                                    None,
                                    // On-demand watchpoint: no periodic
                                    // boundary, so no workload-relative
                                    // offset; the trip phase is authoritative.
                                    None,
                                    trip_phase_step_index,
                                );
                            }
                            FreezeOutcome::Suppressed => {
                                // Unreachable: user-watchpoint passes
                                // gate_on_exit_kind=false. Defensive:
                                // publish a placeholder so the test
                                // sees an entry rather than a missing
                                // snapshot.
                                tracing::warn!(
                                    slot_idx,
                                    %tag,
                                    "freeze-coord: user-watchpoint capture unexpectedly suppressed by gate"
                                );
                                let placeholder =
                                    crate::monitor::dump::FailureDumpReport::placeholder(
                                        "user-watchpoint capture suppressed (gate decision)",
                                    );
                                // Reuse trip_phase_step_index captured
                                // at the top of this iteration. The
                                // Suppressed path is reached on the
                                // unreachable arm (user-watchpoint
                                // passes gate_on_exit_kind=false), but
                                // if it ever fires the placeholder
                                // still needs the TRIP phase, not the
                                // post-rendezvous re-read.
                                freeze_coord_snapshot_bridge.store_with_stats_and_step(
                                    &tag,
                                    placeholder,
                                    None,
                                    None,
                                    // On-demand watchpoint: no periodic
                                    // boundary, so no workload-relative
                                    // offset; the trip phase is authoritative.
                                    None,
                                    trip_phase_step_index,
                                );
                            }
                            FreezeOutcome::KernelOp(_) => capture_only_unreachable!(),
                        }
                        // Release the slot for future arm requests.
                        // `arm_user_watchpoint` finds a free slot by
                        // `request_kva.load(Acquire) == 0`; without
                        // clearing here every fire permanently consumes
                        // its slot, exhausting the cap of three after
                        // three captures and rejecting subsequent
                        // `Op::WatchSnapshot` arms with "no free slot".
                        // Clear `request_kva` and `tag` together so
                        // `arm_user_watchpoint`'s tag publish ordering
                        // (tag first, then `request_kva` Release) sees
                        // a clean slot; vCPU `self_arm_watchpoint` calls
                        // observe the zeroed `request_kva` next iteration
                        // and re-issue `KVM_SET_GUEST_DEBUG` without
                        // this slot's DR/WCR enable so the now-stale
                        // KVA stops trapping. `Release` pairs with the
                        // `Acquire` in `arm_user_watchpoint`'s free-slot
                        // search and the per-vCPU `self_arm_watchpoint`
                        // load.
                        //
                        // Panic-safety: per-slot cleanup runs while
                        // `_gate_guard` is still alive. A panic in
                        // `lock_unpoisoned()` or `tag_guard.clear()`
                        // unwinds through the guard's `Drop`, which
                        // clears the gate before unwinding out of
                        // the slot scope. Closes the deadlock-on-
                        // panic class (see [`gate::OnDemandGateGuard`]
                        // module doc).
                        {
                            let mut tag_guard = freeze_coord_watchpoint
                                .user[slot_idx]
                                .tag
                                .lock_unpoisoned();
                            tag_guard.clear();
                        }
                        freeze_coord_watchpoint.user[slot_idx]
                            .request_kva
                            .store(0, Ordering::Release);
                    }
                    // Once the late snapshot has been emitted, the
                    // coordinator's only remaining job is to keep
                    // the freeze=false invariant clear, service
                    // any pending TLV snapshot requests, and wait
                    // for teardown. Skip the error-trigger paths
                    // below; the next `epoll.wait` at the top of
                    // the loop blocks until kill / bsp_done /
                    // virtio-console TX / watchpoint / scanner
                    // tick — no separate sleep cadence needed.
                    // Goes AFTER the snapshot-request dispatch so
                    // on-demand captures still service post-Done.
                    if freeze_state == FreezeState::Done {
                        continue;
                    }
                    // Early-snapshot trigger: dual_snapshot mode and
                    // we have a working scan context. Mirror the
                    // kernel's `check_rq_for_timeouts` logic — any
                    // task whose `jiffies - p->scx.runnable_at`
                    // exceeds the half-way mark trips the trigger.
                    // Half-way comes from the configured
                    // watchdog_timeout (already plumbed through
                    // `KtstrTestEntry.watchdog_timeout`), so the
                    // early snapshot lands well before the kernel
                    // would emit SCX_EXIT_ERROR_STALL — gives the
                    // operator pre-stall BPF state to diff against
                    // the late snapshot.
                    if scan_tick
                        && freeze_state == FreezeState::Idle
                        && freeze_coord_dual_snapshot
                        && half_threshold_jiffies > 0
                        && let Some(ref ctx) = scan_ctx
                        && let Some(ref mem) = freeze_coord_mem
                    {
                        let jiffies = mem.read_u64(ctx.jiffies_64_pa, 0);
                        let max_age = crate::monitor::runnable_scan::max_runnable_age(
                            mem,
                            ctx.scx_tasks_kva,
                            &ctx.rq_pas,
                            &ctx.offsets,
                            jiffies,
                            ctx.walk,
                            ctx.watchdog_timestamp_pa,
                            ctx.start_kernel_map,
                            ctx.phys_base,
                            ctx.kaslr_offset,
                        );
                        // Track scan trajectory for the diagnostic
                        // logged when err_triggered fires before the
                        // early path captures. peak survives across
                        // iterations even when each individual
                        // max_age dips back to 0 (a task on the list
                        // gets dispatched between two polls), so an
                        // operator viewing the post-hoc warn sees the
                        // closest the run came to tripping the
                        // threshold.
                        early_scan_iters = early_scan_iters.wrapping_add(1);
                        if max_age > early_peak_max_age_jiffies {
                            early_peak_max_age_jiffies = max_age;
                        }
                        if max_age >= half_threshold_jiffies
                            && !freeze_coord_bsp_done.load(Ordering::Acquire)
                        {
                            tracing::info!(
                                max_age,
                                half_threshold_jiffies,
                                "freeze-coord: dual-snapshot early threshold tripped"
                            );
                            // Persist the trigger metric and the
                            // half-way threshold ONLY when the freeze
                            // capture succeeds. The
                            // `DualFailureDumpReport` doc says "Zero
                            // when `early` is `None`", which a
                            // consumer relies on to detect the
                            // capture-failed case from JSON alone:
                            // a `late`-only wrapper with non-zero
                            // metric values would be ambiguous (did
                            // the early capture fail, or did the
                            // trigger never fire?). Co-gating both
                            // sides on `Some(report)` keeps the
                            // invariant.
                            // Early-trigger only persists the report;
                            // the timing summary line is emitted at
                            // the late-trigger emit_json site (which
                            // is where JSON serialisation happens).
                            // Discarding the early `_capture_start`
                            // avoids a separate timing log for the
                            // early path that would not include
                            // json_bytes.
                            // Early trigger uses runnable_at age as
                            // its precondition; exit_kind has not
                            // necessarily been written yet, so pass
                            // `false` to skip the gate. Early
                            // snapshot has no while-frozen work, so
                            // thaw immediately after the dump
                            // returns. A degraded outcome here
                            // (rendezvous timed out during the early
                            // capture) must surface structured
                            // diagnostics rather than be dropped. The
                            // late path will follow up
                            // (FreezeState::TookEarly → Late) so the
                            // dual-snapshot wrapper can still emit a
                            // Single+Degraded composite from the late
                            // arm; here we write a tagged degraded
                            // JSON so the early-half trigger evidence
                            // survives to disk.
                            match freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false }) {
                                FreezeOutcome::Captured(report, _capture_start) => {
                                    early_max_age_jiffies = max_age;
                                    early_threshold_jiffies = half_threshold_jiffies;
                                    early_guard.snapshot = Some(report);
                                }
                                FreezeOutcome::Degraded(degraded) => {
                                    // Stash the degraded reason so the
                                    // late-arm's early_skipped_reason
                                    // calculator can surface it via the
                                    // dual wrapper. Clone before the
                                    // tagged-write block consumes
                                    // `degraded.reason` via `{:?}` and
                                    // the warn's `%` formatting.
                                    early_degraded_reason =
                                        Some(degraded.reason.clone());
                                    if freeze_coord_dump_path.is_some() {
                                        match serde_json::to_string(degraded.as_ref()) {
                                            Ok(json) => {
                                                let _ = write_to_tagged_path(
                                                    freeze_coord_dump_path.as_deref(),
                                                    crate::monitor::dump::SNAPSHOT_TAG_EARLY_DEGRADED,
                                                    &json,
                                                    || {
                                                        format!(
                                                            "{SNAPSHOT_SUMMARY_PREFIX} (on-demand early-snapshot Degraded, write failed): tag={} reason={:?} json_bytes={}",
                                                            crate::monitor::dump::SNAPSHOT_TAG_EARLY_DEGRADED,
                                                            degraded.reason,
                                                            json.len(),
                                                        )
                                                    },
                                                    "freeze-coord: on-demand early-snapshot degraded dump file write failed",
                                                );
                                            }
                                            Err(e) => tracing::error!(
                                                error = %e,
                                                "freeze-coord: on-demand early-snapshot degraded dump (JSON serialization failed)"
                                            ),
                                        }
                                    }
                                    tracing::warn!(
                                        reason = %degraded.reason,
                                        "freeze-coord: early-snapshot degraded (rendezvous timeout); late path will retry"
                                    );
                                }
                                FreezeOutcome::Suppressed => {
                                    // Symmetric with the other three
                                    // `FreezeOutcome::Suppressed`
                                    // arms in this file (TLV CAPTURE
                                    // / periodic / user-watchpoint
                                    // dispatch sites — grep
                                    // `FreezeOutcome::Suppressed`
                                    // to find them), which all log +
                                    // continue rather than panic.
                                    // Suppressed is only producible by
                                    // `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: true })` (the
                                    // exit_kind gate path); the early
                                    // path passes false so this arm is
                                    // unreachable under the current
                                    // invariant. But a future refactor
                                    // changing the bool default — or
                                    // any flag-passing bug — would
                                    // crash the entire VM here instead
                                    // of degrading gracefully. Log
                                    // loudly + continue: early_snapshot
                                    // stays None for this cycle, the
                                    // late path runs normally. `warn`
                                    // matches the other three sites'
                                    // "defensive log + continue" level
                                    // so a monitoring rule catches all
                                    // four uniformly.
                                    tracing::warn!(
                                        "freeze-coord: invariant violated — early-snapshot freeze_and_dispatch(FreezeMode::Capture {{ gate_on_exit_kind: false }}) returned Suppressed (expected only when gate_on_exit_kind=true). Continuing without early snapshot."
                                    );
                                }
                                FreezeOutcome::KernelOp(_) => capture_only_unreachable!(),
                            }
                            thaw_and_barrier();
                            freeze_state = FreezeState::TookEarly;
                        }
                    }
                    // Late-snapshot trigger: err_exit_detected has
                    // flipped. The state-machine guard ensures we
                    // only fire once per VM run — TookEarly → late
                    // is allowed (capturing both halves of the
                    // dual-snapshot wrapper); Done is terminal.
                    if err_triggered
                        && (freeze_state == FreezeState::Idle
                            || freeze_state == FreezeState::TookEarly)
                    {
                        tracing::info!(
                            "freeze-coord: ktstr_err_exit_detected latched, freezing vCPUs"
                        );
                        // When dual-snapshot mode is on but the early
                        // path never captured, surface why so the
                        // operator can act without re-running with
                        // RUST_LOG=ktstr=debug. The three diagnoses
                        // (no scan_ctx, scan ran but always-zero,
                        // scan ran but never crossed threshold) map
                        // to distinct fixes: the first points at
                        // missing kernel symbols / BTF, the second
                        // points at offset/translation bugs in the
                        // scan, the third points at err-class exits
                        // that aren't watchdog stalls (where there
                        // is no half-way state to capture). The warn
                        // fires only when state is genuinely Idle —
                        // a successful TookEarly path has already
                        // logged at info level above.
                        if freeze_coord_dual_snapshot
                            && freeze_state == FreezeState::Idle
                        {
                            tracing::warn!(
                                early_scan_iters,
                                early_peak_max_age_jiffies,
                                half_threshold_jiffies,
                                scan_ctx_resolved = scan_ctx.is_some(),
                                "freeze-coord: dual-snapshot late firing without \
                                 early — runnable_at scan never crossed half-way \
                                 threshold (peak_max_age vs half_threshold tells \
                                 you which case: 0 peak with 0 iters = scan_ctx \
                                 unresolved; 0 peak with non-zero iters = scan ran \
                                 but found no aged tasks; non-zero peak under \
                                 threshold = err-class exit fired before stall \
                                 progressed past half-way)"
                            );
                        }
                        // Gate the dump on `*scx_root->exit_kind`
                        // when the watchpoint was the trigger. The
                        // hardware watchpoint catches every write,
                        // including transient init/teardown writes
                        // setting kind to NONE/DONE; gating on
                        // `kind >= 1024` (SCX_EXIT_ERROR boundary)
                        // suppresses those false positives. The BPF
                        // bss path is its own gate (the tp_btf
                        // handler only latches on error-class kinds),
                        // so when bss alone fired the gate is
                        // redundant and we let the dump run
                        // unconditionally — `bss_state == Triggered`
                        // already proves kind >= 1024.
                        let watchpoint_only_trigger =
                            compute_watchpoint_only_trigger(
                                watchpoint_hit, bss_state,
                            );
                        let late_capture =
                            freeze_and_dispatch(FreezeMode::Capture {
                                gate_on_exit_kind: watchpoint_only_trigger,
                            });
                        // Late-trigger backstop: while guest memory
                        // is still quiesced (vCPUs parked, virtio-blk
                        // worker paused, freeze flag still set), do a
                        // final runnable_at scan and — if it crosses
                        // the threshold and the early snapshot never
                        // captured — clone the just-captured late
                        // report into the early slot. The early and
                        // late slots end up as identical snapshots in
                        // that case, but the wrapper's
                        // `early_max_age_jiffies` /
                        // `early_threshold_jiffies` fields tell the
                        // consumer the trigger condition was met at
                        // freeze time, and the wrapper Display
                        // surfaces "early=present" rather than
                        // "early=absent" so an operator inspecting a
                        // stall dump sees the runnable_at evidence
                        // even when the host coordinator's poll
                        // cadence missed the half-way crossing.
                        //
                        // The backstop runs unconditionally on a
                        // quiesced guest — same memory the dump just
                        // captured — so a positive max_age here is
                        // ground truth for "tasks were stuck on
                        // runnable_list at the error-exit instant",
                        // not a transient observation that could have
                        // dipped before the next poll. Functionally
                        // independent of (and complementary to) the
                        // per-poll early trigger above: the per-poll
                        // path captures the half-way moment; the
                        // backstop captures the late-instant ground
                        // truth.
                        let mut backstop_max_age: u64 = 0;
                        if freeze_coord_dual_snapshot
                            && early_guard.snapshot.is_none()
                            && half_threshold_jiffies > 0
                            && let FreezeOutcome::Captured(ref late, _) = late_capture
                            && let Some(ref ctx) = scan_ctx
                            && let Some(ref mem) = freeze_coord_mem
                        {
                            let jiffies = mem.read_u64(ctx.jiffies_64_pa, 0);
                            backstop_max_age =
                                crate::monitor::runnable_scan::max_runnable_age(
                                    mem,
                                    ctx.scx_tasks_kva,
                                    &ctx.rq_pas,
                                    &ctx.offsets,
                                    jiffies,
                                    ctx.walk,
                                    ctx.watchdog_timestamp_pa,
                                    ctx.start_kernel_map,
                                    ctx.phys_base,
                                    ctx.kaslr_offset,
                                );
                            if backstop_max_age >= half_threshold_jiffies {
                                tracing::info!(
                                    backstop_max_age,
                                    half_threshold_jiffies,
                                    "freeze-coord: late-trigger backstop \
                                     promoting late capture to early slot \
                                     (per-poll early path missed the \
                                     half-way crossing — runnable_at scan \
                                     of frozen guest memory shows the \
                                     stall was real)"
                                );
                                early_guard.snapshot = Some(late.clone());
                                early_max_age_jiffies = backstop_max_age;
                                early_threshold_jiffies = half_threshold_jiffies;
                            }
                        }
                        // Compute the structured early-skip reason
                        // BEFORE thaw, while the relevant state
                        // (peak, threshold, scan_ctx, skip_reason) is
                        // current. The reason is consumed when
                        // building the DualFailureDumpReport below; a
                        // None means "early was captured" or
                        // "single-snapshot mode" — the dual wrapper
                        // serializes None via skip_serializing_if so
                        // a populated `early` keeps the JSON tight.
                        let early_skipped_reason: Option<String> =
                            if !freeze_coord_dual_snapshot
                                || early_guard.snapshot.is_some()
                            {
                                None
                            } else if let Some(reason) =
                                early_degraded_reason.as_deref()
                            {
                                // Early-trigger DID fire (max_age
                                // crossed the half threshold) but
                                // freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false }) returned
                                // Degraded. The early dump landed on
                                // disk at the
                                // SNAPSHOT_TAG_EARLY_DEGRADED sibling
                                // (per the early-Degraded arm above);
                                // surface that here so a dual-wrapper
                                // consumer reading the JSON knows the
                                // structured data exists out-of-band
                                // and the reason field carries the
                                // upstream cause. Branch precedence is
                                // intentional: this fires only when the
                                // early-Degraded arm actually ran, and
                                // ordering BEFORE the
                                // `scan_ctx_skip_reason` branch is safe
                                // because an early-Degraded outcome
                                // implies scan_ctx WAS Some (the
                                // early-trigger guard requires it) —
                                // scan_ctx_skip_reason cannot have
                                // populated.
                                Some(format!(
                                    "early capture degraded ({reason}); \
                                     see tagged sibling at \
                                     SNAPSHOT_TAG_EARLY_DEGRADED"
                                ))
                            } else if let Some(reason) = scan_ctx_skip_reason {
                                Some(format!(
                                    "scan prerequisites unavailable: {reason}"
                                ))
                            } else if early_peak_max_age_jiffies == 0
                                && backstop_max_age == 0
                            {
                                Some(
                                    "scx_tick stall — no per-task \
                                     runnable_at data".to_string(),
                                )
                            } else {
                                Some(format!(
                                    "max_age never crossed threshold \
                                     (peak={early_peak_max_age_jiffies}j, \
                                     threshold={half_threshold_jiffies}j)"
                                ))
                            };
                        // Thaw before emission so a slow JSON
                        // serialise doesn't keep vCPUs parked any
                        // longer than the dump strictly needs. The
                        // backstop above (dual-snapshot only) ran
                        // while still frozen, so the backstop's
                        // runnable_at scan saw the same quiesced
                        // memory the dump captured — thawing here
                        // is safe because every site that depends
                        // on quiesced state has completed.
                        thaw_and_barrier();
                        // Branch on three outcomes of `freeze_and_dispatch`:
                        //   Captured(report, _) → emit the full dump,
                        //                         mark Done, kick kill
                        //                         so the run tears down
                        //                         promptly.
                        //   Degraded(boxed)     → rendezvous timed out;
                        //                         emit SCHEMA_DEGRADED
                        //                         JSON with partial
                        //                         vcpu_regs + trigger-
                        //                         state diagnostics
                        //                         (per the
                        //                         no-silent-drops policy),
                        //                         then mark Done + kick
                        //                         kill — retrying a
                        //                         timed-out rendezvous
                        //                         under a sticky bss
                        //                         latch would just hit
                        //                         the same timeout.
                        //   Suppressed          → gate cross-reference
                        //                         confirmed clean exit
                        //                         (kind &lt; SCX_EXIT_ERROR
                        //                         AND bss not Triggered);
                        //                         reset `watchpoint.hit`
                        //                         and mark Done. The
                        //                         original three-arm
                        //                         shape (separate "reset
                        //                         hit + keep watching"
                        //                         vs "mark Done")
                        //                         collapsed once the
                        //                         cross-reference
                        //                         eliminated the bss-
                        //                         flipped-during-
                        //                         rendezvous race
                        //                         (Triggered bss now
                        //                         routes through
                        //                         Captured/Degraded
                        //                         upstream).
                        match late_capture {
                            FreezeOutcome::Captured(late, capture_start) => {
                                // capture_start anchors the freeze→emit
                                // timing summary; emit_json reads
                                // Instant::now() - capture_start at log
                                // time so it covers serialise + write.
                                let map_count = late.maps.len();
                                let vcpu_regs_count = late.vcpu_regs.len();
                                // Pre-compute vcpu_none_indices so
                                // emit_json's stderr fallback can
                                // surface WHICH vCPUs stalled — same
                                // operator-critical signal the drain's
                                // stderr summary emits. The emit_json
                                // closure receives this as &[usize]
                                // regardless of whether the file write
                                // succeeds; on failure (or no sink)
                                // the indices reach stderr alongside
                                // the structured summary.
                                let vcpu_none_indices = vcpu_none_indices(&late.vcpu_regs);
                                let tasks_enriched = late.task_enrichments.len();
                                let truncated_at_us = late.dump_truncated_at_us;
                                // `to_string` (compact) replaces
                                // `to_string_pretty` to halve
                                // serialization cost on the hot
                                // failure-dump path. JSON consumers
                                // (sidecar tooling, repro probe) all
                                // parse via serde_json which
                                // tolerates either form identically.
                                // Eager `early_guard.snapshot.take()`
                                // here moves the early snapshot into
                                // `dual` BEFORE serde_json runs. If
                                // serde panicked OR returned Err mid-
                                // serialization, the snapshot would
                                // drop with `dual` — panic-unwind
                                // drops it; an Err return exits the
                                // if-block and `dual` drops there too.
                                // Either way `early_guard.snapshot`
                                // ends up None and the Drop is a no-op
                                // — silent loss without even stderr
                                // fallback (the Err arm below surfaces
                                // LATE-capture data only, not the
                                // early snapshot). The type-system
                                // proof that FailureDumpReport
                                // serialization is infallible for the
                                // concrete shape (see test
                                // `failure_dump_report_serialization_is_infallible_for_max_synthetic_input`
                                // in src/monitor/dump/tests.rs) keeps
                                // both paths safe today. A future
                                // field with a fallible Serialize
                                // impl would trip that test first.
                                let json_result = if freeze_coord_dual_snapshot {
                                    let dual = crate::monitor::dump::DualFailureDumpReport {
                                        schema: crate::monitor::dump::SCHEMA_DUAL
                                            .to_string(),
                                        early: early_guard.snapshot.take(),
                                        late,
                                        early_max_age_jiffies,
                                        early_threshold_jiffies,
                                        early_skipped_reason,
                                    };
                                    serde_json::to_string(&dual)
                                } else {
                                    serde_json::to_string(&late)
                                };
                                match json_result {
                                    Ok(json) => emit_json(
                                        &json,
                                        map_count,
                                        vcpu_regs_count,
                                        &vcpu_none_indices,
                                        tasks_enriched,
                                        capture_start.elapsed().as_millis() as u64,
                                        truncated_at_us,
                                    ),
                                    Err(e) => {
                                        // Serialize failure of a
                                        // well-typed FailureDumpReport
                                        // is near-zero probability
                                        // (only non-string-keys in maps
                                        // or custom Serialize errors —
                                        // neither applies here). The
                                        // structured summary is loud-
                                        // emitted to stderr when the
                                        // user wired a sink (so the
                                        // data loss represents a
                                        // captured observation that
                                        // the user asked for); skipped
                                        // when no sink (verifier /
                                        // shell / template paths never
                                        // asked for preservation).
                                        tracing::error!(
                                            error = %e,
                                            map_count,
                                            vcpu_regs_count,
                                            "freeze-coord: failure dump (JSON serialization failed)"
                                        );
                                        if freeze_coord_dump_path.is_some() {
                                            let elapsed_ms = capture_start
                                                .elapsed()
                                                .as_millis()
                                                as u64;
                                            eprintln!(
                                                "{SNAPSHOT_SUMMARY_PREFIX} (Captured, serialize failed): map_count={} vcpu_regs_count={} vcpu_none_indices={:?} tasks_enriched={} elapsed_ms={} serde_error={:?}",
                                                map_count,
                                                vcpu_regs_count,
                                                vcpu_none_indices,
                                                tasks_enriched,
                                                elapsed_ms,
                                                e.to_string(),
                                            );
                                        }
                                    }
                                }
                                freeze_state = FreezeState::Done;
                                // Error-class exit dump complete: the dump
                                // is serialized and emitted above, the probe
                                // ringbuf has drained by the time
                                // sched_ext_exit fired, and serial output is
                                // flushed.
                                //
                                // For a NON-wprof run no useful work remains
                                // in the post-exit window: kill now (the BSP
                                // run loop's `kill.load` and this loop's
                                // `'coord:` guard both observe the edge on
                                // the next wake) rather than looping back to
                                // epoll_wait under EEVDF fallback for the
                                // remainder of the host-watchdog window.
                                //
                                // A wprof run ships its trace LATE (see
                                // `wprof_ship_deadline` at its declaration):
                                // the guest sends the `.pb` over the bulk
                                // port in Phase 5 after the crash, then Phase
                                // 6 reboots (which can hang on some kernels).
                                // Killing here would tear the VM down before
                                // that ship, dropping the `.wprof.pb` /
                                // `.repro.wprof.pb` the artifact needs. For
                                // wprof, arm the bounded ship grace instead:
                                // `freeze_state` is now `Done`, so the loop
                                // keeps running WITHOUT re-freezing
                                // (err-detection is gated on `!= Done`), the
                                // TOKEN_TX drain promotes the kill the instant
                                // the WprofTrace frame lands, and
                                // `WPROF_SHIP_GRACE` is the backstop if the
                                // guest wedges before shipping.
                                if freeze_coord_wprof {
                                    eprintln!(
                                        "freeze-coord: error-exit dump captured (Captured); \
                                         awaiting wprof ship (grace {WPROF_SHIP_GRACE:?})"
                                    );
                                    wprof_ship_deadline =
                                        Some(arm_wprof_grace(run_start, &watchdog_reset_for_coord, &watchdog_reset_tag_for_coord, WPROF_SHIP_GRACE));
                                } else {
                                    eprintln!(
                                        "freeze-coord: kill triggered after \
                                         error-exit dump capture"
                                    );
                                    trigger_freeze_coord_kill(
                                        &freeze_coord_kill,
                                        &freeze_coord_kill_evt,
                                    );
                                }
                            }
                            FreezeOutcome::Degraded(degraded) => {
                                // Rendezvous timeout: emit the
                                // pre-built degraded JSON via the same
                                // atomic-publish helper used for full
                                // dumps so the consumer either sees a
                                // complete degraded file or nothing —
                                // never a torn write. After emit, mark
                                // Done and kick kill: retrying a
                                // rendezvous that already timed out
                                // would just hit the same timeout
                                // (sticky kernel-side bss latch).
                                let elapsed_ms = degraded.elapsed_ms;
                                match serde_json::to_string(degraded.as_ref()) {
                                    Ok(json) => emit_degraded_json(
                                        &json,
                                        &degraded.reason,
                                        elapsed_ms,
                                    ),
                                    Err(e) => tracing::error!(
                                        error = %e,
                                        reason = %degraded.reason,
                                        "freeze-coord: degraded dump (JSON serialization failed)"
                                    ),
                                }
                                // Late Degraded doesn't invalidate a
                                // Captured early either — the early
                                // was a runnable-age spike observation
                                // while the late is "rendezvous timed
                                // out". Both signals matter to an
                                // operator. The main dump path already
                                // holds the degraded JSON (atomic-
                                // published above), so we drop the
                                // early to a tagged sibling path. The
                                // write routes through the shared
                                // `write_to_tagged_path` helper (atomic
                                // publish + parent-dir fsync + stderr
                                // fallback) — same as the
                                // `FreezeOutcome::Degraded` arm of
                                // the early-snapshot dispatch above.
                                // The tag and signal differ: early-
                                // Degraded means the early itself was
                                // Degraded; early-pre-late-degraded
                                // means the early was Captured and the
                                // LATE was Degraded — distinct cases.
                                // Any captured snapshot must reach
                                // disk.
                                //
                                // base_path check FIRST in the let-
                                // chain (no take()), and `.take()` only
                                // AFTER the write succeeds. Two silent-
                                // drop windows closed:
                                //   - Rust let-chain short-circuits
                                //     left-to-right; an early take()
                                //     before the dump_path check would
                                //     consume the snapshot and skip
                                //     the body when dump_path is None.
                                //   - std::fs::write can fail mid-call
                                //     (ENOSPC / EROFS / EACCES). If
                                //     we'd already consumed `early`
                                //     via .take(), the failed write
                                //     would drop the snapshot with
                                //     only a tracing::warn! breadcrumb.
                                //     Holding the snapshot in
                                //     `early_snapshot` until the write
                                //     returns Ok preserves a retry
                                //     path via the end-of-coord drain
                                //     below (different tag, but a
                                //     wrong-tag emit beats a silent
                                //     drop).
                                if freeze_coord_dual_snapshot
                                    && freeze_coord_dump_path.is_some()
                                    && let Some(early) =
                                        early_guard.snapshot.as_ref()
                                {
                                    let write_succeeded = match serde_json::to_string(early) {
                                        Ok(json) => match write_to_tagged_path(
                                            freeze_coord_dump_path.as_deref(),
                                            crate::monitor::dump::SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED,
                                            &json,
                                            || {
                                                let vcpu_none = vcpu_none_indices(&early.vcpu_regs);
                                                format!(
                                                    "{SNAPSHOT_SUMMARY_PREFIX} (early-pre-late-degraded, write failed): schema={} vcpu_regs_count={} vcpu_none_indices={:?} maps_count={} tasks_enriched={} json_bytes={}",
                                                    early.schema,
                                                    early.vcpu_regs.len(),
                                                    vcpu_none,
                                                    early.maps.len(),
                                                    early.task_enrichments.len(),
                                                    json.len(),
                                                )
                                            },
                                            "freeze-coord: early-snapshot pre-late-degraded write failed (early retained for end-of-coord drain retry)",
                                        ) {
                                            Ok(Some(tagged)) => {
                                                tracing::info!(
                                                    path = %tagged.display(),
                                                    "freeze-coord: early snapshot preserved alongside degraded late"
                                                );
                                                true
                                            }
                                            Ok(None) => false,
                                            Err(_) => false,
                                        },
                                        Err(e) => {
                                            tracing::error!(
                                                error = %e,
                                                "freeze-coord: early-snapshot pre-late-degraded JSON serialization failed (early retained for end-of-coord drain retry)"
                                            );
                                            false
                                        }
                                    };
                                    if write_succeeded {
                                        // Consume only on success;
                                        // failure leaves the snapshot
                                        // for the end-of-coord drain
                                        // below to retry.
                                        let _ = early_guard.snapshot.take();
                                    } else {
                                        // Record the tag this arm
                                        // INTENDED so the drain's
                                        // retry lands at the correct
                                        // operator-readable path
                                        // rather than the drain's
                                        // default NEVER_FIRED tag
                                        // (which would misrepresent
                                        // this case — late DID fire
                                        // as Degraded).
                                        early_guard.retain_tag = Some(
                                            crate::monitor::dump::SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED,
                                        );
                                    }
                                }
                                freeze_state = FreezeState::Done;
                                // Degraded means the trigger fired but
                                // capture was lossy (rendezvous timeout).
                                // `warn` (not info) so operators monitoring
                                // scheduler-test runs see the degraded class
                                // — info would bury it; cf. the Captured arm
                                // above which uses info for the canonical
                                // success path.
                                //
                                // Same wprof late-ship handling as the
                                // Captured arm: `thaw_and_barrier` already
                                // ran, so the guest is thawed and can still
                                // ship its trace over the bulk port in Phase
                                // 5; `freeze_state` is now `Done` (no
                                // re-freeze). For a non-wprof run the kill
                                // below is forced teardown, not clean
                                // shutdown. For wprof, arm the bounded ship
                                // grace instead so the `.repro.wprof.pb` is
                                // not dropped — the TOKEN_TX drain or the
                                // `WPROF_SHIP_GRACE` backstop promotes the
                                // kill.
                                if freeze_coord_wprof {
                                    eprintln!(
                                        "freeze-coord: degraded dump captured (Degraded); \
                                         awaiting wprof ship (grace {WPROF_SHIP_GRACE:?})"
                                    );
                                    wprof_ship_deadline =
                                        Some(arm_wprof_grace(run_start, &watchdog_reset_for_coord, &watchdog_reset_tag_for_coord, WPROF_SHIP_GRACE));
                                } else {
                                    eprintln!(
                                        "freeze-coord: kill triggered after \
                                         degraded dump capture"
                                    );
                                    trigger_freeze_coord_kill(
                                        &freeze_coord_kill,
                                        &freeze_coord_kill_evt,
                                    );
                                }
                            }
                            FreezeOutcome::Suppressed => {
                                // Gate cross-reference decided no dump
                                // is warranted: the live
                                // `*scx_root->exit_kind` read is below
                                // SCX_EXIT_ERROR (clean shutdown), and
                                // any BPF `.bss` Triggered observation
                                // was either absent (probe never
                                // latched an error-class exit) or
                                // suppressed via the
                                // `bss_oob_warn_logged` exception
                                // (probe's PA stale post-rebind, so
                                // the cross-reference declined the
                                // override on potentially-recycled
                                // vmalloc bytes — see the `bss_state =
                                // bss_read_state(...)` block inside
                                // `freeze_and_dispatch` above for the
                                // gate decision). Either way the
                                // scheduler shut down cleanly; no late
                                // failure dump is warranted. Reset the
                                // watchpoint hit so the per-iteration
                                // latch read at the top of the loop
                                // does not re-fire on this stale edge;
                                // mark Done so the run terminates
                                // promptly.
                                //
                                // In dual_snapshot mode, an early-
                                // trigger capture may already be in
                                // hand (runnable_at exceeded half-
                                // threshold during the stall, the
                                // early-trigger
                                // `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })` returned
                                // Captured — see the
                                // `FreezeOutcome::Captured` arm
                                // of the early-snapshot dispatch
                                // above). The late gate-Suppressed
                                // decision does NOT invalidate the
                                // early observation — they're
                                // independent events: the early was a
                                // runnable-age spike, the late is the
                                // watchpoint fire interpreted as a
                                // clean exit. Dropping the early
                                // snapshot here would be a silent
                                // drop of captured data.
                                //
                                // Emit to a tagged sibling path rather
                                // than the main dump path. Symmetric
                                // with the Degraded-arm early-pre-
                                // late-degraded emit above: the main
                                // `{stem}.failure-dump.json` is
                                // reserved for "the scheduler had a
                                // failure-class late exit; here's the
                                // dump" semantics — a Suppressed late
                                // means no failure-class late exit
                                // happened, so the main path stays
                                // empty. The tagged sibling carries
                                // the early observation under
                                // [`crate::monitor::dump::SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED`]
                                // so an operator browsing the dump
                                // directory immediately sees the case
                                // from the file name.
                                // base_path check FIRST + `.take()`
                                // only after successful write — see
                                // the Degraded arm above for the
                                // full silent-drop rationale (same
                                // pattern applies here).
                                if freeze_coord_dual_snapshot
                                    && freeze_coord_dump_path.is_some()
                                    && let Some(early) =
                                        early_guard.snapshot.as_ref()
                                {
                                    let write_succeeded = match serde_json::to_string(early) {
                                        Ok(json) => match write_to_tagged_path(
                                            freeze_coord_dump_path.as_deref(),
                                            crate::monitor::dump::SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED,
                                            &json,
                                            || {
                                                let vcpu_none = vcpu_none_indices(&early.vcpu_regs);
                                                format!(
                                                    "{SNAPSHOT_SUMMARY_PREFIX} (early-only-late-suppressed, write failed): schema={} vcpu_regs_count={} vcpu_none_indices={:?} maps_count={} tasks_enriched={} json_bytes={}",
                                                    early.schema,
                                                    early.vcpu_regs.len(),
                                                    vcpu_none,
                                                    early.maps.len(),
                                                    early.task_enrichments.len(),
                                                    json.len(),
                                                )
                                            },
                                            "freeze-coord: early-only (late suppressed) write failed (early retained for end-of-coord drain retry)",
                                        ) {
                                            Ok(Some(tagged)) => {
                                                tracing::info!(
                                                    path = %tagged.display(),
                                                    "freeze-coord: early snapshot preserved (late suppressed, clean exit)"
                                                );
                                                true
                                            }
                                            Ok(None) => false,
                                            Err(_) => false,
                                        },
                                        Err(e) => {
                                            tracing::error!(
                                                error = %e,
                                                "freeze-coord: early-only (late suppressed) JSON serialization failed (early retained for end-of-coord drain retry)"
                                            );
                                            false
                                        }
                                    };
                                    if write_succeeded {
                                        let _ = early_guard.snapshot.take();
                                    } else {
                                        // See Degraded arm above for
                                        // the retain-tag rationale.
                                        early_guard.retain_tag = Some(
                                            crate::monitor::dump::SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED,
                                        );
                                    }
                                }
                                freeze_coord_watchpoint
                                    .hit
                                    .store(false, Ordering::Release);
                                freeze_state = FreezeState::Done;
                            }
                            FreezeOutcome::KernelOp(_) => capture_only_unreachable!(),
                        }
                        continue;
                    }
                    // End of body. Loop back to the `epoll.wait`
                    // at the top, which blocks until any registered
                    // fd fires (kill, bsp_done, virtio-console TX,
                    // watchpoint hit, scanner tick) or
                    // POLL_TIMEOUT_MS elapses.
                    // The watchpoint hit and bss-pa edges are
                    // delivered as eventfd writes from the vCPU
                    // thread, so the trigger latency is bounded by
                    // epoll_wait's microsecond-scale wakeup, NOT by
                    // any host-side polling cadence. Heavy work
                    // (boot-race accessor construction, scan_ctx
                    // resolve, runnable_at scan) remains gated on
                    // `scan_tick`, which only fires on the SCANNER
                    // timerfd edge (every 100 ms).
                }
                // Final drain of any pending user-watchpoint hits.
                // The hot-path for-loop at the end of each
                // coordinator iteration handles slot[i].hit fires
                // synchronously, but two race windows can leave a
                // hit `true` past loop exit:
                //
                //   1. The "already in flight" branch in the
                //      hot-path for-loop re-arms the slot's `hit`
                //      and `continue`s to the next slot when the
                //      gate is held on entry. If kill / bsp_done
                //      flips before the next outer iteration runs,
                //      the re-armed hit is never serviced.
                //   2. A vCPU's `latch_user_hit` Release that
                //      raced the loop exit (kill flipped between
                //      the for-loop terminating and the next
                //      `epoll.wait`).
                //
                // Without this drain the snapshot the test author
                // requested is silently dropped — `Snapshot::watch`
                // produces no entry, which a passing test
                // misinterprets as "the watched address was never
                // written" instead of "the VMM exited before the
                // capture pipeline serviced the fire". Store a
                // "watch-fired-but-coord-exited" placeholder under
                // the slot's tag so the test's lookup gets a
                // distinguishable result. Same minimal-report
                // shape the in-loop "dump prerequisites
                // unavailable" partial path uses, with a
                // dedicated reason string so consumers can tell
                // the two cases apart.
                for slot_idx in 0..freeze_coord_watchpoint.user.len() {
                    if !freeze_coord_watchpoint.user[slot_idx]
                        .hit
                        .swap(false, Ordering::AcqRel)
                    {
                        continue;
                    }
                    let tag = freeze_coord_watchpoint.user[slot_idx]
                        .tag
                        .lock_unpoisoned()
                        .clone();
                    // Skip the placeholder entirely when the bridge
                    // already has a real report under this tag. The
                    // in-loop dispatch publishes via
                    // `snapshot_bridge.store(&tag, report)`; a vCPU
                    // re-arm of `hit=true` after that successful
                    // publish (e.g. a second guest write to the
                    // watched KVA in the same tag, or a vCPU
                    // dispatch racing the in-loop hit.swap) leaves
                    // the slot's hit flag set at coord exit. Without
                    // this guard the final drain stomps the
                    // already-published real report with a hollow
                    // "coord exited before capture" placeholder,
                    // which a test misinterprets as "the watchpoint
                    // mostly didn't fire" rather than "the watch
                    // fired AND was captured." The has() lookup
                    // takes the bridge mutex briefly; teardown is
                    // single-threaded with no concurrent store
                    // (every vCPU thread joins AFTER this drain
                    // returns), so the check is race-free.
                    if freeze_coord_snapshot_bridge.has(&tag) {
                        tracing::debug!(
                            slot_idx,
                            %tag,
                            "freeze-coord: user-watchpoint fire pending at coord \
                             exit, but the bridge already has a real report under \
                             this tag — skipping placeholder to preserve the \
                             captured report"
                        );
                        continue;
                    }
                    tracing::warn!(
                        slot_idx,
                        %tag,
                        "freeze-coord: user-watchpoint fire pending at coord exit; \
                         storing placeholder report (no capture possible during \
                         teardown — vCPU rendezvous would race teardown joins)"
                    );
                    let placeholder = crate::monitor::dump::FailureDumpReport::placeholder(
                        "coord exited before capture",
                    );
                    freeze_coord_snapshot_bridge.store(&tag, placeholder);
                }
                // Post-drain advisory: vCPU threads (BSP + APs) are
                // still alive at this point — they only join inside
                // `collect_results` after the coord thread closure
                // returns (see `run_vm` join sequencing: coord first
                // via `freeze_coord_handle.join()`, AP threads later
                // via `wait_for_exit` + `handle.join` inside
                // `collect_results`). Any vCPU that calls
                // `latch_user_hit` between the drain loop above and
                // its eventual join will set `hit = true` AND
                // increment `hit_evt`, but the coordinator's epoll
                // is already gone — nothing services that hit. The
                // count of slots whose `request_kva != 0` here is
                // the upper bound on hits that could still be lost
                // (each such slot is currently armed in
                // KVM_SET_GUEST_DEBUG on every vCPU and capable of
                // firing on the next guest write to its KVA). This
                // warn surfaces the observability gap so an operator
                // who finds a missing snapshot in
                // `Snapshot::watch_results` can tell "VMM lost the
                // hit during teardown" from "guest never wrote to
                // the watched KVA". Acquire load is overkill (the
                // armed slot publication uses Release / vCPU
                // self-arm uses Acquire) but cheap.
                let still_armed = freeze_coord_watchpoint
                    .user
                    .iter()
                    .filter(|slot| slot.request_kva.load(Ordering::Acquire) != 0)
                    .count();
                if still_armed > 0 {
                    tracing::warn!(
                        still_armed,
                        "freeze-coord: post-drain teardown advisory — {still_armed} \
                         user-watchpoint slot(s) remain armed on every vCPU at \
                         coord exit. Hits latched by a vCPU between this drain \
                         and the eventual vCPU join in collect_results are NOT \
                         serviced (the coord epoll is already gone). Tests \
                         observing a missing snapshot in Snapshot::watch_results \
                         should treat this warn as evidence that the watched \
                         address WAS written to, just past the host-side \
                         capture window."
                    );
                }
                // Drain any held early_snapshot at coord exit. The
                // late-trigger arms (the three `FreezeOutcome`
                // match arms inside the `if err_triggered &&
                // (freeze_state == Idle || freeze_state ==
                // TookEarly)` block above) each take() the early when
                // they fire — but if the late trigger NEVER fires (no
                // err_exit_detected for the full run), the early
                // stays in this closure's local and drops silently
                // when the coord exits. A Captured early IS a real
                // failure-dump observation worth surfacing.
                //
                // Symmetric with the `FreezeOutcome::Suppressed`
                // late-trigger arm above in WRITE PATTERN ONLY (both
                // route through the shared `write_to_tagged_path`
                // helper for atomic publish + parent-dir fsync +
                // stderr fallback). Same operator-discoverable tagged
                // sibling discipline. SIGNAL DIFFERS:
                //   - late-Suppressed: late trigger DID fire, the
                //     gate examined `*scx_root->exit_kind` and
                //     decided clean (kind < SCX_EXIT_ERROR). Tag:
                //     `SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED`.
                //   - drain (this site): late trigger NEVER fired
                //     for the run. freeze_state stayed at Idle or
                //     TookEarly. Tag:
                //     `SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED`.
                // Different tags so an operator browsing the dump dir
                // distinguishes "scheduler recovered AND reached
                // clean late exit" from "scheduler recovered AND was
                // terminated/killed before reaching the late stage."
                // base_path check FIRST — same silent-drop rationale
                // as the Degraded + Suppressed arms above. The drain
                // logic lives in `EarlySnapshotGuard::drain_to_disk`
                // so the same emit runs on panic-unwind via the
                // guard's Drop. The normal-path call here drains
                // explicitly so a tracing::info! "preserved at coord
                // exit" line surfaces in healthy runs; the Drop fires
                // in the panic-unwind path with the same atomic-
                // publish + stderr-fallback semantics (see the
                // EarlySnapshotGuard doc for the full rationale on
                // retain_tag carry, idempotence, and panic-safety).
                early_guard.drain_to_disk();
                // Flush any partial-frame bytes the bulk_assembler
                // is still buffering back into the device's
                // `port1_tx_buf`. The assembler retains tail bytes
                // when a TLV frame straddles two TX wakes — without
                // this push-back the residual is dropped on the
                // floor when the assembler is dropped at closure
                // exit, and `collect_results`'s end-of-run
                // `drain_bulk` + `parse_tlv_stream` path never sees
                // them. Pushing them back means
                // `collect_results`'s drain returns the residual
                // alongside any bytes the device accumulated after
                // the last coordinator drain, and `parse_tlv_stream`
                // completes the frame.
                let coord_exit_t = std::time::Instant::now();
                if crate::vmm::debug_logging_enabled() {
                    eprintln!("CLEANUP: coord loop exited");
                }
                // Periodic-capture teardown summary. When
                // `num_snapshots > 0`, log the fired/total ratio so
                // an operator reading the test's tracing output can
                // tell at a glance whether the periodic-sampling
                // path delivered. The pipeline anchors on
                // `scenario_start_ns` when a CRC-valid
                // `MSG_TYPE_SCENARIO_START` was observed, else on the
                // `watchdog_reset`-derived fallback (the observed
                // scheduler-attach time); the summary keys on the
                // fired count plus which anchor source was available.
                // Three distinct shapes surface:
                //   * `K/N` fired (`K >= 1`) — boundaries were
                //     reached, so an anchor existed. When the primary
                //     `scenario_start_ns` was never stamped the
                //     `watchdog_reset` fallback supplied it and the
                //     message says so. `K < N` means the run
                //     terminated mid-sequence; the doc on
                //     `KtstrTestEntry::num_snapshots` warns the test
                //     author to assert `>= some_lower_bound` rather
                //     than `== num_snapshots` for exactly this case.
                //   * `0/N` with NEITHER anchor available (no
                //     `scenario_start_ns` AND a zero `watchdog_reset`)
                //     — periodic sampling never had a reference point.
                //     Most commonly a guest that crashed mid-boot or a
                //     workload that never reached the host-comms phase.
                //     This is the lone `warn`.
                //   * `0/N` with an anchor available (primary OR the
                //     `watchdog_reset` fallback) but no boundary
                //     reached — a very-short run or a kill before the
                //     first boundary. Same best-effort contract.
                if freeze_coord_num_snapshots > 0 {
                    let scenario_anchor =
                        scenario_start_ns_for_coord.load(Ordering::Acquire);
                    // The periodic pipeline anchors on `scenario_start_ns`
                    // when a CRC-valid MSG_TYPE_SCENARIO_START was observed,
                    // else on the `watchdog_reset`-derived fallback (the
                    // observed scheduler-attach time). A non-zero
                    // `watchdog_reset` means that fallback anchor was
                    // derivable, so the summary must NOT report
                    // "scenario_start_ns never stamped → no anchor" for a run
                    // that actually anchored (and fired) via the fallback.
                    let fallback_anchor =
                        watchdog_reset_for_coord.load(Ordering::Acquire);
                    if next_periodic_idx > 0 {
                        // Snapshots demonstrably fired, so an anchor existed.
                        // When the primary anchor was never stamped the
                        // fallback (the only other source) supplied it — say
                        // so rather than implying the primary path.
                        tracing::info!(
                            target: "ktstr::failure_dump",
                            num_snapshots = freeze_coord_num_snapshots,
                            fired = next_periodic_idx,
                            scenario_anchor_ns = scenario_anchor,
                            fallback_anchor_ns = fallback_anchor,
                            "freeze-coord: {}/{} periodic snapshots fired{}",
                            next_periodic_idx,
                            freeze_coord_num_snapshots,
                            if scenario_anchor == 0 {
                                " (anchored via the watchdog_reset fallback; \
                                 scenario_start_ns was never stamped)"
                            } else {
                                ""
                            },
                        );
                    } else if scenario_anchor == 0 && fallback_anchor == 0 {
                        // No anchor from EITHER source: periodic sampling
                        // never had a reference point. The guest most likely
                        // crashed mid-boot or never reached the host-comms
                        // phase.
                        tracing::warn!(
                            target: "ktstr::failure_dump",
                            num_snapshots = freeze_coord_num_snapshots,
                            fired = next_periodic_idx,
                            "freeze-coord: 0/{} periodic snapshots fired — \
                             neither scenario_start_ns (no CRC-valid \
                             MSG_TYPE_SCENARIO_START observed) nor the \
                             watchdog_reset fallback supplied an anchor; the \
                             guest most likely crashed mid-boot or never \
                             reached the host-comms phase",
                            freeze_coord_num_snapshots,
                        );
                    } else {
                        // An anchor was available (primary or fallback) but no
                        // boundary was reached — a very short run or a kill
                        // before the first boundary. Best-effort contract:
                        // tests assert `>=` a lower bound, not
                        // `== num_snapshots`.
                        tracing::info!(
                            target: "ktstr::failure_dump",
                            num_snapshots = freeze_coord_num_snapshots,
                            fired = next_periodic_idx,
                            scenario_anchor_ns = scenario_anchor,
                            fallback_anchor_ns = fallback_anchor,
                            "freeze-coord: 0/{} periodic snapshots fired — an \
                             anchor was available but no boundary was reached \
                             (short run or kill before the first boundary)",
                            freeze_coord_num_snapshots,
                        );
                    }
                }
                let residual = bulk_assembler.take_residual();
                if !residual.is_empty() {
                    freeze_coord_virtio_con.lock().push_back_bulk(&residual);
                }
                // Take owned_prog_accessor out so it can move into
                // the shared slot below (preferring the live accessor
                // the coord adopted over any leftover the worker
                // published into `accessors_slot` post-reset). Drop
                // the owned map accessor explicitly so the
                // `Arc<GuestMem>` reference-count transition is
                // visible at one site instead of implicit across
                // the join.
                let owned_prog_for_stash = owned_prog_accessor.take();
                let _ = owned_accessor;
                // Signal the accessor-init worker to exit. The worker
                // parks on `poll(kill_evt | reinit_evt, -1)` after each
                // publish (so the coordinator can pulse `reinit_evt`
                // on a scheduler swap and have the worker re-run
                // init), which means a `break 'coord` exit path that
                // didn't already flip `freeze_coord_kill` would leave
                // the park indefinitely. Set the kill flag and pulse
                // `kill_evt` here so every loop-exit path converges
                // on the same teardown contract: the worker's
                // `kill_for_worker.load()` check at the top of the
                // inner retry loop (and immediately after the park
                // wake) observes `true` and returns.
                //
                // Also drop reset triggers — the worker holds an
                // `Arc<GuestMem>` whose host pointer addresses
                // `vm.guest_mem`; that mapping is dropped right after
                // run_vm joins the freeze coordinator thread
                // (`freeze_coord_handle.join()` in run_vm), so any
                // worker still running past this join would dereference
                // freed memory through stale `Arc<GuestMem>` on its
                // next `try_init_*` retry.
                trigger_freeze_coord_kill(&freeze_coord_kill, &freeze_coord_kill_evt);
                if let Some(handle) = accessor_init_handle {
                    let jt = std::time::Instant::now();
                    let _ = handle.join();
                    if crate::vmm::debug_logging_enabled() {
                        eprintln!("CLEANUP: accessor-init worker joined {:?}", jt.elapsed());
                    }
                }
                // Extract the prog accessor for collect_verifier_stats
                // and stash it in the shared slot so run_vm can pass
                // it to VmRunState. Prefer the coord's live
                // `owned_prog_for_stash` (the accessor in active use
                // before teardown); if that's None — e.g. teardown ran
                // before adoption, or a scheduler-detach reset cleared
                // it and the worker hadn't republished — fall back to
                // whatever the worker last left in the slot.
                {
                    let slot = &prog_accessor_slot_for_coord;
                    let extracted = owned_prog_for_stash.or_else(|| {
                        Arc::try_unwrap(accessors_slot)
                            .ok()
                            .and_then(|m| m.into_inner().ok())
                            .and_then(|maybe| maybe)
                            .and_then(|(_map, prog)| prog)
                    });
                    *slot.lock_unpoisoned() = extracted;
                }
                if crate::vmm::debug_logging_enabled() {
                    eprintln!("CLEANUP: coord closure done {:?}", coord_exit_t.elapsed());
                }
            })
            .context("spawn freeze coordinator thread")?;
        // The coordinator holds bsp + AP ImmediateExitHandles and a raw pointer
        // into guest_mem; the guard joins it (before bsp/vm drop) on an
        // early-return past here — e.g. the watchdog spawn `?` just below.
        guard.freeze_coord = Some(freeze_coord_handle);

        let watchdog = std::thread::Builder::new()
            .name("ktstr-watchdog".into())
            .spawn(move || {
                if let Some(cpu) = wd_service_cpu {
                    pin_current_thread(cpu, "ktstr-watchdog");
                }
                // The hang detector's own sensing must not dilate with the
                // load it measures. FIFO-2 is therefore UNCONDITIONAL (not
                // perf-mode-gated): under extreme host dilation a SCHED_OTHER
                // watchdog misses its own 100 ms tick / 2 s monitor-liveness
                // window, so `monitor_live` latches false and the deadman
                // fires on wall-clock against a cell that was actively
                // progressing (an observed dump had a milestone 6 s before
                // the kill). The thread burns ~µs per 100 ms tick, so FIFO-2
                // grants scheduling immunity at no meaningful CPU cost: in
                // perf mode vCPUs at FIFO-1 stay below it; in no-perf/default
                // mode it now outranks SCHED_OTHER vCPUs — intended, sensing
                // must win. Best-effort — warns once/process without
                // CAP_SYS_NICE (see `set_rt_priority`). The service-CPU PIN
                // above stays perf-mode-gated: `wd_service_cpu` is `None`
                // without a reserved CPU, so there is nothing to pin to.
                set_rt_priority(2, "ktstr-watchdog");
                let hard_deadline = Instant::now() + timeout;
                // Soft phase needs enough headroom for the guest to
                // flush serial and reboot. Skip when timeout < 5s.
                let soft_deadline = if timeout > Duration::from_secs(5) {
                    Some(hard_deadline - Duration::from_secs(3))
                } else {
                    None
                };
                let mut soft_fired = false;
                // Progress-tier state. `monitor_liveness` tracks whether
                // the monitor's heartbeat is still advancing (a dead
                // monitor suppresses both progress tiers — see
                // `watchdog_step::MonitorLiveness`). `kill_reason` records
                // which rule fires so the dump names the true cause
                // (Tier-1/2 progress verdict, the Tier-3 hard deadline, or
                // an AP-set kill) — see [`KillReasonTag`]. The Arc is
                // shared with `run_vm`, which loads the final value after
                // this thread's join and surfaces it as
                // `VmResult::watchdog_kill_reason` (no cross-thread
                // ordering subtleties: the join is the synchronization).
                let mut monitor_liveness = watchdog_step::MonitorLiveness::new();
                let kill_reason = watchdog_kill_reason_for_wd;
                // Cached scheduler-attach reset deadline. Decoded
                // lazily from `watchdog_reset_for_wd` after the
                // host monitor stores a non-zero value (the
                // moment `*scx_root` flips from null to non-null
                // in guest memory). `None` means the workload's
                // clock has not started yet, so the original
                // `hard_deadline` (counted from VM boot) still
                // applies. Once `Some(reset)`, the effective
                // deadline becomes the reset value
                // (`reset_deadline.unwrap_or(hard_deadline)` — no
                // min clamp), so boot-time delays do not eat
                // into the workload budget. Cached so the per-tick
                // check is a single compare against
                // `effective_deadline` rather than re-decoding
                // the encoded `Duration::from_nanos` form.
                // Computed only when `workload_duration_for_wd`
                // is set; absent, the load is skipped entirely
                // (no workload duration → nothing to reset to).
                let mut reset_deadline: Option<Instant> = None;
                // Tier-3 deadman deferral bookkeeping. `deadman_deferrals`
                // counts ticks the wall deadline was reached but the
                // deadman deferred (cell alive); folded into the eventual
                // kill dump so a deferred-then-killed cell shows its
                // history.
                let mut deadman_deferrals: u64 = 0;
                if crate::vmm::debug_logging_enabled() {
                    eprintln!("ktstr-watchdog: started, timeout={timeout:?}");
                }

                // Wake plumbing. `tick_tfd` is a periodic 100 ms
                // timerfd that drives the deadline-progress checks
                // (matches the legacy `thread::sleep(100ms)` cadence
                // exactly). `kill_evt_for_watchdog` and
                // `bsp_done_evt_for_wd` are fast-wake fds bumped by
                // the kill / bsp_done setters so the deadline-arm
                // path runs within microseconds of the flip rather
                // than at the next 100 ms tick. Construction failure
                // for any of these means the watchdog cannot
                // observe wake signals; surface as `tracing::error`
                // and return so the symptom is visible — the
                // deadline-armed BSP still gets kicked by the
                // freeze coordinator's own paths if those fire.
                let mut tick_tfd = match TimerFd::new() {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!(err = %e, "ktstr-watchdog: timerfd_create failed");
                        return;
                    }
                };
                let tick = Duration::from_millis(100);
                if let Err(e) = tick_tfd.reset(tick, Some(tick)) {
                    tracing::error!(err = %e, "ktstr-watchdog: timerfd_settime failed");
                    return;
                }
                let epoll = match Epoll::new() {
                    Ok(e) => e,
                    Err(e) => {
                        tracing::error!(err = %e, "ktstr-watchdog: epoll_create1 failed");
                        return;
                    }
                };
                let tick_fd = tick_tfd.as_raw_fd();
                let kill_fd = kill_evt_for_watchdog.as_raw_fd();
                let bsp_done_fd = bsp_done_evt_for_wd.as_raw_fd();
                if let Err(e) = epoll.ctl(
                    ControlOperation::Add,
                    tick_fd,
                    EpollEvent::new(EventSet::IN, tick_fd as u64),
                ) {
                    tracing::error!(err = %e, "ktstr-watchdog: epoll_ctl add timerfd failed");
                    return;
                }
                if let Err(e) = epoll.ctl(
                    ControlOperation::Add,
                    kill_fd,
                    EpollEvent::new(EventSet::IN, kill_fd as u64),
                ) {
                    tracing::error!(err = %e, "ktstr-watchdog: epoll_ctl add kill_evt failed");
                    return;
                }
                if let Err(e) = epoll.ctl(
                    ControlOperation::Add,
                    bsp_done_fd,
                    EpollEvent::new(EventSet::IN, bsp_done_fd as u64),
                ) {
                    tracing::error!(err = %e, "ktstr-watchdog: epoll_ctl add bsp_done_evt failed");
                    return;
                }
                let mut epoll_buf = [EpollEvent::default(); 3];

                loop {
                    if bsp_done_for_wd.load(Ordering::Acquire) {
                        if crate::vmm::debug_logging_enabled() {
                            eprintln!("ktstr-watchdog: BSP done, returning");
                        }
                        return;
                    }
                    // Decode a pending scheduler-attach reset
                    // when not already cached. Skip when the
                    // workload duration was not configured (no
                    // distinct workload budget; `hard_deadline`
                    // is the only deadline that matters).
                    if workload_duration_for_wd.is_some() {
                        let stored_ns = watchdog_reset_for_wd.load(Ordering::Acquire);
                        if stored_ns != 0 {
                            let candidate = run_start
                                .checked_add(Duration::from_nanos(stored_ns))
                                .unwrap_or(hard_deadline);
                            if reset_deadline.is_none()
                                || reset_deadline.is_some_and(|prev| candidate > prev)
                            {
                                reset_deadline = Some(candidate);
                                if crate::vmm::debug_logging_enabled() {
                                    eprintln!(
                                        "ktstr-watchdog: scheduler attach observed, hard \
                                         deadline reset to {:?} from VM start",
                                        candidate.saturating_duration_since(run_start),
                                    );
                                }
                            }
                        }
                    }
                    // Progress-tier evaluation, BEFORE the deadline
                    // compare so a wedge is caught on its phase budget
                    // rather than waiting out the full Tier-3 deadline.
                    // Read the ledger once, fold monitor liveness, and run
                    // the pure `watchdog_step` tiers. `now_wall_ns` is
                    // `run_start`-relative — the same anchor the monitor
                    // stamps `wall_ns_at_progress` with — so the wall
                    // deltas align. `snapshot` and `monitor_live` are read
                    // here and reused by the dump in this same iteration. A
                    // dead monitor makes `evaluate_progress` return None
                    // (both tiers off), so a degraded / no-monitor host
                    // naturally falls back to Tier-3-only.
                    let snapshot = progress_ledger_for_wd.snapshot();
                    let monitor_live = monitor_liveness.observe(snapshot.monitor_heartbeat);
                    let now_wall_ns = run_start.elapsed().as_nanos() as u64;
                    // CPU-trickle verdict for this tick — the Tier-3
                    // deadman's deferral discriminator (its ONLY consumer;
                    // Tier-2 carries no CPU term). The MONITOR computes it
                    // (it owns the per-vCPU CPU data the busiest-vCPU
                    // windowed currency needs — see
                    // `watchdog_step::CpuTrickleTracker`); the watchdog just
                    // reads the published verdict. `true` = the guest's
                    // busiest single vCPU accrued below the currency floor
                    // over the trailing window.
                    let cpu_trickle_stalled = snapshot.cpu_trickle_stalled;
                    // Wall time since the last MILESTONE (progress_epoch
                    // anchor) — milestone-only, so a live kernel's
                    // scheduling noise never resets it. Same quantity as
                    // Tier-2's `wall_in_phase` (a phase entry is a
                    // milestone); reused by the deadman deferral gate.
                    let wall_since_milestone_ns =
                        now_wall_ns.saturating_sub(snapshot.wall_ns_at_progress);
                    let progress_decision =
                        watchdog_step::evaluate_progress(&snapshot, now_wall_ns, monitor_live);
                    let tier_fire = !matches!(progress_decision, watchdog_step::KillDecision::None);
                    let effective_deadline =
                        reset_deadline.map_or(hard_deadline, |r| r.max(hard_deadline));
                    let kill_set = kill_for_watchdog.load(Ordering::Acquire);
                    let hard_deadline_reached = Instant::now() >= effective_deadline;
                    // Tier-3 deadman deferral: reaching the wall deadline
                    // only KILLS when the monitor is dead or the cell is
                    // inert (CPU trickle-stalled AND no milestone within
                    // the grace). A starved-but-alive cell keeps accruing
                    // CPU → not stalled → deferred past the wall deadline
                    // by design (bounded by the harness/operator). A
                    // userspace-wedged cell on a live kernel trickle-stalls
                    // and reaches no milestone → fires here, BOUNDED. See
                    // [`watchdog_step::deadman_should_fire`].
                    let deadman_fire = hard_deadline_reached
                        && watchdog_step::deadman_should_fire(
                            monitor_live,
                            wall_since_milestone_ns,
                            cpu_trickle_stalled,
                        );
                    if kill_set || deadman_fire || tier_fire {
                        // A progress tier fired, an AP set kill, or the
                        // hard timeout expired.
                        // Re-check bsp_done: if the BSP already exited its
                        // run loop, the VcpuFd (and kvm_run mmap backing
                        // bsp_ie) may be dropped. Writing to ie after drop
                        // is a use-after-free.
                        if bsp_done_for_wd.load(Ordering::Acquire) {
                            if crate::vmm::debug_logging_enabled() {
                                eprintln!("ktstr-watchdog: BSP already done, returning");
                            }
                            return;
                        }
                        // The deadman is the CAUSE only when it actually
                        // fired (wall deadline reached AND not deferred),
                        // not merely because the wall deadline passed while
                        // the cell stayed alive. So a kill_set arriving
                        // past a deferred deadline is labeled AP-kill, not
                        // Tier-3, and does not set `timed_out`.
                        let hard_timeout_fired = deadman_fire;
                        // Cause precedence: a progress-tier verdict is the
                        // most specific (it names WHICH wedge), so it wins
                        // over the generic hard deadline; the AP-set kill
                        // is last (and, unlike the tiers/deadline, is not a
                        // timeout). Mirrors the old hard-over-AP order.
                        let reason_tag = match progress_decision {
                            watchdog_step::KillDecision::Tier1CpuBudget => KillReasonTag::Tier1Cpu,
                            watchdog_step::KillDecision::Tier2IdleWedge => KillReasonTag::Tier2Idle,
                            watchdog_step::KillDecision::None if hard_timeout_fired => {
                                KillReasonTag::Tier3Deadman
                            }
                            watchdog_step::KillDecision::None => KillReasonTag::ApKill,
                        };
                        kill_reason.store(reason_tag as u8, Ordering::Relaxed);
                        // Decode back through the atomic the fire path just
                        // stored into, mirroring the reset-tag round-trip;
                        // this token is the dump's authoritative kill cause.
                        let kill_reason_str =
                            KillReasonTag::from_u8(kill_reason.load(Ordering::Relaxed)).render();
                        eprintln!("ktstr-watchdog: {kill_reason_str}, kicking BSP");
                        // Actionable diagnostics. Without this dump the
                        // operator-visible failure is just `timed_out =
                        // true` with no clue why. Print the deadline
                        // values (`effective_deadline` is what actually
                        // fired; `hard_deadline` is the original boot-
                        // anchored deadline before any
                        // scheduler-attach reset) plus the
                        // `timeout`/`workload_duration` knobs the
                        // operator can tune, the cause path
                        // (hard-timeout-expired vs kill-set-by-AP),
                        // and whether the deadline was reset. Both
                        // deadlines are rendered as offsets from
                        // `run_start` so the numbers line up with the
                        // wall-clock the operator sees in the test
                        // output. The `kill_set_by_AP` branch also
                        // ages `effective_deadline` against now so
                        // the operator can see how much budget was
                        // unused when the kill arrived.
                        let now = Instant::now();
                        let effective_offset =
                            effective_deadline.saturating_duration_since(run_start);
                        let hard_offset = hard_deadline.saturating_duration_since(run_start);
                        let elapsed = now.saturating_duration_since(run_start);
                        // Decode WHICH writer last armed the reset
                        // deadline, not merely whether one did. The old
                        // `reset_by_scheduler_attach` key keyed on
                        // `reset_deadline.is_some()`, which reads true
                        // even when the scenario-start arm (not a live
                        // scheduler) armed it — mislabeling dead
                        // schedulers during triage. Relaxed load pairs
                        // with the writers' Relaxed stores; a benign race
                        // with `watchdog_reset_for_wd` can show a fresh
                        // deadline against a stale tag, acceptable for a
                        // human-facing dump.
                        let reset_armed_by = WatchdogResetTag::from_u8(
                            watchdog_reset_tag_for_wd.load(Ordering::Relaxed),
                        )
                        .render();
                        // Evidence for the dump, from this tick's ledger
                        // snapshot. `max_vcpu_cpu_in_phase` (the busiest
                        // vCPU's CPU burned since the phase was entered) is
                        // what Tier-1 compares against the currency-widened
                        // budget — width-independent; the summed `cpu_ns_now`
                        // is rendered alongside as pure context (it feeds no
                        // tier). `wall_in_phase` (wall since the last
                        // milestone) is what Tier-2 / the deadman compare
                        // against the backstop / grace. The `u64::MAX`
                        // sentinel budgets (Body / unmodeled phase) render as
                        // `off` rather than an absurd Duration.
                        let stage = crate::monitor::LifecycleStage::from_u8(snapshot.phase);
                        let max_vcpu_cpu_in_phase = snapshot.max_vcpu_cpu_in_phase_ns;
                        let wall_in_phase =
                            now_wall_ns.saturating_sub(snapshot.wall_ns_at_progress);
                        let cpu_budget = watchdog_step::widened_cpu_budget_ns(
                            snapshot.phase,
                            snapshot.cpu_currency,
                        );
                        let wall_backstop =
                            crate::test_support::runtime::phase_wall_backstop_ns(snapshot.phase);
                        // Sentinel-aware ns→Duration renderer for a budget.
                        let render_budget = |ns: u64| {
                            if ns == u64::MAX {
                                "off".to_string()
                            } else {
                                format!("{:?}", Duration::from_nanos(ns))
                            }
                        };
                        let cpu_currency_str = match snapshot.cpu_currency {
                            crate::monitor::CPU_CURRENCY_PMU => "pmu",
                            crate::monitor::CPU_CURRENCY_PTHREAD => "pthread",
                            _ => "none",
                        };
                        // A pre-attach (Boot) progress-tier kill is an
                        // infrastructure fault, not a test failure — prefix
                        // the header with a greppable marker so triage can
                        // separate framework wedges from workload timeouts.
                        let infra_fault = matches!(
                            reason_tag,
                            KillReasonTag::Tier1Cpu | KillReasonTag::Tier2Idle
                        ) && snapshot.phase
                            == crate::monitor::LifecycleStage::Boot as u8;
                        let header_prefix = if infra_fault {
                            "ktstr infra fault: "
                        } else {
                            ""
                        };
                        eprintln!(
                            "{header_prefix}ktstr-watchdog: deadline expired at {elapsed:?} from VM start"
                        );
                        eprintln!(
                            "  cause={kill_reason_str}, \
                             hard_timeout_fired={hard_timeout_fired}, kill_set_by_AP={}",
                            reason_tag == KillReasonTag::ApKill
                        );
                        // Evidence block. `monitor_live` gates BOTH
                        // progress tiers; `evidence_channels_live` gates
                        // Tier-2 specifically (per-CPU guest reads
                        // resolved this tick) — when false, `runnable_demand`
                        // is blind and Tier-2 was suppressed, so an
                        // early-boot kill here is Tier-1 / Tier-3 only.
                        eprintln!(
                            "  phase={stage:?} ({:?}), monitor_live={monitor_live}, \
                             evidence_channels_live={}",
                            stage.class(),
                            snapshot.evidence_channels_live,
                        );
                        eprintln!(
                            "  max_vcpu_cpu_in_phase={:?} vs budget={} \
                             (currency={cpu_currency_str}), cpu_sum={:?}, \
                             cpu_trickle_stalled={cpu_trickle_stalled}",
                            Duration::from_nanos(max_vcpu_cpu_in_phase),
                            render_budget(cpu_budget),
                            Duration::from_nanos(snapshot.cpu_ns_now),
                        );
                        // Deadman trickle evidence: the BUSIEST single
                        // vCPU's CPU accrued over the last closed 10 s
                        // window (monitor-computed via per-vCPU window
                        // anchors) vs the currency floor — what latched (or
                        // deferred) `cpu_trickle_stalled` above. Deadman-
                        // only; Tier-2 carries no CPU term.
                        eprintln!(
                            "  busiest_vcpu_window={:?} vs trickle_floor={:?}",
                            Duration::from_nanos(snapshot.busiest_vcpu_window_ns),
                            Duration::from_nanos(watchdog_step::trickle_floor_for_currency(
                                snapshot.cpu_currency
                            )),
                        );
                        eprintln!(
                            "  wall_in_phase={:?} vs backstop={}",
                            Duration::from_nanos(wall_in_phase),
                            render_budget(wall_backstop),
                        );
                        eprintln!(
                            "  progress_epoch={} (milestones), wall_since_milestone={:?}, \
                             runnable_demand={}, deadman_deferrals={deadman_deferrals}",
                            snapshot.progress_epoch,
                            Duration::from_nanos(wall_since_milestone_ns),
                            snapshot.runnable_demand,
                        );
                        // KILL-TIME publish-state evidence. The monitor's
                        // pre-latch diag prints the same counters ONCE at
                        // ~10 s; a starved boot can deliver KERN_ADDRS long
                        // after that, so only a kill-time read separates
                        // "the publish NEVER arrived across the whole run"
                        // (frames=0 here — guest never sent, or the frame
                        // was lost/mistyped upstream of the dispatcher)
                        // from "arrived late but a derive input was absent"
                        // (frames>0 with kaslr_raw=0 — e.g. the guest's
                        // kallsyms `_text` read failed so the frame carried
                        // no runtime KVA).
                        eprintln!(
                            "  kern_addrs_frames={} (kill-time), kaslr_raw={:#x}, \
                             phys_base_raw={:#x}, unknown_type_frames={}",
                            kern_addrs_frames_for_wd.load(Ordering::Relaxed),
                            kern_virt_kaslr_for_wd.load(Ordering::Acquire),
                            kern_phys_base_for_wd.load(Ordering::Acquire),
                            unknown_type_frames_for_wd.load(Ordering::Relaxed),
                        );
                        eprintln!(
                            "  effective_deadline={effective_offset:?} from VM start \
                             (reset_armed_by={reset_armed_by})"
                        );
                        eprintln!("  hard_deadline={hard_offset:?} from VM start (timeout knob)");
                        eprintln!(
                            "  timeout={timeout:?}, workload_duration={:?}",
                            workload_duration_for_wd
                        );
                        eprintln!(
                            "  hint: if the test body needs more wall time, increase \
                             duration (the `duration` field on `KtstrTestEntry` / \
                             `#[ktstr_test(duration_ms = ...)]`); the VM timeout is \
                             derived as max(watchdog_timeout, duration) so raising \
                             duration also extends the host watchdog deadline"
                        );
                        // All watchdog VERDICT timeouts set `timed_out`: the
                        // Tier-3 hard deadline AND the Tier-1/2 progress
                        // tiers (a wedge is a timeout by another name). The
                        // AP-set-kill path does NOT — it is a panic-driven
                        // kill, and propagating it as `timed_out=true` would
                        // mislabel it as a deadline expiry.
                        if hard_timeout_fired || tier_fire {
                            timed_out_for_watchdog.store(true, Ordering::Release);
                        }
                        // Propagate kill so handle_freeze's poll loop
                        // exits and the monitor + bpf-write threads stop.
                        trigger_freeze_coord_kill(&kill_for_watchdog, &kill_evt_for_watchdog);
                        if let Some(ref ie) = bsp_ie {
                            ie.set(1);
                            std::sync::atomic::fence(Ordering::Release);
                        }
                        unsafe {
                            libc::pthread_kill(bsp_tid, vcpu_signal());
                        }
                        eprintln!("ktstr-watchdog: BSP kicked");
                        return;
                    }
                    // Soft deadline: request graceful shutdown by
                    // pushing `SIGNAL_VC_SHUTDOWN` into virtio-console
                    // RX. The guest's `hvc0_poll_loop` blocks on
                    // `/dev/hvc0` and recognises the byte directly —
                    // no SHM signal slot needed. The BSP keeps running
                    // so the guest can flush serial and reboot
                    // normally.
                    //
                    // Recompute the soft window from the effective
                    // deadline so a scheduler-attach reset shifts
                    // the soft deadline alongside the hard
                    // deadline. The reset can extend past
                    // hard_deadline (no min clamp), so the
                    // recomputed `effective_deadline - 3s` shifts
                    // forward whenever the reset extends; the
                    // guest still gets its 3s flush window
                    // relative to the deadline that actually
                    // fires. Skip when the original
                    // `soft_deadline` was `None` (timeout < 5s;
                    // no soft phase configured) — the reset path
                    // inherits that decision rather than
                    // synthesising a soft phase out of nothing.
                    //
                    // Tier-3 deferral consistency: the soft request PRECEDES
                    // the hard fire (3 s before), so it is gated by the SAME
                    // deferral predicate — only nudge the guest toward a
                    // flush+reboot when the deadman WOULD fire (monitor dead
                    // or cell inert). A starved-but-alive cell is deferred,
                    // so it must never be told to shut down: that would kill
                    // the very cell the deferral protects. Once the wall
                    // deadline is reached but deferred, count it (no
                    // per-tick log — would spam for the whole deferral span)
                    // and skip soft entirely.
                    let effective_soft = soft_deadline
                        .and_then(|_| effective_deadline.checked_sub(Duration::from_secs(3)));
                    if hard_deadline_reached {
                        // Reached the wall deadline this tick but did not
                        // fire (deadman deferred; kill/tier not set) — the
                        // cell is alive but past its budget.
                        deadman_deferrals = deadman_deferrals.saturating_add(1);
                    } else if !soft_fired
                        && effective_soft.is_some_and(|d| Instant::now() >= d)
                        && watchdog_step::deadman_should_fire(
                            monitor_live,
                            wall_since_milestone_ns,
                            cpu_trickle_stalled,
                        )
                    {
                        soft_fired = true;
                        eprintln!("ktstr-watchdog: soft deadline, requesting graceful shutdown");
                        super::host_comms::request_shutdown(&wd_virtio_con);
                    }
                    // Block until the next tick or a kill_evt /
                    // bsp_done_evt write. -1 timeout: deadlines
                    // (hard + soft) are checked at the top of each
                    // iteration after the wake; the 100 ms timerfd
                    // guarantees the loop wakes at least that often
                    // even when no eventfd writes arrive, which
                    // preserves the legacy cadence exactly.
                    match epoll.wait(-1, &mut epoll_buf) {
                        Ok(n) => {
                            for ev in &epoll_buf[..n] {
                                if ev.fd() == tick_fd {
                                    // Drain the timerfd counter so
                                    // the next epoll_wait blocks
                                    // again instead of returning
                                    // immediately on the residual
                                    // ready bit.
                                    let _ = tick_tfd.wait();
                                }
                                // kill_fd / bsp_done_fd: implicitly
                                // drained because the loop body
                                // re-loads the AtomicBool source of
                                // truth on every iteration. The
                                // EventFd counter accumulates but
                                // is harmless — we only care about
                                // the edge.
                            }
                        }
                        Err(e) => {
                            if e.raw_os_error() != Some(libc::EINTR) {
                                tracing::warn!(err = %e, "ktstr-watchdog: epoll_wait failed");
                                // Fall through to the next iteration
                                // so the deadline check still runs;
                                // a persistent failure is eventually
                                // caught by the hard deadline.
                            }
                        }
                    }
                }
            })
            .context("spawn watchdog thread")?;
        // Last spawn: the watchdog holds bsp's ImmediateExitHandle. With it in
        // the guard, the guard now owns every spawned thread. Under panic=unwind
        // (test profile) a panic in the BSP loop below unwinds through the
        // guard's Drop, which joins them all before bsp/vm drop; under
        // panic=abort (release) the process aborts instead — no detached thread
        // survives to touch freed memory, so that path is UAF-safe too. The
        // normal return disarms the guard just before the watchdog/coordinator
        // joins (deferred past the infallible post-loop teardown, so a panic
        // there is still covered).
        guard.watchdog = Some(watchdog);

        // Boot-ordering gate: block the BSP until every AP host thread has
        // reached the point immediately before its first KVM_RUN (each AP
        // fires its latch at the tail of the closure in `spawn_ap_threads`;
        // see that latch's creation above for the full rationale). The guest
        // kernel's `do_boot_cpu` brings APs up sequentially with a bounded
        // per-CPU wait, and an AP thread the host scheduler hasn't run into
        // KVM_RUN by the time its INIT-SIPI arrives misses the window and goes
        // present-but-offline. Holding guest boot until every AP is in KVM_RUN
        // (where KVM buffers the pending INIT/SIPI) closes that race. On the
        // fast path every latch is already set — APs reach KVM_RUN within a
        // few ms of spawn — so this costs a handful of uncontended lock
        // acquisitions; the timeout is purely a safety net so a wedged or
        // panicked AP can never hang the VM here. `kill` is polled inside the
        // wait so a panicking AP (whose panic hook stores `kill`) releases the
        // gate at once rather than waiting out the full timeout, and the
        // subsequent not-ready check turns that into a propagated error.
        {
            // Milliseconds on the fast path. The run watchdog covers the whole
            // VM at 120s, so a 30s bring-up cap sits comfortably below it while
            // staying generous for a badly oversubscribed host.
            const AP_READY_TIMEOUT: Duration = Duration::from_secs(30);
            // Real gate-start instant so the error can report the ACTUAL wait,
            // not the cap — a kill-break can trip the gate at ~0s elapsed.
            let gate_start = Instant::now();
            let deadline = gate_start + AP_READY_TIMEOUT;
            for latch in &ap_boot_latches {
                while !latch.is_set() {
                    if kill.load(Ordering::Acquire) {
                        break;
                    }
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    latch.wait_timeout((deadline - now).min(Duration::from_millis(100)));
                }
            }
            // Report by guest CPU id: AP index `i` is thread `vcpu-{i+1}`
            // (the BSP is vCPU 0), matching the spawn-loop naming.
            let not_ready: Vec<usize> = ap_boot_latches
                .iter()
                .enumerate()
                .filter(|(_, l)| !l.is_set())
                .map(|(i, _)| i + 1)
                .collect();
            if !not_ready.is_empty() {
                // Evidence dump — runs ONLY on the trip path. The old fixed
                // "host CPU starvation" message was empirically refuted (fair
                // scheduler contention cannot trip this gate even at 25x
                // oversubscription), so instead of asserting a cause we collect
                // per-thread kernel state and let the reader judge. For each AP
                // whose boot latch never fired, read its published TID slot: the
                // TID is stamped as the FIRST act of the AP closure, so a slot
                // still 0 means the host thread ran zero instructions — the pure
                // "never scheduled" signal. Comm-based attribution is NOT used:
                // Rust sets a thread's name via prctl from INSIDE the thread, so
                // a never-scheduled thread still carries the parent's comm. For a
                // thread that did run, read
                // /proc/self/task/{tid}/{stat,schedstat,wchan,status} once and
                // pull the starved-vs-blocked discriminators. All reads are
                // best-effort — a vanished TID yields "?" rather than failing.
                use std::fmt::Write as _;
                let mut evidence = String::new();
                for (i, (tid_slot, _)) in ap_tid_slots.iter().enumerate() {
                    if ap_boot_latches[i].is_set() {
                        continue;
                    }
                    let vcpu_id = i + 1;
                    let tid = tid_slot.load(Ordering::Acquire);
                    if tid == 0 {
                        let _ = writeln!(
                            evidence,
                            "  vCPU {vcpu_id}: never scheduled (no TID stamped) — \
                             the host thread ran zero instructions of its closure \
                             (pure starvation)"
                        );
                        continue;
                    }
                    let base = format!("/proc/self/task/{tid}");
                    let read1 = |f: &str| {
                        std::fs::read_to_string(format!("{base}/{f}"))
                            .map(|s| s.trim().to_string())
                            .unwrap_or_default()
                    };
                    // Split off everything after "(comm) " so a comm containing
                    // ") " (or spaces) can't shift field indices. In that tail
                    // field 0 is `state` (proc stat field 3); `processor` (last
                    // CPU) is stat field 39, i.e. tail index 36.
                    let stat = read1("stat");
                    let tail = stat.rsplit(") ").next().unwrap_or("");
                    let sf: Vec<&str> = tail.split(' ').collect();
                    let state = sf.first().copied().unwrap_or("?");
                    let last_cpu = sf.get(36).copied().unwrap_or("?");
                    // schedstat field 2 = time runnable-but-not-running (ns):
                    // large here means "wanted the CPU, didn't get it"
                    // (starvation); near-zero with a kernel `wchan` means the
                    // thread was blocked in-kernel, not starved.
                    let schedstat = read1("schedstat");
                    let wait_ns = schedstat.split_whitespace().nth(1).unwrap_or("?");
                    let wchan = read1("wchan");
                    let wchan = if wchan.is_empty() {
                        "?"
                    } else {
                        wchan.as_str()
                    };
                    let status = read1("status");
                    let nonvol = status
                        .lines()
                        .find_map(|l| l.strip_prefix("nonvoluntary_ctxt_switches:"))
                        .map(str::trim)
                        .unwrap_or("?");
                    let _ = writeln!(
                        evidence,
                        "  vCPU {vcpu_id} (tid {tid}): state={state} \
                         last_cpu={last_cpu} runnable_wait_ns={wait_ns} \
                         wchan={wchan} nonvoluntary_ctxt_switches={nonvol}"
                    );
                }
                return Err(anyhow::Error::new(ApGateTimeout {
                    not_ready,
                    elapsed: gate_start.elapsed(),
                    killed: kill.load(Ordering::Acquire),
                    evidence,
                }));
            }
        }

        // BSP run loop. Wrapped in the same `with_vcpu_panic_ctx`
        // scope the APs use (symmetric panic-hook signaling) —
        // `kill` plus `bsp_done` are the pair analogous to a
        // vCPU thread's `kill` + `exited` so a BSP panic flips the
        // watchdog-observed flags before the panic=abort teardown.
        // `vcpu_panic::install_once` was already called in
        // `spawn_ap_threads` above, which runs even for a zero-AP VM,
        // so the hook is live by the time BSP enters its loop.
        if crate::vmm::debug_logging_enabled() {
            eprintln!("BSP: entering run loop");
        }
        let (exit_code, timed_out, bsp_exit_reason) = vcpu_panic::with_vcpu_panic_ctx(
            vcpu_panic::VcpuPanicCtx {
                kill: kill.clone(),
                exited: bsp_done.clone(),
                kill_evt: Some(kill_evt.clone()),
                exited_evt: Some(bsp_done_evt.clone()),
                // Hand the BSP's `bsp_alive` flag to the panic hook so a
                // panic-unwind path flips it to `false` BEFORE the
                // stack drop unmaps `bsp`'s `kvm_run` page. The
                // normal-exit path's post-join `bsp_alive.store(false)`
                // (see the `collect_results` finalization block) covers
                // `panic = "abort"` and the no-panic path; the panic
                // hook covers `panic = "unwind"` (test profile) where
                // the post-join store is unreachable. Mirrors
                // the AP-side `alive: Some(alive.clone())` plumbing in
                // spawn_ap_threads — every cross-thread holder of a
                // BSP `ImmediateExitHandle` (the freeze coordinator,
                // the watchdog) gates `ie.set` on this flag's
                // Acquire load, and a panic-released Release store
                // happens-before the unwind drop of `bsp`.
                alive: Some(bsp_alive.clone()),
            },
            || {
                self.run_bsp_loop(
                    &mut bsp,
                    &com1,
                    &com2,
                    Some(&virtio_con),
                    virtio_blk.as_ref(),
                    virtio_net.as_ref(),
                    ioapic_handle.as_ref(),
                    pci_bus_handle.as_ref(),
                    &kill,
                    &freeze,
                    &watchpoint,
                    &bsp_parked,
                    &bsp_regs,
                    has_immediate_exit,
                    run_start,
                    timeout,
                    Some(&parked_evt),
                    Some(&thaw_evt),
                    Some(&kill_evt),
                    tcr_el1_cache.as_ref(),
                    &cr3_cache,
                    &timed_out_flag,
                    &kern_virt_kaslr_for_bsp,
                    &kern_virt_kaslr_evt_for_bsp,
                    entry_syscall_64_link_kva,
                )
            },
        );
        // BSP loop returned normally (Ok path). Keep the guard ARMED through the
        // infallible teardown below (bsp_done / kill stores, eprintln, x86 route
        // tracing) and disarm it only just before the watchdog / coordinator
        // join — so a panic anywhere in this window (defense-in-depth against a
        // future fallible op added here) still joins every spawned thread before
        // bsp / vm drop rather than detaching them.
        //
        // Caller-side log of the loop's exit reason (see the
        // `run_bsp_loop` doc for why the loop doesn't print it
        // itself: the interactive-shell caller must defer this
        // past raw-mode terminal restore; here there is no raw
        // terminal, so print immediately).
        if crate::vmm::debug_logging_enabled() {
            eprintln!("BSP: loop exit reason={bsp_exit_reason:?}");
        }
        bsp_done.store(true, Ordering::Release);
        // Wake the freeze coordinator's epoll loop. Failure
        // (counter overflow / EAGAIN under EFD_NONBLOCK) is benign
        // — the panic-hook path may have already pushed an edge,
        // and the AtomicBool above is still authoritative for
        // `freeze_coord_bsp_done.load(Acquire)` if the eventfd
        // fails to deliver.
        let _ = bsp_done_evt.write(1);
        // Stop the monitor (wakes via kill_evt epoll) and bpf-write
        // thread (observes kill on next 200ms poll cycle).
        // Previously kill was deferred to collect_results, leaving
        // the monitor sampling at 100ms cadence through the entire
        // run_vm cleanup window (watchdog join + coord join).
        kill.store(true, Ordering::Release);
        let _ = kill_evt.write(1);
        // Sample cleanup start at the earliest moment after BSP exit so
        // every host-side teardown step lands inside the window, in
        // execution order: watchdog join (immediately below), AP joins,
        // monitor join, BPF writer join, bulk drain, exit-code and
        // crash-message extraction, and verifier-stat read (the rest
        // run inside `collect_results`). `collect_results` reads
        // `Instant::now()` at the end and the difference becomes
        // `VmResult::cleanup_duration`.
        let cleanup_start = Instant::now();
        // Cleanup-window dilation instrument: snapshot THIS thread's
        // schedstat at the window open. run_vm and collect_results run on
        // the same caller thread (the BSP thread), which performs every
        // join/drain in the window, so the delta across the window is the
        // cleanup performer's own on-CPU + runnable-wait — the evidence
        // the cleanup-budget gate judges overruns against (the per-phase
        // witness CANNOT attest this window: the monitor that feeds it is
        // itself joined inside it).
        let cleanup_sched_t0 = read_self_thread_schedstat();
        // `code` here is the run-loop sentinel (0 only on a BSP-
        // observed `ExitAction::Shutdown`, -1 otherwise — see
        // [`BspExitReason`] and the preceding `BSP: loop exit
        // reason=...` line). The caller-visible exit code is
        // derived from bulk-port `MSG_TYPE_EXIT` or the COM2 `KTSTR_EXIT:`
        // sentinel inside [`KtstrVm::collect_results`], not from
        // this value.
        if crate::vmm::debug_logging_enabled() {
            eprintln!(
                "BSP: exited run loop, code={exit_code} timed_out={timed_out} \
                 (run-loop sentinel — final exit code comes from bulk port / COM2 in collect_results)"
            );
        }

        // Surface IOAPIC routing-install failures (split-irqchip path). A
        // nonzero count means a guest-programmed device IRQ never got its MSI
        // route, so the device hung on first use — report it loudly so a
        // hung-device test shows the cause instead of an opaque timeout.
        // Per-failure errors are already logged in classify_exit; this is the
        // run-level summary. Fires on both clean-exit and watchdog-timeout
        // paths (run_bsp_loop returns from both before reaching here).
        #[cfg(target_arch = "x86_64")]
        if let Some(io) = ioapic_handle.as_ref() {
            let n = io.routing_failures();
            if n > 0 {
                tracing::error!(
                    count = n,
                    "ioapic: {n} KVM_SET_GSI_ROUTING install(s) failed this run — \
                     device IRQs for those pins did not deliver"
                );
            }
        }

        // Same surfacing for the full-irqchip MSI-X route owner: a nonzero count
        // means a guest-unmasked MSI-X vector never got its KVM route, so that
        // vector did not deliver (the NIC would hang on first use of it).
        #[cfg(target_arch = "x86_64")]
        if let Some(owner) = full_route_owner.as_ref() {
            let n = owner.routing_failures();
            if n > 0 {
                tracing::error!(
                    count = n,
                    "full-irqchip: {n} KVM_SET_GSI_ROUTING install(s) failed this \
                     run — MSI-X vectors for those routes did not deliver"
                );
            }
        }

        // Disarm the guard and reclaim every handle for the joins below; from
        // here its Drop is a no-op. Deferred to this point (not right after the
        // BSP loop) so the guard covers the infallible post-loop teardown above.
        // The watchdog/coord joins below consume these handles, so the disarm
        // cannot move past them; the post-join `read_tcr_el1` / `read_cr3`
        // catch-up reads (which must follow the coordinator join to avoid a
        // cache race) then run after this disarm — but they are infallible
        // `.ok()` reads with no panic site, a pre-existing sliver unchanged by
        // this fix. The `watchdog` / `freeze_coord_handle` bindings are `Option`
        // (always `Some` here — reached only after both spawned successfully).
        let RunVmHandles {
            ap_threads,
            monitor: monitor_handle,
            bpf_write: bpf_write_handle,
            freeze_coord: freeze_coord_handle,
            watchdog,
        } = guard.disarm();

        // Join the watchdog before dropping `bsp`. The watchdog holds an
        // ImmediateExitHandle pointing into bsp's kvm_run mmap. If bsp is
        // dropped first, the watchdog may write to unmapped memory.
        // (`Some` here — disarmed from the guard after a successful spawn.)
        if let Some(h) = watchdog {
            let _ = h.join();
        }
        if crate::vmm::debug_logging_enabled() {
            eprintln!("CLEANUP: watchdog joined");
        }

        // Join the freeze coordinator BEFORE `bsp` falls out of scope at
        // the end of this function. The coordinator's captured BSP
        // `ImmediateExitHandle` addresses bsp's kvm_run mmap; reachable
        // from multiple paths inside `freeze_and_dispatch` (TLV-driven
        // CAPTURE, user watchpoint, late-trigger, even after `bsp_done`
        // flips). Without this join, any of those paths can write
        // through a freed kvm_run mapping after bsp drops — a
        // use-after-free with hostile-input semantics.
        //
        // `bsp_done.store(true)` + `bsp_done_evt.write(1)` above
        // (lines around `BSP: exited run loop`) wake the coordinator's
        // epoll loop and break it out of the outer loop on the next
        // iteration, so this join does not deadlock; the watchdog's
        // own kill/bsp_done writes are also covered.
        //
        // Flip `bsp_alive` to `false` AFTER the join completes — at
        // that point the coordinator thread is gone and the gate is
        // belt-and-braces for any future restructuring that could
        // share the BSP IE handle outside this lifecycle.
        // (`Some` here — disarmed from the guard after a successful spawn.)
        if let Some(h) = freeze_coord_handle {
            let _ = h.join();
        }
        if crate::vmm::debug_logging_enabled() {
            eprintln!("CLEANUP: freeze_coord joined");
        }
        bsp_alive.store(false, Ordering::Release);

        // Make sure freeze is cleared before vCPU teardown so the APs
        // don't park-loop after we kick them. The freeze coordinator
        // has already joined above so it cannot re-set freeze=true.

        // Capture the virtio-blk counter Arc before the device's
        // outer `Arc<PiMutex<VirtioBlk>>` falls out of scope. The
        // device's `counters()` accessor clones the inner
        // `Arc<VirtioBlkCounters>`; this transfers a reader-side
        // handle onto `VmRunState` so `collect_results` can attach
        // it to `VmResult` without holding the device alive past
        // its current ownership.
        // Read via `blk_device` (set on both transports), NOT `virtio_blk`
        // (`None` on x86): the counters live in the shared device regardless of
        // whether it is reached through MMIO (aarch64) or the PCI bus (x86).
        let virtio_blk_counters = blk_device.as_ref().map(|d| d.lock().counters());
        // `virtio_net_counters` was captured at device construction (the
        // transport-split init above): on x86_64 from `NetDeviceHandles`
        // (the PCI function owns the device core, so the `virtio_net`
        // MMIO handle is `None` here), on aarch64 from the MMIO handle.

        // Best-effort final TCR_EL1 read from the post-exit BSP.
        // The BSP loop's lazy CAS already populates `tcr_el1_cache`
        // via `read_tcr_el1`; this final read covers the (rare)
        // case where the loop exited before the kernel programmed
        // the MMU (early-boot crash). On x86_64 `read_tcr_el1`
        // returns None and the cache stays None.
        if let Some(ref cache) = tcr_el1_cache
            && cache.load(Ordering::Acquire) == 0
            && let Some(val) = exit_dispatch::read_tcr_el1(&mut bsp)
            && val != 0
        {
            cache.store(val, Ordering::Release);
        }
        // Best-effort final CR3 / TTBR1_EL1 read from the post-exit
        // BSP. Mirrors the TCR_EL1 catch-up above: the BSP loop's
        // lazy CAS populates `cr3_cache` once the kernel installs
        // its post-randomization page tables; this catch-up store
        // covers the (rare) case where the loop exited before
        // `__startup_64` / `__cpu_setup` ran. Failure-dump consumers
        // that read `cr3_cache` post-exit (e.g. for late
        // `phys_base` resolution against a frozen VM) get the live
        // CR3 instead of the bootstrap zero.
        if cr3_cache.load(Ordering::Acquire) == 0
            && let Some(val) = exit_dispatch::read_cr3(&mut bsp)
            && val != 0
        {
            cr3_cache.store(val, Ordering::Release);
        }

        // Host-side vCPU scheduling-dilation sample. Read HERE — after the
        // watchdog/freeze-coord joins above but BEFORE `ap_threads` move
        // into `VmRunState` (they are joined only later, in
        // `collect_results`). A live vCPU is read from proc; an AP that
        // already returned contributes the one-shot self-snapshot it stored
        // immediately before exit. TIDs come from the same
        // `vcpu_tid_slots` the monitor uses — index 0 is the BSP (this
        // thread), 1.. are the AP slots each AP stamped before entering
        // KVM_RUN; a slot still 0 means that AP never scheduled and is
        // skipped by `read_host_vcpu_schedstat`. Each vCPU thread lives
        // exactly one VM run (BSP is this thread; APs are spawned fresh
        // per `run_vm` in `spawn_ap_threads` and never reused), so the
        // whole-thread-life schedstat totals are this run's totals — no
        // baseline subtraction is needed.
        let host_vcpu_tids: Vec<i32> = vcpu_tid_slots
            .iter()
            .map(|(slot, _)| slot.load(Ordering::Acquire))
            .collect();
        let host_vcpu_schedstat =
            read_host_vcpu_schedstat_with_exit_snapshots(&host_vcpu_tids, &vcpu_exit_schedstats);
        // Finalize before the AP handles move into VmRunState. Vanished proc
        // task directories are covered by the aligned exit snapshots. The
        // monitor may take a later no-op sample, but the recorder's finalized
        // latch prevents post-result mutation.
        let contention_witness = contention_recorder.finish();

        // Final ledger snapshot for the VmResult phase/milestone fields.
        // The dispatch consumers that advance these are joined/dead by
        // now, so this is the run's final state.
        let final_ledger = progress_ledger_for_result.snapshot();

        Ok(VmRunState {
            exit_code,
            timed_out,
            // Final kill-reason byte: the watchdog thread joined above, so
            // this load observes its last store (the join synchronizes).
            watchdog_kill_reason_raw: watchdog_kill_reason.load(Ordering::Acquire),
            final_guest_phase_raw: final_ledger.phase,
            final_progress_epoch: final_ledger.progress_epoch,
            bpf_map_write_delivery_raw: bpf_map_write_delivery.load(Ordering::Acquire),
            periodic_prereqs_ready_ns_raw: periodic_prereqs_ready_at.load(Ordering::Acquire),
            periodic_window_end_ns_raw: periodic_window_end_at.load(Ordering::Acquire),
            contention_witness,
            ap_threads,
            monitor_handle,
            bpf_write_handle,
            // Coordinator is already joined above (before `bsp` drops)
            // to prevent UAF on the BSP `ImmediateExitHandle`.
            // `collect_results`'s `if let Some(h) = ...` join is a
            // no-op for the `None` arm.
            freeze_coordinator: None,
            com1,
            com2,
            kill,
            kill_evt,
            freeze,
            vm,
            cleanup_start,
            cleanup_sched_t0,
            virtio_blk_counters,
            virtio_net_counters,
            // Snapshot bridge owning every report stored by the
            // freeze coordinator's TLV-driven snapshot handler
            // over the run's lifetime. Forwarded to
            // `VmResult::snapshot_bridge` by `collect_results`.
            snapshot_bridge,
            tcr_el1: tcr_el1_cache,
            cr3: cr3_cache,
            vmlinux_data: vmlinux_data_for_result,
            prog_accessor: prog_accessor_slot.lock_unpoisoned().take(),
            kern_phys_base: kern_phys_base_for_result.load(Ordering::Acquire),
            // Snapshot the kern_virt_kaslr Arc at run-end. The Arc
            // stores `actual_offset + 1` (bias) so 0 = "never
            // published" and `saturating_sub(1)` recovers the actual
            // value (with 0 meaning either "never published" OR
            // "published as 0", indistinguishable from the consumer's
            // perspective — e2e tests distinguish via the test
            // entry's `kaslr` attribute).
            kern_kaslr_offset: kern_virt_kaslr_for_result
                .load(Ordering::Acquire)
                .saturating_sub(1),
            // Virtio-console handle threaded into `collect_results`
            // for the post-exit `drain_bulk()` call. Carries any
            // port-1 TLV bytes the guest wrote that the freeze
            // coordinator's tx_evt-driven mid-run drain did not
            // already consume; the merge into `guest_messages` keeps
            // existing readers (crate::test_support::eval, sidecar) working without
            // any per-message-type code change.
            virtio_con,
            // Mid-run TLV entries the freeze coordinator already
            // consumed. `collect_results` merges these with the
            // post-exit bulk drain and the COM2 panic-message
            // extraction so every frame the guest published reaches
            // the verdict.
            bulk_messages: freeze_coord_bulk_messages,
            // Scheduler-stats client constructed at the top of
            // `run_vm`. Its drainer thread has been alive since the
            // guest started forwarding stats responses; the client
            // is threaded onto `VmResult` for test-code access.
            stats_client,
            // Periodic-capture count published by the coordinator
            // run-loop after every successful fire / placeholder
            // store. Read AFTER `freeze_coord_handle.join()` ran so
            // the AtomicU32's value is the final advance count;
            // `collect_results` forwards onto
            // `VmResult::periodic_fired`.
            periodic_fired: periodic_fired_slot.load(Ordering::Relaxed),
            // Configured periodic-target plumbed onto KtstrVm via
            // `KtstrVmBuilder::num_snapshots`. Forwarded to
            // `VmResult::periodic_target` so test code can compute
            // coverage as `fired / target`.
            periodic_target: self.num_snapshots,
            // Watchpoint Arc forwarded so `collect_results` can
            // invalidate `kind_host_ptr` and `request_kva` after
            // every vCPU thread joins but before `vm` drops.
            watchpoint,
            // Host-side vCPU dilation sample read just above from live proc
            // entries plus one-shot AP exit snapshots. Forwarded verbatim to
            // `VmResult::host_vcpu_schedstat` by `collect_results`.
            host_vcpu_schedstat,
        })
    }

    /// Spawn AP vCPU threads. Each thread optionally pins itself to a
    /// host CPU from `pin_targets` (indexed by AP order, 0-based), OR
    /// applies a CPU mask from `no_perf_mask` when the no-perf +
    /// `--cpu-cap` path is active. The two are mutually exclusive —
    /// perf-mode produces `pin_targets` via the PinningPlan;
    /// `--cpu-cap` no-perf produces `no_perf_mask` via the LlcPlan.
    ///
    /// Returns `(threads, freeze_handles)`. The freeze handles
    /// (per-AP `parked` flags + register-snapshot slots) are the
    /// freeze coordinator's view of each AP; they live separately
    /// from `VcpuThread` so the thread struct stays minimal —
    /// `VcpuThread` carries only what teardown (kick + join) needs.
    /// Callers that don't run a freeze coordinator (e.g. interactive
    /// shell) discard `freeze_handles`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn spawn_ap_threads(
        &self,
        vcpus: Vec<kvm_ioctls::VcpuFd>,
        has_immediate_exit: bool,
        com1: &Arc<PiMutex<console::Serial>>,
        com2: &Arc<PiMutex<console::Serial>>,
        virtio_con: Option<&Arc<PiMutex<virtio_console::VirtioConsole>>>,
        virtio_blk: Option<&Arc<PiMutex<virtio_blk::VirtioBlk>>>,
        virtio_net: Option<&Arc<PiMutex<virtio_net::VirtioNet>>>,
        ioapic: Option<&Arc<crate::vmm::IoapicHandle>>,
        pci_bus: Option<&Arc<PiMutex<crate::vmm::pci::PciBus>>>,
        kill: &Arc<AtomicBool>,
        kill_evt: &Arc<EventFd>,
        freeze: &Arc<AtomicBool>,
        watchpoint: &Arc<WatchpointArm>,
        pin_targets: &[Option<usize>],
        no_perf_mask: Option<&[usize]>,
        ap_tid_slots: &[(Arc<AtomicI32>, Arc<crate::sync::Latch>)],
        ap_boot_latches: &[Arc<crate::sync::Latch>],
        parked_evt: Option<&Arc<EventFd>>,
        thaw_evt: Option<&Arc<EventFd>>,
    ) -> Result<(Vec<VcpuThread>, ApFreezeHandles)> {
        // Register the process-wide panic hook that flips `kill` +
        // `exited` on a panicking vCPU thread before the
        // panic=abort-induced process teardown. Idempotent via
        // `Once`; safe to call on every VM spawn.
        vcpu_panic::install_once();
        let n = vcpus.len();
        debug_assert_eq!(ap_tid_slots.len(), n);
        debug_assert_eq!(ap_boot_latches.len(), n);
        // Guard the partially-built AP set: a mid-loop `?` below (exit-eventfd
        // alloc / thread spawn) must kick+join the already-spawned APs before
        // the caller's `vm` munmaps guest_mem — the outer RunVmThreadGuard is
        // not armed until spawn_ap_threads returns. Disarmed on success.
        let mut spawn_guard = PartialApSpawnGuard {
            ap_threads: Vec::with_capacity(n),
            kill: kill.clone(),
            kill_evt: kill_evt.clone(),
            freeze: freeze.clone(),
        };
        let mut freeze_parked: Vec<Arc<AtomicBool>> = Vec::with_capacity(n);
        let mut freeze_regs: Vec<Arc<std::sync::Mutex<Option<exit_dispatch::VcpuRegSnapshot>>>> =
            Vec::with_capacity(n);
        for (i, mut vcpu) in vcpus.into_iter().enumerate() {
            let ie_handle = if has_immediate_exit {
                Some(ImmediateExitHandle::from_vcpu(&mut vcpu))
            } else {
                None
            };
            let kill_clone = kill.clone();
            let kill_evt_clone = kill_evt.clone();
            let freeze_clone = freeze.clone();
            let com1_clone = com1.clone();
            let com2_clone = com2.clone();
            let vc_clone = virtio_con.cloned();
            let vblk_clone = virtio_blk.cloned();
            let vnet_clone = virtio_net.cloned();
            let ioapic_clone = ioapic.cloned();
            let pci_bus_clone = pci_bus.cloned();
            let exited = Arc::new(AtomicBool::new(false));
            let exited_clone = exited.clone();
            let schedstat_at_exit = Arc::new(std::sync::Mutex::new(None));
            let schedstat_at_exit_thread = Arc::clone(&schedstat_at_exit);
            let parked = Arc::new(AtomicBool::new(false));
            let parked_clone = parked.clone();
            let regs = Arc::new(std::sync::Mutex::new(None));
            let regs_clone = regs.clone();
            // Per-AP `alive` flag mirroring the BSP `bsp_alive` gate.
            // Initialised to `true`; the AP panic hook (via
            // `VcpuPanicCtx::alive`) flips it to `false` BEFORE
            // unwinding drops `vcpu` and its `kvm_run` mmap, so the
            // freeze coordinator's pass-1 kick loop and the
            // `arm_user_watchpoint` kick gate every `ie.set` on a
            // fresh Acquire load and skip indices whose mmap is
            // about to disappear. Under `panic = "abort"` (release)
            // unwinding never runs and the flag stays `true` for
            // the life of the run; the gate is then a no-op,
            // matching the BSP belt-and-braces semantic.
            let alive = Arc::new(AtomicBool::new(true));
            let has_immediate_exit_clone = has_immediate_exit;
            let pin_cpu = pin_targets.get(i).copied().flatten();
            let mask_for_thread: Option<Vec<usize>> = no_perf_mask.map(|m| m.to_vec());
            // Per-AP shared watchpoint state. Cloned once per AP;
            // the AP polls `wp_clone.request_kva` before each
            // KVM_RUN (via the per-iteration hook in
            // `vcpu_run_loop_unified`) and self-arms via
            // [`self_arm_watchpoint`] when the freeze coordinator
            // publishes the resolved `*scx_root->exit_kind` KVA.
            // The same clone is what the `VcpuExit::Debug` arm in
            // [`exit_dispatch::classify_exit`] uses to set
            // `wp_clone.hit` so the late-trigger poll observes the
            // watchpoint fire.
            let wp_clone = watchpoint.clone();

            let rt = self.performance_mode;
            // Per-AP exit eventfd for `VcpuThread::wait_for_exit` so
            // teardown blocks in `epoll_wait` instead of sleep-polling
            // `exited`. Bumped from inside the closure right after
            // `exited.store(true)` and from the panic hook (via
            // `panic_ctx.exited_evt`) so the parent observes both
            // normal-exit and panic-classified shutdowns through the
            // same fd. EFD_NONBLOCK so a Drop-time write cannot
            // stall.
            let exit_evt =
                Arc::new(EventFd::new(EFD_NONBLOCK).context("create AP vCPU exit eventfd")?);
            let exit_evt_thread = Arc::clone(&exit_evt);
            let panic_ctx = vcpu_panic::VcpuPanicCtx {
                kill: kill.clone(),
                exited: exited.clone(),
                kill_evt: Some(kill_evt.clone()),
                exited_evt: Some(Arc::clone(&exit_evt)),
                // Hand the AP's `alive` flag to the panic hook so a
                // panic-unwind path flips it to `false` BEFORE the
                // stack drop unmaps `vcpu`'s `kvm_run` page. The
                // freeze coordinator's pass-1 kick gates each
                // `ie.set` on this flag's Acquire load.
                alive: Some(alive.clone()),
            };
            let (tid_slot_clone, tid_latch_clone) = {
                let (s, l) = &ap_tid_slots[i];
                (Arc::clone(s), Arc::clone(l))
            };
            let boot_latch_clone = Arc::clone(&ap_boot_latches[i]);
            // Clone the shared parked_evt + thaw_evt for this AP.
            // None when the caller (interactive shell) doesn't run a
            // freeze coordinator; in that case `vcpu_run_loop_unified`
            // never observes a freeze and the eventfd is unused.
            let parked_evt_clone: Option<Arc<EventFd>> = parked_evt.cloned();
            let thaw_evt_clone: Option<Arc<EventFd>> = thaw_evt.cloned();
            let handle = std::thread::Builder::new()
                .name(format!("vcpu-{}", i + 1))
                .spawn(move || {
                    register_vcpu_signal_handler();
                    // Stamp this thread's Linux TID into the per-AP
                    // slot so the monitor can open `perf_event_open`
                    // counters bound to the vCPU thread. Done
                    // BEFORE pinning / RT / KVM_RUN so the value is
                    // visible to any reader the moment the thread is
                    // schedulable. The companion `Latch::set` lets
                    // `open_vcpu_perf_capture` block in
                    // `Latch::wait_timeout` instead of sleep-polling
                    // the atomic. SAFETY: SYS_gettid is the standard
                    // syscall returning this thread's pid_t; no
                    // inputs.
                    let tid = unsafe { libc::syscall(libc::SYS_gettid) } as i32;
                    tid_slot_clone.store(tid, Ordering::Release);
                    tid_latch_clone.set();
                    if let Some(cpu) = pin_cpu {
                        pin_current_thread(cpu, &format!("vCPU {}", i + 1));
                    } else if let Some(mask) = mask_for_thread.as_deref() {
                        set_thread_cpumask(mask, &format!("vCPU {}", i + 1));
                    }
                    if rt {
                        set_rt_priority(1, &format!("vCPU {}", i + 1));
                    }
                    // The watchpoint Arc travels into the run loop
                    // via the `vcpu_run_loop_unified` parameter; the
                    // loop self-arms before each `vcpu.run()` and
                    // sets `watchpoint.hit` on `KVM_EXIT_DEBUG`. The
                    // per-AP `armed_kva` slot that tracks the
                    // currently-programmed `debugreg[0]` lives
                    // inside the loop now, so a single pre-loop
                    // attempt would have been a redundant ioctl
                    // with no effect — the coordinator typically
                    // publishes the resolved KVA AFTER the AP has
                    // entered the loop (once a sched_ext scheduler
                    // attaches and `*scx_root != 0`).
                    // Boot-ordering signal: everything the AP must do before
                    // it can safely receive INIT-SIPI is now complete (signal
                    // handler, affinity, RT prio); the KVM_RUN that follows
                    // will block with MP_STATE_UNINITIALIZED, at which point
                    // KVM buffers the guest's INIT/SIPI. Fire the latch here,
                    // the last statement before the run loop, so the BSP gate
                    // in `run_vm` cannot release until this AP is guaranteed
                    // to catch its bring-up IPI. This is the last point in the
                    // closure that is not inside `vcpu_run_loop_unified`; the
                    // pre-loop steps above are all infallible (they log and
                    // continue on error), so the only way to reach the run
                    // loop without firing this latch is a panic — which the
                    // installed vcpu panic hook turns into a `kill` store that
                    // the BSP gate also observes, so it never hangs.
                    boot_latch_clone.set();
                    vcpu_panic::with_vcpu_panic_ctx(panic_ctx, || {
                        vcpu_run_loop_unified(
                            &mut vcpu,
                            &com1_clone,
                            &com2_clone,
                            vc_clone.as_ref(),
                            vblk_clone.as_ref(),
                            vnet_clone.as_ref(),
                            ioapic_clone.as_ref(),
                            pci_bus_clone.as_ref(),
                            &kill_clone,
                            &kill_evt_clone,
                            &freeze_clone,
                            &parked_clone,
                            &regs_clone,
                            &wp_clone,
                            has_immediate_exit_clone,
                            parked_evt_clone.as_ref(),
                            thaw_evt_clone.as_ref(),
                        );
                    });
                    // `/proc/self/task/<tid>` vanishes as soon as this closure
                    // returns. Preserve one final self-snapshot so host
                    // contention finalization cannot lose an AP that exited
                    // between the last lifecycle frame and result assembly.
                    // This is a single read per AP lifetime, never polling.
                    if let Ok(line) = std::fs::read_to_string("/proc/thread-self/schedstat")
                        && let Some((on_cpu, run_delay)) = parse_schedstat_line(line.trim())
                    {
                        *schedstat_at_exit_thread.lock_unpoisoned() = Some(HostVcpuSchedstat {
                            total_on_cpu_ns: on_cpu,
                            total_run_delay_ns: run_delay,
                            sampled_vcpus: 1,
                        });
                    }
                    // wp_clone is held for the AP's entire lifetime
                    // so the strong count never drops to zero before
                    // the freeze coordinator joins.
                    drop(wp_clone);
                    exited_clone.store(true, Ordering::Release);
                    // Wake any thread blocked in `wait_for_exit` on
                    // this AP's exit_evt. Failure (counter overflow)
                    // is harmless — a previous edge already unblocks
                    // the waiter; only the edge from 0 to non-zero
                    // matters.
                    let _ = exit_evt_thread.write(1);
                    vcpu
                })
                .with_context(|| format!("spawn vCPU {} thread", i + 1))?;

            spawn_guard.ap_threads.push(VcpuThread {
                handle,
                exited,
                immediate_exit: ie_handle,
                exit_evt,
                alive,
                schedstat_at_exit,
            });
            freeze_parked.push(parked);
            freeze_regs.push(regs);
        }
        // All APs spawned — reclaim the set so the guard's Drop is a no-op.
        let ap_threads = spawn_guard.disarm();
        Ok((
            ap_threads,
            ApFreezeHandles {
                parked: freeze_parked,
                regs: freeze_regs,
            },
        ))
    }

    /// Start the monitor thread if vmlinux is available.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn start_monitor(
        &self,
        vm: &kvm::KtstrKvm,
        kill: &Arc<AtomicBool>,
        kill_evt: &Arc<EventFd>,
        run_start: Instant,
        vcpu_pthreads: Vec<libc::pthread_t>,
        contention_recorder: Arc<ContentionWitnessRecorder>,
        perf_capture: Arc<Option<monitor::perf_counters::PerfCountersCapture>>,
        virtio_con: Option<Arc<PiMutex<virtio_console::VirtioConsole>>>,
        sys_rdy_evt: Option<Arc<EventFd>>,
        tcr_el1: Option<Arc<std::sync::atomic::AtomicU64>>,
        cr3: Arc<std::sync::atomic::AtomicU64>,
        watchdog_reset_ns: Arc<std::sync::atomic::AtomicU64>,
        watchdog_reset_tag: Arc<std::sync::atomic::AtomicU8>,
        progress_ledger: Arc<monitor::ProgressLedger>,
        kern_phys_base_shared: Arc<std::sync::atomic::AtomicU64>,
        kern_phys_base_evt: Arc<EventFd>,
        kern_virt_kaslr_shared: Arc<std::sync::atomic::AtomicU64>,
        kern_addrs_frames: Arc<std::sync::atomic::AtomicU64>,
        kern_addrs_crc_bad: Arc<std::sync::atomic::AtomicU64>,
    ) -> Result<Option<JoinHandle<monitor::reader::MonitorLoopResult>>> {
        let Some(vmlinux) = find_vmlinux(&self.kernel) else {
            return Ok(None);
        };
        // Parse the vmlinux once per (path, mtime) per process and
        // share the derived products with the freeze-coord inline
        // link-KVA path (which populated this same cache entry earlier
        // in `run_vm`). The previous structure re-ran
        // `goblin::elf::Elf::parse` over ~50 MB+ of vmlinux and
        // `KernelSymbols::from_elf` here even though the inline path
        // already did both — seconds of duplicate CPU per VM. The BTF
        // parse (previously here, hitting `load_btf_from_elf` +
        // `Btf::from_bytes` once) is now folded into the same cached
        // parse so it still happens exactly once; the boot-window
        // concern (early samples seeing the rq's pre-AP-online state
        // when BTF was parsed twice) is preserved. All products are
        // owned, so the monitor closure below clones them out of the
        // shared `Arc` exactly as if it had parsed them itself.
        let Some(artifacts) = super::vmlinux::cached_vmlinux_artifacts(&vmlinux) else {
            return Ok(None);
        };
        // `monitor: None` means the BTF load or `KernelOffsets`
        // resolution failed — no monitor thread, matching the pre-cache
        // `load_btf_from_elf` / `from_btf` early returns. `symbols`
        // resolving but `monitor` absent is the same "symbols Ok,
        // offsets Err → Ok(None)" the old `(Ok, Ok)` gate produced.
        let Some(mon) = artifacts.monitor.as_ref() else {
            return Ok(None);
        };
        let symbols = artifacts.symbols.clone();
        let offsets = mon.offsets.clone();
        let prog_offsets = mon.prog_offsets.clone();
        // System-wide PSI-irq host-walk offsets (psi_group total/avg + the
        // PSI_IRQ_FULL index). `None` when CONFIG_IRQ_TIME_ACCOUNTING is off
        // (PSI_IRQ_FULL absent from BTF) or psi_group is absent → loud-absent.
        let psi_offsets = mon.psi_offsets;
        let btf = Arc::clone(&mon.btf);
        // Raw vmlinux bytes for the `watch_bpf_maps` per-map program-BTF
        // split-base. A byte-cache hit — `cached_vmlinux_artifacts`
        // above already read and cached these. Only consumed when the
        // test declared watch targets; the `None` arm keeps the
        // pre-cache "unreadable vmlinux → no monitor" early return.
        let vmlinux_data_arc = match super::vmlinux::cached_vmlinux_bytes(&vmlinux) {
            Some(d) => d,
            None => return Ok(None),
        };
        // BTF-capability probe for the `SCX_EV_*` event counters:
        // `event_offsets` resolves only on kernels that expose the
        // `struct scx_sched` / `scx_root` machinery the monitor walks to
        // reach them (a 6.16-cycle addition); on older kernels (e.g.
        // 6.14) it is `None`. Captured here where the resolved
        // `KernelOffsets` is owned, then forwarded onto every
        // `MonitorLoopResult` this closure produces so `MonitorReport`
        // consumers can distinguish "counters absent because the kernel
        // lacks them" from "counters empty because capture regressed".
        let scx_event_counters_supported = offsets.event_offsets.is_some();

        // `watch_bpf_maps` prep: the vmlinux BTF is the split-base for per-map
        // program-BTF loads; `prog_offsets` is cloned because the
        // `prog_stats_ctx` builder below consumes the original; the test
        // entry's `WatchBpfMap` list is lowered to the monitor-local target
        // form. All are moved into the monitor closure and consumed there
        // only when the test declared watch targets. `btf` is already an
        // `Arc<Btf>` cloned from the shared parse artifact above.
        let watch_prog_offsets = prog_offsets.clone();
        let watch_targets: Vec<monitor::reader::WatchBpfMapTarget> = self
            .watch_bpf_maps
            .iter()
            .map(|p| monitor::reader::WatchBpfMapTarget {
                map_suffix: p.map_name_suffix.clone(),
                field: p.field.clone(),
                per_cpu: matches!(
                    p.agg,
                    crate::test_support::BpfMapAgg::PerCpu
                        | crate::test_support::BpfMapAgg::PerCpuCounter
                ),
                counter: matches!(
                    p.agg,
                    crate::test_support::BpfMapAgg::ScalarCounter
                        | crate::test_support::BpfMapAgg::PerCpuCounter
                ),
                label: p.label.clone(),
            })
            .collect();

        let mem = match vm.numa_layout.as_ref() {
            Some(layout) => monitor::reader::GuestMem::from_layout(layout, &vm.guest_mem),
            None => {
                use vm_memory::GuestMemoryRegion;
                let host_base = vm
                    .guest_mem
                    .get_host_address(GuestAddress(DRAM_BASE))
                    .context("resolve guest DRAM base host address (monitor)")?;
                // Size of the first contiguous region only.
                // host_base addresses that single mapping; using the
                // sum of all region lengths would extend past the
                // mapping into host heap when multiple regions exist.
                let mem_size = vm
                    .guest_mem
                    .iter()
                    .next()
                    .context("guest_mem must have at least one region (monitor)")?
                    .len();
                // SAFETY: host_base is from GuestMemoryMmap's mapping,
                // which outlives this GuestMem (owned by `vm` until
                // return).
                unsafe { monitor::reader::GuestMem::new(host_base, mem_size) }
            }
        };
        // Share guest DRAM between the sampling loop (which borrows `&*mem`)
        // and the `watch_bpf_maps` watcher's owned map accessor (which holds
        // an `Arc::clone`) — both read-only over the same backing.
        let mem = Arc::new(mem);
        let num_cpus = self.topology.total_cpus();
        let kill_clone = kill.clone();
        let kill_evt_clone = kill_evt.clone();
        // Clone the boot-complete eventfd handle for the monitor
        // closure. Captured by `move` into the spawned thread so
        // the `epoll_wait` dispatch can register the fd alongside
        // `kill_evt` and the timerfd. `Option::None` short-circuits
        // the pre-sample wait so the test path (no virtio-console)
        // and any `EventFd::new` failure both fall through to the
        // sample loop directly.
        let monitor_sys_rdy_evt = sys_rdy_evt.clone();
        let dump_trigger = self
            .monitor_thresholds
            .map(|thresholds| monitor::reader::DumpTrigger {
                thresholds,
                virtio_con: virtio_con.clone(),
            });

        let hz = monitor::guest_kernel_hz(Some(&self.kernel));
        // ms-precision conversion lives in [`duration_to_jiffies`];
        // see its doc for why the seconds-based form is wrong.
        //
        // Under coverage instrumentation (cargo-llvm-cov builds set
        // `--cfg coverage`) the guest scx watchdog timeout is scaled by
        // [`GUEST_SCX_WATCHDOG_COVERAGE_SCALE`]: the in-guest test binary
        // and fixture schedulers run instrumented (~2x slower), and the
        // coverage CI legs run on colocated runners whose deepened
        // oversubscription starves individual vCPUs asymmetrically — the
        // measured combination that tripped the tight 5 s default on
        // churn-heavy fixtures while the SAME tests pass uninstrumented
        // on the SAME runners. Compile-time-gated so non-coverage builds
        // are byte-identical and the tight default keeps its full bug-
        // sorting value everywhere else; the loud log line below keeps a
        // coverage-log reader from chasing a 15 s eviction as a mystery.
        let watchdog_jiffies = self.watchdog_timeout.map(|d| {
            let j = duration_to_jiffies(d, hz);
            if GUEST_SCX_WATCHDOG_COVERAGE_SCALE > 1 {
                eprintln!(
                    "ktstr: coverage build — guest scx watchdog_timeout scaled \
                     {GUEST_SCX_WATCHDOG_COVERAGE_SCALE}x ({d:?} base)",
                );
            }
            j.saturating_mul(GUEST_SCX_WATCHDOG_COVERAGE_SCALE)
        });
        let preemption_threshold_ns = monitor::vcpu_preemption_threshold_ns(Some(&self.kernel));
        let service_cpu = self.pinning_plan.as_ref().and_then(|p| p.service_cpu);
        // Workload duration captured for the scheduler-attach
        // watchdog reset. `Some(d)` enables the reset; the
        // monitor closure constructs a
        // [`monitor::reader::WatchdogReset`] payload pairing this
        // duration with the resolved `scx_root_pa` and the shared
        // `watchdog_reset_ns` atomic once `symbols.scx_root`
        // resolves below. `None` (the builder default) leaves
        // [`monitor::reader::MonitorConfig::watchdog_reset`] as
        // `None`, and the loop's reset detection short-circuits.
        let workload_duration = self.workload_duration;

        let handle = std::thread::Builder::new()
            .name("vmm-monitor".into())
            .spawn(move || {
                if let Some(cpu) = service_cpu {
                    pin_current_thread(cpu, "monitor");
                }
                // The hang detector's own sensing must not dilate with the
                // load it measures. FIFO-2 is UNCONDITIONAL (not perf-mode-
                // gated): the monitor bumps `monitor_heartbeat` and writes
                // the ledger the watchdog reads, so a SCHED_OTHER monitor
                // that slips its ~100 ms sample under extreme host dilation
                // makes the watchdog latch `monitor_live=false` and fire the
                // deadman on a still-progressing cell (the monitor-starvation
                // residual). The per-tick cost is ~µs, so FIFO-2 grants
                // scheduling immunity at no meaningful CPU cost: perf-mode
                // vCPUs at FIFO-1 stay below it; in no-perf/default mode it
                // now outranks SCHED_OTHER vCPUs — intended, sensing must
                // win. Best-effort — warns once/process without CAP_SYS_NICE
                // (see `set_rt_priority`). The service-CPU PIN above stays
                // perf-mode-gated: `service_cpu` is `None` without a reserved
                // CPU.
                set_rt_priority(2, "monitor");
                // Pre-resolution boot-complete wait, hoisted ABOVE
                // the `phys_base` / `pco_pa` / scx_root_pa /
                // watchdog_pa / `page_offset_base_pa` resolution
                // that follows. Previously this thread either
                // resolved `phys_base` immediately (with `cr3=0` →
                // `phys_base=0` → every KASLR text/data PA wrong)
                // or polled CR3 with a short busy-wait that fires
                // too early — CR3 is set in `__startup_64`, but
                // `setup_per_cpu_areas` (which populates
                // `__per_cpu_offset[]`) and KASLR randomization of
                // `page_offset_base` finish much later in
                // `start_kernel`. Resolving `pco_pa` /
                // `page_offset_base_pa` between those two events
                // produces baked-in stale PAs that the
                // per-iteration refresh inside `monitor_loop` cannot
                // recover from.
                //
                // The `MSG_TYPE_SYS_RDY` TLV frame is emitted by
                // `ktstr_guest_init` after `mount_filesystems()`
                // — strictly AFTER `__startup_64` (CR3 latch),
                // `__cpu_setup` (TCR_EL1 latch), `setup_per_cpu_areas`
                // (`__per_cpu_offset[]` populated), KASLR
                // randomization (`page_offset_base` populated), and
                // userspace init startup. By blocking here on the
                // sys_rdy eventfd, the resolution that follows runs
                // against a guest in steady state: every read in
                // `resolve_phys_base`, `resolve_page_offset_with_tcr`,
                // and the text-mapped PA recomputes lands on
                // populated guest memory.
                //
                // Three exit conditions:
                //   1. sys_rdy fires: proceed to phys_base resolve.
                //   2. kill fires: VM died before booting; return
                //      empty MonitorLoopResult immediately.
                //   3. 5 s timeout: best-effort fall through. The
                //      downstream `data_valid` gate inside
                //      `monitor_loop` still guards every walk, so
                //      reads of pre-boot zeros are tolerated and
                //      the monitor produces an empty sample set
                //      rather than chasing pointers through wrong
                //      PAs.
                //
                // `MonitorConfig::sys_rdy` is set to `None` below
                // because the wait has already happened here —
                // re-running the wait inside `monitor_loop` would
                // be a no-op (sys_rdy is edge-triggered, the eventfd
                // counter has been read by this wait and the
                // `Option::take` in the freeze-coord TOKEN_TX
                // handler also fires only once).
                //
                // `boot_wait_outcome` (declared here so it survives to the
                // result-wrap below) records whether the sys_rdy eventfd
                // fired vs the 5 s ceiling elapsing. It flows ONLY to the
                // closure-wrap on the sampled-run path; every kill early-
                // return below carries `NotConfigured`, because a monitor
                // killed during setup never reaches the sample loop — so
                // the invariant is `Fired` ⟺ the monitor sampled. Surfaced
                // on MonitorReport for the sys_rdy-regression test.
                let mut boot_wait_outcome = monitor::BootWaitOutcome::NotConfigured;
                if let Some(sys_rdy) = monitor_sys_rdy_evt.as_deref() {
                    use std::os::unix::io::AsRawFd;
                    use vmm_sys_util::epoll::{
                        ControlOperation, Epoll, EpollEvent, EventSet,
                    };
                    // Upfront kill check: BSP can exit before the
                    // monitor thread is scheduled (fast 1-CPU tests
                    // that fall through `test_main` in milliseconds).
                    // In that case `run_vm` has already stored
                    // kill + written kill_evt; entering the
                    // boot epoll would still wake immediately on
                    // kill_fd, but skipping the syscall trip
                    // entirely is cheaper and avoids the small
                    // window where epoll_create / epoll_ctl could
                    // race with VM teardown.
                    if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                        return monitor::reader::MonitorLoopResult {
                            samples: Vec::new(),
                            drain: crate::vmm::host_comms::BulkDrainResult {
                                entries: Vec::new(),
                            },
                            watchdog_observation: None,
                            page_offset: 0,
                            preemption_threshold_ns,
                            boot_wait_outcome: monitor::BootWaitOutcome::NotConfigured,
                            scx_event_counters_supported,
                            // Monitor torn down before sampling: no live prog seen.
                            verified_insns: Vec::new(),
                        };
                    }
                    let kill_fd = kill_evt_clone.as_raw_fd();
                    let boot_fd = sys_rdy.as_raw_fd();
                    if let Ok(boot_epoll) = Epoll::new() {
                        let _ = boot_epoll.ctl(
                            ControlOperation::Add,
                            boot_fd,
                            EpollEvent::new(EventSet::IN, boot_fd as u64),
                        );
                        let _ = boot_epoll.ctl(
                            ControlOperation::Add,
                            kill_fd,
                            EpollEvent::new(EventSet::IN, kill_fd as u64),
                        );
                        let mut boot_buf = [EpollEvent::default(); 2];
                        // 5 s ceiling: a healthy guest emits SYS_RDY
                        // within ~3 s of boot; longer is a stuck
                        // guest. Tests that exit without sending
                        // SYS_RDY (e.g. early-init crash) must wait
                        // here only until either the eventfd fires
                        // or `run_vm` propagates the kill
                        // flag — the timeout is the fallback for
                        // the case where neither wake arrives, and
                        // tighter is better because the host VM
                        // teardown waits on this thread joining.
                        // Capture the wake source into the local: Fired iff
                        // the sys_rdy eventfd is among the ready fds, else
                        // TimedOut on Ok(0). On any kill (kill_fd ready here,
                        // or kill set before a return below) the kill early-
                        // returns override to NotConfigured regardless — see
                        // the decl comment's `Fired ⟺ sampled` invariant.
                        match boot_epoll.wait(5_000, &mut boot_buf) {
                            Ok(0) => {
                                boot_wait_outcome = monitor::BootWaitOutcome::TimedOut;
                            }
                            Ok(n) => {
                                if boot_buf[..n].iter().any(|e| e.fd() == boot_fd) {
                                    boot_wait_outcome = monitor::BootWaitOutcome::Fired;
                                }
                            }
                            Err(e) => {
                                tracing::warn!(err = %e, "monitor: boot epoll_wait failed");
                            }
                        }
                    }
                    if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                        return monitor::reader::MonitorLoopResult {
                            samples: Vec::new(),
                            drain: crate::vmm::host_comms::BulkDrainResult {
                                entries: Vec::new(),
                            },
                            watchdog_observation: None,
                            page_offset: 0,
                            preemption_threshold_ns,
                            boot_wait_outcome: monitor::BootWaitOutcome::NotConfigured,
                            scx_event_counters_supported,
                            // Monitor torn down before sampling: no live prog seen.
                            verified_insns: Vec::new(),
                        };
                    }
                }

                // Resolve the kernel image base. On x86_64 this is
                // the compile-time constant; on aarch64 it depends
                // on `VA_BITS_MIN` derived from `TCR_EL1.T1SZ` and
                // `TCR_EL1.TG1` (granule). After the sys_rdy wait
                // the BSP has executed many run-loop iterations
                // and the lazy CAS for `tcr_el1_cache` has fired
                // (kernel programs TCR_EL1 in `__cpu_setup` long
                // before userspace init runs).
                let tcr_el1_value = tcr_el1
                    .as_ref()
                    .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
                    .unwrap_or(0);
                let start_kernel_map_for_thread =
                    monitor::symbols::start_kernel_map_for_tcr(tcr_el1_value)
                        .unwrap_or(monitor::symbols::START_KERNEL_MAP);

                // Resolve the guest kernel's `phys_base` (the physical
                // KASLR load displacement; 0 on non-KASLR boots), used to
                // derive every kernel-text PA the monitor reads. Sourced
                // from the guest's authoritative KERN_ADDRS publish, with
                // a plausibility-gated CR3 page-table walk only as a last
                // resort — see the inner comments for why the walk is not
                // the primary path. This bounded resolve is a FAST PATH
                // only: the monitor loop re-reads `kern_phys_base_shared`
                // per tick while `data_valid` is unlatched and adopts the
                // guest publish the instant it lands (see
                // `RqRefresh::kern_phys_base`), so a slow cold boot that
                // outlives the ~10 s wait no longer freezes the fallback's
                // garbage/zero into every translation for the whole run.
                // `phys_base_guest_published` records which arm supplied
                // the value (diagnostic provenance for the monitor's
                // pre-latch dump).
                let (phys_base, phys_base_guest_published) = {
                    let mut pb = 0u64;
                    let pb_fd = {
                        use std::os::unix::io::AsRawFd;
                        kern_phys_base_evt.as_raw_fd()
                    };
                    let kill_fd = {
                        use std::os::unix::io::AsRawFd;
                        kill_evt_clone.as_raw_fd()
                    };
                    // Wait, event-driven, for the guest's AUTHORITATIVE phys_base.
                    // rust_init (PID 1) reads it from /proc/iomem and publishes it
                    // over KERN_ADDRS very early and ALWAYS (0 for a non-KASLR
                    // kernel, the physical KASLR displacement otherwise), biased
                    // by +1 into `kern_phys_base_shared` with a `kern_phys_base_evt`
                    // signal. Block on that eventfd (waking the instant it lands,
                    // kill-escapable) rather than racing an unreliable CR3
                    // page-table walk: on a slow/cold boot the walk hits an
                    // early/stale CR3 or an unpopulated page and reads GARBAGE
                    // which, latched once into every text-mapped PA, leaves
                    // data_valid unlatched so NO event_counters / schedstat / scx
                    // state is captured for the whole run. The ~10s bound is only a
                    // safety net for the guest-never-published case; warm boots
                    // wake on the first iteration, and scx_root stays set for the
                    // full stall-watchdog window (~20s+) so even a slow publish
                    // leaves ample sampling overlap.
                    let mut guest_published = false;
                    for _ in 0..100 {
                        if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                            break;
                        }
                        let biased = kern_phys_base_shared.load(
                            std::sync::atomic::Ordering::Acquire,
                        );
                        if biased != 0 {
                            pb = biased.wrapping_sub(1);
                            guest_published = true;
                            break;
                        }
                        let mut pfds = [
                            libc::pollfd { fd: pb_fd, events: libc::POLLIN, revents: 0 },
                            libc::pollfd { fd: kill_fd, events: libc::POLLIN, revents: 0 },
                        ];
                        unsafe { libc::poll(pfds.as_mut_ptr(), 2, 100) };
                    }
                    // Last resort: the guest never published (channel failure /
                    // early crash). Fall back to the CR3 page-table walk, but
                    // accept its result ONLY if it is a plausible x86_64 KASLR
                    // phys_base — non-zero, inside guest DRAM, and
                    // CONFIG_PHYSICAL_ALIGN (2 MiB) aligned — never the high-half /
                    // unaligned garbage a stale-CR3 walk can return.
                    if !guest_published
                        && !kill_clone.load(std::sync::atomic::Ordering::Acquire)
                    {
                        let cr3_val = cr3.load(std::sync::atomic::Ordering::Acquire);
                        if cr3_val != 0 {
                            let l5 = monitor::symbols::resolve_pgtable_l5(
                                &mem, &symbols, start_kernel_map_for_thread, 0,
                            );
                            if let Some(v) = monitor::symbols::resolve_phys_base(
                                &mem, &symbols, cr3_val, l5, tcr_el1_value,
                            )
                                && monitor::symbols::plausible_cr3_phys_base(v, mem.size())
                            {
                                pb = v;
                            }
                        }
                    }
                    (pb, guest_published)
                };
                if phys_base != 0 {
                    let _ = kern_phys_base_shared.compare_exchange(
                        0,
                        phys_base.wrapping_add(1),
                        std::sync::atomic::Ordering::Release,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    let _ = kern_phys_base_evt.write(1);
                }

                // The KASLR slide is published into `kern_virt_kaslr_shared`
                // (`+1`-biased) by the BSP MSR_LSTAR readback
                // (`src/vmm/x86_64/msr_kaslr::read_and_derive`, x86_64-only)
                // or the guest-channel KERN_ADDRS handler
                // (`src/vmm/freeze_coord/dispatch.rs`, both arches). It is
                // NOT snapshotted here: the monitor re-reads the Arc every
                // sample iteration via `RqRefresh::kaslr_offset` (the clone
                // below). A once-captured value would be 0 — mis-sliding
                // every per-CPU KVA for the run — whenever the monitor's
                // sys_rdy wait resolves via timeout before the (single,
                // no-retry on aarch64) publisher fires.

                // Kill check between sys_rdy wait and the long-tail
                // setup work below (page-table walks, watchdog override
                // resolve, post-wait re-resolve, BTF prog_offsets
                // consumption, monitor_loop entry). On debug builds
                // with cold caches the resolution path can spend
                // multiple seconds in `resolve_phys_base` /
                // `resolve_pgtable_l5` / `text_kva_to_pa_with_base`,
                // and `run_vm`'s `kill_evt.write(1)` cannot
                // interrupt code that is not blocked on epoll. Sample
                // the kill flag at every major boundary so a VM that
                // exits during setup tears the monitor down within
                // microseconds rather than having `monitor_handle.join`
                // block until the setup runs to completion.
                if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                    return monitor::reader::MonitorLoopResult {
                        samples: Vec::new(),
                        drain: crate::vmm::host_comms::BulkDrainResult {
                            entries: Vec::new(),
                        },
                        watchdog_observation: None,
                        page_offset: 0,
                        preemption_threshold_ns,
                        boot_wait_outcome: monitor::BootWaitOutcome::NotConfigured,
                        scx_event_counters_supported,
                        // Monitor torn down before sampling: no live prog seen.
                        verified_insns: Vec::new(),
                    };
                }

                let page_offset = monitor::symbols::resolve_page_offset_with_tcr(
                    &mem,
                    &symbols,
                    start_kernel_map_for_thread,
                    tcr_el1_value,
                    phys_base,
                );

                // `__per_cpu_offset[]` lives in the kernel image
                // mapping (text PA). `setup_per_cpu_areas` in
                // `start_kernel` populates every slot before SMP
                // bringup IN THE GUEST — but the host monitor thread
                // spawns before the guest BSP enters KVM_RUN, so a
                // pre-loop one-shot read sees BSS zeros. The read PA
                // itself is recomputed per sample inside the monitor
                // loop from `RqRefresh::per_cpu_offset_kva` +
                // `RqRefresh::tcr_el1`, so no base is baked here (a
                // baked base off a not-yet-programmed TCR_EL1 would
                // pin the read at the wrong PA for the whole run —
                // see [`monitor::reader::RqRefresh`]).

                let watchdog_override = watchdog_jiffies.and_then(|jiffies| {
                    // 7.1+ path: deref scx_root -> scx_sched.watchdog_timeout.
                    if let Some((scx_root_kva, wd_offs)) = symbols
                        .scx_root
                        .zip(offsets.watchdog_offsets.as_ref())
                    {
                        let scx_root_pa = monitor::symbols::text_kva_to_pa_with_base(
                            scx_root_kva,
                            start_kernel_map_for_thread,
                            phys_base,
                        );
                        let resolve_pa = |kva| {
                            monitor::symbols::text_kva_to_pa_with_base(
                                kva,
                                start_kernel_map_for_thread,
                                phys_base,
                            )
                        };
                        let interval_pa = symbols.scx_watchdog_interval.map(&resolve_pa);
                        let timestamp_pa = symbols.scx_watchdog_timestamp.map(&resolve_pa);
                        let jiffies_64_pa = symbols.jiffies_64.map(&resolve_pa);
                        return Some(monitor::reader::WatchdogOverride::ScxSched {
                            scx_root_pa,
                            watchdog_offset: wd_offs.scx_sched_watchdog_timeout_off,
                            jiffies,
                            interval_pa,
                            timestamp_pa,
                            jiffies_64_pa,
                        });
                    }
                    if let Some(wdt_kva) = symbols.scx_watchdog_timeout {
                        let resolve_pa = |kva| {
                            monitor::symbols::text_kva_to_pa_with_base(
                                kva,
                                start_kernel_map_for_thread,
                                phys_base,
                            )
                        };
                        let watchdog_timeout_pa = resolve_pa(wdt_kva);
                        let interval_pa = symbols.scx_watchdog_interval.map(&resolve_pa);
                        let timestamp_pa = symbols.scx_watchdog_timestamp.map(&resolve_pa);
                        let jiffies_64_pa = symbols.jiffies_64.map(&resolve_pa);
                        return Some(monitor::reader::WatchdogOverride::StaticGlobal {
                            watchdog_timeout_pa,
                            jiffies,
                            interval_pa,
                            timestamp_pa,
                            jiffies_64_pa,
                        });
                    }
                    None
                });
                if watchdog_jiffies.is_some() && watchdog_override.is_none() {
                    tracing::warn!(
                        "no watchdog override path available — neither scx_sched.watchdog_timeout BTF field nor scx_watchdog_timeout symbol found"
                    );
                }

                // Kill check after watchdog override resolve. The
                // BTF / symbol-table lookups above can themselves
                // touch hundreds of kilobytes of vmlinux ELF, so a
                // VM that exits while we are still here would
                // otherwise have to wait for the entire setup tail
                // to drain before `monitor_handle.join` returns.
                if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                    return monitor::reader::MonitorLoopResult {
                        samples: Vec::new(),
                        drain: crate::vmm::host_comms::BulkDrainResult {
                            entries: Vec::new(),
                        },
                        watchdog_observation: None,
                        page_offset: 0,
                        preemption_threshold_ns,
                        boot_wait_outcome: monitor::BootWaitOutcome::NotConfigured,
                        scx_event_counters_supported,
                        // Monitor torn down before sampling: no live prog seen.
                        verified_insns: Vec::new(),
                    };
                }

                // `event_pcpu_pas` derives from
                // `*scx_root -> scx_sched.pcpu` (or
                // `event_stats_cpu` on pre-6.18 kernels) plus
                // `__per_cpu_offset[]`. Both inputs change with VM
                // lifetime: `*scx_root` is null until a scheduler
                // attaches, and the percpu base table is BSS zero
                // until `setup_per_cpu_areas` runs. Stash the
                // text-mapped PA of `scx_root` plus the BTF offsets
                // and let the monitor loop refresh per-iteration.
                let event_refresh =
                    symbols
                        .scx_root
                        .zip(offsets.event_offsets.as_ref())
                        .map(|(scx_root_kva, ev)| {
                            let scx_root_pa = monitor::symbols::text_kva_to_pa_with_base(
                                scx_root_kva,
                                start_kernel_map_for_thread,
                                phys_base,
                            );
                            monitor::reader::EventRefresh {
                                scx_root_pa,
                                event_offsets: ev.clone(),
                            }
                        });
                // Scheduler-attach watchdog-reset PA, derived
                // independently of `event_refresh` so the reset
                // works on kernels without resolvable
                // `event_offsets` (e.g. older kernels lacking the
                // BTF struct, or stripped vmlinux). Always derives
                // from `symbols.scx_root` directly — the same
                // text-mapped global the kernel itself uses to
                // publish the active `scx_sched`. `None` when the
                // symbol could not be resolved (no scx support in
                // the kernel image, or `KernelSymbols::from_elf`
                // failed to find it); the loop's
                // `cfg.watchdog_reset` short-circuits in that
                // case.
                let scx_root_pa_for_reset = symbols.scx_root.map(|kva| {
                    monitor::symbols::text_kva_to_pa_with_base(
                        kva,
                        start_kernel_map_for_thread,
                        phys_base,
                    )
                });
                // `page_offset_base` is x86_64-only (a KASLR direct-map
                // base randomized by `CONFIG_RANDOMIZE_MEMORY`).
                // `KernelSymbols::from_vmlinux` returns `None` on
                // aarch64 and on kernels built without the symbol —
                // the per-iteration refresh tolerates that and
                // leaves `page_offset` at the pre-loop default. The
                // read PA is recomputed per tick inside the loop (no
                // base is baked here — see `RqRefresh`).
                let rq_refresh = monitor::reader::RqRefresh {
                    // Carry link-time KVAs plus the live-input Arcs
                    // (TCR_EL1 cache, biased kern_phys_base) so the
                    // monitor recomputes every read PA per tick against
                    // the LIVE kernel-image base and phys_base, rather
                    // than the one-shot values this closure resolved —
                    // which can predate the guest's TCR_EL1 program
                    // (aarch64) or its KERN_ADDRS phys_base publish
                    // (both arches) — see `RqRefresh`.
                    per_cpu_offset_kva: symbols.per_cpu_offset,
                    tcr_el1: tcr_el1.clone(),
                    kern_phys_base: Some(kern_phys_base_shared.clone()),
                    phys_base_guest_published,
                    kern_addrs_frames: Some(kern_addrs_frames.clone()),
                    kern_addrs_crc_bad: Some(kern_addrs_crc_bad.clone()),
                    runqueues_kva: symbols.runqueues,
                    kaslr_offset: kern_virt_kaslr_shared.clone(),
                    num_cpus,
                    page_offset_base_kva: symbols.page_offset_base_kva,
                    event: event_refresh,
                };

                let vcpu_timing = monitor::reader::VcpuTiming {
                    pthreads: vcpu_pthreads,
                    contention_recorder: Some(contention_recorder),
                };

                // The legacy SHM signal slot 1 (`SIGNAL_PROBES_READY`)
                // gate before struct_ops discovery has been removed
                // along with the SHM signal-slot infrastructure. The
                // discovery walker tolerates an empty IDR (returns an
                // empty `Vec` when no struct_ops programs are loaded
                // yet) and re-runs every monitor sample, so a race
                // with scheduler BPF program registration recovers on
                // the next cycle.

                // Discover struct_ops programs for per-cycle stats.
                // `cr3_pa` and `l5` are shared with `discover_struct_ops_stats`
                // and `ProgStatsCtx` so per-CPU `bpf_prog_stats` reads can
                // page-walk vmalloc-backed percpu.
                //
                // Re-derive the kernel image base at this point: we
                // just blocked on the guest's slot-1 signal, so the
                // BSP loop has had time to populate the TCR_EL1
                // cache even if it was still 0 at thread start.
                // This is the value that flows into ProgStatsCtx
                // and the GuestKernel constructions below, so a
                // late re-read here gets aarch64 VA_BITS=47 hosts
                // out of the early-boot fallback window.
                let start_kernel_map_post_wait = monitor::symbols::start_kernel_map_for_tcr(
                    tcr_el1
                        .as_ref()
                        .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
                        .unwrap_or(0),
                )
                .unwrap_or(start_kernel_map_for_thread);
                // Use the live BSP CR3 directly (it's already a PA; no
                // `phys_base`-dependent translation needed): re-read
                // `cr3_cache` here and strip its control bits. When the BSP
                // never published a non-zero CR3, fall back to
                // text-translating `init_top_pgt` — historical behaviour,
                // correct on non-KASLR boots — using the `phys_base` the
                // evented guest-publish loop above already resolved (with a
                // plausibility-gated CR3 walk as its last resort).
                let cr3_latest = cr3.load(std::sync::atomic::Ordering::Acquire);
                let cr3_pa = if cr3_latest != 0 {
                    // Strip only the CR3 control bits [11:0] (CR3_ADDR_MASK):
                    // the guest boots mitigations=off (KPTI/PTI disabled) so
                    // bit 12 is a real pgd-PA bit, NOT the PTI user-pgd
                    // selector — clearing it corrupts pgd PAs that are odd
                    // 4 KiB multiples. Mirrors guest.rs walk_cr3 +
                    // resolve_phys_base's first (`& !0xFFF`) attempt.
                    cr3_latest & !0xFFFu64
                } else {
                    monitor::symbols::text_kva_to_pa_with_base(
                        symbols.init_top_pgt.unwrap_or(0),
                        start_kernel_map_post_wait,
                        phys_base,
                    )
                };
                let l5 = monitor::symbols::resolve_pgtable_l5(
                    &mem,
                    &symbols,
                    start_kernel_map_post_wait,
                    phys_base,
                );
                // Kill check after the post-wait re-resolve.
                // `resolve_phys_base` and `resolve_pgtable_l5` are the
                // most expensive operations in the closure on cold
                // caches — each performs a multi-level page-table
                // walk through guest memory. Return promptly if the
                // VM has already torn down.
                if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                    return monitor::reader::MonitorLoopResult {
                        samples: Vec::new(),
                        drain: crate::vmm::host_comms::BulkDrainResult {
                            entries: Vec::new(),
                        },
                        watchdog_observation: None,
                        page_offset: 0,
                        preemption_threshold_ns,
                        boot_wait_outcome: monitor::BootWaitOutcome::NotConfigured,
                        scx_event_counters_supported,
                        // Monitor torn down before sampling: no live prog seen.
                        verified_insns: Vec::new(),
                    };
                }
                // aarch64 TCR_EL1 (granule + T1SZ) for the
                // page-table walker. Threaded through ProgStatsCtx
                // so vmalloc-backed percpu `bpf_prog_stats`
                // translations succeed once the BSP populates the
                // cache. Always 0 on x86_64.
                let tcr_el1_val = tcr_el1
                    .as_ref()
                    .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
                    .unwrap_or(0);
                // `prog_offsets` was resolved up front from the
                // single shared `Btf` parse — see the BTF load at
                // the top of `start_monitor`. A previous version
                // re-parsed BTF here via
                // `BpfProgOffsets::from_vmlinux`, doubling the
                // setup cost on every VM run. Dropping that second
                // parse trims hundreds of ms off monitor-thread
                // startup on debug-built vmlinux, so monitor_loop
                // entry — and the first sample push — lands
                // earlier in the VM lifetime. On short-lived
                // no-scheduler boots where the VM exits within a
                // second, the saved time is the difference between
                // sampling rq_clock pre-tick (zero) and post-tick
                // (real values).
                let prog_stats_ctx = prog_offsets.and_then(|prog_offsets| {
                    let prog_idr_kva = symbols.prog_idr?;
                    // The fused walker
                    // (`walk_struct_ops_runtime_stats`) re-walks
                    // `prog_idr` each sample, which is cheap on
                    // ktstr workloads (idr_next is in the
                    // dozens) and removes the staleness window
                    // the prior cached-discovery design opened.
                    // No upfront discovery — the walker
                    // returns an empty Vec when no struct_ops
                    // programs are loaded yet, and the monitor
                    // sample emits an empty `prog_stats` for
                    // those cycles.
                    //
                    // `per_cpu_offsets` left empty here: when
                    // `rq_refresh` is set on the
                    // [`monitor::reader::MonitorConfig`], the
                    // monitor loop refreshes `__per_cpu_offset[]`
                    // per iteration and threads the live array
                    // through to `walk_struct_ops_runtime_stats`,
                    // ignoring this seed.
                    Some(monitor::reader::ProgStatsCtx {
                        per_cpu_offsets: Vec::new(),
                        walk: monitor::reader::WalkContext {
                            cr3_pa,
                            page_offset,
                            l5,
                            tcr_el1: tcr_el1_val,
                        },
                        prog_idr_kva,
                        offsets: prog_offsets,
                        start_kernel_map: start_kernel_map_post_wait,
                        phys_base,
                        cr3: cr3.clone(),
                    })
                });

                // Kill check between prog_stats_ctx construction and
                // monitor_loop entry. `monitor_loop` itself honours
                // `kill_evt` via its own epoll registration (see
                // `monitor/reader.rs`), so the check here is the
                // last guard that prevents an idle thread closure
                // from racing into the loop after the VM has been
                // told to shut down.
                if kill_clone.load(std::sync::atomic::Ordering::Acquire) {
                    return monitor::reader::MonitorLoopResult {
                        samples: Vec::new(),
                        drain: crate::vmm::host_comms::BulkDrainResult {
                            entries: Vec::new(),
                        },
                        watchdog_observation: None,
                        page_offset: 0,
                        preemption_threshold_ns,
                        boot_wait_outcome: monitor::BootWaitOutcome::NotConfigured,
                        scx_event_counters_supported,
                        // Monitor torn down before sampling: no live prog seen.
                        verified_insns: Vec::new(),
                    };
                }

                // Construct the scheduler-attach reset payload
                // when both ingredients are present: a workload
                // duration on the VM (the test set `duration` →
                // `KtstrVm::workload_duration` is `Some`) AND a
                // resolvable `scx_root` symbol (the kernel image
                // ships scx and the symbol parser found it). Both
                // missing means there is nothing to reset to / no
                // detection point — leave the field `None` so
                // the monitor's per-iteration check
                // short-circuits.
                let watchdog_reset_cfg = workload_duration.zip(scx_root_pa_for_reset).map(
                    |(workload_duration, scx_root_pa)| monitor::reader::WatchdogReset {
                        scx_root_pa,
                        workload_duration,
                        reset_ns: watchdog_reset_ns.as_ref(),
                        reset_tag: watchdog_reset_tag.as_ref(),
                    },
                );
                // Named-BPF-map watch config: present only when the test
                // declared targets AND the BPF-prog offsets + `prog_idr`
                // resolved (both required for the active-scheduler obj-prefix
                // walk that keys the metrics). The watcher lazily builds a
                // guest-memory accessor + reads each target per tick.
                let watch_cfg = if watch_targets.is_empty() {
                    None
                } else {
                    match (watch_prog_offsets, symbols.prog_idr) {
                        (Some(prog_offsets), Some(prog_idr_kva)) => {
                            Some(monitor::reader::WatchBpfMapsCfg {
                                targets: watch_targets,
                                mem: Arc::clone(&mem),
                                vmlinux: vmlinux.clone(),
                                vmlinux_data: Arc::clone(&vmlinux_data_arc),
                                base_btf: Arc::clone(&btf),
                                cr3_pa,
                                cr3: cr3.clone(),
                                tcr_el1: tcr_el1_val,
                                l5,
                                prog_idr_kva,
                                prog_offsets,
                                num_cpus,
                            })
                        }
                        _ => None,
                    }
                };
                let mon_cfg = monitor::reader::MonitorConfig {
                    // `event_pcpu_pas` left `None` here: the loop
                    // recomputes it each iteration via
                    // `rq_refresh.event` so newly attached
                    // schedulers surface event counters from the
                    // first post-attach sample without a restart.
                    event_pcpu_pas: None,
                    dump_trigger: dump_trigger.as_ref(),
                    watchdog_override: watchdog_override.as_ref(),
                    vcpu_timing: Some(&vcpu_timing),
                    // `perf_capture` is `Arc<Option<PerfCountersCapture>>`;
                    // outer deref through `Arc::as_ref` yields
                    // `&Option<PerfCountersCapture>`, inner
                    // `Option::as_ref` yields the
                    // `Option<&PerfCountersCapture>` MonitorConfig wants.
                    perf_capture: (*perf_capture).as_ref(),
                    preemption_threshold_ns,
                    prog_stats_ctx: prog_stats_ctx.as_ref(),
                    page_offset,
                    start_kernel_map: start_kernel_map_post_wait,
                    phys_base,
                    rq_refresh: Some(&rq_refresh),
                    // Enable the system-wide PSI-irq host-walk only when BOTH
                    // the psi_group offsets (config on) and the psi_system
                    // symbol resolve; else the loop emits no psi_irq sample.
                    // `psi_system` is a kernel-image `.data` global, so its PA
                    // is text-mapped once here (`text_kva_to_pa_with_base`,
                    // mirroring `jiffies_64_pa` / `scx_root_pa_for_reset`) — not
                    // per-sample. `start_kernel_map_post_wait` / `phys_base` are
                    // the same bases `MonitorConfig::start_kernel_map` /
                    // `phys_base` carry below.
                    psi: psi_offsets.zip(symbols.psi_system).map(|(offsets, kva)| {
                        monitor::reader::PsiCaptureCfg {
                            offsets,
                            psi_system_pa: monitor::symbols::text_kva_to_pa_with_base(
                                kva,
                                start_kernel_map_post_wait,
                                phys_base,
                            ),
                        }
                    }),
                    watchdog_reset: watchdog_reset_cfg,
                    watch_bpf_maps: watch_cfg.as_ref(),
                    // Producer handle for the progress/liveness ledger.
                    // The `move` closure owns the `progress_ledger` Arc;
                    // borrow it for the loop's per-tick writes.
                    progress_ledger: Some(progress_ledger.as_ref()),
                };
                // `rq_pas` empty: the loop sources every per-CPU
                // PA from `rq_refresh` per iteration so the static
                // slice would be both stale and redundant.
                let mut __mlr = monitor::reader::monitor_loop(
                    &mem,
                    &[],
                    &offsets,
                    Duration::from_millis(100),
                    &kill_clone,
                    &kill_evt_clone,
                    run_start,
                    &mon_cfg,
                );
                // The boot-complete wait happened in this closure (above),
                // not inside monitor_loop (cfg.sys_rdy is None), so stamp
                // the captured outcome onto the result here.
                __mlr.boot_wait_outcome = boot_wait_outcome;
                // Likewise the event-counter capability was decided from
                // the `KernelOffsets` this closure owns, not inside
                // monitor_loop; stamp it onto the result here so it
                // survives to `MonitorReport`.
                __mlr.scx_event_counters_supported = scx_event_counters_supported;
                __mlr
            })
            .context("spawn monitor thread")?;

        Ok(Some(handle))
    }

    /// Spawn a thread that writes to a BPF map in guest memory.
    ///
    /// Event-driven sequence:
    /// 1. Poll `BpfMapAccessorOwned::new` until kernel page tables are up
    /// 2. Poll for an ARRAY map whose program BTF resolves the requested
    ///    field (disambiguates same-suffix `.bss` maps)
    /// 3. Write each queued value, then push `SIGNAL_BPF_WRITE_DONE`
    ///    through virtio-console RX so the guest's `hvc0_poll_loop`
    ///    sets the `bpf_map_write_done` latch; the scenario's
    ///    `wait_for_map_write` gate (`Ctx::wait_for_map_write=true`)
    ///    blocks on that latch until this thread fires.
    ///
    /// `probes_ready_evt` is the EventFd created in `run_vm` and moved in
    /// here — this thread is its sole consumer, so it is not cloned. Phase 1
    /// polls it as a 200 ms backoff and both phases write it on success;
    /// phase 2's retry backoff polls `kill_evt` instead, because the fd
    /// stays level-high once written and re-polling it would spin. The
    /// success-writes wake no other consumer.
    ///
    /// `virtio_con` is the shared virtio-console device used to push
    /// the host→guest wake byte after the writes land. Replaces the
    /// legacy SHM signal slot 0 notification.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn start_bpf_map_write(
        &self,
        vm: &kvm::KtstrKvm,
        kill: &Arc<AtomicBool>,
        kill_evt: &Arc<EventFd>,
        probes_ready_evt: EventFd,
        tcr_el1: Option<Arc<std::sync::atomic::AtomicU64>>,
        cr3: Arc<std::sync::atomic::AtomicU64>,
        virtio_con: Arc<PiMutex<virtio_console::VirtioConsole>>,
        kern_phys_base: Arc<std::sync::atomic::AtomicU64>,
        delivery: Arc<std::sync::atomic::AtomicU8>,
    ) -> Result<Option<JoinHandle<()>>> {
        if self.bpf_map_writes.is_empty() {
            return Ok(None);
        }
        let Some(vmlinux) = find_vmlinux(&self.kernel) else {
            eprintln!("bpf_map_write: vmlinux not found, skipping");
            return Ok(None);
        };

        let mem = match vm.numa_layout.as_ref() {
            Some(layout) => monitor::reader::GuestMem::from_layout(layout, &vm.guest_mem),
            None => {
                use vm_memory::GuestMemoryRegion;
                let host_base = vm
                    .guest_mem
                    .get_host_address(GuestAddress(DRAM_BASE))
                    .context("resolve guest DRAM base host address (bpf-map-write)")?;
                // Size of the first contiguous region only.
                // host_base addresses that single mapping; using the
                // sum of all region lengths would extend past the
                // mapping into host heap when multiple regions exist.
                let mem_size = vm
                    .guest_mem
                    .iter()
                    .next()
                    .context("guest_mem must have at least one region (bpf-map-write)")?
                    .len();
                // SAFETY: host_base is from GuestMemoryMmap's mapping,
                // which outlives this GuestMem (owned by `vm` until
                // return).
                unsafe { monitor::reader::GuestMem::new(host_base, mem_size) }
            }
        };
        let kill_clone = kill.clone();
        let kill_evt_clone = kill_evt.clone();
        let writes = self.bpf_map_writes.clone();

        let handle = std::thread::Builder::new()
            .name("bpf-map-write".into())
            .spawn(move || {
                use crate::monitor::bpf_map::BpfMapAccessor;
                if kill_clone.load(Ordering::Acquire) {
                    return;
                }

                // Phase 1: wait for BPF map accessor (kernel booted, page tables up).
                //
                // Sleeping is replaced by `poll(POLLIN)` against
                // `probes_ready_evt` as a 200 ms-bounded backoff. No
                // sibling writes the fd before this phase succeeds (the
                // monitor holds an unused clone; phase 2 polls `kill_evt`),
                // so in practice it is level-low here and each poll blocks
                // the full 200 ms — the bound is the upper limit for kill /
                // deadline observation while the accessor is still
                // unbuildable. On successful construction we write 1,
                // leaving the fd level-high — but no later phase polls it
                // (phase 2 backs off on `kill_evt`; phase 3 never polls it),
                // so the write wakes no consumer.
                let vmlinux_data_arc = match super::vmlinux::cached_vmlinux_bytes(&vmlinux) {
                    Some(d) => d,
                    None => {
                        eprintln!("bpf_map_write: read vmlinux failed");
                        return;
                    }
                };
                let vmlinux_data = &*vmlinux_data_arc;
                let vmlinux_elf = match goblin::elf::Elf::parse(vmlinux_data) {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("bpf_map_write: parse vmlinux ELF failed: {e:#}");
                        return;
                    }
                };
                let mem = Arc::new(mem);
                // NO wall deadline: this phase's success condition IS guest
                // progress (phys_base published, MMU + page tables up), so
                // it completes exactly when the guest reaches the state the
                // injection targets — the milestone anchoring. A previous
                // 30 s wall deadline here (counted from THREAD START,
                // ticking through the whole guest boot) permanently aborted
                // the injection on saturated hosts where boot alone outran
                // it — the guest then ran wedge-free and the neg_* fixtures'
                // "expected bug did not fire" was an injection no-show, not
                // a detection regression. The retry is kill-bounded (the
                // run's watchdog deadman is the outer clock) and stays loud:
                // a progress line every ~30 s so a genuinely-stuck accessor
                // build (config typo, unresolvable vmlinux) still surfaces
                // in stderr instead of silently spinning to run end.
                let phase1_t0 = std::time::Instant::now();
                let mut phase1_last_log = phase1_t0;
                let owned = loop {
                    let biased = kern_phys_base.load(std::sync::atomic::Ordering::Acquire);
                    if biased == 0 {
                        if kill_clone.load(Ordering::Acquire) {
                            return;
                        }
                        poll_eventfd_until_ready_or_timeout(&probes_ready_evt, 200);
                        continue;
                    }
                    let pb_hint = biased.wrapping_sub(1);
                    let tcr_val = tcr_el1
                        .as_ref()
                        .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
                        .unwrap_or(0);
                    let cr3_val = cr3.load(std::sync::atomic::Ordering::Acquire);
                    match monitor::bpf_map::GuestMemMapAccessorOwned::from_elf_with_hint(Arc::clone(&mem), &vmlinux_elf, vmlinux_data, &vmlinux, tcr_val, cr3_val, pb_hint) {
                        Ok(a) => {
                            let _ = probes_ready_evt.write(1);
                            break a;
                        }
                        Err(e) => {
                            if kill_clone.load(Ordering::Acquire) {
                                return;
                            }
                            if phase1_last_log.elapsed() >= std::time::Duration::from_secs(30) {
                                phase1_last_log = std::time::Instant::now();
                                eprintln!(
                                    "bpf_map_write: accessor init still retrying after {:?} \
                                     (kill-bounded, no wall deadline): {e:#}",
                                    phase1_t0.elapsed(),
                                );
                            }
                            poll_eventfd_until_ready_or_timeout(&probes_ready_evt, 200);
                        }
                    }
                };
                // Phase 2: resolve every queued map before signaling the
                // guest. All-or-nothing: if any map fails to resolve
                // within the deadline, the thread aborts without
                // signaling the guest. The guest then proceeds under its
                // own timeout rather than observing a partial setup.
                // Running writes serially against partially-resolved
                // maps would let a late-discovery failure leave the
                // guest blocked in `wait_for_map_write` with no way to
                // recover.
                //
                // The retry backoff below polls `kill_evt`, NOT
                // `probes_ready_evt` as phase 1 does: once any producer
                // writes `probes_ready_evt` it stays level-high (never
                // `read`), so polling it would return instantly and spin
                // the per-iteration accessor rebind (see the backoff
                // comment below). The success-write to `probes_ready_evt`
                // at each resolution is retained but wakes no consumer
                // today — the monitor holds an unused clone (see
                // `start_monitor`).
                // Parse the host vmlinux BTF once as the split-BTF base for the
                // per-map program-BTF loads below. A scheduler's `.bss` globals
                // live in its PROGRAM BTF (split on vmlinux); resolving the
                // field NAME against that BTF yields the byte offset the
                // compiler placed the var at, instead of a padding-fragile
                // hardcoded constant. Mirrors the monitor read path
                // (`monitor::reader` -> `resolve_map_field_offset_width`).
                // Without a base BTF no field can resolve — abort (the guest
                // proceeds under its own `wait_for_map_write` timeout, the same
                // contract as a map-not-found abort below).
                let base_btf = match monitor::btf_offsets::load_btf_from_elf(
                    &vmlinux_elf,
                    vmlinux_data,
                    &vmlinux,
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!(
                            "bpf_map_write: vmlinux BTF parse failed, no field can resolve: {e:#}"
                        );
                        return;
                    }
                };
                // NO wall deadline (mirrors phase 1): a map with the
                // requested field appears exactly when the scheduler's BPF
                // object loads — guest progress, which saturated hosts
                // dilate arbitrarily. The prior 30 s wall here aborted the
                // injection while the scheduler was still attaching. Kill-
                // bounded, with a loud ~30 s progress line so a genuinely
                // unresolvable field (typo, wrong map suffix) still
                // surfaces in stderr instead of spinning silently.
                let phase2_t0 = std::time::Instant::now();
                let mut phase2_last_log = phase2_t0;
                let mut resolved: Vec<(BpfMapWriteParams, monitor::bpf_map::BpfMapInfo, usize, usize)> =
                    Vec::with_capacity(writes.len());
                for params in writes.iter() {
                    let mut attempt = 0u32;
                    // Multiple ARRAY maps can share a `.bss` suffix — the
                    // scheduler's `bpf_bpf.bss` and the probe's `probe_bp.bss`
                    // both end in `.bss` — so bind the map whose program BTF
                    // actually RESOLVES the requested field, not the first by
                    // IDR order. The read path's `pick_live_map` disambiguates
                    // same-suffix maps by used-maps KVA identity; here we
                    // disambiguate by which map's BTF resolves the field — a
                    // different mechanism, sound because `crash`/`stall` are
                    // unique to the single live scheduler. Retry until that map
                    // appears (the scheduler may still be attaching), rebinding
                    // a fresh accessor each iteration: within a SINGLE
                    // accessor, `maps()` snapshots the map IDR once and caches
                    // it (the per-accessor `maps_cache` is not invalidated on
                    // later `maps()` calls), so a map created after that first
                    // snapshot stays invisible to a REUSED accessor. Building
                    // a fresh `as_accessor()` per iteration starts an empty
                    // cache that re-walks the live IDR — do NOT hoist
                    // `as_accessor()` out of this loop.
                    let (map_info, offset, width) = loop {
                        attempt += 1;
                        let accessor = owned.as_accessor();
                        let hit = accessor.maps().into_iter().find_map(|m| {
                            if m.map_type != monitor::bpf_map::BPF_MAP_TYPE_ARRAY
                                || !m.name().ends_with(&params.map_name_suffix)
                            {
                                return None;
                            }
                            let prog_btf = accessor.load_program_btf(&m, &base_btf);
                            let btf = prog_btf.as_ref().unwrap_or(&base_btf);
                            monitor::btf_offsets::resolve_map_field_offset_width(
                                btf,
                                m.btf_value_type_id,
                                &params.field,
                            )
                            .map(|(off, w)| (m, off, w))
                        });
                        if let Some(hit) = hit {
                            let _ = probes_ready_evt.write(1);
                            break hit;
                        }
                        if kill_clone.load(Ordering::Acquire) {
                            eprintln!("bpf_map_write: VM exited during map/field search");
                            return;
                        }
                        if phase2_last_log.elapsed() >= std::time::Duration::from_secs(30) {
                            phase2_last_log = std::time::Instant::now();
                            eprintln!(
                                "bpf_map_write: field '{}' still unresolved in any '{}' map \
                                 after {} attempts / {:?} (kill-bounded, no wall deadline)",
                                params.field,
                                params.map_name_suffix,
                                attempt,
                                phase2_t0.elapsed(),
                            );
                        }
                        // Back off ~200 ms before re-walking the IDR, waking
                        // early on kill. Poll `kill_evt` rather than the shared,
                        // never-drained `probes_ready_evt` (which stays
                        // level-high once any producer writes it, so polling it
                        // returns instantly and would spin the per-iteration
                        // accessor rebind — an uncached IDR walk + program-BTF
                        // reparse against guest memory — at max rate). `kill_evt`
                        // is low until VM teardown, so this blocks the full
                        // timeout during a normal scheduler attach and wakes
                        // immediately on kill; the `kill_clone` load above is
                        // the authoritative exit (the eventfd is a best-effort
                        // wake, matching the other freeze-coord worker loops).
                        poll_eventfd_until_ready_or_timeout(&kill_evt_clone, 200);
                    };
                    eprintln!(
                        "bpf_map_write: field '{}' resolved to map '{}' off={} width={} after {} attempts",
                        params.field, map_info.name(), offset, width, attempt,
                    );
                    resolved.push((params.clone(), map_info, offset, width));
                }

                // Phase 3: run every queued write.
                //
                // The legacy SHM signal slot 1 (`SIGNAL_PROBES_READY`)
                // gate that waited for the guest's probe pipeline to
                // attach has been removed along with the SHM
                // signal-slot infrastructure. The writes now race
                // against probe attachment; replacing the rendezvous
                // with a virtio-console signal is a follow-up.

                // Rebind a fresh accessor for the writes. As in the retry
                // loop, `maps()`/`load_program_btf` snapshot the map IDR once
                // per accessor, so a reused one could carry a stale snapshot;
                // `read`/`write_value_u32` below address each map by the
                // `BpfMapInfo` captured in phase 2, so any live accessor works.
                let accessor = owned.as_accessor();

                // Log all maps for diagnostic visibility.
                let all_maps = accessor.maps();
                eprintln!(
                    "bpf_map_write: maps() found {} map(s): [{}]",
                    all_maps.len(),
                    all_maps
                        .iter()
                        .map(|m| format!("{}(type={})", m.name(), m.map_type))
                        .collect::<Vec<_>>()
                        .join(", "),
                );

                for (params, map_info, offset, width) in &resolved {
                    // `write_value_u32` is a fixed 4-byte store; a field of a
                    // different width would truncate or clobber adjacent bytes.
                    // crash/stall are `volatile int` (4 bytes) — skip loudly
                    // rather than silently mis-write any non-4-byte field (the
                    // `BpfMapWrite` value is a `u32`). NOTE: a skipped write
                    // still lets `request_bpf_map_write_done` fire below
                    // (signalling the guest all writes completed). This
                    // asymmetry vs phase 2's all-or-nothing resolution abort
                    // is deliberate: phase 2 aborts BEFORE signalling because
                    // it cannot proceed, whereas here the map/field DID
                    // resolve — withholding the signal would hang the guest in
                    // `wait_for_map_write` with no recovery, so a loud skip +
                    // a completed handshake is preferred. Unreachable for the
                    // 4-byte crash/stall fields today; the eprintln surfaces
                    // it if a future non-4-byte field is queued.
                    if *width != 4 {
                        eprintln!(
                            "bpf_map_write: field '{}' width {} != 4 in map '{}' — skipping (value is u32)",
                            params.field, width, map_info.name(),
                        );
                        continue;
                    }
                    let before = accessor.read_value_u32(map_info, *offset);
                    let ok = accessor.write_value_u32(map_info, *offset, params.value);
                    let after = accessor.read_value_u32(map_info, *offset);
                    eprintln!(
                        "bpf_map_write: map '{}' field '{}' off={} write={} (value={} before={:?} after={:?})",
                        map_info.name(), params.field, offset, ok, params.value, before, after,
                    );
                }

                // Notify the guest that every queued write landed by
                // pushing `SIGNAL_BPF_WRITE_DONE` into virtio-console
                // RX. The guest's `hvc0_poll_loop` blocks on
                // `/dev/hvc0`, recognises the byte, and sets the
                // `bpf_map_write_done` latch. A scenario blocked on
                // [`crate::scenario::Ctx::wait_for_map_write`] resumes
                // when the latch fires. Replaces the legacy SHM signal
                // slot 0 notification.
                super::host_comms::request_bpf_map_write_done(&virtio_con);
                // Delivery evidence: every queued write landed and the guest
                // was signalled. The neg_* starvation gates read the decoded
                // `VmResult::bpf_map_writes_delivered` — `Some(true)` here
                // means an absent expected-crash is a REAL detection failure,
                // never an injection no-show.
                delivery.store(2, Ordering::Release);
                let _ = (&kill_clone, &probes_ready_evt, &mem);
            })
            .context("spawn bpf-map-write thread")?;

        Ok(Some(handle))
    }

    /// Unified BSP KVM_RUN loop. Returns `(exit_code, timed_out,
    /// exit_reason)`.
    ///
    /// `exit_code` semantics:
    ///   - `0` only when the BSP itself observed
    ///     [`ExitAction::Shutdown`] from `classify_exit` (i8042 reset
    ///     on x86_64, PSCI SystemEvent on aarch64, or
    ///     `VcpuExit::Shutdown`).
    ///   - `-1` is a sentinel meaning "BSP exited the loop without
    ///     observing Shutdown itself." This does NOT necessarily
    ///     indicate a failure — a peer vCPU that observed Shutdown
    ///     first sets the shared `kill` flag, and the BSP then exits
    ///     via the `kill.load(Acquire)` check at the top of the loop.
    ///     [`super::KtstrVm::collect_results`] overrides the run-loop
    ///     `exit_code` with the bulk-port `MSG_TYPE_EXIT` payload (or the
    ///     COM2 `KTSTR_EXIT:` sentinel) before constructing
    ///     [`super::result::VmResult`], so the value caller-visible
    ///     code reads is the guest's reported exit code, not this
    ///     local sentinel. [`BspExitReason`] is returned so the
    ///     caller can log it (`BSP: loop exit reason=...`) and an
    ///     operator reading stderr can distinguish "AP saw Shutdown
    ///     first" from "BSP itself saw Fatal" or "BSP run() returned
    ///     a permanent error" without correlating to other
    ///     diagnostics. Logging is the caller's job because the
    ///     interactive-shell terminal is still in raw mode when this
    ///     loop exits — a bare `eprintln!` here staircases into the
    ///     session output (LF without CR, glued to the guest's last
    ///     echo); the shell path defers the line until after
    ///     terminal restore and gates it on
    ///     [`BspExitReason::is_abnormal`] so a routine poweroff
    ///     exits silently.
    ///
    /// Handles arch-specific I/O dispatch (port I/O on x86_64, MMIO on
    /// aarch64). HLT/WFI checks the kill flag and continues (both arches).
    /// Shutdown is via PSCI SystemEvent (aarch64) or VcpuExit::Shutdown (x86_64).
    ///
    /// `freeze` and `bsp_parked` plumb the BSP into the failure-dump
    /// rendezvous: when the freeze coordinator latches `freeze=true`
    /// and kicks the BSP out of KVM_RUN, the loop performs the
    /// drain dance (set_immediate_exit(1)→run→set_immediate_exit(0)),
    /// stores `bsp_parked=true` (Release), then polls `freeze` on
    /// `park_timeout(10ms)` until the coordinator clears it. Same
    /// pattern as [`exit_dispatch::vcpu_run_loop_unified`] for APs.
    ///
    /// `watchpoint` carries the failure-dump trigger contract: each
    /// iteration polls `watchpoint.request_kva` and self-arms a
    /// hardware data-write watchpoint on `*scx_root->exit_kind` once
    /// the freeze coordinator has resolved its KVA. When the kernel
    /// later writes the field, KVM exits via `VcpuExit::Debug`; this
    /// loop sets `watchpoint.hit` so the freeze coordinator's
    /// late-trigger poll fires immediately. The arm is one-shot per
    /// KVA value (the per-vCPU `armed_kva` slot suppresses re-arms
    /// after the ioctl lands).
    ///
    /// `tcr_el1_cache` (aarch64 only) is populated lazily on first
    /// successful sysreg read after the guest kernel programs the
    /// MMU; subsequent iterations short-circuit on a non-zero
    /// cached value. Threads that build a `GuestKernel` for
    /// page-table walks load this atomic to feed the
    /// granule-agnostic walker.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_bsp_loop(
        &self,
        bsp: &mut kvm_ioctls::VcpuFd,
        com1: &Arc<PiMutex<console::Serial>>,
        com2: &Arc<PiMutex<console::Serial>>,
        virtio_con: Option<&Arc<PiMutex<virtio_console::VirtioConsole>>>,
        virtio_blk: Option<&Arc<PiMutex<virtio_blk::VirtioBlk>>>,
        virtio_net: Option<&Arc<PiMutex<virtio_net::VirtioNet>>>,
        ioapic: Option<&Arc<crate::vmm::IoapicHandle>>,
        pci_bus: Option<&Arc<PiMutex<crate::vmm::pci::PciBus>>>,
        kill: &Arc<AtomicBool>,
        freeze: &Arc<AtomicBool>,
        watchpoint: &Arc<WatchpointArm>,
        bsp_parked: &Arc<AtomicBool>,
        bsp_regs: &Arc<std::sync::Mutex<Option<exit_dispatch::VcpuRegSnapshot>>>,
        has_immediate_exit: bool,
        _run_start: Instant,
        _timeout: Duration,
        parked_evt: Option<&Arc<EventFd>>,
        thaw_evt: Option<&Arc<EventFd>>,
        kill_evt: Option<&Arc<EventFd>>,
        tcr_el1_cache: Option<&Arc<std::sync::atomic::AtomicU64>>,
        cr3_cache: &Arc<std::sync::atomic::AtomicU64>,
        timed_out_flag: &Arc<AtomicBool>,
        // Shared virt-KASLR offset slot (biased `+1`). The BSP
        // loop attempts a one-shot `KVM_GET_MSRS(MSR_LSTAR)` derive
        // via `msr_kaslr::read_and_derive` on x86_64 between vCPU
        // run iterations; on success it CAS-publishes the
        // `(offset + 1)` here so the monitor + dump pipelines
        // (the dump-path `CpuTimeCapture` construction and the
        // monitor's `RqRefresh` / `text_kva_to_pa_with_base` sites
        // in `start_monitor`)
        // can resolve per-CPU `rq` / `kernel_cpustat` / `kstat`
        // KVAs under `CONFIG_RANDOMIZE_BASE=y`. The guest-channel
        // KERN_ADDRS path also CAS-publishes here (from the coord
        // thread's dispatch); both writers produce the same value
        // because KASLR shifts every text symbol by the same
        // `kaslr_offset`. The CAS-fail branch is a benign no-op.
        // On aarch64 the BSP MSR read is unavailable (no MSR_LSTAR
        // equivalent) so this Arc is read but not written from
        // this path — the guest-channel publisher is the sole
        // source on aarch64.
        kern_virt_kaslr_shared: &Arc<std::sync::atomic::AtomicU64>,
        // EventFd fired when the BSP MSR_LSTAR-derive publishes a
        // non-zero biased value to `kern_virt_kaslr_shared`. Mirror
        // of the KERN_ADDRS-path publish at `dispatch.rs`.
        kern_virt_kaslr_evt: &EventFd,
        // Link-time KVA of `entry_SYSCALL_64` from vmlinux. Passed
        // through to `msr_kaslr::read_and_derive` so the BSP
        // MSR_LSTAR readback can compute the offset. `0` on
        // aarch64 builds (no `entry_SYSCALL_64` symbol) — the
        // BSP-side derive short-circuits in that case and the
        // shared Arc stays at 0 until the guest channel publishes.
        entry_syscall_64_link_kva: u64,
    ) -> (i32, bool, BspExitReason) {
        let mut exit_code: i32 = -1;
        // Track which path drove the BSP out of the loop so the
        // post-loop log line is actionable. Without this, an operator
        // sees `code=-1 timed_out=false` and cannot distinguish
        // "external kill propagated from a peer vCPU's Shutdown" from
        // "BSP itself saw Fatal" — every non-Shutdown exit produces
        // the same `code=-1` sentinel.
        let mut exit_reason = BspExitReason::ExternalKill;
        // Per-BSP `armed_slots` mirrors the AP-side slots — see
        // [`super::vcpu::self_arm_watchpoint`]. Index 0 = slot 0
        // (exit_kind watchpoint); 1..=3 = user watchpoint slots
        // (`Op::WatchSnapshot` arms). All `0` until the coordinator
        // publishes resolved KVAs. `arm_failures` counts consecutive
        // non-EINTR ioctl failures; transient EINTR (signal race
        // with the SIGRTMIN kick path) does NOT increment so a
        // kicked-mid-arm vCPU keeps retrying instead of giving up
        // after the first racey iteration.
        let mut armed_slots: [u64; 4] = [0; 4];
        let mut arm_failures: u8 = 0;
        // aarch64 watchpoint single-step bookkeeping — mirrors the
        // AP-side state in
        // [`super::exit_dispatch::vcpu_run_loop_unified`]. The
        // aarch64 hardware watchpoint trap is taken BEFORE the
        // offending store retires (ARM ARM D2.10.5), so re-entering
        // KVM_RUN replays the same instruction unless we disable
        // the fired slot's WCR.E and assert
        // KVM_GUESTDBG_SINGLESTEP for one KVM_RUN; the next
        // KVM_EXIT_DEBUG carries EC=ESR_ELx_EC_SOFTSTP_LOW (0x32),
        // at which point the dispatch helper clears the flag and
        // `self_arm_watchpoint` restores WCR.E=1. Inert on x86_64
        // (the trap is taken AFTER the store, so re-entry advances
        // normally); the locals still pass through to keep the
        // per-arch helper signatures shared.
        let mut single_step_pending: bool = false;
        let mut single_step_slot: usize = 0;
        let mut armed_single_step: bool = false;

        loop {
            if kill.load(Ordering::Acquire) {
                break;
            }
            // Lazy TCR_EL1 cache populate (aarch64). On x86_64
            // `read_tcr_el1` returns None and the early-exit keeps
            // the atomic untouched. The kernel writes TCR_EL1 in
            // its boot-time MMU bring-up; before that the read
            // returns 0. Skip on subsequent iterations once the
            // atomic carries a non-zero value (CAS prevents races
            // with peer reads from other threads constructing a
            // `GuestKernel`).
            if let Some(cache) = tcr_el1_cache
                && cache.load(Ordering::Acquire) == 0
                && let Some(val) = exit_dispatch::read_tcr_el1(bsp)
                && val != 0
            {
                let _ = cache.compare_exchange(0, val, Ordering::Release, Ordering::Relaxed);
            }
            // CR3 / TTBR1_EL1 cache refresh. KVM_GET_SREGS at BSP
            // entry returns the boot-time CR3 (`PML4_START`, set by
            // `setup_sregs`); the kernel later overwrites this in
            // `__startup_64` after KASLR randomization. We need the
            // POST-randomization value for `phys_base` resolution
            // via page-table walk, so this MUST be a refresh
            // (overwrite each iteration), NOT a one-shot latch:
            // a "skip if non-zero" gate would freeze the cache at
            // the boot CR3 because get_sregs returns it on iter 1
            // before the guest has run `mov cr3, ...`. Accepting
            // every non-zero read also handles process context
            // switches (CR3 swaps to the new task's pgd) — the
            // kernel-half upper PML4 entries are shared across
            // every task's pgd, so any task's CR3 produces a valid
            // walk for kernel symbols. The lazy-CAS pattern still
            // gates on a non-zero `read_cr3` return so a transient
            // EINTR (None) does not zero the cache. Use a Release
            // store (not CAS) so concurrent readers see the latest
            // non-zero value.
            if let Some(val) = exit_dispatch::read_cr3(bsp)
                && val != 0
            {
                cr3_cache.store(val, Ordering::Release);
            }
            // One-shot virt-KASLR derivation from BSP MSR_LSTAR.
            // On x86_64, the kernel writes the post-relocation
            // `entry_SYSCALL_64` KVA into MSR_LSTAR during
            // `cpu_init → syscall_init → idt_syscall_init` (kernel
            // rev 9636d2ea, arch/x86/kernel/cpu/common.c:2257).
            // Reading the MSR back via KVM_GET_MSRS and subtracting
            // the link-time KVA yields the virt-KASLR slide
            // (`msr_kaslr::read_and_derive`). Cross-thread
            // publish target is `kern_virt_kaslr_shared`
            // (Acquire/Release CAS, idempotent vs the guest-channel
            // KERN_ADDRS writer); both paths derive the same offset
            // so the CAS-fail branch is benign.
            //
            // Gates:
            //   * x86_64 only — no MSR_LSTAR on aarch64
            //   * entry_syscall_64_link_kva != 0 — vmlinux must have
            //     the symbol (stripped builds short-circuit)
            //   * `kern_virt_kaslr_shared.load() == 0` — only attempt
            //     until someone publishes (avoids re-running the
            //     ioctl every iteration once the slot is filled)
            //
            // Failure modes (`LstarUnsupported`, `LstarZero`,
            // `NonCanonical`, ...) are retryable per
            // `LstarDeriveError::is_retryable`. We don't track
            // retries explicitly; the loop simply tries again on
            // the next iteration until the read succeeds or kill
            // fires. Steady-state cost is one `.load()` per
            // iteration (one atomic read) once published.
            //
            // Confidential-compute caveat (SEV-ES / SEV-SNP / TDX):
            // `KVM_GET_MSRS` for MSR_LSTAR FAILS on cc guests —
            // SEV-ES via `sev_es_prevent_msr_access` returns
            // -EINVAL when `has_protected_state` (arch/x86/kvm/
            // svm/svm.c:2725-2738); TDX via `tdx_get_msr` returns
            // 1 (ioctl-fail) because `tdx_has_emulated_msr`
            // (arch/x86/kvm/vmx/tdx.c:2104-2149) excludes
            // MSR_LSTAR from the TD-emulated allow-list.
            // Fail-loud, not silent-garbage. The
            // `is_retryable()` / `LstarUnsupported` gate at
            // msr_kaslr.rs:243 surfaces this as
            // `LstarDeriveError`; the wait-loop falls through to
            // the guest-channel KERN_ADDRS `_text` publisher
            // (`dispatch.rs`) or the `nokaslr` cmdline arg
            // (`setup.rs`). ktstr does not target cc today; if a
            // backend ever adds cc support, this chain Just Works
            // — document the assumption at the backend's BSP-init
            // site.
            #[cfg(target_arch = "x86_64")]
            if entry_syscall_64_link_kva != 0
                && kern_virt_kaslr_shared.load(Ordering::Acquire) == 0
                && let Ok(offset) =
                    crate::vmm::x86_64::msr_kaslr::read_and_derive(bsp, entry_syscall_64_link_kva)
                && kern_virt_kaslr_shared
                    .compare_exchange(
                        0,
                        offset.wrapping_add(1),
                        Ordering::Release,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            {
                // Mirror the KERN_ADDRS-path evt fire at
                // dispatch.rs:547 so any future epoll-on-publish
                // consumer (none today, but the eventfd is in
                // the run-loop fdset and the asymmetry would
                // foot-gun a future change). No-op on signal-fd
                // write failure — the atomic is the
                // load-bearing publication.
                let _ = kern_virt_kaslr_evt.write(1);
            }
            // On aarch64 MSR_LSTAR has no equivalent so the BSP
            // path is unavailable; the guest-channel KERN_ADDRS
            // publisher is the sole source on that architecture.
            // Reference the args once so the compiler's
            // dead-code analysis doesn't fire on aarch64 builds.
            #[cfg(not(target_arch = "x86_64"))]
            {
                let _ = entry_syscall_64_link_kva;
                let _ = kern_virt_kaslr_shared;
                let _ = kern_virt_kaslr_evt;
            }
            // Honour a pending freeze before re-entering KVM_RUN.
            // Same drain-dance + park pattern as the AP run loop —
            // delegated to the shared `exit_dispatch::handle_freeze`
            // so the two paths cannot drift.
            if freeze.load(Ordering::Acquire) {
                exit_dispatch::handle_freeze(
                    bsp,
                    has_immediate_exit,
                    kill,
                    freeze,
                    bsp_parked,
                    bsp_regs,
                    parked_evt.map(|a| a.as_ref()),
                    thaw_evt.map(|a| a.as_ref()),
                    kill_evt.map(|a| a.as_ref()),
                );
                if kill.load(Ordering::Acquire) {
                    break;
                }
            }
            // Self-arm the failure-dump watchpoint when the
            // coordinator has resolved a new KVA. Cheap (atomic load
            // and compare) when no new arm is pending. Also drives
            // the aarch64 watchpoint single-step transition: when
            // `single_step_pending` is set by the prior watchpoint
            // exit, this call reissues KVM_SET_GUEST_DEBUG with the
            // fired slot's WCR.E cleared and KVM_GUESTDBG_SINGLESTEP
            // asserted; when the SOFTSTP_LOW exit clears the flag,
            // the next call restores WCR.E=1 and drops the
            // singlestep bit.
            self_arm_watchpoint(
                bsp,
                watchpoint,
                &mut armed_slots,
                &mut arm_failures,
                single_step_pending,
                single_step_slot,
                &mut armed_single_step,
            );

            match bsp.run() {
                Ok(mut exit) => {
                    // HLT/WFI = kernel idle. Check kill flag, then continue.
                    // arm64 shutdown is PSCI reset (SystemEvent), not HLT.
                    if matches!(exit, VcpuExit::Hlt) {
                        if kill.load(Ordering::Acquire) {
                            break;
                        }
                        continue;
                    }
                    // KVM_EXIT_DEBUG fires when the armed hardware
                    // data-write watchpoint trips on a guest write
                    // to `*scx_root->exit_kind`. The kernel writes
                    // the field on BOTH error transitions
                    // (`scx_error -> SCX_EXIT_ERROR/_BPF/_STALL >=
                    // 1024`) AND clean shutdown
                    // (`scx_unregister -> SCX_EXIT_DONE = 1`). Only
                    // the error transitions should trigger the
                    // failure-dump freeze; firing on every clean
                    // test exit is a regression. Read the post-store
                    // value from the host pointer the coordinator
                    // published and gate `hit` on the error
                    // threshold. The watchpoint is left armed
                    // regardless — see the AP-side
                    // `vcpu_run_loop_unified` for the same
                    // rationale.
                    if let VcpuExit::Debug(debug_arch) = &exit {
                        exit_dispatch::dispatch_watchpoint_hit(
                            watchpoint,
                            debug_arch,
                            &armed_slots,
                            &mut single_step_pending,
                            &mut single_step_slot,
                        );
                        if kill.load(Ordering::Acquire) {
                            break;
                        }
                        continue;
                    }
                    match classify_exit(
                        com1,
                        com2,
                        virtio_con.map(|a| a.as_ref()),
                        virtio_blk.map(|a| a.as_ref()),
                        virtio_net.map(|a| a.as_ref()),
                        ioapic.map(|a| a.as_ref()),
                        pci_bus.map(|a| a.as_ref()),
                        &mut exit,
                    ) {
                        Some(ExitAction::Continue) | None => {}
                        Some(ExitAction::Shutdown) => {
                            exit_code = 0;
                            exit_reason = BspExitReason::Shutdown;
                            break;
                        }
                        Some(ExitAction::Fatal(reason)) => {
                            if let Some(r) = reason {
                                tracing::error!(r, "BSP VM entry failed");
                            } else {
                                tracing::error!("BSP internal error");
                            }
                            // Propagate kill to peers and the freeze
                            // coordinator. Unlike the Shutdown arm
                            // (which exits with code=0 and lets
                            // run_vm drive the kill
                            // propagation), Fatal indicates an
                            // unrecoverable hardware/KVM failure and
                            // peers must shut down promptly rather
                            // than spinning until FREEZE_RENDEZVOUS_
                            // TIMEOUT. Mirrors the AP Fatal arm's
                            // kill-propagation in
                            // [`super::exit_dispatch::vcpu_run_loop_unified`].
                            kill.store(true, Ordering::Release);
                            if let Some(kev) = kill_evt {
                                let _ = kev.write(1);
                            }
                            exit_reason = BspExitReason::Fatal;
                            break;
                        }
                    }
                    // Guest kernel panic latched on COM1 at ingest (e.g.
                    // OOM → "...Attempted to kill init! exitcode=0x…")? The
                    // latch fires only once the banner line is newline-
                    // terminated, so classify_exit just drained the '\n'
                    // that completed it.
                    // Abort fast with the cause rather than spinning to the
                    // watchdog / 24h interactive timeout. Propagate kill so
                    // peers + the freeze coordinator tear down promptly,
                    // like the Fatal arm.
                    if let Some(line) = com1.lock().take_panic() {
                        eprintln!(
                            "BSP: guest kernel panic — aborting run: {line} \
                             (if OOM: raise memory_mib or shrink the initramfs)"
                        );
                        kill.store(true, Ordering::Release);
                        if let Some(kev) = kill_evt {
                            let _ = kev.write(1);
                        }
                        exit_reason = BspExitReason::GuestPanic;
                        break;
                    }
                }
                Err(e) => {
                    if e.errno() == libc::EAGAIN || e.errno() == libc::EINTR {
                        if has_immediate_exit {
                            bsp.set_kvm_immediate_exit(0);
                        }
                        continue;
                    }
                    tracing::error!(%e, "BSP run failed");
                    exit_reason = BspExitReason::RunError;
                    break;
                }
            }
        }

        // The watchdog sets `timed_out_flag` only on its hard-
        // deadline branch (NOT on "kill set by AP"). Reading it
        // here propagates the watchdog's hard-timeout verdict
        // through the BSP return tuple → `VmRunState::timed_out`
        // → `VmResult::timed_out` so callers can distinguish a
        // watchdog-driven kill from a clean shutdown or a
        // panic-driven kill.
        let timed_out = timed_out_flag.load(Ordering::Acquire);
        (exit_code, timed_out, exit_reason)
    }

    /// Whether the run has a real BPF `struct_ops` scheduler
    /// attached. Returns true ONLY when `scheduler_binary` is
    /// `Some` — those are the variants that load a userspace
    /// BPF program and call `bpf_program__attach_struct_ops`
    /// (which is what creates `bpf_prog` entries that
    /// [`collect_verifier_stats`](Self::collect_verifier_stats)
    /// walks at cleanup time).
    ///
    /// Returns true for `SchedulerSpec::Discover(name)` and
    /// `SchedulerSpec::Path(p)` — both resolve to a
    /// `scheduler_binary(path)` builder call at
    /// `src/vmm/builder.rs:scheduler_binary` (which sets
    /// `self.scheduler_binary = Some(path)`). Returns false
    /// for `SchedulerSpec::Eevdf` (no scheduler program —
    /// kernel default; builder never calls
    /// `scheduler_binary`) and ALSO false for
    /// `SchedulerSpec::KernelBuiltin { enable, disable }`
    /// (same — its `sched_enable_cmds` are arbitrary
    /// `echo X > /sys/...` shell writes wired into a
    /// separate builder field, sysctl-style toggles around
    /// kernel-builtin schedulers, that load zero BPF
    /// programs). A `verifier_stats` walk against an
    /// Eevdf-or-KernelBuiltin-only run returns an empty Vec
    /// by construction, so the cost-vs-signal tradeoff
    /// favors skipping the walk entirely.
    ///
    /// Sibling to the inline check at `stats_client`
    /// construction (~L1386) which uses `self.scheduler_binary.
    /// is_some()` directly — both gates ask the same
    /// "is there a userspace BPF binary attached" question
    /// and share the same answer.
    fn has_bpf_scheduler_attached(&self) -> bool {
        has_bpf_scheduler_attached_inner(self.scheduler_binary.as_ref(), &self.sched_enable_cmds)
    }

    /// Shutdown threads and collect output.
    pub(super) fn collect_results(&self, start: Instant, run: VmRunState) -> Result<VmResult> {
        // Whole-cleanup timer for the perf-repro tracing pipeline.
        // `cleanup_duration` below already records the post-BSP-exit
        // window via `run.cleanup_start.elapsed()`; this captures the
        // collect_results function span itself so a regression isolates
        // to either the run.cleanup_start window (set by run_vm before
        // it called us) or the function body.
        let collect_results_start = Instant::now();
        let mut exit_code = run.exit_code;
        let timed_out = run.timed_out;
        // Belt-and-braces: kill + kill_evt are already set by run_vm
        // immediately after BSP exits. Re-assert here in case a
        // future code path reaches collect_results without the
        // early-kill having fired. The two consumers that observe
        // kill_evt via epoll are the monitor sampler (reader.rs
        // monitor_loop) and the bpf-map-write thread (start_bpf_map_write).
        // The freeze coordinator is NOT alive here — run_vm joins it
        // before returning VmRunState. kill_evt is level-triggered
        // (EFD_NONBLOCK eventfd); the AtomicBool kill flag is the
        // source of truth that breaks each thread's outer loop.
        run.kill.store(true, Ordering::Release);
        let _ = run.kill_evt.write(1);
        // Clear freeze before kicking APs so any vCPU still in the
        // park loop observes `freeze=false` next iteration and exits
        // toward kill. Without this, an AP parked at the moment the
        // BSP exited would stay parked through the kill check, since
        // park_loop holds park_timeout(10ms) ignoring kill until
        // freeze clears.
        run.freeze.store(false, Ordering::Release);

        // The freeze coordinator was joined inside `run_vm` BEFORE
        // bsp dropped (preventing UAF on the BSP ImmediateExitHandle),
        // so `run.freeze_coordinator` is always `None` here. The
        // `Option`-typed field is preserved for backward compatibility
        // with paths that may construct VmRunState differently in
        // the future; the conditional join below is a no-op for the
        // `None` arm.
        if let Some(h) = run.freeze_coordinator {
            let _ = h.join();
        }
        // Kick + join every vCPU thread via the shared helper — the same
        // deadlock-safe drain (kick + unpark + epoll-wait with deadline, then
        // join) that `RunVmThreadGuard`'s `Drop` runs on the early-return path.
        // `kill` / `kill_evt` / `freeze` were set above, as the helper requires.
        kick_and_join_ap_threads(run.ap_threads);
        if crate::vmm::debug_logging_enabled() {
            eprintln!("CLEANUP: all AP threads joined");
        }

        // Invalidate the watchpoint slots BEFORE `run.vm` drops at
        // the end of this function. `kind_host_ptr` addresses a host
        // u32 inside `vm.guest_mem`'s mmap-backed mapping; once
        // `vm.guest_mem` drops, that mapping unmaps and dereffing
        // `kind_host_ptr` would touch unmapped memory.
        // `request_kva` is the paired guest-side KVA whose
        // translation goes through the same mapping. By this point
        // every vCPU thread has joined (the loop above blocked on
        // each `wait_for_exit` + `handle.join`) and the freeze
        // coordinator joined back in `run_vm` before `bsp` dropped,
        // so no live thread reads either field. The defense-in-depth
        // store here zeroes the slots so a stray future Arc clone
        // (or a follow-up that adds a new reader after teardown)
        // sees a sentinel `null_mut` / `0` that
        // [`super::exit_dispatch::latch_slot0_with_gate`] already
        // gates on, instead of dangling host memory. `Release`
        // ordering pairs with the `Acquire` reads inside the latch
        // path so any future reader sees a coherent view of the
        // invalidation.
        run.watchpoint
            .kind_host_ptr
            .store(std::ptr::null_mut(), Ordering::Release);
        run.watchpoint.request_kva.store(0, Ordering::Release);
        // Mirror the slot-0 invalidation across every user
        // watchpoint slot (1..=3, `Op::WatchSnapshot` arms). A
        // future reader that walks `watchpoint.user[..]` sees the
        // same `request_kva == 0` sentinel as slot 0 — the
        // resolved KVA is no longer reachable from any slot. `hit`
        // is also cleared so a stray Acquire load after teardown
        // observes "no fire pending" instead of a stale latch from
        // an earlier run that no longer has a captured report.
        // `Release` pairs with the `Acquire` reads in
        // `arm_user_watchpoint` and the latch path.
        for slot in &run.watchpoint.user {
            slot.request_kva.store(0, Ordering::Release);
            slot.hit.store(false, Ordering::Release);
        }

        let (monitor_report, mid_flight_drain, mid_run_verified_insns) =
            match run.monitor_handle.and_then(|h| h.join().ok()) {
                Some(monitor::reader::MonitorLoopResult {
                    samples,
                    drain,
                    watchdog_observation,
                    page_offset,
                    preemption_threshold_ns,
                    boot_wait_outcome,
                    scx_event_counters_supported,
                    verified_insns,
                }) => {
                    // `preemption_threshold_ns` was resolved once
                    // inside `start_monitor` (and threaded through
                    // `monitor_loop`'s 0-fallback) so the cleanup
                    // path does NOT re-read the vmlinux to recompute
                    // CONFIG_HZ. The previous structure called
                    // `monitor::vcpu_preemption_threshold_ns(Some(
                    // &self.kernel))` here, which re-read the
                    // vmlinux ELF every cleanup just to derive the
                    // same value the monitor thread already had in
                    // hand.
                    let summary = monitor::MonitorSummary::from_samples_with_threshold(
                        &samples,
                        preemption_threshold_ns,
                    );
                    let report = monitor::MonitorReport {
                        samples,
                        summary,
                        preemption_threshold_ns,
                        watchdog_observation,
                        page_offset,
                        boot_wait_outcome,
                        scx_event_counters_supported,
                    };
                    (Some(report), drain, verified_insns)
                }
                None => (None, BulkDrainResult::default(), Vec::new()),
            };
        if crate::vmm::debug_logging_enabled() {
            eprintln!("CLEANUP: monitor joined");
        }
        let cleanup_t = std::time::Instant::now();

        if let Some(h) = run.bpf_write_handle {
            let _ = h.join();
        }
        if crate::vmm::debug_logging_enabled() {
            eprintln!("CLEANUP: bpf_write joined {:?}", cleanup_t.elapsed());
        }

        // Drain the virtio-console port-1 TX accumulator: the guest
        // wrote bulk TLV-framed messages (STIMULUS, EXIT, SCHED_EXIT,
        // PAYLOAD_METRICS, etc.) to
        // `/dev/vport0p1`; the host side accumulated them into
        // `port1_tx_buf` and we parse them here through
        // `parse_tlv_stream`. Port-1 uses backpressure rather than
        // drops — every byte the guest emitted is delivered, in
        // order.
        //
        // `final_drain` (rather than `drain_bulk`) walks the avail
        // ring once before draining so chains the guest published
        // without a host-observed QUEUE_NOTIFY (the
        // `force_reboot()` race in `rust_init`'s `send_exit`-then-
        // reboot tail) are picked up instead of being lost. See
        // [`crate::vmm::virtio_console::VirtioConsole::final_drain`].
        let bulk_bytes = run.virtio_con.lock().final_drain();
        let mut bulk_drain = host_comms::parse_tlv_stream(&bulk_bytes);
        // Strip coordinator-internal control frames the freeze coord
        // mid-run filter (the TOKEN_TX dispatch in this same file)
        // already drops: SNAPSHOT_REQUEST has its matching reply
        // delivered over port-1 RX; SYS_RDY's only semantic is the
        // eventfd promotion in the coord's TOKEN_TX handler.
        // Without this filter, a late-arriving control frame that
        // the coord had not yet consumed when its outer loop
        // exited would land in `guest_messages` and surface as a
        // phantom verdict entry.
        //
        // Both filters key on
        // [`crate::vmm::wire::MsgType::is_coordinator_internal`] —
        // a single source of truth so adding a new internal control
        // frame is a one-line update at the classifier site.
        bulk_drain.entries.retain(|e| {
            // Keep when the msg_type is NOT a recognised
            // coordinator-internal control frame. Unknown
            // msg_types (None) are preserved verbatim so an
            // operator-side analyser can surface them rather
            // than silently dropping them here.
            match crate::vmm::wire::MsgType::from_wire(e.msg_type) {
                Some(t) => !t.is_coordinator_internal(),
                None => true,
            }
        });
        // Prepend the entries the freeze coordinator already parsed
        // mid-run. The coord's TOKEN_TX handler streams port-1
        // bytes through `HostAssembler` so a SCHED_EXIT can flip
        // the run-wide kill flag without waiting for VM exit;
        // those parsed frames stash here on every drain so
        // `collect_results` can recover them after the coord has
        // joined. Without this merge every guest-side EXIT / TEST
        // / PAYLOAD_METRICS / PROFRAW frame
        // consumed mid-run would be silently lost — `drain_bulk()`
        // above only catches what arrived AFTER the coord stopped
        // polling, which on a typical run is empty. Mid-run
        // entries come first so the merged stream stays in
        // chronological order.
        let mut mid_run_bulk = match run.bulk_messages.lock() {
            Ok(mut g) => std::mem::take(&mut *g),
            Err(p) => std::mem::take(&mut *p.into_inner()),
        };
        mid_run_bulk.extend(bulk_drain.entries);
        bulk_drain.entries = mid_run_bulk;

        // Merge mid-flight drain (from monitor thread, port-1 byte
        // stream) with the post-exit `drain_bulk()`. Mid-flight
        // entries come first since they were drained during
        // execution.
        // The complete TLV log (mid-flight + post-exit). The per-phase
        // stimulus timeline (step frames + scenario-end terminal) is
        // derived on demand from these entries via
        // `VmResult::stimulus_timeline()` — no separate pre-extracted
        // stimulus vec is stored, so every consumer sees the same
        // complete timeline (the previous wire-only field omitted the
        // terminal and silently dropped the last step's iteration_rate
        // for post_vm re-derivation).
        let guest_messages =
            if !mid_flight_drain.entries.is_empty() || !bulk_drain.entries.is_empty() {
                let mut all_entries = mid_flight_drain.entries;
                all_entries.extend(bulk_drain.entries);
                Some(BulkDrainResult {
                    entries: all_entries,
                })
            } else {
                None
            };

        let com2_bytes = run.com2.lock().output();
        let console_output = run.com1.lock().output();

        // Concatenate every CRC-valid `MSG_TYPE_STDOUT` /
        // `MSG_TYPE_STDERR` chunk from the bulk-port drain into a
        // single string and prepend the COM2 capture so panic-hook
        // bytes (the lone remaining COM2 writer) still surface in
        // `result.output`. The bulk-port chunks dominate steady-state
        // test output; COM2 is reserved for fault diagnostics that
        // cannot block on virtio backpressure.
        let mut app_output = String::new();
        if let Some(ref drain) = guest_messages {
            for e in &drain.entries {
                if !e.crc_ok {
                    continue;
                }
                match wire::MsgType::from_wire(e.msg_type) {
                    Some(wire::MsgType::Stdout) | Some(wire::MsgType::Stderr) => {
                        app_output.push_str(&String::from_utf8_lossy(&e.payload));
                    }
                    _ => {}
                }
            }
        }
        if !com2_bytes.is_empty() {
            app_output.push_str(&com2_bytes);
        }

        // Extract exit code: bulk port (primary), COM2 sentinel (fallback).
        let bulk_exit = guest_messages.as_ref().and_then(|d| {
            d.entries
                .iter()
                .rev()
                .find(|e| e.msg_type == wire::MSG_TYPE_EXIT && e.crc_ok && e.payload.len() == 4)
                // LE decode: the guest writes the code via `to_le_bytes`
                // (guest_comms::send_exit), so `from_ne_bytes` would
                // byte-swap on a big-endian host. Matches the shell-mode
                // ExecExit decode in `run_interactive`.
                .map(|e| i32::from_le_bytes(e.payload[..4].try_into().unwrap()))
        });
        // Pre-bulk-port-migration: a COM2 `KTSTR_EXIT=N` sentinel line
        // served as the fallback when no binary `MSG_TYPE_EXIT`
        // frame arrived. The fallback is gone — bulk-port
        // backpressure guarantees delivery, and the guest no longer
        // emits the sentinel. A `None` here keeps `exit_code` at
        // whatever the BSP run-loop's local stored, matching the
        // pre-fallback path.
        if let Some(code) = bulk_exit {
            exit_code = code;
        }

        // Extract crash message from COM2 output. The guest panic
        // hook in `rust_init/init.rs` writes `PANIC: <info>\n<bt>\n` to
        // `/dev/ttyS1`; the host-side parser
        // [`crate::test_support::extract_panic_message`] strips the
        // prefix and returns the trimmed remainder.
        let crash_message =
            crate::test_support::extract_panic_message(&app_output).map(|s| s.to_string());

        // Collect BPF verifier stats from host-side memory reads.
        // Skip when no BPF struct_ops scheduler is attached —
        // verifier_stats walks the kernel's `bpf_prog` table looking
        // for struct_ops progs, and those only exist when a sched_ext
        // BPF binary was loaded and `bpf_program__attach_struct_ops`
        // succeeded. The userspace binary is the only path that
        // creates struct_ops; KernelBuiltin's `sched_enable_cmds`
        // are arbitrary `echo X > /sys/...` writes (sysctl-style
        // toggles around kernel-builtin schedulers like EEVDF) and
        // load zero BPF programs, so the previous gate that OR'd in
        // `!self.sched_enable_cmds.is_empty()` falsely signalled
        // "scheduler attached" for KernelBuiltin shell-toggle
        // configs — the walk found nothing (correct) but spent the
        // syscall + accessor cost, and a downstream `verifier_stats.
        // is_empty()` assertion couldn't distinguish "correctly
        // skipped (no BPF)" from "BPF binary failed to attach".
        let has_bpf_scheduler = self.has_bpf_scheduler_attached();
        let vs_t = std::time::Instant::now();
        let mut vs_path: &'static str = "skipped_no_scheduler";
        // Prefer the mid-run capture taken while the scheduler was
        // attached (see `monitor::reader::MonitorLoopResult::
        // verified_insns`). The post-teardown walk runs AFTER the guest
        // killed the scheduler process, which drops the last prog
        // reference and fires synchronous `bpf_prog_free_id`/`idr_remove`
        // (kernel/bpf/syscall.c) — so the walk finds an empty `prog_idr`
        // and loses `verified_insns`. The mid-run value is captured
        // before that teardown and is fixed once at load, so it is
        // authoritative. The source decision is factored into
        // `choose_verifier_stats_source` (unit-tested); the walk stays
        // inline here because it needs `self`/`run`.
        let verifier_stats = match choose_verifier_stats_source(
            mid_run_verified_insns.is_empty(),
            has_bpf_scheduler,
        ) {
            VerifierStatsSource::MidRun => {
                vs_path = "mid_run_capture";
                mid_run_verified_insns
            }
            VerifierStatsSource::PostTeardownWalk => {
                if let Some(ref prog) = run.prog_accessor {
                    use crate::monitor::bpf_prog::BpfProgAccessor;
                    vs_path = "prebuilt_accessor";
                    if crate::vmm::debug_logging_enabled() {
                        eprintln!("CLEANUP: verifier_stats using pre-built accessor");
                    }
                    let a = prog.as_accessor();
                    a.struct_ops_progs()
                } else {
                    vs_path = "fallback_full_parse";
                    if crate::vmm::debug_logging_enabled() {
                        eprintln!("CLEANUP: verifier_stats fallback (full parse)");
                    }
                    self.collect_verifier_stats(
                        &run.vm,
                        run.tcr_el1.as_ref(),
                        &run.cr3,
                        run.vmlinux_data.as_ref().map(|d| d.as_slice()),
                        run.kern_phys_base,
                    )
                }
            }
            VerifierStatsSource::None => Vec::new(),
        };
        tracing::info!(
            elapsed_ms = vs_t.elapsed().as_millis() as u64,
            path = vs_path,
            n_progs = verifier_stats.len(),
            "auto_repro: collect_verifier_stats",
        );
        if crate::vmm::debug_logging_enabled() {
            eprintln!("CLEANUP: verifier_stats done {:?}", vs_t.elapsed());
        }

        // Sample cleanup elapsed AFTER every blocking step that runs on
        // the post-BSP-exit critical path so the duration captures the
        // full host-side teardown cost, not a partial window. The full
        // ordered set is: watchdog join (in `run_vm`, before
        // `cleanup_start` is stored on `VmRunState`), AP joins, monitor
        // join, BPF writer join, bulk drain, exit-code and crash-message
        // extraction, verifier-stat read. Captured before constructing
        // the result so the `Instant::now()` here is the latest possible
        // read.
        let cleanup_duration = Some(run.cleanup_start.elapsed());
        // Close the cleanup-window dilation instrument on the same
        // thread that opened it (see `cleanup_sched_t0`): the delta's
        // run-delay is the wall the host scheduler ADDED to this
        // window; its dilation D_cleanup feeds the budget gate.
        let cleanup_sched_delta = match (run.cleanup_sched_t0, read_self_thread_schedstat()) {
            (Some(t0), Some(t1)) => Some(t1.delta_from(&t0)),
            _ => None,
        };
        tracing::info!(
            elapsed_ms = collect_results_start.elapsed().as_millis() as u64,
            cleanup_window_ms = cleanup_duration.map(|d| d.as_millis() as u64).unwrap_or(0),
            "auto_repro: collect_results",
        );
        if crate::vmm::debug_logging_enabled() {
            eprintln!("CLEANUP: collect_results done {:?}", cleanup_t.elapsed());
        }

        // Forward the scheduler-stats client. `run.stats_client` is
        // `Some(_)` when the run has a scheduler attached and
        // `None` otherwise; the field on `VmResult.stats_client`
        // mirrors this exactly. The drainer thread (when present)
        // continues to run until the last `Arc<ClientShared>` clone
        // drops; `Drop` on the field then writes the kill eventfd
        // and the drainer thread exits.
        let stats_client = run.stats_client;

        // Count periodic captures that landed REAL BPF state, before
        // the bridge is moved into the result. `periodic_fired`
        // counts every attempted boundary including rendezvous-timeout
        // placeholders; `periodic_real` is the placeholder-excluded
        // floor so the failure display can distinguish "fired but
        // degraded" from genuine coverage. Computed here (run just
        // ended, nothing drained yet) so it is independent of any
        // later test-side drain that would empty the bridge.
        let periodic_real = run.snapshot_bridge.periodic_real_count();

        Ok(VmResult {
            success: !timed_out && exit_code == 0,
            vcpus: self.vcpus,
            cpu_budget: self.effective_cpu_budget,
            // Stamped None here; the host eval layer
            // (run_ktstr_test_inner_impl) sets the scheduler's discovery
            // path post-run. The VM run does not resolve the binary.
            resolve_source: None,
            // Default false at construction — set true (when applicable)
            // by the eval-layer inversion site that runs AFTER
            // evaluate_vm_result, preserving the original success +
            // error chain for diagnostic visibility.
            expect_auto_repro_satisfied: false,
            exit_code,
            duration: start.elapsed(),
            timed_out,
            // Mechanism verdicts for wedge fixtures: WHICH watchdog rule
            // killed the run (the dump's `cause=`) and how far the guest
            // got (final ledger phase / milestone count). Decoded from the
            // raw bytes VmRunState captured post-join.
            watchdog_kill_reason: decode_watchdog_kill_reason(run.watchdog_kill_reason_raw),
            final_guest_phase: decode_guest_phase(run.final_guest_phase_raw),
            final_progress_epoch: run.final_progress_epoch,
            // Injection delivery evidence (see the field doc): 0 = no
            // writes configured, 1 = configured but never delivered,
            // 2 = delivered-and-guest-signalled.
            bpf_map_writes_delivered: decode_bpf_map_write_delivery(run.bpf_map_write_delivery_raw),
            // Readiness-vs-window evidence (0 sentinel → None).
            periodic_prereqs_ready: (run.periodic_prereqs_ready_ns_raw != 0)
                .then(|| Duration::from_nanos(run.periodic_prereqs_ready_ns_raw)),
            periodic_window_end: (run.periodic_window_end_ns_raw != 0)
                .then(|| Duration::from_nanos(run.periodic_window_end_ns_raw)),
            output: app_output,
            stderr: console_output,
            monitor: monitor_report,
            guest_messages,
            verifier_stats,
            kvm_stats: None,
            crash_message,
            cleanup_duration,
            cleanup_sched_delta,
            // Snapshot at assignment, not earlier: VmRunState owns
            // the device's Arc<AtomicU64> counter handle until this
            // point. The sole sources of QUEUE_NOTIFY kicks for both
            // devices are the vCPU threads — the BSP completed its
            // run loop earlier (joined back in `run_vm` before `bsp`
            // dropped) and every AP joined upstream in the
            // `run.ap_threads` join loop above. The virtio-blk
            // worker can therefore receive no new kicks (and the
            // intervening monitor-join + bulk-drain phases give it
            // ample time to drain any in-flight one before settling
            // back on its idle epoll_wait); virtio-net's
            // `process_tx_loopback` is single-threaded on the
            // kicking vCPU so once those joined no writer remains.
            // The relaxed loads inside each `snapshot()` therefore
            // observe the device's final cumulative state.
            // Converting here (rather than at device-build time)
            // keeps the devices incrementing through every vCPU
            // exit path that runs before this function — failure
            // dumps, watchdog timeouts, and the normal exit path
            // all see fully-up-to-date counters.
            virtio_blk_counters: run.virtio_blk_counters.as_deref().map(|c| c.snapshot()),
            virtio_net_counters: virtio_net::VirtioNetCountersSnapshot::aggregate(
                run.virtio_net_counters.iter().map(|c| c.snapshot()),
            ),
            snapshot_bridge: run.snapshot_bridge,
            stats_client,
            periodic_fired: run.periodic_fired,
            periodic_real,
            periodic_target: run.periodic_target,
            // Plumb the virt-KASLR snapshot from VmRunState
            // (populated at `VmRunState { kern_kaslr_offset: ... }`
            // above) through to the public VmResult slot. E2E tests
            // read this to assert the KASLR derivation chain
            // produced a non-zero offset on KASLR-on boots.
            kern_kaslr_offset: run.kern_kaslr_offset,
            // `entry_name` is stamped post-construction in
            // `test_support::eval::run_ktstr_test_inner_impl`
            // immediately after `vm.run()` returns (the
            // `KtstrTestEntry::name` `&'static str` is not in
            // scope here in `freeze_coord::collect_results` and
            // shouldn't be — freeze_coord is entry-agnostic).
            // Leaving it `None` here is correct; the eval-layer
            // stamping happens before any post_vm callback runs.
            entry_name: None,
            // Variant hash stamped by the eval layer (eval/mod.rs) after
            // vm.run() returns, alongside entry_name; freeze_coord is
            // entry-agnostic, so `0` here is correct and overwritten.
            variant_hash: 0,
            // Host-side vCPU scheduling dilation, read at `run_vm` teardown
            // from live proc entries plus one-shot AP exit snapshots. Purely
            // observational — never affects `success` above or the exit code.
            host_vcpu_schedstat: run.host_vcpu_schedstat,
            // Event-anchored per-phase schedstat + constant-cost CPU-pressure
            // series, finalized from live proc entries plus AP exit snapshots.
            contention_witness: run.contention_witness,
            // Empty cache: the single bridge drain is deferred to the
            // first `captures_series()` call on the host (post_vm or
            // evaluate_vm_result). See the `periodic_series_cache`
            // field doc.
            periodic_series_cache: std::sync::OnceLock::new(),
        })
    }

    /// Read BPF verifier stats from guest memory after VM exit.
    ///
    /// Enumerates struct_ops programs in the kernel's `prog_idr` and
    /// reads `bpf_prog_aux->verified_insns` for each.
    pub(super) fn collect_verifier_stats(
        &self,
        vm: &kvm::KtstrKvm,
        tcr_el1: Option<&Arc<std::sync::atomic::AtomicU64>>,
        cr3: &Arc<std::sync::atomic::AtomicU64>,
        cached_vmlinux_data: Option<&[u8]>,
        kern_phys_base_biased: u64,
    ) -> Vec<monitor::bpf_prog::ProgVerifierStats> {
        let vmlinux = match find_vmlinux(&self.kernel) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let mem = match vm.numa_layout.as_ref() {
            Some(layout) => monitor::reader::GuestMem::from_layout(layout, &vm.guest_mem),
            None => {
                use vm_memory::GuestMemoryRegion;
                let host_base = match vm.guest_mem.get_host_address(GuestAddress(DRAM_BASE)) {
                    Ok(ptr) => ptr,
                    Err(_) => return Vec::new(),
                };
                // Size of the first contiguous region only.
                // host_base addresses that single mapping; using the
                // sum of all region lengths would extend past the
                // mapping into host heap when multiple regions exist.
                let mem_size = match vm.guest_mem.iter().next() {
                    Some(r) => r.len(),
                    None => return Vec::new(),
                };
                // SAFETY: host_base is from GuestMemoryMmap's mapping,
                // which outlives this GuestMem (borrowed via `vm` for
                // the body of this function).
                unsafe { monitor::reader::GuestMem::new(host_base, mem_size) }
            }
        };
        // TCR_EL1 (aarch64) drives the granule-agnostic page-table
        // walker. The BSP populates this Arc<AtomicU64> on first
        // successful read post-MMU-bringup; by collect_verifier_stats
        // time it is either set (kernel booted) or 0 (kernel never
        // brought MMU up, e.g. early boot crash). The walker treats
        // 0 as "no TCR available — translation unsupported", which
        // matches the boot-crash case where verifier stats are
        // unavailable anyway.
        let tcr_val = tcr_el1
            .map(|c| c.load(std::sync::atomic::Ordering::Acquire))
            .unwrap_or(0);
        let cr3_val = cr3.load(std::sync::atomic::Ordering::Acquire);
        // Fallback when the caller did not pre-load the vmlinux ELF.
        // Routes through `cached_vmlinux_bytes` so a test process that
        // boots N VMs against the same kernel pays the 50-340 MB read
        // exactly once. Without the cache this path was the dominant
        // cost on `collect_results` cleanup for nextest runs of
        // `#[ktstr_test]` cases that share a kernel.
        let owned_data;
        let vmlinux_data: &[u8] = match cached_vmlinux_data {
            Some(d) => d,
            None => match cached_vmlinux_bytes(&vmlinux) {
                Some(arc) => {
                    owned_data = arc;
                    owned_data.as_slice()
                }
                None => return Vec::new(),
            },
        };
        // Parse the vmlinux ELF once and share the result between
        // `GuestKernel` (kernel symbols + paging state) and
        // `BpfProgOffsets` (BTF section extraction on cache miss).
        // The previous structure parsed the ELF up to three times per
        // call: once inside `GuestKernel::from_vmlinux_bytes`, once
        // again via the nested `KernelSymbols::from_vmlinux_bytes`,
        // and once more via `load_btf_from_bytes` on a sidecar miss.
        // `goblin::elf::Elf::parse` is hundreds of ms on a debug
        // vmlinux, so this single parse is the cheap shared base.
        let elf = match goblin::elf::Elf::parse(vmlinux_data) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        let pb_hint = if kern_phys_base_biased != 0 {
            kern_phys_base_biased.wrapping_sub(1)
        } else {
            0
        };
        let kernel = match monitor::guest::GuestKernel::from_elf_with_hint(
            Arc::new(mem),
            &elf,
            tcr_val,
            cr3_val,
            pb_hint,
        ) {
            Ok(k) => k,
            Err(_) => return Vec::new(),
        };
        // BTF sidecar cache hits skip ELF traversal entirely; on a
        // miss `load_btf_from_elf` reuses the parse above instead of
        // re-running `goblin::elf::Elf::parse(&vmlinux_data)`.
        let offsets =
            match monitor::btf_offsets::BpfProgOffsets::from_elf(&elf, vmlinux_data, &vmlinux) {
                Ok(o) => o,
                Err(_) => return Vec::new(),
            };
        let accessor =
            match monitor::bpf_prog::GuestMemProgAccessor::from_guest_kernel(&kernel, &offsets) {
                Ok(a) => a,
                Err(_) => return Vec::new(),
            };
        // Trait method — `BpfProgAccessor::struct_ops_progs` is in
        // scope via the local `use monitor::bpf_prog::BpfProgAccessor`
        // below; calling it on the concrete type dispatches statically.
        use monitor::bpf_prog::BpfProgAccessor;
        accessor.struct_ops_progs()
    }
}

#[cfg(test)]
mod bss_latch_rescues_suppressed_dump_tests {
    //! Truth table for [`bss_latch_rescues_suppressed_dump`] — the
    //! gate-suppress-but-bss-latch-rescue (F15-class) decision, reached
    //! only when the exit_kind gate decided to suppress. The host/guest
    //! process boundary makes an e2e seam impossible (a static set by
    //! the guest test payload never reaches the host coordinator's own
    //! copy), so this pure-fn table is the honest coverage; the
    //! dump-emit happy path is exercised by the real-stall watchdog
    //! tests in tests/silent_drop_e2e.rs.
    use super::{BssReadState, bss_latch_rescues_suppressed_dump};

    #[test]
    fn triggered_not_oob_rescues() {
        // The probe latch independently observed an error-class exit and
        // the cached `.bss` PA is still valid → rescue the suppressed
        // dump (the latch is the historical authority).
        assert!(bss_latch_rescues_suppressed_dump(
            BssReadState::Triggered,
            false
        ));
    }

    #[test]
    fn triggered_but_oob_honors_suppression() {
        // OOB latched this run: a non-zero read on the cached PA could
        // be recycled vmalloc bytes, not a real fire → do not rescue.
        assert!(!bss_latch_rescues_suppressed_dump(
            BssReadState::Triggered,
            true
        ));
    }

    #[test]
    fn not_triggered_honors_suppression() {
        assert!(!bss_latch_rescues_suppressed_dump(
            BssReadState::NotTriggered,
            false
        ));
    }

    #[test]
    fn out_of_bounds_honors_suppression() {
        assert!(!bss_latch_rescues_suppressed_dump(
            BssReadState::OutOfBounds,
            false
        ));
    }

    #[test]
    fn not_resolved_honors_suppression() {
        assert!(!bss_latch_rescues_suppressed_dump(
            BssReadState::NotResolved,
            false
        ));
    }
}

#[cfg(test)]
mod exit_kind_warrants_dump_tests {
    //! Truth table for [`exit_kind_warrants_dump`] — the pure exit_kind
    //! gate decision. `Some(kind)` = the live `*scx_root->exit_kind`
    //! read through the translated PA; `None` = the KVA no longer
    //! translates (slab page freed mid-teardown). Replaces the deleted
    //! `FREEZE_COORD_TEST_FORCE_TRANSLATE_NONE` host-static seam's
    //! coverage of the translate-fail branch with a host-side unit test.
    use super::exit_kind_warrants_dump;

    #[test]
    fn error_class_warrants_dump() {
        // SCX_EXIT_ERROR = 1024 is the first error-class value; the
        // boundary value and anything above it warrant a dump.
        assert!(exit_kind_warrants_dump(Some(1024)));
        assert!(exit_kind_warrants_dump(Some(2048)));
    }

    #[test]
    fn clean_exit_does_not_warrant_dump() {
        // NONE/DONE and the normal unregister classes all sit below
        // SCX_EXIT_ERROR — a clean shutdown must not emit a dump.
        assert!(!exit_kind_warrants_dump(Some(0)));
        assert!(!exit_kind_warrants_dump(Some(1023)));
    }

    #[test]
    fn translate_failure_does_not_warrant_dump() {
        // None = the exit_kind KVA no longer translates (scheduler torn
        // down); there is no state to capture, so suppress.
        assert!(!exit_kind_warrants_dump(None));
    }
}

#[cfg(test)]
mod wprof_grace_tests {
    //! Unit coverage for the wprof-ship grace predicates the
    //! coordinator's error-exit grace relies on: [`is_wprof_ship_frame`]
    //! (which drained frame ends the grace on a plain wprof run),
    //! [`is_sched_exit_frame`] (which drained frame ARMS the grace on a
    //! wprof run), [`is_probe_output_end_frame`] (which drained frame
    //! ends the grace on an auto-repro probe run), and
    //! [`wprof_grace_should_kill`] (the bounded deadline backstop).
    use super::{
        is_probe_output_end_frame, is_sched_exit_frame, is_wprof_ship_frame,
        wprof_grace_should_kill,
    };
    use crate::vmm::bulk::BulkMessage;
    use crate::vmm::wire::{MSG_TYPE_SCHED_EXIT, MSG_TYPE_STDOUT, MSG_TYPE_WPROF_TRACE};
    use std::time::{Duration, Instant};

    fn frame(msg_type: u32, payload: &[u8], crc_ok: bool) -> BulkMessage {
        BulkMessage {
            msg_type,
            payload: payload.to_vec().into(),
            crc_ok,
        }
    }

    #[test]
    fn is_wprof_ship_frame_accepts_only_valid_nonempty_wprof() {
        assert!(
            is_wprof_ship_frame(&frame(MSG_TYPE_WPROF_TRACE, b"pb", true)),
            "a crc-valid non-empty WprofTrace IS the ship"
        );
        assert!(
            !is_wprof_ship_frame(&frame(MSG_TYPE_WPROF_TRACE, b"pb", false)),
            "torn CRC must not end the grace — a garbled/hostile frame is not the ship"
        );
        assert!(
            !is_wprof_ship_frame(&frame(MSG_TYPE_WPROF_TRACE, b"", true)),
            "empty payload carries no trace — not the ship"
        );
        assert!(
            !is_wprof_ship_frame(&frame(MSG_TYPE_SCHED_EXIT, b"x", true)),
            "a SchedExit is not the ship — it must not terminate the grace early"
        );
    }

    #[test]
    fn is_sched_exit_frame_accepts_only_crc_valid_sched_exit() {
        assert!(
            is_sched_exit_frame(&frame(MSG_TYPE_SCHED_EXIT, b"\0\0\0\0", true)),
            "a crc-valid SCHED_EXIT arms the grace on a wprof run"
        );
        assert!(
            !is_sched_exit_frame(&frame(MSG_TYPE_SCHED_EXIT, b"\0\0\0\0", false)),
            "torn CRC must not arm the grace — a garbled/hostile frame is not a SCHED_EXIT"
        );
        assert!(
            !is_sched_exit_frame(&frame(MSG_TYPE_WPROF_TRACE, b"pb", true)),
            "a WprofTrace is not a SCHED_EXIT"
        );
    }

    #[test]
    fn is_probe_output_end_frame_accepts_only_crc_valid_stdout_with_marker() {
        let end = crate::test_support::PROBE_OUTPUT_END.as_bytes();
        let mut framed = b"...json...\n".to_vec();
        framed.extend_from_slice(end);
        framed.extend_from_slice(b"\n");
        assert!(
            is_probe_output_end_frame(&frame(MSG_TYPE_STDOUT, &framed, true)),
            "a crc-valid stdout frame carrying PROBE_OUTPUT_END ends the grace"
        );
        assert!(
            !is_probe_output_end_frame(&frame(MSG_TYPE_STDOUT, &framed, false)),
            "torn CRC must not end the grace — a garbled/hostile frame is not the payload"
        );
        assert!(
            !is_probe_output_end_frame(&frame(MSG_TYPE_STDOUT, b"ordinary stdout line\n", true)),
            "stdout without the marker is not the probe-payload terminator"
        );
        assert!(
            !is_probe_output_end_frame(&frame(MSG_TYPE_WPROF_TRACE, end, true)),
            "the marker on a non-stdout frame type does not count"
        );
    }

    #[test]
    fn wprof_grace_should_kill_only_on_expired_armed_deadline() {
        let now = Instant::now();
        assert!(
            !wprof_grace_should_kill(None, now),
            "no grace armed => never kill"
        );
        assert!(
            !wprof_grace_should_kill(Some(now + Duration::from_secs(30)), now),
            "future deadline => hold the grace open"
        );
        assert!(
            wprof_grace_should_kill(Some(now - Duration::from_millis(1)), now),
            "expired deadline => kill (bounded fallback)"
        );
        assert!(
            wprof_grace_should_kill(Some(now), now),
            "deadline exactly now => kill (inclusive >= boundary)"
        );
    }
}

#[cfg(test)]
mod has_bpf_scheduler_attached_tests {
    //! 4-case truth table for [`has_bpf_scheduler_attached_inner`]
    //! — the verifier_stats gating predicate. The matrix covers
    //! every cross-product of `(scheduler_binary, sched_enable_cmds)`
    //! so a future regression that re-introduces the
    //! `|| !sched_enable_cmds.is_empty()` clause surfaces here as
    //! the `binary=None, cmds=non-empty` test flipping from `false`
    //! to `true`.
    //!
    //! The KernelBuiltin false-pass class lived precisely in
    //! the `binary=None, cmds=non-empty` cell — KernelBuiltin tests
    //! never set a scheduler binary (they're shell-toggle wrappers
    //! around kernel-builtin EEVDF/CFS) but DO set enable cmds
    //! (sysctl-style `echo X > /sys/...` writes). Pre-fix: the gate
    //! evaluated true and verifier_stats walked the BPF prog table
    //! for nothing; a downstream `verifier_stats.is_empty()`
    //! assertion couldn't distinguish "correctly skipped (no BPF)"
    //! from "BPF binary failed to attach." Post-fix: the gate
    //! evaluates false, verifier_stats stays an empty Vec by
    //! construction, and the assertion semantics align with the
    //! actual kernel state.

    use super::has_bpf_scheduler_attached_inner;
    use std::path::PathBuf;

    /// BPF binary present, no sysctl enable cmds: the canonical
    /// `Discover("scx-foo")` / `Path("/usr/local/bin/scx-foo")`
    /// shape. The gate fires, verifier_stats collection runs.
    #[test]
    fn binary_some_cmds_empty_returns_true() {
        let binary = PathBuf::from("/path/to/scx-foo");
        let cmds: Vec<String> = Vec::new();
        assert!(
            has_bpf_scheduler_attached_inner(Some(&binary), &cmds),
            "Discover/Path with no enable cmds must gate verifier_stats on"
        );
    }

    /// BPF binary present AND sysctl enable cmds set: the rare
    /// shape where a Discover/Path scheduler ALSO ships
    /// `sched_enable_cmds` (e.g. to twiddle a sysctl before
    /// attaching the BPF program). BPF wins — verifier_stats
    /// collection runs, exactly the same as the binary-only case.
    #[test]
    fn binary_some_cmds_non_empty_returns_true() {
        let binary = PathBuf::from("/path/to/scx-foo");
        let cmds: Vec<String> = vec!["echo 1 > /sys/kernel/foo".into()];
        assert!(
            has_bpf_scheduler_attached_inner(Some(&binary), &cmds),
            "BPF binary present must gate verifier_stats on regardless of \
             enable cmds; the `enable cmds` arg is intentionally ignored \
             once the binary signal flips true"
        );
    }

    /// No BPF binary, no sysctl enable cmds: the `SchedulerSpec::
    /// Eevdf` shape (kernel default scheduler, no scheduler-author
    /// involvement). Gate stays off, verifier_stats skipped.
    #[test]
    fn binary_none_cmds_empty_returns_false() {
        let binary: Option<&PathBuf> = None;
        let cmds: Vec<String> = Vec::new();
        assert!(
            !has_bpf_scheduler_attached_inner(binary, &cmds),
            "Eevdf default scheduler (no binary, no cmds) must NOT gate \
             verifier_stats — there's no BPF prog table to walk"
        );
    }

    /// **REGRESSION PIN**: no BPF binary BUT sysctl enable cmds
    /// non-empty. The `SchedulerSpec::KernelBuiltin { enable: [..],
    /// disable: [..] }` shape — shell-toggle wrappers around
    /// kernel-builtin schedulers (EEVDF, CFS) where the test's
    /// only customization is sysctl-style writes. Pre-fix, the
    /// `||` clause on `!sched_enable_cmds.is_empty()` fired this
    /// case `true`, walked the BPF prog table for nothing, and
    /// masked downstream `verifier_stats.is_empty()` assertions.
    /// Post-fix, the predicate ignores the cmds and returns
    /// `false` — verifier_stats stays empty by construction.
    ///
    /// A regression that re-introduces the OR-clause flips this
    /// assertion to `true`. Don't suppress; investigate the
    /// caller-driven reason and either update the comment +
    /// predicate together OR fix the regressing change.
    #[test]
    fn binary_none_cmds_non_empty_returns_false_regression_pin() {
        let binary: Option<&PathBuf> = None;
        let cmds: Vec<String> = vec![
            "echo Y > /sys/kernel/sched/foo".into(),
            "echo bar > /proc/sys/kernel/baz".into(),
        ];
        assert!(
            !has_bpf_scheduler_attached_inner(binary, &cmds),
            "KernelBuiltin (sysctl-toggle, no BPF binary) must NOT gate \
             verifier_stats on — enable cmds don't load BPF programs. \
             A regression flipping this to `true` likely re-introduced \
             the `|| !sched_enable_cmds.is_empty()` clause at the \
             collect_results predicate; that OR clause was the \
             false-pass introduction this fix removed"
        );
    }
}

#[cfg(test)]
mod crc_defense_tests;
#[cfg(test)]
mod rendezvous_tests;
#[cfg(test)]
mod snapshot_tlv_tests;
#[cfg(test)]
mod tx_dispatch_tests;

#[cfg(test)]
mod utf8_safe_truncate_len_tests {
    //! Unit coverage for [`utf8_safe_truncate_len`] — the stderr
    //! last-ditch preservation helper used by the failure-dump
    //! emit_json / emit_degraded_json closures and the end-of-coord
    //! drain. The function exists to prevent a panic when naive
    //! byte-slicing `&json[..cap]` would land inside a multi-byte
    //! UTF-8 sequence (U+FFFD from task `comm` lossy decode, kernel
    //! symbol Unicode, operator-supplied tags). This helper is what
    //! KEEPS the stderr-fallback path from converting a silent drop
    //! into a panic that skips coord cleanup. Coverage here pins the
    //! correctness of every UTF-8 boundary scenario the production
    //! callers can encounter.
    use super::utf8_safe_truncate_len;

    /// Empty input always returns 0 — `&""[..0]` is a valid empty
    /// slice. No walk needed.
    #[test]
    fn empty_input_returns_zero() {
        assert_eq!(utf8_safe_truncate_len("", 0), 0);
        assert_eq!(utf8_safe_truncate_len("", 16), 0);
        assert_eq!(utf8_safe_truncate_len("", usize::MAX), 0);
    }

    /// Cap zero always returns 0 — degenerate "no payload" case.
    /// The drain's stderr-truncation path with cap=0 would emit an
    /// empty head, which is acceptable (the structured summary
    /// line preserves the operator-critical signals).
    #[test]
    fn cap_zero_returns_zero() {
        assert_eq!(utf8_safe_truncate_len("hello", 0), 0);
        assert_eq!(utf8_safe_truncate_len("\u{FFFD}", 0), 0);
    }

    /// Cap >= str length returns str length (entire string is a
    /// valid prefix). No truncation needed; caller emits everything.
    #[test]
    fn cap_at_or_above_len_returns_full_len() {
        assert_eq!(utf8_safe_truncate_len("hi", 2), 2);
        assert_eq!(utf8_safe_truncate_len("hi", 5), 2);
        assert_eq!(utf8_safe_truncate_len("hi", usize::MAX), 2);
    }

    /// Pure-ASCII input: cap inside the string returns cap exactly
    /// (every byte index is a char boundary in ASCII).
    #[test]
    fn ascii_cap_within_len_returns_cap_exactly() {
        let s = "abcdefghij"; // 10 bytes
        assert_eq!(utf8_safe_truncate_len(s, 0), 0);
        assert_eq!(utf8_safe_truncate_len(s, 1), 1);
        assert_eq!(utf8_safe_truncate_len(s, 5), 5);
        assert_eq!(utf8_safe_truncate_len(s, 9), 9);
        assert_eq!(utf8_safe_truncate_len(s, 10), 10);
    }

    /// REPLACEMENT CHARACTER U+FFFD is 3 bytes in UTF-8
    /// (`EF BF BD`). This is the exact byte sequence
    /// `String::from_utf8_lossy` (used by task `comm` decode at
    /// `task_enrichment.rs::read_comm`) substitutes for invalid
    /// input — the most realistic vector for the original panic.
    /// Cap inside the 3-byte sequence must walk back to the
    /// boundary BEFORE the sequence.
    #[test]
    fn walks_back_past_replacement_char() {
        // "a" (1 byte) + U+FFFD (3 bytes) + "b" (1 byte) = 5 bytes
        let s = "a\u{FFFD}b";
        assert_eq!(s.len(), 5);
        assert_eq!(utf8_safe_truncate_len(s, 0), 0);
        // Cap 1 lands at "a" boundary — valid.
        assert_eq!(utf8_safe_truncate_len(s, 1), 1);
        // Cap 2 lands INSIDE U+FFFD (byte 1 of 3) — walks back to 1.
        assert_eq!(utf8_safe_truncate_len(s, 2), 1);
        // Cap 3 lands INSIDE U+FFFD (byte 2 of 3) — walks back to 1.
        assert_eq!(utf8_safe_truncate_len(s, 3), 1);
        // Cap 4 lands at the boundary AFTER U+FFFD — valid.
        assert_eq!(utf8_safe_truncate_len(s, 4), 4);
        // Cap 5 lands at end — valid.
        assert_eq!(utf8_safe_truncate_len(s, 5), 5);
    }

    /// 4-byte UTF-8 sequence (the maximum per RFC 3629). The walk
    /// must retreat at most 3 bytes to find a valid boundary —
    /// pins the worst-case retreat distance the doc claims.
    #[test]
    fn walks_back_past_4byte_sequence() {
        // U+1F600 GRINNING FACE = "\u{1F600}" = 0xF0 0x9F 0x98 0x80
        // = 4 bytes. Surround with ASCII to test boundary walks.
        let s = "x\u{1F600}y";
        assert_eq!(s.len(), 6);
        // Cap 1 at "x" boundary.
        assert_eq!(utf8_safe_truncate_len(s, 1), 1);
        // Caps 2/3/4 inside the 4-byte sequence — all walk back to 1.
        assert_eq!(utf8_safe_truncate_len(s, 2), 1);
        assert_eq!(utf8_safe_truncate_len(s, 3), 1);
        assert_eq!(utf8_safe_truncate_len(s, 4), 1);
        // Cap 5 at the boundary after the 4-byte sequence — valid.
        assert_eq!(utf8_safe_truncate_len(s, 5), 5);
        // Cap 6 at end — valid.
        assert_eq!(utf8_safe_truncate_len(s, 6), 6);
    }

    /// Slicing the returned head must not panic for any input /
    /// cap combination. Property-style coverage of the central
    /// safety guarantee: caller does `&s[..returned]` after the
    /// helper; this must never trigger Rust's char-boundary panic.
    /// Iterates a small set of inputs and caps, asserts no panic.
    #[test]
    fn slice_with_returned_len_never_panics() {
        let inputs = [
            "",
            "a",
            "\u{FFFD}",
            "a\u{FFFD}b",
            "\u{FFFD}\u{FFFD}\u{FFFD}",
            "x\u{1F600}y",
            "Hello, 世界! \u{1F600}",
        ];
        for s in inputs {
            for cap in [0, 1, 2, 3, 4, 5, 10, 100, usize::MAX] {
                let head_end = utf8_safe_truncate_len(s, cap);
                // The actual safety guarantee: this slice must not
                // panic. We assign to `_` to make the side effect
                // explicit; the slice operation itself is the
                // assertion.
                let _ = &s[..head_end];
            }
        }
    }
}

#[cfg(test)]
mod vcpu_none_indices_tests {
    //! Unit coverage for [`vcpu_none_indices`] — surfaces WHICH vCPUs
    //! failed to park during freeze rendezvous (or whose KVM_GET_REGS
    //! errored mid-shutdown) in every structured-summary stderr
    //! preservation line. An operator needs the index list, not just
    //! the count, to triage a hung VM — knowing "vCPUs [1, 3] are
    //! stalled" tells the operator which vCPU threads to inspect in
    //! the kernel log or `perf top -p`; "2 vCPUs missing" does not.
    use super::vcpu_none_indices;
    use crate::monitor::dump::VcpuRegSnapshot;

    // Build a synthetic snapshot for the Some arm. The field values
    // don't matter for this test — only the Some/None distinction
    // drives the helper's filter_map. Constructed in-crate so
    // `#[non_exhaustive]` doesn't block; tcr_el1/user_page_table_root
    // default to None (arch-conditional fields).
    fn snap() -> VcpuRegSnapshot {
        VcpuRegSnapshot {
            instruction_pointer: 0,
            stack_pointer: 0,
            page_table_root: 0,
            user_page_table_root: None,
            tcr_el1: None,
        }
    }

    /// Empty input — degenerate case (would happen if the monitor
    /// crashed before the per-vCPU snapshot vec was allocated).
    /// Returns empty Vec; caller renders `vcpu_none_indices=[]` —
    /// correct, there are no vCPUs to call "none".
    #[test]
    fn empty_vec_returns_empty() {
        let regs: Vec<Option<VcpuRegSnapshot>> = vec![];
        assert_eq!(vcpu_none_indices(&regs), Vec::<usize>::new());
    }

    /// All vCPUs parked + register-captured successfully — happy
    /// path. Returns empty Vec; operator sees `vcpu_none_indices=[]`
    /// signaling zero stalled vCPUs.
    #[test]
    fn all_some_returns_empty() {
        let regs = vec![Some(snap()), Some(snap()), Some(snap())];
        assert_eq!(vcpu_none_indices(&regs), Vec::<usize>::new());
    }

    /// All vCPUs failed (e.g. all `KVM_GET_REGS` errored or rendezvous
    /// timed out before any vCPU parked) — returns every index. The
    /// worst-case stall scenario; operator triages by re-running with
    /// an extended freeze timeout or reading kernel logs for the
    /// affected vCPU threads.
    #[test]
    fn all_none_returns_all_indices() {
        let regs: Vec<Option<VcpuRegSnapshot>> = vec![None, None, None];
        assert_eq!(vcpu_none_indices(&regs), vec![0, 1, 2]);
    }

    /// Mixed Some/None — the realistic partial-stall case. Pins the
    /// asymmetry: indices 1 and 3 stalled, 0 and 2 succeeded. A bug
    /// that swapped the filter polarity (`r.is_some()` instead of
    /// `r.is_none()`) would return [0, 2] here and would vacuously
    /// pass the `all_some` / `all_none` cases — this is the case
    /// that catches a polarity flip.
    #[test]
    fn mixed_returns_only_none_indices() {
        let regs = vec![Some(snap()), None, Some(snap()), None];
        assert_eq!(vcpu_none_indices(&regs), vec![1, 3]);
    }
}

#[cfg(test)]
mod format_path_part_tests {
    //! Unit coverage for [`format_path_part`] — the trace-summary
    //! disambiguator at the tail of the emit_json + emit_degraded_json
    //! closures. Three operator-visible states must render distinctly
    //! so a reader of the tracing::info!/warn! line alone (without
    //! scrolling to the preceding atomic-write-failed tracing::warn!)
    //! can correctly attribute the outcome. Misattribution of an FS
    //! error to "operator forgot to configure a sink" is the
    //! operator-misread class this helper prevents.
    use super::format_path_part;

    /// Successful write to a configured sink — renders ` -> {p}` so
    /// the operator sees the on-disk dump path inline with the
    /// summary.
    #[test]
    fn success_renders_path_arrow() {
        assert_eq!(
            format_path_part(Some("/tmp/dump.json"), false),
            " -> /tmp/dump.json"
        );
    }

    /// Sink configured but atomic write failed (ENOSPC / EROFS /
    /// EACCES) — renders the "see preceding warn" hint so the
    /// operator knows to scroll up for the underlying io::Error.
    #[test]
    fn write_failed_no_path_directs_to_warn() {
        assert_eq!(
            format_path_part(None, true),
            " (atomic write failed; see preceding warn)"
        );
    }

    /// Operator never wired `failure_dump_path` (verifier / shell /
    /// template default = None) — renders the quiet expected-behavior
    /// case. No capture was requested, no error to report.
    #[test]
    fn no_sink_renders_no_file_sink() {
        assert_eq!(format_path_part(None, false), " (no file sink)");
    }

    /// Defensive case — sink configured + write_failed flag set.
    /// UNREACHABLE by construction: the and_then Err arm in both
    /// emit closures sets `write_failed = true` AND returns None,
    /// while the Ok arm returns Some and never sets write_failed.
    /// If a future refactor breaks that invariant, the `_` wildcard
    /// renders the success arrow rather than misclassifying — the
    /// path string is more useful than a write-failure marker when
    /// both are present.
    #[test]
    fn defensive_path_with_write_failed_renders_arrow() {
        assert_eq!(format_path_part(Some("/x"), true), " -> /x");
    }
}

#[cfg(test)]
mod format_truncation_marker_tests {
    //! Unit coverage for [`format_truncation_marker`] — the stderr
    //! payload-head suffix used by all 3 stderr-fallback sites
    //! (emit_json / emit_degraded_json / end-of-coord drain). When
    //! the safe-truncated head drops bytes, the operator must see a
    //! marker that quantifies the loss ("truncated to N bytes; full
    //! M bytes lost — summary above") so they know the preceding
    //! summary line is the authoritative signal source, not the
    //! (truncated) payload head.
    use super::format_truncation_marker;

    /// No truncation — head_end reached the full payload. Returns
    /// empty so the caller's `eprintln!` reads cleanly without a
    /// no-op clause cluttering the output.
    #[test]
    fn no_truncation_returns_empty() {
        assert_eq!(format_truncation_marker(100, 100), "");
    }

    /// Truncation happened — head_end shorter than total. Renders
    /// the "(truncated to N bytes at UTF-8 boundary; D of M payload
    /// bytes dropped — summary above)" marker with the head size,
    /// dropped byte count, and total payload size populated. The
    /// dropped count is `total - head_end` and is rendered explicitly
    /// so an operator does not have to do mental arithmetic to
    /// recover how much of the dump was lost.
    #[test]
    fn truncation_renders_byte_counts() {
        let marker = format_truncation_marker(16, 1024);
        assert_eq!(
            marker,
            " (truncated to 16 bytes at UTF-8 boundary; 1008 of 1024 payload bytes dropped — summary above)"
        );
    }

    /// Empty payload + zero head — degenerate case (no payload to
    /// truncate). Returns empty (no-truncation branch).
    #[test]
    fn empty_payload_zero_head_returns_empty() {
        assert_eq!(format_truncation_marker(0, 0), "");
    }

    /// Defensive case — head_end claims more bytes than total. UNREACHABLE
    /// by construction: utf8_safe_truncate_len caps `bounded` at
    /// `cap.min(s.len())` so the returned head_end is always ≤ total.
    /// If a future refactor breaks that invariant, the helper returns
    /// empty (no-truncation branch) rather than emitting a nonsense
    /// marker with negative loss — the safer degradation.
    #[test]
    fn defensive_head_above_total_returns_empty() {
        assert_eq!(format_truncation_marker(200, 100), "");
    }
}

#[cfg(test)]
mod write_to_tagged_path_tests {
    //! Unit coverage for [`write_to_tagged_path`] — the on-demand
    //! dispatch sites' atomic-publish helper. When an operator wires
    //! `failure_dump_path`, the on-demand snapshot dispatch sites
    //! (TLV CAPTURE / periodic / user-watchpoint / early-snapshot
    //! Degraded) must surface visible signal regardless of whether
    //! the FS write succeeds — same contract emit_json + the drain
    //! enforce. Coverage here pins the no-sink, success, and
    //! write-failure branches.
    use super::write_to_tagged_path;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// `dump_path: None` — verifier / shell / template iteration.
    /// Helper returns `Ok(None)` without touching the stderr_summary
    /// closure. Pins the "no capture requested" silent-skip
    /// invariant — this case is NOT a silent drop because no capture
    /// was ever requested.
    #[test]
    fn no_sink_returns_ok_none_without_invoking_stderr_summary() {
        let summary_called = AtomicBool::new(false);
        let result = write_to_tagged_path(
            None,
            "test_tag",
            "{\"k\":\"v\"}",
            || {
                summary_called.store(true, Ordering::Relaxed);
                String::from("test summary")
            },
            "test warn msg",
        );
        assert!(matches!(result, Ok(None)));
        assert!(
            !summary_called.load(Ordering::Relaxed),
            "stderr_summary closure must NOT fire on no-sink path"
        );
    }

    /// Happy path — sink wired, payload written atomically.
    /// Returns `Ok(Some(tagged))`, file content matches the JSON
    /// byte-for-byte, the stderr_summary closure is NOT invoked
    /// (no fallback needed). Tagged path matches
    /// `snapshot_tagged_path`'s output — the helper does not invent
    /// its own naming scheme, so the on-demand sites' tag layout
    /// stays consistent with what the existing late-trigger path
    /// produces.
    #[test]
    fn successful_write_publishes_tagged_path_with_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().join("dump.failure-dump.json");
        let payload = "{\"schema\":\"single\",\"k\":\"v\"}";
        let summary_called = AtomicBool::new(false);
        let result = write_to_tagged_path(
            Some(&base),
            "ondemand_test",
            payload,
            || {
                summary_called.store(true, Ordering::Relaxed);
                String::from("test summary")
            },
            "test warn msg",
        );
        let tagged = result.expect("write succeeds").expect("path returned");
        assert!(
            !summary_called.load(Ordering::Relaxed),
            "stderr_summary must NOT fire on successful write"
        );
        let written = std::fs::read_to_string(&tagged).expect("read tagged");
        assert_eq!(written, payload);
        let expected = super::snapshot_tagged_path(&base, "ondemand_test");
        assert_eq!(tagged, expected);
        let tmp = tagged.with_extension("json.tmp");
        assert!(!tmp.exists(), "tmp sibling must not linger after rename");
    }

    /// Write-failure path — sink wired, but the parent directory
    /// can't be created because a regular file blocks the path.
    /// `File::create` on the tmp returns ENOTDIR. Helper returns
    /// `Err`, fires the stderr_summary closure so the operator sees
    /// the structured signal even when the FS write didn't land.
    /// Pins stderr-fallback discipline on the on-demand path:
    /// configured-sink-failed never silently drops.
    #[test]
    fn write_failure_invokes_stderr_summary_and_returns_err() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file_in_the_way = dir.path().join("not_a_dir");
        std::fs::write(&file_in_the_way, b"").expect("create blocker file");
        let base = file_in_the_way.join("sub").join("dump.failure-dump.json");
        let summary_called = AtomicBool::new(false);
        let result = write_to_tagged_path(
            Some(&base),
            "test_tag",
            "{\"k\":\"v\"}",
            || {
                summary_called.store(true, Ordering::Relaxed);
                String::from("test summary")
            },
            "test warn msg",
        );
        assert!(result.is_err(), "expected Err on write failure");
        assert!(
            summary_called.load(Ordering::Relaxed),
            "stderr_summary closure MUST fire on write failure (no silent drop)"
        );
    }

    /// Rename-failure tmp-cleanup — File::create + sync_all succeed,
    /// but `rename(tmp, tagged)` fails because the tagged destination
    /// already exists as a non-empty directory. Linux `rename(2)`
    /// returns EISDIR when source is a file and dest is a directory
    /// (empty/non-empty distinction does not matter); ENOTEMPTY
    /// applies for the dir-replacing-non-empty-dir case (both sides
    /// directories). A non-empty dir at the dest adds belt-and-
    /// suspenders against any FS that allows file-over-empty-dir
    /// rename. Helper must clean up the tmp file so a future
    /// operator does not see a stale `.json.tmp` alongside the
    /// blocker. Pins the invariant: tmp cleanup runs on EVERY
    /// failure path, not just File::create failures.
    #[test]
    fn rename_failure_cleans_up_tmp_sibling() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().join("dump.failure-dump.json");
        let tagged = super::snapshot_tagged_path(&base, "rename_fail_test");
        // Pre-create the tagged path as a non-empty directory so
        // rename(tmp, tagged) fails. Empty-dir rename behavior
        // varies across filesystems (Linux ext4/xfs/btrfs accept
        // file-over-empty-dir on the source side via EISDIR; some
        // older kernels rejected it). A non-empty dir is a stable
        // rename failure across every supported FS.
        std::fs::create_dir_all(&tagged).expect("create blocker dir");
        std::fs::write(tagged.join("blocker"), b"").expect("populate dir");
        let summary_called = AtomicBool::new(false);
        let result = write_to_tagged_path(
            Some(&base),
            "rename_fail_test",
            "{\"k\":\"v\"}",
            || {
                summary_called.store(true, Ordering::Relaxed);
                String::from("rename failure summary")
            },
            "rename failure warn",
        );
        assert!(result.is_err(), "expected Err on rename failure");
        assert!(
            summary_called.load(Ordering::Relaxed),
            "stderr_summary closure MUST fire on rename failure"
        );
        let tmp = tagged.with_extension("json.tmp");
        assert!(
            !tmp.exists(),
            "tmp sibling must be cleaned up after rename failure"
        );
    }

    /// UTF-8 boundary at cap — payload contains a 4-byte UTF-8
    /// sequence (U+1F600 GRINNING FACE, encoded as `0xF0 0x9F 0x98
    /// 0x80`) straddling the 16 KiB `ONDEMAND_STDERR_DUMP_CAP`
    /// boundary so the naive `&s[..CAP]` would split mid-char and
    /// panic. The helper's stderr-fallback walks back to a safe
    /// UTF-8 boundary via `utf8_safe_truncate_len`. Pins the UTF-8
    /// char-boundary panic fix on the on-demand path: the payload-
    /// head eprintln! must not panic when truncation lands inside a
    /// multi-byte sequence.
    #[test]
    fn write_failure_handles_utf8_boundary_at_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file_in_the_way = dir.path().join("not_a_dir");
        std::fs::write(&file_in_the_way, b"").expect("create blocker file");
        let base = file_in_the_way.join("sub").join("dump.failure-dump.json");
        // Imports the production cap so a future bump (e.g. 16 KiB
        // → 32 KiB) updates the production behavior AND this test's
        // boundary construction in lockstep. Avoids the silent
        // drift where the test still exercises the old 16 KiB
        // boundary while production truncates at 32 KiB.
        let cap = super::ONDEMAND_STDERR_DUMP_CAP;
        let mut payload = String::with_capacity(cap + 8);
        // Fill to two bytes short of cap, then push a 4-byte UTF-8
        // sequence so bytes cap-2..cap+2 form U+1F600. Bytes cap
        // and cap+1 are mid-char — naive slicing at cap would
        // panic.
        for _ in 0..(cap - 2) {
            payload.push('a');
        }
        payload.push('\u{1F600}');
        let result = write_to_tagged_path(
            Some(&base),
            "utf8_test",
            &payload,
            || String::from("utf8 boundary test summary"),
            "utf8 boundary test warn",
        );
        // Helper must return Err (write failed) WITHOUT panicking.
        // A regression in utf8_safe_truncate_len or the
        // format_truncation_marker integration would surface here as
        // a panic from the payload-head eprintln! rather than a
        // clean Err return.
        assert!(result.is_err(), "expected Err on write failure path");
    }
}

#[cfg(target_arch = "x86_64")]
#[cfg(test)]
mod kvm_clock_save_semantics_tests {
    //! Codifies the take()-once + flags=0-on-restore invariant for
    //! the kvm_clock save/restore bridge between freeze_and_dispatch
    //! (writer) and thaw_and_barrier (reader). The contract: a
    //! freeze captures a kvm_clock_data into the shared RefCell at
    //! freeze entry; the matching thaw take()s the value, zeros
    //! flags, and SET_CLOCKs once. A second take() yields None
    //! (the skip-freeze sentinel — no save → no restore).
    //!
    //! A regression that lost the flags=0 reassignment would still
    //! type-check at the call site (kvm_clock_data is Copy); this
    //! test catches it.
    use kvm_bindings::kvm_clock_data;

    #[test]
    fn take_yields_value_once_then_none() {
        let save: std::cell::RefCell<Option<kvm_clock_data>> = std::cell::RefCell::new(None);
        let captured = kvm_clock_data {
            clock: 12345,
            flags: kvm_bindings::KVM_CLOCK_REALTIME,
            ..Default::default()
        };
        *save.borrow_mut() = Some(captured);

        let taken = save.borrow_mut().take();
        assert!(taken.is_some(), "first take must yield captured value");
        let mut clock = taken.unwrap();
        clock.flags = 0;
        assert_eq!(clock.flags, 0, "flags MUST be zeroed before SET_CLOCK");
        assert_eq!(clock.clock, 12345);

        assert!(
            save.borrow_mut().take().is_none(),
            "second take must yield None — freeze cycle is one-shot per pair"
        );
    }

    #[test]
    fn skip_freeze_path_yields_none_without_write() {
        let save: std::cell::RefCell<Option<kvm_clock_data>> = std::cell::RefCell::new(None);
        assert!(
            save.borrow_mut().take().is_none(),
            "skip-freeze path means no capture → take() must yield None"
        );
    }
}

#[cfg(test)]
mod compute_err_triggered_tests {
    //! Full truth table for [`compute_err_triggered`] — the run-loop's
    //! "fire this iteration" verdict from the `(watchpoint_hit,
    //! BssReadState)` pair. The matrix is the complete cross-product
    //! (2 watchpoint states × every [`BssReadState`] variant = 8 cells)
    //! so a regression that widens the bss arm (e.g. treating
    //! `OutOfBounds` or `NotResolved` as a fire) surfaces here as the
    //! corresponding `(false, …)` cell flipping from `false` to `true`.
    //!
    //! Semantic anchor (`mod.rs`): the verdict is
    //! `watchpoint_hit || matches!(bss_state, Triggered)`. The
    //! production binding at the run loop computes exactly this from the
    //! sticky watchpoint `hit` latch and the `bss_read_state(...)`
    //! result. Only `Triggered` counts on the bss path — `OutOfBounds`,
    //! `NotResolved`, and `NotTriggered` are all "no observable fire",
    //! so a stale cached PA after probe unload cannot synthesise a
    //! phantom fire from arbitrary DRAM bytes.
    use super::{BssReadState, compute_err_triggered};

    /// Every `BssReadState` variant, in declaration order. Drives the
    /// matrix tests so adding a 5th variant to the enum forces a
    /// compile-time decision here (the array literal must be extended)
    /// rather than silently leaving the new variant untested.
    const ALL_STATES: [BssReadState; 4] = [
        BssReadState::NotResolved,
        BssReadState::NotTriggered,
        BssReadState::Triggered,
        BssReadState::OutOfBounds,
    ];

    /// Watchpoint NOT hit: the bss path is the only signal. Only
    /// `Triggered` fires; the other three variants resolve to "no fire".
    /// Pins the bss-only arm of the disjunction — the fallback path used
    /// when the hardware watchpoint could not be armed.
    #[test]
    fn no_watchpoint_only_triggered_bss_fires() {
        assert!(
            !compute_err_triggered(false, BssReadState::NotResolved),
            "wp=false + NotResolved (pre-boot / cache unset) is no fire"
        );
        assert!(
            !compute_err_triggered(false, BssReadState::NotTriggered),
            "wp=false + NotTriggered (latch still 0) is no fire"
        );
        assert!(
            compute_err_triggered(false, BssReadState::Triggered),
            "wp=false + Triggered (bss latch flipped) is THE bss-fallback fire"
        );
        assert!(
            !compute_err_triggered(false, BssReadState::OutOfBounds),
            "wp=false + OutOfBounds (stale PA) must NOT fire — a recycled \
             vmalloc page must not synthesise a phantom fire"
        );
    }

    /// Watchpoint hit: the verdict is the primary path and short-circuits
    /// true regardless of the bss state. Pins that the watchpoint signal
    /// alone is sufficient — including when the bss latch reports
    /// `OutOfBounds` / `NotResolved` (the bss fallback degraded but the
    /// hardware watchpoint still caught the write).
    #[test]
    fn watchpoint_hit_fires_for_every_bss_state() {
        for st in ALL_STATES {
            assert!(
                compute_err_triggered(true, st),
                "wp=true must fire regardless of bss state {st:?}; the \
                 hardware watchpoint is the primary, sufficient signal"
            );
        }
    }

    /// Exhaustive 8-cell cross-product asserted against the independent
    /// reference `watchpoint_hit || st == Triggered`. The reference is
    /// computed WITHOUT `matches!` (direct `==`, which `BssReadState`
    /// supports via `PartialEq`) so it is not a copy of the production
    /// expression — a regression to either side is caught by divergence
    /// from the other.
    #[test]
    fn full_matrix_matches_disjunction_reference() {
        for wp in [false, true] {
            for st in ALL_STATES {
                let expected = wp || st == BssReadState::Triggered;
                assert_eq!(
                    compute_err_triggered(wp, st),
                    expected,
                    "cell (watchpoint_hit={wp}, bss={st:?}) must equal \
                     `wp || st == Triggered`"
                );
            }
        }
    }
}

#[cfg(test)]
mod compute_watchpoint_only_trigger_tests {
    //! Full truth table for [`compute_watchpoint_only_trigger`] — the
    //! predicate that selects the `gate_on_exit_kind` argument at the
    //! `freeze_and_dispatch` Capture call site. The matrix is the
    //! complete cross-product (2 watchpoint states × every
    //! [`BssReadState`] variant = 8 cells).
    //!
    //! Semantic anchor (`mod.rs`): the predicate is
    //! `watchpoint_hit && !matches!(bss_state, Triggered)`. True =
    //! "watchpoint fired but the bss latch did NOT confirm the
    //! error-class exit" → the dispatch GATES on `kind >= SCX_EXIT_ERROR`
    //! because the watchpoint catches every write to `*scx_root->
    //! exit_kind` (including the clean init/teardown NONE/DONE values
    //! that would synthesise a bogus dump without the gate). False on a
    //! bss-confirmed `Triggered` (the tp_btf handler already proved
    //! `kind >= SCX_EXIT_ERROR`, so the gate is redundant) and false
    //! whenever the watchpoint did not fire (there is no
    //! watchpoint-only event to gate).
    //!
    //! Relationship to [`super::compute_err_triggered`]: this is the
    //! AND-NOT counterpart of the OR. The two share the `bss ==
    //! Triggered` term but compose it oppositely, so a regression that
    //! collapsed one into the other would flip the `(true, Triggered)`
    //! cell — caught by `triggered_bss_is_never_watchpoint_only` below.
    use super::{BssReadState, compute_watchpoint_only_trigger};

    /// Every `BssReadState` variant, declaration order — same role as
    /// the sibling module's array: a new enum variant forces a
    /// compile-time extension here rather than a silent coverage gap.
    const ALL_STATES: [BssReadState; 4] = [
        BssReadState::NotResolved,
        BssReadState::NotTriggered,
        BssReadState::Triggered,
        BssReadState::OutOfBounds,
    ];

    /// Watchpoint NOT hit: there is no watchpoint event to gate, so the
    /// predicate is false for every bss state. Pins the left conjunct —
    /// a regression dropping the `watchpoint_hit &&` term would flip
    /// these (the bss-only-fired path must NOT request the exit_kind
    /// gate, because a bss `Triggered` already proved the error class).
    #[test]
    fn no_watchpoint_never_watchpoint_only() {
        for st in ALL_STATES {
            assert!(
                !compute_watchpoint_only_trigger(false, st),
                "wp=false has no watchpoint event to gate; bss state \
                 {st:?} is irrelevant — predicate must be false"
            );
        }
    }

    /// Watchpoint hit, bss NOT `Triggered`: the watchpoint-only case.
    /// True for the three non-`Triggered` states — the watchpoint caught
    /// a write the bss latch has not (yet) confirmed as error-class, so
    /// the dispatch must gate on `kind >= SCX_EXIT_ERROR` to suppress the
    /// clean init/teardown NONE/DONE writes.
    #[test]
    fn watchpoint_hit_without_bss_triggered_is_watchpoint_only() {
        assert!(
            compute_watchpoint_only_trigger(true, BssReadState::NotResolved),
            "wp=true + NotResolved is watchpoint-only → must gate on exit_kind"
        );
        assert!(
            compute_watchpoint_only_trigger(true, BssReadState::NotTriggered),
            "wp=true + NotTriggered is watchpoint-only → must gate on exit_kind"
        );
        assert!(
            compute_watchpoint_only_trigger(true, BssReadState::OutOfBounds),
            "wp=true + OutOfBounds is watchpoint-only (bss fallback stale) → \
             must gate on exit_kind"
        );
    }

    /// Watchpoint hit AND bss `Triggered`: NOT watchpoint-only. The bss
    /// latch independently confirmed the error class, so the dispatch
    /// skips the gate (it is redundant). This is the single cell that
    /// distinguishes this predicate from a plain `watchpoint_hit`
    /// pass-through — the right conjunct's `!Triggered` term.
    #[test]
    fn triggered_bss_is_never_watchpoint_only() {
        assert!(
            !compute_watchpoint_only_trigger(true, BssReadState::Triggered),
            "wp=true + Triggered: bss already proved kind >= SCX_EXIT_ERROR, \
             so the dispatch skips the exit_kind gate — NOT watchpoint-only"
        );
    }

    /// Exhaustive 8-cell cross-product asserted against the independent
    /// reference `watchpoint_hit && st != Triggered` (direct `!=`, not a
    /// `matches!` copy of the production expression). Divergence on any
    /// cell fails the test.
    #[test]
    fn full_matrix_matches_and_not_reference() {
        for wp in [false, true] {
            for st in ALL_STATES {
                let expected = wp && st != BssReadState::Triggered;
                assert_eq!(
                    compute_watchpoint_only_trigger(wp, st),
                    expected,
                    "cell (watchpoint_hit={wp}, bss={st:?}) must equal \
                     `wp && st != Triggered`"
                );
            }
        }
    }
}

/// Which source `collect_results` uses for `VmResult.verifier_stats`.
/// See [`choose_verifier_stats_source`].
#[derive(Debug, PartialEq, Eq)]
enum VerifierStatsSource {
    /// The mid-run capture, taken while the scheduler was attached and
    /// its BPF progs were live in `prog_idr`.
    MidRun,
    /// The post-teardown host walk. Used only when no mid-run capture
    /// exists (e.g. a boot so short the monitor never latched
    /// `data_valid`) but a BPF scheduler was attached. Races synchronous
    /// `idr_remove` at scheduler-process exit, hence the fallback role.
    PostTeardownWalk,
    /// No BPF scheduler was attached — empty `verifier_stats`.
    None,
}

/// Decide where `collect_results` sources `verifier_stats`. Prefer the
/// mid-run capture over the post-teardown walk: the walk runs after the
/// guest killed the scheduler process, which drops the last `bpf_prog`
/// reference and fires synchronous `bpf_prog_free_id` / `idr_remove`
/// (kernel/bpf/syscall.c) BEFORE the RCU-deferred prog free — so it
/// finds an empty `prog_idr` and loses `verified_insns`.
/// A non-empty mid-run capture (read while attached; `verified_insns` is
/// fixed once at load) is therefore authoritative. Fall back to the
/// walk only when there is no mid-run capture, and to empty when no BPF
/// scheduler attached at all.
fn choose_verifier_stats_source(
    mid_run_empty: bool,
    has_bpf_scheduler: bool,
) -> VerifierStatsSource {
    if !mid_run_empty {
        VerifierStatsSource::MidRun
    } else if has_bpf_scheduler {
        VerifierStatsSource::PostTeardownWalk
    } else {
        VerifierStatsSource::None
    }
}

#[cfg(test)]
mod verifier_stats_source_tests {
    use super::{VerifierStatsSource, choose_verifier_stats_source};

    /// The core decision: a non-empty mid-run capture is
    /// preferred over the post-teardown walk (which races `idr_remove`),
    /// even when a scheduler is attached; an empty capture falls back to
    /// the walk when a scheduler attached, else to empty. CI-runnable
    /// (no VM/kernel) — pairs the host-gated demo_verifier e2e.
    #[test]
    fn prefers_mid_run_capture_then_falls_back() {
        // Non-empty mid-run capture wins regardless of has_bpf_scheduler.
        assert_eq!(
            choose_verifier_stats_source(false, true),
            VerifierStatsSource::MidRun,
        );
        assert_eq!(
            choose_verifier_stats_source(false, false),
            VerifierStatsSource::MidRun,
        );
        // Empty capture + scheduler attached -> post-teardown walk.
        assert_eq!(
            choose_verifier_stats_source(true, true),
            VerifierStatsSource::PostTeardownWalk,
        );
        // Empty capture + no scheduler -> empty stats.
        assert_eq!(
            choose_verifier_stats_source(true, false),
            VerifierStatsSource::None,
        );
    }
}

#[cfg(test)]
mod run_vm_thread_guard_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A worker thread that spins until the guard signals `kill`, then records
    /// that it ran. A `Drop` that forgot to set `kill` would hang the join
    /// forever (test timeout); one that detached instead of joining would leave
    /// the `joined` count short.
    fn kill_watching_worker(kill: Arc<AtomicBool>, joined: Arc<AtomicUsize>) -> JoinHandle<()> {
        std::thread::spawn(move || {
            while !kill.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            joined.fetch_add(1, Ordering::Release);
        })
    }

    fn evt() -> Arc<EventFd> {
        Arc::new(EventFd::new(EFD_NONBLOCK).expect("eventfd"))
    }

    /// The core invariant: on the early-return/panic path, `Drop` signals
    /// shutdown (kill + kill_evt + bsp_done + freeze cleared) AND joins every
    /// spawned handle before returning — so no detached thread outlives the
    /// guard (and, in `run_vm`, `guest_mem`). Uses the non-vCPU handles
    /// (monitor/bpf/coord/watchdog); a real `VcpuThread` needs a live VcpuFd,
    /// so the AP path is covered by the shared `kick_and_join_ap_threads` +
    /// `collect_results`.
    ///
    /// This deterministically verifies the CURRENT (joining) `Drop`: `join()`
    /// establishes a happens-before edge from each worker's final
    /// `fetch_add` to the post-drop read, so `joined == 3` is guaranteed. As a
    /// guard against a hypothetical future detach regression it is strong but
    /// not 100% deterministic — a detached worker still observes the
    /// Drop-set `kill` and would eventually increment, so on a sufficiently
    /// fast machine it could reach the counter before the assert. A fully
    /// race-free detach-catch would require a timed wait, which the codebase
    /// forbids (event-driven only); the happens-before proof above is the
    /// strongest sleepless invariant available for a self-terminating worker.
    #[test]
    fn drop_signals_kill_and_joins_every_handle() {
        let kill = Arc::new(AtomicBool::new(false));
        let joined = Arc::new(AtomicUsize::new(0));
        let guard = RunVmThreadGuard {
            ap_threads: Vec::new(),
            monitor: None,
            bpf_write: Some(kill_watching_worker(kill.clone(), joined.clone())),
            freeze_coord: Some(kill_watching_worker(kill.clone(), joined.clone())),
            watchdog: Some(kill_watching_worker(kill.clone(), joined.clone())),
            kill: kill.clone(),
            kill_evt: evt(),
            freeze: Arc::new(AtomicBool::new(true)),
            bsp_done: Arc::new(AtomicBool::new(false)),
            bsp_done_evt: evt(),
        };
        drop(guard);
        assert!(
            kill.load(Ordering::Acquire),
            "Drop sets the kill flag so the workers can exit"
        );
        assert_eq!(
            joined.load(Ordering::Acquire),
            3,
            "Drop JOINS every handle (each worker bumps the counter before exit) \
             — a detach would leave the count below 3"
        );
    }

    /// After `disarm`, the guard is empty: `Drop` must be a no-op (NOT signal
    /// kill, NOT join), and the handles are handed back to the caller. Proves
    /// the Ok path reclaims the threads for the normal teardown instead of the
    /// guard joining them.
    #[test]
    fn disarm_hands_back_handles_and_drop_is_inert() {
        let kill = Arc::new(AtomicBool::new(false));
        let joined = Arc::new(AtomicUsize::new(0));
        // Worker exits on its OWN flag, not the guard's kill, so we can prove the
        // disarmed guard neither signals kill nor joins it.
        let go = Arc::new(AtomicBool::new(false));
        let go_worker = go.clone();
        let joined_worker = joined.clone();
        let h = std::thread::spawn(move || {
            while !go_worker.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            joined_worker.fetch_add(1, Ordering::Release);
        });
        let mut guard = RunVmThreadGuard {
            ap_threads: Vec::new(),
            monitor: None,
            bpf_write: Some(h),
            freeze_coord: None,
            watchdog: None,
            kill: kill.clone(),
            kill_evt: evt(),
            freeze: Arc::new(AtomicBool::new(false)),
            bsp_done: Arc::new(AtomicBool::new(false)),
            bsp_done_evt: evt(),
        };
        let handles = guard.disarm();
        drop(guard); // disarmed -> inert
        assert!(
            !kill.load(Ordering::Acquire),
            "a disarmed guard's Drop does not signal kill"
        );
        assert!(handles.bpf_write.is_some(), "disarm returns the handle");
        // Release + join the reclaimed handle ourselves (the Ok-path role).
        go.store(true, Ordering::Release);
        let _ = handles.bpf_write.unwrap().join();
        assert_eq!(joined.load(Ordering::Acquire), 1);
    }
}
