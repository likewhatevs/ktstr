//! [`SnapshotBridge`] is the request/reply channel between the
//! scenario executor and the host capture pipeline. Implements
//! callbacks ([`CaptureCallback`], [`WatchRegisterCallback`]), the
//! per-thread bridge installation guard ([`BridgeGuard`]), the
//! diagnostic event log ([`SnapshotBridgeEvent`]), and the
//! storage caps ([`MAX_STORED_SNAPSHOTS`], [`MAX_STORED_EVENTS`],
//! [`MAX_WATCH_SNAPSHOTS`]).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::monitor::dump::FailureDumpReport;
use crate::sync::MutexExt;

// ---------------------------------------------------------------------------
// Bridge: request/reply channel between executor and host capture
// ---------------------------------------------------------------------------

/// Closure type the bridge invokes to capture a snapshot.
///
/// Returns `None` when the capture pipeline could not produce a
/// report (rendezvous timed out, capture prerequisites missing, no
/// host-side wiring).
///
/// **Wire shape (locked: ioeventfd doorbell).** The production
/// implementation writes the tag into a small per-call slot inside
/// the SHM region, performs an `mmap`'d `u32` write to the
/// doorbell GPA inside the MMIO gap (KVM dispatches via
/// `KVM_IOEVENTFD` without a userspace exit), then blocks on a
/// per-request reply completion (an eventfd / mpsc receiver paired
/// with the doorbell registration). The freeze coordinator's
/// epoll loop wakes on the doorbell eventfd, reads the tag, runs
/// `freeze_and_dispatch`, and signals the reply completion with
/// the resulting `Option<FailureDumpReport>`.
///
/// On-demand captures are orthogonal to the error-trigger
/// `freeze_state` machine — the request handler in the coordinator
/// must not transition `freeze_state` from Idle, and must service
/// requests even when `freeze_state == Done`. The
/// rendezvous-serialisation invariant is the only constraint: each
/// request waits for `all parked == false` from the previous
/// capture before issuing.
pub type CaptureCallback = Arc<dyn Fn(&str) -> Option<FailureDumpReport> + Send + Sync + 'static>;

/// Closure type the bridge invokes to register a hardware-watchpoint
/// snapshot.
///
/// This callback is the host-side unit-testing seam — it lets
/// in-process executor tests record the symbol and return without
/// arming any hardware. In a booted VM the bridge's
/// `register_watch` is **not** installed; the in-guest
/// `Op::WatchSnapshot` arm rings an SHM doorbell and the host's
/// freeze coordinator runs `arm_user_watchpoint`
/// (`src/vmm/freeze_coord/mod.rs`), which resolves the symbol via a
/// verbatim match against the vmlinux ELF symtab, allocates a
/// free user watchpoint slot (3 user slots are available; slot 0
/// is reserved for the existing `*scx_root->exit_kind` trigger),
/// and arms the hardware watchpoint via `KVM_SET_GUEST_DEBUG`.
///
/// Once armed, the capture tagged with the symbol fires on every
/// guest write without any further userspace round-trip — the
/// debug exit dispatches into the freeze coordinator directly,
/// mirroring the existing reserved-slot path the error-class
/// trigger already uses.
///
/// Returns `Err(reason)` when:
///   - The symbol does not match any vmlinux ELF symtab entry
///     (typo, symbol stripped from the build, or a non-ELF kernel
///     image).
///   - The resolved KVA is not 4-byte aligned (the 4-byte watch
///     length the framework arms requires `addr & 0x3 == 0` on
///     every supported architecture).
///   - All three available user watchpoint slots are already
///     allocated.
///   - `KVM_SET_GUEST_DEBUG` rejected the arm (host kernel
///     limitation).
pub type WatchRegisterCallback =
    Arc<dyn Fn(&str) -> std::result::Result<(), String> + Send + Sync + 'static>;

/// Closure type the bridge invokes for a host-side kernel-memory
/// write or read (`Op::WriteKernel{Hot,Cold}` /
/// `Op::ReadKernel{Hot,Cold}`).
///
/// The host dispatches the request (a sequence of
/// `(KernelTarget, KernelValue)` entries, plus mode/direction/tag
/// metadata in `crate::vmm::wire::KernelOpRequestPayload`) against
/// the resolved guest-memory accessor. Returns
/// `crate::vmm::wire::KernelOpReplyPayload` mirroring the request
/// id, the success/error status, and (for reads) the read-back
/// bytes per entry.
///
/// Test fixtures install a closure that records the request and
/// returns a synthetic reply without touching real guest memory
/// (the in-process bridge surface stays mockable). The in-VM
/// production path goes through the wire layer
/// (`crate::vmm::wire::MsgType::KernelOpRequest`) and the freeze
/// coordinator / hot-path worker, NOT this callback — the bridge
/// keeps it Option<…> so executor tests can install one while real
/// VM runs leave it unset.
pub type KernelOpCallback = Arc<
    dyn Fn(&crate::vmm::wire::KernelOpRequestPayload) -> crate::vmm::wire::KernelOpReplyPayload
        + Send
        + Sync
        + 'static,
>;

/// Shared state owning the capture closure plus the captured-report
/// map.
///
/// Cloneable via the wrapped `Arc`s. The host installs an instance
/// in the executor's thread-local via `Self::set_thread_local`
/// before [`execute_steps`](crate::scenario::ops::execute_steps)
/// runs; the executor's `Op::CaptureSnapshot` arm calls
/// `Self::capture` with the op's name.
/// Maximum number of [`Op::WatchSnapshot`](crate::scenario::ops::Op::WatchSnapshot)
/// ops a single scenario may register.
///
/// This is the framework's per-scenario cap on user watchpoint slots
/// across every supported host architecture, not a count of debug
/// registers on any specific arch. One additional slot (slot 0) is
/// always reserved internally for the `*scx_root->exit_kind`
/// watchpoint that drives the error-class freeze trigger, so a host
/// must expose at least 4 hardware watchpoint slots through
/// `KVM_SET_GUEST_DEBUG` for every user `Op::WatchSnapshot` to arm.
/// Common x86_64 and aarch64 hosts meet that bar.
///
/// The actual host slot count is probed once during VM bring-up via
/// `KVM_CHECK_EXTENSION(KVM_CAP_GUEST_DEBUG_HW_WPS)` in
/// `crate::vmm::freeze_coord` (search for `Cap::DebugHwWps`); a
/// host returning `<= 0` or fewer than 4 slots logs a `tracing::warn!`
/// at coordinator setup. Per-arm failures surface as `tracing::warn!`
/// from `self_arm_watchpoint` with per-vCPU retry capping at
/// `WATCHPOINT_MAX_NON_EINTR_FAILURES`.
pub const MAX_WATCH_SNAPSHOTS: usize = 3;

/// Maximum number of [`FailureDumpReport`]s the bridge keeps. Captures
/// driven by a Loop step with a unique tag per iteration would
/// otherwise grow the storage map without bound — every report
/// renders a full BTF tree (potentially hundreds of KB), so an
/// uncapped bridge under hostile/runaway capture frequency drains
/// host memory. The bridge enforces FIFO eviction at this cap so the
/// most recent captures stay reachable; eviction logs a `tracing::warn!`
/// naming the dropped tag so the operator sees the truncation.
pub const MAX_STORED_SNAPSHOTS: usize = 64;

/// Maximum number of [`SnapshotBridgeEvent`] entries the bridge
/// retains between [`SnapshotBridge::drain_events`] calls. A scenario
/// that triggers many cap-eviction events (a Loop step that captures
/// a unique tag every 30ms for 10 minutes produces ~20 000 events,
/// each ~100 bytes) would otherwise grow the events log without
/// bound. The bridge enforces FIFO eviction at this cap — when push
/// would exceed it, the oldest event is dropped, the dropped count
/// is tracked on `SnapshotStore::events_dropped`, and the next
/// [`SnapshotBridge::drain_events`] call appends a synthetic
/// [`SnapshotBridgeEvent::EventLogTruncated`] entry at the tail so
/// the operator never silently loses events. The cap is loose enough
/// (1024 events × ~100 bytes ≈ 100 KiB) that legitimate scenarios
/// never hit it; only runaway capture frequency does.
pub const MAX_STORED_EVENTS: usize = 1024;

/// A structured event surfaced by the [`SnapshotBridge`] during its
/// own operation (capture, storage, drain). Promotes the previous
/// `tracing::warn!`-only diagnostic channel into an operator-
/// drainable structured row so tests can assert on bridge-side
/// conditions (eviction, missing capture, invariant violations)
/// instead of grepping stderr.
///
/// Distinct from [`crate::assert::AssertDetail`]: an `AssertDetail`
/// is a per-assertion outcome (Starved / Stuck / etc.); a
/// `SnapshotBridgeEvent` is a per-bridge meta-event about the
/// storage pipeline itself. Mixing them at the assertion level
/// would conflate "scheduler behavior failed" with "bridge dropped
/// an entry due to cap" — two orthogonal concerns. Test authors
/// who want to fail their scenario on a bridge event compose the
/// two streams themselves (drain events, convert to `AssertDetail`
/// if needed) — see [`SnapshotBridge::drain_events`].
///
/// Every bridge site that previously emitted only `tracing::warn!`
/// still emits the warn (preserved for stderr visibility) AND
/// appends the structured variant here. "Promote, don't replace."
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum SnapshotBridgeEvent {
    /// Capture callback returned `None` for `tag` — the corresponding
    /// `Op::CaptureSnapshot` was a no-op. Fires from
    /// [`SnapshotBridge::capture`] when the host couldn't freeze /
    /// build the report (scheduler died before the freeze, scan
    /// accessor unavailable, etc.).
    CaptureUnavailable {
        /// Tag the failed capture was attempted under.
        tag: String,
    },
    /// Storage of `tag` overwrote a prior entry. Fires from
    /// [`SnapshotBridge::store`] / [`SnapshotBridge::store_with_stats`]
    /// when `bridge.store(tag, ...)` is called with a tag that
    /// already has a stored report. FIFO order is refreshed to back,
    /// prior `(stats, elapsed_ms)` parallel slots are replaced.
    Overwrite {
        /// Tag whose prior entry was overwritten.
        tag: String,
        /// `schema` of the prior entry — included for diagnostic
        /// context (a schema bump alongside an unintended overwrite
        /// is the textbook double-tag bug).
        prior_schema: String,
    },
    /// FIFO eviction of `evicted_tag` triggered by storing
    /// `new_tag`. Fires from the cap-enforcement loop in
    /// `store_internal` when `reports.len()` exceeds
    /// [`MAX_STORED_SNAPSHOTS`] after insertion. `cap` is the limit
    /// at the time of eviction.
    Eviction {
        /// Tag that was popped from the FIFO to make room.
        evicted_tag: String,
        /// Tag whose storage triggered the cap-overflow.
        new_tag: String,
        /// Cap value at the time — folded in so the operator
        /// doesn't have to cross-reference [`MAX_STORED_SNAPSHOTS`].
        cap: usize,
    },
    /// A drain found `tag` in `reports` but missing from `order` —
    /// internal invariant violation. The report was surfaced at the
    /// tail of the drain output rather than dropped silently; this
    /// event flags the bug so test authors who care can fail their
    /// scenario.
    DrainOrderingInvariantViolation {
        /// Tag whose desynchronised entry was surfaced at the tail.
        tag: String,
        /// Which drain variant fired the warning —
        /// `"drain_ordered"` or `"drain_ordered_with_stats"`. Lets
        /// post-mortem analysis disambiguate the two code paths.
        drain_variant: &'static str,
    },
    /// The cap-enforcement loop in `store_internal` found
    /// `reports.len() > cap` while `order` was empty — a worse
    /// invariant violation than [`Self::DrainOrderingInvariantViolation`]
    /// because the bulk-clear branch nukes ALL reports / stats /
    /// elapsed_ms to restore the invariant. Unreachable through the
    /// current public API (every insert site appends to `order`
    /// alongside `reports`), but recorded for the same future-proofing
    /// reason as the drain variant: a refactor that desynchronised
    /// the two collections must not be allowed to silently drop the
    /// entire bridge state.
    CapInvariantViolation {
        /// `reports.len()` at the moment the bulk-clear was triggered.
        /// Folded in so the operator can see how much state was
        /// nuked.
        reports_len: usize,
        /// Cap value at the time — same definition as
        /// [`Self::Eviction::cap`].
        cap: usize,
    },
    /// The events log itself hit [`MAX_STORED_EVENTS`] and dropped
    /// `dropped_count` oldest events to keep memory bounded. The
    /// bridge appends this variant at the tail of every
    /// [`SnapshotBridge::drain_events`] result whenever
    /// `events_dropped > 0` (resets to 0 after drain), so the
    /// operator never silently loses events — they see a count of
    /// how many were dropped between drains. Test authors who care
    /// about exhaustive coverage should `assert!(!matches!(events
    /// .last(), Some(SnapshotBridgeEvent::EventLogTruncated { .. })))`
    /// to fail when the bridge truncated.
    EventLogTruncated {
        /// Number of events evicted from the front of the log since
        /// the last [`SnapshotBridge::drain_events`] call. Resets to
        /// 0 after drain.
        dropped_count: u64,
    },
}

/// Inner storage for [`SnapshotBridge::snapshots`]. Pairs the
/// HashMap-keyed reports with a [`VecDeque`] tracking insertion
/// order so the FIFO eviction in [`SnapshotBridge::store`] can pop
/// the oldest tag in O(1) when the cap is reached. The optional
/// `stats` map carries the scheduler-stats JSON captured at the
/// same boundary as the snapshot — only periodic captures populate
/// this; on-demand and watchpoint captures leave the slot empty
/// because no stats request is issued.
pub(super) struct SnapshotStore {
    pub(super) reports: HashMap<String, FailureDumpReport>,
    /// scx_stats JSON captured at the same wall-clock as the report
    /// stored under the same tag in `reports`. Periodic captures
    /// populate this when a stats client is wired and the request
    /// succeeds; on-demand / watchpoint paths leave the entry
    /// absent. Sample::stats reads `stats.get(tag)` — `None` is the
    /// expected shape for non-periodic tags or when the scheduler
    /// stats request failed.
    pub(super) stats: HashMap<String, Result<serde_json::Value, super::error::MissingStatsReason>>,
    /// Elapsed milliseconds since `run_start` at the moment the
    /// periodic capture fired. Same key set as `reports` for
    /// periodic tags; absent for non-periodic captures. Read by
    /// [`SnapshotBridge::drain_ordered_with_stats`] to populate
    /// `Sample::elapsed_ms` without recomputing.
    pub(super) elapsed_ms: HashMap<String, u64>,
    /// Workload-relative boundary OFFSET (periodic `boundary_ns -
    /// scenario_anchor_ns`, in ms) this periodic capture was
    /// SCHEDULED for — distinct from `elapsed_ms`, which is the
    /// run_start-relative FIRE time (~uniform across the deferred-fire
    /// burst, so useless for per-phase attribution). Same key set as
    /// `reports` for periodic tags; absent for non-periodic / on-demand
    /// captures. Drained into
    /// [`super::error::DrainedSnapshotEntry::boundary_offset_ms`] and
    /// consumed by `build_phase_buckets[_with_stimulus]` to attribute
    /// the capture to the guest step whose stimulus window contains
    /// this offset, and as the workload-relative bucket start/end.
    pub(super) boundary_offset_ms: HashMap<String, u64>,
    /// Per-VM scenario step index stamped on the capture by the
    /// step-aware entry points
    /// ([`SnapshotBridge::capture_with_step`] /
    /// [`SnapshotBridge::store_with_stats_and_step`]). Absent for
    /// fixture-injected captures stored via the unstamped legacy
    /// paths ([`SnapshotBridge::capture`] / [`SnapshotBridge::store`]
    /// / [`SnapshotBridge::store_with_stats`]). Encoded per the
    /// 1-indexed phase convention (`0` = BASELINE, `1..=N` = Step
    /// ordinals); drained in lock-step with `reports` / `stats` /
    /// `elapsed_ms` and surfaced as the
    /// [`super::error::DrainedSnapshotEntry::step_index`] field so
    /// the phase-aware aggregator can bucket each sample directly.
    pub(super) step_index: HashMap<String, u16>,
    /// Insertion order of currently-resident keys. An overwrite of
    /// an existing key MUST remove the prior entry from this deque
    /// before pushing the fresh occurrence so the `reports.len()`
    /// and `order.len()` invariants stay in lock-step.
    pub(super) order: VecDeque<String>,
    /// Structured bridge-side meta-events appended in insertion
    /// order. Every site that previously emitted only a
    /// `tracing::warn!` also pushes the corresponding
    /// [`SnapshotBridgeEvent`] variant here. Drained by
    /// [`SnapshotBridge::drain_events`] so test authors can assert
    /// on bridge meta-conditions (eviction, overwrite, missing
    /// capture, invariant violation) without grepping stderr.
    /// Capped at [`MAX_STORED_EVENTS`] via FIFO eviction in
    /// `push_event`; dropped count is tracked in `events_dropped`
    /// and surfaced as a synthetic
    /// [`SnapshotBridgeEvent::EventLogTruncated`] appended at the
    /// tail of the next `drain_events` result so no event loss is
    /// silent.
    events: Vec<SnapshotBridgeEvent>,
    /// Number of events evicted from the front of `events` since
    /// the last `drain_events` call. Reset to 0 on drain.
    /// Drain appends [`SnapshotBridgeEvent::EventLogTruncated`] at
    /// the tail when this is non-zero so the operator never silently
    /// loses events — they always see a marker carrying the dropped
    /// count.
    events_dropped: u64,
}

impl SnapshotStore {
    fn new() -> Self {
        Self {
            reports: HashMap::new(),
            stats: HashMap::new(),
            elapsed_ms: HashMap::new(),
            boundary_offset_ms: HashMap::new(),
            step_index: HashMap::new(),
            order: VecDeque::new(),
            events: Vec::new(),
            events_dropped: 0,
        }
    }

    /// Append `event` to `events`, enforcing [`MAX_STORED_EVENTS`]
    /// via FIFO eviction. When push would exceed the cap, the
    /// oldest entry is removed and `events_dropped` is incremented
    /// so a subsequent [`SnapshotBridge::drain_events`] call can
    /// surface a [`SnapshotBridgeEvent::EventLogTruncated`] marker
    /// — the operator never silently loses events. The fast path
    /// (cap not reached) is a single push with no extra allocation.
    fn push_event(&mut self, event: SnapshotBridgeEvent) {
        if self.events.len() >= MAX_STORED_EVENTS {
            // Drop the oldest. Vec::remove(0) is O(n) but the cap
            // is bounded and this branch only fires in pathological
            // runaway-capture scenarios.
            self.events.remove(0);
            self.events_dropped = self.events_dropped.saturating_add(1);
        }
        self.events.push(event);
    }
}

/// RAII guard for a reserved [`SnapshotBridge::watch_count`] slot.
///
/// [`SnapshotBridge::register_watch`] reserves a slot via CAS BEFORE
/// calling the host's watch-register callback so concurrent callers
/// cannot push the count past [`MAX_WATCH_SNAPSHOTS`] even
/// transiently. If the callback panics (rather than returning Err),
/// the prior manual-fetch_sub rollback never ran — the slot would
/// leak permanently and every future `register_watch` call would hit
/// the cap with no real watchpoints armed. This guard releases the
/// reservation on every exit path (Err-return AND unwind); the
/// success path commits the slot via `mem::forget`.
struct WatchSlotGuard<'a> {
    count: &'a std::sync::atomic::AtomicUsize,
}

impl Drop for WatchSlotGuard<'_> {
    fn drop(&mut self) {
        self.count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// A single `cgroup.procs` snapshot captured by
/// [`Op::CaptureCgroupProcs`](crate::scenario::ops::Op::CaptureCgroupProcs).
///
/// Tests drain the per-bridge log of these via
/// [`SnapshotBridge::drain_cgroup_procs`] after the scenario completes,
/// then assert on the captured `pids` (membership, count, identity)
/// against the workload they spawned. Carries:
///
/// - `tag`: the snapshot key supplied in the Op (lets a scenario
///   capture pre/post snapshots of the same cgroup at different
///   points and disambiguate them on drain).
/// - `cgroup`: the cgroup name the kernel read returned PIDs for.
/// - `pids`: the thread-group leaders (PIDs / TGIDs) the kernel
///   reported in `cgroup.procs` at capture time. An empty Vec means
///   the cgroup existed but held no tasks. A read against a missing
///   cgroup directory errors at the dispatch arm and never produces
///   a snapshot — so the presence of a `CgroupProcsSnapshot` in the
///   drain log implies the read succeeded.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct CgroupProcsSnapshot {
    /// Snapshot key supplied to
    /// [`Op::CaptureCgroupProcs`](crate::scenario::ops::Op::CaptureCgroupProcs).
    /// Distinguishes multiple captures of the same cgroup.
    pub tag: String,
    /// Cgroup name the kernel read returned PIDs for.
    pub cgroup: String,
    /// Thread-group leaders in the cgroup at capture time.
    /// `libc::pid_t` is `i32` on Linux; compare against other ktstr
    /// pid surfaces (`WorkloadHandle::worker_pids()`, etc.) via the
    /// matching integer type or `as i32` cast when types differ.
    pub pids: Vec<libc::pid_t>,
}

impl CgroupProcsSnapshot {
    /// Construct a `CgroupProcsSnapshot`. The struct is
    /// `#[non_exhaustive]` so external callers cannot use
    /// struct-literal syntax — use this constructor instead. The
    /// framework's [`SnapshotBridge::record_cgroup_procs`] dispatch
    /// site routes through here so test fixtures that synthesise
    /// snapshots for drain-consumer testing share the same
    /// construction path.
    pub fn new(tag: impl Into<String>, cgroup: impl Into<String>, pids: Vec<libc::pid_t>) -> Self {
        Self {
            tag: tag.into(),
            cgroup: cgroup.into(),
            pids,
        }
    }
}

/// Host-side capture pipeline that the freeze coordinator routes
/// [`Op::CaptureSnapshot`](crate::scenario::ops::Op::CaptureSnapshot) and
/// [`Op::WatchSnapshot`](crate::scenario::ops::Op::WatchSnapshot)
/// requests through.
///
/// Construct via [`SnapshotBridge::new`] (with an explicit capture
/// callback) and optionally [`SnapshotBridge::with_watch_register`]
/// to attach watch support. Install for the current thread via
/// [`SnapshotBridge::set_thread_local`] — see [`BridgeGuard`] for
/// the RAII teardown contract.
#[derive(Clone)]
#[must_use = "dropping a SnapshotBridge discards the capture pipeline"]
pub struct SnapshotBridge {
    capture: CaptureCallback,
    register_watch: Option<WatchRegisterCallback>,
    kernel_op: Option<KernelOpCallback>,
    pub(super) snapshots: Arc<Mutex<SnapshotStore>>,
    /// Per-tag drain log of kernel-op reply payloads. Test fixtures
    /// inspect this via [`SnapshotBridge::drain_kernel_ops`] after
    /// `execute_steps` returns to verify each request's outcome.
    kernel_ops: Arc<Mutex<Vec<(String, crate::vmm::wire::KernelOpReplyPayload)>>>,
    /// Per-tag drain log of cgroup.procs snapshots captured by
    /// [`Op::CaptureCgroupProcs`](crate::scenario::ops::Op::CaptureCgroupProcs).
    /// Stored in insertion order so a scenario capturing the same
    /// cgroup at multiple points (with distinct tags) sees both
    /// snapshots when draining. Test fixtures drain via
    /// [`SnapshotBridge::drain_cgroup_procs`] after the scenario
    /// completes; each entry carries the tag, the cgroup name read,
    /// and the PIDs the kernel reported in `cgroup.procs`.
    cgroup_procs: Arc<Mutex<Vec<CgroupProcsSnapshot>>>,
    watch_count: Arc<std::sync::atomic::AtomicUsize>,
    /// Monotonic counter bumped by the freeze-coordinator's accessor-
    /// init worker on every successful slot publish (initial attach +
    /// each subsequent re-init triggered by scheduler swap). Paired
    /// with [`Self::accessor_worker_state`] so a scheduler-swap op can
    /// wait for the new scheduler's BPF maps to land before returning
    /// success. `None` for bridges built without an accessor (test
    /// fixtures that never trigger reinit); see
    /// [`Self::with_accessor_state`].
    accessor_publish_seqno: Option<Arc<std::sync::atomic::AtomicU64>>,
    /// Liveness sentinel for the accessor-init worker — set by the
    /// worker as it transitions through its retry / publish / exit
    /// states. Values match [`accessor_worker_state`] module constants
    /// (Trying / Succeeded / FailedPermanently). The dispatcher in
    /// `Op::ReplaceScheduler` / `Op::AttachScheduler` reads this on
    /// timeout so the surfaced error distinguishes "worker still
    /// trying — bump deadline" from "worker exited — retry will
    /// hang." `None` mirrors `accessor_publish_seqno`.
    accessor_worker_state: Option<Arc<std::sync::atomic::AtomicU8>>,
    /// Dispatcher wake EventFd. The accessor-init worker pulses
    /// this fd in lock-step with every seqno bump AND on every
    /// terminal worker_state transition (FAILED_PERMANENTLY).
    /// Wait paths in [`Self::wait_for_accessor_publish_advance`] /
    /// [`Self::wait_for_worker_state_not_trying`] `poll(2)` on the
    /// fd with the remaining deadline so the wake latency is one
    /// kernel scheduling tick instead of the 50 ms sleep tail an
    /// atomic-only loop would carry. Distinct from `accessor_ready_evt`
    /// (which the coord epoll drains) so the dispatcher and the
    /// coord don't race for the same wake count.
    ///
    /// **Single-consumer assumption.** The fd is read (drained) only
    /// from the wait paths; the bridge is installed thread-local via
    /// [`Self::set_thread_local`], and each freeze-coordinator
    /// instance constructs one bridge → one wake fd. Multiple
    /// concurrent dispatchers sharing the same fd could race for the
    /// drain edge (the atomic re-check at the top of each wait loop
    /// keeps that benign — atomic is source of truth — but a missed
    /// wake adds latency). Per-VM bridge ownership today guarantees
    /// no such sharing. `None` mirrors the other accessor-state fields.
    accessor_dispatcher_wake_evt: Option<Arc<vmm_sys_util::eventfd::EventFd>>,
}

/// State codes for the accessor-init worker, written to the
/// `SnapshotBridge::accessor_worker_state` atomic. Two-bit
/// encoding leaves the high bits open for future sub-states (e.g.
/// "shutting down").
pub mod accessor_worker_state {
    /// Worker is in its retry loop trying to initialize the BPF
    /// accessor pair. Both initial-attach and post-swap re-init
    /// path through this state before reaching Succeeded.
    pub const TRYING: u8 = 0;
    /// Worker has published at least one accessor pair successfully.
    /// Subsequent re-inits also publish via this transition (the
    /// state stays SUCCEEDED while the seqno bumps).
    pub const SUCCEEDED: u8 = 1;
    /// Worker has exited and will not publish again — ELF parse
    /// failure, the 60 s boot-budget deadline, or some other
    /// terminal condition the worker detected. A dispatcher seeing
    /// this on a timeout knows to surface "worker exited" rather
    /// than "still trying."
    pub const FAILED_PERMANENTLY: u8 = 2;
}

impl std::fmt::Debug for SnapshotBridge {
    /// Debug print does NOT show captured reports (their full
    /// rendering can be hundreds of KB) — only the count and the
    /// presence of callbacks.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnapshotBridge")
            .field("snapshots", &self.len())
            .field("watch_count", &self.watch_count())
            .field("capture", &"<callback>")
            .field(
                "register_watch",
                &if self.register_watch.is_some() {
                    "<callback>"
                } else {
                    "<none>"
                },
            )
            .finish()
    }
}

/// Match the periodic dispatch loop's tag format exactly:
/// `"periodic_"` + 3 ASCII digits. Source of truth is the
/// coordinator's `format!("periodic_{:03}", idx)` emission at
/// `src/vmm/freeze_coord/state.rs`. Distinct from
/// `tag.starts_with("periodic_")` (which would accept arbitrary
/// user `Op::CaptureSnapshot` tags whose names happen to share
/// that prefix and pollute the `periodic_real_count` floor).
fn is_periodic_tag(tag: &str) -> bool {
    match tag.strip_prefix("periodic_") {
        Some(rest) => rest.len() == 3 && rest.bytes().all(|b| b.is_ascii_digit()),
        None => false,
    }
}

impl SnapshotBridge {
    /// Build a bridge from a capture callback. The callback may
    /// freeze the VM, build the report, or return `None` when
    /// capture is unavailable. No watch-register callback —
    /// `Op::WatchSnapshot` returns "not supported" when the host
    /// did not wire one. Use [`Self::with_watch_register`] to
    /// install one.
    pub fn new(capture: CaptureCallback) -> Self {
        Self {
            capture,
            register_watch: None,
            kernel_op: None,
            snapshots: Arc::new(Mutex::new(SnapshotStore::new())),
            kernel_ops: Arc::new(Mutex::new(Vec::new())),
            cgroup_procs: Arc::new(Mutex::new(Vec::new())),
            watch_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            accessor_publish_seqno: None,
            accessor_worker_state: None,
            accessor_dispatcher_wake_evt: None,
        }
    }

    /// Install the accessor-init worker's publish-seqno, worker-state,
    /// and dispatcher-wake EventFd so
    /// [`Self::wait_for_accessor_publish_advance`] /
    /// [`Self::wait_for_worker_state_not_trying`] can block scheduler-
    /// swap ops until the new scheduler's BPF maps land — with kernel-
    /// scheduling-tick wake latency rather than the 50 ms sleep tail
    /// an atomic-only loop would carry. Called by the freeze-coordinator
    /// when it sets up the worker (`vmm-accessor-init` thread); test
    /// bridges that don't drive a real worker can omit this call and
    /// the wait becomes a no-op that returns `Ok(0)` immediately.
    pub fn with_accessor_state(
        mut self,
        publish_seqno: Arc<std::sync::atomic::AtomicU64>,
        worker_state: Arc<std::sync::atomic::AtomicU8>,
        dispatcher_wake_evt: Arc<vmm_sys_util::eventfd::EventFd>,
    ) -> Self {
        self.accessor_publish_seqno = Some(publish_seqno);
        self.accessor_worker_state = Some(worker_state);
        self.accessor_dispatcher_wake_evt = Some(dispatcher_wake_evt);
        self
    }

    /// Snapshot of the accessor-init worker's current publish seqno.
    /// Returns 0 for bridges built without an accessor (test fixtures
    /// — the dispatch wait sees no advance to gate on and the op
    /// returns immediately). Read with `Acquire` so the seqno orders
    /// against the worker's `Release` bump.
    pub fn accessor_publish_seqno(&self) -> u64 {
        self.accessor_publish_seqno
            .as_ref()
            .map(|s| s.load(std::sync::atomic::Ordering::Acquire))
            .unwrap_or(0)
    }

    /// Block until the accessor-init worker exits the TRYING state
    /// (transitions to SUCCEEDED on its first publish, or to
    /// FAILED_PERMANENTLY on a terminal worker exit). Used by
    /// `Op::AttachScheduler` to serialize against the worker's
    /// concurrent boot-publish: capturing the seqno baseline AFTER
    /// the boot publish completes is the only way to ensure the
    /// next observed seqno advance belongs to the just-attached
    /// scheduler rather than to a co-resident boot scheduler that
    /// finished its 60 s init in parallel. Event-driven via
    /// `accessor_dispatcher_wake_evt` — wake latency is one
    /// kernel scheduling tick, not a 50 ms poll. No-op (Ok(())) for
    /// bridges without an accessor.
    pub fn wait_for_worker_state_not_trying(
        &self,
        deadline: std::time::Instant,
        op_label: &str,
    ) -> anyhow::Result<()> {
        let Some(state) = self.accessor_worker_state.as_ref() else {
            return Ok(());
        };
        loop {
            let cur = state.load(std::sync::atomic::Ordering::Acquire);
            if cur != accessor_worker_state::TRYING {
                if cur == accessor_worker_state::FAILED_PERMANENTLY {
                    anyhow::bail!(
                        "{op_label}: accessor-init worker exited \
                         (FAILED_PERMANENTLY) before this op could run — \
                         check freeze-coord 'vmm-accessor-init' logs for ELF \
                         parse or boot-budget failures"
                    );
                }
                return Ok(());
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                anyhow::bail!(
                    "{op_label}: accessor-init worker stayed in TRYING state \
                     past the deadline — the worker's first publish has not \
                     completed. Likely cause: kernel boot stalled before \
                     accessor's bootstrap symbols became readable. Check \
                     freeze-coord logs for `accessor-init:` lines"
                );
            }
            let remaining = deadline.saturating_duration_since(now);
            self.poll_dispatcher_wake(remaining);
            // Loop back to re-check state. The atomic is the source
            // of truth — the eventfd is just a fast wake; a spurious
            // wake (e.g. a publish landing while we're already past
            // TRYING) drops through to the next state check.
        }
    }

    /// Block on the dispatcher-wake EventFd for up to `remaining`.
    /// Caller's atomic re-check after return is the source of truth.
    /// Panics if the with_accessor_state coupling invariant is
    /// violated (see body's unreachable! arm).
    ///
    /// Uses raw libc::poll rather than nix::poll::poll because the
    /// underlying [`vmm_sys_util::eventfd::EventFd`] doesn't impl
    /// `AsFd`, only `AsRawFd`; the cleaner nix path would require
    /// an unsafe `BorrowedFd::borrow_raw` wrapper, leaving the
    /// unsafe surface identical to libc::poll. No net reduction —
    /// keep the simpler libc call.
    fn poll_dispatcher_wake(&self, remaining: std::time::Duration) {
        if remaining.is_zero() {
            return;
        }
        let Some(evt) = self.accessor_dispatcher_wake_evt.as_ref() else {
            unreachable!(
                "poll_dispatcher_wake reached without an installed wake fd — \
                 SnapshotBridge::with_accessor_state stores accessor_worker_state \
                 and accessor_dispatcher_wake_evt together, so any caller that \
                 passed the worker_state gate must also have the wake fd. A None \
                 here indicates a bridge constructor that violated the coupling \
                 invariant"
            );
        };
        let ms = remaining.as_millis().min(i32::MAX as u128) as i32;
        let fd = {
            use std::os::unix::io::AsRawFd;
            evt.as_raw_fd()
        };
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: pollfd is a POD struct; libc::poll reads/writes
        // the slice in-bounds for nfds == 1.
        unsafe {
            libc::poll(&mut pfd, 1, ms);
        }
        // Drain the counter so a subsequent poll doesn't re-fire
        // immediately on the same edge. EFD_NONBLOCK so an empty fd
        // returns EAGAIN — ignore the result.
        let _ = evt.read();
    }

    /// Block until the accessor-init worker's publish seqno advances
    /// past `seqno_before`, or until `deadline` elapses. The dispatch
    /// for `Op::ReplaceScheduler` / `Op::AttachScheduler` calls this
    /// after spawning the new scheduler so the op only returns success
    /// once the new scheduler's BPF accessor pair has been re-published
    /// and a subsequent `Snapshot::active()` reflects the swap.
    ///
    /// On timeout the surfaced error reads the worker-state sentinel
    /// to distinguish "still trying" (transient — operator can bump
    /// the deadline) from "worker exited" (terminal — retry will
    /// hang the same way). Bridges without an accessor (test fixtures
    /// that omit [`Self::with_accessor_state`]) succeed immediately
    /// with `Ok(0)` so unit-test scenarios don't synthesize a worker.
    pub fn wait_for_accessor_publish_advance(
        &self,
        seqno_before: u64,
        deadline: std::time::Instant,
        op_label: &str,
    ) -> anyhow::Result<u64> {
        let (seqno, state) = match (
            self.accessor_publish_seqno.as_ref(),
            self.accessor_worker_state.as_ref(),
        ) {
            (Some(s), Some(w)) => (s, w),
            _ => return Ok(0),
        };
        // Tight initial check so a publish that already landed
        // between the dispatcher's pre-spawn `accessor_publish_seqno()`
        // load and this call returns without sleeping.
        let cur = seqno.load(std::sync::atomic::Ordering::Acquire);
        if cur > seqno_before {
            return Ok(cur);
        }
        loop {
            let cur_state = state.load(std::sync::atomic::Ordering::Acquire);
            if cur_state == accessor_worker_state::FAILED_PERMANENTLY {
                anyhow::bail!(
                    "{op_label}: accessor-init worker exited (FAILED_PERMANENTLY) — \
                     check freeze-coord 'vmm-accessor-init' logs for ELF parse or \
                     boot-budget failures; retrying the op will hit the same wall"
                );
            }
            let cur = seqno.load(std::sync::atomic::Ordering::Acquire);
            if cur > seqno_before {
                return Ok(cur);
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                let remaining_state = state.load(std::sync::atomic::Ordering::Acquire);
                anyhow::bail!(
                    "{op_label}: accessor reinit did not advance publish seqno \
                     from {seqno_before} within deadline (worker state = \
                     {remaining_state}; 0=Trying / 1=Succeeded / 2=FailedPermanently). \
                     A reinit that's stuck in Trying past the deadline indicates the \
                     coord's scan-tick hasn't observed the rebind or the worker's \
                     `from_elf_with_hint` retry is hitting a transient address-space \
                     window — check freeze-coord logs for `accessor-init:` lines"
                );
            }
            // Event-driven via accessor_dispatcher_wake_evt: the
            // worker writes the fd in lock-step with each seqno
            // bump and on FAILED_PERMANENTLY exit, so poll wakes at
            // kernel-scheduling-tick latency. Distinct from the
            // accessor_ready_evt the coord drains (no second-
            // consumer race). Reaching poll_dispatcher_wake without
            // an installed wake fd panics — the early-return at L679
            // gates this branch on accessor_worker_state.is_some()
            // and with_accessor_state stores both together.
            let remaining = deadline.saturating_duration_since(now);
            self.poll_dispatcher_wake(remaining);
        }
    }

    /// Install a kernel-op callback so
    /// `Op::WriteKernel{Hot,Cold}` / `Op::ReadKernel{Hot,Cold}` ops
    /// can dispatch host-side guest-memory writes/reads. Without
    /// one installed, the in-process executor returns "no
    /// SnapshotBridge installed" and the ops fall through to the
    /// virtio-console wire path. Test fixtures use this seam to
    /// record requests and assert on them without touching real
    /// guest memory.
    pub fn with_kernel_op(mut self, callback: KernelOpCallback) -> Self {
        self.kernel_op = Some(callback);
        self
    }

    /// Dispatch a kernel-op request through the installed callback.
    /// Returns `None` when no callback is installed (the executor
    /// then falls through to the wire path); returns
    /// `Some(KernelOpReplyPayload)` otherwise and records the
    /// reply in the per-tag drain log.
    pub fn dispatch_kernel_op(
        &self,
        request: &crate::vmm::wire::KernelOpRequestPayload,
    ) -> Option<crate::vmm::wire::KernelOpReplyPayload> {
        let callback = self.kernel_op.as_ref()?;
        let reply = callback(request);
        self.kernel_ops
            .lock_unpoisoned()
            .push((request.tag.clone(), reply.clone()));
        Some(reply)
    }

    /// Drain the per-tag kernel-op reply log. Returns the accumulated
    /// `(tag, reply)` pairs in insertion order; leaves the bridge's
    /// own copy empty so subsequent calls see only newer entries.
    pub fn drain_kernel_ops(&self) -> Vec<(String, crate::vmm::wire::KernelOpReplyPayload)> {
        std::mem::take(&mut *self.kernel_ops.lock_unpoisoned())
    }

    /// Record a `cgroup.procs` snapshot captured by
    /// [`Op::CaptureCgroupProcs`](crate::scenario::ops::Op::CaptureCgroupProcs).
    ///
    /// Appends to the per-bridge log in insertion order. Multiple
    /// captures of the same `cgroup` (distinguished by `tag`)
    /// surface as separate entries — the bridge does NOT collapse
    /// or overwrite on duplicate cgroup names, which lets a scenario
    /// capture pre/post snapshots of the same cgroup under different
    /// tags. Multiple captures with the same `(tag, cgroup)` also
    /// append rather than overwrite; tag uniqueness is a caller
    /// convention, not a framework-enforced contract.
    pub fn record_cgroup_procs(&self, tag: String, cgroup: String, pids: Vec<libc::pid_t>) {
        self.cgroup_procs
            .lock_unpoisoned()
            .push(CgroupProcsSnapshot::new(tag, cgroup, pids));
    }

    /// Drain the per-bridge `cgroup.procs` snapshot log. Returns the
    /// accumulated [`CgroupProcsSnapshot`]s in insertion order; leaves
    /// the bridge's own copy empty so subsequent calls observe only
    /// newer entries. Mirrors [`Self::drain_kernel_ops`]' contract.
    pub fn drain_cgroup_procs(&self) -> Vec<CgroupProcsSnapshot> {
        std::mem::take(&mut *self.cgroup_procs.lock_unpoisoned())
    }

    /// Look up the first `cgroup.procs` snapshot recorded under
    /// `tag` without consuming the drain log. Returns `None` when no
    /// snapshot was recorded under that tag (the typical case is a
    /// typo, a missing capture op, or capture-before-drain ordering
    /// mistake). For the common single-capture-per-tag pattern this
    /// collapses the `drain_cgroup_procs().iter().find(|s| s.tag ==
    /// tag).expect("missing")` 4-combinator dance into a one-liner.
    ///
    /// **Clone cost.** Each `CgroupProcsSnapshot` carries `tag`,
    /// `cgroup`, and `pids: Vec<libc::pid_t>` — the pids vec is
    /// bounded by the number of tasks in the cgroup at capture
    /// time (kilobytes at most for realistic test scenarios).
    /// Mirrors [`Self::kernel_op_value`]'s non-draining-lookup
    /// shape for sibling-API consistency.
    ///
    /// **Duplicate tag.** Returns the FIRST snapshot recorded under
    /// `tag`. Multiple captures with the same `(tag, cgroup)` are
    /// stored separately (see [`Self::record_cgroup_procs`]); callers
    /// who care about the multiplicity should use
    /// [`Self::drain_cgroup_procs`] and filter the Vec manually.
    pub fn cgroup_procs_by_tag(&self, tag: &str) -> Option<CgroupProcsSnapshot> {
        self.cgroup_procs
            .lock_unpoisoned()
            .iter()
            .find(|s| s.tag == tag)
            .cloned()
    }

    /// Record a kernel-op reply produced by the host-side wire-path
    /// dispatcher into the same per-tag drain log that
    /// [`Self::dispatch_kernel_op`] populates.
    ///
    /// The in-process bridge path stores its replies inside
    /// `dispatch_kernel_op` (which both invokes the callback AND
    /// pushes the reply into the log). The wire-path used by ops
    /// running inside a guest VM produces its reply on the host
    /// freeze-coordinator and frames it back over virtio-console
    /// port 1 — there is no callback to drive the push, so the
    /// host coordinator calls this record-only method directly
    /// after framing each reply. Without this hook, `post_vm`
    /// callbacks observing [`crate::vmm::VmResult::snapshot_bridge`]
    /// would see an empty drain log for any guest-side
    /// `Op::WriteKernel*` / `Op::ReadKernel*` invocation, defeating
    /// the asserts-from-replies pattern the gated cold-path e2e
    /// scaffolding pins.
    pub fn record_kernel_op_reply(
        &self,
        tag: String,
        reply: crate::vmm::wire::KernelOpReplyPayload,
    ) {
        self.kernel_ops.lock_unpoisoned().push((tag, reply));
    }

    /// Look up the first kernel-op reply value recorded under `tag`
    /// in the kernel-op drain log without consuming the log.
    ///
    /// The bulk shape returned by [`Self::drain_kernel_ops`] is
    /// `Vec<(tag, reply)>` with each reply carrying a
    /// `Vec<crate::vmm::wire::KernelOpValue>`. For the common
    /// single-tag single-value read-back lookup, this helper
    /// collapses the 4-layer unwrap (find by tag → check success →
    /// index into read_values → match the variant) into a single
    /// call. Returns `None` when no reply was recorded under `tag`,
    /// when the reply reported `success = false`, or when the
    /// reply's `read_values` is empty (e.g. a write-op reply under
    /// the same tag). Otherwise returns `Some(value)` with the
    /// first `KernelOpValue` of the first matching reply.
    ///
    /// The log is NOT drained — the caller can still inspect via
    /// [`Self::drain_kernel_ops`] to observe the full per-tag
    /// history.
    ///
    /// **Clone cost.** For `U32` / `U64` the clone is 4 / 8 bytes.
    /// For `crate::vmm::wire::KernelOpValue::Bytes` the clone
    /// can be up to `crate::vmm::wire::KERNEL_OP_REPLY_MAX`
    /// (1 MiB). Hot paths that repeatedly inspect the same tag
    /// should prefer [`Self::drain_kernel_ops`] + index into the
    /// returned Vec to avoid the per-call clone.
    pub fn kernel_op_value(&self, tag: &str) -> Option<crate::vmm::wire::KernelOpValue> {
        self.kernel_ops
            .lock_unpoisoned()
            .iter()
            .find(|(t, reply)| t == tag && reply.success && !reply.read_values.is_empty())
            .map(|(_, reply)| reply.read_values[0].clone())
    }

    /// Install a watch-register callback so [`Op::WatchSnapshot`](crate::scenario::ops::Op::WatchSnapshot)
    /// ops can attach hardware-watchpoint snapshots. The callback
    /// is responsible for symbol resolution, watchpoint slot allocation, and
    /// `KVM_SET_GUEST_DEBUG` arming.
    pub fn with_watch_register(mut self, register: WatchRegisterCallback) -> Self {
        self.register_watch = Some(register);
        self
    }

    /// Register a hardware-watchpoint snapshot for `symbol`.
    ///
    /// Enforces the per-scenario [`MAX_WATCH_SNAPSHOTS`] cap before
    /// invoking the host's watch-register callback. Returns
    /// `Err(reason)` when:
    /// - The cap has been reached (slot 0 reserved + 3 user slots
    ///   allocated).
    /// - No watch-register callback was installed via
    ///   [`Self::with_watch_register`].
    /// - The host's callback rejected the request (symbol unresolved,
    ///   alignment violation, ioctl failure).
    pub fn register_watch(&self, symbol: &str) -> std::result::Result<(), String> {
        // Reserve a slot via compare_exchange so concurrent callers
        // can never push the count past MAX_WATCH_SNAPSHOTS even
        // transiently. The previous fetch_add+rollback path let two
        // concurrent threads observe `prev < MAX` and increment past
        // the cap before either rolled back, briefly violating the
        // invariant `watch_count <= MAX_WATCH_SNAPSHOTS`.
        loop {
            let prev = self.watch_count.load(std::sync::atomic::Ordering::Relaxed);
            if prev >= MAX_WATCH_SNAPSHOTS {
                return Err(format!(
                    "Op::WatchSnapshot cap exceeded: scenario already registered \
                     {MAX_WATCH_SNAPSHOTS} watchpoints ({MAX_WATCH_SNAPSHOTS} user \
                     watchpoint slots occupied; slot 0 reserved for the error-class \
                     exit_kind trigger). Drop a watch or use Op::CaptureSnapshot for a \
                     time-driven capture instead."
                ));
            }
            if self
                .watch_count
                .compare_exchange_weak(
                    prev,
                    prev + 1,
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
            // Lost the CAS to a concurrent register/unregister; reload
            // and retry. spurious failures are also retried — that is
            // why this uses the _weak variant inside a loop.
        }
        // Slot reserved. Wrap it in a Drop guard so a panic inside
        // `register(symbol)` releases the reservation on unwind — the
        // previous manual-fetch_sub rollback only ran on the explicit
        // Err(reason) arm, leaking the slot permanently if the
        // callback panicked. The success path commits the slot with
        // mem::forget after register returns Ok.
        let guard = WatchSlotGuard {
            count: &self.watch_count,
        };
        let Some(register) = self.register_watch.as_ref() else {
            drop(guard);
            return Err(format!(
                "Op::WatchSnapshot('{symbol}'): no watch-register callback installed \
                 on this SnapshotBridge — the host wires one via \
                 SnapshotBridge::with_watch_register before execute_steps; \
                 in-guest / no-VM scenarios cannot register hardware watchpoints"
            ));
        };
        register(symbol)?;
        std::mem::forget(guard);
        Ok(())
    }

    /// Number of watchpoint snapshots currently registered.
    pub fn watch_count(&self) -> usize {
        self.watch_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Drive the capture closure and store the result under `name`.
    /// Returns `true` when a report was captured and stored;
    /// `false` when the closure returned `None`.
    pub fn capture(&self, name: &str) -> bool {
        let Some(report) = (self.capture)(name) else {
            tracing::warn!(
                name,
                "SnapshotBridge::capture: capture callback returned None — snapshot unavailable"
            );
            self.snapshots
                .lock_unpoisoned()
                .push_event(SnapshotBridgeEvent::CaptureUnavailable {
                    tag: name.to_string(),
                });
            return false;
        };
        self.store(name, report);
        true
    }

    /// Step-aware variant of [`Self::capture`]: drives the capture
    /// closure and stores the result under `name`, stamping it with
    /// `step_index` so the drained entry's
    /// [`super::error::DrainedSnapshotEntry::step_index`] surfaces
    /// the scenario phase the capture belongs to.
    ///
    /// `step_index` is encoded per the 1-indexed phase convention:
    /// `0` is the BASELINE settle window, `1..=N` align with
    /// scenario Step ordinals. The host-side on-demand-capture
    /// dispatch reads
    /// [`crate::scenario::Ctx::current_step`] just before the
    /// freeze rendezvous and passes the loaded value through this
    /// entry point so the downstream phase aggregator can bucket
    /// the sample directly.
    ///
    /// Returns `true` when a report was captured and stored;
    /// `false` when the closure returned `None`.
    pub fn capture_with_step(&self, name: &str, step_index: u16) -> bool {
        let Some(report) = (self.capture)(name) else {
            tracing::warn!(
                name,
                step_index,
                "SnapshotBridge::capture_with_step: capture callback returned None — snapshot unavailable"
            );
            self.snapshots
                .lock_unpoisoned()
                .push_event(SnapshotBridgeEvent::CaptureUnavailable {
                    tag: name.to_string(),
                });
            return false;
        };
        self.store_internal(name, report, None, None, None, Some(step_index));
        true
    }

    /// Store a pre-built [`FailureDumpReport`] under `name`,
    /// bypassing the capture callback. Used by the host-side freeze
    /// coordinator after it runs `freeze_and_dispatch(FreezeMode::Capture { gate_on_exit_kind: false })` and
    /// wants to publish the resulting report on the bridge for the
    /// test author to drain post-VM-exit.
    ///
    /// Storage is capped at [`MAX_STORED_SNAPSHOTS`] entries to bound
    /// host memory under runaway capture cadence (e.g. a Loop step
    /// firing `Op::CaptureSnapshot` with a unique tag every iteration).
    /// When the cap is reached, the oldest stored entry is evicted
    /// with a `tracing::warn!` naming the dropped tag. An overwrite
    /// of an existing tag also warns and replaces the prior report
    /// in place without disturbing FIFO ordering of other entries.
    pub fn store(&self, name: &str, report: FailureDumpReport) {
        self.store_internal(name, report, None, None, None, None);
    }

    /// Bundle a [`FailureDumpReport`] with the scx_stats JSON and
    /// elapsed-millisecond timestamp captured at the same periodic
    /// boundary. Used by the freeze coordinator's periodic-fire path
    /// so [`Sample`](crate::scenario::sample::Sample) can pair the
    /// frozen BPF state with the running-scheduler stats observed
    /// just before the freeze rendezvous.
    ///
    /// Stats / elapsed are stored in parallel HashMaps keyed by the
    /// same tag as the report. FIFO eviction sweeps all three in
    /// lock-step; an overwrite refreshes order and replaces every
    /// parallel value (or clears it when the new write passes
    /// `None`) so a stale stats / elapsed entry can never accompany
    /// a freshly stored report.
    pub fn store_with_stats(
        &self,
        name: &str,
        report: FailureDumpReport,
        stats: Option<Result<serde_json::Value, super::error::MissingStatsReason>>,
        elapsed_ms: Option<u64>,
    ) {
        self.store_internal(name, report, stats, elapsed_ms, None, None);
    }

    /// Step-aware variant of [`Self::store_with_stats`]: bundles the
    /// scenario phase index alongside the report / stats / elapsed
    /// tuple so the drained entry's
    /// [`super::error::DrainedSnapshotEntry::step_index`] carries
    /// the phase the capture belongs to. The freeze coordinator's
    /// periodic-fire path reads
    /// [`crate::scenario::Ctx::current_step`] just before the
    /// rendezvous and routes the value through this method so each
    /// periodic sample is bucketable per phase without a second
    /// lookup.
    ///
    /// `step_index` is encoded per the 1-indexed phase convention
    /// — `0` is the BASELINE settle window, `1..=N` align with
    /// scenario Step ordinals. All other arguments match
    /// [`Self::store_with_stats`] verbatim.
    pub fn store_with_stats_and_step(
        &self,
        name: &str,
        report: FailureDumpReport,
        stats: Option<Result<serde_json::Value, super::error::MissingStatsReason>>,
        elapsed_ms: Option<u64>,
        boundary_offset_ms: Option<u64>,
        step_index: u16,
    ) {
        self.store_internal(
            name,
            report,
            stats,
            elapsed_ms,
            boundary_offset_ms,
            Some(step_index),
        );
    }

    fn store_internal(
        &self,
        name: &str,
        report: FailureDumpReport,
        stats: Option<Result<serde_json::Value, super::error::MissingStatsReason>>,
        elapsed_ms: Option<u64>,
        boundary_offset_ms: Option<u64>,
        step_index: Option<u16>,
    ) {
        let mut store = self.snapshots.lock_unpoisoned();
        if let Some(existing) = store.reports.insert(name.to_string(), report) {
            tracing::warn!(
                name,
                schema = %existing.schema,
                "SnapshotBridge::store: name already had a stored report; overwriting prior capture"
            );
            store.push_event(SnapshotBridgeEvent::Overwrite {
                tag: name.to_string(),
                prior_schema: existing.schema.clone(),
            });
            // Move this tag to the back of the FIFO order so the
            // overwrite refreshes its position (newest insertion =
            // farthest from eviction). Without this, a hot-rewritten
            // tag would still be the oldest and risk eviction even
            // when actively updated.
            if let Some(pos) = store.order.iter().position(|k| k == name) {
                store.order.remove(pos);
            }
            store.order.push_back(name.to_string());
            // Refresh / clear parallel stats and elapsed entries so
            // the post-overwrite `(report, stats, elapsed)` tuple is
            // self-consistent — a None overwrite must clear the prior
            // value rather than carrying forward a stale match from
            // an earlier capture.
            match stats {
                Some(v) => {
                    store.stats.insert(name.to_string(), v);
                }
                None => {
                    store.stats.remove(name);
                }
            }
            match elapsed_ms {
                Some(v) => {
                    store.elapsed_ms.insert(name.to_string(), v);
                }
                None => {
                    store.elapsed_ms.remove(name);
                }
            }
            match boundary_offset_ms {
                Some(v) => {
                    store.boundary_offset_ms.insert(name.to_string(), v);
                }
                None => {
                    store.boundary_offset_ms.remove(name);
                }
            }
            match step_index {
                Some(v) => {
                    store.step_index.insert(name.to_string(), v);
                }
                None => {
                    store.step_index.remove(name);
                }
            }
            return;
        }
        store.order.push_back(name.to_string());
        if let Some(v) = stats {
            store.stats.insert(name.to_string(), v);
        }
        if let Some(v) = elapsed_ms {
            store.elapsed_ms.insert(name.to_string(), v);
        }
        if let Some(v) = boundary_offset_ms {
            store.boundary_offset_ms.insert(name.to_string(), v);
        }
        if let Some(v) = step_index {
            store.step_index.insert(name.to_string(), v);
        }
        while store.reports.len() > MAX_STORED_SNAPSHOTS {
            let Some(evicted) = store.order.pop_front() else {
                // Defensive: if order is empty while reports is over
                // cap something is desynchronised — clear reports to
                // restore the invariant rather than loop forever.
                let nuked = store.reports.len();
                tracing::warn!(
                    reports_len = nuked,
                    cap = MAX_STORED_SNAPSHOTS,
                    "SnapshotBridge::store: order empty while reports over cap — bulk-clearing to restore invariant"
                );
                store.push_event(SnapshotBridgeEvent::CapInvariantViolation {
                    reports_len: nuked,
                    cap: MAX_STORED_SNAPSHOTS,
                });
                store.reports.clear();
                store.stats.clear();
                store.elapsed_ms.clear();
                store.boundary_offset_ms.clear();
                store.step_index.clear();
                break;
            };
            if store.reports.remove(&evicted).is_some() {
                tracing::warn!(
                    evicted = %evicted,
                    cap = MAX_STORED_SNAPSHOTS,
                    "SnapshotBridge::store: cap reached, evicting oldest captured snapshot"
                );
                store.push_event(SnapshotBridgeEvent::Eviction {
                    evicted_tag: evicted.clone(),
                    new_tag: name.to_string(),
                    cap: MAX_STORED_SNAPSHOTS,
                });
            }
            // Sweep the parallel maps in lock-step so a stranded
            // stats / elapsed / boundary_offset / step_index entry
            // cannot outlive its report.
            store.stats.remove(&evicted);
            store.elapsed_ms.remove(&evicted);
            store.boundary_offset_ms.remove(&evicted);
            store.step_index.remove(&evicted);
        }
    }

    /// Snapshot count for diagnostic logging.
    pub fn len(&self) -> usize {
        self.snapshots.lock_unpoisoned().reports.len()
    }

    /// True when no snapshots have been captured.
    pub fn is_empty(&self) -> bool {
        self.snapshots.lock_unpoisoned().reports.is_empty()
    }

    /// Count of stored periodic-tagged reports that carry REAL BPF
    /// state (not placeholders synthesized by rendezvous timeouts /
    /// gate suppression). Distinct from
    /// [`crate::vmm::VmResult::periodic_fired`], which counts every
    /// periodic boundary the freeze coordinator attempted —
    /// including the ones that landed only a placeholder when the
    /// vCPU rendezvous timed out.
    ///
    /// This is the "useful data produced" floor: a scheduler that
    /// attached but produced nothing but placeholders surfaces as
    /// `periodic_real_count() == 0` here even though
    /// `periodic_fired` may be `target`. Tests that want a
    /// stricter smoke-floor than `periodic_fired >= 1` (which
    /// passes on placeholder-only fills) read this query.
    ///
    /// **Tag-format pin.** The coordinator emits periodic tags via
    /// `format!("periodic_{:03}", idx)` at
    /// `src/vmm/freeze_coord/state.rs` — always
    /// `"periodic_"` + exactly 3 ASCII digits. The match here
    /// enforces that exact shape (NOT a loose prefix) so a user
    /// `Op::CaptureSnapshot { name: "periodic_kaslr" }` cannot
    /// collide with the periodic dispatch namespace and pollute
    /// the count.
    ///
    /// **What "real" measures.** A placeholder report may still
    /// carry `vcpu_regs` from a degraded capture (see
    /// `src/vmm/freeze_coord/mod.rs` periodic-degraded path —
    /// `degraded.vcpu_regs` is preserved into the stored
    /// placeholder). The floor here treats those as "not real"
    /// because the contract is "the test produced BPF-state data"
    /// — vcpu_regs alone don't satisfy that. Tests that want the
    /// looser "any capture-attempt landed" floor read
    /// [`crate::vmm::VmResult::periodic_fired`] instead.
    pub fn periodic_real_count(&self) -> u32 {
        let store = self.snapshots.lock_unpoisoned();
        let mut n: u32 = 0;
        for (tag, report) in &store.reports {
            if is_periodic_tag(tag) && !report.is_placeholder {
                n = n.saturating_add(1);
            }
        }
        n
    }

    /// True when a stored report already exists for `name`. Lets the
    /// freeze coordinator's final-drain placeholder path skip storing
    /// a degraded "coord exited before capture" report on top of a
    /// real capture that the in-loop dispatch landed earlier — without
    /// this gate, a vCPU thread that re-armed `hit=true` after the
    /// in-loop service successfully published the report would have
    /// its tag's stored capture overwritten by the placeholder at
    /// teardown, presenting tests with a hollow snapshot in place of
    /// the real one.
    pub fn has(&self, name: &str) -> bool {
        self.snapshots.lock_unpoisoned().reports.contains_key(name)
    }

    /// Take ownership of the captured snapshots, leaving the bridge
    /// empty. Drops any periodic-capture stats / elapsed metadata
    /// stored alongside reports — callers that need the stats JSON
    /// or per-sample timestamp must use
    /// [`Self::drain_ordered_with_stats`] instead.
    pub fn drain(&self) -> HashMap<String, FailureDumpReport> {
        let mut store = self.snapshots.lock_unpoisoned();
        store.order.clear();
        store.stats.clear();
        store.elapsed_ms.clear();
        store.step_index.clear();
        std::mem::take(&mut store.reports)
    }

    /// Take ownership of the captured snapshots in insertion order,
    /// leaving the bridge empty. The returned `Vec` walks
    /// `SnapshotStore::order` (the FIFO key list maintained by
    /// [`Self::store`]) so periodic captures — whose ordering IS the
    /// signal — are returned `periodic_000` first, `periodic_NNN`
    /// last. [`Self::drain`] returns a `HashMap` and loses ordering;
    /// use this method when ordering matters.
    ///
    /// An overwrite of an existing tag (the `if let Some(existing) =
    /// store.reports.insert(...)` branch in [`Self::store`]) moves
    /// the tag to the back of the FIFO — `drain_ordered` therefore
    /// returns the LATEST capture under each tag exactly once, in
    /// the order of its most-recent insertion.
    ///
    /// FIFO eviction at [`MAX_STORED_SNAPSHOTS`] drops the oldest
    /// tags from `order` AND `reports` together, so a hot run that
    /// fired more than the cap returns the most recent
    /// [`MAX_STORED_SNAPSHOTS`] captures in insertion order; older
    /// captures are gone and [`Self::store`] already logged the
    /// eviction.
    pub fn drain_ordered(&self) -> Vec<(String, FailureDumpReport)> {
        let mut store = self.snapshots.lock_unpoisoned();
        let order = std::mem::take(&mut store.order);
        let mut reports = std::mem::take(&mut store.reports);
        // Stats / elapsed / step_index are dropped with the bridge —
        // callers that need the parallel data must use
        // `drain_ordered_with_stats` instead.
        store.stats.clear();
        store.elapsed_ms.clear();
        store.step_index.clear();
        let mut out: Vec<(String, FailureDumpReport)> = Vec::with_capacity(order.len());
        for tag in order {
            if let Some(report) = reports.remove(&tag) {
                out.push((tag, report));
            }
        }
        // Defensive: if any reports remained outside the order Vec
        // (an invariant violation that would only fire if a future
        // refactor of `store()` desynchronised the two), surface
        // them at the tail rather than dropping silently. Their
        // relative order is HashMap-iteration-arbitrary but at
        // least nothing is lost.
        for (tag, report) in reports {
            tracing::warn!(
                tag,
                "SnapshotBridge::drain_ordered: report present in `reports` \
                 but missing from `order` — surfacing at tail (FIFO \
                 invariant violation; please file)"
            );
            store.push_event(SnapshotBridgeEvent::DrainOrderingInvariantViolation {
                tag: tag.clone(),
                drain_variant: "drain_ordered",
            });
            out.push((tag, report));
        }
        out
    }

    /// Take ownership of the captured snapshots in insertion order
    /// along with the parallel scx_stats JSON and per-sample
    /// elapsed-ms timestamps (`None` per slot when the tag was
    /// captured outside the periodic-capture path or when the stats
    /// request failed). Empties the bridge — every parallel map is
    /// drained in lock-step so a follow-up call returns an empty
    /// vec.
    ///
    /// The returned tuple shape `(tag, report, stats, elapsed_ms)`
    /// is the input to
    /// [`SampleSeries::from_drained`](crate::scenario::sample::SampleSeries::from_drained):
    /// the bridge owns the raw drainable shape, the higher-level
    /// `SampleSeries` view consumes it. Insertion order is the
    /// signal — periodic captures land
    /// `periodic_000`/`periodic_001`/… in monotonic wall-clock
    /// order, and the temporal-assertion patterns walk the vec
    /// expecting that ordering.
    pub fn drain_ordered_with_stats(&self) -> Vec<super::error::DrainedSnapshotEntry> {
        let mut store = self.snapshots.lock_unpoisoned();
        let order = std::mem::take(&mut store.order);
        let mut reports = std::mem::take(&mut store.reports);
        let mut stats = std::mem::take(&mut store.stats);
        let mut elapsed = std::mem::take(&mut store.elapsed_ms);
        let mut boundary_offset = std::mem::take(&mut store.boundary_offset_ms);
        let mut step_index = std::mem::take(&mut store.step_index);
        let mut out: Vec<super::error::DrainedSnapshotEntry> = Vec::with_capacity(order.len());
        // Bridge-absent stats slot collapses to the typed
        // `NoSchedulerBinary` reason: the capture path that produced
        // this tag never bundled a stats Result (non-periodic Op
        // capture, or periodic without a stats client wired). The
        // periodic path always bundles a Some(Result), so a `None`
        // here is always the "no scheduler binary" case.
        let stats_fallback = || Err(super::error::MissingStatsReason::NoSchedulerBinary);
        for tag in order {
            if let Some(report) = reports.remove(&tag) {
                let s = stats.remove(&tag).unwrap_or_else(stats_fallback);
                let e = elapsed.remove(&tag);
                let bo = boundary_offset.remove(&tag);
                let phase = step_index.remove(&tag);
                out.push(super::error::DrainedSnapshotEntry {
                    tag,
                    report,
                    stats: s,
                    elapsed_ms: e,
                    boundary_offset_ms: bo,
                    step_index: phase,
                });
            }
        }
        // Defensive tail for desynchronised maps (matches
        // `drain_ordered`'s tail behaviour). Any stats / elapsed /
        // step_index entries that were not paired with a tag in
        // `order` are dropped because they have no anchoring report
        // — surfacing them as orphaned tuples would invent a structure
        // no consumer expects.
        for (tag, report) in reports {
            tracing::warn!(
                tag,
                "SnapshotBridge::drain_ordered_with_stats: report present in `reports` \
                 but missing from `order` — surfacing at tail (FIFO \
                 invariant violation; please file)"
            );
            store.push_event(SnapshotBridgeEvent::DrainOrderingInvariantViolation {
                tag: tag.clone(),
                drain_variant: "drain_ordered_with_stats",
            });
            let s = stats.remove(&tag).unwrap_or_else(stats_fallback);
            let e = elapsed.remove(&tag);
            let bo = boundary_offset.remove(&tag);
            let phase = step_index.remove(&tag);
            out.push(super::error::DrainedSnapshotEntry {
                tag,
                report,
                stats: s,
                elapsed_ms: e,
                boundary_offset_ms: bo,
                step_index: phase,
            });
        }
        out
    }

    /// Take ownership of all queued [`SnapshotBridgeEvent`]s in
    /// insertion order. Empties the internal log; a follow-up call
    /// returns an empty vec. Test authors call this after
    /// [`Self::drain_ordered`] / [`Self::drain_ordered_with_stats`]
    /// to inspect bridge-side conditions that fired during the
    /// scenario (eviction, overwrite, missing capture, invariant
    /// violation).
    ///
    /// Independent of the report drain — events accumulate even on
    /// scenarios that never call `drain_ordered*`, and reports
    /// remain reachable even on scenarios that never call
    /// `drain_events`. Tests that want to fail on a bridge event
    /// compose the streams: drain events, inspect, fail with
    /// `AssertResult::fail(AssertDetail::new(Other, ...))` if any
    /// variant is unexpected.
    ///
    /// When the events log hit [`MAX_STORED_EVENTS`] and FIFO-evicted
    /// older entries since the previous drain, a synthetic
    /// [`SnapshotBridgeEvent::EventLogTruncated`] is appended at the
    /// tail of the returned vec carrying the dropped count — the
    /// operator never silently loses events. The internal dropped
    /// counter resets to 0 after every drain.
    pub fn drain_events(&self) -> Vec<SnapshotBridgeEvent> {
        let mut store = self.snapshots.lock_unpoisoned();
        let mut events = std::mem::take(&mut store.events);
        if store.events_dropped > 0 {
            events.push(SnapshotBridgeEvent::EventLogTruncated {
                dropped_count: store.events_dropped,
            });
            store.events_dropped = 0;
        }
        events
    }

    /// Non-draining count of queued [`SnapshotBridgeEvent`]s. Useful
    /// for "no bridge events fired" assertions without consuming
    /// the log — `assert_eq!(bridge.event_count(), 0)`. Does NOT
    /// include the synthetic
    /// [`SnapshotBridgeEvent::EventLogTruncated`] marker that
    /// [`Self::drain_events`] would append; that marker is
    /// drain-time-only and `events_dropped > 0` is observable via
    /// the next drain rather than via this counter.
    pub fn event_count(&self) -> usize {
        self.snapshots.lock_unpoisoned().events.len()
    }

    /// Install this bridge as the active bridge for the calling
    /// thread. The bridge stays installed for the lifetime of the
    /// returned [`BridgeGuard`]; on drop the prior bridge (or
    /// `None`) is restored.
    ///
    /// Thread-local because [`execute_steps`](crate::scenario::ops::execute_steps)
    /// runs on the calling thread and `Op::CaptureSnapshot` only makes
    /// sense in that exact thread's call stack — installing a
    /// bridge process-wide would race against parallel test
    /// threads.
    ///
    /// # Cloning for drain
    ///
    /// `set_thread_local` consumes `self`. The bridge is `Clone`
    /// (internal state lives behind `Arc<Mutex<_>>`), so callers
    /// that need to inspect / drain the bridge after the scenario
    /// runs MUST clone before installing:
    ///
    /// ```ignore
    /// let bridge = SnapshotBridge::new(capture_cb);
    /// let drain_handle = bridge.clone();      // share the Arc
    /// let _guard = bridge.set_thread_local(); // consume the original
    /// // ... execute_scenario runs and records on the thread-local ...
    /// let snaps = drain_handle.drain_cgroup_procs();
    /// ```
    ///
    /// Both handles share the same underlying snapshot store
    /// (reports, kernel_ops, cgroup_procs); ops recorded against
    /// the thread-local are observable via the cloned handle.
    /// Forgetting to clone is the single most common bridge-flow
    /// footgun for tests that consume capture results.
    pub fn set_thread_local(self) -> BridgeGuard {
        let prev = ACTIVE_BRIDGE.with(|c| c.borrow_mut().replace(self));
        BridgeGuard { prev }
    }
}

thread_local! {
    static ACTIVE_BRIDGE: std::cell::RefCell<Option<SnapshotBridge>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII guard returned by [`SnapshotBridge::set_thread_local`].
/// Restores the prior thread-local bridge on drop so a nested
/// scenario inside an outer one cannot leak its bridge into the
/// outer scope.
#[must_use = "BridgeGuard restores the prior bridge on drop; bind it"]
pub struct BridgeGuard {
    prev: Option<SnapshotBridge>,
}

impl Drop for BridgeGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        ACTIVE_BRIDGE.with(|c| {
            *c.borrow_mut() = prev;
        });
    }
}

/// Run `f` with the active bridge if one is installed. When no
/// bridge is installed, returns `None` without invoking `f` — the
/// caller's responsibility to fall through to its own no-bridge
/// path.
pub fn with_active_bridge<R>(f: impl FnOnce(&SnapshotBridge) -> R) -> Option<R> {
    ACTIVE_BRIDGE.with(|c| c.borrow().as_ref().map(f))
}

#[cfg(test)]
mod accessor_wait_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU8, AtomicU64};
    use std::time::{Duration, Instant};

    fn bridge_with_accessor_state() -> (
        SnapshotBridge,
        Arc<AtomicU64>,
        Arc<AtomicU8>,
        Arc<vmm_sys_util::eventfd::EventFd>,
    ) {
        let seqno = Arc::new(AtomicU64::new(0));
        let worker_state = Arc::new(AtomicU8::new(accessor_worker_state::TRYING));
        let wake_evt = Arc::new(
            vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK)
                .expect("eventfd for accessor_dispatcher_wake test fixture"),
        );
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb).with_accessor_state(
            seqno.clone(),
            worker_state.clone(),
            wake_evt.clone(),
        );
        (bridge, seqno, worker_state, wake_evt)
    }

    #[test]
    fn wait_no_accessor_state_returns_ok_zero_immediately() {
        // Bridge built without with_accessor_state — wait degenerates
        // to Ok(0) so unit-test scenarios that don't spawn a real
        // worker don't synthesize one.
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        let deadline = Instant::now() + Duration::from_secs(60);
        assert_eq!(
            bridge
                .wait_for_accessor_publish_advance(0, deadline, "Op::Test")
                .unwrap(),
            0
        );
    }

    #[test]
    fn wait_seqno_already_advanced_returns_immediately() {
        // Pre-advance the seqno so the tight initial check returns
        // without polling. Validates the "publish landed between
        // dispatch's load and wait call" race window is handled.
        let (bridge, seqno, _, _) = bridge_with_accessor_state();
        seqno.store(5, std::sync::atomic::Ordering::Release);
        let deadline = Instant::now() + Duration::from_secs(60);
        assert_eq!(
            bridge
                .wait_for_accessor_publish_advance(3, deadline, "Op::Test")
                .unwrap(),
            5
        );
    }

    #[test]
    fn wait_observes_worker_publish() {
        // Pin: worker bumps seqno + pulses dispatcher wake fd on a
        // side thread; dispatch-style wait observes the advance
        // within the deadline. Event-driven path — the wake fd
        // unblocks poll within a kernel-scheduling tick, so the
        // test should complete well under 500 ms even though the
        // deadline is 2 s.
        let (bridge, seqno, _, wake_evt) = bridge_with_accessor_state();
        let bridge_for_thread = bridge.clone();
        let seqno_for_thread = seqno.clone();
        let wake_for_thread = wake_evt.clone();
        let publisher = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            seqno_for_thread.fetch_add(1, std::sync::atomic::Ordering::Release);
            // Pulse the wake fd in lock-step with the seqno bump,
            // matching the worker's contract at the freeze_coord
            // publish site.
            let _ = wake_for_thread.write(1);
            let _ = bridge_for_thread; // keep bridge alive for the thread duration
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        let t0 = Instant::now();
        let observed = bridge
            .wait_for_accessor_publish_advance(0, deadline, "Op::Test")
            .expect("publish observed within deadline");
        let elapsed = t0.elapsed();
        publisher.join().unwrap();
        assert_eq!(observed, 1);
        // Event-driven wake — the wait should return shortly after
        // the 100ms publisher sleep, NOT at the 2s deadline. A 500ms
        // ceiling leaves plenty of slack for slow CI hosts while
        // still catching a regression where wait falls back to a
        // pure-sleep loop.
        assert!(
            elapsed < Duration::from_millis(500),
            "wait did not wake event-driven; took {elapsed:?} \
             (expected close to 100ms)"
        );
    }

    #[test]
    fn wait_bails_on_worker_failed_permanently() {
        // Pin: when worker_state flips to FAILED_PERMANENTLY the
        // wait surfaces the terminal-worker diagnostic instead of
        // blocking the full deadline.
        let (bridge, _, worker_state, _) = bridge_with_accessor_state();
        worker_state.store(
            accessor_worker_state::FAILED_PERMANENTLY,
            std::sync::atomic::Ordering::Release,
        );
        let deadline = Instant::now() + Duration::from_secs(60);
        let t0 = Instant::now();
        let err = bridge
            .wait_for_accessor_publish_advance(0, deadline, "Op::Test")
            .expect_err("expected terminal-worker bail");
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_millis(100),
            "wait did not surface terminal state quickly; took {elapsed:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("FAILED_PERMANENTLY"),
            "bail message missing terminal sentinel: {msg}"
        );
        assert!(msg.contains("Op::Test"), "bail missing op label: {msg}");
    }

    #[test]
    fn wait_bails_on_deadline_with_worker_state_in_diagnostic() {
        // Pin: deadline-exceeded surfaces with the worker state
        // attached so the diagnostic distinguishes stuck-in-Trying
        // (transient) from FAILED_PERMANENTLY (terminal).
        let (bridge, _, _, _) = bridge_with_accessor_state();
        let deadline = Instant::now() + Duration::from_millis(120);
        let err = bridge
            .wait_for_accessor_publish_advance(0, deadline, "Op::Test")
            .expect_err("expected deadline bail");
        let msg = err.to_string();
        assert!(
            msg.contains("worker state = 0"),
            "deadline diagnostic missing worker_state: {msg}"
        );
        assert!(
            msg.contains("Trying"),
            "deadline diagnostic missing state name table: {msg}"
        );
    }

    #[test]
    fn accessor_publish_seqno_returns_zero_without_accessor_state() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        assert_eq!(bridge.accessor_publish_seqno(), 0);
    }

    #[test]
    fn accessor_publish_seqno_reads_atomic() {
        let (bridge, seqno, _, _) = bridge_with_accessor_state();
        assert_eq!(bridge.accessor_publish_seqno(), 0);
        seqno.store(42, std::sync::atomic::Ordering::Release);
        assert_eq!(bridge.accessor_publish_seqno(), 42);
    }
}

#[cfg(test)]
mod periodic_tag_tests {
    use super::*;

    /// `is_periodic_tag` MUST accept the exact format the
    /// coordinator emits: `"periodic_"` + 3 ASCII digits. Pins the
    /// canonical shape so a regression that relaxed the matcher
    /// (e.g., back to `starts_with("periodic_")`) would surface
    /// immediately via the rejected-tag cases below.
    #[test]
    fn is_periodic_tag_accepts_canonical_three_digit_index() {
        assert!(is_periodic_tag("periodic_000"));
        assert!(is_periodic_tag("periodic_007"));
        assert!(is_periodic_tag("periodic_123"));
        assert!(is_periodic_tag("periodic_999"));
    }

    /// User-supplied `Op::CaptureSnapshot` tags whose names start
    /// with `"periodic_"` but DON'T match the strict
    /// `periodic_NNN` shape MUST be rejected. This is the
    /// load-bearing defense against tag collision polluting
    /// `periodic_real_count`.
    #[test]
    fn is_periodic_tag_rejects_user_tag_collisions() {
        assert!(!is_periodic_tag("periodic_kaslr"));
        assert!(!is_periodic_tag("periodic_user_baseline"));
        assert!(!is_periodic_tag("periodic_"));
        assert!(!is_periodic_tag("periodic_1"));
        assert!(!is_periodic_tag("periodic_12"));
        assert!(!is_periodic_tag("periodic_1234"));
        assert!(!is_periodic_tag("periodic_00a"));
        assert!(!is_periodic_tag("periodic_007 "));
        assert!(!is_periodic_tag("PERIODIC_000"));
        assert!(!is_periodic_tag("capture_my_thing"));
        assert!(!is_periodic_tag(""));
        assert!(!is_periodic_tag("periodic"));
    }

    /// `periodic_real_count` MUST count only canonical
    /// `periodic_NNN` tags. A bridge with a real `periodic_000`
    /// alongside a real user `periodic_kaslr` capture MUST
    /// surface count = 1 — the user tag does NOT inflate the
    /// floor.
    #[test]
    fn periodic_real_count_ignores_user_tag_with_periodic_prefix() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        // Real periodic capture from the coordinator.
        let real_periodic = crate::monitor::dump::FailureDumpReport {
            schema: crate::monitor::dump::SCHEMA_SINGLE.to_string(),
            is_placeholder: false,
            ..Default::default()
        };
        bridge.store("periodic_000", real_periodic);
        // User-supplied CaptureSnapshot tag that happens to start
        // with "periodic_" — MUST NOT count.
        let user_capture = crate::monitor::dump::FailureDumpReport {
            schema: crate::monitor::dump::SCHEMA_SINGLE.to_string(),
            is_placeholder: false,
            ..Default::default()
        };
        bridge.store("periodic_kaslr", user_capture);
        // A placeholder under a canonical tag — counted as fired
        // but NOT as real.
        bridge.store(
            "periodic_001",
            crate::monitor::dump::FailureDumpReport::placeholder("rendezvous timed out"),
        );
        assert_eq!(
            bridge.periodic_real_count(),
            1,
            "only the canonical periodic_000 real capture counts; \
             user tag periodic_kaslr (even though real) must be \
             excluded by the strict matcher",
        );
    }
}

#[cfg(test)]
mod cgroup_procs_bridge_tests {
    use super::*;

    /// Pin the Arc-shared-state invariant for `SnapshotBridge.clone()`:
    /// a clone made BEFORE `set_thread_local` consumed the original
    /// sees writes that the consumed bridge made via `record_cgroup_procs`.
    /// This is the user-facing pattern in tests/cgroup_capture_procs_e2e.rs
    /// and the doc example on `Op::CaptureCgroupProcs`. A future refactor
    /// that accidentally turned `cgroup_procs: Arc<Mutex<_>>` into a
    /// non-Arc field would silently break the e2e drain pattern; this
    /// test catches it.
    #[test]
    fn snapshot_bridge_record_cgroup_procs_visible_via_arc_clone() {
        let bridge_a = SnapshotBridge::new(Arc::new(|_| None));
        let bridge_b = bridge_a.clone();
        bridge_a.record_cgroup_procs("tag".into(), "cg".into(), vec![1, 2]);
        let drained_b = bridge_b.drain_cgroup_procs();
        assert_eq!(drained_b.len(), 1);
        assert_eq!(drained_b[0].tag, "tag");
        assert_eq!(drained_b[0].cgroup, "cg");
        assert_eq!(drained_b[0].pids, vec![1, 2]);
        // A drained THROUGH B — pin that A also sees empty afterward
        // (single shared Vec under the Arc<Mutex>; drain via either
        // handle leaves both at empty).
        assert!(
            bridge_a.drain_cgroup_procs().is_empty(),
            "Arc-shared field — B's drain empties A too",
        );
    }

    /// Pin the consume-on-drain semantic of `drain_cgroup_procs`. The
    /// impl uses `std::mem::take` which leaves the source empty after
    /// the call returns. A future refactor swapping `mem::take` for
    /// `.clone()` would silently duplicate snapshots across drains;
    /// this test catches it.
    #[test]
    fn snapshot_bridge_drain_cgroup_procs_consumes_via_mem_take() {
        let bridge = SnapshotBridge::new(Arc::new(|_| None));
        bridge.record_cgroup_procs("t1".into(), "cg".into(), vec![1]);
        bridge.record_cgroup_procs("t2".into(), "cg".into(), vec![2]);
        let drained = bridge.drain_cgroup_procs();
        assert_eq!(drained.len(), 2);
        // Second drain MUST yield empty — mem::take left the source
        // empty after the first drain.
        let second = bridge.drain_cgroup_procs();
        assert!(second.is_empty(), "drain must consume; got: {second:?}");
        // After record-after-drain, only the new entries appear.
        bridge.record_cgroup_procs("t3".into(), "cg".into(), vec![3]);
        let third = bridge.drain_cgroup_procs();
        assert_eq!(third.len(), 1);
        assert_eq!(third[0].tag, "t3");
    }

    /// `cgroup_procs_by_tag` returns the FIRST matching snapshot
    /// without consuming the drain log. Mirrors `kernel_op_value(tag)`
    /// semantics. The drain log persists past the lookup so subsequent
    /// `drain_cgroup_procs` calls observe the same entries.
    #[test]
    fn snapshot_bridge_cgroup_procs_by_tag_is_non_draining() {
        let bridge = SnapshotBridge::new(Arc::new(|_| None));
        bridge.record_cgroup_procs("tag_a".into(), "cg_x".into(), vec![10, 20]);
        bridge.record_cgroup_procs("tag_b".into(), "cg_y".into(), vec![30]);
        // Lookup by tag returns the expected snapshot.
        let snap_a = bridge.cgroup_procs_by_tag("tag_a").expect("tag_a present");
        assert_eq!(snap_a.pids, vec![10, 20]);
        assert_eq!(snap_a.cgroup, "cg_x");
        // Unknown tag → None.
        assert!(bridge.cgroup_procs_by_tag("tag_unknown").is_none());
        // Non-draining: drain still returns both entries.
        let drained = bridge.drain_cgroup_procs();
        assert_eq!(drained.len(), 2);
        // After drain, by_tag finds nothing.
        assert!(bridge.cgroup_procs_by_tag("tag_a").is_none());
    }
}
