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
/// `freeze_and_capture`, and signals the reply completion with
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
/// (`src/vmm/freeze_coord.rs`), which resolves the symbol via a
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

/// Shared state owning the capture closure plus the captured-report
/// map.
///
/// Cloneable via the wrapped `Arc`s. The host installs an instance
/// in the executor's thread-local via `Self::set_thread_local`
/// before [`execute_steps`](crate::scenario::ops::execute_steps)
/// runs; the executor's `Op::Snapshot` arm calls
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
    /// `Op::Snapshot` was a no-op. Fires from
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
    pub(super) stats: HashMap<String, serde_json::Value>,
    /// Elapsed milliseconds since `run_start` at the moment the
    /// periodic capture fired. Same key set as `reports` for
    /// periodic tags; absent for non-periodic captures. Read by
    /// [`SnapshotBridge::drain_ordered_with_stats`] to populate
    /// `Sample::elapsed_ms` without recomputing.
    pub(super) elapsed_ms: HashMap<String, u64>,
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
    /// [`push_event`]; dropped count is tracked in `events_dropped`
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

/// Host-side capture pipeline that the freeze coordinator routes
/// [`Op::Snapshot`](crate::scenario::ops::Op::Snapshot) and
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
    pub(super) snapshots: Arc<Mutex<SnapshotStore>>,
    watch_count: Arc<std::sync::atomic::AtomicUsize>,
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
            snapshots: Arc::new(Mutex::new(SnapshotStore::new())),
            watch_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
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
                     exit_kind trigger). Drop a watch or use Op::Snapshot for a \
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

    /// Store a pre-built [`FailureDumpReport`] under `name`,
    /// bypassing the capture callback. Used by the host-side freeze
    /// coordinator after it runs `freeze_and_capture(false)` and
    /// wants to publish the resulting report on the bridge for the
    /// test author to drain post-VM-exit.
    ///
    /// Storage is capped at [`MAX_STORED_SNAPSHOTS`] entries to bound
    /// host memory under runaway capture cadence (e.g. a Loop step
    /// firing `Op::Snapshot` with a unique tag every iteration).
    /// When the cap is reached, the oldest stored entry is evicted
    /// with a `tracing::warn!` naming the dropped tag. An overwrite
    /// of an existing tag also warns and replaces the prior report
    /// in place without disturbing FIFO ordering of other entries.
    pub fn store(&self, name: &str, report: FailureDumpReport) {
        self.store_internal(name, report, None, None);
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
        stats: Option<serde_json::Value>,
        elapsed_ms: Option<u64>,
    ) {
        self.store_internal(name, report, stats, elapsed_ms);
    }

    fn store_internal(
        &self,
        name: &str,
        report: FailureDumpReport,
        stats: Option<serde_json::Value>,
        elapsed_ms: Option<u64>,
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
            return;
        }
        store.order.push_back(name.to_string());
        if let Some(v) = stats {
            store.stats.insert(name.to_string(), v);
        }
        if let Some(v) = elapsed_ms {
            store.elapsed_ms.insert(name.to_string(), v);
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
            // stats / elapsed entry cannot outlive its report.
            store.stats.remove(&evicted);
            store.elapsed_ms.remove(&evicted);
        }
    }

    /// Snapshot count for diagnostic logging.
    pub fn len(&self) -> usize {
        self.snapshots
            .lock_unpoisoned()
            .reports
            .len()
    }

    /// True when no snapshots have been captured.
    pub fn is_empty(&self) -> bool {
        self.snapshots
            .lock_unpoisoned()
            .reports
            .is_empty()
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
        self.snapshots
            .lock_unpoisoned()
            .reports
            .contains_key(name)
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
        // Stats / elapsed are dropped with the bridge — callers
        // that need the parallel data must use
        // `drain_ordered_with_stats` instead.
        store.stats.clear();
        store.elapsed_ms.clear();
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
    pub fn drain_ordered_with_stats(
        &self,
    ) -> Vec<(
        String,
        FailureDumpReport,
        Option<serde_json::Value>,
        Option<u64>,
    )> {
        let mut store = self.snapshots.lock_unpoisoned();
        let order = std::mem::take(&mut store.order);
        let mut reports = std::mem::take(&mut store.reports);
        let mut stats = std::mem::take(&mut store.stats);
        let mut elapsed = std::mem::take(&mut store.elapsed_ms);
        let mut out: Vec<(
            String,
            FailureDumpReport,
            Option<serde_json::Value>,
            Option<u64>,
        )> = Vec::with_capacity(order.len());
        for tag in order {
            if let Some(report) = reports.remove(&tag) {
                let s = stats.remove(&tag);
                let e = elapsed.remove(&tag);
                out.push((tag, report, s, e));
            }
        }
        // Defensive tail for desynchronised maps (matches
        // `drain_ordered`'s tail behaviour). Any stats / elapsed
        // entries that were not paired with a tag in `order` are
        // dropped because they have no anchoring report — surfacing
        // them as orphaned tuples would invent a structure no
        // consumer expects.
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
            let s = stats.remove(&tag);
            let e = elapsed.remove(&tag);
            out.push((tag, report, s, e));
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
    /// runs on the calling thread and `Op::Snapshot` only makes
    /// sense in that exact thread's call stack — installing a
    /// bridge process-wide would race against parallel test
    /// threads.
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

