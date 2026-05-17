//! Diagnostic snapshot capture and traversal.
//!
//! Test scenarios use [`Op::Snapshot`](crate::scenario::ops::Op::Snapshot)
//! to request a host-side diagnostic capture mid-run. The capture
//! result — a [`FailureDumpReport`] — is keyed by the `name` argument
//! and stored on the scenario's [`SnapshotBridge`], where downstream
//! test code reaches it via [`Snapshot`] for typed traversal of
//! BTF-rendered map values, per-CPU entries, and scalar variables.
//!
//! # Lifecycle
//!
//! 1. **Wire-up.** Before [`execute_steps`](crate::scenario::ops::execute_steps)
//!    runs, host orchestration installs a [`SnapshotBridge`] in the
//!    current thread via [`SnapshotBridge::set_thread_local`]. The
//!    bridge owns the storage map and a callable that performs the
//!    capture.
//!
//! 2. **Capture.** When the executor reaches `Op::Snapshot { name }`,
//!    it invokes [`SnapshotBridge::capture`] with the name. The
//!    closure performs the freeze rendezvous (request/reply with
//!    the freeze coordinator), builds a [`FailureDumpReport`], and
//!    returns it; the bridge stores it under the name.
//!
//! 3. **Inspection.** After the scenario completes, the test author
//!    pulls captured reports out via [`SnapshotBridge::drain`] and
//!    constructs [`Snapshot`] views to assert against rendered
//!    values:
//!    `snapshot.var("nr_cpus_onln").as_u64()? > 0`,
//!    `snapshot.map("scx_per_task")?.find(|e| e.get("tid").as_i64()? == pid)?`.
//!
//! # On-demand vs error-trigger captures
//!
//! `Op::Snapshot` requests are orthogonal to the error-class freeze
//! path. The freeze coordinator's existing state machine for
//! `SCX_EXIT_ERROR` triggers (Idle → TookEarly → Done) governs the
//! *unsolicited* capture pipeline; on-demand captures funnel
//! through a separate request/reply channel and never touch the
//! error-trigger state. The coordinator services on-demand requests
//! even after Done so post-failure scenarios can still snapshot
//! state for context. The serialisation rule: at most one capture in
//! flight at a time — the on-demand path waits for the previous
//! capture's vCPUs to fully return to `parked == false` before
//! issuing the next freeze request, mirroring the rendezvous
//! invariants the error-trigger path already obeys.
//!
//! # Guest → host wire: ioeventfd doorbell (locked)
//!
//! The guest-driven capture trigger uses an in-kernel ioeventfd
//! doorbell, NOT a synchronous MMIO `BusDevice` arm. Per user
//! direction:
//!
//! 1. Host registers an ioeventfd at a dedicated MMIO GPA inside
//!    the existing MMIO gap (e.g. `MMIO_GAP_START + 0x3000`) via
//!    `KVM_IOEVENTFD`. The exact GPA is arch-dependent —
//!    `MMIO_GAP_START + 0x3000` on x86_64,
//!    `VIRTIO_NET_MMIO_BASE + VIRTIO_MMIO_SIZE` on aarch64 — and
//!    the canonical value is exposed as `an internal MMIO doorbell GPA (deleted)`.
//!    The fd is owned by the freeze coordinator and polled
//!    alongside its existing wake sources.
//! 2. Guest [`Op::Snapshot`](crate::scenario::ops::Op::Snapshot)
//!    handler `mmap`s `/dev/mem` to reach the doorbell GPA (same
//!    pattern the SHM ring already uses) and writes the tag value
//!    plus a serial counter into a small per-call slot, then
//!    writes the doorbell. KVM dispatches the write in-kernel and
//!    raises the eventfd; the vCPU thread does NOT exit to
//!    userspace for the doorbell write itself.
//! 3. The freeze coordinator wakes on `eventfd_signal`, reads the
//!    tag from the slot, runs `freeze_and_capture`, builds the
//!    [`FailureDumpReport`], and stores it on the bridge keyed by
//!    that tag. Reply to the guest is implicit — the
//!    [`SnapshotBridge::capture`] callback installed in the
//!    executor's thread-local blocks on a per-request reply
//!    eventfd / completion channel paired with the doorbell.
//!
//! This shape keeps the capture trigger off the vCPU userspace
//! exit path (cleaner — no MMIO `BusDevice` round-trip) and is
//! extensible to higher-rate triggers without redesigning the
//! wire. The [`SnapshotBridge`] surface defined below is the
//! integration point; `ioeventfd` is the wake mechanism that
//! drives the `CaptureCallback` from the guest side. The guest
//! [`Op::WatchSnapshot`](crate::scenario::ops::Op::WatchSnapshot)
//! registration uses the same doorbell at scenario setup
//! (separate tag namespace) so symbol resolution + user
//! watchpoint slot allocation happen on the host without a vCPU
//! userspace exit.
//!
//! # No-bridge fallback
//!
//! When `Op::Snapshot` runs in a context with no installed bridge
//! (e.g. unit tests that exercise the executor without spinning up
//! a VM), the op is a no-op with a `tracing::warn!`. Existing
//! scenarios that do not declare snapshot ops keep working
//! unchanged.
//!
//! # Field accessor traversal
//!
//! [`SnapshotMap`], [`SnapshotEntry`], and [`SnapshotField`] form a
//! lazy borrow chain over the report. Dotted-path lookups (e.g.
//! `entry.get("ctx.weight.value")`) walk
//! [`RenderedValue::Struct`] members by name and follow
//! [`RenderedValue::Ptr`] dereferences transparently — the test
//! author writes the dotted path the BTF source would suggest;
//! pointer chasing is invisible.
//!
//! Missing fields land in [`SnapshotField::Missing`] with an
//! actionable error string identifying the path component that
//! could not be resolved AND the available alternatives at that
//! level. Terminal accessors (`as_u64`, `as_i64`, `as_bool`,
//! `as_str`) return `Result<T, SnapshotError>` so an absent /
//! type-mismatched field bubbles up as a recoverable error rather
//! than panicking.

use crate::monitor::arena::ArenaSnapshot;
use crate::monitor::bpf_prog::ProgRuntimeStats;
use crate::monitor::btf_render::{RenderedMember, RenderedValue};
use crate::monitor::dump::{
    EventCounterSample, FailureDumpEntry, FailureDumpFdArray, FailureDumpMap,
    FailureDumpPercpuEntry, FailureDumpPercpuHashEntry, FailureDumpReport, FailureDumpRingbuf,
    FailureDumpStackTrace, PerCpuTimeStats, PerNodeNumaStats, ProbeBssCounters,
};
use crate::monitor::scx_walker::{DsqState, RqScxState, ScxSchedState};
use crate::monitor::task_enrichment::TaskEnrichment;

/// Maximum number of rendered keys captured into
/// [`SnapshotError::NoMatch::available_keys`] during a failed
/// `find` / `max_by` traversal. Three is a balance between
/// disambiguation power (enough to suggest the keyspace shape) and
/// failure-message readability (does not overrun a terminal line).
const NO_MATCH_KEY_SAMPLE: usize = 3;

/// Maximum number of characters each rendered key in
/// [`SnapshotError::NoMatch::available_keys`] retains before being
/// truncated with a trailing `…`. Wide struct keys (e.g. a
/// 50-field `task_ctx`) would otherwise produce kilobytes of
/// failure text per sampled key.
const NO_MATCH_KEY_CHAR_CAP: usize = 80;

/// Discriminator that [`render_entry_key`]'s fallback path prepends
/// to the raw `key_hex` bytes when an entry's BTF-rendered key was
/// missing at capture time. [`SnapshotError::NoMatch`]'s `Display`
/// impl uses the same prefix as the gate for its BTF-missing hint
/// (when every sampled key starts with this string, BTF was
/// uniformly absent for the map's key type and the hint points the
/// operator at `CONFIG_DEBUG_INFO_BTF=y`). Naming the producer +
/// consumer contract once here keeps a future rename of one side
/// from silently desynchronising the other. Test sites in this
/// module intentionally retain the literal `"hex:"` so they pin the
/// value separately from the const that synchronises production.
pub(super) const HEX_KEY_PREFIX: &str = "hex:";


mod error;

pub use error::{SnapshotError, SnapshotResult};

mod bridge;

pub use bridge::{
    BridgeGuard, CaptureCallback, MAX_STORED_EVENTS, MAX_STORED_SNAPSHOTS, MAX_WATCH_SNAPSHOTS,
    SnapshotBridge, SnapshotBridgeEvent, WatchRegisterCallback, with_active_bridge,
};

// ---------------------------------------------------------------------------
// Snapshot view over a captured FailureDumpReport
// ---------------------------------------------------------------------------

/// Borrowed view over a captured [`FailureDumpReport`] for typed
/// traversal of BTF-rendered map values, per-CPU entries, and
/// scalar variables.
///
/// Constructed from a [`FailureDumpReport`] reference (typically
/// obtained via [`SnapshotBridge::drain`]); the view is cheap to
/// build — it does not copy the underlying report. Accessor
/// methods all return further borrowed views that walk the report
/// in place.
#[derive(Debug)]
#[must_use = "Snapshot is a borrowed view; bind or chain accessors"]
#[non_exhaustive]
pub struct Snapshot<'a> {
    report: &'a FailureDumpReport,
}

impl<'a> Snapshot<'a> {
    /// Build a borrowed view over `report`.
    pub fn new(report: &'a FailureDumpReport) -> Self {
        Self { report }
    }

    /// Underlying [`FailureDumpReport`] borrowed back to the caller.
    ///
    /// **Escape hatch.** Most consumers should reach for the typed
    /// accessors on [`Snapshot`] / [`SnapshotMap`] / [`SnapshotEntry`]
    /// / [`SnapshotField`], which route through [`SnapshotError`] and
    /// compose with the [`crate::assert::temporal`] patterns via
    /// [`SeriesField`](crate::assert::temporal::SeriesField). Use
    /// `report()` only when a [`FailureDumpReport`] field has no
    /// typed accessor yet:
    ///
    /// - `vcpu_regs` — per-vCPU register snapshot captured at the
    ///   freeze instant.
    /// - `vcpu_perf_at_freeze` — per-vCPU hardware perf counter
    ///   snapshot captured at the freeze instant.
    /// - `dump_truncated_at_us` — microseconds-into-the-dump at
    ///   which the soft deadline tripped.
    /// - `sdt_allocations`, `scx_static_ranges` — SDT allocator and
    ///   scx static memory layout snapshots used by the arena /
    ///   pointer-renderer pipelines.
    /// - `schema` — wire-format metadata
    ///   ([`Self::is_placeholder`] already wraps the boolean form).
    ///
    /// All other fields documented as escape-only on
    /// [`FailureDumpReport`] above now have first-class accessors on
    /// [`Snapshot`] (`event_counter_timeline`, `rq_scx_states`,
    /// `dsq_states`, `scx_sched_state`, `per_cpu_time`,
    /// `per_node_numa`, `task_enrichments`, `prog_runtime_stats`,
    /// `probe_counters`) and on [`SnapshotMap`] (`ringbuf`,
    /// `arena`, `fd_array`, `stack_trace`, `map_error`).
    ///
    /// Five `*_unavailable` diagnostic accessors cover the subset of
    /// walker-backed fields the dump pipeline writes a reason string
    /// for: [`Self::scx_walker_unavailable`] (shared by
    /// rq_scx_states / dsq_states / scx_sched_state — the scx
    /// walker writes one reason for the whole group),
    /// [`Self::task_enrichments_unavailable`],
    /// [`Self::prog_runtime_stats_unavailable`],
    /// [`Self::per_node_numa_unavailable`], and
    /// [`Self::sdt_alloc_unavailable`] (for the still-escape-only
    /// `sdt_allocations` field above). The remaining accessors
    /// (`event_counter_timeline`, `per_cpu_time`, `probe_counters`)
    /// have no companion diagnostic — empty / None is their only
    /// "no capture" signal.
    ///
    /// **Caveats of the bypass:**
    /// - No [`SnapshotError`] routing — call-site is on its own to
    ///   handle missing fields / type mismatches / per-CPU
    ///   narrowing.
    /// - No [`SeriesField`](crate::assert::temporal::SeriesField)
    ///   integration — temporal patterns
    ///   ([`nondecreasing`](crate::assert::temporal::SeriesField::nondecreasing),
    ///   [`rate_within`](crate::assert::temporal::SeriesField::rate_within),
    ///   etc.) cannot consume raw `FailureDumpReport` field values.
    /// - No placeholder-sample short-circuit
    ///   ([`Self::is_placeholder`] check is the caller's
    ///   responsibility).
    pub fn report(&self) -> &'a FailureDumpReport {
        self.report
    }

    /// Look up a BPF map by exact name. Returns
    /// [`SnapshotError::MapNotFound`] (with the captured map names
    /// in `available`) when no match is found.
    pub fn map(&self, name: &str) -> SnapshotResult<SnapshotMap<'a>> {
        for m in &self.report.maps {
            if m.name == name {
                return Ok(SnapshotMap { map: m, cpu: None });
            }
        }
        Err(SnapshotError::MapNotFound {
            requested: name.to_string(),
            available: self.report.maps.iter().map(|m| m.name.clone()).collect(),
        })
    }

    /// Walk the BTF-rendered fields of every `*.bss` / `*.data` /
    /// `*.rodata` global-section map for a top-level variable
    /// named `name`. Convenience for `.var("nr_cpus_onln")` style
    /// scalar reads without naming the section explicitly.
    ///
    /// Returns [`SnapshotField::Value`] on a unique match;
    /// [`SnapshotField::Missing`] with
    /// [`SnapshotError::VarNotFound`] (and the union of every
    /// global-section map's top-level member names in `available`)
    /// when no map exposes the name; or
    /// [`SnapshotError::AmbiguousVar`] when more than one
    /// global-section map exposes a top-level member with the same
    /// name. Two BPF objects sharing a global symbol — common when
    /// a scenario loads multiple progs into one report — would
    /// otherwise fall through to an arbitrary first match keyed off
    /// `report.maps` ordering, which depends on kernel IDR
    /// allocation order. Callers disambiguate via
    /// [`Self::map`] and walk the named map directly.
    pub fn var(&self, name: &str) -> SnapshotField<'a> {
        let mut hits: Vec<(&'a str, &'a RenderedValue)> = Vec::new();
        for m in &self.report.maps {
            if !is_global_section_map(&m.name) {
                continue;
            }
            if let Some(v) = m.value.as_ref()
                && let Some(found) = lookup_member(v, name)
            {
                hits.push((m.name.as_str(), found));
            }
        }
        match hits.len() {
            1 => SnapshotField::Value(hits[0].1),
            n if n > 1 => SnapshotField::Missing(SnapshotError::AmbiguousVar {
                requested: name.to_string(),
                found_in: hits.iter().map(|(name, _)| (*name).to_string()).collect(),
            }),
            _ => {
                let mut available: Vec<String> = Vec::new();
                for m in &self.report.maps {
                    if !is_global_section_map(&m.name) {
                        continue;
                    }
                    if let Some(RenderedValue::Struct { members, .. }) = m.value.as_ref() {
                        for member in members {
                            available.push(member.name.clone());
                        }
                    }
                }
                available.sort();
                available.dedup();
                SnapshotField::Missing(SnapshotError::VarNotFound {
                    requested: name.to_string(),
                    available,
                })
            }
        }
    }

    /// Number of maps captured in the report.
    pub fn map_count(&self) -> usize {
        self.report.maps.len()
    }

    /// True when the underlying [`FailureDumpReport`] is a
    /// placeholder produced by [`FailureDumpReport::placeholder`]
    /// — i.e. the freeze-rendezvous capture pipeline could not
    /// produce real data. Periodic-sample temporal patterns use
    /// this to skip the BPF axis on a placeholder sample (the
    /// stats axis, when present, may still be valid). Bypassing
    /// the projection-error path keeps the sample's diagnostic
    /// distinct from "field missing on a real capture".
    pub fn is_placeholder(&self) -> bool {
        self.report.is_placeholder
    }

    // -----------------------------------------------------------------
    // First-class accessors for fields the freeze-coordinator pipeline
    // populates on `FailureDumpReport` outside the BPF-map axis. Each
    // accessor returns either a borrowed slice (whole-vec views) or an
    // `Option<&T>` keyed by the natural identifier. Empty vec is the
    // normal state when the corresponding walker did not run — callers
    // check the companion `*_unavailable` field on the raw report for
    // the diagnostic reason. None on a keyed lookup means "the dump
    // did not capture an entry for that key"; it is not an error.
    //
    // **Keyed-lookup naming convention.** `<base>_at(<key>)` is used
    // when the key is a topology position (CPU index, NUMA node id)
    // that the kernel allocates densely from 0; the `_at` mirrors
    // `Vec::get(idx)` and reads naturally as "the row at this
    // position". `<base>_by_<field>(<value>)` is used when the key is
    // a sparse identifier (pid, program name) — the `_by_<field>`
    // names which field the lookup compares against and reads
    // naturally as "the entry whose <field> matches". The `<base>` is
    // normally the singular form of the plural-vec accessor (e.g.
    // `task_enrichments` → `task_enrichment_by_pid`), but stays
    // plural when the singular reads unnaturally (e.g.
    // `prog_runtime_stats` → `prog_runtime_stats_by_name` — the
    // singular `prog_runtime_stat` would be awkward English; the
    // `Stats` suffix is part of the canonical noun). Each keyed
    // accessor returns the first match in walker enumeration order;
    // production captures do not duplicate keys (kernel walker
    // invariants), but the contract is left first-match-wins so a
    // future duplicate-key scenario surfaces only one row without
    // panicking.
    // -----------------------------------------------------------------

    /// Per-monitor-tick SCX_EV_* event counter samples. Each entry is
    /// the cross-CPU sum of the 13 SCX event counters at one monitor
    /// tick. Empty when no `EventCounterCapture` ran, or every sample
    /// was suppressed (event-stat offsets unresolved, scx_root unset).
    ///
    /// Unlike the walker-backed accessors below, this field carries
    /// no `*_unavailable` companion: an empty timeline is the only
    /// signal for "no capture / no events".
    pub fn event_counter_timeline(&self) -> &'a [EventCounterSample] {
        &self.report.event_counter_timeline
    }

    /// Per-CPU `rq->scx` snapshots — one per CPU walked by
    /// [`crate::monitor::scx_walker`]. Empty when the
    /// `ScxWalkerCapture` was absent or every CPU's translate
    /// failed (see `FailureDumpReport::scx_walker_unavailable`).
    pub fn rq_scx_states(&self) -> &'a [RqScxState] {
        &self.report.rq_scx_states
    }

    /// Per-DSQ snapshots — local, bypass, global, and user DSQs
    /// reachable from `*scx_root`. Each entry carries `nr` (depth),
    /// `seq` (BPF-iter counter), and the queued task KVAs. Empty
    /// when the `ScxWalkerCapture` was absent (see
    /// `FailureDumpReport::scx_walker_unavailable`).
    pub fn dsq_states(&self) -> &'a [DsqState] {
        &self.report.dsq_states
    }

    /// Top-level `scx_sched` state captured from `*scx_root`:
    /// aborting flag, bypass_depth, exit_kind. `None` when no
    /// scheduler is attached or `*scx_root` was unreadable (see
    /// `FailureDumpReport::scx_walker_unavailable`).
    pub fn scx_sched_state(&self) -> Option<&'a ScxSchedState> {
        self.report.scx_sched_state.as_ref()
    }

    /// Per-CPU CPU-time / softirq / IRQ counter rows. One row per
    /// CPU enumerated by [`crate::monitor::dump::CpuTimeCapture`].
    /// Empty when the capture was not wired or symbol/BTF
    /// resolution failed.
    pub fn per_cpu_time(&self) -> &'a [PerCpuTimeStats] {
        &self.report.per_cpu_time
    }

    /// Per-CPU CPU-time row for CPU `cpu`, looked up by the `cpu`
    /// field on each [`PerCpuTimeStats`] (not by vec position).
    /// Returns `None` when no row matches — typical when the
    /// walker skipped that CPU, the capture didn't run, or `cpu`
    /// exceeded the topology. Returns the first match in walker
    /// enumeration order if `cpu` appears more than once.
    pub fn per_cpu_time_at(&self, cpu: u32) -> Option<&'a PerCpuTimeStats> {
        self.report.per_cpu_time.iter().find(|c| c.cpu == cpu)
    }

    /// Per-NUMA-node event counter rows captured from
    /// `pglist_data->node_zones[]->vm_numa_event[]`. Empty until
    /// the host-side NUMA walker lands (see
    /// `FailureDumpReport::per_node_numa_unavailable`).
    pub fn per_node_numa(&self) -> &'a [PerNodeNumaStats] {
        &self.report.per_node_numa
    }

    /// Per-NUMA-node event-counter row for `node`, looked up by
    /// the `node` field on each [`PerNodeNumaStats`]. Returns
    /// `None` when no row matches. Returns the first match in
    /// walker enumeration order if `node` appears more than once.
    pub fn per_node_numa_at(&self, node: u32) -> Option<&'a PerNodeNumaStats> {
        self.report.per_node_numa.iter().find(|n| n.node == node)
    }

    /// Per-task failure-dump enrichments — identity (pid, tgid,
    /// comm), process tree, scheduling priority, sched_class name,
    /// context-switch counters, watchdog disambiguation, lock
    /// slowpath stack matches. Empty when no task walker ran (see
    /// `FailureDumpReport::task_enrichments_unavailable`).
    pub fn task_enrichments(&self) -> &'a [TaskEnrichment] {
        &self.report.task_enrichments
    }

    /// Look up the enrichment for `pid`. The returned reference
    /// matches the first task whose `task_struct.pid` equals `pid`
    /// in walker enumeration order. Returns `None` when no task with
    /// that pid was captured. Production captures dedupe by task_kva
    /// before push, so duplicate-pid rows do not occur in real
    /// dumps.
    pub fn task_enrichment_by_pid(&self, pid: i32) -> Option<&'a TaskEnrichment> {
        self.report.task_enrichments.iter().find(|t| t.pid == pid)
    }

    /// Per-program BPF runtime stats — invocation count, total ns,
    /// recursion misses. One entry per struct_ops program reached
    /// by the prog walker. Empty when no struct_ops programs are
    /// loaded or the prog accessor was unavailable (see
    /// `FailureDumpReport::prog_runtime_stats_unavailable`).
    pub fn prog_runtime_stats(&self) -> &'a [ProgRuntimeStats] {
        &self.report.prog_runtime_stats
    }

    /// Look up the runtime stats for the program registered with
    /// `name` (kernel-side `bpf_prog->aux->name`). Returns `None`
    /// when no program with that name was captured. Returns the
    /// first match in walker enumeration order if `name` appears
    /// more than once — struct_ops programs in real captures use
    /// distinct callback names (`select_cpu`, `enqueue`, etc.) so
    /// duplicates do not occur in production.
    pub fn prog_runtime_stats_by_name(&self, name: &str) -> Option<&'a ProgRuntimeStats> {
        self.report.prog_runtime_stats.iter().find(|p| p.name == name)
    }

    /// Probe BPF program's per-CPU diagnostic counter snapshot.
    /// `None` when the probe's `.bss` map isn't enumerated (probe
    /// not loaded), the program BTF can't be parsed, or the
    /// array's offset doesn't resolve. A populated
    /// `trigger_count > 0` is the structural signal that the
    /// `tp_btf/sched_ext_exit` handler fired during the run.
    pub fn probe_counters(&self) -> Option<&'a ProbeBssCounters> {
        self.report.probe_counters.as_ref()
    }

    // -----------------------------------------------------------------
    // Companion `*_unavailable` diagnostic accessors. Each accessor
    // pairs with the walker-backed slice/option accessor above:
    // when the slice is empty (or the option is None), the matching
    // `*_unavailable()` returns `Some(reason)` if the walker
    // recorded one. `None` from the unavailable accessor means
    // either the walker ran normally (slice populated) or the field
    // is simply absent from the wire format (no reason recorded).
    // -----------------------------------------------------------------

    /// Diagnostic reason recorded when [`Self::rq_scx_states`] /
    /// [`Self::dsq_states`] / [`Self::scx_sched_state`] could not
    /// be populated. `None` when the walker fully succeeded;
    /// otherwise `Some(reason)` (e.g. `"scx_root null"`,
    /// `"no scx walker"`, or a partial-degradation string from the
    /// dump pipeline).
    pub fn scx_walker_unavailable(&self) -> Option<&'a str> {
        self.report.scx_walker_unavailable.as_deref()
    }

    /// Diagnostic reason recorded when [`Self::task_enrichments`]
    /// could not be populated. `None` when the walker yielded at
    /// least one enrichment; otherwise `Some(reason)`
    /// (e.g. `"no task walker available"`,
    /// `"task walker yielded zero tasks"`).
    pub fn task_enrichments_unavailable(&self) -> Option<&'a str> {
        self.report.task_enrichments_unavailable.as_deref()
    }

    /// Diagnostic reason recorded when [`Self::prog_runtime_stats`]
    /// could not be populated. `None` when the walker yielded at
    /// least one program; otherwise `Some(reason)`
    /// (e.g. `"prog accessor unavailable"`,
    /// `"no struct_ops programs loaded"`).
    pub fn prog_runtime_stats_unavailable(&self) -> Option<&'a str> {
        self.report.prog_runtime_stats_unavailable.as_deref()
    }

    /// Diagnostic reason recorded when [`Self::per_node_numa`]
    /// could not be populated — typically `"no NUMA walker"` until
    /// the host-side walker lands.
    pub fn per_node_numa_unavailable(&self) -> Option<&'a str> {
        self.report.per_node_numa_unavailable.as_deref()
    }

    /// Diagnostic reason recorded when the SDT allocator snapshot
    /// (still escape-only via [`Self::report`]) could not be
    /// populated.
    pub fn sdt_alloc_unavailable(&self) -> Option<&'a str> {
        self.report.sdt_alloc_unavailable.as_deref()
    }
}

/// True when a map name matches the libbpf-composed
/// `<obj>.<section>` naming for a global-section map.
fn is_global_section_map(name: &str) -> bool {
    name.ends_with(".bss") || name.ends_with(".data") || name.ends_with(".rodata")
}

// ---------------------------------------------------------------------------
// SnapshotMap
// ---------------------------------------------------------------------------

/// One map's view, possibly narrowed to a specific per-CPU slot via
/// [`Self::cpu`]. Returned by [`Snapshot::map`].
#[derive(Debug)]
#[must_use = "SnapshotMap is a borrowed view; chain accessors"]
#[non_exhaustive]
pub struct SnapshotMap<'a> {
    map: &'a FailureDumpMap,
    /// When `Some(cpu)`, subsequent [`Self::at`] /
    /// [`Self::find`] calls walk only the per-CPU slot for that
    /// CPU; `None` walks the natural (non-per-CPU) entry list.
    cpu: Option<usize>,
}

impl<'a> SnapshotMap<'a> {
    /// Map name as captured.
    pub fn name(&self) -> &'a str {
        &self.map.name
    }

    /// Underlying [`FailureDumpMap`].
    pub fn raw(&self) -> &'a FailureDumpMap {
        self.map
    }

    /// Ringbuf occupancy snapshot for `BPF_MAP_TYPE_RINGBUF` /
    /// `BPF_MAP_TYPE_USER_RINGBUF` maps — capacity, consumer /
    /// producer / pending positions, and the cumulative
    /// `pending_bytes` gap. `None` for non-ringbuf maps or when
    /// the BTF offsets for `bpf_ringbuf_map` / `bpf_ringbuf`
    /// weren't resolvable at capture time.
    pub fn ringbuf(&self) -> Option<&'a FailureDumpRingbuf> {
        self.map.ringbuf.as_ref()
    }

    /// Mapped-page snapshot for `BPF_MAP_TYPE_ARENA` maps. Borrows
    /// the per-page `(user_addr, bytes)` records plus the declared
    /// span / truncation flags. `None` for non-arena maps or when
    /// the arena walker failed to translate the user_vm window.
    pub fn arena(&self) -> Option<&'a ArenaSnapshot> {
        self.map.arena.as_ref()
    }

    /// Populated-slot summary for FD-array families (`PROG_ARRAY`,
    /// `PERF_EVENT_ARRAY`, `ARRAY_OF_MAPS`, `SOCKMAP*`, etc.).
    /// `None` for non-FD-array maps. Surfaces the populated count,
    /// scanned slot count, populated-index list, and the two
    /// truncation flags ([`FailureDumpFdArray::truncated`] for the
    /// scan limit, [`FailureDumpFdArray::indices_truncated`] for the
    /// index list limit).
    pub fn fd_array(&self) -> Option<&'a FailureDumpFdArray> {
        self.map.fd_array.as_ref()
    }

    /// Per-bucket summary for `BPF_MAP_TYPE_STACK_TRACE` maps.
    /// `None` for non-STACK_TRACE maps or when the BTF offsets for
    /// `bpf_stack_map` / `stack_map_bucket` weren't resolvable.
    pub fn stack_trace(&self) -> Option<&'a FailureDumpStackTrace> {
        self.map.stack_trace.as_ref()
    }

    /// Per-map decode-error string set by the freeze coordinator
    /// when this map's contents are missing or partial. `None` on a
    /// successful render. Distinct from [`SnapshotError`] (which
    /// flows through the accessor API) — `map_error` surfaces the
    /// capture-side diagnostic the kernel-walker recorded before
    /// the snapshot was handed to test code.
    pub fn map_error(&self) -> Option<&'a str> {
        self.map.error.as_deref()
    }

    /// Narrow this map view to a specific per-CPU slot. On a
    /// non-per-CPU map this is recorded but ignored when the
    /// underlying entries are not per-CPU. Use on
    /// `BPF_MAP_TYPE_PERCPU_ARRAY` / `BPF_MAP_TYPE_PERCPU_HASH` /
    /// `BPF_MAP_TYPE_LRU_PERCPU_HASH`.
    pub fn cpu(self, n: usize) -> SnapshotMap<'a> {
        SnapshotMap {
            map: self.map,
            cpu: Some(n),
        }
    }

    /// Get an entry by ordinal index.
    ///
    /// For HASH-style entry lists, returns the `n`-th
    /// [`FailureDumpEntry`] in the captured order. For per-CPU
    /// array maps narrowed via [`Self::cpu`], returns the entry
    /// at key `n` with its per-CPU slot pre-resolved. For ARRAY
    /// maps with a single value, `n == 0` returns the value.
    pub fn at(&self, n: usize) -> SnapshotEntry<'a> {
        let resolved = self.entry_at(n);
        match resolved {
            Ok(e) => e,
            Err(err) => SnapshotEntry::Missing(err),
        }
    }

    /// Find the first entry matching `predicate`. Returns
    /// [`SnapshotEntry::Missing`] with [`SnapshotError::NoMatch`]
    /// when no entry matches. The NoMatch payload carries the
    /// total entry count traversed and a small sample of rendered
    /// keys so the failure message can tell `empty map` apart from
    /// `populated map, predicate never matched`.
    pub fn find(&self, predicate: impl Fn(&SnapshotEntry<'a>) -> bool) -> SnapshotEntry<'a> {
        let mut len = 0usize;
        let mut available_keys: Vec<String> = Vec::with_capacity(NO_MATCH_KEY_SAMPLE);
        for entry in self.iter_entries() {
            if predicate(&entry) {
                return entry;
            }
            if available_keys.len() < NO_MATCH_KEY_SAMPLE
                && let Some(k) = render_entry_key(&entry)
            {
                available_keys.push(k);
            }
            len += 1;
        }
        SnapshotEntry::Missing(SnapshotError::NoMatch {
            map: self.map.name.clone(),
            op: "find".to_string(),
            len,
            available_keys,
        })
    }

    /// Collect every entry matching `predicate` into a Vec.
    pub fn filter(&self, predicate: impl Fn(&SnapshotEntry<'a>) -> bool) -> Vec<SnapshotEntry<'a>> {
        self.iter_entries().filter(|e| predicate(e)).collect()
    }

    /// Find the entry whose `key_fn` produces the maximum u64.
    /// Returns [`SnapshotEntry::Missing`] when the map has no
    /// entries. The NoMatch payload's `len` is 0 in that case;
    /// `available_keys` is empty (the map has no keys to sample).
    pub fn max_by(&self, key_fn: impl Fn(&SnapshotEntry<'a>) -> u64) -> SnapshotEntry<'a> {
        let mut best: Option<(u64, SnapshotEntry<'a>)> = None;
        for entry in self.iter_entries() {
            let k = key_fn(&entry);
            let beats = best.as_ref().is_none_or(|(prev, _)| k > *prev);
            if beats {
                best = Some((k, entry));
            }
        }
        match best {
            Some((_, e)) => e,
            None => SnapshotEntry::Missing(SnapshotError::NoMatch {
                map: self.map.name.clone(),
                op: "max_by".to_string(),
                len: 0,
                available_keys: Vec::new(),
            }),
        }
    }

    /// Iterator over every entry under this view. Used by
    /// [`Self::find`] / [`Self::filter`] / [`Self::max_by`].
    fn iter_entries(&self) -> Box<dyn Iterator<Item = SnapshotEntry<'a>> + 'a> {
        if !self.map.percpu_entries.is_empty() {
            let cpu = self.cpu;
            let map = self.map;
            return Box::new(
                map.percpu_entries
                    .iter()
                    .map(move |e| resolve_percpu_entry(map, e, cpu)),
            );
        }
        if !self.map.percpu_hash_entries.is_empty() {
            let cpu = self.cpu;
            let map = self.map;
            return Box::new(
                map.percpu_hash_entries
                    .iter()
                    .map(move |e| resolve_percpu_hash_entry(map, e, cpu)),
            );
        }
        if !self.map.entries.is_empty() {
            return Box::new(self.map.entries.iter().map(SnapshotEntry::Hash));
        }
        if let Some(v) = self.map.value.as_ref() {
            return Box::new(std::iter::once(SnapshotEntry::Value(v)));
        }
        Box::new(std::iter::empty())
    }

    /// Internal entry-by-index resolver returning a structured
    /// error for the surrounding [`Self::at`] arm.
    fn entry_at(&self, n: usize) -> SnapshotResult<SnapshotEntry<'a>> {
        if !self.map.percpu_entries.is_empty() {
            return resolve_percpu_entry_at(self.map, n, self.cpu);
        }
        if !self.map.percpu_hash_entries.is_empty() {
            return resolve_percpu_hash_entry_at(self.map, n, self.cpu);
        }
        if !self.map.entries.is_empty() {
            if n < self.map.entries.len() {
                return Ok(SnapshotEntry::Hash(&self.map.entries[n]));
            }
            return Err(SnapshotError::IndexOutOfRange {
                map: self.map.name.clone(),
                index: n,
                len: self.map.entries.len(),
            });
        }
        if let Some(v) = self.map.value.as_ref() {
            if n == 0 {
                return Ok(SnapshotEntry::Value(v));
            }
            return Err(SnapshotError::IndexOutOfRange {
                map: self.map.name.clone(),
                index: n,
                len: 1,
            });
        }
        Err(SnapshotError::IndexOutOfRange {
            map: self.map.name.clone(),
            index: n,
            len: 0,
        })
    }
}

fn resolve_percpu_entry_at<'a>(
    map: &'a FailureDumpMap,
    n: usize,
    cpu: Option<usize>,
) -> SnapshotResult<SnapshotEntry<'a>> {
    if n >= map.percpu_entries.len() {
        return Err(SnapshotError::IndexOutOfRange {
            map: map.name.clone(),
            index: n,
            len: map.percpu_entries.len(),
        });
    }
    Ok(resolve_percpu_entry(map, &map.percpu_entries[n], cpu))
}

fn resolve_percpu_entry<'a>(
    map: &'a FailureDumpMap,
    entry: &'a FailureDumpPercpuEntry,
    cpu: Option<usize>,
) -> SnapshotEntry<'a> {
    let Some(c) = cpu else {
        return SnapshotEntry::Percpu(entry);
    };
    if c >= entry.per_cpu.len() {
        return SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
            map: map.name.clone(),
            cpu: c,
            len: entry.per_cpu.len(),
            unmapped: false,
        });
    }
    match entry.per_cpu[c].as_ref() {
        Some(v) => SnapshotEntry::Value(v),
        None => SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
            map: map.name.clone(),
            cpu: c,
            len: entry.per_cpu.len(),
            unmapped: true,
        }),
    }
}

fn resolve_percpu_hash_entry_at<'a>(
    map: &'a FailureDumpMap,
    n: usize,
    cpu: Option<usize>,
) -> SnapshotResult<SnapshotEntry<'a>> {
    if n >= map.percpu_hash_entries.len() {
        return Err(SnapshotError::IndexOutOfRange {
            map: map.name.clone(),
            index: n,
            len: map.percpu_hash_entries.len(),
        });
    }
    Ok(resolve_percpu_hash_entry(
        map,
        &map.percpu_hash_entries[n],
        cpu,
    ))
}

fn resolve_percpu_hash_entry<'a>(
    map: &'a FailureDumpMap,
    entry: &'a FailureDumpPercpuHashEntry,
    cpu: Option<usize>,
) -> SnapshotEntry<'a> {
    let Some(c) = cpu else {
        return SnapshotEntry::PercpuHash(entry);
    };
    if c >= entry.per_cpu.len() {
        return SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
            map: map.name.clone(),
            cpu: c,
            len: entry.per_cpu.len(),
            unmapped: false,
        });
    }
    match entry.per_cpu[c].as_ref() {
        Some(v) => SnapshotEntry::Value(v),
        None => SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
            map: map.name.clone(),
            cpu: c,
            len: entry.per_cpu.len(),
            unmapped: true,
        }),
    }
}

/// Render a [`SnapshotEntry`]'s key into a bounded `String` suitable
/// for the [`SnapshotError::NoMatch::available_keys`] sample.
///
/// Returns `None` for [`SnapshotEntry::Value`] (single-value ARRAY
/// maps have no key surface) and [`SnapshotEntry::Missing`] (no
/// entry was produced). Hash / per-CPU-hash entries fall back to
/// the hex-encoded raw key bytes via the `hex:` prefix when BTF
/// rendering was absent at capture time. The result is truncated
/// to [`NO_MATCH_KEY_CHAR_CAP`] chars with a trailing `…` to keep
/// wide struct keys from overrunning failure-message lines.
fn render_entry_key(entry: &SnapshotEntry<'_>) -> Option<String> {
    let key = match entry {
        SnapshotEntry::Hash(e) => match e.key.as_ref() {
            Some(rv) => rv.to_string(),
            None => format!("{HEX_KEY_PREFIX}{}", e.key_hex),
        },
        SnapshotEntry::PercpuHash(e) => match e.key.as_ref() {
            Some(rv) => rv.to_string(),
            None => format!("{HEX_KEY_PREFIX}{}", e.key_hex),
        },
        SnapshotEntry::Percpu(e) => e.key.to_string(),
        SnapshotEntry::Value(_) | SnapshotEntry::Missing(_) => return None,
    };
    // Bytes-per-char is >= 1 in UTF-8, so byte-length <= char-cap implies
    // char-length <= char-cap — short-circuit the O(n) chars().count()
    // walk on the common ASCII case.
    if key.len() <= NO_MATCH_KEY_CHAR_CAP {
        return Some(key);
    }
    if key.chars().count() > NO_MATCH_KEY_CHAR_CAP {
        let mut truncated: String = key
            .chars()
            .take(NO_MATCH_KEY_CHAR_CAP.saturating_sub(1))
            .collect();
        truncated.push('…');
        Some(truncated)
    } else {
        Some(key)
    }
}

// ---------------------------------------------------------------------------
// SnapshotEntry
// ---------------------------------------------------------------------------

/// One entry's view — either a HASH (key, value) pair, a per-CPU
/// array entry, a per-CPU hash entry, a single rendered value, or
/// a missing-entry marker.
#[derive(Debug)]
#[must_use = "SnapshotEntry is a borrowed view; chain accessors"]
#[non_exhaustive]
pub enum SnapshotEntry<'a> {
    /// HASH map entry — `(key, value)` pair.
    Hash(&'a FailureDumpEntry),
    /// PERCPU_ARRAY entry — outer u32 key, inner per-CPU vec.
    Percpu(&'a FailureDumpPercpuEntry),
    /// PERCPU_HASH entry — rendered key, inner per-CPU vec.
    PercpuHash(&'a FailureDumpPercpuHashEntry),
    /// Single rendered value (ARRAY map's `value` field, or a
    /// per-CPU slot resolved via [`SnapshotMap::cpu`]).
    Value(&'a RenderedValue),
    /// No entry matched.
    Missing(SnapshotError),
}

impl<'a> SnapshotEntry<'a> {
    /// True when the lookup succeeded.
    pub fn is_present(&self) -> bool {
        !matches!(self, SnapshotEntry::Missing(_))
    }

    /// Walk into the entry's value side along a dotted path. Each
    /// path component names a [`RenderedValue::Struct`] member;
    /// pointer dereferences are followed transparently. Returns
    /// [`SnapshotField::Missing`] with an actionable error
    /// when the path cannot be resolved.
    pub fn get(&self, path: &str) -> SnapshotField<'a> {
        let value = match self {
            SnapshotEntry::Hash(e) => e.value.as_ref(),
            SnapshotEntry::Percpu(_) | SnapshotEntry::PercpuHash(_) => {
                let map_name = match self {
                    SnapshotEntry::Percpu(_) => "<percpu-array>".to_string(),
                    SnapshotEntry::PercpuHash(_) => "<percpu-hash>".to_string(),
                    _ => String::new(),
                };
                return SnapshotField::Missing(SnapshotError::PerCpuNotNarrowed { map: map_name });
            }
            SnapshotEntry::Value(v) => Some(*v),
            SnapshotEntry::Missing(err) => {
                return SnapshotField::Missing(err.clone());
            }
        };
        let Some(v) = value else {
            return SnapshotField::Missing(SnapshotError::NoRendered {
                map: "<entry>".to_string(),
                side: "value".to_string(),
            });
        };
        walk_dotted_path(v, path)
    }

    /// Look up the entry's KEY side along a dotted path. Mirror
    /// of [`Self::get`] but operates on the key's rendered
    /// structure (HASH / PERCPU_HASH only).
    pub fn key(&self, path: &str) -> SnapshotField<'a> {
        match self {
            SnapshotEntry::Hash(e) => match e.key.as_ref() {
                Some(v) => walk_dotted_path(v, path),
                None => SnapshotField::Missing(SnapshotError::NoRendered {
                    map: "<entry>".to_string(),
                    side: "key".to_string(),
                }),
            },
            SnapshotEntry::PercpuHash(e) => match e.key.as_ref() {
                Some(v) => walk_dotted_path(v, path),
                None => SnapshotField::Missing(SnapshotError::NoRendered {
                    map: "<entry>".to_string(),
                    side: "key".to_string(),
                }),
            },
            SnapshotEntry::Percpu(e) => {
                if path.is_empty() {
                    SnapshotField::PercpuKey { key: e.key }
                } else {
                    SnapshotField::Missing(SnapshotError::TypeMismatch {
                        expected: "Struct".to_string(),
                        actual: "Uint(percpu key)".to_string(),
                        requested: path.to_string(),
                    })
                }
            }
            SnapshotEntry::Value(_) => SnapshotField::Missing(SnapshotError::TypeMismatch {
                expected: "key".to_string(),
                actual: "single Value (no key)".to_string(),
                requested: path.to_string(),
            }),
            SnapshotEntry::Missing(err) => SnapshotField::Missing(err.clone()),
        }
    }

    // -----------------------------------------------------------------
    // Per-CPU aggregators. Apply only to `Percpu` / `PercpuHash`
    // entries; other variants return `Err(TypeMismatch)`. Inside the
    // per_cpu vec, slots whose value is `None` (CPU unmapped / out of
    // range — see `read_percpu_array_value` semantics) skip the
    // aggregation; slots whose rendered value can't decode to the
    // requested scalar return `Err(TypeMismatch)` immediately.
    //
    // `cpu_sum_*` returns `0` when no slot contributes (empty sum
    // identity). `cpu_max_*` / `cpu_min_*` return `Err(NoMatch)`
    // when no slot contributes (max / min of empty set has no
    // meaningful answer).
    // -----------------------------------------------------------------

    /// Sum the per-CPU values at `path` as `u64`. Returns `0` when
    /// every slot is `None` (no slot contributed). A slot whose
    /// rendered value cannot decode to `u64` propagates an Err
    /// immediately and stops the aggregation.
    pub fn cpu_sum_u64(&self, path: &str) -> SnapshotResult<u64> {
        let mut acc: u64 = 0;
        self.try_for_each_cpu_value(path, |v| {
            acc = acc.saturating_add(SnapshotField::Value(v).as_u64()?);
            Ok(())
        })?;
        Ok(acc)
    }

    /// Maximum of per-CPU values at `path` as `u64`. Returns
    /// `Err(NoMatch)` when every slot is `None` (no slot contributed).
    /// A slot whose rendered value cannot decode to `u64` propagates
    /// an Err immediately.
    pub fn cpu_max_u64(&self, path: &str) -> SnapshotResult<u64> {
        let mut best: Option<u64> = None;
        self.try_for_each_cpu_value(path, |v| {
            let n = SnapshotField::Value(v).as_u64()?;
            best = Some(best.map_or(n, |b| b.max(n)));
            Ok(())
        })?;
        best.ok_or_else(|| self.empty_aggregate_error("cpu_max_u64"))
    }

    /// Minimum of per-CPU values at `path` as `u64`. Returns
    /// `Err(NoMatch)` when every slot is `None`. A slot whose
    /// rendered value cannot decode to `u64` propagates an Err
    /// immediately.
    pub fn cpu_min_u64(&self, path: &str) -> SnapshotResult<u64> {
        let mut best: Option<u64> = None;
        self.try_for_each_cpu_value(path, |v| {
            let n = SnapshotField::Value(v).as_u64()?;
            best = Some(best.map_or(n, |b| b.min(n)));
            Ok(())
        })?;
        best.ok_or_else(|| self.empty_aggregate_error("cpu_min_u64"))
    }

    /// Sum the per-CPU values at `path` as `f64`. Returns `0.0`
    /// when every slot is `None`. A slot whose rendered value
    /// cannot decode to `f64` propagates an Err immediately. NaN
    /// slot values propagate through `+=` per IEEE-754 — a single
    /// NaN slot makes the result NaN.
    pub fn cpu_sum_f64(&self, path: &str) -> SnapshotResult<f64> {
        let mut acc: f64 = 0.0;
        self.try_for_each_cpu_value(path, |v| {
            acc += SnapshotField::Value(v).as_f64()?;
            Ok(())
        })?;
        Ok(acc)
    }

    /// Maximum of per-CPU values at `path` as `f64`. Returns
    /// `Err(NoMatch)` when every slot is `None`. A slot whose
    /// rendered value cannot decode to `f64` propagates an Err
    /// immediately. NaN slot values are filtered out per
    /// `f64::max` semantics — `f64::max(NaN, x)` returns `x`, so a
    /// NaN slot never wins against a non-NaN slot. An all-NaN run
    /// is an edge case: the first NaN slot sets `best=NaN`, then
    /// subsequent `NaN.max(NaN)` returns NaN, so the final result
    /// is `Ok(NaN)` rather than NoMatch.
    pub fn cpu_max_f64(&self, path: &str) -> SnapshotResult<f64> {
        let mut best: Option<f64> = None;
        self.try_for_each_cpu_value(path, |v| {
            let n = SnapshotField::Value(v).as_f64()?;
            best = Some(best.map_or(n, |b| b.max(n)));
            Ok(())
        })?;
        best.ok_or_else(|| self.empty_aggregate_error("cpu_max_f64"))
    }

    /// Minimum of per-CPU values at `path` as `f64`. Returns
    /// `Err(NoMatch)` when every slot is `None`. A slot whose
    /// rendered value cannot decode to `f64` propagates an Err
    /// immediately. NaN slot values are filtered out per
    /// `f64::min` semantics — `f64::min(NaN, x)` returns `x`, so a
    /// NaN slot never wins against a non-NaN slot. An all-NaN run
    /// yields `Ok(NaN)` rather than NoMatch — same edge case as
    /// `cpu_max_f64`.
    pub fn cpu_min_f64(&self, path: &str) -> SnapshotResult<f64> {
        let mut best: Option<f64> = None;
        self.try_for_each_cpu_value(path, |v| {
            let n = SnapshotField::Value(v).as_f64()?;
            best = Some(best.map_or(n, |b| b.min(n)));
            Ok(())
        })?;
        best.ok_or_else(|| self.empty_aggregate_error("cpu_min_f64"))
    }

    /// Iterate non-None per-CPU rendered values at `path`. For each
    /// successful slot, invokes `f(cpu_idx, &RenderedValue)`. Slots
    /// whose value is `None` are skipped silently; the iteration
    /// stops at the first slot whose value cannot be reached via
    /// `path` (returning the path-walk error). Returns `Err` for
    /// non-percpu variants.
    pub fn cpu_each<F>(&self, path: &str, mut f: F) -> SnapshotResult<()>
    where
        F: FnMut(usize, &'a RenderedValue) -> SnapshotResult<()>,
    {
        let per_cpu: &[Option<RenderedValue>] = match self {
            SnapshotEntry::Percpu(e) => &e.per_cpu,
            SnapshotEntry::PercpuHash(e) => &e.per_cpu,
            SnapshotEntry::Hash(_) | SnapshotEntry::Value(_) => {
                return Err(SnapshotError::TypeMismatch {
                    expected: "Percpu / PercpuHash".to_string(),
                    actual: self.variant_name().to_string(),
                    requested: path.to_string(),
                });
            }
            SnapshotEntry::Missing(err) => return Err(err.clone()),
        };
        for (cpu_idx, slot) in per_cpu.iter().enumerate() {
            let Some(rendered) = slot.as_ref() else {
                continue;
            };
            let walked = walk_dotted_path(rendered, path);
            let value = match walked {
                SnapshotField::Value(v) => v,
                SnapshotField::PercpuKey { .. } => {
                    return Err(SnapshotError::TypeMismatch {
                        expected: "rendered value".to_string(),
                        actual: "PercpuKey".to_string(),
                        requested: path.to_string(),
                    });
                }
                SnapshotField::Missing(err) => return Err(err),
            };
            f(cpu_idx, value)?;
        }
        Ok(())
    }

    /// Shared walk helper for `cpu_sum_*` / `cpu_max_*` / `cpu_min_*`
    /// — invokes `f` on every non-None slot's rendered value.
    fn try_for_each_cpu_value<F>(&self, path: &str, mut f: F) -> SnapshotResult<()>
    where
        F: FnMut(&'a RenderedValue) -> SnapshotResult<()>,
    {
        self.cpu_each(path, |_, v| f(v))
    }

    /// Name for diagnostic messages. Used by the per-CPU aggregator
    /// `TypeMismatch` paths so the error names the actual variant.
    fn variant_name(&self) -> &'static str {
        match self {
            SnapshotEntry::Hash(_) => "Hash",
            SnapshotEntry::Percpu(_) => "Percpu",
            SnapshotEntry::PercpuHash(_) => "PercpuHash",
            SnapshotEntry::Value(_) => "Value",
            SnapshotEntry::Missing(_) => "Missing",
        }
    }

    /// Build the `NoMatch` error for an empty per-CPU aggregate
    /// (max / min of all-None or all-decode-fail). `op` names the
    /// caller so the error message points at the right method.
    fn empty_aggregate_error(&self, op: &str) -> SnapshotError {
        SnapshotError::NoMatch {
            map: format!("<{}>", self.variant_name()),
            op: op.to_string(),
            len: 0,
            available_keys: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// SnapshotField — terminal traversal value
// ---------------------------------------------------------------------------

/// One field's view at the leaf of a dotted-path walk.
///
/// Returned by [`Snapshot::var`], [`SnapshotEntry::get`], and
/// [`SnapshotEntry::key`]. Terminal `as_*` accessors return
/// [`SnapshotResult`] so a missing or type-mismatched field
/// surfaces as a recoverable error rather than a panic.
#[derive(Debug)]
#[must_use = "SnapshotField is a borrowed view; call as_u64 / as_i64 / etc. to extract"]
#[non_exhaustive]
pub enum SnapshotField<'a> {
    /// Resolved rendered value at the leaf of the path walk.
    Value(&'a RenderedValue),
    /// Dedicated per-CPU array key shape (u32, no struct).
    PercpuKey { key: u32 },
    /// Path could not be resolved.
    Missing(SnapshotError),
}

impl<'a> SnapshotField<'a> {
    /// Walk into a sub-field. Composable with
    /// [`SnapshotEntry::get`].
    pub fn get(&self, path: &str) -> SnapshotField<'a> {
        match self {
            SnapshotField::Value(v) => walk_dotted_path(v, path),
            SnapshotField::PercpuKey { .. } => {
                SnapshotField::Missing(SnapshotError::TypeMismatch {
                    expected: "Struct".to_string(),
                    actual: "Uint(percpu key)".to_string(),
                    requested: path.to_string(),
                })
            }
            SnapshotField::Missing(err) => SnapshotField::Missing(err.clone()),
        }
    }

    /// True when the field resolved successfully.
    pub fn is_present(&self) -> bool {
        !matches!(self, SnapshotField::Missing(_))
    }

    /// Read as `u64`. Accepts [`RenderedValue::Uint`],
    /// [`RenderedValue::Int`] (errors on negative),
    /// [`RenderedValue::Bool`] (0/1), [`RenderedValue::Char`]
    /// (raw byte), [`RenderedValue::Enum`] (raw enum integer),
    /// [`RenderedValue::Ptr`] (pointer value), and the
    /// percpu-array u32 key.
    pub fn as_u64(&self) -> SnapshotResult<u64> {
        match self {
            SnapshotField::Value(v) => render_to_u64(v),
            SnapshotField::PercpuKey { key } => Ok(u64::from(*key)),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `i64`.
    pub fn as_i64(&self) -> SnapshotResult<i64> {
        match self {
            SnapshotField::Value(v) => render_to_i64(v),
            SnapshotField::PercpuKey { key } => Ok(i64::from(*key)),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `bool`. [`RenderedValue::Bool`] direct, ints / enums
    /// non-zero is true.
    pub fn as_bool(&self) -> SnapshotResult<bool> {
        match self {
            SnapshotField::Value(v) => match v {
                RenderedValue::Bool { value } => Ok(*value),
                RenderedValue::Int { value, .. } => Ok(*value != 0),
                RenderedValue::Uint { value, .. } => Ok(*value != 0),
                RenderedValue::Char { value } => Ok(*value != 0),
                RenderedValue::Enum { value, .. } => Ok(*value != 0),
                RenderedValue::Ptr { value, .. } => Ok(*value != 0),
                other => Err(SnapshotError::TypeMismatch {
                    expected: "bool".to_string(),
                    actual: describe_kind(other),
                    requested: String::new(),
                }),
            },
            SnapshotField::PercpuKey { key } => Ok(*key != 0),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `f64`.
    pub fn as_f64(&self) -> SnapshotResult<f64> {
        match self {
            SnapshotField::Value(v) => match v {
                RenderedValue::Float { value, .. } => Ok(*value),
                RenderedValue::Int { value, .. } => Ok(*value as f64),
                RenderedValue::Uint { value, .. } => Ok(*value as f64),
                RenderedValue::Enum { value, .. } => Ok(*value as f64),
                other => Err(SnapshotError::TypeMismatch {
                    expected: "f64".to_string(),
                    actual: describe_kind(other),
                    requested: String::new(),
                }),
            },
            SnapshotField::PercpuKey { key } => Ok(f64::from(*key)),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read the variant string for an [`RenderedValue::Enum`] with
    /// a resolved variant name.
    pub fn as_str(&self) -> SnapshotResult<&'a str> {
        match self {
            SnapshotField::Value(v) => match v {
                RenderedValue::Enum {
                    variant: Some(name),
                    ..
                } => Ok(name.as_str()),
                other => Err(SnapshotError::TypeMismatch {
                    expected: "str (enum variant name)".to_string(),
                    actual: describe_kind(other),
                    requested: String::new(),
                }),
            },
            SnapshotField::PercpuKey { .. } => Err(SnapshotError::TypeMismatch {
                expected: "str".to_string(),
                actual: "Uint(percpu key)".to_string(),
                requested: String::new(),
            }),
            SnapshotField::Missing(err) => Err(err.clone()),
        }
    }

    /// Underlying rendered value if present.
    pub fn rendered(&self) -> Option<&'a RenderedValue> {
        match self {
            SnapshotField::Value(v) => Some(v),
            _ => None,
        }
    }

    /// Error reference when the field is missing; `None`
    /// otherwise.
    pub fn error(&self) -> Option<&SnapshotError> {
        match self {
            SnapshotField::Missing(err) => Some(err),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// JSON dotted-path accessor (mirrors SnapshotField for stats values)
// ---------------------------------------------------------------------------

/// One value's view at the leaf of a dotted-path walk over a
/// [`serde_json::Value`]. Returned by [`stats_path`] / `StatsValue::path`.
///
/// Mirrors the [`SnapshotField`] shape so test authors who already
/// know the BPF-snapshot accessor surface get the same `as_u64` /
/// `as_i64` / `as_f64` / `as_bool` / `as_str` terminals on the
/// scx_stats JSON projection. Errors flow through the same
/// [`SnapshotError`] variants — `FieldNotFound` carries the
/// available object keys, `NotAStruct` flags a non-object cursor,
/// `TypeMismatch` reports the actual JSON shape — so failure-path
/// rendering in temporal assertions is identical regardless of
/// which side of the
/// [`Sample`](crate::scenario::sample::Sample) bundle the lookup
/// originated on.
#[derive(Debug, Clone)]
#[must_use = "JsonField is a borrowed view; call as_u64 / as_i64 / etc. to extract"]
#[non_exhaustive]
pub enum JsonField<'a> {
    /// Resolved JSON value at the leaf of the path walk.
    Value(&'a serde_json::Value),
    /// Path could not be resolved.
    Missing(SnapshotError),
}

impl<'a> JsonField<'a> {
    /// True when the path resolved.
    pub fn is_present(&self) -> bool {
        !matches!(self, JsonField::Missing(_))
    }

    /// Underlying JSON value if present.
    pub fn raw(&self) -> Option<&'a serde_json::Value> {
        match self {
            JsonField::Value(v) => Some(*v),
            JsonField::Missing(_) => None,
        }
    }

    /// Error reference when the path could not be resolved.
    pub fn error(&self) -> Option<&SnapshotError> {
        match self {
            JsonField::Missing(err) => Some(err),
            _ => None,
        }
    }

    /// Walk further into a sub-field. Composable with the result of
    /// [`stats_path`] — `stats_path(v, "layers").path("batch.util")`
    /// is the canonical "drill into a periodic-stats object" shape.
    pub fn path(&self, path: &str) -> JsonField<'a> {
        match self {
            JsonField::Value(v) => walk_json_path(v, path),
            JsonField::Missing(err) => JsonField::Missing(err.clone()),
        }
    }

    /// Read as `u64`. Accepts JSON integers (positive only), JSON
    /// booleans (true → 1, false → 0), and JSON strings whose
    /// content parses as a u64 (scx_stats sometimes stringifies
    /// large counters to avoid 53-bit float collapse). Returns
    /// [`SnapshotError::TypeMismatch`] otherwise.
    pub fn as_u64(&self) -> SnapshotResult<u64> {
        match self {
            JsonField::Value(v) => json_to_u64(v),
            JsonField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `i64`. Accepts JSON integers (any sign), JSON
    /// booleans (true → 1, false → 0), and JSON strings whose
    /// content parses as an i64.
    pub fn as_i64(&self) -> SnapshotResult<i64> {
        match self {
            JsonField::Value(v) => json_to_i64(v),
            JsonField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `f64`. Accepts JSON numbers (integers and
    /// floating-point) and JSON strings whose content parses as
    /// f64.
    pub fn as_f64(&self) -> SnapshotResult<f64> {
        match self {
            JsonField::Value(v) => json_to_f64(v),
            JsonField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `bool`. Accepts JSON booleans directly; rejects
    /// everything else. Distinct from `as_u64() != 0` so the call
    /// site reads honestly: a `bool` claim wants a JSON `true`/
    /// `false`, not a stringified `"1"` that happens to parse.
    pub fn as_bool(&self) -> SnapshotResult<bool> {
        match self {
            JsonField::Value(serde_json::Value::Bool(b)) => Ok(*b),
            JsonField::Value(other) => Err(SnapshotError::TypeMismatch {
                expected: "bool".to_string(),
                actual: describe_json_kind(other),
                requested: String::new(),
            }),
            JsonField::Missing(err) => Err(err.clone()),
        }
    }

    /// Read as `&str`. Accepts JSON strings only.
    pub fn as_str(&self) -> SnapshotResult<&'a str> {
        match self {
            JsonField::Value(serde_json::Value::String(s)) => Ok(s.as_str()),
            JsonField::Value(other) => Err(SnapshotError::TypeMismatch {
                expected: "str".to_string(),
                actual: describe_json_kind(other),
                requested: String::new(),
            }),
            JsonField::Missing(err) => Err(err.clone()),
        }
    }
}

/// Build a [`JsonField`] view rooted at `value` and walk along the
/// dotted path. An empty path returns the root unchanged so a
/// caller writing `stats_path(v, "").as_f64()` (e.g. for a
/// scalar-rooted stats response) hits the typed scalar accessor
/// directly.
///
/// Mirrors [`Snapshot::var`] / [`SnapshotEntry::get`] in error
/// shape: typos and missing keys surface as
/// [`SnapshotError::FieldNotFound`] with the available sibling
/// keys at the failing depth — the same diagnostic experience the
/// BPF-snapshot side already provides. scx_stats payloads commonly
/// nest layer / cgroup / cpu maps under top-level keys, so the
/// dotted form `"layers.batch.util"` is the canonical drill-down
/// for layered scheduler stats.
pub fn stats_path<'a>(value: &'a serde_json::Value, path: &str) -> JsonField<'a> {
    walk_json_path(value, path)
}

fn walk_json_path<'a>(root: &'a serde_json::Value, path: &str) -> JsonField<'a> {
    if path.is_empty() {
        return JsonField::Value(root);
    }
    let mut cursor: &serde_json::Value = root;
    let mut walked = String::new();
    for component in path.split('.') {
        if component.is_empty() {
            return JsonField::Missing(SnapshotError::EmptyPathComponent {
                requested: path.to_string(),
            });
        }
        match cursor {
            serde_json::Value::Object(map) => {
                let Some(next) = map.get(component) else {
                    let mut available: Vec<String> = map.keys().cloned().collect();
                    available.sort();
                    return JsonField::Missing(SnapshotError::FieldNotFound {
                        requested: path.to_string(),
                        walked: walked.clone(),
                        component: component.to_string(),
                        available,
                    });
                };
                cursor = next;
            }
            other => {
                return JsonField::Missing(SnapshotError::NotAStruct {
                    requested: path.to_string(),
                    walked: walked.clone(),
                    component: component.to_string(),
                    kind: describe_json_kind(other),
                });
            }
        }
        if !walked.is_empty() {
            walked.push('.');
        }
        walked.push_str(component);
    }
    JsonField::Value(cursor)
}

fn describe_json_kind(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "Null",
        serde_json::Value::Bool(_) => "Bool",
        serde_json::Value::Number(_) => "Number",
        serde_json::Value::String(_) => "String",
        serde_json::Value::Array(_) => "Array",
        serde_json::Value::Object(_) => "Object",
    }
    .to_string()
}

fn json_to_u64(v: &serde_json::Value) -> SnapshotResult<u64> {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Ok(u)
            } else if let Some(i) = n.as_i64() {
                if i < 0 {
                    Err(SnapshotError::TypeMismatch {
                        expected: "u64".to_string(),
                        actual: "Int(negative)".to_string(),
                        requested: String::new(),
                    })
                } else {
                    Ok(i as u64)
                }
            } else if let Some(f) = n.as_f64() {
                if !f.is_finite() || f < 0.0 {
                    Err(SnapshotError::TypeMismatch {
                        expected: "u64".to_string(),
                        actual: "Float(non-coercible)".to_string(),
                        requested: String::new(),
                    })
                } else if f.fract() != 0.0 {
                    Err(SnapshotError::TypeMismatch {
                        expected: "integer".to_string(),
                        actual: "non-integer float".to_string(),
                        requested: String::new(),
                    })
                } else {
                    Ok(f as u64)
                }
            } else {
                Err(SnapshotError::TypeMismatch {
                    expected: "u64".to_string(),
                    actual: "Number(unrepresentable)".to_string(),
                    requested: String::new(),
                })
            }
        }
        serde_json::Value::Bool(b) => Ok(u64::from(*b)),
        serde_json::Value::String(s) => s.parse::<u64>().map_err(|_| SnapshotError::TypeMismatch {
            expected: "u64".to_string(),
            actual: "String(non-numeric)".to_string(),
            requested: String::new(),
        }),
        other => Err(SnapshotError::TypeMismatch {
            expected: "u64".to_string(),
            actual: describe_json_kind(other),
            requested: String::new(),
        }),
    }
}

fn json_to_i64(v: &serde_json::Value) -> SnapshotResult<i64> {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i)
            } else if let Some(u) = n.as_u64() {
                if u > i64::MAX as u64 {
                    Err(SnapshotError::TypeMismatch {
                        expected: "i64".to_string(),
                        actual: "Uint(>i64::MAX)".to_string(),
                        requested: String::new(),
                    })
                } else {
                    Ok(u as i64)
                }
            } else if let Some(f) = n.as_f64() {
                if !f.is_finite() {
                    Err(SnapshotError::TypeMismatch {
                        expected: "i64".to_string(),
                        actual: "Float(non-finite)".to_string(),
                        requested: String::new(),
                    })
                } else if f.fract() != 0.0 {
                    Err(SnapshotError::TypeMismatch {
                        expected: "integer".to_string(),
                        actual: "non-integer float".to_string(),
                        requested: String::new(),
                    })
                } else {
                    Ok(f as i64)
                }
            } else {
                Err(SnapshotError::TypeMismatch {
                    expected: "i64".to_string(),
                    actual: "Number(unrepresentable)".to_string(),
                    requested: String::new(),
                })
            }
        }
        serde_json::Value::Bool(b) => Ok(i64::from(*b)),
        serde_json::Value::String(s) => s.parse::<i64>().map_err(|_| SnapshotError::TypeMismatch {
            expected: "i64".to_string(),
            actual: "String(non-numeric)".to_string(),
            requested: String::new(),
        }),
        other => Err(SnapshotError::TypeMismatch {
            expected: "i64".to_string(),
            actual: describe_json_kind(other),
            requested: String::new(),
        }),
    }
}

fn json_to_f64(v: &serde_json::Value) -> SnapshotResult<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64().ok_or(SnapshotError::TypeMismatch {
            expected: "f64".to_string(),
            actual: "Number(unrepresentable)".to_string(),
            requested: String::new(),
        }),
        serde_json::Value::String(s) => s.parse::<f64>().map_err(|_| SnapshotError::TypeMismatch {
            expected: "f64".to_string(),
            actual: "String(non-numeric)".to_string(),
            requested: String::new(),
        }),
        other => Err(SnapshotError::TypeMismatch {
            expected: "f64".to_string(),
            actual: describe_json_kind(other),
            requested: String::new(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Dotted-path walker
// ---------------------------------------------------------------------------

/// Walk a [`RenderedValue`] tree along a dotted path. Each
/// component matches a [`RenderedMember::name`] inside a
/// [`RenderedValue::Struct`]; [`RenderedValue::Ptr`] dereferences
/// are followed transparently. An empty path returns the root.
pub(crate) fn walk_dotted_path<'a>(root: &'a RenderedValue, path: &str) -> SnapshotField<'a> {
    if path.is_empty() {
        return SnapshotField::Value(root);
    }
    let mut cursor: &RenderedValue = root;
    let mut walked = String::new();
    for component in path.split('.') {
        if component.is_empty() {
            return SnapshotField::Missing(SnapshotError::EmptyPathComponent {
                requested: path.to_string(),
            });
        }
        cursor = peel_pointer(cursor);
        let RenderedValue::Struct { members, .. } = cursor else {
            return SnapshotField::Missing(SnapshotError::NotAStruct {
                requested: path.to_string(),
                walked: walked.clone(),
                component: component.to_string(),
                kind: describe_kind(cursor),
            });
        };
        let next = members.iter().find(|m| m.name == component);
        let Some(member) = next else {
            let names: Vec<String> = members.iter().map(|m| m.name.clone()).collect();
            return SnapshotField::Missing(SnapshotError::FieldNotFound {
                requested: path.to_string(),
                walked: walked.clone(),
                component: component.to_string(),
                available: names,
            });
        };
        cursor = &member.value;
        if !walked.is_empty() {
            walked.push('.');
        }
        walked.push_str(component);
    }
    SnapshotField::Value(cursor)
}

/// Look up a single top-level member by exact name. Used by
/// [`Snapshot::var`].
fn lookup_member<'a>(value: &'a RenderedValue, name: &str) -> Option<&'a RenderedValue> {
    let v = peel_pointer(value);
    let RenderedValue::Struct { members, .. } = v else {
        return None;
    };
    members
        .iter()
        .find(|m: &&RenderedMember| m.name == name)
        .map(|m| &m.value)
}

/// Peel through any [`RenderedValue::Ptr`] layers whose `deref`
/// is `Some`. Stops at the first non-pointer (or a pointer
/// without a chased deref).
fn peel_pointer(mut v: &RenderedValue) -> &RenderedValue {
    let mut steps = 0;
    while let RenderedValue::Ptr {
        deref: Some(inner), ..
    } = v
    {
        v = inner.as_ref();
        steps += 1;
        if steps > 16 {
            break;
        }
    }
    v
}

/// Human-readable variant name used in error messages.
fn describe_kind(v: &RenderedValue) -> String {
    match v {
        RenderedValue::Int { .. } => "Int",
        RenderedValue::Uint { .. } => "Uint",
        RenderedValue::Bool { .. } => "Bool",
        RenderedValue::Char { .. } => "Char",
        RenderedValue::Float { .. } => "Float",
        RenderedValue::Enum { .. } => "Enum",
        RenderedValue::Struct { .. } => "Struct",
        RenderedValue::Array { .. } => "Array",
        RenderedValue::CpuList { .. } => "CpuList",
        RenderedValue::Ptr { .. } => "Ptr",
        RenderedValue::Bytes { .. } => "Bytes",
        RenderedValue::Truncated { .. } => "Truncated",
        RenderedValue::Unsupported { .. } => "Unsupported",
    }
    .to_string()
}

/// Shared u64 coercion used by [`SnapshotField::as_u64`].
fn render_to_u64(v: &RenderedValue) -> SnapshotResult<u64> {
    match v {
        RenderedValue::Uint { value, .. } => Ok(*value),
        RenderedValue::Int { value, .. } => {
            if *value < 0 {
                Err(SnapshotError::TypeMismatch {
                    expected: "u64".to_string(),
                    actual: "Int(negative)".to_string(),
                    requested: String::new(),
                })
            } else {
                Ok(*value as u64)
            }
        }
        RenderedValue::Bool { value } => Ok(u64::from(*value)),
        RenderedValue::Char { value } => Ok(u64::from(*value)),
        RenderedValue::Enum { value, .. } => {
            if *value < 0 {
                Err(SnapshotError::TypeMismatch {
                    expected: "u64".to_string(),
                    actual: "Enum(negative)".to_string(),
                    requested: String::new(),
                })
            } else {
                Ok(*value as u64)
            }
        }
        RenderedValue::Ptr { value, .. } => Ok(*value),
        other => Err(SnapshotError::TypeMismatch {
            expected: "u64".to_string(),
            actual: describe_kind(other),
            requested: String::new(),
        }),
    }
}

/// Shared i64 coercion used by [`SnapshotField::as_i64`].
fn render_to_i64(v: &RenderedValue) -> SnapshotResult<i64> {
    match v {
        RenderedValue::Int { value, .. } => Ok(*value),
        RenderedValue::Uint { value, .. } => {
            if *value > i64::MAX as u64 {
                Err(SnapshotError::TypeMismatch {
                    expected: "i64".to_string(),
                    actual: "Uint(>i64::MAX)".to_string(),
                    requested: String::new(),
                })
            } else {
                Ok(*value as i64)
            }
        }
        RenderedValue::Bool { value } => Ok(i64::from(*value)),
        RenderedValue::Char { value } => Ok(i64::from(*value)),
        RenderedValue::Enum { value, .. } => Ok(*value),
        other => Err(SnapshotError::TypeMismatch {
            expected: "i64".to_string(),
            actual: describe_kind(other),
            requested: String::new(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::dump::SCHEMA_SINGLE;
    use crate::sync::MutexExt;
    use std::sync::Arc;

    /// Build a synthetic [`FailureDumpReport`] used by every
    /// accessor unit test below.
    fn synthetic_report() -> FailureDumpReport {
        let bss_value = RenderedValue::Struct {
            type_name: Some(".bss".into()),
            members: vec![
                RenderedMember {
                    name: "nr_cpus_onln".into(),
                    value: RenderedValue::Uint { bits: 32, value: 4 },
                },
                RenderedMember {
                    name: "stall".into(),
                    value: RenderedValue::Uint { bits: 8, value: 1 },
                },
                RenderedMember {
                    name: "balance_factor".into(),
                    value: RenderedValue::Float {
                        bits: 64,
                        value: 1.5,
                    },
                },
                RenderedMember {
                    name: "ctx".into(),
                    value: RenderedValue::Struct {
                        type_name: Some("scx_ctx".into()),
                        members: vec![
                            RenderedMember {
                                name: "weight".into(),
                                value: RenderedValue::Uint {
                                    bits: 32,
                                    value: 1024,
                                },
                            },
                            RenderedMember {
                                name: "policy".into(),
                                value: RenderedValue::Enum {
                                    bits: 32,
                                    value: 1,
                                    variant: Some("SCHED_NORMAL".into()),
                                },
                            },
                        ],
                    },
                },
                RenderedMember {
                    name: "leader".into(),
                    value: RenderedValue::Ptr {
                        value: 0xffff_8000_0000_1000,
                        deref: Some(Box::new(RenderedValue::Struct {
                            type_name: Some("task_struct".into()),
                            members: vec![RenderedMember {
                                name: "pid".into(),
                                value: RenderedValue::Int {
                                    bits: 32,
                                    value: 1234,
                                },
                            }],
                        })),
                        deref_skipped_reason: None,
                        cast_annotation: None,
                    },
                },
            ],
        };
        let bss_map = FailureDumpMap {
            name: "bpf.bss".into(),
            map_type: 2,
            value_size: 32,
            max_entries: 1,
            value: Some(bss_value),
            entries: Vec::new(),
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        };
        let hash_map = FailureDumpMap {
            name: "scx_per_task".into(),
            map_type: 1,
            value_size: 8,
            max_entries: 16,
            value: None,
            entries: vec![
                FailureDumpEntry {
                    key: Some(RenderedValue::Uint {
                        bits: 32,
                        value: 100,
                    }),
                    key_hex: "64000000".into(),
                    value: Some(RenderedValue::Struct {
                        type_name: Some("task_ctx".into()),
                        members: vec![
                            RenderedMember {
                                name: "tid".into(),
                                value: RenderedValue::Int {
                                    bits: 32,
                                    value: 100,
                                },
                            },
                            RenderedMember {
                                name: "runtime_ns".into(),
                                value: RenderedValue::Uint {
                                    bits: 64,
                                    value: 5_000_000,
                                },
                            },
                        ],
                    }),
                    value_hex: "0064000000000000".into(),
                    payload: None,
                },
                FailureDumpEntry {
                    key: Some(RenderedValue::Uint {
                        bits: 32,
                        value: 200,
                    }),
                    key_hex: "c8000000".into(),
                    value: Some(RenderedValue::Struct {
                        type_name: Some("task_ctx".into()),
                        members: vec![
                            RenderedMember {
                                name: "tid".into(),
                                value: RenderedValue::Int {
                                    bits: 32,
                                    value: 200,
                                },
                            },
                            RenderedMember {
                                name: "runtime_ns".into(),
                                value: RenderedValue::Uint {
                                    bits: 64,
                                    value: 9_000_000,
                                },
                            },
                        ],
                    }),
                    value_hex: "00c8000000000000".into(),
                    payload: None,
                },
            ],
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        };
        let percpu_map = FailureDumpMap {
            name: "scx_pcpu".into(),
            map_type: 6,
            value_size: 8,
            max_entries: 1,
            value: None,
            entries: Vec::new(),
            percpu_entries: vec![FailureDumpPercpuEntry {
                key: 0,
                per_cpu: vec![
                    Some(RenderedValue::Uint {
                        bits: 64,
                        value: 11,
                    }),
                    Some(RenderedValue::Uint {
                        bits: 64,
                        value: 22,
                    }),
                    None,
                    Some(RenderedValue::Uint {
                        bits: 64,
                        value: 44,
                    }),
                ],
            }],
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        };
        FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: vec![bss_map, hash_map, percpu_map],
            ..Default::default()
        }
    }

    #[test]
    fn snapshot_var_walks_into_bss_struct() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        assert_eq!(snap.var("nr_cpus_onln").as_u64().unwrap(), 4);
        assert!(snap.var("stall").as_bool().unwrap());
        assert!((snap.var("balance_factor").as_f64().unwrap() - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn snapshot_var_dotted_path_walks_nested_struct() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        assert_eq!(snap.var("ctx").get("weight").as_u64().unwrap(), 1024);
        assert_eq!(
            snap.var("ctx").get("policy").as_str().unwrap(),
            "SCHED_NORMAL"
        );
        assert_eq!(snap.var("ctx").get("policy").as_i64().unwrap(), 1);
    }

    #[test]
    fn dotted_path_follows_ptr_deref_transparently() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        assert_eq!(snap.var("leader").get("pid").as_i64().unwrap(), 1234);
    }

    #[test]
    fn missing_var_lists_available_globals() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let f = snap.var("absent");
        let err = f.error().expect("missing field carries an error");
        match err {
            SnapshotError::VarNotFound {
                requested,
                available,
            } => {
                assert_eq!(requested, "absent");
                assert!(available.contains(&"nr_cpus_onln".to_string()));
                assert!(available.contains(&"ctx".to_string()));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
        assert!(f.as_u64().is_err());
        assert!(f.as_i64().is_err());
        assert!(f.as_bool().is_err());
    }

    /// Pin the `Snapshot::var` ambiguity-detection invariant: when
    /// two global-section maps expose a top-level member with the
    /// same name, var() MUST surface AmbiguousVar with both map
    /// names rather than silently returning the first match. The
    /// previous first-match behavior depended on `report.maps`
    /// ordering which mirrors kernel IDR allocation order — a
    /// non-deterministic source. Regression: removing the
    /// hits.len() > 1 arm or short-circuiting on first hit would
    /// surface here as an `Ok` SnapshotField::Value with no error.
    #[test]
    fn snapshot_var_ambiguity_lists_every_match() {
        let mut r = synthetic_report();
        // Add a second .data global-section map that ALSO exposes a
        // top-level `nr_cpus_onln` member. The synthetic report
        // already contains `bpf.bss` with `nr_cpus_onln`; with two
        // maps exposing the name, var() must error.
        let dup_value = RenderedValue::Struct {
            type_name: Some(".data".into()),
            members: vec![RenderedMember {
                name: "nr_cpus_onln".into(),
                value: RenderedValue::Uint {
                    bits: 32,
                    value: 99,
                },
            }],
        };
        r.maps.push(FailureDumpMap {
            name: "other.data".into(),
            map_type: 2,
            value_size: 32,
            max_entries: 1,
            value: Some(dup_value),
            entries: Vec::new(),
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        });
        let snap = Snapshot::new(&r);
        let f = snap.var("nr_cpus_onln");
        let err = f
            .error()
            .expect("duplicate global must surface AmbiguousVar");
        match err {
            SnapshotError::AmbiguousVar {
                requested,
                found_in,
            } => {
                assert_eq!(requested, "nr_cpus_onln");
                assert!(
                    found_in.contains(&"bpf.bss".to_string()),
                    "first map must appear in found_in: {found_in:?}",
                );
                assert!(
                    found_in.contains(&"other.data".to_string()),
                    "second map must appear in found_in: {found_in:?}",
                );
                assert_eq!(
                    found_in.len(),
                    2,
                    "AmbiguousVar must list every map where the name was found, no more no less: {found_in:?}",
                );
            }
            other => panic!("expected AmbiguousVar, got: {other:?}"),
        }
        // Display must mention both map names so the test author
        // can pick the right disambiguation target.
        let rendered = err.to_string();
        assert!(rendered.contains("nr_cpus_onln"), "{rendered}");
        assert!(rendered.contains("bpf.bss"), "{rendered}");
        assert!(rendered.contains("other.data"), "{rendered}");
        // Caller can disambiguate via map() — verify both maps
        // resolve independently.
        let bss = snap
            .map("bpf.bss")
            .unwrap()
            .at(0)
            .get("nr_cpus_onln")
            .as_u64()
            .unwrap();
        let data = snap
            .map("other.data")
            .unwrap()
            .at(0)
            .get("nr_cpus_onln")
            .as_u64()
            .unwrap();
        assert_eq!(bss, 4);
        assert_eq!(data, 99);
    }

    #[test]
    fn missing_field_in_struct_lists_available_members() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let f = snap.var("ctx").get("nonexistent");
        let err = f.error().expect("missing field carries an error");
        match err {
            SnapshotError::FieldNotFound {
                component,
                available,
                ..
            } => {
                assert_eq!(component, "nonexistent");
                assert!(available.contains(&"weight".to_string()));
                assert!(available.contains(&"policy".to_string()));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn missing_map_lists_available_maps() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let err = snap.map("does_not_exist").unwrap_err();
        match err {
            SnapshotError::MapNotFound {
                requested,
                available,
            } => {
                assert_eq!(requested, "does_not_exist");
                assert!(available.contains(&"bpf.bss".to_string()));
                assert!(available.contains(&"scx_per_task".to_string()));
                assert!(available.contains(&"scx_pcpu".to_string()));
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn empty_path_component_returns_error() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let f = snap.var("ctx").get("weight..value");
        match f.error().expect("missing carries error") {
            SnapshotError::EmptyPathComponent { requested } => {
                assert_eq!(requested, "weight..value");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn wrong_kind_at_path_step_explains() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let f = snap.var("ctx").get("weight").get("inner");
        match f.error().expect("missing carries error") {
            SnapshotError::NotAStruct { kind, .. } => {
                assert_eq!(*kind, "Uint");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn map_at_returns_hash_entry() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_per_task").unwrap().at(0);
        assert!(entry.is_present());
        assert_eq!(entry.get("tid").as_i64().unwrap(), 100);
        assert_eq!(entry.get("runtime_ns").as_u64().unwrap(), 5_000_000);
    }

    #[test]
    fn map_at_out_of_range_carries_index_and_len() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_per_task").unwrap().at(99);
        match entry {
            SnapshotEntry::Missing(SnapshotError::IndexOutOfRange { index, len, .. }) => {
                assert_eq!(index, 99);
                assert_eq!(len, 2);
            }
            other => panic!("unexpected entry: present={}", other.is_present()),
        }
    }

    #[test]
    fn map_find_returns_first_match() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let map = snap.map("scx_per_task").unwrap();
        let entry = map.find(|e| e.get("tid").as_i64().unwrap_or(-1) == 200);
        assert!(entry.is_present());
        assert_eq!(entry.get("runtime_ns").as_u64().unwrap(), 9_000_000);
    }

    #[test]
    fn map_find_no_match_carries_op_name_len_and_keys() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let map = snap.map("scx_per_task").unwrap();
        let entry = map.find(|e| e.get("tid").as_i64().unwrap_or(-1) == 999);
        match entry {
            SnapshotEntry::Missing(SnapshotError::NoMatch {
                op,
                len,
                available_keys,
                ..
            }) => {
                assert_eq!(op, "find");
                // synthetic_report's scx_per_task has 2 entries (tid=100 and
                // tid=200) — every traversal that exits via NoMatch must
                // have walked all 2.
                assert_eq!(len, 2);
                // 2 entries, cap is NO_MATCH_KEY_SAMPLE=3, so we keep both.
                assert_eq!(available_keys.len(), 2);
            }
            other => panic!("expected NoMatch, got present={}", other.is_present()),
        }
    }

    #[test]
    fn map_max_by_no_match_reports_empty_map() {
        // Build a fixture whose scx_per_task map has zero entries so
        // max_by's NoMatch arm fires. This is the only path through
        // max_by that produces NoMatch — non-empty maps always have
        // a maximum.
        let mut r = synthetic_report();
        for m in r.maps.iter_mut() {
            if m.name == "scx_per_task" {
                m.entries.clear();
            }
        }
        let snap = Snapshot::new(&r);
        let map = snap.map("scx_per_task").unwrap();
        let entry = map.max_by(|e| e.get("runtime_ns").as_u64().unwrap_or(0));
        match entry {
            SnapshotEntry::Missing(SnapshotError::NoMatch {
                op,
                len,
                available_keys,
                ..
            }) => {
                assert_eq!(op, "max_by");
                assert_eq!(len, 0);
                assert!(available_keys.is_empty());
            }
            other => panic!("expected NoMatch, got present={}", other.is_present()),
        }
    }

    /// Display rendering must surface `len` and `available_keys` for
    /// each of the three cases — empty map, populated map without
    /// sampled keys (all keys were unrenderable), and populated map
    /// with sampled keys. Without this pin, the Display impl could
    /// silently drop the new fields and every structural test would
    /// still pass.
    #[test]
    fn no_match_display_renders_three_arms() {
        let empty = SnapshotError::NoMatch {
            map: "m".to_string(),
            op: "find".to_string(),
            len: 0,
            available_keys: Vec::new(),
        };
        let rendered = format!("{empty}");
        assert!(rendered.contains("'m'"), "{rendered}");
        assert!(rendered.contains("empty"), "{rendered}");

        let unrendered = SnapshotError::NoMatch {
            map: "m".to_string(),
            op: "max_by".to_string(),
            len: 7,
            available_keys: Vec::new(),
        };
        let rendered = format!("{unrendered}");
        assert!(rendered.contains("'m'"), "{rendered}");
        assert!(rendered.contains("7"), "{rendered}");
        assert!(rendered.contains("unavailable"), "{rendered}");

        let sampled = SnapshotError::NoMatch {
            map: "m".to_string(),
            op: "find".to_string(),
            len: 9,
            available_keys: vec!["k0".to_string(), "k1".to_string(), "k2".to_string()],
        };
        let rendered = format!("{sampled}");
        assert!(rendered.contains("'m'"), "{rendered}");
        assert!(rendered.contains("9"), "{rendered}");
        assert!(rendered.contains("k0"), "{rendered}");
        assert!(rendered.contains("k2"), "{rendered}");
    }

    /// `render_entry_key` must truncate wide struct keys (e.g.
    /// a 50-field struct) to [`NO_MATCH_KEY_CHAR_CAP`] chars with a
    /// trailing `…`. Without the cap a single wide-struct key would
    /// blow the failure-message size budget.
    #[test]
    fn render_entry_key_caps_wide_struct_keys() {
        let oversized = "x".repeat(NO_MATCH_KEY_CHAR_CAP * 4);
        let entry_fixture = FailureDumpEntry {
            key: None,
            key_hex: oversized.clone(),
            value: None,
            value_hex: String::new(),
            payload: None,
        };
        let entry = SnapshotEntry::Hash(&entry_fixture);
        let rendered = render_entry_key(&entry).expect("hash entry has a key");
        assert!(rendered.chars().count() <= NO_MATCH_KEY_CHAR_CAP);
        assert!(rendered.ends_with('…'));
        assert!(rendered.starts_with("hex:x"));
    }

    /// Internal-contract pin: when a HASH entry's `key` field is
    /// `None` (BTF was missing at capture for the map's key type),
    /// `render_entry_key` must fall back to a string that BEGINS
    /// with `"hex:"` followed by the raw `key_hex` bytes. The
    /// BTF-missing-hint detection on the NoMatch Display arm uses
    /// `available_keys.iter().all(|k| k.starts_with("hex:"))` as
    /// the discriminator — a silent change to the prefix (drop,
    /// rename to `"0x"`, etc.) would break the hint without
    /// surfacing in any sibling test. `key_hex` is space-separated
    /// hex pairs per `monitor::dump::hex_dump`, so the fixture
    /// uses that format to match what production wire data looks
    /// like.
    #[test]
    fn render_entry_key_hash_fallback_uses_hex_prefix() {
        let entry_fixture = FailureDumpEntry {
            key: None,
            key_hex: "de ad be ef".into(),
            value: None,
            value_hex: String::new(),
            payload: None,
        };
        let entry = SnapshotEntry::Hash(&entry_fixture);
        let rendered = render_entry_key(&entry).expect("hash entry has a key");
        assert!(
            rendered.starts_with("hex:"),
            "Hash fallback must use 'hex:' prefix that NoMatch Display \
             uses as the BTF-missing-hint discriminator; got {rendered:?}",
        );
        assert!(
            rendered.contains("de ad be ef"),
            "key_hex bytes must be preserved verbatim so the operator \
             can disambiguate keys; got {rendered:?}",
        );
    }

    /// Sibling of `render_entry_key_hash_fallback_uses_hex_prefix`
    /// for the [`SnapshotEntry::PercpuHash`] variant. Both Hash and
    /// PercpuHash entries hit the same `hex:{key_hex}` fallback in
    /// `render_entry_key` when their `key` field is `None`; a
    /// regression that altered the fallback shape for ONLY one of
    /// the two variants would silently break the BTF-missing-hint
    /// gate for that variant's map type. The pair pins the
    /// internal contract for both variants explicitly.
    #[test]
    fn render_entry_key_percpu_hash_fallback_uses_hex_prefix() {
        let entry_fixture = FailureDumpPercpuHashEntry {
            key: None,
            key_hex: "ca fe ba be".into(),
            per_cpu: vec![None],
        };
        let entry = SnapshotEntry::PercpuHash(&entry_fixture);
        let rendered = render_entry_key(&entry).expect("percpu-hash entry has a key");
        assert!(
            rendered.starts_with("hex:"),
            "PercpuHash fallback must use 'hex:' prefix; got {rendered:?}",
        );
        assert!(
            rendered.contains("ca fe ba be"),
            "key_hex bytes must be preserved verbatim; got {rendered:?}",
        );
    }

    /// End-to-end pin for the `available_keys.is_empty() && len > 0`
    /// Display arm: a single-value ARRAY map iterates as one
    /// [`SnapshotEntry::Value`], `render_entry_key` returns None for
    /// Value variants, so `find()` traverses one entry, never pushes
    /// a key, and constructs NoMatch with `len = 1, available_keys
    /// = []`. The Display impl's middle arm fires
    /// ("sample keys unavailable"). Complements
    /// [`no_match_display_renders_three_arms`] which exercises the
    /// same arm via direct struct-literal construction — this test
    /// proves the production find/iter_entries path actually reaches
    /// it.
    #[test]
    fn map_find_no_match_on_single_value_array_renders_unavailable_keys() {
        let array_map = FailureDumpMap {
            name: "scx_singleton".into(),
            map_type: 2,
            value_size: 8,
            max_entries: 1,
            value: Some(RenderedValue::Uint {
                bits: 64,
                value: 42,
            }),
            entries: Vec::new(),
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        };
        let r = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: vec![array_map],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_singleton").unwrap().find(|_| false);
        let SnapshotEntry::Missing(err) = entry else {
            panic!("expected Missing, got present={}", entry.is_present());
        };
        match &err {
            SnapshotError::NoMatch {
                op,
                len,
                available_keys,
                ..
            } => {
                assert_eq!(*op, "find");
                assert_eq!(*len, 1, "single Value entry iterated");
                assert!(
                    available_keys.is_empty(),
                    "Value entries are unrenderable for key sampling: {available_keys:?}",
                );
            }
            other => panic!("expected NoMatch, got {other:?}"),
        }
        let rendered = format!("{err}");
        assert!(rendered.contains("'scx_singleton'"), "{rendered}");
        assert!(rendered.contains("matched none of 1"), "{rendered}");
        assert!(rendered.contains("sample keys unavailable"), "{rendered}");
    }

    /// `find()` must clamp `available_keys` at
    /// [`NO_MATCH_KEY_SAMPLE`] regardless of how many entries the
    /// underlying map carries. Pins the cap-clamp inside `find()`
    /// and the FIRST-N-preservation order (keys 0/1/2, not the
    /// last 3). A regression that drops the cap-gate, or that
    /// flips it from `<` to `<=`, would push a 4th key into the
    /// sample and trip the `available_keys.len() ==
    /// NO_MATCH_KEY_SAMPLE` assertion. A regression that reverses
    /// iteration order or swaps to a random sample would trip the
    /// literal-vec assertion below.
    #[test]
    fn map_find_no_match_caps_sampled_keys_at_no_match_key_sample() {
        const N: u32 = 10;
        let entries: Vec<FailureDumpEntry> = (0..N)
            .map(|i| FailureDumpEntry {
                key: Some(RenderedValue::Uint {
                    bits: 32,
                    value: u64::from(i),
                }),
                key_hex: format!("{i:08x}"),
                value: Some(RenderedValue::Uint {
                    bits: 32,
                    value: u64::from(i * 10),
                }),
                value_hex: format!("{:08x}", i * 10),
                payload: None,
            })
            .collect();
        let hash_map = FailureDumpMap {
            name: "scx_big".into(),
            map_type: 1,
            value_size: 4,
            max_entries: 64,
            value: None,
            entries,
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        };
        let r = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: vec![hash_map],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_big").unwrap().find(|_| false);
        match entry {
            SnapshotEntry::Missing(SnapshotError::NoMatch {
                op,
                len,
                available_keys,
                ..
            }) => {
                assert_eq!(op, "find");
                assert_eq!(
                    len,
                    usize::try_from(N).unwrap(),
                    "all {N} entries must be traversed before NoMatch",
                );
                assert_eq!(
                    available_keys.len(),
                    NO_MATCH_KEY_SAMPLE,
                    "cap must clamp sample at NO_MATCH_KEY_SAMPLE",
                );
                assert_eq!(
                    available_keys,
                    vec!["0".to_string(), "1".to_string(), "2".to_string()],
                    "first-N preservation: sample must hold the FIRST 3 keys in \
                     iteration order, not last 3 / random",
                );
            }
            other => panic!("expected NoMatch, got present={}", other.is_present()),
        }
    }

    /// The `find()` `available_keys` sample preserves duplicates in
    /// iteration order — there is no dedup at the snapshot layer.
    /// `iter_entries()` walks `self.map.entries.iter()` directly for
    /// the HASH branch (and `percpu_entries.iter()` /
    /// `percpu_hash_entries.iter()` for the per-CPU branches); all
    /// three are raw `Vec::iter()` calls. The wire format is a `Vec`,
    /// not a `Map`, so duplicate keys are syntactically legal even
    /// though real BPF maps cannot produce them — and the operator's
    /// failure-message contract depends on `len` reflecting EVERY
    /// entry seen and `available_keys` showing the sample as
    /// observed. This test pins the iter-layer no-dedup invariant; a
    /// future "optimization" that adds dedup to `iter_entries()`
    /// would collapse the duplicate `"100"` here and fail loudly.
    #[test]
    fn map_find_no_match_preserves_duplicate_keys_in_sample() {
        let hash_map = FailureDumpMap {
            name: "scx_dup".into(),
            map_type: 1,
            value_size: 4,
            max_entries: 16,
            value: None,
            entries: vec![
                FailureDumpEntry {
                    key: Some(RenderedValue::Uint {
                        bits: 32,
                        value: 100,
                    }),
                    key_hex: "64000000".into(),
                    value: Some(RenderedValue::Uint { bits: 32, value: 1 }),
                    value_hex: "01000000".into(),
                    payload: None,
                },
                FailureDumpEntry {
                    key: Some(RenderedValue::Uint {
                        bits: 32,
                        value: 100,
                    }),
                    key_hex: "64000000".into(),
                    value: Some(RenderedValue::Uint { bits: 32, value: 2 }),
                    value_hex: "02000000".into(),
                    payload: None,
                },
                FailureDumpEntry {
                    key: Some(RenderedValue::Uint {
                        bits: 32,
                        value: 200,
                    }),
                    key_hex: "c8000000".into(),
                    value: Some(RenderedValue::Uint { bits: 32, value: 3 }),
                    value_hex: "03000000".into(),
                    payload: None,
                },
                FailureDumpEntry {
                    key: Some(RenderedValue::Uint {
                        bits: 32,
                        value: 300,
                    }),
                    key_hex: "2c010000".into(),
                    value: Some(RenderedValue::Uint { bits: 32, value: 4 }),
                    value_hex: "04000000".into(),
                    payload: None,
                },
            ],
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        };
        let r = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: vec![hash_map],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_dup").unwrap().find(|_| false);
        match entry {
            SnapshotEntry::Missing(SnapshotError::NoMatch {
                op,
                len,
                available_keys,
                ..
            }) => {
                assert_eq!(op, "find");
                assert_eq!(
                    len, 4,
                    "duplicate-key entries each count toward len; iter_entries \
                     Hash branch iterates the Vec directly with no dedup",
                );
                assert_eq!(available_keys.len(), NO_MATCH_KEY_SAMPLE);
                assert_eq!(
                    available_keys,
                    vec!["100".to_string(), "100".to_string(), "200".to_string()],
                    "no dedup at sample-collection: the duplicate key '100' \
                     appears twice in iteration order before the cap fires \
                     against the third unique key",
                );
                let dup_count = available_keys
                    .iter()
                    .filter(|k| k.as_str() == "100")
                    .count();
                assert_eq!(
                    dup_count, 2,
                    "position-insensitive backup pin: key '100' must appear \
                     exactly twice in available_keys (saw {dup_count} in \
                     {available_keys:?})",
                );
            }
            other => panic!("expected NoMatch, got present={}", other.is_present()),
        }
    }

    /// `find()`'s NoMatch Display appends a BTF-missing hint when
    /// every sampled key carries the `hex:` prefix from
    /// [`render_entry_key`]'s fallback path. Pins both the hint
    /// substring AND that the prior arm's key listing is still
    /// rendered (the hint is additive, not a replacement).
    #[test]
    fn no_match_display_appends_btf_hint_when_all_keys_are_hex() {
        let err = SnapshotError::NoMatch {
            map: "scx_per_task".to_string(),
            op: "find".to_string(),
            len: 5,
            available_keys: vec![
                "hex:64000000".to_string(),
                "hex:c8000000".to_string(),
                "hex:2c010000".to_string(),
            ],
        };
        let rendered = format!("{err}");
        assert!(rendered.contains("'scx_per_task'"), "{rendered}");
        assert!(rendered.contains("matched none of 5"), "{rendered}");
        assert!(
            rendered.contains("hex:64000000"),
            "underlying key list still rendered: {rendered}",
        );
        assert!(rendered.contains("BTF missing"), "{rendered}");
        assert!(rendered.contains("CONFIG_DEBUG_INFO_BTF"), "{rendered}");
    }

    /// The BTF-missing hint MUST NOT fire when the sample contains
    /// any typed (non-`hex:`-prefixed) key. A mixed sample means BTF
    /// IS present and resolving most keys; the unresolved hex ones
    /// are a per-entry capture race, not a uniform-config problem,
    /// so the kernel-rebuild advice would mislead the operator.
    #[test]
    fn no_match_display_omits_btf_hint_when_some_keys_are_typed() {
        let err = SnapshotError::NoMatch {
            map: "mixed".to_string(),
            op: "find".to_string(),
            len: 5,
            available_keys: vec![
                "hex:64000000".to_string(),
                "100".to_string(),
                "hex:2c010000".to_string(),
            ],
        };
        let rendered = format!("{err}");
        assert!(rendered.contains("hex:64000000"), "{rendered}");
        assert!(rendered.contains("100"), "{rendered}");
        assert!(
            !rendered.contains("BTF missing"),
            "BTF hint must not fire on mixed-typed sample: {rendered}",
        );
        assert!(!rendered.contains("CONFIG_DEBUG_INFO_BTF"), "{rendered}");
    }

    /// The BTF-missing hint MUST NOT fire on the `available_keys
    /// .is_empty()` arm even though `[].iter().all(_)` returns true
    /// vacuously. There is no evidence of a BTF problem when no
    /// keys were sampled; the hint here would be a false positive.
    #[test]
    fn no_match_display_omits_btf_hint_when_no_keys_sampled() {
        let err = SnapshotError::NoMatch {
            map: "m".to_string(),
            op: "find".to_string(),
            len: 5,
            available_keys: Vec::new(),
        };
        let rendered = format!("{err}");
        assert!(rendered.contains("sample keys unavailable"), "{rendered}");
        assert!(!rendered.contains("BTF missing"), "{rendered}");
        assert!(!rendered.contains("CONFIG_DEBUG_INFO_BTF"), "{rendered}");
    }

    /// The BTF-missing hint MUST NOT fire on the empty-map arm
    /// (`len == 0`). The empty-map message is its own dedicated
    /// branch; an unconditional hint append would render the
    /// operator-facing message contradictory.
    #[test]
    fn no_match_display_omits_btf_hint_when_map_is_empty() {
        let err = SnapshotError::NoMatch {
            map: "m".to_string(),
            op: "find".to_string(),
            len: 0,
            available_keys: Vec::new(),
        };
        let rendered = format!("{err}");
        assert!(rendered.contains("map is empty"), "{rendered}");
        assert!(!rendered.contains("BTF missing"), "{rendered}");
        assert!(!rendered.contains("CONFIG_DEBUG_INFO_BTF"), "{rendered}");
    }

    /// Boundary case: when entry count equals [`NO_MATCH_KEY_SAMPLE`]
    /// exactly, every entry's key fits in the sample. `len ==
    /// available_keys.len() == NO_MATCH_KEY_SAMPLE` guards against
    /// a regression that gates the push too tightly — e.g.
    /// `available_keys.len() < NO_MATCH_KEY_SAMPLE - 1` would push
    /// only 2 keys at N=3 and trip the
    /// `available_keys.len() == NO_MATCH_KEY_SAMPLE` assertion.
    /// Pairs with
    /// [`map_find_no_match_caps_sampled_keys_at_no_match_key_sample`]
    /// (which catches the OTHER direction — `<=` would push a 4th
    /// key at N >= NO_MATCH_KEY_SAMPLE+1) so both sides of the
    /// gate-condition are pinned.
    #[test]
    fn map_find_no_match_cap_exact_threshold() {
        let entries: Vec<FailureDumpEntry> = (0..NO_MATCH_KEY_SAMPLE as u32)
            .map(|i| FailureDumpEntry {
                key: Some(RenderedValue::Uint {
                    bits: 32,
                    value: u64::from(i),
                }),
                key_hex: format!("{i:08x}"),
                value: Some(RenderedValue::Uint {
                    bits: 32,
                    value: u64::from(i),
                }),
                value_hex: format!("{i:08x}"),
                payload: None,
            })
            .collect();
        let hash_map = FailureDumpMap {
            name: "scx_threshold".into(),
            map_type: 1,
            value_size: 4,
            max_entries: 16,
            value: None,
            entries,
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        };
        let r = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: vec![hash_map],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_threshold").unwrap().find(|_| false);
        match entry {
            SnapshotEntry::Missing(SnapshotError::NoMatch {
                len,
                available_keys,
                ..
            }) => {
                assert_eq!(len, NO_MATCH_KEY_SAMPLE);
                assert_eq!(available_keys.len(), NO_MATCH_KEY_SAMPLE);
            }
            other => panic!("expected NoMatch, got present={}", other.is_present()),
        }
    }

    #[test]
    fn map_filter_collects_matches() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let map = snap.map("scx_per_task").unwrap();
        let matches = map.filter(|e| e.get("runtime_ns").as_u64().unwrap_or(0) > 0);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn map_max_by_picks_largest() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let map = snap.map("scx_per_task").unwrap();
        let busiest = map.max_by(|e| e.get("runtime_ns").as_u64().unwrap_or(0));
        assert!(busiest.is_present());
        assert_eq!(busiest.get("tid").as_i64().unwrap(), 200);
    }

    #[test]
    fn percpu_array_cpu_narrow_reads_per_cpu_slot() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_pcpu").unwrap().cpu(1).at(0);
        assert!(entry.is_present());
        assert_eq!(entry.get("").as_u64().unwrap(), 22);
    }

    #[test]
    fn percpu_array_unmapped_cpu_returns_unmapped_error() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_pcpu").unwrap().cpu(2).at(0);
        match entry {
            SnapshotEntry::Missing(SnapshotError::PerCpuSlot { cpu, unmapped, .. }) => {
                assert_eq!(cpu, 2);
                assert!(unmapped);
            }
            _ => panic!("expected unmapped PerCpuSlot"),
        }
    }

    #[test]
    fn percpu_array_out_of_range_cpu_returns_oor_error() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_pcpu").unwrap().cpu(99).at(0);
        match entry {
            SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
                cpu, unmapped, len, ..
            }) => {
                assert_eq!(cpu, 99);
                assert!(!unmapped);
                assert_eq!(len, 4);
            }
            _ => panic!("expected out-of-range PerCpuSlot"),
        }
    }

    #[test]
    fn percpu_array_get_without_narrow_explains() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_pcpu").unwrap().at(0);
        let f = entry.get("anything");
        match f.error().expect("missing") {
            SnapshotError::PerCpuNotNarrowed { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// The scx_pcpu fixture has per_cpu = [Some(11), Some(22), None,
    /// Some(44)] (snapshot.rs synthetic_report). Per-CPU aggregators
    /// skip the None slot at index 2 and sum / max / min the others.
    #[test]
    fn snapshot_entry_cpu_sum_u64_skips_none_slots() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_pcpu").unwrap().at(0);
        assert_eq!(entry.cpu_sum_u64("").unwrap(), 11 + 22 + 44);
    }

    #[test]
    fn snapshot_entry_cpu_max_u64_returns_largest_present() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_pcpu").unwrap().at(0);
        assert_eq!(entry.cpu_max_u64("").unwrap(), 44);
    }

    #[test]
    fn snapshot_entry_cpu_min_u64_returns_smallest_present() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_pcpu").unwrap().at(0);
        assert_eq!(entry.cpu_min_u64("").unwrap(), 11);
    }

    /// `cpu_max_u64` / `cpu_min_u64` on an all-None per_cpu vec
    /// return NoMatch — no slot contributes a value, so neither max
    /// nor min has a meaningful answer.
    #[test]
    fn snapshot_entry_cpu_max_min_no_slots_returns_no_match() {
        let entry_struct = FailureDumpPercpuEntry {
            key: 0,
            per_cpu: vec![None, None, None],
        };
        let entry = SnapshotEntry::Percpu(&entry_struct);
        match entry.cpu_max_u64("").expect_err("empty must err") {
            SnapshotError::NoMatch { op, len, .. } => {
                assert_eq!(op, "cpu_max_u64");
                assert_eq!(len, 0);
            }
            other => panic!("unexpected: {other:?}"),
        }
        // cpu_sum_u64 returns 0 for all-None (sum identity); this
        // pins the asymmetry between sum (always-defined) and
        // max/min (require >= 1 slot).
        assert_eq!(entry.cpu_sum_u64("").unwrap(), 0);
    }

    #[test]
    fn snapshot_entry_cpu_each_visits_only_present_slots() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let entry = snap.map("scx_pcpu").unwrap().at(0);
        let mut visited: Vec<(usize, u64)> = Vec::new();
        entry
            .cpu_each("", |cpu, v| {
                visited.push((cpu, SnapshotField::Value(v).as_u64()?));
                Ok(())
            })
            .unwrap();
        assert_eq!(visited, vec![(0, 11), (1, 22), (3, 44)]);
    }

    /// Non-percpu variants — Hash, Value — produce TypeMismatch.
    #[test]
    fn snapshot_entry_cpu_sum_on_non_percpu_variant_errors() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let bss_entry = snap.map("bpf.bss").unwrap().at(0);
        // .bss is a Value entry, not Percpu.
        match bss_entry.cpu_sum_u64("").expect_err("non-percpu must err") {
            SnapshotError::TypeMismatch {
                expected, actual, ..
            } => {
                assert!(expected.contains("Percpu"));
                assert_eq!(actual, "Value");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    /// cpu_min_u64 on all-None must also return NoMatch (sibling
    /// of the cpu_max_u64 empty case already pinned).
    #[test]
    fn snapshot_entry_cpu_min_u64_no_slots_returns_no_match() {
        let entry_struct = FailureDumpPercpuEntry {
            key: 0,
            per_cpu: vec![None, None],
        };
        let entry = SnapshotEntry::Percpu(&entry_struct);
        match entry.cpu_min_u64("").expect_err("empty must err") {
            SnapshotError::NoMatch { op, len, .. } => {
                assert_eq!(op, "cpu_min_u64");
                assert_eq!(len, 0);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    fn percpu_entry_with_floats(per_cpu: Vec<Option<f64>>) -> FailureDumpPercpuEntry {
        FailureDumpPercpuEntry {
            key: 0,
            per_cpu: per_cpu
                .into_iter()
                .map(|opt| opt.map(|v| RenderedValue::Float { bits: 64, value: v }))
                .collect(),
        }
    }

    #[test]
    fn snapshot_entry_cpu_sum_f64_skips_none_and_sums() {
        let e = percpu_entry_with_floats(vec![Some(1.5), Some(2.5), None, Some(3.0)]);
        let entry = SnapshotEntry::Percpu(&e);
        let sum = entry.cpu_sum_f64("").unwrap();
        assert!((sum - 7.0).abs() < f64::EPSILON, "sum was {sum}");
    }

    #[test]
    fn snapshot_entry_cpu_max_f64_returns_largest() {
        let e = percpu_entry_with_floats(vec![Some(1.5), Some(9.25), Some(3.0)]);
        let entry = SnapshotEntry::Percpu(&e);
        assert!((entry.cpu_max_f64("").unwrap() - 9.25).abs() < f64::EPSILON);
    }

    #[test]
    fn snapshot_entry_cpu_min_f64_returns_smallest() {
        let e = percpu_entry_with_floats(vec![Some(1.5), Some(9.25), Some(3.0)]);
        let entry = SnapshotEntry::Percpu(&e);
        assert!((entry.cpu_min_f64("").unwrap() - 1.5).abs() < f64::EPSILON);
    }

    /// NaN propagates through f64 sum (per IEEE-754 +=) but is
    /// filtered out by f64::max / f64::min (per Rust stdlib
    /// `maximumNumber` / `minimumNumber` semantics).
    #[test]
    fn snapshot_entry_cpu_f64_aggregators_handle_nan_per_docs() {
        let e = percpu_entry_with_floats(vec![Some(1.0), Some(f64::NAN), Some(3.0)]);
        let entry = SnapshotEntry::Percpu(&e);
        assert!(entry.cpu_sum_f64("").unwrap().is_nan());
        // f64::max filters NaN — the result is the max of 1.0/3.0.
        assert!((entry.cpu_max_f64("").unwrap() - 3.0).abs() < f64::EPSILON);
        // f64::min filters NaN — the result is the min of 1.0/3.0.
        assert!((entry.cpu_min_f64("").unwrap() - 1.0).abs() < f64::EPSILON);
    }

    /// Nested path lookup descends into a Struct rendered value.
    /// Each slot's value is `{count: Uint(N)}` and cpu_sum_u64
    /// walks "count" to get N.
    #[test]
    fn snapshot_entry_cpu_sum_u64_walks_nested_path() {
        let struct_for = |n: u64| RenderedValue::Struct {
            type_name: Some("slot".into()),
            members: vec![RenderedMember {
                name: "count".into(),
                value: RenderedValue::Uint { bits: 64, value: n },
            }],
        };
        let e = FailureDumpPercpuEntry {
            key: 0,
            per_cpu: vec![Some(struct_for(10)), Some(struct_for(20)), None, Some(struct_for(30))],
        };
        let entry = SnapshotEntry::Percpu(&e);
        assert_eq!(entry.cpu_sum_u64("count").unwrap(), 60);
    }

    /// The PercpuHash variant takes the same per_cpu walk as
    /// Percpu (both dispatch through cpu_each's `&e.per_cpu`
    /// arm). Pin both variant dispatch paths so a refactor that
    /// drops the PercpuHash arm surfaces as a TypeMismatch error
    /// the test catches.
    #[test]
    fn snapshot_entry_cpu_aggregators_apply_to_percpu_hash_variant() {
        let e = FailureDumpPercpuHashEntry {
            key: Some(RenderedValue::Uint { bits: 32, value: 0 }),
            key_hex: "00000000".into(),
            per_cpu: vec![
                Some(RenderedValue::Uint { bits: 64, value: 5 }),
                Some(RenderedValue::Uint { bits: 64, value: 7 }),
                None,
            ],
        };
        let entry = SnapshotEntry::PercpuHash(&e);
        assert_eq!(entry.cpu_sum_u64("").unwrap(), 12);
        assert_eq!(entry.cpu_max_u64("").unwrap(), 7);
        assert_eq!(entry.cpu_min_u64("").unwrap(), 5);
    }

    /// A slot whose rendered value is a Struct (not directly an
    /// integer) makes the empty-path cpu_sum_u64 fail with
    /// TypeMismatch — the SnapshotField::Value(v).as_u64() chain
    /// rejects Struct.
    #[test]
    fn snapshot_entry_cpu_sum_u64_struct_slot_errors_with_type_mismatch() {
        let s = RenderedValue::Struct {
            type_name: Some("slot".into()),
            members: vec![RenderedMember {
                name: "count".into(),
                value: RenderedValue::Uint { bits: 64, value: 7 },
            }],
        };
        let e = FailureDumpPercpuEntry {
            key: 0,
            per_cpu: vec![Some(s)],
        };
        let entry = SnapshotEntry::Percpu(&e);
        match entry.cpu_sum_u64("").expect_err("struct cannot as_u64") {
            SnapshotError::TypeMismatch { actual, .. } => {
                assert!(actual.contains("Struct"), "actual was {actual}");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn snapshot_bridge_capture_stores_under_name() {
        let report = synthetic_report();
        let cb: CaptureCallback = Arc::new(move |_name| Some(report.clone()));
        let bridge = SnapshotBridge::new(cb);
        assert!(bridge.is_empty());
        assert!(bridge.capture("test_name"));
        assert_eq!(bridge.len(), 1);
        let drained = bridge.drain();
        assert!(drained.contains_key("test_name"));
        assert_eq!(drained["test_name"].maps.len(), 3);
    }

    #[test]
    fn snapshot_bridge_capture_failure_returns_false() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        assert!(!bridge.capture("oops"));
        assert!(bridge.is_empty());
    }

    #[test]
    fn snapshot_bridge_register_watch_without_callback_errors() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        let err = bridge
            .register_watch("kernel.foo")
            .expect_err("no watch register installed");
        assert!(err.contains("no watch-register callback installed"));
        // Cap rollback: failed register must not consume a slot.
        assert_eq!(bridge.watch_count(), 0);
    }

    #[test]
    fn snapshot_bridge_register_watch_enforces_max_3() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let reg: WatchRegisterCallback = Arc::new(|_symbol| Ok(()));
        let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
        assert!(bridge.register_watch("kernel.a").is_ok());
        assert!(bridge.register_watch("kernel.b").is_ok());
        assert!(bridge.register_watch("kernel.c").is_ok());
        assert_eq!(bridge.watch_count(), MAX_WATCH_SNAPSHOTS);
        let err = bridge
            .register_watch("kernel.d")
            .expect_err("4th watch must be rejected");
        assert!(err.contains("cap exceeded"));
        // Cap rollback: rejection does not consume a slot.
        assert_eq!(bridge.watch_count(), MAX_WATCH_SNAPSHOTS);
    }

    #[test]
    fn snapshot_bridge_register_watch_propagates_callback_error() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let reg: WatchRegisterCallback =
            Arc::new(|symbol| Err(format!("symbol '{symbol}' did not resolve")));
        let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
        let err = bridge
            .register_watch("kernel.nonexistent")
            .expect_err("callback errored");
        assert!(err.contains("kernel.nonexistent"));
        // Failed register must not consume a slot.
        assert_eq!(bridge.watch_count(), 0);
    }

    /// Pin the WatchSlotGuard panic-safety invariant: a panic inside
    /// the watch-register callback must NOT leak the reserved slot.
    /// Before the guard was added, the manual fetch_sub rollback only
    /// ran on the explicit `Err(reason)` arm — a panicking callback
    /// left `watch_count` permanently incremented, eventually exhausting
    /// the cap with no real watchpoints armed. The guard's `Drop` impl
    /// runs on every exit path including unwind; success commits via
    /// `mem::forget`. Regression: removing the guard or moving
    /// `mem::forget` before the callback would surface here as
    /// `watch_count() != 0` after the catch_unwind below.
    #[test]
    fn snapshot_bridge_register_watch_panic_releases_slot() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let reg: WatchRegisterCallback = Arc::new(|_symbol| {
            panic!("synthetic register_watch panic — slot must still release");
        });
        let bridge = SnapshotBridge::new(cb).with_watch_register(reg);
        let bridge_clone = bridge.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = bridge_clone.register_watch("kernel.panic_path");
        }));
        assert!(
            result.is_err(),
            "callback panic must propagate out of register_watch",
        );
        // Slot must be released — guard's Drop ran during unwind.
        assert_eq!(
            bridge.watch_count(),
            0,
            "WatchSlotGuard must release the reserved slot on panic; \
             a non-zero count means the slot leaked and the cap will \
             eventually exhaust with no real watchpoints armed",
        );
        // Cap must remain reachable: a fresh non-panicking callback
        // can now register all 3 user slots.
        let cb2: CaptureCallback = Arc::new(|_| None);
        let reg2: WatchRegisterCallback = Arc::new(|_| Ok(()));
        let bridge2 = SnapshotBridge::new(cb2).with_watch_register(reg2);
        for i in 0..MAX_WATCH_SNAPSHOTS {
            assert!(bridge2.register_watch(&format!("kernel.s{i}")).is_ok());
        }
        assert_eq!(bridge2.watch_count(), MAX_WATCH_SNAPSHOTS);
    }

    #[test]
    fn snapshot_bridge_thread_local_install_and_restore() {
        assert!(with_active_bridge(|_| ()).is_none());
        let report = synthetic_report();
        let cb: CaptureCallback = Arc::new(move |_| Some(report.clone()));
        let bridge = SnapshotBridge::new(cb);
        let bridge_clone = bridge.clone();
        {
            let _g = bridge.set_thread_local();
            let captured = with_active_bridge(|b| b.capture("nested"));
            assert_eq!(captured, Some(true));
        }
        assert!(with_active_bridge(|_| ()).is_none());
        assert_eq!(bridge_clone.len(), 1);
    }

    #[test]
    fn snapshot_bridge_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        assert_send_sync(&bridge);
    }

    /// Filling [`SnapshotBridge`] beyond [`MAX_STORED_SNAPSHOTS`]
    /// must FIFO-evict the oldest tag and keep the newest. Pins
    /// the cap-and-evict invariant the doc on
    /// [`SnapshotBridge::store`] claims (see lines 579–598 / 606–
    /// 621): the `while reports.len() > MAX_STORED_SNAPSHOTS` loop
    /// pops `order.front()` (the oldest insertion) and removes the
    /// corresponding entry from `reports`. A regression that drops
    /// the sweep, replaces FIFO with LIFO, or skips the
    /// `reports.remove` step would surface here as either an
    /// over-cap `len()` or the wrong tag missing/present.
    #[test]
    fn snapshot_bridge_store_fifo_evicts_oldest() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        // Insert exactly MAX_STORED_SNAPSHOTS distinct tags. The
        // store invariant at the cap is `len() == cap`; nothing
        // has been evicted yet.
        for i in 0..MAX_STORED_SNAPSHOTS {
            bridge.store(&format!("tag_{i:04}"), FailureDumpReport::default());
        }
        assert_eq!(
            bridge.len(),
            MAX_STORED_SNAPSHOTS,
            "store at cap must hold exactly {MAX_STORED_SNAPSHOTS} entries",
        );
        // Insert one more — `tag_0000` (the oldest) must be the
        // evicted FIFO front; the freshest tag must now be
        // resident.
        let overflow_tag = format!("tag_{MAX_STORED_SNAPSHOTS:04}");
        bridge.store(&overflow_tag, FailureDumpReport::default());
        assert_eq!(
            bridge.len(),
            MAX_STORED_SNAPSHOTS,
            "post-overflow len must remain at cap (one in, one out)",
        );
        let drained = bridge.drain();
        assert!(
            !drained.contains_key("tag_0000"),
            "FIFO eviction must drop the oldest tag (tag_0000)",
        );
        assert!(
            drained.contains_key(&overflow_tag),
            "newest tag ({overflow_tag}) must be resident after the overflow store",
        );
        // The other 63 originally-inserted tags (tag_0001 ..
        // tag_0063) must all survive — the FIFO is one-in-one-out,
        // not a wholesale flush.
        for i in 1..MAX_STORED_SNAPSHOTS {
            let tag = format!("tag_{i:04}");
            assert!(
                drained.contains_key(&tag),
                "tag {tag} must survive single-overflow eviction",
            );
        }
    }

    /// Storing the same tag twice must REPLACE the report and
    /// move the tag to the BACK of the FIFO order — refreshing
    /// its position so a hot-rewritten tag does not stay near
    /// the eviction front. Pins the overwrite-refresh invariant
    /// the doc at lines 593–603 claims: on insert collision the
    /// loop searches `order` for the existing tag, removes it,
    /// then `push_back`s the fresh occurrence.
    ///
    /// The proof shape: pre-fill to cap with tag_0 .. tag_{cap-1},
    /// re-store tag_0 (refreshing its position to back), then
    /// store one fresh overflow tag. If overwrite-refresh
    /// works, the evicted tag MUST be tag_1 (now the oldest);
    /// without the refresh, tag_0 would stay at front and be
    /// evicted instead.
    #[test]
    fn snapshot_bridge_store_overwrite_refreshes_position() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        for i in 0..MAX_STORED_SNAPSHOTS {
            bridge.store(&format!("tag_{i:04}"), FailureDumpReport::default());
        }
        // Refresh tag_0 by overwriting it. The doc invariant: the
        // overwrite path moves tag_0 from front to back of `order`
        // and replaces its report in `reports`. Use a non-default
        // schema to make the overwrite observable on the value
        // side too.
        let refreshed = FailureDumpReport {
            schema: "refreshed".to_string(),
            ..Default::default()
        };
        bridge.store("tag_0000", refreshed);
        assert_eq!(
            bridge.len(),
            MAX_STORED_SNAPSHOTS,
            "overwrite must not change resident count",
        );
        // Push one fresh overflow tag. With overwrite-refresh,
        // the evicted entry is tag_0001 (now the FIFO front);
        // without it, tag_0000 would still be front and would
        // be evicted instead.
        let overflow_tag = format!("tag_{MAX_STORED_SNAPSHOTS:04}");
        bridge.store(&overflow_tag, FailureDumpReport::default());
        let drained = bridge.drain();
        assert!(
            drained.contains_key("tag_0000"),
            "tag_0000 must survive eviction — overwrite refreshed its FIFO \
             position to the back. A regression to a no-refresh overwrite \
             path would evict tag_0000 instead of tag_0001 here.",
        );
        assert_eq!(
            drained
                .get("tag_0000")
                .expect("tag_0000 resident after overwrite")
                .schema,
            "refreshed",
            "overwrite must replace the report value, not just refresh order",
        );
        assert!(
            !drained.contains_key("tag_0001"),
            "tag_0001 must be the evicted tag — refreshed tag_0000 displaced \
             tag_0001 to the FIFO front",
        );
        assert!(
            drained.contains_key(&overflow_tag),
            "newest tag ({overflow_tag}) must be resident after the overflow store",
        );
    }

    #[test]
    fn enum_variant_round_trips() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let policy = snap.var("ctx").get("policy");
        assert_eq!(policy.as_i64().unwrap(), 1);
        assert_eq!(policy.as_u64().unwrap(), 1);
        assert_eq!(policy.as_str().unwrap(), "SCHED_NORMAL");
    }

    #[test]
    fn rendered_passthrough_returns_raw_value() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let f = snap.var("ctx").get("weight");
        let rendered = f.rendered().expect("weight is a Value");
        match rendered {
            RenderedValue::Uint { bits, value } => {
                assert_eq!(*bits, 32);
                assert_eq!(*value, 1024);
            }
            other => panic!("unexpected rendered shape: {other:?}"),
        }
    }

    #[test]
    fn snapshot_error_display_includes_path_and_alternatives() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        let err = snap.var("ctx").get("nope").error().unwrap().to_string();
        assert!(err.contains("nope"));
        assert!(err.contains("weight"));
    }

    #[test]
    fn var_exact_match_does_not_split_dotted_paths() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        // Chained `var(...).get(...)` walks the rendered struct's
        // members and yields the leaf value — the canonical way to
        // reach a sub-field.
        let chained = snap.var("ctx").get("weight");
        assert_eq!(chained.as_u64().unwrap(), 1024);
        // `Snapshot::var` does not split on `.` — a dotted
        // string is treated as one global variable name. Since
        // no top-level member named `"ctx.weight"` exists, the
        // call resolves to `Missing`.
        let dotted = snap.var("ctx.weight");
        assert!(dotted.error().is_some());
    }

    #[test]
    fn type_mismatch_carries_actual_kind() {
        let r = synthetic_report();
        let snap = Snapshot::new(&r);
        // weight is a Uint — try to read it as bool variant
        // string. Any Value that is not Enum-with-name lands in
        // TypeMismatch.
        let result = snap.var("ctx").get("weight").as_str();
        match result {
            Err(SnapshotError::TypeMismatch {
                expected, actual, ..
            }) => {
                assert_eq!(expected, "str (enum variant name)");
                assert_eq!(actual, "Uint");
            }
            _ => panic!("expected TypeMismatch"),
        }
    }

    /// `SnapshotBridge::drain_ordered` returns every stored
    /// `(name, report)` pair in INSERTION order — the same order the
    /// internal [`SnapshotStore::order`] `VecDeque` records. This is
    /// load-bearing for periodic-capture consumers: the freeze
    /// coordinator's run-loop publishes `periodic_000`, `periodic_001`,
    /// ... at monotonically-increasing wall-clock times, and the test
    /// author needs to walk the captures in the same order to compare
    /// adjacent timeline samples. `drain()` returns a `HashMap` whose
    /// iteration order is non-deterministic across runs, so periodic
    /// consumers MUST go through `drain_ordered` to read the timeline
    /// in cadence order.
    ///
    /// Pin the FIFO contract:
    ///   * insertion order survives through `store()` calls
    ///   * the result is keyed by `String` and carries the full
    ///     `FailureDumpReport` value
    ///   * `drain_ordered()` empties the bridge (matching `drain()`)
    ///     so a follow-up `len()` is 0
    ///   * a tag overwrite refreshes its position to the back, in
    ///     lock-step with the FIFO eviction invariant
    #[test]
    fn snapshot_bridge_drain_ordered_preserves_insertion_order() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        // Insert distinct tags in a non-alphabetical order so an
        // accidental sort-by-key implementation surfaces as a test
        // failure instead of silently appearing to work.
        let inputs: &[&str] = &[
            "periodic_002",
            "periodic_000",
            "periodic_005",
            "periodic_001",
            "periodic_003",
        ];
        for (i, tag) in inputs.iter().enumerate() {
            let r = FailureDumpReport {
                schema: format!("schema_{i}"),
                ..Default::default()
            };
            bridge.store(tag, r);
        }
        let drained: Vec<(String, FailureDumpReport)> = bridge.drain_ordered();
        assert_eq!(
            drained.len(),
            inputs.len(),
            "drain_ordered must yield every stored entry exactly once",
        );
        let drained_names: Vec<&str> = drained.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            drained_names, inputs,
            "drain_ordered must yield insertion order, not sorted or hash order",
        );
        for (i, (_, report)) in drained.iter().enumerate() {
            assert_eq!(
                report.schema,
                format!("schema_{i}"),
                "drained entry {i} must carry the originally-stored report",
            );
        }
        assert_eq!(
            bridge.len(),
            0,
            "drain_ordered must empty the bridge (matching drain())",
        );
        // A subsequent drain_ordered on the empty bridge yields an
        // empty vec — guards against double-drain leaving a stray
        // entry behind in `order` after `reports` is drained.
        let second: Vec<(String, FailureDumpReport)> = bridge.drain_ordered();
        assert!(
            second.is_empty(),
            "second drain_ordered on empty bridge must be empty, got len={}",
            second.len(),
        );
    }

    /// Re-storing an existing tag refreshes its position to the
    /// BACK of the insertion order. This is the same invariant that
    /// `snapshot_bridge_store_overwrite_refreshes_position` pins for
    /// the FIFO eviction path; `drain_ordered` must surface the
    /// refreshed order so downstream consumers see the updated
    /// cadence position. A regression that overwrote the report but
    /// left the order entry in place would surface here as the
    /// refreshed tag still appearing at its original index.
    #[test]
    fn snapshot_bridge_drain_ordered_overwrite_refreshes_position() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        bridge.store("a", FailureDumpReport::default());
        bridge.store("b", FailureDumpReport::default());
        bridge.store("c", FailureDumpReport::default());
        // Overwrite "a" — its position must move from front to
        // back.
        bridge.store(
            "a",
            FailureDumpReport {
                schema: "refreshed".to_string(),
                ..Default::default()
            },
        );
        let drained = bridge.drain_ordered();
        let names: Vec<&str> = drained.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["b", "c", "a"],
            "overwrite of 'a' must move it to the back of the insertion order",
        );
        let a = drained
            .iter()
            .find(|(n, _)| n == "a")
            .expect("'a' resident after overwrite");
        assert_eq!(
            a.1.schema, "refreshed",
            "drain_ordered must surface the refreshed report value, not the prior one",
        );
    }

    /// `store_with_stats` bundles a stats JSON and an elapsed-ms
    /// timestamp alongside the report. `drain_ordered_with_stats`
    /// returns the matching `(tag, report, stats, elapsed)` tuple
    /// per stored entry; non-paired entries (added via plain
    /// `store`) report `None` for both parallel slots.
    #[test]
    fn snapshot_bridge_store_with_stats_round_trips() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        let stats = serde_json::json!({"busy": 75.0});
        bridge.store_with_stats(
            "periodic_000",
            FailureDumpReport::default(),
            Some(stats.clone()),
            Some(123),
        );
        bridge.store("periodic_001", FailureDumpReport::default());
        let drained = bridge.drain_ordered_with_stats();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].0, "periodic_000");
        assert_eq!(drained[0].2, Some(stats));
        assert_eq!(drained[0].3, Some(123));
        assert_eq!(drained[1].0, "periodic_001");
        assert!(drained[1].2.is_none());
        assert!(drained[1].3.is_none());
    }

    /// FIFO eviction at `MAX_STORED_SNAPSHOTS` sweeps the parallel
    /// stats / elapsed maps in lock-step so a stranded entry can
    /// never outlive its report.
    #[test]
    fn snapshot_bridge_store_with_stats_evicts_in_lockstep() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        for i in 0..MAX_STORED_SNAPSHOTS {
            bridge.store_with_stats(
                &format!("tag_{i:04}"),
                FailureDumpReport::default(),
                Some(serde_json::json!({"i": i})),
                Some(i as u64),
            );
        }
        let overflow_tag = format!("tag_{MAX_STORED_SNAPSHOTS:04}");
        bridge.store_with_stats(
            &overflow_tag,
            FailureDumpReport::default(),
            Some(serde_json::json!({"overflow": true})),
            Some(9_999),
        );
        let drained = bridge.drain_ordered_with_stats();
        // tag_0000 must be evicted.
        let names: Vec<&str> = drained.iter().map(|(n, _, _, _)| n.as_str()).collect();
        assert!(!names.contains(&"tag_0000"));
        // Newest must be present with its parallel data.
        let last = drained
            .iter()
            .find(|(n, _, _, _)| n == &overflow_tag)
            .expect("overflow tag resident after evict");
        assert_eq!(last.2, Some(serde_json::json!({"overflow": true})));
        assert_eq!(last.3, Some(9_999));
    }

    /// Overwriting a tag with a `None` stats slot clears the prior
    /// stats — guards against a stale stats / elapsed value
    /// silently surviving across an overwrite that did not bundle
    /// fresh values.
    #[test]
    fn snapshot_bridge_store_with_stats_overwrite_clears_stale_values() {
        let cb: CaptureCallback = Arc::new(|_| None);
        let bridge = SnapshotBridge::new(cb);
        bridge.store_with_stats(
            "periodic_000",
            FailureDumpReport::default(),
            Some(serde_json::json!({"first": true})),
            Some(100),
        );
        // Overwrite via plain `store(...)` — should clear the
        // parallel slots since neither was passed.
        bridge.store("periodic_000", FailureDumpReport::default());
        let drained = bridge.drain_ordered_with_stats();
        assert_eq!(drained.len(), 1);
        assert!(drained[0].2.is_none());
        assert!(drained[0].3.is_none());
    }

    // ---------- stats_path JSON accessor ----------

    /// `stats_path` walks a JSON object along a dotted path and
    /// returns a [`JsonField`] view at the leaf.
    #[test]
    fn stats_path_walks_dotted_path() {
        let v = serde_json::json!({"layers": {"batch": {"util": 75.5}}});
        let f = stats_path(&v, "layers.batch.util");
        assert_eq!(f.as_f64().unwrap(), 75.5);
    }

    /// Empty path returns the root unchanged.
    #[test]
    fn stats_path_empty_returns_root() {
        let v = serde_json::json!(42);
        let f = stats_path(&v, "");
        assert_eq!(f.as_u64().unwrap(), 42);
    }

    /// Missing key surfaces FieldNotFound with the available keys.
    #[test]
    fn stats_path_missing_key_lists_alternatives() {
        let v = serde_json::json!({"busy": 50.0, "antistall": 0});
        let f = stats_path(&v, "missing");
        let err = f.error().expect("missing must error");
        match err {
            SnapshotError::FieldNotFound {
                component,
                available,
                ..
            } => {
                assert_eq!(component, "missing");
                assert!(available.contains(&"busy".to_string()));
                assert!(available.contains(&"antistall".to_string()));
            }
            other => panic!("expected FieldNotFound, got {other:?}"),
        }
    }

    /// Walking through a non-object cursor surfaces NotAStruct.
    #[test]
    fn stats_path_through_scalar_errors_not_a_struct() {
        let v = serde_json::json!({"x": 5});
        let f = stats_path(&v, "x.y");
        match f.error().expect("must error") {
            SnapshotError::NotAStruct { component, .. } => {
                assert_eq!(component, "y");
            }
            other => panic!("expected NotAStruct, got {other:?}"),
        }
    }

    /// Empty path component (`a..b`) reports EmptyPathComponent.
    #[test]
    fn stats_path_empty_component_errors() {
        let v = serde_json::json!({"a": {"b": 1}});
        let f = stats_path(&v, "a..b");
        match f.error().expect("must error") {
            SnapshotError::EmptyPathComponent { requested } => {
                assert_eq!(requested, "a..b");
            }
            other => panic!("expected EmptyPathComponent, got {other:?}"),
        }
    }

    /// String-encoded numeric coerces via as_u64 (scx_stats
    /// stringifies large counters to avoid 53-bit float collapse).
    #[test]
    fn stats_path_string_to_u64_coerces() {
        let v = serde_json::json!({"counter": "12345678901234"});
        let f = stats_path(&v, "counter");
        assert_eq!(f.as_u64().unwrap(), 12_345_678_901_234);
    }

    #[test]
    fn snapshot_error_hash_consistent_with_eq() {
        use std::collections::HashSet;
        let e1 = SnapshotError::VarNotFound {
            requested: "nr_cpus".into(),
            available: vec!["nr_iters".into()],
        };
        let e2 = SnapshotError::VarNotFound {
            requested: "nr_cpus".into(),
            available: vec!["nr_iters".into()],
        };
        let mut set: HashSet<SnapshotError> = HashSet::new();
        set.insert(e1);
        assert!(set.contains(&e2));
    }

    /// Round-trip every SnapshotError variant through serde_json to
    /// pin that the Serialize+Deserialize derives stay byte-stable
    /// across every field shape. Catches a future regression that
    /// reverts a String field to `&'static str` (which would silently
    /// drop the variant from the Deserialize impl).
    #[test]
    fn snapshot_error_serde_round_trip() {
        let cases = vec![
            SnapshotError::MapNotFound {
                requested: "nr_cpus".into(),
                available: vec!["scx_bss".into(), "scx_data".into()],
            },
            SnapshotError::VarNotFound {
                requested: "stall".into(),
                available: vec!["nr_cpus_onln".into()],
            },
            SnapshotError::AmbiguousVar {
                requested: "ctx".into(),
                found_in: vec!["scx_bss".into(), "scx_data".into()],
            },
            SnapshotError::FieldNotFound {
                requested: "scx_bss.ctx.missing".into(),
                walked: "scx_bss.ctx".into(),
                component: "missing".into(),
                available: vec!["nr_cpus".into(), "stall".into()],
            },
            SnapshotError::NotAStruct {
                requested: "scx_bss.stall.x".into(),
                walked: "scx_bss.stall".into(),
                component: "x".into(),
                kind: "Uint".to_string(),
            },
            SnapshotError::TypeMismatch {
                expected: "u64".to_string(),
                actual: "Struct".to_string(),
                requested: "scx_bss.ctx".into(),
            },
            SnapshotError::IndexOutOfRange {
                map: "scx_data".into(),
                index: 5,
                len: 2,
            },
            SnapshotError::PerCpuSlot {
                map: "scx_percpu".into(),
                cpu: 7,
                len: 4,
                unmapped: false,
            },
            SnapshotError::NoMatch {
                map: "scx_data".into(),
                op: "find".to_string(),
                len: 3,
                available_keys: vec!["k0".into(), "k1".into()],
            },
            SnapshotError::EmptyPathComponent {
                requested: "a..b".into(),
            },
            SnapshotError::PerCpuNotNarrowed {
                map: "scx_percpu".into(),
            },
            SnapshotError::NoRendered {
                map: "scx_percpu".into(),
                side: "value".to_string(),
            },
            SnapshotError::PlaceholderSample {
                tag: "primary".into(),
                reason: "vCPU rendezvous timed out".into(),
            },
            SnapshotError::MissingStats {
                tag: "scheduler".into(),
            },
        ];
        for case in cases {
            let json = serde_json::to_string(&case).expect("serialize");
            let back: SnapshotError = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(case, back, "round-trip mismatch for {case:?}");
        }
    }

    /// Contract pin for the documented escape-hatch role of
    /// `Snapshot::report()`. The docstring blesses
    /// `snap.report().<field>` as the official bypass for
    /// FailureDumpReport fields without typed accessors. A
    /// regression that removed `pub fn report()` would silently
    /// break the documented promise without breaking any other
    /// test in the tree (per-field bypass consumers are not yet
    /// landed at the doc time). This test makes the contract
    /// compile-and-test enforced: the method must remain pub, must
    /// pass through the underlying `FailureDumpReport` unchanged.
    #[test]
    fn snapshot_report_escape_hatch_passes_through_underlying_report() {
        let report = FailureDumpReport {
            schema: "escape-hatch-contract-pin".to_string(),
            ..Default::default()
        };
        let snap = Snapshot::new(&report);
        assert_eq!(
            snap.report().schema,
            "escape-hatch-contract-pin",
            "Snapshot::report() must hand back the underlying \
             FailureDumpReport unchanged — this is the documented \
             escape-hatch contract",
        );
    }

    // -------------------------------------------------------------
    // First-class accessor tests — pin every Snapshot / SnapshotMap
    // accessor against the underlying FailureDumpReport / FailureDumpMap
    // field so a rename or removal lights up as a compile error AND
    // the empty-by-default contract holds for callers checking
    // "did the walker run?" via the slice length.
    // -------------------------------------------------------------

    fn task_enrichment_fixture(pid: i32, comm: &str) -> TaskEnrichment {
        // TaskEnrichment derives Default with all zero/empty fields;
        // override only the identity fields the lookup tests pin
        // (and weight/prio/static/normal_prio for the round-trip
        // pin further below — defaults are 0, lookup tests assert
        // non-default values).
        TaskEnrichment {
            pid,
            tgid: pid,
            comm: comm.to_string(),
            weight: 100,
            prio: 120,
            static_prio: 120,
            normal_prio: 120,
            ..Default::default()
        }
    }

    #[test]
    fn snapshot_event_counter_timeline_is_empty_by_default() {
        let r = FailureDumpReport::default();
        let snap = Snapshot::new(&r);
        assert!(snap.event_counter_timeline().is_empty());
    }

    #[test]
    fn snapshot_event_counter_timeline_borrows_populated_vec() {
        let r = FailureDumpReport {
            event_counter_timeline: vec![
                EventCounterSample {
                    elapsed_ms: 10,
                    select_cpu_fallback: 7,
                    ..Default::default()
                },
                EventCounterSample {
                    elapsed_ms: 20,
                    select_cpu_fallback: 11,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        let timeline = snap.event_counter_timeline();
        assert_eq!(timeline.len(), 2);
        assert_eq!(timeline[0].elapsed_ms, 10);
        assert_eq!(timeline[1].select_cpu_fallback, 11);
    }

    #[test]
    fn snapshot_rq_scx_states_borrows_populated_vec() {
        let r = FailureDumpReport {
            rq_scx_states: vec![
                crate::monitor::scx_walker::RqScxState::default(),
                crate::monitor::scx_walker::RqScxState::default(),
            ],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        assert_eq!(snap.rq_scx_states().len(), 2);
    }

    #[test]
    fn snapshot_dsq_states_borrows_populated_vec() {
        let r = FailureDumpReport {
            dsq_states: vec![crate::monitor::scx_walker::DsqState::default()],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        assert_eq!(snap.dsq_states().len(), 1);
    }

    #[test]
    fn snapshot_scx_sched_state_threads_option() {
        let r_absent = FailureDumpReport::default();
        assert!(Snapshot::new(&r_absent).scx_sched_state().is_none());

        let r_present = FailureDumpReport {
            scx_sched_state: Some(crate::monitor::scx_walker::ScxSchedState::default()),
            ..Default::default()
        };
        assert!(Snapshot::new(&r_present).scx_sched_state().is_some());
    }

    #[test]
    fn snapshot_per_cpu_time_at_finds_by_cpu_field_not_position() {
        let r = FailureDumpReport {
            per_cpu_time: vec![
                PerCpuTimeStats {
                    cpu: 2,
                    cpustat_user_ns: 200,
                    ..Default::default()
                },
                PerCpuTimeStats {
                    cpu: 0,
                    cpustat_user_ns: 0,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        assert_eq!(snap.per_cpu_time().len(), 2);
        // Lookup by the `cpu` field, not vec position — first entry
        // has cpu=2 even though it's at index 0.
        let row = snap.per_cpu_time_at(2).expect("cpu 2 present");
        assert_eq!(row.cpustat_user_ns, 200);
        let zero = snap.per_cpu_time_at(0).expect("cpu 0 present");
        assert_eq!(zero.cpustat_user_ns, 0);
        assert!(snap.per_cpu_time_at(99).is_none());
    }

    #[test]
    fn snapshot_per_node_numa_at_finds_by_node_field() {
        let r = FailureDumpReport {
            per_node_numa: vec![
                PerNodeNumaStats {
                    node: 1,
                    numa_hit: 500,
                    ..Default::default()
                },
                PerNodeNumaStats {
                    node: 0,
                    numa_hit: 100,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        assert_eq!(snap.per_node_numa().len(), 2);
        assert_eq!(snap.per_node_numa_at(0).unwrap().numa_hit, 100);
        assert_eq!(snap.per_node_numa_at(1).unwrap().numa_hit, 500);
        assert!(snap.per_node_numa_at(99).is_none());
    }

    #[test]
    fn snapshot_task_enrichment_by_pid_finds_first_match() {
        let r = FailureDumpReport {
            task_enrichments: vec![
                task_enrichment_fixture(42, "alpha"),
                task_enrichment_fixture(7, "beta"),
            ],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        assert_eq!(snap.task_enrichments().len(), 2);
        assert_eq!(snap.task_enrichment_by_pid(7).unwrap().comm, "beta");
        assert_eq!(snap.task_enrichment_by_pid(42).unwrap().comm, "alpha");
        assert!(snap.task_enrichment_by_pid(1).is_none());
    }

    #[test]
    fn snapshot_prog_runtime_stats_by_name_finds_match() {
        let r = FailureDumpReport {
            prog_runtime_stats: vec![
                ProgRuntimeStats {
                    name: "dispatch".into(),
                    cnt: 100,
                    nsecs: 5000,
                    misses: 1,
                },
                ProgRuntimeStats {
                    name: "select_cpu".into(),
                    cnt: 50,
                    nsecs: 2000,
                    misses: 0,
                },
            ],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        assert_eq!(snap.prog_runtime_stats().len(), 2);
        assert_eq!(snap.prog_runtime_stats_by_name("dispatch").unwrap().cnt, 100);
        assert_eq!(snap.prog_runtime_stats_by_name("select_cpu").unwrap().nsecs, 2000);
        assert!(snap.prog_runtime_stats_by_name("nope").is_none());
    }

    /// Pin the first-match-on-duplicate contract on all 4 keyed
    /// lookups. Production captures dedupe before push so this case
    /// shouldn't occur, but the contract is documented as
    /// first-match-wins and a future "return all matches" refactor
    /// without explicit migration would silently surface a
    /// different row.
    #[test]
    fn snapshot_keyed_lookups_return_first_match_on_duplicate() {
        let r = FailureDumpReport {
            per_cpu_time: vec![
                PerCpuTimeStats {
                    cpu: 0,
                    cpustat_user_ns: 11,
                    ..Default::default()
                },
                PerCpuTimeStats {
                    cpu: 0,
                    cpustat_user_ns: 22,
                    ..Default::default()
                },
            ],
            per_node_numa: vec![
                PerNodeNumaStats {
                    node: 0,
                    numa_hit: 100,
                    ..Default::default()
                },
                PerNodeNumaStats {
                    node: 0,
                    numa_hit: 200,
                    ..Default::default()
                },
            ],
            task_enrichments: vec![
                task_enrichment_fixture(7, "first"),
                task_enrichment_fixture(7, "second"),
            ],
            prog_runtime_stats: vec![
                ProgRuntimeStats {
                    name: "p".into(),
                    cnt: 1,
                    nsecs: 10,
                    misses: 0,
                },
                ProgRuntimeStats {
                    name: "p".into(),
                    cnt: 2,
                    nsecs: 20,
                    misses: 0,
                },
            ],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        assert_eq!(snap.per_cpu_time_at(0).unwrap().cpustat_user_ns, 11);
        assert_eq!(snap.per_node_numa_at(0).unwrap().numa_hit, 100);
        assert_eq!(snap.task_enrichment_by_pid(7).unwrap().comm, "first");
        assert_eq!(snap.prog_runtime_stats_by_name("p").unwrap().cnt, 1);
    }

    #[test]
    fn snapshot_probe_counters_threads_option() {
        let r_absent = FailureDumpReport::default();
        assert!(Snapshot::new(&r_absent).probe_counters().is_none());

        let r_present = FailureDumpReport {
            probe_counters: Some(ProbeBssCounters::default()),
            ..Default::default()
        };
        assert!(Snapshot::new(&r_present).probe_counters().is_some());
    }

    fn map_with_shape(
        name: &str,
        ringbuf: Option<FailureDumpRingbuf>,
        arena: Option<ArenaSnapshot>,
        fd_array: Option<FailureDumpFdArray>,
        stack_trace: Option<FailureDumpStackTrace>,
        error: Option<String>,
    ) -> FailureDumpMap {
        FailureDumpMap {
            name: name.into(),
            arena,
            ringbuf,
            stack_trace,
            fd_array,
            error,
            ..Default::default()
        }
    }

    #[test]
    fn snapshot_map_ringbuf_threads_option() {
        let m_absent = map_with_shape("ring", None, None, None, None, None);
        let r1 = FailureDumpReport {
            maps: vec![m_absent],
            ..Default::default()
        };
        let snap1 = Snapshot::new(&r1);
        assert!(snap1.map("ring").unwrap().ringbuf().is_none());

        let m_present = map_with_shape(
            "ring2",
            Some(FailureDumpRingbuf {
                capacity: 4096,
                consumer_pos: 100,
                producer_pos: 500,
                pending_pos: 200,
                pending_bytes: 400,
            }),
            None,
            None,
            None,
            None,
        );
        let r2 = FailureDumpReport {
            maps: vec![m_present],
            ..Default::default()
        };
        let snap2 = Snapshot::new(&r2);
        let rb = snap2.map("ring2").unwrap().ringbuf().expect("present");
        assert_eq!(rb.capacity, 4096);
        assert_eq!(rb.pending_bytes, 400);
    }

    #[test]
    fn snapshot_map_arena_threads_option() {
        let m_absent = map_with_shape("arena_a", None, None, None, None, None);
        let r_a = FailureDumpReport {
            maps: vec![m_absent],
            ..Default::default()
        };
        assert!(Snapshot::new(&r_a).map("arena_a").unwrap().arena().is_none());

        let m_present = map_with_shape(
            "arena_m",
            None,
            Some(ArenaSnapshot::default()),
            None,
            None,
            None,
        );
        let r = FailureDumpReport {
            maps: vec![m_present],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        assert!(snap.map("arena_m").unwrap().arena().is_some());
    }

    #[test]
    fn snapshot_map_fd_array_threads_option() {
        let m_absent = map_with_shape("fda_a", None, None, None, None, None);
        let r_a = FailureDumpReport {
            maps: vec![m_absent],
            ..Default::default()
        };
        assert!(Snapshot::new(&r_a).map("fda_a").unwrap().fd_array().is_none());

        let m_present = map_with_shape(
            "fda",
            None,
            None,
            Some(FailureDumpFdArray {
                populated: 3,
                scanned: 5,
                indices: vec![0, 2, 4],
                truncated: false,
                indices_truncated: false,
            }),
            None,
            None,
        );
        let r = FailureDumpReport {
            maps: vec![m_present],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        let fda = snap.map("fda").unwrap().fd_array().expect("present");
        assert_eq!(fda.populated, 3);
        assert_eq!(fda.indices, vec![0, 2, 4]);
    }

    #[test]
    fn snapshot_map_stack_trace_threads_option() {
        let m_absent = map_with_shape("stack_a", None, None, None, None, None);
        let r_a = FailureDumpReport {
            maps: vec![m_absent],
            ..Default::default()
        };
        assert!(
            Snapshot::new(&r_a)
                .map("stack_a")
                .unwrap()
                .stack_trace()
                .is_none()
        );

        let m_present = map_with_shape(
            "stack",
            None,
            None,
            None,
            Some(FailureDumpStackTrace {
                n_buckets: 32,
                entries: Vec::new(),
                truncated: false,
            }),
            None,
        );
        let r = FailureDumpReport {
            maps: vec![m_present],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        let st = snap.map("stack").unwrap().stack_trace().expect("present");
        assert_eq!(st.n_buckets, 32);
    }

    #[test]
    fn snapshot_map_error_threads_option() {
        let m_absent = map_with_shape("err_a", None, None, None, None, None);
        let r_a = FailureDumpReport {
            maps: vec![m_absent],
            ..Default::default()
        };
        assert!(Snapshot::new(&r_a).map("err_a").unwrap().map_error().is_none());

        let m_present = map_with_shape(
            "err_m",
            None,
            None,
            None,
            None,
            Some("BTF offset unresolved".to_string()),
        );
        let r = FailureDumpReport {
            maps: vec![m_present],
            ..Default::default()
        };
        let snap = Snapshot::new(&r);
        assert_eq!(
            snap.map("err_m").unwrap().map_error(),
            Some("BTF offset unresolved")
        );
    }

    // -------------------------------------------------------------
    // `*_unavailable` companion accessors — pin against the
    // matching `Option<String>` field on FailureDumpReport. Each
    // accessor returns `None` for the default report and
    // `Some(&str)` after the walker writes a reason.
    // -------------------------------------------------------------

    #[test]
    fn snapshot_unavailable_accessors_thread_option() {
        let r_absent = FailureDumpReport::default();
        let s = Snapshot::new(&r_absent);
        assert!(s.scx_walker_unavailable().is_none());
        assert!(s.task_enrichments_unavailable().is_none());
        assert!(s.prog_runtime_stats_unavailable().is_none());
        assert!(s.per_node_numa_unavailable().is_none());
        assert!(s.sdt_alloc_unavailable().is_none());

        let r_present = FailureDumpReport {
            scx_walker_unavailable: Some("scx_root unset".into()),
            task_enrichments_unavailable: Some("no task walker available".into()),
            prog_runtime_stats_unavailable: Some("prog accessor unavailable".into()),
            per_node_numa_unavailable: Some("no NUMA walker".into()),
            sdt_alloc_unavailable: Some("sdt symbol absent".into()),
            ..Default::default()
        };
        let s = Snapshot::new(&r_present);
        assert_eq!(s.scx_walker_unavailable(), Some("scx_root unset"));
        assert_eq!(
            s.task_enrichments_unavailable(),
            Some("no task walker available")
        );
        assert_eq!(
            s.prog_runtime_stats_unavailable(),
            Some("prog accessor unavailable")
        );
        assert_eq!(s.per_node_numa_unavailable(), Some("no NUMA walker"));
        assert_eq!(s.sdt_alloc_unavailable(), Some("sdt symbol absent"));
    }

    // -------------------------------------------------------------
    // Serde round-trip pin for the 13 new accessor target types.
    // Populates a FailureDumpReport with one entry per accessor,
    // serializes to JSON, deserializes back, and hits every new
    // accessor to assert field-level values match. Pins the wire
    // format against silent `#[serde(skip)]` regressions on any
    // of the underlying structs.
    // -------------------------------------------------------------

    #[test]
    fn snapshot_accessor_targets_round_trip_through_serde_json() {
        let original = FailureDumpReport {
            event_counter_timeline: vec![EventCounterSample {
                elapsed_ms: 42,
                select_cpu_fallback: 7,
                ..Default::default()
            }],
            rq_scx_states: vec![crate::monitor::scx_walker::RqScxState {
                cpu: 5,
                ..Default::default()
            }],
            dsq_states: vec![crate::monitor::scx_walker::DsqState {
                nr: 9,
                ..Default::default()
            }],
            scx_sched_state: Some(crate::monitor::scx_walker::ScxSchedState {
                exit_kind: 3,
                ..Default::default()
            }),
            per_cpu_time: vec![PerCpuTimeStats {
                cpu: 3,
                cpustat_user_ns: 9_999,
                ..Default::default()
            }],
            per_node_numa: vec![PerNodeNumaStats {
                node: 1,
                numa_hit: 12_345,
                ..Default::default()
            }],
            task_enrichments: vec![task_enrichment_fixture(101, "rt-test")],
            prog_runtime_stats: vec![ProgRuntimeStats {
                name: "dispatch".into(),
                cnt: 50,
                nsecs: 12_500,
                misses: 2,
            }],
            probe_counters: Some(ProbeBssCounters {
                trigger_count: 17,
                ..Default::default()
            }),
            maps: vec![FailureDumpMap {
                name: "shape_map".into(),
                map_type: 0,
                value_size: 0,
                max_entries: 0,
                value: None,
                entries: Vec::new(),
                percpu_entries: Vec::new(),
                percpu_hash_entries: Vec::new(),
                arena: Some(ArenaSnapshot {
                    declared_pages: 256,
                    ..Default::default()
                }),
                ringbuf: Some(FailureDumpRingbuf {
                    capacity: 8192,
                    consumer_pos: 50,
                    producer_pos: 700,
                    pending_pos: 100,
                    pending_bytes: 650,
                }),
                stack_trace: Some(FailureDumpStackTrace {
                    n_buckets: 16,
                    entries: Vec::new(),
                    truncated: true,
                }),
                fd_array: Some(FailureDumpFdArray {
                    populated: 2,
                    scanned: 4,
                    indices: vec![1, 3],
                    truncated: false,
                    indices_truncated: false,
                }),
                error: Some("decode failed".to_string()),
            }],
            ..Default::default()
        };

        let json = serde_json::to_string(&original).expect("serialize");
        let round_tripped: FailureDumpReport =
            serde_json::from_str(&json).expect("deserialize");
        let snap = Snapshot::new(&round_tripped);

        assert_eq!(snap.event_counter_timeline().len(), 1);
        assert_eq!(snap.event_counter_timeline()[0].elapsed_ms, 42);
        assert_eq!(snap.event_counter_timeline()[0].select_cpu_fallback, 7);

        assert_eq!(snap.rq_scx_states().len(), 1);
        assert_eq!(snap.rq_scx_states()[0].cpu, 5);
        assert_eq!(snap.dsq_states().len(), 1);
        assert_eq!(snap.dsq_states()[0].nr, 9);
        let sched = snap.scx_sched_state().expect("scx_sched_state");
        assert_eq!(sched.exit_kind, 3);

        assert_eq!(snap.per_cpu_time().len(), 1);
        let cpu_row = snap.per_cpu_time_at(3).expect("cpu 3 row");
        assert_eq!(cpu_row.cpustat_user_ns, 9_999);

        assert_eq!(snap.per_node_numa().len(), 1);
        let numa_row = snap.per_node_numa_at(1).expect("node 1 row");
        assert_eq!(numa_row.numa_hit, 12_345);

        assert_eq!(snap.task_enrichments().len(), 1);
        let task = snap.task_enrichment_by_pid(101).expect("pid 101");
        assert_eq!(task.comm, "rt-test");
        // Fixture defaults pin non-pid/comm fields too — verifies
        // they survive serde without #[serde(skip)] regressions.
        assert_eq!(task.weight, 100);
        assert_eq!(task.prio, 120);

        assert_eq!(snap.prog_runtime_stats().len(), 1);
        let prog = snap
            .prog_runtime_stats_by_name("dispatch")
            .expect("dispatch prog");
        assert_eq!(prog.cnt, 50);
        assert_eq!(prog.nsecs, 12_500);
        assert_eq!(prog.misses, 2);

        let probe = snap.probe_counters().expect("probe_counters");
        assert_eq!(probe.trigger_count, 17);

        let map = snap.map("shape_map").expect("map present");
        let rb = map.ringbuf().expect("ringbuf");
        assert_eq!(rb.capacity, 8192);
        assert_eq!(rb.pending_bytes, 650);
        let arena = map.arena().expect("arena");
        assert_eq!(arena.declared_pages, 256);
        let fda = map.fd_array().expect("fd_array");
        assert_eq!(fda.populated, 2);
        assert_eq!(fda.indices, vec![1, 3]);
        let st = map.stack_trace().expect("stack_trace");
        assert_eq!(st.n_buckets, 16);
        assert!(st.truncated);
        assert_eq!(map.map_error(), Some("decode failed"));
    }

    // -- SnapshotBridgeEvent + drain_events -------------------------------

    /// Construct a bridge whose capture callback always returns the
    /// caller-supplied report. Tests that need to trigger
    /// `CaptureUnavailable` build their own bridge with a None-returning
    /// callback inline.
    fn bridge_with_capture_returning(report: FailureDumpReport) -> SnapshotBridge {
        SnapshotBridge::new(std::sync::Arc::new(move |_name| Some(report.clone())))
    }

    /// `capture` whose callback returns `None` records exactly one
    /// `CaptureUnavailable` event under the requested tag and returns
    /// `false` (the Op::Snapshot no-op contract).
    #[test]
    fn snapshot_bridge_event_capture_unavailable_recorded() {
        let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
        assert!(!bridge.capture("tag_x"));
        let events = bridge.drain_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            SnapshotBridgeEvent::CaptureUnavailable { tag } => {
                assert_eq!(tag, "tag_x");
            }
            other => panic!("expected CaptureUnavailable, got {other:?}"),
        }
    }

    /// `store` of a tag that already has a stored report records
    /// exactly one `Overwrite` event with the prior report's `schema`.
    #[test]
    fn snapshot_bridge_event_overwrite_recorded() {
        let bridge = bridge_with_capture_returning(synthetic_report());
        bridge.store("dup_tag", synthetic_report());
        bridge.store("dup_tag", synthetic_report());
        let events = bridge.drain_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            SnapshotBridgeEvent::Overwrite { tag, prior_schema } => {
                assert_eq!(tag, "dup_tag");
                assert_eq!(prior_schema, SCHEMA_SINGLE);
            }
            other => panic!("expected Overwrite, got {other:?}"),
        }
    }

    /// `store` that triggers FIFO cap-eviction records one `Eviction`
    /// event per evicted tag, with `cap` reflecting
    /// [`MAX_STORED_SNAPSHOTS`] and `new_tag` naming the storing
    /// operation that pushed the bridge over the cap.
    #[test]
    fn snapshot_bridge_event_eviction_recorded() {
        let bridge = bridge_with_capture_returning(synthetic_report());
        for i in 0..MAX_STORED_SNAPSHOTS {
            bridge.store(&format!("tag_{i}"), synthetic_report());
        }
        // The first MAX_STORED_SNAPSHOTS stores fit; none should
        // have evicted anything.
        assert_eq!(bridge.event_count(), 0);
        // One more store crosses the cap — exactly one Eviction.
        bridge.store("overflow_tag", synthetic_report());
        let events = bridge.drain_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            SnapshotBridgeEvent::Eviction {
                evicted_tag,
                new_tag,
                cap,
            } => {
                assert_eq!(evicted_tag, "tag_0");
                assert_eq!(new_tag, "overflow_tag");
                assert_eq!(*cap, MAX_STORED_SNAPSHOTS);
            }
            other => panic!("expected Eviction, got {other:?}"),
        }
    }

    /// `drain_events` empties the log — a follow-up call returns an
    /// empty vec without re-yielding the previously-drained events.
    #[test]
    fn snapshot_bridge_drain_events_consumes_log() {
        let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
        bridge.capture("a");
        bridge.capture("b");
        let first = bridge.drain_events();
        assert_eq!(first.len(), 2);
        let second = bridge.drain_events();
        assert!(second.is_empty(), "second drain must return empty vec");
        assert_eq!(bridge.event_count(), 0);
    }

    /// `event_count` reports queued events without consuming them —
    /// a follow-up `drain_events` still yields the same events.
    #[test]
    fn snapshot_bridge_event_count_is_non_draining() {
        let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
        bridge.capture("only");
        assert_eq!(bridge.event_count(), 1);
        // Re-checking does not drain.
        assert_eq!(bridge.event_count(), 1);
        let events = bridge.drain_events();
        assert_eq!(events.len(), 1);
    }

    /// A bridge whose storage and drain paths run cleanly produces
    /// zero events — the structured log is silent on the happy path.
    #[test]
    fn snapshot_bridge_no_events_on_clean_capture_and_drain() {
        let bridge = bridge_with_capture_returning(synthetic_report());
        assert!(bridge.capture("clean_a"));
        assert!(bridge.capture("clean_b"));
        let _ = bridge.drain_ordered_with_stats();
        assert_eq!(
            bridge.event_count(),
            0,
            "clean capture-then-drain must not record any bridge events",
        );
    }

    /// `tracing::warn!` is preserved alongside the event push — the
    /// "promote, don't replace" contract. Cannot directly observe the
    /// warn without a subscriber install; this test pins that the
    /// event is still recorded even after a clean drain (the warn
    /// path is exercised the same way regardless of subscriber state,
    /// so observing the structured event covers both axes).
    #[test]
    fn snapshot_bridge_event_recorded_even_after_drain() {
        let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
        bridge.capture("post_drain_tag");
        let _ = bridge.drain_ordered_with_stats();
        // Events are independent of report drain — they remain
        // queued for inspection until drain_events is called.
        assert_eq!(bridge.event_count(), 1);
        let events = bridge.drain_events();
        assert!(
            matches!(
                events[0],
                SnapshotBridgeEvent::CaptureUnavailable { ref tag } if tag == "post_drain_tag",
            ),
            "post-drain event must remain in the log",
        );
    }

    /// `DrainOrderingInvariantViolation` fires from
    /// [`SnapshotBridge::drain_ordered`] when a report exists in
    /// `reports` without a matching entry in `order`. The
    /// public API keeps the two collections in lock-step, so the
    /// only way to exercise this path is to poke `snapshots`
    /// directly from inside the test module — simulating an internal
    /// refactor bug that desynchronised the two. Pins both the
    /// tag field and the `drain_variant` literal so a regression
    /// that drops the event push (leaving only the warn) surfaces
    /// here.
    #[test]
    fn snapshot_bridge_event_drain_ordering_invariant_violation_drain_ordered() {
        let bridge = bridge_with_capture_returning(synthetic_report());
        {
            let mut store = bridge.snapshots.lock_unpoisoned();
            store
                .reports
                .insert("orphan_tag".to_string(), synthetic_report());
        }
        let drained = bridge.drain_ordered();
        assert_eq!(drained.len(), 1, "orphan must surface at the tail");
        assert_eq!(drained[0].0, "orphan_tag");
        let events = bridge.drain_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            SnapshotBridgeEvent::DrainOrderingInvariantViolation {
                tag,
                drain_variant,
            } => {
                assert_eq!(tag, "orphan_tag");
                assert_eq!(*drain_variant, "drain_ordered");
            }
            other => panic!("expected DrainOrderingInvariantViolation, got {other:?}"),
        }
    }

    /// Parallel test for `drain_ordered_with_stats` — the second
    /// production site for `DrainOrderingInvariantViolation`. The
    /// `drain_variant` discriminator is the only difference; a single
    /// test cannot pin both sites simultaneously because each drain
    /// method consumes the desync.
    #[test]
    fn snapshot_bridge_event_drain_ordering_invariant_violation_drain_ordered_with_stats() {
        let bridge = bridge_with_capture_returning(synthetic_report());
        {
            let mut store = bridge.snapshots.lock_unpoisoned();
            store
                .reports
                .insert("orphan_tag".to_string(), synthetic_report());
        }
        let drained = bridge.drain_ordered_with_stats();
        assert_eq!(drained.len(), 1, "orphan must surface at the tail");
        assert_eq!(drained[0].0, "orphan_tag");
        let events = bridge.drain_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            SnapshotBridgeEvent::DrainOrderingInvariantViolation {
                tag,
                drain_variant,
            } => {
                assert_eq!(tag, "orphan_tag");
                assert_eq!(*drain_variant, "drain_ordered_with_stats");
            }
            other => panic!("expected DrainOrderingInvariantViolation, got {other:?}"),
        }
    }

    /// The cap-enforcement loop supports multi-iteration eviction
    /// (e.g. if `reports.len()` ever exceeds `cap + 1`). The public
    /// API never produces this state because `store_internal`
    /// inserts one entry and pops one per loop iteration, but a
    /// future refactor that pushed N entries in one shot would
    /// rely on the loop running N - cap times. Pre-seed
    /// `reports + order` in lock-step with `cap + 1` entries, then
    /// trigger one store to drive the loop past the cap by 2, and
    /// pin that exactly 2 Eviction events fire — one per iteration,
    /// in FIFO order. A regression that moved the `events.push`
    /// outside the while-loop body (single push after the loop with
    /// the last evicted tag) would only fire 1 event and fail here.
    #[test]
    fn snapshot_bridge_event_eviction_loop_fires_once_per_iteration() {
        let bridge = bridge_with_capture_returning(synthetic_report());
        {
            let mut store = bridge.snapshots.lock_unpoisoned();
            for i in 0..=MAX_STORED_SNAPSHOTS {
                let tag = format!("seed_{i:03}");
                store.reports.insert(tag.clone(), synthetic_report());
                store.order.push_back(tag);
            }
        }
        bridge.store("trigger_tag", synthetic_report());
        let events = bridge.drain_events();
        assert_eq!(
            events.len(),
            2,
            "loop must fire exactly one Eviction per popped entry, not one batched event for the last pop"
        );
        match &events[0] {
            SnapshotBridgeEvent::Eviction {
                evicted_tag,
                new_tag,
                cap,
            } => {
                assert_eq!(evicted_tag, "seed_000", "first pop is FIFO-oldest");
                assert_eq!(new_tag, "trigger_tag");
                assert_eq!(*cap, MAX_STORED_SNAPSHOTS);
            }
            other => panic!("expected Eviction at events[0], got {other:?}"),
        }
        match &events[1] {
            SnapshotBridgeEvent::Eviction {
                evicted_tag,
                new_tag,
                cap,
            } => {
                assert_eq!(evicted_tag, "seed_001", "second pop is FIFO-next-oldest");
                assert_eq!(new_tag, "trigger_tag");
                assert_eq!(*cap, MAX_STORED_SNAPSHOTS);
            }
            other => panic!("expected Eviction at events[1], got {other:?}"),
        }
    }

    /// `CapInvariantViolation` fires when the cap-enforcement loop
    /// in `store_internal` finds `reports.len() > cap` while `order`
    /// is empty — the bulk-clear branch nukes everything to restore
    /// the invariant. Unreachable through the public API (which
    /// always keeps the two collections in lock-step), so this test
    /// pre-seeds the desync directly: `cap + 2` orphan entries in
    /// `reports` with empty `order`, then `bridge.store("trigger_tag",
    /// ...)` to enter `store_internal`. `store_internal` pushes
    /// trigger_tag onto `order` BEFORE the loop, so the first loop
    /// iteration pops trigger_tag (Eviction event with self-eviction
    /// shape — both evicted_tag and new_tag = trigger_tag); the
    /// second iteration finds `order` empty while `reports.len()`
    /// still over cap and fires CapInvariantViolation, bulk-clears,
    /// then breaks. Test asserts both events fire in this order so a
    /// regression in either site surfaces.
    #[test]
    fn snapshot_bridge_event_cap_invariant_violation_recorded() {
        let bridge = bridge_with_capture_returning(synthetic_report());
        {
            let mut store = bridge.snapshots.lock_unpoisoned();
            for i in 0..(MAX_STORED_SNAPSHOTS + 2) {
                store
                    .reports
                    .insert(format!("orphan_{i:03}"), synthetic_report());
            }
        }
        bridge.store("trigger_tag", synthetic_report());
        let events = bridge.drain_events();
        assert_eq!(
            events.len(),
            2,
            "expected one Eviction (self-evicting trigger_tag from order) + one CapInvariantViolation (bulk-clearing the remaining orphans)"
        );
        match &events[0] {
            SnapshotBridgeEvent::Eviction {
                evicted_tag,
                new_tag,
                cap,
            } => {
                assert_eq!(
                    evicted_tag, "trigger_tag",
                    "self-eviction: trigger_tag is the only entry in order"
                );
                assert_eq!(new_tag, "trigger_tag");
                assert_eq!(*cap, MAX_STORED_SNAPSHOTS);
            }
            other => panic!("expected Eviction at events[0], got {other:?}"),
        }
        match &events[1] {
            SnapshotBridgeEvent::CapInvariantViolation { reports_len, cap } => {
                assert_eq!(
                    *reports_len,
                    MAX_STORED_SNAPSHOTS + 2,
                    "reports_len at bulk-clear: 66 orphans remain after trigger_tag was evicted in iteration 1"
                );
                assert_eq!(*cap, MAX_STORED_SNAPSHOTS);
            }
            other => panic!("expected CapInvariantViolation at events[1], got {other:?}"),
        }
        let store = bridge.snapshots.lock_unpoisoned();
        assert_eq!(
            store.reports.len(),
            0,
            "bulk-clear must nuke reports to restore the invariant"
        );
        assert_eq!(store.stats.len(), 0);
        assert_eq!(store.elapsed_ms.len(), 0);
    }

    /// `events` Vec is capped at [`MAX_STORED_EVENTS`] via FIFO
    /// eviction. A scenario that triggers many events without
    /// draining (e.g. a runaway capture loop) must NOT grow the
    /// log unboundedly. Push `cap + 5` events, verify `event_count`
    /// is exactly `cap`, then drain and verify the drained vec is
    /// `cap + 1` long (cap real events + 1 synthetic
    /// [`SnapshotBridgeEvent::EventLogTruncated`] at the tail) with
    /// `dropped_count = 5`. Also pins FIFO eviction order: the
    /// oldest events are dropped, the newest are retained.
    #[test]
    fn snapshot_bridge_events_capped_at_max_stored_events() {
        let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
        let total = MAX_STORED_EVENTS + 5;
        for i in 0..total {
            bridge.capture(&format!("runaway_{i:05}"));
        }
        assert_eq!(
            bridge.event_count(),
            MAX_STORED_EVENTS,
            "events Vec must be capped at MAX_STORED_EVENTS (event_count excludes the synthetic truncation marker)"
        );
        let drained = bridge.drain_events();
        assert_eq!(
            drained.len(),
            MAX_STORED_EVENTS + 1,
            "drain must yield cap real events + 1 synthetic EventLogTruncated tail marker"
        );
        // The first event surviving FIFO eviction is the 6th push
        // (indices 0-4 = 5 events were dropped to make room).
        match &drained[0] {
            SnapshotBridgeEvent::CaptureUnavailable { tag } => {
                assert_eq!(
                    tag, "runaway_00005",
                    "oldest surviving event must be index 5 (indices 0-4 dropped by FIFO eviction)"
                );
            }
            other => panic!("expected CaptureUnavailable at events[0], got {other:?}"),
        }
        // The tail marker carries the dropped count.
        match drained.last().expect("non-empty drain") {
            SnapshotBridgeEvent::EventLogTruncated { dropped_count } => {
                assert_eq!(
                    *dropped_count, 5,
                    "5 events were dropped from the front before the cap held"
                );
            }
            other => panic!("expected EventLogTruncated at events[last], got {other:?}"),
        }
    }

    /// Pushing EXACTLY [`MAX_STORED_EVENTS`] events must NOT trigger
    /// any eviction — the cap-enforcement predicate at L766 is
    /// `>=`, so the cap-th push fits without dropping. Drain then
    /// yields exactly `cap` events with NO `EventLogTruncated` tail
    /// marker. Pins the boundary: a regression that changed `>=`
    /// to `>` would silently shift the cap by one and only be
    /// catchable via tests that exercise the exact-cap case (the
    /// existing over-cap tests would still pass with a one-off
    /// dropped_count). Tester recommended this test in #77 pass 3
    /// coverage-gap analysis.
    #[test]
    fn snapshot_bridge_events_exactly_at_cap_no_truncation_marker() {
        let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
        for i in 0..MAX_STORED_EVENTS {
            bridge.capture(&format!("at_cap_{i:05}"));
        }
        assert_eq!(
            bridge.event_count(),
            MAX_STORED_EVENTS,
            "exact-cap push must NOT trigger eviction — len == cap, dropped == 0"
        );
        let drained = bridge.drain_events();
        assert_eq!(
            drained.len(),
            MAX_STORED_EVENTS,
            "drain must yield exactly cap events with NO synthetic tail marker"
        );
        assert!(
            drained.iter().all(|e| !matches!(
                e,
                SnapshotBridgeEvent::EventLogTruncated { .. }
            )),
            "no EventLogTruncated marker at exact-cap — events_dropped must remain 0"
        );
    }

    /// A second overflow batch after a clean drain must produce a
    /// FRESH truncation marker with its own dropped_count — NOT 0
    /// (missed reset), NOT carried-forward from the prior batch,
    /// NOT accumulated. Pins the events_dropped reset path against
    /// regressions where the counter persists across drains (e.g.
    /// `events_dropped = events_dropped.saturating_sub(...)`
    /// instead of `= 0`, or a reset path that forgets to fire on
    /// the empty-events case). Tester recommended this test in #77
    /// pass 3 coverage-gap analysis.
    #[test]
    fn snapshot_bridge_events_truncation_marker_fresh_after_reset() {
        let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
        // First overflow: cap+5 push → marker with dropped_count=5.
        for i in 0..(MAX_STORED_EVENTS + 5) {
            bridge.capture(&format!("first_{i:05}"));
        }
        let first = bridge.drain_events();
        match first.last().expect("non-empty first drain") {
            SnapshotBridgeEvent::EventLogTruncated { dropped_count } => {
                assert_eq!(*dropped_count, 5, "first batch dropped 5");
            }
            other => panic!("expected EventLogTruncated tail in first drain, got {other:?}"),
        }
        // Second overflow: cap+7 push → marker with dropped_count=7
        // (fresh count, NOT 5+7=12 stale-accumulated, NOT 0
        // missed-reset).
        for i in 0..(MAX_STORED_EVENTS + 7) {
            bridge.capture(&format!("second_{i:05}"));
        }
        let second = bridge.drain_events();
        match second.last().expect("non-empty second drain") {
            SnapshotBridgeEvent::EventLogTruncated { dropped_count } => {
                assert_eq!(
                    *dropped_count, 7,
                    "second batch dropped 7 from a clean post-reset counter"
                );
            }
            other => panic!("expected EventLogTruncated tail in second drain, got {other:?}"),
        }
    }

    /// `events_dropped` resets to 0 after every `drain_events` call —
    /// a subsequent drain with no over-cap pushes in between must NOT
    /// re-emit a stale truncation marker. Push to overflow, drain
    /// (consuming the marker), push within-cap, drain again — the
    /// second drain must NOT include EventLogTruncated.
    #[test]
    fn snapshot_bridge_events_dropped_resets_after_drain() {
        let bridge = SnapshotBridge::new(std::sync::Arc::new(|_name| None));
        for i in 0..(MAX_STORED_EVENTS + 3) {
            bridge.capture(&format!("over_{i:05}"));
        }
        let first = bridge.drain_events();
        assert!(matches!(
            first.last(),
            Some(SnapshotBridgeEvent::EventLogTruncated { dropped_count: 3 })
        ));
        // Second drain immediately — events log is empty, dropped
        // counter reset. Must be empty, NOT carry a stale truncation
        // marker.
        let second = bridge.drain_events();
        assert!(
            second.is_empty(),
            "second drain must NOT re-emit EventLogTruncated — dropped counter reset to 0 after first drain"
        );
        // Push 3 events that fit under the cap. Drain — no
        // truncation marker because nothing was dropped this batch.
        for i in 0..3 {
            bridge.capture(&format!("clean_{i:05}"));
        }
        let third = bridge.drain_events();
        assert_eq!(third.len(), 3, "3 captures, no truncation marker");
        assert!(third.iter().all(|e| !matches!(
            e,
            SnapshotBridgeEvent::EventLogTruncated { .. }
        )));
    }
}
