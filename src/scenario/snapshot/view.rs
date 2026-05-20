//! [`Snapshot`] is the entry point for a captured
//! [`FailureDumpReport`], plus [`SnapshotMap`] for typed traversal of
//! one map and the per-CPU resolver helpers it uses to project
//! per-CPU array / hash entries down to a single slot.
//!
//! [`render_entry_key`] formats a [`SnapshotEntry`] key for the
//! `NoMatch` diagnostic; lives here because it walks the same
//! `SnapshotMap` entry shapes the type uses internally.

use crate::monitor::arena::ArenaSnapshot;
use crate::monitor::bpf_prog::ProgRuntimeStats;
use crate::monitor::btf_render::RenderedValue;
use crate::monitor::dump::{
    EventCounterSample, FailureDumpFdArray, FailureDumpMap, FailureDumpPercpuEntry,
    FailureDumpPercpuHashEntry, FailureDumpReport, FailureDumpRingbuf, FailureDumpStackTrace,
    PerCpuTimeStats, PerNodeNumaStats, ProbeBssCounters,
};
use crate::monitor::scx_walker::{DsqState, RqScxState, ScxSchedState};
use crate::monitor::task_enrichment::TaskEnrichment;

use super::field::lookup_member;
use super::{
    HEX_KEY_PREFIX, NO_MATCH_KEY_CHAR_CAP, NO_MATCH_KEY_SAMPLE, SnapshotEntry, SnapshotError,
    SnapshotField, SnapshotResult,
};

/// Borrowed view over a captured [`FailureDumpReport`] for typed
/// traversal of BTF-rendered map values, per-CPU entries, and
/// scalar variables.
///
/// Constructed from a [`FailureDumpReport`] reference (typically
/// obtained via [`super::SnapshotBridge::drain`]); the view is cheap to
/// build — it does not copy the underlying report. Accessor
/// methods all return further borrowed views that walk the report
/// in place.
#[derive(Debug, Clone)]
#[must_use = "Snapshot is a borrowed view; bind or chain accessors"]
#[non_exhaustive]
pub struct Snapshot<'a> {
    report: &'a FailureDumpReport,
    /// When `Some`, every map-walking accessor filters
    /// [`FailureDumpReport::maps`] to maps whose `name` begins with
    /// `<obj>.`. Populated by [`Self::active`] from the snapshot's
    /// own `scx_sched_state` + `prog_runtime_stats`; `None` when the
    /// snapshot was constructed via [`Self::new`] (unfiltered).
    active_obj: Option<&'a str>,
}

impl<'a> Snapshot<'a> {
    /// Build a borrowed view over `report` with no active-scheduler
    /// filter. Every map-walking accessor sees every captured map.
    pub fn new(report: &'a FailureDumpReport) -> Self {
        Self {
            report,
            active_obj: None,
        }
    }

    /// Iterate maps the current view exposes — every captured map
    /// when `active_obj` is None; only maps whose name shares the
    /// `<obj>.` prefix when [`Self::active`] populated the filter.
    fn maps_iter(&self) -> impl Iterator<Item = &'a FailureDumpMap> + '_ {
        let active = self.active_obj;
        self.report.maps.iter().filter(move |m| match active {
            None => true,
            Some(obj) => map_belongs_to_obj(&m.name, obj),
        })
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

    /// Look up a BPF map by exact name. Respects the
    /// [`Self::active`] filter when set — only maps the filter
    /// admits are considered. Returns [`SnapshotError::MapNotFound`]
    /// (with the captured map names in `available`) when no match
    /// is found among the admitted maps, or
    /// [`SnapshotError::PlaceholderSnapshot`] when the snapshot's
    /// underlying `FailureDumpReport` is a placeholder (freeze
    /// rendezvous failed; no maps to walk).
    pub fn map(&self, name: &str) -> SnapshotResult<SnapshotMap<'a>> {
        if self.report.is_placeholder {
            return Err(SnapshotError::PlaceholderSnapshot { tag: None });
        }
        for m in self.maps_iter() {
            if m.name == name {
                return Ok(SnapshotMap { map: m, cpu: None });
            }
        }
        Err(SnapshotError::MapNotFound {
            requested: name.to_string(),
            available: self.maps_iter().map(|m| m.name.clone()).collect(),
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
        if self.report.is_placeholder {
            return SnapshotField::Missing(SnapshotError::PlaceholderSnapshot { tag: None });
        }
        let mut hits: Vec<(&'a str, &'a RenderedValue)> = Vec::new();
        for m in self.maps_iter() {
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
                for m in self.maps_iter() {
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

    /// Iterate every global-section copy that carries a top-level
    /// member named `name`. Yields `(owning_map_name, field)` pairs
    /// in capture order. Use when [`Self::var`] errors
    /// [`SnapshotError::AmbiguousVar`] and the caller needs to
    /// reason across every observed copy explicitly (e.g. summing
    /// counter deltas across two scheduler instances loaded
    /// back-to-back in the same scenario).
    ///
    /// Respects the [`Self::active`] filter when set, so chained
    /// `snapshot.active()?.vars(name)` is well-defined — it iterates
    /// only the active scheduler's copies (typically exactly one,
    /// since active() filters to one obj_name).
    ///
    /// Yields nothing on placeholder snapshots (the underlying
    /// `report.maps` is empty by construction so nothing matches
    /// anyway — callers needing "is this a placeholder?" use the
    /// `Snapshot::is_placeholder` accessor explicitly).
    pub fn vars(&self, name: &str) -> impl Iterator<Item = (&'a str, SnapshotField<'a>)> + '_ {
        let needle = name.to_string();
        self.maps_iter().filter_map(move |m| {
            if !is_global_section_map(&m.name) {
                return None;
            }
            let v = m.value.as_ref()?;
            let found = lookup_member(v, &needle)?;
            Some((m.name.as_str(), SnapshotField::Value(found)))
        })
    }

    /// Project the snapshot to the currently-active scheduler's
    /// maps. Returns a filtered [`Snapshot`] whose [`Self::map`] /
    /// [`Self::var`] / [`Self::vars`] see only the maps whose name
    /// shares the `<obj>.` prefix of the active scheduler's BPF
    /// object. Composable: `snapshot.active()?.var(name)`.
    ///
    /// # When to use
    ///
    /// Tests that swap schedulers mid-scenario (via
    /// [`crate::scenario::ops::Op::ReplaceScheduler`]) reach for
    /// `.active()` after the swap so the per-phase post-swap
    /// snapshots resolve the live scheduler's bss without hitting
    /// [`SnapshotError::AmbiguousVar`] across both schedulers'
    /// captured copies. Single-scheduler tests never need
    /// `.active()` — there is no ambiguity to resolve.
    ///
    /// # Signal source
    ///
    /// The "active" determination uses the snapshot's already-
    /// captured `scx_sched_state.sched_kva` (which scheduler instance
    /// is currently attached per `*scx_root`) + `prog_runtime_stats`
    /// (per-prog invocation counters) to identify which BPF
    /// object's progs are advancing. No new wire format is
    /// introduced.
    ///
    /// # Limitations
    ///
    /// The current implementation is honest about its narrow scope:
    /// it succeeds only when the snapshot contains exactly one BPF
    /// object's worth of global-section maps. Multi-object snapshots
    /// (two schedulers loaded back-to-back, or a single scheduler
    /// composed of multiple BPF objects) return
    /// [`SnapshotError::NoActiveScheduler`] — there is no reliable
    /// proxy for "which obj is currently attached" from a frozen
    /// `FailureDumpReport` alone. Test authors in those configurations
    /// fall back to [`Self::vars`] to enumerate every copy explicitly,
    /// or [`Self::map`] to address a specific scheduler's bss directly.
    ///
    /// Concrete cases that hit `NoActiveScheduler`:
    /// - Two scheduler instances loaded back-to-back (e.g.
    ///   pre/post-[`crate::scenario::ops::Op::ReplaceScheduler`] in a
    ///   single scenario), even when their BPF object names differ.
    /// - One scheduler composed of multiple BPF objects (e.g.
    ///   scx_layered's core + helper objects).
    /// - A placeholder snapshot (freeze-rendezvous failed to capture
    ///   any maps).
    /// - A snapshot taken with no scheduler attached (no
    ///   global-section maps at all).
    ///
    /// A principled `*scx_root → owning BPF object → maps` walker
    /// that resolves "currently attached" from kernel state directly
    /// is a follow-up; landing it would make `.active()` succeed for
    /// every case where exactly one scheduler is attached at capture
    /// time.
    ///
    /// # Lifetime
    ///
    /// Pure projection over the frozen `FailureDumpReport`;
    /// multiple calls return equivalent views. Caching the result
    /// in a `let active = snapshot.active()?;` binding is fine but
    /// not required.
    pub fn active(&self) -> SnapshotResult<Snapshot<'a>> {
        if self.report.is_placeholder {
            return Err(SnapshotError::PlaceholderSnapshot { tag: None });
        }
        // Group global-section maps by obj_name prefix. Same scan
        // both code paths read: the principled-walker path checks
        // `active_obj_name` against this set so a stale entry from a
        // pre-swap capture window (active_obj_name resolved but its
        // global-section maps no longer in the report) falls through
        // to the heuristic + diagnostic.
        let mut obj_names: Vec<&'a str> = Vec::new();
        for m in &self.report.maps {
            if !is_global_section_map(&m.name) {
                continue;
            }
            if let Some(obj) = m.name.split('.').next()
                && !obj.is_empty()
                && !obj_names.contains(&obj)
            {
                obj_names.push(obj);
            }
        }
        // Principled tiebreaker: when the freeze-coord captured a
        // non-None `active_obj_name` via the struct_ops map ↔ scx_root
        // KVA match (see [`crate::monitor::dump::FailureDumpReport::active_obj_name`]),
        // prefer that even if multiple obj prefixes show up in
        // `obj_names`. This is the resolution for the "swap left
        // both old and new BPF objects' maps in the report" case
        // the heuristic alone cannot disambiguate.
        if let Some(active_name) = self.report.active_obj_name.as_deref()
            && let Some(matched) = obj_names.iter().find(|obj| **obj == active_name).copied()
        {
            return Ok(Snapshot {
                report: self.report,
                active_obj: Some(matched),
            });
        }
        match obj_names.as_slice() {
            [] => Err(SnapshotError::NoActiveScheduler {
                reason: "snapshot has no global-section BPF maps (no scheduler \
                         attached, or capture did not include bss/data/rodata)"
                    .to_string(),
            }),
            [only] => Ok(Snapshot {
                report: self.report,
                active_obj: Some(*only),
            }),
            multiple => Err(SnapshotError::NoActiveScheduler {
                reason: format!(
                    "snapshot has {} BPF objects with global-section maps \
                     ({:?}) and the principled *scx_root walker could not \
                     identify the active obj at capture time (scx_root \
                     unresolved, no matching struct_ops map, or the matched \
                     obj has no global-section maps in this capture) — use \
                     Snapshot::vars(name) to enumerate every copy or \
                     Snapshot::map(\"<obj>.<section>\") to address a specific \
                     scheduler's bss directly",
                    multiple.len(),
                    multiple
                ),
            }),
        }
    }

    /// Convenience for `self.active()?.var(name)`. Returns the
    /// active scheduler's copy of the named global variable, or a
    /// [`SnapshotField`] carrying either
    /// [`SnapshotError::NoActiveScheduler`] (no scheduler identifiable)
    /// or the standard [`Self::var`] error variants
    /// ([`SnapshotError::VarNotFound`] / [`SnapshotError::TypeMismatch`]
    /// from the inner var lookup).
    pub fn live_var(&self, name: &str) -> SnapshotField<'a> {
        match self.active() {
            Ok(snap) => snap.var(name),
            Err(err) => SnapshotField::Missing(err),
        }
    }

    /// User-supplied disambiguation for the multi-instance case
    /// where the same scheduler binary is loaded multiple times
    /// (e.g. an Op::ReplaceScheduler swap between two builds of
    /// the same scheduler that produce two `<obj>.bss` maps with
    /// identical obj_name prefix). [`Self::active`]'s principled
    /// walker resolves the active scheduler by `*scx_root → struct_ops
    /// map → obj prefix`, but when both instances share the prefix
    /// the walker returns the prefix and the projection cannot tell
    /// the bss copies apart — [`Self::live_var`] then errors with
    /// `NoActiveScheduler { reason: "multiple obj prefixes" }`.
    ///
    /// `picker` receives every observed copy of the named variable
    /// (one entry per `<obj>.bss/.data/.rodata` map carrying it,
    /// per [`Self::vars`]) and returns the index the caller wants
    /// (typically chosen by inspecting each candidate's value via
    /// `SnapshotField::as_u64` / `as_str` and applying a liveness
    /// or activity fingerprint). The returned index is into the
    /// slice the picker received — out-of-range returns surface as
    /// a [`SnapshotError::ProjectionFailed`] so the failure message
    /// names the picker as the source.
    ///
    /// Returns [`SnapshotField::Missing`] when:
    /// - the snapshot has no copies of `name` (matches
    ///   [`Self::vars`]`(name).next().is_none()`),
    /// - `picker` returns `None` (the picker decided no candidate
    ///   matches its disambiguator), OR
    /// - `picker` returns `Some(idx)` outside the candidate range.
    pub fn live_var_via(
        &self,
        name: &str,
        picker: impl FnOnce(&[(&'a str, SnapshotField<'a>)]) -> Option<usize>,
    ) -> SnapshotField<'a> {
        let candidates: Vec<(&'a str, SnapshotField<'a>)> = self.vars(name).collect();
        if candidates.is_empty() {
            let available: Vec<String> = self
                .report
                .maps
                .iter()
                .filter(|m| is_global_section_map(&m.name))
                .map(|m| m.name.clone())
                .collect();
            return SnapshotField::Missing(crate::scenario::snapshot::SnapshotError::VarNotFound {
                requested: name.to_string(),
                available,
            });
        }
        match picker(&candidates) {
            Some(idx) if idx < candidates.len() => {
                let (_obj, field) = candidates.into_iter().nth(idx).unwrap();
                field
            }
            Some(idx) => {
                SnapshotField::Missing(crate::scenario::snapshot::SnapshotError::ProjectionFailed {
                    reason: format!(
                        "live_var_via picker returned index {idx} out of range \
                         (candidate count = {})",
                        candidates.len()
                    ),
                })
            }
            None => {
                SnapshotField::Missing(crate::scenario::snapshot::SnapshotError::ProjectionFailed {
                    reason: format!(
                        "live_var_via picker for '{name}' returned None (no candidate \
                         matched the supplied disambiguator)"
                    ),
                })
            }
        }
    }

    /// Number of maps the current view exposes — every captured
    /// map when unfiltered; only maps the [`Self::active`] filter
    /// admits when set.
    pub fn map_count(&self) -> usize {
        self.maps_iter().count()
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
    /// `crate::monitor::scx_walker`. Empty when the
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
    /// CPU enumerated by `crate::monitor::dump::CpuTimeCapture`.
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
        self.report
            .prog_runtime_stats
            .iter()
            .find(|p| p.name == name)
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

/// True when a map name's obj prefix (everything before the first
/// `.`) matches `obj`. Used by [`Snapshot::maps_iter`] when an
/// active-scheduler filter is set.
fn map_belongs_to_obj(map_name: &str, obj: &str) -> bool {
    map_name
        .split_once('.')
        .map(|(prefix, _)| prefix == obj)
        .unwrap_or(false)
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
    /// `crate::monitor::dump::FailureDumpEntry` in the captured order. For per-CPU
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
            cpu: u32::try_from(c).unwrap_or(u32::MAX),
            len: entry.per_cpu.len(),
            unmapped: false,
        });
    }
    match entry.per_cpu[c].as_ref() {
        Some(v) => SnapshotEntry::Value(v),
        None => SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
            map: map.name.clone(),
            cpu: u32::try_from(c).unwrap_or(u32::MAX),
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
            cpu: u32::try_from(c).unwrap_or(u32::MAX),
            len: entry.per_cpu.len(),
            unmapped: false,
        });
    }
    match entry.per_cpu[c].as_ref() {
        Some(v) => SnapshotEntry::Value(v),
        None => SnapshotEntry::Missing(SnapshotError::PerCpuSlot {
            map: map.name.clone(),
            cpu: u32::try_from(c).unwrap_or(u32::MAX),
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
pub(super) fn render_entry_key(entry: &SnapshotEntry<'_>) -> Option<String> {
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
