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
    ExcludedMap, HEX_KEY_PREFIX, NO_MATCH_KEY_CHAR_CAP, NO_MATCH_KEY_SAMPLE, SnapshotEntry,
    SnapshotError, SnapshotField, SnapshotResult,
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
    /// Optional kernel-map-KVA whitelist used alongside
    /// [`Self::active_obj`] to defend against the same-binary case
    /// (two scheduler instances loaded from the same binary, e.g.
    /// MITOSIS_FIXED + MITOSIS_ADAPTIVE both loading `scx_mitosis`,
    /// where the obj prefix matches both copies' bss/data/rodata
    /// maps). When set + non-empty, a map is "active" only if BOTH
    /// `active_obj` matches its prefix AND its
    /// [`FailureDumpMap::map_kva`] appears in the whitelist.
    ///
    /// `&[]` (empty) when [`Self::active`] resolved a prefix via the
    /// Phase-1 name path (no walker run → no KVA set captured) OR
    /// when the snapshot pre-dates the walker plumbing. In that
    /// case `Snapshot::active`'s filter degrades to obj-prefix
    /// matching only — still correct for the different-binary case;
    /// loses the same-binary disambiguation guarantee.
    active_map_kvas: &'a [u64],
}

impl<'a> Snapshot<'a> {
    /// Build a borrowed view over `report` with no active-scheduler
    /// filter. Every map-walking accessor sees every captured map.
    pub fn new(report: &'a FailureDumpReport) -> Self {
        Self {
            report,
            active_obj: None,
            active_map_kvas: &[],
        }
    }

    /// Iterate maps the current view exposes — every captured map
    /// when `active_obj` is None; only maps whose name shares the
    /// `<obj>.` prefix when [`Self::active`] populated the filter.
    /// When [`Self::active_map_kvas`] is also populated, additionally
    /// require the map's [`FailureDumpMap::map_kva`] to be in the
    /// whitelist — this catches the same-binary case where two
    /// scheduler instances' bss maps share an obj prefix but live at
    /// distinct kernel addresses.
    fn maps_iter(&self) -> impl Iterator<Item = &'a FailureDumpMap> + '_ {
        let active = self.active_obj;
        let kva_filter = self.active_map_kvas;
        self.report.maps.iter().filter(move |m| match active {
            None => true,
            Some(obj) => {
                if !map_belongs_to_obj(&m.name, obj) {
                    return false;
                }
                // Empty whitelist = no KVA filter (phase-1 name path
                // OR pre-walker snapshot). Non-empty = require the
                // map's KVA to appear; defends against KVA aliasing
                // and same-binary post-swap ambiguity per the
                // FailureDumpReport::active_map_kvas doc.
                if kva_filter.is_empty() {
                    return true;
                }
                m.map_kva != 0 && kva_filter.contains(&m.map_kva)
            }
        })
    }

    /// Construct [`SnapshotError::ActiveFilterExcludedMaps`] for the
    /// caller IFF the active KVA filter rejected EVERY captured
    /// `<active_obj>.*` map. Returns `None` in every other case:
    ///
    /// - the view is not active-filtered (`active_obj` is `None`),
    /// - the KVA whitelist is empty (no filter active),
    /// - no map shares the active obj prefix at all (the standard
    ///   `MapNotFound` / `VarNotFound` diagnostic carries it),
    /// - at least one captured `<active_obj>.*` map passed the KVA
    ///   whitelist (the admitted set is non-empty, so a lookup miss
    ///   is a real typo / absent symbol — fall through to the
    ///   standard diagnostic, do not falsely steer the operator at
    ///   the filter).
    ///
    /// Only the "admitted set genuinely empty" case fires the rich
    /// diagnostic. Caller is responsible for the `requested` field;
    /// every other field is populated from the snapshot.
    fn excluded_filter_err(&self, requested: String) -> Option<SnapshotError> {
        let obj = self.active_obj?;
        if self.active_map_kvas.is_empty() {
            return None;
        }
        let mut excluded: Vec<ExcludedMap> = Vec::new();
        let mut any_admitted = false;
        for m in &self.report.maps {
            if !map_belongs_to_obj(&m.name, obj) {
                continue;
            }
            if m.map_kva != 0 && self.active_map_kvas.contains(&m.map_kva) {
                any_admitted = true;
                continue;
            }
            excluded.push(ExcludedMap {
                name: m.name.clone(),
                map_kva: m.map_kva,
            });
        }
        if excluded.is_empty() || any_admitted {
            return None;
        }
        Some(SnapshotError::ActiveFilterExcludedMaps {
            requested,
            active_obj: obj.to_string(),
            excluded_maps: excluded,
            whitelist_kvas: self.active_map_kvas.to_vec(),
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
        if let Some(err) = self.excluded_filter_err(name.to_string()) {
            return Err(err);
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
    /// when no map exposes the name; OR — when more than one
    /// global-section map exposes the name — auto-falls-back to
    /// [`Self::live_var`] semantics (delegates to
    /// [`Self::active`] and re-projects) before yielding
    /// [`SnapshotError::AmbiguousVar`].
    ///
    /// # Auto-fallback contract
    ///
    /// When the raw scan finds 2+ hits AND the snapshot is not
    /// already narrowed by [`Self::active`] (i.e.
    /// `self.active_obj` is `None`), `var()` calls
    /// [`Self::active`]: on `Ok` it returns `active.var(name)`
    /// directly — whether [`SnapshotField::Value`],
    /// [`SnapshotError::VarNotFound`], or
    /// [`SnapshotError::AmbiguousVar`] persisting after the
    /// live filter narrowed; on `Err` it falls through to the
    /// pre-filter [`SnapshotError::AmbiguousVar`] (see next
    /// section). The fallback exists so post-
    /// [`crate::scenario::ops::Op::ReplaceScheduler`] callers
    /// who name a global by string don't have to know about
    /// [`Self::live_var`] explicitly — the principled
    /// active-scheduler walker is consulted automatically when
    /// the raw lookup is ambiguous. [`Self::live_var`] remains
    /// the explicit-opt-in form for callers who want the live
    /// filter unconditionally (skip the raw-scan path).
    ///
    /// # When `AmbiguousVar` STILL fires
    ///
    /// After the auto-fallback. The raw scan found 2+ hits AND
    /// `active()` failed (no scheduler attached, multi-obj
    /// without principled walker resolution, etc.). The
    /// `found_in` list names every map the raw scan saw — the
    /// operator needs all of them to reason about which obj
    /// they want to address via [`Self::map`].
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
            n if n > 1 => {
                // Ambiguous at the raw-`var` layer — try the
                // principled active-scheduler resolution before
                // giving up. When `Snapshot::active()` succeeds it
                // restricts the projection to the live scheduler's
                // maps (and, when the walker populated the KVA
                // whitelist, the live scheduler's specific map
                // instances even in the same-binary case). If
                // active() resolves to a Snapshot whose filtered
                // maps_iter yields exactly one hit, return that.
                // When the live filter ALSO can't narrow (e.g.,
                // KVA whitelist excluded every match → narrows to
                // zero, or live obj has 2+ copies of the same
                // global — unusual but possible), surface THE
                // LIVE-FILTERED diagnostic rather than the
                // pre-filter AmbiguousVar list. The operator who
                // hits ambiguity post-disambiguation needs to know the
                // filter ran and what it admitted, not see the
                // raw all-maps "ambiguous between OLD + NEW bss"
                // list that misleads them into reaching for a
                // picker the framework already obviated.
                if self.active_obj.is_none()
                    && let Ok(active) = self.active()
                {
                    return active.var(name);
                }
                SnapshotField::Missing(SnapshotError::AmbiguousVar {
                    requested: name.to_string(),
                    found_in: hits.iter().map(|(name, _)| (*name).to_string()).collect(),
                })
            }
            _ => {
                if let Some(err) = self.excluded_filter_err(name.to_string()) {
                    return SnapshotField::Missing(err);
                }
                // No global-section map yielded the var. If a
                // global-section map's contents failed to render
                // (value absent, `error` set), the search could not
                // confirm the var's absence — that map might hold it.
                // Surface the render failure rather than a false
                // VarNotFound that reads as "the symbol doesn't exist".
                if let Some(m) = self.maps_iter().find(|m| {
                    is_global_section_map(&m.name) && m.value.is_none() && m.error.is_some()
                }) {
                    return SnapshotField::Missing(SnapshotError::MapRenderIncomplete {
                        map: m.name.clone(),
                        error: m.error.clone().unwrap_or_default(),
                    });
                }
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
    /// "Active" comes from two fields the freeze coordinator
    /// populates at capture time:
    /// - [`crate::monitor::dump::FailureDumpReport::active_obj_name`]
    ///   -- set by the target-free `prog_idr` walker (no `scx_root`
    ///   dependency; works pre-6.16) that finds the live
    ///   struct_ops prog's obj prefix (see `monitor/dump/mod.rs`
    ///   `identify_active_obj_from_struct_ops`).
    /// - [`crate::monitor::dump::FailureDumpReport::active_map_kvas`]
    ///   -- the live scheduler's `prog.aux->used_maps` KVA set that
    ///   the same walker publishes. Non-empty iff the walker resolved
    ///   a global-section-bearing prog (the same-binary
    ///   disambiguation case).
    ///
    /// When the walker resolved both fields, `active()` uses them
    /// directly and the obj-prefix scan below is a sanity cross-
    /// check against the captured map set. When the walker was
    /// unavailable (placeholder dump, transient swap window before
    /// the accessor-init worker republished, or kernel built
    /// without struct_ops support), the obj-prefix scan with
    /// per-section count fallback decides.
    ///
    /// # Failure cases
    ///
    /// - [`SnapshotError::PlaceholderSnapshot`]: the snapshot is a
    ///   freeze-rendezvous-failure placeholder.
    /// - [`SnapshotError::NoActiveScheduler`] (no global-section
    ///   maps): the snapshot has no `<obj>.bss/.data/.rodata` —
    ///   either no scheduler is attached, or the capture missed
    ///   the global sections entirely.
    /// - [`SnapshotError::NoActiveScheduler`] (multiple distinct
    ///   obj prefixes, walker unavailable): two scheduler instances
    ///   with DIFFERENT obj names coexist (back-to-back load of
    ///   distinct binaries, or one scheduler composed of multiple
    ///   BPF objects) AND the walker did not publish
    ///   `active_obj_name`. Use [`Self::vars`] to enumerate every
    ///   copy or [`Self::map`] to address a specific scheduler's
    ///   bss directly.
    /// - [`SnapshotError::NoActiveScheduler`] (multi-copy
    ///   same-prefix, walker unavailable): an
    ///   [`crate::scenario::ops::Op::ReplaceScheduler`] swap
    ///   between two builds of the SAME binary left two
    ///   `<obj>.bss` (or `.data` / `.rodata`) copies with
    ///   identical names AND the walker did not publish
    ///   `active_map_kvas` to disambiguate. The obj-prefix filter
    ///   alone cannot pick the live copy without admitting both.
    ///   Use [`Self::live_var_via`] / [`Self::live_vars_via`] with
    ///   `crate::scenario::snapshot::pickers::max_by_sum_u64` to
    ///   pick by counter activity.
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
        // Scan global-section maps to collect:
        //   1. The distinct set of obj_name prefixes (used by the
        //      multi-obj failure diagnostic).
        //   2. Per-(prefix, section) counts (used to detect the
        //      same-binary multi-copy case: two `<prefix>.bss` maps
        //      coexist with identical names but distinct map KVAs).
        // The producer-side helper in
        // `monitor/dump/mod.rs` `count_global_sections_for_prefix`
        // performs the same count; both sites use strict full-name
        // equality to stay in lockstep.
        let mut obj_names: Vec<&'a str> = Vec::new();
        let mut counts: Vec<(&'a str, usize, usize, usize)> = Vec::new();
        for m in &self.report.maps {
            if !is_global_section_map(&m.name) {
                continue;
            }
            let Some(obj) = m.name.split('.').next() else {
                continue;
            };
            if obj.is_empty() {
                continue;
            }
            if !obj_names.contains(&obj) {
                obj_names.push(obj);
                counts.push((obj, 0, 0, 0));
            }
            let entry = counts
                .iter_mut()
                .find(|(o, _, _, _)| *o == obj)
                .expect("obj just pushed");
            // Strict section suffix match — `<obj>.bss` exactly,
            // not `<obj>.bss.shared` or other multi-segment names.
            let section = m.name.split('.').nth(1).unwrap_or("");
            match section {
                "bss" if m.name == format!("{obj}.bss") => entry.1 += 1,
                "data" if m.name == format!("{obj}.data") => entry.2 += 1,
                "rodata" if m.name == format!("{obj}.rodata") => entry.3 += 1,
                _ => {}
            }
        }
        // Principled fast path: when the freeze-coord captured a
        // non-None `active_obj_name` via the target-free prog_idr
        // walker (`prog_idr → prog aux->used_maps → global-section
        // sibling map`; no scx_root), prefer that even if multiple obj
        // prefixes show up in `obj_names`. The KVA whitelist
        // (`active_map_kvas`) pairs with the obj-name filter in
        // `maps_iter` — when populated, same-binary multi-copy
        // resolves to the live copy. When empty AND the matched
        // prefix has any multi-copy section, the obj-prefix filter
        // alone would admit both copies → fail loudly with a
        // multi-copy diagnostic instead of silently surfacing
        // AmbiguousVar at the var lookup.
        if let Some(active_name) = self.report.active_obj_name.as_deref()
            && let Some(matched) = obj_names.iter().find(|obj| **obj == active_name).copied()
        {
            if !self.report.active_map_kvas.is_empty() {
                return Ok(Snapshot {
                    report: self.report,
                    active_obj: Some(matched),
                    active_map_kvas: &self.report.active_map_kvas,
                });
            }
            // Walker did not publish a whitelist. Check the matched
            // prefix's section counts; if any multi-copy, bail.
            if let Some(&(_, b, d, r)) = counts.iter().find(|(o, _, _, _)| *o == matched)
                && (b > 1 || d > 1 || r > 1)
            {
                return Err(SnapshotError::NoActiveScheduler {
                    reason: format_multi_copy_reason(matched, b, d, r),
                });
            }
            return Ok(Snapshot {
                report: self.report,
                active_obj: Some(matched),
                active_map_kvas: &[],
            });
        }
        match (obj_names.as_slice(), counts.as_slice()) {
            ([], _) => Err(SnapshotError::NoActiveScheduler {
                reason: "snapshot has no global-section BPF maps (no scheduler \
                         attached, or capture did not include bss/data/rodata)"
                    .to_string(),
            }),
            ([only], [(_, b, d, r)]) if *b <= 1 && *d <= 1 && *r <= 1 => Ok(Snapshot {
                report: self.report,
                active_obj: Some(*only),
                // Only one obj prefix in the snapshot AND no
                // section has more than one copy — obj-prefix
                // matching uniquely picks the scheduler's maps.
                active_map_kvas: &[],
            }),
            ([only], [(_, b, d, r)]) => Err(SnapshotError::NoActiveScheduler {
                reason: format_multi_copy_reason(only, *b, *d, *r),
            }),
            (multiple, _) => Err(SnapshotError::NoActiveScheduler {
                reason: format!(
                    "snapshot has {} BPF objects with global-section maps \
                     ({:?}) and the principled target-free prog_idr walker \
                     could not identify the active obj at capture time (no \
                     alive struct_ops prog with a `<obj>.bss/.data/.rodata` \
                     sibling in used_maps, or an empty used_maps whitelist) — \
                     use \
                     Snapshot::vars(name) to enumerate every copy or \
                     Snapshot::map(\"<obj>.<section>\") to address a specific \
                     scheduler's bss directly",
                    multiple.len(),
                    multiple
                ),
            }),
        }
    }

    /// Read a single live counter from the active scheduler — the
    /// **default** for single-variable reads. Convenience for
    /// `self.active()?.var(name)`.
    ///
    /// **For multi-variable arithmetic on multiple counters** —
    /// fractions, ratios, deltas computed across more than one
    /// named field — use [`Self::live_vars_via`] instead.
    /// `live_vars_via` resolves the picker ONCE across a name set
    /// so independent per-name picks cannot corrupt the
    /// cross-variable computation by selecting different bss
    /// copies for different names. Repeatedly calling `live_var`
    /// for two counters from the same scheduler is correct in the
    /// walker-resolved case (both reads land in the same scheduler's
    /// bss) but loses that guarantee on the picker-fallback path
    /// — silent corruption of ratios.
    ///
    /// Returns a [`SnapshotField`] carrying either
    /// [`SnapshotError::NoActiveScheduler`] (no scheduler
    /// identifiable) or the standard [`Self::var`] error variants
    /// ([`SnapshotError::VarNotFound`] / [`SnapshotError::TypeMismatch`]
    /// from the inner var lookup).
    pub fn live_var(&self, name: &str) -> SnapshotField<'a> {
        match self.active() {
            Ok(snap) => snap.var(name),
            Err(err) => SnapshotField::Missing(err),
        }
    }

    /// Caller-supplied disambiguator for the multi-bss case where
    /// [`Self::live_var`] cannot resolve a single live copy by itself.
    ///
    /// [`Self::live_var`] delegates to [`Self::active`] to filter the
    /// snapshot to one scheduler's maps. When [`Self::active`] cannot
    /// pick a single scheduler — multiple BPF objects with
    /// global-section maps are present AND the principled
    /// `prog_idr → prog aux->used_maps → global-section map → obj prefix`
    /// walker did not
    /// identify the live one — it errors with
    /// [`SnapshotError::NoActiveScheduler`] (the exact `reason` field
    /// is the long-form message constructed at the bail site listing
    /// the observed obj_names + the walker's failure cause), and
    /// [`Self::live_var`] propagates that as [`SnapshotField::Missing`].
    ///
    /// `live_var_via` is the escape hatch: it skips the [`Self::active`]
    /// filter entirely, enumerates every observed copy of `name` via
    /// [`Self::vars`], and hands the slice to the caller-supplied
    /// `picker` to pick one by index. Common case: an
    /// `Op::ReplaceScheduler` swap between two builds of the same
    /// scheduler that leaves two `<obj>.bss` maps in the snapshot
    /// sharing one obj_name prefix.
    ///
    /// **For multi-variable arithmetic** (ratios, fractions, deltas
    /// computed across more than one named field), use
    /// [`Self::live_vars_via`] instead — it resolves the picker once
    /// across a name set so independent per-name picks cannot
    /// corrupt the cross-variable computation by selecting different
    /// bss copies for different names.
    ///
    /// `picker` receives every observed copy of the named variable
    /// (one entry per `<obj>.bss/.data/.rodata` map carrying it,
    /// per [`Self::vars`]) and returns the index the caller wants
    /// (typically chosen by inspecting each candidate's value via
    /// `SnapshotField::as_u64` / `as_str` and applying a liveness
    /// or activity fingerprint — see
    /// [`crate::scenario::snapshot::pickers`] for predefined
    /// pickers such as `max_by_counter_value`).
    ///
    /// Returns [`SnapshotField::Missing`] when:
    /// - the snapshot's underlying `FailureDumpReport` is a
    ///   placeholder (carrying
    ///   [`SnapshotError::PlaceholderSnapshot`] — matches the
    ///   sibling [`Self::var`] / [`Self::map`] placeholder-first
    ///   contract so callers pattern-matching on the error variant
    ///   distinguish "freeze rendezvous failed" from "name absent
    ///   from a real capture"),
    /// - the snapshot has no copies of `name` (carrying
    ///   [`SnapshotError::VarNotFound`] with the list of available
    ///   global-section maps),
    /// - `picker` returns `None` (carrying
    ///   [`SnapshotError::ProjectionFailed`] naming the picker as
    ///   the source), OR
    /// - `picker` returns `Some(idx)` outside the candidate range
    ///   (carrying [`SnapshotError::ProjectionFailed`] with the bad
    ///   index and the candidate count).
    pub fn live_var_via(
        &self,
        name: &str,
        picker: impl FnOnce(&[(&'a str, SnapshotField<'a>)]) -> Option<usize>,
    ) -> SnapshotField<'a> {
        if self.report.is_placeholder {
            return SnapshotField::Missing(
                crate::scenario::snapshot::SnapshotError::PlaceholderSnapshot { tag: None },
            );
        }
        let candidates: Vec<(&'a str, SnapshotField<'a>)> = self.vars(name).collect();
        if candidates.is_empty() {
            if let Some(err) = self.excluded_filter_err(name.to_string()) {
                return SnapshotField::Missing(err);
            }
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

    /// Caller-supplied disambiguator for the multi-bss case where
    /// **multiple variables from the same scheduler instance** must
    /// be read consistently — e.g. computing
    /// `nr_mig_cross_dispatch / (nr_mig_same_dispatch + nr_mig_cross_dispatch)`
    /// as a cross-LLC dispatch fraction from one scheduler's BPF
    /// counters.
    ///
    /// # Why a separate primitive
    ///
    /// Calling [`Self::live_var_via`] N times independently risks
    /// picking a DIFFERENT bss copy per call: the picker resolves
    /// each name's candidate set independently, so two consecutive
    /// `live_var_via("a", picker)` + `live_var_via("b", picker)`
    /// calls can land on bss copy A for `a` and bss copy B for `b`,
    /// corrupting any cross-variable arithmetic (ratio, fraction,
    /// delta). `live_vars_via` resolves the picker ONCE across the
    /// candidate set for all N names jointly so every returned
    /// [`SnapshotField`] reads from the same source map.
    ///
    /// # Mechanism
    ///
    /// Per global-section map, look up each name in input order;
    /// keep the map as a candidate row iff it has ALL the names
    /// (intersection semantics — partial-coverage maps are absent
    /// from the picker's input). The picker receives
    /// `&[(map_name, fields_in_input_order)]` and returns the
    /// chosen row's index. The returned `Vec<SnapshotField>` is
    /// positional, keyed by the input `names` order — `result[0]`
    /// is `names[0]`'s field from the picked map, `result[1]` is
    /// `names[1]`'s field, etc.
    ///
    /// **Single-section constraint.** All `names` must reside in
    /// the SAME global-section map — typically the scheduler's
    /// `<obj>.bss`. A `bss` counter co-picked with a `data`
    /// constant from the same scheduler obj lands in DIFFERENT
    /// candidate rows (the obj's `.bss` map carries the first
    /// name, its `.data` map carries the second, neither row has
    /// both), the intersection collapses to empty, and the helper
    /// returns [`SnapshotError::VarNotFound`]. If the test reads
    /// from multiple sections, issue separate `live_vars_via`
    /// calls (one per section's name group) and compose the
    /// per-call results caller-side.
    ///
    /// # See also
    ///
    /// - [`Self::live_var_via`] for single-variable disambiguation.
    /// - [`crate::scenario::snapshot::pickers::max_by_sum_u64`] for
    ///   the "max-activity bss" heuristic over co-picked u64
    ///   counters.
    ///
    /// # Errors
    ///
    /// - [`SnapshotError::PlaceholderSnapshot`] — the underlying
    ///   `FailureDumpReport` is a placeholder; matches the sibling
    ///   [`Self::live_var_via`] / [`Self::var`] / [`Self::map`]
    ///   placeholder-first contract.
    /// - [`SnapshotError::ProjectionFailed`] — `names` is empty
    ///   (caller bug: nothing to co-pick), `picker` returns `None`
    ///   (no candidate matched), or `picker` returns an
    ///   out-of-range index.
    /// - [`SnapshotError::VarNotFound`] — no global-section map
    ///   has ALL the requested names. `requested` carries the
    ///   joined name list, `available` carries the global-section
    ///   map names that were scanned.
    pub fn live_vars_via<P>(
        &self,
        names: &[&str],
        picker: P,
    ) -> crate::scenario::snapshot::SnapshotResult<Vec<SnapshotField<'a>>>
    where
        P: FnOnce(&[(&'a str, Vec<SnapshotField<'a>>)]) -> Option<usize>,
    {
        if self.report.is_placeholder {
            return Err(
                crate::scenario::snapshot::SnapshotError::PlaceholderSnapshot { tag: None },
            );
        }
        if names.is_empty() {
            return Err(crate::scenario::snapshot::SnapshotError::ProjectionFailed {
                reason: "live_vars_via called with an empty names slice — \
                         co-pick requires at least one name"
                    .to_string(),
            });
        }
        // Group by MAP: each global-section map becomes a candidate
        // row IFF it has ALL the requested names. Partial-coverage
        // maps are dropped from the picker's input — they cannot
        // answer the co-pick.
        let mut candidates: Vec<(&'a str, Vec<SnapshotField<'a>>)> = Vec::new();
        for m in self.maps_iter() {
            if !is_global_section_map(&m.name) {
                continue;
            }
            let Some(value) = m.value.as_ref() else {
                continue;
            };
            let mut row: Vec<SnapshotField<'a>> = Vec::with_capacity(names.len());
            let mut all_present = true;
            for name in names {
                if let Some(found) = lookup_member(value, name) {
                    row.push(SnapshotField::Value(found));
                } else {
                    all_present = false;
                    break;
                }
            }
            if all_present {
                candidates.push((m.name.as_str(), row));
            }
        }
        if candidates.is_empty() {
            let requested = format!("[{}]", names.join(", "));
            if let Some(err) = self.excluded_filter_err(requested.clone()) {
                return Err(err);
            }
            let available: Vec<String> = self
                .report
                .maps
                .iter()
                .filter(|m| is_global_section_map(&m.name))
                .map(|m| m.name.clone())
                .collect();
            return Err(crate::scenario::snapshot::SnapshotError::VarNotFound {
                requested,
                available,
            });
        }
        match picker(&candidates) {
            Some(idx) if idx < candidates.len() => {
                let (_obj, fields) = candidates.into_iter().nth(idx).unwrap();
                Ok(fields)
            }
            Some(idx) => Err(crate::scenario::snapshot::SnapshotError::ProjectionFailed {
                reason: format!(
                    "live_vars_via picker returned index {idx} out of range \
                     (candidate count = {})",
                    candidates.len()
                ),
            }),
            None => Err(crate::scenario::snapshot::SnapshotError::ProjectionFailed {
                reason: format!(
                    "live_vars_via picker for [{}] returned None (no candidate \
                     matched the supplied disambiguator)",
                    names.join(", ")
                ),
            }),
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

    /// Per-cgroup PSI-irq rows for the test's workload cgroups, host-walked
    /// from the cgroup hierarchy at this freeze (Phase A). One row per
    /// workload-root leaf cgroup with per-cgroup PSI accounting enabled. Empty
    /// when the capture was not wired, the workload root isn't present yet, or
    /// `psi_cgroups_enabled` is off — loud-absent. RAW values; decoded + folded
    /// at the metric layer (see
    /// `crate::monitor::cgroup_walk::CgroupPsiStat`).
    pub fn cgroup_psi(&self) -> &'a [crate::monitor::cgroup_walk::CgroupPsiStat] {
        &self.report.cgroup_psi
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
    /// selected scheduler-exit handler fired during the run.
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

/// Render the multi-copy-same-prefix diagnostic for
/// [`Snapshot::active`]. `(bss, data, rodata)` are full-name
/// equality counts; any value > 1 means the prefix has multiple
/// copies of that section type in the captured `maps[]` (typical
/// cause: `Op::ReplaceScheduler` swap between two builds of the
/// same binary leaves the dying instance's globals adjacent to
/// the new instance's). The message names which section(s) are
/// multi-copy and steers the operator at the picker-based
/// disambiguators.
fn format_multi_copy_reason(prefix: &str, bss: usize, data: usize, rodata: usize) -> String {
    let mut parts: Vec<String> = Vec::new();
    if bss > 1 {
        parts.push(format!("{prefix}.bss × {bss}"));
    }
    if data > 1 {
        parts.push(format!("{prefix}.data × {data}"));
    }
    if rodata > 1 {
        parts.push(format!("{prefix}.rodata × {rodata}"));
    }
    let detail = parts.join(", ");
    format!(
        "snapshot has multiple same-name copies of {prefix}'s global-section maps \
         ({detail}) and the principled target-free prog_idr walker did not \
         publish an active_map_kvas whitelist to disambiguate (transient swap \
         window where \
         the accessor-init worker has not yet republished, or the walker is \
         unavailable on this kernel build) — use \
         `series.live_bpf_vars_via([\"name\"], pickers::max_by_sum_u64)` for \
         multi-variable counter co-pick, or \
         `Snapshot::live_var_via(name, pickers::max_by_counter_value)` for a \
         single-counter pick, to pick by counter activity"
    )
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

    /// When this map's contents failed to render at capture time
    /// (`FailureDumpMap::error` is set), produce the
    /// [`SnapshotError::MapRenderIncomplete`] that callers should
    /// surface in place of an "empty map" verdict. `None` when the
    /// map rendered successfully — the absence is then genuine and
    /// the caller's own "not found" / "out of range" error applies.
    fn render_incomplete_err(&self) -> Option<SnapshotError> {
        self.map
            .error
            .as_ref()
            .map(|error| SnapshotError::MapRenderIncomplete {
                map: self.map.name.clone(),
                error: error.clone(),
            })
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
        // An empty traversal over a map whose contents failed to
        // render is a capture gap, not "predicate never matched".
        if len == 0
            && let Some(err) = self.render_incomplete_err()
        {
            return SnapshotEntry::Missing(err);
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
            None => {
                // No entries to compare. A render failure (contents
                // unreadable at capture) is distinct from an
                // empty map; surface it so the gap is visible.
                if let Some(err) = self.render_incomplete_err() {
                    return SnapshotEntry::Missing(err);
                }
                SnapshotEntry::Missing(SnapshotError::NoMatch {
                    map: self.map.name.clone(),
                    op: "max_by".to_string(),
                    len: 0,
                    available_keys: Vec::new(),
                })
            }
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
        // Nothing to walk. Distinguish a render failure (contents
        // could not be read at capture, `error` set) from a
        // genuinely-empty map so the operator sees the capture gap
        // rather than a misleading "index out of range, len 0".
        if let Some(err) = self.render_incomplete_err() {
            return Err(err);
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
