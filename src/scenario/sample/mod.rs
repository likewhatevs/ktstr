//! Unified periodic-sample bundle and series projection.
//!
//! At every periodic boundary (see [`super::snapshot`] and the
//! freeze coordinator's periodic-capture loop), the framework
//! captures a coupled [`FailureDumpReport`] + scx_stats JSON pair.
//! [`Sample`] is the borrowed-view tuple over that pair plus the
//! per-sample tag and elapsed-millisecond timestamp;
//! [`SampleSeries`] is the ordered sequence of samples drained
//! from a `SnapshotBridge` after VM exit.
//!
//! Test authors do not construct samples manually — they call
//! [`SampleSeries::from_drained`] on the periodic bundle the
//! bridge surfaces via
//! `SnapshotBridge::drain_ordered_with_stats`, then project the
//! series along one of four orthogonal axes:
//!
//!  - **bpf** — kernel BPF state through
//!    [`SampleSeries::bpf`] / the typed
//!    [`SampleSeries::bpf_map`] helper.
//!  - **stats** — userspace scx_stats JSON through
//!    [`SampleSeries::stats`] / the typed
//!    [`SampleSeries::stats_path`] helper.
//!  - **host** — per-sample per-CPU host timeline through
//!    [`SampleSeries::host`] (sourced from
//!    `FailureDumpReport::per_cpu_time`).
//!  - **monitor** — per-VM-run cross-CPU host monitor aggregate
//!    through [`SampleSeries::monitor`] (sourced from
//!    `MonitorReport::summary`).
//!
//! Each projection yields a
//! [`crate::assert::temporal::SeriesField`] that
//! flows into the temporal-assertion patterns
//! (`nondecreasing`, `rate_within`, `steady_within`,
//! `converges_to`, `always_true`, `ratio_within`) defined in
//! [`crate::assert::temporal`].
//!
//! # Lifetime model
//!
//! `SampleSeries` owns the drained `Vec<(tag, report, stats,
//! elapsed_ms)>` so projection closures can borrow into the
//! reports / stats without copying. Constructing a `Sample` only
//! borrows; [`SampleSeries::iter_samples`] yields `Sample<'_>`
//! bound by the series' own lifetime.

use crate::monitor::MonitorReport;
use crate::monitor::dump::FailureDumpReport;

use super::snapshot::{Snapshot, SnapshotResult};
use crate::assert::temporal::SeriesField;

mod bpf;
mod host;
mod monitor;
mod stats;

pub use bpf::{BpfMapCpuProjector, BpfMapProjector};
pub use host::HostView;
pub use monitor::{ERROR_CLASS_NAMES, MonitorView, ScxEventsView};
pub use stats::{StatsPathProjector, StatsValue};

/// One captured periodic sample: a frozen BPF snapshot paired with
/// the scx_stats JSON observed just before the freeze rendezvous,
/// labelled with the periodic tag (`periodic_000` …
/// `periodic_NNN`) and tagged with the elapsed milliseconds since
/// `run_start`.
///
/// Constructed by [`SampleSeries::iter_samples`] — test authors do
/// not invoke `Sample::new` directly. The `'a` lifetime ties the
/// borrowed `tag`, `snapshot`, and `stats` references back to the
/// owning [`SampleSeries`].
#[derive(Debug)]
#[non_exhaustive]
pub struct Sample<'a> {
    /// Periodic tag the freeze coordinator stamped onto this
    /// sample. Always begins with `"periodic_"` followed by a
    /// zero-padded ordinal — see
    /// `crate::vmm::freeze_coord::periodic_tag`.
    pub tag: &'a str,
    /// Wall-clock elapsed milliseconds (pause-adjusted: the
    /// coordinator subtracts cumulative ScenarioPause/Resume
    /// pause time and any in-flight pause window) since the
    /// coordinator's `run_start` instant at stats-request
    /// completion time, pre-freeze. The coordinator captures
    /// this timestamp AFTER the scx_stats request returns
    /// (or fails) and BEFORE entering the freeze rendezvous,
    /// so the value reflects when the running scheduler's
    /// stats were observed. BPF state is observed up to
    /// `FREEZE_RENDEZVOUS_TIMEOUT` later than this anchor.
    /// `None` when the bridge could not record a timestamp
    /// (legacy stores without elapsed metadata, or
    /// non-periodic captures surfaced through the same drain) —
    /// distinct from a measured `Some(0)`.
    pub elapsed_ms: Option<u64>,
    /// Frozen BPF state captured at this boundary. The view is
    /// cheap to build — accessor methods walk the underlying
    /// [`FailureDumpReport`] in place.
    pub snapshot: Snapshot<'a>,
    /// scx_stats JSON observed by a stats request issued just
    /// BEFORE the freeze rendezvous. `Err(reason)` when the stats
    /// client was not wired (`scheduler_binary` is absent) or the
    /// request failed — the carried
    /// [`MissingStatsReason`](crate::scenario::snapshot::MissingStatsReason)
    /// identifies the specific failure mode (no scheduler, relay
    /// rejected, watchdog cancelled, scheduler errno, etc.).
    /// [`SampleSeries::stats`] surfaces this `Err` as a per-sample
    /// [`SnapshotError::MissingStats`](crate::scenario::snapshot::SnapshotError::MissingStats)
    /// slot in the resulting [`SeriesField`] rather than vacuously
    /// skipping; temporal patterns handle that error per their own
    /// policy (gap-tolerant patterns like `nondecreasing`,
    /// `rate_within`, `steady_within`, `converges_to`, and
    /// `ratio_within` skip the sample with a rendered Note, while
    /// strict patterns like `always_true` and `each` fail the
    /// assertion so a stats-coverage gap can never silently slip
    /// past the call site).
    pub stats: Result<&'a serde_json::Value, &'a crate::scenario::snapshot::MissingStatsReason>,
    /// Scenario phase index the freeze coordinator stamped onto
    /// this sample at capture time. Encoded per the framework's
    /// 1-indexed phase convention — `0` is the BASELINE settle
    /// window, `1..=N` align with scenario Step ordinals. `None`
    /// for fixture-injected samples that took the unstamped legacy
    /// bridge paths
    /// ([`super::snapshot::SnapshotBridge::capture`] /
    /// [`super::snapshot::SnapshotBridge::store`] /
    /// [`super::snapshot::SnapshotBridge::store_with_stats`]);
    /// production captures via the periodic-fire path and the
    /// on-demand `Op::CaptureSnapshot` / `Op::WatchSnapshot` apply
    /// arms always carry `Some(idx)`. Read by
    /// [`SampleSeries::by_stamped_phase`] (and as the offset-less
    /// fallback in [`SampleSeries::by_stimulus_phase`]) to bucket
    /// samples per scenario phase for the phase-aware aggregator.
    pub step_index: Option<u16>,
    /// Workload-relative boundary offset (ms) this periodic capture
    /// was scheduled for (`boundary_ns - scenario_anchor_ns`), or
    /// `None` for non-periodic / on-demand captures. Distinct from
    /// `elapsed_ms` (run_start-relative fire time, ~uniform across a
    /// deferred-fire burst). Read by
    /// [`crate::assert::build_phase_buckets`] /
    /// [`crate::assert::build_phase_buckets_with_stimulus`] to
    /// attribute the capture to the guest step whose stimulus window
    /// contains this offset, and as the workload-relative bucket
    /// start/end. `None` falls back to `elapsed_ms` + the stored
    /// `step_index` (today's behavior for on-demand captures).
    pub boundary_offset_ms: Option<u64>,
}

/// Ordered collection of [`Sample`]s drained from a
/// [`SnapshotBridge`](super::snapshot::SnapshotBridge) after a VM
/// run completes. Owns the underlying tuples so projection
/// closures can borrow into the reports / stats without copying.
///
/// Test authors construct a `SampleSeries` from
/// [`super::snapshot::SnapshotBridge::drain_ordered_with_stats`]
/// via [`Self::from_drained`]; non-periodic tags (e.g. `Op::CaptureSnapshot`
/// captures) coexist in the drain output and are tolerated by the
/// projection helpers — the typical pattern is to pre-filter to
/// periodic tags via [`Self::periodic_only`] before asserting.
#[derive(Debug, Clone)]
pub struct SampleSeries {
    rows: Vec<SampleRow>,
    /// Host-side monitor report for the VM run that produced this
    /// series. `None` when the monitor did not run (host-only tests,
    /// early VM failure, or `from_drained` was called with `None`
    /// for the monitor argument). Aggregates inside the report refer
    /// to THAT series' monitoring window only — no cross-series
    /// merge is supported. Surfaced via [`Self::monitor`] which wraps
    /// it in a borrowed [`MonitorView`] for typed projection.
    monitor: Option<MonitorReport>,
}

/// Owned tuple stored inside [`SampleSeries`]. Mirrors the shape of
/// [`super::snapshot::SnapshotBridge::drain_ordered_with_stats`]
/// but carries the timestamp as `Option<u64>` — `None` preserves the
/// bridge's "no timestamp recorded" signal so a not-measured sample
/// stays distinct from a measured `Some(0)`.
#[derive(Debug, Clone)]
struct SampleRow {
    tag: String,
    report: FailureDumpReport,
    stats: Result<serde_json::Value, crate::scenario::snapshot::MissingStatsReason>,
    elapsed_ms: Option<u64>,
    /// Workload-relative boundary offset (ms) for periodic captures;
    /// `None` for non-periodic / on-demand. Mirrored from
    /// [`super::snapshot::DrainedSnapshotEntry::boundary_offset_ms`].
    boundary_offset_ms: Option<u64>,
    /// Scenario phase index stamped at capture time by the
    /// step-aware bridge entry points, mirrored from
    /// [`super::snapshot::DrainedSnapshotEntry::step_index`].
    /// `None` for unstamped legacy / fixture captures (see
    /// [`Sample::step_index`] for the surfaced semantic).
    step_index: Option<u16>,
}

/// Common scaffolding shared by every projector axis (bpf / stats /
/// host per-CPU). Iterates `rows` once, threads each row's
/// `tag` and `elapsed_ms` into the resulting [`SeriesField`], and
/// invokes `row_to_slot` to compute the per-sample value or per-
/// sample `SnapshotError`. Keeps the `tags`/`elapsed`/`values`
/// vec lengths in lock-step so the [`SeriesField::from_parts`]
/// length-parity invariant never triggers.
fn build_series_field<T>(
    rows: &[SampleRow],
    label: impl Into<String>,
    mut row_to_slot: impl FnMut(&SampleRow) -> SnapshotResult<T>,
) -> SeriesField<T> {
    let mut values: Vec<SnapshotResult<T>> = Vec::with_capacity(rows.len());
    let mut tags: Vec<String> = Vec::with_capacity(rows.len());
    let mut elapsed: Vec<Option<u64>> = Vec::with_capacity(rows.len());
    let mut phases: Vec<Option<crate::assert::Phase>> = Vec::with_capacity(rows.len());
    for row in rows {
        tags.push(row.tag.clone());
        elapsed.push(row.elapsed_ms);
        // The drained-bridge step_index is already in the 1-indexed
        // encoding `crate::assert::Phase` wraps (BASELINE = 0, Step[k]
        // = k + 1). Thread it through so `SeriesField::phase` /
        // `value_at_phase` / `last_per_phase` / `ratio_across_phases`
        // see live phase stamps. Synthetic rows (from `from_drained`
        // test path) carry `step_index = None` and stay None here.
        phases.push(row.step_index.map(crate::assert::Phase::from));
        values.push(row_to_slot(row));
    }
    SeriesField::from_parts_with_phases_opt(label, tags, elapsed, values, phases)
}

impl SampleSeries {
    /// Build a series from the bridge's drained tuple. Every entry
    /// is preserved in the order the bridge surfaced, including
    /// non-periodic tags — callers that want the periodic-only
    /// view chain `.periodic_only()`.
    ///
    /// `monitor` is the per-VM-run `MonitorReport` (typically
    /// `result.monitor.clone()` from a `VmResult`). Pass `None`
    /// when the monitor did not run (host-only tests, early VM
    /// failure). Surfaced via [`Self::monitor`] for typed projection
    /// of the summary + scx_events + (future) per-sample timelines.
    pub fn from_drained(
        drained: Vec<(
            String,
            FailureDumpReport,
            Option<serde_json::Value>,
            Option<u64>,
        )>,
        monitor: Option<MonitorReport>,
    ) -> Self {
        let rows = drained
            .into_iter()
            .map(|(tag, report, stats, elapsed_ms)| SampleRow {
                tag,
                report,
                // Test/synthetic caller convention: `None` collapses to
                // the `NoSchedulerBinary` reason because that's the
                // shape every fixture has historically modelled — no
                // scheduler client wired, no stats. Production callers
                // that have a typed [`SchedStatsError`] use
                // [`Self::from_drained_typed`] instead, which preserves
                // the specific failure mode.
                stats: stats.map(Ok).unwrap_or(Err(
                    crate::scenario::snapshot::MissingStatsReason::NoSchedulerBinary,
                )),
                elapsed_ms,
                // Fixture/tuple path carries no scheduled boundary offset.
                boundary_offset_ms: None,
                // Unstamped fixture path: samples surface with
                // `step_index = None` and fall under the
                // by_stamped_phase fallback bucket. Production callers
                // thread the bridge-stamped index via from_drained_typed.
                step_index: None,
            })
            .collect();
        Self { rows, monitor }
    }

    /// Production-path constructor: takes the typed
    /// [`Result<serde_json::Value, MissingStatsReason>`](crate::scenario::snapshot::MissingStatsReason)
    /// shape returned by
    /// [`SnapshotBridge::drain_ordered_with_stats`](crate::scenario::snapshot::SnapshotBridge::drain_ordered_with_stats),
    /// preserving the specific failure mode (relay error, scheduler
    /// errno, watchdog cancellation, etc.). Use this when the caller
    /// has access to the bridge drain output; tests prefer
    /// [`Self::from_drained`] which accepts the simpler `Option`
    /// shape and collapses absent → `NoSchedulerBinary`.
    pub fn from_drained_typed(
        drained: Vec<crate::scenario::snapshot::DrainedSnapshotEntry>,
        monitor: Option<MonitorReport>,
    ) -> Self {
        let rows = drained
            .into_iter()
            .map(|entry| {
                let crate::scenario::snapshot::DrainedSnapshotEntry {
                    tag,
                    report,
                    stats,
                    elapsed_ms,
                    boundary_offset_ms,
                    step_index,
                    ..
                } = entry;
                SampleRow {
                    tag,
                    report,
                    stats,
                    elapsed_ms,
                    boundary_offset_ms,
                    step_index,
                }
            })
            .collect();
        Self { rows, monitor }
    }

    /// Empty series. Useful for tests and for the no-periodic-
    /// capture case where every assertion vacuously passes.
    pub fn empty() -> Self {
        Self {
            rows: Vec::new(),
            monitor: None,
        }
    }

    /// True when no samples are present.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Number of samples in the series.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Filter the series to entries whose tag begins with
    /// `"periodic_"`. Periodic captures are the only entries the
    /// temporal-assertion patterns are designed for; on-demand
    /// `Op::CaptureSnapshot` and watchpoint-fire captures share the
    /// bridge's tag namespace and would otherwise mix into the
    /// timeline as off-cadence outliers. Consumes `self` because
    /// the filter rebuilds the owning row vec — when a borrowed
    /// view is needed instead, see [`Self::periodic_ref`] which
    /// iterates the same rows without taking ownership.
    #[must_use = "periodic_only returns a filtered series; bind the result"]
    pub fn periodic_only(self) -> Self {
        Self {
            rows: self
                .rows
                .into_iter()
                .filter(|r| r.tag.starts_with("periodic_"))
                .collect(),
            monitor: self.monitor,
        }
    }

    /// Borrowed equivalent of [`Self::periodic_only`]: yields a
    /// borrowed-view iterator over [`Sample`]s whose tag starts
    /// with `"periodic_"`, without consuming the series. Use when
    /// a single test asserts on both periodic-only and
    /// all-captures views from the same series.
    pub fn periodic_ref(&self) -> impl Iterator<Item = Sample<'_>> {
        self.iter_samples()
            .filter(|s| s.tag.starts_with("periodic_"))
    }

    /// Iterate over [`Sample`] views borrowing into this series.
    /// Each yielded `Sample<'_>` carries the tag, elapsed-ms,
    /// borrowed [`Snapshot`], borrowed `Option<&Value>` stats,
    /// and the per-sample phase step index.
    pub fn iter_samples(&self) -> impl Iterator<Item = Sample<'_>> {
        self.rows.iter().map(|r| Sample {
            tag: r.tag.as_str(),
            elapsed_ms: r.elapsed_ms,
            snapshot: Snapshot::new(&r.report),
            stats: r.stats.as_ref(),
            step_index: r.step_index,
            boundary_offset_ms: r.boundary_offset_ms,
        })
    }

    /// Group samples by the RAW bridge-stamped scenario phase. The
    /// returned map is keyed by `step_index` (1-indexed phase encoding
    /// — `0` is BASELINE, `1..=N` align with scenario Step ordinals);
    /// each entry is the ordered run of samples that fell in that
    /// phase, preserving the iteration order produced by
    /// [`Self::iter_samples`].
    ///
    /// Samples that lack a stamped step index (the unstamped
    /// fixture path via
    /// [`super::snapshot::SnapshotBridge::capture`] /
    /// [`super::snapshot::SnapshotBridge::store`] /
    /// [`super::snapshot::SnapshotBridge::store_with_stats`]) fall
    /// under key `0` per the "no stamped index" fallback — the same
    /// bucket BASELINE samples land in. The fixture / BASELINE
    /// collision is acceptable because both flavours represent
    /// pre-first-Step (or unstamped) state from the bucketer's
    /// perspective; production callers that need to distinguish
    /// can inspect `Sample::step_index` directly.
    ///
    /// CAVEAT — prefer [`Self::by_stimulus_phase`] when a stimulus
    /// timeline is available: the bridge stamp is the step active at
    /// (deferred) FIRE time, so under the dump-prerequisite gate a
    /// burst of captures can all stamp the same late `CURRENT_STEP`
    /// and collapse every sample into one phase. `by_stimulus_phase`
    /// re-derives the phase from each sample's timing-independent
    /// `boundary_offset_ms`, which is immune to the burst.
    ///
    /// The phase-aware aggregator consumes this map to compute
    /// per-phase metric reductions (Counter `last - first` delta,
    /// Gauge / Peak / Timestamp via `crate::stats::aggregate_samples`).
    pub fn by_stamped_phase(&self) -> std::collections::BTreeMap<u16, Vec<Sample<'_>>> {
        let mut by_phase: std::collections::BTreeMap<u16, Vec<Sample<'_>>> =
            std::collections::BTreeMap::new();
        for sample in self.iter_samples() {
            let key = sample.step_index.unwrap_or(0);
            by_phase.entry(key).or_default().push(sample);
        }
        by_phase
    }

    /// Group samples by the guest step whose stimulus window contains
    /// each sample's workload-relative `boundary_offset_ms`, rather
    /// than the raw bridge-stamped `step_index`
    /// ([`Self::by_stamped_phase`]). The returned map uses the same
    /// 1-indexed phase key (`0` = BASELINE, `1..=N` = Step ordinals)
    /// and preserves [`Self::iter_samples`] order within each bucket.
    ///
    /// This is the correct grouping whenever a stimulus timeline is
    /// available: `boundary_offset_ms` is derived from the scheduled
    /// boundary, NOT the (deferred) fire time, so it survives the
    /// dump-prerequisite-gate burst that makes every periodic capture
    /// stamp the same late `CURRENT_STEP` (the `phases.len() == 1`
    /// collapse `by_stamped_phase` is subject to). Samples with no
    /// `boundary_offset_ms` (on-demand / fixture captures) fall back
    /// to their stamped `step_index`.
    ///
    /// Unlike the folded scalar [`crate::assert::PhaseBucket`]s that
    /// [`crate::assert::build_phase_buckets_with_stimulus`] returns,
    /// this keeps the per-sample [`Sample`] views (full Snapshot / dsq
    /// access) per phase. `build_phase_buckets_with_stimulus` itself
    /// is built on this method.
    pub fn by_stimulus_phase(
        &self,
        stimulus_events: &[crate::timeline::StimulusEvent],
    ) -> std::collections::BTreeMap<u16, Vec<Sample<'_>>> {
        // Step-start timeline in scenario-relative (guest monotonic) ms
        // — the same frame as `boundary_offset_ms`. See
        // [`step_starts_from_stimulus`] for the step-START selection.
        let step_starts = step_starts_from_stimulus(stimulus_events);
        let mut by_phase: std::collections::BTreeMap<u16, Vec<Sample<'_>>> =
            std::collections::BTreeMap::new();
        for sample in self.iter_samples() {
            let key = match sample.boundary_offset_ms {
                Some(offset) => remap_offset_to_step(offset, &step_starts),
                None => sample.step_index.unwrap_or(0),
            };
            by_phase.entry(key).or_default().push(sample);
        }
        by_phase
    }
}

/// Step-start timeline in scenario-relative (guest monotonic) ms — the
/// frame shared with `boundary_offset_ms`. Only step-START stimulus
/// events anchor a step window: the terminal scenario-end event
/// (`step_index` `None`) is dropped by the `filter_map`, and per-step
/// StepEnd events (`is_step_end`, which carry their step's `step_index`)
/// are excluded so a step's window is anchored by its start, not its
/// end-of-hold marker. Returned sorted ascending by `elapsed_ms`.
///
/// Shared by [`SampleSeries::by_stimulus_phase`] (which remaps each
/// capture to the step active at its offset) and
/// [`crate::assert::build_phase_buckets_with_stimulus`] (which enumerates
/// the steps that must have a bucket even when they captured no samples)
/// so the two step-START selections cannot drift.
pub(crate) fn step_starts_from_stimulus(
    stimulus_events: &[crate::timeline::StimulusEvent],
) -> Vec<(u64, u16)> {
    let mut step_starts: Vec<(u64, u16)> = stimulus_events
        .iter()
        .filter(|e| !e.is_step_end)
        .filter_map(|e| e.step_index.map(|k| (e.elapsed_ms, k)))
        .collect();
    step_starts.sort_by_key(|(ms, _)| *ms);
    step_starts
}

/// Map a capture's workload-relative boundary offset (ms since scenario
/// start) to the guest step active at that instant: the `step_index` of
/// the latest stimulus step-start at or before the offset, or `0`
/// (BASELINE) when the offset precedes the first step-start.
/// `step_starts` must be sorted ascending by elapsed_ms.
///
/// This is the timing-independent attribution at the heart of the
/// deferred-fire fix: the scheduled boundary offset is computed from the
/// boundary schedule (not the fire time), so it survives a burst of
/// captures that all fire — and would all stamp the same late
/// CURRENT_STEP — after the dump-prerequisite gate clears.
fn remap_offset_to_step(offset_ms: u64, step_starts: &[(u64, u16)]) -> u16 {
    let mut step = 0u16;
    for (start_ms, k) in step_starts {
        if *start_ms <= offset_ms {
            step = *k;
        } else {
            break;
        }
    }
    step
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::btf_render::{RenderedMember, RenderedValue};
    use crate::monitor::dump::{FailureDumpMap, FailureDumpReport, SCHEMA_SINGLE};

    fn synthetic_report(value: u64) -> FailureDumpReport {
        let bss_value = RenderedValue::Struct {
            type_name: Some(".bss".into()),
            members: vec![
                RenderedMember {
                    name: "nr_dispatched".into(),
                    value: RenderedValue::Uint { bits: 64, value },
                },
                RenderedMember {
                    name: "stall".into(),
                    value: RenderedValue::Uint { bits: 8, value: 0 },
                },
            ],
        };
        let bss_map = FailureDumpMap {
            name: "scx_obj.bss".into(),
            map_kva: 0,
            map_type: 2,
            value_size: 16,
            max_entries: 1,
            value: Some(bss_value),
            entries: Vec::new(),
            array_entries: Vec::new(),
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        };
        FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            active_map_kvas: Vec::new(),
            maps: vec![bss_map],
            ..Default::default()
        }
    }

    fn synthetic_stats(busy: f64) -> serde_json::Value {
        serde_json::json!({
            "busy": busy,
            "antistall": 0,
            "layers": {
                "batch": { "util": busy * 0.5 }
            }
        })
    }

    #[test]
    fn from_drained_preserves_order() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                synthetic_report(10),
                Some(synthetic_stats(50.0)),
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(20),
                Some(synthetic_stats(60.0)),
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None);
        assert_eq!(series.len(), 2);
        let tags: Vec<&str> = series.iter_samples().map(|s| s.tag).collect();
        assert_eq!(tags, vec!["periodic_000", "periodic_001"]);
    }

    #[test]
    fn bpf_member_names_union_not_blinded_by_placeholder_first_sample() {
        // Sample 0 is a placeholder (no maps); sample 1 carries the bss
        // struct. member_names must discover the struct's fields by unioning
        // across samples, not return empty because sample 0 lacked the map
        // (which would silently blind a blanket u64_fields/f64_fields
        // projection).
        let drained = vec![
            (
                "periodic_000".to_string(),
                crate::monitor::dump::FailureDumpReport::default(),
                None,
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(10),
                None,
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None);
        let names = series.bpf_map("scx_obj.bss").member_names();
        assert!(
            names.contains(&"nr_dispatched".to_string()),
            "must discover nr_dispatched from sample 1 despite placeholder sample 0; got {names:?}",
        );
        assert!(
            names.contains(&"stall".to_string()),
            "must discover stall from sample 1; got {names:?}",
        );
    }

    #[test]
    fn stats_key_names_union_not_blinded_by_errored_first_sample() {
        // Sample 0 has no stats (Err); sample 1 carries the scx_stats
        // object. key_names must union across samples so the object's keys
        // are discoverable, not empty because sample 0's stats was Err.
        let drained = vec![
            (
                "periodic_000".to_string(),
                synthetic_report(10),
                None,
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(20),
                Some(synthetic_stats(60.0)),
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None);
        let names = series.stats_path("").key_names();
        assert!(
            names.contains(&"busy".to_string()),
            "must discover the scx_stats keys from sample 1 despite sample 0 having no stats; got {names:?}",
        );
    }

    #[test]
    fn periodic_only_filters_non_periodic_tags() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                synthetic_report(10),
                None,
                Some(100),
            ),
            (
                "user_watchpoint_kind".to_string(),
                synthetic_report(99),
                None,
                Some(150),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(20),
                None,
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None).periodic_only();
        assert_eq!(series.len(), 2);
    }
}
