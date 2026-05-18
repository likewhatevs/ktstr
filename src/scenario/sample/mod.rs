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

pub use bpf::BpfMapProjector;
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
    /// `0` when the bridge could not record a timestamp
    /// (legacy stores without elapsed metadata, or
    /// non-periodic captures surfaced through the same drain).
    pub elapsed_ms: u64,
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
#[derive(Debug)]
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
/// but carries the timestamp explicitly (defaulted to `0` when
/// the bridge omitted it) so iteration does not have to handle
/// the `Option<u64>` repeatedly.
#[derive(Debug)]
struct SampleRow {
    tag: String,
    report: FailureDumpReport,
    stats: Result<serde_json::Value, crate::scenario::snapshot::MissingStatsReason>,
    elapsed_ms: u64,
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
    let mut elapsed: Vec<u64> = Vec::with_capacity(rows.len());
    for row in rows {
        tags.push(row.tag.clone());
        elapsed.push(row.elapsed_ms);
        values.push(row_to_slot(row));
    }
    SeriesField::from_parts(label, tags, elapsed, values)
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
                elapsed_ms: elapsed_ms.unwrap_or(0),
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
            .map(|(tag, report, stats, elapsed_ms)| SampleRow {
                tag,
                report,
                stats,
                elapsed_ms: elapsed_ms.unwrap_or(0),
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
    /// borrowed [`Snapshot`], and borrowed `Option<&Value>` stats.
    pub fn iter_samples(&self) -> impl Iterator<Item = Sample<'_>> {
        self.rows.iter().map(|r| Sample {
            tag: r.tag.as_str(),
            elapsed_ms: r.elapsed_ms,
            snapshot: Snapshot::new(&r.report),
            stats: r.stats.as_ref(),
        })
    }
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
            map_type: 2,
            value_size: 16,
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
        FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
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
