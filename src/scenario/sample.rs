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
//! series along either the BPF or the stats axis through
//! [`SampleSeries::bpf`] / [`SampleSeries::stats`] / the typed
//! [`SampleSeries::bpf_map`] / [`SampleSeries::stats_path`]
//! auto-projection helpers. Each projection yields a
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

use crate::monitor::dump::{FailureDumpReport, PerCpuTimeStats};
use crate::monitor::{MonitorReport, MonitorSummary, ScxEventDeltas};

use super::snapshot::{JsonField, Snapshot, SnapshotField, SnapshotResult, stats_path};
use crate::assert::temporal::SeriesField;

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
    /// BEFORE the freeze rendezvous. `None` when the stats client
    /// was not wired (`scheduler_binary` is absent) or the request
    /// failed (relay rejected, non-zero envelope errno, scheduler
    /// not yet listening). [`SampleSeries::stats`] surfaces this
    /// `None` as a per-sample
    /// [`SnapshotError::MissingStats`](crate::scenario::snapshot::SnapshotError::MissingStats)
    /// slot in the resulting [`SeriesField`] rather than vacuously
    /// skipping; temporal patterns handle that error per their own
    /// policy (gap-tolerant patterns like `nondecreasing`,
    /// `rate_within`, `steady_within`, `converges_to`, and
    /// `ratio_within` skip the sample with a rendered Note, while
    /// strict patterns like `always_true` and `each` fail the
    /// assertion so a stats-coverage gap can never silently slip
    /// past the call site).
    pub stats: Option<&'a serde_json::Value>,
}

/// Ordered collection of [`Sample`]s drained from a
/// [`SnapshotBridge`](super::snapshot::SnapshotBridge) after a VM
/// run completes. Owns the underlying tuples so projection
/// closures can borrow into the reports / stats without copying.
///
/// Test authors construct a `SampleSeries` from
/// [`super::snapshot::SnapshotBridge::drain_ordered_with_stats`]
/// via [`Self::from_drained`]; non-periodic tags (e.g. `Op::Snapshot`
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
    stats: Option<serde_json::Value>,
    elapsed_ms: u64,
}

impl SampleSeries {
    /// Build a series from the bridge's drained tuple. Every entry
    /// is preserved in the order the bridge surfaced, including
    /// non-periodic tags — callers that want the periodic-only
    /// view chain `.periodic_only()`.
    ///
    /// `monitor` is the per-VM-run [`MonitorReport`] (typically
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
    /// `Op::Snapshot` and watchpoint-fire captures share the
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

    /// Project the series along the BPF axis. The closure receives
    /// each sample's [`Snapshot`] and returns a
    /// [`SnapshotResult<T>`] — typically a typed value extracted
    /// via `snap.var(...).as_u64()` or
    /// `snap.map(...).at(...).get(...).as_u64()`. Errors flow
    /// through into the resulting [`SeriesField`] as per-sample
    /// `Err` slots so a temporal-assertion pattern can decide
    /// whether to fail or skip on a missing field.
    ///
    /// `label` is owned (`impl Into<String>`) and lands in
    /// [`crate::assert::temporal::SeriesField::label`] for failure-
    /// message rendering. Callers may pass a `&'static str` literal
    /// or a runtime-built `String` (for auto-discovered struct or
    /// JSON key names).
    pub fn bpf<T, F>(&self, label: impl Into<String>, project: F) -> SeriesField<T>
    where
        F: Fn(&Snapshot<'_>) -> SnapshotResult<T>,
    {
        let mut values: Vec<SnapshotResult<T>> = Vec::with_capacity(self.rows.len());
        let mut tags: Vec<String> = Vec::with_capacity(self.rows.len());
        let mut elapsed: Vec<u64> = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            tags.push(row.tag.clone());
            elapsed.push(row.elapsed_ms);
            // Placeholder reports carry no real BPF state — the
            // freeze rendezvous timed out (or the capture pipeline
            // otherwise failed). Surface a dedicated PlaceholderSample
            // error variant BEFORE invoking the projection closure
            // so the temporal-assertion patterns can branch on
            // "placeholder, skip" distinctly from "field missing,
            // skip" when rendering the verdict's skip-Note.
            if row.report.is_placeholder {
                values.push(Err(
                    crate::scenario::snapshot::SnapshotError::PlaceholderSample {
                        tag: row.tag.clone(),
                        reason: row
                            .report
                            .scx_walker_unavailable
                            .clone()
                            .unwrap_or_else(|| "placeholder report".to_string()),
                    },
                ));
                continue;
            }
            let snap = Snapshot::new(&row.report);
            values.push(project(&snap));
        }
        SeriesField::from_parts(label, tags, elapsed, values)
    }

    /// Project the series along the stats axis. The closure
    /// receives each sample's stats JSON (when present) and
    /// returns a [`SnapshotResult<T>`]. Samples whose `stats` is
    /// `None` get a `Err(MissingStats)` slot — temporal assertions
    /// surface that as a per-sample missing-stats failure rather
    /// than vacuously skipping it, so a coverage gap is never
    /// silent.
    ///
    /// `label` is owned (`impl Into<String>`) and matches the
    /// shape of [`Self::bpf`] — pass a literal or a runtime-built
    /// `String` for auto-discovered keys.
    pub fn stats<T, F>(&self, label: impl Into<String>, project: F) -> SeriesField<T>
    where
        F: Fn(StatsValue<'_>) -> SnapshotResult<T>,
    {
        let mut values: Vec<SnapshotResult<T>> = Vec::with_capacity(self.rows.len());
        let mut tags: Vec<String> = Vec::with_capacity(self.rows.len());
        let mut elapsed: Vec<u64> = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            tags.push(row.tag.clone());
            elapsed.push(row.elapsed_ms);
            let outcome = match row.stats.as_ref() {
                Some(v) => project(StatsValue { value: v }),
                None => Err(crate::scenario::snapshot::SnapshotError::MissingStats {
                    tag: row.tag.clone(),
                }),
            };
            values.push(outcome);
        }
        SeriesField::from_parts(label, tags, elapsed, values)
    }

    /// Auto-project a top-level BPF map's struct members. The
    /// returned [`BpfMapProjector`] auto-discovers struct member
    /// names at sample 0 and exposes them via `.field_u64(name)` /
    /// `.field_i64(name)` / `.field_f64(name)` — a caller that
    /// wants every scalar field of a BSS struct without
    /// enumerating each one by hand calls
    /// `series.bpf_map("scx_obj.bss").at(0)` and then
    /// `.field_u64("nr_dispatched")` for the field of interest.
    ///
    /// **Top-level scalar fields only.** The auto-projector reads
    /// directly-named struct members (e.g. `"nr_dispatched"`,
    /// `"stall"`). Nested struct members (e.g. `"ctx.weight"`) and
    /// deeper paths are NOT auto-discoverable through the typed
    /// `field_*` helpers — for those, use the manual closure
    /// projection [`SampleSeries::bpf`] with
    /// `|snap| snap.var("ctx").get("weight").as_u64()` (or the
    /// equivalent map-walking shape). Per-CPU maps are also out
    /// of scope: they require an explicit `.cpu(N)` narrow on
    /// the [`Snapshot`] accessor surface, so callers route
    /// through the manual closure path for those as well.
    pub fn bpf_map<'a>(&'a self, map_name: &'a str) -> BpfMapProjector<'a> {
        BpfMapProjector {
            series: self,
            map_name,
            entry_index: 0,
        }
    }

    /// Auto-project a stats-JSON sub-tree. The returned
    /// [`StatsPathProjector`] resolves the tree at sample 0 and
    /// exposes object keys via `.key(name)` (for nested layer /
    /// cgroup objects) or `.field(name)` (for scalar leaves).
    /// `path` may be empty — `series.stats_path("")` projects from
    /// the root and is the canonical entry for system-level stats
    /// fields like `busy`, `antistall`, `system_cpu_util_ewma`,
    /// etc.
    pub fn stats_path<'a>(&'a self, path: &str) -> StatsPathProjector<'a> {
        StatsPathProjector {
            series: self,
            path: path.to_string(),
        }
    }

    /// Borrowed view over the per-VM-run host monitor report
    /// associated with this series. `None` when the monitor did
    /// not run (host-only tests, early VM failure, or
    /// [`Self::from_drained`] was called with `None` monitor).
    ///
    /// Monitor is per-series — aggregates inside the returned
    /// [`MonitorView`] refer to THAT series' monitoring window
    /// only; no cross-series merge is supported. A test that
    /// constructs two `SampleSeries` from two VM runs gets two
    /// independent monitors.
    ///
    /// The returned `MonitorView<'_>` borrows from this series,
    /// so the series must outlive any projection chained off the
    /// view (e.g. `series.monitor().map(|m|
    /// m.scx_events()?.total_pairs())` — the whole chain is bound
    /// by `series`'s lifetime).
    pub fn monitor(&self) -> Option<MonitorView<'_>> {
        self.monitor.as_ref().map(|m| MonitorView { report: m })
    }

    /// Borrowed view over the per-sample host-side per-CPU snapshot
    /// data captured into each [`FailureDumpReport::per_cpu_time`](crate::monitor::dump::FailureDumpReport::per_cpu_time).
    /// Returns `None` when the series is empty; otherwise yields a
    /// [`HostView`] that exposes the per-CPU timeline (rows sorted
    /// by elapsed-ms) and a closure-based projector compatible with
    /// the temporal-assertion patterns in
    /// [`crate::assert::temporal`].
    ///
    /// Orthogonal to [`Self::monitor`]: `host()` is the per-sample
    /// per-CPU TIMELINE; `monitor()` is the per-VM-run cross-CPU
    /// AGGREGATE. Tests that want both perspectives chain them
    /// independently from the same series.
    ///
    /// The returned `HostView<'_>` borrows from this series, so the
    /// series must outlive any projection chained off the view.
    pub fn host(&self) -> Option<HostView<'_>> {
        if self.rows.is_empty() {
            None
        } else {
            Some(HostView { rows: &self.rows })
        }
    }
}

/// Borrowed view over a per-VM-run [`MonitorReport`]. Returned by
/// [`SampleSeries::monitor`]; provides typed access to the report's
/// summary statistics + the SCX event-counter deltas.
///
/// Aggregates here refer to the monitoring window of THE SERIES
/// THIS VIEW WAS DRAWN FROM — not the entire test run, not
/// cumulative across series. A test that wants cross-series
/// aggregation must perform it explicitly.
#[derive(Debug, Clone, Copy)]
#[must_use = "MonitorView is a borrowed view; call .summary() or .scx_events() to project"]
#[non_exhaustive]
pub struct MonitorView<'a> {
    report: &'a MonitorReport,
}

impl<'a> MonitorView<'a> {
    /// Aggregate summary statistics: imbalance ratio, nr_running
    /// averages, local DSQ depth, stuck-CPU detection, and
    /// optional schedstat / prog-stats deltas. See
    /// [`MonitorSummary`] for the full field set.
    pub fn summary(&self) -> &'a MonitorSummary {
        &self.report.summary
    }

    /// SCX event-counter accessor. Returns `None` when the monitor
    /// ran but `event_deltas` were not computed (kernel without
    /// event counters, monitoring window too short to compute
    /// first/last deltas) — Option chain matches the source
    /// `MonitorSummary::event_deltas: Option<ScxEventDeltas>` field.
    /// Callers chain `if let Some(evt) = view.scx_events()` to
    /// branch on availability without panicking.
    pub fn scx_events(&self) -> Option<ScxEventsView<'a>> {
        self.report
            .summary
            .event_deltas
            .as_ref()
            .map(|deltas| ScxEventsView { deltas })
    }
}

/// Borrowed view over the [`ScxEventDeltas`] aggregated across the
/// monitor's first/last sample window. Returned by
/// [`MonitorView::scx_events`]; exposes the 14 i64 counter totals
/// via [`Self::total_pairs`] and the 2 f64 derived rates via
/// [`Self::rates_pairs`].
#[derive(Debug, Clone, Copy)]
#[must_use = "ScxEventsView is a borrowed view; call .total_pairs() or .rates_pairs() to project"]
#[non_exhaustive]
pub struct ScxEventsView<'a> {
    deltas: &'a ScxEventDeltas,
}

impl<'a> ScxEventsView<'a> {
    /// All 14 i64 counter totals as `(name, value)` pairs in the
    /// shape that feeds
    /// [`crate::assert::assert_scx_events_clean`]. Order:
    /// `select_cpu_fallback`, `select_cpu_fallback_max_burst`,
    /// `dispatch_local_dsq_offline`, `dispatch_keep_last`,
    /// `enq_skip_exiting`, `enq_skip_migration_disabled`,
    /// `reenq_immed`, `reenq_local_repeat`, `refill_slice_dfl`,
    /// `bypass_duration_ns`, `bypass_dispatch`, `bypass_activate`,
    /// `insert_not_owned`, `sub_bypass_dispatch`.
    ///
    /// **STRICTNESS WARNING:** `assert_scx_events_clean(pairs,
    /// None)` against the full 14-entry slice will spuriously
    /// fail under normal scheduling load — several counters
    /// (`bypass_*`, `dispatch_keep_last`, `refill_slice_dfl`)
    /// legitimately fire on healthy schedulers. Callers either
    /// curate the slice (`pairs.iter().filter(...).collect()`)
    /// or pass `Some(bound)` for non-error-class events. The
    /// projector deliberately does NOT bake "error class" judgment
    /// in — different test scenarios consider different counters
    /// error-class.
    ///
    /// Example — assert only error-class counters are zero by
    /// curating the slice before the assertion:
    ///
    /// ```no_run
    /// # use ktstr::scenario::sample::SampleSeries;
    /// # use ktstr::assert::assert_scx_events_clean;
    /// # fn example(series: &SampleSeries) {
    /// if let Some(view) = series.monitor()
    ///     && let Some(events) = view.scx_events()
    /// {
    ///     let pairs = events.total_pairs();
    ///     let error_only: Vec<(&str, i64)> = pairs
    ///         .into_iter()
    ///         .filter(|(name, _)| {
    ///             matches!(
    ///                 *name,
    ///                 "enq_skip_exiting"
    ///                     | "enq_skip_migration_disabled"
    ///                     | "reenq_immed"
    ///                     | "reenq_local_repeat"
    ///                     | "insert_not_owned"
    ///             )
    ///         })
    ///         .collect();
    ///     assert!(assert_scx_events_clean(&error_only, None).passed);
    /// }
    /// # }
    /// ```
    pub fn total_pairs(&self) -> Vec<(&'static str, i64)> {
        vec![
            ("select_cpu_fallback", self.deltas.total_fallback),
            ("select_cpu_fallback_max_burst", self.deltas.max_fallback_burst),
            ("dispatch_local_dsq_offline", self.deltas.total_dispatch_offline),
            ("dispatch_keep_last", self.deltas.total_dispatch_keep_last),
            ("enq_skip_exiting", self.deltas.total_enq_skip_exiting),
            ("enq_skip_migration_disabled", self.deltas.total_enq_skip_migration_disabled),
            ("reenq_immed", self.deltas.total_reenq_immed),
            ("reenq_local_repeat", self.deltas.total_reenq_local_repeat),
            ("refill_slice_dfl", self.deltas.total_refill_slice_dfl),
            ("bypass_duration_ns", self.deltas.total_bypass_duration),
            ("bypass_dispatch", self.deltas.total_bypass_dispatch),
            ("bypass_activate", self.deltas.total_bypass_activate),
            ("insert_not_owned", self.deltas.total_insert_not_owned),
            ("sub_bypass_dispatch", self.deltas.total_sub_bypass_dispatch),
        ]
    }

    /// Derived per-second rate fields as `(name, value)` pairs.
    /// Separate from [`Self::total_pairs`] because rates have a
    /// different semantic (rate-bounded asserts, not count-bounded)
    /// and a different value type (f64 vs i64). Order:
    /// `select_cpu_fallback_rate`, `dispatch_keep_last_rate`.
    pub fn rates_pairs(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("select_cpu_fallback_rate", self.deltas.fallback_rate),
            ("dispatch_keep_last_rate", self.deltas.keep_last_rate),
        ]
    }
}

/// Borrowed view over the per-sample per-CPU
/// [`PerCpuTimeStats`] data that the host capture pipeline populates
/// into each [`FailureDumpReport::per_cpu_time`](crate::monitor::dump::FailureDumpReport::per_cpu_time). Returned by
/// [`SampleSeries::host`]; exposes a per-CPU timeline (rows sorted
/// ascending by elapsed-ms, stable on ties) plus a closure-based
/// projector that emits a [`SeriesField<u64>`] compatible with the
/// temporal-assertion patterns in [`crate::assert::temporal`].
///
/// Orthogonal to [`MonitorView`]: this view is the per-sample
/// per-CPU TIMELINE source; `MonitorView` exposes the per-VM-run
/// cross-CPU AGGREGATE. The two draw from different fields on the
/// captured reports (`FailureDumpReport::per_cpu_time` here vs
/// `MonitorReport.summary` for the monitor view) and never overlap.
///
/// Placeholder samples (the freeze rendezvous timed out, the
/// capture pipeline otherwise failed) carry an empty `per_cpu_time`
/// slice and naturally drop out of every per-CPU timeline without
/// an explicit filter — temporal-assertion patterns see the
/// surrounding non-placeholder samples in order.
#[derive(Debug, Clone, Copy)]
#[must_use = "HostView is a borrowed view; call .per_cpu_time_timeline() / .per_cpu_field_u64() / .cpus() to project"]
#[non_exhaustive]
pub struct HostView<'a> {
    rows: &'a [SampleRow],
}

impl<'a> HostView<'a> {
    /// Discover every CPU id that appears in at least one sample's
    /// `per_cpu_time` slice. Returned in ascending order, deduped.
    /// Useful for "fan-out over every captured CPU" assertion
    /// loops: `for cpu in host.cpus() { ... }`.
    pub fn cpus(&self) -> Vec<u32> {
        let mut seen = std::collections::BTreeSet::new();
        for row in self.rows {
            for entry in &row.report.per_cpu_time {
                seen.insert(entry.cpu);
            }
        }
        seen.into_iter().collect()
    }

    /// Per-CPU timeline: every sample that captured `cpu`, sorted
    /// ascending by `elapsed_ms`. Ties retain insertion order
    /// (stable sort). Samples whose `per_cpu_time` slice didn't
    /// include `cpu` (placeholder reports, or a kernel without
    /// per-CPU stats) are absent from the returned timeline rather
    /// than producing a default-zero row that would silently advance
    /// counter-style assertions.
    ///
    /// Returns an empty Vec when `cpu` was not captured in any
    /// sample. Test authors that need explicit per-sample
    /// coverage discrimination iterate via
    /// [`SampleSeries::iter_samples`] and consult
    /// [`crate::scenario::snapshot::Snapshot::per_cpu_time_at`] per
    /// sample.
    pub fn per_cpu_time_timeline(&self, cpu: u32) -> Vec<(u64, &'a PerCpuTimeStats)> {
        let mut entries: Vec<(u64, &'a PerCpuTimeStats)> = Vec::new();
        for row in self.rows {
            if let Some(stats) = row.report.per_cpu_time.iter().find(|c| c.cpu == cpu) {
                entries.push((row.elapsed_ms, stats));
            }
        }
        entries.sort_by_key(|(elapsed_ms, _)| *elapsed_ms);
        entries
    }

    /// Project a single u64 field out of each per-sample
    /// `PerCpuTimeStats` row for `cpu` into a [`SeriesField<u64>`]
    /// suitable for the temporal-assertion patterns
    /// (`nondecreasing`, `rate_within`, `steady_within`,
    /// `converges_to`, etc.) in [`crate::assert::temporal`]. Mirrors
    /// the shape of [`SampleSeries::bpf`] so identical assertion
    /// pipelines compose against either axis.
    ///
    /// Samples whose `per_cpu_time` slice didn't include `cpu`
    /// surface as a per-sample
    /// [`SnapshotError::HostFieldUnavailable`](crate::scenario::snapshot::SnapshotError::HostFieldUnavailable)
    /// slot — gap-tolerant temporal patterns skip with a rendered
    /// Note, strict patterns fail the assertion so coverage gaps
    /// can never silently slip past the call site.
    pub fn per_cpu_field_u64(
        &self,
        cpu: u32,
        label: impl Into<String>,
        project: impl Fn(&PerCpuTimeStats) -> u64,
    ) -> SeriesField<u64> {
        let mut values: Vec<SnapshotResult<u64>> = Vec::with_capacity(self.rows.len());
        let mut tags: Vec<String> = Vec::with_capacity(self.rows.len());
        let mut elapsed: Vec<u64> = Vec::with_capacity(self.rows.len());
        for row in self.rows {
            tags.push(row.tag.clone());
            elapsed.push(row.elapsed_ms);
            // Placeholder reports surface as the dedicated
            // PlaceholderSample variant — matching the series.bpf
            // pattern so temporal-assertion sites route placeholder
            // samples through their per-sample skip handling rather
            // than treating them as cpu-coverage gaps. A strict
            // pattern (always_true / each.at_least) would otherwise
            // FAIL on placeholders instead of skipping; gap-tolerant
            // patterns render the right diagnostic Note.
            if row.report.is_placeholder {
                values.push(Err(
                    crate::scenario::snapshot::SnapshotError::PlaceholderSample {
                        tag: row.tag.clone(),
                        reason: row
                            .report
                            .scx_walker_unavailable
                            .clone()
                            .unwrap_or_else(|| "placeholder report".to_string()),
                    },
                ));
                continue;
            }
            let slot = match row.report.per_cpu_time.iter().find(|c| c.cpu == cpu) {
                Some(stats) => Ok(project(stats)),
                None => Err(
                    crate::scenario::snapshot::SnapshotError::HostFieldUnavailable {
                        tag: row.tag.clone(),
                        cpu,
                    },
                ),
            };
            values.push(slot);
        }
        SeriesField::from_parts(label, tags, elapsed, values)
    }
}

/// Newtype carrier handed to the [`SampleSeries::stats`] closure.
/// Wraps a borrowed [`serde_json::Value`] and exposes [`Self::path`]
/// as a thin facade over [`stats_path`] so the closure body reads
/// `s.path("layers.batch.util").as_f64()` without an explicit
/// import.
#[derive(Debug, Clone, Copy)]
pub struct StatsValue<'a> {
    value: &'a serde_json::Value,
}

impl<'a> StatsValue<'a> {
    /// Underlying JSON value.
    pub fn raw(&self) -> &'a serde_json::Value {
        self.value
    }

    /// Walk along a dotted path. Empty path returns the root.
    pub fn path(&self, path: &str) -> JsonField<'a> {
        stats_path(self.value, path)
    }
}

/// Auto-projector handle returned by [`SampleSeries::bpf_map`].
/// Lazily resolves the named map's value at the requested entry
/// index when `Self::field` is invoked.
pub struct BpfMapProjector<'a> {
    series: &'a SampleSeries,
    map_name: &'a str,
    entry_index: usize,
}

impl<'a> BpfMapProjector<'a> {
    /// Pin the entry index for the projection. Defaults to `0`
    /// (typical for ARRAY / `.bss` / `.data` / `.rodata` maps,
    /// which carry a single value at index 0). Use this to walk
    /// into a HASH map at a specific ordinal.
    pub fn at(mut self, index: usize) -> Self {
        self.entry_index = index;
        self
    }

    /// Project a single named struct field as `u64` (the most
    /// common temporal-assertion shape — counters, byte counts).
    /// The label routed onto the resulting [`SeriesField`] is the
    /// caller-supplied field name; combined with the map name in
    /// the diagnostic the failure message reads
    /// `"<map>.<entry_index>.<field>"`.
    pub fn field_u64(&self, field: &str) -> SeriesField<u64> {
        let map_name = self.map_name.to_string();
        let entry_index = self.entry_index;
        let field_owned = field.to_string();
        self.series.bpf(field, move |snap| {
            let entry = match snap.map(&map_name) {
                Ok(m) => m.at(entry_index),
                Err(e) => return Err(e),
            };
            entry.get(&field_owned).as_u64()
        })
    }

    /// Project a single named struct field as `i64`.
    pub fn field_i64(&self, field: &str) -> SeriesField<i64> {
        let map_name = self.map_name.to_string();
        let entry_index = self.entry_index;
        let field_owned = field.to_string();
        self.series.bpf(field, move |snap| {
            let entry = match snap.map(&map_name) {
                Ok(m) => m.at(entry_index),
                Err(e) => return Err(e),
            };
            entry.get(&field_owned).as_i64()
        })
    }

    /// Project a single named struct field as `f64`.
    pub fn field_f64(&self, field: &str) -> SeriesField<f64> {
        let map_name = self.map_name.to_string();
        let entry_index = self.entry_index;
        let field_owned = field.to_string();
        self.series.bpf(field, move |snap| {
            let entry = match snap.map(&map_name) {
                Ok(m) => m.at(entry_index),
                Err(e) => return Err(e),
            };
            entry.get(&field_owned).as_f64()
        })
    }

    /// Discover the struct member names of the map's first
    /// rendered value. Empty when the map is missing in sample 0
    /// or its value is not a struct. Useful for tests that want
    /// to enumerate every scalar field for a blanket assertion.
    pub fn member_names(&self) -> Vec<String> {
        let row = match self.series.rows.first() {
            Some(r) => r,
            None => return Vec::new(),
        };
        let snap = Snapshot::new(&row.report);
        let map = match snap.map(self.map_name) {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        let entry = map.at(self.entry_index);
        // Walk the entry's value — SnapshotEntry doesn't expose
        // its struct members directly, but the rendered_value()
        // accessor on the field-with-empty-path does.
        let field = entry.get("");
        match field {
            SnapshotField::Value(crate::monitor::btf_render::RenderedValue::Struct {
                members,
                ..
            }) => members.iter().map(|m| m.name.clone()).collect(),
            _ => Vec::new(),
        }
    }

    /// Project every struct member that resolves as `u64` for at
    /// least one sample. Iterates [`Self::member_names`], calls
    /// [`Self::field_u64`] for each, and keeps the entries whose
    /// resulting [`SeriesField`] has at least one `Ok` value —
    /// non-numeric members (strings, nested structs, floats) drop
    /// out because their `as_u64()` cast always errors.
    pub fn u64_fields(&self) -> Vec<(String, SeriesField<u64>)> {
        self.member_names()
            .into_iter()
            .filter_map(|name| {
                let field = self.field_u64(&name);
                // Bind the predicate result and drop the
                // values_iter borrow before moving `field`. A
                // chained `.values_iter().any(...).then_some(...)`
                // keeps the iterator alive across the move and
                // fails the borrow check.
                let any_ok = field.values_iter().any(|r| r.is_ok());
                any_ok.then_some((name, field))
            })
            .collect()
    }

    /// Project every struct member that resolves as `f64` for at
    /// least one sample. Mirrors [`Self::u64_fields`] using
    /// [`Self::field_f64`].
    pub fn f64_fields(&self) -> Vec<(String, SeriesField<f64>)> {
        self.member_names()
            .into_iter()
            .filter_map(|name| {
                let field = self.field_f64(&name);
                let any_ok = field.values_iter().any(|r| r.is_ok());
                any_ok.then_some((name, field))
            })
            .collect()
    }
}

/// Auto-projector handle returned by [`SampleSeries::stats_path`].
/// Walks a stats sub-tree per sample and exposes scalar / nested
/// projections for the keys at that level.
pub struct StatsPathProjector<'a> {
    series: &'a SampleSeries,
    path: String,
}

impl<'a> StatsPathProjector<'a> {
    /// Project a JSON key under the resolved path as `u64`.
    pub fn field_u64(&self, key: &str) -> SeriesField<u64> {
        let full_path = join_paths(&self.path, key);
        self.series
            .stats(key, move |sv| sv.path(&full_path).as_u64())
    }

    /// Project a JSON key under the resolved path as `i64`.
    pub fn field_i64(&self, key: &str) -> SeriesField<i64> {
        let full_path = join_paths(&self.path, key);
        self.series
            .stats(key, move |sv| sv.path(&full_path).as_i64())
    }

    /// Project a JSON key under the resolved path as `f64`.
    pub fn field_f64(&self, key: &str) -> SeriesField<f64> {
        let full_path = join_paths(&self.path, key);
        self.series
            .stats(key, move |sv| sv.path(&full_path).as_f64())
    }

    /// Return a sub-projector rooted under `key`. Composable —
    /// `series.stats_path("layers").key("batch").field_f64("util")`
    /// drills into the per-layer scheduler stats one segment at a
    /// time without each call site re-typing the full dotted
    /// path.
    pub fn key(&self, key: &str) -> StatsPathProjector<'a> {
        StatsPathProjector {
            series: self.series,
            path: join_paths(&self.path, key),
        }
    }

    /// Discover the JSON object keys of the resolved path at
    /// sample 0. Empty when the path is missing or resolves to a
    /// non-object; populated when the projection lands on a
    /// `serde_json::Value::Object`.
    pub fn key_names(&self) -> Vec<String> {
        let row = match self.series.rows.first() {
            Some(r) => r,
            None => return Vec::new(),
        };
        let stats = match row.stats.as_ref() {
            Some(s) => s,
            None => return Vec::new(),
        };
        let resolved = stats_path(stats, &self.path);
        let raw = match resolved.raw() {
            Some(v) => v,
            None => return Vec::new(),
        };
        match raw {
            serde_json::Value::Object(map) => {
                let mut names: Vec<String> = map.keys().cloned().collect();
                names.sort();
                names
            }
            _ => Vec::new(),
        }
    }

    /// Project every object key that resolves as `u64` for at
    /// least one sample. Iterates [`Self::key_names`], calls
    /// [`Self::field_u64`] for each, and keeps the entries whose
    /// resulting [`SeriesField`] has at least one `Ok` value —
    /// non-numeric leaves (strings, nested objects, floats) drop
    /// out.
    pub fn u64_fields(&self) -> Vec<(String, SeriesField<u64>)> {
        self.key_names()
            .into_iter()
            .filter_map(|name| {
                let field = self.field_u64(&name);
                // Bind the predicate result and drop the
                // values_iter borrow before moving `field`.
                let any_ok = field.values_iter().any(|r| r.is_ok());
                any_ok.then_some((name, field))
            })
            .collect()
    }

    /// Project every object key that resolves as `f64` for at
    /// least one sample. Mirrors [`Self::u64_fields`] using
    /// [`Self::field_f64`].
    pub fn f64_fields(&self) -> Vec<(String, SeriesField<f64>)> {
        self.key_names()
            .into_iter()
            .filter_map(|name| {
                let field = self.field_f64(&name);
                let any_ok = field.values_iter().any(|r| r.is_ok());
                any_ok.then_some((name, field))
            })
            .collect()
    }
}

fn join_paths(base: &str, leaf: &str) -> String {
    if base.is_empty() {
        leaf.to_string()
    } else if leaf.is_empty() {
        base.to_string()
    } else {
        format!("{base}.{leaf}")
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

    #[test]
    fn bpf_projection_extracts_field_per_sample() {
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
                None,
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None);
        let field: SeriesField<u64> =
            series.bpf("nr_dispatched", |snap| snap.var("nr_dispatched").as_u64());
        let values: Vec<u64> = field
            .values_iter()
            .filter_map(|v| v.as_ref().ok().copied())
            .collect();
        assert_eq!(values, vec![10, 20]);
    }

    #[test]
    fn stats_projection_handles_missing_stats_as_error() {
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
                None,
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None);
        let field: SeriesField<f64> = series.stats("busy", |s| s.path("busy").as_f64());
        let outcomes: Vec<SnapshotResult<f64>> = field.values_iter().cloned().collect();
        assert_eq!(outcomes.len(), 2);
        assert_eq!(
            outcomes[0].as_ref().copied(),
            Ok(50.0),
            "sample with stats present must project the `busy` field verbatim"
        );
        match &outcomes[1] {
            Err(crate::scenario::snapshot::SnapshotError::MissingStats { tag }) => {
                assert_eq!(
                    tag, "periodic_001",
                    "MissingStats tag must identify the sample whose stats slot was None"
                );
            }
            other => panic!(
                "sample with stats=None must surface SnapshotError::MissingStats, got {other:?}"
            ),
        }
    }

    #[test]
    fn bpf_map_projector_field_u64_extracts_field() {
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
                None,
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None);
        let field = series
            .bpf_map("scx_obj.bss")
            .at(0)
            .field_u64("nr_dispatched");
        let values: Vec<u64> = field
            .values_iter()
            .filter_map(|v| v.as_ref().ok().copied())
            .collect();
        assert_eq!(values, vec![10, 20]);
    }

    #[test]
    fn bpf_map_projector_member_names_lists_struct_fields() {
        let drained = vec![(
            "periodic_000".to_string(),
            synthetic_report(10),
            None,
            Some(100),
        )];
        let series = SampleSeries::from_drained(drained, None);
        let names = series.bpf_map("scx_obj.bss").at(0).member_names();
        assert!(names.contains(&"nr_dispatched".to_string()));
        assert!(names.contains(&"stall".to_string()));
    }

    #[test]
    fn stats_path_projector_field_f64_extracts_root_scalar() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                synthetic_report(0),
                Some(synthetic_stats(50.0)),
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(0),
                Some(synthetic_stats(60.0)),
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None);
        let field = series.stats_path("").field_f64("busy");
        let values: Vec<f64> = field
            .values_iter()
            .filter_map(|v| v.as_ref().ok().copied())
            .collect();
        assert_eq!(values.len(), 2);
        assert!((values[0] - 50.0).abs() < f64::EPSILON);
        assert!((values[1] - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_path_projector_key_names_at_root() {
        let drained = vec![(
            "periodic_000".to_string(),
            synthetic_report(0),
            Some(synthetic_stats(50.0)),
            Some(100),
        )];
        let series = SampleSeries::from_drained(drained, None);
        let names = series.stats_path("").key_names();
        assert!(names.contains(&"busy".to_string()));
        assert!(names.contains(&"layers".to_string()));
    }

    #[test]
    fn stats_path_projector_nested_key_drills_in() {
        let drained = vec![(
            "periodic_000".to_string(),
            synthetic_report(0),
            Some(synthetic_stats(50.0)),
            Some(100),
        )];
        let series = SampleSeries::from_drained(drained, None);
        // Note: drilling deeper than 2 levels via key() chain works
        // because key() returns the same kind of projector.
        let field = series.stats_path("layers").key("batch").field_f64("util");
        let values: Vec<f64> = field
            .values_iter()
            .filter_map(|v| v.as_ref().ok().copied())
            .collect();
        assert_eq!(values.len(), 1);
        assert!((values[0] - 25.0).abs() < f64::EPSILON);
    }

    /// Build a synthetic report with mixed-shape members so the
    /// `u64_fields` / `f64_fields` auto-projectors exercise the
    /// "at least one Ok" filter:
    ///   - `nr_dispatched`: Uint — projects Ok as u64.
    ///   - `stall`: Uint — projects Ok as u64.
    ///   - `balance`: Float — projects Err as u64 (TypeMismatch),
    ///     Ok as f64.
    ///   - `flag_str`: Bytes — projects Err as both u64 and f64.
    fn mixed_shape_report(disp: u64, balance: f64) -> FailureDumpReport {
        let bss_value = RenderedValue::Struct {
            type_name: Some(".bss".into()),
            members: vec![
                RenderedMember {
                    name: "nr_dispatched".into(),
                    value: RenderedValue::Uint {
                        bits: 64,
                        value: disp,
                    },
                },
                RenderedMember {
                    name: "stall".into(),
                    value: RenderedValue::Uint { bits: 8, value: 0 },
                },
                RenderedMember {
                    name: "balance".into(),
                    value: RenderedValue::Float {
                        bits: 64,
                        value: balance,
                    },
                },
                RenderedMember {
                    name: "flag_str".into(),
                    value: RenderedValue::Bytes {
                        hex: "de ad".into(),
                    },
                },
            ],
        };
        let bss_map = FailureDumpMap {
            name: "scx_obj.bss".into(),
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
        FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: vec![bss_map],
            ..Default::default()
        }
    }

    /// `BpfMapProjector::u64_fields` keeps every member that yields
    /// at least one `Ok` u64 across the series and drops members
    /// whose every-sample projection errors. The mixed-shape report
    /// above carries two u64 members (`nr_dispatched`, `stall`),
    /// one f64-only member (`balance`) that errors on every u64
    /// projection, and one bytes member (`flag_str`) that also
    /// errors on every u64 projection. The returned vec must
    /// surface only the two u64 names. The `SeriesField::label`
    /// is set to the field name (see `BpfMapProjector::field_u64`),
    /// so the tuple's first slot matches the struct member name
    /// exactly.
    #[test]
    fn bpf_map_projector_u64_fields_keeps_at_least_one_ok_excludes_all_err() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                mixed_shape_report(10, 1.5),
                None,
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                mixed_shape_report(20, 2.5),
                None,
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None);
        let fields = series.bpf_map("scx_obj.bss").at(0).u64_fields();
        let names: Vec<&str> = fields.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"nr_dispatched"),
            "u64-shaped member must be kept: {names:?}",
        );
        assert!(
            names.contains(&"stall"),
            "u64-shaped member must be kept: {names:?}",
        );
        assert!(
            !names.contains(&"balance"),
            "Float-shaped member must be excluded — every u64 projection errors: {names:?}",
        );
        assert!(
            !names.contains(&"flag_str"),
            "Bytes-shaped member must be excluded — every u64 projection errors: {names:?}",
        );
        // The kept fields must carry the projected u64 values
        // verbatim — the tuple's SeriesField is the same object
        // `field_u64(name)` would return.
        let dispatched = fields
            .iter()
            .find(|(n, _)| n == "nr_dispatched")
            .expect("nr_dispatched kept above");
        let values: Vec<u64> = dispatched
            .1
            .values_iter()
            .filter_map(|r| r.as_ref().ok().copied())
            .collect();
        assert_eq!(
            values,
            vec![10, 20],
            "kept SeriesField must carry the per-sample u64 projection",
        );
    }

    /// Mirror of the u64 test for `f64_fields`. Float, Uint, Int,
    /// and Enum members coerce to f64 (see `SnapshotField::as_f64`),
    /// so all three numeric members are kept; the Bytes member
    /// errors and is dropped. This pins the "at least one Ok"
    /// filter for the f64 axis distinctly from the u64 axis.
    #[test]
    fn bpf_map_projector_f64_fields_keeps_at_least_one_ok_excludes_all_err() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                mixed_shape_report(10, 1.5),
                None,
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                mixed_shape_report(20, 2.5),
                None,
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None);
        let fields = series.bpf_map("scx_obj.bss").at(0).f64_fields();
        let names: Vec<&str> = fields.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"nr_dispatched"),
            "Uint coerces to f64 — must be kept: {names:?}",
        );
        assert!(
            names.contains(&"stall"),
            "Uint coerces to f64 — must be kept: {names:?}",
        );
        assert!(
            names.contains(&"balance"),
            "Float coerces to f64 — must be kept: {names:?}",
        );
        assert!(
            !names.contains(&"flag_str"),
            "Bytes does not coerce to f64 — must be excluded: {names:?}",
        );
        let balance = fields
            .iter()
            .find(|(n, _)| n == "balance")
            .expect("balance kept above");
        let values: Vec<f64> = balance
            .1
            .values_iter()
            .filter_map(|r| r.as_ref().ok().copied())
            .collect();
        assert_eq!(values.len(), 2, "balance must surface one f64 per sample",);
        assert!((values[0] - 1.5).abs() < f64::EPSILON);
        assert!((values[1] - 2.5).abs() < f64::EPSILON);
    }

    /// Empty series — no rows to discover member names from, so
    /// `member_names()` returns an empty vec and both auto-projectors
    /// yield empty results without panicking. Pins the "no first
    /// row" branch in `BpfMapProjector::member_names`.
    #[test]
    fn bpf_map_projector_field_helpers_empty_series_yields_empty_vec() {
        let series = SampleSeries::empty();
        let u64s = series.bpf_map("scx_obj.bss").at(0).u64_fields();
        assert!(
            u64s.is_empty(),
            "empty series must yield empty u64_fields, got {} entries",
            u64s.len(),
        );
        let f64s = series.bpf_map("scx_obj.bss").at(0).f64_fields();
        assert!(
            f64s.is_empty(),
            "empty series must yield empty f64_fields, got {} entries",
            f64s.len(),
        );
    }

    /// Build stats payloads with mixed shapes so the
    /// `StatsPathProjector` auto-projectors exercise the same
    /// "at least one Ok" filter on the JSON axis:
    ///   - `busy`: Number — projects Ok as u64 and f64.
    ///   - `count`: Number — projects Ok as u64 and f64.
    ///   - `ratio`: Number(float) — projects Ok as f64;
    ///     u64 errors when the float has a non-zero
    ///     fraction (see `json_to_u64`).
    ///   - `name`: String("nope") — never coerces to numeric.
    fn mixed_stats(busy: u64, count: u64) -> serde_json::Value {
        serde_json::json!({
            "busy": busy,
            "count": count,
            "ratio": 0.5,
            "name": "nope",
        })
    }

    /// `StatsPathProjector::u64_fields` keeps JSON keys whose
    /// per-sample projection lands at least one Ok and drops keys
    /// whose every projection errors. `busy` / `count` are integer
    /// numbers (Ok u64); `ratio` is `0.5` and lands TypeMismatch
    /// on every sample (`json_to_u64` rejects non-integer floats);
    /// `name` is a string that does not parse — also Err.
    #[test]
    fn stats_path_projector_u64_fields_keeps_at_least_one_ok_excludes_all_err() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                synthetic_report(0),
                Some(mixed_stats(50, 7)),
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(0),
                Some(mixed_stats(60, 9)),
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None);
        let fields = series.stats_path("").u64_fields();
        let names: Vec<&str> = fields.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"busy"),
            "Number(integer) key must be kept: {names:?}",
        );
        assert!(
            names.contains(&"count"),
            "Number(integer) key must be kept: {names:?}",
        );
        assert!(
            !names.contains(&"ratio"),
            "Number(non-integer float) errors on every u64 projection — must be excluded: {names:?}",
        );
        assert!(
            !names.contains(&"name"),
            "String('nope') errors on every u64 projection — must be excluded: {names:?}",
        );
        let busy = fields
            .iter()
            .find(|(n, _)| n == "busy")
            .expect("busy kept above");
        let values: Vec<u64> = busy
            .1
            .values_iter()
            .filter_map(|r| r.as_ref().ok().copied())
            .collect();
        assert_eq!(values, vec![50, 60]);
    }

    /// Mirror of the u64 test for `f64_fields`. Numbers coerce to
    /// f64 unconditionally (`json_to_f64`) — `busy`, `count`, and
    /// `ratio` are all kept. `name` is a non-numeric string and
    /// is excluded.
    #[test]
    fn stats_path_projector_f64_fields_keeps_at_least_one_ok_excludes_all_err() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                synthetic_report(0),
                Some(mixed_stats(50, 7)),
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(0),
                Some(mixed_stats(60, 9)),
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None);
        let fields = series.stats_path("").f64_fields();
        let names: Vec<&str> = fields.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"busy"),
            "Number coerces to f64 — must be kept: {names:?}",
        );
        assert!(
            names.contains(&"count"),
            "Number coerces to f64 — must be kept: {names:?}",
        );
        assert!(
            names.contains(&"ratio"),
            "Number(float) coerces to f64 — must be kept: {names:?}",
        );
        assert!(
            !names.contains(&"name"),
            "String('nope') errors on every f64 projection — must be excluded: {names:?}",
        );
        let ratio = fields
            .iter()
            .find(|(n, _)| n == "ratio")
            .expect("ratio kept above");
        let values: Vec<f64> = ratio
            .1
            .values_iter()
            .filter_map(|r| r.as_ref().ok().copied())
            .collect();
        assert_eq!(values.len(), 2);
        assert!((values[0] - 0.5).abs() < f64::EPSILON);
        assert!((values[1] - 0.5).abs() < f64::EPSILON);
    }

    /// Empty series — `key_names()` returns an empty vec because
    /// there is no sample 0 to resolve the path against, so both
    /// auto-projectors yield empty results without panicking.
    #[test]
    fn stats_path_projector_field_helpers_empty_series_yields_empty_vec() {
        let series = SampleSeries::empty();
        let u64s = series.stats_path("").u64_fields();
        assert!(
            u64s.is_empty(),
            "empty series must yield empty u64_fields, got {} entries",
            u64s.len(),
        );
        let f64s = series.stats_path("").f64_fields();
        assert!(
            f64s.is_empty(),
            "empty series must yield empty f64_fields, got {} entries",
            f64s.len(),
        );
    }

    /// `series.monitor()` returns `None` when `from_drained` was
    /// called with `None` monitor (the common test-fixture case).
    /// Pins the Option chain shortcut at the top — caller pattern
    /// `if let Some(view) = series.monitor()` must NOT panic and
    /// must NOT vacuously return default-empty data.
    #[test]
    fn series_monitor_none_when_unset() {
        let series = SampleSeries::from_drained(vec![], None);
        assert!(series.monitor().is_none());
    }

    /// `series.monitor()` returns `Some(view)` when monitor was
    /// supplied; the view wraps the supplied report and the inner
    /// `.summary()` accessor returns a reference to the report's
    /// summary unchanged. Pins the borrow-through-view shape.
    #[test]
    fn series_monitor_view_threads_through_supplied_report() {
        let mut report = MonitorReport::default();
        report.summary.total_samples = 42;
        report.summary.max_imbalance_ratio = 3.14;
        let series = SampleSeries::from_drained(vec![], Some(report));
        let view = series.monitor().expect("monitor must be Some");
        let summary = view.summary();
        assert_eq!(summary.total_samples, 42);
        assert_eq!(summary.max_imbalance_ratio, 3.14);
    }

    /// `view.scx_events()` returns `None` when `event_deltas` is
    /// `None` on the underlying summary (kernel without event
    /// counters, monitoring window too short). Inner-Option chain
    /// must NOT collapse to default-zero pairs — that would be a
    /// silent-loss path the no-silent-drops rule forbids.
    #[test]
    fn series_monitor_scx_events_none_when_event_deltas_absent() {
        let report = MonitorReport::default(); // event_deltas defaults to None
        let series = SampleSeries::from_drained(vec![], Some(report));
        let view = series.monitor().expect("monitor must be Some");
        assert!(
            view.scx_events().is_none(),
            "scx_events must return None when event_deltas is absent — \
             returning Some with zero-default pairs would silently mask the missing-data condition"
        );
    }

    /// `view.scx_events()?.total_pairs()` enumerates all 14 i64
    /// counter fields in the documented order with the documented
    /// names, and `.rates_pairs()` enumerates the 2 f64 derived
    /// rates. Pins the projector's name-to-field mapping against
    /// drift — a regression that reordered fields, renamed a counter,
    /// or accidentally included a rate in total_pairs would fail
    /// here.
    #[test]
    fn series_monitor_scx_events_pairs_map_to_named_counters() {
        let mut report = MonitorReport::default();
        report.summary.event_deltas = Some(ScxEventDeltas {
            total_fallback: 1,
            fallback_rate: 0.5,
            max_fallback_burst: 2,
            total_dispatch_offline: 3,
            total_dispatch_keep_last: 4,
            keep_last_rate: 0.75,
            total_enq_skip_exiting: 5,
            total_enq_skip_migration_disabled: 6,
            total_reenq_immed: 7,
            total_reenq_local_repeat: 8,
            total_refill_slice_dfl: 9,
            total_bypass_duration: 10,
            total_bypass_dispatch: 11,
            total_bypass_activate: 12,
            total_insert_not_owned: 13,
            total_sub_bypass_dispatch: 14,
        });
        let series = SampleSeries::from_drained(vec![], Some(report));
        let view = series.monitor().expect("monitor must be Some");
        let events = view.scx_events().expect("event_deltas were set");
        let totals = events.total_pairs();
        assert_eq!(totals.len(), 14, "exactly 14 i64 counter pairs");
        assert_eq!(
            totals,
            vec![
                ("select_cpu_fallback", 1),
                ("select_cpu_fallback_max_burst", 2),
                ("dispatch_local_dsq_offline", 3),
                ("dispatch_keep_last", 4),
                ("enq_skip_exiting", 5),
                ("enq_skip_migration_disabled", 6),
                ("reenq_immed", 7),
                ("reenq_local_repeat", 8),
                ("refill_slice_dfl", 9),
                ("bypass_duration_ns", 10),
                ("bypass_dispatch", 11),
                ("bypass_activate", 12),
                ("insert_not_owned", 13),
                ("sub_bypass_dispatch", 14),
            ]
        );
        let rates = events.rates_pairs();
        assert_eq!(rates.len(), 2, "exactly 2 f64 rate pairs");
        assert_eq!(
            rates,
            vec![
                ("select_cpu_fallback_rate", 0.5),
                ("dispatch_keep_last_rate", 0.75),
            ]
        );
    }

    /// Pins the STRICTNESS WARNING contract on
    /// [`ScxEventsView::total_pairs`] (sample.rs L429-441): when a
    /// non-error-class counter (`total_bypass_dispatch` here)
    /// legitimately fires alongside an error-class counter at zero,
    /// `assert_scx_events_clean(pairs, None)` against the FULL
    /// 14-entry slice MUST FAIL (because bypass_dispatch > 0); the
    /// CURATED subset of error-class counters MUST PASS (because
    /// every error counter is zero). A future regression that
    /// silently dropped a counter from `total_pairs` or accidentally
    /// curated by the projector would break one of these two
    /// assertions. Adversary O2 fold-in 2026-05-17.
    #[test]
    fn series_monitor_scx_events_strict_zero_misuse_pinning() {
        use crate::assert::assert_scx_events_clean;
        let mut report = MonitorReport::default();
        report.summary.event_deltas = Some(ScxEventDeltas {
            total_bypass_dispatch: 100,
            total_bypass_activate: 50,
            total_dispatch_keep_last: 7,
            ..Default::default()
        });
        let series = SampleSeries::from_drained(vec![], Some(report));
        let view = series.monitor().expect("monitor was set");
        let events = view.scx_events().expect("event_deltas were set");
        let pairs = events.total_pairs();
        // Full slice + strict zero: MUST fail because bypass_*
        // counters fired with legitimate non-zero values.
        let r_full = assert_scx_events_clean(&pairs, None);
        assert!(
            !r_full.passed,
            "strict-zero against full 14-entry slice MUST fail when non-error-class counters legitimately fire — pins the STRICTNESS WARNING design contract"
        );
        // Curated error-class subset: MUST pass because every
        // error-class counter is zero (we only populated bypass_*
        // and dispatch_keep_last, neither of which is error class).
        let error_only: Vec<(&str, i64)> = pairs
            .into_iter()
            .filter(|(name, _)| {
                matches!(
                    *name,
                    "enq_skip_exiting"
                        | "enq_skip_migration_disabled"
                        | "reenq_immed"
                        | "reenq_local_repeat"
                        | "insert_not_owned"
                )
            })
            .collect();
        let r_curated = assert_scx_events_clean(&error_only, None);
        assert!(
            r_curated.passed,
            "curated error-class subset MUST pass when every error counter is zero — pins the curate-then-assert documented workaround"
        );
    }

    /// `series.host()` returns `None` on an empty series (no
    /// samples) — Option chain shortcut, no panic.
    #[test]
    fn series_host_empty_series_returns_none() {
        let series = SampleSeries::from_drained(vec![], None);
        assert!(series.host().is_none());
    }

    /// Single-sample series with N captured CPUs:
    /// `per_cpu_time_timeline(cpu)` returns exactly 1 row for each
    /// captured cpu, empty Vec for any other cpu. Pins the
    /// per-CPU filter — placeholder-or-absent CPUs MUST NOT
    /// surface default-zero rows that would silently advance
    /// counter-style assertions.
    #[test]
    fn series_host_per_cpu_time_timeline_single_sample() {
        let mut report = FailureDumpReport::default();
        report.per_cpu_time = vec![
            PerCpuTimeStats {
                cpu: 0,
                cpustat_user_ns: 100,
                ..Default::default()
            },
            PerCpuTimeStats {
                cpu: 3,
                cpustat_user_ns: 300,
                ..Default::default()
            },
        ];
        let series = SampleSeries::from_drained(
            vec![("periodic_000".to_string(), report, None, Some(50u64))],
            None,
        );
        let host = series.host().expect("non-empty series");
        let t0 = host.per_cpu_time_timeline(0);
        assert_eq!(t0.len(), 1);
        assert_eq!(t0[0].0, 50);
        assert_eq!(t0[0].1.cpustat_user_ns, 100);
        let t3 = host.per_cpu_time_timeline(3);
        assert_eq!(t3.len(), 1);
        assert_eq!(t3[0].1.cpustat_user_ns, 300);
        let t99 = host.per_cpu_time_timeline(99);
        assert!(
            t99.is_empty(),
            "cpu not captured in any sample MUST yield empty timeline (not default-zero)"
        );
        assert_eq!(host.cpus(), vec![0, 3]);
    }

    /// Multi-sample series with NON-monotonic elapsed_ms:
    /// `per_cpu_time_timeline` returns rows sorted ascending by
    /// elapsed_ms; ties retain insertion order (stable sort).
    /// Pins the sort contract against drift to unstable sort or
    /// reverse order.
    #[test]
    fn series_host_per_cpu_time_timeline_sorts_by_elapsed_ms_stable() {
        let mk = |val: u64| {
            let mut r = FailureDumpReport::default();
            r.per_cpu_time = vec![PerCpuTimeStats {
                cpu: 0,
                cpustat_user_ns: val,
                ..Default::default()
            }];
            r
        };
        let series = SampleSeries::from_drained(
            vec![
                ("a".to_string(), mk(100), None, Some(100u64)),
                ("b".to_string(), mk(200), None, Some(50u64)),
                ("c".to_string(), mk(300), None, Some(100u64)),
                ("d".to_string(), mk(400), None, Some(25u64)),
            ],
            None,
        );
        let host = series.host().expect("non-empty");
        let timeline = host.per_cpu_time_timeline(0);
        assert_eq!(timeline.len(), 4);
        assert_eq!(timeline[0].0, 25);
        assert_eq!(timeline[0].1.cpustat_user_ns, 400);
        assert_eq!(timeline[1].0, 50);
        assert_eq!(timeline[1].1.cpustat_user_ns, 200);
        assert_eq!(
            timeline[2].0, 100,
            "first of the tied-elapsed-ms pair: insertion order = 'a'"
        );
        assert_eq!(timeline[2].1.cpustat_user_ns, 100);
        assert_eq!(
            timeline[3].0, 100,
            "second of the tied-elapsed-ms pair: insertion order = 'c'"
        );
        assert_eq!(timeline[3].1.cpustat_user_ns, 300);
    }

    /// Placeholder samples (empty per_cpu_time) naturally drop
    /// from the timeline without an explicit filter. Pins the
    /// "no explicit placeholder-skip needed" contract: a
    /// placeholder mid-stream MUST NOT inject a default-zero
    /// row that would silently advance counter-style assertions.
    #[test]
    fn series_host_placeholder_naturally_drops_without_explicit_filter() {
        let mk_real = |val: u64| {
            let mut r = FailureDumpReport::default();
            r.per_cpu_time = vec![PerCpuTimeStats {
                cpu: 0,
                cpustat_user_ns: val,
                ..Default::default()
            }];
            r
        };
        let placeholder = FailureDumpReport::placeholder("freeze rendezvous timed out");
        let series = SampleSeries::from_drained(
            vec![
                ("real_pre".to_string(), mk_real(10), None, Some(10u64)),
                ("placeholder_mid".to_string(), placeholder, None, Some(20u64)),
                ("real_post".to_string(), mk_real(30), None, Some(30u64)),
            ],
            None,
        );
        let host = series.host().expect("non-empty");
        let timeline = host.per_cpu_time_timeline(0);
        assert_eq!(
            timeline.len(),
            2,
            "placeholder MUST drop from the timeline naturally — pins the no-explicit-filter contract"
        );
        assert_eq!(timeline[0].0, 10);
        assert_eq!(timeline[1].0, 30);
    }

    /// Closure-based `per_cpu_field_u64` projector emits a
    /// [`SeriesField<u64>`] with one slot per sample. Samples
    /// where `cpu` was captured produce `Ok(value)`; samples where
    /// `cpu` was absent surface as
    /// [`SnapshotError::HostFieldUnavailable`] (NOT silently
    /// dropped, NOT default-zero) so coverage gaps reach the
    /// temporal-assertion layer.
    #[test]
    fn series_host_per_cpu_field_u64_closure_projection() {
        let mk = |val: u64| {
            let mut r = FailureDumpReport::default();
            r.per_cpu_time = vec![PerCpuTimeStats {
                cpu: 1,
                cpustat_system_ns: val,
                ..Default::default()
            }];
            r
        };
        let mk_missing = || {
            let mut r = FailureDumpReport::default();
            r.per_cpu_time = vec![PerCpuTimeStats {
                cpu: 0,
                cpustat_system_ns: 999,
                ..Default::default()
            }];
            r
        };
        let series = SampleSeries::from_drained(
            vec![
                ("a".to_string(), mk(100), None, Some(10u64)),
                ("b".to_string(), mk_missing(), None, Some(20u64)),
                ("c".to_string(), mk(300), None, Some(30u64)),
            ],
            None,
        );
        let host = series.host().expect("non-empty");
        let field = host.per_cpu_field_u64(1, "system_ns_cpu1", |stats| stats.cpustat_system_ns);
        let slots: Vec<_> = field.values_iter().collect();
        assert_eq!(slots.len(), 3);
        assert_eq!(*slots[0].as_ref().expect("cpu 1 captured in sample a"), 100);
        match slots[1] {
            Err(crate::scenario::snapshot::SnapshotError::HostFieldUnavailable { tag, cpu }) => {
                assert_eq!(tag, "b");
                assert_eq!(*cpu, 1);
            }
            other => panic!(
                "cpu 1 absent in sample b MUST surface as HostFieldUnavailable, got {other:?}"
            ),
        }
        assert_eq!(*slots[2].as_ref().expect("cpu 1 captured in sample c"), 300);
    }

    /// `per_cpu_field_u64` on a PLACEHOLDER sample surfaces
    /// [`SnapshotError::PlaceholderSample`] — NOT
    /// `HostFieldUnavailable`. Mirrors the [`SampleSeries::bpf`]
    /// placeholder-gate pattern so temporal-assertion sites route
    /// placeholders through their per-sample skip handling (cleaner
    /// F2 fold-in 2026-05-17).
    #[test]
    fn series_host_per_cpu_field_u64_placeholder_surfaces_placeholder_sample_variant() {
        let mk = |val: u64| {
            let mut r = FailureDumpReport::default();
            r.per_cpu_time = vec![PerCpuTimeStats {
                cpu: 0,
                cpustat_user_ns: val,
                ..Default::default()
            }];
            r
        };
        let placeholder = FailureDumpReport::placeholder("freeze rendezvous timed out");
        let series = SampleSeries::from_drained(
            vec![
                ("real".to_string(), mk(100), None, Some(10u64)),
                ("placeholder".to_string(), placeholder, None, Some(20u64)),
            ],
            None,
        );
        let host = series.host().expect("non-empty");
        let field = host.per_cpu_field_u64(0, "user_ns_cpu0", |s| s.cpustat_user_ns);
        let slots: Vec<_> = field.values_iter().collect();
        assert_eq!(slots.len(), 2);
        assert_eq!(*slots[0].as_ref().expect("real sample Ok"), 100);
        match slots[1] {
            Err(crate::scenario::snapshot::SnapshotError::PlaceholderSample { tag, .. }) => {
                assert_eq!(tag, "placeholder");
            }
            other => panic!(
                "placeholder sample MUST surface as PlaceholderSample (not HostFieldUnavailable), got {other:?}"
            ),
        }
    }

    /// `cpus()` returns an empty Vec on a series where every
    /// sample is a placeholder (rows non-empty, every per_cpu_time
    /// is empty). Pins the all-placeholder edge case at unit-test
    /// granularity (tester T6 fold-in 2026-05-17).
    #[test]
    fn series_host_cpus_empty_when_all_samples_are_placeholders() {
        let series = SampleSeries::from_drained(
            vec![
                (
                    "p0".to_string(),
                    FailureDumpReport::placeholder("t1"),
                    None,
                    Some(10u64),
                ),
                (
                    "p1".to_string(),
                    FailureDumpReport::placeholder("t2"),
                    None,
                    Some(20u64),
                ),
                (
                    "p2".to_string(),
                    FailureDumpReport::placeholder("t3"),
                    None,
                    Some(30u64),
                ),
            ],
            None,
        );
        let host = series.host().expect("rows non-empty");
        assert!(
            host.cpus().is_empty(),
            "all-placeholder series MUST surface cpus() as empty (no per_cpu_time data anywhere)"
        );
    }

    /// Multi-sample × multi-CPU with VARIABLE per-sample coverage
    /// (sample A: cpus 0,1; sample B: cpus 1,2; sample C: cpus 0,2).
    /// Pins the BTreeSet-dedup union from `cpus()` AND per-CPU
    /// filtering in `per_cpu_time_timeline` AND mixed Ok/Err
    /// pattern in `per_cpu_field_u64` simultaneously. Tester T7
    /// fold-in 2026-05-17.
    #[test]
    fn series_host_interleaved_multi_cpu_multi_sample_coverage() {
        let mk = |cpus: &[(u32, u64)]| {
            let mut r = FailureDumpReport::default();
            r.per_cpu_time = cpus
                .iter()
                .map(|(c, v)| PerCpuTimeStats {
                    cpu: *c,
                    cpustat_user_ns: *v,
                    ..Default::default()
                })
                .collect();
            r
        };
        let series = SampleSeries::from_drained(
            vec![
                ("A".to_string(), mk(&[(0, 10), (1, 100)]), None, Some(10u64)),
                ("B".to_string(), mk(&[(1, 200), (2, 300)]), None, Some(20u64)),
                ("C".to_string(), mk(&[(0, 50), (2, 600)]), None, Some(30u64)),
            ],
            None,
        );
        let host = series.host().expect("non-empty");
        // cpus() union: {0, 1, 2} sorted
        assert_eq!(host.cpus(), vec![0, 1, 2]);
        // per_cpu_time_timeline(0): rows from A + C (B has no cpu 0)
        let t0 = host.per_cpu_time_timeline(0);
        assert_eq!(t0.len(), 2);
        assert_eq!(t0[0].0, 10);
        assert_eq!(t0[0].1.cpustat_user_ns, 10);
        assert_eq!(t0[1].0, 30);
        assert_eq!(t0[1].1.cpustat_user_ns, 50);
        // per_cpu_time_timeline(1): rows from A + B (C has no cpu 1)
        let t1 = host.per_cpu_time_timeline(1);
        assert_eq!(t1.len(), 2);
        assert_eq!(t1[0].1.cpustat_user_ns, 100);
        assert_eq!(t1[1].1.cpustat_user_ns, 200);
        // per_cpu_time_timeline(2): rows from B + C (A has no cpu 2)
        let t2 = host.per_cpu_time_timeline(2);
        assert_eq!(t2.len(), 2);
        assert_eq!(t2[0].1.cpustat_user_ns, 300);
        assert_eq!(t2[1].1.cpustat_user_ns, 600);
        // per_cpu_field_u64(1): A=Ok(100), B=Ok(200), C=Err(HostFieldUnavailable cpu=1)
        let field1 = host.per_cpu_field_u64(1, "cpu1_user", |s| s.cpustat_user_ns);
        let slots: Vec<_> = field1.values_iter().collect();
        assert_eq!(slots.len(), 3);
        assert_eq!(*slots[0].as_ref().unwrap(), 100);
        assert_eq!(*slots[1].as_ref().unwrap(), 200);
        match slots[2] {
            Err(crate::scenario::snapshot::SnapshotError::HostFieldUnavailable { tag, cpu }) => {
                assert_eq!(tag, "C");
                assert_eq!(*cpu, 1);
            }
            other => panic!("expected HostFieldUnavailable for C/cpu=1, got {other:?}"),
        }
    }

    /// `cpus()` is sorted ascending (BTreeSet semantic) regardless
    /// of per_cpu_time insertion order. Pins against a regression
    /// that switched BTreeSet → HashSet → Vec without an explicit
    /// sort step. Tester T8 fold-in 2026-05-17.
    #[test]
    fn series_host_cpus_sorted_ascending_independent_of_insertion_order() {
        let mut report = FailureDumpReport::default();
        report.per_cpu_time = vec![
            PerCpuTimeStats {
                cpu: 5,
                ..Default::default()
            },
            PerCpuTimeStats {
                cpu: 1,
                ..Default::default()
            },
            PerCpuTimeStats {
                cpu: 3,
                ..Default::default()
            },
        ];
        let series = SampleSeries::from_drained(
            vec![("s".to_string(), report, None, Some(0u64))],
            None,
        );
        let host = series.host().expect("non-empty");
        assert_eq!(
            host.cpus(),
            vec![1, 3, 5],
            "cpus() MUST return ascending-sorted distinct CPU ids regardless of per_cpu_time insertion order"
        );
    }
}
