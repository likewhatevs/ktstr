//! Per-VM-run host monitor projection.
//!
//! The host-side monitor (see [`crate::monitor`]) aggregates sampling
//! observations across the VM run and produces a [`MonitorReport`]
//! exposing summary statistics and SCX event-counter deltas. This
//! module wraps that report in borrowed views ([`MonitorView`] +
//! [`ScxEventsView`]) returned from [`SampleSeries::monitor`].
//!
//! Orthogonal to [`super::host`]: monitor exposes the per-VM-run
//! cross-CPU AGGREGATE; the host view exposes the per-SAMPLE per-CPU
//! TIMELINE. The two draw from different fields on the captured
//! reports (`MonitorReport.summary` here vs
//! `FailureDumpReport::per_cpu_time` for the host view) and never
//! overlap.

use crate::monitor::{MonitorReport, MonitorSummary, ScxEventDeltas};

use super::SampleSeries;

/// Borrowed view over a per-VM-run `MonitorReport`. Returned by
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
    /// averages, local DSQ depth, stuck-CPU count, and
    /// optional schedstat / prog-stats deltas. See
    /// `MonitorSummary` for the full field set.
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

    /// Borrowed per-tick monitor samples. Each
    /// `crate::monitor::MonitorSample` is one host-side
    /// observation of the guest's per-CPU runqueue state
    /// (`nr_running`, `local_dsq_depth`, `rq_clock`, optional
    /// event counters). The monitor thread captures these on a
    /// fixed cadence independent of the snapshot bridge's
    /// freeze-rendezvous captures; samples carry their own
    /// `elapsed_ms` timestamp for windowing.
    ///
    /// Empty when the monitor ran but produced no samples (very
    /// short run, monitor thread exited early). The slot is
    /// always present — `MonitorView` itself only exists when a
    /// `MonitorReport` was attached at series construction.
    ///
    /// Live caller: [`crate::assert::build_phase_buckets`] windows
    /// these samples per phase to compute metrics like
    /// `avg_imbalance_ratio` that need per-CPU `rq.nr_running`
    /// (full-class count), which the bridge-captured
    /// [`crate::scenario::snapshot::Snapshot`] does NOT expose
    /// (Snapshot carries only `scx_rq.nr_running`, the SCX-only
    /// subset). The two data axes are complementary: Snapshot for
    /// frozen BPF state at capture instants, MonitorSample for
    /// per-tick observations across the whole window.
    pub fn samples(&self) -> &'a [crate::monitor::MonitorSample] {
        &self.report.samples
    }

    /// vCPU-preemption exemption window (ns) for stall detection,
    /// derived from the guest kernel's `CONFIG_HZ` at run time. `0`
    /// means "derive from a default" — callers folding per-phase
    /// stall counts pass it through to
    /// `crate::timeline::compute_metrics` so the per-phase predicate
    /// matches the run-level `MonitorSummary::stuck_count` one.
    pub fn preemption_threshold_ns(&self) -> u64 {
        self.report.preemption_threshold_ns
    }
}

/// Default curated subset of [`ScxEventsView::total_pairs`] counter
/// names that signal genuine scheduler-class errors when non-zero.
/// Used to filter the full 14-entry total slice down to the entries
/// that callers conventionally bound at zero with
/// [`crate::assert::assert_scx_events_clean`].
///
/// Membership is the documented intersection of the kernel-side
/// `SCX_EV_*` counters whose non-zero firing is exclusively
/// pathological (skipped enqueue paths, repeated re-enqueue cycles,
/// owner-mismatched inserts) — the `bypass_*`,
/// `dispatch_keep_last`, `refill_slice_dfl` counters that
/// legitimately fire on healthy schedulers are deliberately
/// excluded. Different test scenarios may consider different
/// counters error-class; the projector exposes the full slice via
/// [`ScxEventsView::total_pairs`] so callers can override this
/// default by filtering on their own set.
pub const ERROR_CLASS_NAMES: &[&str] = &[
    "enq_skip_exiting",
    "enq_skip_migration_disabled",
    "reenq_immed",
    "reenq_local_repeat",
    "insert_not_owned",
];

/// Borrowed view over the `ScxEventDeltas` aggregated across the
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
    /// # use ktstr::scenario::sample::ERROR_CLASS_NAMES;
    /// # use ktstr::assert::assert_scx_events_clean;
    /// # fn example(series: &SampleSeries) {
    /// if let Some(view) = series.monitor()
    ///     && let Some(events) = view.scx_events()
    /// {
    ///     let pairs = events.total_pairs();
    ///     let error_only: Vec<(&str, i64)> = pairs
    ///         .into_iter()
    ///         .filter(|(name, _)| ERROR_CLASS_NAMES.contains(name))
    ///         .collect();
    ///     assert!(assert_scx_events_clean(&error_only, None).is_pass());
    /// }
    /// # }
    /// ```
    pub fn total_pairs(&self) -> Vec<(&'static str, i64)> {
        vec![
            ("select_cpu_fallback", self.deltas.total_fallback),
            (
                "select_cpu_fallback_max_burst",
                self.deltas.max_fallback_burst,
            ),
            (
                "dispatch_local_dsq_offline",
                self.deltas.total_dispatch_offline,
            ),
            ("dispatch_keep_last", self.deltas.total_dispatch_keep_last),
            ("enq_skip_exiting", self.deltas.total_enq_skip_exiting),
            (
                "enq_skip_migration_disabled",
                self.deltas.total_enq_skip_migration_disabled,
            ),
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

impl SampleSeries {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `series.monitor()` returns `None` when no monitor was
    /// supplied (host-only tests, early VM failure, or
    /// `from_drained` was called with `None` monitor). Pins the
    /// Option chain — callers reaching for monitor metrics via
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
        report.summary.max_imbalance_ratio = 2.5;
        let series = SampleSeries::from_drained(vec![], Some(report));
        let view = series.monitor().expect("monitor must be Some");
        let summary = view.summary();
        assert_eq!(summary.total_samples, 42);
        assert_eq!(summary.max_imbalance_ratio, 2.5);
    }

    /// `view.scx_events()` returns `None` when `event_deltas` is
    /// `None` on the underlying summary (kernel without event
    /// counters, monitoring window too short). Inner-Option chain
    /// must NOT collapse to default-zero pairs — silently masking
    /// the missing-data condition would be a silent-loss path.
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
    /// [`ScxEventsView::total_pairs`]: when a non-error-class
    /// counter (`total_bypass_dispatch` here) legitimately fires
    /// alongside an error-class counter at zero,
    /// `assert_scx_events_clean(pairs, None)` against the FULL
    /// 14-entry slice MUST FAIL (because bypass_dispatch > 0); the
    /// CURATED subset of error-class counters MUST PASS (because
    /// every error counter is zero). A future regression that
    /// silently dropped a counter from `total_pairs` or accidentally
    /// curated by the projector would break one of these two
    /// assertions.
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
            !r_full.is_pass(),
            "strict-zero against full 14-entry slice MUST fail when non-error-class counters legitimately fire — pins the STRICTNESS WARNING design contract"
        );
        // Curated error-class subset: MUST pass because every
        // error-class counter is zero (we only populated bypass_*
        // and dispatch_keep_last, neither of which is error class).
        let error_only: Vec<(&str, i64)> = pairs
            .into_iter()
            .filter(|(name, _)| ERROR_CLASS_NAMES.contains(name))
            .collect();
        let r_curated = assert_scx_events_clean(&error_only, None);
        assert!(
            r_curated.is_pass(),
            "curated error-class subset MUST pass when every error counter is zero — pins the curate-then-assert documented workaround"
        );
    }
}
