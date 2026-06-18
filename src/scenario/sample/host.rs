//! Per-sample, per-CPU host-side timeline projection.
//!
//! The host-capture pipeline (see [`crate::monitor::dump`]) populates
//! each [`FailureDumpReport::per_cpu_time`](crate::monitor::dump::FailureDumpReport::per_cpu_time)
//! with a slice of [`PerCpuTimeStats`] taken at sample time. This
//! module exposes those slices as a borrowed-view timeline
//! ([`HostView`]) keyed by CPU id, with a closure-based projector
//! that emits a [`SeriesField<u64>`] compatible with the temporal-
//! assertion patterns in [`crate::assert::temporal`].
//!
//! Orthogonal to [`super::monitor`]: this view is the per-SAMPLE
//! per-CPU TIMELINE source; the monitor view exposes the per-VM-RUN
//! cross-CPU AGGREGATE. The two never overlap — they draw from
//! different fields on the captured reports
//! (`FailureDumpReport::per_cpu_time` here vs `MonitorReport.summary`
//! for the monitor view).

use crate::assert::temporal::SeriesField;
use crate::monitor::dump::PerCpuTimeStats;

use super::{SampleRow, SampleSeries, build_series_field};

/// Borrowed view over the per-sample per-CPU [`PerCpuTimeStats`] data
/// that the host capture pipeline populates into each
/// [`FailureDumpReport::per_cpu_time`](crate::monitor::dump::FailureDumpReport::per_cpu_time). Returned by
/// [`SampleSeries::host`]; exposes a per-CPU timeline (rows sorted
/// ascending by elapsed-ms, stable on ties) plus a closure-based
/// projector that emits a [`SeriesField<u64>`] compatible with the
/// temporal-assertion patterns in [`crate::assert::temporal`].
///
/// Orthogonal to [`super::MonitorView`]: this view is the per-sample
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
    ///
    /// Inherits the first-match-wins contract for duplicate-cpu
    /// entries from
    /// [`crate::scenario::snapshot::Snapshot::per_cpu_time_at`]:
    /// production walker (`collect_per_cpu_time`) enforces one
    /// entry per cpu per sample, but the lookup leaves the contract
    /// first-match for graceful degradation on a malformed report.
    /// Samples whose `elapsed_ms` is `None` (the bridge recorded no
    /// timestamp) are EXCLUDED: a timestamp-less sample has
    /// no position on a time-ordered axis, and placing it at a
    /// fabricated `0` would corrupt the ascending-by-time contract.
    pub fn per_cpu_time_timeline(&self, cpu: u32) -> Vec<(u64, &'a PerCpuTimeStats)> {
        let mut entries: Vec<(u64, &'a PerCpuTimeStats)> = Vec::new();
        for row in self.rows {
            let Some(elapsed_ms) = row.elapsed_ms else {
                continue;
            };
            if let Some(stats) = row.report.per_cpu_time.iter().find(|c| c.cpu == cpu) {
                entries.push((elapsed_ms, stats));
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
        build_series_field(self.rows, label, |row| {
            // Placeholder reports surface as the dedicated
            // PlaceholderSample variant — matching the series.bpf
            // pattern so temporal-assertion sites route placeholder
            // samples through their per-sample skip handling rather
            // than treating them as cpu-coverage gaps. A strict
            // pattern (always_true / each.at_least) would otherwise
            // FAIL on placeholders instead of skipping; gap-tolerant
            // patterns render the right diagnostic Note.
            if row.report.is_placeholder {
                return Err(
                    crate::scenario::snapshot::SnapshotError::PlaceholderSample {
                        tag: row.tag.clone(),
                        reason: row
                            .report
                            .scx_walker_unavailable
                            .clone()
                            .unwrap_or_else(|| "placeholder report".to_string()),
                    },
                );
            }
            // Inherits the first-match-wins contract from
            // [`crate::scenario::snapshot::Snapshot::per_cpu_time_at`]:
            // production walker (`collect_per_cpu_time` at
            // `crate::monitor::dump`) enforces one entry per cpu,
            // but the closure leaves the contract first-match for
            // graceful degradation on a malformed report.
            match row.report.per_cpu_time.iter().find(|c| c.cpu == cpu) {
                Some(stats) => Ok(project(stats)),
                None => Err(
                    crate::scenario::snapshot::SnapshotError::HostFieldUnavailable {
                        tag: row.tag.clone(),
                        cpu,
                    },
                ),
            }
        })
    }
}

impl SampleSeries {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::dump::FailureDumpReport;

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
        let report = FailureDumpReport {
            per_cpu_time: vec![
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
            ],
            ..Default::default()
        };
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

    /// REGRESSION: a sample whose `elapsed_ms` is `None` (the
    /// bridge recorded no timestamp) is DROPPED from the time-ordered
    /// timeline — it has no position on a time axis and must not be
    /// placed at a fabricated `0` (which would sort first and corrupt the
    /// ascending contract).
    #[test]
    fn series_host_per_cpu_time_timeline_drops_none_elapsed() {
        let mk = |user_ns: u64| FailureDumpReport {
            per_cpu_time: vec![PerCpuTimeStats {
                cpu: 0,
                cpustat_user_ns: user_ns,
                ..Default::default()
            }],
            ..Default::default()
        };
        let series = SampleSeries::from_drained(
            vec![
                ("a".to_string(), mk(10), None, Some(100u64)),
                // No recorded timestamp: must be dropped from the timeline.
                ("b".to_string(), mk(20), None, None),
                ("c".to_string(), mk(30), None, Some(300u64)),
            ],
            None,
        );
        let host = series.host().expect("non-empty series");
        let t0 = host.per_cpu_time_timeline(0);
        assert_eq!(
            t0.len(),
            2,
            "the None-elapsed row must be dropped, not placed at a fabricated 0",
        );
        assert_eq!(t0[0].0, 100);
        assert_eq!(t0[0].1.cpustat_user_ns, 10);
        assert_eq!(t0[1].0, 300);
        assert_eq!(t0[1].1.cpustat_user_ns, 30);
    }

    /// Multi-sample series with NON-monotonic elapsed_ms:
    /// `per_cpu_time_timeline` returns rows sorted ascending by
    /// elapsed_ms; ties retain insertion order (stable sort).
    /// Pins the sort contract against drift to unstable sort or
    /// reverse order.
    #[test]
    fn series_host_per_cpu_time_timeline_sorts_by_elapsed_ms_stable() {
        let mk = |val: u64| FailureDumpReport {
            per_cpu_time: vec![PerCpuTimeStats {
                cpu: 0,
                cpustat_user_ns: val,
                ..Default::default()
            }],
            ..Default::default()
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
        let mk_real = |val: u64| FailureDumpReport {
            per_cpu_time: vec![PerCpuTimeStats {
                cpu: 0,
                cpustat_user_ns: val,
                ..Default::default()
            }],
            ..Default::default()
        };
        let placeholder = FailureDumpReport::placeholder("freeze rendezvous timed out");
        let series = SampleSeries::from_drained(
            vec![
                ("real_pre".to_string(), mk_real(10), None, Some(10u64)),
                (
                    "placeholder_mid".to_string(),
                    placeholder,
                    None,
                    Some(20u64),
                ),
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
        let mk = |val: u64| FailureDumpReport {
            per_cpu_time: vec![PerCpuTimeStats {
                cpu: 1,
                cpustat_system_ns: val,
                ..Default::default()
            }],
            ..Default::default()
        };
        let mk_missing = || FailureDumpReport {
            per_cpu_time: vec![PerCpuTimeStats {
                cpu: 0,
                cpustat_system_ns: 999,
                ..Default::default()
            }],
            ..Default::default()
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
    /// placeholders through their per-sample skip handling.
    #[test]
    fn series_host_per_cpu_field_u64_placeholder_surfaces_placeholder_sample_variant() {
        let mk = |val: u64| FailureDumpReport {
            per_cpu_time: vec![PerCpuTimeStats {
                cpu: 0,
                cpustat_user_ns: val,
                ..Default::default()
            }],
            ..Default::default()
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
    /// granularity.
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
    /// pattern in `per_cpu_field_u64` simultaneously.
    #[test]
    fn series_host_interleaved_multi_cpu_multi_sample_coverage() {
        let mk = |cpus: &[(u32, u64)]| FailureDumpReport {
            per_cpu_time: cpus
                .iter()
                .map(|(c, v)| PerCpuTimeStats {
                    cpu: *c,
                    cpustat_user_ns: *v,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        let series = SampleSeries::from_drained(
            vec![
                ("A".to_string(), mk(&[(0, 10), (1, 100)]), None, Some(10u64)),
                (
                    "B".to_string(),
                    mk(&[(1, 200), (2, 300)]),
                    None,
                    Some(20u64),
                ),
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
    /// sort step.
    #[test]
    fn series_host_cpus_sorted_ascending_independent_of_insertion_order() {
        let report = FailureDumpReport {
            per_cpu_time: vec![
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
            ],
            ..Default::default()
        };
        let series =
            SampleSeries::from_drained(vec![("s".to_string(), report, None, Some(0u64))], None);
        let host = series.host().expect("non-empty");
        assert_eq!(
            host.cpus(),
            vec![1, 3, 5],
            "cpus() MUST return ascending-sorted distinct CPU ids regardless of per_cpu_time insertion order"
        );
    }

    /// Duplicate-cpu first-match-wins contract. The production
    /// walker (`collect_per_cpu_time`) enforces one entry per cpu
    /// per sample, but `HostView::per_cpu_time_timeline` and
    /// `HostView::per_cpu_field_u64` both use `iter().find(|c|
    /// c.cpu == cpu)` which returns the FIRST match — silently
    /// dropping subsequent entries for the same cpu. Pins the
    /// first-match-wins contract so a regression to `last_match`,
    /// panic-on-dup, or any other handling surfaces here.
    #[test]
    fn series_host_per_cpu_time_timeline_first_match_wins_on_duplicate_cpu() {
        let report = FailureDumpReport {
            per_cpu_time: vec![
                PerCpuTimeStats {
                    cpu: 0,
                    cpustat_user_ns: 100,
                    ..Default::default()
                },
                PerCpuTimeStats {
                    cpu: 0,
                    cpustat_user_ns: 200,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let series =
            SampleSeries::from_drained(vec![("s".to_string(), report, None, Some(0u64))], None);
        let host = series.host().expect("non-empty");
        let timeline = host.per_cpu_time_timeline(0);
        assert_eq!(timeline.len(), 1, "first-match-wins: timeline pushes once");
        assert_eq!(
            timeline[0].1.cpustat_user_ns, 100,
            "first-match-wins: timeline returns FIRST entry (100), not second (200)"
        );
        let field = host.per_cpu_field_u64(0, "user_ns", |s| s.cpustat_user_ns);
        let slots: Vec<_> = field.values_iter().collect();
        assert_eq!(slots.len(), 1);
        assert_eq!(
            *slots[0].as_ref().expect("Ok(first match value)"),
            100,
            "first-match-wins: per_cpu_field_u64 also returns FIRST entry"
        );
    }

    /// elapsed_ms plumbing through `per_cpu_field_u64` is verified
    /// via `iter_full()` — pins both the elapsed_ms VALUE per slot
    /// AND the tag string per slot against value-corruption
    /// regressions (e.g. `elapsed.push(0)` instead of
    /// `elapsed.push(row.elapsed_ms)`, or tag/elapsed vec swap).
    #[test]
    fn series_host_per_cpu_field_u64_iter_full_threads_tag_and_elapsed_correctly() {
        let mk = |val: u64| FailureDumpReport {
            per_cpu_time: vec![PerCpuTimeStats {
                cpu: 0,
                cpustat_user_ns: val,
                ..Default::default()
            }],
            ..Default::default()
        };
        let series = SampleSeries::from_drained(
            vec![
                ("alpha".to_string(), mk(10), None, Some(100u64)),
                ("beta".to_string(), mk(20), None, Some(200u64)),
                ("gamma".to_string(), mk(30), None, Some(300u64)),
            ],
            None,
        );
        let host = series.host().expect("non-empty");
        let field = host.per_cpu_field_u64(0, "user_ns", |s| s.cpustat_user_ns);
        let full: Vec<_> = field.iter_full().collect();
        assert_eq!(full.len(), 3);
        assert_eq!(full[0].0, "alpha");
        assert_eq!(full[0].1, Some(100));
        assert_eq!(*full[0].2.as_ref().unwrap(), 10);
        assert_eq!(full[1].0, "beta");
        assert_eq!(full[1].1, Some(200));
        assert_eq!(*full[1].2.as_ref().unwrap(), 20);
        assert_eq!(full[2].0, "gamma");
        assert_eq!(full[2].1, Some(300));
        assert_eq!(*full[2].2.as_ref().unwrap(), 30);
    }

    /// Non-placeholder sample with EMPTY per_cpu_time (real capture
    /// succeeded, BPF axis populated, but CpuTimeCapture didn't
    /// run or returned an empty Vec) MUST surface as
    /// `HostFieldUnavailable`, NOT `PlaceholderSample`. Pins the
    /// `is_placeholder` gate predicate against drift to
    /// `per_cpu_time.is_empty() || is_placeholder` (which would
    /// mis-classify "real but no data" as a placeholder).
    #[test]
    fn series_host_per_cpu_field_u64_non_placeholder_empty_per_cpu_time_surfaces_host_field_unavailable()
     {
        // FailureDumpReport::default() has is_placeholder=false +
        // empty per_cpu_time.
        let report = FailureDumpReport::default();
        let series = SampleSeries::from_drained(
            vec![("real_no_cpu_data".to_string(), report, None, Some(10u64))],
            None,
        );
        let host = series.host().expect("non-empty");
        let field = host.per_cpu_field_u64(0, "user_ns", |s| s.cpustat_user_ns);
        let slots: Vec<_> = field.values_iter().collect();
        assert_eq!(slots.len(), 1);
        match slots[0] {
            Err(crate::scenario::snapshot::SnapshotError::HostFieldUnavailable { tag, cpu }) => {
                assert_eq!(tag, "real_no_cpu_data");
                assert_eq!(*cpu, 0);
            }
            Err(crate::scenario::snapshot::SnapshotError::PlaceholderSample { .. }) => {
                panic!(
                    "non-placeholder sample with empty per_cpu_time MUST surface as HostFieldUnavailable, NOT PlaceholderSample (regression: empty-per_cpu_time gating as placeholder)"
                )
            }
            other => panic!("expected HostFieldUnavailable, got {other:?}"),
        }
    }
}
