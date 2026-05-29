//! Stats-JSON-axis projection for [`SampleSeries`].
//!
//! Each [`Sample`](super::Sample) carries an optional `scx_stats` JSON
//! value captured from a scx_stats request issued just BEFORE the
//! freeze rendezvous. This module exposes the closure-based
//! [`SampleSeries::stats`] projection (manual path access via
//! [`StatsValue`]) and the auto-discovering
//! [`SampleSeries::stats_path`] → [`StatsPathProjector`] pair that
//! walks a stats sub-tree, enumerates object keys, and projects each
//! as `SeriesField<u64>` / `SeriesField<i64>` / `SeriesField<f64>`.
//!
//! Orthogonal to [`super::bpf`]: the stats axis sources its values
//! from the userspace scheduler's `scx_stats` JSON; the BPF axis
//! sources from kernel-side BPF state. Tests typically use both.

use crate::assert::temporal::SeriesField;
use crate::scenario::snapshot::{JsonField, SnapshotResult, stats_path};

use super::{SampleSeries, build_series_field};

impl SampleSeries {
    /// Project the series along the stats axis. The closure
    /// receives each sample's stats JSON (when present) and
    /// returns a [`SnapshotResult<T>`]. Samples whose `stats` is
    /// `Err(reason)` get a `Err(MissingStats { reason })` slot —
    /// temporal assertions surface that as a per-sample
    /// missing-stats failure rather than vacuously skipping it,
    /// so a coverage gap is never silent and the operator sees
    /// the *why* (no scheduler binary configured, relay timed
    /// out, scheduler returned errno, etc.).
    ///
    /// `label` is owned (`impl Into<String>`) and matches the
    /// shape of [`Self::bpf`] — pass a literal or a runtime-built
    /// `String` for auto-discovered keys.
    pub fn stats<T, F>(&self, label: impl Into<String>, project: F) -> SeriesField<T>
    where
        F: Fn(StatsValue<'_>) -> SnapshotResult<T>,
    {
        build_series_field(&self.rows, label, |row| match row.stats.as_ref() {
            Ok(v) => project(StatsValue { value: v }),
            Err(reason) => Err(crate::scenario::snapshot::SnapshotError::MissingStats {
                tag: row.tag.clone(),
                reason: reason.clone(),
            }),
        })
    }

    /// Project the live scheduler's stats JSON field at `path` as
    /// `u64`. Per-row equivalent of `series.stats(label, |s|
    /// s.get(path).as_u64())` with the boilerplate elided. Mirrors
    /// [`Self::bpf_live_u64`] for naming parity across axes.
    ///
    /// **Why "live" applies — per-request freshness, not a buffer.**
    /// Each periodic snapshot issues a FRESH `scx_stats` request
    /// just before the freeze rendezvous fires; the response in
    /// `row.stats` came from whichever scheduler was alive at
    /// request-issue time. There is no relay buffer of "the last
    /// stats we saw" — a stale-pre-swap response cannot land in
    /// a post-swap sample. After `Op::ReplaceScheduler` the host
    /// reconnects to the new scheduler's `scx_stats` endpoint
    /// before the next periodic boundary issues its request, so
    /// post-swap samples carry the new scheduler's data. The
    /// `_live` suffix matches the BPF axis naming for cross-axis
    /// vocabulary consistency AND describes the actual freshness
    /// guarantee — same semantic across both axes.
    pub fn stats_live_u64(&self, path: &str) -> SeriesField<u64> {
        let path_owned = path.to_string();
        self.stats(path_owned.clone(), move |s| s.get(&path_owned).as_u64())
    }

    /// Sibling of [`Self::stats_live_u64`] projecting as `i64`.
    pub fn stats_live_i64(&self, path: &str) -> SeriesField<i64> {
        let path_owned = path.to_string();
        self.stats(path_owned.clone(), move |s| s.get(&path_owned).as_i64())
    }

    /// Sibling of [`Self::stats_live_u64`] projecting as `f64`.
    pub fn stats_live_f64(&self, path: &str) -> SeriesField<f64> {
        let path_owned = path.to_string();
        self.stats(path_owned.clone(), move |s| s.get(&path_owned).as_f64())
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
}

/// Newtype carrier handed to the [`SampleSeries::stats`] closure.
/// Wraps a borrowed [`serde_json::Value`] and exposes [`Self::get`]
/// as a thin facade over [`stats_path`] so the closure body reads
/// `s.get("layers.batch.util").as_f64()` without an explicit
/// import. The `.get(path)` name mirrors
/// [`crate::scenario::snapshot::SnapshotField::get`] and
/// [`crate::scenario::snapshot::JsonField::get`] so test authors
/// see one navigator vocabulary across every accessor surface.
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
    pub fn get(&self, path: &str) -> JsonField<'a> {
        stats_path(self.value, path)
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
            .stats(key, move |sv| sv.get(&full_path).as_u64())
    }

    /// Project a JSON key under the resolved path as `i64`.
    pub fn field_i64(&self, key: &str) -> SeriesField<i64> {
        let full_path = join_paths(&self.path, key);
        self.series
            .stats(key, move |sv| sv.get(&full_path).as_i64())
    }

    /// Project a JSON key under the resolved path as `f64`.
    pub fn field_f64(&self, key: &str) -> SeriesField<f64> {
        let full_path = join_paths(&self.path, key);
        self.series
            .stats(key, move |sv| sv.get(&full_path).as_f64())
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
            Ok(s) => s,
            Err(_) => return Vec::new(),
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

    #[test]
    fn stats_projection_handles_missing_stats_as_error() {
        use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
        let drained = vec![
            DrainedSnapshotEntry {
                tag: "periodic_000".to_string(),
                report: synthetic_report(10),
                stats: Ok(synthetic_stats(50.0)),
                elapsed_ms: Some(100),
                step_index: None,
            },
            DrainedSnapshotEntry {
                tag: "periodic_001".to_string(),
                report: synthetic_report(20),
                stats: Err(MissingStatsReason::NoSchedulerBinary),
                elapsed_ms: Some(200),
                step_index: None,
            },
        ];
        let series = SampleSeries::from_drained_typed(drained, None);
        let field: SeriesField<f64> = series.stats("busy", |s| s.get("busy").as_f64());
        let outcomes: Vec<SnapshotResult<f64>> = field.values_iter().cloned().collect();
        assert_eq!(outcomes.len(), 2);
        assert_eq!(
            outcomes[0].as_ref().copied(),
            Ok(50.0),
            "sample with stats present must project the `busy` field verbatim"
        );
        match &outcomes[1] {
            Err(crate::scenario::snapshot::SnapshotError::MissingStats { tag, reason }) => {
                assert_eq!(
                    tag, "periodic_001",
                    "MissingStats tag must identify the sample whose stats slot was Err"
                );
                assert_eq!(
                    reason,
                    &MissingStatsReason::NoSchedulerBinary,
                    "MissingStats reason must propagate the carried MissingStatsReason verbatim"
                );
            }
            other => panic!(
                "sample with stats=Err must surface SnapshotError::MissingStats, got {other:?}"
            ),
        }
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
            "String key must be excluded — every u64 projection errors: {names:?}",
        );
    }

    /// Mirror of the u64 test for `f64_fields`. `busy`, `count`,
    /// and `ratio` all coerce to f64; only `name` errors. Pins the
    /// "at least one Ok" filter for the f64 axis distinctly from
    /// the u64 axis.
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
            "Number(integer) coerces to f64 — must be kept: {names:?}",
        );
        assert!(
            names.contains(&"count"),
            "Number(integer) coerces to f64 — must be kept: {names:?}",
        );
        assert!(
            names.contains(&"ratio"),
            "Number(non-integer float) coerces to f64 — must be kept: {names:?}",
        );
        assert!(
            !names.contains(&"name"),
            "String key must be excluded — every f64 projection errors: {names:?}",
        );
    }

    /// Empty series — no rows to discover JSON keys from, so
    /// `key_names()` returns an empty vec and both auto-projectors
    /// yield empty results without panicking. Pins the "no first
    /// row" branch in `StatsPathProjector::key_names`.
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
}
