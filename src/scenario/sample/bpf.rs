//! BPF-axis projection for [`SampleSeries`].
//!
//! Each [`Sample`](super::Sample) carries a frozen [`Snapshot`] over
//! BPF program state captured at the freeze rendezvous. This module
//! exposes the closure-based [`SampleSeries::bpf`] projection (manual
//! field access via the Snapshot accessor surface) and the auto-
//! discovering [`SampleSeries::bpf_map`] → [`BpfMapProjector`] pair
//! that enumerates a map's struct members and projects each as
//! `SeriesField<u64>` / `SeriesField<i64>` / `SeriesField<f64>`.
//!
//! Orthogonal to [`super::stats`]: the BPF axis sources its values
//! from the kernel-side BPF state (counters, ringbuf items, struct
//! members); the stats axis sources from the userspace scheduler's
//! `scx_stats` JSON. Tests typically use both — BPF for low-level
//! state, stats for scheduler-author-defined metrics.

use crate::assert::temporal::SeriesField;
use crate::scenario::snapshot::{Snapshot, SnapshotField, SnapshotResult};

use super::{SampleSeries, build_series_field};

impl SampleSeries {
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
        build_series_field(&self.rows, label, |row| {
            // Placeholder reports carry no real BPF state — the
            // freeze rendezvous timed out (or the capture pipeline
            // otherwise failed). Surface a dedicated PlaceholderSample
            // error variant BEFORE invoking the projection closure
            // so the temporal-assertion patterns can branch on
            // "placeholder, skip" distinctly from "field missing,
            // skip" when rendering the verdict's skip-Note.
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
            let snap = Snapshot::new(&row.report);
            project(&snap)
        })
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
}
