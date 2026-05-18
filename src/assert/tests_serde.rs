//! Wire-format tests for `CgroupStats`, `ScenarioStats`, and
//! `AssertResult`: round-trip + strict-schema rejection of any
//! omitted required field. No `#[serde(default)]` shims — sidecar
//! data is disposable per the project pre-1.0 stance, so every
//! field is required on the wire.

use super::*;

#[test]
fn scenario_stats_serde_roundtrip() {
    let s = ScenarioStats {
        cgroups: vec![CgroupStats {
            num_workers: 4,
            num_cpus: 2,
            avg_off_cpu_pct: 50.0,
            min_off_cpu_pct: 40.0,
            max_off_cpu_pct: 60.0,
            spread: 20.0,
            max_gap_ms: 150,
            max_gap_cpu: 3,
            total_migrations: 10,
            ..Default::default()
        }],
        total_workers: 4,
        total_cpus: 2,
        total_migrations: 10,
        worst_spread: 20.0,
        worst_gap_ms: 150,
        worst_gap_cpu: 3,
        ..Default::default()
    };
    let json = serde_json::to_string(&s).unwrap();
    let s2: ScenarioStats = serde_json::from_str(&json).unwrap();
    assert_eq!(s.total_workers, s2.total_workers);
    assert_eq!(s.worst_gap_ms, s2.worst_gap_ms);
    assert_eq!(s.cgroups.len(), s2.cgroups.len());
    assert_eq!(s.cgroups[0].num_workers, s2.cgroups[0].num_workers);
}

#[test]
fn assert_result_serde_roundtrip() {
    let r = AssertResult {
        outcomes: vec![Outcome::Fail(AssertDetail::new(DetailKind::Other, "test"))],
        passes: vec![],
        stats: Default::default(),
        measurements: std::collections::BTreeMap::new(),
        info_notes: vec![InfoNote::new("ctx=42")],
    };
    let json = serde_json::to_string(&r).unwrap();
    let r2: AssertResult = serde_json::from_str(&json).unwrap();
    assert_eq!(r.is_pass(), r2.is_pass());
    assert_eq!(r.is_fail(), r2.is_fail());
    let r_details: Vec<&AssertDetail> = r.failure_details().collect();
    let r2_details: Vec<&AssertDetail> = r2.failure_details().collect();
    assert_eq!(r_details.len(), r2_details.len());
    assert_eq!(r_details[0].message, r2_details[0].message);
    assert_eq!(r.passes, r2.passes);
    assert_eq!(r.info_notes.len(), r2.info_notes.len());
    assert_eq!(r.info_notes[0].message, r2.info_notes[0].message);
}

/// Postcard roundtrip pins the wire format the freeze-coord drain
/// reads via
/// [`crate::test_support::output::parse_assert_result_from_drain`]
/// (MSG_TYPE_TEST_RESULT TLV). A regression that re-adds
/// `#[serde(tag = "kind", content = "data")]` to `Outcome` (or any
/// nested type inside `AssertResult`) breaks postcard's
/// externally-tagged enum decoder silently — caught here at test
/// time, not at runtime by a failing TLV drain that surfaces as
/// `ERR_NO_TEST_FUNCTION_OUTPUT`.
#[test]
fn assert_result_postcard_roundtrip() {
    // Populate every NoteValue variant in measurements — guards
    // against a future regression that re-adds `#[serde(untagged)]`
    // to NoteValue (postcard cannot decode untagged enums under the
    // same self-describing-format constraint that drove the Outcome
    // tagging choice). If the attr slips back in, postcard decode
    // here fails with `WontImplement` at test time rather than
    // silently dropping measurement data on the wire.
    let mut measurements = std::collections::BTreeMap::new();
    measurements.insert("pid".to_string(), NoteValue::Int(-1));
    measurements.insert("bytes".to_string(), NoteValue::Uint(4096));
    measurements.insert("rate".to_string(), NoteValue::Float(3.14));
    measurements.insert("ok".to_string(), NoteValue::Bool(true));
    measurements.insert("label".to_string(), NoteValue::Text("benchmark".to_string()));
    let r = AssertResult {
        outcomes: vec![
            Outcome::Fail(AssertDetail::new(DetailKind::Other, "fail msg")),
            Outcome::Skip(AssertDetail::new(DetailKind::Skip, "skip msg")),
            Outcome::Pass,
        ],
        passes: vec![],
        stats: Default::default(),
        measurements,
        info_notes: vec![InfoNote::new("ctx=42")],
    };
    let bytes = postcard::to_allocvec(&r).expect("postcard encode");
    let r2: AssertResult = postcard::from_bytes(&bytes).expect("postcard decode");
    assert_eq!(r.is_fail(), r2.is_fail());
    assert_eq!(r.is_skip(), r2.is_skip());
    assert_eq!(r.outcomes.len(), r2.outcomes.len());
    let r_fails: Vec<_> = r.failure_details().collect();
    let r2_fails: Vec<_> = r2.failure_details().collect();
    assert_eq!(r_fails.len(), r2_fails.len());
    assert_eq!(r_fails[0].message, r2_fails[0].message);
    let r_skips: Vec<_> = r.skip_reasons().collect();
    let r2_skips: Vec<_> = r2.skip_reasons().collect();
    assert_eq!(r_skips.len(), r2_skips.len());
    assert_eq!(r_skips[0].message, r2_skips[0].message);
    assert_eq!(r.info_notes.len(), r2.info_notes.len());
    assert_eq!(r.info_notes[0].message, r2.info_notes[0].message);
    // Verify every NoteValue variant roundtripped — guards against
    // a future `#[serde(untagged)]` regression on NoteValue (or any
    // nested measurement type) that postcard can't decode.
    assert_eq!(r.measurements.len(), r2.measurements.len());
    assert_eq!(r2.measurements.get("pid"), Some(&NoteValue::Int(-1)));
    assert_eq!(r2.measurements.get("bytes"), Some(&NoteValue::Uint(4096)));
    if let Some(NoteValue::Float(f)) = r2.measurements.get("rate") {
        assert!((f - 3.14).abs() < 1e-9);
    } else {
        panic!("rate must decode to NoteValue::Float, got {:?}", r2.measurements.get("rate"));
    }
    assert_eq!(r2.measurements.get("ok"), Some(&NoteValue::Bool(true)));
    assert_eq!(r2.measurements.get("label"), Some(&NoteValue::Text("benchmark".to_string())));
}

/// Strict-schema rejection sibling for `CgroupStats`. The
/// sidecar wire format persists one
/// [`CgroupStats`](crate::assert::CgroupStats) per entry inside
/// the [`ScenarioStats::cgroups`] vec, so the same schema-
/// symmetry invariant that `ScenarioStats` enforces applies here
/// one level deep. A regression that softened a required field
/// on `CgroupStats` alone would slip past the sibling
/// `ScenarioStats` test.
#[test]
fn cgroup_stats_missing_required_field_rejected_by_deserialize() {
    const REQUIRED_FIELDS: &[&str] = &[
        "num_workers",
        "num_cpus",
        "avg_off_cpu_pct",
        "min_off_cpu_pct",
        "max_off_cpu_pct",
        "spread",
        "max_gap_ms",
        "max_gap_cpu",
        "total_migrations",
        "migration_ratio",
        "p99_wake_latency_us",
        "median_wake_latency_us",
        "wake_latency_cv",
        "total_iterations",
        "mean_run_delay_us",
        "worst_run_delay_us",
        "page_locality",
        "cross_node_migration_ratio",
        "ext_metrics",
    ];
    // `wake_latency_tail_ratio` and `iterations_per_worker` are
    // method-only on CgroupStats and DO NOT appear in the JSON
    // wire format; they are recomputed on read from p99/median
    // and total_iterations/num_workers respectively.

    let cg = CgroupStats::default();
    let full = match serde_json::to_value(&cg).unwrap() {
        serde_json::Value::Object(m) => m,
        other => panic!("expected object, got {other:?}"),
    };

    for field in REQUIRED_FIELDS {
        let mut obj = full.clone();
        assert!(
            obj.remove(*field).is_some(),
            "CgroupStats must emit `{field}` for its rejection \
             case to be meaningful — the field list in this test \
             has drifted from the struct definition",
        );
        let json = serde_json::Value::Object(obj).to_string();
        let err = serde_json::from_str::<CgroupStats>(&json)
            .err()
            .unwrap_or_else(|| {
                panic!("deserialize must reject CgroupStats with `{field}` removed, but succeeded",)
            });
        let msg = format!("{err}");
        assert!(
            msg.contains(field),
            "missing-field error for `{field}` must name the field; got: {msg}",
        );
    }
}

/// Strict-schema rejection: a `ScenarioStats` JSON with a
/// required scalar field omitted (here: `total_workers`) must
/// fail deserialization. `ScenarioStats` carries `Default` for
/// struct construction ergonomics, but that does NOT imply
/// `#[serde(default)]` on each field — and the sidecar schema
/// policy requires serialize/deserialize symmetry. A regression
/// that added `#[serde(default)]` to a scalar field (e.g. to
/// soften a schema migration) would make the `from_str` call
/// below succeed silently, defaulting to 0 without notifying the
/// consumer that the producer omitted data.
#[test]
fn scenario_stats_missing_required_scalar_rejected_by_deserialize() {
    // Table-driven expansion covering EVERY required scalar field
    // instead of a single `total_workers` sentinel. Each removal
    // must produce a missing-field error naming the removed
    // field. The loop forces a pass-or-fail result per field, so
    // a regression that softens just one field (e.g. adds
    // `#[serde(default)]` to `worst_gap_cpu` alone) trips this
    // test with a field-level assertion message — the old single-
    // sentinel form would have passed silently on any field
    // other than `total_workers`.
    const REQUIRED_FIELDS: &[&str] = &[
        "cgroups",
        "total_workers",
        "total_cpus",
        "total_migrations",
        "worst_spread",
        "worst_gap_ms",
        "worst_gap_cpu",
        "worst_migration_ratio",
        "worst_p99_wake_latency_us",
        "worst_median_wake_latency_us",
        "worst_wake_latency_cv",
        "total_iterations",
        "worst_mean_run_delay_us",
        "worst_run_delay_us",
        "worst_page_locality",
        "worst_cross_node_migration_ratio",
        "worst_wake_latency_tail_ratio",
        "worst_iterations_per_worker",
        "ext_metrics",
    ];

    let s = ScenarioStats::default();
    let full = match serde_json::to_value(&s).unwrap() {
        serde_json::Value::Object(m) => m,
        other => panic!("expected object, got {other:?}"),
    };

    for field in REQUIRED_FIELDS {
        let mut obj = full.clone();
        assert!(
            obj.remove(*field).is_some(),
            "ScenarioStats must emit `{field}` for its rejection case to be meaningful — \
             the field list in this test has drifted from the struct definition",
        );
        let json = serde_json::Value::Object(obj).to_string();
        let err = serde_json::from_str::<ScenarioStats>(&json)
            .err()
            .unwrap_or_else(|| {
                panic!(
                    "deserialize must reject ScenarioStats with `{field}` removed, but succeeded",
                )
            });
        let msg = format!("{err}");
        assert!(
            msg.contains(field),
            "missing-field error for `{field}` must name the field; got: {msg}",
        );
    }
}

/// Strict-schema rejection: an `AssertResult` JSON with a
/// required field omitted (here: `passed`) must fail
/// deserialization. `AssertResult` has NO `Default` derive and no
/// `#[serde(default)]` — every field is required on the wire.
/// Pinned so a regression that softens any of passed / skipped /
/// details / stats trips this test.
#[test]
fn assert_result_missing_required_field_rejected_by_deserialize() {
    // All five `AssertResult` fields are wire-required (the struct
    // has no `Default` derive and no `#[serde(default)]` on any
    // field). Loop over each; each removal must fail deserialize
    // with a missing-field error naming the removed field.
    const REQUIRED_FIELDS: &[&str] = &[
        "outcomes",
        "passes",
        "stats",
        "measurements",
        "info_notes",
    ];

    let r = AssertResult {
        outcomes: vec![Outcome::Fail(AssertDetail::new(DetailKind::Other, "detail"))],
        passes: vec![],
        stats: ScenarioStats::default(),
        measurements: std::collections::BTreeMap::new(),
        info_notes: vec![],
    };
    let full = match serde_json::to_value(&r).unwrap() {
        serde_json::Value::Object(m) => m,
        other => panic!("expected object, got {other:?}"),
    };

    for field in REQUIRED_FIELDS {
        let mut obj = full.clone();
        assert!(
            obj.remove(*field).is_some(),
            "AssertResult must emit `{field}` for its rejection case to be meaningful",
        );
        let json = serde_json::Value::Object(obj).to_string();
        let err = serde_json::from_str::<AssertResult>(&json).err().unwrap_or_else(
            || panic!(
                "deserialize must reject AssertResult with `{field}` removed, but succeeded",
            ),
        );
        let msg = format!("{err}");
        assert!(
            msg.contains(field),
            "missing-field error for `{field}` must name the field; got: {msg}",
        );
    }
}
