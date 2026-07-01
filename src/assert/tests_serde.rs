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
            cgroup_name: "cg_0".to_string(),
            num_workers: 4,
            num_cpus: 3,
            cpus_used: [0usize, 1, 2].into_iter().collect(),
            avg_off_cpu_pct: Some(50.0),
            min_off_cpu_pct: Some(40.0),
            max_off_cpu_pct: Some(60.0),
            spread: Some(20.0),
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
    // Full-value roundtrip, not just a 4-field spot check: a serde
    // regression (a rename typo, a precision-losing serializer, or an
    // Option mis-encode turning Some(50.0) into None/0.0) must surface.
    // The nested CgroupStats Option<f64> off-CPU% fields are the
    // load-bearing case — they are legitimately optional on the wire,
    // so a Some->None corruption deserializes cleanly and only an
    // explicit value check catches it.
    assert_eq!(s, s2, "ScenarioStats must roundtrip every field verbatim");
    // Spot-pin the optional off-CPU% fields so a reader sees exactly
    // what the equality guards.
    let c = &s2.cgroups[0];
    assert_eq!(c.avg_off_cpu_pct, Some(50.0));
    assert_eq!(c.min_off_cpu_pct, Some(40.0));
    assert_eq!(c.max_off_cpu_pct, Some(60.0));
    assert_eq!(c.spread, Some(20.0));
    // The new wire fields: non-empty so a BTreeSet ordering / String
    // encode regression on cpus_used / cgroup_name surfaces here, not
    // only via the render path.
    assert_eq!(c.cgroup_name, "cg_0");
    assert_eq!(
        c.cpus_used,
        [0usize, 1, 2]
            .into_iter()
            .collect::<std::collections::BTreeSet<usize>>(),
    );
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
    measurements.insert("rate".to_string(), NoteValue::Float(3.15));
    measurements.insert("ok".to_string(), NoteValue::Bool(true));
    measurements.insert(
        "label".to_string(),
        NoteValue::Text("benchmark".to_string()),
    );
    let r = AssertResult {
        outcomes: vec![
            Outcome::Fail(AssertDetail::new(DetailKind::Other, "fail msg")),
            Outcome::Inconclusive(AssertDetail::new(DetailKind::Other, "inconclusive msg")),
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
    assert_eq!(r.is_inconclusive(), r2.is_inconclusive());
    assert_eq!(r.outcomes.len(), r2.outcomes.len());
    let r_fails: Vec<_> = r.failure_details().collect();
    let r2_fails: Vec<_> = r2.failure_details().collect();
    assert_eq!(r_fails.len(), r2_fails.len());
    assert_eq!(r_fails[0].message, r2_fails[0].message);
    let r_incons: Vec<_> = r.inconclusive_details().collect();
    let r2_incons: Vec<_> = r2.inconclusive_details().collect();
    assert_eq!(r_incons.len(), r2_incons.len());
    assert_eq!(r_incons[0].message, r2_incons[0].message);
    let r_skips: Vec<_> = r.skip_details().collect();
    let r2_skips: Vec<_> = r2.skip_details().collect();
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
        assert!((f - 3.15).abs() < 1e-9);
    } else {
        panic!(
            "rate must decode to NoteValue::Float, got {:?}",
            r2.measurements.get("rate")
        );
    }
    assert_eq!(r2.measurements.get("ok"), Some(&NoteValue::Bool(true)));
    assert_eq!(
        r2.measurements.get("label"),
        Some(&NoteValue::Text("benchmark".to_string()))
    );
}

/// Strict-type rejection on `Assert` deserialize. The serde
/// derive emits typed `invalid type ...: expected $T` errors when
/// the wire payload supplies the wrong type — a regression that
/// softened any field to `#[serde(deserialize_with = lenient_*)]`
/// or `#[serde(default)]` on a typed-wrong input would silently
/// accept garbage and downgrade test thresholds. Pins three
/// representative fields across distinct shapes:
/// `enforce_monitor_thresholds` (bool) + `min_page_locality` +
/// `max_slow_tier_ratio` (Option<f64>). Each case verifies both
/// shape and locality: the error must be a typed-mismatch (not a
/// "missing field" or generic parse error — the typed-mismatch
/// shape is what discriminates strict from softened) AND the
/// `serde_path_to_error` wrapper must surface the offending
/// field name so an operator pasting a malformed config can
/// diagnose at the source, not at the call site that consumed a
/// silently-defaulted value.
#[test]
fn assert_strict_type_rejection_names_offending_field() {
    let base = serde_json::to_value(Assert::NO_OVERRIDES).expect("baseline serializes");
    let serde_json::Value::Object(base_obj) = base else {
        panic!("Assert must serialize as a JSON object");
    };

    let cases: &[(&str, serde_json::Value)] = &[
        ("enforce_monitor_thresholds", serde_json::json!(0)),
        ("enforce_monitor_thresholds", serde_json::json!("true")),
        ("enforce_monitor_thresholds", serde_json::Value::Null),
        ("min_page_locality", serde_json::json!("0.95")),
        ("max_slow_tier_ratio", serde_json::json!(true)),
    ];

    for (field, wrong_val) in cases {
        let mut obj = base_obj.clone();
        obj.insert((*field).to_string(), wrong_val.clone());
        let value = serde_json::Value::Object(obj);
        let err = serde_path_to_error::deserialize::<_, Assert>(value)
            .expect_err(&format!("deserialize must reject {field} = {wrong_val}"));
        let path = err.path().to_string();
        let inner = err.inner().to_string();
        assert!(
            inner.contains("invalid type"),
            "deserialize error for `{field} = {wrong_val}` must be a typed-mismatch \
             (`invalid type ...`), not a softened-default acceptance; got inner: {inner}"
        );
        assert_eq!(
            path, *field,
            "serde_path_to_error must surface the offending field path for \
             `{field} = {wrong_val}`; got path `{path}` with inner `{inner}`"
        );
    }
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
    // The off-CPU% fields (avg / min / max_off_cpu_pct, spread) are
    // intentionally OMITTED: they are `Option<f64>`, where `None`
    // encodes "off-CPU% not measured" (no worker with positive wall
    // time). serde maps a missing key on an `Option` field to `None`,
    // which is the correct semantic here (absent == not reported), so
    // these are legitimately optional on the wire — not a softened
    // required scalar.
    const REQUIRED_FIELDS: &[&str] = &[
        "cgroup_name",
        "num_workers",
        "num_cpus",
        "cpus_used",
        "max_gap_ms",
        "max_gap_cpu",
        "total_migrations",
        "migration_ratio",
        "p99_wake_latency_us",
        "median_wake_latency_us",
        "wake_latency_cv",
        "wake_measured",
        "median_timer_latency_us",
        "p99_timer_latency_us",
        "p999_timer_latency_us",
        "worst_timer_latency_us",
        "timer_measured",
        "total_iterations",
        "total_cpu_time_ns",
        "mean_run_delay_us",
        "worst_run_delay_us",
        "run_delay_measured",
        "page_locality",
        "cross_node_migration_ratio",
        "ext_metrics",
    ];
    // The legitimately-optional wire fields, where a missing key maps to `None`:
    // the off-CPU% `Option<f64>` family ("not measured"), and `taobench_whole`
    // (`Option<TaobenchStats>`, `None` for a non-Taobench cgroup). serde maps a
    // missing key on an `Option` field to `None`, the correct absent semantic, so
    // these are legitimately optional on the wire. Every other emitted field is a
    // required scalar.
    const OPTIONAL_FIELDS: &[&str] = &[
        "avg_off_cpu_pct",
        "min_off_cpu_pct",
        "max_off_cpu_pct",
        "spread",
        "taobench_whole",
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

    // Completeness guard: every emitted wire field must be classified as either
    // required or (Option) optional. This catches the inverse drift the
    // per-field loop below misses — a NEW struct field that nobody added to
    // REQUIRED_FIELDS would otherwise silently escape the strict-schema check
    // (exactly how the `*_measured` bools and the timer reductions slipped).
    for key in full.keys() {
        assert!(
            REQUIRED_FIELDS.contains(&key.as_str()) || OPTIONAL_FIELDS.contains(&key.as_str()),
            "CgroupStats wire field `{key}` is in neither REQUIRED_FIELDS nor \
             OPTIONAL_FIELDS — a new field escaped the strict-schema test; \
             classify it (required scalar, or Option → optional)",
        );
    }

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
        "total_iterations",
        // The wake / run-delay (worst_p99/median/cv, worst_mean_run_delay_us,
        // worst_run_delay_us), iteration-efficiency
        // (worst_iterations_per_worker/_per_cpu_sec), both NUMA roll-ups
        // (worst_page_locality, worst_cross_node_migration_ratio), and
        // wake-latency tail-ratio (worst_wake_latency_tail_ratio) roll-ups are
        // intentionally omitted: they are no longer typed ScenarioStats fields —
        // they are `MetricKind::Distribution` / `WorstLowest` /
        // `WorstCrossNodeRatio` / `WakeLatencyTailRatio` metrics re-pooled into
        // `ext_metrics` post-merge by `populate_run_distribution_metrics`.
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
    const REQUIRED_FIELDS: &[&str] = &["outcomes", "passes", "stats", "measurements", "info_notes"];

    let r = AssertResult {
        outcomes: vec![Outcome::Fail(AssertDetail::new(
            DetailKind::Other,
            "detail",
        ))],
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

/// `#[serde(skip)]` semantic pin for the two reproducer-matcher
/// fields on [`Assert`] (`expect_scx_bpf_error_contains` +
/// `expect_scx_bpf_error_matches`). Both fields are intentionally
/// dropped from the wire format — the `&'static str` shape cannot
/// roundtrip through a borrowed deserializer (no source-string
/// lifetime to bind to) and the matcher patterns are test-author
/// static literals, not per-run data the sidecar needs to
/// roundtrip (see the docstrings on each field for the full
/// rationale).
///
/// This test pins the BYPASS semantic: an `Assert` constructed
/// with both matcher fields populated MUST serialize to JSON that
/// OMITS them, AND the JSON must deserialize back into an `Assert`
/// whose matcher fields are `None`. Together those two properties
/// prove the `#[serde(skip)]` is wired on BOTH the serialize and
/// deserialize sides — a regression that dropped the attribute
/// from either direction would silently start sending matcher
/// strings on the wire (serialize side) or attempt to deserialize
/// them and fail at the borrow-lifetime gate (deserialize side).
#[test]
fn assert_reproducer_matcher_fields_serde_skip_bypass() {
    use crate::assert::Assert;

    let with_matchers = Assert::NO_OVERRIDES
        .expect_scx_bpf_error_contains("apply_cell_config")
        .expect_scx_bpf_error_matches(r"(?m)^apply_cell_config$");

    assert_eq!(
        with_matchers.expect_scx_bpf_error_contains,
        Some("apply_cell_config"),
        "constructed value must carry the contains matcher",
    );
    assert_eq!(
        with_matchers.expect_scx_bpf_error_matches,
        Some(r"(?m)^apply_cell_config$"),
        "constructed value must carry the regex matcher",
    );

    let json =
        serde_json::to_string(&with_matchers).expect("Assert with matchers must serialize cleanly");
    assert!(
        !json.contains("expect_scx_bpf_error_contains"),
        "serialized JSON must OMIT expect_scx_bpf_error_contains \
         (#[serde(skip)] regressed on serialize side); got: {json}",
    );
    assert!(
        !json.contains("expect_scx_bpf_error_matches"),
        "serialized JSON must OMIT expect_scx_bpf_error_matches \
         (#[serde(skip)] regressed on serialize side); got: {json}",
    );

    let roundtrip: Assert =
        serde_json::from_str(&json).expect("serialized matcher-bearing Assert must deserialize");
    assert_eq!(
        roundtrip.expect_scx_bpf_error_contains, None,
        "deserialized contains matcher must be None — \
         #[serde(skip)] should default-init Option to None per \
         Option::default(); regression would either deserialize \
         the omitted field with a non-None value (impossible per \
         the skip contract) or fail the deserialize entirely.",
    );
    assert_eq!(
        roundtrip.expect_scx_bpf_error_matches, None,
        "deserialized regex matcher must be None — same rationale \
         as expect_scx_bpf_error_contains above.",
    );
}
