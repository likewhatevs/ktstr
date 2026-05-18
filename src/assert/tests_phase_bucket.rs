//! Unit tests for [`PhaseBucket`] and the [`ScenarioStats`]
//! per-phase accessor surface. Verifies the
//! `Op::ReadKernel*`-style discoverability path
//! (`phase` / `phase_metric`) plus the serde round-trip every new
//! pub serialized type needs.

use std::collections::BTreeMap;

use super::{PhaseBucket, ScenarioStats};

/// `PhaseBucket` serde round-trip covering every field including
/// the `end_ms == u64::MAX` open-ended sentinel and a populated
/// metrics map. Pins the wire shape against any schema drift (a
/// future field rename or kind tag change surfaces here, not via
/// the wider SidecarResult round-trip which carries many fields).
#[test]
fn phase_bucket_json_round_trips_all_fields() {
    let mut metrics = BTreeMap::new();
    metrics.insert("worst_spread".to_string(), 0.42);
    metrics.insert("dsq_depth_max".to_string(), 12.0);
    let bucket = PhaseBucket {
        step_index: 7,
        label: "Step[6]".to_string(),
        start_ms: 1500,
        end_ms: u64::MAX,
        sample_count: 42,
        metrics,
    };
    let json = serde_json::to_string(&bucket).expect("serialize");
    let back: PhaseBucket = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, bucket);
}

/// Empty `metrics` BTreeMap serializes as a present-but-empty
/// `"metrics": {}` field, not as absent. Pins the distinction
/// between "phase had no samples for any metric" (empty map,
/// present) and "deserialization dropped the field" (absent).
#[test]
fn phase_bucket_empty_metrics_round_trips_as_empty_object() {
    let bucket = PhaseBucket {
        step_index: 0,
        label: "BASELINE".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 0,
        metrics: BTreeMap::new(),
    };
    let json = serde_json::to_string(&bucket).expect("serialize");
    assert!(
        json.contains(r#""metrics":{}"#),
        "empty metrics must serialize as present `metrics: {{}}`, got: {json}"
    );
    let back: PhaseBucket = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, bucket);
}

/// `step_index = u16::MAX` round-trips losslessly through serde_json.
/// Pins the type-width contract: any future `#[serde(with = ...)]`
/// or accidental narrowing to i16 corrupts at the boundary.
#[test]
fn phase_bucket_step_index_u16_max_round_trips() {
    let bucket = PhaseBucket {
        step_index: u16::MAX,
        label: "Step[65534]".to_string(),
        start_ms: 0,
        end_ms: 1,
        sample_count: 0,
        metrics: BTreeMap::new(),
    };
    let json = serde_json::to_string(&bucket).expect("serialize");
    let back: PhaseBucket = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.step_index, u16::MAX);
    assert_eq!(back, bucket);
}

/// Empty `label` string serializes as a present-but-empty field,
/// not dropped. Pins against a future
/// `#[serde(skip_serializing_if = "String::is_empty")]` regression
/// that would silently change the wire shape.
#[test]
fn phase_bucket_empty_label_round_trips_as_present_field() {
    let bucket = PhaseBucket {
        step_index: 0,
        label: String::new(),
        start_ms: 0,
        end_ms: 0,
        sample_count: 0,
        metrics: BTreeMap::new(),
    };
    let json = serde_json::to_string(&bucket).expect("serialize");
    assert!(
        json.contains(r#""label":"""#),
        "empty label must serialize as present `label: \"\"`, got: {json}"
    );
    let back: PhaseBucket = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.label, "");
    assert_eq!(back, bucket);
}

/// `PhaseBucket::get` returns the value when the key is present
/// and `None` when absent. The absence semantic is load-bearing —
/// the per-phase aggregator emits absent keys for "no finite
/// samples for this metric in this phase," distinct from `Some(0.0)`
/// which means the reducer produced a real zero.
#[test]
fn phase_bucket_get_distinguishes_absent_from_zero() {
    let mut metrics = BTreeMap::new();
    metrics.insert("present".to_string(), 0.0);
    let bucket = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 1000,
        sample_count: 10,
        metrics,
    };
    assert_eq!(bucket.get("present"), Some(0.0));
    assert_eq!(bucket.get("absent"), None);
}

/// `ScenarioStats::Default` yields an empty `phases` vec. Existing
/// scenarios that don't construct phases explicitly get the
/// flat-bucket-only shape with zero per-phase data.
#[test]
fn scenario_stats_default_has_empty_phases() {
    let stats = ScenarioStats::default();
    assert!(stats.phases.is_empty());
    assert_eq!(stats.phase(0), None);
    assert_eq!(stats.phase_metric(0, "any"), None);
}

/// `ScenarioStats::phase` looks up by `step_index` rather than vec
/// position. A non-contiguous phases vec (e.g. with BASELINE plus
/// Step[2] only, skipping Step[0] and Step[1] entries) still
/// resolves correctly by step_index — the lookup uses the field,
/// not the slot.
#[test]
fn scenario_stats_phase_lookup_by_step_index_not_position() {
    let mut metrics_baseline = BTreeMap::new();
    metrics_baseline.insert("worst_spread".to_string(), 0.10);
    let mut metrics_step2 = BTreeMap::new();
    metrics_step2.insert("worst_spread".to_string(), 0.42);
    let stats = ScenarioStats {
        phases: vec![
            PhaseBucket {
                step_index: 0,
                label: "BASELINE".to_string(),
                start_ms: 0,
                end_ms: 100,
                sample_count: 2,
                metrics: metrics_baseline,
            },
            PhaseBucket {
                step_index: 3,
                label: "Step[2]".to_string(),
                start_ms: 200,
                end_ms: 300,
                sample_count: 5,
                metrics: metrics_step2,
            },
        ],
        ..Default::default()
    };
    assert_eq!(stats.phase(0).map(|p| p.step_index), Some(0));
    assert_eq!(stats.phase(3).map(|p| p.step_index), Some(3));
    assert_eq!(stats.phase(1), None);
    assert_eq!(stats.phase(2), None);
}

/// `ScenarioStats::phase_metric` is the typed shortcut for
/// `phase(idx).and_then(|p| p.get(metric))`. Returns the value
/// when both the step and metric are present; `None` when either
/// is missing.
#[test]
fn scenario_stats_phase_metric_resolves_typed_lookup() {
    let mut metrics = BTreeMap::new();
    metrics.insert("worst_spread".to_string(), 0.42);
    metrics.insert("dsq_depth_max".to_string(), 12.0);
    let stats = ScenarioStats {
        phases: vec![PhaseBucket {
            step_index: 1,
            label: "Step[0]".to_string(),
            start_ms: 100,
            end_ms: 200,
            sample_count: 3,
            metrics,
        }],
        ..Default::default()
    };
    assert_eq!(stats.phase_metric(1, "worst_spread"), Some(0.42));
    assert_eq!(stats.phase_metric(1, "dsq_depth_max"), Some(12.0));
    assert_eq!(stats.phase_metric(1, "absent"), None);
    assert_eq!(stats.phase_metric(99, "worst_spread"), None);
}

/// `ScenarioStats::step` translates 0-indexed scenario Step number
/// to the 1-indexed phase encoding: scenario-Step N lives at
/// `step_index = N + 1`. The accessor hides the 1-indexing trap.
#[test]
fn scenario_stats_step_translates_scenario_step_idx_to_phase_index() {
    let stats = ScenarioStats {
        phases: vec![
            PhaseBucket {
                step_index: 0, // BASELINE
                label: "BASELINE".to_string(),
                ..Default::default()
            },
            PhaseBucket {
                step_index: 1, // Step 0 of scenario
                label: "Step[0]".to_string(),
                ..Default::default()
            },
            PhaseBucket {
                step_index: 2, // Step 1 of scenario
                label: "Step[1]".to_string(),
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    // step(0) = "Step[0]" (scenario-side first Step, NOT BASELINE)
    assert_eq!(stats.step(0).map(|p| p.label.as_str()), Some("Step[0]"));
    assert_eq!(stats.step(1).map(|p| p.label.as_str()), Some("Step[1]"));
    // Out-of-range scenario Step returns None
    assert_eq!(stats.step(99), None);
    // u16::MAX + 1 saturates via checked_add → None
    assert_eq!(stats.step(u16::MAX), None);
}

/// `ScenarioStats::step_metric` is the sibling shortcut to
/// `phase_metric` taking a 0-indexed scenario-Step number.
#[test]
fn scenario_stats_step_metric_resolves_scenario_indexed_lookup() {
    let mut metrics = BTreeMap::new();
    metrics.insert("worst_spread".to_string(), 0.42);
    let stats = ScenarioStats {
        phases: vec![PhaseBucket {
            step_index: 1, // Scenario Step 0
            label: "Step[0]".to_string(),
            metrics,
            ..Default::default()
        }],
        ..Default::default()
    };
    assert_eq!(stats.step_metric(0, "worst_spread"), Some(0.42));
    assert_eq!(stats.step_metric(0, "absent"), None);
    assert_eq!(stats.step_metric(1, "worst_spread"), None);
}

/// `ScenarioStats::is_known_metric` lets the test author
/// distinguish a typo (`"worts_spread"`) from legitimate-absent
/// data (the metric simply had no finite samples in the phase).
#[test]
fn scenario_stats_is_known_metric_distinguishes_typo_from_absent_data() {
    // "worst_spread" is a registered METRICS entry.
    assert!(ScenarioStats::is_known_metric("worst_spread"));
    // A typo / unknown metric name is NOT registered.
    assert!(!ScenarioStats::is_known_metric("worts_spread"));
    assert!(!ScenarioStats::is_known_metric(""));
    assert!(!ScenarioStats::is_known_metric("totally_made_up"));
}

/// `ScenarioStats::known_metrics` yields the same set of names
/// that `is_known_metric` validates positively. Round-trip
/// consistency: every yielded name passes is_known_metric, and
/// the count matches the METRICS registry length.
#[test]
fn scenario_stats_known_metrics_iterates_registry() {
    let names: Vec<&'static str> = ScenarioStats::known_metrics().collect();
    assert!(!names.is_empty(), "METRICS registry must have entries");
    assert_eq!(names.len(), crate::stats::METRICS.len());
    for name in names {
        assert!(
            ScenarioStats::is_known_metric(name),
            "every known_metrics() entry must pass is_known_metric: {name}"
        );
    }
}
