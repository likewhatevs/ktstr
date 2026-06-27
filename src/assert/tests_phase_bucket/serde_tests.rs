use super::*;

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
        per_cgroup: Default::default(),
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

/// `PhaseBucket.per_cgroup` serde round-trip with a fully-populated
/// [`PhaseCgroupStats`] (every field type: the sample Vecs, the cpus_used set,
/// and the counters) pins the per-phase per-cgroup wire shape. Also asserts
/// the default carrier is empty — the structural-carrier invariant before any
/// capture path populates it.
#[test]
fn phase_bucket_per_cgroup_round_trips_and_defaults_empty() {
    use super::PhaseCgroupStats;
    use std::collections::BTreeSet;
    assert!(
        PhaseBucket::default().per_cgroup.is_empty(),
        "the structural carrier defaults to an empty per_cgroup map",
    );
    let mut bucket = PhaseBucket {
        per_cgroup: Default::default(),
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 1000,
        sample_count: 3,
        metrics: BTreeMap::new(),
    };
    bucket.per_cgroup.insert(
        "cg_0".to_string(),
        PhaseCgroupStats {
            num_workers: 3,
            cpus_used: BTreeSet::from([2, 5, 6]),
            wake_latencies_ns: vec![10, 20, 30],
            wake_sample_total: 3,
            run_delays_ns: vec![1_500, 2_500],
            off_cpu_pcts: vec![1.5, 11.0, 22.5],
            total_migrations: 7,
            total_iterations: 4200,
            total_cpu_time_ns: 9_000_000,
            numa_pages_local: 90,
            numa_pages_total: 100,
            cross_node_migrated: 4,
            max_gap_ms: 13,
            max_gap_cpu: 2,
            stripped: false,
            metrics: std::collections::BTreeMap::new(),
            schbench: None,
            taobench: None,
        },
    );
    let json = serde_json::to_string(&bucket).expect("serialize");
    let back: PhaseBucket = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, bucket);
    assert_eq!(back.per_cgroup["cg_0"].total_iterations, 4200);
    assert_eq!(back.per_cgroup["cg_0"].wake_latencies_ns, vec![10, 20, 30]);
    assert_eq!(back.per_cgroup["cg_0"].off_cpu_pcts, vec![1.5, 11.0, 22.5]);
    assert_eq!(back.per_cgroup["cg_0"].cpus_used, BTreeSet::from([2, 5, 6]),);
}

/// Empty `metrics` BTreeMap serializes as a present-but-empty
/// `"metrics": {}` field, not as absent. Pins the distinction
/// between "phase had no samples for any metric" (empty map,
/// present) and "deserialization dropped the field" (absent).
#[test]
fn phase_bucket_empty_metrics_round_trips_as_empty_object() {
    let bucket = PhaseBucket {
        per_cgroup: Default::default(),
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
        per_cgroup: Default::default(),
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
        per_cgroup: Default::default(),
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
        per_cgroup: Default::default(),
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
    assert_eq!(stats.phase(crate::assert::Phase::BASELINE), None);
    assert_eq!(
        stats.phase_metric(crate::assert::Phase::BASELINE, "any"),
        None
    );
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
                per_cgroup: Default::default(),
                step_index: 0,
                label: "BASELINE".to_string(),
                start_ms: 0,
                end_ms: 100,
                sample_count: 2,
                metrics: metrics_baseline,
            },
            PhaseBucket {
                per_cgroup: Default::default(),
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
    assert_eq!(
        stats
            .phase(crate::assert::Phase::BASELINE)
            .map(|p| p.step_index),
        Some(0)
    );
    assert_eq!(
        stats
            .phase(crate::assert::Phase::step(2))
            .map(|p| p.step_index),
        Some(3)
    );
    assert_eq!(stats.phase(crate::assert::Phase::step(0)), None);
    assert_eq!(stats.phase(crate::assert::Phase::step(1)), None);
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
            per_cgroup: Default::default(),
            step_index: 1,
            label: "Step[0]".to_string(),
            start_ms: 100,
            end_ms: 200,
            sample_count: 3,
            metrics,
        }],
        ..Default::default()
    };
    assert_eq!(
        stats.phase_metric(crate::assert::Phase::step(0), "worst_spread"),
        Some(0.42)
    );
    // A typed BuiltinMetric resolves identically to its &str wire name.
    assert_eq!(
        stats.phase_metric(
            crate::assert::Phase::step(0),
            crate::stats::BuiltinMetric::WorstSpread
        ),
        Some(0.42),
    );
    // A non-registered key stays a dynamic lookup against the bucket.
    assert_eq!(
        stats.phase_metric(crate::assert::Phase::step(0), "dsq_depth_max"),
        Some(12.0)
    );
    assert_eq!(
        stats.phase_metric(crate::assert::Phase::step(0), "absent"),
        None
    );
    assert_eq!(
        stats.phase_metric(crate::assert::Phase::step(98), "worst_spread"),
        None
    );
}

/// `ScenarioStats::phase(Phase::step(k))` resolves to the k-th scenario Step's
/// bucket — the typed `Phase` hides the 1-indexed encoding (Step k lives at the
/// underlying `step_index = k + 1`). Out-of-range Steps and the `u16::MAX`
/// saturation both resolve to `None` (no such bucket).
#[test]
fn scenario_stats_phase_step_resolves_to_step_bucket() {
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
    // Phase::step(0) is the scenario's first Step (Step[0]), NOT BASELINE.
    assert_eq!(
        stats
            .phase(crate::assert::Phase::step(0))
            .map(|p| p.label.as_str()),
        Some("Step[0]")
    );
    assert_eq!(
        stats
            .phase(crate::assert::Phase::step(1))
            .map(|p| p.label.as_str()),
        Some("Step[1]")
    );
    // Out-of-range Step -> None.
    assert_eq!(stats.phase(crate::assert::Phase::step(99)), None);
    // Phase::step(u16::MAX) saturates to step_index u16::MAX -> no bucket -> None.
    assert_eq!(stats.phase(crate::assert::Phase::step(u16::MAX)), None);
}

/// `ScenarioStats::run_metric` resolves the run-level ext-sourced
/// metric family by registry name (the typed-method replacement for the
/// deleted `worst_*` fields, so code holding the run's `AssertResult`
/// never reaches into the raw `ext_metrics` map by string).
/// Sentinel-free: an absent name is `None`, a measured `Some(0.0)` is a
/// real zero. The typed cross-cgroup fields (`worst_spread` etc.) and
/// the monitor-sourced metrics are NOT in `ext_metrics`, so they
/// resolve to `None` here — read those via their named fields /
/// `phase_metric`.
#[test]
fn scenario_stats_run_metric_resolves_ext_family_sentinel_free() {
    let mut ext = BTreeMap::new();
    // A reclassified Distribution metric (was a typed field pre-Item-7).
    ext.insert("worst_run_delay_us".to_string(), 48.0);
    // A reclassified WorstLowest metric.
    ext.insert("worst_iterations_per_cpu_sec".to_string(), 12345.0);
    // A real measured zero — must resolve to Some(0.0), not None.
    ext.insert("worst_wake_latency_cv".to_string(), 0.0);
    // A user-defined extensible-metric key resolves too.
    ext.insert("my_custom_metric".to_string(), 7.0);
    let stats = ScenarioStats {
        ext_metrics: ext,
        // Typed cross-cgroup field set, but NOT mirrored into ext_metrics:
        // run_metric() must NOT resolve it (read via the named field instead).
        worst_spread: 0.99,
        ..Default::default()
    };
    assert_eq!(stats.run_metric("worst_run_delay_us"), Some(48.0));
    assert_eq!(
        stats.run_metric("worst_iterations_per_cpu_sec"),
        Some(12345.0)
    );
    // Sentinel-free: real measured zero is Some(0.0), distinct from absent.
    assert_eq!(stats.run_metric("worst_wake_latency_cv"), Some(0.0));
    assert_eq!(stats.run_metric("my_custom_metric"), Some(7.0));
    // Absent ext key (no contributing cgroup/carrier, or a typo) -> None.
    assert_eq!(stats.run_metric("worst_p99_wake_latency_us"), None);
    assert_eq!(stats.run_metric("totally_made_up"), None);
    // Typed cross-cgroup field is not in ext_metrics -> None here even
    // though the run carries it (worst_spread == 0.99): read via the field.
    assert_eq!(stats.run_metric("worst_spread"), None);
    assert_eq!(stats.worst_spread, 0.99);
    // Monitor-sourced run-level metric is not held on ScenarioStats -> None.
    assert_eq!(stats.run_metric("max_imbalance_ratio"), None);
}

/// The metric accessors take `impl Into<MetricId>`: a typed
/// [`crate::stats::BuiltinMetric`], a `&str`, a `String`, and a `&String` all
/// resolve identically (`From<&str>`/`From<String>` canonicalize a registered
/// name to the typed `Builtin`), and an unregistered scheduler-runtime key
/// resolves to `None` rather than a guessed value (the no-guessed-kind
/// guardrail, reached through an accessor).
#[test]
fn scenario_stats_accessors_accept_typed_and_string_metric_ids() {
    use crate::stats::BuiltinMetric;
    let mut ext = BTreeMap::new();
    ext.insert("worst_run_delay_us".to_string(), 48.0);
    let stats = ScenarioStats {
        ext_metrics: ext,
        ..Default::default()
    };
    let owned = "worst_run_delay_us".to_string();
    // Typed, &str, String, and &String ids all resolve to the SAME value.
    assert_eq!(stats.run_metric(BuiltinMetric::WorstRunDelayUs), Some(48.0));
    assert_eq!(stats.run_metric("worst_run_delay_us"), Some(48.0));
    assert_eq!(stats.run_metric(owned.clone()), Some(48.0));
    assert_eq!(stats.run_metric(&owned), Some(48.0));
    // An unregistered scheduler-runtime key resolves to None, not a guess.
    assert_eq!(stats.run_metric("scx_runtime_only_key"), None);
}
