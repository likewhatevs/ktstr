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
            timer_latencies_ns: vec![],
            timer_sample_total: 0,
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
/// real zero. The monitor-sourced metrics are NOT in `ext_metrics` (read via
/// `phase_metric`); the typed cross-cgroup fields (`worst_spread` etc.) re-derive
/// None-aware from the per-cgroup carriers — `None` here since this fixture
/// carries no cgroups.
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
        // Typed cross-cgroup struct field set directly, but run_metric re-derives
        // from the (here empty) cgroups carriers, not this field, so it resolves
        // None-aware-absent below.
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
    // Typed cross-cgroup field re-derives from the per-cgroup carriers
    // (self.cgroups), NOT from the struct field: this fixture has no cgroups, so
    // run_metric reports None-aware ABSENT (not-measured), independent of the
    // struct field worst_spread == 0.99 (which the field read still returns).
    assert_eq!(stats.run_metric("worst_spread"), None);
    assert_eq!(stats.worst_spread, 0.99);
    // Monitor-sourced run-level metric is not held on ScenarioStats -> None.
    assert_eq!(stats.run_metric("max_imbalance_ratio"), None);
}

/// The typed cross-cgroup fields re-derive None-aware from `self.cgroups`: a
/// MEASURED zero resolves to `Some(0.0)`, a never-measured carrier to `None` —
/// the distinction the bare 0.0-sentinel struct fields cannot carry.
#[test]
fn run_metric_typed_cgroup_fields_distinguish_measured_zero_from_absent() {
    use crate::stats::BuiltinMetric as B;
    let cg = |spread: Option<f64>, iters: u64, migr: u64, gap_ms: u64, workers: usize| CgroupStats {
        num_workers: workers,
        spread,
        total_iterations: iters,
        total_migrations: migr,
        migration_ratio: if iters > 0 { migr as f64 / iters as f64 } else { 0.0 },
        max_gap_ms: gap_ms,
        ..Default::default()
    };
    // MEASURED zeros -> Some(0.0): spread measured 0; 0 migrations over 1000
    // iters; 0 gap with 4 workers; the totals SUM to a measured 0 / 1000.
    let m = ScenarioStats {
        cgroups: vec![cg(Some(0.0), 1000, 0, 0, 4)],
        ..Default::default()
    };
    assert_eq!(m.run_metric(B::WorstSpread), Some(0.0));
    assert_eq!(m.run_metric(B::WorstMigrationRatio), Some(0.0));
    assert_eq!(m.run_metric(B::TotalMigrations), Some(0.0));
    assert_eq!(m.run_metric(B::TotalIterations), Some(1000.0));
    assert_eq!(m.run_metric(B::WorstGapMs), Some(0.0));
    // NEVER-MEASURED -> None: no spread; 0 iters (ratio undefined); no workers
    // (gap unmeasured). The SUM totals stay Some(0.0) — a cgroup contributed.
    let a = ScenarioStats {
        cgroups: vec![cg(None, 0, 0, 0, 0)],
        ..Default::default()
    };
    assert_eq!(a.run_metric(B::WorstSpread), None);
    assert_eq!(a.run_metric(B::WorstMigrationRatio), None);
    assert_eq!(a.run_metric(B::WorstGapMs), None);
    assert_eq!(a.run_metric(B::TotalMigrations), Some(0.0));
    assert_eq!(a.run_metric(B::TotalIterations), Some(0.0));
    // No cgroups at all -> the SUM totals are None (no contributor).
    let e = ScenarioStats::default();
    assert_eq!(e.run_metric(B::TotalMigrations), None);
    assert_eq!(e.run_metric(B::TotalIterations), None);
    // Worst-across-cgroups: max spread / max gap over the measured cgroups.
    let two = ScenarioStats {
        cgroups: vec![cg(Some(0.1), 100, 5, 3, 2), cg(Some(0.4), 100, 1, 9, 2)],
        ..Default::default()
    };
    assert_eq!(two.run_metric(B::WorstSpread), Some(0.4), "max spread across cgroups");
    assert_eq!(two.run_metric(B::WorstGapMs), Some(9.0), "max gap across cgroups");
    assert_eq!(two.run_metric(B::TotalMigrations), Some(6.0), "5 + 1 summed");
}

/// The two NUMA fields re-derive from the per-phase carriers with the ASYMMETRIC
/// cross-phase fold — LATEST residency snapshot (a gauge, NOT summed) + SUMmed
/// cross-node migration deltas — and a measured-0.0 locality (all pages off-node,
/// the worst cell) WINS the lowest fold rather than being skipped as a sentinel
/// (the read-side fix). Never-measured (no NUMA pages) and the phase-less path
/// resolve to None.
#[test]
fn run_metric_numa_fields_latest_residency_summed_migrations_none_aware() {
    use crate::stats::BuiltinMetric as B;
    let pcg = |local: u64, total: u64, migr: u64| PhaseCgroupStats {
        numa_pages_local: local,
        numa_pages_total: total,
        cross_node_migrated: migr,
        ..Default::default()
    };
    let phase = |step: u16, cgs: &[(&str, PhaseCgroupStats)]| {
        let mut b = PhaseBucket {
            step_index: step,
            ..Default::default()
        };
        for (n, pc) in cgs {
            b.per_cgroup.insert(n.to_string(), pc.clone());
        }
        b
    };
    // One cgroup A over two phases: residency is a GAUGE -> LATEST (phase 2:
    // 250/1000 = 0.25), NOT summed ((900+250)/(1000+1000) = 0.575). Migrations
    // are per-phase DELTAS -> SUM (10 + 20 = 30) over the LATEST total 1000.
    let g = ScenarioStats {
        phases: vec![
            phase(1, &[("A", pcg(900, 1000, 10))]),
            phase(2, &[("A", pcg(250, 1000, 20))]),
        ],
        ..Default::default()
    };
    assert_eq!(
        g.run_metric(B::WorstPageLocality),
        Some(0.25),
        "LATEST residency 250/1000, NOT summed (which would give 0.575)"
    );
    let cross = g.run_metric(B::WorstCrossNodeMigrationRatio).unwrap();
    assert!(
        (cross - 0.03).abs() < 1e-9,
        "summed migrations 30 / latest total 1000 = 0.03; got {cross}"
    );
    // A measured-0.0 locality (all off-node) is the WORST and wins the lowest
    // fold — NOT skipped as a sentinel (the fold_lowest_nonzero read-side bug).
    let z = ScenarioStats {
        phases: vec![phase(1, &[("A", pcg(0, 1000, 0)), ("B", pcg(1000, 1000, 0))])],
        ..Default::default()
    };
    assert_eq!(
        z.run_metric(B::WorstPageLocality),
        Some(0.0),
        "measured 0.0 (all off-node) is the worst — wins the lowest, not skipped"
    );
    // Never-measured (no NUMA pages, total == 0) -> None, distinct from Some(0.0).
    let n = ScenarioStats {
        phases: vec![phase(1, &[("A", pcg(0, 0, 0))])],
        ..Default::default()
    };
    assert_eq!(n.run_metric(B::WorstPageLocality), None, "no NUMA pages -> not measured");
    assert_eq!(n.run_metric(B::WorstCrossNodeMigrationRatio), None);
    // cross_node is a CHURN ratio (cumulative migration EVENTS / snapshot
    // residency), intentionally UNBOUNDED: more migrations than resident pages
    // (heavy re-migration) -> ratio > 1.0. page_locality stays bounded [0,1].
    let churn = ScenarioStats {
        phases: vec![phase(1, &[("A", pcg(500, 1000, 3000))])],
        ..Default::default()
    };
    let r = churn.run_metric(B::WorstCrossNodeMigrationRatio).unwrap();
    assert!((r - 3.0).abs() < 1e-9, "3000 migrations / 1000 pages = 3.0 (>1.0 churn); got {r}");
    assert_eq!(
        churn.run_metric(B::WorstPageLocality),
        Some(0.5),
        "page_locality 500/1000 stays a bounded [0,1] fraction"
    );
    // Cross-phase None-aware LATEST: a cgroup measured in phase 1 (700/1000)
    // whose phase-2 carrier measured NO NUMA pages (total == 0) keeps the
    // phase-1 residency — the not-measured later phase does NOT overwrite it to
    // 0/0 and drop the cgroup from the worst-pool (the cross-phase analogue of
    // the measured-vs-unmeasured discipline; LATEST means latest MEASURED).
    let latest_unmeasured = ScenarioStats {
        phases: vec![
            phase(1, &[("A", pcg(700, 1000, 0))]),
            phase(2, &[("A", pcg(0, 0, 0))]),
        ],
        ..Default::default()
    };
    assert_eq!(
        latest_unmeasured.run_metric(B::WorstPageLocality),
        Some(0.7),
        "a later not-measured (total == 0) phase must not erase phase 1's measured residency",
    );

    // Phase-less (direct-assertion path): no NUMA carriers -> None (loud-absent).
    let p = ScenarioStats {
        cgroups: vec![CgroupStats::default()],
        ..Default::default()
    };
    assert_eq!(p.run_metric(B::WorstPageLocality), None, "no phases -> no NUMA carriers");
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
