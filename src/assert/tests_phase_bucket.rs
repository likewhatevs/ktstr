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
    assert_eq!(stats.run_metric("worst_iterations_per_cpu_sec"), Some(12345.0));
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

// -- build_phase_buckets pipeline tests ------------------------------
//
// These tests construct synthetic `SampleSeries`'s with explicit
// `step_index` stamping and run them through `build_phase_buckets`
// end-to-end to verify the bucket-shape contract:
// * one bucket per observed step_index
// * label encodes the 1-indexed convention (`BASELINE` /
//   `Step[k-1]`)
// * start_ms / end_ms span first..last sample in the bucket
// * sample_count matches the input count
//
// Metric population is exercised separately at the
// per-metric-arm tests in `src/stats.rs`; these tests verify the
// bucketing skeleton independent of metric data.

use crate::monitor::dump::{FailureDumpReport, SCHEMA_SINGLE};
use crate::scenario::sample::SampleSeries;
use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};

/// Build a minimal `FailureDumpReport` placeholder for tests.
/// Carries no BPF state — `MetricDef::read_sample` returns `None`
/// for every metric on this report, so the resulting
/// `PhaseBucket.metrics` map is empty. The test exercises the
/// bucketing shape, not the metric extraction.
fn fixture_report() -> FailureDumpReport {
    FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        ..Default::default()
    }
}

/// Build a synthetic `DrainedSnapshotEntry` with the given
/// `step_index` stamp and `elapsed_ms` anchor.
fn fixture_entry(tag: &str, step_index: u16, elapsed_ms: u64) -> DrainedSnapshotEntry {
    DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(elapsed_ms),
        boundary_offset_ms: None,
        step_index: Some(step_index),
    }
}

/// Empty `SampleSeries` -> empty `phases` vec. No BASELINE
/// bucket is synthesised from nothing; the aggregator yields
/// the empty shape so the renderer downstream can paint the
/// "no per-phase data" path correctly (distinct from "BASELINE
/// existed but had no metrics").
#[test]
fn build_phase_buckets_empty_series_yields_empty_phases() {
    let samples = SampleSeries::from_drained_typed(Vec::new(), None);
    let phases = crate::assert::build_phase_buckets(&samples);
    assert!(
        phases.is_empty(),
        "empty input must yield empty phases, got {phases:?}"
    );
}

/// Three samples all stamped under BASELINE (`step_index = 0`)
/// produce a single PhaseBucket with `label = "BASELINE"`,
/// `sample_count = 3`, and start/end_ms spanning the first/last
/// sample's elapsed_ms. Pins the BASELINE label convention.
#[test]
fn build_phase_buckets_baseline_only_yields_single_bucket() {
    let drained = vec![
        fixture_entry("periodic_000", 0, 100),
        fixture_entry("periodic_001", 0, 200),
        fixture_entry("periodic_002", 0, 300),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    assert_eq!(phases.len(), 1, "single phase observed -> single bucket");
    let bucket = &phases[0];
    assert_eq!(bucket.step_index, 0);
    assert_eq!(bucket.label, "BASELINE");
    assert_eq!(bucket.sample_count, 3);
    assert_eq!(bucket.start_ms, 100);
    assert_eq!(bucket.end_ms, 300);
    assert!(
        bucket.metrics.is_empty(),
        "synthetic fixture report carries no BPF state -> metrics empty"
    );
}

/// Three phases (BASELINE + Step[0] + Step[1]) round-trip
/// correctly: 3 buckets emitted in step_index order, labels are
/// "BASELINE" / "Step[0]" / "Step[1]" per the 1-indexed
/// convention (scenario Step k lives at step_index k+1), each
/// bucket counts its own samples, start/end_ms spans the
/// bucket's window.
#[test]
fn build_phase_buckets_three_phases_round_trip_with_correct_labels() {
    let drained = vec![
        fixture_entry("periodic_000", 0, 10),  // BASELINE
        fixture_entry("periodic_001", 0, 20),  // BASELINE
        fixture_entry("periodic_002", 1, 100), // Step[0]
        fixture_entry("periodic_003", 1, 200), // Step[0]
        fixture_entry("periodic_004", 1, 300), // Step[0]
        fixture_entry("periodic_005", 2, 400), // Step[1]
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    assert_eq!(phases.len(), 3);

    // Buckets returned in step_index order because
    // SampleSeries::by_stamped_phase returns a BTreeMap keyed by step_index.
    assert_eq!(phases[0].step_index, 0);
    assert_eq!(phases[0].label, "BASELINE");
    assert_eq!(phases[0].sample_count, 2);
    assert_eq!(phases[0].start_ms, 10);
    assert_eq!(phases[0].end_ms, 20);

    assert_eq!(phases[1].step_index, 1);
    assert_eq!(phases[1].label, "Step[0]");
    assert_eq!(phases[1].sample_count, 3);
    assert_eq!(phases[1].start_ms, 100);
    assert_eq!(phases[1].end_ms, 300);

    assert_eq!(phases[2].step_index, 2);
    assert_eq!(phases[2].label, "Step[1]");
    assert_eq!(phases[2].sample_count, 1);
    // Single sample in the bucket: start_ms == end_ms.
    assert_eq!(phases[2].start_ms, 400);
    assert_eq!(phases[2].end_ms, 400);
}

/// Unstamped samples (DrainedSnapshotEntry.step_index = None)
/// fall under key `0` per SampleSeries::by_stamped_phase's
/// "no stamped index" fallback. The resulting bucket is
/// labelled "BASELINE" because step_index = 0 is the BASELINE
/// encoding regardless of whether the original stamp was Some(0)
/// or None. Pins the fallback semantic — fixture / legacy /
/// pre-step_index samples don't disappear, they cluster into
/// BASELINE.
#[test]
fn build_phase_buckets_unstamped_samples_cluster_under_baseline() {
    let unstamped = DrainedSnapshotEntry {
        tag: "periodic_000".to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(50),
        boundary_offset_ms: None,
        step_index: None,
    };
    let samples = SampleSeries::from_drained_typed(vec![unstamped], None);
    let phases = crate::assert::build_phase_buckets(&samples);
    assert_eq!(phases.len(), 1);
    assert_eq!(phases[0].step_index, 0);
    assert_eq!(phases[0].label, "BASELINE");
    assert_eq!(phases[0].sample_count, 1);
}

/// Non-contiguous step_index sequence (BASELINE + Step[2],
/// skipping Step[0] and Step[1]) yields exactly the observed
/// phases — the aggregator does not synthesise empty buckets
/// for skipped Step ordinals. A test author whose scenario
/// somehow produced a sparse step_index sequence sees the sparse
/// shape on the output, not a fictitious dense fill.
#[test]
fn build_phase_buckets_skipped_steps_yield_sparse_output() {
    let drained = vec![
        fixture_entry("periodic_000", 0, 10),
        fixture_entry("periodic_001", 3, 500), // Step[2]
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    assert_eq!(phases.len(), 2);
    assert_eq!(phases[0].step_index, 0);
    assert_eq!(phases[1].step_index, 3);
    assert_eq!(phases[1].label, "Step[2]");
}

/// Every wired per-sample metric arm extracts its value end-to-end
/// through the phase aggregator. Builds a 2-phase SampleSeries whose
/// snapshots
/// carry KNOWN dsq_states + event_counter_timeline values, runs
/// build_phase_buckets, and asserts each PhaseBucket.metrics
/// map contains the three wired keys (max_dsq_depth /
/// total_fallback / total_keep_last) with values matching the
/// per-kind reduction (Peak max-of-max for max_dsq_depth;
/// Counter last-first delta for total_fallback / total_keep_last).
///
/// Pins the wiring between MetricDef::read_sample's per-metric
/// dispatch (stats.rs:read_sample at L315+) and the per-phase
/// reduction (aggregate_samples_for_phase at L225). A future
/// refactor that drops a metric from the dispatch silently
/// produces a missing-key in PhaseBucket.metrics — which the
/// renderer paints as "absent" but is actually a regression;
/// without this test, that silent drop is invisible until
/// caught by an operator manually checking the compare output.
#[test]
fn build_phase_buckets_extracts_wired_metric_arms_end_to_end() {
    use crate::monitor::dump::{EventCounterSample, FailureDumpReport, SCHEMA_SINGLE};
    use crate::monitor::scx_walker::DsqState;

    // Sample helper that builds a FailureDumpReport carrying
    // explicit per-CPU dsq depth and cumulative event counters.
    // local_dsq_depth -> max_dsq_depth Peak (per-CPU max).
    // fallback / keep_last -> total_fallback / total_keep_last
    // Counter (cumulative since boot; per-phase delta is the
    // last-first across phase samples).
    fn report_with(dsq_depths: &[u32], fallback: i64, keep_last: i64) -> FailureDumpReport {
        let dsq_states = dsq_depths
            .iter()
            .enumerate()
            .map(|(cpu, &nr)| DsqState {
                id: 0,
                origin: format!("local cpu {cpu}"),
                nr,
                seq: 0,
                task_kvas: Vec::new(),
                truncated: false,
            })
            .collect();
        let event_counter_timeline = vec![EventCounterSample {
            elapsed_ms: 0,
            select_cpu_fallback: fallback,
            dispatch_local_dsq_offline: 0,
            dispatch_keep_last: keep_last,
            enq_skip_exiting: 0,
            enq_skip_migration_disabled: 0,
            reenq_immed: 0,
            reenq_local_repeat: 0,
            refill_slice_dfl: 0,
            bypass_duration: 0,
            bypass_dispatch: 0,
            bypass_activate: 0,
            insert_not_owned: 0,
            sub_bypass_dispatch: 0,
        }];
        FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            dsq_states,
            event_counter_timeline,
            ..Default::default()
        }
    }

    fn entry_with(
        tag: &str,
        step_index: u16,
        elapsed_ms: u64,
        dsq_depths: &[u32],
        fallback: i64,
        keep_last: i64,
    ) -> DrainedSnapshotEntry {
        DrainedSnapshotEntry {
            tag: tag.to_string(),
            report: report_with(dsq_depths, fallback, keep_last),
            stats: Err(MissingStatsReason::NoSchedulerBinary),
            elapsed_ms: Some(elapsed_ms),
            boundary_offset_ms: None,
            step_index: Some(step_index),
        }
    }

    // Phase 0 (BASELINE): 2 samples
    //   sample[0]: dsq depths [5, 3] -> max 5; fallback=10; keep_last=20
    //   sample[1]: dsq depths [4, 8] -> max 8; fallback=15; keep_last=30
    // Phase 1 (Step[0]): 2 samples
    //   sample[2]: dsq depths [12, 7] -> max 12; fallback=18; keep_last=35
    //   sample[3]: dsq depths [9, 11] -> max 11; fallback=25; keep_last=42
    let drained = vec![
        entry_with("periodic_000", 0, 10, &[5, 3], 10, 20),
        entry_with("periodic_001", 0, 20, &[4, 8], 15, 30),
        entry_with("periodic_002", 1, 100, &[12, 7], 18, 35),
        entry_with("periodic_003", 1, 200, &[9, 11], 25, 42),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    assert_eq!(phases.len(), 2, "BASELINE + Step[0] -> 2 buckets");

    // BASELINE bucket (step_index = 0):
    //   max_dsq_depth: per-sample max -> [5, 8]; Peak reduction
    //     across the phase via aggregate_samples (max-of-max) -> 8
    //   total_fallback: Counter delta last - first = 15 - 10 = 5
    //   total_keep_last: Counter delta last - first = 30 - 20 = 10
    let baseline = &phases[0];
    assert_eq!(baseline.step_index, 0);
    assert_eq!(
        baseline.metrics.get("max_dsq_depth").copied(),
        Some(8.0),
        "BASELINE max_dsq_depth: Peak reduction over per-sample [5, 8] yields max 8"
    );
    assert_eq!(
        baseline.metrics.get("total_fallback").copied(),
        Some(5.0),
        "BASELINE total_fallback: Counter delta 15 - 10 = 5"
    );
    assert_eq!(
        baseline.metrics.get("total_keep_last").copied(),
        Some(10.0),
        "BASELINE total_keep_last: Counter delta 30 - 20 = 10"
    );

    // Step[0] bucket (step_index = 1):
    //   max_dsq_depth: per-sample max -> [12, 11]; Peak max 12
    //   total_fallback: 25 - 18 = 7
    //   total_keep_last: 42 - 35 = 7
    let step0 = &phases[1];
    assert_eq!(step0.step_index, 1);
    assert_eq!(step0.label, "Step[0]");
    assert_eq!(
        step0.metrics.get("max_dsq_depth").copied(),
        Some(12.0),
        "Step[0] max_dsq_depth: Peak max of [12, 11] = 12"
    );
    assert_eq!(
        step0.metrics.get("total_fallback").copied(),
        Some(7.0),
        "Step[0] total_fallback: Counter delta 25 - 18 = 7"
    );
    assert_eq!(
        step0.metrics.get("total_keep_last").copied(),
        Some(7.0),
        "Step[0] total_keep_last: Counter delta 42 - 35 = 7"
    );

    // No host-only metric should appear in metrics maps —
    // worst_spread, worst_gap_ms, etc. are cross-cgroup folds
    // with no per-sample reading and stay absent.
    for host_only in [
        "worst_spread",
        "worst_gap_ms",
        "worst_migration_ratio",
        "max_imbalance_ratio",
        "worst_p99_wake_latency_us",
        "worst_iterations_per_worker",
        "worst_page_locality",
    ] {
        assert!(
            !baseline.metrics.contains_key(host_only),
            "BASELINE must not carry host-only metric {host_only}"
        );
        assert!(
            !step0.metrics.contains_key(host_only),
            "Step[0] must not carry host-only metric {host_only}"
        );
    }
}

/// Off-cadence captures (on-demand `Op::CaptureSnapshot` / watchpoint
/// fires, tagged non-`periodic_`) must NOT pollute per-phase metric
/// folds — the production accessors (`VmResult::phase_buckets` /
/// `evaluate_vm_result`) bucket `periodic_only()`. A full-series bucket
/// folds the off-cadence outlier into the Peak; a periodic-only bucket
/// excludes it.
#[test]
fn periodic_only_excludes_off_cadence_captures_from_phase_buckets() {
    use crate::monitor::dump::{FailureDumpReport, SCHEMA_SINGLE};
    use crate::monitor::scx_walker::DsqState;

    fn entry(tag: &str, elapsed_ms: u64, dsq_depth: u32) -> DrainedSnapshotEntry {
        DrainedSnapshotEntry {
            tag: tag.to_string(),
            report: FailureDumpReport {
                schema: SCHEMA_SINGLE.to_string(),
                dsq_states: vec![DsqState {
                    id: 0,
                    origin: "local cpu 0".to_string(),
                    nr: dsq_depth,
                    seq: 0,
                    task_kvas: Vec::new(),
                    truncated: false,
                }],
                ..Default::default()
            },
            stats: Err(MissingStatsReason::NoSchedulerBinary),
            elapsed_ms: Some(elapsed_ms),
            boundary_offset_ms: None,
            step_index: Some(1),
        }
    }

    // Step[0]: two periodic captures (dsq 5, 8) + one off-cadence
    // on-demand capture with a wild dsq depth (1000). max_dsq_depth is a
    // Peak (order-independent), so the outlier wins a full-series max.
    let drained = vec![
        entry("periodic_000", 100, 5),
        entry("periodic_001", 200, 8),
        entry("ondemand_000", 300, 1000),
    ];
    let series = SampleSeries::from_drained_typed(drained, None);

    // Sanity: the full series folds the off-cadence outlier into the Peak.
    let full = crate::assert::build_phase_buckets(&series);
    assert_eq!(
        full.iter()
            .find(|p| p.step_index == 1)
            .expect("Step[0]")
            .metrics
            .get("max_dsq_depth")
            .copied(),
        Some(1000.0),
        "full series includes the off-cadence capture's dsq depth",
    );

    // Periodic-only (the production path) EXCLUDES the off-cadence
    // capture: the Peak is the periodic max (8), not the 1000 outlier.
    let periodic = crate::assert::build_phase_buckets(&series.clone().periodic_only());
    assert_eq!(
        periodic
            .iter()
            .find(|p| p.step_index == 1)
            .expect("Step[0]")
            .metrics
            .get("max_dsq_depth")
            .copied(),
        Some(8.0),
        "periodic_only excludes the off-cadence outlier: max is 8, NOT 1000",
    );
}

/// `system_time_ns` / `user_time_ns` are injected per-phase as a
/// per-thread-GROUP delta (each tgid's `thread_group_cputime` at first
/// vs last appearance, summed), NOT a per-sample cross-task sum then a
/// Counter delta. Pins three properties:
///   * the delta is `last - first` of the live `task_struct` counter per
///     tgid (3000 ns system, 7000 ns user for the persistent group);
///   * a high-cumulative-counter task that appears in only ONE sample
///     contributes 0 (it never reaches two readable boundaries) — a
///     sum-then-delta would have inflated the phase by ~1e6 ns;
///   * a phase with fewer than two readable samples for any group omits
///     the key (absent != real 0).
///
/// signal_{u,s}time are `Some(0)` here, matching production: a readable
/// signal_struct with no exited-thread time reads `Some(0)`, not `None`
/// (`None` is reserved for a translate miss — see the dedicated test).
#[test]
fn build_phase_buckets_injects_per_group_cpu_time_delta() {
    use crate::monitor::task_enrichment::TaskEnrichment;

    fn task(tgid: i32, utime: u64, stime: u64) -> TaskEnrichment {
        TaskEnrichment {
            pid: tgid,
            tgid,
            utime,
            stime,
            signal_utime: Some(0),
            signal_stime: Some(0),
            ..Default::default()
        }
    }
    fn entry(
        tag: &str,
        step_index: u16,
        elapsed_ms: u64,
        tasks: Vec<TaskEnrichment>,
    ) -> DrainedSnapshotEntry {
        DrainedSnapshotEntry {
            tag: tag.to_string(),
            report: FailureDumpReport {
                schema: SCHEMA_SINGLE.to_string(),
                task_enrichments: tasks,
                ..Default::default()
            },
            stats: Err(MissingStatsReason::NoSchedulerBinary),
            elapsed_ms: Some(elapsed_ms),
            boundary_offset_ms: None,
            step_index: Some(step_index),
        }
    }

    let drained = vec![
        // BASELINE: a single enriched sample -> no delta measurable.
        entry("periodic_000", 0, 10, vec![task(100, 2000, 1000)]),
        // Step[0]: two samples. tgid=100 persists (utime 2000->9000,
        // stime 1000->4000). tgid=200 (huge cumulative history) appears
        // ONLY in the later sample -> single-appearance -> contributes 0.
        entry("periodic_001", 1, 100, vec![task(100, 2000, 1000)]),
        entry(
            "periodic_002",
            1,
            200,
            vec![task(100, 9000, 4000), task(200, 2_000_000, 1_000_000)],
        ),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    assert_eq!(phases.len(), 2, "BASELINE + Step[0]");

    let baseline = &phases[0];
    assert_eq!(baseline.step_index, 0);
    assert!(
        !baseline.metrics.contains_key("system_time_ns")
            && !baseline.metrics.contains_key("user_time_ns"),
        "single enriched sample -> CPU-time key omitted (absent != real 0)",
    );

    let step0 = &phases[1];
    assert_eq!(step0.step_index, 1);
    assert_eq!(
        step0.metrics.get("system_time_ns").copied(),
        Some(3000.0),
        "system delta = tgid100 (4000-1000) + tgid200 (single-appearance \
         -> 0); NOT 1_003_000 (sum-then-delta inflation)",
    );
    assert_eq!(
        step0.metrics.get("user_time_ns").copied(),
        Some(7000.0),
        "user delta = tgid100 (9000-2000); tgid200 single-appearance -> 0",
    );
}

/// The per-phase CPU-time fold includes the thread-group
/// `signal_struct` accumulator (a dying thread's time moves there at
/// exit), counted once per tgid, so a mid-phase thread exit does not
/// dip the group total below its live-thread-only sum.
#[test]
fn build_phase_buckets_cpu_time_includes_signal_accumulator() {
    use crate::monitor::task_enrichment::TaskEnrichment;

    fn entry(
        tag: &str,
        elapsed_ms: u64,
        stime: u64,
        signal_stime: Option<u64>,
    ) -> DrainedSnapshotEntry {
        DrainedSnapshotEntry {
            tag: tag.to_string(),
            report: FailureDumpReport {
                schema: SCHEMA_SINGLE.to_string(),
                task_enrichments: vec![TaskEnrichment {
                    pid: 100,
                    tgid: 100,
                    stime,
                    signal_stime,
                    ..Default::default()
                }],
                ..Default::default()
            },
            stats: Err(MissingStatsReason::NoSchedulerBinary),
            elapsed_ms: Some(elapsed_ms),
            boundary_offset_ms: None,
            step_index: Some(1),
        }
    }

    // tgid=100 group total = live stime + the shared signal accumulator.
    // Sample A's signal is Some(0) (a genuine zero — no exited-thread time
    // yet — NOT None, which would mean an unreadable signal_struct):
    //   sample A: 1000 + 0    = 1000
    //   sample B: 2000 + 3000 = 5000   (a thread exited; 3000 -> signal)
    //   delta = 4000  (without the signal fold it would read 1000)
    let drained = vec![
        entry("periodic_000", 100, 1000, Some(0)),
        entry("periodic_001", 200, 2000, Some(3000)),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    assert_eq!(
        step0.metrics.get("system_time_ns").copied(),
        Some(4000.0),
        "signal accumulator (3000) folds into the group total: \
         (2000+3000) - (1000+0) = 4000",
    );
}

/// A `None` signal_field is an unreadable signal_struct (translate miss),
/// NOT a real zero. A group whose signal is None at one of its endpoints
/// is EXCLUDED from the delta there (its full thread_group_cputime is
/// unmeasurable) rather than counted live-only — otherwise a None-at-first
/// / Some(large)-at-last pair would leak the cumulative accumulator as a
/// phantom positive. Pins that the phantom does not leak.
#[test]
fn build_phase_buckets_cpu_time_excludes_group_with_unreadable_signal() {
    use crate::monitor::task_enrichment::TaskEnrichment;

    fn t(tgid: i32, stime: u64, signal_stime: Option<u64>) -> TaskEnrichment {
        TaskEnrichment {
            pid: tgid,
            tgid,
            stime,
            signal_stime,
            ..Default::default()
        }
    }
    fn entry(tag: &str, elapsed_ms: u64, tasks: Vec<TaskEnrichment>) -> DrainedSnapshotEntry {
        DrainedSnapshotEntry {
            tag: tag.to_string(),
            report: FailureDumpReport {
                schema: SCHEMA_SINGLE.to_string(),
                task_enrichments: tasks,
                ..Default::default()
            },
            stats: Err(MissingStatsReason::NoSchedulerBinary),
            elapsed_ms: Some(elapsed_ms),
            boundary_offset_ms: None,
            step_index: Some(1),
        }
    }

    // tgid=100: signal UNREADABLE (None) at the first sample, Some(5e6) at
    //   the last, live stime flat (1000) -> the None endpoint is omitted,
    //   leaving one readable sample -> EXCLUDED (must not add 5e6).
    // tgid=200: signal Some(0) both samples, stime 1000 -> 3000 -> 2000.
    let drained = vec![
        entry(
            "periodic_000",
            100,
            vec![t(100, 1000, None), t(200, 1000, Some(0))],
        ),
        entry(
            "periodic_001",
            200,
            vec![t(100, 1000, Some(5_000_000)), t(200, 3000, Some(0))],
        ),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    assert_eq!(
        step0.metrics.get("system_time_ns").copied(),
        Some(2000.0),
        "only tgid200 (readable at both) contributes (3000-1000=2000); \
         tgid100's unreadable-signal first endpoint excludes it — the 5e6 \
         accumulator must NOT leak as a phantom positive",
    );
}

/// A numeric tgid reused within the phase (process exit + PID realloc to a
/// fresh group starting near zero) reads a LOWER thread_group_cputime at
/// the last sample than the first; `saturating_sub` clamps the per-group
/// delta to 0 rather than wrapping u128 to a phantom huge value. The group
/// still qualifies (two readable samples), so the result is a real
/// `Some(0.0)`, not absent.
#[test]
fn build_phase_buckets_cpu_time_clamps_counter_decrease() {
    use crate::monitor::task_enrichment::TaskEnrichment;
    fn entry(tag: &str, elapsed_ms: u64, stime: u64) -> DrainedSnapshotEntry {
        DrainedSnapshotEntry {
            tag: tag.to_string(),
            report: FailureDumpReport {
                schema: SCHEMA_SINGLE.to_string(),
                task_enrichments: vec![TaskEnrichment {
                    pid: 100,
                    tgid: 100,
                    stime,
                    signal_stime: Some(0),
                    ..Default::default()
                }],
                ..Default::default()
            },
            stats: Err(MissingStatsReason::NoSchedulerBinary),
            elapsed_ms: Some(elapsed_ms),
            boundary_offset_ms: None,
            step_index: Some(1),
        }
    }
    // stime decreases 5000 -> 1000 across the phase (tgid reuse).
    let drained = vec![
        entry("periodic_000", 100, 5000),
        entry("periodic_001", 200, 1000),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    assert_eq!(
        step0.metrics.get("system_time_ns").copied(),
        Some(0.0),
        "a last < first read clamps to 0 (qualifying group, counters did \
         not advance) — a real Some(0.0), never a wrapped huge value",
    );
}

/// Two readable samples whose tgid sets are DISJOINT (every group appears
/// in exactly one sample) yield no group with two readable boundaries, so
/// the key is ABSENT (unmeasurable) — distinct from a real `Some(0.0)`.
#[test]
fn build_phase_buckets_cpu_time_disjoint_groups_yield_absent_not_zero() {
    use crate::monitor::task_enrichment::TaskEnrichment;
    fn entry(tag: &str, elapsed_ms: u64, tgid: i32) -> DrainedSnapshotEntry {
        DrainedSnapshotEntry {
            tag: tag.to_string(),
            report: FailureDumpReport {
                schema: SCHEMA_SINGLE.to_string(),
                task_enrichments: vec![TaskEnrichment {
                    pid: tgid,
                    tgid,
                    stime: 1000,
                    signal_stime: Some(0),
                    ..Default::default()
                }],
                ..Default::default()
            },
            stats: Err(MissingStatsReason::NoSchedulerBinary),
            elapsed_ms: Some(elapsed_ms),
            boundary_offset_ms: None,
            step_index: Some(1),
        }
    }
    // sample A: only tgid=100; sample B: only tgid=200 (disjoint sets).
    let drained = vec![
        entry("periodic_000", 100, 100),
        entry("periodic_001", 200, 200),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    assert!(
        !step0.metrics.contains_key("system_time_ns"),
        "disjoint groups -> no group spans two readable samples -> absent, \
         not a sentinel Some(0.0)",
    );
}

/// An unanchored sample (no boundary offset AND no measured `elapsed_ms`)
/// must sort LAST in the per-group CPU-time delta, never
/// first: with the pre-Option behavior it coerced to `0` and became the
/// spurious earliest `first_seen` endpoint, so a high cumulative counter
/// in that sample masked real in-phase growth (last <= first ->
/// saturating_sub -> 0). Pins that the unanchored sample sorts to MAX so
/// `first_seen` comes from the earliest TIMED sample.
#[test]
fn build_phase_buckets_cpu_time_unanchored_sample_sorts_last_not_first() {
    use crate::monitor::task_enrichment::TaskEnrichment;
    fn task(stime: u64) -> TaskEnrichment {
        TaskEnrichment {
            pid: 100,
            tgid: 100,
            stime,
            signal_stime: Some(0),
            ..Default::default()
        }
    }
    fn entry(tag: &str, elapsed_ms: Option<u64>, stime: u64) -> DrainedSnapshotEntry {
        DrainedSnapshotEntry {
            tag: tag.to_string(),
            report: FailureDumpReport {
                schema: SCHEMA_SINGLE.to_string(),
                task_enrichments: vec![task(stime)],
                ..Default::default()
            },
            stats: Err(MissingStatsReason::NoSchedulerBinary),
            elapsed_ms,
            boundary_offset_ms: None,
            step_index: Some(1),
        }
    }
    // tgid=100, signal Some(0) -> group total == live stime. Drain order
    // leads with the UNANCHORED sample (elapsed None, stime 3000) to
    // simulate an unsorted grouped vec; the timed samples grow 1000 ->
    // 3000.
    //   * fix (None -> u64::MAX): sort [A(100), B(200), X(None)] ->
    //     first_seen = 1000 (A), last_seen = 3000 -> delta = 2000.
    //   * bug (None -> 0): sort [X, A(100), B(200)] -> first_seen = 3000
    //     (X), last_seen = 3000 -> saturating_sub -> 0 (real growth lost).
    let drained = vec![
        entry("on_demand_000", None, 3000),
        entry("periodic_001", Some(100), 1000),
        entry("periodic_002", Some(200), 3000),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    assert_eq!(
        step0.metrics.get("system_time_ns").copied(),
        Some(2000.0),
        "unanchored sample sorts LAST -> first_seen from the earliest \
         timed sample (1000), delta = 2000; if it sorted first the delta \
         would clamp to 0",
    );
}

/// A phase whose every sample is unanchored (no boundary offset AND no
/// measured `elapsed_ms`) yields the inverted window
/// `(start_ms = u64::MAX, end_ms = 0)`: no sample contributes a time
/// anchor, so `lo`/`hi` keep their identity seeds. The half-open window
/// test then folds zero monitor samples — the correct "no placeable
/// samples" outcome — rather than coercing the missing anchors to 0 and
/// over-folding from the run start.
#[test]
fn build_phase_buckets_all_unanchored_phase_yields_inverted_window() {
    fn unanchored(tag: &str) -> DrainedSnapshotEntry {
        DrainedSnapshotEntry {
            tag: tag.to_string(),
            report: fixture_report(),
            stats: Err(MissingStatsReason::NoSchedulerBinary),
            elapsed_ms: None,
            boundary_offset_ms: None,
            step_index: Some(1),
        }
    }
    let drained = vec![unanchored("on_demand_000"), unanchored("on_demand_001")];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("phase bucket present even when every sample is unanchored");
    assert_eq!(step0.sample_count, 2, "both unanchored samples counted");
    assert_eq!(
        (step0.start_ms, step0.end_ms),
        (u64::MAX, 0),
        "all-unanchored phase -> inverted window that folds nothing, \
         not a (0, x) window anchored at the run start",
    );
}

/// `ScenarioStats::phase` lookup against the phases built by
/// `build_phase_buckets` returns the bucket whose step_index
/// matches, not by vec position. Confirms the integration
/// between the aggregator output and the accessor surface from
/// step 1.
#[test]
fn build_phase_buckets_integration_with_scenario_stats_phase_accessor() {
    let drained = vec![
        fixture_entry("periodic_000", 0, 10),
        fixture_entry("periodic_001", 1, 100),
        fixture_entry("periodic_002", 2, 200),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    let stats = ScenarioStats {
        phases,
        ..Default::default()
    };
    assert_eq!(stats.phase(0).map(|p| p.label.as_str()), Some("BASELINE"));
    assert_eq!(stats.phase(1).map(|p| p.label.as_str()), Some("Step[0]"));
    assert_eq!(stats.phase(2).map(|p| p.label.as_str()), Some("Step[1]"));
    assert_eq!(stats.phase(3), None);
    // `step(0)` is the scenario-side 0-indexed accessor: maps to
    // phase index 1 (scenario Step 0 lives at step_index 1).
    assert_eq!(stats.step(0).map(|p| p.label.as_str()), Some("Step[0]"));
    assert_eq!(stats.step(1).map(|p| p.label.as_str()), Some("Step[1]"));
}

// ---------- Phase newtype ----------

#[test]
fn phase_baseline_const_is_u16_zero() {
    assert_eq!(crate::assert::Phase::BASELINE.as_u16(), 0);
    assert!(crate::assert::Phase::BASELINE.is_baseline());
}

#[test]
fn phase_step_zero_indexed_constructor_encodes_1_indexed() {
    assert_eq!(crate::assert::Phase::step(0).as_u16(), 1);
    assert_eq!(crate::assert::Phase::step(1).as_u16(), 2);
    assert_eq!(crate::assert::Phase::step(7).as_u16(), 8);
    assert!(!crate::assert::Phase::step(0).is_baseline());
}

#[test]
fn phase_step_saturating_at_u16_max_does_not_overflow() {
    // u16::MAX - 1 + 1 saturates to u16::MAX rather than wrapping
    // to 0 (which would collide with BASELINE).
    let saturated = crate::assert::Phase::step(u16::MAX - 1);
    assert_eq!(saturated.as_u16(), u16::MAX);
    let still_saturated = crate::assert::Phase::step(u16::MAX);
    assert_eq!(still_saturated.as_u16(), u16::MAX);
    assert!(
        !saturated.is_baseline(),
        "saturating MUST NOT collide with BASELINE"
    );
}

#[test]
fn phase_display_baseline_step_format() {
    assert_eq!(format!("{}", crate::assert::Phase::BASELINE), "BASELINE");
    assert_eq!(format!("{}", crate::assert::Phase::step(0)), "Step[0]");
    assert_eq!(format!("{}", crate::assert::Phase::step(7)), "Step[7]");
}

#[test]
fn phase_serde_transparent_round_trips_as_bare_u16() {
    // #[serde(transparent)] means the wire format is the inner
    // u16, not a tagged struct. Pin both directions: a Phase
    // serializes as a JSON number, and a bare JSON number
    // deserializes as a Phase.
    let phase = crate::assert::Phase::step(4);
    let json = serde_json::to_string(&phase).unwrap();
    assert_eq!(
        json, "5",
        "wire format must be the inner 1-indexed u16, not a tagged struct"
    );
    let round_tripped: crate::assert::Phase = serde_json::from_str(&json).unwrap();
    assert_eq!(round_tripped, phase);
    // And the reverse: a raw number deserializes as a Phase.
    let from_raw: crate::assert::Phase = serde_json::from_str("0").unwrap();
    assert_eq!(from_raw, crate::assert::Phase::BASELINE);
}

#[test]
fn phase_from_u16_wraps_raw_value() {
    let from: crate::assert::Phase = 3u16.into();
    assert_eq!(from.as_u16(), 3);
    let to: u16 = crate::assert::Phase::step(2).into();
    assert_eq!(to, 3);
}

// ---------- ScenarioStats::has_steps ----------

#[test]
fn scenario_stats_has_steps_false_for_empty_phases() {
    let stats = ScenarioStats::default();
    assert!(!stats.has_steps());
}

#[test]
fn scenario_stats_has_steps_false_when_only_baseline() {
    let stats = ScenarioStats {
        phases: vec![crate::assert::PhaseBucket {
            step_index: 0,
            label: "BASELINE".to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };
    assert!(
        !stats.has_steps(),
        "BASELINE-only must NOT count as 'has steps'"
    );
}

#[test]
fn scenario_stats_has_steps_true_when_any_step_phase_present() {
    let stats = ScenarioStats {
        phases: vec![
            crate::assert::PhaseBucket {
                step_index: 0,
                label: "BASELINE".to_string(),
                ..Default::default()
            },
            crate::assert::PhaseBucket {
                step_index: 1,
                label: "Step[0]".to_string(),
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    assert!(stats.has_steps());
}

// ---------- PhaseBucket::expect_metric ----------

#[test]
#[should_panic(expected = "metric 'missing' absent from phase step_index=1")]
fn phase_bucket_expect_metric_panics_with_diagnostic_when_absent() {
    let bucket = crate::assert::PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        sample_count: 3,
        metrics: std::collections::BTreeMap::from([("throughput".to_string(), 42.0)]),
        ..Default::default()
    };
    bucket.expect_metric("missing");
}

#[test]
fn phase_bucket_expect_metric_returns_value_when_present() {
    let bucket = crate::assert::PhaseBucket {
        metrics: std::collections::BTreeMap::from([("throughput".to_string(), 42.5)]),
        ..Default::default()
    };
    assert_eq!(bucket.expect_metric("throughput"), 42.5);
}

// ---------- PhaseGuard (RAII auto-stamp) ----------

#[test]
fn phase_guard_outside_scope_returns_none() {
    // No guard installed → current_phase_label is None and a
    // freshly-constructed AssertDetail inherits None.
    assert!(crate::assert::current_phase_label().is_none());
    let d = crate::assert::AssertDetail::new(crate::assert::DetailKind::Other, "no guard");
    assert!(
        d.phase.is_none(),
        "AssertDetail constructed outside any PhaseGuard must stamp phase=None"
    );
}

#[test]
fn phase_guard_install_step_sets_active_label() {
    let _g = crate::assert::PhaseGuard::install_step(0);
    assert_eq!(
        crate::assert::current_phase_label().as_deref(),
        Some("Step[0]"),
    );
    let d = crate::assert::AssertDetail::new(crate::assert::DetailKind::Other, "under Step[0]");
    assert_eq!(d.phase.as_deref(), Some("Step[0]"));
}

#[test]
fn phase_guard_install_baseline_sets_active_label() {
    let _g = crate::assert::PhaseGuard::install_baseline();
    assert_eq!(
        crate::assert::current_phase_label().as_deref(),
        Some("BASELINE"),
    );
}

#[test]
fn phase_guard_drop_restores_prior_label() {
    {
        let _outer = crate::assert::PhaseGuard::install_step(0); // "Step[0]"
        assert_eq!(
            crate::assert::current_phase_label().as_deref(),
            Some("Step[0]"),
        );
        {
            let _inner = crate::assert::PhaseGuard::install_step(2); // "Step[2]"
            assert_eq!(
                crate::assert::current_phase_label().as_deref(),
                Some("Step[2]"),
            );
        } // inner drops → restore Step[0]
        assert_eq!(
            crate::assert::current_phase_label().as_deref(),
            Some("Step[0]"),
            "inner guard's Drop must restore the outer guard's label",
        );
    } // outer drops → restore None
    assert!(
        crate::assert::current_phase_label().is_none(),
        "outermost guard's Drop must restore None",
    );
}

#[test]
fn phase_guard_passdetail_binary_auto_stamps() {
    let _g = crate::assert::PhaseGuard::install_step(1);
    let p = crate::assert::PassDetail::binary("metric", "ge", "10.0", "5.0");
    assert_eq!(p.phase.as_deref(), Some("Step[1]"));
}

#[test]
fn phase_guard_passdetail_unary_auto_stamps() {
    let _g = crate::assert::PhaseGuard::install_step(2);
    let p = crate::assert::PassDetail::unary("metric", "is_finite", "42.0");
    assert_eq!(p.phase.as_deref(), Some("Step[2]"));
}

#[test]
fn phase_guard_infonote_auto_stamps() {
    let _g = crate::assert::PhaseGuard::install_baseline();
    let n = crate::assert::InfoNote::new("settle observed");
    assert_eq!(n.phase.as_deref(), Some("BASELINE"));
}

#[test]
fn phase_guard_with_phase_builder_overrides_auto_stamp() {
    let _g = crate::assert::PhaseGuard::install_step(0); // "Step[0]"
    let d = crate::assert::AssertDetail::new(crate::assert::DetailKind::Other, "override")
        .with_phase("explicit_override");
    assert_eq!(
        d.phase.as_deref(),
        Some("explicit_override"),
        "with_phase builder must override the auto-stamp default",
    );
}

/// `populate_run_ext_metrics` is a no-op for an empty SampleSeries:
/// `read_sample` returns `None` for every registered metric on the
/// empty fixture (no DSQ states, no event counters), so nothing
/// lands in `ext_metrics`. Pins the contract that the helper
/// never synthesises sentinel zeros from no-data input.
#[test]
fn populate_run_ext_metrics_empty_series_inserts_nothing() {
    let samples = SampleSeries::from_drained_typed(Vec::new(), None);
    let mut target = std::collections::BTreeMap::new();
    crate::assert::populate_run_ext_metrics(&samples, &mut target);
    assert!(
        target.is_empty(),
        "no input samples must produce no ext_metrics entries, got {target:?}",
    );
}

/// `populate_run_ext_metrics` never overwrites a key already
/// present in `target` — a typed GauntletRow field that produced
/// a value via the MetricDef accessor stays untouched. Pins the
/// "fill the gap, never clobber" contract: cross-RUN comparison
/// expects the typed-field value when present and the
/// helper-computed value only when not.
#[test]
fn populate_run_ext_metrics_does_not_overwrite_existing_keys() {
    let samples = SampleSeries::from_drained_typed(Vec::new(), None);
    let mut target = std::collections::BTreeMap::new();
    target.insert("avg_dsq_depth".to_string(), 42.0);
    crate::assert::populate_run_ext_metrics(&samples, &mut target);
    assert_eq!(
        target.get("avg_dsq_depth").copied(),
        Some(42.0),
        "existing key must survive populate_run_ext_metrics",
    );
}

/// `build_phase_buckets` populates `avg_imbalance_ratio` from
/// MonitorSamples windowed by phase. Synthesised samples land in
/// the Step[0] window; the per-phase mean of their
/// `imbalance_ratio()` readings (max(nr_running)/max(1, min(nr_running))
/// per CPU) is stamped on PhaseBucket.metrics. Confirms the F-A
/// fix wiring: imbalance now flows through PhaseBucket per-phase
/// rather than only at the run-aggregate MonitorSummary level.
#[test]
fn build_phase_buckets_avg_imbalance_ratio_from_monitor_samples() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    // Three monitor samples covering [50..250 ms]. Sample CPU vecs
    // produce known imbalance ratios:
    //   s_50:  cpus=[nr=2, nr=2] -> ratio = 2 / max(1, 2) = 1.0
    //   s_100: cpus=[nr=4, nr=2] -> ratio = 4 / max(1, 2) = 2.0
    //   s_200: cpus=[nr=6, nr=2] -> ratio = 6 / max(1, 2) = 3.0
    // Mean across all three = (1.0 + 2.0 + 3.0) / 3 = 2.0
    let cpu = |nr: u32| CpuSnapshot {
        nr_running: nr,
        ..Default::default()
    };
    let mon = MonitorReport {
        samples: vec![
            MonitorSample::new(50, vec![cpu(2), cpu(2)]),
            MonitorSample::new(100, vec![cpu(4), cpu(2)]),
            MonitorSample::new(200, vec![cpu(6), cpu(2)]),
        ],
        ..Default::default()
    };
    // Two snapshot bridge entries fence the Step[0] window at
    // elapsed_ms [50..250]; all three monitor samples land inside.
    let drained = vec![
        fixture_entry("periodic_000", 1, 50),
        fixture_entry("periodic_001", 1, 250),
    ];
    let samples = SampleSeries::from_drained_typed(drained, Some(mon));
    let phases = crate::assert::build_phase_buckets(&samples);
    assert_eq!(phases.len(), 1, "single phase from two same-step samples");
    let step0 = &phases[0];
    let avg = step0
        .metrics
        .get("avg_imbalance_ratio")
        .copied()
        .expect("avg_imbalance_ratio must be populated from MonitorSamples");
    assert!(
        (avg - 2.0).abs() < f64::EPSILON,
        "expected mean = 2.0, got {avg}",
    );
}

/// MonitorSamples whose elapsed_ms falls OUTSIDE the phase window
/// (`[start_ms, end_ms]`) are excluded from the avg_imbalance_ratio
/// reduction. A sample at elapsed_ms = 9999 with a wildly
/// different imbalance must not contaminate the in-window mean.
#[test]
fn build_phase_buckets_avg_imbalance_excludes_out_of_window_monitor_samples() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    let cpu = |nr: u32| CpuSnapshot {
        nr_running: nr,
        ..Default::default()
    };
    let mon = MonitorReport {
        samples: vec![
            MonitorSample::new(100, vec![cpu(4), cpu(2)]),
            MonitorSample::new(150, vec![cpu(4), cpu(2)]),
            MonitorSample::new(200, vec![cpu(4), cpu(2)]),
            MonitorSample::new(9999, vec![cpu(100), cpu(2)]),
        ],
        ..Default::default()
    };
    let drained = vec![
        fixture_entry("periodic_000", 1, 100),
        fixture_entry("periodic_001", 1, 200),
    ];
    let samples = SampleSeries::from_drained_typed(drained, Some(mon));
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = &phases[0];
    let avg = step0
        .metrics
        .get("avg_imbalance_ratio")
        .copied()
        .expect("avg_imbalance_ratio populated");
    assert!(
        (avg - 2.0).abs() < f64::EPSILON,
        "out-of-window sample must not contaminate in-window mean (got {avg})",
    );
    // max_imbalance_ratio now folds on captured buckets too, so the
    // elapsed=9999 outlier (imbalance 100/2 = 50) is out-of-window and must
    // be excluded, so the per-phase Peak is the in-window max (all 2.0), NOT
    // 50. Pins that the windowing gate applies to the newly-folded key.
    let max_imb = step0
        .metrics
        .get("max_imbalance_ratio")
        .copied()
        .expect("max_imbalance_ratio populated on captured bucket");
    assert!(
        (max_imb - 2.0).abs() < f64::EPSILON,
        "out-of-window outlier (50) must be excluded from max_imbalance_ratio (got {max_imb})",
    );
}

/// A CAPTURED bucket (`sample_count > 0`) must carry
/// `max_imbalance_ratio` (Peak) and `stuck_count` (Counter) folded from its
/// in-window monitor samples. Neither has a `read_sample` dispatch arm (both
/// fall to `_ => None`), so before the gate fix they surfaced ONLY on
/// synthesized buckets and a captured (common-case) phase dropped them.
/// max_imbalance = max sample ratio; stuck_count = consecutive
/// frozen-`rq_clock` non-idle CPU stalls.
#[test]
fn build_phase_buckets_captured_bucket_carries_max_imbalance_and_stuck() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    // cpu(nr, clk): nr_running + a frozen rq_clock so two consecutive
    // samples sharing the SAME non-zero clock register a per-CPU stall.
    let cpu = |nr: u32, clk: u64| CpuSnapshot {
        nr_running: nr,
        rq_clock: clk,
        ..Default::default()
    };
    // s_60:  [nr=4, nr=2] -> imbalance 4 / max(1,2) = 2.0; rq_clock 1000
    // s_120: [nr=6, nr=2] -> imbalance 6 / max(1,2) = 3.0; rq_clock 1000
    //   frozen vs s_60 + both CPUs non-idle -> stall_count = 2.
    let mon = MonitorReport {
        samples: vec![
            MonitorSample::new(60, vec![cpu(4, 1000), cpu(2, 1000)]),
            MonitorSample::new(120, vec![cpu(6, 1000), cpu(2, 1000)]),
        ],
        ..Default::default()
    };
    // Two snapshot captures fence the Step[0] window at [50..250] and make
    // the bucket CAPTURED (sample_count == 2), not synthesized.
    let drained = vec![
        fixture_entry("periodic_000", 1, 50),
        fixture_entry("periodic_001", 1, 250),
    ];
    let samples = SampleSeries::from_drained_typed(drained, Some(mon));
    let phases = crate::assert::build_phase_buckets(&samples);
    assert_eq!(phases.len(), 1, "single phase");
    let step0 = &phases[0];
    assert!(
        step0.sample_count > 0,
        "bucket must be CAPTURED (not synthesized) for this test to mean anything",
    );
    let max_imb = step0
        .metrics
        .get("max_imbalance_ratio")
        .copied()
        .expect("captured bucket must now carry max_imbalance_ratio");
    assert!(
        (max_imb - 3.0).abs() < f64::EPSILON,
        "max imbalance = max(2.0, 3.0) = 3.0, got {max_imb}",
    );
    let stuck = step0
        .metrics
        .get("stuck_count")
        .copied()
        .expect("captured bucket must now carry stuck_count");
    assert!(
        (stuck - 2.0).abs() < f64::EPSILON,
        "two frozen-clock non-idle CPUs => stall_count 2, got {stuck}",
    );
}

/// Boundary: `stuck_count` needs `windows(2)` of monitor samples, so
/// a phase with a single in-window monitor sample has NO stall to report —
/// `stuck_count` must be ABSENT (not present as 0), while `max_imbalance_ratio`
/// (computed from the single sample) is still folded. Pins that the
/// `if pm.stall_count > 0` gate leaves the key out rather than writing a
/// misleading zero.
#[test]
fn build_phase_buckets_single_in_window_sample_has_no_stuck_count() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    let cpu = |nr: u32, clk: u64| CpuSnapshot {
        nr_running: nr,
        rq_clock: clk,
        ..Default::default()
    };
    // ONE in-window sample: imbalance 4/2 = 2.0; no second sample => no
    // windows(2) => stall_count 0 => stuck_count not written.
    let mon = MonitorReport {
        samples: vec![MonitorSample::new(100, vec![cpu(4, 1000), cpu(2, 1000)])],
        ..Default::default()
    };
    let drained = vec![
        fixture_entry("periodic_000", 1, 50),
        fixture_entry("periodic_001", 1, 250),
    ];
    let samples = SampleSeries::from_drained_typed(drained, Some(mon));
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = &phases[0];
    let max_imb = step0
        .metrics
        .get("max_imbalance_ratio")
        .copied()
        .expect("single in-window sample still yields max_imbalance_ratio");
    assert!(
        (max_imb - 2.0).abs() < f64::EPSILON,
        "max imbalance = 2.0 from the single sample, got {max_imb}",
    );
    assert!(
        !step0.metrics.contains_key("stuck_count"),
        "no consecutive-sample stall => stuck_count must be ABSENT, not 0: {:?}",
        step0.metrics.get("stuck_count"),
    );
}

/// Boundary: stuck_count scopes to in-window samples. The in_window
/// filter runs BEFORE compute_metrics' `windows(2)` stall detection, so a
/// stall pair never forms across the phase edge — an out-of-window sample
/// cannot pair with the last in-window sample. Two frozen-clock in-window
/// samples plus one far-out sample must yield stuck_count = 2 (the single
/// fully-in-window pair x 2 CPUs), NOT 4. Guards against a regression that
/// windowed AFTER stall computation and mis-attributed a cross-phase stall.
#[test]
fn build_phase_buckets_stuck_count_excludes_cross_window_stall_pair() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    let cpu = |nr: u32, clk: u64| CpuSnapshot {
        nr_running: nr,
        rq_clock: clk,
        ..Default::default()
    };
    // s_100 + s_200 are in [50,250); their frozen rq_clock pairs them ->
    // stall_count 2 (both CPUs). s_9999 is OUT of window; it is filtered
    // before compute_metrics, so the (s_200, s_9999) pair never forms. If
    // windowing ran AFTER stall detection, that pair would add 2 more.
    let mon = MonitorReport {
        samples: vec![
            MonitorSample::new(100, vec![cpu(4, 1000), cpu(2, 1000)]),
            MonitorSample::new(200, vec![cpu(6, 1000), cpu(2, 1000)]),
            MonitorSample::new(9999, vec![cpu(8, 1000), cpu(2, 1000)]),
        ],
        ..Default::default()
    };
    let drained = vec![
        fixture_entry("periodic_000", 1, 50),
        fixture_entry("periodic_001", 1, 250),
    ];
    let samples = SampleSeries::from_drained_typed(drained, Some(mon));
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = &phases[0];
    let stuck = step0
        .metrics
        .get("stuck_count")
        .copied()
        .expect("in-window stall pair yields stuck_count");
    assert!(
        (stuck - 2.0).abs() < f64::EPSILON,
        "only the fully-in-window pair counts => stuck_count 2 (not 4 with a cross-window pair), got {stuck}",
    );
}

/// Tester B14 BLOCKING: avg_dsq_depth end-to-end pin through
/// the registry → build_phase_buckets → PhaseBucket.metrics
/// path. Without this, a regression where the read_sample
/// dispatch arm at src/stats.rs returns None silently produces
/// an empty per-phase entry — operator-visible drop. Synthetic
/// Snapshot DSQ states produce a known mean across local-cpu
/// entries.
#[test]
fn build_phase_buckets_avg_dsq_depth_from_snapshot_dsq_states() {
    use crate::monitor::dump::FailureDumpReport;
    use crate::monitor::scx_walker::DsqState;
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    // Two periodic captures, each carrying 3 local-cpu DSQ
    // states with depths 2/4/6 → per-sample mean 4.0. Two
    // identical samples → per-phase mean 4.0.
    let mk_entry = |tag: &str, ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            dsq_states: vec![
                DsqState {
                    origin: "local cpu 0".to_string(),
                    nr: 2,
                    ..Default::default()
                },
                DsqState {
                    origin: "local cpu 1".to_string(),
                    nr: 4,
                    ..Default::default()
                },
                DsqState {
                    origin: "local cpu 2".to_string(),
                    nr: 6,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(ms),
        boundary_offset_ms: None,
        step_index: Some(1),
    };
    let drained = vec![mk_entry("periodic_000", 100), mk_entry("periodic_001", 200)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    let avg = step0
        .metrics
        .get("avg_dsq_depth")
        .copied()
        .expect("avg_dsq_depth populated from local-cpu DSQ states");
    assert!(
        (avg - 4.0).abs() < f64::EPSILON,
        "expected per-phase avg of mean(2,4,6)=4.0, got {avg}",
    );
    // Also verify max_dsq_depth shipped correctly through the
    // same DSQ-walker axis.
    let max = step0
        .metrics
        .get("max_dsq_depth")
        .copied()
        .expect("max_dsq_depth populated alongside avg");
    assert!(
        (max - 6.0).abs() < f64::EPSILON,
        "expected max=6.0, got {max}"
    );
}

/// Tester B15 BLOCKING: iteration_rate per-phase population via
/// build_phase_buckets_with_stimulus. Synthetic StimulusEvents
/// with total_iterations deltas at known boundaries produce a
/// known per-phase rate.
#[test]
fn build_phase_buckets_with_stimulus_populates_iteration_rate() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    // Snapshot bridge entries fence two Step windows: Step[0]
    // at [100, 1100], Step[1] at [1100, 2100]. Stimulus events
    // carry total_iterations at each boundary. iteration_rate
    // for Step[1] (curr.elapsed_ms=2100, prev.elapsed_ms=1100,
    // iter delta 2000) → 2000 / (1000ms/1000) = 2000.0/s.
    let mk_entry = |tag: &str, step: u16, ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(ms),
        boundary_offset_ms: None,
        step_index: Some(step),
    };
    let drained = vec![
        mk_entry("periodic_000", 1, 100),
        mk_entry("periodic_001", 1, 1100),
        mk_entry("periodic_002", 2, 1100),
        mk_entry("periodic_003", 2, 2100),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let stimulus = vec![
        StimulusEvent {
            elapsed_ms: 100,
            label: "Step[0]".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: Some(0),
            step_index: None,
            is_terminal: false,
            is_step_end: false,
        },
        StimulusEvent {
            elapsed_ms: 1100,
            label: "Step[1]".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: Some(1000),
            step_index: None,
            is_terminal: false,
            is_step_end: false,
        },
        StimulusEvent {
            elapsed_ms: 2100,
            label: "end".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: Some(3000),
            step_index: None,
            is_terminal: false,
            is_step_end: false,
        },
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("Step[1] bucket present");
    let rate = step1
        .metrics
        .get("iteration_rate")
        .copied()
        .expect("iteration_rate populated for Step[1]");
    assert!(
        (rate - 2000.0).abs() < f64::EPSILON,
        "expected iteration_rate=2000.0 iter/s, got {rate}",
    );
}

/// The deferred-fire fix. When the dump-prerequisite gate holds the
/// periodic boundaries until the accessor adopts, they fire in a burst
/// and every capture reads the same late `CURRENT_STEP`, so the stamped
/// step_index collapses to one value (the `phases.len() == 1` bug). The
/// workload-relative `boundary_offset_ms` — computed from the boundary
/// schedule, not the fire time — must instead drive attribution. Four
/// captures all stamped step_index=3 but scheduled across BASELINE +
/// three step windows must land in four distinct buckets.
#[test]
fn build_phase_buckets_with_stimulus_remaps_by_boundary_offset_over_stamped_step() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    // All four stamp step_index=3 (the burst bug) and share a ~uniform
    // run-relative fire time (elapsed_ms), but their SCHEDULED offsets
    // fall before step 1 (BASELINE) and inside steps 1, 2, 3.
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(3),
    };
    let drained = vec![
        mk("periodic_base", 500),  // before step 1 start (1000) -> BASELINE
        mk("periodic_000", 1_500), // step 1 window [1000, 2000)
        mk("periodic_001", 2_500), // step 2 window [2000, 3000)
        mk("periodic_002", 3_500), // step 3 window [3000, ..)
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    // Step-start timeline (scenario-relative): step k starts at k*1000 ms.
    let stim = |elapsed_ms: u64, k: u16| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let stimulus = vec![stim(1000, 1), stim(2000, 2), stim(3000, 3)];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let idxs: Vec<u16> = phases.iter().map(|p| p.step_index).collect();
    assert_eq!(
        idxs,
        vec![0, 1, 2, 3],
        "boundary_offset_ms must drive grouping (BASELINE + one capture \
         per step), NOT the uniformly-wrong stamped step_index=3 which \
         would collapse all four into a single bucket; got {idxs:?}",
    );
    for p in &phases {
        assert_eq!(
            p.sample_count, 1,
            "each remapped bucket holds exactly its one scheduled capture; \
             step_index={} count={}",
            p.step_index, p.sample_count,
        );
    }
}

/// #4 residual fix: a scenario step whose periodic-boundary window
/// captured ZERO samples (the uniform whole-workload boundary placement
/// skipped it) still gets a PhaseBucket carrying its capture-independent
/// iteration_rate. Steps 1 and 3 capture; step 2 captures nothing. The
/// stimulus carries a StepStart per step with total_iterations, so step
/// 2's rate (StepStart[2]=1000 -> StepStart[3]=2000 over 1000 ms =
/// 1000/s) is measurable from the timeline alone. Before the synthesize
/// seam, step 2 produced no bucket and its rate was silently dropped.
/// Also pins dedup: a captured step is never given a duplicate bucket.
#[test]
fn build_phase_buckets_with_stimulus_synthesizes_zero_capture_step_bucket() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    // Captures: one in step 1's window, one in step 3's window. NONE in
    // step 2's window [2000, 3000).
    let cap = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(1),
    };
    let drained = vec![cap("periodic_000", 1_500), cap("periodic_001", 3_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    // StepStart[k] at k*1000 ms with cumulative iterations; the rate loop
    // pairs consecutive starts (delta iters / delta s).
    let start = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let stimulus = vec![
        start(1000, 1, 0),
        start(2000, 2, 1000),
        start(3000, 3, 2000),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);

    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("zero-capture step 2 must still produce a synthesized bucket");
    assert_eq!(step2.sample_count, 0, "synthesized bucket is capture-free");
    let rate = step2
        .metrics
        .get("iteration_rate")
        .copied()
        .expect("step 2's capture-independent iteration_rate must be recovered");
    assert!(
        (rate - 1000.0).abs() < f64::EPSILON,
        "step 2 rate = (2000-1000) iters / 1000 ms = 1000/s, got {rate}",
    );
    // Dedup: each captured step keeps a single bucket (no duplicate
    // synthesized bucket for steps 1 and 3) and the vec is step-sorted.
    let idxs: Vec<u16> = phases.iter().map(|p| p.step_index).collect();
    assert_eq!(
        idxs,
        vec![1, 2, 3],
        "captured steps 1/3 keep their single bucket; step 2 synthesized; \
         sorted by step_index; got {idxs:?}",
    );
}

/// #4 fuller variant: a synthesized zero-capture step bucket also
/// recovers its monitor-derived `avg_imbalance_ratio` from the
/// MonitorSamples in its window — not just iteration_rate. The anchor
/// monitor sample at the first stimulus elapsed pins
/// `monitor_clock_offset` to 0 so the in-step-2 sample lands in [2000,
/// 3000).
#[test]
fn build_phase_buckets_with_stimulus_synthesized_bucket_folds_monitor_imbalance() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let cpu = |nr: u32| CpuSnapshot {
        nr_running: nr,
        ..Default::default()
    };
    let mon = MonitorReport {
        samples: vec![
            // Anchor (>500 ms) at the first stimulus elapsed -> offset 0.
            MonitorSample::new(1000, vec![cpu(2), cpu(2)]), // step 1, imbalance 1.0
            MonitorSample::new(2500, vec![cpu(6), cpu(2)]), // step 2, imbalance 3.0
        ],
        ..Default::default()
    };
    // Captures in steps 1 and 3 only; step 2 captures nothing.
    let cap = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(1),
    };
    let drained = vec![cap("periodic_000", 1_500), cap("periodic_001", 3_500)];
    let samples = SampleSeries::from_drained_typed(drained, Some(mon));
    let start = |elapsed_ms: u64, k: u16| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let stimulus = vec![start(1000, 1), start(2000, 2), start(3000, 3)];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("synthesized zero-capture step 2 bucket present");
    assert_eq!(step2.sample_count, 0);
    let avg = step2
        .metrics
        .get("avg_imbalance_ratio")
        .copied()
        .expect("synthesized bucket must recover avg_imbalance_ratio from monitor samples");
    assert!(
        (avg - 3.0).abs() < f64::EPSILON,
        "step 2 in-window monitor imbalance = 6 / max(1, 2) = 3.0, got {avg}",
    );
    // The stimulus carried total_iterations: None, so no rate is fabricated.
    assert!(
        !step2.metrics.contains_key("iteration_rate"),
        "None total_iterations must yield NO iteration_rate (no fabrication); got {:?}",
        step2.metrics,
    );
}

/// A single-step run with a capture in its window produces exactly one
/// Step bucket and synthesizes nothing extra: the synthesize loop only
/// fires for a step that has a StepStart but no bucket.
#[test]
fn build_phase_buckets_with_stimulus_single_captured_step_no_spurious_synthesis() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let cap = DrainedSnapshotEntry {
        tag: "periodic_000".to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(1_500),
        step_index: Some(1),
    };
    let samples = SampleSeries::from_drained_typed(vec![cap], None);
    let stimulus = vec![StimulusEvent {
        elapsed_ms: 1000,
        label: "StepStart[1]".to_string(),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(1),
        is_terminal: false,
        is_step_end: false,
    }];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let idxs: Vec<u16> = phases.iter().map(|p| p.step_index).collect();
    assert_eq!(
        idxs,
        vec![1],
        "exactly the one captured step; the synthesize loop adds no extras",
    );
    assert_eq!(phases[0].sample_count, 1);
}

/// A sched-died last step (a StepStart with no StepEnd, no successor
/// start, and no captures) still produces a present-but-empty bucket:
/// open-ended window (u64::MAX), sample_count 0, and NO iteration_rate
/// (rate_to has no right boundary) — no panic, no phantom rate.
#[test]
fn build_phase_buckets_with_stimulus_sched_died_last_step_yields_empty_present_bucket() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let cap = DrainedSnapshotEntry {
        tag: "periodic_000".to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(1_500),
        step_index: Some(1),
    };
    let samples = SampleSeries::from_drained_typed(vec![cap], None);
    let start = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    // step 1 captured; step 2 started then died (no StepEnd, no successor).
    let stimulus = vec![start(1000, 1, 0), start(2000, 2, 1000)];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("sched-died last step still gets a present bucket");
    assert_eq!(step2.sample_count, 0);
    assert_eq!(
        step2.end_ms,
        u64::MAX,
        "no StepEnd / successor start -> open-ended window",
    );
    assert!(
        !step2.metrics.contains_key("iteration_rate"),
        "no right boundary -> no rate (absent, not a phantom 0); got {:?}",
        step2.metrics,
    );
}

/// A synthesized (sample_count == 0) bucket between two captured phases:
/// its STIMULUS-DERIVED throughput (iteration_rate) is a real measurement
/// that survives the capture gap, so a collapse INTO it and a recovery
/// OUT of it ARE flagged (the #12 fix). Its monitor-derived metrics
/// (avg_imbalance) come from a different sampling basis on a zero-sample
/// phase, so they stay SUPPRESSED — a wild synthesized imbalance never
/// flags a phantom change while a real throughput change does. Pins the
/// behavior the pre-#12 blanket sample_count gate got wrong.
#[test]
fn synthesized_zero_sample_bucket_flags_throughput_not_phantom_monitor() {
    use crate::timeline::{ChangeDirection, Timeline, TimelineContext};
    let captured = |k: u16, imbalance: f64, rate: f64| crate::assert::PhaseBucket {
        per_cgroup: Default::default(),
        step_index: k,
        label: format!("Step[{}]", k.saturating_sub(1)),
        start_ms: (k as u64) * 1000,
        end_ms: (k as u64) * 1000 + 1000,
        sample_count: 5,
        metrics: std::collections::BTreeMap::from([
            ("avg_imbalance_ratio".to_string(), imbalance),
            ("iteration_rate".to_string(), rate),
        ]),
    };
    // step 1 + step 3 steady & captured; step 2 synthesized with a WILD
    // imbalance (monitor — must stay gated) AND a real collapsed
    // stimulus-derived throughput (must flag).
    let buckets = vec![
        captured(1, 1.0, 1000.0),
        crate::assert::PhaseBucket {
            per_cgroup: Default::default(),
            step_index: 2,
            label: "Step[1]".to_string(),
            start_ms: 2000,
            end_ms: 3000,
            sample_count: 0,
            metrics: std::collections::BTreeMap::from([
                ("avg_imbalance_ratio".to_string(), 100.0),
                ("iteration_rate".to_string(), 10.0),
            ]),
        },
        captured(3, 1.0, 1000.0),
    ];
    let timeline = Timeline::from_phase_buckets(&buckets, &[], &TimelineContext::default());
    // No phase flags a monitor (imbalance) change — the wild synthesized
    // imbalance is gated off on both boundaries it touches.
    assert!(
        timeline
            .phases
            .iter()
            .all(|p| !p.changes.iter().any(|c| c.metric == "imbalance")),
        "a zero-sample bucket's wild imbalance must not flag a phantom change; got {:?}",
        timeline
            .phases
            .iter()
            .map(|p| (p.index, p.changes.clone()))
            .collect::<Vec<_>>(),
    );
    // Phase index 1 = boundary INTO the synthesized step: throughput
    // 1000 -> 10 is a real collapse, flagged Degraded. Pin the phase
    // index so a reorder can't mis-target the boundary while finding a
    // throughput change of the expected direction by coincidence.
    let into_synth = &timeline.phases[1];
    assert_eq!(
        into_synth.index, 2,
        "phases[1] is step_index 2 (the synthesized step)"
    );
    assert!(
        into_synth
            .changes
            .iter()
            .any(|c| c.metric == "throughput" && c.direction == ChangeDirection::Degraded),
        "throughput collapse into the synthesized step must flag Degraded; got {:?}",
        into_synth.changes,
    );
    // Phase index 2 = boundary OUT of the synthesized step: throughput
    // 10 -> 1000 recovery flagged Improved (before == 10 > 0).
    let out_of_synth = &timeline.phases[2];
    assert_eq!(
        out_of_synth.index, 3,
        "phases[2] is step_index 3 (the captured step after)"
    );
    assert!(
        out_of_synth
            .changes
            .iter()
            .any(|c| c.metric == "throughput" && c.direction == ChangeDirection::Improved),
        "throughput recovery out of the synthesized step must flag Improved; got {:?}",
        out_of_synth.changes,
    );
}

/// Event-sort tie-stability: when StepEnd[k] and StepStart[k+1] share
/// the same elapsed_ms (a zero-length inter-step gap at the guest's
/// coarse-ms clock), the rate attributed to step k must be the
/// step-LOCAL StepStart[k]->StepEnd[k] delta, NOT the cross-step
/// StepStart[k]->StepStart[k+1] delta. The total-ordered sort (StepEnd
/// before StepStart on a tie) guarantees the step-local pairing.
#[test]
fn build_phase_buckets_with_stimulus_step_end_tie_attributes_step_local_rate() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let cap = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(1),
    };
    let drained = vec![cap("periodic_000", 1_500), cap("periodic_001", 2_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    // StepEnd[1] and StepStart[2] BOTH at elapsed 2000 (the tie). Step 1's
    // local delta is 500 iters / 1s = 500/s; the cross-step
    // StepStart[1]->StepStart[2] delta would be 9000 iters / 1s = 9000/s.
    let ev = |elapsed_ms: u64, k: u16, iters: u64, is_step_end: bool| StimulusEvent {
        elapsed_ms,
        label: if is_step_end {
            format!("StepEnd[{k}]")
        } else {
            format!("StepStart[{k}]")
        },
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end,
    };
    let stimulus = vec![
        ev(1000, 1, 0, false),    // StepStart[1]
        ev(2000, 1, 500, true),   // StepEnd[1]   (ties with StepStart[2])
        ev(2000, 2, 9000, false), // StepStart[2] (ties with StepEnd[1])
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket present");
    let rate = step1
        .metrics
        .get("iteration_rate")
        .copied()
        .expect("step 1 iteration_rate populated");
    assert!(
        (rate - 500.0).abs() < f64::EPSILON,
        "step 1 must get its LOCAL StepStart[1]->StepEnd[1] rate (500/s), \
         not the cross-step StepStart[1]->StepStart[2] delta (9000/s); got {rate}",
    );
}

/// Hostile input: a synthesized bucket's end_ms clamps to the next
/// step-start, so a non-monotonic (corrupt-wire) StepEnd[k] >=
/// StepStart[k+1] cannot extend the synthesized window past the next
/// step's start (which would over-fold monitor samples into two adjacent
/// synthesized buckets).
#[test]
fn build_phase_buckets_with_stimulus_synthesized_end_ms_clamped_to_next_start() {
    use crate::scenario::snapshot::DrainedSnapshotEntry;
    use crate::timeline::StimulusEvent;
    // No captures -> all steps synthesize.
    let samples = SampleSeries::from_drained_typed(Vec::<DrainedSnapshotEntry>::new(), None);
    let ev = |elapsed_ms: u64, k: u16, is_step_end: bool| StimulusEvent {
        elapsed_ms,
        label: format!("Step{}[{k}]", if is_step_end { "End" } else { "Start" }),
        op_kind: None,
        detail: None,
        total_iterations: Some(0),
        step_index: Some(k),
        is_terminal: false,
        is_step_end,
    };
    // StepEnd[1] at 5000 is CORRUPT — it lands AFTER StepStart[2] (2000).
    let stimulus = vec![
        ev(1000, 1, false), // StepStart[1]
        ev(2000, 2, false), // StepStart[2]
        ev(5000, 1, true),  // StepEnd[1] — non-monotonic
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("synthesized step 1 bucket present");
    assert_eq!(
        step1.end_ms, 2000,
        "synthesized step 1 end_ms must clamp to StepStart[2]=2000, NOT the \
         corrupt StepEnd[1]=5000 that would overlap step 2; got {}",
        step1.end_ms,
    );
}

/// #4 regression fix: a synthesized zero-capture bucket folds the FULL
/// monitor-derived metric set (not just avg_imbalance_ratio), restoring
/// parity with the legacy Timeline::build fallback for a
/// zero-capture-with-monitor run. Before the fix the synthesize path
/// dropped max_imbalance / dsq / fallback / keep_last from the rendered
/// timeline (the path-flip regression pass 3 caught).
#[test]
fn build_phase_buckets_with_stimulus_synthesized_bucket_folds_full_monitor_set() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample, ScxEventCounters};
    use crate::scenario::snapshot::DrainedSnapshotEntry;
    use crate::timeline::StimulusEvent;
    let cpu = |nr: u32, dsq: u32, rq: u64, ev: Option<ScxEventCounters>| CpuSnapshot {
        nr_running: nr,
        local_dsq_depth: dsq,
        rq_clock: rq,
        scx_nr_running: 0,
        scx_flags: 0,
        event_counters: ev,
        schedstat: None,
        vcpu_cpu_time_ns: None,
        vcpu_perf: None,
        sched_domains: None,
    };
    let evc = |fb: i64, kl: i64| {
        Some(ScxEventCounters {
            select_cpu_fallback: fb,
            dispatch_keep_last: kl,
            ..Default::default()
        })
    };
    // Two in-window samples; anchor at the first stimulus elapsed (1000)
    // pins monitor_clock_offset to 0. Per-CPU nr_running 4/2 -> imbalance
    // 2.0; per-CPU dsq (3,1) -> avg 2.0, max 3; event counters
    // 10->110 fallback (delta 100), 5->55 keep_last (delta 50).
    let mon = MonitorReport {
        samples: vec![
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 1000,
                cpus: vec![cpu(4, 3, 100, evc(10, 5)), cpu(2, 1, 100, None)],
            },
            MonitorSample {
                prog_stats: None,
                elapsed_ms: 1500,
                cpus: vec![cpu(4, 3, 100, evc(110, 55)), cpu(2, 1, 200, None)],
            },
        ],
        ..Default::default()
    };
    // No captures -> step 1 synthesizes; StepStart[2] bounds its window.
    let samples = SampleSeries::from_drained_typed(Vec::<DrainedSnapshotEntry>::new(), Some(mon));
    let start = |elapsed_ms: u64, k: u16| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let stimulus = vec![start(1000, 1), start(2000, 2)];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("synthesized step 1 bucket present");
    assert_eq!(step1.sample_count, 0, "synthesized bucket is capture-free");
    let g = |k: &str| step1.metrics.get(k).copied();
    // The monitor-derived metrics the pre-fix synthesize path DROPPED,
    // now restored to parity with Timeline::build's compute_metrics:
    assert_eq!(g("max_imbalance_ratio"), Some(2.0), "max imbalance");
    assert_eq!(g("avg_dsq_depth"), Some(2.0), "avg dsq depth");
    assert_eq!(g("max_dsq_depth"), Some(3.0), "max dsq depth");
    assert_eq!(
        g("total_fallback"),
        Some(100.0),
        "fallback counter delta 110-10"
    );
    assert_eq!(
        g("total_keep_last"),
        Some(50.0),
        "keep_last counter delta 55-5"
    );
    // avg_imbalance_ratio was folded pre-fix too; still present.
    assert_eq!(g("avg_imbalance_ratio"), Some(2.0), "avg imbalance");
}

/// Hostile input: a corrupt StepEnd[k] BEFORE its StepStart[k] yields an
/// inverted synthesized window (end_ms < start_ms). fold_monitor_into_bucket
/// folds nothing (the half-open `m >= start_ms && m < end_ms` is vacuously
/// false) and the bucket is present without panic — no over-fold, no crash.
#[test]
fn build_phase_buckets_with_stimulus_synthesized_inverted_window_folds_nothing() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    use crate::scenario::snapshot::DrainedSnapshotEntry;
    use crate::timeline::StimulusEvent;
    let cpu = |nr: u32| CpuSnapshot {
        nr_running: nr,
        ..Default::default()
    };
    let mon = MonitorReport {
        samples: vec![MonitorSample::new(1500, vec![cpu(6), cpu(2)])],
        ..Default::default()
    };
    let samples = SampleSeries::from_drained_typed(Vec::<DrainedSnapshotEntry>::new(), Some(mon));
    let ev = |elapsed_ms: u64, k: u16, is_step_end: bool| StimulusEvent {
        elapsed_ms,
        label: format!("Step{}[{k}]", if is_step_end { "End" } else { "Start" }),
        op_kind: None,
        detail: None,
        total_iterations: Some(0),
        step_index: Some(k),
        is_terminal: false,
        is_step_end,
    };
    // StepEnd[1] at 1000 is BEFORE StepStart[1] at 2000 (corrupt wire).
    let stimulus = vec![ev(2000, 1, false), ev(1000, 1, true)];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("synthesized step 1 bucket present despite inverted window");
    assert!(
        step1.start_ms > step1.end_ms,
        "window is inverted: start {} end {}",
        step1.start_ms,
        step1.end_ms,
    );
    assert!(
        !step1.metrics.contains_key("avg_imbalance_ratio"),
        "an inverted window folds no monitor samples; got {:?}",
        step1.metrics,
    );
}

/// Math-protocol boundary coverage for the synthesize end_ms terminal-clamp
/// arm: a last step with no StepEnd and no successor but a ScenarioEnd
/// terminal present clamps the window to the terminal (not u64::MAX), so a
/// monitor sample AFTER the terminal is excluded from the fold (the
/// pass-4 over-fold guard).
#[test]
fn build_phase_buckets_with_stimulus_synthesized_end_ms_clamps_to_terminal() {
    use crate::monitor::{CpuSnapshot, MonitorReport, MonitorSample};
    use crate::scenario::snapshot::DrainedSnapshotEntry;
    use crate::timeline::StimulusEvent;
    let cpu = |nr: u32| CpuSnapshot {
        nr_running: nr,
        ..Default::default()
    };
    // s1 @1000 (anchor -> monitor_clock_offset 0), imbalance 2/2 = 1.0,
    // IN [1000, 3000). s2 @4000 (AFTER the terminal 3000), imbalance
    // 10/2 = 5.0, must be EXCLUDED by the terminal clamp.
    let mon = MonitorReport {
        samples: vec![
            MonitorSample::new(1000, vec![cpu(2), cpu(2)]),
            MonitorSample::new(4000, vec![cpu(10), cpu(2)]),
        ],
        ..Default::default()
    };
    let samples = SampleSeries::from_drained_typed(Vec::<DrainedSnapshotEntry>::new(), Some(mon));
    let start = StimulusEvent {
        elapsed_ms: 1000,
        label: "StepStart[1]".to_string(),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(1),
        is_terminal: false,
        is_step_end: false,
    };
    let terminal = StimulusEvent {
        elapsed_ms: 3000,
        label: "ScenarioEnd".to_string(),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: None,
        is_terminal: true,
        is_step_end: false,
    };
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &[start, terminal]);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("synthesized step 1 bucket present");
    assert_eq!(
        step1.end_ms, 3000,
        "end_ms clamps to the terminal (3000), not u64::MAX",
    );
    assert_eq!(
        step1.metrics.get("avg_imbalance_ratio").copied(),
        Some(1.0),
        "only the in-window pre-terminal sample folds (imbalance 1.0); the \
         post-terminal sample (5.0) is excluded — avg is 1.0, not 3.0",
    );
}

/// The public SampleSeries grouping methods: by_stamped_phase
/// COLLAPSES a deferred-fire burst (every capture stamped the same late
/// step) into one phase, while by_stimulus_phase re-derives the correct
/// per-phase grouping from each sample's timing-independent
/// boundary_offset_ms. Pins both new public entry points and the
/// difference that motivates by_stimulus_phase.
#[test]
fn by_stimulus_phase_separates_what_by_stamped_phase_collapses() {
    use crate::scenario::sample::SampleSeries;
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    // All four captures stamp the SAME late step (the burst) but their
    // SCHEDULED offsets fall in distinct step windows.
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(3),
    };
    let drained = vec![
        mk("p0", 500),   // before step 1 start (1000) -> BASELINE (0)
        mk("p1", 1_500), // step 1 window
        mk("p2", 2_500), // step 2 window
        mk("p3", 3_500), // step 3 window
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let stim = |elapsed_ms: u64, k: u16| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let stimulus = vec![stim(1000, 1), stim(2000, 2), stim(3000, 3)];

    // by_stamped_phase: all four collapse into the single stamped key 3.
    let stamped = samples.by_stamped_phase();
    assert_eq!(stamped.keys().copied().collect::<Vec<_>>(), vec![3]);
    assert_eq!(stamped[&3].len(), 4, "stamped grouping collapses the burst");

    // by_stimulus_phase: re-derived from boundary_offset -> 4 phases.
    let by_stim = samples.by_stimulus_phase(&stimulus);
    assert_eq!(
        by_stim.keys().copied().collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    for k in [0u16, 1, 2, 3] {
        assert_eq!(
            by_stim[&k].len(),
            1,
            "phase {k} should hold exactly one sample"
        );
    }
}

/// Fallback + synthesize interaction: a capture with no
/// `boundary_offset_ms` (on-demand / fixture) keeps its stamped
/// step_index even when a stimulus timeline is present — the offset
/// remap only overrides captures that carry a scheduled offset, so
/// legacy / non-periodic entries are untouched. The step-5 StepStart
/// does NOT pull the None-offset captures to step 5 (they stay at their
/// stamped 1 / 2); it does, per the #4 synthesize seam, produce its OWN
/// capture-free step-5 bucket — a StepStart marks a step that ran, so
/// every StepStart-step gets a bucket even with zero captures.
#[test]
fn build_phase_buckets_with_stimulus_none_offset_falls_back_to_stamped_step() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, step: u16| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(100),
        boundary_offset_ms: None,
        step_index: Some(step),
    };
    let drained = vec![mk("periodic_000", 1), mk("periodic_001", 2)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    // A step-5 stimulus is present: it must NOT pull the None-offset
    // captures to step 5 (they keep their stamped 1 / 2), but it DOES
    // synthesize its own capture-free step-5 bucket.
    let stimulus = vec![StimulusEvent {
        elapsed_ms: 0,
        label: "StepStart[5]".to_string(),
        op_kind: None,
        detail: None,
        total_iterations: None,
        step_index: Some(5),
        is_terminal: false,
        is_step_end: false,
    }];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let idxs: Vec<u16> = phases.iter().map(|p| p.step_index).collect();
    assert_eq!(
        idxs,
        vec![1, 2, 5],
        "None-offset captures keep their stamped step_index (1, 2), NOT \
         remapped to the step-5 stimulus; step 5's StepStart synthesizes its \
         own capture-free bucket; got {idxs:?}",
    );
    // The captures stayed in buckets 1/2 (one each) — NOT collapsed into a
    // single step-5 bucket. Step 5 is the synthesized, capture-free one.
    let count = |k: u16| {
        phases
            .iter()
            .find(|p| p.step_index == k)
            .map(|p| p.sample_count)
    };
    assert_eq!(
        count(1),
        Some(1),
        "capture stamped 1 stays in its own bucket"
    );
    assert_eq!(
        count(2),
        Some(1),
        "capture stamped 2 stays in its own bucket"
    );
    assert_eq!(
        count(5),
        Some(0),
        "step 5 is the synthesized capture-free bucket"
    );
}

/// Iteration-rate attribution regression: in the production shape (periodic captures carry
/// workload-relative boundary offsets in the step INTERIOR + stimulus
/// events carry step_index), iteration_rate must attach to the step the
/// rate was measured DURING — by step_index, NOT by a timestamp-window
/// match against the capture-derived (interior) bucket window. The
/// step-START (prev.elapsed_ms) falls in the inter-step gap, inside no
/// interior bucket window, so the old window match dropped every rate.
/// This pins the step_index attribution; it FAILS on the window match.
#[test]
fn build_phase_buckets_with_stimulus_iteration_rate_attaches_by_step_not_interior_window() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    // One capture per step, each at the step INTERIOR (offset strictly
    // greater than the step-start), mirroring compute_periodic_boundaries_ns
    // (10-90% interior). Stamped step_index intentionally wrong (all 9) so
    // the OFFSET drives grouping and the stimulus event's step_index drives
    // rate attribution.
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    let drained = vec![
        mk("periodic_000", 1_500), // step 1 interior (window [1000,2000))
        mk("periodic_001", 2_500), // step 2 interior (window [2000,3000))
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    // Step-starts at 1000/2000/3000; cumulative iterations 0/1000/3000.
    let stim = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let stimulus = vec![stim(1000, 1, 0), stim(2000, 2, 1000), stim(3000, 3, 3000)];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    // Step 1 bucket (step_index==1): rate for (step1@1000 -> step2@2000),
    // iters 0->1000 over 1s = 1000/s. Its window is the single interior
    // capture [1500,1500], which does NOT contain the step-start 1000 —
    // the old window match dropped this rate.
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(1000.0),
        "step 1 iteration_rate must attach by step_index to the interior \
         bucket; got {:?} (start_ms={}, end_ms={})",
        step1.metrics.get("iteration_rate"),
        step1.start_ms,
        step1.end_ms,
    );
    // Step 2 bucket: rate for (step2@2000 -> step3@3000), iters
    // 1000->3000 over 1s = 2000/s.
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("Step[1] bucket present");
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        Some(2000.0),
        "step 2 iteration_rate must attach by step_index; got {:?}",
        step2.metrics.get("iteration_rate"),
    );
}

/// Each step's `iteration_rate` is the STEP-LOCAL
/// `StepStart[k]` -> `StepEnd[k]` delta (its OWN workers, start-to-end of
/// hold), NOT the cross-step `StepStart[k]` -> `StepStart[k+1]` delta.
/// Workers respawned per step read ~0 at every StepStart, so the
/// cross-step delta is `0 - 0` and yields no rate (the old cross-step bug:
/// every fresh-per-step phase silently had no throughput). Pairing each
/// step's own StepStart -> StepEnd recovers the real per-step rate. The
/// elapsed-sorted `windows(2)` walk pairs `StepStart[k]` -> `StepEnd[k]`
/// first (both carry step_index `k`); `or_insert` keeps that step-local
/// rate, and the intervening `StepEnd[k]` -> `StepStart[k+1]` pair reads
/// `0 <= end` so `rate_to` returns None and never overwrites.
#[test]
fn build_phase_buckets_with_stimulus_pairs_step_local_for_respawned_workers() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    let drained = vec![
        mk("periodic_000", 1_500), // step 1 interior (window [1000,2100))
        mk("periodic_001", 2_500), // step 2 interior (window [2100,..))
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let start = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let end = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepEnd[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: true,
    };
    // Step 1: 0 -> 10000 over its 1s hold (10000/s). Step 2 workers are
    // RESPAWNED, so StepStart[2] reads 0 again; step 2: 0 -> 5000 over 1s
    // (5000/s). 1000ms windows keep the division exact (1.0s denominator).
    let stimulus = vec![
        start(1_000, 1, 0),
        end(2_000, 1, 10_000),
        start(2_100, 2, 0),
        end(3_100, 2, 5_000),
        StimulusEvent::terminal(3_200, 5_000),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket present");
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("step 2 bucket present");
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(10_000.0),
        "step 1 must report its step-local StepStart->StepEnd rate, not the \
         cross-step 0->0 delta (the old cross-step silent None); got {:?}",
        step1.metrics.get("iteration_rate"),
    );
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        Some(5_000.0),
        "step 2 (respawned workers) must report its step-local rate; got {:?}",
        step2.metrics.get("iteration_rate"),
    );
}

/// A PERSISTENT (Backdrop) population keeps iterating through
/// the inter-step teardown, so `StepStart[k+1]` reads MORE than
/// `StepEnd[k]` and the cross-step `StepEnd[k]` -> `StepStart[k+1]` pair
/// WOULD yield a rate. But that pair has an `is_step_end` `prev`, so the
/// attribution loop's guard skips it before `rate_components`/`or_insert` ever
/// run — the cross-step rate is never even computed, and each step
/// reports only its own step-local `StepStart[k]` -> `StepEnd[k]`
/// throughput. (`or_insert` is a redundant secondary safety here, not
/// the operative mechanism — the guard is.)
#[test]
fn build_phase_buckets_with_stimulus_step_local_wins_over_persistent_cross_step() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    let drained = vec![mk("periodic_000", 1_500), mk("periodic_001", 2_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let start = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let end = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepEnd[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: true,
    };
    // Step 1 local: 0 -> 10000 over 1s (10000/s). Persistent workers add
    // 500 in the 100ms teardown gap, so StepStart[2] reads 10500 — the
    // cross-step StepEnd[1] -> StepStart[2] pair would compute 500/0.1s =
    // 5000/s, but its is_step_end prev is skipped by the guard so that
    // rate is never computed for step 1.
    // Step 2 local: 10500 -> 15500 over 1s (5000/s).
    let stimulus = vec![
        start(1_000, 1, 0),
        end(2_000, 1, 10_000),
        start(2_100, 2, 10_500),
        end(3_100, 2, 15_500),
        StimulusEvent::terminal(3_200, 15_500),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket present");
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("step 2 bucket present");
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(10_000.0),
        "step-local rate must win over the 5000/s persistent cross-step \
         delta (the is_step_end guard skips the cross-step pair); got {:?}",
        step1.metrics.get("iteration_rate"),
    );
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        Some(5_000.0),
        "step 2 must report its own start-to-end-of-hold rate; got {:?}",
        step2.metrics.get("iteration_rate"),
    );
}

/// A STALLED step (its own `StepStart[k] -> StepEnd[k]` delta is zero)
/// must report its MEASURED-ZERO rate `Some(0.0)` — it must NOT leak the
/// inter-step teardown-gap rate that the `StepEnd[k] -> StepStart[k+1]`
/// pair would otherwise produce. StepEnd[k] carries step_index `k`, so
/// without the `is_step_end` guard in the attribution loop that cross-step
/// pair (prev = StepEnd[k], also step_index `k`) would `or_insert` a gap
/// rate into bucket `k`. This pins the guard: bucket `k` is sourced ONLY
/// by its own StepStart -> StepEnd pair (which here is a measured zero).
#[test]
fn build_phase_buckets_with_stimulus_stalled_step_reports_measured_zero() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    let drained = vec![mk("periodic_000", 1_500), mk("periodic_001", 2_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let start = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    let end = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepEnd[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: true,
    };
    // Step 1 STALLED: StepStart[1] == StepEnd[1] == 0, so its step-local
    // pair is rate_components (0.0, secs), derived iteration_rate 0.0 —
    // measured zero throughput. A persistent
    // population then advances 500 during the 100ms teardown gap, so the
    // cross-step StepEnd[1](0) -> StepStart[2](500) pair WOULD compute
    // 500/0.1s = 5000/s — which must NOT land in bucket 1 (it is
    // guard-skipped). Step 2 runs normally: 500 -> 1500 over 1s = 1000/s.
    let stimulus = vec![
        start(1_000, 1, 0),
        end(2_000, 1, 0),
        start(2_100, 2, 500),
        end(3_100, 2, 1_500),
        StimulusEvent::terminal(3_200, 1_500),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket present");
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("step 2 bucket present");
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(0.0),
        "a stalled step reports measured-zero throughput, not the leaked \
         5000/s teardown gap rate from the StepEnd[1] -> StepStart[2] pair; \
         got {:?}",
        step1.metrics.get("iteration_rate"),
    );
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        Some(1_000.0),
        "step 2 still reports its own step-local rate; got {:?}",
        step2.metrics.get("iteration_rate"),
    );
}

/// Pins the ms→s unit conversion + end-to-end component emission for the
/// iteration_rate Rate: a step over a 2000ms window with a 1000-iteration
/// delta must emit total_phase_duration_sec == 2.0 (NOT 2000 — the /1000
/// lives in the component because derive_rate_metrics does a bare num/den)
/// and total_phase_iterations == 1000, so the re-derived iteration_rate is
/// 1000 / 2.0 = 500/s. A regression leaving the denominator in ms would
/// derive 1000/2000 = 0.5 and pass every type/naming gate silently.
#[test]
fn build_phase_buckets_with_stimulus_emits_rate_components_in_seconds() {
    use crate::timeline::StimulusEvent;
    // No capture samples for step 1 → the synthesized-bucket seam creates
    // it and the attribution loop fills the components from the stimulus.
    let samples = SampleSeries::from_drained_typed(vec![], None);
    let stimulus = vec![
        StimulusEvent {
            elapsed_ms: 1_000,
            label: "StepStart[1]".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: Some(0),
            step_index: Some(1),
            is_terminal: false,
            is_step_end: false,
        },
        StimulusEvent {
            elapsed_ms: 3_000, // 2000ms = 2.0s window
            label: "StepEnd[1]".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: Some(1_000), // 1000-iteration delta
            step_index: Some(1),
            is_terminal: false,
            is_step_end: true,
        },
        StimulusEvent::terminal(3_100, 1_000),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket synthesized from stimulus");
    assert_eq!(
        step1.metrics.get("total_phase_duration_sec").copied(),
        Some(2.0),
        "duration component is SECONDS (2000ms / 1000), not ms; got {:?}",
        step1.metrics.get("total_phase_duration_sec"),
    );
    assert_eq!(
        step1.metrics.get("total_phase_iterations").copied(),
        Some(1_000.0),
        "iteration component is the 1000-iteration delta",
    );
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(500.0),
        "derived iteration_rate = 1000 iters / 2.0 s = 500/s (NOT 0.5 from a \
         ms denominator); got {:?}",
        step1.metrics.get("iteration_rate"),
    );
}

/// The LAST step has no successor step event, so its
/// iteration_rate needs the terminal scenario-end boundary. The
/// terminal supplies that boundary (last step's rate = delta to the
/// terminal count) and must NOT seed a phantom bucket — the bucket set
/// stays equal to the sample-derived phases.
#[test]
fn build_phase_buckets_with_stimulus_terminal_gives_last_step_rate_no_phantom_bucket() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    // Two captures: step 1 interior, step 2 interior.
    let drained = vec![mk("periodic_000", 1_500), mk("periodic_001", 2_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let stim = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    // Step starts at 1000/2000 (iters 0/1000); terminal at 3000 with
    // final iters 3000 — the right boundary the LAST step (step 2)
    // needs. The terminal carries step_index None + is_terminal so it
    // seeds no bucket.
    let stimulus = vec![
        stim(1000, 1, 0),
        stim(2000, 2, 1000),
        StimulusEvent::terminal(3000, 3000),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let idxs: Vec<u16> = phases.iter().map(|p| p.step_index).collect();
    assert_eq!(
        idxs,
        vec![1, 2],
        "terminal must not add a phantom bucket; got {idxs:?}",
    );
    // Step 2 is the LAST step: its rate comes from the terminal,
    // 1000 -> 3000 over 1s = 2000/s. Without the terminal it would be
    // None (the bug this fixes).
    let step2 = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("step 2 bucket present");
    assert_eq!(
        step2.metrics.get("iteration_rate").copied(),
        Some(2000.0),
        "last step's iteration_rate must come from the terminal boundary",
    );
}

/// First-step zero-baseline rate, through the production aggregator + the FULL from_wire
/// path (unit tests previously injected Some(0) directly, masking the
/// wire 0->None collapse). The first step frame reads 0 cumulative
/// iterations; after dropping the sentinel the first step's bucket gets
/// a rate.
#[test]
fn build_phase_buckets_with_stimulus_first_step_zero_baseline_from_wire() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    let drained = vec![mk("periodic_000", 1_500), mk("periodic_001", 2_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let wire = |elapsed_ms: u32, step_index: u16, iters: u64| crate::vmm::wire::StimulusEvent {
        elapsed_ms,
        step_index,
        op_count: 0,
        op_kinds: 0,
        cgroup_count: 0,
        worker_count: 1,
        total_iterations: iters,
    };
    // First step frame reads 0 cumulative iters (workers just spawned).
    let stimulus: Vec<StimulusEvent> = [wire(1000, 1, 0), wire(2000, 2, 2000)]
        .iter()
        .map(StimulusEvent::from_wire)
        .collect();
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket present");
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(2000.0),
        "first step's iteration_rate must compute from the 0 baseline",
    );
}

/// Loop-hold attribution: once the guest emits a start frame
/// for a Loop step (the run_step Loop arm), a scenario ending on a Loop
/// step must attribute the final window's throughput to the LOOP step,
/// NOT graft it onto the prior step. This pins the host-side contract
/// the guest fix relies on: with the loop step's own start frame
/// present, (loop_start -> terminal) lands on the loop step's bucket and
/// (prior_start -> loop_start) lands on the prior step's bucket.
#[test]
fn build_phase_buckets_with_stimulus_loop_step_rate_no_prior_graft() {
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    use crate::timeline::StimulusEvent;
    let mk = |tag: &str, offset_ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: fixture_report(),
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(9_000),
        boundary_offset_ms: Some(offset_ms),
        step_index: Some(9),
    };
    // Prior step (step 1) interior + Loop step (step 2) interior.
    let drained = vec![mk("periodic_000", 1_500), mk("periodic_001", 2_500)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let stim = |elapsed_ms: u64, k: u16, iters: u64| StimulusEvent {
        elapsed_ms,
        label: format!("StepStart[{k}]"),
        op_kind: None,
        detail: None,
        total_iterations: Some(iters),
        step_index: Some(k),
        is_terminal: false,
        is_step_end: false,
    };
    // step1@1000 (iters 0), loop step2@2000 (iters 1000, its OWN start
    // frame — the loop-hold attribution fix), terminal@3000 (iters 3000).
    let stimulus = vec![
        stim(1000, 1, 0),
        stim(2000, 2, 1000),
        StimulusEvent::terminal(3000, 3000),
    ];
    let phases = crate::assert::build_phase_buckets_with_stimulus(&samples, &stimulus);
    let step1 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("step 1 bucket");
    let loop_step = phases
        .iter()
        .find(|p| p.step_index == 2)
        .expect("loop step bucket");
    // Prior step gets ONLY its own window (0 -> 1000 over 1s = 1000/s),
    // NOT the loop window grafted on.
    assert_eq!(
        step1.metrics.get("iteration_rate").copied(),
        Some(1000.0),
        "prior step must not absorb the loop step's window",
    );
    // Loop step gets the (loop_start -> terminal) rate: 1000 -> 3000
    // over 1s = 2000/s.
    assert_eq!(
        loop_step.metrics.get("iteration_rate").copied(),
        Some(2000.0),
        "loop step must get its own throughput from its start frame + terminal",
    );
}

/// `populate_run_ext_metrics` on a populated series produces
/// the expected entries. Without this, the empty test and the
/// no-overwrite test pass vacuously and the load-bearing happy
/// path is uncovered.
#[test]
fn populate_run_ext_metrics_populated_series_inserts_expected_keys() {
    use crate::monitor::dump::FailureDumpReport;
    use crate::monitor::scx_walker::DsqState;
    use crate::scenario::snapshot::{DrainedSnapshotEntry, MissingStatsReason};
    let mk_entry = |tag: &str, ms: u64| DrainedSnapshotEntry {
        tag: tag.to_string(),
        report: FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            dsq_states: vec![DsqState {
                origin: "local cpu 0".to_string(),
                nr: 5,
                ..Default::default()
            }],
            ..Default::default()
        },
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(ms),
        boundary_offset_ms: None,
        step_index: Some(0),
    };
    let drained = vec![mk_entry("periodic_000", 100), mk_entry("periodic_001", 200)];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let mut target = std::collections::BTreeMap::new();
    crate::assert::populate_run_ext_metrics(&samples, &mut target);
    // avg_dsq_depth has no typed GauntletRow field → populated.
    // mean of (5, 5) = 5.0.
    let avg = target
        .get("avg_dsq_depth")
        .copied()
        .expect("avg_dsq_depth populated for populated series");
    assert!(
        (avg - 5.0).abs() < f64::EPSILON,
        "expected avg_dsq_depth=5.0, got {avg}",
    );
    // max_dsq_depth has a typed field → skipped by populate.
    assert!(
        !target.contains_key("max_dsq_depth"),
        "max_dsq_depth has a typed GauntletRow field; must not leak into ext_metrics",
    );
}

/// populate_run_ext_metrics_from_phases folds per-phase metrics that
/// have no read_sample dispatch (e.g. avg_imbalance_ratio) by a
/// Gauge(Avg) weighted mean across phases. (iteration_rate, also without
/// read_sample dispatch, is a MetricKind::Rate — re-pooled from its
/// summed components, not weighted-mean folded.)
#[test]
fn populate_run_ext_metrics_from_phases_folds_per_phase_keys() {
    use crate::assert::PhaseBucket;
    use std::collections::BTreeMap;
    let mut m0 = BTreeMap::new();
    m0.insert("avg_imbalance_ratio".to_string(), 2.0);
    let mut m1 = BTreeMap::new();
    m1.insert("avg_imbalance_ratio".to_string(), 4.0);
    let phases = vec![
        PhaseBucket {
            per_cgroup: Default::default(),
            step_index: 1,
            label: "Step[0]".to_string(),
            start_ms: 0,
            end_ms: 100,
            sample_count: 5,
            metrics: m0,
        },
        PhaseBucket {
            per_cgroup: Default::default(),
            step_index: 2,
            label: "Step[1]".to_string(),
            start_ms: 100,
            end_ms: 200,
            sample_count: 15,
            metrics: m1,
        },
    ];
    let mut target = BTreeMap::new();
    crate::assert::populate_run_ext_metrics_from_phases(&phases, &mut target);
    // avg_imbalance_ratio is Gauge(Avg) — weighted mean by
    // sample_count: (2.0*5 + 4.0*15) / 20 = 70/20 = 3.5.
    let avg = target
        .get("avg_imbalance_ratio")
        .copied()
        .expect("avg_imbalance_ratio folded from per-phase");
    assert!(
        (avg - 3.5).abs() < f64::EPSILON,
        "expected weighted mean 3.5, got {avg}",
    );
}

/// Run-level guard: populate_run_ext_metrics_from_phases must SKIP keys with
/// a typed GauntletRow field (TYPED_FIELD_NAMES) so the phase fold never
/// re-injects them into ext_metrics. The monitor fold writes max_imbalance_ratio +
/// stuck_count onto CAPTURED buckets; both are typed-backed (their accessor
/// wins on read), so writing them to ext would be unread bloat AND a cross-RUN
/// drift trap (stuck_count: typed whole-run flag vs ext per-phase
/// stall-count SUM). avg_imbalance_ratio (genuinely ext-only) must still fold.
#[test]
fn populate_run_ext_metrics_from_phases_skips_typed_backed_keys() {
    use crate::assert::PhaseBucket;
    use std::collections::BTreeMap;
    let mut m = BTreeMap::new();
    m.insert("avg_imbalance_ratio".to_string(), 2.0); // ext-only -> folded
    m.insert("max_imbalance_ratio".to_string(), 3.0); // typed-backed -> skipped
    m.insert("stuck_count".to_string(), 2.0); // typed-backed -> skipped
    let phases = vec![PhaseBucket {
        per_cgroup: Default::default(),
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 5,
        metrics: m,
    }];
    let mut target = BTreeMap::new();
    crate::assert::populate_run_ext_metrics_from_phases(&phases, &mut target);
    assert!(
        target.contains_key("avg_imbalance_ratio"),
        "avg_imbalance_ratio is ext-only and must be folded into ext_metrics",
    );
    assert!(
        !target.contains_key("max_imbalance_ratio"),
        "max_imbalance_ratio is typed-backed; must NOT leak into ext_metrics from the phase fold",
    );
    assert!(
        !target.contains_key("stuck_count"),
        "stuck_count is typed-backed; must NOT leak into ext_metrics (typed flag vs ext-sum drift)",
    );
}

/// A synthesized zero-capture phase (sample_count==0) still folds into
/// the run aggregate — its capture-independent iteration_rate is
/// INCLUDED, not dropped. iteration_rate is now a MetricKind::Rate, so
/// inclusion is via its Counter components (total_phase_iterations /
/// total_phase_duration_sec) SUMMING across phases (weights ignored — see
/// aggregate_finite) and the rate re-deriving as Σiters/Σseconds. A
/// regression dropping the synthesized phase would silently re-drop its
/// iterations from the sidecar aggregate: the run-level variant of the #4
/// bug. Unequal phase durations make the re-pool (450) distinct from both
/// a mean-of-ratios (500) and a dropped-synthesized result (400).
#[test]
fn populate_run_ext_metrics_repools_synthesized_zero_capture_phase() {
    use crate::assert::PhaseBucket;
    use std::collections::BTreeMap;
    // Captured phase: 1200 iters over 3s = 400/s.
    let cap = BTreeMap::from([
        ("total_phase_iterations".to_string(), 1200.0),
        ("total_phase_duration_sec".to_string(), 3.0),
    ]);
    // Synthesized zero-capture phase: 600 iters over 1s = 600/s.
    let synth = BTreeMap::from([
        ("total_phase_iterations".to_string(), 600.0),
        ("total_phase_duration_sec".to_string(), 1.0),
    ]);
    let phases = vec![
        PhaseBucket {
            per_cgroup: Default::default(),
            step_index: 1,
            label: "Step[0]".to_string(),
            start_ms: 0,
            end_ms: 100,
            sample_count: 5,
            metrics: cap,
        },
        PhaseBucket {
            per_cgroup: Default::default(),
            step_index: 2,
            label: "Step[1]".to_string(),
            start_ms: 100,
            end_ms: 200,
            sample_count: 0, // synthesized zero-capture step
            metrics: synth,
        },
    ];
    let mut target = BTreeMap::new();
    crate::assert::populate_run_ext_metrics_from_phases(&phases, &mut target);
    // Re-pool: Σiters / Σseconds = (1200 + 600) / (3 + 1) = 1800/4 = 450/s.
    // The synthesized phase's 600 iters are SUMMED in (Counters ignore the
    // sample_count weight), so 450 — NOT 400 (synthesized dropped, 1200/3)
    // and NOT 500 (mean of the two ready ratios 400 and 600).
    let r = target
        .get("iteration_rate")
        .copied()
        .expect("synthesized zero-capture phase's components must re-pool into iteration_rate");
    assert!(
        (r - 450.0).abs() < f64::EPSILON,
        "expected re-pooled 450/s (synthesized 600 iters summed in), NOT \
         400 (dropped) or 500 (mean of ratios); got {r}",
    );
    // The summed components survive for any further re-derivation.
    assert_eq!(
        target.get("total_phase_iterations").copied(),
        Some(1800.0),
        "components sum across phases (weights ignored for Counters)",
    );
}

// ---- #7 Item 6b: step-local per-cgroup capture ----
//
// phase_cgroup_stats extracts the RAW components every cgroup_stats reduction
// re-pools from; step_per_cgroup_bucket wraps them in the guest-side carrier;
// fold_guest_per_cgroup_into_host_buckets merges the carrier into the
// host-rebuilt buckets by step_index. These tests pin (1) raw-component parity
// vs cgroup_stats, (2) the math-boundary edges (not-measured off-CPU%, raw-ns
// run delays, the numa partition, MAX-not-SUM cross-node, argmax gap coupling),
// (3) the carrier shape, and (4) the fold (union / orphan / pass-through).

use super::tests_common::rpt;
use super::{
    PhaseCgroupStats, cgroup_stats, fold_guest_per_cgroup_into_host_buckets, percentile,
    phase_cgroup_stats, step_per_cgroup_bucket,
};
use crate::workload::WorkerReport;
use std::collections::BTreeSet;

/// Every [`super::CgroupStats`] reduction re-pools EXACTLY from the
/// [`PhaseCgroupStats`] raw components: avg/min/max/spread off-CPU% from
/// `off_cpu_pcts`; p99/median from the pooled `wake_latencies_ns`; mean/worst
/// run-delay from `run_delays_ns` (RAW ns ÷ 1000); migration_ratio /
/// cross_node_migration_ratio from the counters; cpus_used / num_cpus from the
/// union. The one addition is `numa_pages_local` (cgroup_stats has no node
/// context, leaving page_locality 0.0). Built on the SAME two reports so the
/// re-pool MUST reproduce cgroup_stats's reductions value-for-value.
#[test]
fn phase_cgroup_stats_components_repool_to_cgroup_stats() {
    let w1 = WorkerReport {
        schedstat_run_delay_ns: 3000,
        schedstat_cpu_time_ns: 1_000_000,
        migration_count: 2,
        iterations: 100,
        wake_latencies_ns: vec![1000, 2000],
        wake_sample_total: 5,
        vmstat_numa_pages_migrated: 10,
        numa_pages: BTreeMap::from([(0usize, 100u64), (1, 50)]),
        // wall 1_000_000, off 200_000 -> off-CPU% 20.0; gap 40 on cpu 0.
        ..rpt(1, 1000, 1_000_000, 200_000, &[0, 1], 40)
    };
    let w2 = WorkerReport {
        schedstat_run_delay_ns: 5000,
        schedstat_cpu_time_ns: 2_000_000,
        migration_count: 3,
        iterations: 200,
        wake_latencies_ns: vec![3000],
        wake_sample_total: 7,
        vmstat_numa_pages_migrated: 8,
        numa_pages: BTreeMap::from([(1usize, 80u64), (2, 20)]),
        // wall 2_000_000, off 100_000 -> off-CPU% 5.0; gap 60 on cpu 2.
        ..rpt(2, 2000, 2_000_000, 100_000, &[2, 3], 60)
    };
    let reports = vec![w1, w2];
    let nodes: BTreeSet<usize> = [0, 1].into_iter().collect();
    let pcs = phase_cgroup_stats(&reports, Some(&nodes));
    let cg = cgroup_stats(&reports);

    // cpus_used union + worker count.
    assert_eq!(pcs.cpus_used, cg.cpus_used);
    assert_eq!(pcs.cpus_used.len(), cg.num_cpus);
    assert_eq!(pcs.num_workers, cg.num_workers);

    // off-CPU% re-pool to avg/min/max/spread.
    assert_eq!(pcs.off_cpu_pcts.len(), 2);
    let avg = pcs.off_cpu_pcts.iter().sum::<f64>() / pcs.off_cpu_pcts.len() as f64;
    let min = pcs.off_cpu_pcts.iter().cloned().reduce(f64::min).unwrap();
    let max = pcs.off_cpu_pcts.iter().cloned().reduce(f64::max).unwrap();
    assert!((avg - cg.avg_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((min - cg.min_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((max - cg.max_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!(((max - min) - cg.spread.unwrap()).abs() < 1e-9);

    // Pooled wake latencies re-pool to p99/median via the same percentile fn.
    let mut pooled = pcs.wake_latencies_ns.clone();
    pooled.sort_unstable();
    assert_eq!(pooled, vec![1000u64, 2000, 3000]);
    let p99 = percentile(&pooled, 0.99) as f64 / 1000.0;
    let median = percentile(&pooled, 0.5) as f64 / 1000.0;
    assert!((p99 - cg.p99_wake_latency_us).abs() < 1e-9);
    assert!((median - cg.median_wake_latency_us).abs() < 1e-9);
    assert_eq!(pcs.wake_sample_total, 12);
    // CV re-pools over the SAME pooled set with n = wake_latencies_ns.len()
    // (the reservoir-CLAMPED pool size), NOT wake_sample_total — exactly as
    // cgroup_stats computes it (cv = stddev/mean, n = all_latencies.len()).
    // Here len() == 3 but wake_sample_total == 12, so a re-pool that wrongly
    // divided by wake_sample_total would NOT reproduce cg.wake_latency_cv.
    let n = pcs.wake_latencies_ns.len() as f64;
    let mean_ns = pcs.wake_latencies_ns.iter().sum::<u64>() as f64 / n;
    let cv = if mean_ns > 0.0 {
        let variance = pcs
            .wake_latencies_ns
            .iter()
            .map(|&v| (v as f64 - mean_ns).powi(2))
            .sum::<f64>()
            / n;
        variance.sqrt() / mean_ns
    } else {
        0.0
    };
    assert!(
        (cv - cg.wake_latency_cv).abs() < 1e-9,
        "wake_latency_cv re-pools with n = len() ({}), NOT wake_sample_total \
         ({}); got {cv} vs cg {}",
        pcs.wake_latencies_ns.len(),
        pcs.wake_sample_total,
        cg.wake_latency_cv,
    );

    // Run delays stored RAW ns; the re-pool divides by 1000 ONCE to match
    // cgroup_stats's µs mean/worst (pre-dividing here would be a 1000x error).
    assert_eq!(pcs.run_delays_ns, vec![3000u64, 5000]);
    let rd_us: Vec<f64> = pcs.run_delays_ns.iter().map(|&v| v as f64 / 1000.0).collect();
    let mean_rd = rd_us.iter().sum::<f64>() / rd_us.len() as f64;
    let worst_rd = rd_us.iter().cloned().reduce(f64::max).unwrap();
    assert!((mean_rd - cg.mean_run_delay_us).abs() < 1e-9);
    assert!((worst_rd - cg.worst_run_delay_us).abs() < 1e-9);

    // Counters + the ratios they re-pool.
    assert_eq!(pcs.total_migrations, cg.total_migrations);
    assert_eq!(pcs.total_iterations, cg.total_iterations);
    assert_eq!(pcs.total_cpu_time_ns, cg.total_cpu_time_ns);
    assert!(
        (pcs.total_migrations as f64 / pcs.total_iterations as f64 - cg.migration_ratio).abs()
            < 1e-9
    );

    // NUMA totals: total = sum, cross-node = MAX (system-wide delta observed
    // redundantly), re-pooling to cross_node_migration_ratio.
    assert_eq!(pcs.numa_pages_total, 250);
    assert_eq!(pcs.cross_node_migrated, 10);
    assert!(
        (pcs.cross_node_migrated as f64 / pcs.numa_pages_total as f64
            - cg.cross_node_migration_ratio)
            .abs()
            < 1e-9
    );

    // numa_pages_local: the 6b addition cgroup_stats lacks. Pages on the
    // expected nodes {0,1}: w1 100+50=150, w2 80 (node 1) -> 230.
    assert_eq!(pcs.numa_pages_local, 230);
    assert_eq!(
        cg.page_locality, 0.0,
        "cgroup_stats has no node context; 6b captures the locality numerator",
    );
    // numa_pages_local's authority is assert_cgroup (the page_locality numerator),
    // NOT cgroup_stats. Cross-check the re-pool against the real reduction
    // value-for-value so a divergence in the partition node set would be caught.
    let cg_numa =
        crate::assert::Assert::NO_OVERRIDES.assert_cgroup_with_numa(&reports, None, Some(&nodes));
    let page_locality = cg_numa.stats.cgroups[0].page_locality;
    assert!(
        (pcs.numa_pages_local as f64 / pcs.numa_pages_total as f64 - page_locality).abs() < 1e-9,
        "numa_pages_local/total re-pools to assert_cgroup's page_locality ({page_locality})",
    );

    // Worst gap stays coupled to its CPU (argmax of the pair, not two maxes).
    assert_eq!((pcs.max_gap_ms, pcs.max_gap_cpu), (cg.max_gap_ms, cg.max_gap_cpu));
    assert_eq!((pcs.max_gap_ms, pcs.max_gap_cpu), (60, 2));
}

/// Boundary: NO worker has measurable wall time -> `off_cpu_pcts` is EMPTY (the
/// not-measured state), so the re-pool yields None for avg/min/max/spread —
/// preserving the not-measured-vs-measured-zero distinction cgroup_stats keeps
/// (`avg_off_cpu_pct` None), NOT a measured-0% / perfectly-fair reading.
#[test]
fn phase_cgroup_stats_off_cpu_pcts_empty_when_no_wall_time() {
    let reports = vec![rpt(1, 1000, 0, 0, &[0], 10), rpt(2, 1000, 0, 0, &[1], 10)];
    let pcs = phase_cgroup_stats(&reports, None);
    assert!(
        pcs.off_cpu_pcts.is_empty(),
        "no measurable wall time -> not-measured (empty), not a measured zero",
    );
    let cg = cgroup_stats(&reports);
    assert!(cg.avg_off_cpu_pct.is_none(), "mirrors cgroup_stats None");
}

/// Boundary (the OTHER half of the not-measured-vs-measured-zero distinction): a
/// worker with measurable wall time (wall_time_ns > 0) and off_cpu_ns == 0 yields
/// a PRESENT off_cpu_pcts sample of 0.0 — a measured zero, NOT an absent/empty
/// vec. cgroup_stats correspondingly reports `Some(0.0)`, not `None`. A future
/// tightening of the wall_time_ns>0 filter to off_cpu_ns>0 would collapse this
/// into the not-measured case; this pins against that silent regression.
#[test]
fn phase_cgroup_stats_off_cpu_pcts_measured_zero_is_present() {
    // wall 5000 > 0, off 0 -> off-CPU% == 0.0 (measured), not filtered out.
    let reports = vec![rpt(1, 1000, 5000, 0, &[0], 10)];
    let pcs = phase_cgroup_stats(&reports, None);
    assert_eq!(
        pcs.off_cpu_pcts,
        vec![0.0],
        "measured zero is a PRESENT 0.0 sample, distinct from the empty not-measured vec",
    );
    let cg = cgroup_stats(&reports);
    assert_eq!(cg.avg_off_cpu_pct, Some(0.0), "mirrors cgroup_stats Some(0.0), not None");
}

/// Boundary (MIXED): a cgroup with SOME workers wall_time_ns==0 and others
/// wall_time_ns>0. The `wall_time_ns > 0` filter must EXCLUDE the zero-wall
/// workers from off_cpu_pcts (no div-by-zero NaN) while still counting them in
/// num_workers. The surviving sample matches the wall>0 worker, and the re-pool
/// equals cgroup_stats over the same mixed reports value-for-value.
#[test]
fn phase_cgroup_stats_off_cpu_pcts_mixed_filters_zero_wall_workers() {
    let reports = vec![
        rpt(1, 1000, 0, 0, &[0], 0),               // wall 0 -> excluded from off_cpu_pcts
        rpt(2, 1000, 1_000_000, 200_000, &[1], 0), // wall>0 -> off-CPU% 20.0
    ];
    let pcs = phase_cgroup_stats(&reports, None);
    assert_eq!(pcs.off_cpu_pcts, vec![20.0], "only the wall>0 worker contributes a sample");
    assert_eq!(pcs.num_workers, 2, "the zero-wall worker still counts toward num_workers");
    // Re-pool matches cgroup_stats over the same mixed reports.
    let cg = cgroup_stats(&reports);
    let avg = pcs.off_cpu_pcts.iter().sum::<f64>() / pcs.off_cpu_pcts.len() as f64;
    assert!((avg - cg.avg_off_cpu_pct.unwrap()).abs() < 1e-9, "avg re-pools to cgroup_stats");
    assert!((20.0 - cg.max_off_cpu_pct.unwrap()).abs() < 1e-9, "max matches the lone sample");
}

/// Boundary: without an expected-node set `numa_pages_local` is 0 (mirrors
/// cgroup_stats leaving page_locality 0.0 absent NUMA context), while
/// `numa_pages_total` is still the sum so cross_node_migration_ratio re-pools.
#[test]
fn phase_cgroup_stats_numa_local_zero_without_expected_nodes() {
    let reports = vec![WorkerReport {
        numa_pages: BTreeMap::from([(0usize, 100u64), (1, 40)]),
        ..rpt(1, 1, 1000, 0, &[0], 0)
    }];
    let pcs = phase_cgroup_stats(&reports, None);
    assert_eq!(
        pcs.numa_pages_local, 0,
        "no node set -> 0 numerator (page_locality re-pools to 0.0)",
    );
    assert_eq!(pcs.numa_pages_total, 140, "total still computed without node context");
}

/// Boundary: the partition counts ONLY pages on the expected nodes, exactly as
/// `AssertPlan::assert_cgroup` does (`nodes.contains` gate, summed across
/// workers and numa_maps node entries).
#[test]
fn phase_cgroup_stats_numa_local_partitions_on_expected_nodes() {
    let reports = vec![
        WorkerReport {
            numa_pages: BTreeMap::from([(0usize, 100u64), (3, 25)]),
            ..rpt(1, 1, 1000, 0, &[0], 0)
        },
        WorkerReport {
            numa_pages: BTreeMap::from([(1usize, 60u64), (2, 15)]),
            ..rpt(2, 1, 1000, 0, &[1], 0)
        },
    ];
    let nodes: BTreeSet<usize> = [0, 1].into_iter().collect();
    let pcs = phase_cgroup_stats(&reports, Some(&nodes));
    // local = node0(100) + node1(60) = 160; off-node 3(25)+2(15) excluded.
    assert_eq!(pcs.numa_pages_local, 160);
    assert_eq!(pcs.numa_pages_total, 200);
}

/// Boundary: `vmstat_numa_pages_migrated` is a SYSTEM-WIDE delta each worker
/// observes redundantly, so the fold is MAX, not SUM (summing would inflate by
/// the worker count).
#[test]
fn phase_cgroup_stats_cross_node_migrated_is_max_not_sum() {
    let reports = vec![
        WorkerReport { vmstat_numa_pages_migrated: 30, ..rpt(1, 1, 1000, 0, &[0], 0) },
        WorkerReport { vmstat_numa_pages_migrated: 20, ..rpt(2, 1, 1000, 0, &[1], 0) },
    ];
    let pcs = phase_cgroup_stats(&reports, None);
    assert_eq!(pcs.cross_node_migrated, 30, "MAX (30), not SUM (50)");
}

/// Boundary: the worst scheduling gap and its CPU are taken TOGETHER from the
/// argmax worker — never two independent maxes (which would pair 90 with cpu 7).
#[test]
fn phase_cgroup_stats_gap_argmax_couples_ms_and_cpu() {
    let reports = vec![
        WorkerReport { max_gap_ms: 40, max_gap_cpu: 7, ..rpt(1, 1, 1000, 0, &[0], 0) },
        WorkerReport { max_gap_ms: 90, max_gap_cpu: 3, ..rpt(2, 1, 1000, 0, &[1], 0) },
        WorkerReport { max_gap_ms: 10, max_gap_cpu: 5, ..rpt(3, 1, 1000, 0, &[2], 0) },
    ];
    let pcs = phase_cgroup_stats(&reports, None);
    assert_eq!(
        (pcs.max_gap_ms, pcs.max_gap_cpu),
        (90, 3),
        "argmax keeps the gap bound to its CPU (independent maxes would give (90, 7))",
    );
}

/// num_workers is a Counter (SUM), not a Peak (MAX): a multi-WorkSpec cgroup
/// (`.work(4).work(2)`) emits ONE carrier per handle under the SAME name
/// (`apply_setup` -> `collect_handles`), and `AssertResult::merge` folds those
/// same-name carriers via `PhaseCgroupStats::merge`. The carriers cover DISJOINT
/// worker subsets, so num_workers must SUM (4 + 2 = 6) to match `cgroup_stats`
/// over the pooled reports (`reports.len()` = 6); a MAX (4) would inflate the
/// re-pooled `iterations_per_worker` 1.5x.
#[test]
fn phase_cgroup_stats_num_workers_sums_across_same_name_carriers() {
    // Two handles for one cgroup: 4 workers (100 iters each) + 2 (200 each).
    let reports1: Vec<WorkerReport> = (0..4)
        .map(|i| WorkerReport { iterations: 100, ..rpt(i, 1, 1000, 0, &[i as usize], 0) })
        .collect();
    let reports2: Vec<WorkerReport> = (4..6)
        .map(|i| WorkerReport { iterations: 200, ..rpt(i, 1, 1000, 0, &[i as usize], 0) })
        .collect();
    let pcs1 = phase_cgroup_stats(&reports1, None);
    let pcs2 = phase_cgroup_stats(&reports2, None);
    assert_eq!(pcs1.num_workers, 4);
    assert_eq!(pcs2.num_workers, 2);
    let merged = PhaseCgroupStats::merge(pcs1, pcs2);
    assert_eq!(merged.num_workers, 6, "disjoint worker subsets SUM, not MAX(4,2)=4");
    assert_eq!(merged.total_iterations, 4 * 100 + 2 * 200);

    // The merged components re-pool to the cgroup_stats reduction over the
    // POOLED reports (the run-level authority): num_workers = reports.len() = 6,
    // and iterations_per_worker = total_iterations / num_workers value-for-value.
    let mut pooled = reports1.clone();
    pooled.extend(reports2.clone());
    let cg = cgroup_stats(&pooled);
    assert_eq!(merged.num_workers, cg.num_workers);
    assert_eq!(merged.total_iterations, cg.total_iterations);
    let repooled = merged.total_iterations as f64 / merged.num_workers as f64;
    let cg_ipw = cg.iterations_per_worker().expect("pooled cgroup has workers");
    assert!(
        (repooled - cg_ipw).abs() < 1e-9,
        "re-pooled iterations_per_worker {repooled} must match cgroup_stats {cg_ipw} \
         (800/6); a MAX num_workers would give 800/4",
    );
}

/// The carrier RE-CAPS the pooled wake_latencies_ns at MAX_WAKE_SAMPLES so the
/// serialized AssertResult stays within the guest bulk-port frame even on a
/// many-core cgroup. Without the re-cap the pool would be
/// workers × MAX_WAKE_SAMPLES (here 120k) and could overrun the 16 MiB frame,
/// flipping a PASS to a truncated FAIL. wake_sample_total keeps the true count.
#[test]
fn phase_cgroup_stats_caps_pooled_wake_latencies() {
    use crate::workload::MAX_WAKE_SAMPLES;
    // DISTINCT values so distribution-preservation is testable: worker 1 carries
    // 0..60k, worker 2 carries 60k..120k -> population 0..120_000 (mean 59999.5),
    // pool 120k > MAX_WAKE_SAMPLES (100k). A constant fill would only pin len().
    let mk = |tid: i32, lo: u64, hi: u64| WorkerReport {
        wake_latencies_ns: (lo..hi).collect(),
        wake_sample_total: hi - lo,
        ..rpt(tid, 1, 1000, 0, &[0], 0)
    };
    let reports = vec![mk(1, 0, 60_000), mk(2, 60_000, 120_000)];
    let pcs = phase_cgroup_stats(&reports, None);
    assert_eq!(
        pcs.wake_latencies_ns.len(),
        MAX_WAKE_SAMPLES,
        "pooled wake_latencies re-capped to MAX_WAKE_SAMPLES (not the 120k concat)",
    );
    assert_eq!(
        pcs.wake_sample_total, 120_000,
        "true pre-cap population preserved for the re-pool",
    );
    // PARITY BOUNDARY (>cap): the carrier holds a reservoir SUBSAMPLE (len == cap)
    // of a larger population while cgroup_stats reduces over the FULL concat — the
    // documented distribution-equivalent (not value-for-value) divergence.
    assert!(
        pcs.wake_latencies_ns.len() < pcs.wake_sample_total as usize,
        "carrier is a subsample of the true population — the >cap divergence",
    );
    // DISTRIBUTION PRESERVATION (the load-bearing >cap claim). A
    // distribution-destroying reservoir bug (first-K, constant overwrite, fixed
    // replacement index) would shift the mean or collapse the range — caught
    // here, where the earlier constant-fill version was blind.
    //
    // (1) STATISTICAL tolerance: a uniform 100k-of-120k reservoir has a sample
    // mean tightly tracking the population mean (59999.5); the ±3000 band is ~27×
    // the ~110-wide sample-mean std error, so a valid reservoir never trips it
    // while a shifted distribution does.
    let mean = pcs.wake_latencies_ns.iter().map(|&v| v as f64).sum::<f64>()
        / pcs.wake_latencies_ns.len() as f64;
    assert!(
        (mean - 59_999.5).abs() < 3000.0,
        "reservoir mean {mean} tracks population mean 59999.5 (statistical tolerance)",
    );
    // (2) DETERMINISTIC bound: the reservoir holds 100k of 120k values, excluding
    // only 20k, so by pigeonhole min ≤ 20000 and max ≥ 99999 → range ≥ 99999
    // ALWAYS. Assert > 90_000 (comfortably inside the guaranteed 99999, never a
    // flake) to catch a constant/degenerate fill.
    let min = *pcs.wake_latencies_ns.iter().min().unwrap();
    let max = *pcs.wake_latencies_ns.iter().max().unwrap();
    assert!(max < 120_000, "every sample drawn from the 0..120_000 population");
    assert!(
        max - min > 90_000,
        "reservoir spans the population (guaranteed range ≥ 99999), not a constant fill",
    );
}

/// PhaseCgroupStats::merge RE-CAPS the merged wake_latencies_ns at
/// MAX_WAKE_SAMPLES. Same-name carriers (a multi-WorkSpec cgroup's per-handle
/// carriers) merge ON THE GUEST before the AssertResult is serialized, so without
/// the re-cap the merged pool would be K × MAX_WAKE_SAMPLES and could overrun the
/// 16 MiB bulk frame (a PASS flipped to a truncated FAIL). A concat already ≤ cap
/// passes through unchanged (value-for-value parity for small pools).
#[test]
fn phase_cgroup_stats_merge_caps_pooled_wake_latencies() {
    use crate::workload::MAX_WAKE_SAMPLES;
    // DISTINCT value ranges so the merged reservoir's distribution is testable.
    let carrier = |lo: u64, hi: u64| PhaseCgroupStats {
        wake_latencies_ns: (lo..hi).collect(),
        wake_sample_total: hi - lo,
        ..Default::default()
    };
    // Two carriers each AT the cap (0..100k and 100k..200k, each ≤cap so neither
    // is pre-subsampled) -> concat 200k = the TRUE 0..200_000 population (mean
    // 99999.5) -> re-capped back to cap.
    let merged = PhaseCgroupStats::merge(carrier(0, 100_000), carrier(100_000, 200_000));
    assert_eq!(
        merged.wake_latencies_ns.len(),
        MAX_WAKE_SAMPLES,
        "merged wake_latencies re-capped to MAX_WAKE_SAMPLES (not 2×cap)",
    );
    assert_eq!(
        merged.wake_sample_total,
        2 * MAX_WAKE_SAMPLES as u64,
        "true pre-cap population SUMs across carriers",
    );
    // Distribution preservation: the merged reservoir's mean tracks the 0..200_000
    // population mean (99999.5). (Both inputs were ≤cap so the concat is the true
    // population; the reservoir-of-reservoirs bias only appears when EACH input is
    // itself >cap with differing populations — see the merge bias note.)
    let mean = merged.wake_latencies_ns.iter().map(|&v| v as f64).sum::<f64>()
        / merged.wake_latencies_ns.len() as f64;
    assert!(
        (mean - 99_999.5).abs() < 3000.0,
        "merged reservoir mean {mean} tracks population mean 99999.5",
    );
    // A merge whose concat is ≤ cap passes through unchanged (no re-sample).
    let small = PhaseCgroupStats::merge(carrier(0, 10), carrier(100, 120));
    assert_eq!(small.wake_latencies_ns.len(), 30, "≤cap concat not re-sampled");
}

/// `AssertResult::strip_phase_cgroup_samples` — the graceful-degradation lever
/// `send_test_result` uses when the serialized result would overrun the bulk
/// frame: it drops every per_cgroup carrier's RAW sample vectors but PRESERVES
/// the counters, `wake_sample_total`, the coupled gap, and the verdict (no
/// PASS→FAIL flip). Pins the drop scope + the dropped-sample count.
#[test]
fn strip_phase_cgroup_samples_drops_only_sample_vecs_preserving_verdict() {
    let mut pc = BTreeMap::new();
    pc.insert(
        "cg".to_string(),
        PhaseCgroupStats {
            num_workers: 3,
            wake_latencies_ns: vec![1, 2, 3],
            wake_sample_total: 99,
            run_delays_ns: vec![10, 20],
            off_cpu_pcts: vec![5.0],
            total_iterations: 42,
            max_gap_ms: 7,
            max_gap_cpu: 2,
            ..Default::default()
        },
    );
    let mut r = crate::assert::AssertResult::pass();
    r.stats.phases = vec![PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        metrics: BTreeMap::new(),
        per_cgroup: pc,
    }];
    let dropped = r.strip_phase_cgroup_samples();
    assert_eq!(dropped, 3 + 2 + 1, "wake(3) + run(2) + off(1) samples dropped");
    let cg = &r.stats.phases[0].per_cgroup["cg"];
    assert!(cg.wake_latencies_ns.is_empty(), "wake samples dropped");
    assert!(cg.run_delays_ns.is_empty(), "run-delay samples dropped");
    assert!(cg.off_cpu_pcts.is_empty(), "off-CPU samples dropped");
    // Counters + reduced fields preserved.
    assert_eq!(cg.num_workers, 3);
    assert_eq!(cg.wake_sample_total, 99, "true population preserved");
    assert_eq!(cg.total_iterations, 42);
    assert_eq!((cg.max_gap_ms, cg.max_gap_cpu), (7, 2));
    assert!(r.is_pass(), "verdict preserved (no PASS->FAIL flip)");
}

/// PhaseCgroupStats::merge breaks a max_gap_ms TIE toward `b` (last-wins),
/// matching the builders' `max_by_key` (which returns the LAST max on a tie). On
/// equal gaps with different CPUs the later carrier's CPU wins — keeping the
/// cross-carrier merge consistent with a single cgroup_stats over pooled reports.
#[test]
fn phase_cgroup_stats_merge_gap_tie_breaks_to_b() {
    let a = PhaseCgroupStats { max_gap_ms: 5, max_gap_cpu: 3, ..Default::default() };
    let b = PhaseCgroupStats { max_gap_ms: 5, max_gap_cpu: 7, ..Default::default() };
    let merged = PhaseCgroupStats::merge(a, b);
    assert_eq!(
        (merged.max_gap_ms, merged.max_gap_cpu),
        (5, 7),
        "equal gap -> b (last) wins, matching max_by_key last-wins",
    );
}

/// Cross-carrier gap parity: merging two same-name carriers with EQUAL max_gap_ms
/// (different CPU) yields the SAME (ms, cpu) as cgroup_stats over the concatenated
/// reports IN FOLD ORDER — pinning that the merge's last-wins tie-break stays
/// coupled to the pooled-report order (a reordered fold would desync the CPU).
#[test]
fn phase_cgroup_stats_merge_gap_tie_matches_pooled_cgroup_stats() {
    let r1 = vec![WorkerReport { max_gap_ms: 8, max_gap_cpu: 1, ..rpt(1, 1, 1000, 0, &[1], 0) }];
    let r2 = vec![WorkerReport { max_gap_ms: 8, max_gap_cpu: 9, ..rpt(2, 1, 1000, 0, &[9], 0) }];
    let merged =
        PhaseCgroupStats::merge(phase_cgroup_stats(&r1, None), phase_cgroup_stats(&r2, None));
    // cgroup_stats over the concatenation (r1 ++ r2, fold order): max_by_key
    // returns the LAST max on the gap-8 tie -> cpu 9.
    let mut pooled = r1.clone();
    pooled.extend(r2.clone());
    let cg = cgroup_stats(&pooled);
    assert_eq!(
        (merged.max_gap_ms, merged.max_gap_cpu),
        (cg.max_gap_ms, cg.max_gap_cpu),
        "cross-carrier gap argmax matches cgroup_stats over pooled reports in fold order",
    );
    assert_eq!((merged.max_gap_ms, merged.max_gap_cpu), (8, 9));
}

/// The guest-side carrier: keyed by cgroup name, stamped with the 1-indexed
/// step_index, labeled via [`Phase`] Display, with the merge-neutral
/// `(u64::MAX, 0)` window and empty metrics — it carries ONLY per_cgroup.
#[test]
fn step_per_cgroup_bucket_keys_by_name_with_sentinel_window() {
    let reports = vec![WorkerReport {
        iterations: 10,
        schedstat_cpu_time_ns: 500,
        ..rpt(1, 1, 1000, 100, &[0], 5)
    }];
    let nodes: BTreeSet<usize> = [0].into_iter().collect();
    let b = step_per_cgroup_bucket("cg_step", &reports, Some(&nodes), 3);
    assert_eq!(b.step_index, 3);
    assert_eq!(b.label, "Step[2]", "Phase Display: 1-indexed step_index 3 -> Step[2]");
    assert_eq!(b.start_ms, u64::MAX, "merge-neutral: min() against host start is a no-op");
    assert_eq!(b.end_ms, 0, "merge-neutral: max() against host end is a no-op");
    assert_eq!(b.sample_count, 0);
    assert!(b.metrics.is_empty(), "carrier contributes only per_cgroup");
    assert_eq!(b.per_cgroup.len(), 1);
    let pc = &b.per_cgroup["cg_step"];
    assert_eq!(pc.total_iterations, 10);
    assert_eq!(pc.num_workers, 1);
}

/// The carrier's label for the BASELINE encoding (step_index 0) reads
/// "BASELINE", pinning the 1-indexed Phase encoding at the boundary.
#[test]
fn step_per_cgroup_bucket_baseline_label() {
    let b = step_per_cgroup_bucket("cg", &[], None, 0);
    assert_eq!(b.label, "BASELINE");
    assert_eq!(b.step_index, 0);
    assert_eq!(b.per_cgroup["cg"].num_workers, 0, "empty reports -> zero-component carrier");
}

/// The fold unions a guest carrier's per_cgroup into the host bucket of the
/// SAME step_index, leaving the host's window / metrics / sample_count intact
/// (the carrier's MAX/0 window and empty metrics are merge-neutral).
#[test]
fn fold_unions_guest_per_cgroup_into_matching_host_bucket() {
    // Host bucket as build_phase_buckets_with_stimulus produces it: the
    // iteration_rate Rate ALONGSIDE its two Counter components (50 = 1000/20),
    // plus a plain non-rate metric. merge_matched_phase_buckets skips Rate keys
    // and re-derives them from the components, so a realistic host bucket keeps
    // its rate; the components must be present (a bare rate would be dropped).
    let host = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 100,
        end_ms: 200,
        sample_count: 3,
        metrics: BTreeMap::from([
            ("iteration_rate".to_string(), 50.0),
            ("total_phase_iterations".to_string(), 1000.0),
            ("total_phase_duration_sec".to_string(), 20.0),
            ("worst_spread".to_string(), 0.42),
        ]),
        per_cgroup: BTreeMap::new(),
    };
    let mut g_pc = BTreeMap::new();
    g_pc.insert("cgA".to_string(), PhaseCgroupStats { total_iterations: 42, ..Default::default() });
    let guest = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: BTreeMap::new(),
        per_cgroup: g_pc,
    };
    let out = fold_guest_per_cgroup_into_host_buckets(vec![host], vec![guest]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].step_index, 1);
    assert_eq!(out[0].start_ms, 100, "host window start preserved (min vs MAX)");
    assert_eq!(out[0].end_ms, 200, "host window end preserved (max vs 0)");
    assert_eq!(out[0].sample_count, 3, "host sample_count preserved (+0)");
    assert_eq!(
        out[0].metrics.get("worst_spread").copied(),
        Some(0.42),
        "host non-rate metric preserved through the carrier merge",
    );
    assert_eq!(
        out[0].metrics.get("iteration_rate").copied(),
        Some(50.0),
        "host Rate re-derived from its carried components (1000/20), not dropped",
    );
    assert_eq!(out[0].per_cgroup["cgA"].total_iterations, 42, "guest per_cgroup unioned in");
}

/// Negative direction of the Rate-survival contract: a host bucket carrying a
/// Rate metric WITHOUT its Counter components LOSES the rate through the carrier
/// merge — merge_matched_phase_buckets skips Rate keys and re-derives them, and
/// derive_rate_metrics cannot re-derive from absent components. This drop is
/// SAFE in production only because build_phase_buckets_with_stimulus co-inserts
/// total_phase_iterations + total_phase_duration_sec alongside iteration_rate
/// (and iterations_per_cpu_sec lands only in ext_metrics, never PhaseBucket
/// metrics), so no production host bucket carries a component-less Rate. This
/// test pins the failure mode so a future producer violating that contract
/// surfaces here rather than as a silently-dropped metric.
#[test]
fn fold_drops_host_rate_lacking_its_components() {
    let host = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        // iteration_rate WITHOUT its total_phase_iterations /
        // total_phase_duration_sec components — a contract violation no
        // production path produces.
        metrics: BTreeMap::from([("iteration_rate".to_string(), 50.0)]),
        per_cgroup: BTreeMap::new(),
    };
    let mut g_pc = BTreeMap::new();
    g_pc.insert("cg".to_string(), PhaseCgroupStats { total_iterations: 1, ..Default::default() });
    let guest = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: BTreeMap::new(),
        per_cgroup: g_pc,
    };
    let out = fold_guest_per_cgroup_into_host_buckets(vec![host], vec![guest]);
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].metrics.get("iteration_rate"),
        None,
        "a component-less Rate is dropped by the carrier merge's re-derive; \
         production never produces this (build_phase_buckets co-inserts the \
         components), so the drop is unreachable in practice",
    );
    // The per_cgroup payload still folds in regardless of the dropped Rate.
    assert_eq!(out[0].per_cgroup["cg"].total_iterations, 1);
}

/// A guest step_index with no host bucket — the DEFENSIVE case where the
/// carrier's step has no StepStart frame in the host stimulus (the host
/// synthesizes a bucket for every StepStart-step, so a captured-but-short step
/// takes the matched arm, not this) — is carried verbatim, NOT dropped, with its
/// window normalized to (0, 0) so duration consumers don't underflow the
/// merge-neutral sentinel. Output stays sorted by step_index.
#[test]
fn fold_carries_orphan_guest_step_index_with_normalized_window() {
    let host = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        metrics: BTreeMap::new(),
        per_cgroup: BTreeMap::new(),
    };
    let mut g_pc = BTreeMap::new();
    g_pc.insert("cgB".to_string(), PhaseCgroupStats { total_migrations: 7, ..Default::default() });
    let guest = PhaseBucket {
        step_index: 5,
        label: "Step[4]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: BTreeMap::new(),
        per_cgroup: g_pc,
    };
    let out = fold_guest_per_cgroup_into_host_buckets(vec![host], vec![guest]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].step_index, 1, "sorted by step_index");
    assert_eq!(out[1].step_index, 5);
    let orphan = &out[1];
    assert_eq!(orphan.start_ms, 0, "orphan window normalized (no underflow)");
    assert_eq!(orphan.end_ms, 0, "orphan window normalized");
    assert_eq!(orphan.label, "Step[4]", "orphan carrier label preserved");
    assert_eq!(orphan.per_cgroup["cgB"].total_migrations, 7, "orphan per_cgroup not dropped");

    // Saturation boundary: a >65k-step scenario collapses step_index to u16::MAX
    // (the step loop + build_stimulus both saturate). An orphan there is carried
    // identically — (0,0) window, verbatim per_cgroup — since the fold's
    // BTreeMap<u16,_> keys any value, the sentinel included.
    let mut sat_pc = BTreeMap::new();
    sat_pc.insert(
        "cgSat".to_string(),
        PhaseCgroupStats { total_iterations: 3, ..Default::default() },
    );
    let sat_guest = PhaseBucket {
        step_index: u16::MAX,
        label: "Step[65534]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: BTreeMap::new(),
        per_cgroup: sat_pc,
    };
    let sat_out = fold_guest_per_cgroup_into_host_buckets(vec![], vec![sat_guest]);
    assert_eq!(sat_out.len(), 1);
    assert_eq!(sat_out[0].step_index, u16::MAX);
    assert_eq!(
        (sat_out[0].start_ms, sat_out[0].end_ms),
        (0, 0),
        "u16::MAX orphan window normalized",
    );
    assert_eq!(
        sat_out[0].per_cgroup["cgSat"].total_iterations, 3,
        "u16::MAX orphan per_cgroup carried verbatim",
    );
}

/// With NO guest carriers (a run with no step-local cgroups) the fold returns
/// the host buckets unchanged — the pre-6b behavior, which the
/// `phase_buckets_equals_stats_phases*` eval pins depend on.
#[test]
fn fold_empty_guest_passes_host_through_unchanged() {
    let host = vec![
        PhaseBucket {
            step_index: 0,
            label: "BASELINE".to_string(),
            start_ms: 0,
            end_ms: 50,
            sample_count: 2,
            metrics: BTreeMap::from([("k".to_string(), 1.0)]),
            per_cgroup: BTreeMap::new(),
        },
        PhaseBucket {
            step_index: 1,
            label: "Step[0]".to_string(),
            start_ms: 50,
            end_ms: 100,
            sample_count: 3,
            metrics: BTreeMap::new(),
            per_cgroup: BTreeMap::new(),
        },
    ];
    let out = fold_guest_per_cgroup_into_host_buckets(host.clone(), vec![]);
    assert_eq!(out, host, "no guest carriers -> host buckets unchanged");
}

/// Multiple cgroups in one step land in a single carrier bucket (collect_handles
/// merges them); the fold carries all of them into the host bucket.
#[test]
fn fold_multiple_cgroups_in_one_step_all_carried() {
    let mut g_pc = BTreeMap::new();
    g_pc.insert("cgX".to_string(), PhaseCgroupStats { total_iterations: 10, ..Default::default() });
    g_pc.insert("cgY".to_string(), PhaseCgroupStats { total_iterations: 20, ..Default::default() });
    let guest = PhaseBucket {
        step_index: 2,
        label: "Step[1]".to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: BTreeMap::new(),
        per_cgroup: g_pc,
    };
    let host = PhaseBucket {
        step_index: 2,
        label: "Step[1]".to_string(),
        start_ms: 0,
        end_ms: 10,
        sample_count: 1,
        metrics: BTreeMap::new(),
        per_cgroup: BTreeMap::new(),
    };
    let out = fold_guest_per_cgroup_into_host_buckets(vec![host], vec![guest]);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].per_cgroup.len(), 2);
    assert_eq!(out[0].per_cgroup["cgX"].total_iterations, 10);
    assert_eq!(out[0].per_cgroup["cgY"].total_iterations, 20);
}

/// Defensive: two guest carriers sharing a step_index (the serialized guest
/// result is already per-step-merged, but the fold must not double-count or drop
/// either) merge their same-named per_cgroup components sequentially.
#[test]
fn fold_duplicate_guest_step_index_merges_sequentially() {
    let make = |iters: u64| {
        let mut pc = BTreeMap::new();
        pc.insert(
            "cgZ".to_string(),
            PhaseCgroupStats { total_iterations: iters, ..Default::default() },
        );
        PhaseBucket {
            step_index: 3,
            label: "Step[2]".to_string(),
            start_ms: u64::MAX,
            end_ms: 0,
            sample_count: 0,
            metrics: BTreeMap::new(),
            per_cgroup: pc,
        }
    };
    let out = fold_guest_per_cgroup_into_host_buckets(vec![], vec![make(5), make(8)]);
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].per_cgroup["cgZ"].total_iterations,
        13,
        "5 + 8 summed (Counter): neither carrier dropped or double-counted",
    );
    // The orphan window stays normalized (0,0) through the re-merge: the first
    // carrier hits the orphan arm and is normalized to (0,0) BEFORE the second
    // carrier merges via min(0, MAX)=0 / max(0, 0)=0 — so the MAX sentinel never
    // reaches a duration consumer.
    assert_eq!((out[0].start_ms, out[0].end_ms), (0, 0), "orphan window normalized");
}

/// Defensive boundary: a guest carrier MUST carry the merge-neutral
/// `(u64::MAX, 0)` sentinel window. The fold validates this BEFORE the
/// matched/orphan dispatch, so BOTH arms are guarded — the matched arm relies on
/// the window being merge-neutral (min/max no-ops against the host window) and
/// the orphan arm normalizes it to (0,0). A carrier with a real window matching a
/// host bucket would otherwise silently corrupt the merged window; the
/// `debug_assert!` trips loudly in test builds instead. Host bucket present at the
/// same step_index so the carrier WOULD take the (dangerous) matched arm — the
/// assert fires first.
#[test]
#[should_panic(expected = "guest carrier must carry the merge-neutral")]
fn fold_panics_on_non_sentinel_guest_window() {
    let host = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        metrics: BTreeMap::new(),
        per_cgroup: BTreeMap::new(),
    };
    let mut g_pc = BTreeMap::new();
    g_pc.insert("cg".to_string(), PhaseCgroupStats { total_iterations: 1, ..Default::default() });
    // A guest carrier with a REAL window (not the (u64::MAX, 0) sentinel) — the
    // step_per_cgroup_bucket invariant violated.
    let bad = PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 100,
        end_ms: 200,
        sample_count: 0,
        metrics: BTreeMap::new(),
        per_cgroup: g_pc,
    };
    let _ = fold_guest_per_cgroup_into_host_buckets(vec![host], vec![bad]);
}

/// Defensive boundary: host buckets MUST have unique step_index. The fold keys
/// them into a `BTreeMap<u16, _>`, so a duplicate would silently MERGE one bucket
/// into the other (a dropped phase, not a panic, in release); the
/// `debug_assert_eq!` on the map size vs the input count trips loudly in test
/// builds instead.
#[test]
#[should_panic(expected = "host buckets must have unique step_index")]
fn fold_panics_on_duplicate_host_step_index() {
    let dup = || PhaseBucket {
        step_index: 1,
        label: "Step[0]".to_string(),
        start_ms: 0,
        end_ms: 100,
        sample_count: 1,
        metrics: BTreeMap::new(),
        per_cgroup: BTreeMap::new(),
    };
    let _ = fold_guest_per_cgroup_into_host_buckets(vec![dup(), dup()], vec![]);
}

// -- Item 7 run-level distributional re-pool (populate_run_distribution_metrics) --

use super::{CgroupStats, populate_run_distribution_metrics, populate_run_distribution_metrics_from};
use crate::stats::{MetricKind, SampleReduction, SampleSource};

/// Helper: a `ScenarioStats` whose single phase carries `per_cgroup` keyed by
/// name, plus optional run-level `cgroups` for the WorstLowest path / the
/// stripped-run Distribution fallback.
fn repool_stats(
    carriers: Vec<(&str, PhaseCgroupStats)>,
    cgroups: Vec<CgroupStats>,
) -> ScenarioStats {
    let mut bucket = PhaseBucket::default();
    for (name, pcg) in carriers {
        bucket.per_cgroup.insert(name.to_string(), pcg);
    }
    ScenarioStats {
        phases: vec![bucket],
        cgroups,
        ..ScenarioStats::default()
    }
}

/// THE THESIS: the run-level wake p99 is the percentile over the COMBINED
/// cross-cgroup sample set, NOT the max of per-cgroup p99s. cg_a carries 100
/// low samples (1..100 µs); cg_b carries 3 high samples (1000/2000/3000 µs).
/// The pooled p99 over the 103-sample union is 2000 µs, while the deleted
/// max-of-per-cgroup fold would report cg_b's own p99 (3000 µs).
#[test]
fn repool_distribution_pools_wake_across_cgroups_not_max_of_per_cgroup() {
    let cg_a = PhaseCgroupStats {
        wake_latencies_ns: (1..=100u64).map(|v| v * 1000).collect(),
        wake_sample_total: 100,
        ..PhaseCgroupStats::default()
    };
    let cg_b = PhaseCgroupStats {
        wake_latencies_ns: vec![1_000_000, 2_000_000, 3_000_000],
        wake_sample_total: 3,
        ..PhaseCgroupStats::default()
    };
    let mut stats = repool_stats(vec![("a", cg_a), ("b", cg_b)], vec![]);
    populate_run_distribution_metrics(&mut stats);

    let p99 = stats.ext_metrics.get("worst_p99_wake_latency_us").copied();
    assert_eq!(
        p99,
        Some(2000.0),
        "pooled cross-cgroup p99 over the 103-sample union, got {p99:?}",
    );
    assert_ne!(
        p99,
        Some(3000.0),
        "must NOT be the max of per-cgroup p99s (cg_b's 3000 µs) — the thesis",
    );
}

/// Empty input (no phases, no cgroups) writes NO Distribution or WorstLowest
/// key — absence is preserved, never a synthesized 0.0 (mirrors the
/// derive_rate_metrics both-or-neither contract).
#[test]
fn repool_distribution_empty_inserts_no_keys() {
    let mut stats = ScenarioStats::default();
    populate_run_distribution_metrics(&mut stats);
    for name in [
        "worst_p99_wake_latency_us",
        "worst_median_wake_latency_us",
        "worst_wake_latency_cv",
        "worst_mean_run_delay_us",
        "worst_run_delay_us",
        "worst_iterations_per_worker",
        "worst_iterations_per_cpu_sec",
        "worst_wake_latency_tail_ratio",
    ] {
        assert!(
            !stats.ext_metrics.contains_key(name),
            "{name} must be absent for empty input",
        );
    }
}

/// Single-sample boundary: p99 == median == the sole sample (µs), CV == 0
/// (stddev of one sample is 0).
#[test]
fn repool_distribution_single_sample() {
    let cg = PhaseCgroupStats {
        wake_latencies_ns: vec![42_000],
        wake_sample_total: 1,
        ..PhaseCgroupStats::default()
    };
    let mut stats = repool_stats(vec![("a", cg)], vec![]);
    populate_run_distribution_metrics(&mut stats);
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(42.0),
    );
    assert_eq!(
        stats
            .ext_metrics
            .get("worst_median_wake_latency_us")
            .copied(),
        Some(42.0),
    );
    assert_eq!(
        stats.ext_metrics.get("worst_wake_latency_cv").copied(),
        Some(0.0),
    );
}

/// Value-for-value: build the carrier from the SAME reports cgroup_stats
/// reduces (single cgroup, ≤ cap so the carrier IS the full pool), run the
/// actual re-pool, and assert ext_metrics reproduces cgroup_stats's wake /
/// run-delay reductions exactly.
#[test]
fn repool_distribution_value_for_value_with_cgroup_stats() {
    let reports = vec![
        WorkerReport {
            wake_latencies_ns: vec![1000, 2000, 3000, 4000, 5000],
            schedstat_run_delay_ns: 7000,
            schedstat_cpu_time_ns: 1_000_000,
            ..rpt(1, 1000, 1_000_000, 0, &[0], 0)
        },
        WorkerReport {
            wake_latencies_ns: vec![6000, 7000, 8000, 9000, 10000],
            schedstat_run_delay_ns: 3000,
            schedstat_cpu_time_ns: 1_000_000,
            ..rpt(2, 1000, 1_000_000, 0, &[1], 0)
        },
    ];
    let cg = cgroup_stats(&reports);
    let carrier = phase_cgroup_stats(&reports, None);
    let mut stats = repool_stats(vec![("a", carrier)], vec![]);
    populate_run_distribution_metrics(&mut stats);
    let ext = |n: &str| stats.ext_metrics.get(n).copied().unwrap();
    assert!((ext("worst_p99_wake_latency_us") - cg.p99_wake_latency_us).abs() < 1e-9);
    assert!((ext("worst_median_wake_latency_us") - cg.median_wake_latency_us).abs() < 1e-9);
    assert!((ext("worst_wake_latency_cv") - cg.wake_latency_cv).abs() < 1e-9);
    assert!((ext("worst_mean_run_delay_us") - cg.mean_run_delay_us).abs() < 1e-9);
    assert!((ext("worst_run_delay_us") - cg.worst_run_delay_us).abs() < 1e-9);
}

/// `WakeLatencyTailRatio` producer contract: `populate_run_distribution_metrics`
/// emits the `worst_wake_latency_tail_ratio` ext key as the MAX over the
/// per-cgroup `CgroupStats::wake_latency_tail_ratio` values — but ONLY when the
/// run cleared the `WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS` floor AND at least
/// one cgroup carried a measurable tail (median > 0). Below the floor, or with
/// no measurable tail, the key is ABSENT (excluded from the cross-RUN mean,
/// read as None by compare) — never a 0.0 sentinel. This is the floor gate the
/// deleted typed-field accessor used to enforce, now moved to the producer.
#[test]
fn wake_latency_tail_ratio_producer_floor_gates_and_maxes() {
    use crate::stats::WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS as MIN;
    let key = "worst_wake_latency_tail_ratio";
    // Two cgroups with measurable tails: ratios 10.0 (20/2) and 4.0 (8/2).
    let tail_cgroups = || {
        vec![
            CgroupStats {
                cgroup_name: "a".to_string(),
                p99_wake_latency_us: 20.0,
                median_wake_latency_us: 2.0,
                ..CgroupStats::default()
            },
            CgroupStats {
                cgroup_name: "b".to_string(),
                p99_wake_latency_us: 8.0,
                median_wake_latency_us: 2.0,
                ..CgroupStats::default()
            },
        ]
    };

    // Below the floor: no key, even though both cgroups carry a tail.
    let mut below = repool_stats(vec![], tail_cgroups());
    below.total_iterations = MIN - 1;
    populate_run_distribution_metrics(&mut below);
    assert_eq!(
        below.ext_metrics.get(key).copied(),
        None,
        "sub-threshold run must emit no tail-ratio key (floor gate at the producer)",
    );

    // Above the floor: key present, the MAX over per-cgroup ratios (10.0 > 4.0).
    let mut above = repool_stats(vec![], tail_cgroups());
    above.total_iterations = MIN;
    populate_run_distribution_metrics(&mut above);
    assert_eq!(
        above.ext_metrics.get(key).copied(),
        Some(10.0),
        "above-floor key must be the MAX over per-cgroup p99/median ratios",
    );

    // Above the floor but NO measurable tail (median 0 -> per-cgroup ratio 0.0):
    // absent, not a 0.0 sentinel.
    let mut no_tail = repool_stats(
        vec![],
        vec![CgroupStats {
            cgroup_name: "a".to_string(),
            p99_wake_latency_us: 0.0,
            median_wake_latency_us: 0.0,
            ..CgroupStats::default()
        }],
    );
    no_tail.total_iterations = MIN;
    populate_run_distribution_metrics(&mut no_tail);
    assert_eq!(
        no_tail.ext_metrics.get(key).copied(),
        None,
        "a run with no measurable tail (median 0) must emit no key, not Some(0.0)",
    );

    // Above the floor, MIXED: one cgroup with a tail (10/2 = 5.0), one with
    // median 0 (ratio 0.0). The `r > 0.0` guard skips the zero so it is NOT
    // folded into the max; the key is the surviving cgroup's tail, 5.0.
    let mut mixed = repool_stats(
        vec![],
        vec![
            CgroupStats {
                cgroup_name: "a".to_string(),
                p99_wake_latency_us: 10.0,
                median_wake_latency_us: 2.0,
                ..CgroupStats::default()
            },
            CgroupStats {
                cgroup_name: "b".to_string(),
                p99_wake_latency_us: 0.0,
                median_wake_latency_us: 0.0,
                ..CgroupStats::default()
            },
        ],
    );
    mixed.total_iterations = MIN;
    populate_run_distribution_metrics(&mut mixed);
    assert_eq!(
        mixed.ext_metrics.get(key).copied(),
        Some(5.0),
        "a median-0 cgroup (ratio 0.0) is skipped by the r>0.0 guard; the key is \
         the surviving cgroup's tail (10/2=5.0), not folded with the 0.0",
    );
}

// -- Item 8: per-phase per-cgroup display-time reductions (PhaseCgroupStats) --

/// `off_cpu_summary` boundary contract: `None` on empty (NOT measured), `Some`
/// on data — INCLUDING a measured zero, distinct from the `None` state.
/// avg/min/max/spread mirror cgroup_stats's off-CPU reduction (spread=max-min).
#[test]
fn phase_cgroup_off_cpu_summary_boundaries() {
    assert_eq!(PhaseCgroupStats::default().off_cpu_summary(), None);
    let one = PhaseCgroupStats {
        off_cpu_pcts: vec![42.0],
        ..Default::default()
    };
    assert_eq!(one.off_cpu_summary(), Some((42.0, 42.0, 42.0, 0.0)));
    let zeros = PhaseCgroupStats {
        off_cpu_pcts: vec![0.0, 0.0],
        ..Default::default()
    };
    assert_eq!(
        zeros.off_cpu_summary(),
        Some((0.0, 0.0, 0.0, 0.0)),
        "measured zeros are Some((0,..)), distinct from None (not-measured)",
    );
    let multi = PhaseCgroupStats {
        off_cpu_pcts: vec![10.0, 20.0, 30.0],
        ..Default::default()
    };
    assert_eq!(multi.off_cpu_summary(), Some((20.0, 10.0, 30.0, 20.0)));
}

/// `wake_summary` boundary: `None` on empty; single-sample p99 == median == the
/// sole sample (µs); nearest-rank percentile, ns→µs once.
#[test]
fn phase_cgroup_wake_summary_boundaries() {
    assert_eq!(PhaseCgroupStats::default().wake_summary(), None);
    let one = PhaseCgroupStats {
        wake_latencies_ns: vec![5000],
        ..Default::default()
    };
    assert_eq!(one.wake_summary(), Some((5.0, 5.0)));
    // 1000..10000ns: p99 nearest-rank idx ceil(10*0.99)-1=9 -> 10000ns=10µs;
    // median idx ceil(10*0.5)-1=4 -> 5000ns=5µs.
    let ten = PhaseCgroupStats {
        wake_latencies_ns: (1..=10u64).map(|v| v * 1000).collect(),
        ..Default::default()
    };
    assert_eq!(ten.wake_summary(), Some((10.0, 5.0)));
}

/// `run_delay_summary` boundary: `None` on empty; divides raw ns→µs ONCE
/// (mean 200µs / worst 300µs over [100_000, 300_000] ns) — not double-divided
/// (0.2) or un-divided (100_000).
#[test]
fn phase_cgroup_run_delay_summary_boundaries() {
    assert_eq!(PhaseCgroupStats::default().run_delay_summary(), None);
    let two = PhaseCgroupStats {
        run_delays_ns: vec![100_000, 300_000],
        ..Default::default()
    };
    assert_eq!(two.run_delay_summary(), Some((200.0, 300.0)));
}

/// Parity: a carrier built from the SAME reports cgroup_stats reduces yields
/// off-cpu / wake / run-delay summaries equal value-for-value (≤cap) to
/// cgroup_stats's fields — the per-phase render reproduces the whole-run
/// reduction when the phase spans the whole run.
#[test]
fn phase_cgroup_summaries_match_cgroup_stats() {
    let reports = vec![
        WorkerReport {
            wake_latencies_ns: vec![1000, 2000, 3000, 4000, 5000],
            schedstat_run_delay_ns: 7000,
            iterations: 100,
            ..rpt(1, 1000, 10_000_000, 4_000_000, &[0], 0)
        },
        WorkerReport {
            wake_latencies_ns: vec![6000, 7000, 8000, 9000, 10000],
            schedstat_run_delay_ns: 3000,
            iterations: 100,
            ..rpt(2, 1000, 10_000_000, 1_000_000, &[1], 0)
        },
    ];
    let cg = cgroup_stats(&reports);
    let carrier = phase_cgroup_stats(&reports, None);
    let (avg, min, max, spread) = carrier.off_cpu_summary().expect("off-cpu measured");
    assert!((avg - cg.avg_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((min - cg.min_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((max - cg.max_off_cpu_pct.unwrap()).abs() < 1e-9);
    assert!((spread - cg.spread.unwrap()).abs() < 1e-9);
    let (p99, median) = carrier.wake_summary().expect("wake measured");
    assert!((p99 - cg.p99_wake_latency_us).abs() < 1e-9);
    assert!((median - cg.median_wake_latency_us).abs() < 1e-9);
    let (mean, worst) = carrier.run_delay_summary().expect("run-delay measured");
    assert!((mean - cg.mean_run_delay_us).abs() < 1e-9);
    assert!((worst - cg.worst_run_delay_us).abs() < 1e-9);
}

/// run_delay_summary's mean is f64-ULP-equivalent (not bit-exact) to
/// cgroup_stats — Σns/n/1000 vs Σ(ns/1000)/n reassociate differently. This
/// pins the documented 1e-9 bound with a DIVERGENT input (the value-for-value
/// parity test above uses run-delays that are bit-exact in BOTH reassociations,
/// so its tolerance is dead): these three schedstat_run_delay_ns values make
/// the two means differ at the float level (~1e-12), so the < 1e-9 assert is
/// load-bearing — a reassociation/precision regression would exceed it.
#[test]
fn phase_cgroup_run_delay_mean_within_ulp_of_cgroup_stats() {
    let reports: Vec<WorkerReport> = [8_865_093u64, 9_991_834, 9_627_760]
        .iter()
        .enumerate()
        .map(|(i, &rd)| WorkerReport {
            schedstat_run_delay_ns: rd,
            ..rpt(i as i32 + 1, 1000, 1_000_000, 0, &[i], 0)
        })
        .collect();
    let cg = cgroup_stats(&reports);
    let carrier = phase_cgroup_stats(&reports, None);
    let (mean, _worst) = carrier.run_delay_summary().expect("run-delay measured");
    let delta = (mean - cg.mean_run_delay_us).abs();
    assert!(delta < 1e-9, "mean within 1e-9 of cgroup_stats; delta={delta:e}");
    assert!(
        delta > 0.0,
        "inputs must actually DIVERGE so the 1e-9 tolerance is load-bearing, not dead; delta={delta:e}",
    );
}

/// Carrier-name dedup across STEPS: a cgroup NAME that carries samples in
/// ANY phase is in `*_carriers`, so EVERY `stats.cgroups` entry with that
/// name — including a separate (handle, step) entry merged in for a later
/// step, as `AssertResult::merge` produces — is skipped from the
/// carrier-less worst-wins fold. The pooled percentile stays the union of
/// the carried samples; bogus typed reductions on the same-named cgroups
/// entries never leak in. Pins the disjointness-holds-within-a-step-not-
/// across-steps contract documented on populate_run_distribution_metrics.
#[test]
fn repool_distribution_carrier_name_dedup_skips_same_name_cgroups_across_steps() {
    let step0 = PhaseCgroupStats {
        wake_latencies_ns: vec![1000, 2000],
        wake_sample_total: 2,
        ..PhaseCgroupStats::default()
    };
    let step1 = PhaseCgroupStats {
        wake_latencies_ns: vec![3000, 4000],
        wake_sample_total: 2,
        ..PhaseCgroupStats::default()
    };
    let mut b0 = PhaseBucket {
        step_index: 1,
        ..PhaseBucket::default()
    };
    b0.per_cgroup.insert("a".to_string(), step0);
    let mut b1 = PhaseBucket {
        step_index: 2,
        ..PhaseBucket::default()
    };
    b1.per_cgroup.insert("a".to_string(), step1);
    // Two same-named cgroups entries (one per step, as AssertResult::merge
    // produces) carrying BOGUS huge p99 reductions that must NOT leak in.
    let cgroups = vec![
        CgroupStats {
            cgroup_name: "a".to_string(),
            p99_wake_latency_us: 9999.0,
            ..CgroupStats::default()
        },
        CgroupStats {
            cgroup_name: "a".to_string(),
            p99_wake_latency_us: 8888.0,
            ..CgroupStats::default()
        },
    ];
    let mut stats = ScenarioStats {
        phases: vec![b0, b1],
        cgroups,
        ..ScenarioStats::default()
    };
    populate_run_distribution_metrics(&mut stats);
    // Pooled p99 over the cross-step union [1,2,3,4] µs (nearest-rank index
    // ceil(4*0.99)-1 = 3) = 4.0 µs; the 9999/8888 bogus per-cgroup reductions
    // are skipped because "a" is in wake_carriers.
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(4.0),
    );
}

/// Registry-impossible misauthoring detector: a Distribution pairing a wake
/// source with the Worst reduction (no `CgroupStats` wake-worst field) hits
/// the cross-source arm of `distribution_cgroup_reduction` via the
/// carrier-less fold (empty pool + a carrier-less cgroup), which
/// `debug_assert!(false, ...)`s in test builds — catching the misauthored
/// registry entry in CI. In release that arm returns `f64::NAN`, which the
/// producer's `is_finite` insert guard (the pass-8 should-fix) drops to
/// ABSENCE rather than writing a NaN that would fail the whole sidecar
/// write. This exercises the `_from` split directly (the testability the
/// doc claims) over a deliberately non-registry (name, kind) pair.
#[test]
#[should_panic(expected = "no CgroupStats wake reduction")]
fn repool_distribution_cross_source_arm_debug_asserts_in_test_build() {
    let mut target = BTreeMap::new();
    let empty_carriers = std::collections::BTreeSet::new();
    // Empty wake pool -> no pooled value -> the carrier-less fold runs over
    // `cgroups`; "a" is absent from (empty) wake_carriers, so it is folded via
    // distribution_cgroup_reduction(cg, WakeLatencyNs, Worst) -> cross-source arm.
    populate_run_distribution_metrics_from(
        &mut target,
        std::iter::once((
            "worst_p99_wake_latency_us",
            MetricKind::Distribution {
                source: SampleSource::WakeLatencyNs,
                reduction: SampleReduction::Worst,
            },
        )),
        &[],
        &empty_carriers,
        &[],
        &empty_carriers,
        &[CgroupStats {
            cgroup_name: "a".to_string(),
            ..CgroupStats::default()
        }],
        0,
    );
}

/// Distribution measured-zero contract: a cgroups-present cohort with NO
/// carrier samples whose per-cgroup reductions are all 0.0 folds to
/// `Some(0.0)` — a measured zero, NOT absence (matching the deleted
/// 0.0-sentinel typed field). Contrast WorstLowest, which yields `None` for
/// an all-`None` cohort. Guards against a future "zero-as-sentinel" refactor
/// silently flipping a quiet run's Distribution from `Some(0.0)` to absent.
#[test]
fn repool_distribution_all_zero_reductions_is_measured_zero_not_absent() {
    let cg = CgroupStats {
        cgroup_name: "x".to_string(),
        p99_wake_latency_us: 0.0,
        worst_run_delay_us: 0.0,
        ..CgroupStats::default()
    };
    // Stripped/empty carrier named "x" -> "x" not in *_carriers -> the
    // carrier-less fold reads cg's 0.0 reductions.
    let mut stats = repool_stats(vec![("x", PhaseCgroupStats::default())], vec![cg]);
    populate_run_distribution_metrics(&mut stats);
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(0.0),
        "all-zero-reduction cohort -> measured Some(0.0), not absent",
    );
    assert_eq!(
        stats.ext_metrics.get("worst_run_delay_us").copied(),
        Some(0.0),
    );
    // WorstLowest contrast: cg has no workers -> iterations_per_worker None ->
    // all-None cohort -> absent (NOT Some(0.0)) — the kind-specific boundary.
    assert!(
        !stats
            .ext_metrics
            .contains_key("worst_iterations_per_worker"),
        "all-None WorstLowest cohort stays absent, distinct from Distribution's 0.0",
    );
}

/// Mixed across-steps dedup (the subtlest documented boundary): a cgroup
/// name that carries samples in ONE phase is in `*_carriers`, so its
/// `stats.cgroups[]` entry is skipped from the carrier-less fold EVEN when
/// another phase's same-name carrier is empty — the bogus per-cgroup
/// reduction never leaks into the pooled value.
#[test]
fn repool_distribution_name_in_carriers_skips_cgroup_even_with_empty_sibling_phase() {
    let with_samples = PhaseCgroupStats {
        wake_latencies_ns: vec![1000, 2000],
        wake_sample_total: 2,
        ..PhaseCgroupStats::default()
    };
    let mut b0 = PhaseBucket {
        step_index: 1,
        ..PhaseBucket::default()
    };
    b0.per_cgroup.insert("a".to_string(), with_samples);
    // phase[1]: same name "a", EMPTY carrier (collected no samples).
    let mut b1 = PhaseBucket {
        step_index: 2,
        ..PhaseBucket::default()
    };
    b1.per_cgroup.insert("a".to_string(), PhaseCgroupStats::default());
    // A stats.cgroups "a" entry with a bogus high p99 that must NOT leak in
    // ("a" is in wake_carriers via phase[0], so this entry is skipped).
    let cgroups = vec![CgroupStats {
        cgroup_name: "a".to_string(),
        p99_wake_latency_us: 9999.0,
        ..CgroupStats::default()
    }];
    let mut stats = ScenarioStats {
        phases: vec![b0, b1],
        cgroups,
        ..ScenarioStats::default()
    };
    populate_run_distribution_metrics(&mut stats);
    // Pooled p99 over phase[0]'s [1,2] µs only (nearest-rank index
    // ceil(2*0.99)-1 = 1) = 2.0 µs; the 9999 bogus reduction is skipped.
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(2.0),
    );
}

/// Graceful degradation: when the bulk frame stripped the phase sample pools
/// (carriers empty), the Distribution re-pool falls back to the worst-wins
/// (max — LowerBetter) over the SURVIVING per-cgroup CgroupStats reductions,
/// reproducing the pre-Item-7 cross-cgroup max — never a silent vanish.
#[test]
fn repool_distribution_falls_back_to_cgroup_reductions_when_stripped() {
    // Distinct non-empty cgroup_names mirror production (collect_handles
    // labels every stats.cgroups entry); the carrier-less fold is
    // name-keyed, so production-shaped fixtures keep the dedup realistic.
    let cg0 = CgroupStats {
        cgroup_name: "a".to_string(),
        p99_wake_latency_us: 30.0,
        worst_run_delay_us: 80.0,
        ..CgroupStats::default()
    };
    let cg1 = CgroupStats {
        cgroup_name: "b".to_string(),
        p99_wake_latency_us: 70.0,
        worst_run_delay_us: 50.0,
        ..CgroupStats::default()
    };
    // Phase carrier with EMPTY sample vecs (the stripped state). Named "a"
    // but empty, so "a" is NOT in *_carriers — both cgroups fall to the
    // carrier-less fallback fold regardless of name.
    let mut stats = repool_stats(
        vec![("a", PhaseCgroupStats::default())],
        vec![cg0, cg1],
    );
    populate_run_distribution_metrics(&mut stats);
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(70.0),
        "stripped → fallback max over cgroup p99 reductions",
    );
    assert_eq!(
        stats.ext_metrics.get("worst_run_delay_us").copied(),
        Some(80.0),
        "stripped → fallback max over cgroup worst_run_delay reductions",
    );
}

/// Run-delay re-pool divides RAW ns by 1000 ONCE: mean / worst over
/// `run_delays_ns` [100_000, 300_000] ns are 200 µs / 300 µs, not 0.2/0.3
/// (double-divided) or 100_000/300_000 (forgot to divide).
#[test]
fn repool_run_delay_divides_ns_by_1000_once() {
    let cg = PhaseCgroupStats {
        run_delays_ns: vec![100_000, 300_000],
        ..PhaseCgroupStats::default()
    };
    let mut stats = repool_stats(vec![("a", cg)], vec![]);
    populate_run_distribution_metrics(&mut stats);
    assert_eq!(
        stats.ext_metrics.get("worst_mean_run_delay_us").copied(),
        Some(200.0),
    );
    assert_eq!(
        stats.ext_metrics.get("worst_run_delay_us").copied(),
        Some(300.0),
    );
}

/// CROSS-PHASE pooling (the epic thesis on the phase dimension): the run-level
/// re-pool unions a cgroup's raw samples across MULTIPLE phases, so the pooled
/// p99 is the percentile over BOTH phases' samples combined — not phase[0]
/// alone, not a fold of per-phase reductions. populate_run_distribution_metrics
/// iterates every phase bucket, so a regression that pooled only one phase
/// surfaces here (the single-phase repool_* tests cannot catch it).
#[test]
fn repool_distribution_pools_wake_across_phases() {
    let phase0 = PhaseCgroupStats {
        wake_latencies_ns: (1..=50u64).map(|v| v * 1000).collect(), // 1..50 µs
        wake_sample_total: 50,
        ..PhaseCgroupStats::default()
    };
    let phase1 = PhaseCgroupStats {
        wake_latencies_ns: (51..=100u64).map(|v| v * 1000).collect(), // 51..100 µs
        wake_sample_total: 50,
        ..PhaseCgroupStats::default()
    };
    let mut b0 = PhaseBucket {
        step_index: 0,
        ..PhaseBucket::default()
    };
    b0.per_cgroup.insert("a".to_string(), phase0);
    let mut b1 = PhaseBucket {
        step_index: 1,
        ..PhaseBucket::default()
    };
    b1.per_cgroup.insert("a".to_string(), phase1);
    let mut stats = ScenarioStats {
        phases: vec![b0, b1],
        ..ScenarioStats::default()
    };
    populate_run_distribution_metrics(&mut stats);

    // Union over BOTH phases = [1..100] µs (100 samples); p99 = sorted[98] = 99 µs.
    let union: Vec<u64> = (1..=100u64).map(|v| v * 1000).collect();
    let expected = percentile(&union, 0.99) as f64 / 1000.0;
    assert_eq!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(expected),
        "pooled p99 must be over the cross-PHASE union (99 µs), not phase[0] alone",
    );
    // Distinct from phase[0]-only (1..50 µs → p99 = 50 µs) — proves both phases pool.
    let p0: Vec<u64> = (1..=50u64).map(|v| v * 1000).collect();
    let phase0_only = percentile(&p0, 0.99) as f64 / 1000.0;
    assert_ne!(
        stats.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(phase0_only),
        "must NOT pool only phase[0]",
    );
}

/// Mean-reduction thesis — the reduction with the LARGEST divergence from the deleted
/// max-of-per-cgroup-means fold: worst_mean_run_delay_us is the MEAN over the
/// POOLED cross-cgroup run-delay set, NOT the max of per-cgroup means. cg_a: 4
/// workers @ 10 µs; cg_b: 1 worker @ 200 µs. Pooled mean = (10*4 + 200)/5 =
/// 48 µs; max-of-per-cgroup-means = max(10, 200) = 200 µs. They differ, so a
/// regression re-introducing the max-of-means fold for the run-delay source
/// would fail here (the single-cgroup parity tests cannot catch it).
#[test]
fn repool_mean_run_delay_pools_across_cgroups_not_max_of_per_cgroup() {
    let cg_a = PhaseCgroupStats {
        run_delays_ns: vec![10_000, 10_000, 10_000, 10_000],
        ..PhaseCgroupStats::default()
    };
    let cg_b = PhaseCgroupStats {
        run_delays_ns: vec![200_000],
        ..PhaseCgroupStats::default()
    };
    let mut stats = repool_stats(vec![("a", cg_a), ("b", cg_b)], vec![]);
    populate_run_distribution_metrics(&mut stats);
    // Pooled mean over the 5-sample union = (40_000 + 200_000)/5 / 1000 = 48 µs.
    assert_eq!(
        stats.ext_metrics.get("worst_mean_run_delay_us").copied(),
        Some(48.0),
        "pooled cross-cgroup mean run-delay over the union, not max-of-per-cgroup",
    );
    // max-of-per-cgroup-means = max(10 µs, 200 µs) = 200 µs — the deleted fold.
    assert_ne!(
        stats.ext_metrics.get("worst_mean_run_delay_us").copied(),
        Some(200.0),
        "must NOT be max of per-cgroup means (the pre-Item-7 fold)",
    );
}

/// Carrier-less cgroups (a backdrop — collected with step_index None, so no
/// per-phase carrier; 6c/#36) are NOT dropped from the run-level Distribution:
/// their surviving per-cgroup CgroupStats reduction folds worst-wins into the
/// pooled value. Here cgroup "a" carries low wake samples (pooled p99 ~50 µs)
/// and is also in stats.cgroups with a bogus p99=9999 that MUST be ignored
/// (it is pooled, not reduction-folded); backdrop "bd" has NO carrier and a
/// p99=500 µs that MUST be folded in. Result = max(pooled 50, bd 500) = 500.
#[test]
fn repool_distribution_folds_carrierless_backdrop_not_dropped() {
    let carrier_a = PhaseCgroupStats {
        wake_latencies_ns: (1..=50u64).map(|v| v * 1000).collect(), // 1..50 µs
        wake_sample_total: 50,
        ..PhaseCgroupStats::default()
    };
    // "a" is carrier-bearing AND in stats.cgroups: its typed p99 (9999) must be
    // IGNORED (pooled, not folded). "bd" has no carrier: its p99 (500) folds in.
    let cg_a = CgroupStats {
        cgroup_name: "a".to_string(),
        p99_wake_latency_us: 9999.0,
        ..CgroupStats::default()
    };
    let cg_bd = CgroupStats {
        cgroup_name: "bd".to_string(),
        p99_wake_latency_us: 500.0,
        ..CgroupStats::default()
    };
    let mut stats = repool_stats(vec![("a", carrier_a)], vec![cg_a, cg_bd]);
    populate_run_distribution_metrics(&mut stats);
    let p99 = stats.ext_metrics.get("worst_p99_wake_latency_us").copied();
    assert_eq!(
        p99,
        Some(500.0),
        "backdrop p99 (500) folds worst-wins with the pooled carrier p99 (~50); \
         got {p99:?}",
    );
    // The carrier-bearing cgroup "a" is POOLED (its samples), NOT reduction-
    // folded: its bogus typed p99=9999 must not appear.
    assert_ne!(p99, Some(9999.0), "carrier-bearing cgroup must be pooled, not reduction-folded");
}

/// Per-SOURCE carrier independence: the run-delay carrier set is consulted
/// separately from the wake set. carrier "a" carries run_delays (NO wake
/// samples) and is also in stats.cgroups with bogus run-delay reductions that
/// must be IGNORED (pooled, since it IS in the run-delay carrier set); backdrop
/// "bd" has no carrier and folds its run-delay reductions worst-wins. Pins that
/// the run-delay carrier set is checked independently of the (empty) wake set.
#[test]
fn repool_run_delay_folds_carrierless_backdrop_independently_of_wake() {
    let carrier_a = PhaseCgroupStats {
        run_delays_ns: vec![10_000, 10_000], // 2 workers @ 10 µs, pooled mean 10 µs
        // no wake_latencies_ns: "a" is carrier-bearing for run-delay only.
        ..PhaseCgroupStats::default()
    };
    let cg_a = CgroupStats {
        cgroup_name: "a".to_string(),
        mean_run_delay_us: 9999.0, // carrier-bearing for run-delay → IGNORED (pooled)
        worst_run_delay_us: 9999.0,
        ..CgroupStats::default()
    };
    let cg_bd = CgroupStats {
        cgroup_name: "bd".to_string(),
        mean_run_delay_us: 500.0, // no carrier → folds worst-wins
        worst_run_delay_us: 700.0,
        ..CgroupStats::default()
    };
    let mut stats = repool_stats(vec![("a", carrier_a)], vec![cg_a, cg_bd]);
    populate_run_distribution_metrics(&mut stats);
    // Mean: max(pooled carrier mean 10, bd 500) = 500; cg_a's 9999 ignored (pooled).
    assert_eq!(
        stats.ext_metrics.get("worst_mean_run_delay_us").copied(),
        Some(500.0),
    );
    // Worst (max): max(pooled carrier max 10, bd 700) = 700; cg_a's 9999 ignored.
    assert_eq!(
        stats.ext_metrics.get("worst_run_delay_us").copied(),
        Some(700.0),
    );
}
