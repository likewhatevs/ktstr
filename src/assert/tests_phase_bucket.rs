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

    // Buckets returned in step_index order because SampleSeries::by_phase
    // returns a BTreeMap keyed by step_index.
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
/// fall under key `0` per SampleSeries::by_phase's
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
    assert!(!saturated.is_baseline(), "saturating MUST NOT collide with BASELINE");
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
    assert_eq!(json, "5", "wire format must be the inner 1-indexed u16, not a tagged struct");
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
    assert!(!stats.has_steps(), "BASELINE-only must NOT count as 'has steps'");
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
    let d = crate::assert::AssertDetail::new(
        crate::assert::DetailKind::Other,
        "no guard",
    );
    assert!(d.phase.is_none(), "AssertDetail constructed outside any PhaseGuard must stamp phase=None");
}

#[test]
fn phase_guard_install_step_sets_active_label() {
    let _g = crate::assert::PhaseGuard::install_step(0);
    assert_eq!(
        crate::assert::current_phase_label().as_deref(),
        Some("Step[0]"),
    );
    let d = crate::assert::AssertDetail::new(
        crate::assert::DetailKind::Other,
        "under Step[0]",
    );
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
    let d = crate::assert::AssertDetail::new(
        crate::assert::DetailKind::Other,
        "override",
    )
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
    assert!((max - 6.0).abs() < f64::EPSILON, "expected max=6.0, got {max}");
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
        },
        StimulusEvent {
            elapsed_ms: 1100,
            label: "Step[1]".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: Some(1000),
        },
        StimulusEvent {
            elapsed_ms: 2100,
            label: "end".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: Some(3000),
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

/// Tester B17 BLOCKING: populate_run_ext_metrics on a populated
/// series produces the expected entries. Without this, the empty
/// + no-overwrite tests pass vacuously and the load-bearing
/// happy path is uncovered.
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

/// populate_run_ext_metrics_from_phases populates per-phase
/// metrics that have no read_sample dispatch (avg_imbalance_ratio,
/// iteration_rate). Weighted-mean fold across phases.
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
            step_index: 1,
            label: "Step[0]".to_string(),
            start_ms: 0,
            end_ms: 100,
            sample_count: 5,
            metrics: m0,
        },
        PhaseBucket {
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
