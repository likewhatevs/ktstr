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
