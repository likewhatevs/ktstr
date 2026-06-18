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
    assert_eq!(by_stim.keys().copied().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    for k in [0u16, 1, 2, 3] {
        assert_eq!(by_stim[&k].len(), 1, "phase {k} should hold exactly one sample");
    }
}

/// Fallback: a capture with no `boundary_offset_ms` (on-demand /
/// fixture) keeps its stamped step_index even when a stimulus timeline
/// is present — the offset remap only overrides captures that carry a
/// scheduled offset, so legacy / non-periodic entries are untouched.
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
    // A step-5 stimulus is present but must NOT pull the None-offset
    // captures to step 5 — they keep their stamped 1 / 2.
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
        vec![1, 2],
        "None-offset captures keep their stamped step_index (1, 2), not \
         remapped to the step-5 stimulus; got {idxs:?}",
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
/// attribution loop's guard skips it before `rate_to`/`or_insert` ever
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
    // pair is rate_to Some(0.0) — measured zero throughput. A persistent
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
    let step1 = phases.iter().find(|p| p.step_index == 1).expect("step 1 bucket");
    let loop_step = phases.iter().find(|p| p.step_index == 2).expect("loop step bucket");
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
