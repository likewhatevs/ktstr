use super::*;

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
// per-metric-arm tests in `src/stats/metric.rs`; these tests verify the
// bucketing skeleton independent of metric data.

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
/// dispatch (read_sample in `src/stats/metric.rs`) and the per-phase
/// reduction (aggregate_samples_for_phase in `src/stats/metric.rs`). A future
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
    assert_eq!(
        stats
            .phase(crate::assert::Phase::BASELINE)
            .map(|p| p.label.as_str()),
        Some("BASELINE")
    );
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
    assert_eq!(stats.phase(crate::assert::Phase::step(2)), None);
    // Phase::step(0) is the scenario's first Step (the typed accessor hides the
    // 1-indexed encoding: Step 0 lives at the underlying step_index 1).
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

// ---------- avg_cpu_util_comp_scale (scx_layered util-compensation) ----------

/// Build a `PerCpuTimeStats` with the eight `cpustat[]` ns slots set explicitly
/// (order: user, nice, system, idle, iowait, irq, softirq, steal); other fields
/// default. In-crate construction of the `#[non_exhaustive]` type.
#[allow(clippy::too_many_arguments)]
fn cpustat_row(
    cpu: u32,
    user: u64,
    nice: u64,
    system: u64,
    idle: u64,
    iowait: u64,
    irq: u64,
    softirq: u64,
    steal: u64,
) -> crate::monitor::dump::PerCpuTimeStats {
    crate::monitor::dump::PerCpuTimeStats {
        cpu,
        cpustat_user_ns: user,
        cpustat_nice_ns: nice,
        cpustat_system_ns: system,
        cpustat_idle_ns: idle,
        cpustat_iowait_ns: iowait,
        cpustat_irq_ns: irq,
        cpustat_softirq_ns: softirq,
        cpustat_steal_ns: steal,
        ..Default::default()
    }
}

/// The pure per-CPU scale mirrors scx_layered's `compute_scale(delta_total,
/// overhead)`: split a window of `delta_total` ns into `overhead` ns of IRQ and
/// the rest idle, so `scale = delta_total / (delta_total - overhead)`. Pins the
/// six scx_layered util_compensation reference cases: (1000,0)->1.0,
/// (1000,500)->2.0, (1000,900)->10.0, (1000,950)->20.0, (1000,980)->20.0 (ratio
/// 50 clamped to the 20.0 cap), (1000,1000)->1.0 (available 0 -> the floor).
#[test]
fn cpu_util_comp_scale_matches_scx_layered_reference_cases() {
    let zero = crate::monitor::dump::PerCpuTimeStats {
        cpu: 0,
        ..Default::default()
    };
    // overhead carried in irq; the remaining available carried in idle.
    let case = |delta_total: u64, overhead: u64| {
        let last = cpustat_row(0, 0, 0, 0, delta_total - overhead, 0, overhead, 0, 0);
        crate::assert::cpu_util_comp_scale(&zero, &last)
    };
    assert!((case(1000, 0) - 1.0).abs() < 1e-9, "no overhead -> 1.0");
    assert!((case(1000, 500) - 2.0).abs() < 1e-9, "50% overhead -> 2.0");
    assert!(
        (case(1000, 900) - 10.0).abs() < 1e-9,
        "90% overhead -> 10.0"
    );
    assert!(
        (case(1000, 950) - 20.0).abs() < 1e-9,
        "95% overhead -> 20.0 (cap)"
    );
    assert!(
        (case(1000, 980) - 20.0).abs() < 1e-9,
        "98% overhead -> ratio 50 clamped to 20.0"
    );
    assert!(
        (case(1000, 1000) - 1.0).abs() < 1e-9,
        "all overhead -> available 0 -> 1.0 floor"
    );
}

/// `delta_total` sums ALL 8 cpustat slots; `overhead` is exactly
/// irq+softirq+steal. Every one of the 8 slots is set to a DISTINCT non-zero
/// value, so dropping any slot from `delta_total`, dropping any of the three
/// from `overhead`, or swapping a slot between the two groups moves the result
/// off 2.0. Window: user 100 + nice 150 + system 200 + idle 250 + iowait 300
/// (available 1000); irq 120 + softirq 380 + steal 500 (overhead 1000) ->
/// delta_total 2000, available 1000, scale 2.0.
#[test]
fn cpu_util_comp_scale_uses_all_eight_slots_and_three_overhead_slots() {
    let zero = crate::monitor::dump::PerCpuTimeStats {
        cpu: 0,
        ..Default::default()
    };
    let last = cpustat_row(0, 100, 150, 200, 250, 300, 120, 380, 500);
    // delta_total = 100+150+200+250+300 (available 1000) + 120+380+500
    // (overhead 1000) = 2000; available = 1000; scale = 2000/1000 = 2.0.
    assert!((crate::assert::cpu_util_comp_scale(&zero, &last) - 2.0).abs() < 1e-9);
}

/// A cpustat counter that REGRESSES across the span (a CPU-hotplug reset, or a
/// corrupt/hostile guest read where last < first) saturates each delta to 0
/// rather than wrapping to ~`u64::MAX`: delta_total 0 -> available 0 -> the 1.0
/// floor, no panic, no wrap-driven garbage scale.
#[test]
fn cpu_util_comp_scale_saturates_regressing_counters_to_floor() {
    let first = cpustat_row(0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000);
    let last = crate::monitor::dump::PerCpuTimeStats {
        cpu: 0,
        ..Default::default()
    };
    assert_eq!(crate::assert::cpu_util_comp_scale(&first, &last), 1.0);
}

/// A hostile per-slot `u64::MAX` must not overflow-panic the `delta_total` /
/// `overhead` saturating sums — the result is a finite clamped scale, not a
/// panic. (Here every slot saturates: delta_total and overhead both pin to
/// `u64::MAX`, so available 0 -> the 1.0 floor.)
#[test]
fn cpu_util_comp_scale_saturates_hostile_max_slot_no_panic() {
    let zero = crate::monitor::dump::PerCpuTimeStats {
        cpu: 0,
        ..Default::default()
    };
    let last = cpustat_row(
        0,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
    );
    let s = crate::assert::cpu_util_comp_scale(&zero, &last);
    assert!(
        s.is_finite() && (1.0..=20.0).contains(&s),
        "hostile max slots -> finite clamped scale, got {s}"
    );
}

/// End-to-end through `build_phase_buckets`: two `per_cpu_time` freezes in one
/// phase, two CPUs with distinct per-CPU scales, asserts the fold injects
/// `avg_cpu_util_comp_scale` as the MEAN across CPUs (not the max or sum).
/// First freeze all-zero; last freeze CPU0 idle 500 / irq 500 -> scale 2.0,
/// CPU1 idle 250 / irq 750 -> available 250, delta_total 1000 -> scale 4.0;
/// mean = 3.0. Pins the wiring between `fold_util_comp_scale` (phase_build) and
/// the first->last per_cpu_time freeze span.
#[test]
fn build_phase_buckets_injects_avg_cpu_util_comp_scale_mean_across_cpus() {
    use crate::monitor::dump::{FailureDumpReport, PerCpuTimeStats, SCHEMA_SINGLE};

    fn entry(tag: &str, elapsed_ms: u64, rows: Vec<PerCpuTimeStats>) -> DrainedSnapshotEntry {
        DrainedSnapshotEntry {
            tag: tag.to_string(),
            report: FailureDumpReport {
                schema: SCHEMA_SINGLE.to_string(),
                per_cpu_time: rows,
                ..Default::default()
            },
            stats: Err(MissingStatsReason::NoSchedulerBinary),
            elapsed_ms: Some(elapsed_ms),
            boundary_offset_ms: None,
            step_index: Some(1),
        }
    }
    let row = |cpu, idle, irq| PerCpuTimeStats {
        cpu,
        cpustat_idle_ns: idle,
        cpustat_irq_ns: irq,
        ..Default::default()
    };
    let drained = vec![
        entry("periodic_000", 100, vec![row(0, 0, 0), row(1, 0, 0)]),
        entry(
            "periodic_001",
            200,
            vec![row(0, 500, 500), row(1, 250, 750)],
        ),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    let got = step0.metrics.get("avg_cpu_util_comp_scale").copied();
    assert_eq!(
        got,
        Some(3.0),
        "mean of per-CPU scales (2.0, 4.0) = 3.0, got {got:?}"
    );
}

// ---------- avg_task_lat_cri / max_task_lat_cri (scx_lavd per-task lat_cri) ----

/// Build an sdt_alloc snapshot entry whose BTF-rendered payload carries a
/// `normalized_lat_cri` u16 member (the scx_lavd task_ctx field), so the
/// `fold_lat_cri` arena walk extracts it via `RenderedValue::get`.
fn lat_cri_entry(value: u64) -> crate::monitor::sdt_alloc::SdtAllocEntry {
    use crate::monitor::btf_render::{RenderedMember, RenderedValue};
    crate::monitor::sdt_alloc::SdtAllocEntry {
        idx: 0,
        genn: 0,
        user_addr: 0,
        payload: RenderedValue::Struct {
            type_name: Some("task_ctx".to_string()),
            members: vec![RenderedMember {
                name: "normalized_lat_cri".to_string(),
                value: RenderedValue::Uint { bits: 16, value },
            }],
        },
    }
}

/// End-to-end through `build_phase_buckets`: two per-phase freezes each carrying
/// an sdt_alloc snapshot with per-task `normalized_lat_cri` values, asserts
/// `fold_lat_cri` injects `avg_task_lat_cri` (mean over (freeze, task)) +
/// `max_task_lat_cri` (max). Freeze 1: [100, 200]; freeze 2: [300, 400] ->
/// mean (100+200+300+400)/4 = 250, max 400. Pins the gauge fold across ALL
/// samples (not first/last) and the cross-(freeze, task) mean+max.
#[test]
fn build_phase_buckets_injects_task_lat_cri_mean_and_max() {
    use crate::monitor::dump::{FailureDumpReport, SCHEMA_SINGLE};
    use crate::monitor::sdt_alloc::SdtAllocatorSnapshot;

    fn entry(tag: &str, elapsed_ms: u64, lat_cris: &[u64]) -> DrainedSnapshotEntry {
        let alloc = SdtAllocatorSnapshot {
            allocator_name: "scx_task_allocator".to_string(),
            entries: lat_cris.iter().map(|&v| lat_cri_entry(v)).collect(),
            ..Default::default()
        };
        DrainedSnapshotEntry {
            tag: tag.to_string(),
            report: FailureDumpReport {
                schema: SCHEMA_SINGLE.to_string(),
                sdt_allocations: vec![alloc],
                ..Default::default()
            },
            stats: Err(MissingStatsReason::NoSchedulerBinary),
            elapsed_ms: Some(elapsed_ms),
            boundary_offset_ms: None,
            step_index: Some(1),
        }
    }
    let drained = vec![
        entry("periodic_000", 100, &[100, 200]),
        entry("periodic_001", 200, &[300, 400]),
    ];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    assert_eq!(
        step0.metrics.get("avg_task_lat_cri").copied(),
        Some(250.0),
        "mean of (100,200,300,400) over (freeze, task) = 250"
    );
    assert_eq!(
        step0.metrics.get("max_task_lat_cri").copied(),
        Some(400.0),
        "max over (freeze, task) = 400"
    );
}

/// Loud-absent: an arena payload WITHOUT a `normalized_lat_cri` member (a
/// non-lavd scheduler's task_ctx) yields NO lat_cri key — `RenderedValue::get`
/// returns None, the fold inserts nothing. Pins the member-name-only gate (no
/// scheduler-name hardcoding) and the absent-vs-zero distinction.
#[test]
fn build_phase_buckets_lat_cri_absent_for_non_lavd_payload() {
    use crate::monitor::btf_render::{RenderedMember, RenderedValue};
    use crate::monitor::dump::{FailureDumpReport, SCHEMA_SINGLE};
    use crate::monitor::sdt_alloc::{SdtAllocEntry, SdtAllocatorSnapshot};

    let alloc = SdtAllocatorSnapshot {
        allocator_name: "scx_task_allocator".to_string(),
        entries: vec![SdtAllocEntry {
            idx: 0,
            genn: 0,
            user_addr: 0,
            payload: RenderedValue::Struct {
                type_name: Some("other_taskc".to_string()),
                members: vec![RenderedMember {
                    name: "some_other_field".to_string(),
                    value: RenderedValue::Uint { bits: 32, value: 7 },
                }],
            },
        }],
        ..Default::default()
    };
    let drained = vec![DrainedSnapshotEntry {
        tag: "periodic_000".to_string(),
        report: FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            sdt_allocations: vec![alloc],
            ..Default::default()
        },
        stats: Err(MissingStatsReason::NoSchedulerBinary),
        elapsed_ms: Some(100),
        boundary_offset_ms: None,
        step_index: Some(1),
    }];
    let samples = SampleSeries::from_drained_typed(drained, None);
    let phases = crate::assert::build_phase_buckets(&samples);
    let step0 = phases
        .iter()
        .find(|p| p.step_index == 1)
        .expect("Step[0] bucket present");
    assert!(
        !step0.metrics.contains_key("avg_task_lat_cri"),
        "non-lavd payload (no normalized_lat_cri member) must not emit the key"
    );
    assert!(!step0.metrics.contains_key("max_task_lat_cri"));
}
