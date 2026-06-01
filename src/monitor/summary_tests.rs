//! Unit tests for `MonitorSummary::from_samples`: imbalance ratio,
//! local-DSQ depth, stall detection, average fields, and the
//! field-shape regression guard.
//! Co-located via the sibling `*_tests.rs` pattern.

#![cfg(test)]

use super::*;

#[test]
fn empty_samples_default_summary() {
    let summary = MonitorSummary::from_samples(&[]);
    assert_eq!(summary.total_samples, 0);
    assert_eq!(summary.max_imbalance_ratio, 0.0);
    assert_eq!(summary.max_local_dsq_depth, 0);
    assert!(!summary.stuck_detected);
    assert_eq!(summary.avg_imbalance_ratio, 0.0);
    assert_eq!(summary.avg_nr_running, 0.0);
    assert_eq!(summary.avg_local_dsq_depth, 0.0);
}

#[test]
fn single_sample_imbalanced_cpus() {
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                local_dsq_depth: 3,
                rq_clock: 1000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 4,
                local_dsq_depth: 1,
                rq_clock: 2000,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert_eq!(summary.total_samples, 1);
    assert!((summary.max_imbalance_ratio - 4.0).abs() < f64::EPSILON);
    assert_eq!(summary.max_local_dsq_depth, 3);
    assert!(!summary.stuck_detected);
    // avg fields: single sample with cpus [nr_running=1, nr_running=4]
    assert!((summary.avg_imbalance_ratio - 4.0).abs() < f64::EPSILON);
    assert!((summary.avg_nr_running - 2.5).abs() < f64::EPSILON);
    assert!((summary.avg_local_dsq_depth - 2.0).abs() < f64::EPSILON);
}

#[test]
fn stuck_detected_when_clock_stuck() {
    let s1 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 5000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 6000,
                ..Default::default()
            },
        ],
    };
    let s2 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 200,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 5000, // stuck
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 7000,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[s1, s2]);
    assert!(summary.stuck_detected);
}

#[test]
fn balanced_cpus_ratio_one() {
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 50,
        cpus: vec![
            CpuSnapshot {
                nr_running: 3,
                rq_clock: 100,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 3,
                rq_clock: 200,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    assert!(!summary.stuck_detected);
    assert!((summary.avg_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    assert!((summary.avg_nr_running - 3.0).abs() < f64::EPSILON);
    assert!((summary.avg_local_dsq_depth - 0.0).abs() < f64::EPSILON);
}

#[test]
fn single_cpu_no_division_by_zero() {
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 10,
        cpus: vec![CpuSnapshot {
            nr_running: 5,
            local_dsq_depth: 2,
            rq_clock: 1000,
            ..Default::default()
        }],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert_eq!(summary.total_samples, 1);
    // Single CPU: min == max, ratio = 1.0
    assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    assert_eq!(summary.max_local_dsq_depth, 2);
    assert!(!summary.stuck_detected);
}

#[test]
fn all_zero_snapshots() {
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![CpuSnapshot::default(), CpuSnapshot::default()],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert_eq!(summary.total_samples, 1);
    // nr_running=0 for all CPUs: max/max(min,1) = 0/1 = 0.0, but
    // initial max_imbalance_ratio is 1.0 and 0.0 < 1.0, so stays 1.0.
    assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    assert_eq!(summary.max_local_dsq_depth, 0);
    // rq_clock=0 is excluded from stall detection
    assert!(!summary.stuck_detected);
    // avg: valid sample with 2 all-zero CPUs
    assert_eq!(summary.avg_imbalance_ratio, 0.0);
    assert_eq!(summary.avg_nr_running, 0.0);
    assert_eq!(summary.avg_local_dsq_depth, 0.0);
}

#[test]
fn empty_cpus_in_sample() {
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 10,
        cpus: vec![],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert_eq!(summary.total_samples, 1);
    // Empty cpus slice is skipped via `continue`
    assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    // avg: sample skipped (empty cpus), no valid readings
    assert_eq!(summary.avg_imbalance_ratio, 0.0);
    assert_eq!(summary.avg_nr_running, 0.0);
    assert_eq!(summary.avg_local_dsq_depth, 0.0);
}

#[test]
fn min_nr_zero_division_guard() {
    // All CPUs have nr_running=0. The code uses min_nr.max(1) as
    // divisor, so ratio = 0/1 = 0.0, which is < initial 1.0.
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 10,
        cpus: vec![
            CpuSnapshot {
                nr_running: 0,
                rq_clock: 100,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 0,
                rq_clock: 200,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    // Should not panic from division by zero.
    // max_imbalance_ratio stays at initial 1.0 since 0/1=0 < 1.0.
    assert!((summary.max_imbalance_ratio - 1.0).abs() < f64::EPSILON);
}

#[test]
fn min_nr_zero_max_nr_nonzero() {
    // min_nr=0, max_nr=5: ratio = 5/max(0,1) = 5.0
    let sample = MonitorSample {
        prog_stats: None,
        elapsed_ms: 10,
        cpus: vec![
            CpuSnapshot {
                nr_running: 0,
                rq_clock: 100,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 5,
                rq_clock: 200,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert!((summary.max_imbalance_ratio - 5.0).abs() < f64::EPSILON);
}

#[test]
fn advancing_clocks_no_stuck() {
    let s1 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 1000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 2000,
                ..Default::default()
            },
        ],
    };
    let s2 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 200,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 1500,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 2500,
                ..Default::default()
            },
        ],
    };
    let s3 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 300,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 2000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 3000,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[s1, s2, s3]);
    assert!(!summary.stuck_detected);
    assert_eq!(summary.total_samples, 3);
}

#[test]
fn different_length_cpu_vecs() {
    // First sample has 2 CPUs, second has 3. Stall detection uses
    // min(prev.len, curr.len) = 2, so only CPUs 0-1 are compared.
    let s1 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 1000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 2000,
                ..Default::default()
            },
        ],
    };
    let s2 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 200,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 1500,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 2500,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 3000,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[s1, s2]);
    assert!(!summary.stuck_detected);
    assert_eq!(summary.total_samples, 2);
    // max_local_dsq_depth comes from all CPUs in all samples.
    assert_eq!(summary.max_local_dsq_depth, 0);
}

// -- MonitorSummary field value assertions --

#[test]
fn from_samples_fields_sane_values() {
    let samples: Vec<_> = (0..5u64)
        .map(|i| MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: (i as u32 + 1),
                    scx_nr_running: i as u32,
                    local_dsq_depth: (i as u32) % 3,
                    rq_clock: 1000 + i * 500,
                    scx_flags: 0,
                    event_counters: Some(ScxEventCounters {
                        select_cpu_fallback: i as i64 * 2,
                        dispatch_keep_last: i as i64,
                        ..Default::default()
                    }),
                    schedstat: None,
                    vcpu_cpu_time_ns: None,
                    vcpu_perf: None,
                    sched_domains: None,
                },
                CpuSnapshot {
                    nr_running: (i as u32 + 2),
                    scx_nr_running: i as u32 + 1,
                    local_dsq_depth: 0,
                    rq_clock: 1100 + i * 600,
                    scx_flags: 0,
                    event_counters: Some(ScxEventCounters {
                        select_cpu_fallback: i as i64 * 3,
                        dispatch_keep_last: i as i64 * 2,
                        ..Default::default()
                    }),
                    schedstat: None,
                    vcpu_cpu_time_ns: None,
                    vcpu_perf: None,
                    sched_domains: None,
                },
            ],
        })
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    // total_samples matches input count
    assert_eq!(summary.total_samples, 5);
    // max_imbalance_ratio: all samples have nr_running differing by 1,
    // worst case is sample 0: nr_running=[1,2] -> ratio=2.0
    assert!(
        summary.max_imbalance_ratio >= 1.0,
        "ratio must be >= 1.0: {}",
        summary.max_imbalance_ratio
    );
    assert!(
        summary.max_imbalance_ratio <= 10.0,
        "ratio must be reasonable: {}",
        summary.max_imbalance_ratio
    );
    // max_local_dsq_depth: worst is (4 % 3) = 1 on cpu0 at i=4, or (3 % 3)=0 at i=3, (2%3)=2 at i=2
    assert!(
        summary.max_local_dsq_depth <= DSQ_PLAUSIBILITY_CEILING,
        "dsq depth must be below plausibility ceiling: {}",
        summary.max_local_dsq_depth
    );
    assert!(
        summary.max_local_dsq_depth <= 10,
        "dsq depth must be small in this controlled test: {}",
        summary.max_local_dsq_depth
    );
    // stuck_detected: rq_clock advances each sample, so no stuck
    assert!(
        !summary.stuck_detected,
        "no stuck expected with advancing rq_clock"
    );
    // event_deltas: should be computed
    let deltas = summary
        .event_deltas
        .as_ref()
        .expect("event deltas must be present");
    assert!(
        deltas.total_fallback >= 0,
        "fallback count must be non-negative"
    );
    assert!(
        deltas.total_dispatch_keep_last >= 0,
        "keep_last count must be non-negative"
    );
    assert!(
        deltas.fallback_rate >= 0.0,
        "fallback rate must be non-negative"
    );
    assert!(
        deltas.keep_last_rate >= 0.0,
        "keep_last rate must be non-negative"
    );
    // avg fields: must be positive with non-zero nr_running input
    assert!(
        summary.avg_imbalance_ratio >= 1.0,
        "avg imbalance must be >= 1.0: {}",
        summary.avg_imbalance_ratio,
    );
    assert!(
        summary.avg_nr_running > 0.0,
        "avg nr_running must be positive: {}",
        summary.avg_nr_running,
    );
    assert!(
        summary.avg_local_dsq_depth >= 0.0,
        "avg dsq_depth must be non-negative: {}",
        summary.avg_local_dsq_depth,
    );
}

#[test]
fn from_samples_empty_all_defaults() {
    // Check that every field of MonitorSummary defaults correctly for empty input,
    // including event_deltas which empty_samples_default_summary does not check.
    let summary = MonitorSummary::from_samples(&[]);
    assert_eq!(summary.total_samples, 0);
    assert_eq!(summary.max_imbalance_ratio, 0.0);
    assert_eq!(summary.max_local_dsq_depth, 0);
    assert!(!summary.stuck_detected);
    assert_eq!(summary.avg_imbalance_ratio, 0.0);
    assert_eq!(summary.avg_nr_running, 0.0);
    assert_eq!(summary.avg_local_dsq_depth, 0.0);
    assert!(
        summary.event_deltas.is_none(),
        "empty input must not produce event deltas"
    );
}
