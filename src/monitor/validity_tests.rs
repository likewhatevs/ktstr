//! Unit tests for the validity helpers:
//! `MonitorThresholds::data_looks_valid`, `sample_looks_valid`,
//! `MonitorSample::imbalance_ratio`, and
//! `MonitorSample::sum_event_field`.
//! Co-located via the sibling `*_tests.rs` pattern.

#![cfg(test)]

use super::test_util::balanced_sample;
use super::*;

// -- data_looks_valid tests --

#[test]
fn data_looks_valid_empty() {
    assert!(MonitorThresholds::data_looks_valid(&[]));
}

#[test]
fn data_looks_valid_normal() {
    let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 2000)];
    assert!(MonitorThresholds::data_looks_valid(&samples));
}

#[test]
fn data_looks_valid_all_same_clocks() {
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    rq_clock: 5000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    rq_clock: 5000,
                    ..Default::default()
                },
            ],
        },
    ];
    assert!(!MonitorThresholds::data_looks_valid(&samples));
}

#[test]
fn data_looks_valid_dsq_over_ceiling() {
    let samples = vec![MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            local_dsq_depth: 50000,
            rq_clock: 1000,
            ..Default::default()
        }],
    }];
    assert!(!MonitorThresholds::data_looks_valid(&samples));
}

// -- MonitorSample::imbalance_ratio tests --

#[test]
fn imbalance_ratio_empty_cpus() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![],
    };
    assert!((s.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn imbalance_ratio_single_cpu() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![CpuSnapshot {
            nr_running: 5,
            ..Default::default()
        }],
    };
    assert!((s.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn imbalance_ratio_balanced() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![
            CpuSnapshot {
                nr_running: 3,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 3,
                ..Default::default()
            },
        ],
    };
    assert!((s.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn imbalance_ratio_imbalanced() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![
            CpuSnapshot {
                nr_running: 2,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 8,
                ..Default::default()
            },
        ],
    };
    assert!((s.imbalance_ratio() - 4.0).abs() < f64::EPSILON);
}

#[test]
fn imbalance_ratio_zero_min() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![
            CpuSnapshot {
                nr_running: 0,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 5,
                ..Default::default()
            },
        ],
    };
    // min=0, max(0,1)=1, ratio=5/1=5.0
    assert!((s.imbalance_ratio() - 5.0).abs() < f64::EPSILON);
}

// -- MonitorSample::sum_event_field tests --

#[test]
fn sum_event_field_none_when_no_counters() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![CpuSnapshot::default(), CpuSnapshot::default()],
    };
    assert!(s.sum_event_field(|e| e.select_cpu_fallback).is_none());
}

#[test]
fn sum_event_field_sums_across_cpus() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![
            CpuSnapshot {
                event_counters: Some(ScxEventCounters {
                    select_cpu_fallback: 10,
                    ..Default::default()
                }),
                ..Default::default()
            },
            CpuSnapshot {
                event_counters: Some(ScxEventCounters {
                    select_cpu_fallback: 20,
                    ..Default::default()
                }),
                ..Default::default()
            },
        ],
    };
    assert_eq!(s.sum_event_field(|e| e.select_cpu_fallback), Some(30));
}

#[test]
fn sum_event_field_mixed_some_none() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![
            CpuSnapshot {
                event_counters: Some(ScxEventCounters {
                    dispatch_keep_last: 7,
                    ..Default::default()
                }),
                ..Default::default()
            },
            CpuSnapshot::default(),
        ],
    };
    assert_eq!(s.sum_event_field(|e| e.dispatch_keep_last), Some(7));
}

// -- sample_looks_valid tests --

#[test]
fn sample_looks_valid_normal() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            local_dsq_depth: 5,
            ..Default::default()
        }],
    };
    assert!(sample_looks_valid(&s));
}

#[test]
fn sample_looks_valid_at_ceiling() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            local_dsq_depth: DSQ_PLAUSIBILITY_CEILING,
            ..Default::default()
        }],
    };
    assert!(sample_looks_valid(&s));
}

#[test]
fn sample_looks_valid_over_ceiling() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            local_dsq_depth: DSQ_PLAUSIBILITY_CEILING + 1,
            ..Default::default()
        }],
    };
    assert!(!sample_looks_valid(&s));
}

#[test]
fn sample_looks_valid_empty_cpus() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![],
    };
    assert!(sample_looks_valid(&s));
}

#[test]
fn sample_looks_valid_zero_initialized() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 0,
        cpus: vec![CpuSnapshot::default(), CpuSnapshot::default()],
    };
    // All fields zero, local_dsq_depth=0 <= DSQ_PLAUSIBILITY_CEILING.
    assert!(sample_looks_valid(&s));
}

#[test]
fn sample_looks_valid_multiple_cpus_one_over() {
    let s = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                local_dsq_depth: 5,
                ..Default::default()
            },
            CpuSnapshot {
                local_dsq_depth: DSQ_PLAUSIBILITY_CEILING + 1,
                ..Default::default()
            },
        ],
    };
    // One CPU over ceiling invalidates the entire sample.
    assert!(!sample_looks_valid(&s));
}
