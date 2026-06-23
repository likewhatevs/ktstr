//! Unit tests for stall detection behavior, plus the negative
//! threshold cases (imbalance / DSQ / stuck rq_clock) and the
//! vCPU CPU-time gating that suppresses false-positive stalls
//! when the vCPU was preempted by the host.
//! Co-located via the sibling `*_tests.rs` pattern.

#![cfg(test)]

use super::*;

// ---------------------------------------------------------------
// Negative tests: check monitor diagnostics catch controlled failures
// ---------------------------------------------------------------

#[test]
fn neg_tight_imbalance_threshold_catches_mild_imbalance() {
    let t = MonitorThresholds {
        max_imbalance_ratio: 1.0,
        sustained_samples: 2,
        fail_on_stall: false,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..3u64)
        .map(|i| MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 3,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        })
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    assert!(
        summary.max_imbalance_ratio >= 1.5,
        "summary must capture ratio"
    );
    assert_eq!(summary.stuck_count, 0, "no stall in this scenario");
    assert_eq!(summary.total_samples, 3);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "imbalance=1.5 must fail threshold=1.0");
    // Format: "imbalance ratio 1.5 exceeded threshold 1.0 for 2 consecutive samples (ending at sample 2)"
    let detail = v
        .failure_details()
        .find(|d| d.contains("imbalance"))
        .unwrap();
    assert!(detail.contains("ratio"), "must include 'ratio': {detail}");
    assert!(
        detail.contains("exceeded threshold"),
        "must include threshold: {detail}"
    );
    assert!(
        detail.contains("1.0"),
        "must show threshold value: {detail}"
    );
    assert!(
        detail.contains("consecutive samples"),
        "must show sustained count: {detail}"
    );
    assert!(
        detail.contains("ending at sample"),
        "must show sample index: {detail}"
    );
    assert!(
        v.summary.contains("FAILED"),
        "summary must say FAILED: {}",
        v.summary
    );
}

#[test]
fn neg_tight_dsq_threshold_catches_small_depth() {
    let t = MonitorThresholds {
        max_local_dsq_depth: 1,
        sustained_samples: 2,
        fail_on_stall: false,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..3u64)
        .map(|i| MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    local_dsq_depth: 3,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    local_dsq_depth: 0,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        })
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    assert_eq!(
        summary.max_local_dsq_depth, 3,
        "summary must capture max depth"
    );
    assert!(
        summary.max_local_dsq_depth <= DSQ_PLAUSIBILITY_CEILING,
        "depth must be plausible"
    );
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "dsq_depth=3 must fail threshold=1");
    // Format: "local DSQ depth 3 on cpu0 exceeded threshold 1 for 2 consecutive samples (ending at sample 2)"
    let detail = v
        .failure_details()
        .find(|d| d.contains("DSQ depth"))
        .unwrap();
    assert!(detail.contains("3"), "must show depth value: {detail}");
    assert!(detail.contains("cpu0"), "must show CPU number: {detail}");
    assert!(
        detail.contains("threshold 1"),
        "must show threshold: {detail}"
    );
    assert!(
        detail.contains("consecutive samples"),
        "must show count: {detail}"
    );
}

#[test]
fn neg_stuck_detection_catches_frozen_rq_clock() {
    // Stuck checks use sustained_samples window. sustained_samples=1 means
    // a single stuck pair triggers failure.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
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
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    assert_eq!(
        summary.stuck_count, 1,
        "one stuck (cpu, pair) event: cpu0 frozen across the single window"
    );
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "frozen rq_clock must be detected");
    let detail = v
        .failure_details()
        .find(|d| d.contains("rq_clock stall"))
        .unwrap();
    assert!(detail.contains("cpu0"), "must name frozen CPU: {detail}");
    assert!(
        detail.contains("consecutive samples"),
        "must show sustained count: {detail}"
    );
    assert!(
        detail.contains("clock=5000"),
        "must include frozen clock value: {detail}"
    );
}

#[test]
fn stuck_count_accumulates_across_consecutive_windows() {
    // cpu0 frozen across TWO consecutive windows (3 samples) counts TWICE.
    // Pins removal of the per-RUN early break: the old 0/1 path stopped at
    // the first stuck window. cpu1 advances so each sample stays plausible.
    let mk = |ms: u64, rq1: u64| MonitorSample {
        prog_stats: None,
        elapsed_ms: ms,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 5000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: rq1,
                ..Default::default()
            },
        ],
    };
    let samples = vec![mk(100, 6000), mk(200, 6100), mk(300, 6200)];
    let summary = MonitorSummary::from_samples(&samples);
    assert_eq!(
        summary.stuck_count, 2,
        "cpu0 frozen across both windows -> 2 observations (per-run break removed)"
    );
}

#[test]
fn stuck_count_accumulates_across_cpus_in_one_window() {
    // TWO CPUs frozen in the SAME window count TWICE. Pins removal of the
    // per-CPU inner early break: the old path stopped at the first stuck
    // CPU. cpu2 advances so the sample stays plausible.
    let mk = |ms: u64, rq2: u64| MonitorSample {
        prog_stats: None,
        elapsed_ms: ms,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 5000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 7000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: rq2,
                ..Default::default()
            },
        ],
    };
    let samples = vec![mk(100, 9000), mk(200, 9100)];
    let summary = MonitorSummary::from_samples(&samples);
    assert_eq!(
        summary.stuck_count, 2,
        "cpu0 + cpu1 both frozen in the one window -> 2 observations (per-cpu break removed)"
    );
}

#[test]
fn neg_combined_imbalance_and_stuck_both_reported() {
    let t = MonitorThresholds {
        max_imbalance_ratio: 2.0,
        sustained_samples: 1,
        fail_on_stall: true,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 10,
                    rq_clock: 2000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 10,
                    rq_clock: 3000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    assert_eq!(summary.stuck_count, 1);
    assert!(summary.max_imbalance_ratio >= 10.0);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.is_fail());
    let imb = v
        .failure_details()
        .find(|d| d.contains("imbalance"))
        .unwrap();
    assert!(
        imb.contains("exceeded threshold 2.0"),
        "imbalance format: {imb}"
    );
    let stall = v
        .failure_details()
        .find(|d| d.contains("rq_clock stall"))
        .unwrap();
    assert!(stall.contains("cpu0"), "stall format: {stall}");
    assert!(
        v.details.len() >= 2,
        "both violations must be reported, got {}",
        v.details.len()
    );
    assert!(v.summary.contains("FAILED"), "summary: {}", v.summary);
}

#[test]
fn stuck_idle_cpu_exempt() {
    // nr_running==0 on both samples: idle CPU, NOHZ tick stopped.
    // rq_clock not advancing is expected, not a stall.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 5000, // stuck but idle
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    assert_eq!(
        summary.stuck_count, 0,
        "idle CPU should not trigger stall in summary"
    );
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.is_pass(),
        "idle CPU should not trigger stall: {:?}",
        v.details
    );
}

#[test]
fn stuck_idle_to_busy_not_exempt() {
    // nr_running transitions from 0 to 1 — the CPU woke up but
    // rq_clock didn't advance. This IS a stall (the CPU is now
    // busy but the scheduler tick hasn't fired).
    // Second CPU has a different clock value so data_looks_valid passes.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000, // stuck, but now busy
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    assert_eq!(
        summary.stuck_count, 1,
        "busy CPU with frozen clock is a stall"
    );
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        !v.is_pass(),
        "busy CPU with frozen clock must fail: {:?}",
        v.details
    );
}

#[test]
fn stuck_sustained_window_filters_transient() {
    // With sustained_samples=3, a 2-sample stall doesn't trigger.
    // Second CPU has a different clock value so data_looks_valid passes.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 3,
        ..Default::default()
    };
    let mut samples = Vec::new();
    // 3 samples: 2 consecutive stall pairs for cpu0, then clock advances.
    for i in 0..3u64 {
        samples.push(MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000, // stuck for all 3
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000 + i * 500, // advancing
                    ..Default::default()
                },
            ],
        });
    }
    // Break the streak: clock advances in 4th sample.
    samples.push(MonitorSample {
        prog_stats: None,
        elapsed_ms: 300,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 6000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 7500,
                ..Default::default()
            },
        ],
    });
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    // 2 consecutive stall pairs < sustained_samples=3
    assert!(v.passed, "2 stall pairs < sustained=3: {:?}", v.details);
}

#[test]
fn stuck_sustained_window_catches_real_stuck() {
    // With sustained_samples=3, 3+ consecutive stall pairs trigger.
    // Second CPU has a different clock value so data_looks_valid passes.
    // `enforce: true` opts the verdict out of report-only mode (the
    // default), so the recorded stall violation flips `passed` to
    // false.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 3,
        enforce: true,
        ..Default::default()
    };
    // 4 samples = 3 consecutive stall pairs for cpu0. cpu1 advances.
    let samples: Vec<_> = (0..4u64)
        .map(|i| MonitorSample {
            prog_stats: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000, // stuck
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000 + i * 500, // advancing
                    ..Default::default()
                },
            ],
        })
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "3 consecutive stall pairs must fail");
    assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
}

#[test]
fn from_samples_idle_cpu_no_stuck() {
    // from_samples should not flag stall when both samples have
    // nr_running==0 on the stuck CPU.
    let s1 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 0,
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
                nr_running: 0,
                rq_clock: 5000, // stuck but idle
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
    assert_eq!(summary.stuck_count, 0);
}

#[test]
fn stuck_below_sustained_passes() {
    // 1 stall pair with sustained_samples=5 should pass.
    // Second CPU has a different clock value so data_looks_valid passes.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 5,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
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
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        },
        // Clock recovers.
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 300,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 8000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "1 stall < sustained=5: {:?}", v.details);
}

// -- vCPU CPU time gating tests --

#[test]
fn evaluate_suppresses_stuck_when_vcpu_preempted() {
    // vcpu_cpu_time_ns shows < threshold advancement -> vCPU was
    // preempted, stall should be suppressed. Use explicit threshold
    // (10ms) to avoid host CONFIG_HZ dependency.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    vcpu_cpu_time_ns: Some(1_000_000_000),
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    vcpu_cpu_time_ns: Some(1_000_000_000),
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,                        // stuck
                    vcpu_cpu_time_ns: Some(1_000_500_000), // 0.5ms < 10ms threshold
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    vcpu_cpu_time_ns: Some(1_010_000_000),
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples_with_threshold(&samples, 10_000_000);
    assert_eq!(
        summary.stuck_count, 0,
        "preempted vCPU should not flag stall in summary"
    );
    let report = MonitorReport {
        samples,
        summary,
        preemption_threshold_ns: 10_000_000,
        watchdog_observation: None,
        page_offset: 0,
        boot_wait_outcome: BootWaitOutcome::NotConfigured,
    };
    let v = t.evaluate(&report);
    assert!(
        v.is_pass(),
        "preempted vCPU should suppress stall: {:?}",
        v.details
    );
}

#[test]
fn evaluate_catches_stuck_when_vcpu_running() {
    // vcpu_cpu_time_ns shows advancement >= threshold -> vCPU was
    // running, stall is real. Use explicit threshold (10ms) to avoid
    // host CONFIG_HZ dependency (DEFAULT_HZ=250 gives 40ms threshold,
    // which would mask the 10ms advance).
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    vcpu_cpu_time_ns: Some(1_000_000_000),
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000,
                    vcpu_cpu_time_ns: Some(1_000_000_000),
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,                        // stuck
                    vcpu_cpu_time_ns: Some(1_010_000_000), // 10ms advance
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    vcpu_cpu_time_ns: Some(1_010_000_000),
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples_with_threshold(&samples, 10_000_000);
    assert_eq!(
        summary.stuck_count, 1,
        "running vCPU with stuck clock is a stall"
    );
    let report = MonitorReport {
        samples,
        summary,
        preemption_threshold_ns: 10_000_000,
        watchdog_observation: None,
        page_offset: 0,
        boot_wait_outcome: BootWaitOutcome::NotConfigured,
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "running vCPU stall must fail: {:?}", v.details);
    assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
}

#[test]
fn evaluate_stuck_none_vcpu_time_falls_back_to_current_behavior() {
    // vcpu_cpu_time_ns is None -> assume vCPU was running (don't suppress).
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
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
        },
        MonitorSample {
            prog_stats: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000, // stuck, no vcpu_cpu_time_ns
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
                    ..Default::default()
                },
            ],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    assert_eq!(
        summary.stuck_count, 1,
        "None vcpu time should not suppress stall"
    );
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        !v.is_pass(),
        "None vcpu time should detect stall: {:?}",
        v.details
    );
}

#[test]
fn from_samples_suppresses_stuck_when_vcpu_preempted() {
    // from_samples_with_threshold should respect vcpu_cpu_time_ns
    // gating. Use explicit threshold to avoid host CONFIG_HZ dependency.
    let s1 = MonitorSample {
        prog_stats: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 5000,
                vcpu_cpu_time_ns: Some(1_000_000_000),
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 6000,
                vcpu_cpu_time_ns: Some(1_000_000_000),
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
                rq_clock: 5000,                        // stuck
                vcpu_cpu_time_ns: Some(1_000_100_000), // 0.1ms < 10ms threshold
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 7000,
                vcpu_cpu_time_ns: Some(1_010_000_000),
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples_with_threshold(&[s1, s2], 10_000_000);
    assert_eq!(
        summary.stuck_count, 0,
        "preempted vCPU should not flag stall"
    );
}
