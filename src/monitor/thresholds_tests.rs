//! Unit tests for `MonitorThresholds::evaluate`: default values,
//! balanced/imbalanced load, DSQ depth, stuck rq_clock, validity
//! guards, and the `enforce` flag default.
//! Co-located via the sibling `*_tests.rs` pattern.

#![cfg(test)]

use super::test_util::balanced_sample;
use super::*;

/// Wire-format pin: `enforce` defaults to `false` (report-only
/// mode). Tests that want hard pass/fail must explicitly set
/// `enforce: true` in their `MonitorThresholds` literal OR use the
/// `Assert` builder's `.with_monitor_defaults()` helper (which sets
/// `enforce_monitor_thresholds = true`).
///
/// Without this canary, a future commit that flips the default back
/// to `enforce: true` (or any other value) would silently re-enable
/// enforcement for all tests using `..Default::default()` or
/// `MonitorThresholds::new()` — masking real bugs that the
/// report-only mode is designed to surface as warnings instead of
/// failures.
#[test]
fn enforce_defaults_to_false() {
    let t = MonitorThresholds::default();
    assert!(
        !t.enforce,
        "enforce must default to false (report-only mode)"
    );
    let d = MonitorThresholds::new();
    assert!(!d.enforce, "new().enforce must match default()");
}

#[test]
fn thresholds_default_values() {
    // Regression guard for `MonitorThresholds::new()`. Every
    // field is asserted: changing a default silently shifts what
    // "passes by default" across every test that inherits
    // defaults via `Assert::default_checks()` + per-scheduler
    // merge. If a default moves, the rationale belongs in the
    // doc comment on `new()` first; the test failure then
    // prompts the rationale update.
    let t = MonitorThresholds::default();
    assert!(
        (t.max_imbalance_ratio - 4.0).abs() < f64::EPSILON,
        "default max_imbalance_ratio drifted: {}",
        t.max_imbalance_ratio,
    );
    assert_eq!(
        t.max_local_dsq_depth, 50,
        "default max_local_dsq_depth drifted",
    );
    assert!(t.fail_on_stall, "default fail_on_stall drifted");
    assert_eq!(t.sustained_samples, 5, "default sustained_samples drifted");
    assert!(
        (t.max_fallback_rate - 200.0).abs() < f64::EPSILON,
        "default max_fallback_rate drifted: {}",
        t.max_fallback_rate,
    );
    assert!(
        (t.max_keep_last_rate - 100.0).abs() < f64::EPSILON,
        "default max_keep_last_rate drifted: {}",
        t.max_keep_last_rate,
    );
}

#[test]
fn thresholds_default_matches_const() {
    // `Default::default()` and `new()` must agree — the impl
    // forwards, but the forward is a single expression that a
    // drive-by refactor could break.
    let a = MonitorThresholds::default();
    let b = MonitorThresholds::new();
    assert!((a.max_imbalance_ratio - b.max_imbalance_ratio).abs() < f64::EPSILON);
    assert_eq!(a.max_local_dsq_depth, b.max_local_dsq_depth);
    assert_eq!(a.fail_on_stall, b.fail_on_stall);
    assert_eq!(a.sustained_samples, b.sustained_samples);
    assert!((a.max_fallback_rate - b.max_fallback_rate).abs() < f64::EPSILON);
    assert!((a.max_keep_last_rate - b.max_keep_last_rate).abs() < f64::EPSILON);
}

#[test]
fn thresholds_balanced_samples_pass() {
    let t = MonitorThresholds::default();
    let samples: Vec<_> = (0..10)
        .map(|i| balanced_sample(i * 100, 1000 + i * 500))
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "balanced samples should pass: {:?}", v.details);
}

#[test]
fn thresholds_imbalance_below_sustained_passes() {
    let t = MonitorThresholds {
        sustained_samples: 5,
        max_imbalance_ratio: 4.0,
        ..Default::default()
    };
    // 4 consecutive imbalanced samples (below sustained_samples=5).
    let mut samples = Vec::new();
    for i in 0..4 {
        samples.push(MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 10,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        });
    }
    // Then a balanced one to break the streak.
    samples.push(balanced_sample(400, 3000));
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.is_pass(),
        "4 imbalanced < sustained_samples=5: {:?}",
        v.details
    );
}

#[test]
fn thresholds_imbalance_at_sustained_fails() {
    let t = MonitorThresholds {
        sustained_samples: 5,
        max_imbalance_ratio: 4.0,
        enforce: true,
        ..Default::default()
    };
    // 5 consecutive imbalanced samples (ratio=10, threshold=4).
    let mut samples = Vec::new();
    for i in 0..5u64 {
        samples.push(MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 10,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        });
    }
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.is_fail());
    assert!(v.details.iter().any(|d| d.contains("imbalance")));
}

#[test]
fn thresholds_dsq_depth_sustained_fails() {
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_local_dsq_depth: 10,
        fail_on_stall: false,
        enforce: true,
        ..Default::default()
    };
    let mut samples = Vec::new();
    for i in 0..3u64 {
        samples.push(MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    local_dsq_depth: 20,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 2,
                    local_dsq_depth: 5,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        });
    }
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.is_fail());
    assert!(v.details.iter().any(|d| d.contains("DSQ depth")));
}

#[test]
fn thresholds_dsq_depth_below_sustained_passes() {
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_local_dsq_depth: 10,
        fail_on_stall: false,
        ..Default::default()
    };
    // Only 2 consecutive DSQ violations, then a clean sample.
    let mut samples = Vec::new();
    for i in 0..2u64 {
        samples.push(MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 2,
                    local_dsq_depth: 20,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 2,
                    local_dsq_depth: 5,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        });
    }
    samples.push(balanced_sample(200, 2000));
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "2 DSQ violations < sustained=3: {:?}", v.details);
}

#[test]
fn thresholds_stuck_fails() {
    // Stuck checks use the sustained_samples window. With sustained_samples=1,
    // a single stuck pair triggers failure. `enforce: true` opts out of
    // the report-only default so the violation flips `passed` to false.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
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
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                }, // stuck
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7000,
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
    assert!(v.is_fail());
    assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
}

#[test]
fn thresholds_stuck_disabled_passes() {
    // Test the `fail_on_stall: false` toggle: a CPU with a
    // stuck `rq_clock` must NOT fail the verdict when stall
    // detection is disabled. The setup uses 2 CPUs with mixed
    // rq_clock (cpu0 stuck at 5000/5000, cpu1 advancing
    // 6000/7500) so `data_looks_valid` returns true (rq_clocks
    // are not all identical) and the no-signal Inconclusive
    // arm doesn't fire — the test isolates the
    // `fail_on_stall=false` toggle from the data-validity
    // path.
    let t = MonitorThresholds {
        fail_on_stall: false,
        sustained_samples: 100,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
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
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                }, // cpu0 stuck but stall check disabled
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 7500,
                    ..Default::default()
                }, // cpu1 advancing so data_looks_valid passes
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
    assert!(
        v.is_pass(),
        "fail_on_stall=false must pass even with stuck cpu0: {:?}",
        v.details
    );
}

#[test]
fn thresholds_imbalance_interrupted_by_balanced_resets() {
    // 3 imbalanced, 1 balanced, 3 imbalanced — never reaches sustained=5.
    let t = MonitorThresholds {
        sustained_samples: 5,
        max_imbalance_ratio: 4.0,
        fail_on_stall: false,
        ..Default::default()
    };
    let mut samples = Vec::new();
    for i in 0..3u64 {
        samples.push(MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 10,
                    rq_clock: 1100 + i * 500,
                    ..Default::default()
                },
            ],
        });
    }
    samples.push(balanced_sample(300, 2500));
    for i in 4..7u64 {
        samples.push(MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 3000 + i * 500,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 10,
                    rq_clock: 3100 + i * 500,
                    ..Default::default()
                },
            ],
        });
    }
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.is_pass(),
        "interrupted imbalance should pass: {:?}",
        v.details
    );
}

#[test]
fn thresholds_multiple_violations() {
    // Both imbalance and stall in the same report. Both need to
    // reach sustained_samples to trigger. 3 samples = 2 consecutive
    // stall pairs for cpu0 (clock stuck at 1000), 2 consecutive
    // imbalance violations (ratio=5.0 > 2.0).
    let t = MonitorThresholds {
        sustained_samples: 2,
        max_imbalance_ratio: 2.0,
        fail_on_stall: true,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 5,
                    rq_clock: 2000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                }, // stall + imbalance
                CpuSnapshot {
                    nr_running: 5,
                    rq_clock: 3000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: 300,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 1000,
                    ..Default::default()
                }, // stall continues
                CpuSnapshot {
                    nr_running: 5,
                    rq_clock: 4000,
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
    assert!(v.is_fail());
    assert!(v.details.iter().any(|d| d.contains("imbalance")));
    assert!(v.details.iter().any(|d| d.contains("rq_clock stall")));
}

#[test]
fn thresholds_empty_samples_vec_yields_inconclusive() {
    // Empty samples vec — monitor produced no data at all (the
    // earliest Inconclusive arm in evaluate). Pin that this surfaces
    // as Inconclusive (not Pass), distinct from the populated-but-
    // empty-cpus case which legitimately passes.
    let t = MonitorThresholds::default();
    let report = MonitorReport {
        samples: vec![],
        summary: MonitorSummary::from_samples(&[]),
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.is_inconclusive(),
        "no samples must surface as Inconclusive (not Pass) — silent-pass guard. \
         Got is_pass={}, is_fail={}, summary={}",
        v.is_pass(),
        v.is_fail(),
        v.summary,
    );
    assert_eq!(
        v.summary, "no monitor samples",
        "summary text must match the dedicated no-samples narrative",
    );
}

#[test]
fn thresholds_empty_cpus_samples_pass() {
    let t = MonitorThresholds::default();
    let samples = vec![
        MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: 100,
            cpus: vec![],
        },
        MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: 200,
            cpus: vec![],
        },
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.is_pass());
}

#[test]
fn thresholds_uninitialized_memory_yields_inconclusive() {
    // Simulates what happens when monitor reads guest memory before
    // kernel initialization: all rq_clocks identical, DSQ depths
    // garbage. data_looks_valid rejects the sample, so the verdict
    // must be Inconclusive (not Pass) — passing on no-signal would
    // let a CI gate keying off is_pass treat "couldn't measure"
    // identically to "measured and OK", the silent-pass class the
    // Inconclusive arm exists to prevent.
    let t = MonitorThresholds::default();
    let garbage_clock = 10314579376562252011u64;
    let samples: Vec<_> = (0..10)
        .map(|i| MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: garbage_clock,
                    local_dsq_depth: 1550435906,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 0,
                    rq_clock: garbage_clock,
                    local_dsq_depth: 1550435906,
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
    assert!(
        v.is_inconclusive(),
        "uninitialized guest memory must surface as Inconclusive (not Pass) — \
         silent-pass guard at the MonitorVerdict surface. Got is_pass={}, is_fail={}, summary={}",
        v.is_pass(),
        v.is_fail(),
        v.summary,
    );
    assert!(
        !v.is_pass(),
        "is_pass must be false on the Inconclusive arm (strict mutex)",
    );
}

#[test]
fn thresholds_all_same_clocks_yields_inconclusive() {
    // All clocks identical across all CPUs and samples = uninitialized.
    // data_looks_valid rejects, so the verdict is Inconclusive (not
    // Pass) — same silent-pass guard as
    // [`thresholds_uninitialized_memory_yields_inconclusive`].
    let t = MonitorThresholds {
        fail_on_stall: true,
        ..Default::default()
    };
    let samples = vec![
        MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
            ],
        },
        MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: 200,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
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
    assert!(
        v.is_inconclusive(),
        "all-same clocks must surface as Inconclusive (not Pass) — \
         the no-signal arm cannot masquerade as a measured pass. Got is_pass={}, summary={}",
        v.is_pass(),
        v.summary,
    );
}

#[test]
fn thresholds_dsq_over_plausibility_ceiling_yields_inconclusive() {
    // sample_looks_valid rejects when any DSQ depth exceeds the
    // plausibility ceiling — the verdict is Inconclusive (not
    // Pass) so the operator sees "couldn't measure" rather than
    // a silent green light on garbage data.
    let t = MonitorThresholds::default();
    let samples = vec![MonitorSample {
        bpf_map_fields: Vec::new(),
        prog_stats: None,
        psi_irq: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 1000,
                local_dsq_depth: 50000,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 1,
                rq_clock: 2000,
                local_dsq_depth: 5,
                ..Default::default()
            },
        ],
    }];
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(
        v.is_inconclusive(),
        "implausible DSQ depth must surface as Inconclusive (not Pass) — \
         silent-pass guard. Got is_pass={}, summary={}",
        v.is_pass(),
        v.summary,
    );
}

#[test]
fn thresholds_single_cpu_single_sample_valid() {
    // A single reading cannot be compared, so all_clocks_same with
    // total_readings=1 should still be treated as valid.
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 1,
        enforce: true,
        ..Default::default()
    };
    let samples = vec![MonitorSample {
        bpf_map_fields: Vec::new(),
        prog_stats: None,
        psi_irq: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: 5000,
            ..Default::default()
        }],
    }];
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "single reading should be valid: {:?}", v.details);
}
