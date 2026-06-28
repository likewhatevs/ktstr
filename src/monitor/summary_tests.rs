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
    assert_eq!(summary.stuck_count, 0);
    assert_eq!(summary.avg_imbalance_ratio, 0.0);
    assert_eq!(summary.avg_nr_running, 0.0);
    assert_eq!(summary.avg_local_dsq_depth, 0.0);
}

#[test]
fn single_sample_imbalanced_cpus() {
    let sample = MonitorSample {
        prog_stats: None,
        psi_irq: None,
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
    assert_eq!(summary.stuck_count, 0);
    // avg fields: single sample with cpus [nr_running=1, nr_running=4]
    assert!((summary.avg_imbalance_ratio - 4.0).abs() < f64::EPSILON);
    assert!((summary.avg_nr_running - 2.5).abs() < f64::EPSILON);
    assert!((summary.avg_local_dsq_depth - 2.0).abs() < f64::EPSILON);
}

#[test]
fn stuck_count_when_clock_stuck() {
    let s1 = MonitorSample {
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
    };
    let s2 = MonitorSample {
        prog_stats: None,
        psi_irq: None,
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
    assert_eq!(summary.stuck_count, 1);
}

#[test]
fn balanced_cpus_ratio_one() {
    let sample = MonitorSample {
        prog_stats: None,
        psi_irq: None,
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
    assert_eq!(summary.stuck_count, 0);
    assert!((summary.avg_imbalance_ratio - 1.0).abs() < f64::EPSILON);
    assert!((summary.avg_nr_running - 3.0).abs() < f64::EPSILON);
    assert!((summary.avg_local_dsq_depth - 0.0).abs() < f64::EPSILON);
}

#[test]
fn single_cpu_no_division_by_zero() {
    let sample = MonitorSample {
        prog_stats: None,
        psi_irq: None,
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
    assert_eq!(summary.stuck_count, 0);
}

#[test]
fn all_zero_snapshots() {
    let sample = MonitorSample {
        prog_stats: None,
        psi_irq: None,
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
    assert_eq!(summary.stuck_count, 0);
    // avg: valid sample with 2 all-zero CPUs
    assert_eq!(summary.avg_imbalance_ratio, 0.0);
    assert_eq!(summary.avg_nr_running, 0.0);
    assert_eq!(summary.avg_local_dsq_depth, 0.0);
}

#[test]
fn empty_cpus_in_sample() {
    let sample = MonitorSample {
        prog_stats: None,
        psi_irq: None,
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
        psi_irq: None,
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
        psi_irq: None,
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
        psi_irq: None,
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
        psi_irq: None,
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
        psi_irq: None,
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
    assert_eq!(summary.stuck_count, 0);
    assert_eq!(summary.total_samples, 3);
}

#[test]
fn different_length_cpu_vecs() {
    // First sample has 2 CPUs, second has 3. Stall detection uses
    // min(prev.len, curr.len) = 2, so only CPUs 0-1 are compared.
    let s1 = MonitorSample {
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
                nr_running: 1,
                rq_clock: 2000,
                ..Default::default()
            },
        ],
    };
    let s2 = MonitorSample {
        prog_stats: None,
        psi_irq: None,
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
    assert_eq!(summary.stuck_count, 0);
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
            psi_irq: None,
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
                    avg_irq_util: None,
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
                    avg_irq_util: None,
                    sched_domains: None,
                },
            ],
        })
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    // total_samples matches input count.
    assert_eq!(summary.total_samples, 5);

    // max_imbalance_ratio = max over samples of (max_nr / max(1,min_nr)).
    // Each sample i: cpu0 nr=i+1, cpu1 nr=i+2 ⇒ ratio=(i+2)/(i+1),
    // descending from 2.0 (i=0) to 1.2 (i=4). The peak is sample 0 = 2.0.
    assert!(
        (summary.max_imbalance_ratio - 2.0).abs() < f64::EPSILON,
        "peak ratio is sample 0's 2/1: {}",
        summary.max_imbalance_ratio
    );

    // max_local_dsq_depth = max over all CPU readings of local_dsq_depth.
    // cpu0 carries i%3 (0,1,2,0,1) and cpu1 carries 0 ⇒ peak is i=2's 2.
    assert_eq!(
        summary.max_local_dsq_depth, 2,
        "peak local_dsq_depth is i=2's (2 % 3) = 2"
    );
    assert!(
        summary.max_local_dsq_depth <= DSQ_PLAUSIBILITY_CEILING,
        "must stay below the plausibility ceiling that gates validity",
    );

    // stuck_count: rq_clock advances each sample, so no stuck.
    assert_eq!(
        summary.stuck_count, 0,
        "no stuck expected with advancing rq_clock"
    );

    // event_deltas are end-minus-start over first/last samples with
    // counters (sample 0 and sample 4), exactly computable here:
    //   select_cpu_fallback sum: s0 = 0*2 + 0*3 = 0; s4 = 4*2 + 4*3 = 20
    //   dispatch_keep_last sum:  s0 = 0   + 0   = 0; s4 = 4   + 4*2 = 12
    //   window = last.elapsed_ms - first.elapsed_ms = 400 - 0 = 400ms = 0.4s
    let deltas = summary
        .event_deltas
        .as_ref()
        .expect("event deltas must be present");
    assert_eq!(
        deltas.total_fallback, 20,
        "total_fallback = last_sum(20) - first_sum(0)"
    );
    assert_eq!(
        deltas.total_dispatch_keep_last, 12,
        "total_dispatch_keep_last = last_sum(12) - first_sum(0)"
    );
    assert!(
        (deltas.fallback_rate - 50.0).abs() < f64::EPSILON,
        "fallback_rate = 20 / 0.4s = 50.0: {}",
        deltas.fallback_rate
    );
    assert!(
        (deltas.keep_last_rate - 30.0).abs() < f64::EPSILON,
        "keep_last_rate = 12 / 0.4s = 30.0: {}",
        deltas.keep_last_rate
    );
    // The per-sample burst max equals the largest consecutive-sample
    // fallback delta. Sums per sample: 0,5,10,15,20 (i*2 + i*3 = 5i),
    // so every consecutive delta is exactly 5.
    assert_eq!(
        deltas.max_fallback_burst, 5,
        "each consecutive fallback delta is 5i - 5(i-1) = 5"
    );

    // avg fields over all 10 valid CPU readings:
    //   avg_nr_running   = (sum cpu0 1..5=15 + sum cpu1 2..6=20) / 10 = 3.5
    //   avg_local_dsq    = (sum cpu0 i%3=4 + sum cpu1 0=0)       / 10 = 0.4
    //   avg_imbalance    = mean of (i+2)/(i+1) over i=0..4 = 437/300
    assert!(
        (summary.avg_nr_running - 3.5).abs() < f64::EPSILON,
        "avg_nr_running = 35 / 10 readings: {}",
        summary.avg_nr_running,
    );
    assert!(
        (summary.avg_local_dsq_depth - 0.4).abs() < f64::EPSILON,
        "avg_local_dsq_depth = 4 / 10 readings: {}",
        summary.avg_local_dsq_depth,
    );
    let expected_avg_imbalance = (2.0 + 1.5 + 4.0 / 3.0 + 1.25 + 1.2) / 5.0;
    assert!(
        (summary.avg_imbalance_ratio - expected_avg_imbalance).abs() < 1e-12,
        "avg_imbalance = mean of (i+2)/(i+1): got {} want {}",
        summary.avg_imbalance_ratio,
        expected_avg_imbalance,
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
    assert_eq!(summary.stuck_count, 0);
    assert_eq!(summary.avg_imbalance_ratio, 0.0);
    assert_eq!(summary.avg_nr_running, 0.0);
    assert_eq!(summary.avg_local_dsq_depth, 0.0);
    assert_eq!(summary.psi_irq_full_avg10, None);
    assert_eq!(summary.total_irq_pressure_us, None);
    assert!(
        summary.event_deltas.is_none(),
        "empty input must not produce event deltas"
    );
}

// -- avg_irq_util (PELT IRQ load) fold --

#[test]
fn avg_irq_util_means_reporting_cpus_and_skips_none() {
    // CPUs that report avg_irq_util feed the mean + peak; a CPU with None
    // (CONFIG_HAVE_SCHED_AVG_IRQ off / unresolved offset) is SKIPPED, not
    // counted as 0 — the divisor is the reporting-reading count, so a
    // partial-report kernel is neither diluted nor false-zeroed.
    let s1 = MonitorSample {
        prog_stats: None,
        psi_irq: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                avg_irq_util: Some(40),
                ..Default::default()
            },
            CpuSnapshot {
                avg_irq_util: Some(60),
                ..Default::default()
            },
        ],
    };
    let s2 = MonitorSample {
        prog_stats: None,
        psi_irq: None,
        elapsed_ms: 200,
        cpus: vec![
            CpuSnapshot {
                avg_irq_util: Some(80),
                ..Default::default()
            },
            CpuSnapshot {
                avg_irq_util: None,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[s1, s2]);
    // 3 reporting readings (40, 60, 80); the None is skipped from BOTH the
    // numerator and the divisor: mean = 180 / 3 = 60.0.
    let avg = summary
        .avg_irq_util
        .expect("some CPU reported avg_irq_util");
    assert!(
        (avg - 60.0).abs() < f64::EPSILON,
        "mean over reporting CPUs only (None skipped): {avg}",
    );
    assert_eq!(
        summary.max_avg_irq_util,
        Some(80.0),
        "peak across reporting CPUs/samples",
    );
}

#[test]
fn avg_irq_util_none_when_no_cpu_reports() {
    // No CPU reports avg_irq_util (a non-HAVE_SCHED_AVG_IRQ kernel): both the
    // mean and the peak are None (loud-absent), never a false 0.0.
    let sample = MonitorSample {
        prog_stats: None,
        psi_irq: None,
        elapsed_ms: 100,
        cpus: vec![
            CpuSnapshot {
                nr_running: 2,
                rq_clock: 1000,
                avg_irq_util: None,
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 2,
                rq_clock: 2000,
                avg_irq_util: None,
                ..Default::default()
            },
        ],
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert_eq!(
        summary.avg_irq_util, None,
        "no reporting CPU -> None, not 0.0",
    );
    assert_eq!(
        summary.max_avg_irq_util, None,
        "no reporting CPU -> None, not 0.0",
    );
    // The sample WAS valid (avg_nr_running computed) — proving the None is
    // the avg_irq gate, not the whole sample being skipped.
    assert!((summary.avg_nr_running - 2.0).abs() < f64::EPSILON);
}

// -- PSI-irq run-level fold (system-wide psi_system host-walk) --

/// `psi_irq_full_avg10` is the MEAN of the decoded avg10 EWMA (raw/2048 = %)
/// across the samples that reported PSI-irq (a Gauge), and
/// `total_irq_pressure_us` is the end-start delta of the cumulative `total` ns
/// (decoded ns→µs, a Counter). Pins both folds with exactly computable values.
#[test]
fn psi_irq_folds_avg10_mean_and_total_delta() {
    // avg10 decode = raw / 2048 (FIXED_1): 51200 → 25%, 153600 → 75% ⇒ mean 50%.
    // total decode = ns / 1000: delta (3_500_000 - 1_000_000) ns = 2500 µs.
    let s1 = MonitorSample {
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: 1000,
            ..Default::default()
        }],
        prog_stats: None,
        psi_irq: Some(PsiIrqSample {
            avg10_raw: 51_200,
            total_ns: 1_000_000,
        }),
    };
    let s2 = MonitorSample {
        elapsed_ms: 200,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: 2000,
            ..Default::default()
        }],
        prog_stats: None,
        psi_irq: Some(PsiIrqSample {
            avg10_raw: 153_600,
            total_ns: 3_500_000,
        }),
    };
    let summary = MonitorSummary::from_samples(&[s1, s2]);
    let avg10 = summary
        .psi_irq_full_avg10
        .expect("samples reported PSI-irq");
    assert!(
        (avg10 - 50.0).abs() < f64::EPSILON,
        "avg10 mean = (25 + 75) / 2 = 50.0: {avg10}",
    );
    let total = summary
        .total_irq_pressure_us
        .expect("samples reported PSI-irq");
    assert!(
        (total - 2500.0).abs() < f64::EPSILON,
        "total delta = (3_500_000 - 1_000_000) ns / 1000 = 2500 µs: {total}",
    );
}

/// No sample carries PSI-irq (a kernel without CONFIG_PSI /
/// CONFIG_IRQ_TIME_ACCOUNTING, or no `psi_system` symbol): BOTH run-level PSI
/// metrics are `None` (loud-absent), never a false 0.0 — the same
/// absent-vs-measured-zero contract as `avg_irq_util`.
#[test]
fn psi_irq_none_when_no_sample_reports() {
    let sample = MonitorSample {
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            nr_running: 2,
            rq_clock: 1000,
            ..Default::default()
        }],
        prog_stats: None,
        psi_irq: None,
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert_eq!(
        summary.psi_irq_full_avg10, None,
        "no PSI reading -> None, not 0.0",
    );
    assert_eq!(
        summary.total_irq_pressure_us, None,
        "no PSI reading -> None, not 0.0",
    );
    // The sample WAS valid (avg_nr_running computed) — proving the None is the
    // PSI gate, not the whole sample being skipped.
    assert!((summary.avg_nr_running - 2.0).abs() < f64::EPSILON);
}

/// A PSI / scheduler reset rewinds the monotonic cumulative `total[]` between
/// samples. `total_irq_pressure_us` uses `saturating_sub`, so the delta clamps
/// to 0 rather than underflowing into a giant bogus pressure — mirroring the
/// event-delta counter-reset clamp.
#[test]
fn psi_irq_total_saturates_on_counter_reset() {
    let s1 = MonitorSample {
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: 1000,
            ..Default::default()
        }],
        prog_stats: None,
        psi_irq: Some(PsiIrqSample {
            avg10_raw: 0,
            total_ns: 5_000_000,
        }),
    };
    let s2 = MonitorSample {
        elapsed_ms: 200,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: 2000,
            ..Default::default()
        }],
        prog_stats: None,
        psi_irq: Some(PsiIrqSample {
            avg10_raw: 0,
            total_ns: 1_000_000, // rewound below the first reading
        }),
    };
    let summary = MonitorSummary::from_samples(&[s1, s2]);
    assert_eq!(
        summary.total_irq_pressure_us,
        Some(0.0),
        "counter reset must clamp the delta to 0, not underflow",
    );
}

/// A single PSI-reporting sample: the total end-start delta is 0 (first ==
/// last) and the avg10 mean is exactly that one sample's decoded value — the
/// single-sample boundary of both folds.
#[test]
fn psi_irq_single_sample_zero_total_delta() {
    let sample = MonitorSample {
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: 1000,
            ..Default::default()
        }],
        prog_stats: None,
        psi_irq: Some(PsiIrqSample {
            avg10_raw: 102_400, // 102400 / 2048 = 50.0%
            total_ns: 7_000_000,
        }),
    };
    let summary = MonitorSummary::from_samples(&[sample]);
    assert!(
        (summary.psi_irq_full_avg10.unwrap() - 50.0).abs() < f64::EPSILON,
        "single-sample avg10 = the lone decoded value",
    );
    assert_eq!(
        summary.total_irq_pressure_us,
        Some(0.0),
        "single sample: first == last ⇒ delta 0",
    );
}

/// A monitor sample with `psi_irq = None` interleaved between reporting samples
/// (a sample where the read couldn't resolve, or a gap) is FILTERED from the
/// fold — NOT treated as a 0% reading that drags the mean down, and NOT breaking
/// the first→last cumulative delta. Pins `from_samples`'s `filter_map` compaction
/// so a future refactor to a zero-fill can't silently bias the gauge.
#[test]
fn psi_irq_interleaved_none_is_filtered_not_zeroed() {
    let mk = |elapsed_ms: u64, psi: Option<PsiIrqSample>| MonitorSample {
        elapsed_ms,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: elapsed_ms * 10,
            ..Default::default()
        }],
        prog_stats: None,
        psi_irq: psi,
    };
    let samples = vec![
        mk(
            100,
            Some(PsiIrqSample {
                avg10_raw: 102_400, // 50%
                total_ns: 1_000_000,
            }),
        ),
        mk(200, None),
        mk(
            300,
            Some(PsiIrqSample {
                avg10_raw: 102_400, // 50%
                total_ns: 3_000_000,
            }),
        ),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    // Mean over the 2 REPORTING samples = 50.0, NOT (50 + 0 + 50)/3 = 33.3 — the
    // None is filtered (filter_map compacts it out), never zero-filled.
    assert!(
        (summary.psi_irq_full_avg10.unwrap() - 50.0).abs() < f64::EPSILON,
        "interleaved None must be filtered from the mean, not counted as 0%",
    );
    // Delta spans first-reporting (1e6) → last-reporting (3e6) = 2e6 ns = 2000 µs;
    // the mid-gap None does not break the monotonic cumulative delta.
    assert_eq!(
        summary.total_irq_pressure_us,
        Some(2000.0),
        "delta spans first→last reporting sample across the None gap",
    );
}

// -- fold_run_level_ext (the shared monitor-summary → ext fold, used by both
// group::sidecar_to_row and VmResult::run_metric) --

/// fold_run_level_ext folds the 5 ext-only run-level monitor metrics
/// (avg_nr_running + the PELT IRQ load pair + the PSI-irq pair) into the ext
/// map. Pins: all 5 inserted with a sampled summary; the Option IRQ fields are
/// loud-absent (key omitted, never a false 0.0) when None; a 0-sample summary is
/// a no-op. This is the shared path the sidecar row and the in-test run_metric
/// accessor both use, so the key list + guard can't drift between them.
#[test]
fn fold_run_level_ext_folds_the_five_monitor_metrics() {
    use std::collections::BTreeMap;
    let mut s = MonitorSummary {
        total_samples: 5,
        avg_nr_running: 2.5,
        avg_irq_util: Some(40.0),
        max_avg_irq_util: Some(80.0),
        psi_irq_full_avg10: Some(12.5),
        total_irq_pressure_us: Some(3000.0),
        ..Default::default()
    };
    let mut ext = BTreeMap::new();
    s.fold_run_level_ext(&mut ext);
    assert_eq!(ext.get("avg_nr_running"), Some(&2.5));
    assert_eq!(ext.get("avg_irq_util"), Some(&40.0));
    assert_eq!(ext.get("max_avg_irq_util"), Some(&80.0));
    assert_eq!(ext.get("psi_irq_full_avg10"), Some(&12.5));
    assert_eq!(ext.get("total_irq_pressure_us"), Some(&3000.0));

    // Loud-absent: None Option fields → keys omitted (never a false 0.0);
    // avg_nr_running (plain f64) is still inserted.
    s.avg_irq_util = None;
    s.psi_irq_full_avg10 = None;
    let mut ext2 = BTreeMap::new();
    s.fold_run_level_ext(&mut ext2);
    assert_eq!(ext2.get("avg_nr_running"), Some(&2.5));
    assert_eq!(ext2.get("avg_irq_util"), None, "None → absent, not 0.0");
    assert_eq!(ext2.get("psi_irq_full_avg10"), None, "None → absent, not 0.0");
    assert_eq!(ext2.get("max_avg_irq_util"), Some(&80.0));
    assert_eq!(ext2.get("total_irq_pressure_us"), Some(&3000.0));

    // total_samples == 0 → no-op (a 0-sample run carries no occupancy/IRQ signal).
    let empty = MonitorSummary {
        total_samples: 0,
        avg_nr_running: 9.9,
        avg_irq_util: Some(50.0),
        ..Default::default()
    };
    let mut ext3 = BTreeMap::new();
    empty.fold_run_level_ext(&mut ext3);
    assert!(ext3.is_empty(), "0-sample summary folds nothing");

    // entry().or_insert(): a value already present is NOT overwritten (an earlier
    // populator wins; no key overlaps in production, but the guard is defensive).
    let mut ext4 = BTreeMap::new();
    ext4.insert("avg_nr_running".to_string(), 1.0);
    s.fold_run_level_ext(&mut ext4);
    assert_eq!(ext4.get("avg_nr_running"), Some(&1.0), "pre-set value wins");
}

#[test]
fn fold_run_level_ext_folds_per_domain_lb_keys() {
    use std::collections::BTreeMap;
    let s = MonitorSummary {
        total_samples: 3,
        sched_domain_lb: Some(vec![SchedDomainLbDelta {
            level: "MC".into(),
            lb_count: 20,
            lb_failed: 6,
            lb_gained: 12,
            lb_imbalance_load: 100,
            lb_imbalance_util: 200,
            lb_imbalance_task: 3,
            lb_imbalance_misfit: 0,
            alb_count: 1,
            alb_pushed: 2,
        }]),
        ..Default::default()
    };
    let mut ext = BTreeMap::new();
    s.fold_run_level_ext(&mut ext);
    // Level-suffixed (lowercased), one key per curated counter. The four
    // imbalance accumulators are emitted as separate same-unit keys.
    assert_eq!(ext.get("lb_count_mc"), Some(&20.0));
    assert_eq!(ext.get("lb_failed_mc"), Some(&6.0));
    assert_eq!(ext.get("lb_gained_mc"), Some(&12.0));
    assert_eq!(ext.get("lb_imbalance_load_mc"), Some(&100.0));
    assert_eq!(ext.get("lb_imbalance_util_mc"), Some(&200.0));
    assert_eq!(ext.get("lb_imbalance_task_mc"), Some(&3.0));
    assert_eq!(
        ext.get("lb_imbalance_misfit_mc"),
        Some(&0.0),
        "a present level's zero counter is a measured 0, emitted (not absent)"
    );
    assert_eq!(ext.get("alb_count_mc"), Some(&1.0));
    assert_eq!(ext.get("alb_pushed_mc"), Some(&2.0));

    // None sched_domain_lb → NO per-domain keys (level-granularity absence:
    // a level not in the run's topology emits nothing, distinct from a
    // present level's measured zero above).
    let empty = MonitorSummary {
        total_samples: 3,
        ..Default::default()
    };
    let mut ext2 = BTreeMap::new();
    empty.fold_run_level_ext(&mut ext2);
    assert!(
        !ext2.keys().any(|k| k.starts_with("lb_") || k.starts_with("alb_")),
        "no per-domain keys when sched_domain_lb is None"
    );
}
