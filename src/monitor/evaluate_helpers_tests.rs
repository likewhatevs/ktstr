//! Direct unit tests for the per-aspect helpers
//! [`MonitorThresholds::track_imbalance_and_dsq`],
//! [`MonitorThresholds::track_stall`],
//! [`MonitorThresholds::track_event_rates`], and
//! [`MonitorThresholds::summarize`].
//!
//! `evaluate` itself is covered end-to-end by the threshold,
//! stall_detection, event_rates, and enforce sibling test modules.
//! These tests anchor each helper to its specific contract so a
//! regression in one aspect surfaces here directly — rather than as
//! a churn of failures across every evaluate-based fixture that
//! happens to exercise the broken path.
//! Co-located via the sibling `*_tests.rs` pattern.

#![cfg(test)]

use super::test_util::balanced_sample;
use super::*;

/// `track_imbalance_and_dsq` must run BOTH trackers across the same
/// sample stream and report `worst_dsq_cpu` as the CPU index whose
/// `local_dsq_depth` set the worst-value high-water mark — not the
/// first violating CPU, not the last. Pin this here so a future
/// refactor that swaps the "is new value worse than `worst_value`"
/// check for a "first violation seen" heuristic surfaces as a
/// direct helper test rather than as a confusing format-string
/// regression in `neg_tight_dsq_threshold_catches_small_depth`.
#[test]
fn track_imbalance_and_dsq_reports_worst_cpu_not_first() {
    let t = MonitorThresholds {
        max_imbalance_ratio: 100.0, // imbalance check disabled by ceiling
        max_local_dsq_depth: 1,
        ..Default::default()
    };
    // CPU 0 violates with depth=3; CPU 1 violates with depth=5. The
    // worst per-sample depth is 5 on cpu1 — `worst_dsq_cpu` must point
    // to cpu1.
    let samples = vec![MonitorSample {
        bpf_map_fields: Vec::new(),
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
                nr_running: 1,
                local_dsq_depth: 5,
                rq_clock: 2000,
                ..Default::default()
            },
        ],
    }];
    let (imbalance, dsq, worst_cpu) = t.track_imbalance_and_dsq(&samples);
    // Imbalance disabled by ceiling — no violation recorded.
    assert_eq!(imbalance.worst_run, 0, "imbalance must not be flagged");
    // DSQ violation: worst value is 5, attributed to cpu1.
    assert!(
        (dsq.worst_value - 5.0).abs() < f64::EPSILON,
        "dsq worst_value must be 5, got {}",
        dsq.worst_value
    );
    assert_eq!(worst_cpu, 1, "worst_dsq_cpu must be 1 (deepest violator)");
}

/// `track_imbalance_and_dsq` must NOT advance either tracker's
/// consecutive run when a sample has `cpus.is_empty()` (treat as a
/// non-violation, NOT a missing reading that breaks the sustained
/// streak). The continue arm records `(false, 0.0, i)` on both —
/// this test pins that empty-cpus does not silently advance a
/// violation streak through a gap.
#[test]
fn track_imbalance_and_dsq_empty_cpus_records_non_violation() {
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
    let (imbalance, dsq, _) = t.track_imbalance_and_dsq(&samples);
    assert_eq!(imbalance.consecutive, 0);
    assert_eq!(dsq.consecutive, 0);
    assert_eq!(imbalance.worst_run, 0);
    assert_eq!(dsq.worst_run, 0);
}

/// `track_stall` must size its per-CPU tracker vector to the
/// maximum `cpus.len()` across all samples in the report, NOT just
/// the first sample. A regression that uses `samples[0].cpus.len()`
/// would silently drop stalls on CPUs that appear later (e.g. CPU
/// hotplug increased the count mid-run).
#[test]
fn track_stall_sizes_vec_to_max_cpu_count_across_samples() {
    let t = MonitorThresholds {
        fail_on_stall: true,
        ..Default::default()
    };
    let s1 = MonitorSample {
        bpf_map_fields: Vec::new(),
        prog_stats: None,
        psi_irq: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: 1000,
            ..Default::default()
        }],
    };
    // Second sample has 3 CPUs — the tracker vec must size to 3.
    let s2 = MonitorSample {
        bpf_map_fields: Vec::new(),
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
    let report = MonitorReport {
        samples: vec![s1, s2],
        summary: MonitorSummary::default(),
        ..Default::default()
    };
    let stall = t.track_stall(&report);
    assert_eq!(
        stall.len(),
        3,
        "must size vec to max cpu count across samples"
    );
}

/// `track_stall` must suppress a frozen `rq_clock` when the vCPU
/// `cpu_time_ns` shows the vCPU was preempted by the host (advance
/// < threshold). This is the regression that
/// `evaluate_suppresses_stuck_when_vcpu_preempted` covers
/// end-to-end; pinning it at the helper boundary catches it without
/// the verdict-formatting layer in the way.
#[test]
fn track_stall_respects_vcpu_preemption_threshold() {
    let t = MonitorThresholds {
        fail_on_stall: true,
        ..Default::default()
    };
    let s1 = MonitorSample {
        bpf_map_fields: Vec::new(),
        prog_stats: None,
        psi_irq: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: 5000,
            vcpu_cpu_time_ns: Some(1_000_000_000),
            ..Default::default()
        }],
    };
    let s2 = MonitorSample {
        bpf_map_fields: Vec::new(),
        prog_stats: None,
        psi_irq: None,
        elapsed_ms: 200,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: 5000,                        // stuck
            vcpu_cpu_time_ns: Some(1_000_500_000), // 0.5ms < 10ms threshold → preempted
            ..Default::default()
        }],
    };
    let report = MonitorReport {
        samples: vec![s1, s2],
        summary: MonitorSummary::default(),
        preemption_threshold_ns: 10_000_000, // explicit 10ms
        watchdog_observation: None,
        page_offset: 0,
        boot_wait_outcome: BootWaitOutcome::NotConfigured,
        scx_event_counters_supported: false,
    };
    let stall = t.track_stall(&report);
    assert_eq!(stall.len(), 1);
    assert_eq!(
        stall[0].worst_run, 0,
        "preempted vCPU must not register as a stall"
    );
}

/// `track_event_rates` must zero-record the trackers across an
/// interval where either side is missing event counters. A
/// regression that simply skips the interval (no `record` call)
/// would silently advance an in-progress violation streak through
/// the missing interval — pinning this here guards the "missing
/// data is a reset, not a gap" contract.
#[test]
fn track_event_rates_missing_counters_records_non_violation() {
    let t = MonitorThresholds {
        max_fallback_rate: 0.0,
        max_keep_last_rate: 0.0,
        ..Default::default()
    };
    let samples = vec![balanced_sample(0, 1000), balanced_sample(100, 1500)];
    let (fallback_rate, keep_last_rate) = t.track_event_rates(&samples);
    assert_eq!(fallback_rate.consecutive, 0);
    assert_eq!(keep_last_rate.consecutive, 0);
    assert_eq!(fallback_rate.worst_run, 0);
    assert_eq!(keep_last_rate.worst_run, 0);
}

/// `track_event_rates` must not divide by zero when two consecutive
/// samples share the same `elapsed_ms`. The `interval_s <= 0.0`
/// guard records a non-violation and continues; a regression that
/// dropped the guard would divide by the zero interval -- `+inf` on
/// the fallback path (delta 0->1000) tripping a false violation that
/// the `fb.worst_run == 0` assertion catches, and `NaN` on the
/// keep_last path (0/0).
#[test]
fn track_event_rates_zero_interval_does_not_panic() {
    let t = MonitorThresholds::default();
    // Two samples with identical elapsed_ms → 0-second interval.
    let make = |fb| MonitorSample {
        bpf_map_fields: Vec::new(),
        prog_stats: None,
        psi_irq: None,
        elapsed_ms: 100,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: 1000,
            event_counters: Some(ScxEventCounters {
                select_cpu_fallback: fb,
                ..Default::default()
            }),
            ..Default::default()
        }],
    };
    let samples = vec![make(0), make(1000)];
    let (fb, kl) = t.track_event_rates(&samples);
    assert_eq!(fb.worst_run, 0);
    assert_eq!(kl.worst_run, 0);
}

/// Pin each of the three arms of `summarize`.
#[test]
fn summarize_three_arms() {
    assert_eq!(MonitorThresholds::summarize(false, true, 0), "monitor OK");
    assert_eq!(MonitorThresholds::summarize(false, false, 0), "monitor OK");
    assert_eq!(
        MonitorThresholds::summarize(true, true, 3),
        "monitor FAILED: 3 violation(s)"
    );
    let report_only = MonitorThresholds::summarize(true, false, 2);
    assert!(
        report_only.contains("flagged 2"),
        "report-only must say flagged: {report_only}"
    );
    assert!(
        report_only.contains("report-only"),
        "report-only must carry advisory: {report_only}"
    );
}
