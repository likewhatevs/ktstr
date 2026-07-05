//! Unit tests for the `enforce` flag in `MonitorThresholds`:
//! the same violation must yield `passed=false` when
//! `enforce=true` and `passed=true` (report-only) when
//! `enforce=false`, with details and summary set accordingly.
//! Co-located via the sibling `*_tests.rs` pattern.

#![cfg(test)]

use super::*;

/// Behavioral pin: with `enforce: true`, a recorded violation flips
/// `verdict.passed` to false. The enforcement arithmetic at
/// src/monitor/mod.rs (passed = !failed || !enforce) reduces to
/// `passed = !failed` when enforce is true. A regression that
/// inverts this polarity (e.g. accidentally drops the negation on
/// `!self.enforce`) would let violations silently pass even under
/// enforce=true, defeating the test author's opt-in.
///
/// Parallel to `stuck_sustained_window_catches_real_stuck` in
/// `stall_detection_tests.rs` (same fixture) but explicitly asserts
/// the end-to-end arithmetic
/// rather than the canary-style "stall must fail" framing.
#[test]
fn enforce_true_with_violation_yields_passed_false() {
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 3,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..4u64)
        .map(|i| MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000 + i * 500,
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
        !v.is_pass(),
        "enforce=true + violation must yield passed=false: {:?}",
        v.details
    );
    assert!(
        !v.details.is_empty(),
        "violation must be recorded in details even when enforce=true"
    );
    assert!(
        v.summary.contains("FAILED"),
        "summary must indicate failure: {}",
        v.summary
    );
}

/// Behavioral pin: with `enforce: false` (the report-only default),
/// the SAME violation that fails the enforced fixture above MUST
/// yield `verdict.passed == true`. The violation is still recorded
/// in `details` and the summary carries the report-only advisory;
/// only the boolean verdict flips. Catches a regression in the
/// `passed = !failed || !enforce` arithmetic at src/monitor/mod.rs
/// that would silently fail tests using the report-only default —
/// exactly the silent-bug class the report-only mode is designed to
/// prevent.
///
/// Same fixture as `enforce_true_with_violation_yields_passed_false`
/// above with enforce flipped to false; pair-tested to pin both
/// polarities of the same end-to-end evaluation arithmetic.
#[test]
fn enforce_false_with_violation_yields_passed_true_and_records_details() {
    let t = MonitorThresholds {
        fail_on_stall: true,
        sustained_samples: 3,
        enforce: false,
        ..Default::default()
    };
    let samples: Vec<_> = (0..4u64)
        .map(|i| MonitorSample {
            bpf_map_fields: Vec::new(),
            prog_stats: None,
            psi_irq: None,
            elapsed_ms: i * 100,
            cpus: vec![
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 5000,
                    ..Default::default()
                },
                CpuSnapshot {
                    nr_running: 1,
                    rq_clock: 6000 + i * 500,
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
        v.is_pass(),
        "enforce=false + violation must yield passed=true (report-only mode)"
    );
    assert!(
        !v.details.is_empty(),
        "violation must STILL be recorded in details even under report-only mode"
    );
    assert!(
        v.details.iter().any(|d| d.contains("rq_clock stuck")),
        "details must carry the rq_clock stuck violation: {:?}",
        v.details
    );
    assert!(
        v.summary.contains("report-only") || v.summary.contains("flagged"),
        "summary must carry the report-only advisory: {}",
        v.summary
    );
}
