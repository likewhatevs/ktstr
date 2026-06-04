//! Unit tests for the `SustainedViolationTracker` state machine
//! (consecutive-violation counter, worst-run preservation).
//! Co-located via the sibling `*_tests.rs` pattern.

#![cfg(test)]

use super::*;

#[test]
fn sustained_tracker_no_violations() {
    let t = SustainedViolationTracker::default();
    assert!(!t.sustained(3));
    assert_eq!(t.worst_run, 0);
}

#[test]
fn sustained_tracker_single_violation_not_sustained() {
    let mut t = SustainedViolationTracker::default();
    t.record(true, 5.0, 0);
    assert!(!t.sustained(3));
    assert_eq!(t.worst_run, 1);
    assert_eq!(t.worst_at, 0);
    assert!((t.worst_value - 5.0).abs() < f64::EPSILON);
}

#[test]
fn sustained_tracker_meets_threshold() {
    let mut t = SustainedViolationTracker::default();
    t.record(true, 2.0, 0);
    t.record(true, 3.0, 1);
    t.record(true, 4.0, 2);
    assert!(t.sustained(3));
    assert_eq!(t.worst_run, 3);
    assert_eq!(t.worst_at, 2);
    assert!((t.worst_value - 4.0).abs() < f64::EPSILON);
}

#[test]
fn sustained_tracker_reset_on_non_violation() {
    let mut t = SustainedViolationTracker::default();
    t.record(true, 1.0, 0);
    t.record(true, 2.0, 1);
    t.record(false, 0.0, 2); // reset
    t.record(true, 3.0, 3);
    assert!(!t.sustained(3));
    assert_eq!(t.worst_run, 2); // longest consecutive run was 2
    assert_eq!(t.consecutive, 1); // current run is 1
}

#[test]
fn sustained_tracker_worst_run_preserved_after_reset() {
    let mut t = SustainedViolationTracker::default();
    for i in 0..5 {
        t.record(true, i as f64, i);
    }
    t.record(false, 0.0, 5);
    t.record(true, 99.0, 6);
    t.record(true, 100.0, 7);
    // Worst run is 5 from the first sequence.
    assert_eq!(t.worst_run, 5);
    assert!(t.sustained(5));
    assert!(!t.sustained(6));
}
