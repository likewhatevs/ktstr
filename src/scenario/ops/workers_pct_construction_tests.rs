
use super::types::CgroupDef;
use crate::workload::WorkSpec;

/// Builder rejects NaN at construction.
#[test]
#[should_panic(expected = "must be finite and > 0.0")]
fn cgroup_def_workers_pct_panics_on_nan() {
    let _ = CgroupDef::named("x").workers_pct(f64::NAN);
}

#[test]
#[should_panic(expected = "must be finite and > 0.0")]
fn cgroup_def_workers_pct_panics_on_inf() {
    let _ = CgroupDef::named("x").workers_pct(f64::INFINITY);
}

#[test]
#[should_panic(expected = "must be finite and > 0.0")]
fn cgroup_def_workers_pct_panics_on_zero() {
    let _ = CgroupDef::named("x").workers_pct(0.0);
}

#[test]
#[should_panic(expected = "must be finite and > 0.0")]
fn cgroup_def_workers_pct_panics_on_negative() {
    let _ = CgroupDef::named("x").workers_pct(-0.5);
}

#[test]
#[should_panic(expected = "must be finite and > 0.0")]
fn work_spec_workers_pct_panics_on_nan() {
    let _ = WorkSpec::default().workers_pct(f64::NAN);
}

#[test]
#[should_panic(expected = "must be finite and > 0.0")]
fn work_spec_workers_pct_panics_on_inf() {
    let _ = WorkSpec::default().workers_pct(f64::INFINITY);
}

#[test]
#[should_panic(expected = "must be finite and > 0.0")]
fn work_spec_workers_pct_panics_on_zero() {
    let _ = WorkSpec::default().workers_pct(0.0);
}

#[test]
#[should_panic(expected = "must be finite and > 0.0")]
fn work_spec_workers_pct_panics_on_negative() {
    let _ = WorkSpec::default().workers_pct(-0.5);
}
