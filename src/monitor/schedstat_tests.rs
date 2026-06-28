//! Unit tests for `SchedstatDeltas` aggregation in
//! `MonitorSummary::from_samples`.
//! Co-located via the sibling `*_tests.rs` pattern.

#![cfg(test)]

use super::test_util::balanced_sample;
use super::*;

fn sample_with_schedstat(
    elapsed_ms: u64,
    clock_base: u64,
    run_delay: u64,
    pcount: u64,
    sched_count: u32,
    ttwu_count: u32,
) -> MonitorSample {
    MonitorSample {
        prog_stats: None,
        psi_irq: None,
        elapsed_ms,
        cpus: vec![
            CpuSnapshot {
                nr_running: 2,
                rq_clock: clock_base,
                schedstat: Some(RqSchedstat {
                    run_delay,
                    pcount,
                    sched_count,
                    ttwu_count,
                    ..Default::default()
                }),
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 2,
                rq_clock: clock_base + 100,
                schedstat: Some(RqSchedstat {
                    run_delay,
                    pcount,
                    sched_count,
                    ttwu_count,
                    ..Default::default()
                }),
                ..Default::default()
            },
        ],
    }
}

#[test]
fn schedstat_deltas_computed_from_samples() {
    // 2 CPUs, each starting at run_delay=1000, ending at 5000.
    // Total delta = 2 * (5000 - 1000) = 8000.
    let samples = vec![
        sample_with_schedstat(0, 1000, 1000, 10, 50, 30),
        sample_with_schedstat(1000, 2000, 5000, 20, 100, 60),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let d = summary.schedstat_deltas.unwrap();
    assert_eq!(d.total_run_delay, 8000);
    assert_eq!(d.total_pcount, 20);
    assert_eq!(d.total_sched_count, 100);
    assert_eq!(d.total_ttwu_count, 60);
    // Rate: 8000 ns / 1.0 s = 8000.0 ns/s.
    assert!((d.run_delay_rate - 8000.0).abs() < f64::EPSILON);
    assert!((d.sched_count_rate - 100.0).abs() < f64::EPSILON);
}

#[test]
fn schedstat_deltas_none_without_schedstat() {
    let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 1500)];
    let summary = MonitorSummary::from_samples(&samples);
    assert!(summary.schedstat_deltas.is_none());
}

#[test]
fn schedstat_deltas_single_sample() {
    // Single sample -> first == last, duration=0, rates=0.
    let samples = vec![sample_with_schedstat(100, 1000, 5000, 10, 50, 30)];
    let summary = MonitorSummary::from_samples(&samples);
    let d = summary.schedstat_deltas.unwrap();
    assert_eq!(d.run_delay_rate, 0.0);
    assert_eq!(d.sched_count_rate, 0.0);
    assert_eq!(d.total_run_delay, 0);
}

#[test]
fn schedstat_deltas_rates() {
    // 1 CPU, 500ms window. run_delay increases by 2000, sched_count by 40.
    // run_delay_rate = 2000 / 0.5 = 4000.0 ns/s.
    // sched_count_rate = 40 / 0.5 = 80.0 /s.
    let samples = vec![
        sample_with_schedstat(0, 1000, 1000, 5, 10, 20),
        sample_with_schedstat(500, 2000, 3000, 15, 50, 40),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let d = summary.schedstat_deltas.unwrap();
    // 2 CPUs, each delta = 2000, total = 4000.
    assert_eq!(d.total_run_delay, 4000);
    // rate = 4000 / 0.5s = 8000.0
    assert!((d.run_delay_rate - 8000.0).abs() < f64::EPSILON);
    // 2 CPUs, each sched_count delta = 40, total = 80.
    assert_eq!(d.total_sched_count, 80);
    // rate = 80 / 0.5s = 160.0
    assert!((d.sched_count_rate - 160.0).abs() < f64::EPSILON);
}

#[test]
fn schedstat_deltas_all_fields() {
    let make = |elapsed_ms, rd, pc, yc, sc, sg, tc, tl| MonitorSample {
        prog_stats: None,
        psi_irq: None,
        elapsed_ms,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: elapsed_ms * 10,
            schedstat: Some(RqSchedstat {
                run_delay: rd,
                pcount: pc,
                yld_count: yc,
                sched_count: sc,
                sched_goidle: sg,
                ttwu_count: tc,
                ttwu_local: tl,
            }),
            ..Default::default()
        }],
    };
    let samples = vec![
        make(100, 100, 10, 1, 20, 5, 30, 15),
        make(200, 500, 25, 4, 50, 12, 70, 35),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let d = summary.schedstat_deltas.unwrap();
    assert_eq!(d.total_run_delay, 400);
    assert_eq!(d.total_pcount, 15);
    assert_eq!(d.total_yld_count, 3);
    assert_eq!(d.total_sched_count, 30);
    assert_eq!(d.total_sched_goidle, 7);
    assert_eq!(d.total_ttwu_count, 40);
    assert_eq!(d.total_ttwu_local, 20);
}
