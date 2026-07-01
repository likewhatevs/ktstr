//! Unit tests for `SchedstatDeltas` and per-domain `SchedDomainLbDelta`
//! aggregation in `MonitorSummary::from_samples`.
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
        bpf_map_fields: Vec::new(),
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
    // Window = elapsed 0->1000ms = 1.0s; the per-second rates derive in the
    // registry as total_X / total_schedstat_wall_sec (8000/1 = 8000 ns/s,
    // 100/1 = 100 csw/s).
    assert!((d.total_schedstat_wall_sec - 1.0).abs() < f64::EPSILON);
}

#[test]
fn schedstat_deltas_none_without_schedstat() {
    let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 1500)];
    let summary = MonitorSummary::from_samples(&samples);
    assert!(summary.schedstat_deltas.is_none());
}

#[test]
fn schedstat_deltas_single_sample() {
    // Single sample -> first == last, duration=0 -> window 0.0 (the registry
    // Rate derivation skips a zero denominator, so no per-second rate emits).
    let samples = vec![sample_with_schedstat(100, 1000, 5000, 10, 50, 30)];
    let summary = MonitorSummary::from_samples(&samples);
    let d = summary.schedstat_deltas.unwrap();
    assert_eq!(d.total_schedstat_wall_sec, 0.0);
    assert_eq!(d.total_run_delay, 0);
}

#[test]
fn schedstat_deltas_rates() {
    // 500ms window; per-CPU run_delay +2000, sched_count +40. The per-second
    // rates re-derive in the registry as total_X / total_schedstat_wall_sec.
    let samples = vec![
        sample_with_schedstat(0, 1000, 1000, 5, 10, 20),
        sample_with_schedstat(500, 2000, 3000, 15, 50, 40),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let d = summary.schedstat_deltas.unwrap();
    // Window = 0->500ms = 0.5s.
    assert!((d.total_schedstat_wall_sec - 0.5).abs() < f64::EPSILON);
    // 2 CPUs, each delta = 2000, total = 4000 (registry rate 4000/0.5 = 8000 ns/s).
    assert_eq!(d.total_run_delay, 4000);
    // 2 CPUs, each sched_count delta = 40, total = 80 (registry rate 80/0.5 = 160 csw/s).
    assert_eq!(d.total_sched_count, 80);
}

#[test]
fn schedstat_deltas_all_fields() {
    let make = |elapsed_ms, rd, pc, yc, sc, sg, tc, tl| MonitorSample {
        bpf_map_fields: Vec::new(),
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

// -- Per-domain LB delta aggregation -----------------------------

/// A `SchedDomainStats` with the per-idle-type counters placed in the
/// `CPU_IDLE` slot (\[1\]) so "sum across idle types" equals the supplied
/// value; the imbalance accumulators stay zero unless the caller overrides.
fn sd_stats(lb_count: u32, lb_failed: u32, lb_gained: u32) -> SchedDomainStats {
    SchedDomainStats {
        lb_count: [0, lb_count, 0],
        lb_failed: [0, lb_failed, 0],
        lb_gained: [0, lb_gained, 0],
        ..Default::default()
    }
}

/// `MonitorSample` whose `n_cpus` CPUs each carry the SAME single domain
/// (level `name`, stats `st`). Models the per-CPU domain replication the
/// reducer sums across (struct sched_domain is per-CPU).
fn sample_with_domain(
    elapsed_ms: u64,
    name: &str,
    st: SchedDomainStats,
    n_cpus: usize,
) -> MonitorSample {
    MonitorSample {
        bpf_map_fields: Vec::new(),
        prog_stats: None,
        psi_irq: None,
        elapsed_ms,
        cpus: (0..n_cpus)
            .map(|_| CpuSnapshot {
                nr_running: 1,
                sched_domains: Some(vec![SchedDomainSnapshot {
                    name: name.to_string(),
                    stats: Some(st.clone()),
                    ..Default::default()
                }]),
                ..Default::default()
            })
            .collect(),
    }
}

#[test]
fn sched_domain_deltas_summed_per_level_across_cpus() {
    // 2 CPUs, MC domain. Per-CPU delta: lb_count 15-5=10, lb_failed 4-1=3,
    // lb_gained 8-2=6. Summed across the 2 CPUs (per-CPU sd structs): 20/6/12.
    let samples = vec![
        sample_with_domain(0, "MC", sd_stats(5, 1, 2), 2),
        sample_with_domain(1000, "MC", sd_stats(15, 4, 8), 2),
    ];
    let domains = MonitorSummary::from_samples(&samples)
        .sched_domain_lb
        .unwrap();
    assert_eq!(domains.len(), 1);
    let mc = &domains[0];
    assert_eq!(mc.level, "MC");
    assert_eq!(mc.lb_count, 20);
    assert_eq!(mc.lb_failed, 6);
    assert_eq!(mc.lb_gained, 12);
}

#[test]
fn sched_domain_deltas_none_without_domains() {
    let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 1500)];
    assert!(
        MonitorSummary::from_samples(&samples)
            .sched_domain_lb
            .is_none()
    );
}

#[test]
fn sched_domain_deltas_sum_across_idle_types_and_imbalance_variants() {
    // lb_count spread across all 3 idle types (2+3+5=10), summed into one
    // per-level total. The four imbalance accumulators are placed one per idle
    // slot (load=1, util=2, task=3, misfit=4) and stay SEPARATE — each is summed
    // only across its own idle slots, never across variants (incommensurable
    // units). One CPU; first sample all-zero.
    let last = SchedDomainStats {
        lb_count: [2, 3, 5],
        lb_imbalance_load: [1, 0, 0],
        lb_imbalance_util: [0, 2, 0],
        lb_imbalance_task: [0, 0, 3],
        lb_imbalance_misfit: [4, 0, 0],
        ..Default::default()
    };
    let samples = vec![
        sample_with_domain(0, "MC", SchedDomainStats::default(), 1),
        sample_with_domain(1000, "MC", last, 1),
    ];
    let d = MonitorSummary::from_samples(&samples)
        .sched_domain_lb
        .unwrap();
    assert_eq!(d[0].lb_count, 10, "summed across 3 idle types");
    // The four imbalance accumulators stay SEPARATE (incommensurable units),
    // each summed across the 3 idle types: load=1, util=2, task=3, misfit=4.
    assert_eq!(d[0].lb_imbalance_load, 1);
    assert_eq!(d[0].lb_imbalance_util, 2);
    assert_eq!(d[0].lb_imbalance_task, 3);
    assert_eq!(d[0].lb_imbalance_misfit, 4);
}

#[test]
fn sched_domain_deltas_keyed_by_level_name_not_index() {
    // One CPU with two domains SMT (index 0) + MC (index 1). Each level is a
    // distinct keyed entry — the per-CPU-relative index is not used as the key.
    let mk = |elapsed_ms: u64, smt: SchedDomainStats, mc: SchedDomainStats| MonitorSample {
        bpf_map_fields: Vec::new(),
        prog_stats: None,
        psi_irq: None,
        elapsed_ms,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            sched_domains: Some(vec![
                SchedDomainSnapshot {
                    level: 0,
                    name: "SMT".into(),
                    stats: Some(smt),
                    ..Default::default()
                },
                SchedDomainSnapshot {
                    level: 1,
                    name: "MC".into(),
                    stats: Some(mc),
                    ..Default::default()
                },
            ]),
            ..Default::default()
        }],
    };
    let samples = vec![
        mk(0, sd_stats(1, 0, 0), sd_stats(10, 0, 0)),
        mk(1000, sd_stats(4, 0, 0), sd_stats(40, 0, 0)),
    ];
    let domains = MonitorSummary::from_samples(&samples)
        .sched_domain_lb
        .unwrap();
    let by: std::collections::BTreeMap<&str, u64> = domains
        .iter()
        .map(|d| (d.level.as_str(), d.lb_count))
        .collect();
    assert_eq!(by.get("SMT"), Some(&3), "SMT delta 4-1");
    assert_eq!(by.get("MC"), Some(&30), "MC delta 40-10");
}

#[test]
fn sched_domain_deltas_empty_series_is_none() {
    assert!(MonitorSummary::from_samples(&[]).sched_domain_lb.is_none());
}

#[test]
fn sched_domain_deltas_single_sample_yields_zero_deltas_not_none() {
    // A single sample → first == last → all-zero deltas, but Some(vec) (a
    // measured level with no activity over the window), NOT None — None would
    // mistreat a one-sample run as no-data.
    let samples = vec![sample_with_domain(100, "MC", sd_stats(7, 2, 3), 2)];
    let d = MonitorSummary::from_samples(&samples)
        .sched_domain_lb
        .expect("single sample is data, not None");
    assert_eq!(d.len(), 1);
    assert_eq!(d[0].level, "MC");
    assert_eq!(d[0].lb_count, 0, "first==last → zero delta");
    assert_eq!(d[0].lb_failed, 0);
    assert_eq!(d[0].lb_gained, 0);
}

#[test]
fn sched_domain_deltas_counter_reset_saturates_to_zero() {
    // last < first (counter rollover or sd re-init across the window) →
    // saturating_sub clamps to 0, never a wrapped huge value or debug panic.
    let samples = vec![
        sample_with_domain(0, "MC", sd_stats(100, 50, 40), 1),
        sample_with_domain(1000, "MC", sd_stats(10, 5, 4), 1),
    ];
    let d = MonitorSummary::from_samples(&samples)
        .sched_domain_lb
        .unwrap();
    assert_eq!(d[0].lb_count, 0, "last<first saturates to 0");
    assert_eq!(d[0].lb_failed, 0);
    assert_eq!(d[0].lb_gained, 0);
}

#[test]
fn sched_domain_deltas_level_absent_from_first_baselines_to_zero() {
    // A level present only in the LAST sample (no first baseline) → baseline 0
    // → delta is the full last counter. Built by giving the first sample a
    // different level (SMT) than the last (MC). Output is keyed off the last
    // sample's levels, so SMT (first-only) is not emitted.
    let samples = vec![
        sample_with_domain(0, "SMT", sd_stats(1, 1, 1), 1),
        sample_with_domain(1000, "MC", sd_stats(40, 9, 8), 1),
    ];
    let d = MonitorSummary::from_samples(&samples)
        .sched_domain_lb
        .unwrap();
    assert!(
        d.iter().all(|x| x.level != "SMT"),
        "output is keyed off the last sample; a first-only level is not emitted",
    );
    let mc = d
        .iter()
        .find(|x| x.level == "MC")
        .expect("MC present in last");
    assert_eq!(
        mc.lb_count, 40,
        "absent-in-first baseline 0 → full last value"
    );
    assert_eq!(mc.lb_failed, 9);
    assert_eq!(mc.lb_gained, 8);
}
