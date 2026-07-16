//! Unit tests for event-counter rate thresholds and
//! `compute_event_deltas` edge cases (counter resets, single
//! samples, missing counters).
//! Co-located via the sibling `*_tests.rs` pattern.

#![cfg(test)]

use super::test_util::balanced_sample;
use super::*;

/// Build a sample with event counters. Each CPU gets the same counter
/// values so the total across CPUs = ncpus * per_cpu_value.
fn sample_with_events(
    elapsed_ms: u64,
    clock_base: u64,
    fallback: i64,
    keep_last: i64,
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
                vcpu_cpu_time_ns: Some(1_000_000_000 + elapsed_ms * 1_000_000),
                event_counters: Some(ScxEventCounters {
                    select_cpu_fallback: fallback,
                    dispatch_keep_last: keep_last,
                    ..Default::default()
                }),
                ..Default::default()
            },
            CpuSnapshot {
                nr_running: 2,
                rq_clock: clock_base + 100,
                vcpu_cpu_time_ns: Some(2_000_000_000 + elapsed_ms * 1_000_000),
                event_counters: Some(ScxEventCounters {
                    select_cpu_fallback: fallback,
                    dispatch_keep_last: keep_last,
                    ..Default::default()
                }),
                ..Default::default()
            },
        ],
    }
}

#[test]
fn event_rates_ignore_elapsed_wall_dilation() {
    let first = sample_with_events(0, 1_000, 0, 0);
    let mut undilated = sample_with_events(100, 2_000, 10, 5);
    let mut dilated = sample_with_events(10_000, 2_000, 10, 5);
    // Both runs received the same 100 ms on each vCPU. Only host wall elapsed
    // differs, modeling 100x preemption dilation.
    for (cpu, ns) in undilated
        .cpus
        .iter_mut()
        .zip([1_100_000_000, 2_100_000_000])
    {
        cpu.vcpu_cpu_time_ns = Some(ns);
    }
    for (cpu, ns) in dilated.cpus.iter_mut().zip([1_100_000_000, 2_100_000_000]) {
        cpu.vcpu_cpu_time_ns = Some(ns);
    }

    let a = MonitorSummary::from_samples(&[first.clone(), undilated])
        .event_deltas
        .expect("event deltas");
    let b = MonitorSummary::from_samples(&[first, dilated])
        .event_deltas
        .expect("event deltas");
    assert_eq!(a.fallback_rate, Some(200.0));
    assert_eq!(a.keep_last_rate, Some(100.0));
    assert_eq!(a.fallback_rate, b.fallback_rate);
    assert_eq!(a.keep_last_rate, b.keep_last_rate);
    assert_eq!(a.total_event_vcpu_sec, b.total_event_vcpu_sec);
}

#[test]
fn event_rates_require_every_vcpu_clock() {
    let first = sample_with_events(0, 1_000, 0, 0);
    let mut last = sample_with_events(100, 2_000, 10, 5);
    last.cpus[1].vcpu_cpu_time_ns = None;
    let deltas = MonitorSummary::from_samples(&[first, last])
        .event_deltas
        .expect("raw event deltas remain available");
    assert_eq!(deltas.total_fallback, 20);
    assert_eq!(deltas.fallback_rate, None);
    assert_eq!(deltas.keep_last_rate, None);
    assert_eq!(deltas.total_event_vcpu_sec, None);
}

#[test]
fn thresholds_fallback_rate_sustained_fails() {
    // sustained_samples=3, max_fallback_rate=10.0.
    // 100ms intervals, 2 CPUs. Each CPU increments fallback by 10
    // per sample -> delta = 20 total per interval / 0.1s = 200/s > 10.
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_fallback_rate: 10.0,
        fail_on_rq_clock_stuck: false,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..4)
        .map(|i| sample_with_events(i * 100, 1000 + i * 500, i as i64 * 10, 0))
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.is_fail());
    assert!(v.details.iter().any(|d| d.contains("fallback rate")));
}

#[test]
fn thresholds_fallback_rate_below_sustained_passes() {
    // 2 violating intervals then a clean one — below sustained=3.
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_fallback_rate: 10.0,
        fail_on_rq_clock_stuck: false,
        ..Default::default()
    };
    let mut samples: Vec<_> = (0..3)
        .map(|i| sample_with_events(i * 100, 1000 + i * 500, i as i64 * 10, 0))
        .collect();
    // 4th sample: same fallback as 3rd -> rate = 0.
    samples.push(sample_with_events(300, 2500, 20, 0));
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "2 violations < sustained=3: {:?}", v.details);
}

#[test]
fn thresholds_keep_last_rate_sustained_fails() {
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_keep_last_rate: 10.0,
        fail_on_rq_clock_stuck: false,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..4)
        .map(|i| sample_with_events(i * 100, 1000 + i * 500, 0, i as i64 * 10))
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.is_fail());
    assert!(v.details.iter().any(|d| d.contains("keep_last rate")));
}

#[test]
fn thresholds_keep_last_rate_below_sustained_passes() {
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_keep_last_rate: 10.0,
        fail_on_rq_clock_stuck: false,
        ..Default::default()
    };
    let mut samples: Vec<_> = (0..3)
        .map(|i| sample_with_events(i * 100, 1000 + i * 500, 0, i as i64 * 10))
        .collect();
    // Reset: same keep_last as previous -> rate = 0.
    samples.push(sample_with_events(300, 2500, 0, 20));
    let summary = MonitorSummary::from_samples(&samples);
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(v.passed, "2 violations < sustained=3: {:?}", v.details);
}

#[test]
fn thresholds_event_rate_interrupted_resets() {
    // 2 violating intervals, 1 clean, 2 violating — never reaches sustained=3.
    let t = MonitorThresholds {
        sustained_samples: 3,
        max_fallback_rate: 10.0,
        fail_on_rq_clock_stuck: false,
        ..Default::default()
    };
    let mut samples = Vec::new();
    // 3 samples = 2 intervals of high fallback rate.
    for i in 0..3u64 {
        samples.push(sample_with_events(
            i * 100,
            1000 + i * 500,
            i as i64 * 10,
            0,
        ));
    }
    // Clean interval: same fallback -> rate = 0.
    samples.push(sample_with_events(300, 2500, 20, 0));
    // 2 more samples = 2 intervals of high fallback rate (not 3).
    // Production sums select_cpu_fallback across the 2 CPUs, so the
    // interval 3->4 rate is (80-40)/0.1 = 400/s (per-CPU fallback goes
    // 20 -> 40), violating; interval 4->5 is also violating. That's 2
    // intervals, below sustained=3.
    for i in 0..2u64 {
        samples.push(sample_with_events(
            400 + i * 100,
            3000 + i * 500,
            30 + (i + 1) as i64 * 10,
            0,
        ));
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
        "interrupted rate violations should pass: {:?}",
        v.details
    );
}

#[test]
fn thresholds_no_event_counters_skips_rate_check() {
    // Samples without event counters should not trigger rate violations.
    let t = MonitorThresholds {
        sustained_samples: 1,
        max_fallback_rate: 0.0, // any rate would fail
        max_keep_last_rate: 0.0,
        fail_on_rq_clock_stuck: false,
        ..Default::default()
    };
    let samples: Vec<_> = (0..5)
        .map(|i| balanced_sample(i * 100, 1000 + i * 500))
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
        "no event counters should skip rate check: {:?}",
        v.details
    );
}

#[test]
fn thresholds_default_event_rate_values() {
    let t = MonitorThresholds::default();
    assert!((t.max_fallback_rate - 200.0).abs() < f64::EPSILON);
    assert!((t.max_keep_last_rate - 100.0).abs() < f64::EPSILON);
}

#[test]
fn summary_keep_last_rate_computed() {
    // 2 CPUs, each with keep_last incrementing by 5 per sample.
    // 3 samples over 200ms -> total delta = 2*10 = 20, rate = 20/0.2 = 100.
    let samples = vec![
        sample_with_events(0, 1000, 0, 0),
        sample_with_events(100, 1500, 0, 5),
        sample_with_events(200, 2000, 0, 10),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let deltas = summary.event_deltas.unwrap();
    assert!((deltas.keep_last_rate.unwrap() - 100.0).abs() < f64::EPSILON);
}

// -- compute_event_deltas edge cases --

#[test]
fn event_deltas_none_without_counters() {
    let samples = vec![balanced_sample(100, 1000), balanced_sample(200, 1500)];
    let summary = MonitorSummary::from_samples(&samples);
    assert!(summary.event_deltas.is_none());
}

#[test]
fn event_deltas_single_sample() {
    // Only one sample with events -> first == last, so no CPU-time denominator.
    let samples = vec![sample_with_events(100, 1000, 50, 25)];
    let summary = MonitorSummary::from_samples(&samples);
    let deltas = summary.event_deltas.unwrap();
    assert_eq!(deltas.fallback_rate, None);
    assert_eq!(deltas.keep_last_rate, None);
}

#[test]
fn event_deltas_max_fallback_burst() {
    // 3 samples: burst between samples 1 and 2.
    let samples = vec![
        sample_with_events(0, 1000, 0, 0),
        sample_with_events(100, 1500, 5, 0),
        sample_with_events(200, 2000, 100, 0),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let deltas = summary.event_deltas.unwrap();
    // Per-CPU: burst is (100-5)*2 = 190 across 2 CPUs.
    assert!(deltas.max_fallback_burst > 0);
}

#[test]
fn event_deltas_counter_reset_clamps_to_zero() {
    // A scheduler restart between samples resets the per-CPU
    // counters to smaller (or zero) values. The raw delta
    // `last - first` is then negative — which would flow through
    // as a negative fallback_rate / negative total. Clamp to zero
    // so the downstream rate is sane.
    //
    // Sample 0 at t=0ms has high counters (pre-restart).
    // Sample 1 at t=1000ms has low counters (post-restart).
    let samples = vec![
        sample_with_events(0, 1000, 1000, 500),
        sample_with_events(1000, 2000, 5, 2),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let deltas = summary.event_deltas.unwrap();
    assert!(
        deltas.total_fallback >= 0,
        "reset must not produce negative total_fallback, got {}",
        deltas.total_fallback
    );
    assert!(
        deltas.fallback_rate.unwrap() >= 0.0,
        "reset must not produce negative fallback_rate, got {}",
        deltas.fallback_rate.unwrap()
    );
    assert!(
        deltas.total_dispatch_keep_last >= 0,
        "reset must not produce negative keep_last total, got {}",
        deltas.total_dispatch_keep_last
    );
    assert!(
        deltas.keep_last_rate.unwrap() >= 0.0,
        "reset must not produce negative keep_last_rate, got {}",
        deltas.keep_last_rate.unwrap()
    );
}

#[test]
fn event_deltas_all_counters_computed() {
    let make = |elapsed_ms, fb, kl, dsq_off, exit, migdis| MonitorSample {
        bpf_map_fields: Vec::new(),
        prog_stats: None,
        psi_irq: None,
        elapsed_ms,
        cpus: vec![CpuSnapshot {
            nr_running: 1,
            rq_clock: elapsed_ms * 10,
            event_counters: Some(ScxEventCounters {
                select_cpu_fallback: fb,
                dispatch_local_dsq_offline: dsq_off,
                dispatch_keep_last: kl,
                enq_skip_exiting: exit,
                enq_skip_migration_disabled: migdis,
                ..Default::default()
            }),
            ..Default::default()
        }],
    };
    let samples = vec![
        make(100, 10, 20, 30, 40, 50),
        make(200, 110, 120, 130, 140, 150),
    ];
    let summary = MonitorSummary::from_samples(&samples);
    let d = summary.event_deltas.unwrap();
    assert_eq!(d.total_fallback, 100);
    assert_eq!(d.total_dispatch_keep_last, 100);
    assert_eq!(d.total_dispatch_offline, 100);
    assert_eq!(d.total_enq_skip_exiting, 100);
    assert_eq!(d.total_enq_skip_migration_disabled, 100);
}

#[test]
fn neg_fallback_rate_threshold_fires() {
    let t = MonitorThresholds {
        sustained_samples: 2,
        max_fallback_rate: 5.0,
        fail_on_rq_clock_stuck: false,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..3u64)
        .map(|i| sample_with_events(i * 100, 1000 + i * 500, i as i64 * 10, 0))
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    assert!(
        summary.event_deltas.is_some(),
        "event deltas must be computed"
    );
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "fallback rate must be caught");
    // Format: "fallback rate 200.0/vcpu-s exceeded threshold 5.0/vcpu-s ..."
    let detail = v
        .failure_details()
        .find(|d| d.contains("fallback rate"))
        .unwrap();
    assert!(
        detail.contains("/vcpu-s"),
        "must include rate unit: {detail}"
    );
    assert!(
        detail.contains("exceeded threshold"),
        "must state threshold: {detail}"
    );
    assert!(
        detail.contains("5.0/vcpu-s"),
        "must show threshold value: {detail}"
    );
    assert!(
        detail.contains("consecutive intervals"),
        "must show sustained count: {detail}"
    );
}

#[test]
fn neg_keep_last_rate_threshold_fires() {
    let t = MonitorThresholds {
        sustained_samples: 2,
        max_keep_last_rate: 5.0,
        fail_on_rq_clock_stuck: false,
        enforce: true,
        ..Default::default()
    };
    let samples: Vec<_> = (0..3u64)
        .map(|i| sample_with_events(i * 100, 1000 + i * 500, 0, i as i64 * 10))
        .collect();
    let summary = MonitorSummary::from_samples(&samples);
    assert!(summary.event_deltas.is_some());
    let report = MonitorReport {
        samples,
        summary,
        ..Default::default()
    };
    let v = t.evaluate(&report);
    assert!(!v.passed, "keep_last rate must be caught");
    // Format: "keep_last rate .../vcpu-s exceeded threshold 5.0/vcpu-s ..."
    let detail = v
        .failure_details()
        .find(|d| d.contains("keep_last rate"))
        .unwrap();
    assert!(
        detail.contains("/vcpu-s"),
        "must include rate unit: {detail}"
    );
    assert!(
        detail.contains("exceeded threshold"),
        "must state threshold: {detail}"
    );
    assert!(
        detail.contains("5.0/vcpu-s"),
        "must show threshold value: {detail}"
    );
}
