//! Unit tests for the schbench per-phase host fold + derivation (layer 2):
//! - [`PhaseCgroupStats::merge`] pooling the `schbench` carrier across workers
//!   (histogram bucket-add + integer-sum of the run-delay raw pairs/loop_count).
//! - `derive_schbench_phase_metrics` re-deriving the per-phase percentile /
//!   run-delay-mean / loop-count scalars into `PhaseBucket::metrics`, with the
//!   pcount==0 / empty-histogram ABSENT guards and cross-cgroup pooling.
//! - the [`crate::stats::MetricKind::PerPhase`] registry entries.

use super::*;
use crate::assert::derive_schbench_phase_metrics;
use crate::stats::{MetricKind, metric_def};
use crate::workload::PhaseSlice;
use crate::workload::schbench::plat::{Pct, PlatStats};
use crate::workload::schbench::run::SchbenchPhaseStats;

/// A wakeup histogram: 99 samples at 10µs + 1 at 10000µs — the
/// `plat::percentiles_split_across_buckets` shape (p50/p90/p99 = 10, p99.9 in
/// the 10000µs tail), 100 samples total.
fn wakeup_hist() -> PlatStats {
    let mut h = PlatStats::default();
    for _ in 0..99 {
        h.add_lat(10);
    }
    h.add_lat(10_000);
    h
}

/// A per-phase schbench aggregate: a known 100-sample wakeup histogram, an
/// EMPTY request histogram (so request keys stay absent), msg run-delay
/// 50000ns over `msg_pc` schedules, worker 80000ns over `worker_pc`, and
/// `loop_count` cycles.
fn sps(loop_count: u64, msg_pc: u64, worker_pc: u64) -> SchbenchPhaseStats {
    SchbenchPhaseStats {
        wakeup: wakeup_hist(),
        request: PlatStats::default(),
        rps: PlatStats::default(),
        msg_run_delay_ns: 50_000,
        msg_pcount: msg_pc,
        worker_run_delay_ns: 80_000,
        worker_pcount: worker_pc,
        loop_count,
    }
}

fn slice_with(schbench: Option<SchbenchPhaseStats>) -> PhaseSlice {
    PhaseSlice {
        schbench,
        ..Default::default()
    }
}

#[test]
fn phase_cgroup_merge_pools_schbench_carrier() {
    // Two backdrop slices (same epoch, different workers) each carry a schbench
    // aggregate; phase_slice_to_cgroup_stats carries it through and merge pools.
    // Each worker also carries a per-phase rps histogram with a distinct extreme,
    // so merge's rps.combine is exercised on non-empty operands.
    let mut sa = sps(10, 5, 0);
    sa.rps.add_lat(700);
    let mut sb = sps(20, 3, 4);
    sb.rps.add_lat(900);
    let a = phase_slice_to_cgroup_stats(&slice_with(Some(sa)), None);
    let b = phase_slice_to_cgroup_stats(&slice_with(Some(sb)), None);
    let merged = PhaseCgroupStats::merge(a, b)
        .schbench
        .expect("both Some -> Some");
    // Histograms combine by bucket-count add: 100 + 100 = 200 wakeup samples.
    assert_eq!(merged.wakeup.sample_count(), 200);
    // Run-delay raw pairs + loop_count integer-add (population-weighted
    // sufficient statistics — divide once at the derivation, never average).
    assert_eq!(merged.msg_run_delay_ns, 100_000);
    assert_eq!(merged.msg_pcount, 8);
    assert_eq!(merged.worker_run_delay_ns, 160_000);
    assert_eq!(merged.worker_pcount, 4);
    assert_eq!(merged.loop_count, 30);
    // rps pools across the two workers: 1 + 1 = 2 samples, cross-operand extremes
    // preserved (min from a, max from b) — proves merge folds rps via combine,
    // not a drop or a sum.
    assert_eq!(merged.rps.sample_count(), 2);
    let r = merged.rps.percentiles();
    assert_eq!(r.min, 700, "merged rps min = lower operand's extreme");
    assert_eq!(r.max, 900, "merged rps max = higher operand's extreme");
}

#[test]
fn phase_cgroup_merge_schbench_option_or() {
    let some = || phase_slice_to_cgroup_stats(&slice_with(Some(sps(7, 1, 1))), None);
    let none = || phase_slice_to_cgroup_stats(&slice_with(None), None);
    assert!(
        PhaseCgroupStats::merge(some(), none()).schbench.is_some(),
        "Some+None -> Some"
    );
    assert!(
        PhaseCgroupStats::merge(none(), some()).schbench.is_some(),
        "None+Some -> Some"
    );
    assert!(
        PhaseCgroupStats::merge(none(), none()).schbench.is_none(),
        "None+None (non-schbench cgroup) -> None"
    );
}

#[test]
fn derive_writes_perphase_scalars_with_absent_guards() {
    let pc = PhaseCgroupStats {
        schbench: Some(sps(42, 5, 0)), // msg pcount 5, worker pcount 0
        ..Default::default()
    };
    let mut bucket = PhaseBucket::default();
    bucket.per_cgroup.insert("cg".to_string(), pc);
    derive_schbench_phase_metrics(std::slice::from_mut(&mut bucket));
    // Wakeup percentiles re-derived from the merged histogram (µs).
    assert_eq!(bucket.metrics.get("wakeup_p50_latency_us"), Some(&10.0));
    assert_eq!(bucket.metrics.get("wakeup_p99_latency_us"), Some(&10.0));
    assert!(*bucket.metrics.get("wakeup_p999_latency_us").unwrap() >= 10_000.0);
    // Wakeup min/max from the fixture (99×10µs + 1×10000µs): min=10, max=10000.
    assert_eq!(bucket.metrics.get("wakeup_min_latency_us"), Some(&10.0));
    assert_eq!(bucket.metrics.get("wakeup_max_latency_us"), Some(&10_000.0));
    // EMPTY request histogram -> request keys ABSENT (no false 0).
    assert!(!bucket.metrics.contains_key("request_p99_latency_us"));
    assert!(!bucket.metrics.contains_key("request_min_latency_us"));
    assert!(!bucket.metrics.contains_key("request_max_latency_us"));
    // EMPTY rps histogram (sps() default) -> rps keys ABSENT.
    assert!(!bucket.metrics.contains_key("rps_p20"));
    assert!(!bucket.metrics.contains_key("rps_min"));
    // msg sched-delay = 50000ns / 5 = 10000ns -> 10µs (sample-weighted, ns->µs).
    assert_eq!(bucket.metrics.get("sched_delay_msg_us"), Some(&10.0));
    // worker pcount == 0 -> ABSENT, not a false 0.
    assert!(!bucket.metrics.contains_key("sched_delay_worker_us"));
    // loop_count is always present for a schbench phase.
    assert_eq!(bucket.metrics.get("schbench_loop_count"), Some(&42.0));
}

#[test]
fn derive_is_noop_for_non_schbench_phase() {
    // A phase whose only cgroup carries no schbench aggregate gets no schbench
    // keys — distinct from a schbench phase with 0 cycles (which gets loop_count
    // 0). Exercises the `pooled is None -> continue` path.
    let mut bucket = PhaseBucket::default();
    bucket
        .per_cgroup
        .insert("cg".to_string(), PhaseCgroupStats::default());
    derive_schbench_phase_metrics(std::slice::from_mut(&mut bucket));
    assert!(
        bucket.metrics.is_empty(),
        "no schbench carrier -> no schbench keys"
    );
}

#[test]
fn derive_pools_schbench_across_cgroups() {
    // Two schbench cgroups in one phase: the flat metrics map holds one set, so
    // percentiles come from the POOLED histogram and the scalars pool. Pinning
    // that the derivation combines per_cgroup, never averages per-cgroup values.
    let mk = || PhaseCgroupStats {
        schbench: Some(sps(10, 5, 5)),
        ..Default::default()
    };
    let mut bucket = PhaseBucket::default();
    bucket.per_cgroup.insert("cg_a".to_string(), mk());
    bucket.per_cgroup.insert("cg_b".to_string(), mk());
    derive_schbench_phase_metrics(std::slice::from_mut(&mut bucket));
    // loop_count pools: 10 + 10 = 20.
    assert_eq!(bucket.metrics.get("schbench_loop_count"), Some(&20.0));
    // sched_delay_msg = Σrd/Σpc = (50000+50000)/(5+5) = 10000ns -> 10µs.
    assert_eq!(bucket.metrics.get("sched_delay_msg_us"), Some(&10.0));
    // Pooled wakeup is 200 samples, still p99 = 10µs (the 10µs bucket holds 198).
    assert_eq!(bucket.metrics.get("wakeup_p99_latency_us"), Some(&10.0));
}

#[test]
fn derive_rps_distribution_values_and_absent_guard() {
    // The per-phase RPS derivation reads the pooled rps PlatStats: p20/p50/p90 from
    // the PLIST_FOR_RPS slots, min/max from the histogram extremes. A spanning
    // distribution gives DISTINCT p20/p50/p90, so a transposed Pct->key mapping (e.g.
    // rps_p20 reading value_at(P90)) would change the asserted value and fail. Each
    // key is compared against the independently-computed percentile, so the assert is
    // robust to the histogram's bucket arithmetic.
    let mut rps = PlatStats::default();
    for _ in 0..20 {
        rps.add_lat(100);
    }
    for _ in 0..30 {
        rps.add_lat(2_000);
    }
    for _ in 0..50 {
        rps.add_lat(40_000);
    }
    let r = rps.percentiles();
    assert!(
        r.value_at(Pct::P20) < r.value_at(Pct::P90),
        "spanning distribution yields distinct percentiles (transpose-detecting)"
    );

    let mut carrier = sps(10, 5, 3);
    carrier.rps = rps;
    let pc = PhaseCgroupStats {
        schbench: Some(carrier),
        ..Default::default()
    };
    let mut bucket = PhaseBucket::default();
    bucket.per_cgroup.insert("cg".to_string(), pc);
    derive_schbench_phase_metrics(std::slice::from_mut(&mut bucket));
    // Each key reads its OWN percentile slot — pins the Pct->key wiring.
    assert_eq!(
        bucket.metrics.get("rps_p20"),
        Some(&(r.value_at(Pct::P20) as f64))
    );
    assert_eq!(
        bucket.metrics.get("rps_p50"),
        Some(&(r.value_at(Pct::P50) as f64))
    );
    assert_eq!(
        bucket.metrics.get("rps_p90"),
        Some(&(r.value_at(Pct::P90) as f64))
    );
    // min/max are the histogram extremes: smallest sample (100), largest (40000).
    // (The 0-as-unset min sentinel is pinned separately by plat.rs's
    // zero_sample_records_bucket_but_not_min.)
    assert_eq!(bucket.metrics.get("rps_min"), Some(&100.0));
    assert_eq!(bucket.metrics.get("rps_max"), Some(&40_000.0));
}

#[test]
fn derive_pools_rps_across_cgroups() {
    // Two cgroups with DISJOINT rps ranges: the derived rps_min comes from cg_a's low
    // range and rps_max from cg_b's high range, proving the rps histogram POOLS across
    // cgroups (combine) before the percentile derivation — not per-cgroup-averaged or
    // single-cgroup.
    let mk = |lo: u32, hi: u32| {
        let mut c = sps(10, 5, 5);
        c.rps.add_lat(lo);
        c.rps.add_lat(hi);
        PhaseCgroupStats {
            schbench: Some(c),
            ..Default::default()
        }
    };
    let mut bucket = PhaseBucket::default();
    bucket.per_cgroup.insert("cg_a".to_string(), mk(100, 200));
    bucket
        .per_cgroup
        .insert("cg_b".to_string(), mk(10_000, 20_000));
    derive_schbench_phase_metrics(std::slice::from_mut(&mut bucket));
    // Pooled extremes span BOTH cgroups -> min from cg_a, max from cg_b.
    assert_eq!(bucket.metrics.get("rps_min"), Some(&100.0));
    assert_eq!(bucket.metrics.get("rps_max"), Some(&20_000.0));
}

#[test]
fn derive_request_min_max_value_pins() {
    // request_min/max derivation mirrors wakeup's (q.min/q.max under
    // p.request.sample_count()>0); pin the values from a populated request histogram so
    // a transposed insert (request_min <- q.max) is caught. The wakeup pair is
    // value-pinned in derive_writes_perphase_scalars_with_absent_guards and the rps pair
    // in derive_rps_distribution_values_and_absent_guard; this closes the request gap.
    let mut carrier = sps(5, 1, 1);
    let mut req = PlatStats::default();
    for _ in 0..50 {
        req.add_lat(30);
    }
    req.add_lat(9_000);
    carrier.request = req;
    let pc = PhaseCgroupStats {
        schbench: Some(carrier),
        ..Default::default()
    };
    let mut bucket = PhaseBucket::default();
    bucket.per_cgroup.insert("cg".to_string(), pc);
    derive_schbench_phase_metrics(std::slice::from_mut(&mut bucket));
    // 50×30µs + 1×9000µs: min=30, max=9000.
    assert_eq!(bucket.metrics.get("request_min_latency_us"), Some(&30.0));
    assert_eq!(bucket.metrics.get("request_max_latency_us"), Some(&9_000.0));
}

#[test]
fn phase_cgroup_schbench_serde_roundtrips() {
    // The schbench carrier (incl. the [u32;4864] histograms via plat's serde
    // adapter) rides PhaseCgroupStats inside AssertResult; pin the roundtrip,
    // including the widened per-phase `rps` histogram field.
    let mut carrier = sps(42, 5, 3);
    let mut rps = PlatStats::default();
    for v in [700u32, 720, 740] {
        rps.add_lat(v);
    }
    carrier.rps = rps;
    let pc = PhaseCgroupStats {
        schbench: Some(carrier),
        ..Default::default()
    };
    let json = serde_json::to_string(&pc).expect("serialize PhaseCgroupStats");
    let back: PhaseCgroupStats = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, pc, "PhaseCgroupStats roundtrips incl. schbench");
    let s = back.schbench.expect("schbench survives the roundtrip");
    assert_eq!(s.wakeup.sample_count(), 100);
    assert_eq!(
        s.rps.sample_count(),
        3,
        "the widened per-phase rps field roundtrips"
    );
    assert_eq!(s.loop_count, 42);
    assert_eq!(s.msg_pcount, 5);
    assert_eq!(s.worker_pcount, 3);
}

#[test]
fn strip_phase_cgroup_samples_drops_schbench_carrier() {
    // The robustness fix: strip_phase_cgroup_samples must also null the
    // schbench carrier (a worker_count>1 schbench cgroup multiplies the ~19 KiB
    // histograms by carrier count), counting its samples into `dropped` and
    // flagging `stripped`. Dropping schbench loses only the per-phase
    // percentiles (-> ABSENT metric, a loud degradation), never the verdict.
    let mut carrier = sps(7, 1, 1); // wakeup 100 samples, request 0
    for v in [700u32, 720, 740] {
        carrier.rps.add_lat(v); // 3 per-phase rps samples (nulled with the carrier)
    }
    let pc = PhaseCgroupStats {
        schbench: Some(carrier),
        wake_latencies_ns: vec![1, 2], // generic samples too
        ..Default::default()
    };
    let mut bucket = PhaseBucket::default();
    bucket.per_cgroup.insert("cg".to_string(), pc);
    let mut r = crate::assert::AssertResult::pass();
    r.stats.phases = vec![bucket];
    let dropped = r.strip_phase_cgroup_samples();
    let cg = &r.stats.phases[0].per_cgroup["cg"];
    assert!(cg.schbench.is_none(), "schbench carrier dropped to None");
    assert!(cg.stripped, "carrier flagged stripped");
    // 2 generic wake + 100 schbench wakeup + 3 rps samples (request empty). The rps
    // term pins the strip tally counting the rps field symmetrically with
    // wakeup/request (the whole carrier is nulled, so all three must be counted).
    assert_eq!(
        dropped,
        2 + 100 + 3,
        "dropped counts generic + schbench wakeup + rps histogram samples"
    );
    assert!(r.is_pass(), "verdict preserved (no PASS->FAIL flip)");
}

#[test]
fn schbench_perphase_metrics_registered() {
    // Every derived key is a registered MetricKind::PerPhase (so it satisfies the
    // keys=MetricDef::name contract + carries polarity/threshold) and is_derived
    // (so the within-run reducers, the phase-bucket merge, and the cross-RUN ext
    // fold all skip it — the derivation pass is its sole producer).
    for key in [
        "wakeup_p50_latency_us",
        "wakeup_p90_latency_us",
        "wakeup_p99_latency_us",
        "wakeup_p999_latency_us",
        "wakeup_min_latency_us",
        "wakeup_max_latency_us",
        "request_p50_latency_us",
        "request_p90_latency_us",
        "request_p99_latency_us",
        "request_p999_latency_us",
        "request_min_latency_us",
        "request_max_latency_us",
        "sched_delay_msg_us",
        "sched_delay_worker_us",
        "schbench_loop_count",
        "rps_p20",
        "rps_p50",
        "rps_p90",
        "rps_min",
        "rps_max",
    ] {
        let def = metric_def(key).unwrap_or_else(|| panic!("{key} must be registered"));
        assert!(
            matches!(def.kind, MetricKind::PerPhase),
            "{key} kind is PerPhase"
        );
        assert!(def.kind.is_derived(), "{key} is is_derived");
    }
    use crate::test_support::Polarity;
    assert!(
        matches!(
            metric_def("wakeup_p99_latency_us").unwrap().polarity,
            Polarity::LowerBetter
        ),
        "latency is LowerBetter"
    );
    assert!(
        matches!(
            metric_def("sched_delay_worker_us").unwrap().polarity,
            Polarity::LowerBetter
        ),
        "sched-delay is LowerBetter"
    );
    assert!(
        matches!(
            metric_def("schbench_loop_count").unwrap().polarity,
            Polarity::HigherBetter
        ),
        "loop_count is HigherBetter"
    );
    // The new latency extremes are LowerBetter (a higher min/max latency is worse).
    assert!(
        matches!(
            metric_def("wakeup_min_latency_us").unwrap().polarity,
            Polarity::LowerBetter
        ),
        "latency min is LowerBetter"
    );
    assert!(
        matches!(
            metric_def("wakeup_max_latency_us").unwrap().polarity,
            Polarity::LowerBetter
        ),
        "latency max is LowerBetter"
    );
    // The 5 rps metrics are HigherBetter — the min/max INVERT the latency-extreme
    // polarity (a higher worst/best second rate is better).
    assert!(
        matches!(
            metric_def("rps_p50").unwrap().polarity,
            Polarity::HigherBetter
        ),
        "rps percentiles are HigherBetter"
    );
    assert!(
        matches!(
            metric_def("rps_min").unwrap().polarity,
            Polarity::HigherBetter
        ),
        "rps_min inverts latency-min polarity (HigherBetter)"
    );
    assert!(
        matches!(
            metric_def("rps_max").unwrap().polarity,
            Polarity::HigherBetter
        ),
        "rps_max inverts latency-max polarity (HigherBetter)"
    );
}
