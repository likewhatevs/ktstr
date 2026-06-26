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
use crate::workload::schbench::plat::PlatStats;
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
    let a = phase_slice_to_cgroup_stats(&slice_with(Some(sps(10, 5, 0))), None);
    let b = phase_slice_to_cgroup_stats(&slice_with(Some(sps(20, 3, 4))), None);
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
    // EMPTY request histogram -> request keys ABSENT (no false 0).
    assert!(!bucket.metrics.contains_key("request_p99_latency_us"));
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
fn phase_cgroup_schbench_serde_roundtrips() {
    // The schbench carrier (incl. the [u32;4864] histograms via plat's serde
    // adapter) rides PhaseCgroupStats inside AssertResult; pin the roundtrip.
    let pc = PhaseCgroupStats {
        schbench: Some(sps(42, 5, 3)),
        ..Default::default()
    };
    let json = serde_json::to_string(&pc).expect("serialize PhaseCgroupStats");
    let back: PhaseCgroupStats = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, pc, "PhaseCgroupStats roundtrips incl. schbench");
    let s = back.schbench.expect("schbench survives the roundtrip");
    assert_eq!(s.wakeup.sample_count(), 100);
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
    let pc = PhaseCgroupStats {
        schbench: Some(sps(7, 1, 1)),  // wakeup 100 samples, request 0
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
    // 2 generic wake samples + 100 schbench wakeup samples (request empty).
    assert_eq!(
        dropped,
        2 + 100,
        "dropped counts generic + schbench histogram samples"
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
        "request_p50_latency_us",
        "request_p90_latency_us",
        "request_p99_latency_us",
        "request_p999_latency_us",
        "sched_delay_msg_us",
        "sched_delay_worker_us",
        "schbench_loop_count",
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
}
