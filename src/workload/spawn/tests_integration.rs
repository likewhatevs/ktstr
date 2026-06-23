//! Spawn-pipeline tests — integration group.

#![cfg(test)]
#![allow(unused_imports)]

use super::super::affinity::*;
use super::super::config::*;
use super::super::types::*;
use super::super::worker::*;
use super::testing::*;
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[test]
fn workload_config_default() {
    let c = WorkloadConfig::default();
    assert_eq!(c.num_workers, 1);
    assert!(matches!(c.work_type, WorkType::SpinWait));
    assert!(matches!(c.sched_policy, SchedPolicy::Normal));
    assert!(matches!(c.affinity, AffinityIntent::Inherit));
    // Default nice is None — the `worker_main` gate skips the
    // `setpriority(2)` call entirely so the worker inherits the
    // parent's nice value.
    assert_eq!(c.nice, None);
}
#[test]
fn workload_config_builder_setters_chain() {
    let cfg = WorkloadConfig::default()
        .workers(7)
        .work_type(WorkType::SpinWait)
        .sched_policy(SchedPolicy::Batch)
        .nice(5);
    assert_eq!(cfg.num_workers, 7);
    assert!(matches!(cfg.work_type, WorkType::SpinWait));
    assert!(matches!(cfg.sched_policy, SchedPolicy::Batch));
    assert_eq!(cfg.nice, Some(5));
}
#[test]
fn worker_report_serde_roundtrip() {
    let r = WorkerReport {
        tid: 42,
        work_units: 1000,
        cpu_time_ns: 5_000_000_000,
        wall_time_ns: 10_000_000_000,
        off_cpu_ns: 5_000_000_000,
        migration_count: 3,
        cpus_used: [0, 1, 2].into_iter().collect(),
        migrations: vec![Migration {
            at_ns: 100,
            from_cpu: 0,
            to_cpu: 1,
        }],
        max_gap_ms: 50,
        max_gap_cpu: 1,
        max_gap_at_ms: 500,
        wake_latencies_ns: vec![1000, 2000],
        wake_sample_total: 2,
        iteration_costs_ns: vec![3000, 4000, 5000],
        iteration_cost_sample_total: 3,
        iterations: 10,
        schedstat_run_delay_ns: 500_000,
        schedstat_run_count: 20,
        schedstat_cpu_time_ns: 4_000_000_000,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        // Non-default so the serde roundtrip proves the field
        // survives, not just that Default's value matches on
        // both sides.
        is_messenger: true,
        // Non-zero so the serde roundtrip proves group_idx
        // serializes/deserializes correctly. The composed
        // dispatch path tags reports with their group_idx; a
        // silent default-zero on serde would lose that tag.
        group_idx: 7,
        affinity_error: None,
        phase_slices: vec![PhaseSlice {
            phase_epoch: 1,
            cpus_used: [2usize].into_iter().collect(),
            wake_latencies_ns: vec![700, 900],
            wake_sample_total: 2,
            run_delay_ns: 12_000,
            off_cpu_ns: 3_000,
            wall_ns: 8_000,
            migration_count: 1,
            iterations: 64,
            schedstat_cpu_time_ns: 5_000,
            numa_pages: [(0usize, 7u64)].into_iter().collect(),
            vmstat_numa_pages_migrated: 2,
            max_gap_ms: 9,
            max_gap_cpu: 2,
        }],
    };
    let json = serde_json::to_string(&r).unwrap();
    let r2: WorkerReport = serde_json::from_str(&json).unwrap();
    assert_eq!(r.tid, r2.tid);
    assert_eq!(r.work_units, r2.work_units);
    assert_eq!(r.migration_count, r2.migration_count);
    assert_eq!(r.cpus_used, r2.cpus_used);
    assert_eq!(r.max_gap_ms, r2.max_gap_ms);
    assert_eq!(r.wake_sample_total, r2.wake_sample_total);
    assert_eq!(r.iteration_costs_ns, r2.iteration_costs_ns);
    assert_eq!(
        r.iteration_cost_sample_total,
        r2.iteration_cost_sample_total
    );
    assert_eq!(r.completed, r2.completed);
    assert_eq!(r.is_messenger, r2.is_messenger);
    assert_eq!(r.group_idx, r2.group_idx);
    // phase_slices survive the serde roundtrip in full: PhaseSlice derives
    // PartialEq, so this compares every field (not just a hand-picked
    // subset) and stays complete as the struct grows.
    assert_eq!(r.phase_slices, r2.phase_slices);
}
// -- Migration value-type coverage --------------------------------
//
// `Migration` is the wire format emitted by workers when they
// observe a CPU migration. It is `Copy` and intentionally omits a
// `Default` impl (a zeroed Migration is a self-migration, which is
// semantically invalid — see the type doc on `super::Migration`).
// The pins below cover: `::new(..)` constructor produces the same
// field layout as the direct struct-literal form; serde roundtrip
// across the three numeric fields; `::new` is `const fn` so it can
// seed `static` items; the type has NO `Default` impl.

// Compile-time pin: `Migration::new` is `pub const fn`. A regression
// dropping `const` would silently break const-context use.
const _: Migration = Migration::new(0, 1, 2);

// Future contributor: if you hit an AmbiguousIfImpl compile error
// here after adding `derive(Default)` (or a manual Default impl) on
// Migration, the rationale is in the doc comment on the `Migration`
// struct in `spawn::mod`
// (TL;DR: a zeroed Migration is `{at_ns: 0, from_cpu: 0, to_cpu: 0}`
// which is a self-migration where source == dest — NOT a real
// migration. Downstream analysis that assumes `from_cpu != to_cpu`
// would misread default values as real migrations). Construct every
// Migration explicitly via `Migration::new`.
assert_not_impl_default!(Migration);

// Future contributor: if you hit an AmbiguousIfImpl compile error
// here after adding `derive(Default)` (or a manual Default impl) on
// WorkerExitInfo, the rationale is in the doc comment on the
// `WorkerExitInfo` enum in `spawn::mod`
// (TL;DR: every variant carries observed-outcome state; a default
// would have to pick TimedOut, but a test using `..Default::default()`
// would get "worker never exited within the deadline" silently,
// which an operator triaging the failure would chase for minutes
// before realizing the value came from a missing field). Construct
// every WorkerExitInfo explicitly via the variant the scenario
// expects (e.g. `WorkerExitInfo::Exited(0)` for a clean success).
assert_not_impl_default!(WorkerExitInfo);

#[test]
fn migration_new_matches_struct_literal() {
    let from_ctor = Migration::new(1_000_000_000, 0, 1);
    let from_literal = Migration {
        at_ns: 1_000_000_000,
        from_cpu: 0,
        to_cpu: 1,
    };
    assert_eq!(from_ctor, from_literal);
    assert_eq!(from_ctor.at_ns, 1_000_000_000);
    assert_eq!(from_ctor.from_cpu, 0);
    assert_eq!(from_ctor.to_cpu, 1);
}

#[test]
fn migration_serde_roundtrip() {
    let original = Migration::new(42, 3, 7);
    let bytes = serde_json::to_vec(&original).expect("serialize");
    let restored: Migration = serde_json::from_slice(&bytes).expect("deserialize");
    assert_eq!(restored, original);
    assert_eq!(restored.at_ns, 42);
    assert_eq!(restored.from_cpu, 3);
    assert_eq!(restored.to_cpu, 7);
}
#[test]
fn spawn_start_collect_integration() {
    let config = WorkloadConfig {
        num_workers: 2,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    assert_eq!(h.worker_pids().len(), 2);
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(r.work_units > 0, "worker {} did no work", r.tid);
        assert!(r.wall_time_ns > 0);
        assert!(!r.cpus_used.is_empty());
    }
}

/// End-to-end producer -> host-fold for the per-phase backdrop capture,
/// WITHOUT a VM: spawn REAL backdrop workers, drive them
/// through phase epochs exactly as the scenario engine does
/// (set_phase_epoch at each Step boundary, the u32::MAX inter-step-gap
/// sentinel at the end), collect their REAL phase_slices, then run the same
/// host fold (expand_backdrop_phase_buckets) the None-arm of collect_handles
/// uses in production. The build_phase_slice / backdrop.rs unit tests cover
/// the math over HAND-BUILT slices; this is the only test that exercises the
/// worker-side state machine — the epoch poll, drain-on-change, the
/// 9-variable re-baseline, and the final drain — and proves its output
/// partitions the run and survives the host fold into per-cgroup buckets.
#[test]
fn backdrop_worker_phase_slices_partition_and_fold() {
    let config = WorkloadConfig {
        num_workers: 2,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    // BASELINE (epoch 0) work, then two real phases, then the inter-step
    // gap sentinel. CPU-bound SpinWait polls the epoch every ~1024 spins,
    // so 150ms per window is ample for both workers to observe each
    // transition and accumulate work in every phase.
    let dwell = Duration::from_millis(150);
    std::thread::sleep(dwell);
    h.set_phase_epoch(1);
    std::thread::sleep(dwell);
    h.set_phase_epoch(2);
    std::thread::sleep(dwell);
    h.set_phase_epoch(u32::MAX);
    std::thread::sleep(Duration::from_millis(40));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);

    for r in &reports {
        // The phase windows partition the whole run: every iteration the
        // worker did falls in exactly one phase, so the per-phase iteration
        // deltas sum EXACTLY to the whole-run total (phase_iterations_start
        // inits to 0 and WorkerReport.iterations is the same counter). A
        // broken re-baseline (leaked or double-counted delta) breaks this.
        let slice_iters: u64 = r.phase_slices.iter().map(|s| s.iterations).sum();
        assert_eq!(
            slice_iters, r.iterations,
            "phase-slice iterations must partition the whole-run total \
             (worker {}): slices sum {} vs report {}",
            r.tid, slice_iters, r.iterations,
        );
        // Both real phase epochs were observed. Every epoch change fires
        // drain-on-change for the phase just ended: 0->1 emits epoch 0,
        // 1->2 emits epoch 1, 2->u32::MAX emits epoch 2; the final drain
        // after the loop exits emits the still-open u32::MAX gap slice
        // (which the host discards).
        let real: BTreeSet<u32> = r
            .phase_slices
            .iter()
            .map(|s| s.phase_epoch)
            .filter(|&e| e != 0 && e != u32::MAX)
            .collect();
        assert!(
            real.contains(&1) && real.contains(&2),
            "worker {} did not emit both real phase epochs; saw {:?}",
            r.tid,
            r.phase_slices
                .iter()
                .map(|s| s.phase_epoch)
                .collect::<Vec<_>>(),
        );
        // Each real-phase slice measured a positive window, and off_cpu never
        // exceeds wall (the saturating off_cpu invariant, on live data).
        for s in r
            .phase_slices
            .iter()
            .filter(|s| s.phase_epoch == 1 || s.phase_epoch == 2)
        {
            assert!(
                s.wall_ns > 0,
                "worker {} epoch {} zero wall",
                r.tid,
                s.phase_epoch
            );
            assert!(
                s.off_cpu_ns <= s.wall_ns,
                "worker {} epoch {}: off_cpu {} > wall {}",
                r.tid,
                s.phase_epoch,
                s.off_cpu_ns,
                s.wall_ns,
            );
        }
    }

    // The production host fold (collect_handles None-arm) over the REAL
    // reports: epoch 0 (BASELINE) and u32::MAX (gap) are dropped; the two
    // real epochs become per-cgroup buckets at step_index 1 and 2, each
    // pooling both workers.
    let buckets = crate::assert::expand_backdrop_phase_buckets("cg_bg", &reports, None);
    let steps: BTreeSet<u16> = buckets.iter().map(|b| b.step_index).collect();
    assert_eq!(steps, [1u16, 2].into_iter().collect::<BTreeSet<u16>>());
    for b in &buckets {
        let cg = b.per_cgroup.get("cg_bg").expect("cg_bg carrier present");
        assert_eq!(
            cg.num_workers, 2,
            "step {} pools both workers",
            b.step_index
        );
        assert!(
            cg.total_iterations > 0,
            "step {} backdrop carrier has zero iterations",
            b.step_index,
        );
    }
}

#[test]
fn spawn_auto_start_on_collect() {
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let h = WorkloadHandle::spawn(&config).unwrap();
    // Don't call start() - collect should auto-start
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
}
#[test]
fn spawn_yield_heavy_produces_work() {
    let reports = spawn_and_collect_after(WorkType::YieldHeavy, 1, 200);
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
#[test]
fn spawn_mixed_produces_work() {
    let reports = spawn_and_collect_after(WorkType::Mixed, 1, 200);
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
/// Regression guard for the sign-cast bug: every pid returned
/// from `worker_pids()` must be a positive, live `pid_t` that
/// round-trips through `Pid::from_raw` + `kill(_, None)` (the
/// "exists" probe). A negative pid would silently broadcast
/// SIGKILL to a process group; a stale/reaped pid would fail the
/// probe with ESRCH. Either indicates storage upstream
/// re-introduced the u32 wraparound or dropped a child on the
/// floor.
#[test]
fn spawn_pids_fit_in_pid_t() {
    let config = WorkloadConfig {
        num_workers: 4,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let h = WorkloadHandle::spawn(&config).unwrap();
    for pid in h.worker_pids() {
        assert!(pid > 0, "child pid must be positive, got {pid}");
        // Signal 0 (None) only checks existence; it does not
        // deliver anything. Proves the pid is a real, live
        // process we can address — not a negative-cast bomb.
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None)
            .unwrap_or_else(|e| panic!("spawned child pid {pid} not addressable: {e}"));
    }
}
#[test]
fn spawn_multiple_workers_distinct_pids() {
    let config = WorkloadConfig {
        num_workers: 4,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    let pids = h.worker_pids();
    assert_eq!(pids.len(), 4);
    let unique: std::collections::HashSet<libc::pid_t> = pids.iter().copied().collect();
    assert_eq!(unique.len(), 4, "all worker PIDs should be distinct");
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(500));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 4);
}
/// Spawn-time affinity gate: every accepted variant resolves to
/// the matching [`ResolvedAffinity`] shape, every rejected variant
/// bails with an actionable diagnostic. Pins the gate's accept /
/// reject contract so adding a new [`AffinityIntent`] variant
/// forces a deliberate decision here.
#[test]
fn resolve_spawn_affinity_accepts_no_context_variants() {
    // Inherit -> ResolvedAffinity::None
    let r =
        GroupParams::resolve_spawn_affinity(&AffinityIntent::Inherit, "WorkloadConfig::affinity")
            .expect("Inherit must resolve");
    assert!(matches!(r, ResolvedAffinity::None));

    // Exact -> ResolvedAffinity::Fixed (set preserved)
    let r = GroupParams::resolve_spawn_affinity(
        &AffinityIntent::exact([0, 2, 4]),
        "WorkloadConfig::affinity",
    )
    .expect("Exact must resolve");
    match r {
        ResolvedAffinity::Fixed(set) => {
            assert_eq!(set.len(), 3);
            assert!(set.contains(&0) && set.contains(&2) && set.contains(&4));
        }
        other => panic!("expected Fixed, got {:?}", other),
    }

    // RandomSubset -> ResolvedAffinity::Random (pool + count preserved)
    let r = GroupParams::resolve_spawn_affinity(
        &AffinityIntent::random_subset([0usize, 1, 2, 3], 2),
        "WorkloadConfig::affinity",
    )
    .expect("RandomSubset must resolve");
    match r {
        ResolvedAffinity::Random { from, count } => {
            assert_eq!(from.len(), 4);
            assert_eq!(count, 2);
        }
        other => panic!("expected Random, got {:?}", other),
    }
}
#[test]
fn resolve_spawn_affinity_rejects_topology_variants() {
    for variant in [
        AffinityIntent::SingleCpu,
        AffinityIntent::LlcAligned,
        AffinityIntent::CrossCgroup,
        AffinityIntent::SmtSiblingPair,
    ] {
        let err = GroupParams::resolve_spawn_affinity(&variant, "WorkloadConfig::affinity")
            .expect_err("topology-aware variant must bail at gate");
        let msg = err.to_string();
        assert!(
            msg.contains("requires scenario"),
            "diagnostic must mention scenario context, got: {msg}"
        );
        assert!(
            msg.contains("WorkloadConfig::affinity"),
            "diagnostic must include site, got: {msg}"
        );
    }
}
/// Empty `Exact` would yield a zero-mask `sched_setaffinity` call
/// that the kernel rejects with EINVAL. The gate bails with an
/// actionable diagnostic pointing the caller at `Inherit`.
#[test]
fn resolve_spawn_affinity_rejects_empty_exact() {
    let err = GroupParams::resolve_spawn_affinity(
        &AffinityIntent::Exact(BTreeSet::new()),
        "WorkloadConfig::affinity",
    )
    .expect_err("empty Exact must bail at gate");
    let msg = err.to_string();
    assert!(
        msg.contains("empty CPU set"),
        "diagnostic must name the empty-set condition, got: {msg}"
    );
    assert!(
        msg.contains("Inherit"),
        "diagnostic must point caller at Inherit, got: {msg}"
    );
    assert!(
        msg.contains("WorkloadConfig::affinity"),
        "diagnostic must include site, got: {msg}"
    );
}
/// `RandomSubset` with an empty pool leaves the spawn-time gate
/// nothing to sample from. The gate bails rather than silently
/// resolving to no affinity.
#[test]
fn resolve_spawn_affinity_rejects_empty_random_pool() {
    let err = GroupParams::resolve_spawn_affinity(
        &AffinityIntent::RandomSubset {
            from: BTreeSet::new(),
            count: 2,
        },
        "WorkloadConfig::affinity",
    )
    .expect_err("empty RandomSubset pool must bail at gate");
    let msg = err.to_string();
    assert!(
        msg.contains("empty pool"),
        "diagnostic must name the empty-pool condition, got: {msg}"
    );
    assert!(
        msg.contains("Inherit"),
        "diagnostic must point caller at Inherit, got: {msg}"
    );
    assert!(
        msg.contains("WorkloadConfig::affinity"),
        "diagnostic must include site, got: {msg}"
    );
}
/// `RandomSubset { count: 0 }` would draw zero CPUs per worker —
/// equivalent to no constraint. The gate bails rather than
/// silently resolving to no affinity.
#[test]
fn resolve_spawn_affinity_rejects_zero_count_random() {
    let err = GroupParams::resolve_spawn_affinity(
        &AffinityIntent::RandomSubset {
            from: BTreeSet::from([0usize, 1, 2]),
            count: 0,
        },
        "WorkloadConfig::affinity",
    )
    .expect_err("RandomSubset count=0 must bail at gate");
    let msg = err.to_string();
    assert!(
        msg.contains("count=0"),
        "diagnostic must name the zero-count condition, got: {msg}"
    );
    assert!(
        msg.contains("Inherit"),
        "diagnostic must point caller at Inherit, got: {msg}"
    );
    assert!(
        msg.contains("WorkloadConfig::affinity"),
        "diagnostic must include site, got: {msg}"
    );
}
/// Direct `WorkloadHandle::spawn` rejects each topology-aware
/// variant the gate guards. Verifies the bail propagates through
/// the spawn pipeline and the error message identifies the
/// offending field.
#[test]
fn spawn_rejects_topology_aware_variants_at_primary() {
    for variant in [
        AffinityIntent::SingleCpu,
        AffinityIntent::LlcAligned,
        AffinityIntent::CrossCgroup,
        AffinityIntent::SmtSiblingPair,
    ] {
        let label = format!("{variant:?}");
        let config = WorkloadConfig::default()
            .work_type(WorkType::SpinWait)
            .affinity(variant);
        // WorkloadHandle does not impl Debug, so expect_err is
        // unavailable — match on the Result directly.
        let err = match WorkloadHandle::spawn(&config) {
            Ok(_) => panic!(
                "topology-aware variant {label} must reject at \
                 WorkloadHandle::spawn"
            ),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("WorkloadConfig::affinity"),
            "diagnostic must name the field for {label}, got: {msg}"
        );
        assert!(
            msg.contains("requires scenario"),
            "diagnostic must mention scenario context for {label}, got: {msg}"
        );
    }
}
/// Direct `WorkloadHandle::spawn` accepts `RandomSubset` because
/// the caller supplies the `from` pool, so the gate has every
/// resolved field it needs without scenario context. Each worker
/// gets an independent draw at spawn time; this test verifies the
/// resolved affinity falls inside the pool.
#[test]
fn spawn_accepts_random_subset_directly() {
    let pool: Vec<usize> = (0..2).collect();
    let config = WorkloadConfig::default()
        .work_type(WorkType::SpinWait)
        .workers(2)
        .affinity(AffinityIntent::random_subset(pool.iter().copied(), 1));
    let mut h = WorkloadHandle::spawn(&config).expect("RandomSubset with explicit pool must spawn");
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(
            !r.cpus_used.is_empty(),
            "RandomSubset worker must run somewhere"
        );
        for cpu in &r.cpus_used {
            assert!(
                pool.contains(cpu),
                "worker used CPU {cpu} outside pool {pool:?}"
            );
        }
    }
}
#[test]
fn spawn_io_sync_write_produces_work() {
    let reports = spawn_and_collect_after(WorkType::IoSyncWrite, 1, 200);
    assert_eq!(reports.len(), 1);
    assert!(
        reports[0].work_units > 0,
        "IoSyncWrite worker {} did no work",
        reports[0].tid
    );
}
#[test]
fn spawn_io_rand_read_produces_work() {
    let reports = spawn_and_collect_after(WorkType::IoRandRead, 1, 200);
    assert_eq!(reports.len(), 1);
    assert!(
        reports[0].work_units > 0,
        "IoRandRead worker {} did no work",
        reports[0].tid
    );
}
#[test]
fn spawn_io_convoy_produces_work() {
    let reports = spawn_and_collect_after(WorkType::IoConvoy, 1, 200);
    assert_eq!(reports.len(), 1);
    assert!(
        reports[0].work_units > 0,
        "IoConvoy worker {} did no work",
        reports[0].tid
    );
}
#[test]
fn spawn_bursty_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::Bursty {
            burst_duration: Duration::from_millis(50),
            sleep_duration: Duration::from_millis(50),
        },
        1,
        300,
    );
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
#[test]
fn spawn_pipeio_produces_work() {
    let reports = spawn_and_collect_after(WorkType::PipeIo { burst_iters: 1024 }, 2, 300);
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(r.work_units > 0, "PipeIo worker {} did no work", r.tid);
    }
}
#[test]
fn spawn_pipeio_odd_workers_fails() {
    let config = WorkloadConfig {
        num_workers: 3,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::PipeIo { burst_iters: 1024 },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let result = WorkloadHandle::spawn(&config);
    assert!(result.is_err(), "PipeIo with odd workers should fail");
    let msg = format!("{:#}", result.err().unwrap());
    assert!(
        msg.contains("divisible by 2"),
        "expected divisibility error: {msg}"
    );
}
// `spawn_zero_workers` removed: same root cause as the
// `snapshot_iterations_empty_handle` deletion below — `WorkloadHandle::
// spawn(&cfg)` rejects `num_workers = 0` via `WorkloadConfig::validate()`
// before any handle exists. The validate gate (`WorkloadConfig::validate`'s
// `num_workers > 0` check) has dedicated coverage in this module.

#[test]
fn worker_pids_count_matches_num_workers() {
    for n in [1, 3, 5] {
        let config = WorkloadConfig {
            num_workers: n,
            ..Default::default()
        };
        let h = WorkloadHandle::spawn(&config).unwrap();
        assert_eq!(
            h.worker_pids().len(),
            n,
            "worker_pids().len() should match num_workers={n}"
        );
        drop(h);
    }
}
#[test]
fn worker_report_serde_edge_cases() {
    // Empty migrations and cpus_used
    let r = WorkerReport {
        tid: 0,
        work_units: 0,
        cpu_time_ns: 0,
        wall_time_ns: 0,
        off_cpu_ns: 0,
        migration_count: 0,
        cpus_used: BTreeSet::new(),
        migrations: vec![],
        max_gap_ms: 0,
        max_gap_cpu: 0,
        max_gap_at_ms: 0,
        wake_latencies_ns: vec![],
        wake_sample_total: 0,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        iterations: 0,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
        affinity_error: None,
        phase_slices: vec![],
    };
    let json = serde_json::to_string(&r).unwrap();
    let r2: WorkerReport = serde_json::from_str(&json).unwrap();
    assert_eq!(r2.tid, 0);
    assert!(r2.cpus_used.is_empty());
    assert!(r2.migrations.is_empty());

    // Max u64 values
    let r = WorkerReport {
        tid: i32::MAX,
        work_units: u64::MAX,
        cpu_time_ns: u64::MAX,
        wall_time_ns: u64::MAX,
        off_cpu_ns: u64::MAX,
        migration_count: u64::MAX,
        cpus_used: [0, usize::MAX].into_iter().collect(),
        migrations: vec![],
        max_gap_ms: u64::MAX,
        max_gap_cpu: usize::MAX,
        max_gap_at_ms: u64::MAX,
        wake_latencies_ns: vec![],
        wake_sample_total: u64::MAX,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: u64::MAX,
        iterations: u64::MAX,
        schedstat_run_delay_ns: u64::MAX,
        schedstat_run_count: u64::MAX,
        schedstat_cpu_time_ns: u64::MAX,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: usize::MAX,
        affinity_error: None,
        phase_slices: vec![],
    };
    let json = serde_json::to_string(&r).unwrap();
    let r2: WorkerReport = serde_json::from_str(&json).unwrap();
    assert_eq!(r2.work_units, u64::MAX);
    assert_eq!(r2.tid, i32::MAX);
}
#[test]
fn spawn_pipeio_four_workers() {
    let config = WorkloadConfig {
        num_workers: 4,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::PipeIo { burst_iters: 512 },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    assert_eq!(h.worker_pids().len(), 4);
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(300));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 4);
    for r in &reports {
        assert!(
            r.work_units > 0,
            "PipeIo 4-worker worker {} did no work",
            r.tid
        );
    }
}
#[test]
fn workload_config_debug_shows_field_values() {
    let c = WorkloadConfig {
        num_workers: 7,
        affinity: AffinityIntent::Exact([3].into_iter().collect()),
        work_type: WorkType::YieldHeavy,
        sched_policy: SchedPolicy::Batch,
        ..Default::default()
    };
    let s = format!("{:?}", c);
    assert!(s.contains("7"), "must show num_workers value");
    assert!(s.contains("Exact"), "must show affinity variant");
    assert!(s.contains("3"), "must show affinity CPU set");
    assert!(s.contains("YieldHeavy"), "must show work_type variant");
    assert!(s.contains("Batch"), "must show sched_policy variant");
}
#[test]
fn migration_debug_shows_field_values() {
    let m = Migration {
        at_ns: 99999,
        from_cpu: 3,
        to_cpu: 7,
    };
    let s = format!("{:?}", m);
    assert!(s.contains("99999"), "must show at_ns value");
    assert!(s.contains("3"), "must show from_cpu value");
    assert!(s.contains("7"), "must show to_cpu value");
    let m2 = Migration {
        at_ns: 1,
        from_cpu: 0,
        to_cpu: 1,
    };
    let s2 = format!("{:?}", m2);
    assert_ne!(
        s, s2,
        "different field values must produce different debug output"
    );
}
#[test]
fn worker_report_debug_shows_field_values() {
    let r = WorkerReport {
        tid: 42,
        work_units: 12345,
        cpu_time_ns: 1000,
        wall_time_ns: 2000,
        off_cpu_ns: 1000,
        migration_count: 3,
        cpus_used: [0, 5].into_iter().collect(),
        migrations: vec![],
        max_gap_ms: 77,
        max_gap_cpu: 5,
        max_gap_at_ms: 500,
        wake_latencies_ns: vec![],
        wake_sample_total: 0,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        iterations: 0,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
        affinity_error: None,
        phase_slices: vec![],
    };
    let s = format!("{:?}", r);
    assert!(s.contains("42"), "must show tid value");
    assert!(s.contains("12345"), "must show work_units value");
    assert!(s.contains("77"), "must show max_gap_ms value");
    assert!(s.contains("5"), "must show max_gap_cpu value");
}
// -- WorkerReport edge cases --

#[test]
fn worker_report_off_cpu_ns_calculation() {
    // Drive the production derivation `derive_off_cpu_ns`
    // (worker/mod.rs) — the same function `worker_main` uses to
    // populate `WorkerReport::off_cpu_ns`. The prior version of this
    // test hand-built a WorkerReport literal and asserted three
    // self-chosen constants were arithmetically consistent, touching
    // no production code.
    //
    // Normal case: off_cpu = wall - cpu.
    assert_eq!(
        derive_off_cpu_ns(5_000_000_000, 3_000_000_000),
        2_000_000_000,
        "off_cpu_ns must be wall_time_ns - cpu_time_ns",
    );
    // Zero off-CPU: a worker pinned on-CPU for its whole life.
    assert_eq!(
        derive_off_cpu_ns(4_000_000_000, 4_000_000_000),
        0,
        "wall == cpu must yield zero off-CPU time",
    );
    // Saturating boundary: cpu_time_ns slightly exceeds wall_time_ns
    // (the two clocks — CLOCK_THREAD_CPUTIME_ID vs the monotonic
    // start.elapsed() — can skew). The derivation must clamp to 0,
    // NOT wrap to a near-u64::MAX phantom off-CPU figure. A
    // regression that swapped the saturating subtraction for a plain
    // `wall - cpu` would panic (overflow) in debug and wrap in
    // release — this case catches both.
    assert_eq!(
        derive_off_cpu_ns(3_000_000_000, 3_000_000_001),
        0,
        "cpu > wall (clock skew) must saturate to 0, not wrap",
    );
    assert_eq!(
        derive_off_cpu_ns(0, u64::MAX),
        0,
        "extreme cpu > wall must saturate to 0",
    );
}
#[test]
fn migration_serde_multiple() {
    let migrations = vec![
        Migration {
            at_ns: 100,
            from_cpu: 0,
            to_cpu: 1,
        },
        Migration {
            at_ns: 200,
            from_cpu: 1,
            to_cpu: 2,
        },
        Migration {
            at_ns: 300,
            from_cpu: 2,
            to_cpu: 0,
        },
    ];
    let json = serde_json::to_string(&migrations).unwrap();
    let m2: Vec<Migration> = serde_json::from_str(&json).unwrap();
    assert_eq!(m2.len(), 3);
    assert_eq!(m2[0].from_cpu, 0);
    assert_eq!(m2[2].to_cpu, 0);
}
// -- snapshot_iterations tests --
//
// `snapshot_iterations_empty_handle` was removed: `WorkloadHandle::
// spawn(&cfg)` rejects `num_workers = 0` via `WorkloadConfig::
// validate()` before any handle exists — so the "empty handle"
// (zero-workers) state is no longer reachable via the public API.
// The validate gate itself (`WorkloadConfig::validate`'s `num_workers > 0`
// check) is covered in this module.

#[test]
fn snapshot_iterations_running_workers() {
    let config = WorkloadConfig {
        num_workers: 2,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::SpinWait,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let iters = h.snapshot_iterations();
    assert_eq!(iters.len(), 2);
    // After 200ms of SpinWait, workers should have done iterations.
    for (i, &v) in iters.iter().enumerate() {
        assert!(v > 0, "worker {i} should have iterations > 0, got {v}");
    }
    drop(h);
}
#[test]
fn spawn_cache_pressure_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::CachePressure {
            size_kib: 32,
            stride: 64,
        },
        1,
        200,
    );
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
#[test]
fn spawn_cache_yield_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::CacheYield {
            size_kib: 32,
            stride: 64,
        },
        1,
        200,
    );
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
#[test]
fn spawn_cache_pipe_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::CachePipe {
            size_kib: 32,
            burst_iters: 1024,
        },
        2,
        300,
    );
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(r.work_units > 0, "CachePipe worker {} did no work", r.tid);
    }
}
#[test]
fn spawn_sequence_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::Sequence {
            first: WorkPhase::Spin(Duration::from_millis(10)),
            rest: vec![WorkPhase::Yield(Duration::from_millis(10))],
        },
        1,
        200,
    );
    assert_eq!(reports.len(), 1);
    assert!(reports[0].work_units > 0);
}
#[test]
fn spawn_custom_produces_work() {
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::custom("test_spin", custom_spin_fn),
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 1);
    assert!(
        reports[0].work_units > 0,
        "Custom worker did no work: work_units={}",
        reports[0].work_units
    );
    assert!(reports[0].wall_time_ns > 0);
    assert!(
        reports.iter().all(|r| r.completed),
        "every worker report on the live / non-sentinel path \
         must carry completed=true — pairs with the
         completed=false assertion in \
         stop_and_collect_reaps_grandchild_from_panicking_custom_closure",
    );
}
/// `CloneMode::Fork + WorkType::ForkExit` is the well-tested
/// pair (existing test
/// `stop_and_collect_reaps_grandchild_from_panicking_custom_closure`
/// pins the fork mode's panic shape). This regression guard
/// proves the new D5 incompatibility check does NOT also reject
/// the legitimate Fork+ForkExit combination.
#[test]
fn spawn_fork_with_forkexit_succeeds() {
    let config = WorkloadConfig {
        num_workers: 1,
        clone_mode: CloneMode::Fork,
        work_type: WorkType::ForkExit,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("Fork + ForkExit must remain valid");
    h.start();
    std::thread::sleep(std::time::Duration::from_millis(100));
    let _ = h.stop_and_collect();
}
/// Guards three invariants of [`WorkType::PageFaultChurn`]:
///
/// 1. Every spawned worker produces non-zero `work_units` and
///    `iterations` (sanity — holds under the pre-fix bug too,
///    so it's a basic progress check, not a regression guard).
/// 2. `iter_slot` (host-side iteration sampling, read via
///    [`WorkloadHandle::snapshot_iterations`]) ADVANCES during
///    the run. Asserted as a positive delta between two
///    snapshots taken at 100 ms and 250 ms. A delta is
///    insensitive to worker start-up latency (the test would
///    otherwise race against workers whose first outer iter
///    lands after the first snapshot). Pre-fix, PageFaultChurn
///    used an inner `while !STOP` loop that bypassed the
///    iter_slot publish in the outer `worker_main` loop, so
///    both snapshots were pinned at 0 and the delta would be 0.
/// 3. On multi-CPU hosts, at least one worker records ≥ 1
///    migration. With `num_workers = available_parallelism() + 1`
///    the workload oversubscribes by one, forcing at least one
///    context switch and CPU re-dispatch in any realistic
///    scheduler; combined with the migration check in the
///    outer `worker_main` loop (gated on
///    `work_units.is_multiple_of(1024)`) firing every 64 outer
///    iters for this test's parameters (touches_per_cycle=16 +
///    spin_iters=32 = 48 work_units/iter,
///    gcd(48, 1024) = 16, period = 1024/16 = 64; the default
///    16-iter period documented in
///    doc/guide/src/architecture/workers.md assumes
///    default params 256+64=320 instead), this puts the
///    assertion well above the flake threshold. Gated on
///    `available_parallelism() > 1` because single-CPU
///    sandboxes legitimately report 0 migrations.
#[test]
fn spawn_page_fault_churn_produces_work() {
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    // Oversubscribe by one to force CPU sharing even on fully
    // idle hosts, so the migration-count assertion below has
    // a reliable signal.
    let num_workers = num_cpus + 1;
    let config = WorkloadConfig {
        num_workers,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::PageFaultChurn {
            region_kib: 64,
            touches_per_cycle: 16,
            spin_iters: 32,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).unwrap();
    h.start();
    // Delta-based iter_slot assertion. Pre-fix these snapshots
    // were both 0 for PageFaultChurn (inner `while !STOP`
    // blocked the iter_slot publish in the outer `worker_main`
    // loop). Post-fix the outer loop updates iter_slot every
    // iteration. We snapshot at t=100 ms, then poll every 50 ms
    // up to a 5 s ceiling for the delta to clear 0 on every
    // worker — that's enough headroom for the oversubscribed
    // `num_cpus + 1` workers to each get CPU at least once even
    // under heavy nextest parallel load. The original 150 ms
    // hard wait flaked under contention; the regression this
    // test guards (inner-while bug → permanent iter_slot=0) is
    // unaffected by the longer ceiling because the polled
    // assertion still fires if ANY worker stays at 0
    // indefinitely.
    std::thread::sleep(std::time::Duration::from_millis(100));
    let snap1 = h.snapshot_iterations();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    let snap2 = loop {
        let snap = h.snapshot_iterations();
        let all_advanced = snap
            .iter()
            .zip(snap1.iter())
            .all(|(b, a)| b.saturating_sub(*a) > 0);
        if all_advanced {
            break snap;
        }
        if std::time::Instant::now() >= deadline {
            break snap;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), num_workers);
    assert_eq!(snap1.len(), num_workers);
    assert_eq!(snap2.len(), num_workers);
    for i in 0..num_workers {
        let delta = snap2[i].saturating_sub(snap1[i]);
        assert!(
            delta > 0,
            "worker {i} iter_slot delta within 5 s budget was 0 \
             (snap1={}, snap2={}); outer loop is not advancing, \
             indicating a regression that restores the \
             inner-`while !STOP` bug",
            snap1[i],
            snap2[i],
        );
    }
    // Basic progress sanity — holds even under the pre-fix
    // bug (inner loop still incremented work_units and
    // iterations), so this is not a regression guard for the
    // inner-while bug. Delta assertion above covers that.
    for r in &reports {
        assert!(
            r.work_units > 0,
            "PageFaultChurn worker {} did no work",
            r.tid
        );
        assert!(
            r.iterations > 0,
            "PageFaultChurn worker {} final iterations = 0",
            r.tid
        );
    }
    if num_cpus > 1 {
        let total_migrations: u64 = reports.iter().map(|r| r.migration_count).sum();
        assert!(
            total_migrations > 0,
            "expected ≥ 1 migration across {num_workers} \
             oversubscribed workers on {num_cpus}-cpu host; 0 \
             total migrations suggests the outer migration \
             check at work_units.is_multiple_of(1024) isn't \
             firing, indicating a regression that restores the \
             inner-`while !STOP` bug"
        );
    }
}
#[test]
fn spawn_mutex_contention_produces_work() {
    let reports = spawn_and_collect_after(
        WorkType::MutexContention {
            contenders: 4,
            hold_iters: 64,
            work_iters: 256,
        },
        4,
        500,
    );
    assert_eq!(reports.len(), 4);
    for r in &reports {
        assert!(
            r.work_units > 0,
            "MutexContention worker {} did no work",
            r.tid
        );
    }
}
#[test]
fn spawn_mutex_contention_bad_worker_count_fails() {
    let config = WorkloadConfig {
        num_workers: 3,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::MutexContention {
            contenders: 4,
            hold_iters: 256,
            work_iters: 1024,
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let result = WorkloadHandle::spawn(&config);
    assert!(result.is_err());
    let msg = format!("{:#}", result.err().unwrap());
    assert!(
        msg.contains("divisible by 4"),
        "expected divisibility error: {msg}"
    );
}
/// `WorkType::IpcVariance` spawn-side rejection mirrors the
/// constructor: a struct-literal with zero `hot_iters`
/// fails at [`WorkloadHandle::spawn`] with the typed error.
#[test]
fn ipc_variance_spawn_rejects_zero_hot_iters() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::IpcVariance {
            hot_iters: 0,
            cold_iters: 1,
            period_iters: 1,
        },
        ..Default::default()
    };
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("IpcVariance hot_iters=0 must be rejected at spawn");
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "hot_iters",
                group_idx: 0,
            }
        ),
        "expected ZeroIpcVarianceParam {{ hot_iters }} at spawn; got: {typed:?}",
    );
}
/// `WorkType::IpcVariance` spawn-side rejection mirrors the
/// constructor for `cold_iters`. Zero `cold_iters` would
/// produce a cold phase that does no memory work — the
/// scheduler-observable IPC variance the variant is named
/// for would not exist. The same `ZeroIpcVarianceParam`
/// variant fires from both the constructor and the spawn
/// gate.
#[test]
fn ipc_variance_spawn_rejects_zero_cold_iters() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::IpcVariance {
            hot_iters: 1,
            cold_iters: 0,
            period_iters: 1,
        },
        ..Default::default()
    };
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("IpcVariance cold_iters=0 must be rejected at spawn");
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "cold_iters",
                group_idx: 0,
            }
        ),
        "expected ZeroIpcVarianceParam {{ cold_iters }} at spawn; got: {typed:?}",
    );
}
/// `WorkType::IpcVariance` spawn-side rejection mirrors the
/// constructor for `period_iters`. Zero `period_iters`
/// would skip the inner loop entirely so the variant
/// produces no hot/cold alternation — the worker still
/// iterates the outer loop but performs no work. Pinning
/// the rejection prevents that silent degeneration.
#[test]
fn ipc_variance_spawn_rejects_zero_period_iters() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::IpcVariance {
            hot_iters: 1,
            cold_iters: 1,
            period_iters: 0,
        },
        ..Default::default()
    };
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("IpcVariance period_iters=0 must be rejected at spawn");
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "period_iters",
                group_idx: 0,
            }
        ),
        "expected ZeroIpcVarianceParam {{ period_iters }} at spawn; got: {typed:?}",
    );
}

/// Build a fully populated `WorkerReport` with non-default values
/// in every field. Anchoring tests on this shape proves the wire
/// format carries every byte the worker writes — a missing field on
/// either side would shift the positional postcard decoder onto the
/// next field's bytes (silent corruption per the doc on
/// `exit_info` / `affinity_error`).
fn fully_populated_report() -> WorkerReport {
    WorkerReport {
        tid: 12345,
        work_units: 7_777_777,
        cpu_time_ns: 3_141_592_653,
        wall_time_ns: 6_283_185_307,
        off_cpu_ns: 3_141_592_654,
        migration_count: 9,
        cpus_used: [0usize, 3, 5, 7].into_iter().collect(),
        migrations: vec![
            Migration {
                at_ns: 100,
                from_cpu: 0,
                to_cpu: 3,
            },
            Migration {
                at_ns: 250,
                from_cpu: 3,
                to_cpu: 5,
            },
        ],
        max_gap_ms: 42,
        max_gap_cpu: 5,
        max_gap_at_ms: 999,
        wake_latencies_ns: vec![1_000, 2_000, 3_000, 4_000],
        wake_sample_total: 4,
        iteration_costs_ns: vec![10, 20, 30],
        iteration_cost_sample_total: 3,
        iterations: 1024,
        schedstat_run_delay_ns: 555_000,
        schedstat_run_count: 73,
        schedstat_cpu_time_ns: 8_000_000_000,
        completed: true,
        numa_pages: [(0usize, 100u64), (1usize, 200u64)].into_iter().collect(),
        vmstat_numa_pages_migrated: 17,
        exit_info: None,
        is_messenger: true,
        group_idx: 4,
        affinity_error: None,
        phase_slices: vec![PhaseSlice {
            phase_epoch: 2,
            cpus_used: [1usize, 4, 6].into_iter().collect(),
            wake_latencies_ns: vec![500, 1500, 2500],
            wake_sample_total: 8,
            run_delay_ns: 444_000,
            off_cpu_ns: 1_234_567,
            wall_ns: 9_876_543,
            migration_count: 3,
            iterations: 256,
            schedstat_cpu_time_ns: 7_000_000,
            numa_pages: [(0usize, 11u64), (1, 22)].into_iter().collect(),
            vmstat_numa_pages_migrated: 13,
            max_gap_ms: 27,
            max_gap_cpu: 4,
        }],
    }
}

/// Compare two `WorkerReport`s field-by-field. `WorkerReport` does
/// not derive `PartialEq`, so the roundtrip tests must check every
/// field explicitly. A missing assertion would silently let a
/// mismatched field through — the same hazard the production
/// postcard pipe avoids by emitting every field on every call.
fn assert_worker_report_eq(a: &WorkerReport, b: &WorkerReport) {
    assert_eq!(a.tid, b.tid, "tid");
    assert_eq!(a.work_units, b.work_units, "work_units");
    assert_eq!(a.cpu_time_ns, b.cpu_time_ns, "cpu_time_ns");
    assert_eq!(a.wall_time_ns, b.wall_time_ns, "wall_time_ns");
    assert_eq!(a.off_cpu_ns, b.off_cpu_ns, "off_cpu_ns");
    assert_eq!(a.migration_count, b.migration_count, "migration_count");
    assert_eq!(a.cpus_used, b.cpus_used, "cpus_used");
    assert_eq!(a.migrations.len(), b.migrations.len(), "migrations.len");
    for (i, (am, bm)) in a.migrations.iter().zip(b.migrations.iter()).enumerate() {
        assert_eq!(am.at_ns, bm.at_ns, "migrations[{i}].at_ns");
        assert_eq!(am.from_cpu, bm.from_cpu, "migrations[{i}].from_cpu");
        assert_eq!(am.to_cpu, bm.to_cpu, "migrations[{i}].to_cpu");
    }
    assert_eq!(a.max_gap_ms, b.max_gap_ms, "max_gap_ms");
    assert_eq!(a.max_gap_cpu, b.max_gap_cpu, "max_gap_cpu");
    assert_eq!(a.max_gap_at_ms, b.max_gap_at_ms, "max_gap_at_ms");
    assert_eq!(
        a.wake_latencies_ns, b.wake_latencies_ns,
        "wake_latencies_ns"
    );
    assert_eq!(
        a.wake_sample_total, b.wake_sample_total,
        "wake_sample_total"
    );
    assert_eq!(
        a.iteration_costs_ns, b.iteration_costs_ns,
        "iteration_costs_ns"
    );
    assert_eq!(
        a.iteration_cost_sample_total, b.iteration_cost_sample_total,
        "iteration_cost_sample_total"
    );
    assert_eq!(a.iterations, b.iterations, "iterations");
    assert_eq!(
        a.schedstat_run_delay_ns, b.schedstat_run_delay_ns,
        "schedstat_run_delay_ns"
    );
    assert_eq!(
        a.schedstat_run_count, b.schedstat_run_count,
        "schedstat_run_count"
    );
    assert_eq!(
        a.schedstat_cpu_time_ns, b.schedstat_cpu_time_ns,
        "schedstat_cpu_time_ns"
    );
    assert_eq!(a.completed, b.completed, "completed");
    assert_eq!(a.numa_pages, b.numa_pages, "numa_pages");
    assert_eq!(
        a.vmstat_numa_pages_migrated, b.vmstat_numa_pages_migrated,
        "vmstat_numa_pages_migrated"
    );
    match (&a.exit_info, &b.exit_info) {
        (None, None) => {}
        (Some(WorkerExitInfo::Exited(x)), Some(WorkerExitInfo::Exited(y))) => {
            assert_eq!(x, y, "exit_info Exited code");
        }
        (Some(WorkerExitInfo::Signaled(x)), Some(WorkerExitInfo::Signaled(y))) => {
            assert_eq!(x, y, "exit_info Signaled signum");
        }
        (Some(WorkerExitInfo::TimedOut), Some(WorkerExitInfo::TimedOut)) => {}
        (Some(WorkerExitInfo::WaitFailed(x)), Some(WorkerExitInfo::WaitFailed(y))) => {
            assert_eq!(x, y, "exit_info WaitFailed message");
        }
        (Some(WorkerExitInfo::Panicked(x)), Some(WorkerExitInfo::Panicked(y))) => {
            assert_eq!(x, y, "exit_info Panicked message");
        }
        (other_a, other_b) => {
            panic!("exit_info variant mismatch: a={other_a:?} b={other_b:?}");
        }
    }
    assert_eq!(a.is_messenger, b.is_messenger, "is_messenger");
    assert_eq!(a.group_idx, b.group_idx, "group_idx");
    assert_eq!(a.affinity_error, b.affinity_error, "affinity_error");
    // PhaseSlice derives PartialEq, so a single assert_eq! compares every
    // field and self-maintains when a field is added — no hand-rolled
    // per-field list to fall out of sync. The backdrop per-phase telemetry
    // crosses the same postcard wire.
    assert_eq!(a.phase_slices, b.phase_slices, "phase_slices");
}

/// Roundtrip a fully populated `WorkerReport` (`exit_info=None`)
/// through `postcard::to_stdvec` → `postcard::from_bytes` —
/// the exact codec the worker→parent report pipe uses (mod.rs:
/// `to_stdvec` at the worker child, `from_bytes` at
/// `stop_and_collect`). Every field is asserted equal post-decode;
/// a missing field on either side would corrupt subsequent fields
/// silently per the `exit_info` / `affinity_error` doc warnings.
#[test]
fn worker_report_postcard_roundtrip() {
    let report = fully_populated_report();
    let bytes = postcard::to_stdvec(&report).expect("encode");
    let decoded: WorkerReport = postcard::from_bytes(&bytes).expect("decode");
    assert_worker_report_eq(&report, &decoded);
}

/// Roundtrip a sentinel-shaped `WorkerReport`: `exit_info =
/// Some(Exited(1))` (the catch_unwind panic-arm shape per
/// stop_and_collect's sentinel doc) and `affinity_error =
/// Some("EINVAL")` (the EINVAL-from-cpuset shape per the
/// `affinity_error` doc). Confirms the `Option<…>` tag bytes
/// round-trip through postcard without losing the inner payload —
/// the postcard positional encoding emits the tag whether or not
/// the option is populated, so a sentinel and a live-worker frame
/// must each decode with their original shape.
#[test]
fn worker_report_postcard_sentinel_roundtrip() {
    let mut report = fully_populated_report();
    report.exit_info = Some(WorkerExitInfo::Exited(1));
    report.affinity_error = Some("EINVAL".to_string());
    let bytes = postcard::to_stdvec(&report).expect("encode");
    let decoded: WorkerReport = postcard::from_bytes(&bytes).expect("decode");
    assert!(
        matches!(decoded.exit_info, Some(WorkerExitInfo::Exited(1))),
        "exit_info must roundtrip as Exited(1); got {:?}",
        decoded.exit_info
    );
    assert_eq!(decoded.affinity_error.as_deref(), Some("EINVAL"));
    assert_worker_report_eq(&report, &decoded);
}

/// Roundtrip-verify that `Vec<WorkerReport>` survives postcard
/// encode/decode, guarding the wire format for the fork-mode
/// report path which uses postcard. Pins per-element field
/// fidelity through the postcard codec across a multi-report
/// container.
#[test]
fn vec_worker_report_postcard_roundtrip() {
    let mut second = fully_populated_report();
    second.tid = 67890;
    second.group_idx = 5;
    second.is_messenger = false;
    second.exit_info = Some(WorkerExitInfo::Signaled(9));
    let reports: Vec<WorkerReport> = vec![fully_populated_report(), second];
    let bytes = postcard::to_stdvec(&reports).expect("encode");
    let decoded: Vec<WorkerReport> = postcard::from_bytes(&bytes).expect("decode");
    assert_eq!(decoded.len(), reports.len(), "vec length must roundtrip");
    for (i, (a, b)) in reports.iter().zip(decoded.iter()).enumerate() {
        assert_worker_report_eq(a, b);
        assert_eq!(a.tid, b.tid, "report[{i}] tid");
    }
}

/// Roundtrip every `WorkerExitInfo` variant — Exited, Signaled,
/// TimedOut, WaitFailed(String), Panicked(String) — through the
/// postcard codec. Each variant carries a distinct payload shape
/// (i32 / unit / String) and a serde-tagged enum encoding emits a
/// discriminant byte plus the inner payload; a missing variant on
/// either side would shift every downstream field per the
/// positional decoder, so the roundtrip must cover all five.
#[test]
fn worker_report_postcard_all_exit_info_variants_roundtrip() {
    let variants = [
        WorkerExitInfo::Exited(1),
        WorkerExitInfo::Signaled(9),
        WorkerExitInfo::TimedOut,
        WorkerExitInfo::WaitFailed("ECHILD".to_string()),
        WorkerExitInfo::Panicked("custom worker panicked".to_string()),
    ];
    for variant in variants {
        let mut report = fully_populated_report();
        report.exit_info = Some(variant);
        let bytes = postcard::to_stdvec(&report).expect("encode");
        let decoded: WorkerReport = postcard::from_bytes(&bytes).expect("decode");
        assert_worker_report_eq(&report, &decoded);
    }
}

/// Roundtrip a `WorkerReport::default()` shape through postcard.
/// Production sentinels are constructed via
/// `WorkerReport { ..WorkerReport::default() }` with select fields
/// overridden (mod.rs uses this shape at the catch_unwind arm,
/// pcomm-decode-failure arm, and pcomm-empty-payload arm). A
/// silent codec regression on the default shape would corrupt
/// every sentinel without surfacing in tests that only encode
/// fully-populated reports.
#[test]
fn worker_report_postcard_default_roundtrip() {
    let sentinel = WorkerReport::default();
    let bytes = postcard::to_stdvec(&sentinel).expect("encode");
    let decoded: WorkerReport = postcard::from_bytes(&bytes).expect("decode");
    assert_worker_report_eq(&sentinel, &decoded);
}

/// `postcard::from_bytes` rejects a truncated frame (first half of
/// a fully-populated `WorkerReport` encoding) with `Err`. Pins the
/// codec-level rejection only — does not exercise the parent's
/// sentinel-synthesis path, which is covered by dedicated
/// `stop_and_collect` tests.
#[test]
fn truncated_frame_decodes_to_err() {
    let report = fully_populated_report();
    let bytes = postcard::to_stdvec(&report).expect("encode");
    assert!(
        bytes.len() >= 2,
        "encoded report must be at least 2 bytes; got {}",
        bytes.len()
    );
    let truncated = &bytes[..bytes.len() / 2];
    let result: Result<WorkerReport, _> = postcard::from_bytes(truncated);
    assert!(
        result.is_err(),
        "truncated frame must decode to Err; got Ok({:?})",
        result.ok()
    );
}

/// `postcard::from_bytes` rejects an empty input slice with `Err`.
/// Pins the codec-level rejection only — does not exercise the
/// parent's sentinel-synthesis path, which is covered by dedicated
/// `stop_and_collect` tests.
#[test]
fn empty_payload_decodes_to_err() {
    let result: Result<WorkerReport, _> = postcard::from_bytes(&[]);
    assert!(
        result.is_err(),
        "empty payload must decode to Err; got Ok({:?})",
        result.ok()
    );
}
