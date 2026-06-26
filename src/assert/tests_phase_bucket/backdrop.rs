//! Unit tests for the backdrop per-phase host fold:
//! [`phase_slice_to_cgroup_stats`] (one `PhaseSlice` -> per-cgroup
//! carrier), [`pool_phase_slice_stats`] (pool same-epoch slices across
//! workers), and [`expand_backdrop_phase_buckets`] (group a worker set's
//! slices into per-epoch `PhaseBucket`s, skipping the BASELINE / gap
//! sentinels). These pin the per-phase aggregation a backdrop worker
//! feeds the host — the per-phase backdrop-capture surface.

use super::*;
use crate::workload::PhaseSlice;

/// A fully-populated `PhaseSlice` (every field non-default so the mapping
/// is exercised, not masked by a zero); `epoch` tags the phase.
fn slice(epoch: u32) -> PhaseSlice {
    PhaseSlice {
        phase_epoch: epoch,
        cpus_used: [0usize, 1].into_iter().collect(),
        wake_latencies_ns: vec![1000, 2000],
        wake_sample_total: 5,
        run_delay_ns: 3000,
        off_cpu_ns: 200_000,
        wall_ns: 1_000_000,
        migration_count: 2,
        iterations: 100,
        schedstat_cpu_time_ns: 1_000_000,
        numa_pages: [(0usize, 100u64), (1, 50)].into_iter().collect(),
        vmstat_numa_pages_migrated: 10,
        max_gap_ms: 40,
        max_gap_cpu: 0,
        // Generic-fold tests; the schbench per-phase carrier is covered by its
        // own host-fold tests in layer 2.
        schbench: None,
    }
}

fn report_with_slices(slices: Vec<PhaseSlice>) -> WorkerReport {
    WorkerReport {
        phase_slices: slices,
        ..WorkerReport::default()
    }
}

// -- phase_slice_to_cgroup_stats (one slice -> carrier) --

#[test]
fn phase_slice_maps_one_slice() {
    let cg = phase_slice_to_cgroup_stats(&slice(1), None);
    assert_eq!(cg.num_workers, 1);
    assert!(!cg.stripped);
    // off-CPU% = 200_000 / 1_000_000 * 100 = 20.0
    assert_eq!(cg.off_cpu_pcts, vec![20.0]);
    // run_delays raw ns, one per worker (NOT divided).
    assert_eq!(cg.run_delays_ns, vec![3000]);
    // Coupled gap pair carried as-is.
    assert_eq!((cg.max_gap_ms, cg.max_gap_cpu), (40, 0));
    assert_eq!(cg.total_migrations, 2);
    assert_eq!(cg.total_iterations, 100);
    assert_eq!(cg.total_cpu_time_ns, 1_000_000);
    assert_eq!(cg.wake_sample_total, 5);
    assert_eq!(cg.wake_latencies_ns, vec![1000, 2000]);
    assert_eq!(cg.cpus_used, [0, 1].into_iter().collect());
    // numa total = 100 + 50; no expected_nodes => local 0.
    assert_eq!(cg.numa_pages_total, 150);
    assert_eq!(cg.numa_pages_local, 0);
    // vmstat migrated carried for a single slice (the pool MAXes it).
    assert_eq!(cg.cross_node_migrated, 10);
}

#[test]
fn phase_slice_off_cpu_pcts_empty_when_wall_zero() {
    let s = PhaseSlice {
        wall_ns: 0,
        off_cpu_ns: 0,
        ..slice(1)
    };
    // wall_ns == 0 => the worker never ran this phase => not measured =>
    // EMPTY (distinct from a measured zero).
    assert!(
        phase_slice_to_cgroup_stats(&s, None)
            .off_cpu_pcts
            .is_empty()
    );
}

#[test]
fn phase_slice_off_cpu_pcts_measured_zero_present() {
    let s = PhaseSlice {
        wall_ns: 5000,
        off_cpu_ns: 0,
        ..slice(1)
    };
    // wall measured, off_cpu 0 => a measured zero, NOT empty.
    assert_eq!(
        phase_slice_to_cgroup_stats(&s, None).off_cpu_pcts,
        vec![0.0]
    );
}

#[test]
fn phase_slice_numa_local_partition() {
    let s = PhaseSlice {
        numa_pages: [(0usize, 100u64), (1, 60), (2, 15)].into_iter().collect(),
        ..slice(1)
    };
    // No expected_nodes => local 0, total full.
    let none = phase_slice_to_cgroup_stats(&s, None);
    assert_eq!(none.numa_pages_local, 0);
    assert_eq!(none.numa_pages_total, 175);
    // expected {0,1} => local 100 + 60 = 160; total still 175.
    let nodes: BTreeSet<usize> = [0, 1].into_iter().collect();
    let some = phase_slice_to_cgroup_stats(&s, Some(&nodes));
    assert_eq!(some.numa_pages_local, 160);
    assert_eq!(some.numa_pages_total, 175);
}

// -- pool_phase_slice_stats (same-epoch slices across workers) --

#[test]
fn pool_phase_slice_stats_pools_three() {
    // Three slices, same epoch; vary gap / cross_node / off_cpu / cpu / wake.
    let a = PhaseSlice {
        max_gap_ms: 30,
        max_gap_cpu: 1,
        vmstat_numa_pages_migrated: 30,
        off_cpu_ns: 100_000,
        wall_ns: 1_000_000,
        cpus_used: [0usize].into_iter().collect(),
        wake_latencies_ns: vec![1000],
        ..slice(1)
    };
    let b = PhaseSlice {
        max_gap_ms: 90,
        max_gap_cpu: 3,
        vmstat_numa_pages_migrated: 20,
        off_cpu_ns: 200_000,
        wall_ns: 1_000_000,
        cpus_used: [1usize].into_iter().collect(),
        wake_latencies_ns: vec![2000],
        ..slice(1)
    };
    let c = PhaseSlice {
        max_gap_ms: 50,
        max_gap_cpu: 7,
        vmstat_numa_pages_migrated: 10,
        off_cpu_ns: 300_000,
        wall_ns: 1_000_000,
        cpus_used: [2usize].into_iter().collect(),
        wake_latencies_ns: vec![3000],
        ..slice(1)
    };
    let pool = pool_phase_slice_stats(&[&a, &b, &c], None);
    assert_eq!(pool.num_workers, 3);
    // counters SUM (each base slice has migration 2 / iters 100 / cpu 1M / wake_total 5).
    assert_eq!(pool.total_migrations, 6);
    assert_eq!(pool.total_iterations, 300);
    assert_eq!(pool.total_cpu_time_ns, 3_000_000);
    assert_eq!(pool.wake_sample_total, 15);
    // cross_node MAX (30,20,10) == 30, NOT summed (60).
    assert_eq!(pool.cross_node_migrated, 30);
    // argmax gap: the worst worker's (90, 3) pair, not (90, 7).
    assert_eq!((pool.max_gap_ms, pool.max_gap_cpu), (90, 3));
    // cpus_used union {0,1,2}.
    assert_eq!(pool.cpus_used, [0, 1, 2].into_iter().collect());
    // off_cpu_pcts: each worker's value pushed (10,20,30 in some order).
    let mut pcts = pool.off_cpu_pcts.clone();
    pcts.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert_eq!(pcts, vec![10.0, 20.0, 30.0]);
    // wake samples concatenated (≤ cap => value-for-value).
    let mut w = pool.wake_latencies_ns.clone();
    w.sort_unstable();
    assert_eq!(w, vec![1000, 2000, 3000]);
}

#[test]
fn pool_phase_slice_stats_numa_local_across_pool() {
    let a = PhaseSlice {
        numa_pages: [(0usize, 100u64)].into_iter().collect(),
        ..slice(1)
    };
    let b = PhaseSlice {
        numa_pages: [(1usize, 60u64), (2, 15)].into_iter().collect(),
        ..slice(1)
    };
    let nodes: BTreeSet<usize> = [0, 1].into_iter().collect();
    let pool = pool_phase_slice_stats(&[&a, &b], Some(&nodes));
    // local = 100 (node 0) + 60 (node 1) = 160; total = 175.
    assert_eq!(pool.numa_pages_local, 160);
    assert_eq!(pool.numa_pages_total, 175);
}

#[test]
fn pool_phase_slice_stats_empty_is_zero_worker() {
    let pool = pool_phase_slice_stats(&[], None);
    assert_eq!(pool.num_workers, 0);
    assert!(pool.wake_latencies_ns.is_empty());
    assert_eq!(pool.wake_sample_total, 0);
    assert!(pool.off_cpu_pcts.is_empty());
    assert!(pool.run_delays_ns.is_empty());
    assert_eq!(pool.total_migrations, 0);
    assert_eq!(pool.total_iterations, 0);
    assert_eq!(pool.total_cpu_time_ns, 0);
    assert_eq!(pool.cross_node_migrated, 0);
    assert_eq!((pool.max_gap_ms, pool.max_gap_cpu), (0, 0));
    assert!(pool.cpus_used.is_empty());
    assert!(!pool.stripped);
}

#[test]
fn pool_single_slice_equals_mapper() {
    let s = slice(1);
    // A one-element pool is value-identical to the mapper output.
    assert_eq!(
        pool_phase_slice_stats(&[&s], None),
        phase_slice_to_cgroup_stats(&s, None)
    );
}

// -- expand_backdrop_phase_buckets (grouping + per-epoch pooling) --

#[test]
fn expand_groups_by_epoch_skipping_sentinels() {
    let reports = vec![report_with_slices(vec![
        slice(0), // BASELINE -> skipped
        slice(1), // epoch 1, worker 1
        PhaseSlice {
            iterations: 999,
            ..slice(1)
        }, // epoch 1, worker 2
        slice(2), // epoch 2
        slice(u32::MAX), // inter-step gap -> skipped
    ])];
    let buckets = expand_backdrop_phase_buckets("cg_bg", &reports, None);
    // Only the two real epochs survive, in ascending order.
    let idxs: Vec<u16> = buckets.iter().map(|b| b.step_index).collect();
    assert_eq!(idxs, vec![1, 2]);
    // label via Phase::Display; window is the merge-neutral sentinel;
    // metrics empty; sample_count 0 (per_cgroup is the only payload).
    assert_eq!(
        buckets[0].label,
        crate::assert::Phase::from(1u16).to_string()
    );
    assert_eq!(
        buckets[1].label,
        crate::assert::Phase::from(2u16).to_string()
    );
    assert_eq!((buckets[0].start_ms, buckets[0].end_ms), (u64::MAX, 0));
    assert_eq!(buckets[0].sample_count, 0);
    assert!(buckets[0].metrics.is_empty());
    // per_cgroup keyed by name; epoch 1 pooled 2 slices, epoch 2 pooled 1.
    assert_eq!(buckets[0].per_cgroup["cg_bg"].num_workers, 2);
    assert_eq!(buckets[1].per_cgroup["cg_bg"].num_workers, 1);
}

#[test]
fn expand_no_real_epochs_yields_no_buckets() {
    // Only sentinel epochs -> empty.
    let reports = vec![report_with_slices(vec![slice(0), slice(u32::MAX)])];
    assert!(expand_backdrop_phase_buckets("cg_bg", &reports, None).is_empty());
    // No slices at all -> empty.
    let reports2 = vec![report_with_slices(vec![])];
    assert!(expand_backdrop_phase_buckets("cg_bg", &reports2, None).is_empty());
}
