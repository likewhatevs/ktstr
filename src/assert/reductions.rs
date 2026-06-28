use super::*;

/// Check that workers only ran on CPUs in `expected`.
///
/// Any worker that used a CPU outside the expected set produces a
/// failure with the unexpected CPU IDs listed.
///
/// ```
/// # use ktstr::assert::assert_isolation;
/// # use ktstr::workload::WorkerReport;
/// # use std::collections::BTreeSet;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0, 1].into_iter().collect(),
/// #     work_units: 100, cpu_time_ns: 1_000_000, wall_time_ns: 2_000_000,
/// #     off_cpu_ns: 1_000_000, migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 0, max_gap_cpu: 0, max_gap_at_ms: 0,
/// #     wake_latencies_ns: vec![], wake_sample_total: 0,
/// #     iteration_costs_ns: vec![], iteration_cost_sample_total: 0,
/// #     iterations: 0,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     is_messenger: false,
/// #     ..Default::default()
/// # };
/// let expected: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
/// assert!(assert_isolation(&[report], &expected).is_pass());
/// ```
pub fn assert_isolation(reports: &[WorkerReport], expected: &BTreeSet<usize>) -> AssertResult {
    let mut r = AssertResult::pass();
    for w in reports {
        let bad: BTreeSet<usize> = w.cpus_used.difference(expected).copied().collect();
        if !bad.is_empty() {
            r.record_fail(AssertDetail::new(
                DetailKind::Isolation,
                format!("tid {} ran on unexpected CPUs {:?}", w.tid, bad),
            ));
        }
    }
    r
}

/// Nearest-rank percentile of a sorted slice (`p` in `[0.0, 1.0]`).
///
/// Returns the value at index `ceil(n * p) - 1`, clamped into
/// `[0, n-1]`. For `n = 100` and `p = 0.99` this is `sorted[98]` (the
/// 99th element in 1-indexed order), not `sorted[99]` (the max). The
/// previous formulation, `ceil(n * 0.99)` without the `-1`, was
/// off-by-one and returned the max for `n = 100`.
///
/// # Preconditions
///
/// `sorted` must be non-decreasing. The function indexes by rank
/// without checking order, so an unsorted input silently returns
/// the value at the computed index — a meaningless number. A
/// `debug_assert!` enforces this in debug builds; release builds
/// skip the check (the production callers sort immediately upstream
/// — `assert_not_starved` and `assert_benchmarks` both
/// `sorted.sort_unstable()` before this call — so the runtime
/// guard is unnecessary in production paths).
///
/// An empty slice yields `0` (the caller should short-circuit
/// before invoking).
pub(crate) fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    debug_assert!(
        sorted.windows(2).all(|w| w[0] <= w[1]),
        "percentile() requires sorted input; got slice with out-of-order pair",
    );
    let n = sorted.len();
    let idx = ((n as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    sorted[idx]
}

/// Build per-cgroup telemetry (pure measurement, no assertions) from
/// worker reports. This is the SINGLE telemetry builder on the assertion
/// path: `AssertPlan::assert_cgroup` calls it unconditionally and
/// [`assert_not_starved`] wraps it with the default fairness checks, so
/// per-cgroup [`CgroupStats`] is never gated behind whether a worker-check
/// assertion was configured. Empty `reports` yield a `num_workers == 0`
/// `CgroupStats` (the reduces below collapse to 0.0/0), so a declared
/// cgroup that collected no reports surfaces as a zero-worker entry rather
/// than silently vanishing from [`ScenarioStats::cgroups`].
pub fn cgroup_stats(reports: &[WorkerReport]) -> CgroupStats {
    let cpus: BTreeSet<usize> = reports
        .iter()
        .flat_map(|w| w.cpus_used.iter().copied())
        .collect();
    let pcts: Vec<f64> = reports
        .iter()
        .filter(|w| w.wall_time_ns > 0)
        .map(|w| w.off_cpu_ns as f64 / w.wall_time_ns as f64 * 100.0)
        .collect();

    // None when no worker had measurable wall time (pcts empty):
    // off-CPU% is undefined, and a not-measured cgroup must not read
    // as a measured 0% / spread-0 (perfectly fair) one. Some(_)
    // otherwise, including a real measured zero.
    let min = pcts.iter().cloned().reduce(f64::min);
    let max = pcts.iter().cloned().reduce(f64::max);
    let avg = if pcts.is_empty() {
        None
    } else {
        Some(pcts.iter().sum::<f64>() / pcts.len() as f64)
    };
    let spread = match (min, max) {
        (Some(lo), Some(hi)) => Some(hi - lo),
        _ => None,
    };

    let worst_gap = reports.iter().max_by_key(|w| w.max_gap_ms);
    let (gap_ms, gap_cpu) = worst_gap
        .map(|w| (w.max_gap_ms, w.max_gap_cpu))
        .unwrap_or((0, 0));

    // Compute benchmarking stats from worker reports.
    let all_latencies: Vec<u64> = reports
        .iter()
        .flat_map(|w| w.wake_latencies_ns.iter().copied())
        .collect();
    let (p99_us, median_us, lat_cv) = if all_latencies.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let mut sorted = all_latencies.clone();
        sorted.sort_unstable();
        let p99 = percentile(&sorted, 0.99) as f64 / 1000.0;
        // Median routes through `percentile(sorted, 0.5)` so the
        // nearest-rank algorithm matches every other percentile in
        // the project (p99, schbench's `lat99`, the BPF latency
        // histograms). A bare `sorted[n/2]` would pick the upper of
        // the two middle samples for even `n`, while `percentile`
        // returns the value at `ceil(n * 0.5) - 1` — the lower of
        // the two middles — and that lower-bound convention is what
        // the docs on [`CgroupStats::median_wake_latency_us`] and
        // the schbench cross-reference promise.
        let median = percentile(&sorted, 0.5) as f64 / 1000.0;
        let n = all_latencies.len() as f64;
        let mean_ns = all_latencies.iter().sum::<u64>() as f64 / n;
        let cv = if mean_ns > 0.0 {
            let variance = all_latencies
                .iter()
                .map(|&v| (v as f64 - mean_ns).powi(2))
                .sum::<f64>()
                / n;
            variance.sqrt() / mean_ns
        } else {
            0.0
        };
        (p99, median, cv)
    };

    // Timer-latency reductions (WorkType::TimerLatency), pooled across the
    // cgroup's workers exactly like the wake-latency block above but over the
    // distinct `timer_latencies_ns` reservoir. p999 is the deep RT tail; worst
    // is the single maximum. All `0.0` when no worker recorded timer samples.
    let all_timer: Vec<u64> = reports
        .iter()
        .flat_map(|w| w.timer_latencies_ns.iter().copied())
        .collect();
    let (median_timer, p99_timer, p999_timer, worst_timer) = if all_timer.is_empty() {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let mut sorted = all_timer.clone();
        sorted.sort_unstable();
        (
            percentile(&sorted, 0.5) as f64 / 1000.0,
            percentile(&sorted, 0.99) as f64 / 1000.0,
            percentile(&sorted, 0.999) as f64 / 1000.0,
            *sorted.last().expect("non-empty: checked above") as f64 / 1000.0,
        )
    };

    // saturating folds throughout this builder: pool the per-worker guest-runtime
    // counters (iterations, migrations, cpu-time-ns, sample totals, numa pages)
    // overflow-safely. This worker->cgroup fold is the FIRST cross-source sum and
    // the one most exposed to a corrupt/hostile WorkerReport; a u64::MAX component
    // clamps to u64::MAX instead of debug-panicking / release-wrapping a per-cgroup
    // total (and the derived migration-ratio / page-locality) to a silently-wrong
    // value. saturating_add is exact for every in-range value.
    let total_iters: u64 = reports
        .iter()
        .map(|w| w.iterations)
        .fold(0u64, u64::saturating_add);
    let run_delays: Vec<f64> = reports
        .iter()
        .map(|w| w.schedstat_run_delay_ns as f64 / 1000.0)
        .collect();
    let mean_run_delay = if run_delays.is_empty() {
        0.0
    } else {
        run_delays.iter().sum::<f64>() / run_delays.len() as f64
    };
    let worst_run_delay = run_delays.iter().cloned().reduce(f64::max).unwrap_or(0.0);

    let total_mig: u64 = reports
        .iter()
        .map(|w| w.migration_count)
        .fold(0u64, u64::saturating_add);
    let mig_ratio = migration_ratio_of(total_mig, total_iters);

    // Cross-node page-migration ratio: pages migrated cross-node over the
    // cgroup's total allocated pages. `vmstat_numa_pages_migrated` is a
    // system-wide delta each worker captured over its own loop; concurrent
    // workers observe overlapping deltas, so take the MAX across the cgroup
    // (summing would inflate by the worker count) over the cgroup-wide
    // total of allocated pages. Pure measurement — populated whenever NUMA
    // pages were seen, 0.0 otherwise. (The `max_cross_node_migration_ratio`
    // CHECK in `AssertPlan::assert_cgroup` recomputes the same raw counts
    // for its diagnostic; this is the always-on telemetry.)
    let total_numa_pages: u64 = reports
        .iter()
        .map(|w| w.numa_pages.values().copied().fold(0u64, u64::saturating_add))
        .fold(0u64, u64::saturating_add);
    let migrated_pages: u64 = reports
        .iter()
        .map(|w| w.vmstat_numa_pages_migrated)
        .max()
        .unwrap_or(0);
    let cross_node_ratio = cross_node_migration_ratio_of(migrated_pages, total_numa_pages);

    // Whole-run taobench engine aggregate, pooled across this cgroup's
    // `WorkType::Taobench` workers: Σ ops, MAX wall window (shared across the
    // concurrent workers, per `TaobenchStats::merge`). `None` for a cgroup
    // with no Taobench worker. The run-level cross-cgroup pool + the qps/hit
    // Rate derivation happen in `populate_run_pooled_taobench`; this is the
    // per-cgroup raw carrier, the taobench analogue of `total_iterations`.
    let taobench_whole = reports.iter().filter_map(|w| w.taobench_whole.as_ref()).fold(
        None,
        |acc: Option<crate::workload::taobench::run::TaobenchStats>, t| {
            Some(match acc {
                Some(mut a) => {
                    a.merge(t);
                    a
                }
                None => *t,
            })
        },
    );

    CgroupStats {
        // Empty here; collect_handles labels the entry post-hoc (it has
        // the cgroup name in scope, this reports-only builder does not).
        cgroup_name: String::new(),
        num_workers: reports.len(),
        num_cpus: cpus.len(),
        cpus_used: cpus,
        avg_off_cpu_pct: avg,
        min_off_cpu_pct: min,
        max_off_cpu_pct: max,
        spread,
        max_gap_ms: gap_ms,
        max_gap_cpu: gap_cpu,
        total_migrations: total_mig,
        migration_ratio: mig_ratio,
        p99_wake_latency_us: p99_us,
        median_wake_latency_us: median_us,
        wake_latency_cv: lat_cv,
        // `0.0` above is a not-measured sentinel iff no worker recorded a
        // wake/timer sample; the flag distinguishes that from a measured zero
        // so the run-level re-pool excludes (not folds in) a no-sample cgroup.
        wake_measured: !all_latencies.is_empty(),
        median_timer_latency_us: median_timer,
        p99_timer_latency_us: p99_timer,
        p999_timer_latency_us: p999_timer,
        worst_timer_latency_us: worst_timer,
        timer_measured: !all_timer.is_empty(),
        total_iterations: total_iters,
        total_cpu_time_ns: reports
            .iter()
            .map(|w| w.schedstat_cpu_time_ns)
            .fold(0u64, u64::saturating_add),
        mean_run_delay_us: mean_run_delay,
        worst_run_delay_us: worst_run_delay,
        // Run-delay is measured whenever a worker exists (a worker with `0.0`
        // delay is a real measured zero); only a worker-less cohort is
        // not-measured.
        run_delay_measured: !run_delays.is_empty(),
        // page_locality requires the expected NUMA node set (the cpuset's
        // nodes), which this reports-only builder does not have. It is
        // populated by `AssertPlan::assert_cgroup` when `numa_nodes` is
        // supplied; left 0.0 here (no NUMA context).
        page_locality: 0.0,
        cross_node_migration_ratio: cross_node_ratio,
        taobench_whole,
        ext_metrics: BTreeMap::new(),
    }
}

/// Migrations per iteration; `0.0` when no iterations ran (a measured-zero,
/// not a rate over zero). Single-sourced so the per-phase carrier
/// (`write_carrier_scalars`) and [`cgroup_stats`] (whose `migration_ratio`
/// feeds the run-level `worst_migration_ratio` fold) compute it identically.
pub(crate) fn migration_ratio_of(total_migrations: u64, total_iterations: u64) -> f64 {
    if total_iterations > 0 {
        total_migrations as f64 / total_iterations as f64
    } else {
        0.0
    }
}

/// Cross-node migrated pages over the cgroup-wide total allocated pages; `0.0`
/// when no NUMA pages were seen. Single-sourced with [`cgroup_stats`]. A CHURN
/// ratio, NOT a bounded `[0,1]` fraction: the numerator is cumulative migration
/// EVENTS (`/proc/vmstat numa_pages_migrated`, which counts each migration, so a
/// page can be counted more than once) and the denominator is a residency
/// SNAPSHOT — so the result can legitimately exceed 1.0 under heavy re-migration.
pub(crate) fn cross_node_migration_ratio_of(migrated_pages: u64, total_numa_pages: u64) -> f64 {
    if total_numa_pages > 0 {
        migrated_pages as f64 / total_numa_pages as f64
    } else {
        0.0
    }
}

/// Fraction of allocated pages on the expected NUMA nodes; `0.0` when no NUMA
/// pages were seen. Single-sourced with the per-phase carrier.
pub(crate) fn page_locality_of(numa_pages_local: u64, numa_pages_total: u64) -> f64 {
    if numa_pages_total > 0 {
        numa_pages_local as f64 / numa_pages_total as f64
    } else {
        0.0
    }
}

/// Iterations per worker; `None` when there are no workers (undefined,
/// distinct from a measured zero). Single-sourced with
/// [`CgroupStats::iterations_per_worker`](crate::assert::CgroupStats::iterations_per_worker)
/// so the run-level WorstLowest fold stays bit-identical.
pub(crate) fn iterations_per_worker_of(num_workers: usize, total_iterations: u64) -> Option<f64> {
    if num_workers > 0 {
        Some(total_iterations as f64 / num_workers as f64)
    } else {
        None
    }
}

/// Worker iterations per CPU-second (`total_iterations / (total_cpu_time_ns /
/// 1e9)`); `None` when there are no workers or no on-CPU time captured.
/// Single-sourced with
/// [`CgroupStats::iterations_per_cpu_sec`](crate::assert::CgroupStats::iterations_per_cpu_sec).
pub(crate) fn iterations_per_cpu_sec_of(
    num_workers: usize,
    total_cpu_time_ns: u64,
    total_iterations: u64,
) -> Option<f64> {
    if num_workers == 0 || total_cpu_time_ns == 0 {
        return None;
    }
    Some(total_iterations as f64 / (total_cpu_time_ns as f64 / 1e9))
}

/// Per-phase per-cgroup RAW-component builder — the sibling of [`cgroup_stats`]
/// that emits [`PhaseCgroupStats`]'s un-reduced components instead of the
/// reduced ratios/percentiles, so the distributional re-pool recomputes each
/// aggregate from the pooled components at every level. Every [`CgroupStats`]
/// reduction re-pools from these fields: avg/min/max/spread off-CPU% from
/// `off_cpu_pcts`; p99/median/CV from `wake_latencies_ns`; mean/worst run-delay
/// from `run_delays_ns` (RAW ns, the re-pool divides by 1000); migration_ratio
/// / iterations_per_cpu_sec / iterations_per_worker from the counters;
/// page_locality / cross_node_migration_ratio from the numa counters; the
/// coupled worst gap from the argmax pair; cpus_used / num_cpus from `cpus_used`.
///
/// `expected_nodes` is this cgroup's cpuset NUMA-node set (from
/// [`crate::topology::TestTopology::numa_nodes_for_cpuset`]); `numa_pages_local`
/// is the page count on those nodes (0 when `None`, mirroring [`cgroup_stats`]
/// leaving `page_locality` 0.0 without NUMA context — the partition lives with
/// the caller that has the node set, as [`AssertPlan::assert_cgroup`] does).
/// The whole-run [`cgroup_stats`] reductions stay the run-level authority; this
/// feeds the per-phase [`PhaseBucket::per_cgroup`] carrier.
///
/// RE-POOL GUARD CONTRACT: the div-by-zero / not-measured guards live in
/// [`cgroup_stats`], NOT in these raw components — a future re-pool over them
/// MUST mirror them exactly or ship a NaN/Inf or a not-measured-vs-zero
/// collapse: `migration_ratio` only when `total_iterations > 0`;
/// `cross_node_migration_ratio` / `page_locality` only when `numa_pages_total >
/// 0`; mean/worst run-delay only when `run_delays_ns` is non-empty; and
/// avg/min/max/spread off-CPU% return None (not 0.0) when `off_cpu_pcts` is
/// empty (the not-measured state).
pub(crate) fn phase_cgroup_stats(
    reports: &[WorkerReport],
    expected_nodes: Option<&BTreeSet<usize>>,
) -> PhaseCgroupStats {
    let cpus_used: BTreeSet<usize> = reports
        .iter()
        .flat_map(|w| w.cpus_used.iter().copied())
        .collect();
    // Per-worker off-CPU% (only workers with measurable wall time), un-reduced.
    // EMPTY = not measured: the re-pool then yields None for avg/min/max/spread,
    // preserving the not-measured-vs-measured-zero distinction cgroup_stats keeps.
    let off_cpu_pcts: Vec<f64> = reports
        .iter()
        .filter(|w| w.wall_time_ns > 0)
        .map(|w| w.off_cpu_ns as f64 / w.wall_time_ns as f64 * 100.0)
        .collect();
    // Pool every worker's already per-worker-capped wake-latency vec, RE-CAPPING
    // the concatenation at MAX_WAKE_SAMPLES via the same Algorithm-R reservoir the
    // per-worker path uses. This carrier is the FIRST to serialize raw samples
    // over the size-limited guest bulk port (the AssertResult); without the
    // re-cap the pool would be workers × MAX_WAKE_SAMPLES and could overrun the
    // 16 MiB frame on a many-core host, flipping a PASS to a truncated FAIL. The
    // reservoir is distribution-preserving, so p99 / median / CV re-pool over it
    // as cgroup_stats does over the per-worker pool; `wake_sample_total` keeps the
    // TRUE pre-cap population for the re-pool. PARITY: for pools ≤
    // MAX_WAKE_SAMPLES the reservoir is the full concatenation, so the re-pool is
    // VALUE-FOR-VALUE with cgroup_stats; above the cap it is a distribution-
    // preserving SUBSAMPLE (cgroup_stats keeps the full concat), so the re-pool is
    // distribution-equivalent, not byte-identical — see the wake_latencies_ns
    // field doc for the full contract. PhaseCgroupStats::merge keeps same-name
    // carriers bounded too: ≤cap it concatenates (value-for-value), >cap it uses a
    // population-WEIGHTED reservoir merge (weighted_merge_reservoirs) so the merged
    // subsample is unbiased rather than length-skewed toward the smaller carrier.
    let mut wake_latencies_ns: Vec<u64> = Vec::new();
    let mut pooled_wake_count: u64 = 0;
    for w in reports {
        for &sample in &w.wake_latencies_ns {
            crate::workload::reservoir_push(
                &mut wake_latencies_ns,
                &mut pooled_wake_count,
                sample,
                crate::workload::MAX_WAKE_SAMPLES,
            );
        }
    }
    // saturating folds (per cgroup_stats's rationale): overflow-safe pool of the
    // per-worker guest-runtime counters into this per-phase carrier.
    let wake_sample_total: u64 = reports
        .iter()
        .map(|w| w.wake_sample_total)
        .fold(0u64, u64::saturating_add);
    // Distinct timer reservoir, pooled across workers exactly like
    // wake_latencies_ns; timer_sample_total keeps the true pre-cap population.
    let mut timer_latencies_ns: Vec<u64> = Vec::new();
    let mut pooled_timer_count: u64 = 0;
    for w in reports {
        for &sample in &w.timer_latencies_ns {
            crate::workload::reservoir_push(
                &mut timer_latencies_ns,
                &mut pooled_timer_count,
                sample,
                crate::workload::MAX_WAKE_SAMPLES,
            );
        }
    }
    let timer_sample_total: u64 = reports
        .iter()
        .map(|w| w.timer_sample_total)
        .fold(0u64, u64::saturating_add);
    // RAW ns, one per worker — NOT divided by 1000. cgroup_stats divides at
    // reduction time; the re-pool over the concatenated samples divides once,
    // so pre-dividing here would double-divide (a 1000x error).
    let run_delays_ns: Vec<u64> = reports.iter().map(|w| w.schedstat_run_delay_ns).collect();
    // Coupled worst gap: take (ms, cpu) TOGETHER from the worst worker (argmax),
    // never two independent maxes — keeps the gap bound to its CPU.
    let (max_gap_ms, max_gap_cpu) = reports
        .iter()
        .max_by_key(|w| w.max_gap_ms)
        .map(|w| (w.max_gap_ms, w.max_gap_cpu))
        .unwrap_or((0, 0));
    let total_migrations: u64 = reports
        .iter()
        .map(|w| w.migration_count)
        .fold(0u64, u64::saturating_add);
    let total_iterations: u64 = reports
        .iter()
        .map(|w| w.iterations)
        .fold(0u64, u64::saturating_add);
    // schedstat_cpu_time_ns (task->se.sum_exec_runtime), NOT cpu_time_ns
    // (CLOCK_THREAD_CPUTIME_ID) — matches cgroup_stats's total_cpu_time_ns.
    let total_cpu_time_ns: u64 = reports
        .iter()
        .map(|w| w.schedstat_cpu_time_ns)
        .fold(0u64, u64::saturating_add);
    let numa_pages_total: u64 = reports
        .iter()
        .map(|w| w.numa_pages.values().copied().fold(0u64, u64::saturating_add))
        .fold(0u64, u64::saturating_add);
    // System-wide /proc/vmstat numa_pages_migrated delta each worker observes
    // redundantly -> MAX, not SUM (summing inflates by the worker count).
    let cross_node_migrated: u64 = reports
        .iter()
        .map(|w| w.vmstat_numa_pages_migrated)
        .max()
        .unwrap_or(0);
    // Pages on the cgroup's expected NUMA nodes (page_locality numerator),
    // partitioned exactly as AssertPlan::assert_cgroup does; 0 without a node
    // set (mirrors cgroup_stats leaving page_locality 0.0 absent NUMA context).
    let numa_pages_local: u64 = expected_nodes
        .map(|nodes| {
            let mut local = 0u64;
            for w in reports {
                for (&node, &count) in &w.numa_pages {
                    if nodes.contains(&node) {
                        local = local.saturating_add(count);
                    }
                }
            }
            local
        })
        .unwrap_or(0);
    PhaseCgroupStats {
        num_workers: reports.len(),
        cpus_used,
        wake_latencies_ns,
        wake_sample_total,
        timer_latencies_ns,
        timer_sample_total,
        run_delays_ns,
        off_cpu_pcts,
        total_migrations,
        total_iterations,
        total_cpu_time_ns,
        numa_pages_local,
        numa_pages_total,
        cross_node_migrated,
        max_gap_ms,
        max_gap_cpu,
        // Fresh carrier built from worker reports — never stripped.
        stripped: false,
        // Derived scalars are filled post-build by derive_phase_metrics.
        metrics: std::collections::BTreeMap::new(),
        // schbench rides PhaseSlice (the backdrop per-phase carrier), not the
        // whole-run WorkerReports this fn pools, so it is always None here.
        schbench: None,
        taobench: None,
    }
}

/// Build a per-cgroup carrier from ONE already-per-phase backdrop
/// [`crate::workload::PhaseSlice`] (no whole-run differencing — the
/// slice's counter fields are already per-phase deltas). The single-worker
/// analog of [`phase_cgroup_stats`]: `num_workers` is 1 and each list field
/// carries this one worker's value, so [`PhaseCgroupStats::merge`] pools
/// slices across backdrop workers into a per-epoch carrier identically to
/// how [`phase_cgroup_stats`] pools whole-run reports.
pub(crate) fn phase_slice_to_cgroup_stats(
    slice: &crate::workload::PhaseSlice,
    expected_nodes: Option<&BTreeSet<usize>>,
) -> PhaseCgroupStats {
    // Per-phase off-CPU%, one value, only when wall time was measured
    // (wall_ns == 0 => the worker never ran this phase => EMPTY = not
    // measured, matching phase_cgroup_stats's not-measured contract).
    let off_cpu_pcts: Vec<f64> = if slice.wall_ns > 0 {
        vec![slice.off_cpu_ns as f64 / slice.wall_ns as f64 * 100.0]
    } else {
        Vec::new()
    };
    // saturating folds: overflow-safe pool of this slice's per-node page counts.
    let numa_pages_total: u64 = slice
        .numa_pages
        .values()
        .copied()
        .fold(0u64, u64::saturating_add);
    let numa_pages_local: u64 = expected_nodes
        .map(|nodes| {
            slice
                .numa_pages
                .iter()
                .filter(|(node, _)| nodes.contains(node))
                .map(|(_, &count)| count)
                .fold(0u64, u64::saturating_add)
        })
        .unwrap_or(0);
    PhaseCgroupStats {
        num_workers: 1,
        cpus_used: slice.cpus_used.clone(),
        wake_latencies_ns: slice.wake_latencies_ns.clone(),
        wake_sample_total: slice.wake_sample_total,
        timer_latencies_ns: slice.timer_latencies_ns.clone(),
        timer_sample_total: slice.timer_sample_total,
        // RAW ns, one per worker (NOT divided) — same contract as
        // phase_cgroup_stats::run_delays_ns.
        run_delays_ns: vec![slice.run_delay_ns],
        off_cpu_pcts,
        total_migrations: slice.migration_count,
        total_iterations: slice.iterations,
        total_cpu_time_ns: slice.schedstat_cpu_time_ns,
        numa_pages_local,
        numa_pages_total,
        cross_node_migrated: slice.vmstat_numa_pages_migrated,
        max_gap_ms: slice.max_gap_ms,
        max_gap_cpu: slice.max_gap_cpu,
        stripped: false,
        // Derived scalars are filled post-build by derive_phase_metrics.
        metrics: std::collections::BTreeMap::new(),
        // Carry the per-phase schbench engine metrics through (None for every
        // non-schbench backdrop slice); PhaseCgroupStats::merge pools them.
        schbench: slice.schbench.clone(),
        // Carry the per-phase taobench engine metrics through (None for every
        // non-taobench backdrop slice); PhaseCgroupStats::merge pools them.
        taobench: slice.taobench,
    }
}

/// Pool a set of backdrop [`crate::workload::PhaseSlice`]s (all for
/// the SAME epoch, one per worker) into a single per-cgroup carrier via
/// [`PhaseCgroupStats::merge`] — the per-phase analog of
/// [`phase_cgroup_stats`] pooling whole-run reports. An empty input yields
/// a zero-worker carrier (all fields empty/0, `stripped: false`) so a phase
/// no backdrop worker observed renders as not-measured rather than
/// panicking.
pub(crate) fn pool_phase_slice_stats(
    slices: &[&crate::workload::PhaseSlice],
    expected_nodes: Option<&BTreeSet<usize>>,
) -> PhaseCgroupStats {
    let mut iter = slices
        .iter()
        .map(|s| phase_slice_to_cgroup_stats(s, expected_nodes));
    match iter.next() {
        Some(first) => iter.fold(first, PhaseCgroupStats::merge),
        None => PhaseCgroupStats {
            num_workers: 0,
            cpus_used: BTreeSet::new(),
            wake_latencies_ns: Vec::new(),
            wake_sample_total: 0,
            timer_latencies_ns: Vec::new(),
            timer_sample_total: 0,
            run_delays_ns: Vec::new(),
            off_cpu_pcts: Vec::new(),
            total_migrations: 0,
            total_iterations: 0,
            total_cpu_time_ns: 0,
            numa_pages_local: 0,
            numa_pages_total: 0,
            cross_node_migrated: 0,
            max_gap_ms: 0,
            max_gap_cpu: 0,
            stripped: false,
            metrics: std::collections::BTreeMap::new(),
            schbench: None,
            taobench: None,
        },
    }
}

/// Expand a backdrop worker set's per-phase
/// [`crate::workload::PhaseSlice`]s into one [`PhaseBucket`] per epoch,
/// keyed by the epoch as `step_index`, each pooling that epoch's slices
/// across workers via [`pool_phase_slice_stats`]. BASELINE (epoch 0) and
/// inter-step-gap (`u32::MAX`) epochs are skipped — they have no paired
/// host bucket and the host fold discards them. Called by the backdrop
/// (None-`step_index`) arm of [`crate::scenario::collect_handles`]; the
/// host's [`fold_guest_per_cgroup_into_host_buckets`] then unions these
/// into the host-rebuilt buckets (matched epochs) or surfaces them as
/// orphan not-measured windows. Extracted (rather than inlined in
/// collect_handles, which calls `stop_and_collect`) so the grouping +
/// per-epoch pooling is unit-testable directly.
pub(crate) fn expand_backdrop_phase_buckets(
    name: &str,
    reports: &[WorkerReport],
    expected_nodes: Option<&BTreeSet<usize>>,
) -> Vec<PhaseBucket> {
    let mut by_epoch: std::collections::BTreeMap<u32, Vec<&crate::workload::PhaseSlice>> =
        std::collections::BTreeMap::new();
    for report in reports {
        for slice in &report.phase_slices {
            if slice.phase_epoch == 0 || slice.phase_epoch == u32::MAX {
                continue;
            }
            by_epoch.entry(slice.phase_epoch).or_default().push(slice);
        }
    }
    by_epoch
        .into_iter()
        .map(|(epoch, slices)| {
            let mut per_cgroup = std::collections::BTreeMap::new();
            per_cgroup.insert(
                name.to_string(),
                pool_phase_slice_stats(&slices, expected_nodes),
            );
            // Lossless: a real epoch == u32::from(phase_step_index: u16),
            // and 0 / u32::MAX are filtered above.
            let step_index = epoch as u16;
            PhaseBucket {
                step_index,
                label: Phase::from(step_index).to_string(),
                start_ms: u64::MAX,
                end_ms: 0,
                sample_count: 0,
                metrics: std::collections::BTreeMap::new(),
                per_cgroup,
            }
        })
        .collect()
}

/// Build the single-bucket guest-side per-phase carrier for one step-local
/// cgroup: a [`PhaseBucket`] at `step_index` whose only payload is the
/// `per_cgroup` entry `name -> phase_cgroup_stats(reports, expected_nodes)`.
///
/// The guest emits one of these per step-local cgroup at `collect_step`
/// teardown ([`crate::scenario::collect_handles`]). The window is the
/// merge-neutral `(u64::MAX, 0)` sentinel and `metrics` is empty: the carrier
/// contributes ONLY `per_cgroup`. When folded into the host-rebuilt bucket of
/// the same `step_index` ([`fold_guest_per_cgroup_into_host_buckets`] via
/// [`merge_matched_phase_buckets`]) the `MAX`/`0` window is a no-op against the
/// host's real window (`min`/`max`), so the host's window and metrics win and
/// only `per_cgroup` is carried. The `label` uses [`Phase`]'s `Display` so an
/// orphan carrier (no host bucket) still reads `BASELINE`/`Step[k]`.
pub(crate) fn step_per_cgroup_bucket(
    name: &str,
    reports: &[WorkerReport],
    expected_nodes: Option<&BTreeSet<usize>>,
    step_index: u16,
) -> PhaseBucket {
    let mut per_cgroup = std::collections::BTreeMap::new();
    per_cgroup.insert(
        name.to_string(),
        phase_cgroup_stats(reports, expected_nodes),
    );
    PhaseBucket {
        step_index,
        label: Phase::from(step_index).to_string(),
        start_ms: u64::MAX,
        end_ms: 0,
        sample_count: 0,
        metrics: std::collections::BTreeMap::new(),
        per_cgroup,
    }
}

/// Roll a single cgroup's [`CgroupStats`] up into a one-cgroup
/// [`ScenarioStats`]. The KEPT typed `worst_*` fields carry this cgroup's
/// values and fold across cgroups in [`AssertResult::merge`] by max (all are
/// higher-is-worse): `worst_spread`, `worst_migration_ratio`, and the coupled
/// `worst_gap_ms` / `worst_gap_cpu`. The two NUMA roll-ups
/// (`worst_page_locality`, `worst_cross_node_migration_ratio`), the wake-latency
/// / run-delay distributions, the iteration efficiencies, and the wake-latency
/// tail ratio are NOT carried here — they have no typed field and re-pool
/// run-level POST-merge in [`populate_run_distribution_metrics`]:
/// `worst_page_locality` (WorstLowest) re-pools None-aware from
/// `stats.phases[].per_cgroup` NUMA carriers (lowest per-cgroup locality, a
/// measured 0.0 winning the lowest rather than being skipped as a sentinel),
/// `worst_cross_node_migration_ratio` (WorstCrossNodeRatio) re-pools the MAX
/// per-cgroup churn ratio from the same carriers, the tail ratio is the max over
/// the per-cgroup `CgroupStats::wake_latency_tail_ratio`, and the distributions /
/// efficiencies pool from `stats.phases[].per_cgroup` / `stats.cgroups`.
/// `cgroups` carries exactly this one entry so merge appends one per
/// handle without double-counting.
pub(crate) fn scenario_stats_for_cgroup(cg: &CgroupStats) -> ScenarioStats {
    ScenarioStats {
        total_workers: cg.num_workers,
        total_cpus: cg.num_cpus,
        total_migrations: cg.total_migrations,
        // worst_spread is higher-is-worse (merge takes max). A
        // not-measured cgroup (`spread == None`) maps to 0.0 — the
        // neutral element for max, and the gauntlet layer's
        // documented no-data convention. A measured zero stays 0.0.
        worst_spread: cg.spread.unwrap_or(0.0),
        worst_gap_ms: cg.max_gap_ms,
        worst_gap_cpu: cg.max_gap_cpu,
        worst_migration_ratio: cg.migration_ratio,
        total_iterations: cg.total_iterations,
        // worst_page_locality and worst_cross_node_migration_ratio are no longer
        // typed fields — both re-pool from the per-phase NUMA carriers in
        // populate_run_distribution_metrics. cg.page_locality is structurally 0.0
        // here (it needs an expected-node set this reports-only builder lacks);
        // cg.cross_node_migration_ratio IS populated but is a single pre-folded
        // scalar that cannot carry the cross-phase SUM-migrated / LATEST-total
        // fold, so the per-phase carriers own both.
        ext_metrics: cg.ext_metrics.clone(),
        cgroups: vec![cg.clone()],
        phases: Vec::new(),
    }
}

/// Record the DEFAULT fairness outcomes (Starved / Unfair / Stuck) for one
/// cgroup against the framework default thresholds
/// ([`spread_threshold_pct`] / [`gap_threshold_ms`]). Telemetry is built
/// separately by [`cgroup_stats`]; this only appends fail outcomes, so it
/// is shared by [`assert_not_starved`] and the `not_starved` arm of
/// [`AssertPlan::assert_cgroup`] without rebuilding stats.
pub(crate) fn record_default_fairness(
    r: &mut AssertResult,
    cg: &CgroupStats,
    reports: &[WorkerReport],
) {
    for w in reports {
        if w.work_units == 0 {
            r.record_fail(AssertDetail::new(
                DetailKind::Starved,
                format!("tid {} starved (0 work units)", w.tid),
            ));
        }
    }
    // Off-cpu spread above the default threshold, gated on >=2 workers
    // with measurable wall time (the historical `pcts.len() >= 2`).
    // `cg.spread` is None when off-CPU% was not measured — inconclusive,
    // never flagged unfair.
    let measurable = reports.iter().filter(|w| w.wall_time_ns > 0).count();
    let spread_limit = spread_threshold_pct();
    if let Some(spread) = cg.spread
        && spread > spread_limit
        && measurable >= 2
    {
        r.record_fail(AssertDetail::new(
            DetailKind::Unfair,
            format!(
                "unfair cgroup: spread={:.0}% ({:.0}-{:.0}%) {} workers on {} cpus (threshold {:.0}%)",
                spread,
                cg.min_off_cpu_pct.unwrap_or(0.0),
                cg.max_off_cpu_pct.unwrap_or(0.0),
                cg.num_workers,
                cg.num_cpus,
                spread_limit,
            ),
        ));
    }
    let gap_limit = gap_threshold_ms();
    for w in reports {
        if w.max_gap_ms > gap_limit {
            r.record_fail(AssertDetail::new(
                DetailKind::Stuck,
                format!(
                    "tid {} stuck {}ms on cpu{} at +{}ms (threshold {}ms)",
                    w.tid, w.max_gap_ms, w.max_gap_cpu, w.max_gap_at_ms, gap_limit,
                ),
            ));
        }
    }
}

/// Default fairness check for one cgroup's worker reports: builds the
/// per-cgroup telemetry ([`cgroup_stats`]) and records Starved / Unfair /
/// Stuck against the framework default thresholds. Telemetry is ALWAYS
/// populated — including a `num_workers == 0` entry for empty reports — so
/// `r.stats.cgroups` is never empty for a declared cgroup, independent of
/// whether any fail outcome fired.
pub fn assert_not_starved(reports: &[WorkerReport]) -> AssertResult {
    let cg = cgroup_stats(reports);
    let mut r = AssertResult::pass();
    record_default_fairness(&mut r, &cg, reports);
    r.stats = scenario_stats_for_cgroup(&cg);
    r
}

/// Check throughput parity across workers: coefficient of variation and
/// minimum work rate.
///
/// `max_cv`: maximum allowed coefficient of variation (stddev/mean) for
/// work_units / cpu_time_ns across workers. `None` skips the CV check.
///
/// `min_rate`: minimum work_units per CPU-second. `None` skips the floor check.
///
/// When every worker recorded `cpu_time_ns == 0`, both gates record
/// their OWN Inconclusive outcome (the CV gate emits a "CV cannot be
/// computed" detail; the min_rate gate emits a "rates cannot be
/// computed" detail). Each gate carries its own diagnostic so a
/// caller that supplies only one of the two threshold parameters
/// sees the matching Inconclusive message and an operator reading
/// [`AssertResult::inconclusive_details`] can identify which gate(s)
/// misfired without re-deriving the inputs.
///
/// ```
/// # use ktstr::assert::assert_throughput_parity;
/// # use ktstr::workload::WorkerReport;
/// # let mk = |units, cpu_ns| WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: units, cpu_time_ns: cpu_ns, wall_time_ns: cpu_ns,
/// #     off_cpu_ns: cpu_ns, migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 0, max_gap_cpu: 0, max_gap_at_ms: 0,
/// #     wake_latencies_ns: vec![], wake_sample_total: 0,
/// #     iteration_costs_ns: vec![], iteration_cost_sample_total: 0,
/// #     iterations: 0,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     is_messenger: false,
/// #     ..Default::default()
/// # };
/// // Equal throughput -> low CV -> passes.
/// let reports = [mk(1000, 1_000_000_000), mk(1000, 1_000_000_000)];
/// assert!(assert_throughput_parity(&reports, Some(0.5), None).is_pass());
/// ```
pub fn assert_throughput_parity(
    reports: &[WorkerReport],
    max_cv: Option<f64>,
    min_rate: Option<f64>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if reports.is_empty() {
        return r;
    }

    // Compute per-worker throughput: work_units / cpu_seconds
    let rates: Vec<f64> = reports
        .iter()
        .map(|w| {
            if w.cpu_time_ns == 0 {
                0.0
            } else {
                w.work_units as f64 / (w.cpu_time_ns as f64 / 1e9)
            }
        })
        .collect();

    // CV is computed over MEASURED workers only (cpu_time_ns > 0). A zero-cpu
    // worker's rate is unknowable — it is forced to 0.0 above so the min_rate
    // gate below can index `reports[i]`, but that 0.0 is NOT a measured rate.
    // Folding it into the mean/variance inflates the spread and FAILs a uniform
    // workload that merely had one worker record no CPU time — the same exclusion
    // the min_rate gate applies (it skips zero-cpu workers as "rate unknowable,
    // not failing").
    let measured: Vec<f64> = reports
        .iter()
        .zip(&rates)
        .filter(|(w, _)| w.cpu_time_ns > 0)
        .map(|(_, &rate)| rate)
        .collect();
    let mean = if measured.is_empty() {
        0.0
    } else {
        measured.iter().sum::<f64>() / measured.len() as f64
    };

    // Detect the all-zero-cpu condition once so a call with both
    // `max_cv` and `min_rate` set surfaces a single Inconclusive
    // listing every threshold that couldn't evaluate, rather than
    // emitting one record per gate (which produced duplicate
    // "denominator is zero" diagnostics for the same root cause).
    let all_zero_cpu = reports.iter().all(|w| w.cpu_time_ns == 0);

    if all_zero_cpu && (max_cv.is_some() || min_rate.is_some()) {
        let mut limits: Vec<String> = Vec::with_capacity(2);
        if let Some(cv_limit) = max_cv {
            limits.push(format!("max_cv {cv_limit:.3}"));
        }
        if let Some(floor) = min_rate {
            limits.push(format!("min_rate {floor:.0}"));
        }
        r.record_inconclusive(AssertDetail::new(
            DetailKind::Benchmark,
            format!(
                "throughput parity inconclusive: all {} workers recorded zero cpu_time_ns — \
                 denominator is zero, rates cannot be computed; {} neither pass nor fail \
                 (was the workload able to run?)",
                reports.len(),
                limits.join(" + "),
            ),
        ));
        return r;
    }

    if let Some(cv_limit) = max_cv
        && mean > 0.0
        && measured.len() >= 2
    {
        let n_measured = measured.len() as f64;
        let variance = measured.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n_measured;
        let stddev = variance.sqrt();
        let cv = stddev / mean;
        if cv > cv_limit {
            r.record_fail(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "throughput CV {cv:.3} exceeds limit {cv_limit:.3} (mean={mean:.0} work/cpu_s)"
                ),
            ));
        }
    }

    if let Some(floor) = min_rate {
        // Skip per-worker zero-cpu cases: their rate is forced to
        // 0.0 above, and comparing that to `floor` would synthesize
        // a guaranteed Fail with a misleading "below floor" message
        // when the real story is "this worker recorded no CPU time
        // — the rate is unknowable, not failing". The all-zero-cpu
        // case is already handled at the top of the function as a
        // single combined Inconclusive.
        for (i, &rate) in rates.iter().enumerate() {
            if reports[i].cpu_time_ns == 0 {
                continue;
            }
            if rate < floor {
                r.record_fail(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "worker {} throughput {rate:.0} work/cpu_s below floor {floor:.0}",
                        reports[i].tid
                    ),
                ));
            }
        }
    }

    r
}

/// Check benchmarking metrics: p99 wake latency, wake latency CV,
/// and minimum iteration rate.
///
/// ```
/// # use ktstr::assert::assert_benchmarks;
/// # use ktstr::workload::WorkerReport;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: 1000, cpu_time_ns: 2_500_000_000,
/// #     wall_time_ns: 5_000_000_000, off_cpu_ns: 2_500_000_000,
/// #     migration_count: 0, migrations: vec![],
/// #     max_gap_ms: 50, max_gap_cpu: 0, max_gap_at_ms: 1000,
/// #     wake_latencies_ns: vec![100, 200, 300, 400, 500],
/// #     wake_sample_total: 5,
/// #     iteration_costs_ns: vec![], iteration_cost_sample_total: 0,
/// #     iterations: 1000,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     is_messenger: false,
/// #     ..Default::default()
/// # };
/// // p99 = 500ns, well under 10000ns limit.
/// assert!(assert_benchmarks(&[report], Some(10000), None, None).is_pass());
/// ```
pub fn assert_benchmarks(
    reports: &[WorkerReport],
    max_p99_ns: Option<u64>,
    max_cv: Option<f64>,
    min_iter_rate: Option<f64>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if reports.is_empty() {
        // No worker reports means nothing to measure — any benchmark
        // threshold the caller supplied cannot be evaluated. A silent
        // pass would let thresholds look "green" on a broken run that
        // never produced signal; surface it as skip so the operator
        // knows the benchmark was not actually exercised.
        return AssertResult::skip("no worker reports — benchmark skipped");
    }

    // Collect all wake latencies across workers.
    let all_latencies: Vec<u64> = reports
        .iter()
        .flat_map(|w| w.wake_latencies_ns.iter().copied())
        .collect();

    if let Some(p99_limit) = max_p99_ns
        && !all_latencies.is_empty()
    {
        let mut sorted = all_latencies.clone();
        sorted.sort_unstable();
        let p99 = percentile(&sorted, 0.99);
        if p99 > p99_limit {
            r.record_fail(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "p99 wake latency {p99}ns exceeds limit {p99_limit}ns ({} samples)",
                    sorted.len()
                ),
            ));
        }
    }

    if let Some(cv_limit) = max_cv
        && all_latencies.len() >= 2
    {
        let n = all_latencies.len() as f64;
        let mean = all_latencies.iter().sum::<u64>() as f64 / n;
        if mean > 0.0 {
            let variance = all_latencies
                .iter()
                .map(|&v| (v as f64 - mean).powi(2))
                .sum::<f64>()
                / n;
            let cv = variance.sqrt() / mean;
            if cv > cv_limit {
                r.record_fail(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "wake latency CV {cv:.3} exceeds limit {cv_limit:.3} (mean={mean:.0}ns)"
                    ),
                ));
            }
        } else {
            // CV is dispersion / mean. With mean == 0 every captured
            // wake-latency sample was zero, so the denominator is
            // zero and CV is undefined — neither pass nor fail is
            // truthful. The same workload that fails to record
            // measurable wake latency at all (typically: nothing
            // actually woke, or every wake landed at <1ns and
            // truncated to zero in the ns counter) previously slid
            // past the gate as a silent pass; surface it as
            // Inconclusive so a broken benchmarking run does not
            // masquerade as a CV-compliant one.
            r.record_inconclusive(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "wake latency CV inconclusive: all {} sample(s) had zero mean wake \
                     latency — denominator is zero, CV cannot be computed; limit \
                     {cv_limit:.3} neither pass nor fail (did any wake event capture a \
                     non-zero latency?)",
                    all_latencies.len(),
                ),
            ));
        }
    }

    if let Some(rate_floor) = min_iter_rate {
        // Skip per-worker zero-wall cases (rate is unknowable when
        // wall_time_ns == 0) but count them: if every worker had
        // zero wall_time, the gate silently passed before — record
        // Inconclusive instead so a broken run that produced no
        // signal at all doesn't masquerade as a passing benchmark.
        let mut zero_wall_count = 0usize;
        for w in reports {
            if w.wall_time_ns == 0 {
                zero_wall_count += 1;
                continue;
            }
            let rate = w.iterations as f64 / (w.wall_time_ns as f64 / 1e9);
            if rate < rate_floor {
                r.record_fail(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "worker {} iteration rate {rate:.1}/s below floor {rate_floor:.1}/s",
                        w.tid
                    ),
                ));
            }
        }
        if zero_wall_count == reports.len() {
            r.record_inconclusive(AssertDetail::new(
                DetailKind::Benchmark,
                format!(
                    "min iteration rate inconclusive: all {} workers recorded zero wall_time_ns — \
                     denominator is zero, rate cannot be computed; floor {rate_floor:.1}/s \
                     neither pass nor fail (was the workload able to run?)",
                    reports.len()
                ),
            ));
        }
    }

    r
}

/// Assert that every SCX event counter in `events` is at or below
/// `max_count`. `events` is a slice of `(name, count)` pairs sourced
/// from the kernel's per-task `scx_event_stats` (see `kernel/sched/ext.c`,
/// `SCX_EV_*` macros) — typically aggregated and surfaced via
/// `monitor::ScxEventDeltas` or sidecar `GauntletRow.fallback_count` /
/// `keep_last_count` fields. Pass `None` for `max_count` to require zero
/// (the strict default — error-class events should not fire under a
/// healthy scheduler).
///
/// The assertion is decoupled from the `monitor` module on purpose:
/// callers harvest the counters they care about (via the live monitor
/// path or by reading sidecar JSON post-hoc) and feed name/count
/// pairs in. This keeps the assert API surface decoupled from the
/// kernel-side counter inventory, which evolves across kernel
/// versions — adding a new `SCX_EV_*` does not force an API change
/// here.
///
/// Returns a passing result if every counter is within bound; failures
/// concatenate one [`AssertDetail`] per offending counter under
/// [`DetailKind::SchedulerEvent`] so an operator can identify which
/// events fired without scanning the full counter set.
///
/// ```
/// # use ktstr::assert::assert_scx_events_clean;
/// // Strict default — every counter must be zero.
/// let r = assert_scx_events_clean(&[("enq_skip_exiting", 0), ("dispatch_local_dsq_offline", 0)], None);
/// assert!(r.is_pass());
///
/// // A non-zero error-class counter fails.
/// let r = assert_scx_events_clean(&[("enq_skip_exiting", 7)], None);
/// assert!(r.is_fail());
///
/// // Caller-supplied bound tolerates small counts.
/// let r = assert_scx_events_clean(&[("dispatch_keep_last", 3)], Some(10));
/// assert!(r.is_pass());
/// ```
pub fn assert_scx_events_clean(events: &[(&str, i64)], max_count: Option<i64>) -> AssertResult {
    let mut r = AssertResult::pass();
    for (name, count) in events {
        // Kernel `scx_event_stats` counters are monotonic u64 — a
        // negative i64 here means the source data is corrupted
        // (counter reset, wraparound on a signed conversion, or
        // sidecar JSON bit-loss). Treat negatives as failures rather
        // than letting them silently pass `*count > bound` for any
        // non-negative bound.
        let failed = match max_count {
            // Strict default: every counter must be exactly zero.
            // `*count > 0` would let -5 slip through.
            None => *count != 0,
            // Bounded: reject negatives explicitly, then enforce
            // the upper bound.
            Some(bound) => *count < 0 || *count > bound,
        };
        if failed {
            let bound_desc = match max_count {
                None => "0".to_string(),
                Some(b) => b.to_string(),
            };
            r.record_fail(AssertDetail::new(
                DetailKind::SchedulerEvent,
                format!("scx event `{name}` count {count} exceeds bound {bound_desc}",),
            ));
        }
    }
    r
}

/// Threshold-preset bundle for [`assert_thresholds`]. Captures the
/// guarantees a scheduler-under-test should meet on a healthy run:
/// wake latency stays within bound, per-iteration compute cost stays
/// within bound, CPU migrations stay within bound, and every worker
/// makes some forward progress.
///
/// Each `Option` field is independent — `None` skips that check. A
/// `AbsoluteThresholds` with every field `None` is a no-op (the
/// returned [`AssertResult`] always passes), useful as a starting
/// point for builder-style composition. Construct the all-`None`
/// thresholds via `AbsoluteThresholds::default()` and chain the
/// `max_*` / `min_*` setters (e.g. `AbsoluteThresholds::default().max_migrations(5)`)
/// or spread into a struct literal (`AbsoluteThresholds { max_migrations: Some(5), ..Default::default() }`).
/// Use [`Self::strict`] for the "every check enabled with sane defaults" preset.
///
/// Distinct from [`Assert`]: `Assert` is the merge-tree threshold
/// config consumed by the worker-side `AssertPlan`; `AbsoluteThresholds`
/// is a flat preset designed for direct invocation in test bodies
/// where the test author wants a one-call multi-field check without
/// engaging the merge chain. The two surfaces compose — a test can
/// run `assert_thresholds` against a worker-report slice AND merge the
/// `Assert`-derived result into the same accumulator via
/// [`AssertResult::merge`].
#[must_use = "AbsoluteThresholds only takes effect when passed to assert_thresholds"]
#[derive(Debug, Clone, Copy, Default)]
pub struct AbsoluteThresholds {
    /// Maximum acceptable p99 wake latency (nanoseconds). Compared
    /// against the pooled p99 across every worker's
    /// [`WorkerReport::wake_latencies_ns`]. `None` skips the check.
    /// Same units / semantics as [`Assert::max_p99_wake_latency_ns`].
    pub max_p99_wake_latency_ns: Option<u64>,
    /// Maximum acceptable p99 per-iteration compute cost (nanoseconds).
    /// Compared against the pooled p99 across every worker's
    /// [`WorkerReport::iteration_costs_ns`]. `None` skips the check.
    /// Only meaningful for compute work types that populate the
    /// reservoir (`AluHot`, `SmtSiblingSpin`, `IpcVariance`); blocking
    /// variants report empty `iteration_costs_ns` and the check is a
    /// no-op for those.
    pub max_iteration_cost_p99_ns: Option<u64>,
    /// Maximum acceptable total CPU migrations across every worker.
    /// Compared against the sum of [`WorkerReport::migration_count`].
    /// `None` skips the check. Distinct from
    /// [`Assert::max_migration_ratio`] (migrations per iteration) —
    /// this is an absolute count, useful when the test pins a known
    /// workload size and migrations should stay below a fixed ceiling
    /// regardless of how many iterations completed.
    pub max_migrations: Option<u64>,
    /// Minimum acceptable per-worker work_units. Every worker must
    /// have completed at least this many work units; one starved
    /// worker fails the check. `None` skips. Distinct from
    /// [`assert_not_starved`]'s zero-work-units check, which gates
    /// only against literal zero — this gate accepts a non-zero
    /// floor so a test can reject "barely made progress" runs that
    /// pass the strict starvation gate.
    pub min_work_units: Option<u64>,
}

impl AbsoluteThresholds {
    /// Sane-default preset: p99 wake latency under 10ms, p99
    /// iteration cost under 1ms, total migrations under 1000, every
    /// worker completes ≥1 work unit. The defaults are deliberately
    /// loose — a threshold set tight enough to catch egregious
    /// regressions without flagging every routine scheduler
    /// perturbation. Tests
    /// that need tighter bounds should set the fields explicitly via
    /// the bare-verb builder methods rather than tuning these constants.
    pub const fn strict() -> Self {
        Self {
            max_p99_wake_latency_ns: Some(10_000_000),
            max_iteration_cost_p99_ns: Some(1_000_000),
            max_migrations: Some(1000),
            min_work_units: Some(1),
        }
    }

    /// Builder setter for [`Self::max_p99_wake_latency_ns`].
    pub const fn max_p99_wake_latency_ns(mut self, v: u64) -> Self {
        self.max_p99_wake_latency_ns = Some(v);
        self
    }

    /// Builder setter for [`Self::max_iteration_cost_p99_ns`].
    pub const fn max_iteration_cost_p99_ns(mut self, v: u64) -> Self {
        self.max_iteration_cost_p99_ns = Some(v);
        self
    }

    /// Builder setter for [`Self::max_migrations`].
    pub const fn max_migrations(mut self, v: u64) -> Self {
        self.max_migrations = Some(v);
        self
    }

    /// Builder setter for [`Self::min_work_units`].
    pub const fn min_work_units(mut self, v: u64) -> Self {
        self.min_work_units = Some(v);
        self
    }
}

/// Run every check in `thresholds` against `reports`, merging results
/// into a single [`AssertResult`]. A `None` field on the thresholds
/// skips that check.
///
/// An empty `reports` slice short-circuits to a skip (`"no worker
/// reports to evaluate"`) regardless of thresholds content — silently
/// passing thresholds against zero samples would let them look
/// "green" on a run that produced no measurement.
///
/// Field-to-check mapping:
/// - `max_p99_wake_latency_ns` -> pooled p99 across every worker's
///   `wake_latencies_ns`; tagged [`DetailKind::Benchmark`].
/// - `max_iteration_cost_p99_ns` -> pooled p99 across every worker's
///   `iteration_costs_ns`; tagged [`DetailKind::Benchmark`].
/// - `max_migrations` -> sum of `migration_count` across workers;
///   tagged [`DetailKind::Migration`].
/// - `min_work_units` -> per-worker `work_units >= floor`; tagged
///   [`DetailKind::Starved`] when a worker is below the floor.
///
/// The wake-latency check delegates to [`assert_benchmarks`] for the
/// percentile path so the same nearest-rank algorithm applies; the
/// iteration-cost check uses an inline percentile call against the
/// pooled `iteration_costs_ns` reservoir.
///
/// ```
/// # use ktstr::assert::{AbsoluteThresholds, assert_thresholds};
/// # use ktstr::workload::WorkerReport;
/// # let report = WorkerReport {
/// #     tid: 1, cpus_used: [0].into_iter().collect(),
/// #     work_units: 1000, cpu_time_ns: 2_500_000_000,
/// #     wall_time_ns: 5_000_000_000, off_cpu_ns: 2_500_000_000,
/// #     migration_count: 5, migrations: vec![],
/// #     max_gap_ms: 50, max_gap_cpu: 0, max_gap_at_ms: 1000,
/// #     wake_latencies_ns: vec![100, 200, 300, 400, 500],
/// #     wake_sample_total: 5,
/// #     iteration_costs_ns: vec![1000, 2000, 3000, 4000, 5000],
/// #     iteration_cost_sample_total: 5,
/// #     iterations: 1000,
/// #     schedstat_run_delay_ns: 0, schedstat_run_count: 0,
/// #     schedstat_cpu_time_ns: 0,
/// #     completed: true,
/// #     numa_pages: std::collections::BTreeMap::new(),
/// #     vmstat_numa_pages_migrated: 0,
/// #     exit_info: None,
/// #     affinity_error: None,
/// #     is_messenger: false,
/// #     group_idx: 0,
/// #     phase_slices: vec![],
/// # };
/// // Strict preset on a healthy run — passes.
/// let r = assert_thresholds(&[report], &AbsoluteThresholds::strict());
/// assert!(r.is_pass());
/// ```
pub fn assert_thresholds(
    reports: &[WorkerReport],
    thresholds: &AbsoluteThresholds,
) -> AssertResult {
    // Empty `reports` means nothing was measured. Returning a fresh
    // `pass()` here would silently green-light a broken run that
    // produced no signal; delegating to `assert_benchmarks` and
    // merging its skip would lose the skip flag (`AssertResult::merge`
    // ANDs `skipped`, so `pass.merge(skip) == passed-not-skipped`).
    // Surface the skip directly so the operator sees the thresholds
    // weren't actually exercised.
    if reports.is_empty() {
        return AssertResult::skip("no worker reports to evaluate");
    }

    let mut r = AssertResult::pass();

    // Wake-latency p99: reuse the existing `assert_benchmarks` path
    // so the percentile algorithm stays unified. With `reports`
    // non-empty here, `assert_benchmarks` cannot return a skip —
    // the merge sees only pass/fail, preserving thresholds semantics.
    if thresholds.max_p99_wake_latency_ns.is_some() {
        r.merge(assert_benchmarks(
            reports,
            thresholds.max_p99_wake_latency_ns,
            None,
            None,
        ));
    }

    // Iteration-cost p99: pooled across every worker's reservoir.
    // Skipped when no samples are present — compute work types that
    // populate `iteration_costs_ns` are sparse, so an empty pooled
    // set is the common case for blocking variants and not a failure.
    if let Some(cost_limit) = thresholds.max_iteration_cost_p99_ns {
        let all_costs: Vec<u64> = reports
            .iter()
            .flat_map(|w| w.iteration_costs_ns.iter().copied())
            .collect();
        if !all_costs.is_empty() {
            let mut sorted = all_costs.clone();
            sorted.sort_unstable();
            let p99 = percentile(&sorted, 0.99);
            if p99 > cost_limit {
                r.record_fail(AssertDetail::new(
                    DetailKind::Benchmark,
                    format!(
                        "p99 iteration cost {p99}ns exceeds limit {cost_limit}ns ({} samples)",
                        sorted.len(),
                    ),
                ));
            }
        }
    }

    // Total migrations across all workers: absolute-count gate
    // (distinct from migration_ratio which is a per-iteration rate).
    if let Some(max_mig) = thresholds.max_migrations {
        // saturating: a corrupt/hostile per-worker migration_count must clamp,
        // not wrap, this absolute-count gate — a wrapped small total would
        // silently PASS a max_migrations limit it should fail.
        let total_mig: u64 = reports
            .iter()
            .map(|w| w.migration_count)
            .fold(0u64, u64::saturating_add);
        if total_mig > max_mig {
            r.record_fail(AssertDetail::new(
                DetailKind::Migration,
                format!(
                    "total migrations {total_mig} exceeds limit {max_mig} ({} workers)",
                    reports.len(),
                ),
            ));
        }
    }

    // Per-worker work_units floor: every worker must have completed
    // at least `min` work units. One starved worker fails the check.
    if let Some(min_units) = thresholds.min_work_units {
        for w in reports {
            if w.work_units < min_units {
                r.record_fail(AssertDetail::new(
                    DetailKind::Starved,
                    format!(
                        "tid {} work_units {} below floor {min_units}",
                        w.tid, w.work_units,
                    ),
                ));
            }
        }
    }

    r
}

// (The legacy `Expect` / `Checks` / `CheckBuilder` types previously
// living here were replaced by the [`Verdict`]-based claim API
// (defined further up in this file). The new flow is
// `Assert::default_checks().verdict().claim_<field>(stats).at_most(N)` for
// stats-struct-derived accessors, or `claim!(verdict, expr)` for
// expression-labeled claims. Both produce
// [`ClaimBuilder`]/[`SetClaim`]/[`SeqClaim`] under the hood and
// record outcomes onto the same [`AssertResult`] envelope that
// `assert_not_starved` / `assert_isolation` produce, so the two
// paths compose via [`Verdict::merge`].)
