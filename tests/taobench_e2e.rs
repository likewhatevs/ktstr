//! End-to-end proof that the taobench key-value-cache workload runs inside a real
//! VM. Boots with scx-ktstr, runs a backdrop taobench workload (a bounded,
//! evicting cache with a fast in-cache hit path and a slow backing-store-miss
//! path), holds one phase, and reads the per-phase qps metric back in post_vm.
//!
//! The taobench engine is entirely in-process: a closed-loop client population
//! drives a sharded cache; fast workers serve hits (touching the stored value
//! bytes); slow dispatcher threads serve misses (sleep + fill + wake). The client
//! and dispatcher threads make the fast-hit / slow-miss handoffs genuinely
//! concurrent across the guest's cores. `WorkType::Taobench` mandates a single
//! ktstr worker -- the client/fast/slow parallelism is the engine's internal
//! thread topology, not ktstr worker processes.
//!
//! The gate is the FRAMEWORK INVARIANT: the taobench workload boots, runs its
//! topology end-to-end, and emits its per-phase `taobench_total_qps` metric --
//! proof the cache served ops, the slow-miss path filled + woke clients, and the
//! per-phase op counts reduced through the metric API into a `PhaseBucket`. It does
//! NOT gate on absolute qps (scx-ktstr is a fixture, not a performance scheduler);
//! the hit-ratio equilibrium (~target, not drifting to 1.0) is pinned by the
//! in-process unit test `engine_serves_ops_and_hit_ratio_settles_near_target_not_one`,
//! which runs WITHOUT a VM -- so the engine logic stays covered even when this
//! host-gated e2e is skipped. A deadlock, panic, or stats-plumbing break would
//! produce no metric and fail the gate.

use anyhow::Result;
use ktstr::assert::{AssertResult, Phase};
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::test_support::{Scheduler, SchedulerSpec};

const TAO_SCX: Scheduler = Scheduler::named("tao_scx").binary(SchedulerSpec::Discover("scx-ktstr"));

/// The per-phase metric the taobench run must emit: total throughput (qps). Its
/// presence proves the engine's client / fast-worker / slow-dispatcher topology
/// ran end-to-end in the VM and the per-phase op counts reduced through the metric
/// API. Present iff the phase had a measured wall window -- always true for a run
/// that served any ops.
const QPS_METRIC: &str = "taobench_total_qps";

/// post_vm: the framework-invariant gate. The taobench workload produced its
/// per-phase qps -- proves the cache + fast/slow tiers ran to completion in the
/// guest (no deadlock, no panic, no stats-plumbing break).
fn assert_taobench_workload_ran(result: &VmResult) -> Result<()> {
    let phase = Phase::step(0); // the lone hold step (step_index 1)
    anyhow::ensure!(
        result.phase_metric(phase, QPS_METRIC).is_some(),
        "taobench phase produced no {QPS_METRIC} -- the taobench workload did not run"
    );
    Ok(())
}

// Single 6s hold step; duration_s covers it with margin and watchdog_timeout_s
// lifts the VM hard deadline to cover boot + the hold.
#[ktstr_test(
    scheduler = TAO_SCX,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 8,
    watchdog_timeout_s = 30,
    cleanup_budget_ms = 5000,
    num_snapshots = 3,
    post_vm = assert_taobench_workload_ran,
)]
fn taobench_runs_in_vm(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::backdrop::Backdrop;
    use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_scenario};
    use ktstr::workload::{TaobenchConfig, WorkType};
    use std::time::Duration;

    // Small bounded cache (8 MiB) so the eviction<->refill equilibrium reaches the
    // target hit ratio quickly within the short hold. 4 client + 2 slow threads
    // make the fast-hit / slow-miss handoffs genuinely concurrent across the 2
    // cores. slow_path_sleep_us(50) keeps the slow tier a real off-CPU cost.
    let cfg = TaobenchConfig::default()
        .client_threads(4)
        .slow_threads(2)
        .cache_capacity_mib(8)
        .target_hit_pct(90)
        .slow_path_sleep_us(50);
    let backdrop = Backdrop::new().push_cgroup(
        CgroupDef::named("tao_bench")
            .work_type(WorkType::taobench(cfg))
            .workers(1),
    );

    let steps = vec![Step::new(vec![], HoldSpec::fixed(Duration::from_secs(6)))];

    execute_scenario(ctx, backdrop, steps)
}
