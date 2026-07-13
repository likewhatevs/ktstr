//! e2e for the `CrossAffinityChurn` WorkType: a worker that rewrites
//! its cgroup siblings' CPU affinity at high rate.

use ktstr::prelude::*;

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// `CrossAffinityChurn` runs end-to-end in a VM. Two flipper workers
/// (`cross_affinity_churn`) discover their cgroup siblings — the eight
/// `FutexPingPong` workers, a DIFFERENT WorkSpec in the same dedicated
/// cgroup — via `cgroup.procs` and rewrite their affinity at high
/// rate, toggling two cpuset sub-masks one CPU apart. This exercises
/// the cross-WorkSpec sibling discovery (the reason the WorkType reads
/// cgroup membership rather than its own handle's pids) and the flip
/// loop against a live scheduler.
///
/// The scx-ktstr fixture does NOT implement `ops.set_cpumask`, so the
/// flips drive only the generic kernel migration path
/// (`set_cpus_allowed` + `affine_move_task`, then `select_cpu` on the
/// next wake) — no `set_cpumask` hook, hence no scheduler crash. The
/// test's contract is therefore that the scenario completes cleanly
/// under the scheduler (the mechanism works in-VM). The
/// scheduler-crash reproduction that USES this WorkType lives in
/// scx_lavd's ktstr tests, where `ops.set_cpumask` is implemented and
/// the cross-CPU flip races the scheduler's flag bookkeeping.
///
/// Fairness rate ceilings are raised (mirroring `cover_sched_mixed`):
/// the flippers and the futex workers have deliberately divergent work
/// profiles, so the per-cgroup fairness checks would otherwise flag
/// the imbalance that is the whole point of the scenario.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 2048,
    sustained_samples = 25,
    // Churn-heavy multi-profile fixture: the guest scx watchdog's 5s
    // default evicts it under contended-runner starvation (the same
    // margin its scenario_coverage siblings already declare).
    watchdog_timeout_s = 15,
    max_keep_last_rate = 1000000000.0,
    max_fallback_rate = 1000000000.0
)]
fn cross_affinity_churn_runs_in_vm(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(
        ctx,
        vec![
            CgroupDef::named("race")
                .cpuset(CpusetSpec::range(0.0, 1.0))
                // Targets: the workers whose affinity gets rewritten.
                .work(
                    WorkSpec::default()
                        .workers(8)
                        .work_type(WorkType::FutexPingPong { spin_iters: 0 }),
                )
                // Flippers: the cross-task affinity-churn driver.
                // Declared AFTER the targets on purpose — apply_setup
                // spawns+moves+starts WorkSpecs serially in declaration
                // order, so the flippers find the FutexPingPong workers
                // already in cgroup.procs; reversing the order would
                // leave the flippers with only their co-flippers as
                // siblings — the eight FutexPingPong targets would not
                // yet be in cgroup.procs, so the intended cross-WorkSpec
                // churn would not happen.
                .work(
                    WorkSpec::default()
                        .workers(2)
                        .work_type(WorkType::cross_affinity_churn(0)),
                ),
        ],
    )
}
