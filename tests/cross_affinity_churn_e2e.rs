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
/// Converted from `#[ktstr_test]` to `#[ktstr_scenario]` so the workload is a
/// value a second backend can run.
///
/// THE CONVERSION IS A PURE RESHAPE — no `&Ctx` read is resolved and no
/// substitution is made, so there is no condition to state. The body never
/// dereferenced `ctx`; it only handed it to `execute_defs`, and
/// `execute_defs(ctx, defs)` is defined as
/// `execute_steps(ctx, vec![Step::with_defs(defs, HoldSpec::FULL)])` while
/// `ScenarioDef::with_defs(defs)` is defined as
/// `Self::new(vec![Step::with_defs(defs, HoldSpec::FULL)])`. Same step, same
/// hold, same defs.
///
/// THE ORACLE IS UNCHANGED. It lives entirely in the attribute above --
/// `sustained_samples`, `max_keep_last_rate`, `max_fallback_rate` -- which the
/// macro carries through untouched. Nothing moved into or out of the body.
///
/// WHAT THE SIMULATOR WILL NOT SEE, stated here rather than left to the
/// fidelity report alone: `CrossAffinityChurn` lowers to `Run(n)` with NO
/// affinity change (`Cause::DynamicSchedAttr`), so in-sim the two flippers
/// become plain spin loops and the cross-task affinity rewriting this test is
/// named for DOES NOT HAPPEN. The eight `FutexPingPong` targets are faithful
/// (`Cause::BlockingMechanism` — the futex wait/wake structure is preserved).
/// AND THE WORKER COUNT NARROWS, WHICH NOTHING RECORDS. `FutexPingPong` lowers
/// to `Plan { tasks: 2 }` -- a hardcoded pair -- so the declared `workers(8)`
/// becomes TWO tasks in-sim. With the two flippers that is 4 simulated tasks
/// against 10 in the VM. The fidelity report says `BlockingMechanism`, which
/// describes the wait/wake MECHANISM and says nothing about the count, so the
/// narrowing is invisible in every artifact: the exported record still says
/// `workers: 8`, and the cross-backend check compares per-cgroup CPU SHARE
/// against a single cgroup, which is 1.0 on both sides no matter how many
/// tasks produced it.
///
/// So a green simulator run of this scenario is evidence about futex wait/wake
/// STRUCTURE under a shared cpuset. It is NOT evidence about affinity churn,
/// and it is NOT evidence at the declared concurrency. Read the fidelity report
/// before quoting it as either, and do not read the record's worker counts as
/// what the simulator ran.
#[ktstr_scenario(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 4,
    threads = 1,
    sustained_samples = 25,
    // Churn-heavy multi-profile fixture: the guest scx watchdog's 5s
    // default evicts it under contended-runner starvation (the same
    // margin its scenario_coverage siblings already declare).
    watchdog_timeout_s = 15,
    max_keep_last_rate = 1000000000.0,
    max_fallback_rate = 1000000000.0
)]
fn cross_affinity_churn_runs_in_vm() -> ScenarioDef {
    ScenarioDef::with_defs(vec![
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
    ])
}

/// The converted scenario is the workload a backend actually receives.
///
/// Reached through `KTSTR_SCENARIOS` — the exporter's own path — rather than by
/// calling the builder directly, so it asserts on the value the export walks,
/// not on a local copy that could agree while the registry disagreed.
#[test]
fn converted_scenario_is_the_workload_a_backend_receives() {
    let entry = ktstr::test_support::find_scenario("cross_affinity_churn_runs_in_vm")
        .expect("the converted scenario must register in KTSTR_SCENARIOS");
    let def = (entry.build)();

    let steps = def.steps();
    assert_eq!(
        steps.len(),
        1,
        "execute_defs is one step; got {}",
        steps.len()
    );
    assert_eq!(
        steps[0].hold,
        ktstr::scenario::ops::HoldSpec::FULL,
        "execute_defs holds FULL, and ScenarioDef::with_defs must too -- if this \
         fails the reshape changed the run window",
    );

    let defs = match &steps[0].setup {
        ktstr::scenario::ops::Setup::Defs(d) => d,
        ktstr::scenario::ops::Setup::Factory(_) => panic!("must be static Defs, not a Factory"),
    };
    assert_eq!(defs.len(), 1, "one cgroup");
    assert_eq!(defs[0].name.as_ref(), "race");

    // The two WorkSpecs, IN DECLARATION ORDER -- the body comment explains why
    // the order is load-bearing (apply_setup starts them serially, so the
    // flippers must find the targets already in cgroup.procs).
    let works = &defs[0].works;
    assert_eq!(works.len(), 2, "targets + flippers");
    assert_eq!(works[0].num_workers, Some(8), "eight futex targets");
    assert_eq!(
        works[0].work_type,
        ktstr::workload::WorkType::FutexPingPong { spin_iters: 0 },
    );
    assert_eq!(works[1].num_workers, Some(2), "two flippers");
    assert_eq!(
        works[1].work_type,
        ktstr::workload::WorkType::cross_affinity_churn(0),
    );
}

/// Export this binary's scenarios. Required of every binary that declares one --
/// `every_scenario_binary_exports` fails by filename otherwise, because
/// `KTSTR_SCENARIOS` is per link unit.
#[test]
fn export_registered_scenarios() {
    let _ = ktstr::test_support::export_registered_scenarios(
        ktstr::test_support::DEFAULT_WORKERS_PER_CGROUP,
    );
}
