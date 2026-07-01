//! e2e for the `survives_storm` entry assertion (the positive inverse of
//! `expect_err`): a scx scheduler that SURVIVES a storm yields EXIT_PASS
//! with `survives_storm` set.

use ktstr::prelude::*;

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// `#[ktstr_test(survives_storm)]` over a Step scenario that holds a
/// `CgroupAttachStorm` workload (fork → migrate child into a sibling
/// `dest` cgroup → exit racing the reap) against the scx-ktstr fixture.
/// scx-ktstr has no crashing `cgroup_attach_task` handler, so it survives
/// the storm and the run PASSES — the symmetric fixed-direction pin to the
/// crash / `expect_err` direction.
///
/// What `survives_storm` adds over the plain
/// `tests/cgroup_attach_storm_e2e.rs` companion: it DECLARES the survival
/// intent in source and exercises the full attribute → `KtstrTestEntry`
/// flag → dispatch projection. If the scheduler had died during the hold,
/// the scenario liveness probe would record a `DetailKind::Scheduler*`
/// fail, the eval layer would attach the `SurvivesStormViolated` marker,
/// and dispatch would force EXIT_FAIL with a survival-specific explainer
/// (before any inversion arm); a clean survival is EXIT_PASS unchanged.
/// The death direction itself is covered by the existing crash /
/// `expect_err` tests; the dispatch exit-code projection is unit-pinned in
/// `src/test_support/dispatch_tests.rs`.
///
/// `DEST` is created by `Op::add_cgroup` on the step; a non-Loop step
/// applies its ops BEFORE the `CgroupDef` setup spawns the workers, so the
/// sibling exists and is writable (the guest runs as root) at worker
/// entry.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    survives_storm,
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 1024,
    duration_s = 10,
    watchdog_timeout_s = 30
)]
fn scheduler_survives_storm_e2e(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("race")
                    .workers(8)
                    .work_type(WorkType::cgroup_attach_storm("dest", ReapMode::SigIgn)),
            ],
            HoldSpec::FULL,
        )
        .set_ops(vec![Op::add_cgroup("dest")]),
    ];
    execute_steps(ctx, steps)
}
