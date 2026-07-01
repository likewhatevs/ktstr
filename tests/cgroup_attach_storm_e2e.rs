//! e2e for the `CgroupAttachStorm` WorkType: workers fork transient
//! children and migrate them — the whole process — into a sibling
//! cgroup, driving the kernel cgroup-attach threadgroup path against a
//! live scheduler.

use ktstr::prelude::*;

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Sibling cgroup the storm migrates its forked children into.
const DEST: &str = "dest";
/// Storm workers — each forks + migrates + reaps independently.
const WORKERS: usize = 8;

/// `CgroupAttachStorm` runs end-to-end in a VM against the scx-ktstr
/// fixture. `WORKERS` storm workers live in a dedicated cgroup
/// (`"race"`) and each iteration `fork` a transient child, write its pid
/// to the sibling `DEST` cgroup's `cgroup.procs` (resolved as
/// `<race-parent>/dest/cgroup.procs`), and let the child `_exit` racing
/// the `SIGCHLD = SIG_IGN` auto-reap. The `cgroup.procs` write drives
/// the kernel's `cgroup_attach_task` threadgroup walk →
/// `TRACE_CGROUP_PATH(attach_task)` (`kernel/cgroup/cgroup.c`) — the
/// leader-acquire surface a scheduler's cgroup-attach handler must
/// survive (migrating a task that is concurrently tearing down).
///
/// `DEST` is created by `Op::add_cgroup(DEST)` on the step. A non-Loop
/// step applies its ops BEFORE the `CgroupDef` setup spawns the workers
/// (`src/scenario/ops/mod.rs` — "Ops first … then CgroupDef setup"), so
/// the sibling exists and is writable (the guest runs as root) when the
/// workers pre-flight it at entry: the storm cannot vacuously no-op.
///
/// The scx-ktstr fixture does NOT implement a `cgroup_attach_task`
/// (`tp_btf`) handler that crashes on the race, so this test's contract
/// is that the scenario completes cleanly under the scheduler (the
/// mechanism works in-VM and the scheduler survives the storm). The
/// scheduler-crash reproduction that USES this WorkType lives in
/// scx_layered's ktstr tests, where the cgroup-attach handler is
/// implemented and the migrate-vs-exit race trips the leader acquire;
/// the recipe for hosting that test in the external scheduler crate is
/// documented in `doc/guide`. The serde / name / from_name / ctor
/// surface and the `CloneMode::Thread` spawn-reject are pinned by the
/// CI-runnable unit tests in `src/workload/types/tests.rs` and
/// `src/workload/spawn/tests_thread_mode.rs`.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 1024,
    duration_s = 10,
    watchdog_timeout_s = 30
)]
fn cgroup_attach_storm_runs_in_vm(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("race")
                    .workers(WORKERS)
                    .work_type(WorkType::cgroup_attach_storm(DEST, ReapMode::SigIgn)),
            ],
            HoldSpec::FULL,
        )
        .set_ops(vec![Op::add_cgroup(DEST)]),
    ];
    execute_steps(ctx, steps)
}
