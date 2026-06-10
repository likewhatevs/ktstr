//! e2e for `WorkerCtx`: a `Custom` worker discovers its cgroup cpuset
//! and sibling pids THROUGH the ctx (not raw `sched_getaffinity` /
//! `cgroup.procs` scraping) and uses them to churn its siblings'
//! affinity. The payoff: a custom probe gets the same
//! runtime environment the built-in `WorkType::CrossAffinityChurn`
//! computes, without re-rolling the syscalls.

use ktstr::prelude::*;
use std::sync::atomic::Ordering;

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

// A cross-task affinity flipper built ENTIRELY on `WorkerCtx`: `cpus()`
// supplies the cpuset to derive two one-CPU-apart masks, and
// `sibling_pids()` supplies the targets. Mirrors
// `WorkType::CrossAffinityChurn` but shows a `Custom` worker can do it
// from the ctx alone. The asserts below are ADVISORY: an empty
// cpus/siblings (a wiring failure) panics the flipper → a sentinel
// report, but the scenario verdict does NOT gate on worker completion
// (src/scenario never inspects it), so they do not fail the test.
// Deterministic ctx-wiring correctness is pinned host-side by
// custom_worker_receives_wired_ctx; this e2e's job is "the ctx-driven
// flipper MECHANISM runs end-to-end against a live scheduler." Masks
// come from the flipper's own cpuset and the kernel clamps each
// sched_setaffinity to the target's cpuset (cpumask_and); a fully-
// disjoint sibling clamps to empty → -EINVAL, silently skipped — the
// shared CgroupDef below avoids that.
fn ctx_flipper(ctx: &WorkerCtx) -> WorkerReport {
    let cpus = ctx.cpus();
    assert!(
        cpus.len() >= 2,
        "ctx.cpus() must expose the cgroup's >=2-CPU cpuset; got {cpus:?}"
    );
    assert!(
        !ctx.sibling_pids().is_empty(),
        "ctx.sibling_pids() must see the cgroup peers in a dedicated cgroup"
    );
    let sz = std::mem::size_of::<libc::cpu_set_t>();
    let make = |drop_last: bool| -> libc::cpu_set_t {
        let take = if drop_last {
            cpus.len() - 1
        } else {
            cpus.len()
        };
        let mut s: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe { libc::CPU_ZERO(&mut s) };
        for &c in cpus.iter().take(take) {
            unsafe { libc::CPU_SET(c, &mut s) };
        }
        s
    };
    let (mask_a, mask_b) = (make(false), make(true));
    let mut toggle = false;
    while !ctx.stop().load(Ordering::Relaxed) {
        let mask: *const libc::cpu_set_t = if toggle { &mask_a } else { &mask_b };
        for &pid in ctx.sibling_pids() {
            // SAFETY: pid is a live cgroup sibling; sched_setaffinity
            // with a valid mask pointer and size. Best-effort — a
            // sibling that exited mid-run returns ESRCH, which we ignore.
            unsafe {
                libc::sched_setaffinity(pid, sz, mask);
            }
        }
        toggle = !toggle;
    }
    WorkerReport::default()
}

/// A `Custom` worker drives cross-task affinity churn using ONLY its
/// `WorkerCtx` — `cpus()` for the masks, `sibling_pids()` for the
/// targets — in a real scx-ktstr VM: the ctx-driven flipper MECHANISM
/// runs end-to-end against a live scheduler. This does NOT prove
/// ctx-wiring correctness in-VM (a wiring bug panics the flipper's
/// advisory asserts → sentinel report, but the scenario can still run
/// cleanly — the verdict doesn't gate on worker completion); the
/// deterministic wiring guard is the host-side
/// `custom_worker_receives_wired_ctx`. Eight `FutexPingPong` targets
/// are declared BEFORE the two flippers so the declaration-order
/// contract makes them visible (see `WorkType::CrossAffinityChurn`).
/// Verdict mirrors `cross_affinity_churn_e2e`: the scenario runs
/// cleanly under the scheduler. Fairness rate ceilings are raised
/// because the flipper and futex profiles diverge by design.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 4,
    threads = 1,
    memory_mib = 2048,
    sustained_samples = 25,
    max_keep_last_rate = 1000000000.0,
    max_fallback_rate = 1000000000.0
)]
fn worker_ctx_drives_affinity_churn_in_vm(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(
        ctx,
        vec![
            CgroupDef::named("race")
                .cpuset(CpusetSpec::range(0.0, 1.0))
                // Targets: the workers whose affinity the flippers rewrite.
                .work(
                    WorkSpec::default()
                        .workers(8)
                        .work_type(WorkType::FutexPingPong { spin_iters: 0 }),
                )
                // Flippers: a Custom worker reading cpus()/sibling_pids()
                // from its WorkerCtx — declared AFTER the targets so the
                // serial-start contract makes the futex workers visible.
                .work(
                    WorkSpec::default()
                        .workers(2)
                        .work_type(WorkType::custom("ctx_flipper", ctx_flipper)),
                ),
        ],
    )
}
