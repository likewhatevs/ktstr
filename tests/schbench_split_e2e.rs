//! End-to-end proof that schbench's `--split` mode runs inside a real VM. Boots
//! with scx-ktstr, runs a backdrop schbench workload configured with
//! `split_percent(Some(50))` and two internal worker threads, holds one phase,
//! and reads the per-phase wakeup/request percentiles back in post_vm.
//!
//! `--split` is the deliberate-cache-contention mode: each cycle runs some matrix
//! multiplies on ONE process-global shared matrix (modeled with `AtomicU64`
//! `Relaxed` load/store, so the shared race is sound) and the rest on a per-worker
//! private matrix (schbench.c:1390-1404). Two worker threads concurrently
//! RMW-multiply the shared matrix, so this exercises the real concurrent shared
//! path in a guest. The gate is the FRAMEWORK INVARIANT: the split workload boots,
//! runs, and emits its per-phase latency metrics -- it does NOT gate on absolute
//! latency (scx-ktstr is a fixture, not a performance scheduler). A deadlock,
//! panic, or data-race trap in the shared-matrix path would produce no metric and
//! fail Gate 1.

use anyhow::Result;
use ktstr::assert::{AssertResult, Phase};
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::test_support::{Scheduler, SchedulerSpec};

const SPLIT_SCX: Scheduler =
    Scheduler::named("split_scx").binary(SchedulerSpec::Discover("scx-ktstr"));

/// The two per-phase percentiles the split run must emit (both LowerBetter
/// latencies). Their presence proves the split engine ran end-to-end in the VM
/// and recorded both the wakeup (message-handshake) and request latencies.
const WAKEUP_METRIC: &str = "wakeup_p99_latency_us";
const REQUEST_METRIC: &str = "request_p99_latency_us";

/// post_vm: the framework-invariant gate. The split workload produced both
/// per-phase percentiles -- proves the shared-matrix concurrent path ran to
/// completion in the guest (no deadlock, no panic, no data-race trap).
fn assert_split_workload_ran(result: &VmResult) -> Result<()> {
    let split = Phase::step(0); // the lone hold step (step_index 1)
    anyhow::ensure!(
        result.phase_metric(split, WAKEUP_METRIC).is_some(),
        "split phase produced no {WAKEUP_METRIC} -- the split workload did not record latencies"
    );
    anyhow::ensure!(
        result.phase_metric(split, REQUEST_METRIC).is_some(),
        "split phase produced no {REQUEST_METRIC} -- the split workload did not record latencies"
    );
    Ok(())
}

// Single 6s hold step; duration_s covers it with margin and watchdog_timeout_s
// lifts the VM hard deadline to cover boot + the hold (vm_timeout_from_entry,
// test_support/runtime.rs).
#[ktstr_test(
    scheduler = SPLIT_SCX,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 8,
    watchdog_timeout_s = 30,
    cleanup_budget_ms = 5000,
    num_snapshots = 3,
    post_vm = assert_split_workload_ran,
)]
fn schbench_split_runs_in_vm(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::backdrop::Backdrop;
    use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_scenario};
    use ktstr::workload::{SchbenchConfig, WorkType};
    use std::time::Duration;

    // split_percent(Some(50)): half the cache footprint is the per-thread private
    // matrix, half is the ONE shared matrix every worker contends on. Two worker
    // threads make the shared-matrix RMW genuinely concurrent. WorkType::Schbench
    // mandates a single ktstr worker (the message/worker topology is internal).
    let cfg = SchbenchConfig::default()
        .worker_threads(2)
        .split_percent(Some(50));
    let backdrop = Backdrop::new().push_cgroup(
        CgroupDef::named("split_bench")
            .work_type(WorkType::schbench(cfg))
            .workers(1),
    );

    let steps = vec![Step::new(vec![], HoldSpec::fixed(Duration::from_secs(6)))];

    execute_scenario(ctx, backdrop, steps)
}
