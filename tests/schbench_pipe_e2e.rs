//! End-to-end proof that schbench's `-p` pipe (memory-transfer) mode runs inside
//! a real VM. Boots with scx-ktstr, runs a backdrop schbench workload configured
//! with `pipe_transfer_bytes(65536)` and two internal worker threads, holds one
//! phase, and reads the per-phase wakeup percentile back in post_vm.
//!
//! Pipe mode REPLACES the matrix work with schbench's memory-transfer simulation
//! (schbench.c:177): the message thread memsets each woken worker's per-thread
//! page to `1` (schbench.c:980-981) and the worker memsets its own page to `2`
//! before blocking (schbench.c:1003-1004), then `do_work` + the think-sleep are
//! skipped (schbench.c:1448). The two worker threads make the waker->worker page
//! writes genuinely concurrent across cores, exercising the cross-thread
//! `pipe_page` `UnsafeCell` whose race-freedom rests on the wait-list + futex
//! happens-before ordering (the run.rs `Sync` SAFETY note).
//!
//! The gate is the FRAMEWORK INVARIANT: the pipe workload boots, runs, and emits
//! its per-phase WAKEUP latency metric -- proof the message-handshake transfer
//! loop ran end-to-end in the guest (each cycle is a wakeup + the two memsets).
//! It does NOT gate on absolute latency (scx-ktstr is a fixture, not a
//! performance scheduler), and it does NOT gate on request latency: pipe mode
//! skips `do_work`, so the request-latency delta is ~0 and filtered out
//! (schbench reports only wakeup + transfer throughput in pipe mode,
//! schbench.c:1972-1982). A deadlock, panic, or data-race trap in the
//! cross-thread pipe-page path would produce no metric and fail the gate.

use anyhow::Result;
use ktstr::assert::{AssertResult, Phase};
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::test_support::{Scheduler, SchedulerSpec};

const PIPE_SCX: Scheduler =
    Scheduler::named("pipe_scx").binary(SchedulerSpec::Discover("scx-ktstr"));

/// The per-phase percentile the pipe run must emit: the wakeup (message-handshake)
/// latency. Its presence proves the pipe transfer loop ran end-to-end in the VM
/// (each transfer cycle is a handshake wakeup plus the worker/waker page memsets).
/// Request latency is deliberately NOT required -- pipe mode skips `do_work`, so
/// that delta is ~0 and filtered (matching schbench's pipe-mode report).
const WAKEUP_METRIC: &str = "wakeup_p99_latency_us";

/// post_vm: the framework-invariant gate. The pipe workload produced the wakeup
/// percentile -- proves the cross-thread pipe-page transfer path ran to completion
/// in the guest (no deadlock, no panic, no data-race trap).
fn assert_pipe_workload_ran(result: &VmResult) -> Result<()> {
    let pipe = Phase::step(0); // the lone hold step (step_index 1)
    anyhow::ensure!(
        result.phase_metric(pipe, WAKEUP_METRIC).is_some(),
        "pipe phase produced no {WAKEUP_METRIC} -- the pipe transfer workload did not run"
    );
    Ok(())
}

// Single 6s hold step; duration_s covers it with margin and watchdog_timeout_s
// lifts the VM hard deadline to cover boot + the hold (vm_timeout_from_entry,
// test_support/runtime.rs).
#[ktstr_test(
    scheduler = PIPE_SCX,
    llcs = 1,
    cores = 2,
    threads = 1,
    memory_mib = 512,
    duration_s = 8,
    watchdog_timeout_s = 30,
    cleanup_budget_ms = 5000,
    num_snapshots = 3,
    post_vm = assert_pipe_workload_ran,
)]
fn schbench_pipe_runs_in_vm(ctx: &Ctx) -> Result<AssertResult> {
    use ktstr::scenario::backdrop::Backdrop;
    use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_scenario};
    use ktstr::workload::{SchbenchConfig, WorkType};
    use std::time::Duration;

    // pipe_transfer_bytes(65536): 64 KiB per-thread transfer page (well under the
    // 1 MiB clamp). Two worker threads make the waker->worker page memsets
    // genuinely concurrent. sleep_usec(0) drops the think-sleep (pipe mode skips
    // it anyway). WorkType::Schbench mandates a single ktstr worker (the
    // message/worker topology is internal).
    let cfg = SchbenchConfig::default()
        .worker_threads(2)
        .sleep_usec(0)
        .pipe_transfer_bytes(65536);
    let backdrop = Backdrop::new().push_cgroup(
        CgroupDef::named("pipe_bench")
            .work_type(WorkType::schbench(cfg))
            .workers(1),
    );

    let steps = vec![Step::new(vec![], HoldSpec::fixed(Duration::from_secs(6)))];

    execute_scenario(ctx, backdrop, steps)
}
