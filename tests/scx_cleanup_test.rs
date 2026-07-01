use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::test_support::{Scheduler, SchedulerSpec};

const SCX_CLEANUP_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Boots a VM with the scx-ktstr scheduler attached, runs no
/// workload, and exits cleanly. Counterpart to
/// `eevdf_empty_run_exits_under_watchdog` in `tests/eevdf_tests.rs`.
///
/// Unlike the EEVDF version, this test boots WITH the scx-ktstr
/// scheduler attached, but it exercises the same non-blocking
/// poll-based `trace_pipe` reader (`start_trace_pipe`) on the
/// scheduler-attached teardown path. On a clean run no
/// `sched_ext_dump` event fires — that tracepoint emits only as a
/// one-shot exit dump, and clean runs never start one — so the
/// reader forwards nothing and joins within one poll cycle of the
/// stop flag. A regression that re-wedges teardown lands either as
/// a `cleanup_budget_ms = 5000` overshoot (caught by
/// `evaluate_vm_result` against
/// [`ktstr::prelude::VmResult::cleanup_duration`]) or, in the
/// catastrophic case, as a host VM timeout
/// (`vm_timeout_from_entry` in `src/test_support/runtime.rs`). The
/// cleanup duration is also persisted to the sidecar so stats
/// tooling can flag drift across runs.
///
/// Body returns `Ok(AssertResult::pass())` because the assertion
/// of interest — that VM teardown completes within the cleanup
/// budget — is enforced host-side after the body returns; if the
/// budget is exceeded the framework folds a failing detail into
/// the verdict.
#[ktstr_test(
    scheduler = SCX_CLEANUP_SCHED,
    llcs = 1,
    cores = 1,
    threads = 1,
    memory_mib = 256,
    cleanup_budget_ms = 5000,
)]
fn scx_empty_run_exits_under_watchdog(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}
