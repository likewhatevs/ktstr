//! Consolidated end-to-end coverage of the cold-path
//! [`Op::WriteKernelCold`] / [`Op::ReadKernelCold`] dispatch path
//! under scx-ktstr. Exercises the three resolution arms the v1
//! dispatcher implements:
//!
//! - [`KernelTarget::TaskField`] — per-task `task_struct` field
//!   writes under the dispatcher's 8-layer SCX validation chain
//!   (pid match, `start_time` identity, lifetime, `on_rq == 0`,
//!   scx queued-empty, `ext_sched_class`, `policy == SCHED_EXT`,
//!   `start_boottime != 0`). Seeds an SCX-managed worker's
//!   `scx.dsq_vtime` to ~30 days and asserts the read-back value
//!   either equals the seed or has advanced forward (no regression).
//! - [`KernelTarget::PerCpuField`] — per-CPU struct field read via
//!   the hardcoded v1 symbol→struct mapping (`runqueues→rq.nr_running`).
//! - [`KernelTarget::symbol`] — direct symbol read of `jiffies_64`,
//!   the kernel's 64-bit jiffies counter.
//!
//! Every Op fires through the freeze-rendezvous-and-thaw cold path,
//! so the test doubles as a smoke test for the freeze coord's
//! deferred-queue + accessor-adoption handshake under a live
//! scheduler — cold-ops dispatched before the host's `owned_accessor`
//! adoption defer cleanly instead of failing with the synthetic
//! "owned_accessor not yet initialised" error reply.
//!
//! Architecture: the scenario body runs INSIDE the guest VM and is
//! the only place that can read worker pids + `/proc/<pid>/stat`.
//! The host freeze-coord records every cold-op reply into the host
//! snapshot bridge (record_kernel_op_reply hook); the `post_vm`
//! callback drains that bridge HOST-side to assert on the captured
//! `read_values`.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{KernelOpReplyPayload, KernelOpValue, VmResult};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{
    HoldSpec, KernelTarget, KernelValue, KernelValueWidth, Op, Step, execute_steps,
};
use ktstr::test_support::{Scheduler, SchedulerSpec};
use ktstr::workload::{
    AffinityIntent, Phase, SchedPolicy, WorkType, WorkloadConfig, WorkloadHandle,
};
use std::time::Duration;

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Seed value for the `scx.dsq_vtime` write. Picks ~30 days in
/// nanoseconds — far above any vtime an scx-ktstr worker would
/// accumulate during the short test duration — so the post-read
/// assertion's "vtime advanced from seed" branch is unambiguous
/// (any organic vtime growth lands between SEED and SEED+test_dur).
const SEED_VTIME_NS: u64 = 30 * 86_400 * 1_000_000_000;

const TAG_TASK_FIELD: &str = "verify_dsq_vtime";
const TAG_PER_CPU_FIELD: &str = "verify_rq_nr_running";
const TAG_SYMBOL: &str = "verify_jiffies_64";

/// Read `/proc/<pid>/stat` field 22 (`starttime`, jiffies since boot)
/// and convert to nanoseconds via `sysconf(_SC_CLK_TCK)`.
///
/// Field 2 (`comm`) is parenthesized and may contain spaces, so the
/// jiffies-tokenization splits AFTER the rightmost `)` to avoid
/// miscounting embedded spaces inside `comm`. Returns the value in
/// nanoseconds since boot to match the dispatcher's
/// `expected_start_time_ns` parameter (the kernel populates
/// `task->start_time` via `ktime_get_ns()` in
/// `kernel/fork.c::copy_process`).
fn read_start_time_ns(pid: libc::pid_t) -> Result<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let after_comm = stat
        .rsplit_once(')')
        .ok_or_else(|| anyhow::anyhow!("/proc/{pid}/stat missing closing ')' for comm"))?
        .1;
    // Layout after comm closing paren: ` state ppid pgrp session
    // tty_nr tpgid flags minflt cminflt majflt cmajflt utime stime
    // cutime cstime priority nice num_threads itrealvalue starttime`.
    // Iterator's nth(19) lands on starttime (0-based).
    let token = after_comm
        .split_whitespace()
        .nth(19)
        .ok_or_else(|| anyhow::anyhow!("/proc/{pid}/stat truncated before starttime"))?;
    let jiffies: u64 = token
        .parse()
        .map_err(|e| anyhow::anyhow!("starttime '{token}' not a u64: {e}"))?;
    let clk_tck = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    anyhow::ensure!(clk_tck > 0, "sysconf(_SC_CLK_TCK) returned {clk_tck}");
    Ok(jiffies
        .saturating_mul(1_000_000_000)
        .saturating_div(clk_tck as u64))
}

/// Boot scx-ktstr, spawn one SCX-managed worker that blocks for the
/// whole scenario, then seed its `scx.dsq_vtime` to a known sentinel
/// and read it back via the cold-path dispatcher. Folds Symbol +
/// PerCpuField reads into the same Step so all three resolution
/// arms exercise the freeze-rendezvous-and-thaw path in one boot.
///
/// The worker uses [`WorkType::Sequence`] with a single
/// [`Phase::Sleep`] longer than the scenario's total budget so the
/// task stays in `TASK_INTERRUPTIBLE` (off-rq, `scx.dsq == NULL`,
/// `scx.runnable_node` empty) across every cold-path Op fire —
/// required for the dispatcher's L4 (`on_rq == 0`) + L5 (scx queued-
/// empty) gates to pass.
#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 5,
    watchdog_timeout_s = 60,
    auto_repro = false,
    post_vm = assert_cold_path_roundtrips,
)]
fn cold_path_op_handler_roundtrips(ctx: &Ctx) -> Result<AssertResult> {
    let block_dur = ctx.duration + Duration::from_secs(10);
    let config = WorkloadConfig {
        num_workers: 1,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::Sequence {
            first: Phase::Sleep(block_dur),
            rest: vec![],
        },
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    // Allow the worker to enter Phase::Sleep so the dispatcher's
    // on_rq=0 + scx-queued-empty gates pass. 500ms covers the start
    // handshake + scx-ktstr's dispatch latency under a 1-worker,
    // 0-load scenario.
    std::thread::sleep(Duration::from_millis(500));
    let pids = handle.worker_pids();
    let pid_i32 = *pids
        .first()
        .ok_or_else(|| anyhow::anyhow!("worker_pids returned no workers"))?;
    anyhow::ensure!(pid_i32 > 0, "worker pid {pid_i32} unset");
    let pid: u32 = u32::try_from(pid_i32)
        .map_err(|e| anyhow::anyhow!("worker pid {pid_i32} does not fit u32: {e}"))?;
    let start_time_ns = read_start_time_ns(pid_i32)?;

    let task_target = || KernelTarget::task_field(pid, start_time_ns, "scx.dsq_vtime");
    let steps = vec![Step::new(
        vec![
            Op::write_kernel_cold(task_target(), KernelValue::u64(SEED_VTIME_NS)),
            Op::read_kernel_cold(TAG_TASK_FIELD, task_target(), KernelValueWidth::u64()),
            Op::read_kernel_cold(
                TAG_PER_CPU_FIELD,
                KernelTarget::per_cpu_field("runqueues", "nr_running", 0),
                KernelValueWidth::u32(),
            ),
            Op::read_kernel_cold(
                TAG_SYMBOL,
                KernelTarget::symbol("jiffies_64"),
                KernelValueWidth::u64(),
            ),
        ],
        HoldSpec::FULL,
    )];
    let result = execute_steps(ctx, steps);
    let _ = handle.stop_and_collect();
    result
}

fn assert_cold_path_roundtrips(result: &VmResult) -> Result<()> {
    let replies = result.snapshot_bridge.drain_kernel_ops();
    anyhow::ensure!(!result.timed_out, "guest timed out under the watchdog");
    anyhow::ensure!(
        result.crash_message.is_none(),
        "guest panicked: crash_message = {:?}",
        result.crash_message,
    );
    anyhow::ensure!(
        result.exit_code == 0,
        "guest exit_code = {} (expected 0)",
        result.exit_code,
    );
    assert_task_field(&replies)?;
    assert_per_cpu_field(&replies)?;
    assert_symbol(&replies)?;
    Ok(())
}

fn find_reply<'a>(
    replies: &'a [(String, KernelOpReplyPayload)],
    tag: &str,
) -> Result<&'a KernelOpReplyPayload> {
    replies
        .iter()
        .find(|(t, _)| t == tag)
        .map(|(_, reply)| reply)
        .ok_or_else(|| {
            let tags: Vec<&str> = replies.iter().map(|(t, _)| t.as_str()).collect();
            anyhow::anyhow!("no reply for tag `{tag}`; captured={tags:?}")
        })
}

fn assert_task_field(replies: &[(String, KernelOpReplyPayload)]) -> Result<()> {
    let reply = find_reply(replies, TAG_TASK_FIELD)?;
    anyhow::ensure!(reply.success, "TaskField read rejected: {}", reply.reason);
    let value = reply
        .read_values
        .first()
        .ok_or_else(|| anyhow::anyhow!("TaskField reply read_values empty"))?;
    let observed = match value {
        KernelOpValue::U64(v) => *v,
        other => anyhow::bail!("TaskField expected U64, got {other:?}"),
    };
    anyhow::ensure!(
        observed >= SEED_VTIME_NS,
        "TaskField scx.dsq_vtime regressed: seed={SEED_VTIME_NS}, observed={observed} — \
         cold-path write did not stick, or the scheduler reset the value"
    );
    Ok(())
}

fn assert_per_cpu_field(replies: &[(String, KernelOpReplyPayload)]) -> Result<()> {
    let reply = find_reply(replies, TAG_PER_CPU_FIELD)?;
    anyhow::ensure!(
        reply.success,
        "PerCpuField read rejected: {}",
        reply.reason
    );
    let value = reply
        .read_values
        .first()
        .ok_or_else(|| anyhow::anyhow!("PerCpuField reply read_values empty"))?;
    match value {
        KernelOpValue::U32(_) => Ok(()),
        other => anyhow::bail!("PerCpuField expected U32, got {other:?}"),
    }
}

fn assert_symbol(replies: &[(String, KernelOpReplyPayload)]) -> Result<()> {
    let reply = find_reply(replies, TAG_SYMBOL)?;
    anyhow::ensure!(reply.success, "Symbol read rejected: {}", reply.reason);
    let value = reply
        .read_values
        .first()
        .ok_or_else(|| anyhow::anyhow!("Symbol reply read_values empty"))?;
    match value {
        KernelOpValue::U64(v) => {
            anyhow::ensure!(*v > 0, "jiffies_64 read as 0");
            Ok(())
        }
        other => anyhow::bail!("Symbol expected U64, got {other:?}"),
    }
}
