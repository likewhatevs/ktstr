//! End-to-end coverage of kvm_clock save/restore across freeze
//! rendezvous. Observed via `runqueues.clock` (per-rq
//! sched_clock-ns counter, TSC-backed on KVM x86) because the
//! synthetic `__ktstr_test_now_ns` symbol the original gated
//! skeleton wanted never landed in scx-ktstr's probe.
//!
//! ## Coverage delta from the original design
//!
//! The original gated skeleton wanted to detect three failure
//! classes: (1) clock backwards, (2) clock paused-and-reset, (3)
//! phantom time. The migration covers (1) and (3) but NOT (2):
//!
//! - **(1) Clock backwards** — covered. T1 >= T0 monotonicity
//!   assertion catches any fresh-zero restore.
//! - **(2) Clock paused-and-reset** — NOT covered by this test.
//!   Requires an observable that advances at wall-clock rate
//!   regardless of guest scheduler activity. `rq.clock` only
//!   advances inside `update_rq_clock()` (called on scheduler
//!   dispatch); under scx-ktstr the dispatch is infrequent, so
//!   `rq.clock` advances by ~14ms across a 3s hold — too noisy to
//!   distinguish "paused-and-reset" from normal sparse dispatch.
//!   `jiffies_64` was attempted first but `NO_HZ_IDLE` suppresses
//!   it. A proper fix requires the synthetic probe symbol the
//!   original skeleton was designed around (`ktime_get_ns()` on
//!   every scx-ktstr tracepoint entry) OR a different observable
//!   that advances unconditionally with wall time (e.g. raw
//!   `tk_core.timekeeper.tkr_mono.base` via a BTF nested-path
//!   read).
//! - **(3) Phantom time** — covered. T1 - T0 < 30s ceiling catches
//!   any restore that injects multi-second fake advancement.
//!
//! ## How the test exercises kvm_clock save/restore
//!
//! Each `Op::ReadKernelCold` triggers a freeze rendezvous; the
//! freeze coord saves guest TSC via `KVM_GET_CLOCK` before parking
//! vCPUs and restores via `KVM_SET_CLOCK` with elapsed wall time
//! after thaw. The two reads bracket a 3s hold; a regression that
//! corrupts the save/restore polarity surfaces as T1 < T0; a
//! regression that injects extra time surfaces as T1 - T0 >> 30s.
//!
//! ## Migration note
//!
//! Migrated from gated `tests/kvm_clock_freeze_e2e.rs` skeleton.
//! The synthetic `__ktstr_test_now_ns` mechanism it was designed
//! around requires scx-ktstr probe additions out of scope here.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{KernelOpReplyPayload, KernelOpValue, VmResult};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{
    HoldSpec, KernelTarget, KernelValueWidth, Op, Step, execute_steps,
};
use ktstr::test_support::{Scheduler, SchedulerSpec};
use ktstr::workload::{
    AffinityIntent, SchedPolicy, WorkType, WorkloadConfig, WorkloadHandle,
};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Maximum acceptable `rq.clock` advance. The 30-second ceiling
/// (10x the 3s hold) catches phantom-time regressions that would
/// inject seconds of fake elapsed time during the freeze
/// save/restore (a bad TSC-adjust path or `KVMCLOCK_CTRL` race).
///
/// No matching MIN bound is enforced — see the doc-comment
/// header's "Coverage delta" section: scx-ktstr's sparse dispatch
/// means `rq.clock` only advances by ~14ms across a 3s hold even
/// with oversubscribed runnable load, which is indistinguishable
/// from a paused-and-reset regression. A proper MIN check needs a
/// wall-clock-rate observable that doesn't depend on scheduler
/// dispatch (synthetic probe symbol from the original skeleton's
/// design, or BTF nested-path read of `tk_core.timekeeper.tkr_mono.base`).
const MAX_CLOCK_ADVANCE_NS: u64 = 30_000_000_000;

const TAG_T0: &str = "jiffies_t0";
const TAG_T1: &str = "jiffies_t1";

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    llcs = 1,
    cores = 2,
    threads = 1,
    duration_s = 3,
    watchdog_timeout_s = 60,
    auto_repro = false,
    post_vm = assert_clock_advance,
)]
fn kvm_clock_freeze_clock_advance(ctx: &Ctx) -> Result<AssertResult> {
    // Oversubscribe vCPUs 4:1 with spinners so the scheduler MUST
    // round-robin dispatch on every CPU — each context switch
    // calls `update_rq_clock()`, which is what advances rq.clock.
    // A single spinner per CPU stays running but rarely triggers a
    // dispatch, so rq.clock advances only when scx-ktstr's tick
    // observation fires (rare). Oversubscription forces continuous
    // dispatch.
    let total_cpus = ctx.topo.total_cpus();
    let config = WorkloadConfig {
        num_workers: total_cpus * 4,
        affinity: AffinityIntent::Inherit,
        work_type: WorkType::YieldHeavy,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    // Give the scheduler one dispatch round so rq.clock is in
    // steady-state by T0 capture.
    std::thread::sleep(std::time::Duration::from_millis(100));
    // T0 read: forces a freeze rendezvous via cold-path dispatch.
    // The dispatcher saves + restores kvm_clock (via KVM_GET_CLOCK
    // / KVM_SET_CLOCK) around the park; TSC-derived rq.clock
    // follows the same save/restore path.
    //
    // After T0, hold for the full scenario duration (3s). T1 read
    // forces another freeze rendezvous; the observed delta is
    // T1 - T0 in nanoseconds.
    let steps = vec![Step::new(
        vec![
            Op::read_kernel_cold(
                TAG_T0,
                KernelTarget::per_cpu_field("runqueues", "clock", 1),
                KernelValueWidth::u64(),
            ),
            Op::read_kernel_cold(
                TAG_T1,
                KernelTarget::per_cpu_field("runqueues", "clock", 1),
                KernelValueWidth::u64(),
            ),
        ],
        HoldSpec::FULL,
    )];
    let result = execute_steps(ctx, steps);
    let _ = handle.stop_and_collect();
    result
}

fn assert_clock_advance(result: &VmResult) -> Result<()> {
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
    let replies = result.snapshot_bridge.drain_kernel_ops();
    let t0 = read_u64_tag(&replies, TAG_T0)?;
    let t1 = read_u64_tag(&replies, TAG_T1)?;
    anyhow::ensure!(
        t1 >= t0,
        "rq.clock went BACKWARDS across freeze rendezvous: t0={t0}ns t1={t1}ns \
         — kvm_clock save/restore broke monotonicity (likely fresh-zero on restore)"
    );
    let delta = t1 - t0;
    anyhow::ensure!(
        delta <= MAX_CLOCK_ADVANCE_NS,
        "rq.clock jumped {delta}ns (> {MAX_CLOCK_ADVANCE_NS}ns) — restore added \
         phantom time (bad TSC-adjust or KVMCLOCK_CTRL race)"
    );
    Ok(())
}

fn read_u64_tag(replies: &[(String, KernelOpReplyPayload)], tag: &str) -> Result<u64> {
    let (_t, reply) = replies
        .iter()
        .find(|(t, _)| t == tag)
        .ok_or_else(|| {
            let tags: Vec<&str> = replies.iter().map(|(t, _)| t.as_str()).collect();
            anyhow::anyhow!("no reply for tag `{tag}`; captured={tags:?}")
        })?;
    anyhow::ensure!(reply.success, "{tag} read rejected: {}", reply.reason);
    match reply.read_values.first() {
        Some(KernelOpValue::U64(v)) => Ok(*v),
        Some(other) => anyhow::bail!("{tag} expected U64, got {other:?}"),
        None => anyhow::bail!("{tag} reply read_values empty"),
    }
}
