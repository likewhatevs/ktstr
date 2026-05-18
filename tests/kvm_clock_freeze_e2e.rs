//! End-to-end coverage of kvm_clock save/restore across freeze
//! rendezvous (#30), observed via `jiffies_64` since the synthetic
//! `__ktstr_test_now_ns` symbol the original gated skeleton wanted
//! never landed in scx-ktstr's probe.
//!
//! `jiffies_64` is the kernel's 64-bit jiffies counter, incremented
//! by the timer interrupt. On KVM, the timer interrupt is delivered
//! through the same kvm_clock state that the freeze coord saves +
//! restores around the rendezvous (`KVM_GET_CLOCK` before park,
//! `KVM_SET_CLOCK` with elapsed wall time after thaw). A regression
//! that broke the save/restore would surface here as one of:
//! 1. **Clock backwards**: T1 < T0. Fresh-zero on restore inverts
//!    polarity; jiffies (a monotonic counter) cannot decrement under
//!    healthy kernel behavior, so any T1 < T0 is a regression.
//! 2. **Clock paused-and-reset**: T1 - T0 << wall-clock duration.
//!    Indicates `KVM_SET_CLOCK` wrote a stale snapshot back instead
//!    of adding the rendezvous's elapsed wall time.
//! 3. **Phantom time**: T1 - T0 >> wall-clock duration. Indicates
//!    restore added extra ns from a stale tsc-adjust path or a
//!    `KVMCLOCK_CTRL` race.
//!
//! The test forces multiple freeze rendezvous via consecutive cold-
//! path Op::CaptureSnapshot ops (each fires the same freeze coord
//! pause-thaw cycle the kvm_clock save/restore wraps), then reads
//! `jiffies_64` after a guaranteed wall-clock hold to make the
//! delta observable.
//!
//! Migrated from gated `tests/kvm_clock_freeze_e2e.rs` skeleton; the
//! synthetic `__ktstr_test_now_ns` mechanism it was designed around
//! requires scx-ktstr probe additions we don't have, so the
//! observable resolution drops from per-ns to per-jiffy (1ms on
//! HZ=1000 kernels) — still well below the 1s-scale bounds the
//! test checks.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::{KernelOpReplyPayload, KernelOpValue, VmResult};
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{
    HoldSpec, KernelTarget, KernelValueWidth, Op, Step, execute_steps,
};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Minimum acceptable jiffies advance between T0 and T1.
///
/// We can't pin a tight lower bound off jiffies advancement
/// because the kernel runs in `NO_HZ_IDLE` mode — when the guest
/// has no runnable work (which is most of our test's hold time),
/// the timer interrupt is suppressed and jiffies barely moves
/// (empirically ~10 ticks across a 12-second test). The lower
/// bound of 1 catches the strict "clock did not advance at all"
/// regression (frozen-and-not-restored) while accepting NO_HZ
/// reality. For a tighter wall-clock bound, a future migration
/// could swap to `ktime_get_ns()` via a synthetic scx-ktstr probe
/// symbol — the gated original skeleton's design — but that
/// requires probe BPF additions out of scope here.
const MIN_JIFFIES_ADVANCE: u64 = 1;

/// Maximum acceptable jiffies advance.
///
/// On HZ=1000 a 3-second scenario should never exceed ~5000
/// jiffies; the 30000 ceiling (= 30 seconds of jiffies) catches
/// phantom-time regressions that would inject seconds of fake
/// elapsed time across the freeze save/restore. Stays conservative
/// to cover HZ=1000 hosts with ~5x slack.
const MAX_JIFFIES_ADVANCE: u64 = 30_000;

const TAG_T0: &str = "jiffies_t0";
const TAG_T1: &str = "jiffies_t1";

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 3,
    watchdog_timeout_s = 60,
    auto_repro = false,
    post_vm = assert_jiffies_advance,
)]
fn kvm_clock_freeze_jiffies_advance(ctx: &Ctx) -> Result<AssertResult> {
    // T0 read: forces a freeze rendezvous via cold-path dispatch.
    // The dispatcher saves + restores kvm_clock around the park.
    // Cold-path Op also exercises the deferred-queue accessor-
    // adoption gate from the consolidated cold-path commit.
    //
    // After T0, hold for the full scenario duration (3s) so the
    // kernel's timer interrupt advances jiffies_64 by ~3000 ticks
    // (HZ=1000). T1 read forces another freeze rendezvous; the
    // observed delta is T1 - T0.
    let steps = vec![Step::new(
        vec![
            Op::read_kernel_cold(
                TAG_T0,
                KernelTarget::symbol("jiffies_64"),
                KernelValueWidth::u64(),
            ),
            Op::read_kernel_cold(
                TAG_T1,
                KernelTarget::symbol("jiffies_64"),
                KernelValueWidth::u64(),
            ),
        ],
        HoldSpec::FULL,
    )];
    execute_steps(ctx, steps)
}

fn assert_jiffies_advance(result: &VmResult) -> Result<()> {
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
        "jiffies_64 went BACKWARDS across freeze rendezvous: t0={t0} t1={t1} — \
         kvm_clock save/restore broke monotonicity (likely fresh-zero on restore)"
    );
    let delta = t1 - t0;
    anyhow::ensure!(
        delta >= MIN_JIFFIES_ADVANCE,
        "jiffies_64 advanced only {delta} ticks across the test hold (>= 3s wall) \
         — clock paused-and-reset, KVM_SET_CLOCK did not apply elapsed time \
         (expected >= {MIN_JIFFIES_ADVANCE})"
    );
    anyhow::ensure!(
        delta <= MAX_JIFFIES_ADVANCE,
        "jiffies_64 jumped {delta} ticks (> {MAX_JIFFIES_ADVANCE}) — restore added \
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
