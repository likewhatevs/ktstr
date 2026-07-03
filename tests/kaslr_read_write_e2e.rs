//! Op::ReadKernel / Op::WriteKernel round-trip pins under KASLR-on.
//! Exercises the kaslr_offset thread-through in kernel_op_dispatch.rs
//! on both the read AND the write paths via:
//!
//! 1. Symbol read — `jiffies_64` (.data u64); validates the kernel-
//!    text slide path for an in-image global.
//! 2. Symbol write round-trip — `panic_on_oops` (kernel/panic.c, u32);
//!    read pre, write sentinel, read post. Pin: post == SENTINEL.
//!    Mutation is harmless past test teardown (controls oops behavior
//!    only).
//! 3. Per-CPU OrU32 round-trip — `runqueues.scx.flags` cpu 0; HIGH_BIT
//!    is reserved test space (sched_ext core touches only the low bits
//!    defined by `enum scx_rq_flags`). Pin: post & HIGH_BIT == HIGH_BIT AND
//!    (post ^ pre) == HIGH_BIT (no scheduler-side bits drifted).
//!
//! All assertions are EXPLICIT-VALUE (no panic-on-bad-read). The
//! chain is silent-on-wrong-offset (read_scalar
//! returns zeros, kva_to_pa wraps), so the test infrastructure must
//! distinguish "wrong-PA-zero" from "correct-PA-zero" via either
//! (a) a sentinel write that produces a known non-zero value, OR
//! (b) a kaslr-on precondition combined with plausibility bounds.

use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::prelude::*;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{
    HoldSpec, KernelTarget, KernelValue, KernelValueWidth, Op, Step, execute_steps,
};

declare_scheduler!(KTSTR_SCHED, {
    name = "ktstr_sched",
    binary = "scx-ktstr",
});

// =========================================================================
// Op::WriteKernelCold(symbol) round-trip — panic_on_oops.
// =========================================================================

const TAG_PRE: &str = "panic_on_oops_pre";
const TAG_POST: &str = "panic_on_oops_post";
const SENTINEL: u32 = 0xDEAD_BEEF;

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 3,
    auto_repro = false,
    post_vm = assert_panic_on_oops_roundtrip,
)]
fn kaslr_write_symbol_roundtrip_panic_on_oops(ctx: &Ctx) -> Result<AssertResult> {
    let target = || KernelTarget::symbol("panic_on_oops");
    let steps = vec![Step::new(
        vec![
            Op::read_kernel_cold(TAG_PRE, target(), KernelValueWidth::u32()),
            Op::write_kernel_cold(target(), KernelValue::u32(SENTINEL)),
            Op::read_kernel_cold(TAG_POST, target(), KernelValueWidth::u32()),
        ],
        HoldSpec::FULL,
    )];
    execute_steps(ctx, steps)
}

fn assert_panic_on_oops_roundtrip(result: &VmResult) -> Result<()> {
    // A scheduler that never attached (SchedulerNotAttached, e.g. scx
    // enable stalled under host oversubscription) yields success ==
    // false with an empty bridge; the runner already renders that with
    // the real scheduler-log cause. Return early so the assertions below
    // don't fire a misleading "kern_kaslr_offset == 0" / "no reply for
    // tag" on top of it (mirrors default_post_vm_periodic_fired).
    if !result.success {
        return Ok(());
    }
    anyhow::ensure!(
        result.kern_kaslr_offset != 0,
        "kern_kaslr_offset == 0 — test must run under KASLR-on to \
         validate slide-aware symbol resolution for writes"
    );
    let replies = result.snapshot_bridge.drain_kernel_ops();
    let find = |tag: &str| {
        replies
            .iter()
            .find(|(t, _)| t == tag)
            .map(|(_, r)| r)
            .ok_or_else(|| anyhow::anyhow!("no reply for tag '{tag}'"))
    };
    let pre = find(TAG_PRE)?;
    let post = find(TAG_POST)?;
    anyhow::ensure!(pre.success, "pre read failed: {}", pre.reason);
    anyhow::ensure!(post.success, "post read failed: {}", post.reason);
    let pre_val = match pre.read_values.first() {
        Some(KernelOpValue::U32(v)) => *v,
        other => anyhow::bail!("pre read_values[0] not U32: {other:?}"),
    };
    let post_val = match post.read_values.first() {
        Some(KernelOpValue::U32(v)) => *v,
        other => anyhow::bail!("post read_values[0] not U32: {other:?}"),
    };
    anyhow::ensure!(
        pre_val <= 2,
        "pre panic_on_oops = {pre_val} (sane values are 0, 1, 2) — \
         read PA wrong under KASLR-on"
    );
    anyhow::ensure!(
        post_val == SENTINEL,
        "post panic_on_oops = {post_val:#x}, expected SENTINEL = \
         {SENTINEL:#x}. If post == pre ({pre_val}), the write \
         missed. Otherwise read and write resolved to different \
         PAs (asymmetric slide application — the slide thread-through \
         did not reach the write side OR direct-symbol resolution \
         lost the slide)"
    );
    Ok(())
}

// =========================================================================
// Op::ReadKernelCold(symbol("jiffies_64")) under KASLR-on.
// =========================================================================

const TAG_JIFFIES: &str = "jiffies_64_read";

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 3,
    auto_repro = false,
    post_vm = assert_jiffies_read,
)]
fn kaslr_read_symbol_jiffies_64(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::new(
        vec![Op::read_kernel_cold(
            TAG_JIFFIES,
            KernelTarget::symbol("jiffies_64"),
            KernelValueWidth::u64(),
        )],
        HoldSpec::FULL,
    )];
    execute_steps(ctx, steps)
}

fn assert_jiffies_read(result: &VmResult) -> Result<()> {
    // See assert_panic_on_oops_roundtrip: skip the KASLR/reply checks
    // when the scheduler never attached, so the real failure surfaces.
    if !result.success {
        return Ok(());
    }
    anyhow::ensure!(result.kern_kaslr_offset != 0, "KASLR-on required");
    let replies = result.snapshot_bridge.drain_kernel_ops();
    let reply = replies
        .iter()
        .find(|(t, _)| t == TAG_JIFFIES)
        .map(|(_, r)| r)
        .ok_or_else(|| anyhow::anyhow!("no jiffies reply"))?;
    anyhow::ensure!(reply.success, "jiffies read failed: {}", reply.reason);
    let v = match reply.read_values.first() {
        Some(KernelOpValue::U64(v)) => *v,
        other => anyhow::bail!("expected U64, got {other:?}"),
    };
    anyhow::ensure!(v != 0, "jiffies_64 = 0 — wrong PA or pre-boot read");
    anyhow::ensure!(
        v != u64::MAX,
        "jiffies_64 = u64::MAX — uninitialized slab page"
    );
    eprintln!("jiffies_64 = {v} (raw) — KASLR-on read succeeded");
    Ok(())
}

// =========================================================================
// Op::WriteKernelCold(per_cpu_field, OrU32) round-trip — exercises the
// per-CPU write path under KASLR-on. Target: runqueues.scx.flags cpu 0;
// HIGH_BIT (bit 31) is reserved test space (sched_ext core touches the
// low bits only per `enum scx_rq_flags`).
// =========================================================================

const TAG_PCPU_PRE: &str = "rq_flags_pre";
const TAG_PCPU_POST: &str = "rq_flags_post";
const HIGH_BIT: u32 = 1 << 31;

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 3,
    auto_repro = false,
    post_vm = assert_rq_flags_or_roundtrip,
)]
fn kaslr_write_per_cpu_field_or_roundtrip(ctx: &Ctx) -> Result<AssertResult> {
    let target = || KernelTarget::per_cpu_field("runqueues", "scx.flags", 0);
    let steps = vec![Step::new(
        vec![
            Op::read_kernel_cold(TAG_PCPU_PRE, target(), KernelValueWidth::u32()),
            Op::write_kernel_cold(target(), KernelValue::or_u32(HIGH_BIT)),
            Op::read_kernel_cold(TAG_PCPU_POST, target(), KernelValueWidth::u32()),
        ],
        HoldSpec::FULL,
    )];
    execute_steps(ctx, steps)
}

fn assert_rq_flags_or_roundtrip(result: &VmResult) -> Result<()> {
    // See assert_panic_on_oops_roundtrip: skip the KASLR/reply checks
    // when the scheduler never attached, so the real failure surfaces.
    if !result.success {
        return Ok(());
    }
    anyhow::ensure!(
        result.kern_kaslr_offset != 0,
        "kern_kaslr_offset == 0 — test requires KASLR-on"
    );
    let replies = result.snapshot_bridge.drain_kernel_ops();
    let find = |tag: &str| {
        replies
            .iter()
            .find(|(t, _)| t == tag)
            .map(|(_, r)| r)
            .ok_or_else(|| anyhow::anyhow!("no reply for tag '{tag}'"))
    };
    let pre = find(TAG_PCPU_PRE)?;
    let post = find(TAG_PCPU_POST)?;
    anyhow::ensure!(pre.success, "pre read failed: {}", pre.reason);
    anyhow::ensure!(post.success, "post read failed: {}", post.reason);
    let pre_v = match pre.read_values.first() {
        Some(KernelOpValue::U32(v)) => *v,
        other => anyhow::bail!("pre not U32: {other:?}"),
    };
    let post_v = match post.read_values.first() {
        Some(KernelOpValue::U32(v)) => *v,
        other => anyhow::bail!("post not U32: {other:?}"),
    };
    anyhow::ensure!(
        pre_v < (1 << 16),
        "pre rq.scx.flags = {pre_v:#x} (>16 bits) — likely wrong PA"
    );
    anyhow::ensure!(
        (post_v & HIGH_BIT) == HIGH_BIT,
        "post rq.scx.flags = {post_v:#x} missing HIGH_BIT {HIGH_BIT:#x} — \
         OrU32 RMW did not land. Either the read leg or the write leg \
         resolved to the wrong PA under KASLR-on."
    );
    anyhow::ensure!(
        (post_v ^ pre_v) == HIGH_BIT,
        "post ^ pre = {:#x}, expected {HIGH_BIT:#x} — other bits \
         changed between the write and post-read rendezvous while \
         vCPUs ran (each cold op is a separate rendezvous).",
        post_v ^ pre_v
    );
    Ok(())
}

// =========================================================================
// rq.cpu ground-truth — exercises the per-CPU read path under KASLR-on
// for TWO distinct CPUs (cpu 0 + cpu 1). The field rq.cpu is set ONCE
// by sched_init at boot to the CPU index. Pin: cpu0_read == 0 AND
// cpu1_read == 1. Without cpu1 the cpu0==0 assertion is vacuous (any
// wrong-PA zero would also match).
// =========================================================================

const TAG_RQ_CPU0: &str = "rq_cpu_index_0";
const TAG_RQ_CPU1: &str = "rq_cpu_index_1";

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    duration_s = 3,
    auto_repro = false,
    post_vm = assert_rq_cpu_field,
)]
fn kaslr_compute_rq_pas_ground_truth(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::new(
        vec![
            Op::read_kernel_cold(
                TAG_RQ_CPU0,
                KernelTarget::per_cpu_field("runqueues", "cpu", 0),
                KernelValueWidth::u32(),
            ),
            Op::read_kernel_cold(
                TAG_RQ_CPU1,
                KernelTarget::per_cpu_field("runqueues", "cpu", 1),
                KernelValueWidth::u32(),
            ),
        ],
        HoldSpec::FULL,
    )];
    execute_steps(ctx, steps)
}

fn assert_rq_cpu_field(result: &VmResult) -> Result<()> {
    // See assert_panic_on_oops_roundtrip: skip the KASLR/reply checks
    // when the scheduler never attached, so the real failure surfaces.
    if !result.success {
        return Ok(());
    }
    anyhow::ensure!(
        result.kern_kaslr_offset != 0,
        "KASLR-on required to validate compute_rq_pas under slide"
    );
    let replies = result.snapshot_bridge.drain_kernel_ops();
    let read = |tag: &str| -> Result<u32> {
        let r = replies
            .iter()
            .find(|(t, _)| t == tag)
            .map(|(_, r)| r)
            .ok_or_else(|| anyhow::anyhow!("no reply for {tag}"))?;
        anyhow::ensure!(r.success, "{tag} read failed: {}", r.reason);
        match r.read_values.first() {
            Some(KernelOpValue::U32(v)) => Ok(*v),
            other => anyhow::bail!("{tag} not U32: {other:?}"),
        }
    };
    let cpu0 = read(TAG_RQ_CPU0)?;
    let cpu1 = read(TAG_RQ_CPU1)?;
    anyhow::ensure!(
        cpu0 == 0,
        "rq.cpu at cpu 0 = {cpu0} (expected 0). compute_rq_pas resolved \
         to wrong per-CPU page OR the slide thread-through did not \
         reach the per-CPU read path."
    );
    anyhow::ensure!(
        cpu1 == 1,
        "rq.cpu at cpu 1 = {cpu1} (expected 1). per-CPU offset array \
         indexing wrong (or only 1 CPU in this test fixture)."
    );
    Ok(())
}
