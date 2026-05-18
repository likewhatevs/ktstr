// Gated until B7-handler-internals (task #48) implements the
// referenced surface (`KernelValueOp`, `WriteableField`,
// `Step.pre_step_writes`). See cold_path_op_e2e.rs header for the
// `cfg(any())` rationale.
#![cfg(any())]

//! End-to-end coverage of `Step.kernel_writes` setup slot (#29).
//!
//! The setup slot is a new `Step` field that lands kernel-memory
//! writes BEFORE the first op in the step's ops vec executes. The
//! observable invariant is per-step ordering: a sentinel value
//! installed via setup must be visible to the first op's read.
//!
//! Why the ordering matters: scenarios that pre-configure kernel
//! flags (e.g. `SCX_RQ_CLK_VALID`) and immediately run a workload
//! whose dispatch path observes those flags rely on setup having
//! landed at every CPU's view BEFORE the workload sees its first
//! schedule decision. A regression that interleaved setup with the
//! first op would silently break the precondition.
//!
//! Skeleton-only. Wires to `Step.kernel_writes` and the in-ops
//! `Op::ReadKernelCold` paired observation pattern.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{
    HoldSpec, KernelTarget, KernelValueOp, KernelWrite, Op, Step, execute_steps,
};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

// ---------------------------------------------------------------------
// T29.1 — Setup writes happen BEFORE the first op observes them.
//
// Installs a sentinel via `Step.kernel_writes`, then immediately
// reads the same symbol in `ops[0]`. The read must observe the
// sentinel value — proves setup landed before the first op
// dispatch entered freeze rendezvous.
//
// A regression that lazily applied setup (e.g. deferred to the
// step's HoldSpec exit) would surface here as "first op read 0x0,
// setup did not execute before first op".
// ---------------------------------------------------------------------

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    num_snapshots = 0,
    duration_s = 5,
    watchdog_timeout_s = 15,
    auto_repro = false,
    post_vm = assert_setup_before_ops,
)]
fn setup_writes_happen_before_first_op(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        // TODO(B7-impl): `kernel_writes` is the new Step field that
        //   lands with #29. The semantics: every write here applies
        //   BEFORE ops[0] dispatches.
        kernel_writes: vec![KernelWrite {
            target: KernelTarget::Symbol("__ktstr_test_sentinel".into()),
            value: KernelValueOp::Set(ktstr::scenario::ops::KernelValue::U64(0x5E70_BEEF)),
        }],
        ops: vec![Op::ReadKernelCold {
            tag: "first_op_reads_setup_sentinel".into(),
            target: KernelTarget::Symbol("__ktstr_test_sentinel".into()),
            width: 8,
        }],
        hold: HoldSpec::FULL,
        ..Default::default()
    }];
    execute_steps(ctx, steps)
}

fn assert_setup_before_ops(r: &VmResult) -> AssertResult {
    let mut v = AssertResult::pass();
    let Some((_tag, reply)) = r
        .snapshot_bridge
        .drain_kernel_ops()
        .into_iter()
        .find(|(t, _)| t == "first_op_reads_setup_sentinel")
    else {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            "expected reply tagged `first_op_reads_setup_sentinel` not captured",
        ));
        return v;
    };
    if reply.value.len() < 8 {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!("reply too short: {} bytes", reply.value.len()),
        ));
        return v;
    }
    let read = u64::from_le_bytes(reply.value[..8].try_into().unwrap());
    if read != 0x5E70_BEEF {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "first op read 0x{read:x}; expected 0x5E70_BEEF — \
                 setup_slot did not execute before first op",
            ),
        ));
    }
    v
}

// ---------------------------------------------------------------------
// T29.2 — Empty setup is a no-op (no spurious dispatch).
//
// `Step { kernel_writes: vec![], ops: vec![...], .. }` must NOT
// produce any setup-phase wire traffic. Pins that the default-
// empty case doesn't accidentally inject a sentinel write or
// trigger a freeze rendezvous before ops[0].
// ---------------------------------------------------------------------

#[test]
fn empty_setup_produces_no_dispatch() {
    // TODO(B7-impl): unit-level via MockSnapshotBridge. Build a Step
    //   with kernel_writes: vec![] + ops: vec![] + assert
    //   bridge.captured_requests.len() == 0 (no kernel-op traffic at
    //   all). Empty setup must be a structural no-op.
    let _step = Step {
        kernel_writes: vec![],
        ops: vec![],
        hold: HoldSpec::FULL,
        ..Default::default()
    };
    todo!("B7-impl: assert bridge.captured_requests is empty");
}
