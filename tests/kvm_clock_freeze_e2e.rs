// Gated until B7-handler-internals (task #48) wires kvm_clock
// save/restore into freeze_coord. See cold_path_op_e2e.rs header
// for the `cfg(any())` rationale.
#![cfg(any())]

//! End-to-end coverage of kvm_clock save/restore across freeze (#30).
//!
//! The freeze coordinator parks every vCPU at the rendezvous point,
//! performs the host-side work (cold-path Ops, BPF reads, etc.),
//! then resumes. During that interval the guest's view of
//! monotonic time MUST advance — kvm_clock is the canonical
//! source on x86_64. Wire path: `KVM_GET_CLOCK` before resume,
//! `KVM_SET_CLOCK` after work, optionally `KVM_KVMCLOCK_CTRL` to
//! ack the pause to the guest.
//!
//! Three observable failure modes:
//!   1. **Clock backwards**: T1 < T0 means save/restore inverted
//!      polarity — a fresh-zero on restore would surface as
//!      "T0 huge, T1 small".
//!   2. **Clock paused-and-reset**: T1 - T0 << freeze duration
//!      means restore wrote a stale snapshot (didn't add elapsed
//!      wall-clock). Guest sees frozen time, breaks userspace
//!      timing assumptions.
//!   3. **Phantom time**: T1 - T0 >> freeze duration means restore
//!      added extra ns from a stale tsc-adjust path or
//!      KVMCLOCK_CTRL fired without proper guest acknowledgment.
//!
//! Skeleton-only.

use anyhow::Result;
use std::collections::HashMap;

use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::prelude::VmResult;
use ktstr::scenario::Ctx;
use ktstr::scenario::ops::{HoldSpec, KernelTarget, Op, Step, execute_steps};
use ktstr::test_support::{Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::named("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));

/// Lower bound on the freeze rendezvous wall-clock duration. The
/// test forces freeze to exceed this via repeated cold-ops. A
/// guest clock advance below this means restore did not apply
/// elapsed time.
const FREEZE_LOWER_NS: u64 = 1_000_000_000;

/// Upper bound on the expected guest clock advance (freeze
/// duration + epsilon). Above this means restore added phantom
/// time — a bad TSC-adjust restore or a KVMCLOCK_CTRL race.
const FREEZE_UPPER_NS: u64 = 30_000_000_000;

// ---------------------------------------------------------------------
// T30.1 — kvm_clock monotonic across freeze rendezvous.
//
// Synthetic guest-side symbol `__ktstr_test_now_ns` is populated
// by the scx-ktstr probe's tracepoint handler on every entry —
// the value is `ktime_get_ns()` (monotonic, kvm_clock-backed on
// x86_64). The test captures T0 / T1 around a deliberately-slow
// freeze block and asserts the three failure-mode bounds.
// ---------------------------------------------------------------------

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    num_snapshots = 0,
    duration_s = 15,
    watchdog_timeout_s = 30,
    auto_repro = false,
    post_vm = assert_kvm_clock_monotonic_across_freeze,
)]
fn kvm_clock_monotonic_across_freeze(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        ops: vec![
            Op::ReadKernelCold {
                tag: "t0_kvm_clock_ns".into(),
                target: KernelTarget::Symbol("__ktstr_test_now_ns".into()),
                width: 8,
            },
            // TODO(B7-impl): force a >=1s freeze rendezvous. Options:
            //   - Op::Sleep { freeze_for_ms: 1500 } (new) — the most
            //     direct route; spec a freeze-only sleep op so the
            //     test doesn't depend on workload pacing.
            //   - Or chain N Op::CaptureSnapshot rendezvous, each
            //     burning N00ms, until cumulative >= 1s.
            //   The exact mechanism is the implementer's call; the
            //   test invariant is "freeze duration observable here
            //   is >= FREEZE_LOWER_NS".
            Op::ReadKernelCold {
                tag: "t1_kvm_clock_ns".into(),
                target: KernelTarget::Symbol("__ktstr_test_now_ns".into()),
                width: 8,
            },
        ],
        hold: HoldSpec::FULL,
        ..Default::default()
    }];
    execute_steps(ctx, steps)
}

fn assert_kvm_clock_monotonic_across_freeze(r: &VmResult) -> AssertResult {
    let mut v = AssertResult::pass();
    let replies: HashMap<String, _> = r.snapshot_bridge.drain_kernel_ops().into_iter().collect();
    let (Some(t0_reply), Some(t1_reply)) = (
        replies.get("t0_kvm_clock_ns"),
        replies.get("t1_kvm_clock_ns"),
    ) else {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            "missing t0 or t1 reply — Read dispatch did not land at one or both boundaries",
        ));
        return v;
    };
    let t0 = u64::from_le_bytes(t0_reply.value[..8].try_into().unwrap());
    let t1 = u64::from_le_bytes(t1_reply.value[..8].try_into().unwrap());
    if t1 <= t0 {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "kvm_clock went BACKWARDS across freeze: t0=0x{t0:x} t1=0x{t1:x} — \
                 save/restore broke monotonicity (likely fresh-zero on restore)",
            ),
        ));
        return v;
    }
    let delta = t1 - t0;
    if delta < FREEZE_LOWER_NS {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "kvm_clock advanced only {delta}ns across >=1s freeze — \
                 clock paused-and-reset, KVM_SET_CLOCK did not apply elapsed time",
            ),
        ));
    } else if delta > FREEZE_UPPER_NS {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "kvm_clock jumped {delta}ns (>{FREEZE_UPPER_NS}ns / 30s) — \
                 restore added phantom time (bad TSC-adjust or KVMCLOCK_CTRL race)",
            ),
        ));
    }
    v
}

// ---------------------------------------------------------------------
// T30.2 — Negative control: vacuity guard.
//
// Without a knob to disable kvm_clock save/restore, T30.1 could
// vacuously pass — kvm_clock is intrinsically monotonic on KVM
// even without explicit save/restore, so the test would not
// actually exercise the new code path. This test is the negative
// control: with save/restore disabled (via cmdline knob landing
// with #42 kill-switch matrix infra), T30.1 must FAIL.
//
// If the knob doesn't land in B7's scope, taskify as a follow-up
// — the vacuity-guard concern is the second-order test that
// proves T30.1 is exercising the right code.
// ---------------------------------------------------------------------

#[test]
#[ignore = "B7-impl: vacuity-guard requires #42 kill-switch matrix infra (no-kvm-clock-restore knob); taskify if absent"]
fn kvm_clock_freeze_without_save_restore_fails() {
    // TODO(B7-impl): boot VM with ktstr.no_kvm_clock_save_restore=1
    //   cmdline knob (matches #42 kill-switch convention). Re-run
    //   the T30.1 sequence. Assert at LEAST one of the three failure
    //   modes fires — the test for the test.
    todo!("B7-impl: requires #42 cmdline knob infra; conditional on landing");
}
