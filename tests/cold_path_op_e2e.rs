// Gated until B7-handler-internals (task #48) implements the
// referenced surface (`KernelValueOp`, `WriteableField`,
// `KernelOpError`, `Step.pre_step_writes`). `cfg(any())` is the
// canonical "always-false" — the file is parsed but not type-
// checked, so the test contract stays visible while the impl is
// in flight.
#![cfg(any())]

//! End-to-end coverage of the cold-path Op handler (#19).
//!
//! Cold-path Ops (`Op::WriteKernelCold` / `Op::ReadKernelCold`)
//! dispatch kernel-memory write/read requests through the
//! freeze-coordinator while every vCPU is parked at the freeze
//! rendezvous. The encoder runs host-side and synthesizes a
//! `KernelOpRequestPayload`; the dispatch consumer at
//! `src/vmm/freeze_coord/dispatch.rs:265` decodes it and walks the
//! frozen guest memory.
//!
//! These tests pin three orthogonal invariants:
//!   1. **Encoder shape** (T19.1, unit): the apply_ops path
//!      produces a payload whose `writes` vec preserves the user-
//!      supplied target / value tuples without reordering or
//!      variant coercion.
//!   2. **Roundtrip observability** (T19.2, e2e): a write at a
//!      known-symbol target is observable via a paired read in the
//!      same step — proves the cold-path actually mutates frozen
//!      guest memory and the kernel sees it after resume. Uses a
//!      synthetic test symbol (`__ktstr_test_word`) so the test
//!      never stomps on a real kernel clocksource.
//!   3. **Hostile-input rejection** (T19.3, unit): wrong-half PAs,
//!      unmapped kernel-half PAs, and `u64::MAX` overflow boundaries
//!      must surface as typed `KernelOpError` variants — never as a
//!      silent zero-read. Extends the KernAddrs byte-layout pin
//!      pattern at `src/vmm/wire.rs:1741`.
//!
//! Skeleton-only. Bodies wire to types (`KernelWrite`,
//! `KernelTarget::Symbol`, `KernelOpError::PaWrongHalf` etc.) that
//! land with B7's implementation. The placeholders are intentional —
//! the test file is the explicit pre-impl design surface.

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
// T19.1 — Unit: Op::WriteKernelCold synthesizes correct request payload.
//
// Constructs a 3-write Op::WriteKernelCold with varied target
// addresses + KernelValueOp variants. Routes through apply_op with
// a mock SnapshotBridge that captures the dispatched
// `KernelOpRequestPayload`. Asserts writes.len() + per-write target
// + per-write value variant survive the encoder unchanged.
//
// Pins the encoder side without booting a guest. A regression that
// reordered writes, coerced variants (e.g. silently promoted U8 to
// U64), or dropped the last write would surface here.
// ---------------------------------------------------------------------

#[test]
fn op_write_kernel_cold_dispatches_writes_payload() {
    // TODO(B7-impl): construct MockSnapshotBridge that records every
    //   dispatched KernelOpRequestPayload; route apply_op through it.
    //   The mock infrastructure lands with #19; this test signature
    //   pins the surface name.
    let _writes = vec![
        KernelWrite {
            target: KernelTarget::PhysicalAddress(0x1000),
            value: KernelValueOp::Set(ktstr::scenario::ops::KernelValue::U64(0xDEAD_BEEF)),
        },
        KernelWrite {
            target: KernelTarget::PhysicalAddress(0x2000),
            value: KernelValueOp::Set(ktstr::scenario::ops::KernelValue::U32(0xCAFE)),
        },
        KernelWrite {
            target: KernelTarget::PhysicalAddress(0x3008),
            value: KernelValueOp::Set(ktstr::scenario::ops::KernelValue::U8(0xAB)),
        },
    ];
    // Assertions (post-impl):
    //   - bridge.captured_requests.len() == 1
    //   - bridge.captured_requests[0].writes.len() == 3
    //   - matches!(writes[0].value, KernelValueOp::Set(KernelValue::U64(0xDEAD_BEEF)))
    //   - matches!(writes[1].value, KernelValueOp::Set(KernelValue::U32(0xCAFE)))
    //   - matches!(writes[2].value, KernelValueOp::Set(KernelValue::U8(0xAB)))
    //   - writes[N].target preserves PhysicalAddress payload byte-equal.
    todo!("B7-impl: wire MockSnapshotBridge + assertions");
}

// ---------------------------------------------------------------------
// T19.2 — E2E roundtrip: write → freeze → resume → kernel observes.
//
// Boots a real VM with a synthetic test symbol `__ktstr_test_word`
// added to the scx-ktstr probe BSS. Dispatches a paired
// Write/Read sequence in one Step: the write lands during cold-path
// freeze rendezvous; the subsequent read re-enters freeze and pulls
// the now-stored value back. The post_vm callback drains the
// snapshot bridge's kernel-op reply log and asserts the read
// observed the sentinel value the write installed.
//
// This is the load-bearing pin for #19: a regression that
// silently no-op'd the write (e.g. KernelOpMode::Cold lookup
// resolved the wrong bridge callback) would surface as
// "guest read 0x0, expected 0xDEAD_BEEF_CAFE_BABE" — a clear
// operator-facing diagnostic vs. silent green test.
// ---------------------------------------------------------------------

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    num_snapshots = 0,
    duration_s = 5,
    watchdog_timeout_s = 15,
    auto_repro = false,
    post_vm = assert_cold_write_roundtrip,
)]
fn cold_write_roundtrip_observes_in_guest(ctx: &Ctx) -> Result<AssertResult> {
    // TODO(B7-impl): `Step.kernel_writes` is the new Setup-slot field
    //   landing with #29; this test uses the in-ops Write/Read form
    //   directly since the roundtrip observation is the
    //   payload-level invariant, not the setup-ordering one.
    let steps = vec![Step {
        ops: vec![
            Op::WriteKernelCold {
                writes: vec![KernelWrite {
                    target: KernelTarget::Symbol("__ktstr_test_word".into()),
                    value: KernelValueOp::Set(
                        ktstr::scenario::ops::KernelValue::U64(0xDEAD_BEEF_CAFE_BABE),
                    ),
                }],
            },
            Op::ReadKernelCold {
                tag: "verify".into(),
                target: KernelTarget::Symbol("__ktstr_test_word".into()),
                width: 8,
            },
        ],
        hold: HoldSpec::FULL,
        ..Default::default()
    }];
    execute_steps(ctx, steps)
}

fn assert_cold_write_roundtrip(r: &VmResult) -> AssertResult {
    let mut v = AssertResult::pass();
    let replies = r.snapshot_bridge.drain_kernel_ops();
    let Some((_tag, reply)) = replies.into_iter().find(|(t, _)| t == "verify") else {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            "no `verify` read reply captured — cold-path Read dispatch did not land",
        ));
        return v;
    };
    if reply.value.len() < 8 {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "reply value too short: got {} bytes, expected >= 8",
                reply.value.len()
            ),
        ));
        return v;
    }
    let read_u64 = u64::from_le_bytes(reply.value[..8].try_into().expect("8-byte slice"));
    if read_u64 != 0xDEAD_BEEF_CAFE_BABE {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "guest read 0x{read_u64:016x}, expected 0xDEAD_BEEF_CAFE_BABE — \
                 cold-path Write did not mutate guest memory or Read read the \
                 wrong PA",
            ),
        ));
    }
    v
}

// ---------------------------------------------------------------------
// T19.3 — Error paths: PA validation rejection.
//
// Three orthogonal unit tests pinning the typed-error contract:
//   - Wrong-half PA (user-half address) rejected before dispatch.
//   - Kernel-half-but-unmapped PA rejected after dispatch (the
//     decoder must surface "unmapped" distinctly from "zero-read").
//   - u64::MAX overflow boundary: pa + width wraps, must reject.
//
// Crucial: the rejection-vs-zero-read distinction is the silent-
// data-corruption defense. A regression that bounds-rejected to a
// zero read would compile and pass roundtrip tests on benign inputs
// but silently lose data on hostile inputs.
// ---------------------------------------------------------------------

#[test]
fn op_write_kernel_cold_rejects_user_half_pa() {
    // TODO(B7-impl): route apply_op through MockSnapshotBridge; assert
    //   the returned Err matches KernelOpError::PaWrongHalf. The
    //   wrong-half PA is the canonical user-mapped boundary —
    //   anything below 0xffff_8000_0000_0000 on x86_64.
    let _user_half_pa: u64 = 0x4000_0000_0000;
    todo!("B7-impl: assert Err(KernelOpError::PaWrongHalf)");
}

#[test]
fn op_write_kernel_cold_rejects_unmapped_kernel_half_pa() {
    // TODO(B7-impl): kernel-half PA outside guest DRAM region; the
    //   decoder's bounds check must surface as PaUnmapped (NOT
    //   silently zero-read).
    let _unmapped_kernel_pa: u64 = 0xffff_ffff_ffff_e000;
    todo!("B7-impl: assert Err(KernelOpError::PaUnmapped)");
}

#[test]
fn op_write_kernel_cold_rejects_pa_overflow_at_u64_max() {
    // TODO(B7-impl): pa + width wraps under wrapping_add; explicit
    //   reject required so the decoder never reads from address 0
    //   after a malicious wrap.
    let _pa: u64 = u64::MAX;
    let _width: usize = 8;
    todo!("B7-impl: assert Err(KernelOpError::PaOverflow)");
}
