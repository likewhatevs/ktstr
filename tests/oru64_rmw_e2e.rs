// Gated until B7-handler-internals (task #48) implements the
// referenced surface. See cold_path_op_e2e.rs header for the
// `cfg(any())` rationale.
#![cfg(any())]

//! End-to-end coverage of `KernelValueOp::OrU64` RMW semantics (#33).
//!
//! `OrU64(mask)` performs `*pa |= mask` at the target — a read-
//! modify-write that sets the supplied bits without clobbering
//! the other bits already in the word. The classic use case is
//! `SCX_RQ_CLK_VALID` and other per-rq flag bits where the host
//! must enable a feature without racing the scheduler's own
//! writes to neighboring bits.
//!
//! Three invariants pinned here:
//!   1. **Bit-set without other-bit clobber** (T33.1): pre-populate
//!      a flag word with alternating bits, OrU64 a single new bit,
//!      verify the union — not just the OR mask alone.
//!   2. **Idempotent on already-set bit** (T33.2): OrU64 a bit
//!      that's already set must produce the unchanged value. Pins
//!      that the RMW path doesn't accidentally toggle / clear.
//!   3. **Postcard roundtrip** (T33.3): the `KernelValueOp::OrU64`
//!      variant must survive postcard encode/decode unchanged —
//!      guards against a `#[serde(untagged)]` regression on
//!      `KernelValueOp` (the #37 audit pattern).
//!
//! Skeleton-only.

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

/// Pre-populated alternating bit pattern. Chosen so a bug that
/// returned just the OR mask (instead of the union) would surface
/// as 0x0000_0001_0000_0000 != 0xAAAA_AAAB_AAAA_AAAA.
const INITIAL_FLAGS: u64 = 0xAAAA_AAAA_AAAA_AAAA;

/// Single bit (bit 32) deliberately NOT set in INITIAL_FLAGS so
/// the OR is observable. Choice of bit 32 picks a position whose
/// presence is unambiguous on hex inspection.
const OR_MASK: u64 = 0x0000_0001_0000_0000;

// ---------------------------------------------------------------------
// T33.1 — Host OR'd bit visible to kernel, other bits preserved.
// ---------------------------------------------------------------------

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    num_snapshots = 0,
    duration_s = 5,
    watchdog_timeout_s = 15,
    auto_repro = false,
    post_vm = assert_oru64_preserves_other_bits,
)]
fn oru64_sets_target_bit_preserves_others(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        // TODO(B7-impl): Step.kernel_writes (the #29 setup slot)
        //   pre-installs the alternating pattern so the OR has a
        //   non-trivial baseline to merge against.
        kernel_writes: vec![KernelWrite {
            target: KernelTarget::Symbol("__ktstr_test_flags".into()),
            value: KernelValueOp::Set(ktstr::scenario::ops::KernelValue::U64(INITIAL_FLAGS)),
        }],
        ops: vec![
            Op::WriteKernelCold {
                writes: vec![KernelWrite {
                    target: KernelTarget::Symbol("__ktstr_test_flags".into()),
                    value: KernelValueOp::OrU64(OR_MASK),
                }],
            },
            Op::ReadKernelCold {
                tag: "post_or".into(),
                target: KernelTarget::Symbol("__ktstr_test_flags".into()),
                width: 8,
            },
        ],
        hold: HoldSpec::FULL,
        ..Default::default()
    }];
    execute_steps(ctx, steps)
}

fn assert_oru64_preserves_other_bits(r: &VmResult) -> AssertResult {
    let mut v = AssertResult::pass();
    let Some((_tag, reply)) = r
        .snapshot_bridge
        .drain_kernel_ops()
        .into_iter()
        .find(|(t, _)| t == "post_or")
    else {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            "no `post_or` reply captured",
        ));
        return v;
    };
    let observed = u64::from_le_bytes(reply.value[..8].try_into().unwrap());
    let expected = INITIAL_FLAGS | OR_MASK;
    if observed != expected {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "OrU64 observed=0x{observed:016x} expected=0x{expected:016x} — \
                 wrong bits set or other bits clobbered",
            ),
        ));
    }
    v
}

// ---------------------------------------------------------------------
// T33.2 — Idempotent: OrU64 with already-set bit leaves value
// unchanged. Pins the "redundant bit" RMW path so a regression
// that toggled / cleared on already-set wouldn't slip past T33.1
// (which uses a new bit).
// ---------------------------------------------------------------------

#[ktstr_test(
    scheduler = KTSTR_SCHED,
    num_snapshots = 0,
    duration_s = 5,
    watchdog_timeout_s = 15,
    auto_repro = false,
    post_vm = assert_oru64_already_set_is_noop,
)]
fn oru64_already_set_bit_is_noop(ctx: &Ctx) -> Result<AssertResult> {
    // Choose a bit that IS set in INITIAL_FLAGS (bit 1 — high in
    // 0xA = 1010). OR'ing it should leave the value unchanged.
    let already_set: u64 = 0x0000_0000_0000_0002;
    assert_eq!(INITIAL_FLAGS & already_set, already_set, "test setup bug");
    let steps = vec![Step {
        kernel_writes: vec![KernelWrite {
            target: KernelTarget::Symbol("__ktstr_test_flags".into()),
            value: KernelValueOp::Set(ktstr::scenario::ops::KernelValue::U64(INITIAL_FLAGS)),
        }],
        ops: vec![
            Op::WriteKernelCold {
                writes: vec![KernelWrite {
                    target: KernelTarget::Symbol("__ktstr_test_flags".into()),
                    value: KernelValueOp::OrU64(already_set),
                }],
            },
            Op::ReadKernelCold {
                tag: "post_idempotent_or".into(),
                target: KernelTarget::Symbol("__ktstr_test_flags".into()),
                width: 8,
            },
        ],
        hold: HoldSpec::FULL,
        ..Default::default()
    }];
    execute_steps(ctx, steps)
}

fn assert_oru64_already_set_is_noop(r: &VmResult) -> AssertResult {
    let mut v = AssertResult::pass();
    let Some((_, reply)) = r
        .snapshot_bridge
        .drain_kernel_ops()
        .into_iter()
        .find(|(t, _)| t == "post_idempotent_or")
    else {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            "no `post_idempotent_or` reply captured",
        ));
        return v;
    };
    let observed = u64::from_le_bytes(reply.value[..8].try_into().unwrap());
    if observed != INITIAL_FLAGS {
        v.record_fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "OrU64 of already-set bit changed value: \
                 observed=0x{observed:016x} expected=0x{INITIAL_FLAGS:016x} — \
                 RMW path toggled or cleared instead of OR-ing",
            ),
        ));
    }
    v
}

// ---------------------------------------------------------------------
// T33.3 — Postcard roundtrip for `KernelValueOp::OrU64`.
//
// Wire-format regression pin. A regression that re-added
// `#[serde(untagged)]` to KernelValueOp would silently break
// postcard decode (postcard cannot decode untagged enums under
// the externally-tagged constraint the wire path requires).
// Catches the bug class at test time rather than runtime TLV
// drain. Mirrors the assert_result_postcard_roundtrip pattern at
// src/assert/tests_serde.rs:72.
// ---------------------------------------------------------------------

#[test]
fn kernel_value_op_oru64_postcard_roundtrip() {
    // TODO(B7-impl): the KernelOpRequestPayload type that wraps
    //   writes is what actually crosses the wire; encode a single-
    //   write payload with OrU64, decode, assert variant + mask
    //   survive. Direct KernelValueOp roundtrip is the minimal pin
    //   if the wrapping payload isn't available at test time.
    let _value = KernelValueOp::OrU64(0xDEAD_BEEF_CAFE_BABE);
    // Assertions (post-impl):
    //   let bytes = postcard::to_allocvec(&value).expect("encode");
    //   let v2: KernelValueOp = postcard::from_bytes(&bytes).expect("decode");
    //   assert!(matches!(v2, KernelValueOp::OrU64(0xDEAD_BEEF_CAFE_BABE)));
    todo!("B7-impl: postcard encode + decode + variant match");
}
