// Gated until B7-handler-internals (task #48) implements the
// referenced surface. See cold_path_op_e2e.rs header for the
// `cfg(any())` rationale.
#![cfg(any())]

//! Runtime fallback coverage for the `WriteableField` allow-list (#44).
//!
//! The canonical pin for "writing rq.nr_running is forbidden" is
//! `tests/compile_fail/nr_running_write_forbidden.rs` — if the
//! typed allow-list (`WriteableField` enum) has no `NrRunning`
//! variant, the compile_fail test guarantees no code path can
//! construct one.
//!
//! This file holds the RUNTIME-ONLY fallback tests in case the
//! allow-list shape is string-keyed instead of variant-typed (or
//! to pin the runtime-rejection behavior even when the compile-
//! time pin exists — defense in depth).
//!
//! Per advphd-kernel-31: writing rq.nr_running corrupts the
//! scheduler's runnable counter and surfaces as silently-wrong
//! load-balancing decisions on every subsequent dispatch. The
//! v1 allow-list intentionally excludes it.
//!
//! Skeleton-only.

use ktstr::scenario::ops::{KernelTarget, KernelValueOp, KernelWrite, Op};

// ---------------------------------------------------------------------
// T44.2 — Runtime rejection BEFORE dispatch.
//
// Constructs an `Op::WriteKernelCold` targeting `rq.nr_running`
// via whatever string-keyed or fallback path the implementation
// exposes. apply_op must return Err synchronously, BEFORE any
// wire request is sent — the rejection is a static-list lookup,
// not a runtime kernel response.
// ---------------------------------------------------------------------

#[test]
fn op_write_kernel_cold_rejects_nr_running_target() {
    // TODO(B7-impl): if WriteableField is typed-enum-only (no
    //   string fallback), this test becomes unreachable —
    //   compile-fail pin at tests/compile_fail/
    //   nr_running_write_forbidden.rs is the canonical guard.
    //   If a string-keyed escape hatch exists, this test pins
    //   the runtime rejection.
    //
    //   Expected error: KernelOpError::FieldNotInAllowList { field: "nr_running" }
    //   Expected wire effect: zero dispatched requests.
    let _op = Op::WriteKernelCold {
        writes: vec![KernelWrite {
            // Placeholder target — the actual API may be
            // KernelTarget::FieldOf { struct: "rq", field: "nr_running" }
            // or similar. Wire to whatever the impl exposes.
            target: KernelTarget::Symbol("rq.nr_running".into()),
            value: KernelValueOp::Set(ktstr::scenario::ops::KernelValue::U64(42)),
        }],
    };
    todo!("B7-impl: assert apply_op returns FieldNotInAllowList + bridge captured 0 requests");
}

// ---------------------------------------------------------------------
// T44.3 — Sibling test: the ALLOWED fields pass apply_op.
//
// Without this, T44.2 / the compile_fail test could pass against
// an over-restrictive allow-list that rejects everything. Pin one
// representative allowed field (e.g. rq.clock) to prove the
// gating is selective, not blanket.
// ---------------------------------------------------------------------

#[test]
fn op_write_kernel_cold_accepts_rq_clock_target() {
    // TODO(B7-impl): the typed allow-list per advphd-kernel-31's
    //   v1 list: RqClock, RqClockTask, RqScxClock, RqScxFlags,
    //   Jiffies64. Verify rq.clock passes apply_op into the
    //   dispatch path (mock bridge captures one request).
    let _op = Op::WriteKernelCold {
        writes: vec![KernelWrite {
            target: KernelTarget::Symbol("rq.clock".into()),
            value: KernelValueOp::Set(ktstr::scenario::ops::KernelValue::U64(1_000_000_000)),
        }],
    };
    todo!("B7-impl: assert apply_op succeeds + bridge captured 1 request");
}
