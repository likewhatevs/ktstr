//! Coverage for the kernel-op dispatch surface: the four
//! `KernelTarget`/`KernelValue` conversion helpers plus
//! `dispatch_kernel_op_request`'s bridge-first /
//! wire-fallback / hard-bail routing.
//!
//! These exercise the host-side dispatch surface only. The
//! in-guest wire path (`request_kernel_op` end-to-end) is
//! covered separately in `src/vmm/guest_comms.rs::tests` and
//! the future end-to-end integration suite.

use std::sync::Arc;

use super::{
    KernelTarget, KernelValue, Op, build_kernel_op_request, dispatch_kernel_op_request,
    merge_adjacent_cold_writes, wait_for_accessor_publish_or_bail,
    wait_for_worker_state_not_trying_or_bail, write_entries_from_writes,
};
use crate::scenario::snapshot::{CaptureCallback, KernelOpCallback, SnapshotBridge};

/// 1:1 mapping of every `KernelTarget` variant via the
/// `From<&KernelTarget> for KernelOpTarget` impl — pins the
/// Cow-to-String coercion + per-CPU field decomposition. A
/// regression that flipped a variant tag, dropped a Cow→String
/// conversion, or swapped per-cpu-field fields surfaces here.
#[test]
fn kernel_target_into_wire_maps_every_variant() {
    let cases: &[(KernelTarget, crate::vmm::wire::KernelOpTarget)] = &[
        (
            KernelTarget::symbol("jiffies"),
            crate::vmm::wire::KernelOpTarget::Symbol("jiffies".into()),
        ),
        (
            KernelTarget::direct(0xffff_8000_0000_2000),
            crate::vmm::wire::KernelOpTarget::Direct(0xffff_8000_0000_2000),
        ),
        (
            KernelTarget::kva(0xffff_c000_dead_beef),
            crate::vmm::wire::KernelOpTarget::Kva(0xffff_c000_dead_beef),
        ),
        (
            KernelTarget::per_cpu_field("runqueues", "clock", 5),
            crate::vmm::wire::KernelOpTarget::PerCpuField {
                symbol: "runqueues".into(),
                field: "clock".into(),
                cpu: 5,
            },
        ),
        (
            KernelTarget::task_field(42, 1_700_000_000_000, "scx.dsq_vtime"),
            crate::vmm::wire::KernelOpTarget::TaskField {
                pid: 42,
                expected_start_time_ns: 1_700_000_000_000,
                field: "scx.dsq_vtime".into(),
            },
        ),
    ];
    for (src, want) in cases {
        let got: crate::vmm::wire::KernelOpTarget = src.into();
        assert_eq!(&got, want, "wire mapping mismatch for {src:?}");
    }
}

/// 1:1 mapping of every `KernelValue` variant via the
/// `From<&KernelValue> for KernelOpValue` impl — pins the
/// Bytes-clone semantic and the numeric-width identity. A
/// regression that swapped U32/U64 width or skipped the Bytes
/// clone surfaces here.
#[test]
fn kernel_value_into_wire_maps_every_variant() {
    let u32_val: crate::vmm::wire::KernelOpValue = (&KernelValue::u32(42)).into();
    assert_eq!(u32_val, crate::vmm::wire::KernelOpValue::U32(42));
    let u64_val: crate::vmm::wire::KernelOpValue =
        (&KernelValue::u64(0xDEAD_BEEF_CAFE_F00D)).into();
    assert_eq!(
        u64_val,
        crate::vmm::wire::KernelOpValue::U64(0xDEAD_BEEF_CAFE_F00D)
    );
    let bytes = vec![1u8, 2, 3, 4, 5];
    let bytes_val: crate::vmm::wire::KernelOpValue = (&KernelValue::bytes(bytes.clone())).into();
    assert_eq!(bytes_val, crate::vmm::wire::KernelOpValue::Bytes(bytes));
    // OrU32: width is u32 not u64 (struct scx_rq.flags is u32
    // per kernel/sched/sched.h:803); a regression that mapped
    // OrU32 to wire-side U32 or OrU64 (had one existed) would
    // silently lose the RMW intent at the dispatcher and
    // either drop the OR or corrupt the adjacent field.
    let or_val: crate::vmm::wire::KernelOpValue = (&KernelValue::or_u32(1 << 5)).into();
    assert_eq!(or_val, crate::vmm::wire::KernelOpValue::OrU32(1 << 5));
    // Edge mask values per the test spec: degenerate
    // zero mask (a tempting wrong-optimization to skip the OR
    // on `mask == 0` would trip here), all-bits-set mask
    // (RMW degenerates to a full overwrite — still must
    // route through OrU32 wire variant, not U32), and a
    // multi-bit non-power-of-2 mask (catches a regression
    // that treated OrU32 as single-bit-only).
    let zero_or: crate::vmm::wire::KernelOpValue = (&KernelValue::or_u32(0)).into();
    assert_eq!(zero_or, crate::vmm::wire::KernelOpValue::OrU32(0));
    let max_or: crate::vmm::wire::KernelOpValue = (&KernelValue::or_u32(u32::MAX)).into();
    assert_eq!(max_or, crate::vmm::wire::KernelOpValue::OrU32(u32::MAX));
    let multi_bit_or: crate::vmm::wire::KernelOpValue = (&KernelValue::or_u32(0xA5A5_A5A5)).into();
    assert_eq!(
        multi_bit_or,
        crate::vmm::wire::KernelOpValue::OrU32(0xA5A5_A5A5)
    );
}

/// `KernelValue::OrU32` participates in `PartialEq` distinct
/// from its `U32` sibling and distinct from other OrU32 masks.
/// A regression that custom-impl'd PartialEq ignoring the
/// variant tag (so `OrU32(1) == U32(1)`) would silently
/// conflate write modes — `assert_eq!` on collected
/// `Vec<KernelValue>` would compare-equal across variants.
#[test]
fn kernel_value_partial_eq_distinguishes_oru32_from_u32_and_other_masks() {
    assert_eq!(KernelValue::or_u32(1), KernelValue::or_u32(1));
    assert_ne!(KernelValue::or_u32(1), KernelValue::or_u32(2));
    assert_ne!(KernelValue::or_u32(1), KernelValue::u32(1));
}

/// `KernelValue::or_u32` is `const fn` so static contexts can
/// declare flag constants without a runtime construction. A
/// regression that dropped the `const` modifier would push
/// every static use to a runtime build (or a compile error
/// at the callsite). This compile-time `const _` assertion
/// trips on the regression at the file boundary, not at the
/// downstream caller.
#[test]
fn kernel_value_or_u32_is_const_constructible() {
    const _OR_AT_COMPILE_TIME: KernelValue = KernelValue::or_u32(1 << 5);
}

/// `write_entries_from_writes` preserves order and produces one
/// wire entry per source pair — the cold-batch contract relies
/// on every entry surviving the conversion in the supplied
/// sequence (a reorder would change which CPU's `rq.clock`
/// landed in which freeze-rendezvous slot).
#[test]
fn write_entries_from_writes_preserves_order_and_count() {
    let writes = vec![
        (
            KernelTarget::per_cpu_field("runqueues", "clock", 0),
            KernelValue::u64(100),
        ),
        (
            KernelTarget::per_cpu_field("runqueues", "clock", 1),
            KernelValue::u64(200),
        ),
        (KernelTarget::symbol("jiffies"), KernelValue::u32(0xDEAD)),
    ];
    let entries = write_entries_from_writes(&writes);
    assert_eq!(entries.len(), 3);
    match (&entries[0].target, &entries[0].value) {
        (
            crate::vmm::wire::KernelOpTarget::PerCpuField { cpu: 0, .. },
            crate::vmm::wire::KernelOpValue::U64(100),
        ) => {}
        other => panic!("entry[0] mismatch: {other:?}"),
    }
    match (&entries[1].target, &entries[1].value) {
        (
            crate::vmm::wire::KernelOpTarget::PerCpuField { cpu: 1, .. },
            crate::vmm::wire::KernelOpValue::U64(200),
        ) => {}
        other => panic!("entry[1] mismatch: {other:?}"),
    }
    match (&entries[2].target, &entries[2].value) {
        (
            crate::vmm::wire::KernelOpTarget::Symbol(s),
            crate::vmm::wire::KernelOpValue::U32(0xDEAD),
        ) if s == "jiffies" => {}
        other => panic!("entry[2] mismatch: {other:?}"),
    }
}

/// `wait_for_worker_state_not_trying_or_bail`'s
/// `None`-bridge arm must `anyhow::bail!` loudly. A regression
/// that reverted to the earlier silent-Ok shape would let
/// AttachScheduler / ReplaceScheduler proceed without quiescing
/// the worker state, baselining the post-spawn seqno against an
/// in-flight publish from the prior scheduler — downstream
/// scheduler-liveness assertions then vacuously pass against
/// stale state.
#[test]
fn wait_for_worker_state_not_trying_or_bail_no_bridge_bails_loudly() {
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(1);
    let err = wait_for_worker_state_not_trying_or_bail("Op::TestNoBridge", deadline)
        .expect_err("no-bridge path must bail loudly, not silent-Ok");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Op::TestNoBridge"),
        "error must name the op label: {msg}",
    );
    assert!(
        msg.contains("no SnapshotBridge installed"),
        "error must cite the bridge-missing root cause: {msg}",
    );
    assert!(
        msg.contains("worker-state-not-trying"),
        "error must cite the specific gate that would have been skipped: {msg}",
    );
    assert!(
        msg.contains("set_thread_local"),
        "error must enumerate the install-bridge recovery path: {msg}",
    );
}

/// `wait_for_accessor_publish_or_bail`'s
/// `None`-bridge arm must `anyhow::bail!` loudly, not silently
/// return Ok. A regression that reverted to the earlier
/// `Some(Ok(_)) | None => Ok(())` shape would let
/// AttachScheduler / ReplaceScheduler appear to succeed in
/// contexts where no SnapshotBridge is installed — the
/// scheduler-liveness verification would never run and
/// downstream assertions would vacuously pass.
#[test]
fn wait_for_accessor_publish_or_bail_no_bridge_bails_loudly() {
    let err = wait_for_accessor_publish_or_bail(
        "Op::TestNoBridge",
        0,
        std::time::Duration::from_millis(1),
    )
    .expect_err("no-bridge path must bail loudly, not silent-Ok");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("Op::TestNoBridge"),
        "error must name the op label: {msg}",
    );
    assert!(
        msg.contains("no SnapshotBridge installed"),
        "error must cite the bridge-missing root cause: {msg}",
    );
    assert!(
        msg.contains("host_only"),
        "error must enumerate the host_only recovery path: {msg}",
    );
    assert!(
        msg.contains("set_thread_local"),
        "error must enumerate the install-bridge recovery path: {msg}",
    );
}

/// `dispatch_kernel_op_request` hard-bails when no bridge is
/// installed AND we're not in a guest VM — per the project
/// "no silent drops" rule. The bail message must name both
/// recovery paths (install bridge callback / run in guest VM)
/// so the misconfigured-test signal is actionable. A regression
/// that reverted to silent warn-skip would re-introduce the
/// vacuous-test footgun.
#[test]
fn dispatch_kernel_op_request_no_bridge_no_guest_hard_bails() {
    let payload = build_kernel_op_request(
        crate::vmm::wire::KernelOpMode::Hot,
        crate::vmm::wire::KernelOpDirection::Write,
        "missing_setup".into(),
        vec![],
    );
    let r = dispatch_kernel_op_request("Op::TestNoBridge", payload);
    let err = r.expect_err("no-bridge/non-guest must bail loudly, not warn-skip");
    let msg = err.to_string();
    assert!(
        msg.contains("Op::TestNoBridge"),
        "error must name the op label: {msg}"
    );
    assert!(
        msg.contains("missing_setup"),
        "error must name the request tag: {msg}"
    );
    assert!(
        msg.contains("with_kernel_op"),
        "error must point at SnapshotBridge::with_kernel_op recovery path: {msg}"
    );
    assert!(
        msg.contains("guest VM"),
        "error must mention the guest-VM recovery path: {msg}"
    );
}

/// `dispatch_kernel_op_request` invokes the bridge callback on
/// success, propagates `reply.success == true` as `Ok(())`, and
/// the bridge's drain log captures the (tag, reply) pair.
#[test]
fn dispatch_kernel_op_request_bridge_success_path() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let kernel_op_cb: KernelOpCallback = Arc::new(|req| crate::vmm::wire::KernelOpReplyPayload {
        request_id: req.request_id,
        success: true,
        reason: String::new(),
        read_values: vec![],
    });
    let bridge = SnapshotBridge::new(cb).with_kernel_op(kernel_op_cb);
    let bridge_clone = bridge.clone();
    let _g = bridge.set_thread_local();
    let payload = build_kernel_op_request(
        crate::vmm::wire::KernelOpMode::Cold,
        crate::vmm::wire::KernelOpDirection::Write,
        "test_tag".into(),
        vec![crate::vmm::wire::KernelOpEntry {
            target: crate::vmm::wire::KernelOpTarget::Symbol("jiffies".into()),
            value: crate::vmm::wire::KernelOpValue::U64(42),
        }],
    );
    let r = dispatch_kernel_op_request("Op::TestSuccess", payload);
    assert!(r.is_ok(), "bridge success path must Ok, got {r:?}");
    let log = bridge_clone.drain_kernel_ops();
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].0, "test_tag");
    assert!(log[0].1.success);
}

/// `dispatch_kernel_op_request` propagates `reply.success ==
/// false` as an `anyhow::Error` carrying the reason — the
/// executor's `?` propagation thus surfaces host-side op
/// failures (e.g. "symbol not found") to the caller.
#[test]
fn dispatch_kernel_op_request_bridge_failure_path_bails() {
    let cb: CaptureCallback = Arc::new(|_| None);
    let kernel_op_cb: KernelOpCallback = Arc::new(|req| crate::vmm::wire::KernelOpReplyPayload {
        request_id: req.request_id,
        success: false,
        reason: "host: symbol 'bogus' not found".into(),
        read_values: vec![],
    });
    let bridge = SnapshotBridge::new(cb).with_kernel_op(kernel_op_cb);
    let _g = bridge.set_thread_local();
    let payload = build_kernel_op_request(
        crate::vmm::wire::KernelOpMode::Hot,
        crate::vmm::wire::KernelOpDirection::Read,
        "failing_tag".into(),
        vec![crate::vmm::wire::KernelOpEntry {
            target: crate::vmm::wire::KernelOpTarget::Symbol("bogus".into()),
            value: crate::vmm::wire::KernelOpValue::U64(0),
        }],
    );
    let r = dispatch_kernel_op_request("Op::TestFailure", payload);
    let err = r.expect_err("reply.success=false must bail");
    let msg = err.to_string();
    assert!(
        msg.contains("Op::TestFailure"),
        "error must name the op label: {msg}"
    );
    assert!(
        msg.contains("failing_tag"),
        "error must name the request tag: {msg}"
    );
    assert!(
        msg.contains("symbol 'bogus' not found"),
        "error must surface the host reason: {msg}"
    );
}

/// Three adjacent `Op::WriteKernelCold` singletons fold into
/// one merged op with the 3 writes concatenated in input
/// order. Pins the multi-CPU-seed shape — `with_uptime`
/// writing per-CPU `rq.clock` on every CPU must land in ONE
/// freeze rendezvous, not N.
#[test]
fn merge_adjacent_cold_writes_folds_three_singletons() {
    let ops = vec![
        Op::write_kernel_cold(
            KernelTarget::per_cpu_field("runqueues", "clock", 0),
            KernelValue::u64(100),
        ),
        Op::write_kernel_cold(
            KernelTarget::per_cpu_field("runqueues", "clock", 1),
            KernelValue::u64(200),
        ),
        Op::write_kernel_cold(
            KernelTarget::per_cpu_field("runqueues", "clock", 2),
            KernelValue::u64(300),
        ),
    ];
    let merged = merge_adjacent_cold_writes(&ops);
    assert_eq!(
        merged.len(),
        1,
        "3 adjacent cold-write singletons must fold to 1 op"
    );
    match &merged[0] {
        Op::WriteKernelCold { writes } => {
            assert_eq!(writes.len(), 3, "merged batch must carry all 3 writes");
            match &writes[0].1 {
                KernelValue::U64(100) => {}
                other => panic!("first entry value mismatch: {other:?}"),
            }
            match &writes[2].1 {
                KernelValue::U64(300) => {}
                other => panic!("third entry value mismatch: {other:?}"),
            }
        }
        other => panic!("expected merged WriteKernelCold, got {other:?}"),
    }
}

/// An `Op::WriteKernelHot` between two cold-write singletons
/// is a hard barrier — the two cold-write singletons emerge as
/// 2 separate `Op::WriteKernelCold` ops (one on each side of
/// the hot barrier), preserving rendezvous boundaries.
#[test]
fn merge_adjacent_cold_writes_hot_is_barrier() {
    let ops = vec![
        Op::write_kernel_cold(KernelTarget::symbol("a"), KernelValue::u64(1)),
        Op::write_kernel_hot(KernelTarget::symbol("h"), KernelValue::u64(2)),
        Op::write_kernel_cold(KernelTarget::symbol("b"), KernelValue::u64(3)),
    ];
    let merged = merge_adjacent_cold_writes(&ops);
    assert_eq!(merged.len(), 3, "hot barrier must split cold writes");
    assert!(matches!(merged[0], Op::WriteKernelCold { ref writes } if writes.len() == 1));
    assert!(matches!(merged[1], Op::WriteKernelHot { .. }));
    assert!(matches!(merged[2], Op::WriteKernelCold { ref writes } if writes.len() == 1));
}

/// An `Op::ReadKernelCold` between two cold-write singletons
/// is a hard barrier — reads don't fold into write batches
/// in this pre-pass (a follow-up adds per-entry direction +
/// tag for mixed-direction folding). The two cold-write
/// singletons emerge as 2 separate ops + the read in between.
#[test]
fn merge_adjacent_cold_writes_read_is_barrier() {
    let ops = vec![
        Op::write_kernel_cold(KernelTarget::symbol("a"), KernelValue::u64(1)),
        Op::read_kernel_cold(
            "r",
            KernelTarget::symbol("r"),
            super::KernelValueWidth::u64(),
        ),
        Op::write_kernel_cold(KernelTarget::symbol("b"), KernelValue::u64(2)),
    ];
    let merged = merge_adjacent_cold_writes(&ops);
    assert_eq!(merged.len(), 3, "cold-read barrier must split cold writes");
    assert!(matches!(merged[0], Op::WriteKernelCold { ref writes } if writes.len() == 1));
    assert!(matches!(merged[1], Op::ReadKernelCold { .. }));
    assert!(matches!(merged[2], Op::WriteKernelCold { ref writes } if writes.len() == 1));
}

/// Caller-supplied multi-write `Op::WriteKernelCold` (via
/// `Op::write_kernel_cold_batch`) merges with adjacent
/// singletons — the multi-write's `writes` vec appends onto
/// the running batch in input order.
#[test]
fn merge_adjacent_cold_writes_appends_multi_write_op() {
    let ops = vec![
        Op::write_kernel_cold(KernelTarget::symbol("pre"), KernelValue::u64(0)),
        Op::write_kernel_cold_batch(vec![
            (KernelTarget::symbol("a"), KernelValue::u64(1)),
            (KernelTarget::symbol("b"), KernelValue::u64(2)),
        ]),
        Op::write_kernel_cold(KernelTarget::symbol("post"), KernelValue::u64(3)),
    ];
    let merged = merge_adjacent_cold_writes(&ops);
    assert_eq!(
        merged.len(),
        1,
        "singleton+batch+singleton must fold to 1 op"
    );
    match &merged[0] {
        Op::WriteKernelCold { writes } => {
            assert_eq!(writes.len(), 4);
            let names: Vec<&str> = writes
                .iter()
                .map(|(t, _)| match t {
                    KernelTarget::Symbol(s) => s.as_ref(),
                    _ => panic!("non-Symbol target"),
                })
                .collect();
            assert_eq!(names, vec!["pre", "a", "b", "post"]);
        }
        other => panic!("expected merged WriteKernelCold, got {other:?}"),
    }
}

/// Empty input → empty output. Single cold-write → single
/// cold-write (no spurious wrapping). Pre-pass is structurally
/// equivalent on inputs with no fold opportunities.
#[test]
fn merge_adjacent_cold_writes_passes_through_when_nothing_to_fold() {
    assert!(merge_adjacent_cold_writes(&[]).is_empty());
    let single = vec![Op::write_kernel_cold(
        KernelTarget::symbol("x"),
        KernelValue::u64(42),
    )];
    let merged = merge_adjacent_cold_writes(&single);
    assert_eq!(merged.len(), 1);
    match &merged[0] {
        Op::WriteKernelCold { writes } => assert_eq!(writes.len(), 1),
        other => panic!("expected single WriteKernelCold, got {other:?}"),
    }
}
