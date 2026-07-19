use super::super::*;
use super::common::{
    DEFAULT_START_TIME, EXT_KVA, build_test_kernel, paint_valid_task, synth_task,
    synth_validation_offsets,
};

// ---- TaskField walker + validation tests ----
//
// Synthetic task_struct fixtures (offset layout, valid-task painter,
// synthetic GuestKernel builder) live in `super::common` and are
// shared with the `dispatch_arms` walker tests.

fn validate(
    kernel: &crate::monitor::guest::GuestKernel,
    task_pa: u64,
    pid: u32,
    expected_start_time_ns: u64,
    offs: &TaskValidationOffsets,
) -> Result<(), String> {
    validate_task_for_field_op(kernel, task_pa, pid, expected_start_time_ns, offs, EXT_KVA)
}

/// L1-L8 all pass on a freshly-painted valid SCX task.
#[test]
fn validate_task_happy_path_accepts() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 12345);
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    assert!(validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs).is_ok());
}

/// Validation scalar reads follow each member's BTF width. All ordinary
/// integer widths supported by GuestMem round-trip, while an unusual width is
/// rejected explicitly instead of extending into adjacent guest state.
#[test]
fn validation_integer_reads_supported_widths_and_rejects_unknown_width() {
    let mut buf = vec![0u8; 4096];
    buf[0x10] = 0xAB;
    buf[0x20..0x22].copy_from_slice(&0xCDEFu16.to_ne_bytes());
    buf[0x30..0x34].copy_from_slice(&0x1234_5678u32.to_ne_bytes());
    buf[0x40..0x48].copy_from_slice(&0x0123_4567_89AB_CDEFu64.to_ne_bytes());
    let kernel = build_test_kernel(&mut buf, Default::default());
    let read = |offset, width| {
        read_validation_integer(kernel.mem(), 0, "fixture", IntegerField { offset, width })
    };

    assert_eq!(read(0x10, 1).unwrap(), 0xAB);
    assert_eq!(read(0x20, 2).unwrap(), 0xCDEF);
    assert_eq!(read(0x30, 4).unwrap(), 0x1234_5678);
    assert_eq!(read(0x40, 8).unwrap(), 0x0123_4567_89AB_CDEF);

    let err = read(0x50, 3).expect_err("unsupported integer width must reject");
    assert!(err.contains("task_struct.fixture"), "{err}");
    assert!(err.contains("width=3"), "{err}");
    assert!(err.contains("supported widths: 1, 2, 4, 8"), "{err}");
}

/// L1 (pid mismatch): walker matched a task whose pid changed
/// between walker scan and validation read. Defense against slab
/// recycle.
#[test]
fn validate_task_rejects_pid_mismatch() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 99);
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs)
        .expect_err("pid mismatch must reject");
    assert!(err.contains("pid mismatch"), "must name layer: {err}");
    assert!(err.contains("read pid=99"));
    assert!(err.contains("expected 12345"));
}

/// L2 (start_time identity mismatch): the kernel sets start_time
/// ONCE at fork. If the original task exited and the kernel
/// recycled the PID, the new task's start_time will differ from
/// what we captured at spawn. The L2 gate accepts a window of
/// `[expected, expected + START_TIME_PROC_TICK_NS)` to absorb
/// the userspace /proc tick quantization (see L2 doc); this
/// test pins the rejection for an observed value BELOW expected
/// (the only direction recycle can drift), and the next two
/// tests pin the in-window accept + out-of-window reject above.
#[test]
fn validate_task_rejects_start_time_below_window() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 12345);
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    // observed=DEFAULT_START_TIME, expected=DEFAULT_START_TIME + 1ms
    // → observed < expected, must reject.
    let too_high_expected = DEFAULT_START_TIME + 1_000_000;
    let err = validate(&kernel, 0, 12345, too_high_expected, &offs)
        .expect_err("start_time below window must reject");
    assert!(err.contains("start_time identity mismatch"));
    assert!(err.contains(&format!("observed={DEFAULT_START_TIME}")));
    assert!(err.contains(&format!("expected in [{too_high_expected}")));
    assert!(err.contains("recycled"));
}

/// L2 accepts an observed start_time within the userspace-tick
/// window above the expected value. Pins the legitimate
/// jiffies-quantization gap that /proc/<pid>/stat field 22
/// introduces.
#[test]
fn validate_task_accepts_start_time_within_tick_window() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 12345);
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    // observed=DEFAULT_START_TIME, expected=DEFAULT_START_TIME-5ms
    // → observed - expected = +5ms (within 10ms window).
    let expected_within_window = DEFAULT_START_TIME - 5_000_000;
    validate(&kernel, 0, 12345, expected_within_window, &offs)
        .expect("start_time within tick window must accept");
}

/// L2 rejects an observed start_time MORE than one tick above
/// the expected value — characteristic of a PID-recycled task
/// whose start_time is fundamentally newer.
#[test]
fn validate_task_rejects_start_time_above_window() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 12345);
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    // observed=DEFAULT_START_TIME, expected=DEFAULT_START_TIME-20ms
    // → observed - expected = +20ms (above 10ms window).
    let expected_below_window = DEFAULT_START_TIME - 20_000_000;
    let err = validate(&kernel, 0, 12345, expected_below_window, &offs)
        .expect_err("start_time above window must reject");
    assert!(err.contains("start_time identity mismatch"));
    assert!(err.contains("recycled"));
}

/// L3 (TASK_DEAD): task in final teardown — fields mid-cleanup.
#[test]
fn validate_task_rejects_task_dead() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 12345);
    buf[synth_task::STATE_OFF..synth_task::STATE_OFF + 4].copy_from_slice(&0x80u32.to_le_bytes());
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let err =
        validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs).expect_err("TASK_DEAD must reject");
    assert!(err.contains("TASK_DEAD"));
    assert!(err.contains("state=0x80"));
}

/// L4 (on_rq != 0): task is queued — writing scheduler fields
/// would corrupt rb-tree / DSQ ordering.
#[test]
fn validate_task_rejects_on_rq_queued() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 12345);
    buf[synth_task::ON_RQ_OFF..synth_task::ON_RQ_OFF + 4].copy_from_slice(&1u32.to_le_bytes());
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let err =
        validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs).expect_err("on_rq=1 must reject");
    assert!(err.contains("on_rq=1"));
    assert!(err.contains("rb-tree"));
    assert!(err.contains("WorkType::FutexPingPong"));
}

/// L5 part-1 (scx.dsq != NULL): task is queued on an SCX DSQ.
#[test]
fn validate_task_rejects_scx_dsq_populated() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 12345);
    buf[synth_task::SCX_DSQ_OFF..synth_task::SCX_DSQ_OFF + 8]
        .copy_from_slice(&0xFFFF_DEAD_BEEFu64.to_le_bytes());
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs)
        .expect_err("scx.dsq non-NULL must reject");
    assert!(err.contains("scx.dsq=0xffffdeadbeef"));
    assert!(err.contains("SCX DSQ"));
    assert!(err.contains("WorkType::FutexPingPong"));
}

/// L5 part-2 (scx.runnable_node linked): task is on a per-rq
/// runnable_list even though scx.dsq is NULL.
#[test]
fn validate_task_rejects_scx_runnable_node_linked() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 12345);
    // Point runnable_node.next at a non-self address (linked).
    buf[synth_task::SCX_RUNNABLE_NODE_OFF..synth_task::SCX_RUNNABLE_NODE_OFF + 8]
        .copy_from_slice(&0xFFFF_8881_DEAD_C0DEu64.to_le_bytes());
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs)
        .expect_err("runnable_node linked must reject");
    assert!(err.contains("scx.runnable_node is linked"));
    assert!(err.contains("WorkType::FutexPingPong"));
}

/// L6 (sched_class != ext_sched_class): non-SCX task rejected.
#[test]
fn validate_task_rejects_non_ext_sched_class() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 12345);
    // Set sched_class to a fake fair_sched_class KVA.
    let fair_kva: u64 = 0xFFFF_FFFF_8200_0000;
    buf[synth_task::SCHED_CLASS_OFF..synth_task::SCHED_CLASS_OFF + 8]
        .copy_from_slice(&fair_kva.to_le_bytes());
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs)
        .expect_err("non-ext sched_class must reject");
    assert!(err.contains(&format!("sched_class={fair_kva:#x}")));
    assert!(err.contains("SCX-managed tasks only"));
    assert!(err.contains("SchedPolicy::Normal"));
}

/// L8 (start_boottime == 0): probable slab-recycle survivor that
/// passed L1 + L2 by coincidence.
#[test]
fn validate_task_rejects_zero_start_boottime() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 12345);
    buf[synth_task::START_BOOTTIME_OFF..synth_task::START_BOOTTIME_OFF + 8]
        .copy_from_slice(&0u64.to_le_bytes());
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME, &offs)
        .expect_err("start_boottime=0 must reject");
    assert!(err.contains("start_boottime=0"));
    assert!(err.contains("slab-recycle"));
}

/// Layer ordering: L1 (pid) fires BEFORE L2 (start_time). A task
/// with BOTH pid mismatch AND start_time mismatch surfaces the
/// pid error.
#[test]
fn validate_task_layer_order_pid_before_start_time() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 99);
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    // Both mismatched — expect pid error.
    let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME + 1, &offs).expect_err("must reject");
    assert!(err.contains("pid mismatch"), "L1 must fire first: {err}");
    assert!(!err.contains("start_time identity mismatch"));
}

/// Layer ordering: L2 (start_time) fires BEFORE L3 (TASK_DEAD).
#[test]
fn validate_task_layer_order_start_time_before_dead() {
    let mut buf = vec![0u8; 4096];
    paint_valid_task(&mut buf, 0, 12345);
    buf[synth_task::STATE_OFF..synth_task::STATE_OFF + 4].copy_from_slice(&0x80u32.to_le_bytes());
    let kernel = build_test_kernel(&mut buf, Default::default());
    let offs = synth_validation_offsets();
    let err = validate(&kernel, 0, 12345, DEFAULT_START_TIME + 1, &offs).expect_err("must reject");
    assert!(
        err.contains("start_time identity mismatch"),
        "L2 must fire first: {err}"
    );
    assert!(!err.contains("TASK_DEAD"));
}

/// `KernelValue::OrU32` RMW: sets the supplied bits without
/// clobbering bits already set in the target word. Pre-populates
/// a synthetic symbol with an alternating bit pattern and ORs a
/// single unset bit; the read-back must show the union, not just
/// the OR mask alone.
///
/// Migrated from `tests/oru64_rmw_e2e.rs` (gated skeleton); the
/// dispatcher narrowed OrU64 to OrU32 because the canonical
/// scheduler-flags use case (`struct scx_rq.flags`) is u32 per
/// `kernel/sched/sched.h:803`. Test runs as a host-side unit
/// test against `build_test_kernel`'s synthetic guest memory —
/// no VM boot — because the RMW correctness is pure dispatcher
/// arithmetic.
#[test]
fn oru32_sets_target_bits_preserves_others() {
    // KVA = start_kernel_map + PA; the Symbol path's
    // `text_kva_to_pa` translates via start_kernel_map, NOT
    // page_offset.  The const value of start_kernel_map is
    // arch-specific (0xFFFF_FFFF_8000_0000 on x86_64,
    // 0xFFFF_8000_8000_0000 on aarch64), so the SYMBOL_KVA is
    // derived rather than hardcoded — a hardcoded x86_64
    // value puts the translated PA outside the buffer on
    // aarch64 and the read silently returns 0.
    const SYMBOL_PA: u64 = 0x40;
    let symbol_kva: u64 = crate::monitor::symbols::START_KERNEL_MAP + SYMBOL_PA;
    const INITIAL_FLAGS: u32 = 0xAAAA_AAAA;
    const OR_MASK: u32 = 0x0000_0001;
    let mut buf = vec![0u8; 4096];
    buf[SYMBOL_PA as usize..SYMBOL_PA as usize + 4].copy_from_slice(&INITIAL_FLAGS.to_le_bytes());
    let mut symbols = std::collections::HashMap::new();
    symbols.insert("test_flags".to_string(), symbol_kva);
    let kernel = build_test_kernel(&mut buf, symbols);
    dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Symbol("test_flags".into()),
        &KernelOpValue::OrU32(OR_MASK),
    )
    .expect("OrU32 RMW dispatch must succeed against painted symbol");
    let observed = kernel
        .read_symbol_u32("test_flags")
        .expect("read-back must succeed");
    assert_eq!(
        observed,
        INITIAL_FLAGS | OR_MASK,
        "OrU32 must set 0x{OR_MASK:08x} without clobbering 0x{INITIAL_FLAGS:08x}"
    );
}

/// `KernelValue::OrU32` is idempotent on already-set bits: OR'ing
/// a bit that is already 1 leaves the word unchanged. Pins that
/// the RMW path never accidentally toggles or clears a bit it
/// was asked to OR — a regression that flipped `|=` to `^=`
/// would slip past the [`oru32_sets_target_bits_preserves_others`]
/// test (which uses a NEW bit) and surface only here.
///
/// Migrated from `tests/oru64_rmw_e2e.rs`.
#[test]
fn oru32_idempotent_on_already_set_bit() {
    // KVA = start_kernel_map + PA; the Symbol path's
    // `text_kva_to_pa` translates via start_kernel_map, NOT
    // page_offset.  The const value of start_kernel_map is
    // arch-specific (0xFFFF_FFFF_8000_0000 on x86_64,
    // 0xFFFF_8000_8000_0000 on aarch64), so the SYMBOL_KVA is
    // derived rather than hardcoded — a hardcoded x86_64
    // value puts the translated PA outside the buffer on
    // aarch64 and the read silently returns 0.
    const SYMBOL_PA: u64 = 0x40;
    let symbol_kva: u64 = crate::monitor::symbols::START_KERNEL_MAP + SYMBOL_PA;
    const INITIAL_FLAGS: u32 = 0xAAAA_AAAA;
    // Bit 1 is set in 0xA = 1010 — picking 0x2 means OR-ing an
    // already-set bit.
    const ALREADY_SET: u32 = 0x0000_0002;
    let mut buf = vec![0u8; 4096];
    buf[SYMBOL_PA as usize..SYMBOL_PA as usize + 4].copy_from_slice(&INITIAL_FLAGS.to_le_bytes());
    let mut symbols = std::collections::HashMap::new();
    symbols.insert("test_flags".to_string(), symbol_kva);
    let kernel = build_test_kernel(&mut buf, symbols);
    // Sanity: the chosen bit IS set in the initial value (test
    // construction guard).
    assert_eq!(
        INITIAL_FLAGS & ALREADY_SET,
        ALREADY_SET,
        "test setup bug: chose a bit that is not pre-set"
    );
    dispatch_one_write(
        &kernel,
        None,
        0,
        &KernelOpTarget::Symbol("test_flags".into()),
        &KernelOpValue::OrU32(ALREADY_SET),
    )
    .expect("OrU32 with already-set bit must succeed");
    let observed = kernel
        .read_symbol_u32("test_flags")
        .expect("read-back must succeed");
    assert_eq!(
        observed, INITIAL_FLAGS,
        "OrU32 of already-set bit must leave value unchanged \
         (regression: bit was toggled or cleared instead of OR'd)"
    );
}

/// `KernelOpValue::OrU32` survives a postcard round-trip embedded
/// inside a full `KernelOpRequestPayload`. Pins the wire-format
/// shape against a `#[serde(untagged)]` regression (untagged enums
/// break postcard's externally-tagged constraint — the wire-path
/// reader at the host would silently drop the variant tag).
///
/// Migrated from `tests/oru64_rmw_e2e.rs`. The existing
/// `kernel_op_request_payload_postcard_round_trip` test in
/// `src/vmm/wire.rs` covers U32/U64/Bytes/PerCpuField/TaskField
/// but does not include an OrU32 entry — this fills that gap.
#[test]
fn oru32_postcard_round_trip_through_payload() {
    const MASK: u32 = 0xDEAD_BEEF;
    let payload = crate::vmm::wire::KernelOpRequestPayload {
        request_id: 0xABCD,
        mode: crate::vmm::wire::KernelOpMode::Cold,
        direction: crate::vmm::wire::KernelOpDirection::Write,
        tag: "oru32_roundtrip_pin".into(),
        entries: vec![crate::vmm::wire::KernelOpEntry {
            target: KernelOpTarget::Symbol("any_symbol".into()),
            value: KernelOpValue::OrU32(MASK),
        }],
    };
    let bytes = postcard::to_allocvec(&payload).expect("encode");
    let back: crate::vmm::wire::KernelOpRequestPayload =
        postcard::from_bytes(&bytes).expect("decode");
    assert_eq!(back.entries.len(), 1);
    match &back.entries[0].value {
        KernelOpValue::OrU32(observed_mask) => {
            assert_eq!(*observed_mask, MASK, "OrU32 mask must survive round-trip");
        }
        other => panic!("expected OrU32 variant after round-trip, got {other:?}"),
    }
}
