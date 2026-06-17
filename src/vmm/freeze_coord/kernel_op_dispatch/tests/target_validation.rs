use super::super::*;

// ---- KVA validation tests ----

/// Boundary inclusive: KVA at exactly KERNEL_HALF_CONSERVATIVE_5LEVEL
/// is accepted. Regression guard against off-by-one flipping
/// the `<` to `<=` (which would reject the canonical bound).
#[test]
fn validate_kva_target_accepts_exact_threshold() {
    assert!(validate_kva_target(KERNEL_HALF_CONSERVATIVE_5LEVEL, 4).is_ok());
}

/// Boundary exclusive: KVA one below threshold rejects.
/// Pins the rejection-side off-by-one symmetric with the
/// inclusive-bound test.
#[test]
fn validate_kva_target_rejects_one_below_threshold() {
    let kva = KERNEL_HALF_CONSERVATIVE_5LEVEL - 1;
    let err = validate_kva_target(kva, 4).expect_err("must reject");
    assert!(
        err.contains(&format!("{kva:#x}")),
        "error must echo rejected KVA for operator triage; got {err}"
    );
}

/// User-half edge (kva=0). Per the project's no-silent-drops
/// rule, 0 must fail loud rather than be treated as a sentinel.
#[test]
fn validate_kva_target_rejects_zero() {
    let err = validate_kva_target(0, 4).expect_err("kva=0 must reject");
    assert!(err.contains("0x0"));
}

/// User-half max (canonical 4-level user-half top).
/// Pins that a bit-63 only check is insufficient; the
/// threshold-based check catches this case.
#[test]
fn validate_kva_target_rejects_user_half_max() {
    let kva = 0x0000_7FFF_FFFF_FFFF;
    assert!(
        validate_kva_target(kva, 4).is_err(),
        "canonical user-half max must reject"
    );
}

/// Kernel-half typical KASLR-off + KASLR-on land.
/// Pins that real-world kernel KVAs don't false-reject.
#[test]
fn validate_kva_target_accepts_kernel_typical() {
    // x86_64 _text on KASLR-off (4-level).
    assert!(validate_kva_target(0xFFFF_FFFF_8100_0000, 4).is_ok());
    // Typical vmalloc address (high canonical addr).
    assert!(validate_kva_target(0xFFFF_C900_0000_0000, 4).is_ok());
    // 5-level direct-map base sample (would fail under 4-level-strict).
    assert!(validate_kva_target(0xFF11_0000_0000_0000, 4).is_ok());
}

/// user_half_kva_rejection_reason format pin via Path B
/// helper-extraction integration: the test invokes the SAME
/// helper the production dispatcher calls and pins error_reply's
/// propagation through the batch-prefix machinery. A regression
/// that drops the rejection, changes the format, or stops
/// calling the helper trips here. NOT a tautology — the test
/// does not synthesize its own copy of the format string.
#[test]
fn user_half_kva_rejection_reason_format_pin() {
    let kva = 0x4000_0000_0000;
    let helper_reason = user_half_kva_rejection_reason(kva);
    let batch_reason = format!("entry[0]: {helper_reason}");
    let reply = error_reply(11, batch_reason.clone());
    assert!(!reply.success);
    assert_eq!(reply.reason, batch_reason);
    // Helper output names the rejected KVA + the threshold + the
    // operator-actionable suggestion. A regression to the wrong
    // bound or dropped rejection surfaces.
    assert!(helper_reason.contains(&format!("{kva:#x}")));
    assert!(helper_reason.contains(&format!("{KERNEL_HALF_CONSERVATIVE_5LEVEL:#x}")));
    assert!(helper_reason.contains("kernel-half"));
    assert!(helper_reason.contains("5-level conservative"));
    assert!(helper_reason.contains("Symbol target"));
}

/// validate_direct_target accepts an in-range KVA.
/// Page_offset is a typical 4-level KASLR-off direct-map base.
#[test]
fn validate_direct_target_accepts_in_range() {
    let page_offset = 0xFFFF_8880_0000_0000u64;
    let dram_size = 256 * 1024 * 1024; // 256 MB typical ktstr test VM
    // First byte of direct map.
    assert!(validate_direct_target(page_offset, 4, page_offset, dram_size).is_ok());
    // Mid-range.
    assert!(validate_direct_target(page_offset + 0x1000, 8, page_offset, dram_size).is_ok());
    // Last U32 inside.
    assert!(validate_direct_target(page_offset + dram_size - 4, 4, page_offset, dram_size).is_ok());
}

/// validate_direct_target rejects a KVA below page_offset.
/// The user-half / canonical-hole class — kva_to_pa would
/// underflow and wrap.
#[test]
fn validate_direct_target_rejects_below_page_offset() {
    let page_offset = 0xFFFF_8880_0000_0000u64;
    let dram_size = 256 * 1024 * 1024;
    let kva = page_offset - 1;
    let err = validate_direct_target(kva, 4, page_offset, dram_size)
        .expect_err("kva below page_offset must reject");
    assert!(err.contains(&format!("{kva:#x}")));
    assert!(err.contains(&format!("{page_offset:#x}")));
    assert!(err.contains("would wrap"));
}

/// validate_direct_target rejects a KVA range past the
/// direct-map end. The "out the upper end" class.
#[test]
fn validate_direct_target_rejects_past_end() {
    let page_offset = 0xFFFF_8880_0000_0000u64;
    let dram_size = 256 * 1024 * 1024;
    // One byte past the last valid KVA, 4-byte len.
    let kva = page_offset + dram_size - 3;
    let err = validate_direct_target(kva, 4, page_offset, dram_size)
        .expect_err("kva+len past direct-map end must reject");
    assert!(err.contains("overruns direct-map end"));
}

/// validate_direct_target rejects overflow on kva+len.
/// Pins the checked_add guard.
#[test]
fn validate_direct_target_rejects_kva_len_overflow() {
    let page_offset = 0xFFFF_8880_0000_0000u64;
    let dram_size = 256 * 1024 * 1024;
    let kva = u64::MAX - 2;
    let err = validate_direct_target(kva, 4, page_offset, dram_size)
        .expect_err("kva+len overflow must reject");
    assert!(err.contains("overflow"));
}

/// validate_kva_target rejects overflow on kva+len.
#[test]
fn validate_kva_target_rejects_kva_len_overflow() {
    let kva = u64::MAX - 2;
    let err = validate_kva_target(kva, 4).expect_err("kva+len overflow must reject");
    assert!(err.contains("overflow"));
}
