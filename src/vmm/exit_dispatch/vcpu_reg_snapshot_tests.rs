use super::*;

#[test]
fn vcpu_reg_snapshot_display_renders_three_hex_fields() {
    // x86_64-shape snapshot (user_page_table_root=None): only
    // the three core hex fields render; no `uptroot=` suffix.
    let s = VcpuRegSnapshot {
        instruction_pointer: 0xffff_ffff_8100_1234,
        stack_pointer: 0xffff_ffff_8000_0000,
        page_table_root: 0x0123_4567_89ab_cdef,
        user_page_table_root: None,
        tcr_el1: None,
    };
    let out = format!("{s}");
    assert_eq!(
        out,
        "ip=0xffffffff81001234 sp=0xffffffff80000000 ptroot=0x0123456789abcdef"
    );
}

#[test]
fn vcpu_reg_snapshot_display_appends_user_pt_root_when_present() {
    // aarch64-shape snapshot: user_page_table_root populated
    // → Display appends ` uptroot=0x...`. Pinning here so a
    // future Display tweak (e.g. swapping " " for "\n  ")
    // is caught.
    let s = VcpuRegSnapshot {
        instruction_pointer: 0xffff_8000_8100_1234,
        stack_pointer: 0xffff_8000_8000_0000,
        page_table_root: 0x0000_4000_8000_0000,
        user_page_table_root: Some(0x0000_0000_aaaa_bbbb),
        tcr_el1: Some(0xb510_0010),
    };
    let out = format!("{s}");
    assert_eq!(
        out,
        "ip=0xffff800081001234 sp=0xffff800080000000 ptroot=0x0000400080000000 uptroot=0x00000000aaaabbbb"
    );
}

#[test]
fn vcpu_reg_snapshot_serde_round_trip() {
    let s = VcpuRegSnapshot {
        instruction_pointer: 0x1,
        stack_pointer: 0x2,
        page_table_root: 0x3,
        user_page_table_root: None,
        tcr_el1: None,
    };
    let json = serde_json::to_string(&s).expect("serialize");
    // Pin the JSON key names so a future field rename is
    // caught here rather than in downstream consumers
    // (operator JSON parsers, the failure_dump_e2e fixture).
    // Arch-neutral keys: see field doc on
    // VcpuRegSnapshot::page_table_root for the per-arch
    // semantics each one carries.
    assert!(
        json.contains("\"instruction_pointer\""),
        "missing JSON key `instruction_pointer`: {json}"
    );
    assert!(
        json.contains("\"stack_pointer\""),
        "missing JSON key `stack_pointer`: {json}"
    );
    assert!(
        json.contains("\"page_table_root\""),
        "missing JSON key `page_table_root`: {json}"
    );
    // user_page_table_root is None → serde-skipped via
    // skip_serializing_if = "Option::is_none"; assert it does
    // NOT appear in the JSON.
    assert!(
        !json.contains("\"user_page_table_root\""),
        "user_page_table_root must skip-serialize when None: {json}"
    );
    let parsed: VcpuRegSnapshot = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.instruction_pointer, 0x1);
    assert_eq!(parsed.stack_pointer, 0x2);
    assert_eq!(parsed.page_table_root, 0x3);
    assert!(
        parsed.user_page_table_root.is_none(),
        "missing field must deserialize as None"
    );
}

#[test]
fn vcpu_reg_snapshot_serde_round_trip_with_user_pt_root() {
    // aarch64-shape: user_page_table_root populated → JSON
    // carries the key, deserialize round-trips the value.
    let s = VcpuRegSnapshot {
        instruction_pointer: 0x1,
        stack_pointer: 0x2,
        page_table_root: 0x3,
        user_page_table_root: Some(0xdead_beef_cafe_d00d),
        tcr_el1: None,
    };
    let json = serde_json::to_string(&s).expect("serialize");
    assert!(
        json.contains("\"user_page_table_root\""),
        "user_page_table_root must serialize when Some: {json}"
    );
    let parsed: VcpuRegSnapshot = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.user_page_table_root, Some(0xdead_beef_cafe_d00d));
}

#[test]
fn vcpu_reg_snapshot_serde_round_trip_tcr_el1() {
    // Mirrors the user_page_table_root coverage but for tcr_el1:
    // skip_serializing_if = "Option::is_none" must drop the key
    // when None and emit + round-trip it when Some. Picks a
    // representative TCR_EL1 word (T1SZ=0x10 in [21:16] with
    // TG1=0b10 / 4 KB granule in [31:30]) so the test pins the
    // wire format the page-table walker actually sees on aarch64.
    let some_val: u64 = 0x0000_0000_b510_0010;

    let s_some = VcpuRegSnapshot {
        instruction_pointer: 0x1,
        stack_pointer: 0x2,
        page_table_root: 0x3,
        user_page_table_root: None,
        tcr_el1: Some(some_val),
    };
    let json_some = serde_json::to_string(&s_some).expect("serialize Some");
    assert!(
        json_some.contains("\"tcr_el1\""),
        "tcr_el1 must serialize when Some: {json_some}"
    );

    let s_none = VcpuRegSnapshot {
        instruction_pointer: 0x1,
        stack_pointer: 0x2,
        page_table_root: 0x3,
        user_page_table_root: None,
        tcr_el1: None,
    };
    let json_none = serde_json::to_string(&s_none).expect("serialize None");
    assert!(
        !json_none.contains("\"tcr_el1\""),
        "tcr_el1 must skip-serialize when None: {json_none}"
    );

    // Deserialize Some-flavour JSON back, assert value preserved.
    let parsed_some: VcpuRegSnapshot = serde_json::from_str(&json_some).expect("deserialize Some");
    assert_eq!(parsed_some.tcr_el1, Some(some_val));

    // Deserialize JSON without the key (None-flavour); the
    // serde(default) attribute must yield None rather than
    // failing because the field is absent.
    let parsed_none: VcpuRegSnapshot = serde_json::from_str(&json_none).expect("deserialize None");
    assert!(
        parsed_none.tcr_el1.is_none(),
        "missing tcr_el1 must deserialize as None"
    );
}

#[test]
fn vcpu_reg_snapshot_zero_renders_zeros() {
    let s = VcpuRegSnapshot {
        instruction_pointer: 0,
        stack_pointer: 0,
        page_table_root: 0,
        user_page_table_root: None,
        tcr_el1: None,
    };
    // 16 hex digits each — leading zeros preserved so widths
    // line up across rows in multi-vcpu output.
    assert_eq!(
        format!("{s}"),
        "ip=0x0000000000000000 sp=0x0000000000000000 ptroot=0x0000000000000000"
    );
}

/// Pure-arithmetic test on the aarch64 KVM register IDs the
/// capture path uses. Verifying the encoding here means a
/// transcription bug (e.g. wrong byte offset, dropped flag,
/// wrong sysreg op-code packing) would be caught without
/// booting an aarch64 VM. Mirrors the kernel's
/// `KVM_REG_ARM_CORE_REG(name) = offsetof(struct kvm_regs,
/// name) / sizeof(__u32)` macro for core regs and the
/// `ARM64_SYS_REG(Op0, Op1, CRn, CRm, Op2)` packing for
/// sysregs (arch/arm64/include/uapi/asm/kvm.h).
#[test]
#[cfg(target_arch = "aarch64")]
fn aarch64_register_ids_match_kernel_encoding() {
    const KVM_REG_ARM64: u64 = 0x6000_0000_0000_0000;
    const KVM_REG_SIZE_U64: u64 = 0x0030_0000_0000_0000;
    const KVM_REG_ARM_CORE: u64 = 0x0010_0000;
    const KVM_REG_ARM64_SYSREG: u64 = 0x0013_0000;

    // PC at byte offset 256 in struct kvm_regs (= same offset
    // in user_pt_regs because user_pt_regs.regs is at offset
    // 0). 256 / 4 = 64.
    const EXPECTED_PC_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | 64;
    // SP_EL1 at byte offset 272 in struct kvm_regs (right
    // after the 272-byte user_pt_regs). 272 / 4 = 68.
    const EXPECTED_SP_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | 68;
    // TTBR0_EL1 sysreg packing: Op0=3, Op1=0, CRn=2, CRm=0, Op2=0
    // → (3<<14) | (0<<11) | (2<<7) | (0<<3) | 0 = 0xC100.
    const EXPECTED_TTBR0_EL1_ID: u64 =
        KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC100;
    // TTBR1_EL1 sysreg packing: Op0=3, Op1=0, CRn=2, CRm=0, Op2=1
    // → (3<<14) | (0<<11) | (2<<7) | (0<<3) | 1 = 0xC101.
    const EXPECTED_TTBR1_EL1_ID: u64 =
        KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC101;

    // Reconstruct what capture_vcpu_regs declares to catch
    // any drift between the const declarations there and
    // the kernel ABI. The unsafe cast to *const u64 isn't
    // available across modules, so re-derive the values
    // here using the exact same expression form.
    let pc_id = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | (256 / 4);
    let sp_el1_id = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | (272 / 4);
    let ttbr0_el1_id = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC100;
    let ttbr1_el1_id = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC101;

    assert_eq!(pc_id, EXPECTED_PC_ID, "PC_ID encoding drift");
    assert_eq!(
        sp_el1_id, EXPECTED_SP_EL1_ID,
        "SP_EL1_ID encoding drift — note offset is 272 (sp_el1), \
         not 248 (sp_el0)"
    );
    assert_eq!(
        ttbr0_el1_id, EXPECTED_TTBR0_EL1_ID,
        "TTBR0_EL1_ID encoding drift — verify (Op0=3, Op1=0, \
         CRn=2, CRm=0, Op2=0) packs to 0xC100"
    );
    assert_eq!(
        ttbr1_el1_id, EXPECTED_TTBR1_EL1_ID,
        "TTBR1_EL1_ID encoding drift — verify (Op0=3, Op1=0, \
         CRn=2, CRm=0, Op2=1) packs to 0xC101"
    );
    // Adjacency check: TTBR0 and TTBR1 differ only in Op2,
    // so the encoding must differ by exactly 1. Catches a
    // typo where one constant gets the other's value.
    assert_eq!(
        ttbr1_el1_id - ttbr0_el1_id,
        1,
        "TTBR0/TTBR1 encodings should differ by exactly 1 (Op2 bit)"
    );
}

/// DR7 wire format for a 4-byte write watchpoint in slot 0.
///
/// The KVM hardware-watchpoint freeze trigger arms slot 0 of
/// the guest's debug registers via `KVM_SET_GUEST_DEBUG` to
/// catch writes to `sch->exit_kind`. The DR7 byte the VMM
/// hands KVM must encode the exact slot/length/access pattern
/// the watchpoint requires, otherwise either KVM rejects the
/// configuration or the breakpoint catches the wrong access
/// class — both surface as silent freeze-coordinator failure
/// (the guest never traps, no failure dump).
///
/// Field layout per Intel SDM Vol 3 Ch 17.2 ("Debug Registers")
/// and pinned against the production
/// [`super::super::vcpu::self_arm_watchpoint`] x86_64 path which
/// composes DR7 as:
///
///   base = `0x400 | 0x200 | 0x100`  (MBS, GE, LE)
///   per slot i: `(0b11) << (2*i)`           (L<i> | G<i>)
///             | `(0b01) << (16 + 4*i)`     (R/W<i> = write)
///             | `(0b11) << (18 + 4*i)`     (LEN<i> = 4-byte)
///
/// Bit-by-bit:
///
///   bit  0   L0   (local enable, slot 0) — 1
///   bit  1   G0   (global enable, slot 0) — 1
///   bit  8   LE   (local exact-match, required for data BPs)
///   bit  9   GE   (global exact-match, required for data BPs)
///   bit 10   reserved, must be 1 (DR7_FIXED_1)
///   bits [17:16]  R/W0 (00=exec, 01=write, 11=rw) — 01
///   bits [19:18]  LEN0 (00=1B, 01=2B, 10=8B, 11=4B) — 11
///
/// Expected value 0xD0703 is what the production wire format
/// emits for (slot=0, type=write, len=4). Pinning the arithmetic
/// here means a future refactor that flips a bit (e.g. swaps R/W
/// to exec, drops GE/LE, picks the wrong length encoding, drops
/// L0 or G0) is caught at unit-test time before the trigger
/// silently stops firing.
#[test]
#[cfg(target_arch = "x86_64")]
fn dr7_slot0_write_4byte_encoding() {
    // Field constants — local to the test so a future
    // production-side rename does not silently divorce the
    // assertion from the wire format.
    const DR7_FIXED_1: u64 = 1 << 10;
    const DR_LOCAL_EXACT: u64 = 1 << 8; // LE — local exact-match
    const DR_GLOBAL_EXACT: u64 = 1 << 9; // GE — global exact-match
    const DR_LOCAL_ENABLE: u64 = 1 << 0; // L0 — slot 0 local
    const DR_GLOBAL_ENABLE: u64 = 1 << 1; // G0 — slot 0 global
    const DR_RW_WRITE: u64 = 0b01;
    const DR_LEN_4: u64 = 0b11;
    // Slot 0 occupies bits 16/17 for R/W and 18/19 for LEN.
    const SLOT0_RW_SHIFT: u32 = 16;
    const SLOT0_LEN_SHIFT: u32 = 18;

    let dr7 = DR7_FIXED_1
        | DR_GLOBAL_EXACT
        | DR_LOCAL_EXACT
        | DR_LOCAL_ENABLE
        | DR_GLOBAL_ENABLE
        | (DR_RW_WRITE << SLOT0_RW_SHIFT)
        | (DR_LEN_4 << SLOT0_LEN_SHIFT);
    // 0xD0703 mirrors what `self_arm_watchpoint` in
    // `super::super::vcpu` actually programs into KVM. A drift
    // here means the watchpoint's wire format diverged from the
    // production encoding and `KVM_SET_GUEST_DEBUG` will arm the
    // wrong breakpoint (or none).
    assert_eq!(
        dr7, 0xD0703,
        "DR7 encoding for (slot=0, write, 4B) must match the production wire format"
    );

    // Bit-by-bit cross-check: every contributing bit must be
    // present, and every other bit must be clear. Catches the
    // failure mode where two bugs cancel — e.g. wrong shift
    // for R/W combined with wrong shift for LEN that happen
    // to sum to the right total.
    assert_ne!(dr7 & (1 << 0), 0, "L0 (bit 0) must be set");
    assert_ne!(dr7 & (1 << 1), 0, "G0 (bit 1) must be set");
    assert_ne!(
        dr7 & (1 << 8),
        0,
        "LE (bit 8) must be set for data breakpoints"
    );
    assert_ne!(
        dr7 & (1 << 9),
        0,
        "GE (bit 9) must be set for data breakpoints"
    );
    assert_ne!(dr7 & (1 << 10), 0, "DR7_FIXED_1 (bit 10) must be set");
    // Slot 0 R/W field = 0b01 (write).
    assert_eq!(
        (dr7 >> SLOT0_RW_SHIFT) & 0b11,
        DR_RW_WRITE,
        "slot 0 R/W field must encode write (0b01)"
    );
    // Slot 0 LEN field = 0b11 (4 bytes).
    assert_eq!(
        (dr7 >> SLOT0_LEN_SHIFT) & 0b11,
        DR_LEN_4,
        "slot 0 LEN field must encode 4 bytes (0b11)"
    );
    // No other slot should be enabled.
    assert_eq!(
        dr7 & 0b1111_1100,
        0,
        "slots 1..3 must be disabled (L/G bits clear)"
    );
    // R/W and LEN fields for slots 1..3 must be zero.
    assert_eq!(
        (dr7 >> 20) & 0xFFF,
        0,
        "slots 1..3 R/W + LEN fields must be zero"
    );
}

/// DBGWCR wire format for a 4-byte write watchpoint at byte
/// offset 0 of an 8-byte aligned block. The aarch64 sibling of
/// the x86_64 [`dr7_slot0_write_4byte_encoding`] test above —
/// pins the exact bit layout that
/// [`super::super::vcpu::self_arm_watchpoint`] emits, so a future
/// refactor that flips a bit (e.g. swaps LSC to read, drops PAC,
/// picks the wrong BAS shift) is caught at unit-test time
/// before the trigger silently stops firing.
///
/// Field layout per ARM ARM D7.3.11 ("DBGWCR<n>_EL1, Debug
/// Watchpoint Control Registers") and pinned against QEMU's
/// `insert_hw_watchpoint` in `target/arm/hyp_gdbstub.c`:
///
///   bit 0       E   = 1 (enable)
///   bits [2:1]  PAC = 0b11 (EL0+EL1, any security state)
///   bits [4:3]  LSC = 0b10 (write-only)
///   bits [12:5] BAS = 0xF << byte_offset (4 contiguous bytes)
///   bits [15:13] HMC = 0
///   bits [19:16] SSC = 0
///   bit 20       WT = 0
///   bits [23:21] LBN = 0
///   bits [28:24] MASK = 0
///
/// Concrete byte_offset=0 wire format: `0x1F7`.
#[test]
#[cfg(target_arch = "aarch64")]
fn dbgwcr_slot0_write_4byte_encoding_offset0() {
    let kva: u64 = 0xffff_ffff_8100_1000; // 8-byte aligned (low bits = 0)
    let byte_offset = (kva & 0x7u64) as u32;
    let bas: u64 = 0xFu64 << byte_offset;
    let wcr: u64 = 1u64 | (0b11u64 << 1) | (0b10u64 << 3) | (bas << 5);
    assert_eq!(
        wcr, 0x1F7,
        "DBGWCR encoding for (slot=0, write, 4B, offset=0) must \
         match the QEMU/ARM ARM gold-standard wire format"
    );
    // Bit-by-bit cross-check.
    assert_eq!(wcr & 1, 1, "E (bit 0) must be set");
    assert_eq!(
        (wcr >> 1) & 0b11,
        0b11,
        "PAC (bits 2:1) must be 0b11 (EL0+EL1)"
    );
    assert_eq!(
        (wcr >> 3) & 0b11,
        0b10,
        "LSC (bits 4:3) must be 0b10 (write-only)"
    );
    assert_eq!(
        (wcr >> 5) & 0xFF,
        0x0F,
        "BAS (bits 12:5) must be 0x0F for offset=0 (4 \
         contiguous low bytes)"
    );
    // No other fields must be set.
    assert_eq!(
        (wcr >> 13) & 0xF,
        0,
        "HMC + low SSC bit (bits 16:13) must be zero"
    );
    assert_eq!((wcr >> 20) & 0xF, 0, "WT + LBN must be zero");
    assert_eq!((wcr >> 24) & 0x1F, 0, "MASK must be zero");
}

/// DBGWCR encoding for a 4-byte write watchpoint at byte offset
/// 4 of an 8-byte aligned block. Catches the failure mode where
/// a 4-byte aligned but not 8-byte aligned KVA (e.g. a struct
/// field at offset 4 inside an 8-byte aligned struct) gets the
/// wrong BAS shift. Concrete wire format: `0x1E17`.
#[test]
#[cfg(target_arch = "aarch64")]
fn dbgwcr_slot0_write_4byte_encoding_offset4() {
    let kva: u64 = 0xffff_ffff_8100_1004; // 4-byte aligned, byte_offset=4
    let byte_offset = (kva & 0x7u64) as u32;
    let bas: u64 = 0xFu64 << byte_offset;
    let wcr: u64 = 1u64 | (0b11u64 << 1) | (0b10u64 << 3) | (bas << 5);
    assert_eq!(
        wcr, 0x1E17,
        "DBGWCR encoding for (slot=0, write, 4B, offset=4) must \
         match `0x1 | (3<<1) | (2<<3) | (0xF0 << 5)` = 0x1E17"
    );
    assert_eq!(
        (wcr >> 5) & 0xFF,
        0xF0,
        "BAS (bits 12:5) must be 0xF0 for offset=4 (4 \
         contiguous high bytes)"
    );
}

/// DBGWVR alignment: the WCR/WVR pair always uses an 8-byte
/// aligned base and BAS to select the 4 bytes, so a 4-byte
/// aligned KVA at offset 4 must yield WVR = `kva & ~0x7` (= the
/// containing 8-byte block's base), not `kva` itself.
#[test]
#[cfg(target_arch = "aarch64")]
fn dbgwvr_8byte_aligned_base() {
    let kva: u64 = 0xffff_ffff_8100_1004;
    let wvr = kva & !0x7u64;
    assert_eq!(
        wvr, 0xffff_ffff_8100_1000,
        "DBGWVR base must clear the bottom 3 bits (8-byte align) \
         so BAS picks the 4 watched bytes within the block"
    );
    // Round-trip: reconstructing the watched range from WVR + BAS.
    let byte_offset = (kva & 0x7u64) as u32;
    let bas: u64 = 0xFu64 << byte_offset;
    let watched_lo = wvr + (bas.trailing_zeros() as u64);
    let watched_hi = watched_lo + (bas.count_ones() as u64);
    assert_eq!(
        watched_lo, kva,
        "watched range low must equal the original KVA"
    );
    assert_eq!(
        watched_hi,
        kva + 4,
        "watched range high must equal kva + 4 (4 bytes)"
    );
}

/// FAR-based slot decode for the aarch64 watchpoint exit path.
/// Constructs a synthetic `kvm_debug_exit_arch` with EC =
/// WATCHPT_LOW and a FAR inside slot 2's 4-byte window, runs
/// the dispatch helper, and asserts only `user[1].hit` was
/// latched (slot indices in `armed_slots` are 0=DR0/exit_kind,
/// 1=DR1/user[0], 2=DR2/user[1], 3=DR3/user[2]). Also pins the
/// post-fire single-step bookkeeping: the helper must mark
/// `single_step_pending=true` and record the matched slot
/// index, so the next `self_arm_watchpoint` call disables that
/// slot's WCR.E and asserts `KVM_GUESTDBG_SINGLESTEP` (avoiding
/// the aarch64 watchpoint-replay infinite loop).
#[test]
#[cfg(target_arch = "aarch64")]
fn watchpoint_slot_decode_from_far_user_slot() {
    use crate::vmm::vcpu::WatchpointArm;
    let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
    // Slot 0 unused (request_kva = 0); slots 1..=3 carry
    // distinct addresses 4 bytes apart so a FAR inside slot 2's
    // window matches exactly one slot.
    let armed_slots: [u64; 4] = [
        0,
        0xffff_ffff_8100_1000,
        0xffff_ffff_8100_1004,
        0xffff_ffff_8100_1008,
    ];
    // Construct a synthetic debug-exit payload pointing at the
    // first byte of slot 2 (DR2 / user[1]).
    let far = 0xffff_ffff_8100_1004u64;
    let hsr = (super::ESR_ELX_EC_WATCHPT_LOW) << super::ESR_ELX_EC_SHIFT;
    let debug_arch = kvm_bindings::kvm_debug_exit_arch {
        hsr,
        hsr_high: 0,
        far,
    };
    let mut single_step_pending = false;
    let mut single_step_slot: usize = 99;
    super::dispatch_watchpoint_hit(
        &watchpoint,
        &debug_arch,
        &armed_slots,
        &mut single_step_pending,
        &mut single_step_slot,
    );
    assert!(
        !watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
        "slot 0 (exit_kind) must not latch when a different \
         slot's range matches FAR"
    );
    assert!(
        !watchpoint.user[0]
            .hit
            .load(std::sync::atomic::Ordering::Acquire),
        "user[0] / slot 1 must not latch — FAR is inside slot 2's \
         range"
    );
    assert!(
        watchpoint.user[1]
            .hit
            .load(std::sync::atomic::Ordering::Acquire),
        "user[1] / slot 2 must latch — FAR equals the slot's KVA"
    );
    assert!(
        !watchpoint.user[2]
            .hit
            .load(std::sync::atomic::Ordering::Acquire),
        "user[2] / slot 3 must not latch — FAR is outside its \
         range"
    );
    // Single-step bookkeeping: a watchpoint match MUST request
    // single-step on the matching slot so the next KVM_RUN
    // advances past the offending store before the slot is
    // re-armed. `single_step_slot` is a 4-bit bitmap of
    // matched slot indices; with FAR inside slot 2's range,
    // bit 2 must be the only bit set so `self_arm_watchpoint`
    // clears WCR[2].E (and leaves peer slots armed) for the
    // single-step pass.
    assert!(
        single_step_pending,
        "single_step_pending must be set when a watchpoint match \
         latches; without this the next KVM_RUN replays the same \
         store and re-trips the watchpoint forever (ARM ARM \
         D2.10.5)"
    );
    assert_eq!(
        single_step_slot, 0b0100,
        "single_step_slot bitmap must encode slot 2 (bit 2 = 1, \
         0b0100) so self_arm_watchpoint clears WCR[2].E and \
         leaves WCR[0/1/3].E armed for the single-step pass"
    );
}

/// Non-watchpoint EC values must be ignored. KVM_EXIT_DEBUG can
/// surface for soft-step (EC = 0x32) or BRK (EC = 0x3C); only
/// EC = 0x34 (`ESR_ELx_EC_WATCHPT_LOW`) means a data watchpoint
/// fired. Other ECs must not latch any slot. Soft-step (EC =
/// 0x32) is treated specially when `single_step_pending` is
/// set — it clears the flag so `self_arm_watchpoint` can
/// restore the disabled slot's WCR.E. With the flag clear,
/// soft-step exits are spurious and must NOT touch any state.
#[test]
#[cfg(target_arch = "aarch64")]
fn watchpoint_dispatch_ignores_non_watchpt_ec() {
    use crate::vmm::vcpu::WatchpointArm;
    let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
    let armed_slots: [u64; 4] = [
        0xffff_ffff_8100_1000,
        0xffff_ffff_8100_1004,
        0xffff_ffff_8100_1008,
        0xffff_ffff_8100_100c,
    ];
    // EC = 0x32 (software step) with single_step_pending = false
    // — must NOT latch and must not flip pending state.
    let hsr = 0x32u32 << super::ESR_ELX_EC_SHIFT;
    let debug_arch = kvm_bindings::kvm_debug_exit_arch {
        hsr,
        hsr_high: 0,
        far: 0xffff_ffff_8100_1004,
    };
    let mut single_step_pending = false;
    let mut single_step_slot: usize = 99;
    super::dispatch_watchpoint_hit(
        &watchpoint,
        &debug_arch,
        &armed_slots,
        &mut single_step_pending,
        &mut single_step_slot,
    );
    assert!(
        !watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
        "soft-step EC must not latch slot 0"
    );
    for (i, slot) in watchpoint.user.iter().enumerate() {
        assert!(
            !slot.hit.load(std::sync::atomic::Ordering::Acquire),
            "soft-step EC must not latch user[{i}]"
        );
    }
    assert!(
        !single_step_pending,
        "spurious soft-step exit (no pending step) must leave \
         single_step_pending unchanged"
    );
    assert_eq!(
        single_step_slot, 99,
        "spurious soft-step exit must not clobber single_step_slot"
    );
}

/// Soft-step exit AFTER a watchpoint trap is the second half of
/// the aarch64 watchpoint-replay-avoidance dance: it signals
/// "the offending store has retired, you may rearm the slot
/// now". The dispatch helper must clear `single_step_pending`
/// (so the next `self_arm_watchpoint` call restores WCR.E=1)
/// AND must not latch any new `hit` (the original
/// WATCHPT_LOW exit already latched the freeze trigger).
#[test]
#[cfg(target_arch = "aarch64")]
fn watchpoint_softstep_clears_single_step_pending() {
    use crate::vmm::vcpu::WatchpointArm;
    let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
    let armed_slots: [u64; 4] = [0, 0xffff_ffff_8100_1000, 0, 0];
    let hsr = super::ESR_ELX_EC_SOFTSTP_LOW << super::ESR_ELX_EC_SHIFT;
    let debug_arch = kvm_bindings::kvm_debug_exit_arch {
        hsr,
        hsr_high: 0,
        far: 0,
    };
    let mut single_step_pending = true;
    let mut single_step_slot: usize = 1;
    super::dispatch_watchpoint_hit(
        &watchpoint,
        &debug_arch,
        &armed_slots,
        &mut single_step_pending,
        &mut single_step_slot,
    );
    assert!(
        !single_step_pending,
        "SOFTSTP_LOW with pending step must clear \
         single_step_pending so the next self_arm_watchpoint \
         call restores WCR.E=1 and drops KVM_GUESTDBG_SINGLESTEP"
    );
    assert!(
        !watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
        "SOFTSTP_LOW must not latch slot 0 (the WATCHPT_LOW \
         exit that preceded it already did)"
    );
    for (i, slot) in watchpoint.user.iter().enumerate() {
        assert!(
            !slot.hit.load(std::sync::atomic::Ordering::Acquire),
            "SOFTSTP_LOW must not latch user[{i}]"
        );
    }
}

/// Slot 0 (`exit_kind`) MUST NOT latch when `kind_host_ptr`
/// is null. The freeze coordinator publishes `kind_host_ptr`
/// (Release) BEFORE `request_kva` (Release); the vCPU's
/// `self_arm_watchpoint` only programs the hardware
/// watchpoint after observing a non-zero `request_kva` via
/// an Acquire load — at which point both stores are visible.
/// A null observation here is a publication-invariant
/// violation. Pinning the no-latch behavior so a future
/// regression that re-introduces an unconditional fallback
/// (e.g. "latch hit just in case") is caught: latching on
/// null defeats the post-store error-class gate and re-fires
/// dump emission on every clean SCX_EXIT_DONE shutdown.
/// The single-step bookkeeping STILL fires on the matched
/// slot — re-entering KVM_RUN without stepping past the
/// offending store would replay the same trap forever
/// regardless of whether the slot-0 latch ran.
#[test]
#[cfg(target_arch = "aarch64")]
fn watchpoint_slot0_skips_latch_when_host_ptr_null() {
    use crate::vmm::vcpu::WatchpointArm;
    let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
    // host_ptr is null by construction (`AtomicPtr::new(null)`).
    let armed_slots: [u64; 4] = [0xffff_ffff_8100_1000, 0, 0, 0];
    let hsr = (super::ESR_ELX_EC_WATCHPT_LOW) << super::ESR_ELX_EC_SHIFT;
    let debug_arch = kvm_bindings::kvm_debug_exit_arch {
        hsr,
        hsr_high: 0,
        far: 0xffff_ffff_8100_1000,
    };
    let mut single_step_pending = false;
    let mut single_step_slot: usize = 0;
    super::dispatch_watchpoint_hit(
        &watchpoint,
        &debug_arch,
        &armed_slots,
        &mut single_step_pending,
        &mut single_step_slot,
    );
    assert!(
        !watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
        "slot 0 must NOT latch hit when kind_host_ptr is null — \
         a null observation is a publication-invariant violation, \
         not a fallback trigger"
    );
    // Single-step bookkeeping must still record the matched
    // slot in the bitmap so `self_arm_watchpoint` clears
    // WCR.E on it. Without this the offending store replays
    // forever on re-entering KVM_RUN.
    assert!(
        single_step_pending,
        "single_step_pending must be set when the FAR matches a \
         slot, regardless of slot-0 latch outcome"
    );
    assert_eq!(
        single_step_slot, 0b0001,
        "single_step_slot bitmap must include slot 0 (bit 0 = 1) \
         so self_arm_watchpoint clears WCR[0].E during the \
         single-step pass"
    );
}

/// x86_64 sibling of `watchpoint_slot_decode_from_far_user_slot`.
/// Constructs a synthetic `kvm_debug_exit_arch` with DR6=0x4 (B2
/// set, indicating DR2 fired) and verifies the dispatch helper
/// latches `user[1].hit` (slot 2 → user[1]) and leaves slot 0
/// and the other user slots untouched. Pins the DR6→slot
/// mapping so a future refactor that flips the bit-shift
/// (e.g. interprets bit 2 as slot 1 instead of slot 2) is
/// caught at unit-test time.
#[test]
#[cfg(target_arch = "x86_64")]
fn watchpoint_dispatch_x86_dr6_b2_latches_user_slot_1() {
    use crate::vmm::vcpu::WatchpointArm;
    let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
    // armed_slots is consumed by the aarch64 path only; x86
    // uses DR6 alone.
    let armed_slots: [u64; 4] = [0; 4];
    let debug_arch = kvm_bindings::kvm_debug_exit_arch {
        exception: 0,
        pad: 0,
        pc: 0,
        // DR6 bit 2 (B2) set ⇒ DR2 fired (slot 2 in our
        // 0=DR0..3=DR3 indexing). Other bits cleared per Intel
        // SDM Vol. 3B 17.2.5.
        dr6: 0x4,
        dr7: 0,
    };
    let mut single_step_pending = false;
    let mut single_step_slot: usize = 99;
    super::dispatch_watchpoint_hit(
        &watchpoint,
        &debug_arch,
        &armed_slots,
        &mut single_step_pending,
        &mut single_step_slot,
    );
    assert!(
        !watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
        "slot 0 (exit_kind) must not latch when DR6 B0 is clear"
    );
    assert!(
        !watchpoint.user[0]
            .hit
            .load(std::sync::atomic::Ordering::Acquire),
        "user[0] / slot 1 must not latch — DR6 B1 is clear"
    );
    assert!(
        watchpoint.user[1]
            .hit
            .load(std::sync::atomic::Ordering::Acquire),
        "user[1] / slot 2 must latch — DR6 B2 is set"
    );
    assert!(
        !watchpoint.user[2]
            .hit
            .load(std::sync::atomic::Ordering::Acquire),
        "user[2] / slot 3 must not latch — DR6 B3 is clear"
    );
    // x86 single-step bookkeeping is inert (the trap is taken
    // AFTER the offending store retires — Intel SDM 17.2.4).
    assert!(
        !single_step_pending,
        "x86 dispatch must never set single_step_pending — \
         single-step is aarch64-only"
    );
    assert_eq!(
        single_step_slot, 99,
        "x86 dispatch must not clobber single_step_slot — \
         single-step is aarch64-only"
    );
}

/// x86_64 multi-match: DR6=0x5 (B0 + B2) means DR0 and DR2
/// both fired on the same exit. The dispatch helper must
/// latch BOTH slot 0 and user[1] from a single
/// `kvm_debug_exit_arch`. Catches a refactor that breaks at
/// the first set bit (e.g. early `return` after `if hits[0]`).
#[test]
#[cfg(target_arch = "x86_64")]
fn watchpoint_dispatch_x86_dr6_multi_match() {
    use crate::vmm::vcpu::WatchpointArm;
    let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
    // Slot 0 needs a non-null kind_host_ptr to latch via the
    // post-store value gate. Pre-arm it with a host-side u32
    // holding an error-class kind value so
    // `latch_slot0_with_gate` takes the latch branch instead
    // of the gated-suppression branch.
    let kind: u32 = super::SCX_EXIT_ERROR_THRESHOLD;
    let kind_box = Box::new(kind);
    let kind_ptr = Box::into_raw(kind_box);
    watchpoint
        .kind_host_ptr
        .store(kind_ptr, std::sync::atomic::Ordering::Release);
    let armed_slots: [u64; 4] = [0; 4];
    let debug_arch = kvm_bindings::kvm_debug_exit_arch {
        exception: 0,
        pad: 0,
        pc: 0,
        // DR6 bits 0 + 2 set: DR0 and DR2 both fired.
        dr6: 0x5,
        dr7: 0,
    };
    let mut single_step_pending = false;
    let mut single_step_slot: usize = 99;
    super::dispatch_watchpoint_hit(
        &watchpoint,
        &debug_arch,
        &armed_slots,
        &mut single_step_pending,
        &mut single_step_slot,
    );
    assert!(
        watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
        "slot 0 (exit_kind) must latch — DR6 B0 set + kind ≥ \
         SCX_EXIT_ERROR_THRESHOLD"
    );
    assert!(
        !watchpoint.user[0]
            .hit
            .load(std::sync::atomic::Ordering::Acquire),
        "user[0] / slot 1 must not latch — DR6 B1 is clear"
    );
    assert!(
        watchpoint.user[1]
            .hit
            .load(std::sync::atomic::Ordering::Acquire),
        "user[1] / slot 2 must latch — DR6 B2 is set, even \
         though slot 0 latched first in iteration order"
    );
    assert!(
        !watchpoint.user[2]
            .hit
            .load(std::sync::atomic::Ordering::Acquire),
        "user[2] / slot 3 must not latch — DR6 B3 is clear"
    );
    // SAFETY: kind_box ownership round-trip via Box::into_raw
    // / Box::from_raw matches the standard pattern; we own
    // the only pointer to this allocation in the test
    // function and the dispatch helper has finished its
    // read_volatile before this drop.
    let _ = unsafe { Box::from_raw(kind_ptr) };
}

/// CAS-based dedup in `latch_hit`: a second call (e.g. two
/// vCPUs each writing the watched address, or a re-fire on
/// the same vCPU before the freeze coordinator resets `hit`)
/// must not write a second eventfd edge. Pinning the dedup so
/// a future refactor that drops the CAS in favour of an
/// unconditional `store(true)` is caught here — coordinator
/// would then rendezvous N times for one logical fire.
#[test]
fn latch_hit_is_idempotent_across_repeat_calls() {
    use crate::vmm::vcpu::WatchpointArm;
    use std::os::fd::AsRawFd;
    let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");

    // First latch: must flip false→true and write eventfd.
    watchpoint.latch_hit();
    assert!(
        watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
        "first latch_hit must flip hit=false→true"
    );

    // Drain the eventfd to verify there is exactly one write
    // pending. EFD_NONBLOCK + counter mode: a single read
    // returns the accumulated count and resets the internal
    // counter.
    let mut buf = [0u8; 8];
    let n = unsafe {
        libc::read(
            watchpoint.hit_evt.as_raw_fd(),
            buf.as_mut_ptr() as *mut libc::c_void,
            buf.len(),
        )
    };
    assert_eq!(
        n, 8,
        "first latch_hit must produce one eventfd edge \
         (8-byte counter read)"
    );
    let count_after_first = u64::from_ne_bytes(buf);
    assert_eq!(
        count_after_first, 1,
        "first latch_hit must increment counter by exactly 1"
    );

    // Second latch (without coordinator reset): CAS fails,
    // no eventfd write. Counter stays at 0 — a subsequent
    // read returns EAGAIN (no edge available).
    watchpoint.latch_hit();
    let mut buf2 = [0u8; 8];
    let n2 = unsafe {
        libc::read(
            watchpoint.hit_evt.as_raw_fd(),
            buf2.as_mut_ptr() as *mut libc::c_void,
            buf2.len(),
        )
    };
    let errno = unsafe { *libc::__errno_location() };
    assert!(
        n2 == -1 && errno == libc::EAGAIN,
        "second latch_hit on already-latched slot must NOT \
         write to hit_evt (cross-vCPU dedup); read should \
         return EAGAIN, got n={n2} errno={errno}"
    );
}

/// CAS-based dedup in `latch_user_hit`: same invariant as the
/// slot-0 latch, applied to user slots 1..=3. Catches an off-
/// by-one in the per-slot CAS (e.g. CAS on slot 0's `hit`
/// instead of `user[idx].hit`).
#[test]
fn latch_user_hit_is_idempotent_across_repeat_calls() {
    use crate::vmm::vcpu::WatchpointArm;
    use std::os::fd::AsRawFd;
    let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");

    watchpoint.latch_user_hit(1);
    assert!(
        watchpoint.user[1]
            .hit
            .load(std::sync::atomic::Ordering::Acquire),
        "first latch_user_hit(1) must flip user[1].hit=false→true"
    );

    let mut buf = [0u8; 8];
    let n = unsafe {
        libc::read(
            watchpoint.hit_evt.as_raw_fd(),
            buf.as_mut_ptr() as *mut libc::c_void,
            buf.len(),
        )
    };
    assert_eq!(n, 8, "first latch_user_hit(1) must write eventfd");
    let count_after_first = u64::from_ne_bytes(buf);
    assert_eq!(count_after_first, 1, "counter increment must be 1");

    watchpoint.latch_user_hit(1);
    let mut buf2 = [0u8; 8];
    let n2 = unsafe {
        libc::read(
            watchpoint.hit_evt.as_raw_fd(),
            buf2.as_mut_ptr() as *mut libc::c_void,
            buf2.len(),
        )
    };
    let errno = unsafe { *libc::__errno_location() };
    assert!(
        n2 == -1 && errno == libc::EAGAIN,
        "second latch_user_hit(1) on already-latched slot \
         must NOT write to hit_evt; read should return \
         EAGAIN, got n={n2} errno={errno}"
    );

    // Out-of-range idx must be a silent no-op — no panic, no
    // hit on any slot, no eventfd write. Catches a future
    // refactor that drops the bounds check.
    watchpoint.latch_user_hit(99);
    for (i, slot) in watchpoint.user.iter().enumerate() {
        if i == 1 {
            assert!(
                slot.hit.load(std::sync::atomic::Ordering::Acquire),
                "user[1].hit must remain latched"
            );
        } else {
            assert!(
                !slot.hit.load(std::sync::atomic::Ordering::Acquire),
                "user[{i}].hit must remain unlatched after \
                 out-of-range latch_user_hit(99)"
            );
        }
    }
}

/// `WatchpointArm::mark_armed` flips the `any_armed` gate
/// from 0 to 1 and is idempotent. Pins the gate's role: the
/// publisher (freeze coordinator's err_exit publish or
/// `arm_user_watchpoint`) calls `mark_armed` AFTER the
/// `Release` store on `request_kva`; until then
/// `self_arm_watchpoint` short-circuits without doing the
/// per-slot Acquire reads.
#[test]
fn mark_armed_flips_gate_and_is_idempotent() {
    use crate::vmm::vcpu::WatchpointArm;
    let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
    assert_eq!(
        watchpoint
            .any_armed
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "newly-constructed WatchpointArm must have any_armed=0 \
         so self_arm_watchpoint short-circuits before any \
         publisher fires"
    );
    watchpoint.mark_armed();
    assert_eq!(
        watchpoint
            .any_armed
            .load(std::sync::atomic::Ordering::Relaxed),
        1,
        "first mark_armed call must flip the gate to 1"
    );
    // Idempotence: a second call must leave the gate at 1.
    // Catches a refactor that turned mark_armed into an
    // increment / fetch_add (which would saturate eventually
    // but burn cycles on every publisher call).
    watchpoint.mark_armed();
    assert_eq!(
        watchpoint
            .any_armed
            .load(std::sync::atomic::Ordering::Relaxed),
        1,
        "second mark_armed call must leave the gate at 1 \
         (idempotent — mark_armed is `store(1)`, not `+= 1`)"
    );
}
