use super::*;

// ----- Register protection tests ------------------------------
//
// R10 is the read-only frame pointer per the BPF ABI; the
// analyzer enforces the invariant that `regs[10]` is always
// Unknown by guarding MOV and LDX early. Out-of-range register
// indices (11..=15) can appear in a malformed instruction stream
// because BpfInsn packs each field into 4 bits; the bounds check
// at the top of step() and each handle_* routine rejects without
// panicking.

/// MOV with `dst == r10` is rejected by the production guard in
/// the ALU64-MOV-X arm (`if dst == BPF_REG_R10 { return; }`).
/// Verifying r10's state directly is impossible because the
/// stack-spill / reload path keys on the register index, not on
/// `regs[10]`'s `RegState`. The probe instead routes through a
/// second MOV: `MOV r3, r10` copies `regs[10]` into `regs[3]`,
/// then a deref chain through r3 records a cast iff `regs[10]`
/// was typed. With the rejection working, `regs[10]` stays
/// Unknown, so r3 stays Unknown, and the deref chain produces no
/// record.
#[test]
fn mov_to_r10_rejected_keeps_r10_unknown() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // r2 = *(u64 *)(r1 + 8)   -- r2 = LoadedU64Field{T, 8}
    // r10 = r2                -- REJECTED, r10 stays Unknown
    // r3 = r10                -- r3 = regs[10] = Unknown
    // r4 = *(u64 *)(r3 + 0)   -- r3 Unknown, no record
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        mov_x(10, 2),
        mov_x(3, 10),
        ldx(BPF_SIZE_DW, 4, 3, 0),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: t_id,
        }],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "MOV r10, r2 must be rejected so r10 stays Unknown: {map:?}"
    );
}

/// LDX with `dst == r10` is rejected by the production guard
/// (`if dst == BPF_REG_R10 { return; }` in `handle_ldx`). The
/// same routing trick as `mov_to_r10_rejected_keeps_r10_unknown`
/// observes the rejection: a successful LDX into r10 would have
/// seeded `regs[10] = LoadedU64Field`, and a follow-up
/// `MOV r3, r10; LDX r4, [r3+0]` chain would record a cast.
/// With the guard active, r10 stays Unknown and the chain
/// produces nothing.
#[test]
fn ldx_into_r10_rejected_keeps_r10_unknown() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // r10 = *(u64 *)(r1 + 8)  -- REJECTED, r10 stays Unknown
    // r3 = r10                -- r3 = regs[10] = Unknown
    // r4 = *(u64 *)(r3 + 0)   -- r3 Unknown, no record
    let insns = vec![
        ldx(BPF_SIZE_DW, 10, 1, 8),
        mov_x(3, 10),
        ldx(BPF_SIZE_DW, 4, 3, 0),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: t_id,
        }],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "LDX r10, [r1+8] must be rejected so r10 stays Unknown: {map:?}"
    );
}

/// `BPF_STX | BPF_DW | BPF_MEM` with both `dst` and `src` out of
/// the 0..=10 valid register range (encoded as 15) must NOT panic.
/// `BpfInsn::new` masks each register field to 4 bits, so 15
/// round-trips through `dst_reg()` / `src_reg()` as 15. The bounds
/// check at the top of `step()` (and the redundant guard in
/// `handle_stx`) reject before any array indexing.
#[test]
fn oob_stx_reg_does_not_panic() {
    let (blob, _t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let bad = BpfInsn::new(BPF_CLASS_STX | BPF_SIZE_DW | BPF_MODE_MEM, 15, 15, 0, 0);
    let insns = vec![bad, exit()];
    let map = analyze_casts(&insns, &btf, &[], &[], &[], &[]);
    assert!(
        map.is_empty(),
        "OOB STX (dst=15, src=15) must not panic: {map:?}"
    );
}

/// `BPF_ALU64 | BPF_OP_MOV | BPF_SRC_X` with `dst == 15` (out of
/// range) must NOT panic. The bounds check at the top of `step()`
/// rejects before the MOV branch executes.
#[test]
fn oob_mov_reg_does_not_panic() {
    let (blob, _t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let bad = BpfInsn::new(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 15, 0, 0, 0);
    let insns = vec![bad, exit()];
    let map = analyze_casts(&insns, &btf, &[], &[], &[], &[]);
    assert!(map.is_empty(), "OOB MOV (dst=15) must not panic: {map:?}");
}

/// `BPF_STX | BPF_DW | BPF_ATOMIC` with `dst` and `src` out of
/// range (15) must NOT panic. The bounds check at the top of
/// `step()` rejects before dispatch into `handle_atomic`; the
/// redundant guard at the top of `handle_atomic` is a defense-in-
/// depth backstop.
#[test]
fn oob_atomic_reg_does_not_panic() {
    let (blob, _t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let bad = BpfInsn::new(
        BPF_CLASS_STX | BPF_SIZE_DW | BPF_MODE_ATOMIC,
        15,
        15,
        0,
        BPF_FETCH | 0xe0,
    );
    let insns = vec![bad, exit()];
    let map = analyze_casts(&insns, &btf, &[], &[], &[], &[]);
    assert!(
        map.is_empty(),
        "OOB ATOMIC (dst=15, src=15) must not panic: {map:?}"
    );
}

/// `MOV dst, src` where the source register is `Unknown`
/// overwrites the destination's typed state with `Unknown`.
/// Production unconditionally copies `regs[src]` into `regs[dst]`,
/// so a previously-typed dst loses its `Pointer{T}` /
/// `LoadedU64Field` state when an Unknown source is moved in. A
/// subsequent deref through dst must NOT record a cast.
#[test]
fn mov_x_unknown_source_overwrites_typed_dst() {
    // Seed r2 with a load chain so it carries LoadedU64Field{T, 8}.
    // Then MOV r2 = r3, where r3 stays Unknown. r2 becomes Unknown.
    // The follow-up deref chain through r2 must produce no record.
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![
        // r2 = *(u64*)(r1+8)  -- r2 = LoadedU64Field{T, 8}
        ldx(BPF_SIZE_DW, 2, 1, 8),
        // r2 = r3            -- r3 Unknown -> r2 becomes Unknown
        mov_x(2, 3),
        // r4 = *(u64*)(r2+0) -- r2 Unknown, no record
        ldx(BPF_SIZE_DW, 4, 2, 0),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: t_id,
        }],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "MOV with Unknown source must overwrite typed dst: {map:?}"
    );
}

/// Self-copy `MOV r2, r2` preserves the register's state because
/// production reads and writes `regs[2]` with no intermediate
/// transformation. A typed register that self-copies continues
/// to carry its typed state into subsequent operations.
#[test]
fn mov_x_self_copy_preserves_state() {
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![
        // r2 = *(u64*)(r1+8)  -- r2 = LoadedU64Field{T, 8}
        ldx(BPF_SIZE_DW, 2, 1, 8),
        // r2 = arena_cast(r2) -- arena_confirmed evidence (arena evidence)
        addr_space_cast(2, 2, 1),
        // r2 = r2             -- self-copy, r2 stays LoadedU64Field
        mov_x(2, 2),
        // r3 = *(u64*)(r2+0)  -- records access, resolves to Q
        ldx(BPF_SIZE_DW, 3, 2, 0),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: t_id,
        }],
        &[],
        &[],
        &[],
    );
    assert_eq!(
        map.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "MOV self-copy must preserve LoadedU64Field state: {map:?}"
    );
}

/// 32-bit MOV (`BPF_CLASS_ALU | BPF_OP_MOV | BPF_SRC_X`) destroys
/// typed-pointer state in the destination register because a
/// 32-bit move truncates the upper 32 bits of any 64-bit pointer.
/// Production routes 32-bit MOV to `set_reg(dst, Unknown)`
/// regardless of source state. A subsequent deref through the
/// destination must record nothing.
#[test]
fn mov32_destroys_typed_state() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // Build a 32-bit MOV: ALU (not ALU64) | MOV | SRC_X.
    let mov32 = mk_insn(BPF_CLASS_ALU | BPF_OP_MOV | BPF_SRC_X, 4, 2, 0, 0);
    let insns = vec![
        // r2 = *(u64*)(r1+8)  -- r2 = LoadedU64Field{T, 8}
        ldx(BPF_SIZE_DW, 2, 1, 8),
        // r4 = (u32) r2       -- 32-bit MOV truncates -> r4 = Unknown
        mov32,
        // r5 = *(u64*)(r4+0)  -- r4 Unknown, no record
        ldx(BPF_SIZE_DW, 5, 4, 0),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: t_id,
        }],
        &[],
        &[],
        &[],
    );
    assert!(map.is_empty(), "32-bit MOV must drop typed state: {map:?}");
}

// ----- ALU op tests -------------------------------------------
//
// All non-MOV ALU ops (ADD, SUB, AND, OR, LSH, etc.) and
// immediate-source MOV (`BPF_OP_MOV | BPF_SRC_K`) destroy the
// typed-pointer property of the destination register. Production
// handles every such case via the catch-all in the ALU dispatch:
// drop dst to Unknown.
//
// Tests verify destruction by setting up a Pointer{T} register
// that would normally produce a kptr finding when stored into a
// P struct's u64 slot, then applying the ALU op, then performing
// the STX. With the destruction working, the STX records nothing.

/// `BPF_ALU64 | BPF_ADD | BPF_SRC_X` with a typed pointer in dst
/// destroys the pointer state. Pointer + integer is no longer a
/// pointer to the same struct (it's a derived address), so the
/// kptr finding must drop.
#[test]
fn alu64_add_x_destroys_typed_pointer() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // ALU64 ADD X: code = BPF_CLASS_ALU64 | BPF_ADD | BPF_SRC_X.
    // BPF_ADD = 0x00, so code = 0x07 | 0x00 | 0x08 = 0x0f.
    let add_x = mk_insn(
        BPF_CLASS_ALU64 | (bs::BPF_ADD as u8) | BPF_SRC_X,
        1,
        3,
        0,
        0,
    );
    // r1 starts Pointer{T}. ADD r1, r3 -> r1 Unknown.
    // STX *(r6+slot_off) = r1 -> no record.
    let insns = vec![add_x, stx(BPF_SIZE_DW, 6, 1, slot_off as i16), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 1,
                struct_type_id: t_id,
            },
            InitialReg {
                reg: 6,
                struct_type_id: p_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "ALU64 ADD X must destroy typed pointer: {map:?}"
    );
}

/// `BPF_ALU64 | BPF_SUB | BPF_SRC_X` destroys the typed pointer
/// state of the destination register. Same shape as ADD: any
/// arithmetic on a pointer produces an integer, not a typed
/// pointer.
#[test]
fn alu64_sub_x_destroys_typed_pointer() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    let sub_x = mk_insn(
        BPF_CLASS_ALU64 | (bs::BPF_SUB as u8) | BPF_SRC_X,
        1,
        3,
        0,
        0,
    );
    let insns = vec![sub_x, stx(BPF_SIZE_DW, 6, 1, slot_off as i16), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 1,
                struct_type_id: t_id,
            },
            InitialReg {
                reg: 6,
                struct_type_id: p_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "ALU64 SUB X must destroy typed pointer: {map:?}"
    );
}

/// `BPF_ALU64 | BPF_AND | BPF_SRC_X` destroys the typed pointer
/// state of the destination register. Bitwise AND on a pointer
/// produces a masked integer, not a typed pointer.
#[test]
fn alu64_and_x_destroys_typed_pointer() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    let and_x = mk_insn(
        BPF_CLASS_ALU64 | (bs::BPF_AND as u8) | BPF_SRC_X,
        1,
        3,
        0,
        0,
    );
    let insns = vec![and_x, stx(BPF_SIZE_DW, 6, 1, slot_off as i16), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 1,
                struct_type_id: t_id,
            },
            InitialReg {
                reg: 6,
                struct_type_id: p_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "ALU64 AND X must destroy typed pointer: {map:?}"
    );
}

/// `BPF_ALU64 | BPF_ADD | BPF_SRC_K` (immediate ADD) destroys
/// typed-pointer state in the destination register. Same code
/// path as ADD with register source -- production drops dst to
/// Unknown for any non-MOV-X-ALU64 op.
#[test]
fn alu64_add_k_destroys_typed_pointer() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // ALU64 ADD K: BPF_SRC_K is 0, so source field is unused.
    // imm carries the constant.
    let add_k = mk_insn(BPF_CLASS_ALU64 | (bs::BPF_ADD as u8), 1, 0, 0, 8);
    let insns = vec![add_k, stx(BPF_SIZE_DW, 6, 1, slot_off as i16), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 1,
                struct_type_id: t_id,
            },
            InitialReg {
                reg: 6,
                struct_type_id: p_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "ALU64 ADD K must destroy typed pointer: {map:?}"
    );
}

/// Immediate MOV (`mov_k`, `BPF_OP_MOV | BPF_SRC_K`) destroys the
/// destination register's typed-pointer state. The catch-all in
/// the ALU dispatch handles this case because the `BPF_OP_MOV +
/// BPF_SRC_X` short-circuit only matches the register source
/// variant; the immediate variant lands in the destruction branch.
#[test]
fn mov_k_destroys_typed_pointer() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // mov_k r1, 42 -> r1 Unknown.
    // STX *(r6+slot_off) = r1 -> no record.
    let insns = vec![
        mov_k(1, 42),
        stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 1,
                struct_type_id: t_id,
            },
            InitialReg {
                reg: 6,
                struct_type_id: p_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert!(map.is_empty(), "mov_k must destroy typed pointer: {map:?}");
}

// ----- Additional BPF_ADDR_SPACE_CAST tests -------------------

/// `BPF_ADDR_SPACE_CAST` with a reserved imm value (`imm == 2`,
/// neither `1` nor `0x10000`) drops the destination register to
/// Unknown. The verifier in `kernel/bpf/verifier.c check_alu_op`
/// rejects programs that use any other imm for
/// `BPF_ADDR_SPACE_CAST`, so seeing it in pre-verification
/// bytecode is malformed; treating dst as Unknown is the
/// conservative direction.
#[test]
fn addr_space_cast_unknown_imm_drops_dst() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // r3 = *(u64*)(r1+8)             -- r3 = LoadedU64Field{T, 8}
    // r4 = (cast imm=2) r3            -- imm=2 is reserved, r4 Unknown
    // r5 = *(u64*)(r4+0)              -- no record
    let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 1, 2);
    let insns = vec![
        ldx(BPF_SIZE_DW, 3, 1, 8),
        cast,
        ldx(BPF_SIZE_DW, 5, 4, 0),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: t_id,
        }],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "BPF_ADDR_SPACE_CAST with reserved imm must drop dst: {map:?}"
    );
}

/// `BPF_ADDR_SPACE_CAST` arena -> kernel (`imm == 1`) on a
/// `Pointer{T}` source (rather than a `LoadedU64Field`) propagates
/// the typed pointer state into the destination register.
/// Production unconditionally copies `regs[src]` into `regs[dst]`;
/// the LoadedU64Field-only branch merely populates
/// `arena_confirmed` as a side effect when the source matches
/// that variant. Since Pointer{T} sources skip the
/// `arena_confirmed` insertion, no false-positive arena evidence
/// is recorded, and the typed pointer survives the cast for use
/// as a kptr value in a subsequent STX.
#[test]
fn addr_space_cast_arena_imm1_on_pointer_propagates() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // r4 = (cast imm=1) r3   -- r3 = Pointer{T}, r4 = Pointer{T}
    // STX *(r6+slot_off) = r4 -- records (P, slot_off) -> T kptr finding
    let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 1, 1);
    let insns = vec![cast, stx(BPF_SIZE_DW, 6, 4, slot_off as i16), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 3,
                struct_type_id: t_id,
            },
            InitialReg {
                reg: 6,
                struct_type_id: p_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert_eq!(
        map.get(&(p_id, slot_off)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
        "ADDR_SPACE_CAST imm=1 on Pointer{{T}} must propagate state: {map:?}"
    );
}

/// `BPF_ADDR_SPACE_CAST` kernel -> arena (`imm == 0x10000`) on a
/// `Pointer{T}` source PROPAGATES the typed kernel pointer into the
/// destination register. Production (cast_analysis/mod.rs, the
/// `imm == 1 << 16` arm) does `self.regs[dst] = self.regs[src]` when
/// the source is a `Pointer{..}` (or `ArenaU64FromAlloc`) — it does
/// NOT clear dst. A subsequent kptr STX through the cast result
/// therefore records the finding.
///
/// Sibling test `addr_space_cast_kernel_to_arena_propagates_loaded_field`
/// covers the `LoadedU64Field` source case; this test verifies that
/// `Pointer{T}` survives `addr_space_cast(imm=0x10000)` so subsequent
/// field loads through the cast destination produce `LoadedU64Field`
/// entries (needed for cross-subprog arena pointer detection where a
/// callee casts a forwarded Pointer parameter).
#[test]
fn addr_space_cast_kernel_arena_preserves_pointer_source() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // r4 = (cast imm=0x10000) r3   -- r3 = Pointer{T}, r4 = Pointer{T}
    // STX *(r6+slot_off) = r4      -- r4 Pointer{T}, kptr record
    let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 1, 0x10000);
    let insns = vec![cast, stx(BPF_SIZE_DW, 6, 4, slot_off as i16), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 3,
                struct_type_id: t_id,
            },
            InitialReg {
                reg: 6,
                struct_type_id: p_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert_eq!(
        map.len(),
        1,
        "Pointer through addr_space_cast should produce a kptr CastHit: {map:?}"
    );
}

// ----- Misc tests ---------------------------------------------

/// `BPF_LD | BPF_W | BPF_ABS` (`code == 0x20`) is the legacy
/// packet-data load mode kept for socket filters; it loads from
/// packet data into r0. Production sets r0 to Unknown for any LD
/// mode that is not the LD_IMM64 two-slot form. A previously-
/// typed r0 must lose its typed state, so a follow-up kptr STX
/// through r0 produces no record.
#[test]
fn bpf_ld_abs_clears_r0() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // BPF_LD | BPF_W | BPF_ABS = 0x00 | 0x00 | 0x20 = 0x20.
    let ld_abs = mk_insn(BPF_CLASS_LD | BPF_SIZE_W | (bs::BPF_ABS as u8), 0, 0, 0, 0);
    // r0 starts Pointer{T}. After BPF_LD_ABS, r0 is Unknown.
    // STX *(r6+slot_off) = r0 -> no record.
    let insns = vec![ld_abs, stx(BPF_SIZE_DW, 6, 0, slot_off as i16), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 0,
                struct_type_id: t_id,
            },
            InitialReg {
                reg: 6,
                struct_type_id: p_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "BPF_LD_ABS must clear r0 typed state: {map:?}"
    );
}

/// `BPF_LD | BPF_W | BPF_IND` (`code == 0x40`) is the indirect
/// packet-data load mode for socket filters; it loads from
/// `packet[src + imm]` into r0. Production treats it the same way
/// as `BPF_LD_ABS`: r0 becomes Unknown. A previously-typed r0
/// must lose its state.
#[test]
fn bpf_ld_ind_clears_r0() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // BPF_LD | BPF_W | BPF_IND = 0x00 | 0x00 | 0x40 = 0x40.
    let ld_ind = mk_insn(BPF_CLASS_LD | BPF_SIZE_W | (bs::BPF_IND as u8), 0, 0, 0, 0);
    let insns = vec![ld_ind, stx(BPF_SIZE_DW, 6, 0, slot_off as i16), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 0,
                struct_type_id: t_id,
            },
            InitialReg {
                reg: 6,
                struct_type_id: p_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "BPF_LD_IND must clear r0 typed state: {map:?}"
    );
}

/// A program consisting of a single `EXIT` instruction must not
/// panic and must produce an empty `CastMap`. EXIT is in
/// `BPF_CLASS_JMP` with `op == BPF_OP_EXIT`, which production
/// explicitly leaves unmodified. The empty instruction-stream
/// behavior is the baseline; this test guards against regressions
/// where a "no recognizable ops" program produces phantom output.
#[test]
fn single_exit_does_not_panic() {
    let (blob, _t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![exit()];
    let map = analyze_casts(&insns, &btf, &[], &[], &[], &[]);
    assert!(map.is_empty(), "single EXIT must yield empty map: {map:?}");
}

/// A program of only jump / branch instructions (no LDX, STX, MOV,
/// or call) carries no data flow that the analyzer could track.
/// Production processes each insn through `step()` but the JMP
/// arm only mutates state on CALL -- branches and EXIT are no-ops
/// for state. The forward walk completes without panicking; the
/// output map is empty even though every PC is processed.
#[test]
fn jumps_only_program_does_not_panic() {
    let (blob, _t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // BPF_JEQ_K = 0x10. Construct a sequence of conditional and
    // unconditional jumps that branch within bounds.
    // pc 0: if r1 == 0 goto +1 (target = pc 2)
    // pc 1: ja +1               (target = pc 3)
    // pc 2: ja -2               (target = pc 1)
    // pc 3: exit
    let jeq = mk_insn(BPF_CLASS_JMP | 0x10, 1, 0, 1, 0);
    let ja_plus = mk_insn(BPF_CLASS_JMP, 0, 0, 1, 0);
    let ja_minus = mk_insn(BPF_CLASS_JMP, 0, 0, -2, 0);
    let insns = vec![jeq, ja_plus, ja_minus, exit()];
    let map = analyze_casts(&insns, &btf, &[], &[], &[], &[]);
    assert!(
        map.is_empty(),
        "all-jumps program must yield empty map: {map:?}"
    );
}
