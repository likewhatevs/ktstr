use super::*;

// ----- BPF_ATOMIC tests ---------------------------------------

/// Helper: build a `BPF_STX | BPF_DW | BPF_ATOMIC` instruction
/// with the given atomic-op `imm`. Encoding per kernel uapi
/// `bpf.h`: `code = STX | DW | ATOMIC = 0x03 | 0x18 | 0xc0 = 0xdb`.
fn atomic_stx(dst: u8, src: u8, off: i16, imm: i32) -> BpfInsn {
    mk_insn(
        BPF_CLASS_STX | BPF_SIZE_DW | BPF_MODE_ATOMIC,
        dst,
        src,
        off,
        imm,
    )
}

/// `BPF_XCHG` (`imm == 0xe0 | BPF_FETCH = 0xe1`) overwrites the
/// source register with the prior memory value per kernel uapi
/// `bpf.h`. The analyzer cannot type the prior memory contents,
/// so the source register's typed state is clobbered. A
/// subsequent plain STX of that register into a `u64` slot must
/// NOT produce a kptr finding.
#[test]
fn atomic_xchg_clobbers_src() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // R1 = Pointer{T}; XCHG src=R1 to [R2+0] -> R1 = Unknown.
    // Then STX R1 into P.slot must NOT record because R1 is
    // Unknown post-xchg.
    let insns = vec![
        atomic_stx(2, 1, 0, 0xe0 | BPF_FETCH),
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
    assert!(
        map.is_empty(),
        "XCHG must clobber src R1 typed state: {map:?}"
    );
}

/// `BPF_CMPXCHG` (`imm == 0xf0 | BPF_FETCH = 0xf1`) overwrites R0
/// with the prior memory value per kernel uapi `bpf.h`,
/// regardless of whether the compare-and-write succeeded. R0's
/// typed state must be clobbered. A subsequent STX of R0 into a
/// `u64` slot must NOT produce a kptr finding.
#[test]
fn atomic_cmpxchg_clobbers_r0() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // Seed R0 = Pointer{T} (analyzer accepts InitialReg.reg in
    // 0..=9). CMPXCHG dst=2, src=1 with imm=0xf1 clobbers R0.
    // Subsequent STX of R0 into P.slot must NOT record.
    let insns = vec![
        atomic_stx(2, 1, 0, 0xf0 | BPF_FETCH),
        stx(BPF_SIZE_DW, 6, 0, slot_off as i16),
        exit(),
    ];
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
        "CMPXCHG must clobber R0 typed state: {map:?}"
    );
}

/// Non-fetch atomic ops (plain `BPF_ADD`/`AND`/`OR`/`XOR` without
/// the `BPF_FETCH` bit) read-modify-write memory but do not
/// overwrite any register. Source register typed state must
/// survive intact, so a subsequent STX of the source into a
/// `u64` slot still records the kptr finding. Per linux uapi
/// `bpf_common.h`: BPF_ADD=0x00, BPF_OR=0x40, BPF_AND=0x50,
/// BPF_XOR=0xa0. All four flavours must round-trip the source
/// register's `Pointer{T}` state.
#[test]
fn atomic_non_fetch_preserves_regs() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // Iterate every non-fetch atomic op to defend against the
    // failure mode where the analyzer accidentally clobbers
    // `src` for some imm encodings but not others (e.g. a future
    // refactor that special-cases BPF_ADD only). All four ops
    // share the kernel verifier's RMW semantics: mutate memory,
    // no register output. Encoding is the bare top nibble — adding
    // BPF_FETCH (0x01) shifts to the clobbering branch tested by
    // `atomic_xchg_clobbers_src`.
    const BPF_ATOMIC_ADD: i32 = 0x00;
    const BPF_ATOMIC_OR: i32 = 0x40;
    const BPF_ATOMIC_AND: i32 = 0x50;
    const BPF_ATOMIC_XOR: i32 = 0xa0;
    for imm in [
        BPF_ATOMIC_ADD,
        BPF_ATOMIC_OR,
        BPF_ATOMIC_AND,
        BPF_ATOMIC_XOR,
    ] {
        let insns = vec![
            atomic_stx(2, 1, 0, imm),
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
        assert_eq!(
            map.len(),
            1,
            "imm=0x{imm:02x}: exactly one kptr finding expected, got: {map:?}"
        );
        assert_eq!(
            map.get(&(p_id, slot_off)),
            Some(&CastHit {
                alloc_size: None,
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel,
            }),
            "imm=0x{imm:02x}: non-fetch ATOMIC must preserve src register: {map:?}"
        );
    }
}

/// An `ATOMIC` op targeting `[r10 + neg_off]` mutates a stack
/// slot the analyzer was tracking from a prior spill. The slot's
/// saved value is overwritten by the atomic operation, so a
/// subsequent reload through `r10` must NOT resurrect the
/// pre-atomic typed-pointer state.
#[test]
fn atomic_on_stack_invalidates_slot() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // *(u64 *)(r10 - 8) = R1   ; spill Pointer{T} -> stack[-8]
    // ATOMIC XCHG [r10 - 8] = R2   ; mutates the stack slot
    // R3 = *(u64 *)(r10 - 8)   ; reload must yield Unknown
    // *(u64 *)(R6 + slot_off) = R3 ; R3 Unknown -> no record
    let insns = vec![
        stx(BPF_SIZE_DW, 10, 1, -8),
        atomic_stx(10, 2, -8, 0xe0 | BPF_FETCH),
        ldx(BPF_SIZE_DW, 3, 10, -8),
        stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
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
    assert!(
        map.is_empty(),
        "ATOMIC on stack slot must invalidate, reload yields Unknown: {map:?}"
    );
}

/// `BPF_LOAD_ACQ` (`imm == 0x100`) loads a memory value with
/// acquire-ordered semantics into `dst` per kernel
/// `include/linux/filter.h`. The analyzer cannot type a memory
/// value pulled out via this path, so `dst` is clobbered to
/// `Unknown`. A subsequent STX of `dst` into a `u64` slot must
/// NOT record a kptr finding even though `dst` held a typed
/// pointer prior to the load-acquire.
#[test]
fn atomic_load_acq_clobbers_dst() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // R1 = Pointer{T}; LOAD_ACQ targets dst=1, src=2 (address
    // base) -> R1 = Unknown. Then STX R1 into P.slot must NOT
    // record because R1 is Unknown post-load-acquire.
    let insns = vec![
        atomic_stx(1, 2, 0, BPF_LOAD_ACQ_IMM),
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
    assert!(
        map.is_empty(),
        "LOAD_ACQ must clobber dst R1 typed state: {map:?}"
    );
}

/// `BPF_STORE_REL` (`imm == 0x110`) stores `src` to memory with
/// release-ordered semantics per kernel
/// `include/linux/filter.h`. `dst` is the address-base register
/// and `src` is the value being stored — neither is overwritten.
/// Both registers' typed-pointer state must survive intact: a
/// subsequent plain STX of either typed pointer into another
/// `u64` slot still records the kptr finding.
#[test]
fn atomic_store_rel_preserves_src_and_dst() {
    // BTF: u64(1), T(2, u64@0), T*(3), P(4, u64@slot_off1, u64@slot_off2).
    // Two distinct slots so we can verify BOTH registers'
    // typed states by storing each into a separate slot.
    let slot_off1: u32 = 16;
    let slot_off2: u32 = 24;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_p = push_name(&mut strings, "P");
    let n_x = push_name(&mut strings, "x");
    let n_slot1 = push_name(&mut strings, "slot1");
    let n_slot2 = push_name(&mut strings, "slot2");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynType::Ptr { type_id: 2 },
        SynType::Struct {
            name_off: n_p,
            size: slot_off2 + 8,
            members: vec![
                SynMember {
                    name_off: n_slot1,
                    type_id: 1,
                    byte_offset: slot_off1,
                },
                SynMember {
                    name_off: n_slot2,
                    type_id: 1,
                    byte_offset: slot_off2,
                },
            ],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let p_id = 4;
    // R1 = Pointer{T}, R2 = Pointer{T}, R6 = Pointer{P}.
    // STORE_REL dst=R7 src=R1: address-base R7, value R1.
    //   R1 must remain Pointer{T}; R7 is uninvolved here.
    // STX *(R6 + slot1) = R1: records (P, slot1) -> T.
    // STX *(R6 + slot2) = R2: records (P, slot2) -> T.
    // If STORE_REL had clobbered R1, the first kptr write
    // would have dropped — but R2 (the unused-by-STORE_REL
    // typed pointer) would still record, giving a partial map.
    // The two-slot assertion discriminates: both slots present
    // proves STORE_REL left R1 alone; only-slot2 present would
    // catch a regression that clobbers R1 specifically.
    let insns = vec![
        atomic_stx(7, 1, 0, BPF_STORE_REL_IMM),
        stx(BPF_SIZE_DW, 6, 1, slot_off1 as i16),
        stx(BPF_SIZE_DW, 6, 2, slot_off2 as i16),
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
                reg: 2,
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
        map.get(&(p_id, slot_off1)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
        "STORE_REL must preserve src R1 typed state (slot1 missing): {map:?}"
    );
    assert_eq!(
        map.get(&(p_id, slot_off2)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
        "STORE_REL must not affect uninvolved R2 (slot2 missing): {map:?}"
    );
}

/// `BPF_STORE_REL` through `r10` writes the stack slot at
/// `[r10 + off]`. Even though STORE_REL has no per-register
/// clobber effect, the stack-slot invalidation arm at the head
/// of `handle_atomic` runs for every atomic flavor except
/// LOAD_ACQ when `dst == r10` (STORE_REL is not that exception).
/// A prior spill of a typed pointer
/// into the slot is overwritten by the release store, so a
/// subsequent reload through `r10` must NOT resurrect the
/// pre-store-release typed-pointer state.
#[test]
fn atomic_store_rel_invalidates_stack_slot() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // *(u64 *)(r10 - 8) = R1     ; spill Pointer{T} -> stack[-8]
    // STORE_REL [r10 - 8] = R2   ; release-store overwrites slot
    // R3 = *(u64 *)(r10 - 8)     ; reload must yield Unknown
    // *(u64 *)(R6 + slot_off) = R3 ; R3 Unknown -> no record
    let insns = vec![
        stx(BPF_SIZE_DW, 10, 1, -8),
        atomic_stx(10, 2, -8, BPF_STORE_REL_IMM),
        ldx(BPF_SIZE_DW, 3, 10, -8),
        stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
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
    assert!(
        map.is_empty(),
        "STORE_REL through r10 must invalidate slot, reload Unknown: {map:?}"
    );
}

/// `BPF_ADD | BPF_FETCH` (`imm == 0x01`) is an atomic
/// fetch-and-add: src receives the prior memory value, memory
/// receives `memory + src`. Per kernel uapi `bpf.h` and the
/// `has_fetch` arm in `handle_atomic`, src's typed-pointer state
/// is dropped to `Unknown`. A subsequent STX of src into a `u64`
/// slot must NOT record a kptr finding.
#[test]
fn atomic_add_fetch_clobbers_src() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // R1 = Pointer{T}; ADD|FETCH src=R1 to [R2+0] -> R1 = Unknown.
    // Then STX R1 into P.slot must NOT record because R1 is
    // Unknown post-fetch-add. BPF_ADD = 0x00 (linux uapi
    // bpf_common.h) | BPF_FETCH = 0x01.
    let insns = vec![
        atomic_stx(2, 1, 0, BPF_FETCH),
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
    assert!(
        map.is_empty(),
        "ADD|FETCH must clobber src R1 typed state: {map:?}"
    );
}

/// `BPF_AND | BPF_FETCH` (`imm == 0x51`) is an atomic
/// fetch-and-and: src receives the prior memory value. The
/// `has_fetch` arm in `handle_atomic` drops src to `Unknown`.
/// A subsequent STX of src into a `u64` slot must NOT record.
#[test]
fn atomic_and_fetch_clobbers_src() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // BPF_AND = 0x50 (linux uapi bpf_common.h) | BPF_FETCH = 0x51.
    let insns = vec![
        atomic_stx(2, 1, 0, 0x50 | BPF_FETCH),
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
    assert!(
        map.is_empty(),
        "AND|FETCH must clobber src R1 typed state: {map:?}"
    );
}

/// `BPF_OR | BPF_FETCH` (`imm == 0x41`) is an atomic
/// fetch-and-or: src receives the prior memory value. The
/// `has_fetch` arm in `handle_atomic` drops src to `Unknown`.
/// A subsequent STX of src into a `u64` slot must NOT record.
#[test]
fn atomic_or_fetch_clobbers_src() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // BPF_OR = 0x40 (linux uapi bpf_common.h) | BPF_FETCH = 0x41.
    let insns = vec![
        atomic_stx(2, 1, 0, 0x40 | BPF_FETCH),
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
    assert!(
        map.is_empty(),
        "OR|FETCH must clobber src R1 typed state: {map:?}"
    );
}

/// `BPF_XOR | BPF_FETCH` (`imm == 0xa1`) is an atomic
/// fetch-and-xor: src receives the prior memory value. The
/// `has_fetch` arm in `handle_atomic` drops src to `Unknown`.
/// A subsequent STX of src into a `u64` slot must NOT record.
#[test]
fn atomic_xor_fetch_clobbers_src() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // BPF_XOR = 0xa0 (linux uapi bpf_common.h) | BPF_FETCH = 0xa1.
    let insns = vec![
        atomic_stx(2, 1, 0, 0xa0 | BPF_FETCH),
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
    assert!(
        map.is_empty(),
        "XOR|FETCH must clobber src R1 typed state: {map:?}"
    );
}

/// `BPF_ATOMIC` with `BPF_W` (4-byte) size targeting `[r10+off]`
/// must invalidate the stack slot the same way a DW atomic does.
/// Per `handle_atomic`, the stack-invalidation arm runs on
/// `dst == r10` for non-LOAD_ACQ atomics regardless of the size
/// bits in the opcode — a 4-byte atomic write into a slot that
/// formerly held a 64-bit typed pointer truncates the slot's
/// content. A subsequent DW reload must NOT resurrect the
/// pre-atomic typed-pointer state.
#[test]
fn atomic_w_size_invalidates_stack_slot() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // *(u64 *)(r10 - 8) = R1                 ; spill Pointer{T} -> stack[-8]
    // ATOMIC<W> XCHG [r10 - 8] = R2          ; W-size atomic on slot
    // R3 = *(u64 *)(r10 - 8)                 ; reload must yield Unknown
    // *(u64 *)(R6 + slot_off) = R3           ; R3 Unknown -> no record
    //
    // Constructed with `mk_insn` directly because the `atomic_stx`
    // helper hard-codes `BPF_SIZE_DW`. Code = STX | W | ATOMIC.
    let atomic_w = mk_insn(
        BPF_CLASS_STX | BPF_SIZE_W | BPF_MODE_ATOMIC,
        10,
        2,
        -8,
        0xe0 | BPF_FETCH,
    );
    let insns = vec![
        stx(BPF_SIZE_DW, 10, 1, -8),
        atomic_w,
        ldx(BPF_SIZE_DW, 3, 10, -8),
        stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
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
    assert!(
        map.is_empty(),
        "W-size ATOMIC on stack slot must invalidate, reload Unknown: {map:?}"
    );
}

/// `BPF_CMPXCHG` (`imm == 0xf1`) overwrites BOTH `R0` (with the
/// prior memory value) AND `src` (because `BPF_FETCH` is set,
/// the second-stage `has_fetch` arm in `handle_atomic` runs in
/// addition to the CMPXCHG-specific R0 clobber). The existing
/// `atomic_cmpxchg_clobbers_r0` test guards the R0 path; this
/// test guards the src path. Per kernel uapi `bpf.h` and the
/// final fall-through `if has_fetch` arm in `handle_atomic`,
/// src's typed-pointer state is dropped to `Unknown` regardless
/// of which atomic-op top nibble was used. A subsequent STX of
/// src into a `u64` slot must NOT record a kptr finding.
#[test]
fn atomic_cmpxchg_clobbers_src() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // R1 = Pointer{T}; CMPXCHG dst=R2, src=R1 with imm=0xf1.
    // R0 (the cmpxchg "expected" register, not seeded here) is
    // clobbered to Unknown by the CMPXCHG-specific arm; R1 is
    // clobbered to Unknown by the final has_fetch arm. STX R1
    // into P.slot must NOT record.
    let insns = vec![
        atomic_stx(2, 1, 0, 0xf0 | BPF_FETCH),
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
    assert!(
        map.is_empty(),
        "CMPXCHG must clobber src R1 typed state via has_fetch arm: {map:?}"
    );
}

// ----- Stack tests --------------------------------------------

/// A second spill to the same stack slot overwrites the prior
/// saved state. Reload restores the latest typed pointer, not
/// the original one. Production line `self.stack_slots.insert`
/// replaces by key.
#[test]
fn stack_spill_overwrite_uses_latest() {
    // BTF: u64(1), T1(2, u64@0), T2(3, u64@0), P(4, u64@slot_off).
    // T1 and T2 are distinguishable by id; the test seeds R1=T1,
    // R2=T2, then spills T2 last so reload should yield T2.
    let slot_off: u32 = 16;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t1 = push_name(&mut strings, "T1");
    let n_t2 = push_name(&mut strings, "T2");
    let n_p = push_name(&mut strings, "P");
    let n_x = push_name(&mut strings, "x");
    let n_slot = push_name(&mut strings, "slot");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Struct {
            name_off: n_t1,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynType::Struct {
            name_off: n_t2,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynType::Struct {
            name_off: n_p,
            size: slot_off + 8,
            members: vec![SynMember {
                name_off: n_slot,
                type_id: 1,
                byte_offset: slot_off,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t1_id = 2;
    let t2_id = 3;
    let p_id = 4;
    // Spill R1 (T1*) to [r10-8]
    // Spill R2 (T2*) to [r10-8] -- overwrite
    // Reload to R3 (must be T2*)
    // Store R3 into P.slot
    let insns = vec![
        stx(BPF_SIZE_DW, 10, 1, -8),
        stx(BPF_SIZE_DW, 10, 2, -8),
        ldx(BPF_SIZE_DW, 3, 10, -8),
        stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 1,
                struct_type_id: t1_id,
            },
            InitialReg {
                reg: 2,
                struct_type_id: t2_id,
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
            target_type_id: t2_id,
            addr_space: AddrSpace::Kernel,
        }),
        "second spill to same slot must win: {map:?}"
    );
}

/// `BPF_CALL` clobbers `r0..r5` per the BPF ABI but does NOT
/// invalidate stack slots. A typed pointer parked in `[r10-N]`
/// before a helper call must reload as the same typed pointer
/// after the call returns.
#[test]
fn stack_spill_survives_helper_call() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // Spill R1 (T*) to [r10-8]
    // CALL helper (clobbers R0..R5, R6 untouched)
    // Reload from [r10-8] to R3 (must restore T*)
    // Store R3 into P.slot
    let insns = vec![
        stx(BPF_SIZE_DW, 10, 1, -8),
        call(),
        ldx(BPF_SIZE_DW, 3, 10, -8),
        stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
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
    assert_eq!(
        map.get(&(p_id, slot_off)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
        "stack-spilled pointer must survive helper call: {map:?}"
    );
}

/// A sub-DW (4-byte W) store through `r10` to a slot previously
/// holding a typed pointer truncates the stored value. Per
/// `handle_stx`, the slot is removed so a later DW reload
/// returns Unknown rather than resurrecting the stale typed
/// state.
#[test]
fn sub_dw_spill_invalidates() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // *(u64 *)(r10 - 8) = R1   ; spill T* to slot
    // *(u32 *)(r10 - 8) = R1   ; sub-DW store, slot removed
    // R3 = *(u64 *)(r10 - 8)   ; reload must yield Unknown
    // *(u64 *)(R6 + slot_off) = R3 ; no record
    let insns = vec![
        stx(BPF_SIZE_DW, 10, 1, -8),
        stx(BPF_SIZE_W, 10, 1, -8),
        ldx(BPF_SIZE_DW, 3, 10, -8),
        stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
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
    assert!(
        map.is_empty(),
        "sub-DW store must invalidate slot, reload Unknown: {map:?}"
    );
}

/// `BPF_ST` (immediate store, `code = BPF_ST | BPF_MEM | BPF_DW
/// = 0x7A`) writes a constant immediate to memory through `r10`.
/// The constant is never a typed pointer, but the store overlays
/// any prior typed value the analyzer was tracking in the
/// stack slot. Per the `BPF_CLASS_ST` arm in `step()`, the
/// stack slot is removed when `dst == r10 && mode == BPF_MEM`
/// so a subsequent reload through `r10` must NOT resurrect the
/// pre-immediate-store typed-pointer state.
#[test]
fn st_imm_invalidates_stack_slot() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // *(u64 *)(r10 - 8) = R1     ; spill Pointer{T} -> stack[-8]
    // *(u64 *)(r10 - 8) = imm 0  ; ST overwrites slot with constant
    // R3 = *(u64 *)(r10 - 8)     ; reload must yield Unknown
    // *(u64 *)(R6 + slot_off) = R3 ; R3 Unknown -> no record
    //
    // Constructed with `mk_insn` directly — there is no
    // helper for BPF_ST class instructions. Code per linux
    // uapi `bpf_common.h` and `bpf.h`: BPF_ST | BPF_MEM | BPF_DW
    // = 0x02 | 0x60 | 0x18 = 0x7A. dst=r10 (frame pointer),
    // src is unused (encoded as 0), off=-8 (slot key matching
    // the prior spill), imm=0 (constant value, irrelevant —
    // any constant overwrites the typed slot the same way).
    let st_imm_dw = mk_insn(BPF_CLASS_ST | BPF_MODE_MEM | BPF_SIZE_DW, 10, 0, -8, 0);
    let insns = vec![
        stx(BPF_SIZE_DW, 10, 1, -8),
        st_imm_dw,
        ldx(BPF_SIZE_DW, 3, 10, -8),
        stx(BPF_SIZE_DW, 6, 3, slot_off as i16),
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
    assert!(
        map.is_empty(),
        "BPF_ST imm to stack slot must invalidate, reload Unknown: {map:?}"
    );
}
