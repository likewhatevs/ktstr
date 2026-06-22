use super::*;

// ----- BTF type edge cases ------------------------------------

/// `struct_member_at` skips bitfield members (members with
/// `bitfield_size > 0`) even when their byte offset matches the
/// query. A LDX through a `Pointer{T}` register at the bitfield's
/// byte offset must NOT seed a `LoadedU64Field` state, so no
/// pattern accumulates and no cast emits.
#[test]
fn struct_member_at_skips_bitfield_at_target_offset() {
    // BTF: u64(1), T(kind_flag=1) with a u64 bitfield at byte
    // offset 8 (bitfield_size = 32 bits). The byte offset matches
    // the LDX target but the bitfield_size > 0 makes
    // struct_member_at return None.
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_f = push_name(&mut strings, "f");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::StructBitfields {
            name_off: n_t,
            size: 16,
            members: vec![SynMemberBits {
                name_off: n_f,
                type_id: 1,
                bit_offset: 8 * 8,      // byte offset 8
                bitfield_size_bits: 32, // bitfield -> skip
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    // r2 = *(u64*)(r1 + 8): handle_ldx queries struct_member_at(T, 8).
    // The bitfield at 8 is skipped; struct_member_at returns None;
    // r2 becomes Unknown; no LoadedU64Field; no pattern recorded.
    // r3 = *(u64*)(r2 + 0) on Unknown source records nothing.
    let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
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
        "bitfield at target offset must not seed cast: {map:?}"
    );
}

/// `struct_member_at` skips members whose bit offset is not a
/// multiple of 8 (`bit_off % 8 != 0`) -- bit-position members
/// inside a kind_flag=1 struct that happen to lie between byte
/// boundaries cannot serve as a u64 LDX source. Even though the
/// byte derived from the bit position would round to the LDX
/// target offset, the alignment guard rejects.
#[test]
fn struct_member_at_skips_non_byte_aligned_member() {
    // T (kind_flag=1) with a u64 member at bit_offset = 65 (NOT
    // a multiple of 8; integer-divided by 8 gives byte 8). The
    // alignment guard rejects regardless of bitfield_size_bits.
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_f = push_name(&mut strings, "f");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::StructBitfields {
            name_off: n_t,
            size: 24,
            members: vec![SynMemberBits {
                name_off: n_f,
                type_id: 1,
                bit_offset: 65,        // NOT a multiple of 8
                bitfield_size_bits: 0, // not a bitfield
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    // r2 = *(u64*)(r1 + 8): struct_member_at scans T's members,
    // sees bit_offset 65 % 8 = 1, skips. r2 stays Unknown.
    let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
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
        "non-byte-aligned member must not seed cast: {map:?}"
    );
}

/// `member_size_bytes` returns `None` for terminal types whose
/// size is not resolvable from BTF alone (Func, FuncProto, Void,
/// Fwd, Var, Datasec). The `build_layout_index` loop must handle
/// `None` by skipping the member rather than panicking. Stress
/// the path by constructing a candidate struct whose members
/// include one each of Fwd, Func, and Void; the matcher must not
/// include those member positions in the (offset, size) layout
/// map. Without this guard the matcher would either crash on the
/// `expect`-style unwrap or emit a candidate the renderer cannot
/// chase.
#[test]
fn member_size_bytes_unsupported_terminals_skipped() {
    // BTF:
    //   id 1: u64
    //   id 2: T   { u64 f @ 8 }       -- source struct
    //   id 3: Fwd struct (kind_flag=0)
    //   id 4: FuncProto returning void, no params
    //   id 5: Func -> id 4
    //   id 6: U  { fwd_ref @ 0; func_ref @ 8; void_ref @ 16; u64 v @ 24 }
    //
    //   The members typed as Fwd / Func / Void all return None
    //   from member_size_bytes, so layout_index for U skips them.
    //   The single u64 at offset 24 makes U a candidate at
    //   (offset=24, size=8) ONLY. The LDX pattern accesses
    //   offset=24 (size 8); intersection -> {U}.
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_u = push_name(&mut strings, "U");
    let n_fwd_target = push_name(&mut strings, "fwd_struct");
    let n_func = push_name(&mut strings, "fn");
    let n_fwd_ref = push_name(&mut strings, "fwd_ref");
    let n_func_ref = push_name(&mut strings, "func_ref");
    let n_void_ref = push_name(&mut strings, "void_ref");
    let n_v = push_name(&mut strings, "v");
    let n_f = push_name(&mut strings, "f");
    let types = vec![
        // id 1: u64
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: source T { u64 f @ 8 }
        SynType::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // id 3: Fwd (struct).
        SynType::Fwd {
            name_off: n_fwd_target,
            kind_flag: 0,
        },
        // id 4: FuncProto -> void, no params.
        SynType::FuncProto {
            return_type_id: 0,
            params: vec![],
        },
        // id 5: Func -> id 4.
        SynType::Func {
            name_off: n_func,
            type_id: 4,
            linkage: 1,
        },
        // id 6: U with members of unsupported sizes plus one
        // sized u64 member. Production must skip the unsupported
        // ones and include only (offset=24, size=8).
        SynType::Struct {
            name_off: n_u,
            size: 32,
            members: vec![
                SynMember {
                    name_off: n_fwd_ref,
                    type_id: 3, // Fwd -> None size
                    byte_offset: 0,
                },
                SynMember {
                    name_off: n_func_ref,
                    type_id: 5, // Func -> None size
                    byte_offset: 8,
                },
                SynMember {
                    name_off: n_void_ref,
                    type_id: 0, // Void -> None size
                    byte_offset: 16,
                },
                SynMember {
                    name_off: n_v,
                    type_id: 1, // u64 -> Some(8)
                    byte_offset: 24,
                },
            ],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let u_id = 6;
    // r2 = *(u64*)(r1 + 8); cast r2 (arena evidence); r3 =
    // *(u64*)(r2 + 24). The pattern (offset=24, size=8) must
    // intersect to {U} only -- Fwd / Func / Void members at
    // offsets 0/8/16 are skipped during layout indexing.
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(2, 2, 1),
        ldx(BPF_SIZE_DW, 3, 2, 24),
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
            target_type_id: u_id,
            addr_space: AddrSpace::Arena,
        }),
        "unsupported terminals must be skipped without crashing: {map:?}"
    );
}

/// `build_layout_index` skips bitfield members (those with
/// `bitfield_size > 0`). A struct whose only u64 sits as a
/// bitfield does NOT register in the (offset, size) layout map,
/// so it cannot be a candidate even when the access pattern
/// "would" match its byte position. The matcher converges on the
/// struct that has a NON-bitfield member at the queried position.
#[test]
fn build_layout_index_skips_bitfields_in_candidates() {
    // BTF: u64(1), T(2, u64@8) source. Q1(3, kind_flag=1) with a
    // u64 BITFIELD at byte 0 (size 32 bits) -- must NOT register
    // as a candidate. Q2(4) with a u64 NON-bitfield at byte 0 --
    // sole candidate. Pattern (offset=0, size=8) -> {Q2} only.
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q1 = push_name(&mut strings, "Q1");
    let n_q2 = push_name(&mut strings, "Q2");
    let n_f = push_name(&mut strings, "f");
    let n_a = push_name(&mut strings, "a");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // T has u64@8 source field.
        SynType::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // Q1 (kind_flag=1) with bitfield u64 at byte 0
        // (bitfield_size = 32 bits). Production must skip during
        // layout indexing because bitfield_size > 0.
        SynType::StructBitfields {
            name_off: n_q1,
            size: 8,
            members: vec![SynMemberBits {
                name_off: n_a,
                type_id: 1,
                bit_offset: 0,
                bitfield_size_bits: 32,
            }],
        },
        // Q2 with a normal u64 at byte 0 -- included in layout.
        SynType::Struct {
            name_off: n_q2,
            size: 8,
            members: vec![SynMember {
                name_off: n_a,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let q2_id = 4;
    // Sequence: r2 = *(u64*)(r1 + 8); cast r2 (arena evidence);
    // r3 = *(u64*)(r2 + 0).
    // Pattern (0, 8): layout includes Q2 only (Q1's bitfield
    // skipped). Map records (T, 8) -> (Q2, Arena).
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(2, 2, 1),
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
            target_type_id: q2_id,
            addr_space: AddrSpace::Arena,
        }),
        "bitfield candidate must be skipped: {map:?}"
    );
}

/// `BTF_KIND_UNION` participates in `build_layout_index` and
/// `struct_member_at` identically to `BTF_KIND_STRUCT` -- `btf-rs`
/// aliases `Union = Struct`, and production matches both via
/// `Type::Struct(s) | Type::Union(s)`. Verify a kptr STX through
/// a parent typed as a union records correctly, AND a candidate
/// search resolves to a union when its layout uniquely matches.
#[test]
fn union_works_like_struct_for_layout_and_member_lookup() {
    // BTF:
    //   id 1: u64
    //   id 2: T (struct, kptr target) { u64 x @ 0 }
    //   id 3: T*
    //   id 4: P (UNION) { u64 slot @ 16 }
    //   id 5: SourceU (struct) { u64 f @ 8 }
    //   id 6: TargetU (UNION) { u64 a @ 0 } -- candidate target
    //
    // Two checks in one test:
    //   (a) STX through a Pointer{P=union} into P.slot at offset
    //       16 records (P, 16) -> T (Kernel).
    //   (b) LDX through a SourceU producing LoadedU64Field then
    //       deref at offset 0 (size 8) finds TargetU (the only
    //       struct/union with u64@0 in this BTF after dropping
    //       SourceU which has u64@8 not u64@0).
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_p = push_name(&mut strings, "P");
    let n_su = push_name(&mut strings, "SourceU");
    let n_tu = push_name(&mut strings, "TargetU");
    let n_x = push_name(&mut strings, "x");
    let n_slot = push_name(&mut strings, "slot");
    let n_f = push_name(&mut strings, "f");
    let n_a = push_name(&mut strings, "a");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // T has its u64 at offset 8 (NOT 0) so layout(0, 8) is
        // uniquely satisfied by TargetU below — production must
        // converge on TargetU only when intersecting candidates.
        SynType::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        SynType::Ptr { type_id: 2 }, // id 3: T*
        // id 4: P as a UNION with a u64 slot at byte 16.
        SynType::Union {
            name_off: n_p,
            size: 24,
            members: vec![SynMember {
                name_off: n_slot,
                type_id: 1,
                byte_offset: 16,
            }],
        },
        // id 5: SourceU as a struct with u64 source field at
        // offset 8.
        SynType::Struct {
            name_off: n_su,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // id 6: TargetU as a UNION with u64 a at offset 0. Sole
        // candidate for the (0, 8) access pattern (after dropping
        // the source struct SourceU which has u64@8 not u64@0).
        // Ensures the layout index walks Union the same as Struct.
        SynType::Union {
            name_off: n_tu,
            size: 8,
            members: vec![SynMember {
                name_off: n_a,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let p_id = 4;
    let source_u_id = 5;
    let target_u_id = 6;
    // Block 1: kptr through union parent.
    //   r1 = Pointer{P=union}; r6 = Pointer{T}.
    //   *(u64*)(r1 + 16) = r6  -> kptr_findings[(P,16)] = T.
    // Block 2: arena LDX through union target.
    //   r2 = Pointer{SourceU}; r3 = *(u64*)(r2 + 8); r4 = *(u64*)(r3 + 0).
    //   Pattern (0, 8) -> {TargetU}; (SourceU, 8) -> (TargetU, Arena).
    // Add arena_confirmed evidence (arena evidence) on r3 between the load
    // of SourceU.f and the deref through it.
    let insns = vec![
        stx(BPF_SIZE_DW, 1, 6, 16),
        ldx(BPF_SIZE_DW, 3, 2, 8),
        addr_space_cast(3, 3, 1),
        ldx(BPF_SIZE_DW, 4, 3, 0),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 1,
                struct_type_id: p_id,
            },
            InitialReg {
                reg: 6,
                struct_type_id: t_id,
            },
            InitialReg {
                reg: 2,
                struct_type_id: source_u_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert_eq!(
        map.get(&(p_id, 16)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
        "kptr through union parent must record: {map:?}"
    );
    assert_eq!(
        map.get(&(source_u_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: target_u_id,
            addr_space: AddrSpace::Arena,
        }),
        "union target must be a layout candidate: {map:?}"
    );
}

/// `build_layout_index`'s walk advances type ids `1..=max_id`
/// using `consecutive_fail` to short-circuit pathological /
/// synthesized BTFs. After 256 consecutive failed
/// `resolve_type_by_id` calls the loop breaks. To exercise this
/// path the test relies on the fact that `max_seen_type_id +
/// CANDIDATE_SEARCH_SLACK` (= 65538 here) is far above the
/// legitimate ids in the BTF, so the walk WOULD iterate ~65k
/// failed lookups without the cap; the consecutive fail cap of
/// 256 short-circuits early. Verify that:
///   - the matcher still finds the legitimate candidate (loop
///     visits valid ids before the cap kicks in),
///   - the matcher does not panic when many failed ids accumulate.
#[test]
fn build_layout_index_consecutive_fail_cap_short_circuits() {
    // BTF: u64(1), T(2, u64@8 source), Q(3, u64@0 unique target).
    // Type ids 4..=65538 would all fail -- production stops after
    // 256 consecutive fails. The legitimate candidate at id 3 is
    // found before the cap activates, so the cast still emits.
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(2, 2, 1),
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
    // Even though ids 4..=65538 (max_seen=2 + slack=65536) all
    // resolve to errors, the consecutive_fail cap (256) stops the
    // loop early without panic, and Q(3) is recorded normally.
    assert_eq!(
        map.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "valid candidate must be found before fail cap; sparse \
             BTF must not panic: {map:?}"
    );
}

/// Non-bitfield members (`bitfield_size_bits = 0`) inside a
/// `kind_flag=1` struct ARE included in `build_layout_index`. The
/// production guard
/// `matches!(m.bitfield_size(), Some(s) if s > 0)` only matches
/// when the size is strictly positive. With kind_flag=1 every
/// member exposes `bitfield_size = Some(0)` for non-bitfield
/// members; production must NOT skip them.
#[test]
fn kind_flag_struct_includes_non_bitfield_members() {
    // BTF:
    //   id 1: u64
    //   id 2: T (kind_flag=0) { u64 src @ 8 }   -- source struct
    //   id 3: Q (kind_flag=1) { u64 a @ 0,
    //                            bf u64 b @ 64 (bitfield 32) }
    // Q has a non-bitfield u64 at byte 0. Production must include
    // it in layout (kind_flag=1, but bitfield_size=Some(0) since
    // bitfield_size_bits=0). The bitfield member at byte 8 is
    // skipped (bitfield_size_bits=32 > 0). Pattern (0, 8) -> {Q}.
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q = push_name(&mut strings, "Q");
    let n_src = push_name(&mut strings, "src");
    let n_a = push_name(&mut strings, "a");
    let n_b = push_name(&mut strings, "b");
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
            size: 16,
            members: vec![SynMember {
                name_off: n_src,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // Q (kind_flag=1) has a non-bitfield u64 at byte 0 AND a
        // bitfield u64 at byte 8 (32-bit field). The non-bitfield
        // member must remain in layout despite kind_flag=1.
        SynType::StructBitfields {
            name_off: n_q,
            size: 16,
            members: vec![
                SynMemberBits {
                    name_off: n_a,
                    type_id: 1,
                    bit_offset: 0,
                    bitfield_size_bits: 0,
                },
                SynMemberBits {
                    name_off: n_b,
                    type_id: 1,
                    bit_offset: 64,
                    bitfield_size_bits: 32,
                },
            ],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let q_id = 3;
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(2, 2, 1),
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
        "non-bitfield member of kind_flag=1 struct must be a \
             layout candidate: {map:?}"
    );
}

// ----- Stack edge cases ---------------------------------------

/// `handle_stx`'s r10 spill guard treats a store with `off >= 0`
/// as out-of-spec for BPF (the stack grows DOWN; slots live at
/// negative offsets). The guard removes any prior slot at that
/// offset rather than saving state. Symmetrically, `handle_ldx`'s
/// r10 reload guard rejects loads with `off >= 0` and produces
/// Unknown. Verify the STX path's invalidation: a write with
/// `off >= 0` through r10 must remove any previously saved slot
/// state at that offset so a later DW reload returns Unknown.
#[test]
fn stack_off_non_negative_through_r10_invalidates() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // *(u64 *)(r10 + 0) = R1   ; off >= 0 spill: dropped, no
    //                            ; state saved at slot 0
    // R3 = *(u64 *)(r10 + 0)   ; off >= 0 reload: returns Unknown
    // *(u64 *)(R6 + slot_off) = R3
    //                          ; R3 Unknown -> no kptr finding
    let insns = vec![
        stx(BPF_SIZE_DW, 10, 1, 0),
        ldx(BPF_SIZE_DW, 3, 10, 0),
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
        "non-negative r10 store must not save state, reload \
             returns Unknown: {map:?}"
    );
}

/// `field_byte_offset` returns `None` for negative offsets when
/// the base register is NOT r10 (stack-relative loads are handled
/// separately via the r10 fast path). On a struct-pointer base,
/// a negative `off` is undefined behavior -- kernel struct fields
/// have non-negative byte offsets relative to the struct base.
/// Production drops dst to Unknown via
/// `field_byte_offset(off) -> None` and the LDX records nothing.
#[test]
fn negative_off_in_non_r10_context_drops() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // r2 = *(u64 *)(r1 - 8): r1 is Pointer{T}, NOT r10. Negative
    // off goes to the Pointer arm of handle_ldx and produces
    // None in field_byte_offset, dropping dst to Unknown. No
    // pattern recorded.
    // r3 = *(u64 *)(r2 + 0): r2 Unknown -> no record.
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, -8),
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
    assert!(
        map.is_empty(),
        "negative offset through Pointer{{T}} must drop, no \
             pattern recorded: {map:?}"
    );
}

/// Two consecutive STX-through-r10 spills with the SAME source
/// register state (Pointer{T}) overwrite the slot with a value
/// indistinguishable from the prior contents. Production stores
/// the second spill via `self.stack_slots.insert(off, regs[src])`
/// which replaces by key but does not collapse to Conflicting.
/// A later reload restores the same Pointer{T}; a subsequent STX
/// of the reloaded register into a parent slot must record the
/// kptr finding as `Single(T)` -- NOT `Conflicting`.
#[test]
fn stack_spill_same_target_stays_single() {
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // *(u64 *)(r10 - 8) = R1   ; spill Pointer{T} (slot=Pointer{T})
    // *(u64 *)(r10 - 8) = R1   ; spill Pointer{T} again -- same
    //                            ; target type, slot replaced
    //                            ; with same value, NOT Conflicting
    // R3 = *(u64 *)(r10 - 8)   ; reload as Pointer{T}
    // *(u64 *)(R6 + slot_off) = R3
    //                          ; records (P, slot_off) -> T
    let insns = vec![
        stx(BPF_SIZE_DW, 10, 1, -8),
        stx(BPF_SIZE_DW, 10, 1, -8),
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
        "spill of identical Pointer{{T}} must reload as Pointer{{T}}, \
             kptr stays Single: {map:?}"
    );
}

// ----- kfunc edge cases ---------------------------------------

/// `handle_kfunc_call` walks the FuncProto's return type through
/// `bpf_map::resolve_to_struct_id`. A return type that resolves
/// to a non-struct pointer (e.g. `int *`, `void *`) yields `None`
/// from the resolver, so R0 stays Unknown. A subsequent STX of
/// R0 into a u64 slot must NOT record a kptr finding.
#[test]
fn kfunc_call_returning_int_ptr_leaves_r0_unknown() {
    let slot_off: u32 = 16;
    // BTF:
    //   id 1: u64
    //   id 2: P (struct with u64 slot @ slot_off) -- kptr parent
    //         seed type for the post-call STX
    //   id 3: int* (Ptr -> u64). Pointee peels to Type::Int, so
    //         resolve_to_struct_id returns None.
    //   id 4: FuncProto returning id 3 (int*)
    //   id 5: Func -> id 4
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_p = push_name(&mut strings, "P");
    let n_slot = push_name(&mut strings, "slot");
    let n_kfunc = push_name(&mut strings, "bpf_returns_int_ptr");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
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
        SynType::Ptr { type_id: 1 }, // id 3: u64*
        SynType::FuncProto {
            return_type_id: 3,
            params: vec![],
        },
        SynType::Func {
            name_off: n_kfunc,
            type_id: 4,
            linkage: 1,
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let p_id = 2;
    let kfunc_id = 5;
    // call kfunc returning int*; *(P + slot) = R0. R0 must be
    // Unknown (the return type's pointee resolves to Int, not
    // Struct, so resolve_to_struct_id returns None).
    let insns = vec![
        kfunc_call(kfunc_id),
        stx(BPF_SIZE_DW, 6, 0, slot_off as i16),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: p_id,
        }],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "kfunc returning int* must leave R0 Unknown: {map:?}"
    );
}

/// `handle_kfunc_call` short-circuits when the FuncProto's
/// `return_type_id == 0` (void return). R0 stays Unknown after
/// the standard r0..r5 clobber. A subsequent STX of R0 into a
/// u64 slot must NOT record a kptr finding.
#[test]
fn kfunc_call_void_return_leaves_r0_unknown() {
    let slot_off: u32 = 16;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_p = push_name(&mut strings, "P");
    let n_slot = push_name(&mut strings, "slot");
    let n_kfunc = push_name(&mut strings, "bpf_void_return");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
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
        // id 3: FuncProto -> void (return_type_id = 0).
        SynType::FuncProto {
            return_type_id: 0,
            params: vec![],
        },
        // id 4: Func -> id 3.
        SynType::Func {
            name_off: n_kfunc,
            type_id: 3,
            linkage: 1,
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let p_id = 2;
    let kfunc_id = 4;
    let insns = vec![
        kfunc_call(kfunc_id),
        stx(BPF_SIZE_DW, 6, 0, slot_off as i16),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: p_id,
        }],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "kfunc with void return must leave R0 Unknown: {map:?}"
    );
}

/// `handle_kfunc_call`'s `imm` may resolve directly to a
/// `Type::FuncProto` (no `Type::Func` wrapper). Production peels
/// `Type::Func -> Type::FuncProto` when needed but also accepts a
/// FuncProto id directly. A kfunc call with `imm` == FuncProto id
/// must seed R0 from the proto's return type just as it would
/// from a Func wrapper.
#[test]
fn kfunc_call_with_funcproto_id_directly() {
    let slot_off: u32 = 16;
    // BTF:
    //   id 1: u64
    //   id 2: T (kptr target struct) { u64 x @ 0 }
    //   id 3: T*
    //   id 4: P (struct holding the kptr slot)
    //   id 5: FuncProto -> T*  (no Func wrapper)
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
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
            name_off: n_t,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynType::Ptr { type_id: 2 }, // id 3: T*
        SynType::Struct {
            name_off: n_p,
            size: slot_off + 8,
            members: vec![SynMember {
                name_off: n_slot,
                type_id: 1,
                byte_offset: slot_off,
            }],
        },
        // id 5: FuncProto returning T* -- pass id=5 directly to
        // kfunc_call's imm so the resolver hits the Type::FuncProto
        // arm, not Type::Func -> peel.
        SynType::FuncProto {
            return_type_id: 3,
            params: vec![],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let p_id = 4;
    let proto_id = 5;
    // kfunc call with imm = proto_id (id 5 = FuncProto). R0 must
    // be set to Pointer{T} via the FuncProto-direct path.
    let insns = vec![
        kfunc_call(proto_id),
        stx(BPF_SIZE_DW, 6, 0, slot_off as i16),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: p_id,
        }],
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
        "kfunc with direct FuncProto id must seed R0 from return \
             type: {map:?}"
    );
}
