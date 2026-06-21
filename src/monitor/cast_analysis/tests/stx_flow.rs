use super::*;

// ----- STX-flow arena cast (allocator-return seed path) ------
//
// The STX-flow path detects allocator-return values stored into
// u64 slots: at a `BPF_PSEUDO_CALL` site flagged via
// [`SubprogReturn::insn_offset`], the analyzer seeds R0 to
// `RegState::ArenaU64FromAlloc` after the standard R0..=R5
// clobber. The next STX of R0 (or its propagation through MOV /
// stack spill) into a u64 field of a typed `Pointer{P}` parent
// records `(P, off)` as an Arena cast finding with
// `target_type_id == 0` (the renderer's resolve_arena_type
// bridge supplies the actual payload type at chase time).

/// Allocator-return → STX path: a `BPF_PSEUDO_CALL` flagged by
/// `SubprogReturn` seeds R0 to `ArenaU64FromAlloc`; the
/// subsequent `STX [R1+8] = R0` records `(M, 8)` as an Arena
/// finding with `target_type_id == 0`. The renderer's
/// `resolve_arena_type` bridge resolves the actual payload type
/// at chase time.
#[test]
fn stx_flow_alloc_return_records_arena_finding() {
    // BTF: u64(1), M(2, u64@8 — the `cgx_raw` slot).
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_m = push_name(&mut strings, "M");
    let n_cgx = push_name(&mut strings, "cgx_raw");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Struct {
            name_off: n_m,
            size: 16,
            members: vec![SynMember {
                name_off: n_cgx,
                type_id: 1,
                byte_offset: 8,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let m_id = 2;
    // R6 is callee-saved per the BPF ABI — survives the
    // R0..=R5 clobber the call applies. Seed R6 = Pointer{M}
    // so the post-call STX through R6 still has a typed parent
    // base. R1 cannot be used as the parent (the call clobbers
    // it as an argument register), and the test specifically
    // wants the analyzer to record `(M, 8)` AFTER the call.
    //
    // Insn 0: BPF_PSEUDO_CALL — SubprogReturn seed at
    //         insn_offset=0 tags R0 as ArenaU64FromAlloc
    //         after the R0..=R5 clobber.
    // Insn 1: STX [R6 + 8] = R0 — records (M, 8) -> Arena
    //         via the STX-flow path.
    let pseudo_call = mk_insn(
        BPF_CLASS_JMP | BPF_OP_CALL,
        0,
        BPF_PSEUDO_CALL,
        0,
        0, // pc-relative offset to the subprog (irrelevant here)
    );
    let insns = vec![pseudo_call, stx(BPF_SIZE_DW, 6, 0, 8), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: m_id,
        }],
        &[],
        &[],
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 0,
        }],
    );
    assert_eq!(
        map.get(&(m_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "STX-flow alloc-return must record Arena finding with \
             target_type_id=0: {map:?}"
    );
}

/// Allocator-return seed propagates through `mov_x` so a register
/// rename before the STX still records the Arena finding.
#[test]
fn stx_flow_alloc_return_propagates_through_mov() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_m = push_name(&mut strings, "M");
    let n_cgx = push_name(&mut strings, "cgx_raw");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Struct {
            name_off: n_m,
            size: 16,
            members: vec![SynMember {
                name_off: n_cgx,
                type_id: 1,
                byte_offset: 8,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let m_id = 2;
    let pseudo_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    // r7 = r0  (mov_x). r7 is callee-saved per the BPF ABI;
    // mov_x propagates ArenaU64FromAlloc from R0 to R7. R6 is
    // the parent base (callee-saved, seeded as `Pointer{M}`),
    // so the STX through [R6 + 8] = R7 records the Arena
    // finding. The post-call STX cannot use R1 as the parent —
    // the call clobbered R1 along with the rest of R0..=R5.
    let insns = vec![pseudo_call, mov_x(7, 0), stx(BPF_SIZE_DW, 6, 7, 8), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: m_id,
        }],
        &[],
        &[],
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 0,
        }],
    );
    assert_eq!(
        map.get(&(m_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "MOV must propagate ArenaU64FromAlloc through r7: {map:?}"
    );
}

/// Allocator-return seed propagates through stack spill / reload.
/// Mirrors the `register_stack_spill_round_trip` pattern: STX
/// through r10 saves the register state, LDX through r10
/// restores it.
#[test]
fn stx_flow_alloc_return_round_trips_through_stack() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_m = push_name(&mut strings, "M");
    let n_cgx = push_name(&mut strings, "cgx_raw");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Struct {
            name_off: n_m,
            size: 16,
            members: vec![SynMember {
                name_off: n_cgx,
                type_id: 1,
                byte_offset: 8,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let m_id = 2;
    let pseudo_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    // *(u64 *)(r10 - 8) = r0   ; spill R0 (ArenaU64FromAlloc)
    // r7 = *(u64 *)(r10 - 8)   ; reload — R7 holds ArenaU64FromAlloc
    // *(u64 *)(R6 + 8) = R7    ; records (M, 8) -> Arena
    //
    // R6 is the parent base, seeded as `Pointer{M}`. Both R6
    // and R7 are callee-saved — they survive the R0..=R5
    // clobber the call applied at PC 0.
    let insns = vec![
        pseudo_call,
        stx(BPF_SIZE_DW, 10, 0, -8),
        ldx(BPF_SIZE_DW, 7, 10, -8),
        stx(BPF_SIZE_DW, 6, 7, 8),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: m_id,
        }],
        &[],
        &[],
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 0,
        }],
    );
    assert_eq!(
        map.get(&(m_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "Stack spill/reload must round-trip ArenaU64FromAlloc: {map:?}"
    );
}

/// Subsequent LDX from a previously-arena-tagged slot inherits
/// `ArenaU64FromAlloc` (alias-set tracking via the source field).
/// Storing the inherited tag into another u64 slot also records
/// the Arena finding.
#[test]
fn stx_flow_alias_tracking_propagates_via_ldx() {
    // M: { u64 src_slot @ 0; u64 dst_slot @ 8 }
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_m = push_name(&mut strings, "M");
    let n_src = push_name(&mut strings, "src_slot");
    let n_dst = push_name(&mut strings, "dst_slot");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Struct {
            name_off: n_m,
            size: 16,
            members: vec![
                SynMember {
                    name_off: n_src,
                    type_id: 1,
                    byte_offset: 0,
                },
                SynMember {
                    name_off: n_dst,
                    type_id: 1,
                    byte_offset: 8,
                },
            ],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let m_id = 2;
    let pseudo_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    // pc 0: pseudo_call -> R0 = ArenaU64FromAlloc { source: None }.
    // pc 1: stx [R6 + 0] = R0 -> records (M, 0) -> Arena.
    // pc 2: ldx R7, [R6 + 0]   -> R7 inherits ArenaU64FromAlloc
    //                              { source: Some((M, 0)) } via
    //                              alias-set tracking.
    // pc 3: stx [R6 + 8] = R7 -> records (M, 8) -> Arena.
    //
    // R6 is the parent base (callee-saved, seeded as
    // `Pointer{M}`); R7 is also callee-saved and survives
    // any future call clobber. R1 cannot be the parent here —
    // the call at PC 0 clobbered it.
    let insns = vec![
        pseudo_call,
        stx(BPF_SIZE_DW, 6, 0, 0),
        ldx(BPF_SIZE_DW, 7, 6, 0),
        stx(BPF_SIZE_DW, 6, 7, 8),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: m_id,
        }],
        &[],
        &[],
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 0,
        }],
    );
    assert_eq!(
        map.get(&(m_id, 0)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "first STX must record (M, 0) -> Arena: {map:?}"
    );
    assert_eq!(
        map.get(&(m_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "alias-tracked LDX from (M, 0) must propagate to (M, 8) STX: {map:?}"
    );
}

/// Conflict between arena STX and kernel kptr STX on the same
/// slot drops both observations from the output map. Mirrors the
/// existing arena/kptr conflict invariant for the new path.
#[test]
fn stx_flow_conflict_with_kptr_drops_both() {
    // BTF: u64(1), T(2, struct), T*(3), M(4, u64@0).
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_m = push_name(&mut strings, "M");
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
        SynType::Ptr { type_id: 2 },
        SynType::Struct {
            name_off: n_m,
            size: 8,
            members: vec![SynMember {
                name_off: n_slot,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let m_id = 4;
    let t_id = 2;
    let pseudo_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    // R6 = M* (parent base, callee-saved → survives the call).
    // R7 = T* (kptr value source, also callee-saved → survives
    // the call). pc 0: pseudo_call → R0 = arena.
    // pc 1: stx [R6+0] = R0 → arena_stx_findings[(M,0)] = Pending.
    // pc 2: stx [R6+0] = R7 → kptr_findings[(M,0)] = Single(T).
    // Conflict: both observations on (M, 0) — drop both.
    let insns = vec![
        pseudo_call,
        stx(BPF_SIZE_DW, 6, 0, 0),
        stx(BPF_SIZE_DW, 6, 7, 0),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 6,
                struct_type_id: m_id,
            },
            InitialReg {
                reg: 7,
                struct_type_id: t_id,
            },
        ],
        &[],
        &[],
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 0,
        }],
    );
    assert!(
        !map.contains_key(&(m_id, 0)),
        "arena/kptr conflict must drop the slot from output: {map:?}"
    );
}

/// Same slot triggers both detection paths in one program: the
/// shape-inference path's LDX accumulates into `patterns`, AND a
/// `pseudo_call` + STX-flow path inserts into `arena_stx_findings`.
/// The two evidence sources combine: STX-flow proves the slot
/// holds an arena pointer (gates emission past the arena-evidence direct-
/// evidence requirement), and shape-inference resolves the
/// target struct from the observed dereference pattern. The
/// arena STX-flow loop in [`Analyzer::finalize`] runs the same
/// shape-inference intersection across `patterns[key]` as the
/// shape-inference loop does, so a slot with BOTH evidence
/// sources emits a `CastHit` whose `target_type_id` is the
/// uniquely-resolved struct id (when shape inference resolves)
/// — not the deferred-resolve sentinel.
///
/// Pre-fix: the LDX alias-tracking arm at
/// [`Analyzer::handle_ldx`] re-typed any LDX off a slot already
/// in `arena_stx_findings` to `RegState::ArenaU64FromAlloc`,
/// which suppressed downstream access recording (the arena arm
/// in `handle_ldx` drops dst without populating patterns) — so
/// even when struct shape uniquely identified the target,
/// `target_type_id` stayed 0. Post-fix the LDX always emits
/// `LoadedU64Field`, downstream LDXs through it record the
/// access pattern, and shape inference resolves the target the
/// renderer can chase against without consulting the
/// `MemReader::resolve_arena_type` bridge. The bridge stays in
/// place for slots whose access pattern doesn't uniquely
/// resolve.
///
/// Test fixture: struct Q has `u64@0` + `u64@8` (size 16), so
/// the access pattern at offsets 0 and 8 with size 8 uniquely
/// identifies Q. The assertion pins that target_type_id ==
/// Q's BTF id, NOT 0.
#[test]
fn stx_flow_resolves_target_via_shape_inference_under_alias_tracking() {
    // BTF: u64(1), P(2, u64@8 source), Q(3, u64@0+u64@8). Q is the
    // unique candidate matching both pattern accesses (offset=0,
    // size=8) and (offset=8, size=8). Both detection paths fire and
    // combine: STX-flow proves the slot holds an arena pointer and
    // gates emission, shape inference resolves the target struct
    // from the access pattern.
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_p = push_name(&mut strings, "P");
    let n_q = push_name(&mut strings, "Q");
    let n_f = push_name(&mut strings, "f");
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
            name_off: n_p,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        SynType::Struct {
            name_off: n_q,
            size: 16,
            members: vec![
                SynMember {
                    name_off: n_a,
                    type_id: 1,
                    byte_offset: 0,
                },
                SynMember {
                    name_off: n_b,
                    type_id: 1,
                    byte_offset: 8,
                },
            ],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let p_id = 2;
    let q_id = 3;
    let pseudo_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    // Phase 1 — shape-inference accesses for (P, 8). Both passes
    // record the same pattern: the LDX at PC 0 emits
    // `LoadedU64Field{P, 8}` regardless of whether
    // `arena_stx_findings` already contains the slot, and
    // downstream LDXs through that register record accesses
    // (0, 8) and (8, 8) into `patterns[(P, 8)]`. Q is the
    // unique candidate matching both accesses (it has u64@0
    // and u64@8, size 16), so finalize's intersection resolves
    // `target_type_id = q_id`.
    //   r2 = LDX[r1 + 8]   ; LoadedU64Field{P, 8}
    //   r3 = LDX[r2 + 0]   ; records access (0, 8)
    //   r4 = LDX[r2 + 8]   ; records access (8, 8)
    // Phase 2 — preserve r1 across the call clobber by stashing
    // it in r6 (callee-saved per BPF ABI, R0..R5 only clobbered):
    //   r6 = r1
    // Phase 3 — STX-flow tags the same slot:
    //   pseudo_call (PC 4)  ; SubprogReturn at PC=4 sets R0 to
    //                         RegState::ArenaU64FromAlloc after
    //                         the standard R0..=R5 clobber
    //   STX[r6 + 8] = r0    ; arena_stx_findings.insert((P, 8),
    //                         Pending)
    let insns = vec![
        // Phase 1: shape inference seeding.
        ldx(BPF_SIZE_DW, 2, 1, 8),
        ldx(BPF_SIZE_DW, 3, 2, 0),
        ldx(BPF_SIZE_DW, 4, 2, 8),
        // Phase 2: preserve P* across the call.
        mov_x(6, 1),
        // Phase 3: STX-flow tag.
        pseudo_call,
        stx(BPF_SIZE_DW, 6, 0, 8),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: p_id,
        }],
        &[],
        &[],
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 4,
        }],
    );
    assert_eq!(
        map.get(&(p_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "STX-flow gates emission past the arena-evidence requirement; shape inference resolves \
             target_type_id from the recorded access pattern (Q is the \
             only struct of size 16 with u64@0 and u64@8): {map:?}"
    );
}

/// Long stream of `BPF_LD_IMM64` two-slot instructions back-to-
/// back, terminated by `exit`. Every `lo` slot sets `skip_next`,
/// and every `hi` slot is the upper-immediate placeholder that
/// the analyzer must not interpret. Verifies the
/// `skip_next`-driven decode path does not run off the end of the
/// slice or misaccount its position when the program is densely
/// packed with two-slot ops. Also exercises the same pattern in
/// `jump_targets`'s pre-pass — both must agree on which slots are
/// second-half placeholders.
/// Helper: build a BTF blob carrying:
///   - `u64` (id 1)
///   - struct `M` (id 2) with one `u64 cgx_raw` member at byte
///     offset 8
///   - `void *` (id 3) — `Ptr -> 0` (the BTF void marker)
///   - `FuncProto` returning id 3 with no params (id 4)
///   - `BTF_KIND_FUNC` named `func_name` extern-linkage referring
///     to FuncProto id 4 (id 5)
///
/// Returns the byte blob plus `(M_id, kfunc_id) = (2, 5)`. Used
/// by the kfunc-allocator-arm tests below — the analyzer's
/// `handle_kfunc_call` arm peels the kfunc's return type to
/// `Ptr -> Void`, looks up the kfunc's name, and applies the
/// `ARENA_ALLOC_KFUNC_NAMES` allowlist to decide whether to tag
/// R0 as `ArenaU64FromAlloc`.
fn btf_with_arena_alloc_kfunc(func_name: &str) -> (Vec<u8>, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_m = push_name(&mut strings, "M");
    let n_cgx = push_name(&mut strings, "cgx_raw");
    let n_func = push_name(&mut strings, func_name);
    let types = vec![
        // id 1: u64
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct M { u64 cgx_raw @ 8 } size=16
        SynType::Struct {
            name_off: n_m,
            size: 16,
            members: vec![SynMember {
                name_off: n_cgx,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // id 3: Ptr -> Void (pointee type id == 0). The
        // `void *` shape kfunc allocators declare.
        SynType::Ptr { type_id: 0 },
        // id 4: FuncProto returning id 3 with no params.
        SynType::FuncProto {
            return_type_id: 3,
            params: vec![],
        },
        // id 5: Func named func_name, extern, type id = 4.
        SynType::Func {
            name_off: n_func,
            type_id: 4,
            linkage: 2, // BTF_FUNC_EXTERN
        },
    ];
    let blob = build_btf(&types, &strings);
    (blob, 2, 5)
}

/// Kfunc allocator arm — happy path. Calling
/// `bpf_arena_alloc_pages` (an allowlisted kfunc whose return
/// peels to `Ptr -> Void`) tags R0 as
/// [`RegState::ArenaU64FromAlloc`]; the subsequent STX of R0
/// into a u64 slot of a typed parent records an Arena finding.
#[test]
fn kfunc_arena_alloc_allowlist_records_arena_finding() {
    let (blob, m_id, kfunc_id) = btf_with_arena_alloc_kfunc("bpf_arena_alloc_pages");
    let btf = Btf::from_bytes(&blob).unwrap();
    // R6 holds `Pointer{M}` and is callee-saved per the BPF
    // ABI, so it survives the R0..=R5 clobber the kfunc call
    // applies. Seeding R1 instead would be wiped by the call
    // and the post-call STX would have an Unknown parent.
    //
    // Insn 0: BPF_PSEUDO_KFUNC_CALL with imm=kfunc_id. The
    //         analyzer's `handle_kfunc_call` resolves the
    //         kfunc, peels the return through `Ptr -> Void`,
    //         matches the name against
    //         `ARENA_ALLOC_KFUNC_NAMES`, and tags R0 as
    //         ArenaU64FromAlloc.
    // Insn 1: STX [R6 + 8] = R0 — records (M, 8) -> Arena.
    let kfunc_call = mk_insn(
        BPF_CLASS_JMP | BPF_OP_CALL,
        0,
        BPF_PSEUDO_KFUNC_CALL,
        0,
        kfunc_id as i32,
    );
    let insns = vec![kfunc_call, stx(BPF_SIZE_DW, 6, 0, 8), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: m_id,
        }],
        &[],
        &[],
        &[],
    );
    assert_eq!(
        map.get(&(m_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "allowlisted kfunc with `Ptr -> Void` return must seed R0 \
             as ArenaU64FromAlloc; subsequent STX must record an Arena \
             finding: {map:?}"
    );
}

/// Kfunc allocator arm — strict gate: a kfunc whose name is on
/// the allowlist BUT whose return type is NOT `Ptr -> Void`
/// (e.g. a same-named kfunc with a typed-pointer return) must
/// fall through to the typed-pointer arm OR no-op. Drift between
/// the kernel BTF and the analyzer's allowlist must not produce
/// a false positive.
#[test]
fn kfunc_arena_alloc_typed_return_falls_through() {
    // BTF: u64(1), M(2), struct R(3, u64@0), Ptr->R(4),
    // FuncProto returning Ptr->R(5), Func named
    // "bpf_arena_alloc_pages" -> proto 5 (id 6).
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_m = push_name(&mut strings, "M");
    let n_cgx = push_name(&mut strings, "cgx_raw");
    let n_r = push_name(&mut strings, "R");
    let n_x = push_name(&mut strings, "x");
    let n_func = push_name(&mut strings, "bpf_arena_alloc_pages");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Struct {
            name_off: n_m,
            size: 16,
            members: vec![SynMember {
                name_off: n_cgx,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // id 3: struct R { u64 x @ 0 } — a TYPED struct, not Void.
        SynType::Struct {
            name_off: n_r,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 4: Ptr -> R (pointee type id == 3, NOT 0).
        SynType::Ptr { type_id: 3 },
        // id 5: FuncProto returning id 4.
        SynType::FuncProto {
            return_type_id: 4,
            params: vec![],
        },
        // id 6: Func named the allowlisted name, but with a
        // typed-pointer return.
        SynType::Func {
            name_off: n_func,
            type_id: 5,
            linkage: 2,
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let m_id = 2;
    let kfunc_id = 6;
    let kfunc_call = mk_insn(
        BPF_CLASS_JMP | BPF_OP_CALL,
        0,
        BPF_PSEUDO_KFUNC_CALL,
        0,
        kfunc_id,
    );
    // R6 holds `Pointer{M}` (callee-saved, survives the call).
    // The post-call STX through R6 records the kptr finding
    // for the (M, 8) slot.
    let insns = vec![kfunc_call, stx(BPF_SIZE_DW, 6, 0, 8), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: m_id,
        }],
        &[],
        &[],
        &[],
    );
    // The typed-pointer arm fires first and tags R0 as
    // `Pointer{R}` (not ArenaU64FromAlloc). The subsequent STX
    // records a Kernel kptr finding `(M, 8) -> R`, NOT an
    // Arena finding. The allowlist arm DID NOT execute because
    // arm 1 returned first.
    assert_eq!(
        map.get(&(m_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 3, // R's id
            addr_space: AddrSpace::Kernel,
        }),
        "kfunc whose return is typed (Ptr -> Struct) must take the \
             typed-pointer arm, NOT the arena allocator arm; the \
             allowlist arm must not produce a false-positive Arena \
             finding: {map:?}"
    );
}

/// Kfunc allocator arm — strict gate: a kfunc whose return
/// peels to `Ptr -> Void` but whose name is NOT on the
/// allowlist must NOT seed R0 as ArenaU64FromAlloc. Without
/// the name gate, every `void *`-returning kfunc would tag
/// arbitrary u64 slots as arena-shaped.
#[test]
fn kfunc_arena_alloc_non_allowlist_name_drops() {
    // Use an unrelated kfunc name. The BTF still has a
    // `Ptr -> Void` return.
    let (blob, m_id, kfunc_id) = btf_with_arena_alloc_kfunc("ktstr_unlisted_kfunc");
    let btf = Btf::from_bytes(&blob).unwrap();
    let kfunc_call = mk_insn(
        BPF_CLASS_JMP | BPF_OP_CALL,
        0,
        BPF_PSEUDO_KFUNC_CALL,
        0,
        kfunc_id as i32,
    );
    // R6 (callee-saved) holds `Pointer{M}`; survives the call
    // clobber. The post-call STX through R6 has a typed parent;
    // the only way the assertion (`map.is_empty()`) could fail
    // is if the analyzer mistakenly tagged R0 as
    // ArenaU64FromAlloc despite the non-allowlist name. Using
    // R6 ensures the test is not falsely passing because the
    // STX itself failed (R1 clobbered).
    let insns = vec![kfunc_call, stx(BPF_SIZE_DW, 6, 0, 8), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: m_id,
        }],
        &[],
        &[],
        &[],
    );
    // R0 stays Unknown (arm 1 sees a non-Struct return and
    // returns; arm 2 sees a non-allowlist name and drops).
    // The STX with R0=Unknown produces no finding regardless
    // of whether the parent base register is typed.
    assert!(
        map.is_empty(),
        "kfunc with `Ptr -> Void` return but non-allowlist name must \
             NOT seed an arena finding: {map:?}"
    );
}

