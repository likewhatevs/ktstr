use super::*;

// -----------------------------------------------------------------
//
// Edge-arm coverage for three under-tested branches of the cast
// analyzer's forward walk:
//
//   1. The `ALU64 | ADD | K` arm on a `RegState::FrameAddr`
//      register (mod.rs step()): the `i16::try_from(new_off)`
//      overflow -> Unknown branch, and the else-fall-through when
//      the destination is NOT a FrameAddr.
//   2. The `handle_kfunc_call` inner `_ => return` arm: a
//      `BTF_KIND_FUNC` whose type-field resolves to a non-FuncProto
//      type, and an allowlisted kfunc whose return is a typed
//      `Ptr -> Int` (neither `Ptr -> Struct` nor `Ptr -> Void`).
//   3. The `handle_stx` `StxValueKind::Unknown` no-op arm: a later
//      untyped DW store to a slot already in `arena_stx_findings`
//      must NOT erase the recorded Arena finding (may-analysis
//      persistence).
//
// All fixtures are synthetic BTF built via `build_btf`; no VM.

// ----- FrameAddr ADD-K arm ------------------------------------

/// Build a `.maps`-style fixture for the `bpf_map_update_elem`
/// bridge path: value struct `V` with one `u64` field at byte
/// offset 8, an outer parent struct `P` (so the parent register is
/// typed), and a `.maps` datasec declaring a map whose `value`
/// member peels to `Ptr -> V`. Returns the byte blob plus
/// `(datasec_id, var_off, v_id, p_id)`.
///
/// Mirrors the shape `helper_map_update_then_lookup_propagates_arena_through_map_value`
/// uses, trimmed to the single-`update_elem` form the FrameAddr arm
/// needs: the bridge walks `V`'s members at `r3_base + member_off`,
/// so the test controls the `r3 += imm` value that produces (or
/// overflows) `r3_base`.
fn btf_map_update_value_v(slot_off: u32) -> (Vec<u8>, u32, u32, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_v = push_name(&mut strings, "V");
    let n_v_field = push_name(&mut strings, "v_field");
    let n_p = push_name(&mut strings, "P");
    let n_p_field = push_name(&mut strings, "p_field");
    let n_type = push_name(&mut strings, "type");
    let n_value = push_name(&mut strings, "value");
    let n_map_def = push_name(&mut strings, "anon_map_def");
    let n_map_var = push_name(&mut strings, "the_map");
    let n_maps = push_name(&mut strings, ".maps");
    let types = vec![
        // id 1: u64.
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct V { u64 v_field @ slot_off }.
        SynType::Struct {
            name_off: n_v,
            size: slot_off + 8,
            members: vec![SynMember {
                name_off: n_v_field,
                type_id: 1,
                byte_offset: slot_off,
            }],
        },
        // id 3: struct P { u64 p_field @ 0 } — parent, distinct
        // from V so the analyzer's self-store rejection does not
        // fire on the bridge-recorded finding.
        SynType::Struct {
            name_off: n_p,
            size: 8,
            members: vec![SynMember {
                name_off: n_p_field,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 4: Ptr -> V (the map_def's `value` member type).
        SynType::Ptr { type_id: 2 },
        // id 5: anonymous map_def { u32 type @ 0; V *value @ 8 }.
        SynType::Struct {
            name_off: n_map_def,
            size: 16,
            members: vec![
                SynMember {
                    name_off: n_type,
                    type_id: 1,
                    byte_offset: 0,
                },
                SynMember {
                    name_off: n_value,
                    type_id: 4,
                    byte_offset: 8,
                },
            ],
        },
        // id 6: Var "the_map" of type map_def, GLOBAL linkage.
        SynType::Var {
            name_off: n_map_var,
            type_id: 5,
            linkage: 1,
        },
        // id 7: Datasec ".maps" containing the_map at offset 0.
        SynType::Datasec {
            name_off: n_maps,
            size: 16,
            entries: vec![SynVarSecinfo {
                type_id: 6,
                offset: 0,
                size: 16,
            }],
        },
    ];
    (build_btf(&types, &strings), 7, 0, 2, 3)
}

/// Build the single-`update_elem` instruction stream parameterised
/// by the `r3 += imm` immediate. `pseudo_call` (annotated as a
/// `SubprogReturn`) seeds R0 to `ArenaU64FromAlloc` and flips
/// `func_has_alloc`; the spill places that arena value on the stack
/// at `spill_off`; `r3 = r10; r3 += add_k_imm` builds the value
/// pointer the bridge resolves against; `call bpf_map_update_elem`
/// drives `bridge_map_value_spill`.
///
/// PC layout (LD_IMM64 second slot skipped via skip_next):
///   0 pseudo_call
///   1 stx [r10 + spill_off] = r0
///   2 ld_imm64 r1, .maps@0    (3 = hi slot)
///   4 mov r2, r10             (key pointer; content irrelevant)
///   5 mov r3, r10             (FrameAddr{0})
///   6 r3 += add_k_imm         (FrameAddr arm under test)
///   7 call bpf_map_update_elem
///   8 exit
fn map_update_with_r3_add_k(spill_off: i16, add_k_imm: i32) -> (Vec<BpfInsn>, Vec<DatasecPointer>) {
    let pseudo_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    let stx_spill = stx(BPF_SIZE_DW, 10, 0, spill_off);
    let [ld_lo, ld_hi] = ld_imm64(1, 0);
    let mov_r2 = mov_x(2, 10);
    let mov_r3 = mov_x(3, 10);
    // r3 += add_k_imm (ALU64 | ADD | K). src field unused; imm
    // carries the constant. This is the FrameAddr ADD-K arm.
    let r3_add_k = mk_insn(BPF_CLASS_ALU64 | (bs::BPF_ADD as u8), 3, 0, 0, add_k_imm);
    let call_update = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, 0, 0, BPF_FUNC_MAP_UPDATE_ELEM);
    let insns = vec![
        pseudo_call,
        stx_spill,
        ld_lo,
        ld_hi,
        mov_r2,
        mov_r3,
        r3_add_k,
        call_update,
        exit(),
    ];
    // PC 2 is the LD_IMM64 lo slot for r1 -> .maps@0.
    let datasec_pointers = vec![DatasecPointer {
        insn_offset: 2,
        datasec_type_id: 7,
        base_offset: 0,
    }];
    (insns, datasec_pointers)
}

/// In-range control for [`step_alu64_add_k_overflow_drops_frameaddr_to_unknown`]:
/// with `r3 += -16` the saturating add `0 + (-16) = -16` round-trips
/// cleanly through `i16::try_from`, so r3 stays
/// `FrameAddr{offset: -16}`. The `bpf_map_update_elem` bridge then
/// resolves V's u64 field at `r3_base(-16) + member_off(8) = -8`,
/// finds the arena value spilled there, and records `(V, 8) -> Arena`
/// with `target_type_id == 0`. Proves the ONLY difference from the
/// overflow case is the immediate.
#[test]
fn step_alu64_add_k_in_range_keeps_frameaddr_and_records() {
    let (blob, _datasec_id, _var_off, v_id, _p_id) = btf_map_update_value_v(8);
    let btf = Btf::from_bytes(&blob).unwrap();
    // Spill the arena value at r10-8; V.v_field@8 with r3_base=-16
    // -> slot_off = -16 + 8 = -8.
    let (insns, datasec_pointers) = map_update_with_r3_add_k(-8, -16);
    let map = analyze_casts(
        &insns,
        &btf,
        &[],
        &[],
        &datasec_pointers,
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 0,
        }],
    );
    assert_eq!(
        map.get(&(v_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "in-range r3 += -16 keeps FrameAddr{{-16}}; the update_elem bridge \
             must record (V, 8) -> Arena from the spilled arena slot: {map:?}"
    );
}

/// `ALU64 | ADD | K` on a `RegState::FrameAddr` whose saturating-added
/// offset exceeds `i16::MAX` (32767) must drop the register to
/// `Unknown` rather than wrap (mod.rs step(), the
/// `i16::try_from(new_off)` Err arm). With `r3 += 40000` the add
/// `0 + 40000 = 40000 > i16::MAX`, so `i16::try_from` fails and r3
/// becomes `Unknown`. At the `bpf_map_update_elem` call site the
/// `RegState::FrameAddr` guard on the value pointer then fails, so
/// `bridge_map_value_spill` is never called and no Arena finding is
/// recorded — the output map is empty.
///
/// Same fixture, same spill, same SubprogReturn seed as
/// [`step_alu64_add_k_in_range_keeps_frameaddr_and_records`]; the only
/// difference is the ADD-K immediate (40000 vs -16). The in-range
/// sibling records `(V, 8) -> Arena`, so an empty map here proves the
/// overflow arm dropped r3 to Unknown.
#[test]
fn step_alu64_add_k_overflow_drops_frameaddr_to_unknown() {
    let (blob, _datasec_id, _var_off, _v_id, _p_id) = btf_map_update_value_v(8);
    let btf = Btf::from_bytes(&blob).unwrap();
    // 40000 > i16::MAX (32767), so 0 + 40000 overflows i16 -> r3
    // drops to Unknown. The spill slot is irrelevant because the
    // bridge never runs (the FrameAddr guard on r3 fails).
    let (insns, datasec_pointers) = map_update_with_r3_add_k(-8, 40000);
    let map = analyze_casts(
        &insns,
        &btf,
        &[],
        &[],
        &datasec_pointers,
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 0,
        }],
    );
    assert!(
        map.is_empty(),
        "ALU64 ADD K overflowing i16::MAX on a FrameAddr must drop r3 to \
             Unknown so the update_elem bridge records nothing: {map:?}"
    );
}

/// `ALU64 | ADD | K` whose destination is NOT a `RegState::FrameAddr`
/// must drop the register to `Unknown` (mod.rs step(), the
/// else-fall-through after the FrameAddr let-chain guard fails). The
/// offset arithmetic only applies to frame addresses; any other
/// typed register receiving an ADD-K becomes an integer.
///
/// Distinct from `alu64_add_k_destroys_typed_pointer` (which drops a
/// `Pointer{T}`): here the register is a `LoadedU64Field` that has
/// already accrued `arena_confirmed` evidence via `addr_space_cast`.
/// The ADD-K must still drop it, and the prior arena confirmation
/// must NOT keep the downstream deref alive — proving the dst-drop
/// happens before shape inference can record an access through the
/// now-Unknown register.
///
/// Sequence (contrast with `mov_x_propagates_loaded_state`, which
/// keeps state through MOV):
///   r2 = *(u64 *)(r1 + 8)        ; r2 = LoadedU64Field{T, 8}
///   r2 = (cast as(1) -> as(0)) r2 ; arena_confirmed += (T, 8)
///   r2 += 8                       ; ADD-K on non-FrameAddr -> Unknown
///   r3 = *(u64 *)(r2 + 0)        ; deref through Unknown -> no access
/// The deref records no pattern, so no Arena hit emits despite the
/// prior arena_confirmed.
#[test]
fn step_alu64_add_k_on_non_frameaddr_drops_dst() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // ALU64 | ADD | K r2, imm=8 — r2 is LoadedU64Field (not
    // FrameAddr), so the dst is dropped to Unknown.
    let add_k = mk_insn(BPF_CLASS_ALU64 | (bs::BPF_ADD as u8), 2, 0, 0, 8);
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(2, 2, 1),
        add_k,
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
        "ALU64 ADD K on a non-FrameAddr (LoadedU64Field) register must \
             drop dst to Unknown; the deref through it records no access \
             despite the prior arena_confirmed: {map:?}"
    );
}

// ----- handle_kfunc_call resolution-failure arms --------------

/// Build a `kptr_base`-style BTF plus a `BTF_KIND_FUNC` (id 5) whose
/// type field points at `func_type_id`. With `func_type_id` set to a
/// non-FuncProto type id (e.g. a Struct), the analyzer's
/// `handle_kfunc_call` resolves `Func -> func_type_id` to a
/// non-FuncProto and bails at the inner `_ => return` arm. Returns
/// the byte blob plus `(p_id, kfunc_id)`.
fn btf_kfunc_pointing_at(slot_off: u32, func_type_id: u32) -> (Vec<u8>, u32, u32) {
    // Same id layout as `btf_kptr_base` (ids 1..=4: u64, T, T*, P)
    // plus a Func as id 5 so `func_type_id` can reference any of
    // 1..=4. Built inline rather than via `btf_kptr_base` because
    // the Func append needs the same `build_btf` call.
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_p = push_name(&mut strings, "P");
    let n_x = push_name(&mut strings, "x");
    let n_slot = push_name(&mut strings, "slot");
    let n_func = push_name(&mut strings, "ktstr_kfunc");
    let types = vec![
        // id 1: u64
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct T { u64 x @ 0 }
        SynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: T*
        SynType::Ptr { type_id: 2 },
        // id 4: struct P { u64 slot @ slot_off }
        SynType::Struct {
            name_off: n_p,
            size: slot_off + 8,
            members: vec![SynMember {
                name_off: n_slot,
                type_id: 1,
                byte_offset: slot_off,
            }],
        },
        // id 5: Func "ktstr_kfunc" whose type field is func_type_id.
        SynType::Func {
            name_off: n_func,
            type_id: func_type_id,
            linkage: 2, // BTF_FUNC_EXTERN
        },
    ];
    (build_btf(&types, &strings), 4, 5)
}

/// `handle_kfunc_call` inner `_ => return` arm: a `BTF_KIND_FUNC`
/// whose type field resolves to something other than a FuncProto
/// (here a Struct) must leave R0 Unknown. The analyzer resolves
/// `Func -> get_type_id() -> resolve_type_by_id()`, sees a
/// non-FuncProto, and returns without typing R0. The subsequent STX
/// of the (Unknown) R0 into P's u64 slot records no finding.
///
/// (A Func with type field == 0 does NOT exercise a distinct `None`
/// arm: btf-rs `Func::get_type_id()` returns `Some(0)` because
/// `BTF_KIND_FUNC` has `has_type()`, and `resolve_type_by_id(0)`
/// returns `Type::Void`, which also lands in this same `_ => return`
/// arm. Pointing the Func at a Struct keeps the fixture
/// unambiguous.)
#[test]
fn kfunc_call_func_pointing_to_non_funcproto_leaves_r0_unknown() {
    let slot_off: u32 = 8;
    // Func id 5 points at the struct T (id 2), not a FuncProto.
    let (blob, p_id, kfunc_id) = btf_kfunc_pointing_at(slot_off, 2);
    let btf = Btf::from_bytes(&blob).unwrap();
    // R6 holds Pointer{P} (callee-saved, survives the call clobber).
    let kfunc = kfunc_call(kfunc_id);
    let insns = vec![kfunc, stx(BPF_SIZE_DW, 6, 0, slot_off as i16), exit()];
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
        None,
        "kfunc whose Func type field points at a non-FuncProto must leave \
             R0 Unknown (inner _ => return); the STX records nothing: {map:?}"
    );
    assert!(
        map.is_empty(),
        "no other findings expected from the non-FuncProto kfunc: {map:?}"
    );
}

/// Build a fixture for the allowlisted-kfunc + `Ptr -> Int` return
/// case. Func named `bpf_arena_alloc_pages` (on
/// `ARENA_ALLOC_KFUNC_NAMES`) whose FuncProto returns `Ptr -> u64`
/// (a typed scalar pointer — neither `Ptr -> Struct` nor
/// `Ptr -> Void`). Returns the blob plus `(m_id, kfunc_id)`.
fn btf_arena_kfunc_ptr_to_int() -> (Vec<u8>, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_m = push_name(&mut strings, "M");
    let n_slot = push_name(&mut strings, "slot");
    let n_func = push_name(&mut strings, "bpf_arena_alloc_pages");
    let types = vec![
        // id 1: u64
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct M { u64 slot @ 8 }
        SynType::Struct {
            name_off: n_m,
            size: 16,
            members: vec![SynMember {
                name_off: n_slot,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // id 3: Ptr -> u64 (pointee id == 1, NOT 0). A typed scalar
        // pointer: resolve_to_struct_id returns None (arm 1 miss),
        // and return_peels_to_ptr_void returns false because the
        // pointee id != 0 (arm 2 gate rejects).
        SynType::Ptr { type_id: 1 },
        // id 4: FuncProto returning id 3 (Ptr -> u64).
        SynType::FuncProto {
            return_type_id: 3,
            params: vec![],
        },
        // id 5: Func named the allowlisted name -> proto id 4.
        SynType::Func {
            name_off: n_func,
            type_id: 4,
            linkage: 2,
        },
    ];
    (build_btf(&types, &strings), 2, 5)
}

/// `handle_kfunc_call` combined drop: an allowlisted kfunc name
/// whose return peels to `Ptr -> Int` declines on BOTH arms. Arm 1
/// (`resolve_to_struct_id`) returns None because the pointee is an
/// Int, not a Struct. Arm 2's `return_peels_to_ptr_void` gate
/// returns false because the pointee id != 0 (it is the u64 id), so
/// the allowlist match never fires. R0 stays Unknown, no finding
/// emits.
///
/// Contrast: `kfunc_arena_alloc_typed_return_falls_through` uses
/// `Ptr -> Struct` (arm 1 fires -> Kernel finding);
/// `kfunc_arena_alloc_non_allowlist_name_drops` uses `Ptr -> Void`
/// with a non-allowlist name. This is the allowlisted-name +
/// `Ptr -> NonVoid-NonStruct` intersection that exercises
/// `return_peels_to_ptr_void`'s false-on-typed-pointee arm under an
/// allowlisted name.
#[test]
fn kfunc_allowlist_name_ptr_to_nonvoid_nonstruct_drops_r0() {
    let (blob, m_id, kfunc_id) = btf_arena_kfunc_ptr_to_int();
    let btf = Btf::from_bytes(&blob).unwrap();
    // R6 holds Pointer{M} (callee-saved, survives the call clobber),
    // so the STX through it has a typed parent: the only way
    // map.is_empty() could fail is if R0 was mistakenly typed.
    let kfunc = kfunc_call(kfunc_id);
    let insns = vec![kfunc, stx(BPF_SIZE_DW, 6, 0, 8), exit()];
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
        None,
        "allowlisted kfunc returning Ptr -> Int declines on both arms \
             (no struct, non-void pointee); R0 stays Unknown: {map:?}"
    );
    assert!(
        map.is_empty(),
        "Ptr -> Int return must record neither an Arena nor a kptr \
             finding: {map:?}"
    );
}

// ----- handle_stx StxValueKind::Unknown persistence -----------

/// `handle_stx` `StxValueKind::Unknown` arm for an
/// `arena_stx_findings` slot: a later DW store of an Unknown-typed
/// register to a slot ALREADY recorded as Arena via the STX-flow
/// path must NOT erase the finding (may-analysis persistence). The
/// arm is a deliberate no-op.
///
/// Sequence:
///   pseudo_call (SubprogReturn @ 0) -> R0 = ArenaU64FromAlloc
///   stx [r6 + 8] = r0   ; arena_stx_findings[(M, 8)] = Pending
///   stx [r6 + 8] = r5   ; r5 never set -> StxValueKind::Unknown,
///                         same key -> no-op
/// The (M, 8) Arena finding must SURVIVE the untyped store.
///
/// Distinct from `addr_space_cast_arena_survives_unknown_stx`, which
/// covers the `arena_confirmed` (shape-inference) survival path; this
/// pins the `arena_stx_findings` survival under a subsequent Unknown
/// STX to the same key.
#[test]
fn handle_stx_unknown_value_does_not_invalidate_prior_arena_stx_finding() {
    // BTF: u64(1), M(2, u64@8).
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_m = push_name(&mut strings, "M");
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
            name_off: n_m,
            size: 16,
            members: vec![SynMember {
                name_off: n_slot,
                type_id: 1,
                byte_offset: 8,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let m_id = 2;
    // R6 = Pointer{M} (callee-saved, survives the call clobber). R5
    // is never set, so it stays Unknown after the R0..=R5 clobber ->
    // the second STX takes the StxValueKind::Unknown arm.
    let pseudo_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    let insns = vec![
        pseudo_call,
        stx(BPF_SIZE_DW, 6, 0, 8),
        stx(BPF_SIZE_DW, 6, 5, 8),
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
        "a later Unknown-valued STX to the same slot must NOT erase the \
             prior arena_stx_findings Arena hit: {map:?}"
    );
}
