use super::*;

// -----------------------------------------------------------------
//
// Plain-helper return tests: bpf_map_lookup_elem (helper id 1)
//
// These tests exercise the analyzer's `BPF_OP_CALL` plain-helper
// arm. The arm fires when `src_reg == 0` (helper-call pseudo per
// linux uapi `bpf.h`), `imm == BPF_FUNC_map_lookup_elem` (== 1),
// and the saved-pre-clobber R1 was a [`RegState::DatasecPointer`]
// into a `BTF_KIND_DATASEC` named `.maps`. The analyzer types R0
// as `Pointer{value_struct_id}` only when the targeted map's BTF
// declaration carries a `value` member whose type peels to
// `Ptr -> Struct/Union`. Stat-counter maps and other shapes drop.
//
// Fixtures synthesise a `.maps` datasec with one map declaration
// matching the libbpf `parse_btf_map_def` shape (a per-map struct
// whose `value` member type is `typeof(T) *` per the
// `__type(value, T)` macro in `tools/lib/bpf/bpf_helpers.h`).

/// `BPF_CALL` for the plain-helper form: `code = BPF_JMP|BPF_CALL`,
/// `src_reg == 0` (helper, not pseudo), `imm == helper_id`. Mirrors
/// what clang emits for `bpf_map_lookup_elem(...)` and friends.
fn helper_call(helper_id: i32) -> BpfInsn {
    mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, 0, 0, helper_id)
}

/// Selector for the `value` member's underlying shape in the
/// synthetic `.maps` BTF fixture built by
/// [`btf_with_maps_and_task_ctx`].
///
/// - `Struct`: `Ptr -> struct cbw_cgrp_entry { u64 cgx @ 0 }`
///   (the canonical case — should resolve to the entry struct id).
/// - `U64`: `Ptr -> u64` (stat-counter map — should drop because
///   `resolve_to_struct_id` rejects a non-struct pointee).
/// - `Typedef`: `Ptr -> Typedef -> struct cbw_cgrp_entry`
///   (typedef-wrapped struct; resolution must peel the typedef).
/// - `Void`: `Ptr -> void` (`__type(value, void)` — not a real
///   libbpf shape but exercises the `resolve_to_struct_id`
///   `pointee == 0` reject path).
enum MapValueShape {
    Struct,
    U64,
    Typedef,
    #[allow(dead_code)] // Void shape is reserved for a follow-up
    // void-pointee assertion; the existing tests do not exercise
    // it directly because the U64 case already covers the
    // `resolve_to_struct_id` reject path.
    Void,
}

/// Happy path: `entry = bpf_map_lookup_elem(&map, &key)` types
/// R0 as `Pointer{cbw_cgrp_entry}`, then a follow-up STX of R0
/// into a u64 slot of an outer `Pointer{task_ctx}` records a
/// kernel kptr finding.
///
/// This is the load-bearing end-to-end test: it proves the
/// helper-return arm types R0 AND that the typed R0 flows into
/// the existing kptr STX path without any other plumbing.
/// Without Delta A landing this test would fail because R0
/// would stay Unknown after the call, and the STX would record
/// nothing.
///
/// Sequence (mirrors clang's `__always_inline`'d inline
/// `cbw_get_cgroup_ctx_raw` codegen):
///   r1 = LD_IMM64(.maps, 0)        ; r1 = DatasecPointer{maps,0}
///   r2 = key                       ; analyzer ignores R2 (the
///                                  ; ARG_PTR_TO_MAP_KEY is a
///                                  ; verifier-side gate, not a
///                                  ; cast-finding signal)
///   call helper(BPF_FUNC_map_lookup_elem)
///   *(u64 *)(r6 + 8) = r0          ; STX R0 into task_ctx.cgx_raw
///
/// R6 is seeded as `Pointer{task_ctx}` so the STX's parent base
/// resolves to a struct with a u64 field at offset 8. The
/// expected finding keys on `(task_ctx_id, 8) ->
/// (cbw_cgrp_entry_id, Kernel)`.
#[test]
fn helper_map_lookup_elem_typed_value_seeds_r0() {
    let (blob, datasec_id, var_off, value_sid, parent_id) =
        btf_with_maps_and_task_ctx(MapValueShape::Struct);
    let btf = Btf::from_bytes(&blob).unwrap();

    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    let mov_key = mov_k(2, 0); // R2 = key (Unknown — analyzer ignores)
    let call_lookup = helper_call(BPF_FUNC_MAP_LOOKUP_ELEM);
    // *(u64 *)(R6 + 8) = R0 — STX R0 into task_ctx.cgx_raw.
    let stx_kptr = stx(BPF_SIZE_DW, 6, 0, 8);

    let insns = vec![ld_lo, ld_hi, mov_key, call_lookup, stx_kptr, exit()];
    // PC numbering: 0=ld_lo, 1=ld_hi (skip), 2=mov_key,
    // 3=call_lookup, 4=stx_kptr, 5=exit. The DatasecPointer
    // marks PC=0 (the LD_IMM64 lo slot) as targeting `.maps`.
    let datasec_pointers = vec![DatasecPointer {
        insn_offset: 0,
        datasec_type_id: datasec_id,
        base_offset: var_off,
    }];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: parent_id,
        }],
        &[],
        &datasec_pointers,
        &[],
    );
    assert_eq!(
        map.get(&(parent_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: value_sid,
            addr_space: AddrSpace::Kernel,
        }),
        "lookup-derived R0 stored into task_ctx.cgx_raw must record \
             (task_ctx, 8) -> (cbw_cgrp_entry, Kernel): {map:?}"
    );
}

/// Stat-counter map shape: `__type(value, u64)` produces a
/// `value` member of type `Ptr -> u64`. `resolve_to_struct_id`
/// returns None (u64 is not a struct), so the helper-return
/// arm leaves R0 Unknown. The follow-up STX records nothing.
#[test]
fn helper_map_lookup_elem_value_type_unresolvable_keeps_r0_unknown() {
    let (blob, datasec_id, var_off, _value_sid, parent_id) =
        btf_with_maps_and_task_ctx(MapValueShape::U64);
    let btf = Btf::from_bytes(&blob).unwrap();
    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    let mov_key = mov_k(2, 0);
    let call_lookup = helper_call(BPF_FUNC_MAP_LOOKUP_ELEM);
    let stx_kptr = stx(BPF_SIZE_DW, 6, 0, 8);
    let insns = vec![ld_lo, ld_hi, mov_key, call_lookup, stx_kptr, exit()];
    let datasec_pointers = vec![DatasecPointer {
        insn_offset: 0,
        datasec_type_id: datasec_id,
        base_offset: var_off,
    }];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: parent_id,
        }],
        &[],
        &datasec_pointers,
        &[],
    );
    assert!(
        map.is_empty(),
        "stat-counter map (`__type(value, u64)`) must keep R0 Unknown \
             so the STX records nothing: {map:?}"
    );
}

#[test]
fn helper_map_lookup_elem_value_type_void_keeps_r0_unknown() {
    let (blob, datasec_id, var_off, _value_sid, parent_id) =
        btf_with_maps_and_task_ctx(MapValueShape::Void);
    let btf = Btf::from_bytes(&blob).unwrap();
    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    let mov_key = mov_k(2, 0);
    let call_lookup = helper_call(BPF_FUNC_MAP_LOOKUP_ELEM);
    let stx_kptr = stx(BPF_SIZE_DW, 6, 0, 8);
    let insns = vec![ld_lo, ld_hi, mov_key, call_lookup, stx_kptr, exit()];
    let datasec_pointers = vec![DatasecPointer {
        insn_offset: 0,
        datasec_type_id: datasec_id,
        base_offset: var_off,
    }];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: parent_id,
        }],
        &[],
        &datasec_pointers,
        &[],
    );
    assert!(
        map.is_empty(),
        "void-value map (`Ptr -> Void`) must keep R0 Unknown: {map:?}"
    );
}

/// Typedef-wrapped value type: `__type(value, cbw_cgrp_entry_t)`
/// where `typedef struct cbw_cgrp_entry cbw_cgrp_entry_t;`. The
/// analyzer must peel the typedef via `peel_modifiers` /
/// `resolve_to_struct_id` and resolve to the underlying struct id.
#[test]
fn helper_map_lookup_elem_value_type_struct_via_typedef() {
    let (blob, datasec_id, var_off, value_sid, parent_id) =
        btf_with_maps_and_task_ctx(MapValueShape::Typedef);
    let btf = Btf::from_bytes(&blob).unwrap();
    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    let mov_key = mov_k(2, 0);
    let call_lookup = helper_call(BPF_FUNC_MAP_LOOKUP_ELEM);
    let stx_kptr = stx(BPF_SIZE_DW, 6, 0, 8);
    let insns = vec![ld_lo, ld_hi, mov_key, call_lookup, stx_kptr, exit()];
    let datasec_pointers = vec![DatasecPointer {
        insn_offset: 0,
        datasec_type_id: datasec_id,
        base_offset: var_off,
    }];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: parent_id,
        }],
        &[],
        &datasec_pointers,
        &[],
    );
    assert_eq!(
        map.get(&(parent_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: value_sid,
            addr_space: AddrSpace::Kernel,
        }),
        "typedef-wrapped value type must peel to the underlying struct id: {map:?}"
    );
}

/// Missing-`.maps`-datasec case: the call site's R1 is Unknown
/// (no DatasecPointer annotation was emitted), so the
/// helper-return arm cannot resolve R0. Analogous to the
/// `ld_imm64_without_annotation_no_record` test for the kptr
/// path.
#[test]
fn helper_map_lookup_elem_no_map_metadata_keeps_r0_unknown() {
    let (blob, _datasec_id, var_off, _value_sid, parent_id) =
        btf_with_maps_and_task_ctx(MapValueShape::Struct);
    let btf = Btf::from_bytes(&blob).unwrap();
    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    let mov_key = mov_k(2, 0);
    let call_lookup = helper_call(BPF_FUNC_MAP_LOOKUP_ELEM);
    let stx_kptr = stx(BPF_SIZE_DW, 6, 0, 8);
    let insns = vec![ld_lo, ld_hi, mov_key, call_lookup, stx_kptr, exit()];
    // Empty datasec_pointers — analyzer leaves R1 Unknown
    // through the LD_IMM64; helper-return arm sees a non-
    // DatasecPointer and falls through.
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: parent_id,
        }],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "without DatasecPointer annotation R1 stays Unknown so the \
             helper-return arm cannot type R0: {map:?}"
    );
}

/// Helper allowlist gate: a non-`bpf_map_lookup_elem` helper id
/// (e.g. `BPF_FUNC_get_current_task = 35`) must NOT seed R0 even
/// when R1 is a valid `.maps` DatasecPointer. The arm keys on
/// `imm == BPF_FUNC_map_lookup_elem` exactly.
#[test]
fn helper_not_in_allowlist_keeps_r0_unknown() {
    let (blob, datasec_id, var_off, _value_sid, parent_id) =
        btf_with_maps_and_task_ctx(MapValueShape::Struct);
    let btf = Btf::from_bytes(&blob).unwrap();
    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    let mov_key = mov_k(2, 0);
    // BPF_FUNC_get_current_task = 35 per linux uapi `bpf.h`.
    // The arm rejects this id even though R1 is a valid `.maps`
    // descriptor.
    let call_other = helper_call(35);
    let stx_kptr = stx(BPF_SIZE_DW, 6, 0, 8);
    let insns = vec![ld_lo, ld_hi, mov_key, call_other, stx_kptr, exit()];
    let datasec_pointers = vec![DatasecPointer {
        insn_offset: 0,
        datasec_type_id: datasec_id,
        base_offset: var_off,
    }];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: parent_id,
        }],
        &[],
        &datasec_pointers,
        &[],
    );
    assert!(
        map.is_empty(),
        "non-bpf_map_lookup_elem helper must not type R0 even with \
             a valid `.maps` R1: {map:?}"
    );
}

/// Boundary check: `imm <= 0` (negative or zero) must drop. A
/// pre-relocation kfunc placeholder uses `imm == -1`, but those
/// carry `src_reg == BPF_PSEUDO_CALL` so the helper arm's
/// `pseudo == 0` gate already rejects them. This test exercises
/// the explicit boundary on a synthetic plain helper with
/// negative or zero imm — `BPF_FUNC_unspec == 0` and any
/// negative value must NOT match BPF_FUNC_map_lookup_elem (== 1).
#[test]
fn helper_imm_negative_or_zero_keeps_r0_unknown() {
    let (blob, datasec_id, var_off, _value_sid, parent_id) =
        btf_with_maps_and_task_ctx(MapValueShape::Struct);
    let btf = Btf::from_bytes(&blob).unwrap();
    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    let mov_key = mov_k(2, 0);
    let stx_kptr = stx(BPF_SIZE_DW, 6, 0, 8);
    let datasec_pointers = vec![DatasecPointer {
        insn_offset: 0,
        datasec_type_id: datasec_id,
        base_offset: var_off,
    }];
    // Try imm == 0 (BPF_FUNC_unspec).
    {
        let call_zero = helper_call(0);
        let insns = vec![ld_lo, ld_hi, mov_key, call_zero, stx_kptr, exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 6,
                struct_type_id: parent_id,
            }],
            &[],
            &datasec_pointers,
            &[],
        );
        assert!(
            map.is_empty(),
            "helper id 0 (BPF_FUNC_unspec) must not seed R0: {map:?}"
        );
    }
    // Try imm == -1 (placeholder shape with plain pseudo).
    {
        let call_neg = helper_call(-1);
        let insns = vec![ld_lo, ld_hi, mov_key, call_neg, stx_kptr, exit()];
        let map = analyze_casts(
            &insns,
            &btf,
            &[InitialReg {
                reg: 6,
                struct_type_id: parent_id,
            }],
            &[],
            &datasec_pointers,
            &[],
        );
        assert!(map.is_empty(), "helper id -1 must not seed R0: {map:?}");
    }
}

/// End-to-end: lookup → STX into typed slot → CastMap entry.
///
/// This is the load-bearing test for the user's stated goal: the
/// `bpf_map_lookup_elem` returned pointer must thread through the
/// existing kptr STX path and produce a CastMap entry the
/// renderer can chase. Distinct from
/// `helper_map_lookup_elem_typed_value_seeds_r0` (which uses the
/// same flow but is named for the seeding step) by adding a MOV
/// between the call and the STX, exercising the typed-pointer
/// propagation path the analyzer already supports.
#[test]
fn stx_through_helper_returned_pointer_records_finding() {
    let (blob, datasec_id, var_off, value_sid, parent_id) =
        btf_with_maps_and_task_ctx(MapValueShape::Struct);
    let btf = Btf::from_bytes(&blob).unwrap();
    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    let mov_key = mov_k(2, 0);
    let call_lookup = helper_call(BPF_FUNC_MAP_LOOKUP_ELEM);
    // r7 = r0 — propagate the typed pointer through a callee-
    // saved register before the STX. Verifies the typed-pointer
    // state survives an ALU64|MOV|X.
    let mov_r7 = mov_x(7, 0);
    let stx_kptr = stx(BPF_SIZE_DW, 6, 7, 8);
    let insns = vec![ld_lo, ld_hi, mov_key, call_lookup, mov_r7, stx_kptr, exit()];
    let datasec_pointers = vec![DatasecPointer {
        insn_offset: 0,
        datasec_type_id: datasec_id,
        base_offset: var_off,
    }];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: parent_id,
        }],
        &[],
        &datasec_pointers,
        &[],
    );
    assert_eq!(
        map.get(&(parent_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: value_sid,
            addr_space: AddrSpace::Kernel,
        }),
        "lookup -> mov -> stx into task_ctx.cgx_raw must record a kernel \
             cast finding: {map:?}"
    );
}

/// Build a BTF that includes a `.maps` datasec with a single
/// map declaration AND a separate `task_ctx { u64 cgx_raw @ 8 }`
/// parent struct used as the STX destination base for the
/// end-to-end tests above. Returns `(blob, datasec_id,
/// var_offset, value_struct_id, parent_struct_id)`.
///
/// `value_struct_id` is `0` when the resolution is expected to
/// drop ([`MapValueShape::U64`], [`MapValueShape::Void`]) — the
/// callers compare the analyzer's CastMap against an empty map
/// in those branches and never deref the id.
///
/// The parent struct is distinct from the map's value type so
/// the test's STX target is a different struct id, avoiding
/// the analyzer's self-store rejection in [`Analyzer::handle_stx`].
fn btf_with_maps_and_task_ctx(value_kind: MapValueShape) -> (Vec<u8>, u32, u32, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_entry = push_name(&mut strings, "cbw_cgrp_entry");
    let n_cgx = push_name(&mut strings, "cgx");
    let n_value = push_name(&mut strings, "value");
    let n_type = push_name(&mut strings, "type");
    let n_map_def = push_name(&mut strings, "anon_map_def");
    let n_map_var = push_name(&mut strings, "cbw_cgrp_map");
    let n_maps = push_name(&mut strings, ".maps");
    let n_entry_typedef = push_name(&mut strings, "cbw_cgrp_entry_t");
    let n_task_ctx = push_name(&mut strings, "task_ctx");
    let n_cgx_raw = push_name(&mut strings, "cgx_raw");

    let mut types = vec![
        // id 1: u64
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct cbw_cgrp_entry { u64 cgx @ 0 }
        SynType::Struct {
            name_off: n_entry,
            size: 8,
            members: vec![SynMember {
                name_off: n_cgx,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: struct task_ctx { u64 cgx_raw @ 8 } (size = 16
        // so the cgx_raw slot at offset 8 has a backing byte).
        SynType::Struct {
            name_off: n_task_ctx,
            size: 16,
            members: vec![SynMember {
                name_off: n_cgx_raw,
                type_id: 1,
                byte_offset: 8,
            }],
        },
    ];
    let parent_id = 3u32;
    let (value_ptr_id, expected_struct_id) = match value_kind {
        MapValueShape::Struct => {
            // id 4: Ptr -> id 2
            types.push(SynType::Ptr { type_id: 2 });
            (4u32, 2u32)
        }
        MapValueShape::U64 => {
            // id 4: Ptr -> id 1 (u64). resolve_to_struct_id
            // returns None on a non-struct pointee.
            types.push(SynType::Ptr { type_id: 1 });
            (4u32, 0u32)
        }
        MapValueShape::Typedef => {
            // id 4: Ptr -> id 5 (typedef cbw_cgrp_entry_t -> 2)
            // id 5: Typedef cbw_cgrp_entry_t -> 2
            types.push(SynType::Ptr { type_id: 5 });
            types.push(SynType::Typedef {
                name_off: n_entry_typedef,
                type_id: 2,
            });
            (4u32, 2u32)
        }
        MapValueShape::Void => {
            // id 4: Ptr -> 0 (BTF void marker).
            types.push(SynType::Ptr { type_id: 0 });
            (4u32, 0u32)
        }
    };
    let map_def_id = types.len() as u32 + 1;
    types.push(SynType::Struct {
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
                type_id: value_ptr_id,
                byte_offset: 8,
            },
        ],
    });
    let map_var_id = map_def_id + 1;
    types.push(SynType::Var {
        name_off: n_map_var,
        type_id: map_def_id,
        linkage: 1,
    });
    let datasec_id = map_var_id + 1;
    types.push(SynType::Datasec {
        name_off: n_maps,
        size: 16,
        entries: vec![SynVarSecinfo {
            type_id: map_var_id,
            offset: 0,
            size: 16,
        }],
    });
    let blob = build_btf(&types, &strings);
    (blob, datasec_id, 0, expected_struct_id, parent_id)
}

/// Datasec name gate: a non-`.maps` datasec must NOT drive the
/// helper-return arm even when the structural shape (Var ->
/// Struct -> Ptr -> Struct member named `value`) coincidentally
/// matches. Pins the strict `name == ".maps"` check. Without
/// this gate, a `.bss` global whose underlying type happened to
/// be a struct with a `value` member of pointer type would
/// silently drive a false-positive kptr finding.
///
/// This test reuses [`btf_with_maps_and_task_ctx`] but the
/// caller mis-identifies the datasec section name. We synthesize
/// a fresh BTF with the datasec named `.bss` instead.
#[test]
fn helper_map_lookup_elem_non_dot_maps_datasec_drops() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_entry = push_name(&mut strings, "cbw_cgrp_entry");
    let n_cgx = push_name(&mut strings, "cgx");
    let n_value = push_name(&mut strings, "value");
    let n_type = push_name(&mut strings, "type");
    let n_map_def = push_name(&mut strings, "anon_map_def");
    let n_map_var = push_name(&mut strings, "fake_map");
    let n_bss = push_name(&mut strings, ".bss");
    let n_task_ctx = push_name(&mut strings, "task_ctx");
    let n_cgx_raw = push_name(&mut strings, "cgx_raw");

    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Struct {
            name_off: n_entry,
            size: 8,
            members: vec![SynMember {
                name_off: n_cgx,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynType::Struct {
            name_off: n_task_ctx,
            size: 16,
            members: vec![SynMember {
                name_off: n_cgx_raw,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // id 4: Ptr -> id 2 (cbw_cgrp_entry)
        SynType::Ptr { type_id: 2 },
        // id 5: map-def-shaped struct in `.bss` (NOT `.maps`).
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
        // id 6: Var
        SynType::Var {
            name_off: n_map_var,
            type_id: 5,
            linkage: 1,
        },
        // id 7: Datasec ".bss" (NOT `.maps`)
        SynType::Datasec {
            name_off: n_bss,
            size: 16,
            entries: vec![SynVarSecinfo {
                type_id: 6,
                offset: 0,
                size: 16,
            }],
        },
    ];
    let parent_id = 3u32;
    let datasec_id = 7u32;
    let var_off = 0u32;
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    let mov_key = mov_k(2, 0);
    let call_lookup = helper_call(BPF_FUNC_MAP_LOOKUP_ELEM);
    let stx_kptr = stx(BPF_SIZE_DW, 6, 0, 8);
    let insns = vec![ld_lo, ld_hi, mov_key, call_lookup, stx_kptr, exit()];
    let datasec_pointers = vec![DatasecPointer {
        insn_offset: 0,
        datasec_type_id: datasec_id,
        base_offset: var_off,
    }];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: parent_id,
        }],
        &[],
        &datasec_pointers,
        &[],
    );
    assert!(
        map.is_empty(),
        "non-`.maps` datasec must not drive the helper-return arm even \
             with a structurally matching map-def shape: {map:?}"
    );
}

#[test]
fn empty_access_pattern_does_not_trigger_conflict_with_kptr() {
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![ldx(BPF_SIZE_DW, 3, 1, 8), stx(BPF_SIZE_DW, 1, 5, 8), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 1,
                struct_type_id: t_id,
            },
            InitialReg {
                reg: 5,
                struct_type_id: q_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert!(
        map.contains_key(&(t_id, 8)),
        "kptr finding on slot with empty-access pattern (LDX without deref) \
             must NOT be dropped by conflict detection: {map:?}"
    );
}

#[test]
fn only_ld_imm64_no_oob() {
    const N_PAIRS: usize = 50;
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let mut insns: Vec<BpfInsn> = Vec::with_capacity(2 * N_PAIRS + 1);
    let lo = mk_insn(BPF_CLASS_LD | BPF_SIZE_DW | BPF_MODE_IMM, 2, 0, 0, 0);
    let hi = mk_insn(0, 0, 0, 0, 0);
    for _ in 0..N_PAIRS {
        insns.push(lo);
        insns.push(hi);
    }
    insns.push(exit());
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
        "all-LD_IMM64 stream must produce no findings, no OOB panic: {map:?}"
    );
}

#[test]
fn arena_stx_pending_then_duplicate_is_idempotent() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let pseudo_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    let insns = vec![
        pseudo_call,
        stx(BPF_SIZE_DW, 6, 0, 8),
        stx(BPF_SIZE_DW, 6, 0, 8),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: t_id,
        }],
        &[],
        &[],
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 0,
        }],
    );
    assert!(
        !map.is_empty(),
        "duplicate STX to same slot must not conflict; map: {map:?}"
    );
}

#[test]
fn three_way_conflict_arena_kptr_pattern_drops_all() {
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let cast = addr_space_cast(3, 2, 1);
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        cast,
        stx(BPF_SIZE_DW, 1, 3, 8),
        stx(BPF_SIZE_DW, 1, 5, 8),
        ldx(BPF_SIZE_DW, 6, 1, 8),
        ldx(BPF_SIZE_DW, 7, 6, 0),
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
                reg: 5,
                struct_type_id: q_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "arena + kptr + pattern on same slot must all drop: {map:?}"
    );
}

/// Verify `struct_member_at` recognises offsets that land INSIDE
/// an array member as the array's element type. The peel detects
/// `Type::Array(arr)`, computes `arr.len()` × `elem_size` to bound
/// `byte_offset` relative to `member_off`, and returns
/// `MemberAt::Struct { member_type_id: elem_tid }` for queried
/// offsets that fall within the array's bytes AND are
/// `elem_size`-aligned, so the LDX / STX paths see the element's
/// BTF type rather than the array's.
///
/// Pattern: `struct T { u64 history[4] @ 0 }`. The third element
/// (`history[2]`) sits at byte offset 16. The BPF program loads
/// `history[2]` through the typed pointer in r1, casts the loaded
/// u64 through `bpf_addr_space_cast(off=1, imm=1)` so the source
/// slot tags `arena_confirmed`, stores the cast result back into
/// the same slot, then dereferences the cast result with a u64
/// load at offset 0:
///
/// ```text
/// r2 = *(u64 *)(r1 + 16)              -- load history[2]
/// r4 = bpf_addr_space_cast(r2, 0, 1)  -- arena_confirmed += (T, 16)
/// *(u64 *)(r1 + 16) = r4              -- STX back into the same slot
/// r3 = *(u64 *)(r4 + 0)               -- deref the cast result
/// ```
///
/// The first LDX is the array-recognition exercise: with r1 typed
/// `Pointer{T}`, `struct_member_at(btf, T_id, 16)` walks T's
/// single member at offset 0, peels its type to
/// `Type::Array(u64; 4)`, computes `elem_size = 8 × nelems = 4 ⇒
/// arr_byte_size = 32`, checks `rel = 16 < 32` AND `rel %
/// elem_size == 0`, and returns
/// `MemberAt::Struct { member_type_id = u64_id }`. The Plain u64
/// field arm in `handle_ldx` then tags r2 as
/// `LoadedU64Field { source_struct_id: T_id, field_offset: 16 }`
/// — `canonical_field_off = 16` because `MemberAt::Struct` keys
/// on the queried offset directly.
///
/// `addr_space_cast(4, 2, 1)` propagates r2's state into r4 and
/// inserts `(T_id, 16)` into `arena_confirmed` (the arena-evidence
/// gate's "direct evidence" channel). The STX-back of r4 mirrors
/// real schedulers' arena-pointer write-back idiom; its src is
/// `LoadedU64Field` (cast propagates state verbatim) so the
/// production STX path's value-state arm does not match
/// `Pointer{T}` / `ArenaU64FromAlloc` and the store has no
/// side-effect on the analyzer state — `addr_space_cast_arena_alone_does_not_emit`
/// pins this exact "cast alone produces no map entry" property.
///
/// The trailing `r3 = *(u64 *)(r4 + 0)` records an `Access {
/// offset: 0, size: 8 }` on `patterns[(T_id, 16)]` (because r4 is
/// `LoadedU64Field`, the LDX-through-`LoadedU64Field` arm is the
/// only path that adds accesses). Combined with the
/// `arena_confirmed` evidence, `finalize`'s shape-inference loop
/// intersects the BTF layout for `(off=0, size=8)`: only
/// `Q { u64 @ 0 }` matches that shape, so the arena-evidence-gated emit fires
/// with `target_type_id = Q`. The presence of `(T_id, 16) →
/// CastHit { alloc_size: None, Q_id, Arena }` in the resulting `CastMap` is the
/// witness that the array peel produced
/// `MemberAt::Struct { member_type_id = u64_id }`: without it the
/// LDX would have dropped r2 to `Unknown`, neither the cast nor
/// the deref would have keyed `(T_id, 16)`, and the assertion
/// would fail.
#[test]
fn struct_member_at_resolves_array_element_offset() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q = push_name(&mut strings, "Q");
    let n_history = push_name(&mut strings, "history");
    let n_x = push_name(&mut strings, "x");

    // id 1: u64 (size=8, bits=64). Doubles as element type for the
    //       array AND index-type stand-in — the analyzer never
    //       inspects the index type, and `Array::index_type` is
    //       not consulted along the `struct_member_at` path under
    //       test, so reusing id 1 keeps the BTF minimal without
    //       changing behaviour.
    // id 2: u64[4] — 32-byte total array-of-u64. The new
    //       `struct_member_at` array arm peels this when the
    //       queried offset lands inside the array's bytes.
    // id 3: struct T { u64 history[4] @ 0 } — size 32. Single
    //       member that the LDX walks; the queried offset 16
    //       falls into the array's range and the test verifies
    //       it resolves to the u64 element type.
    // id 4: struct Q { u64 x @ 0 } — size 8. Unique-shape target
    //       so the arena-evidence-gated shape inference resolves the
    //       cast-confirmed slot to a single candidate. T's only
    //       member is the array (size 32), so T is not a
    //       candidate for the (0, 8) access shape and the
    //       intersection collapses to {Q} alone.
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Array {
            type_id: 1,
            index_type_id: 1,
            nelems: 4,
        },
        SynType::Struct {
            name_off: n_t,
            size: 32,
            members: vec![SynMember {
                name_off: n_history,
                type_id: 2,
                byte_offset: 0,
            }],
        },
        SynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 3u32;
    let q_id = 4u32;

    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 16),
        addr_space_cast(4, 2, 1),
        stx(BPF_SIZE_DW, 1, 4, 16),
        ldx(BPF_SIZE_DW, 3, 4, 0),
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
        map.get(&(t_id, 16)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "(T={t_id}, 16) — third u64 element of `history[4]` — must \
         appear in the cast map with target=Q ({q_id}) after \
         struct_member_at peels the array member type to `u64`: \
         {map:?}"
    );
}

#[test]
fn stx_nested_struct_arena_finding_keys_on_inner() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_inner = push_name(&mut strings, "Inner");
    let n_cgx = push_name(&mut strings, "cgx_raw");
    let n_llcx = push_name(&mut strings, "llcx_raw");
    let n_outer = push_name(&mut strings, "Outer");
    let n_inner_field = push_name(&mut strings, "inner");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Struct {
            name_off: n_inner,
            size: 16,
            members: vec![
                SynMember {
                    name_off: n_cgx,
                    type_id: 1,
                    byte_offset: 0,
                },
                SynMember {
                    name_off: n_llcx,
                    type_id: 1,
                    byte_offset: 8,
                },
            ],
        },
        SynType::Struct {
            name_off: n_outer,
            size: 16,
            members: vec![SynMember {
                name_off: n_inner_field,
                type_id: 2,
                byte_offset: 0,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let inner_id: u32 = 2;
    let outer_id: u32 = 3;
    let pseudo_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    let insns = vec![
        pseudo_call,
        stx(BPF_SIZE_DW, 6, 0, 0),
        mov_x(7, 0),
        stx(BPF_SIZE_DW, 6, 7, 8),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: outer_id,
        }],
        &[],
        &[],
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 0,
        }],
    );
    assert_eq!(
        map.get(&(inner_id, 0)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "nested struct STX must key on (Inner, 0) not (Outer, 0): {map:?}"
    );
    assert_eq!(
        map.get(&(inner_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "nested struct STX must key on (Inner, 8) not (Outer, 8): {map:?}"
    );
    assert!(
        !map.contains_key(&(outer_id, 0)),
        "outer struct id must NOT appear as key: {map:?}"
    );
    assert!(
        !map.contains_key(&(outer_id, 8)),
        "outer struct id must NOT appear as key: {map:?}"
    );
}

#[test]
fn ldx_nested_struct_loads_inner_key() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_inner = push_name(&mut strings, "Inner");
    let n_field = push_name(&mut strings, "ptr_field");
    let n_outer = push_name(&mut strings, "Outer");
    let n_embed = push_name(&mut strings, "embed");
    let n_target = push_name(&mut strings, "Target");
    let n_x = push_name(&mut strings, "x");
    let types = vec![
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Struct {
            name_off: n_inner,
            size: 16,
            members: vec![SynMember {
                name_off: n_field,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        SynType::Struct {
            name_off: n_outer,
            size: 16,
            members: vec![SynMember {
                name_off: n_embed,
                type_id: 2,
                byte_offset: 0,
            }],
        },
        SynType::Struct {
            name_off: n_target,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let inner_id: u32 = 2;
    let outer_id: u32 = 3;
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
            struct_type_id: outer_id,
        }],
        &[],
        &[],
        &[],
    );
    assert!(
        map.contains_key(&(inner_id, 8)),
        "nested LDX + deref must key on (Inner={inner_id}, 8) \
         not (Outer={outer_id}, 8): {map:?}"
    );
    assert!(
        !map.contains_key(&(outer_id, 8)),
        "outer id must NOT appear as key for nested member: {map:?}"
    );
}

#[test]
fn cross_function_u64_param_inherits_caller_pointer_type() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_m = push_name(&mut strings, "M");
    let n_cgx = push_name(&mut strings, "cgx_raw");
    let n_caller = push_name(&mut strings, "caller");
    let n_callee = push_name(&mut strings, "callee");
    let n_taskc_raw = push_name(&mut strings, "taskc_raw");
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
        SynType::FuncProto {
            return_type_id: 0,
            params: vec![SynParam {
                name_off: n_taskc_raw,
                type_id: 1,
            }],
        },
        SynType::Func {
            name_off: n_callee,
            type_id: 3,
            linkage: 1,
        },
        SynType::Func {
            name_off: n_caller,
            type_id: 3,
            linkage: 1,
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let m_id = 2;
    // Caller at PC 0..2: R1 = Pointer{M}, BPF_PSEUDO_CALL to callee at PC 4.
    // Callee at PC 4..6 (func_entry): FuncProto says param 0 is u64,
    // but caller had R1 = Pointer{M}. Cross-function propagation
    // should type R1 as Pointer{M} in the callee. Then:
    // PC 4: (func entry, R1 = Pointer{M} via caller propagation)
    // PC 5: STX [R1 + 8] = R6 (R6 = ArenaU64FromAlloc from seed)
    // PC 6: EXIT
    let insns = vec![
        // PC 0: caller func entry (via FuncEntry at pc=0)
        // R1 is seeded as Pointer{M} from InitialReg
        mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 2),
        // PC 1: EXIT
        exit(),
        // PC 2: padding (unreachable)
        exit(),
        // PC 3: callee func entry (via FuncEntry at pc=3)
        // R1 typed as Pointer{M} via cross-function propagation
        // Save R1 to R6 before inner call clobbers R1
        mov_x(6, 1),
        // PC 4: inner BPF_PSEUDO_CALL (allocator) → R0 = ArenaU64FromAlloc
        mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0),
        // PC 5: STX [R6 + 8] = R0 (store arena alloc return to
        //        the u64 param's struct field via callee-saved R6)
        stx(BPF_SIZE_DW, 6, 0, 8),
        // PC 6: EXIT
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: m_id,
        }],
        &[FuncEntry {
            insn_offset: 3,
            func_proto_id: 3,
        }],
        &[],
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 4,
        }],
    );
    assert_eq!(
        map.get(&(m_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "cross-function u64 param must inherit caller's Pointer{{M}} \
         and record arena STX at (M, 8): {map:?}"
    );
}

/// Map-value arena propagation: an arena pointer stashed into a
/// `u64` field of a stack-local map-value struct V at
/// `bpf_map_update_elem(map, key, &V)` must round-trip through the
/// map so that the matching `bpf_map_lookup_elem(map)` returned R0
/// carries V-typed arena evidence on the same field. A subsequent
/// `LDX r2 = *(u64 *)(r0 + 8)` must therefore tag r2 as
/// [`RegState::ArenaU64FromAlloc`], and a follow-up
/// `STX *(u64 *)(r6 + 0) = r2` into a typed `Pointer{P}` parent
/// must record `(P, 0) -> Arena` via the existing
/// `arena_stx_findings` finalize path.
///
/// This is the load-bearing contract for the map-value arena
/// propagation feature: scx schedulers stash `scx_static_alloc` /
/// `scx_alloc_internal` returns inside map-value structs so the
/// arena pointer survives across BPF program invocations. The
/// renderer chases the recovered pointer from a parent struct
/// field that received the lookup-loaded value — without the
/// round-trip propagation the field renders as a raw u64 counter
/// even though the runtime byte pattern is an arena VA.
///
/// Synthesized program (clang-style codegen for the
/// "stack-local V, update_elem(&V), then lookup_elem to read V
/// back" pattern):
///
/// ```text
///   pc 0: pseudo_call          ; SubprogReturn @ pc=0 -> R0 = ArenaU64FromAlloc
///   pc 1: stx [r10 + (-16)] = r0
///                              ; spill arena value at the V.field
///                              ; offset within the stack-local V
///                              ; (V base lives at r10-24, V.field
///                              ; at +8 -> r10-16)
///   pc 2-3: ld_imm64 r1, .maps@0
///                              ; r1 = DatasecPointer{maps, 0}
///   pc 4: r2 = r10             ; key pointer (content irrelevant)
///   pc 5: r2 += -24
///   pc 6: r3 = r10             ; value pointer = &V on the stack
///   pc 7: r3 += -24
///   pc 8: call helper(BPF_FUNC_map_update_elem)
///                              ; FUTURE FEATURE: walks V's u64
///                              ; fields, sees stack_slot[-16] is
///                              ; ArenaU64FromAlloc, tags map's
///                              ; (V, 8) as arena-backed.
///   pc 9-10: ld_imm64 r1, .maps@0
///   pc 11: call helper(BPF_FUNC_map_lookup_elem)
///                              ; R0 = Pointer{V}.
///   pc 12: ldx r2 = *(u64 *)(r0 + 8)
///                              ; FUTURE FEATURE: V.field is the
///                              ; map's tagged arena field, so r2
///                              ; takes ArenaU64FromAlloc instead
///                              ; of LoadedU64Field{V, 8}.
///   pc 13: stx *(u64 *)(r6 + 0) = r2
///                              ; records (P, 0) -> Arena via the
///                              ; arena_stx_findings path.
///   pc 14: exit
/// ```
///
/// R6 is seeded `Pointer{P}` via [`InitialReg`]; R6 is callee-
/// saved per the BPF ABI so it survives both helper-call clobbers.
/// The map's value type V is distinct from the parent struct P so
/// the analyzer's self-store rejection in [`Analyzer::handle_stx`]
/// does not fire — the recorded finding is `parent=P, target=V`'s
/// value, not `P -> P`.
///
/// Helper id 2 is `BPF_FUNC_map_update_elem` per linux uapi
/// `bpf.h` (`FN(map_update_elem, 2, ##ctx)`); helper id 1 is
/// `BPF_FUNC_map_lookup_elem`. Both are plain helpers (`src_reg
/// == 0`). The numeric literal 2 is used here directly because
/// the analyzer does not yet expose a `BPF_FUNC_MAP_UPDATE_ELEM`
/// constant — that constant is part of the feature this test
/// guards.
#[test]
fn helper_map_update_then_lookup_propagates_arena_through_map_value() {
    // BTF: u64(1), V(2, u64@8), P(3, u64@0), Ptr->V(4),
    // map_def(5, type@0 + V*@8), Var "the_map"(6),
    // Datasec ".maps"(7).
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_v = push_name(&mut strings, "V");
    let n_v_field = push_name(&mut strings, "field");
    let n_p = push_name(&mut strings, "P");
    let n_p_field = push_name(&mut strings, "field");
    let n_type = push_name(&mut strings, "type");
    let n_value = push_name(&mut strings, "value");
    let n_map_def = push_name(&mut strings, "anon_map_def");
    let n_map_var = push_name(&mut strings, "the_map");
    let n_maps = push_name(&mut strings, ".maps");
    let types = vec![
        // id 1: u64 (size=8, bits=64).
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct V { u64 field @ 8 }, size=16. The map's
        // value type — V.field at offset 8 is the slot the
        // pre-update STX spills the arena value into.
        SynType::Struct {
            name_off: n_v,
            size: 16,
            members: vec![SynMember {
                name_off: n_v_field,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // id 3: struct P { u64 field @ 0 }, size=8. The parent
        // struct that receives the post-lookup arena STX —
        // distinct from V so the analyzer's self-store rejection
        // in handle_stx does not fire.
        SynType::Struct {
            name_off: n_p,
            size: 8,
            members: vec![SynMember {
                name_off: n_p_field,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 4: Ptr -> V. The map_def's `value` member type per
        // libbpf's `__type(value, V)` macro expansion (`typeof(V) *`).
        SynType::Ptr { type_id: 2 },
        // id 5: anonymous map_def struct {
        //   u32 type @ 0;  /* placeholder for __uint(type, ...) */
        //   V *value @ 8;
        // } size=16. Mirrors the per-map struct shape libbpf's
        // `parse_btf_map_def` consumes.
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
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let v_id = 2u32;
    let p_id = 3u32;
    let datasec_id = 7u32;
    let var_off = 0u32;

    let pseudo_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    let stx_spill = stx(BPF_SIZE_DW, 10, 0, -16);
    let [ld_lo_pre, ld_hi_pre] = ld_imm64(1, var_off as i32);
    let mov_r2_from_r10 = mov_x(2, 10);
    // R2 += -24 (ALU64 ADD K). Encoded as
    // `BPF_CLASS_ALU64 | BPF_ADD | BPF_SRC_K` per linux uapi
    // `bpf.h`; src field is unused, imm carries the constant.
    let r2_minus_24 = mk_insn(BPF_CLASS_ALU64 | BPF_OP_ADD, 2, 0, 0, -24);
    let mov_r3_from_r10 = mov_x(3, 10);
    let r3_minus_24 = mk_insn(BPF_CLASS_ALU64 | BPF_OP_ADD, 3, 0, 0, -24);
    // BPF_FUNC_map_update_elem == 2 per linux uapi `bpf.h`
    // (`FN(map_update_elem, 2, ...)`). The analyzer does not yet
    // export a BPF_FUNC_MAP_UPDATE_ELEM constant — that addition
    // is part of the feature this test guards.
    let call_update = helper_call(2);
    let [ld_lo_post, ld_hi_post] = ld_imm64(1, var_off as i32);
    let call_lookup = helper_call(BPF_FUNC_MAP_LOOKUP_ELEM);
    let ldx_v_field = ldx(BPF_SIZE_DW, 2, 0, 8);
    let stx_into_p = stx(BPF_SIZE_DW, 6, 2, 0);

    let insns = vec![
        // pc 0: pseudo_call -> R0 = ArenaU64FromAlloc.
        pseudo_call,
        // pc 1: spill R0 to stack at the V.field offset.
        stx_spill,
        // pc 2-3: ld_imm64 r1, .maps@0 (DatasecPointer for update).
        ld_lo_pre,
        ld_hi_pre,
        // pc 4-5: r2 = r10 + (-24) (key pointer; content
        // irrelevant — `bpf_map_update_elem`'s key arg is a
        // verifier-side gate, not a cast-finding signal).
        mov_r2_from_r10,
        r2_minus_24,
        // pc 6-7: r3 = r10 + (-24) (value pointer = &V on stack).
        mov_r3_from_r10,
        r3_minus_24,
        // pc 8: call bpf_map_update_elem.
        call_update,
        // pc 9-10: ld_imm64 r1, .maps@0 again (the post-update
        // R1 was clobbered by the helper call — pre-relocation
        // bytecode reloads the descriptor).
        ld_lo_post,
        ld_hi_post,
        // pc 11: call bpf_map_lookup_elem -> R0 = Pointer{V}.
        call_lookup,
        // pc 12: ldx r2 = *(u64 *)(r0 + 8) — load V.field.
        ldx_v_field,
        // pc 13: stx *(u64 *)(r6 + 0) = r2 — store into P.field.
        stx_into_p,
        // pc 14: exit.
        exit(),
    ];
    // PC layout summary (LD_IMM64 second slots are skipped via
    // skip_next; numbering above counts every BpfInsn slot):
    //   0 pseudo_call
    //   1 stx [r10-16] = r0
    //   2 ld_lo_pre, 3 ld_hi_pre
    //   4 mov r2,r10, 5 r2 += -24
    //   6 mov r3,r10, 7 r3 += -24
    //   8 call(2)        -- bpf_map_update_elem
    //   9 ld_lo_post, 10 ld_hi_post
    //   11 call(1)       -- bpf_map_lookup_elem
    //   12 ldx r2, [r0+8]
    //   13 stx [r6+0] = r2
    //   14 exit
    let datasec_pointers = vec![
        DatasecPointer {
            insn_offset: 2,
            datasec_type_id: datasec_id,
            base_offset: var_off,
        },
        DatasecPointer {
            insn_offset: 9,
            datasec_type_id: datasec_id,
            base_offset: var_off,
        },
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 6,
            struct_type_id: p_id,
        }],
        &[],
        &datasec_pointers,
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 0,
        }],
    );
    // The end-to-end contract: the post-lookup LDX of V.field
    // produces R2 = ArenaU64FromAlloc (because the prior
    // update_elem tagged V's u64@8 slot as arena-backed via the
    // stack-spilled value), and the subsequent STX into P.field
    // records (P, 0) -> Arena via the arena_stx_findings finalize
    // path. `target_type_id == 0` because the renderer's
    // `MemReader::resolve_arena_type` bridge supplies the actual
    // payload BTF id at chase time — same convention every
    // arena-STX finding uses (see
    // `stx_flow_alloc_return_records_arena_finding`).
    //
    // Until the map-value propagation feature lands, this test
    // fails — the analyzer treats `bpf_map_update_elem` as an
    // ordinary helper that clobbers R0..=R5 without inspecting
    // R3's pointee, so V's (2, 8) slot never receives the arena
    // tag, the post-lookup LDX produces `LoadedU64Field{V, 8}`
    // instead of `ArenaU64FromAlloc`, and the final STX records
    // nothing in `arena_stx_findings`. Reference is V's id
    // (2) for the Var var_off==0 → the_map → V layout.
    let _ = v_id;
    assert_eq!(
        map.get(&(p_id, 0)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "map-value arena propagation must surface `(P, 0) -> Arena` \
         after update_elem(&V_with_arena_at_off_8) -> lookup_elem -> \
         LDX V.field -> STX into P.field: {map:?}"
    );
}

/// Cross-subprog fixpoint: callee at LOWER PC than caller. Single-pass
/// misses this because caller_arg_types isn't populated when the callee's
/// FuncEntry runs. The fixpoint carries caller_arg_types from pass 1 into
/// pass 2 so the callee inherits the caller's typed arguments.
///
/// Pattern mirrors scx_cgroup_bw_consume (callee, PC 8064) called from
/// account_task_runtime (caller, PC 18292) in lavd — the exact chain
/// that required the fixpoint to detect cgx_raw as an arena pointer.
#[test]
fn cross_function_fixpoint_callee_before_caller() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_parent = push_name(&mut strings, "Parent");
    let n_field = push_name(&mut strings, "arena_field");
    let n_caller = push_name(&mut strings, "caller");
    let n_callee = push_name(&mut strings, "callee");
    let n_p1 = push_name(&mut strings, "parent_raw");
    let n_p2 = push_name(&mut strings, "val_raw");

    let types = vec![
        // id 1: u64
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct Parent { u64 arena_field @ offset 8 }
        SynType::Struct {
            name_off: n_parent,
            size: 16,
            members: vec![SynMember {
                name_off: n_field,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // id 3: Ptr -> Parent
        SynType::Ptr { type_id: 2 },
        // id 4: callee FuncProto: (u64 parent_raw, u64 val_raw)
        SynType::FuncProto {
            return_type_id: 0,
            params: vec![
                SynParam {
                    name_off: n_p1,
                    type_id: 1,
                },
                SynParam {
                    name_off: n_p2,
                    type_id: 1,
                },
            ],
        },
        // id 5: Func "callee" -> proto 4
        SynType::Func {
            name_off: n_callee,
            type_id: 4,
            linkage: 1,
        },
        // id 6: caller FuncProto: (Ptr->Parent)
        SynType::FuncProto {
            return_type_id: 0,
            params: vec![SynParam {
                name_off: n_p1,
                type_id: 3,
            }],
        },
        // id 7: Func "caller" -> proto 6
        SynType::Func {
            name_off: n_caller,
            type_id: 6,
            linkage: 1,
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let parent_id: u32 = 2;

    // Callee at PC 0..1 (BEFORE caller — exercises fixpoint):
    //   PC 0: STX [R1 + 8] = R2
    //         R1 = Pointer{Parent} (from caller_arg_types, pass 2)
    //         R2 = ArenaU64FromAlloc (from caller_arg_types, pass 2)
    //   PC 1: EXIT
    //
    // Caller at PC 2..5 (AFTER callee):
    //   PC 2: FuncEntry. R1 = Pointer{Parent} from FuncProto.
    //   PC 2: allocator call → R0 = ArenaU64FromAlloc
    //   PC 3: R2 = R0 (move arena result to R2 for callee arg)
    //   PC 4: BPF_PSEUDO_CALL to callee (PC 0). imm = -5.
    //         caller_arg_types[0] = [Pointer{Parent}, ArenaU64FromAlloc, ...]
    //   PC 5: EXIT
    let alloc_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    let callee_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, -7);

    let insns = vec![
        // --- callee at PC 0..1 ---
        stx(BPF_SIZE_DW, 1, 2, 8), // PC 0: STX [R1+8] = R2
        exit(),                    // PC 1
        // --- caller at PC 2..7 ---
        mov_x(6, 1), // PC 2: save R1 to callee-saved R6
        alloc_call,  // PC 3: allocator → R0=ArenaU64FromAlloc
        mov_x(2, 0), // PC 4: R2 = R0 (arena result)
        mov_x(1, 6), // PC 5: R1 = R6 (restore parent ptr)
        callee_call, // PC 6: call callee at PC 0
        exit(),      // PC 7
    ];

    let map = analyze_casts(
        &insns,
        &btf,
        &[],
        &[
            FuncEntry {
                insn_offset: 0,
                func_proto_id: 4,
            }, // callee (u64, u64)
            FuncEntry {
                insn_offset: 2,
                func_proto_id: 6,
            }, // caller (Ptr->Parent)
        ],
        &[],
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 3,
        }], // allocator at PC 3
    );

    assert_eq!(
        map.get(&(parent_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "fixpoint must propagate caller's [Pointer{{Parent}}, ArenaU64FromAlloc] \
         into callee at lower PC, producing (Parent, 8) -> Arena: {map:?}"
    );
}

// -- finalize-order tests ----------------------------------------
//
// `Analyzer::finalize` runs the arena_stx loop (cast_analysis/mod.rs:2672)
// BEFORE the standalone shape-inference loop (cast_analysis/mod.rs:2746).
// The standalone loop's `if out.contains_key(&key) { continue; }`
// gate at line 2759 prevents it from overwriting an arena_stx
// emission for the same key. The internal shape-inference inside
// the arena_stx loop (lines 2685-2710) computes the same candidate
// intersection, so the two paths produce the same target_type_id
// when shape inference is unique.
//
// The bug surface for the ambiguous-shape fix is
// the AMBIGUOUS-shape case: when shape inference yields 0 or 2+
// candidates, the arena_stx loop emits with target_type_id=0
// (deferred resolve sentinel) AND the standalone loop's gate
// preserves that emission. Without this test, a refactor that
// swapped the loop order (running shape-inference first) would
// drop the deferred-resolve entry entirely for ambiguous-shape
// slots — every cgx_raw / llcx_raw chase would silently miss
// because the analyzer never emitted the slot.

/// Ambiguous shape preserves arena_stx deferred-resolve emit.
///
/// BTF: u64(1), P(2, u64@8 source) parent struct, two structs Q
/// and R both 16 bytes with `u64@0 + u64@8` shape. Shape inference
/// on accesses (0, 8) and (8, 8) intersects to {Q, R} — TWO
/// candidates, ambiguous. The standalone shape-inference loop drops
/// the slot per cast_analysis/mod.rs:2810-2811 ("0 or 2+ candidates
/// -> drop silently"). The arena_stx loop's internal inference at
/// 2685-2709 also yields `inferred_target = None`, so it falls back
/// to `target_type_id = 0` per line 2710 (`unwrap_or(0)`). The map
/// must contain `(P, 8) -> CastHit { alloc_size: None, target_type_id: 0,
/// addr_space: Arena }` — the deferred-resolve sentinel — so the
/// renderer's `chase_arena_pointer` special case (btf_render/mod.rs:3392)
/// can call `MemReader::resolve_arena_type` at chase time.
///
/// Pin both:
///   (a) the entry IS present (arena_stx loop emitted it), and
///   (b) target_type_id == 0 (shape inference did not resolve).
///
/// A regression that ran shape-inference FIRST would drop this slot
/// entirely (no entry in `out`), and the chase-arm `MemReader::
/// resolve_arena_type` bridge would have nothing to resolve against.
/// A regression that emitted with one of {Q, R} arbitrarily would
/// produce a wrong-type render at chase time.
#[test]
fn finalize_arena_stx_emits_deferred_resolve_when_shape_inference_ambiguous() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_p = push_name(&mut strings, "P");
    let n_q = push_name(&mut strings, "Q");
    let n_r = push_name(&mut strings, "R");
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
        // id 2: P { u64 f @ 8 } — parent struct, slot at offset 8.
        SynType::Struct {
            name_off: n_p,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // id 3: Q { u64 a @ 0; u64 b @ 8 } — candidate A.
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
        // id 4: R { u64 a @ 0; u64 b @ 8 } — candidate B with same
        // shape. Two candidates collide on the access pattern; shape
        // inference yields {Q, R} — ambiguous.
        SynType::Struct {
            name_off: n_r,
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
    let pseudo_call = mk_insn(BPF_CLASS_JMP | BPF_OP_CALL, 0, BPF_PSEUDO_CALL, 0, 0);
    // Shape-inference accesses (0, 8) and (8, 8) on (P, 8) plus
    // an STX-flow tag on (P, 8) — both Q and R match the shape.
    //
    //   r2 = LDX[r1 + 8]   ; LoadedU64Field{P, 8}
    //   r3 = LDX[r2 + 0]   ; records access (0, 8) into patterns[(P, 8)]
    //   r4 = LDX[r2 + 8]   ; records access (8, 8) into patterns[(P, 8)]
    //   r6 = r1            ; preserve P* across the call (R6 callee-saved)
    //   pseudo_call (PC 4) ; SubprogReturn at PC=4 → R0 = ArenaU64FromAlloc
    //   STX[r6 + 8] = r0   ; arena_stx_findings.insert((P, 8), Pending)
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        ldx(BPF_SIZE_DW, 3, 2, 0),
        ldx(BPF_SIZE_DW, 4, 2, 8),
        mov_x(6, 1),
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
    // The slot MUST be emitted. Drop = silent failure — chase_arena_pointer
    // would never enter the deferred-resolve special case for this slot.
    let hit = map
        .get(&(p_id, 8))
        .expect("(P, 8) must be in CastMap even when shape inference is ambiguous");
    assert_eq!(
        hit.addr_space,
        AddrSpace::Arena,
        "ambiguous-shape STX-flow tag must still emit AddrSpace::Arena: {map:?}"
    );
    assert_eq!(
        hit.target_type_id, 0,
        "ambiguous shape (Q and R both 16-byte u64@0+u64@8) must yield \
         target_type_id=0 (deferred resolve via MemReader::resolve_arena_type \
         bridge at chase time). target_type_id={} suggests one of Q/R was \
         picked arbitrarily — false-positive render of the wrong struct \
         shape: {map:?}",
        hit.target_type_id
    );
    // Cross-check: assert the picked id is NOT one of the candidate
    // structs. A regression that picked Q or R would set
    // target_type_id to 3 or 4.
    assert!(
        hit.target_type_id != 3 && hit.target_type_id != 4,
        "target_type_id={} must NOT be one of the ambiguous candidates Q (3) or R (4); \
         picking one arbitrarily would render the slot's payload against the wrong \
         struct shape at chase time: {map:?}",
        hit.target_type_id
    );
}

/// Order-independence: when the STX-flow tag for a slot
/// fires BEFORE the dereference pattern through the same slot, the
/// dereference accesses must still flow into shape inference. The
/// failing pre-fix behaviour was: pseudo_call + STX(slot) tagged
/// the slot in pass 1; pass 2's LDX off the slot inherited
/// `RegState::ArenaU64FromAlloc` via alias-tracking; downstream
/// LDXs through the alias-tagged register dropped accesses (the
/// arena arm in `handle_ldx` set dst to Unknown without populating
/// patterns), so shape inference saw an empty access set even
/// though the access pattern uniquely identified the target
/// struct.
///
/// Post-fix the alias-tagged variant carries the source slot
/// identity (`source: Some((parent, off))`) and the LDX arm
/// records the access against that slot. Shape inference at
/// finalize sees the access pattern and resolves
/// `target_type_id` to the unique BTF id when the pattern
/// uniquely matches one struct.
///
/// Test fixture: struct Q { u64@0; u64@8 } is the unique 16-byte
/// candidate matching accesses (0, 8) and (8, 8). The bug
/// surface for cgx_raw / llcx_raw in `lib/cgroup_bw.bpf.c`'s
/// `scx_static_alloc()`-backed pointers — the STX-flow tag fires
/// at the allocator-return STX, the deref pattern fires later
/// when the cached pointer is consulted, and shape inference
/// must bridge the two evidence sources to recover the per-cgroup
/// payload type.
#[test]
fn stx_flow_stx_before_deref_resolves_target_via_shape_inference() {
    // P { u64 cgx_raw @ 8 }, Q { u64@0; u64@8 } — Q is the unique
    // 16-byte candidate.
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_p = push_name(&mut strings, "P");
    let n_q = push_name(&mut strings, "Q");
    let n_cgx = push_name(&mut strings, "cgx_raw");
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
                name_off: n_cgx,
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
    // Order: STX-flow tag FIRST, then deref pattern. The deref
    // happens AFTER the slot is already in `arena_stx_findings`.
    //
    //   pseudo_call (PC 0)  ; SubprogReturn at PC=0 sets R0 to
    //                         RegState::ArenaU64FromAlloc { source: None }
    //                         after the standard R0..=R5 clobber.
    //   STX[r6 + 8] = r0    ; (P, 8) → arena_stx_findings.insert(Pending).
    //   r2 = LDX[r6 + 8]    ; pass 2 sees (P, 8) in arena_stx_findings,
    //                         re-types as ArenaU64FromAlloc { source:
    //                         Some((P, 8)) }.
    //   r3 = LDX[r2 + 0]    ; records access (0, 8) against (P, 8)
    //                         via the source field.
    //   r4 = LDX[r2 + 8]    ; records access (8, 8).
    //
    // R6 must be Pointer{P} (callee-saved → survives the call).
    let insns = vec![
        pseudo_call,
        stx(BPF_SIZE_DW, 6, 0, 8),
        ldx(BPF_SIZE_DW, 2, 6, 8),
        ldx(BPF_SIZE_DW, 3, 2, 0),
        ldx(BPF_SIZE_DW, 4, 2, 8),
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
        &[SubprogReturn {
            alloc_size: None,
            insn_offset: 0,
        }],
    );
    let hit = map
        .get(&(p_id, 8))
        .expect("(P, 8) must be in CastMap — STX-flow gates emission");
    assert_eq!(
        hit.addr_space,
        AddrSpace::Arena,
        "STX-flow tag must yield AddrSpace::Arena: {map:?}"
    );
    assert_eq!(
        hit.target_type_id, q_id,
        "post-fix shape inference must resolve target_type_id=q_id ({q_id}) \
         even when the STX-flow tag fires BEFORE the deref pattern \
         (pre-fix bug: alias-tagged register dropped accesses, leaving \
         target_type_id=0): {map:?}"
    );
}
