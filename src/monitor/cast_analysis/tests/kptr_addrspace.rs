use super::*;

// ----- Kptr detection tests -----------------------------------
//
// Kptr tests share a BTF shape: a "task_struct"-like target T,
// a parent struct P with a `u64 slot @ off` field, and the
// appropriate FuncProto / Func types when a function-entry
// seeding test calls them out. The shared `btf_kptr_base` helper
// (in the facade) keeps each test focused on the instruction
// sequence under examination.

#[test]
fn kptr_from_function_param_stored_to_u64_field() {
    // R1 starts as T* (param[0]).
    // R6 = R1 (preserve across the rest).
    // R2 = some P* (the parent struct holding the kptr slot).
    //   We don't have a separate P param so seed R2 directly.
    // *(u64 *)(R2 + slot_off) = R6
    //   - R2 is Pointer{P}, R6 is Pointer{T}, field is u64 ->
    //     map records (P, slot_off) -> (T, AddrSpace::Kernel).
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![mov_x(6, 1), stx(BPF_SIZE_DW, 2, 6, slot_off as i16), exit()];
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
        "kptr STX must record kernel-space cast: {map:?}"
    );
}

#[test]
fn kptr_through_stack_spill() {
    // R1 starts as T*; spill to [r10-8]; reload into R3; store
    // R3 into the parent slot. Tests that stack spill / reload
    // preserves the typed-pointer state.
    //
    //   *(u64 *)(r10 - 8) = R1     ; spill T*
    //   R3 = *(u64 *)(r10 - 8)     ; reload as T*
    //   *(u64 *)(R4 + slot_off) = R3
    let slot_off: u32 = 24;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![
        stx(BPF_SIZE_DW, 10, 1, -8), // spill R1
        ldx(BPF_SIZE_DW, 3, 10, -8), // reload to R3
        stx(BPF_SIZE_DW, 4, 3, slot_off as i16),
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
                reg: 4,
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
        "stack spill must preserve typed pointer: {map:?}"
    );
}

#[test]
fn kptr_from_kfunc_return() {
    // BTF layout reused across this test:
    //   id 1: u64
    //   id 2: struct T { u64 x @ 0 }
    //   id 3: T*
    //   id 4: struct P { u64 slot @ 16 }
    //   id 5: FuncProto returning T*  (return_type_id = 3)
    //   id 6: Func("bpf_task_acquire") -> id 5
    //
    // Sequence:
    //   call kfunc id=6
    //   *(u64 *)(R6 + 16) = R0   ; R6 is P*
    let slot_off: u32 = 16;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_p = push_name(&mut strings, "P");
    let n_x = push_name(&mut strings, "x");
    let n_slot = push_name(&mut strings, "slot");
    let n_kfunc = push_name(&mut strings, "bpf_task_acquire");
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
            size: slot_off + 8,
            members: vec![SynMember {
                name_off: n_slot,
                type_id: 1,
                byte_offset: slot_off,
            }],
        },
        // id 5: FuncProto -> T*
        SynType::FuncProto {
            return_type_id: 3,
            params: vec![],
        },
        // id 6: Func bpf_task_acquire (linkage = global)
        SynType::Func {
            name_off: n_kfunc,
            type_id: 5,
            linkage: 1,
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let p_id = 4;
    let kfunc_id = 6;
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
    assert_eq!(
        map.get(&(p_id, slot_off)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
        "kfunc-returned T* stored to P.slot must record: {map:?}"
    );
}

#[test]
fn kptr_clobbered_by_call() {
    // R1 starts as T*. A non-kfunc BPF_CALL clobbers R0..R5.
    // The post-call STX of R1 must NOT record a kptr — R1 is
    // Unknown after the helper call.
    //
    //   call helper        ; clobbers R0..R5
    //   *(u64 *)(R6 + 16) = R1   ; R1 was clobbered
    let slot_off: u32 = 16;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![call(), stx(BPF_SIZE_DW, 6, 1, slot_off as i16), exit()];
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
        "post-call clobbered R1 must not record kptr: {map:?}"
    );
}

#[test]
fn mixed_arena_and_kptr_in_one_program() {
    // Single BTF, single instruction sequence: trigger BOTH
    // detection paths.
    //
    // BTF:
    //   id 1: u64
    //   id 2: struct T { u64 x @ 0 }     (kernel kptr target)
    //   id 3: T*
    //   id 4: struct A { u64 a0 @ 0; u64 a1 @ 8 }  (arena target)
    //   id 5: struct M {                ; map value
    //           u64 arena_ptr @ 0;     ; carries A*
    //           u64 kptr      @ 16;    ; carries T*
    //         }
    //
    // Instructions:
    //   r1 := M*        (seed via InitialReg)
    //   r6 := T*        (seed via InitialReg, separate value)
    //   r2 = *(u64 *)(r1 + 0)    ; load M.arena_ptr -> r2 = LoadedU64Field
    //   r3 = *(u64 *)(r2 + 0)    ; deref @0 (u64) -> records access
    //   r4 = *(u64 *)(r2 + 8)    ; deref @8 (u64) -> records access
    //                            ;   intersection -> A (unique match)
    //   *(u64 *)(r1 + 16) = r6   ; STX of T* into M.kptr -> Kernel cast
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_a = push_name(&mut strings, "A");
    let n_m = push_name(&mut strings, "M");
    let n_x = push_name(&mut strings, "x");
    let n_a0 = push_name(&mut strings, "a0");
    let n_a1 = push_name(&mut strings, "a1");
    let n_arena_ptr = push_name(&mut strings, "arena_ptr");
    let n_kptr = push_name(&mut strings, "kptr");
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
            name_off: n_a,
            size: 16,
            members: vec![
                SynMember {
                    name_off: n_a0,
                    type_id: 1,
                    byte_offset: 0,
                },
                SynMember {
                    name_off: n_a1,
                    type_id: 1,
                    byte_offset: 8,
                },
            ],
        },
        SynType::Struct {
            name_off: n_m,
            size: 24,
            members: vec![
                SynMember {
                    name_off: n_arena_ptr,
                    type_id: 1,
                    byte_offset: 0,
                },
                SynMember {
                    name_off: n_kptr,
                    type_id: 1,
                    byte_offset: 16,
                },
            ],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let a_id = 4;
    let m_id = 5;
    let insns = vec![
        // Arena LDX path with arena_confirmed evidence (arena evidence).
        ldx(BPF_SIZE_DW, 2, 1, 0),
        addr_space_cast(2, 2, 1),
        ldx(BPF_SIZE_DW, 3, 2, 0),
        ldx(BPF_SIZE_DW, 4, 2, 8),
        // Kernel STX path.
        stx(BPF_SIZE_DW, 1, 6, 16),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 1,
                struct_type_id: m_id,
            },
            InitialReg {
                reg: 6,
                struct_type_id: t_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert_eq!(
        map.get(&(m_id, 0)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: a_id,
            addr_space: AddrSpace::Arena,
        }),
        "arena cast missing: {map:?}"
    );
    assert_eq!(
        map.get(&(m_id, 16)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
        "kernel kptr missing: {map:?}"
    );
}

#[test]
fn func_entry_seeding_from_btf() {
    // FuncProto with two parameters: param 0 = T* (typed
    // source), param 1 = P* (parent base). FuncEntry seeds
    // R1 = Pointer{T} and R2 = Pointer{P}. R3..R5 must remain
    // Unknown — the FuncProto only describes two parameters,
    // and `seed_from_func_proto()` walks `proto.parameters`
    // (not R3..R5 unconditionally). InitialReg state set
    // before the run does not survive into a function entry
    // at PC 0.
    //
    // The strengthened test verifies BOTH halves:
    //   1. R1 and R2 are typed → STX through R2 records
    //      (P, slot1) -> T at the slot dedicated to the seeded
    //      param.
    //   2. R3, R4, R5 stay Unknown → STX through each into a
    //      distinct u64 slot in P records nothing. If
    //      `seed_from_func_proto()` accidentally typed R3..R5
    //      from leftover state or over-walked the parameter
    //      list, those stores would record (P, slotN) -> T
    //      and the count assertion would fire.
    let slot1: u32 = 16; // store R1 -> P (typed, must record)
    let slot3: u32 = 24; // store R3 -> P (must NOT record)
    let slot4: u32 = 32; // store R4 -> P (must NOT record)
    let slot5: u32 = 40; // store R5 -> P (must NOT record)
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_p = push_name(&mut strings, "P");
    let n_x = push_name(&mut strings, "x");
    let n_s1 = push_name(&mut strings, "s1");
    let n_s3 = push_name(&mut strings, "s3");
    let n_s4 = push_name(&mut strings, "s4");
    let n_s5 = push_name(&mut strings, "s5");
    let n_arg_t = push_name(&mut strings, "task");
    let n_arg_p = push_name(&mut strings, "parent");
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
            size: slot5 + 8,
            members: vec![
                SynMember {
                    name_off: n_s1,
                    type_id: 1,
                    byte_offset: slot1,
                },
                SynMember {
                    name_off: n_s3,
                    type_id: 1,
                    byte_offset: slot3,
                },
                SynMember {
                    name_off: n_s4,
                    type_id: 1,
                    byte_offset: slot4,
                },
                SynMember {
                    name_off: n_s5,
                    type_id: 1,
                    byte_offset: slot5,
                },
            ],
        },
        SynType::Ptr { type_id: 4 }, // id 5: P*
        // id 6: FuncProto(T*, P*) -> void. Only two params, so
        // FuncEntry only seeds R1 and R2 — R3, R4, R5 must
        // stay Unknown.
        SynType::FuncProto {
            return_type_id: 0,
            params: vec![
                SynParam {
                    name_off: n_arg_t,
                    type_id: 3,
                },
                SynParam {
                    name_off: n_arg_p,
                    type_id: 5,
                },
            ],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let p_id = 4;
    let proto_id = 6;
    // STX *(R2 + slot1) = R1   ; R1=T, R2=P → records (P, slot1) -> T
    // STX *(R2 + slot3) = R3   ; R3=Unknown → no record
    // STX *(R2 + slot4) = R4   ; R4=Unknown → no record
    // STX *(R2 + slot5) = R5   ; R5=Unknown → no record
    let insns = vec![
        stx(BPF_SIZE_DW, 2, 1, slot1 as i16),
        stx(BPF_SIZE_DW, 2, 3, slot3 as i16),
        stx(BPF_SIZE_DW, 2, 4, slot4 as i16),
        stx(BPF_SIZE_DW, 2, 5, slot5 as i16),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[],
        &[FuncEntry {
            insn_offset: 0,
            func_proto_id: proto_id,
        }],
        &[],
        &[],
    );
    assert_eq!(
        map.len(),
        1,
        "FuncEntry must seed only R1 and R2; R3..R5 stay Unknown so \
             only the R1->slot1 STX records: {map:?}"
    );
    assert_eq!(
        map.get(&(p_id, slot1)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
        "FuncEntry param seeding must populate R1 and R2: {map:?}"
    );
    // Adversary check: the failure modes we are guarding against
    // would emit (P, slot3/4/5) -> T. Assert each absent.
    assert!(
        !map.contains_key(&(p_id, slot3)),
        "R3 must remain Unknown post-FuncEntry: {map:?}"
    );
    assert!(
        !map.contains_key(&(p_id, slot4)),
        "R4 must remain Unknown post-FuncEntry: {map:?}"
    );
    assert!(
        !map.contains_key(&(p_id, slot5)),
        "R5 must remain Unknown post-FuncEntry: {map:?}"
    );
}

// ----- BPF_ADDR_SPACE_CAST tests ------------------------------

/// `BPF_ADDR_SPACE_CAST` arena -> kernel (`imm == 1`) on a
/// `LoadedU64Field` source populates `arena_confirmed` but does
/// NOT produce a standalone map entry when no subsequent deref
/// refines the target via shape inference. Without a resolved
/// target type, the renderer cannot chase — emitting a
/// placeholder would produce worse output than the raw u64
/// fallback. The cast evidence participates only in conflict
/// detection (preventing a kptr finding from claiming the slot).
///
/// The "no emit alone" fact must be distinguished from the failure
/// mode where the cast is silently ignored — both produce an
/// empty map. The test runs three analyses to nail down which
/// branch is exercised:
///   1. cast alone           → empty (arena_confirmed populated,
///      but no deref pattern).
///   2. cast + same-slot STX → empty (arena_confirmed conflicts
///      with kptr_findings, both drop).
///   3. same-slot STX alone  → kptr finding emits.
///
/// (1) + (2) - (3) prove arena_confirmed was populated by the
/// cast: if it were not, (2) would emit the kptr finding just as
/// (3) does, contradicting the empty result.
#[test]
fn addr_space_cast_arena_alone_does_not_emit() {
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();

    // (1) cast alone — current behavior. arena_confirmed must
    // be populated for (T, 8) but no map entry emitted.
    // r3 = *(u64 *)(r1 + 8)         ; r3 = LoadedU64Field{T, 8}
    // r4 = (cast as(1) -> as(0)) r3 ; arena_confirmed += (T, 8)
    let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 1, 1);
    let insns_cast_only = vec![ldx(BPF_SIZE_DW, 3, 1, 8), cast, exit()];
    let map_cast_only = analyze_casts(
        &insns_cast_only,
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
        map_cast_only.is_empty(),
        "arena_confirmed alone (no deref pattern) must not emit: {map_cast_only:?}"
    );

    // (2) cast + STX of Pointer{Q} into the same slot. If
    // arena_confirmed for (T, 8) was populated by the cast,
    // the conflict-detection chain in `finalize()` drops both
    // observations and the map stays empty. If the cast did
    // NOT populate arena_confirmed, no conflict and the kptr
    // finding (T, 8) -> Q emits.
    // r3 = *(u64 *)(r1 + 8)            ; LoadedU64Field source
    // r4 = (cast as(1) -> as(0)) r3    ; arena_confirmed += (T, 8)
    // *(u64 *)(r1 + 8) = r5            ; kptr_findings += (T, 8) -> Q
    let insns_cast_plus_kptr = vec![
        ldx(BPF_SIZE_DW, 3, 1, 8),
        cast,
        stx(BPF_SIZE_DW, 1, 5, 8),
        exit(),
    ];
    let map_cast_plus_kptr = analyze_casts(
        &insns_cast_plus_kptr,
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
        map_cast_plus_kptr.is_empty(),
        "cast + same-slot STX must conflict-drop both observations \
             (proves arena_confirmed was populated): {map_cast_plus_kptr:?}"
    );

    // (3) STX alone — no cast, so arena_confirmed stays empty
    // and the kptr finding emits. Establishes the baseline
    // "STX would have recorded" so that (2)'s empty result is
    // attributable to the conflict, not to a non-functional
    // STX path.
    let insns_kptr_only = vec![stx(BPF_SIZE_DW, 1, 5, 8), exit()];
    let map_kptr_only = analyze_casts(
        &insns_kptr_only,
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
    assert_eq!(
        map_kptr_only.len(),
        1,
        "STX-only baseline must record exactly one kptr finding: {map_kptr_only:?}"
    );
    assert_eq!(
        map_kptr_only.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Kernel,
        }),
        "STX-only baseline records (T, 8) -> (Q, Kernel): {map_kptr_only:?}"
    );
}

/// A `BPF_ADDR_SPACE_CAST` arena confirmation (`imm == 1`) must
/// SURVIVE a later untyped (`StxValueKind::Unknown`) store to the
/// same slot. This is the `.bss` arena-holder pattern: one BPF
/// program casts the slot (recording `arena_confirmed` for it)
/// while another stores the live arena VA into the same global as
/// a plain `u64`. May-analysis: the cast evidence persists across
/// the untyped store, so shape inference still emits the chase
/// finding and the renderer chases the pointer.
///
/// Regression test for the deterministic-drop bug where the
/// `StxValueKind::Unknown` STX arm removed the `arena_confirmed`
/// entry, silently demoting the rendered pointer to a raw `u64`.
/// Without the cast, the untyped store records nothing; with the
/// cast but the old invalidation, the store cleared the evidence
/// and the map went empty -- so a non-empty `(T, 8) -> (Q, Arena)`
/// result proves the confirmation outlived the untyped store.
#[test]
fn addr_space_cast_arena_survives_unknown_stx() {
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // r3 = *(u64 *)(r1 + 8)            ; r3 = LoadedU64Field{T, 8}
    // r4 = (cast as(1) -> as(0)) r3    ; arena_confirmed += (T, 8)
    // r6 = *(u64 *)(r3 + 0)            ; deref -> shape candidate (T,8)->Q
    // *(u64 *)(r1 + 8) = r9            ; r9 never set -> StxValueKind::Unknown
    // The untyped store must NOT clear arena_confirmed for (T, 8),
    // otherwise shape inference loses its arena evidence and drops
    // the finding (see shape_inference_alone_drops_without_arena_confirmed).
    let insns = vec![
        ldx(BPF_SIZE_DW, 3, 1, 8),
        addr_space_cast(4, 3, 1),
        ldx(BPF_SIZE_DW, 6, 3, 0),
        stx(BPF_SIZE_DW, 1, 9, 8),
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
        map.len(),
        1,
        "exactly the (T, 8) chase finding expected: {map:?}"
    );
    assert_eq!(
        map.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "arena_confirmed must survive the untyped STX so shape \
             inference still emits (T,8)->(Q, Arena): {map:?}"
    );
}

/// The deferred-resolve emit loop (cast_analysis/mod.rs) must emit a
/// `target_type_id == 0` arena `CastHit` when a slot has arena evidence
/// (`addr_space_cast` -> `arena_confirmed`) and a deref pattern, but shape
/// inference cannot UNIQUELY resolve the pointee (>=2 BTF structs match the
/// deref shape, so the `candidates.len() == 1` check fails). The renderer
/// then chases via the runtime arena VA (`resolve_arena_type`) at dump time
/// rather than a static target type.
///
/// Distinct from `addr_space_cast_arena_survives_unknown_stx`, whose
/// single-candidate BTF lets shape inference resolve uniquely (so the
/// deferred loop never fires). Two same-shape targets Q1/Q2 force the
/// ambiguous branch -- the only focused unit coverage of the
/// `target_type_id == 0` emit path (otherwise exercised only end-to-end by
/// `cast_analysis_chases_bss_to_arena`).
#[test]
fn addr_space_cast_ambiguous_shape_emits_deferred_resolve() {
    let (blob, t_id) = btf_source_and_two_targets(8);
    let btf = Btf::from_bytes(&blob).unwrap();
    // r3 = *(u64 *)(r1 + 8)            ; r3 = LoadedU64Field{T, 8}
    // r4 = (cast as(1) -> as(0)) r3    ; arena_confirmed += (T, 8)
    // r6 = *(u64 *)(r3 + 0)            ; deref -> pattern (T,8); shape
    //                                    matches Q1 AND Q2 -> ambiguous
    // Shape inference drops (T,8) (candidates != 1); the deferred loop
    // emits target_type_id=0 from arena_confirmed + the pattern.
    let insns = vec![
        ldx(BPF_SIZE_DW, 3, 1, 8),
        addr_space_cast(4, 3, 1),
        ldx(BPF_SIZE_DW, 6, 3, 0),
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
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        }),
        "ambiguous shape + arena_confirmed must emit a deferred-resolve \
             (target_type_id=0) arena CastHit: {map:?}"
    );
}

/// `BPF_ADDR_SPACE_CAST` kernel -> arena (`imm == 0x10000`) on a
/// `LoadedU64Field` source PROPAGATES the source's `RegState` into
/// the destination register (and tags `arena_confirmed` for the
/// source slot). Production (cast_analysis/mod.rs, the `imm == 1
/// << 16` arm) does `self.regs[dst] = self.regs[src]` for a
/// `LoadedU64Field` source — it does NOT drop dst to Unknown. A
/// subsequent deref through the cast RESULT register therefore
/// records the same chase finding the source slot would, which is
/// the analyzer's may-analysis behavior for cross-subprog arena
/// pointer detection (see sibling
/// `addr_space_cast_kernel_arena_preserves_pointer_source` for the
/// `Pointer{T}` source case).
///
/// Note: this propagation intentionally over-tracks relative to the
/// kernel verifier, which `mark_reg_unknown`s the cast_user
/// (`imm == 1U << 16`) destination with no type
/// (kernel/bpf/verifier.c::check_alu_op). The analyzer keeps the
/// typed state so a later deref through the cast result still
/// attributes to the source struct field.
#[test]
fn addr_space_cast_kernel_to_arena_propagates_loaded_field() {
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // r3 = *(u64 *)(r1 + 8)            ; r3 = LoadedU64Field{T, 8}
    // r4 = (cast as(0) -> as(1)) r3    ; arena_confirmed += (T, 8),
    //                                    r4 = LoadedU64Field{T, 8}
    //                                    (propagated from r3)
    // r5 = *(u64 *)(r4 + 0)            ; deref through the cast
    //                                    RESULT -> records (T,8)->Q
    // The deref is through r4 ALONE (no redundant r3 deref), so the
    // recorded finding can ONLY come from r4 retaining the
    // propagated LoadedU64Field state. If production had dropped r4
    // to Unknown, the deref would attribute to no source and the
    // map would be empty.
    let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 1, 0x10000);
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
    // The deref through the propagated r4 unique-resolves to Q (the
    // only BTF struct with a u64 at offset 0). The cast itself
    // populated arena_confirmed for (T, 8), so shape inference
    // emits the (T, 8) -> (Q, Arena) finding.
    assert_eq!(
        map.len(),
        1,
        "exactly one finding (via the propagated cast result r4) expected: {map:?}"
    );
    assert_eq!(
        map.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "kernel->arena cast propagates src LoadedU64Field into dst, so the \
         deref through the cast result records (T, 8) -> (Q, Arena): {map:?}"
    );
}

/// Sign-extending MOV (`off in {8, 16, 32}`) destroys the typed-
/// pointer property — a sign-extended s8/s16/s32 cannot survive
/// as a 64-bit pointer. Production drops dst to Unknown; a
/// subsequent deref through the resulting register must not
/// record any cast.
#[test]
fn sign_extend_mov_drops_state() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // r3 = *(u64 *)(r1 + 8)         ; r3 = LoadedU64Field{T, 8}
    // r4 = (s32) r3                  ; off=8 sign-extend -> Unknown
    // r5 = *(u64 *)(r4 + 0)         ; r4 Unknown -> no record
    let sxt = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 8, 0);
    let insns = vec![
        ldx(BPF_SIZE_DW, 3, 1, 8),
        sxt,
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
        "sign-extend MOV must drop typed state: {map:?}"
    );
}
