use super::*;

// ----- Tests --------------------------------------------------

#[test]
fn empty_insns_yields_empty_map() {
    let (blob, _t, _q) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let map = analyze_casts(&[], &btf, &[], &[], &[], &[]);
    assert!(map.is_empty());
}

#[test]
fn no_initial_seed_yields_empty_map() {
    // Without seeding any register as a struct pointer, the
    // analyzer cannot identify the source type of an LDX.
    let (blob, _t, _q) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
    let map = analyze_casts(&insns, &btf, &[], &[], &[], &[]);
    assert!(map.is_empty());
}

#[test]
fn simple_cast_recovers_target() {
    // r1 -> *(T *).
    // r2 = *(u64 *)(r1 + 8)   -- "load u64 at T.f"
    // r4 = bpf_addr_space_cast(r2, 0, 1)  -- arena_confirmed evidence (arena evidence)
    // r3 = *(u64 *)(r4 + 0)   -- "use loaded value as Q*"
    //
    // The arena_space_cast on the LoadedU64Field register is the
    // arena-evidence prerequisite: shape inference alone is not
    // enough evidence to emit a finding. The `bpf_addr_space_cast`
    // tags `(t_id, 8)` as arena-confirmed, after which the
    // shape-inference finding can fire when exactly one struct
    // matches the access pattern.
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(4, 2, 1),
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
        map.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "got: {map:?}"
    );
}

/// Arena-evidence mitigation: shape-inference candidates without direct
/// arena evidence (no `BPF_ADDR_SPACE_CAST` AND no STX-flow
/// allocator-return tag) must drop, even when the (offset, size)
/// access pattern uniquely matches one BTF candidate. Pin the
/// drop-without-evidence behaviour against a regression that
/// would re-enable shape-inference-only emits, which the
/// arena-evidence mitigation explicitly forbids: a 33-bit-shaped
/// counter on aarch64 falls inside the 4 GiB arena window and
/// would render as a chased pointer if the arena-evidence gate weren't here.
///
/// Same instruction shape as `simple_cast_recovers_target` minus
/// the `BPF_ADDR_SPACE_CAST`. The companion proof — re-running
/// with the addr_space_cast added DOES emit — anchors that the
/// drop is the arena-evidence gate, not a separate analysis defect that
/// would also have rejected the cast-augmented sequence.
#[test]
fn shape_inference_alone_drops_without_arena_confirmed() {
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();

    // (a) WITHOUT addr_space_cast: shape inference recognizes
    // the (offset=0, size=8) access against Q's layout, but the
    // arena-evidence gate at `finalize`'s arena loop demands direct evidence
    // (`arena_confirmed` OR `arena_stx_findings`). Neither is
    // populated, so the slot drops and the map stays empty.
    let insns_no_evidence = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
    let map_no_evidence = analyze_casts(
        &insns_no_evidence,
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
        map_no_evidence.is_empty(),
        "shape inference without `arena_confirmed` / `arena_stx_findings` \
             must drop per the arena-evidence mitigation: {map_no_evidence:?}"
    );

    // (b) WITH addr_space_cast on the LoadedU64Field source: the
    // cast populates `arena_confirmed` for (T, 8); the same
    // (offset=0, size=8) access is now matched against Q with
    // direct evidence, so the finding emits. Establishes that
    // (a)'s empty result is attributable specifically to the
    // arena-evidence gate — without this companion the drop could be explained
    // by an unrelated analysis defect (e.g. shape inference itself
    // failing to match Q's layout for some other reason).
    let insns_with_evidence = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(4, 2, 1),
        ldx(BPF_SIZE_DW, 3, 4, 0),
        exit(),
    ];
    let map_with_evidence = analyze_casts(
        &insns_with_evidence,
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
        map_with_evidence.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "with addr_space_cast evidence the same shape MUST emit, \
             proving (a)'s empty result is the arena-evidence gate firing: \
             {map_with_evidence:?}"
    );
}

/// Arena-evidence mitigation, multi-offset disambiguation form: a program
/// whose access pattern uniquely intersects to one candidate
/// (`Q { u64 @ 0; u32 @ 8 }`) but lacks ANY direct arena
/// evidence — neither a `BPF_ADDR_SPACE_CAST` nor an STX-flow
/// allocator-return tag — must drop. Pin the gate against
/// hostile-aarch64 regressions where a 33-bit-shaped counter
/// would fall inside the 4 GiB arena window at chase time and
/// render as a chased pointer if shape inference alone were
/// allowed to emit.
///
/// Distinct from `shape_inference_alone_drops_without_arena_confirmed`:
/// that test uses a single-access (size=8 @ 0) shape and pairs
/// the empty-without-evidence with a with-evidence companion.
/// This test exercises the multi-access intersection path —
/// `(offset=0, size=8)` AND `(offset=8, size=4)` together pin
/// the candidate set to exactly Q via shape inference, so the
/// drop guards the gate against any future rewrite that
/// preserves single-offset rejection but allows multi-offset
/// shapes to slip through.
#[test]
fn arena_evidence_rejects_shape_inference_without_evidence() {
    // BTF: u64(1), u32(2), T(3, u64@8), Q(4, u64@0+u32@8). Q is
    // the ONLY struct in the BTF whose layout satisfies both
    // (offset=0, size=8) and (offset=8, size=4), so shape
    // inference would resolve to Q if the arena-evidence gate weren't here.
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_u32 = push_name(&mut strings, "u32");
    let n_t = push_name(&mut strings, "T");
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
        SynType::Int {
            name_off: n_u32,
            size: 4,
            encoding: 0,
            offset: 0,
            bits: 32,
        },
        SynType::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        SynType::Struct {
            name_off: n_q,
            size: 12,
            members: vec![
                SynMember {
                    name_off: n_a,
                    type_id: 1,
                    byte_offset: 0,
                },
                SynMember {
                    name_off: n_b,
                    type_id: 2,
                    byte_offset: 8,
                },
            ],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 3;
    // Sequence: r2 = T.f, r3 = *(u64*)(r2 + 0), r4 = *(u32*)(r2 + 8).
    // Pattern entries: {(0, 8), (8, 4)}; intersection in
    // `build_layout_index` resolves to {Q} uniquely. NO
    // addr_space_cast, NO pseudo_call+SubprogReturn — neither
    // `arena_confirmed` nor `arena_stx_findings` populated for
    // (T, 8). The arena-evidence gate at the head of the arena-emit loop
    // drops the slot.
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        ldx(BPF_SIZE_DW, 3, 2, 0),
        ldx(BPF_SIZE_W, 4, 2, 8),
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
        "multi-offset shape inference with NO direct arena evidence \
             (no addr_space_cast, no STX-flow tag) must drop per the \
             arena-evidence mitigation: {map:?}"
    );
}

#[test]
fn ambiguous_targets_drop_silently() {
    // Build BTF with two structs having a u64 at offset 0
    // (both Q1 and Q2 match the access pattern). Cast must NOT
    // be recorded because false positives are unacceptable.
    let mut strings: Vec<u8> = vec![0];
    let n_int = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q1 = push_name(&mut strings, "Q1");
    let n_q2 = push_name(&mut strings, "Q2");
    let n_f = push_name(&mut strings, "f");
    let n_x = push_name(&mut strings, "x");
    let types = vec![
        SynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        SynType::Struct {
            name_off: n_q1,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynType::Struct {
            name_off: n_q2,
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
    let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: 2,
        }],
        &[],
        &[],
        &[],
    );
    assert!(map.is_empty(), "ambiguous candidates must drop: {map:?}");
}

#[test]
fn multi_offset_disambiguates_target() {
    // Two Q-shaped structs differ by their second offset:
    //   Q1: { u64 @0; u64 @8 }
    //   Q2: { u64 @0; u32 @8 }
    // When the BPF program reads both Q->@0 (u64) and
    // Q->@8 (u64), only Q1 fits. The intersection-based
    // matcher must converge to Q1.
    let mut strings: Vec<u8> = vec![0];
    let n_u32 = push_name(&mut strings, "u32");
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q1 = push_name(&mut strings, "Q1");
    let n_q2 = push_name(&mut strings, "Q2");
    let n_f = push_name(&mut strings, "f");
    let n_a = push_name(&mut strings, "a");
    let n_b = push_name(&mut strings, "b");
    let types = vec![
        SynType::Int {
            name_off: n_u32,
            size: 4,
            encoding: 0,
            offset: 0,
            bits: 32,
        },
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
                name_off: n_f,
                type_id: 2,
                byte_offset: 8,
            }],
        },
        SynType::Struct {
            name_off: n_q1,
            size: 16,
            members: vec![
                SynMember {
                    name_off: n_a,
                    type_id: 2,
                    byte_offset: 0,
                },
                SynMember {
                    name_off: n_b,
                    type_id: 2,
                    byte_offset: 8,
                },
            ],
        },
        SynType::Struct {
            name_off: n_q2,
            size: 16,
            members: vec![
                SynMember {
                    name_off: n_a,
                    type_id: 2,
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
    let t_id = 3;
    let q1_id = 4;
    // Sequence: load r1->T.f (offset 8) into r2; cast r2 to
    // arena_confirmed (arena-evidence prerequisite); then deref
    // the cast result at offset 0 (8 bytes) and offset 8 (8 bytes).
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(2, 2, 1),
        ldx(BPF_SIZE_DW, 3, 2, 0),
        ldx(BPF_SIZE_DW, 4, 2, 8),
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
            target_type_id: q1_id,
            addr_space: AddrSpace::Arena,
        }),
        "map: {map:?}"
    );
}

#[test]
fn multiple_distinct_casts_recorded() {
    // T has TWO u64 fields, each loaded and dereferenced as a
    // distinct target struct. Q1 has u64@8 only (no @0). Q2
    // has u64@0 + u32@8. The two cast access patterns each
    // narrow to a single candidate.
    let mut strings: Vec<u8> = vec![0];
    let n_u32 = push_name(&mut strings, "u32");
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q1 = push_name(&mut strings, "Q1");
    let n_q2 = push_name(&mut strings, "Q2");
    let n_f1 = push_name(&mut strings, "f1");
    let n_f2 = push_name(&mut strings, "f2");
    let n_a = push_name(&mut strings, "a");
    let n_b = push_name(&mut strings, "b");
    let types = vec![
        SynType::Int {
            name_off: n_u32,
            size: 4,
            encoding: 0,
            offset: 0,
            bits: 32,
        },
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynType::Struct {
            name_off: n_t,
            size: 24,
            members: vec![
                SynMember {
                    name_off: n_f1,
                    type_id: 2,
                    byte_offset: 8,
                },
                SynMember {
                    name_off: n_f2,
                    type_id: 2,
                    byte_offset: 16,
                },
            ],
        },
        SynType::Struct {
            name_off: n_q1,
            size: 16,
            members: vec![SynMember {
                name_off: n_a,
                type_id: 2,
                byte_offset: 8,
            }],
        },
        SynType::Struct {
            name_off: n_q2,
            size: 12,
            members: vec![
                SynMember {
                    name_off: n_a,
                    type_id: 2,
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
    // Type ids per `types` order: u32=1, u64=2, T=3, Q1=4, Q2=5.
    let t_id = 3;
    let q1_id = 4;
    let q2_id = 5;

    // Cast 1: T.f1 -> *(Q1*). Read at offset 8 (8 bytes) only;
    // Q1 matches, Q2 has u32@8 (4 bytes) so does not match.
    // Cast 2: T.f2 -> *(Q2*). Read at offset 0 (8 bytes) and
    // offset 8 (4 bytes). Q1 lacks @0 → only Q2 matches.
    // Each LoadedU64Field gets an addr_space_cast applied to it
    // (arena-evidence prerequisite for shape-inference findings).
    let insns = vec![
        // r2 = *(u64 *)(r1 + 8)  -- T.f1 → r2
        ldx(BPF_SIZE_DW, 2, 1, 8),
        // r2 = arena_cast(r2)    -- arena_confirmed for (T, 8)
        addr_space_cast(2, 2, 1),
        // r3 = *(u64 *)(r2 + 8)  -- (T.f1 → Q1).a (offset 8, size 8)
        ldx(BPF_SIZE_DW, 3, 2, 8),
        // Reset r2's loaded-state by overwriting via mov_k.
        mov_k(2, 0),
        // r2 = *(u64 *)(r1 + 16) -- T.f2 → r2
        ldx(BPF_SIZE_DW, 2, 1, 16),
        // r2 = arena_cast(r2)    -- arena_confirmed for (T, 16)
        addr_space_cast(2, 2, 1),
        // r4 = *(u64 *)(r2 + 0)  -- (T.f2 → Q2).a (offset 0, size 8)
        ldx(BPF_SIZE_DW, 4, 2, 0),
        // r5 = *(u32 *)(r2 + 8)  -- (T.f2 → Q2).b (offset 8, size 4)
        ldx(BPF_SIZE_W, 5, 2, 8),
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
            target_type_id: q1_id,
            addr_space: AddrSpace::Arena,
        }),
        "f1: {map:?}"
    );
    assert_eq!(
        map.get(&(t_id, 16)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q2_id,
            addr_space: AddrSpace::Arena,
        }),
        "f2: {map:?}"
    );
}

#[test]
fn register_reuse_after_call_clears_state() {
    // Load T.f into r2, then BPF_CALL clobbers r0..r5. The
    // dereference of the post-call r2 must NOT be attributed
    // to the pre-call source.
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8), // r2 = T.f
        call(),                    // clobbers r0..r5
        ldx(BPF_SIZE_DW, 3, 2, 0), // r2 is Unknown now
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
        "post-call r2 must not retain T.f source: {map:?}"
    );
}

#[test]
fn nondw_load_does_not_track_u64_field() {
    // r2 = *(u32 *)(r1 + 8)  -- not a 64-bit load, cannot carry
    // a pointer. Subsequent deref must not be attributed.
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![ldx(BPF_SIZE_W, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
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
    assert!(map.is_empty(), "32-bit load must not seed cast: {map:?}");
}

#[test]
fn ptr_field_tracked_as_typed_pointer_not_cast() {
    // T.field is declared as `Q *` in BTF (already typed). The
    // analyzer follows the chain to mark the loaded register
    // as a Q*, but does NOT record a cast (renderer already
    // chases declared Ptr fields).
    let mut strings: Vec<u8> = vec![0];
    let n_int = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q = push_name(&mut strings, "Q");
    let n_f = push_name(&mut strings, "f");
    let n_x = push_name(&mut strings, "x");
    let types = vec![
        // id 1: u64
        SynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct Q { u64 x @0 }
        SynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: Q* (pointer to id=2)
        SynType::Ptr { type_id: 2 },
        // id 4: struct T { Q* f @8 }
        SynType::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 3,
                byte_offset: 8,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 4;
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8), // r2 = T.f -- typed Q* per BTF
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
        "typed Ptr field must not be recorded as cast: {map:?}"
    );
}

#[test]
fn null_check_fall_through_preserves_state() {
    // if r2 <COND> 0 goto SKIP; deref r2; SKIP: exit.
    // The deref happens at the FALL-THROUGH after the
    // conditional jump, so the state survives. The analyzer
    // should still record the cast across every supported
    // conditional-jump op-code: per linux uapi `bpf_common.h`
    // and `bpf.h` the JMP class accepts JEQ=0x10, JGT=0x20,
    // JGE=0x30, JNE=0x50, JSGT=0x60, JSGE=0x70, JLT=0xa0,
    // JLE=0xb0, JSLT=0xc0, JSLE=0xd0 (JSET=0x40 also branches
    // but takes a bitmask not a comparison; covered too). Each
    // pairs with BPF_SRC_K (0x00) for an immediate operand; the
    // K and X variants share the same off-relative branch
    // encoding so testing K covers both as far as
    // `jump_targets()` is concerned. JMP32 class mirrors the
    // op codes; covered with BPF_CLASS_JMP32 | BPF_JEQ to verify
    // class-bit independence. None of these touch register
    // state in `step()` (only BPF_OP_CALL clears registers per
    // line ~726), so every variant must preserve the
    // pre-jump LoadedU64Field on the fall-through path.
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // BPF_SRC_K is 0; OR with BPF_SRC_X (0x08) for the X-form
    // smoke test on JEQ to confirm src-kind has no effect on
    // jump-target detection or fall-through state preservation.
    let variants: &[(u8, &str)] = &[
        (BPF_CLASS_JMP | 0x10, "JEQ_K"),
        (BPF_CLASS_JMP | 0x10 | BPF_SRC_X, "JEQ_X"),
        (BPF_CLASS_JMP | 0x20, "JGT_K"),
        (BPF_CLASS_JMP | 0x30, "JGE_K"),
        (BPF_CLASS_JMP | 0x40, "JSET_K"),
        (BPF_CLASS_JMP | 0x50, "JNE_K"),
        (BPF_CLASS_JMP | 0x60, "JSGT_K"),
        (BPF_CLASS_JMP | 0x70, "JSGE_K"),
        (BPF_CLASS_JMP | 0xa0, "JLT_K"),
        (BPF_CLASS_JMP | 0xb0, "JLE_K"),
        (BPF_CLASS_JMP | 0xc0, "JSLT_K"),
        (BPF_CLASS_JMP | 0xd0, "JSLE_K"),
        (BPF_CLASS_JMP32 | 0x10, "JEQ32_K"),
    ];
    for (code, label) in variants {
        // pc 0: r2 = T.f
        // pc 1: if r2 <COND> 0 goto +1 (jump to pc=3, skip deref)
        // pc 2: r3 = *r2  (fall-through; r2 still LoadedU64Field)
        // pc 3: exit.
        // BPF_ADDR_SPACE_CAST adds arena_confirmed evidence
        // (arena-evidence prerequisite). Apply BEFORE the
        // conditional jump so the cast lands on the source
        // u64 value, not the (already-typed) cast result.
        let jcc = mk_insn(*code, 2, 0, 1, 0);
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8),
            addr_space_cast(2, 2, 1),
            jcc,
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
            map.len(),
            1,
            "{label}: exactly one cast expected on fall-through, got: {map:?}"
        );
        assert_eq!(
            map.get(&(t_id, 8)),
            Some(&CastHit {
                alloc_size: None,
                target_type_id: q_id,
                addr_space: AddrSpace::Arena,
            }),
            "{label}: fall-through deref must record: {map:?}"
        );
    }
}

#[test]
fn deref_at_jump_target_is_dropped() {
    // if r2 != 0 goto USE; ... USE: deref r2.
    // The deref is at the branch target, where state is reset
    // by the conservative join handler. False negative is
    // acceptable.
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // pc 0: r2 = T.f
    // pc 1: if r2 != 0 goto +1 (= pc 3, the deref)
    // pc 2: exit (skipped on the taken branch)
    // pc 3: r3 = *r2  -- STATE WAS RESET at pc 3 (target).
    // pc 4: exit.
    let jne = mk_insn(BPF_CLASS_JMP | 0x50, 2, 0, 1, 0); // BPF_JNE_K = 0x50
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        jne,
        exit(),
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
    assert!(map.is_empty(), "deref at branch target must drop: {map:?}");
}

#[test]
fn mov_x_propagates_loaded_state() {
    // r2 = T.f; r2 = arena_cast(r2); r4 = r2; deref r4 at offset 0.
    // The MOV r4, r2 propagates LoadedU64Field after the
    // arena_confirmed evidence is recorded (arena-evidence mitigation).
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(2, 2, 1),
        mov_x(4, 2),
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
        map.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "MOV must propagate: {map:?}"
    );
}

#[test]
fn ld_imm64_skips_second_slot() {
    // BPF_LD_IMM64 is two slots; the second slot's `code` is 0.
    // A bare 0-code insn must not be misinterpreted as anything
    // active. After the two-slot insn, normal flow continues.
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let ld_imm64_lo = mk_insn(BPF_CLASS_LD | BPF_SIZE_DW | BPF_MODE_IMM, 6, 0, 0, 0);
    let ld_imm64_hi = mk_insn(0, 0, 0, 0, 0);
    let insns = vec![
        ld_imm64_lo,
        ld_imm64_hi,
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
        "LD_IMM64 second slot must skip: {map:?}"
    );
}

#[test]
fn r10_seed_rejected() {
    // Seeding the frame pointer is silently dropped — even
    // when the BTF type id is valid. Nothing tracks r10.
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 10, 8),
        ldx(BPF_SIZE_DW, 3, 2, 0),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 10,
            struct_type_id: t_id,
        }],
        &[],
        &[],
        &[],
    );
    assert!(map.is_empty(), "r10 seed must be ignored: {map:?}");
}

#[test]
fn nonu64_field_at_source_offset_not_tracked() {
    // T has a u32 at offset 8 (not u64). Loading from there
    // and treating as a pointer is meaningless — the analyzer
    // must not seed LoadedU64Field.
    let mut strings: Vec<u8> = vec![0];
    let n_u32 = push_name(&mut strings, "u32");
    let n_t = push_name(&mut strings, "T");
    let n_f = push_name(&mut strings, "f");
    let types = vec![
        SynType::Int {
            name_off: n_u32,
            size: 4,
            encoding: 0,
            offset: 0,
            bits: 32,
        },
        SynType::Struct {
            name_off: n_t,
            size: 12,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
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
        "u32-typed field must not seed cast: {map:?}"
    );
}
