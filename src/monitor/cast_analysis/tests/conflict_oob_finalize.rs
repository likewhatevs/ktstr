use super::*;

// ----- Conflict tests -----------------------------------------

/// A single `(struct, offset)` slot observed by BOTH the arena
/// LDX path (loaded then dereferenced) AND the kernel STX path
/// (typed `Pointer{T}` stored) is ambiguous. The same byte cannot
/// simultaneously hold an arena VA and a kernel VA. Per
/// `finalize`, both observations drop and the slot does not
/// appear in the output map.
#[test]
fn arena_and_kptr_same_field_drops_both() {
    // BTF: u64(1), T(2, u64@0), T*(3), P(4, u64@8).
    // T is the unique candidate for shape pattern (offset=0, size=8).
    // Arena: load P.u64@8 -> r2, deref r2+0 -> patterns[(P,8)]={(0,8)}.
    // Kptr: STX *(P+8) = R6 (Pointer{T}) -> kptr_findings[(P,8)] = T.
    // Conflict on (P, 8) drops both.
    let slot_off: u32 = 8;
    let (blob, t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    // Arena LDX path: r2 = *(u64*)(r1+8); r3 = *(u64*)(r2+0).
    // Kernel STX path: *(u64*)(r1+8) = r6.
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, slot_off as i16),
        ldx(BPF_SIZE_DW, 3, 2, 0),
        stx(BPF_SIZE_DW, 1, 6, slot_off as i16),
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
        ],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "arena+kptr conflict on same slot must drop both: {map:?}"
    );
}

/// Two STX writes to the same `(struct, offset)` slot with
/// different target struct ids collapse the kptr finding to
/// `KptrEntry::Conflicting`. `finalize` skips conflicting
/// entries, so the slot does not appear in the output map.
///
/// Bare `map.is_empty()` cannot distinguish "Conflicting state
/// was reached" from "the analyzer never recorded either STX".
/// Both yield empty maps. The strengthened test runs THREE
/// analyses to triangulate the production path:
///   (a) baseline single-STX of T1: must record (P, slot) -> T1.
///       Establishes that the STX path is functional and that
///       T1 is recoverable from R1.
///   (b) STX T1 then STX T2: must drop (collapse to Conflicting).
///       Same as the original test.
///   (c) STX T1, STX T2, STX T1 again: must STILL drop. Once a
///       slot transitions to Conflicting, every subsequent STX
///       (even of the original target) preserves Conflicting per
///       the `Some(_)` arm of the match in `handle_stx()` —
///       proves the slot did NOT revert to `Single(T1)` after
///       the third store. If the analyzer instead overwrote
///       Conflicting back to Single on a same-target restore,
///       (c) would emit (P, slot) -> T1 like (a) does.
#[test]
fn kptr_conflict_two_targets_drops() {
    // BTF: u64(1), T1(2, u64@0), T2(3, u64@0), P(4, u64@slot_off).
    // Seed R1=Pointer{T1}, R2=Pointer{T2}, R6=Pointer{P}.
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
    let seeds = [
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
    ];

    // (a) Baseline: single STX of T1 records (P, slot) -> T1.
    // Without this anchor, the empty map from (b)/(c) below
    // could be explained by a non-functional STX path rather
    // than the Conflicting transition.
    let insns_single = vec![stx(BPF_SIZE_DW, 6, 1, slot_off as i16), exit()];
    let map_single = analyze_casts(&insns_single, &btf, &seeds, &[], &[], &[]);
    assert_eq!(
        map_single.len(),
        1,
        "(a) single STX must record exactly one finding: {map_single:?}"
    );
    assert_eq!(
        map_single.get(&(p_id, slot_off)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t1_id,
            addr_space: AddrSpace::Kernel,
        }),
        "(a) baseline records (P, slot) -> (T1, Kernel): {map_single:?}"
    );

    // (b) Two distinct targets — collapses to Conflicting and
    // finalize drops. Same shape as the historical test.
    let insns_conflict = vec![
        stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
        stx(BPF_SIZE_DW, 6, 2, slot_off as i16),
        exit(),
    ];
    let map_conflict = analyze_casts(&insns_conflict, &btf, &seeds, &[], &[], &[]);
    assert!(
        map_conflict.is_empty(),
        "(b) two distinct kptr targets on same slot must collapse to \
             Conflicting and drop: {map_conflict:?}"
    );

    // (c) Append a third STX of T1. If the slot is Conflicting,
    // the `Some(_)` arm preserves Conflicting (no revert to
    // Single). If the production code instead reset to
    // Single(T1) on a same-target restore, the map would
    // emit (P, slot) -> T1 like (a) does. Empty map confirms
    // Conflicting was reached and is sticky.
    let insns_three = vec![
        stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
        stx(BPF_SIZE_DW, 6, 2, slot_off as i16),
        stx(BPF_SIZE_DW, 6, 1, slot_off as i16),
        exit(),
    ];
    let map_three = analyze_casts(&insns_three, &btf, &seeds, &[], &[], &[]);
    assert!(
        map_three.is_empty(),
        "(c) Conflicting state must be sticky across same-target \
             restore — third STX of T1 must not resurrect: {map_three:?}"
    );
}

// ----- OOB tests ----------------------------------------------

/// A malformed `BpfInsn` with `dst >= 11` (out of the 0..=10
/// valid register range) must NOT panic. The bounds check at
/// the top of `step()` and `handle_*` rejects early. The
/// analyzer treats the instruction as a no-op; output map is
/// empty.
#[test]
fn oob_dst_reg_does_not_panic() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // LDX with dst=11 (invalid). Construct via BpfInsn::new
    // which masks to 4 bits (11 & 0x0f == 11). No panic; map empty.
    let bad = BpfInsn::new(BPF_CLASS_LDX | BPF_SIZE_DW | BPF_MODE_MEM, 11, 1, 8, 0);
    let insns = vec![bad, exit()];
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
    assert!(map.is_empty(), "OOB dst must not panic, map empty: {map:?}");
}

/// A malformed `BpfInsn` with `src == 15` (out of the 0..=10
/// valid register range) must NOT panic. `BpfInsn::new` packs
/// 15 into the 4-bit src field; `src_reg()` decodes back to 15.
/// The bounds check rejects early; output map is empty.
#[test]
fn oob_src_reg_does_not_panic() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // LDX with src=15 (invalid). No panic; map empty.
    let bad = BpfInsn::new(BPF_CLASS_LDX | BPF_SIZE_DW | BPF_MODE_MEM, 2, 15, 8, 0);
    let insns = vec![bad, exit()];
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
    assert!(map.is_empty(), "OOB src must not panic, map empty: {map:?}");
}

// ----- Other tests --------------------------------------------

/// Storing a `Pointer{P}` into a `u64` field of struct `P`
/// itself is rejected: `parent == target` is almost always a
/// structural error from ambiguous pointer aliasing in the
/// analyzer, not a real kptr write. Production line
/// `if canonical_parent == target { return; }`
/// drops the finding; output map is empty.
#[test]
fn self_store_rejected() {
    // BTF: u64(1), P(2, u64@slot_off). No separate target type.
    let slot_off: u32 = 8;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_p = push_name(&mut strings, "P");
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
    let p_id = 2;
    // R1 = Pointer{P}. STX *(R1 + slot_off) = R1. Self-store.
    let insns = vec![stx(BPF_SIZE_DW, 1, 1, slot_off as i16), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: p_id,
        }],
        &[],
        &[],
        &[],
    );
    assert!(map.is_empty(), "self-store must be rejected: {map:?}");
}

/// A FuncProto with a variadic sentinel parameter (`name_off=0
/// AND type_id=0` per `Parameter::is_variadic`) must terminate
/// the parameter scan: no parameter slot past the sentinel
/// reseeds a register. With params `[T*, P*, variadic]` the
/// non-variadic prefix seeds R1 = Pointer{T} and R2 = Pointer{P};
/// the variadic sentinel terminates the scan so R3 stays Unknown
/// even though a real BTF parameter slot follows. A subsequent
/// STX through R3 must not record a kptr finding.
#[test]
fn variadic_param_breaks_seeding() {
    // BTF: u64(1), T(2, u64@0), T*(3), P(4, u64@slot_off1, u64@slot_off2),
    //      P*(5), FuncProto(6, params=[T*, P*, variadic, T*]).
    let slot_off1: u32 = 16;
    let slot_off2: u32 = 24;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_p = push_name(&mut strings, "P");
    let n_x = push_name(&mut strings, "x");
    let n_slot1 = push_name(&mut strings, "slot1");
    let n_slot2 = push_name(&mut strings, "slot2");
    let n_arg_t = push_name(&mut strings, "task");
    let n_arg_p = push_name(&mut strings, "parent");
    let n_arg_after = push_name(&mut strings, "after_variadic");
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
        SynType::Ptr { type_id: 4 }, // id 5: P*
        // FuncProto with [T*, P*, variadic, T*]. The trailing
        // T* slot is BTF-reachable but unreachable in the BPF
        // calling convention because the variadic sentinel
        // terminates the scan; the analyzer must NOT seed R4
        // from it.
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
                SynParam {
                    name_off: 0,
                    type_id: 0,
                },
                SynParam {
                    name_off: n_arg_after,
                    type_id: 3,
                },
            ],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let p_id = 4;
    let proto_id = 6;
    // FuncEntry seeds R1 = Pointer{T} (param 0), R2 = Pointer{P}
    // (param 1). R3 stays Unknown because the variadic sentinel
    // terminates the scan before param 3.
    // STX *(R2 + slot1) = R1 records (P, slot1) -> T.
    // STX *(R2 + slot2) = R3 must NOT record (R3 Unknown).
    let insns = vec![
        stx(BPF_SIZE_DW, 2, 1, slot_off1 as i16),
        stx(BPF_SIZE_DW, 2, 3, slot_off2 as i16),
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
        map.get(&(p_id, slot_off1)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
        "non-variadic params must seed R1 and R2: {map:?}"
    );
    assert!(
        !map.contains_key(&(p_id, slot_off2)),
        "variadic sentinel must terminate scan, R3 must stay Unknown: {map:?}"
    );
}

/// `FuncEntry` clears ALL registers (R0..R10) and the stack before
/// seeding R1..R5 from the FuncProto. A typed pointer parked in
/// any register by `InitialReg` is dropped at the entry PC —
/// including callee-saved R6..R9 (the linear walk has no real
/// caller, so preserving them would leak stale state).
#[test]
fn func_entry_clears_all_regs() {
    // BTF: u64(1), T(2, u64@0), T*(3), P(4, u64@slot_off),
    //      FuncProto(5, params=[T*]).
    let slot_off: u32 = 16;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_p = push_name(&mut strings, "P");
    let n_x = push_name(&mut strings, "x");
    let n_slot = push_name(&mut strings, "slot");
    let n_arg = push_name(&mut strings, "arg");
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
        // FuncProto(T*) -> void.
        SynType::FuncProto {
            return_type_id: 0,
            params: vec![SynParam {
                name_off: n_arg,
                type_id: 3,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let p_id = 4;
    let proto_id = 5;
    // Seed R3 = Pointer{T} and R6 = Pointer{P} via InitialReg.
    // FuncEntry at PC 0 clears ALL registers (R0..R10), then
    // seeds R1 from param 0. Both R3 and R6 are now Unknown.
    // STX *(R6 + slot) = R3 must NOT record (both cleared).
    let insns = vec![stx(BPF_SIZE_DW, 6, 3, slot_off as i16), exit()];
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
        &[FuncEntry {
            insn_offset: 0,
            func_proto_id: proto_id,
        }],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "FuncEntry pre-clear must drop R3 typed state: {map:?}"
    );
}

/// `BPF_PROBE_MEM` (`mode = 0x20`) is a post-verifier marker
/// per linux `include/linux/filter.h` and never appears in
/// pre-verification bytecode. Production treats any LDX with
/// `mode != BPF_MODE_MEM` as Unknown; a subsequent deref
/// through the resulting register records nothing.
#[test]
fn probe_mem_load_treated_as_unknown() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // PROBE_MEM mode = 0x20. code = LDX | DW | PROBE_MEM = 0x39.
    // dst=2, src=1, off=8 mimics the arena LDX shape but the
    // mode bits divert to the Unknown branch.
    const BPF_MODE_PROBE_MEM: u8 = 0x20;
    let probe_load = mk_insn(BPF_CLASS_LDX | BPF_SIZE_DW | BPF_MODE_PROBE_MEM, 2, 1, 8, 0);
    let insns = vec![probe_load, ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
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
        "BPF_PROBE_MEM load must mark dst Unknown: {map:?}"
    );
}

// ----- Finalize edge cases ------------------------------------

/// `BPF_ADDR_SPACE_CAST` arena->kernel populates `arena_confirmed`
/// even when the originating LDX never produces a downstream
/// dereference, so `patterns[(T,8)]` carries an EMPTY access set.
/// If the same `(T, off)` slot is also the parent of a STX-source
/// kptr write, `finalize` must treat the slot as conflicting and
/// drop BOTH observations: the kptr loop skips the conflicted key
/// and the arena loop skips the empty-access entry independently.
/// Without the `arena_confirmed`-side participation in the
/// conflict set the kptr finding would emit even though the cast
/// instruction proves the slot holds an arena address.
#[test]
fn finalize_arena_confirmed_conflicts_with_kptr() {
    // BTF: u64(1), T(2, u64@8), T*(3), Q(4, u64@0). T also acts
    // as the parent for the kptr STX (the slot at T+8). Q is the
    // distinct value type to keep the self-store rejection from
    // firing.
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q = push_name(&mut strings, "Q");
    let n_f = push_name(&mut strings, "f");
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
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        SynType::Ptr { type_id: 2 },
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
    let t_id = 2;
    let q_id = 4;
    // r2 = *(u64*)(r1 + 8)        ; r2 = LoadedU64Field{T, 8},
    //                               patterns[(T,8)] = {} (no deref)
    // r4 = (cast as(1)->as(0)) r2 ; arena_confirmed += (T, 8)
    // *(u64*)(r1 + 8) = r3        ; r1 still Pointer{T},
    //                               r3 = Pointer{Q} ->
    //                               kptr_findings[(T,8)] = Single(Q)
    let cast = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 2, 1, 1);
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        cast,
        stx(BPF_SIZE_DW, 1, 3, 8),
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
                reg: 3,
                struct_type_id: q_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert!(
        !map.contains_key(&(t_id, 8)),
        "arena_confirmed + kptr conflict on (T, 8) must drop both: {map:?}"
    );
    assert!(map.is_empty(), "no other entries expected: {map:?}");
}

/// Loading a u64 field without dereferencing through it leaves
/// `patterns[(T, off)]` populated but with an EMPTY access set.
/// `finalize`'s arena loop short-circuits via
/// `if accesses.is_empty() { continue }` and emits nothing. The
/// slot stays absent from the output map even though the source
/// register held a `LoadedU64Field` state at one point.
#[test]
fn finalize_empty_access_set_does_not_emit() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // r2 = *(u64 *)(r1 + 8)  -- patterns[(T,8)] = {} (loaded only)
    // exit                    -- never dereferenced
    let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), exit()];
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
        !map.contains_key(&(t_id, 8)),
        "empty access set must not emit: {map:?}"
    );
    assert!(map.is_empty(), "no other entries expected: {map:?}");
}

/// The candidate-search intersection drops the source struct
/// from the candidate set. When the original candidate set is
/// `{source, X}` (source plus exactly one foreign struct that also
/// matches), `finalize` removes the source, leaving `{X}`, and
/// emits `X` as the sole remaining candidate. Dropping the entry
/// happens only when 0 or 2+ non-source candidates remain.
#[test]
fn finalize_source_in_candidates_with_others_emits_other() {
    // BTF: u64(1), T(2, u64@0 + u64@8 -- same shape T matches its
    // own access pattern), Q(3, u64@0 + u64@8). Loading T.f at
    // offset 0 then dereferencing through it at offsets 0 and 8
    // gives candidates {T, Q} -- both have u64s at those offsets.
    // Source T is removed from the candidate set; Q is the sole
    // remaining candidate and is emitted.
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q = push_name(&mut strings, "Q");
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
    let t_id = 2;
    // Sequence: r2 = *(u64*)(r1 + 0); r3 = *(u64*)(r2 + 0);
    // r4 = *(u64*)(r2 + 8). Candidates for (offset=0, size=8) and
    // (offset=8, size=8) intersect to {T, Q}. Source T is removed
    // from candidates; Q remains as the sole non-source candidate
    // and is emitted.
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 0),
        addr_space_cast(2, 2, 1),
        ldx(BPF_SIZE_DW, 3, 2, 0),
        ldx(BPF_SIZE_DW, 4, 2, 8),
        exit(),
    ];
    let q_id = 3;
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
        map.get(&(t_id, 0)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "source removed, sole non-source candidate Q emitted: {map:?}"
    );
}

/// When the only candidate matching the access pattern is the
/// source struct itself, `finalize` removes it via
/// `candidates.remove(source)` and the resulting set
/// is empty -- nothing emits. This guards against self-typed
/// casts (`source.f` -> `source*`) where a self-referential layout
/// would silently win the intersection without disambiguating
/// evidence.
#[test]
fn finalize_only_source_candidate_drops() {
    // BTF: u64(1), T(2, u64@8). T is the only struct in the BTF;
    // its layout matches the access pattern (offset=8, size=8).
    // After remove(source) the candidate set is empty -> skip emit.
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
        // T has u64@8 only; (offset=8, size=8) matches T.
        SynType::Struct {
            name_off: n_t,
            size: 16,
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
    // Sequence: r2 = *(u64*)(r1 + 8); r3 = *(u64*)(r2 + 8).
    // Pattern recorded: source=(T,8), access=(8,8). Layout maps
    // (8,8) -> {T}. After remove(source), candidates empty ->
    // skip emit.
    let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 8), exit()];
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
        "candidate set containing only the source must drop: {map:?}"
    );
}

/// The candidate-search loop walks BTF ids `1..=max_id` where
/// `max_id = max_seen_type_id + CANDIDATE_SEARCH_SLACK` capped at
/// `MAX_BTF_ID_PROBE`. When the source struct id is small (e.g.
/// 2), `max_seen_type_id` picks up the source id (via
/// `note_type_id`) and the slack carries the search several
/// thousand ids further. A target struct that lives WELL beyond
/// the source id but well inside the slack window must still be
/// found. This guards against a regression that would shrink the
/// loop bound to `max_seen_type_id` itself, AND verifies that the
/// `MAX_BTF_ID_PROBE` cap leaves room for the slack on small
/// max_seen values.
#[test]
fn finalize_max_seen_type_id_slack_finds_distant_candidate() {
    // BTF: many filler Ptr types between T and Q so that Q's id
    // is far above max_seen_type_id (which the analyzer sets to
    // T's id when seeding R1). The slack (CANDIDATE_SEARCH_SLACK
    // = 65_536) more than covers any practical BTF id space, so
    // Q at id 203 must be found even though only T (id 2) is
    // touched during the forward pass. The MAX_BTF_ID_PROBE cap
    // (100_000) is far above id 203, so this also exercises the
    // .min() arm without truncating the search.
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q = push_name(&mut strings, "Q");
    let n_f = push_name(&mut strings, "f");
    let n_x = push_name(&mut strings, "x");
    // Build: id 1 = u64, id 2 = T (struct with u64@8), then 200
    // filler Ptr-to-u64 types (ids 3..=202), then id 203 = Q
    // (struct with u64@0). max_seen ends up at 2 (T), slack pushes
    // search through id 65_538, so id 203 is well within range.
    let mut types: Vec<SynType> = Vec::new();
    types.push(SynType::Int {
        name_off: n_u64,
        size: 8,
        encoding: 0,
        offset: 0,
        bits: 64,
    });
    types.push(SynType::Struct {
        name_off: n_t,
        size: 16,
        members: vec![SynMember {
            name_off: n_f,
            type_id: 1,
            byte_offset: 8,
        }],
    });
    for _ in 0..200 {
        types.push(SynType::Ptr { type_id: 1 });
    }
    types.push(SynType::Struct {
        name_off: n_q,
        size: 8,
        members: vec![SynMember {
            name_off: n_x,
            type_id: 1,
            byte_offset: 0,
        }],
    });
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let q_id = 203;
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
        "slack must carry search well past max_seen, capped within \
             MAX_BTF_ID_PROBE: {map:?}"
    );
}
