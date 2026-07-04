use super::*;

// ----- Stress / boundary tests -------------------------------
//
// These tests target performance regressions (quadratic blowups
// over the BTF id walk, the patterns set, the layout index) and
// boundary panics (OOB indexing on max-stack, depth-limit chains
// through `peel_modifiers`). Assertions verify exact
// `CastMap` contents — counting only would mask both spurious
// entries and missed entries.

/// 10,000-instruction program: stuffed with `r0 = 0` no-ops with
/// a single arena cast pattern buried near the middle. Verifies
/// the analyzer's forward-pass cost remains linear in instruction
/// count and that its register tracking does not lose the typed
/// state across thousands of unrelated instructions. Single-slot
/// `r0 = 0` only clobbers `r0`, so the seeded `r1 = T*` and the
/// loaded `r2 = LoadedU64Field{T, 8}` survive across the no-op
/// padding and the cast resolves uniquely to `Q`. `mov_k(0, 0)` is
/// used (not a `BPF_JA +0` no-op) because it clobbers only `r0` and
/// is ALU64-class, so it adds no jump targets. A `BPF_JA +0` would
/// add every PC to the jump-target set, but the branch-merge only
/// saves/restores register state at conditional-branch targets — no
/// state is reset at unconditional-JA targets — so an all-`BPF_JA +0`
/// stream would carry `r1`/`r2` through unchanged rather than
/// clobbering them.
#[test]
fn large_program_buried_cast_recorded() {
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let mut insns: Vec<BpfInsn> = Vec::with_capacity(10_001);
    for _ in 0..4_999 {
        insns.push(mov_k(0, 0));
    }
    insns.push(ldx(BPF_SIZE_DW, 2, 1, 8));
    // arena-evidence mitigation: arena_confirmed evidence on r2.
    insns.push(addr_space_cast(2, 2, 1));
    insns.push(ldx(BPF_SIZE_DW, 3, 2, 0));
    for _ in 0..4_997 {
        insns.push(mov_k(0, 0));
    }
    insns.push(exit());
    assert_eq!(insns.len(), 10_000);
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
        "exactly one cast in 10k-insn program: {map:?}"
    );
    assert_eq!(
        map.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "buried cast must resolve: {map:?}"
    );
}

/// 100 distinct `FuncEntry` records at consecutive PCs, each
/// pointing at a distinct `FuncProto(T_i*, P*) -> void`. After
/// each entry's reseeding, the single instruction at that PC is
/// `STX *(R2 + slot_off_i) = R1`, recording `(P, slot_off_i) ->
/// (T_i, Kernel)`. Verifies that the analyzer applies every
/// `FuncEntry` (no off-by-one, no early-exit on the entry list
/// scan) and that 100 distinct kptr findings land in the output.
/// Sized below the `i16` byte-offset bound (100 * 8 = 800).
#[test]
fn many_func_entries_each_seeds() {
    const N: usize = 100;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_p = push_name(&mut strings, "P");
    let n_arg_t = push_name(&mut strings, "task");
    let n_arg_p = push_name(&mut strings, "parent");
    let mut t_name_offs = Vec::with_capacity(N);
    for i in 0..N {
        t_name_offs.push(push_name(&mut strings, &format!("T{i}")));
    }
    let mut slot_name_offs = Vec::with_capacity(N);
    for i in 0..N {
        slot_name_offs.push(push_name(&mut strings, &format!("slot{i}")));
    }
    // Type id layout (1-indexed; id 0 is Void):
    //   1: u64; 2..=N+1: T_i (each with u64@0);
    //   N+2..=2N+1: T_i*; 2N+2: P (N u64 fields); 2N+3: P*;
    //   2N+4..=3N+3: N FuncProtos (T_i*, P*) -> void.
    let mut types: Vec<SynType> = Vec::new();
    types.push(SynType::Int {
        name_off: n_u64,
        size: 8,
        encoding: 0,
        offset: 0,
        bits: 64,
    });
    for &name_off in t_name_offs.iter().take(N) {
        types.push(SynType::Struct {
            name_off,
            size: 8,
            members: vec![SynMember {
                name_off: 0,
                type_id: 1,
                byte_offset: 0,
            }],
        });
    }
    for i in 0..N {
        types.push(SynType::Ptr {
            type_id: (2 + i) as u32,
        });
    }
    let p_size: u32 = 8 * (N as u32);
    let p_members: Vec<SynMember> = (0..N)
        .map(|i| SynMember {
            name_off: slot_name_offs[i],
            type_id: 1,
            byte_offset: 8 * i as u32,
        })
        .collect();
    types.push(SynType::Struct {
        name_off: n_p,
        size: p_size,
        members: p_members,
    });
    let p_id: u32 = 2 + 2 * N as u32;
    types.push(SynType::Ptr { type_id: p_id });
    let p_ptr_id: u32 = 2 * N as u32 + 3;
    for i in 0..N {
        types.push(SynType::FuncProto {
            return_type_id: 0,
            params: vec![
                SynParam {
                    name_off: n_arg_t,
                    type_id: (N as u32 + 2 + i as u32),
                },
                SynParam {
                    name_off: n_arg_p,
                    type_id: p_ptr_id,
                },
            ],
        });
    }
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let mut insns: Vec<BpfInsn> = Vec::with_capacity(N + 1);
    let mut func_entries: Vec<FuncEntry> = Vec::with_capacity(N);
    for i in 0..N {
        insns.push(stx(BPF_SIZE_DW, 2, 1, (8 * i) as i16));
        let proto_id: u32 = 2 * N as u32 + 4 + i as u32;
        func_entries.push(FuncEntry {
            insn_offset: i,
            func_proto_id: proto_id,
        });
    }
    insns.push(exit());
    let map = analyze_casts(&insns, &btf, &[], &func_entries, &[], &[]);
    assert_eq!(map.len(), N, "expected {N} kptr findings: {map:?}");
    for i in 0..N {
        let t_id = (2 + i) as u32;
        assert_eq!(
            map.get(&(p_id, 8 * i as u32)),
            Some(&CastHit {
                alloc_size: None,
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel,
            }),
            "FuncEntry #{i} at PC {i} must record (P, {}) -> T{i}: {map:?}",
            8 * i as u32
        );
    }
}

/// 500 distinct struct types in the BTF; only one matches the
/// observed access pattern. Verifies that the matcher's
/// intersection over `build_layout_index` correctly narrows the
/// candidate set when nearly every other type matches a
/// disambiguating-but-not-target shape.
///
/// Layout: `Qtarget` has `(u64@40, u32@80)`. The other 499 each
/// carry only a single `u64@0` — they match neither `(40, 8)` nor
/// `(80, 4)`, so the intersection collapses to `Qtarget`. Source
/// `T` has a single u64@8, which matches neither observed access
/// `(40, 8)` nor `(80, 4)`, so `T` never enters the candidate set.
#[test]
fn many_struct_types_unique_match_resolves() {
    const N_FILLER: usize = 499;
    let mut strings: Vec<u8> = vec![0];
    let n_u32 = push_name(&mut strings, "u32");
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_qtarget = push_name(&mut strings, "Qtarget");
    let n_filler_a = push_name(&mut strings, "a");
    let n_filler_b = push_name(&mut strings, "b");
    let n_f = push_name(&mut strings, "f");
    let mut filler_name_offs = Vec::with_capacity(N_FILLER);
    for i in 0..N_FILLER {
        filler_name_offs.push(push_name(&mut strings, &format!("Q{i}")));
    }
    let mut types: Vec<SynType> = vec![
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
            name_off: n_qtarget,
            size: 84,
            members: vec![
                SynMember {
                    name_off: n_filler_a,
                    type_id: 2,
                    byte_offset: 40,
                },
                SynMember {
                    name_off: n_filler_b,
                    type_id: 1,
                    byte_offset: 80,
                },
            ],
        },
    ];
    for &name_off in filler_name_offs.iter().take(N_FILLER) {
        types.push(SynType::Struct {
            name_off,
            size: 8,
            members: vec![SynMember {
                name_off: n_filler_a,
                type_id: 2,
                byte_offset: 0,
            }],
        });
    }
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id: u32 = 3;
    let qtarget_id: u32 = 4;
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(2, 2, 1),
        ldx(BPF_SIZE_DW, 3, 2, 40),
        ldx(BPF_SIZE_W, 4, 2, 80),
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
        "single unique cast across 500 candidates: {map:?}"
    );
    assert_eq!(
        map.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: qtarget_id,
            addr_space: AddrSpace::Arena,
        }),
        "unique match must resolve to Qtarget: {map:?}"
    );
}

/// 30-level chain of cycling `Typedef -> Const -> Volatile`
/// modifiers wrapping a `u64` `Int`. `peel_modifiers` walks 30
/// peel iterations (well below the `MAX_MODIFIER_DEPTH = 32`
/// cap) before resolving the underlying type. The struct member
/// at `T.f` carries this deep chain as its declared type; the
/// analyzer's cast path must still recognize the field as a
/// plain `u64` and seed `LoadedU64Field` on the LDX. The
/// follow-up deref then records `(T, 8) -> (Q, Arena)`.
#[test]
fn deep_modifier_chain_resolves_to_u64() {
    const CHAIN_LEN: usize = 30;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q = push_name(&mut strings, "Q");
    let n_f = push_name(&mut strings, "f");
    let n_x = push_name(&mut strings, "x");
    let n_typedef = push_name(&mut strings, "alias_t");
    let mut types: Vec<SynType> = Vec::new();
    types.push(SynType::Int {
        name_off: n_u64,
        size: 8,
        encoding: 0,
        offset: 0,
        bits: 64,
    });
    for i in 0..CHAIN_LEN {
        let inner_id = 1 + i as u32;
        let kind = i % 3;
        let chain_node = match kind {
            0 => SynType::Typedef {
                name_off: n_typedef,
                type_id: inner_id,
            },
            1 => SynType::Const { type_id: inner_id },
            _ => SynType::Volatile { type_id: inner_id },
        };
        types.push(chain_node);
    }
    let chain_head_id: u32 = (CHAIN_LEN as u32) + 1;
    types.push(SynType::Struct {
        name_off: n_t,
        size: 16,
        members: vec![SynMember {
            name_off: n_f,
            type_id: chain_head_id,
            byte_offset: 8,
        }],
    });
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
    let t_id: u32 = chain_head_id + 1;
    let q_id: u32 = chain_head_id + 2;
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
        "30-level modifier chain must peel to u64 and seed cast: {map:?}"
    );
}

/// All 64 stack slots filled with distinct typed pointers (each
/// produced by a kfunc-call return), then all reloaded and stored
/// into 64 distinct `(P, slot_off)` slots — yielding 64 kernel
/// kptr findings. Verifies that `stack_slots` (a `BTreeMap`)
/// handles a fully-loaded BPF stack frame (512 bytes at 8 bytes/slot)
/// with no slot lost or aliased on reload. Each kfunc returns a
/// different `T_i*` so the assertion validates that saved register
/// state per slot is preserved independently.
#[test]
fn maximum_stack_slots_all_recorded() {
    const N: usize = 64;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_p = push_name(&mut strings, "P");
    let mut t_names = Vec::with_capacity(N);
    let mut slot_names = Vec::with_capacity(N);
    let mut kfunc_names = Vec::with_capacity(N);
    for i in 0..N {
        t_names.push(push_name(&mut strings, &format!("T{i}")));
        slot_names.push(push_name(&mut strings, &format!("slot{i}")));
        kfunc_names.push(push_name(&mut strings, &format!("kfunc_acquire_{i}")));
    }
    // Type id layout:
    //   1: u64; 2..=N+1: T_i; N+2..=2N+1: T_i*;
    //   2N+2: P (N u64 fields); 2N+3..=3N+2: FuncProtos returning T_i*;
    //   3N+3..=4N+2: Func entries.
    let mut types: Vec<SynType> = Vec::new();
    types.push(SynType::Int {
        name_off: n_u64,
        size: 8,
        encoding: 0,
        offset: 0,
        bits: 64,
    });
    for &name_off in t_names.iter().take(N) {
        types.push(SynType::Struct {
            name_off,
            size: 8,
            members: vec![SynMember {
                name_off: 0,
                type_id: 1,
                byte_offset: 0,
            }],
        });
    }
    for i in 0..N {
        types.push(SynType::Ptr {
            type_id: (2 + i) as u32,
        });
    }
    let p_members: Vec<SynMember> = (0..N)
        .map(|i| SynMember {
            name_off: slot_names[i],
            type_id: 1,
            byte_offset: 8 * i as u32,
        })
        .collect();
    types.push(SynType::Struct {
        name_off: n_p,
        size: 8 * N as u32,
        members: p_members,
    });
    for i in 0..N {
        types.push(SynType::FuncProto {
            return_type_id: (N as u32 + 2 + i as u32),
            params: vec![],
        });
    }
    for (i, &name_off) in kfunc_names.iter().enumerate().take(N) {
        types.push(SynType::Func {
            name_off,
            type_id: (2 * N as u32 + 3 + i as u32),
            linkage: 1,
        });
    }
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let p_id: u32 = 2 * N as u32 + 2;
    // r6 is seeded with Pointer{P} via InitialReg and is callee-
    // saved across kfunc CALL (per BPF ABI, R6..R9 are not clobbered).
    let mut insns: Vec<BpfInsn> = Vec::with_capacity(4 * N + 1);
    for i in 0..N {
        let func_id: u32 = 3 * N as u32 + 3 + i as u32;
        insns.push(kfunc_call(func_id));
        insns.push(stx(BPF_SIZE_DW, 10, 0, -((i as i16 + 1) * 8)));
    }
    for i in 0..N {
        insns.push(ldx(BPF_SIZE_DW, 3, 10, -((i as i16 + 1) * 8)));
        insns.push(stx(BPF_SIZE_DW, 6, 3, (8 * i) as i16));
    }
    insns.push(exit());
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
    assert_eq!(map.len(), N, "expected {N} kptr findings: {map:?}");
    for i in 0..N {
        let t_id: u32 = (2 + i) as u32;
        assert_eq!(
            map.get(&(p_id, 8 * i as u32)),
            Some(&CastHit {
                alloc_size: None,
                target_type_id: t_id,
                addr_space: AddrSpace::Kernel,
            }),
            "stack slot {i} (off={}) must record T{i}: {map:?}",
            -((i as i16 + 1) * 8)
        );
    }
}

/// Source struct `T` with 100 `u64` members; cast pattern triggers
/// at `f50` (offset 400) and `f99` (offset 792). Each load enters
/// `LoadedU64Field`, then a follow-up deref records a unique-shape
/// access against a single matching candidate (`Q50` for `f50`,
/// `Q99` for `f99`). The dereference offsets/sizes are chosen so
/// `T`'s u64-at-multiple-of-8 layout matches NEITHER pattern,
/// avoiding the ambiguity drop. Verifies the matcher scales when
/// the source struct has many fields.
#[test]
fn many_field_struct_records_two_distinct_casts() {
    const N: u32 = 100;
    let mut strings: Vec<u8> = vec![0];
    let n_u8 = push_name(&mut strings, "u8");
    let n_u32 = push_name(&mut strings, "u32");
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_q50 = push_name(&mut strings, "Q50");
    let n_q99 = push_name(&mut strings, "Q99");
    let n_x = push_name(&mut strings, "x");
    let mut t_field_names = Vec::with_capacity(N as usize);
    for i in 0..N {
        t_field_names.push(push_name(&mut strings, &format!("f{i}")));
    }
    let t_members: Vec<SynMember> = (0..N)
        .map(|i| SynMember {
            name_off: t_field_names[i as usize],
            type_id: 3,
            byte_offset: 8 * i,
        })
        .collect();
    // Type ids:
    //   1: u8, 2: u32, 3: u64;
    //   4: T (100 u64 fields at 0, 8, ..., 792);
    //   5: Q50 (single u32@4 — pattern (4, 4) matches only this);
    //   6: Q99 (single u8@5 — pattern (5, 1) matches only this).
    let types = vec![
        SynType::Int {
            name_off: n_u8,
            size: 1,
            encoding: 0,
            offset: 0,
            bits: 8,
        },
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
            size: 8 * N,
            members: t_members,
        },
        SynType::Struct {
            name_off: n_q50,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 2,
                byte_offset: 4,
            }],
        },
        SynType::Struct {
            name_off: n_q99,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 5,
            }],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id: u32 = 4;
    let q50_id: u32 = 5;
    let q99_id: u32 = 6;
    // f50 at offset 400; f99 at offset 792.
    // Each LoadedU64Field gets an addr_space_cast applied for
    // arena-evidence mitigation: arena_confirmed evidence.
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 400),
        addr_space_cast(2, 2, 1),
        ldx(BPF_SIZE_W, 3, 2, 4),
        ldx(BPF_SIZE_DW, 2, 1, 792),
        addr_space_cast(2, 2, 1),
        ldx(BPF_SIZE_B, 4, 2, 5),
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
    assert_eq!(map.len(), 2, "two distinct casts expected: {map:?}");
    assert_eq!(
        map.get(&(t_id, 400)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q50_id,
            addr_space: AddrSpace::Arena,
        }),
        "f50 at offset 400: {map:?}"
    );
    assert_eq!(
        map.get(&(t_id, 792)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q99_id,
            addr_space: AddrSpace::Arena,
        }),
        "f99 at offset 792: {map:?}"
    );
}

/// 20 distinct `(source_struct, field_offset)` cast patterns in a
/// single program. Source struct `T` has 20 `u64` fields at
/// offsets `0, 8, ..., 152`. Each `T.f_i` is loaded then
/// dereferenced at offset `(i+1)` size 1, matching exactly one of
/// 20 distinct `Q_i` target structs (each with a single `u8` at
/// the matching offset). Verifies that the analyzer's `patterns`
/// map and the matcher's per-pattern intersection scale to many
/// distinct cast emissions in one walk.
#[test]
fn many_cast_patterns_in_one_program() {
    const N: u32 = 20;
    let mut strings: Vec<u8> = vec![0];
    let n_u8 = push_name(&mut strings, "u8");
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_x = push_name(&mut strings, "x");
    let mut t_field_names = Vec::with_capacity(N as usize);
    let mut q_names = Vec::with_capacity(N as usize);
    for i in 0..N {
        t_field_names.push(push_name(&mut strings, &format!("f{i}")));
        q_names.push(push_name(&mut strings, &format!("Q{i}")));
    }
    // Type ids:
    //   1: u8, 2: u64; 3: T (N u64 fields);
    //   4..=3+N: Q_i, each with single u8@(i+1).
    let t_members: Vec<SynMember> = (0..N)
        .map(|i| SynMember {
            name_off: t_field_names[i as usize],
            type_id: 2,
            byte_offset: 8 * i,
        })
        .collect();
    let mut types: Vec<SynType> = vec![
        SynType::Int {
            name_off: n_u8,
            size: 1,
            encoding: 0,
            offset: 0,
            bits: 8,
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
            size: 8 * N,
            members: t_members,
        },
    ];
    for i in 0..N {
        types.push(SynType::Struct {
            name_off: q_names[i as usize],
            size: i + 2, // u8@(i+1) requires size >= i+2
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: i + 1,
            }],
        });
    }
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id: u32 = 3;
    // Emit one (load + arena_cast + deref) triple per pattern for
    // arena-evidence mitigation: arena_confirmed evidence on each slot.
    let mut insns: Vec<BpfInsn> = Vec::with_capacity(3 * N as usize + 1);
    for i in 0..N {
        insns.push(ldx(BPF_SIZE_DW, 2, 1, (8 * i) as i16));
        insns.push(addr_space_cast(2, 2, 1));
        insns.push(ldx(BPF_SIZE_B, 3, 2, (i + 1) as i16));
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
    assert_eq!(map.len(), N as usize, "expected {N} cast patterns: {map:?}");
    for i in 0..N {
        let q_id: u32 = 4 + i;
        assert_eq!(
            map.get(&(t_id, 8 * i)),
            Some(&CastHit {
                alloc_size: None,
                target_type_id: q_id,
                addr_space: AddrSpace::Arena,
            }),
            "pattern #{i} at (T, {}) must resolve to Q{i}: {map:?}",
            8 * i
        );
    }
}

/// BTF with no struct types at all (only a single `u64` Int).
/// `build_layout_index` walks the id space without finding any
/// struct/union; `finalize` emits no findings. Verifies the
/// analyzer does not panic on a degenerate BTF that contains no
/// aggregate types, and that the empty layout index correctly
/// produces an empty `CastMap` even when the instruction stream
/// contains LDX patterns. The seed of `r1 = struct_type_id 1` is
/// silently dropped by `resolve_to_struct_id` (id 1 is `u64`).
#[test]
fn empty_btf_no_panic() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let types = vec![SynType::Int {
        name_off: n_u64,
        size: 8,
        encoding: 0,
        offset: 0,
        bits: 64,
    }];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: 1,
        }],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "no struct types in BTF must produce empty CastMap: {map:?}"
    );
}

/// BTF containing only `Int` types and no structs. The seed
/// targets a non-struct id, so `resolve_to_struct_id` returns
/// `None` and the seed is dropped. The instruction stream's LDX
/// cannot type any register, the `patterns` map stays empty, and
/// `build_layout_index` finds no struct/union to index. Verifies
/// the analyzer handles a scalar-only BTF without panic.
#[test]
fn btf_only_ints_no_panic() {
    let mut strings: Vec<u8> = vec![0];
    let n_u8 = push_name(&mut strings, "u8");
    let n_u16 = push_name(&mut strings, "u16");
    let n_u32 = push_name(&mut strings, "u32");
    let n_u64 = push_name(&mut strings, "u64");
    let n_s32 = push_name(&mut strings, "s32");
    // BTF int encoding bit `BTF_INT_SIGNED` per linux uapi `btf.h`.
    const BTF_INT_SIGNED: u32 = 1;
    let types = vec![
        SynType::Int {
            name_off: n_u8,
            size: 1,
            encoding: 0,
            offset: 0,
            bits: 8,
        },
        SynType::Int {
            name_off: n_u16,
            size: 2,
            encoding: 0,
            offset: 0,
            bits: 16,
        },
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
        SynType::Int {
            name_off: n_s32,
            size: 4,
            encoding: BTF_INT_SIGNED,
            offset: 0,
            bits: 32,
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    // Seed targets the u64 id — `resolve_to_struct_id` walks
    // Int as terminal and returns None. The seed silently drops.
    let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: 4,
        }],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "Int-only BTF must produce empty CastMap: {map:?}"
    );
}
