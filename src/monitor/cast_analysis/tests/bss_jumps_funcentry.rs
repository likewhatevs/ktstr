use super::*;

// ----- BSS / Datasec kptr detection ---------------------------
//
// These tests exercise the `BPF_PSEUDO_MAP_VALUE` path in the
// `BPF_LD_IMM64` arm. The tests build a synthetic BTF with a
// `BTF_KIND_DATASEC` over a `BTF_KIND_VAR` whose underlying
// type is a plain u64 — the BSS layout libbpf generates for
// `__u64 my_kptr;`. The `DatasecPointer` annotation passed to
// `analyze_casts` mirrors what the host-side cast loader
// emits after walking `.rel.text` against the program's
// datasec sections.

/// Build a synthetic BTF that declares a `BTF_KIND_DATASEC`
/// (`.bss`) containing a single u64 global variable
/// (`my_kptr`) at offset 0. Returns `(blob, datasec_id,
/// kptr_target_id, var_byte_offset, kfunc_btf_id)` where
/// `kptr_target_id` is a separate struct
/// (`task_struct`-stand-in) that the STX path stores INTO
/// the u64 slot.
///
/// Layout (BTF type ids assigned in order):
/// - id 1: int u64 (size=8, bits=64)
/// - id 2: struct task_struct { u64 x @ 0 }   -- kptr target
/// - id 3: T*
/// - id 4: BTF_KIND_VAR(name="my_kptr", type=1, linkage=GLOBAL)
/// - id 5: BTF_KIND_DATASEC(name=".bss", size=8, entries=[
///   {type=4, offset=0, size=8}])
/// - id 6: FuncProto returning T*
/// - id 7: Func("bpf_task_acquire") -> id 6
fn btf_bss_with_kptr() -> (Vec<u8>, u32, u32, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "task_struct");
    let n_x = push_name(&mut strings, "x");
    let n_kptr = push_name(&mut strings, "my_kptr");
    let n_bss = push_name(&mut strings, ".bss");
    let n_kfunc = push_name(&mut strings, "bpf_task_acquire");
    let types = vec![
        // id 1: u64
        SynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct task_struct { u64 x @ 0 }
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
        // id 4: Var("my_kptr", type=u64=1, linkage=GLOBAL)
        SynType::Var {
            name_off: n_kptr,
            type_id: 1,
            linkage: 1,
        },
        // id 5: Datasec(".bss") containing my_kptr at offset 0
        SynType::Datasec {
            name_off: n_bss,
            size: 8,
            entries: vec![SynVarSecinfo {
                type_id: 4,
                offset: 0,
                size: 8,
            }],
        },
        // id 6: FuncProto -> T*
        SynType::FuncProto {
            return_type_id: 3,
            params: vec![],
        },
        // id 7: Func bpf_task_acquire (linkage = global)
        SynType::Func {
            name_off: n_kfunc,
            type_id: 6,
            linkage: 1,
        },
    ];
    let blob = build_btf(&types, &strings);
    (blob, 5, 2, 0, 7)
}

/// `BPF_LD_IMM64` with a `DatasecPointer` annotation must type
/// the destination register as a typed pointer into the
/// datasec. The follow-up STX through that register records a
/// kptr finding keyed on `(datasec_id, var_byte_offset)`.
///
/// Sequence (mirrors clang's `my_kptr = bpf_task_acquire(...)`
/// codegen):
///   call kfunc bpf_task_acquire   ; r0 = T*
///   r1 = LD_IMM64(.bss, 0)        ; r1 = DatasecPointer{bss, 0}
///   *(u64 *)(r1 + 0) = r0         ; STX r0 into .bss[my_kptr]
///
/// Expected: CastMap entry
/// `(datasec_id, 0) -> (T, AddrSpace::Kernel)`.
#[test]
fn bss_kptr_records_kernel_cast() {
    let (blob, datasec_id, t_id, var_off, kfunc_id) = btf_bss_with_kptr();
    let btf = Btf::from_bytes(&blob).unwrap();
    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    let stx_kptr = stx(BPF_SIZE_DW, 1, 0, 0);
    let insns = vec![kfunc_call(kfunc_id), ld_lo, ld_hi, stx_kptr, exit()];
    // PC numbering: 0=call, 1=ld_lo, 2=ld_hi (skipped via
    // skip_next), 3=stx, 4=exit. The DatasecPointer marks PC=1
    // (the BPF_LD_IMM64 lo slot) as targeting the .bss
    // datasec at the my_kptr offset.
    let datasec_pointers = vec![DatasecPointer {
        insn_offset: 1,
        datasec_type_id: datasec_id,
        base_offset: var_off,
    }];
    let map = analyze_casts(&insns, &btf, &[], &[], &datasec_pointers, &[]);
    assert_eq!(
        map.get(&(datasec_id, var_off)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
        "kfunc-returned T* stored into .bss[my_kptr] must record \
             (datasec_id, 0) -> (T, Kernel): {map:?}"
    );
}

/// `BPF_LD_IMM64` WITHOUT a `DatasecPointer` annotation leaves
/// the destination register as `Unknown`, so a follow-up STX
/// through it cannot record a kptr finding. This guards
/// against a regression where the analyzer accidentally types
/// the LD_IMM64 destination as the global variable's
/// underlying integer type just from the BTF.
#[test]
fn ld_imm64_without_annotation_no_record() {
    let (blob, _datasec_id, _t_id, var_off, kfunc_id) = btf_bss_with_kptr();
    let btf = Btf::from_bytes(&blob).unwrap();
    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    let stx_kptr = stx(BPF_SIZE_DW, 1, 0, 0);
    let insns = vec![kfunc_call(kfunc_id), ld_lo, ld_hi, stx_kptr, exit()];
    // Empty datasec_pointers — analyzer has no way to recover
    // the parent datasec id, so the LD_IMM64 destination
    // stays Unknown.
    let map = analyze_casts(&insns, &btf, &[], &[], &[], &[]);
    assert!(
        map.is_empty(),
        "LD_IMM64 without DatasecPointer annotation must not record \
             a kptr finding: {map:?}"
    );
}

/// `BPF_LD_IMM64` with a `DatasecPointer` annotation but the
/// follow-up STX uses an untyped value register (literal
/// constant via mov_k) records nothing. The kptr path
/// requires both base AND value registers to be typed.
#[test]
fn bss_stx_with_untyped_value_no_record() {
    let (blob, datasec_id, _t_id, var_off, _kfunc_id) = btf_bss_with_kptr();
    let btf = Btf::from_bytes(&blob).unwrap();
    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    // r0 = literal 0 (mov_k clobbers any prior typed state)
    // *(u64 *)(r1 + 0) = r0      ; r0 Unknown -> no record
    let mov_zero = mov_k(0, 0);
    let stx_kptr = stx(BPF_SIZE_DW, 1, 0, 0);
    let insns = vec![ld_lo, ld_hi, mov_zero, stx_kptr, exit()];
    let datasec_pointers = vec![DatasecPointer {
        insn_offset: 0,
        datasec_type_id: datasec_id,
        base_offset: var_off,
    }];
    let map = analyze_casts(&insns, &btf, &[], &[], &datasec_pointers, &[]);
    assert!(
        map.is_empty(),
        "STX with untyped value register must not record kptr: {map:?}"
    );
}

/// Multi-variable BSS layout: a single datasec contains TWO
/// u64 globals at distinct offsets. The analyzer must key
/// each kptr finding on the right `(datasec_id, var_offset)`
/// pair without conflating them.
#[test]
fn bss_multi_variable_layout() {
    // BTF: u64(1), T(2, u64@0), T*(3), Var "kptr_a"(4),
    // Var "kptr_b"(5), Datasec(6, [(4,0,8), (5,16,8)]),
    // FuncProto(7), Func(8).
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "task_struct");
    let n_x = push_name(&mut strings, "x");
    let n_a = push_name(&mut strings, "kptr_a");
    let n_b = push_name(&mut strings, "kptr_b");
    let n_bss = push_name(&mut strings, ".bss");
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
        SynType::Var {
            name_off: n_a,
            type_id: 1,
            linkage: 1,
        },
        SynType::Var {
            name_off: n_b,
            type_id: 1,
            linkage: 1,
        },
        SynType::Datasec {
            name_off: n_bss,
            size: 24,
            entries: vec![
                SynVarSecinfo {
                    type_id: 4,
                    offset: 0,
                    size: 8,
                },
                SynVarSecinfo {
                    type_id: 5,
                    offset: 16,
                    size: 8,
                },
            ],
        },
        SynType::FuncProto {
            return_type_id: 3,
            params: vec![],
        },
        SynType::Func {
            name_off: n_kfunc,
            type_id: 7,
            linkage: 1,
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let datasec_id = 6;
    let t_id = 2;
    let kfunc_id = 8;
    let [ld_a_lo, ld_a_hi] = ld_imm64(1, 0);
    let [ld_b_lo, ld_b_hi] = ld_imm64(2, 16);
    let insns = vec![
        kfunc_call(kfunc_id),
        ld_a_lo,
        ld_a_hi,
        stx(BPF_SIZE_DW, 1, 0, 0),
        kfunc_call(kfunc_id),
        ld_b_lo,
        ld_b_hi,
        stx(BPF_SIZE_DW, 2, 0, 0),
        exit(),
    ];
    // PC numbering: 0=call, 1=ld_a_lo, 2=ld_a_hi, 3=stx_a,
    // 4=call, 5=ld_b_lo, 6=ld_b_hi, 7=stx_b, 8=exit.
    let datasec_pointers = vec![
        DatasecPointer {
            insn_offset: 1,
            datasec_type_id: datasec_id,
            base_offset: 0,
        },
        DatasecPointer {
            insn_offset: 5,
            datasec_type_id: datasec_id,
            base_offset: 16,
        },
    ];
    let map = analyze_casts(&insns, &btf, &[], &[], &datasec_pointers, &[]);
    assert_eq!(
        map.get(&(datasec_id, 0)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
        "kptr_a at offset 0: {map:?}"
    );
    assert_eq!(
        map.get(&(datasec_id, 16)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
        "kptr_b at offset 16: {map:?}"
    );
}

/// `struct_member_at` on a Datasec parent finds the variable
/// whose byte range contains the queried offset. A query
/// inside a multi-byte variable's range returns the
/// variable's start offset, NOT the queried offset, in
/// `MemberAt::Datasec::var_byte_offset`. A query that lands
/// outside any variable's range returns None.
#[test]
fn struct_member_at_datasec_resolves_variables() {
    let (blob, datasec_id, _t_id, _var_off, _kfunc_id) = btf_bss_with_kptr();
    let btf = Btf::from_bytes(&blob).unwrap();
    // Exact-offset hit: my_kptr starts at byte 0.
    let m0 = struct_member_at(&btf, datasec_id, 0).expect("byte 0 must hit my_kptr");
    match m0 {
        MemberAt::Datasec {
            var_byte_offset, ..
        } => assert_eq!(var_byte_offset, 0),
        MemberAt::Struct { .. } => panic!("Datasec parent must yield Datasec match"),
    }
    // Mid-variable hit: byte 4 lands inside my_kptr's [0, 8)
    // range; should return the variable's start (0).
    let m4 = struct_member_at(&btf, datasec_id, 4).expect("byte 4 must hit my_kptr range");
    match m4 {
        MemberAt::Datasec {
            var_byte_offset, ..
        } => assert_eq!(var_byte_offset, 0),
        MemberAt::Struct { .. } => panic!("Datasec parent must yield Datasec match"),
    }
    // Out-of-range hit: byte 100 is past the section.
    assert!(
        struct_member_at(&btf, datasec_id, 100).is_none(),
        "byte 100 outside section must return None"
    );
}

/// End-to-end: a BSS u64 stores a kfunc-returned pointer
/// (mirrors `__u64 my_kptr; my_kptr = bpf_task_acquire(...)`
/// at the analyzer level). Produces exactly one CastMap entry
/// keyed on `(datasec_id, 0)` -> `(task_struct, Kernel)`.
#[test]
fn end_to_end_bss_global_stores_kfunc_pointer() {
    let (blob, datasec_id, t_id, var_off, kfunc_id) = btf_bss_with_kptr();
    let btf = Btf::from_bytes(&blob).unwrap();
    let [ld_lo, ld_hi] = ld_imm64(1, var_off as i32);
    let insns = vec![
        kfunc_call(kfunc_id),
        ld_lo,
        ld_hi,
        stx(BPF_SIZE_DW, 1, 0, 0),
        exit(),
    ];
    let datasec_pointers = vec![DatasecPointer {
        insn_offset: 1,
        datasec_type_id: datasec_id,
        base_offset: var_off,
    }];
    let map = analyze_casts(&insns, &btf, &[], &[], &datasec_pointers, &[]);
    assert_eq!(map.len(), 1, "exactly one finding expected: {map:?}");
    assert_eq!(
        map.get(&(datasec_id, var_off)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Kernel,
        }),
    );
}

// ----- Edge case tests: kfunc imm=0 ---------------------------

/// `handle_kfunc_call` short-circuits on `imm <= 0`. Typically
/// `imm = -1` for an unrelocated kfunc placeholder; `imm = 0`
/// also hits the short-circuit. R0 stays Unknown after the
/// standard R0..R5 clobber.
#[test]
fn kfunc_call_imm_zero_leaves_r0_unknown() {
    let slot_off: u32 = 16;
    let (blob, _t_id, p_id, _t_ptr_id) = btf_kptr_base(slot_off);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![
        kfunc_call(0),
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
        "kfunc_call imm=0 must leave R0 Unknown: {map:?}"
    );
}

// ----- Edge case tests: jumps ---------------------------------

/// `BPF_JMP32 | BPF_JA` (gotol, op=0x00) uses `insn.imm` as the
/// 32-bit jump offset per `jump_targets`. The target PC is reset.
#[test]
fn jmp32_gotol_resets_state_at_target() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // pc 0: r2 = T.f (LoadedU64Field).
    // pc 1: gotol +1 (JMP32|JA, imm=1). Target = pc 3.
    // pc 2: exit (skipped).
    // pc 3: r3 = *(u64 *)(r2 + 0) — state reset, no record.
    let gotol = mk_insn(BPF_CLASS_JMP32, 0, 0, 0, 1);
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        gotol,
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
    assert!(map.is_empty(), "JMP32|JA target must reset state: {map:?}");
}

/// Out-of-range jump targets (negative resolved address, or past
/// `insns.len()`) are silently dropped. State survives.
#[test]
fn out_of_range_jump_targets_dropped() {
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let jeq_neg = mk_insn(BPF_CLASS_JMP | 0x10, 2, 0, -100, 0);
    let jeq_pos = mk_insn(BPF_CLASS_JMP | 0x10, 2, 0, 100, 0);
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 8),
        addr_space_cast(2, 2, 1),
        jeq_neg,
        jeq_pos,
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
        "out-of-range jumps must drop, state survives: {map:?}"
    );
}

/// All conditional jump opcodes register their targets per
/// `jump_targets`. JEQ, JGT, JGE, JSET, JNE, JSGT, JSGE, JLT,
/// JLE, JSLT, JSLE — each one's target PC must reset state.
#[test]
fn all_conditional_jumps_register_targets() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let ops: [u8; 11] = [
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0xa0, 0xb0, 0xc0, 0xd0,
    ];
    for op in ops {
        let cond = mk_insn(BPF_CLASS_JMP | op, 2, 0, 1, 0);
        let insns = vec![
            ldx(BPF_SIZE_DW, 2, 1, 8),
            cond,
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
        assert!(
            map.is_empty(),
            "JMP op 0x{op:02x} target must reset state: {map:?}"
        );
    }
}

// ----- Edge case tests: FuncEntry -----------------------------

/// Multiple `FuncEntry` entries at the same PC are processed in
/// order — last one wins. Each entry's `seed_from_func_proto`
/// clears all registers before seeding.
/// Two FuncProtos at PC 0:
///   A: ([T*, P*]) — seeds R1=T*, R2=P*.
///   B: ([P*, T*]) — seeds R1=P*, R2=T*.
/// With B processed second, R1=P* and R2=T*. Records (P, slot) -> T.
#[test]
fn func_entry_multiple_at_same_pc_last_wins() {
    let slot_off: u32 = 16;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_p = push_name(&mut strings, "P");
    let n_x = push_name(&mut strings, "x");
    let n_slot = push_name(&mut strings, "slot");
    let n_arg_t = push_name(&mut strings, "arg_t");
    let n_arg_p = push_name(&mut strings, "arg_p");
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
        SynType::Ptr { type_id: 4 },
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
        SynType::FuncProto {
            return_type_id: 0,
            params: vec![
                SynParam {
                    name_off: n_arg_p,
                    type_id: 5,
                },
                SynParam {
                    name_off: n_arg_t,
                    type_id: 3,
                },
            ],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let p_id = 4;
    let proto_a = 6;
    let proto_b = 7;
    let insns = vec![stx(BPF_SIZE_DW, 1, 2, slot_off as i16), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[],
        &[
            FuncEntry {
                insn_offset: 0,
                func_proto_id: proto_a,
            },
            FuncEntry {
                insn_offset: 0,
                func_proto_id: proto_b,
            },
        ],
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
        "later FuncEntry at same PC must win: {map:?}"
    );
}

/// `FuncEntry` with `insn_offset` past `insns.len()` is silently
/// skipped — the loop never finds a matching PC.
#[test]
fn func_entry_past_insns_len_no_op() {
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
        &[FuncEntry {
            insn_offset: 999,
            func_proto_id: 1,
        }],
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
        "FuncEntry past insns.len() must not affect run: {map:?}"
    );
}

/// `FuncEntry` at PC 0 with no params clears all registers, then
/// iterates an empty param list (no seeding). InitialReg state
/// is wiped.
#[test]
fn func_entry_pc0_no_params_clears_initial_regs() {
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
        SynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynType::FuncProto {
            return_type_id: 0,
            params: vec![],
        },
    ];
    let blob = build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).unwrap();
    let t_id = 2;
    let proto_id = 4;
    let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: t_id,
        }],
        &[FuncEntry {
            insn_offset: 0,
            func_proto_id: proto_id,
        }],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "FuncEntry with empty params must clear all regs: {map:?}"
    );
}

/// `FuncEntry` with `func_proto_id == 0` (Void) hits the
/// `_ => return` arm — but only AFTER all registers are cleared.
#[test]
fn func_entry_proto_id_zero_clears_regs_no_seed() {
    let (blob, t_id, _q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: t_id,
        }],
        &[FuncEntry {
            insn_offset: 0,
            func_proto_id: 0,
        }],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "FuncEntry with proto_id=0 must clear regs and not seed: {map:?}"
    );
}

/// `FuncEntry` at PC > 0 reseeds at the matching PC mid-stream.
#[test]
fn func_entry_pc_gt_0_reseeds_mid_stream() {
    let slot_off: u32 = 16;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_p = push_name(&mut strings, "P");
    let n_x = push_name(&mut strings, "x");
    let n_slot = push_name(&mut strings, "slot");
    let n_arg_t = push_name(&mut strings, "arg_t");
    let n_arg_p = push_name(&mut strings, "arg_p");
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
        SynType::Ptr { type_id: 4 },
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
    // pc 0: exit. pc 1: STX *(R2 + slot_off) = R1.
    // FuncEntry at PC 1 reseeds R1=T*, R2=P*. Records (P, slot) -> T.
    let insns = vec![exit(), stx(BPF_SIZE_DW, 2, 1, slot_off as i16), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[],
        &[FuncEntry {
            insn_offset: 1,
            func_proto_id: proto_id,
        }],
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
        "FuncEntry at PC>0 must reseed: {map:?}"
    );
}

// ----- Misc edge case tests -----------------------------------

/// The second slot of `BPF_LD_IMM64` is skipped per `skip_next`.
/// Even non-zero content must not be interpreted as instruction.
#[test]
fn ld_imm64_second_slot_with_non_zero_content_skipped() {
    let (blob, t_id, q_id) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    // pc 0: LD_IMM64 first slot (with non-zero imm = 42).
    // pc 1: second slot — emit a fake "instruction" that LOOKS
    //       like an ALU64|MOV|X. It must be skipped.
    // pc 2: r2 = T.f
    // pc 3: r3 = *r2
    let ld_imm64_lo = mk_insn(BPF_CLASS_LD | BPF_SIZE_DW | BPF_MODE_IMM, 6, 0, 0, 42);
    let fake_mov = mk_insn(BPF_CLASS_ALU64 | BPF_OP_MOV | BPF_SRC_X, 4, 3, 0, 0);
    let insns = vec![
        ld_imm64_lo,
        fake_mov,
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
        "non-zero LD_IMM64 second slot must skip: {map:?}"
    );
}

/// `seed` iterates `initial_regs` in order; later seeds for the
/// same register overwrite earlier ones (last wins).
#[test]
fn initial_reg_duplicate_seeds_last_wins() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = push_name(&mut strings, "u64");
    let n_s1 = push_name(&mut strings, "S1");
    let n_s2 = push_name(&mut strings, "S2");
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
        // id 2: S1 { u64 f @ 8 }, size 16
        SynType::Struct {
            name_off: n_s1,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // id 3: S2 { u64 f @ 16 }, size 24
        SynType::Struct {
            name_off: n_s2,
            size: 24,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 16,
            }],
        },
        // id 4: Q { u64 x @ 0 }, size 8
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
    let s1_id = 2;
    let s2_id = 3;
    let q_id = 4;
    // Seed R1 first as S1, then as S2 — last wins, R1 = S2.
    // Sequence: r2 = *(u64*)(r1+16) = S2.f, cast (arena evidence),
    // then r3 = *r2 at 0. Records (S2, 16) -> Q. If first seed
    // had won, S1 has no field at offset 16, so no record.
    let insns = vec![
        ldx(BPF_SIZE_DW, 2, 1, 16),
        addr_space_cast(2, 2, 1),
        ldx(BPF_SIZE_DW, 3, 2, 0),
        exit(),
    ];
    let map = analyze_casts(
        &insns,
        &btf,
        &[
            InitialReg {
                reg: 1,
                struct_type_id: s1_id,
            },
            InitialReg {
                reg: 1,
                struct_type_id: s2_id,
            },
        ],
        &[],
        &[],
        &[],
    );
    assert_eq!(
        map.get(&(s2_id, 16)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "duplicate InitialReg seed must use last value: {map:?}"
    );
    assert!(
        !map.contains_key(&(s1_id, 16)),
        "first InitialReg seed must NOT take effect: {map:?}"
    );
}

/// `InitialReg` with `struct_type_id == 0` is silently dropped.
#[test]
fn initial_reg_struct_type_id_zero_dropped() {
    let (blob, _t, _q) = btf_with_source_and_target(8, 0);
    let btf = Btf::from_bytes(&blob).unwrap();
    let insns = vec![ldx(BPF_SIZE_DW, 2, 1, 8), ldx(BPF_SIZE_DW, 3, 2, 0), exit()];
    let map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: 0,
        }],
        &[],
        &[],
        &[],
    );
    assert!(
        map.is_empty(),
        "InitialReg with struct_type_id=0 must be dropped: {map:?}"
    );
}

