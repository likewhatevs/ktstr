use super::*;

// ---- Cast pipeline integration ---------------------------------
//
// The tests above stub `cast_lookup` with a fixed [`CastHit`]. The
// integration tests below run real
// [`crate::monitor::cast_analysis::analyze_casts`] over a synthetic
// BPF program, feed the resulting [`crate::monitor::cast_analysis::CastMap`]
// into [`CastStubReader::cast_map`], and render a struct against
// the analyzer's actual output. The point: verify the
// `(parent_type_id, member_byte_offset)` keys the analyzer emits
// match the keys [`render_member`]'s cast intercept queries via
// [`MemReader::cast_lookup`]. A drift between analyzer and
// renderer (e.g. analyzer keys on a typedef-wrapped surface id,
// renderer queries with the peeled struct id) would surface as a
// missed intercept here even though both halves work in
// isolation.
//
// The fixtures use the same [`cast_build_btf`] /
// [`CastSynType`] machinery the gate-focused tests above use,
// extended with `Typedef` and `Const` for the modifier-chain case.
// BPF instruction encoding goes through [`BpfInsn::new`]; opcode
// constants come from `libbpf_rs::libbpf_sys` (the same source
// the analyzer's own private constants use, so the test
// instruction stream stays in lock-step with the analyzer's
// decode tables).

use crate::monitor::cast_analysis::{BpfInsn, CastMap, InitialReg, analyze_casts};

/// `BPF_LDX | BPF_DW | BPF_MEM` in the [`BpfInsn::code`] byte. The
/// arena-cast LDX shape the analyzer's `handle_ldx` matches: load
/// 8 bytes through a typed pointer base. Constants pulled from
/// `libbpf_rs::libbpf_sys` so the test encoding stays in lock-step
/// with the analyzer's own decode tables.
fn cast_ldx_dw_mem_code() -> u8 {
    use libbpf_rs::libbpf_sys as bs;
    (bs::BPF_LDX | bs::BPF_DW | bs::BPF_MEM) as u8
}

/// `BPF_JMP | BPF_EXIT` in the [`BpfInsn::code`] byte. Terminator
/// for synthetic programs.
fn cast_exit_code() -> u8 {
    use libbpf_rs::libbpf_sys as bs;
    (bs::BPF_JMP | bs::BPF_EXIT) as u8
}

/// One-shot helper: emit `r{dst} = *(u64 *)(r{src} + off)` as a
/// single [`BpfInsn`] for cast-integration tests.
fn cast_ldx_dw(dst: u8, src: u8, off: i16) -> BpfInsn {
    BpfInsn::new(cast_ldx_dw_mem_code(), dst, src, off, 0)
}

/// One-shot helper: emit `exit` as a single [`BpfInsn`].
fn cast_exit() -> BpfInsn {
    BpfInsn::new(cast_exit_code(), 0, 0, 0, 0)
}

/// One-shot helper: emit `BPF_ADDR_SPACE_CAST` (ALU64 | MOV | X
/// with `off=1`). The analyzer treats `imm=1` as the as(1)→as(0)
/// cast (arena→kernel), which adds the source's `(struct,
/// field_offset)` to `arena_confirmed` — the arena-evidence
/// prerequisite for shape-inference findings.
fn cast_addr_space_cast(dst: u8, src: u8, imm: i32) -> BpfInsn {
    use libbpf_rs::libbpf_sys as bs;
    let code = (bs::BPF_ALU64 | bs::BPF_MOV | bs::BPF_X) as u8;
    BpfInsn::new(code, dst, src, 1, imm)
}

/// Build a BTF blob shaped to drive [`analyze_casts`] to a unique
/// (T, 8) → (Q, Arena) finding when run against the canonical
/// `r2 = T.f; r3 = *r2` pair. Layout:
///
///   id=1: u64 (size 8, plain unsigned)
///   id=2: struct T { u64 f @ offset 8 }, size 16
///   id=3: struct Q { u64 x @ offset 0 }, size 8
///
/// T's u64 lives at offset 8 (not 0), so the access pattern
/// `(offset=0, size=8)` from `*r2` only matches Q in the layout
/// index — the analyzer's source-removal step doesn't fire (T is
/// not in the candidate set), so the single-candidate condition
/// emits the entry.
fn cast_btf_t_at_offset_8_q_at_offset_0() -> (Vec<u8>, u32, u32) {
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let types = vec![
        // id 1: u64 plain unsigned.
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct T { u64 f @ 8 }, size 16. Offset 8 keeps T
        // out of the (offset=0, size=8) layout-index bucket, so the
        // analyzer's intersection collapses to {Q} cleanly.
        CastSynType::Struct {
            name_off: n_t,
            size: 16,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        // id 3: struct Q { u64 x @ 0 }, size 8.
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    (cast_build_btf(&types, &strings), 2, 3)
}

/// First true integration test: real [`analyze_casts`] output drives
/// the renderer's cast intercept end-to-end. Verifies that the
/// `(parent_type_id, member_byte_offset)` keys the analyzer emits
/// match the keys [`render_member`] queries via
/// [`MemReader::cast_lookup`], and that the renderer chases the
/// recovered target through [`render_cast_pointer`] to produce a
/// `Ptr` with a populated `deref`.
///
/// A drift between analyzer key format and renderer query format
/// would surface here as a missed intercept (the u64 field
/// rendered as `Uint` instead of `Ptr`), even though both halves
/// work in isolation.
#[test]
fn cast_pipeline_analyzer_output_drives_renderer_intercept() {
    let (blob, t_id, q_id) = cast_btf_t_at_offset_8_q_at_offset_0();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // BPF program:
    //   r2 = *(u64 *)(r1 + 8)   ; load T.f → r2 = LoadedU64Field{T, 8}
    //   r2 = arena_cast(r2)     ; arena_confirmed evidence (arena evidence)
    //   r3 = *(u64 *)(r2 + 0)   ; deref @0 records access (0, 8) under (T, 8)
    //   exit
    let insns = vec![
        cast_ldx_dw(2, 1, 8),
        cast_addr_space_cast(2, 2, 1),
        cast_ldx_dw(3, 2, 0),
        cast_exit(),
    ];
    let cast_map = analyze_casts(
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
    // Analyzer must produce exactly one finding: (T, 8) → (Q, Arena).
    assert_eq!(
        cast_map.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        "analyzer must emit (T, 8) → (Q, Arena); got: {cast_map:?}"
    );

    // Render T's bytes (16 bytes: 8 padding + arena address at
    // offset 8) with the analyzer's CastMap as the lookup source.
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let mut outer_bytes = vec![0u8; 16];
    outer_bytes[8..16].copy_from_slice(&TARGET_ADDR.to_le_bytes());
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();

    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct {
        type_name,
        ref members,
    } = v
    else {
        panic!("expected outer Struct render, got {v:?}");
    };
    assert_eq!(type_name.as_deref(), Some("T"));
    assert_eq!(members.len(), 1, "T has a single u64 member at offset 8");
    assert_eq!(members[0].name, "f");
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "(T, 8) cast hit must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR, "Ptr value must be the loaded u64");
    assert!(
        deref_skipped_reason.is_none(),
        "successful chase must carry no skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("chase succeeded → deref must be Some");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the rendered Q struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "deref payload must carry Q's name"
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "x");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x42);
}

/// Modifier-chain integration: render a `Typedef → Const → Struct(T)`
/// surface type. The renderer's [`peel_modifiers_with_id`] must
/// peel both wrappers and emit `parent_type_id = T_id` (the
/// underlying struct) so [`MemReader::cast_lookup`] queries with
/// the same id the analyzer keyed on. Catches the fragile coupling
/// where a future change to one peel path (renderer or analyzer)
/// drifts away from the other.
#[test]
fn cast_pipeline_modifier_chain_renderer_peels_to_analyzer_struct_id() {
    // Layout extends `cast_btf_t_at_offset_8_q_at_offset_0` with two
    // modifier wrappers: id=4 = Const(T), id=5 = Typedef(Const(T)).
    // The analyzer still keys on T_id=2 (struct id) because both
    // `bpf_map::resolve_to_struct_id` (used by InitialReg seeding)
    // and `peel_modifiers_with_id` (used by render_struct) collapse
    // the wrapper chain.
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    // Add a typedef name distinct from the existing strings so the
    // wrapper carries an identifiable name on the wire (the
    // renderer doesn't surface it because peel collapses the
    // wrapper, but the BTF blob round-trips correctly).
    let mut strings = strings;
    let n_typedef = push(&mut strings, "T_alias");
    let types = vec![
        // id 1..3 mirror cast_btf_t_at_offset_8_q_at_offset_0.
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 16,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 4: const(T) — wraps T_id=2.
        CastSynType::Const { type_id: 2 },
        // id 5: typedef T_alias = const(T) — wraps id 4. Render via
        // this id; peel_modifiers_with_id collapses 5→4→2.
        CastSynType::Typedef {
            name_off: n_typedef,
            type_id: 4,
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;
    let typedef_id: u32 = 5;

    // Analyzer is run against the SAME BTF; the wrappers are inert
    // for analyze_casts (they're not Struct/Union, so the layout
    // index skips them). InitialReg seeds with the typedef wrapper
    // to verify the analyzer's `resolve_to_struct_id` peels through
    // the same chain the renderer does.
    // arena-evidence mitigation: include arena_space_cast on r2 to
    // populate arena_confirmed for (T, 8) so the shape-inference
    // finding emits.
    let insns = vec![
        cast_ldx_dw(2, 1, 8),
        cast_addr_space_cast(2, 2, 1),
        cast_ldx_dw(3, 2, 0),
        cast_exit(),
    ];
    let cast_map = analyze_casts(
        &insns,
        &btf,
        &[InitialReg {
            reg: 1,
            struct_type_id: typedef_id,
        }],
        &[],
        &[],
        &[],
    );
    assert_eq!(
        cast_map.get(&(t_id, 8)),
        Some(&CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena
        }),
        "analyzer must peel typedef→const→struct and key on T_id={t_id}; got: {cast_map:?}"
    );

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let mut outer_bytes = vec![0u8; 16];
    outer_bytes[8..16].copy_from_slice(&TARGET_ADDR.to_le_bytes());
    let inner_bytes = 0x99u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    // Render via the typedef id — the renderer's peel must produce
    // parent_type_id=T_id so the cast lookup hits.
    let v = render_value_with_mem(&btf, typedef_id, &outer_bytes, &reader);
    let RenderedValue::Struct {
        type_name,
        ref members,
    } = v
    else {
        panic!("expected Struct render after peel, got {v:?}");
    };
    assert_eq!(
        type_name.as_deref(),
        Some("T"),
        "renderer must collapse typedef/const wrappers to underlying T name"
    );
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "modifier-chain peel must reach the cast intercept; got {:?}. \
             A failure here means the renderer's peel diverges from the \
             analyzer's — the integration is broken.",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(deref_skipped_reason.is_none());
    let inner = deref.as_deref().expect("chase deref Some");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        ..
    } = *inner
    else {
        panic!("deref payload must be Q struct, got {inner:?}");
    };
    assert_eq!(inner_name.as_deref(), Some("Q"));
}

/// Multi-field integration: a struct with three u64 fields where
/// only two are flagged by the analyzer must render the flagged
/// fields as `Ptr` and the third as `Uint`. The point: per-member
/// cast lookup is independent — a hit on one offset must not
/// promote unrelated u64 fields.
#[test]
fn cast_pipeline_multi_field_only_flagged_offsets_render_as_ptr() {
    // Layout: T has u64 fields at offsets 0, 8, 16. Q is a generic
    // 8-byte target (u64 @ 0).
    //   id=1: u64
    //   id=2: struct T { u64 f0 @ 0; u64 f1 @ 8; u64 f2 @ 16 }, size 24
    //   id=3: struct Q { u64 x @ 0 }, size 8
    let (strings, n_int, n_t, n_q, _n_f, n_x) = cast_strings_for_t_q();
    // Add three field names distinct from `f`/`x`.
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let mut strings = strings;
    let n_f0 = push(&mut strings, "f0");
    let n_f1 = push(&mut strings, "f1");
    let n_f2 = push(&mut strings, "f2");
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 24,
            members: vec![
                CastSynMember {
                    name_off: n_f0,
                    type_id: 1,
                    byte_offset: 0,
                },
                CastSynMember {
                    name_off: n_f1,
                    type_id: 1,
                    byte_offset: 8,
                },
                CastSynMember {
                    name_off: n_f2,
                    type_id: 1,
                    byte_offset: 16,
                },
            ],
        },
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;

    // Build a CastMap by hand with entries for offsets 0 and 8
    // ONLY (NOT 16). This mirrors a partial-coverage analyzer
    // result: some u64 fields recovered, others missed (false
    // negatives are the safe direction for the analyzer).
    let mut cast_map: CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );
    cast_map.insert(
        (t_id, 8),
        CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );

    // Outer T bytes: arena addresses at offsets 0 and 8, plain
    // u64 counter at offset 16. The renderer must NOT chase the
    // counter even though it could be misinterpreted as an arena
    // address (it falls in the arena window for stress purposes).
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const ADDR_F0: u64 = 0x10_0000_1000;
    const ADDR_F1: u64 = 0x10_0000_2000;
    const COUNTER_F2: u64 = 0x10_0000_3000;
    let mut outer_bytes = vec![0u8; 24];
    outer_bytes[0..8].copy_from_slice(&ADDR_F0.to_le_bytes());
    outer_bytes[8..16].copy_from_slice(&ADDR_F1.to_le_bytes());
    outer_bytes[16..24].copy_from_slice(&COUNTER_F2.to_le_bytes());

    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(ADDR_F0, 0xAAu64.to_le_bytes().to_vec());
    arena_bytes.insert(ADDR_F1, 0xBBu64.to_le_bytes().to_vec());
    // Note: deliberately NO entry at COUNTER_F2 — even if the
    // intercept were buggy and fired for f2, the chase would
    // surface a `read_arena returned None` skip reason rather
    // than a deref payload. The test's primary check is that f2
    // renders as Uint (no intercept attempted).
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    assert_eq!(members.len(), 3, "T has three u64 members");
    assert_eq!(members[0].name, "f0");
    assert_eq!(members[1].name, "f1");
    assert_eq!(members[2].name, "f2");

    // f0 (cast hit at offset 0): Ptr with deref Some.
    let RenderedValue::Ptr {
        value: f0_value,
        ref deref,
        ..
    } = members[0].value
    else {
        panic!(
            "f0 (offset 0) must render as Ptr (cast map hit); got {:?}",
            members[0].value
        );
    };
    assert_eq!(f0_value, ADDR_F0);
    assert!(deref.is_some(), "f0 chase must succeed (deref Some)");

    // f1 (cast hit at offset 8): Ptr with deref Some.
    let RenderedValue::Ptr {
        value: f1_value,
        ref deref,
        ..
    } = members[1].value
    else {
        panic!(
            "f1 (offset 8) must render as Ptr (cast map hit); got {:?}",
            members[1].value
        );
    };
    assert_eq!(f1_value, ADDR_F1);
    assert!(deref.is_some(), "f1 chase must succeed (deref Some)");

    // f2 (NO cast entry at offset 16): plain Uint counter.
    let RenderedValue::Uint {
        bits: f2_bits,
        value: f2_value,
    } = members[2].value
    else {
        panic!(
            "f2 (offset 16) must render as Uint (no cast map entry); \
             got {:?}. A failure here means a hit on one offset is \
             contaminating unrelated offsets in the same struct.",
            members[2].value
        );
    };
    assert_eq!(f2_bits, 64);
    assert_eq!(f2_value, COUNTER_F2);
}

/// Empty-CastMap integration: a [`CastMap`] with no entries (the
/// no-cast case for a scheduler whose program contains zero
/// recovered casts) must leave every u64 field rendering as a
/// plain `Uint`. Verifies the lookup miss path through the real
/// `BTreeMap::get` returns `None` exactly as the trait-default
/// `cast_lookup` does, so deploying an empty analyzer result is
/// behaviorally indistinguishable from no analyzer at all.
#[test]
fn cast_pipeline_empty_cast_map_renders_uint() {
    let (blob, t_id, _q_id) = cast_btf_t_at_offset_8_q_at_offset_0();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Empty CastMap. cast_lookup must return None for every query.
    let cast_map: CastMap = std::collections::BTreeMap::new();
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        ..Default::default()
    };

    // T has u64 at offset 8; render with a counter-shaped value
    // there. With no cast entries, the field must render as a
    // plain Uint regardless of the value's numeric range.
    let mut outer_bytes = vec![0u8; 16];
    outer_bytes[8..16].copy_from_slice(&0xCAFE_F00Du64.to_le_bytes());
    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Uint { bits, value } = members[0].value else {
        panic!(
            "empty cast map must leave u64 as Uint; got {:?}. A \
             failure here means an empty BTreeMap is being treated \
             as 'wildcard hit' instead of 'no hits' — a regression \
             that would promote every u64 to a phantom pointer.",
            members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0xCAFE_F00D);
}

/// Wrong-struct integration: a [`CastMap`] entry keyed on a
/// struct id DIFFERENT from the one being rendered must NOT
/// fire. The renderer queries `cast_lookup` with the parent
/// struct's id; an entry for an unrelated struct (even at the
/// same byte offset, with the same target type) is a miss.
/// Catches the regression where the lookup ignored
/// `parent_type_id` and matched on offset alone.
#[test]
fn cast_pipeline_wrong_struct_id_does_not_intercept() {
    // Layout: two distinct structs with identical shape (u64 @ 0).
    // The CastMap entry is keyed on U; the test renders T. The
    // intercept must NOT fire on T even though T's offset-0 u64
    // is structurally identical to U's.
    //   id=1: u64
    //   id=2: struct T { u64 f @ 0 }, size 8
    //   id=3: struct Q { u64 x @ 0 }, size 8 (cast target)
    //   id=4: struct U { u64 g @ 0 }, size 8 (the "wrong" parent)
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let mut strings = strings;
    let n_u = push(&mut strings, "U");
    let n_g = push(&mut strings, "g");
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: T { u64 f @ 0 }
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: Q { u64 x @ 0 } (cast target — present so the
        // CastMap entry references a real id, even though the
        // intercept must not fire for this test).
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 4: U { u64 g @ 0 } — the unrelated parent the cast
        // map is keyed on.
        CastSynType::Struct {
            name_off: n_u,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_g,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;
    let u_id: u32 = 4;

    // CastMap entry keyed on (U, 0) — NOT T. Rendering T must
    // miss the lookup.
    let mut cast_map: CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (u_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );

    // Reader is configured with an arena window and bytes for the
    // value, so a buggy renderer that ignored parent_type_id and
    // intercepted anyway would chase successfully — surfacing the
    // bug as a Ptr render. The correct renderer treats the field
    // as Uint.
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const VAL: u64 = 0x10_0000_1000;
    let outer_bytes = VAL.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(VAL, 0x77u64.to_le_bytes().to_vec());
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Uint { bits, value } = members[0].value else {
        panic!(
            "(T, 0) must miss the (U, 0) cast entry → render as \
             Uint; got {:?}. A failure here means cast_lookup is \
             ignoring parent_type_id, which would promote every \
             u64 at the entry's offset across every struct in the \
             scheduler.",
            members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, VAL);
}

// ---- render_cast_pointer runtime address-space dispatch --------
//
// `render_cast_pointer` dispatches on [`MemReader::is_arena_addr`]
// against the actual pointer value, not the analyzer's `AddrSpace`
// hint: an in-window value enters [`chase_arena_pointer`], an
// out-of-window value enters the kernel arm regardless of the hint
// (the hint is preserved in `cast_annotation` only when the chase
// could not be performed). These tests pin the runtime fallthrough
// so a regression that re-introduced an early "outside arena
// window" skip would surface here.

/// Arena cast hit whose value falls OUTSIDE the configured arena
/// window: the runtime `is_arena_addr` check in
/// [`render_cast_pointer`] is false, so the renderer falls through
/// to the kernel arm via runtime address-space detection. With no
/// kva entry configured, `read_kva` returns None and the renderer
/// surfaces `Ptr{ deref: None, deref_skipped_reason:
/// Some("kernel read_kva failed at 0x...") }` with
/// `cast_annotation: Some("cast→kernel")`. Pins the runtime
/// fallthrough — a regression that re-introduced an early "outside
/// arena window" skip would block legitimate kernel chases
/// whenever the analyzer's `AddrSpace` tag drifted.
#[test]
fn cast_chase_arena_hint_with_non_arena_value_falls_through_to_kernel_arm() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    // Value below the arena window; `is_arena_addr` returns false,
    // dispatching to the kernel arm. No kva entry → `read_kva`
    // fails, surfacing as a labelled skip with cast→kernel
    // annotation.
    const OUT_OF_WINDOW: u64 = 0x0F_FFFF_FFFF;
    let outer_bytes = OUT_OF_WINDOW.to_le_bytes().to_vec();
    let mut cast_map: crate::monitor::cast_analysis::CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        ..Default::default()
    };
    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!("must surface as Ptr; got {:?}", members[0].value);
    };
    assert_eq!(value, OUT_OF_WINDOW);
    assert!(deref.is_none());
    let reason = deref_skipped_reason.as_deref().expect("skip reason");
    // Runtime fallthrough → kernel arm; read_kva has no entry, so
    // the kernel-read-failed reason fires. The reason includes the
    // "(cast analysis may have flagged a non-pointer field)" suffix
    // because the original hint was Arena, indicating the analyzer
    // may have mis-tagged the slot.
    assert!(
        reason.contains("read_kva failed"),
        "skip reason must mention 'read_kva failed' (kernel arm); got: {reason}"
    );
    assert!(
        reason.contains("cast analysis may have flagged"),
        "Arena→kernel runtime dispatch must annotate suffix; got: {reason}"
    );
    // cast_annotation reflects the runtime decision (kernel), not
    // the analyzer's hint (Arena).
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→kernel"),
        "runtime kernel dispatch must produce cast→kernel annotation"
    );
}
