use super::*;

// ---- Fwd → complete-sibling resolution ------------------------
//
// `peel_modifiers_resolving_fwd` looks up a [`Type::Fwd`] terminal
// by name in the same BTF and prefers a complete
// [`Type::Struct`] / [`Type::Union`] sibling of matching aggregate
// kind. The chase pipeline uses this so a forward-declared
// pointee whose body lives one BTF id away (a routine outcome of
// concatenated BPF object files where one .bpf.c only declares
// the type while another defines it) renders as a chased struct
// rather than skipping with "forward declaration; body not in
// this BTF". Tests below exercise:
//   - Fwd + complete Struct siblings of the same name → chase
//     succeeds, deref carries the Struct render.
//   - Fwd with no sibling → chase still skips with the
//     descriptive reason from `unsizable_chase_reason`.
//   - Fwd-struct vs Union same-name → renderer must NOT collapse
//     across aggregate kinds (BTF wire format permits same-name
//     struct + union; collapsing would mis-render the wrong
//     layout).
//   - Anonymous Fwd → no resolution attempted (no name to look
//     up); descriptive reason still surfaces.

/// Arena chase: Type::Ptr arm. The pointee is a [`Type::Fwd`]
/// `task_ctx`; the SAME BTF carries a complete [`Type::Struct`]
/// `task_ctx` at a different id. Without the Fwd shortcut,
/// `chase_arena_pointer` would skip with "forward declaration;
/// body not in this BTF". With the shortcut, it lands on the
/// complete struct and renders the member values from arena
/// bytes.
#[test]
fn arena_chase_pointee_fwd_resolves_to_complete_struct_sibling() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "scx_task_map_val");
    let n_data = push(&mut strings, "data");
    let n_task_ctx = push(&mut strings, "task_ctx");
    let n_field = push(&mut strings, "field");
    // BTF layout:
    //   id=1: u64
    //   id=2: BTF_KIND_FWD struct task_ctx (no body — emulates
    //         what clang emits when only a pointer to task_ctx
    //         is referenced in the unit; the body is in a
    //         sibling unit's BTF)
    //   id=3: BTF_KIND_PTR -> id=2
    //   id=4: struct scx_task_map_val { struct task_ctx *data @ 0 }
    //   id=5: BTF_KIND_STRUCT task_ctx { u64 field @ 0 } size=8
    //         — the COMPLETE shape `peel_modifiers_resolving_fwd`
    //         lands on after looking up "task_ctx" in the BTF.
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Fwd {
            name_off: n_task_ctx,
            is_union: false,
        },
        CastSynType::Ptr { type_id: 2 },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_data,
                type_id: 3,
                byte_offset: 0,
            }],
        },
        CastSynType::Struct {
            name_off: n_task_ctx,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_field,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let outer_id: u32 = 4;

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x77u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "data field must render as Ptr (BTF Type::Ptr arm); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref_skipped_reason.is_none(),
        "Fwd-resolved chase must succeed; got skip reason: {deref_skipped_reason:?}"
    );
    let payload = deref
        .as_deref()
        .expect("Fwd-resolved chase must produce a deref payload");
    let RenderedValue::Struct {
        ref type_name,
        members: ref inner_members,
    } = *payload
    else {
        panic!("deref must be Struct render; got {payload:?}");
    };
    assert_eq!(
        type_name.as_deref(),
        Some("task_ctx"),
        "deref must carry the resolved Struct name"
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "field");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!(
            "inner field must decode as Uint; got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x77);
}

/// Cast intercept arena arm: a [`CastHit`] whose `target_type_id`
/// resolves to a [`Type::Fwd`] with a complete [`Type::Struct`]
/// sibling. Mirrors the BTF Type::Ptr test above but exercises
/// `render_cast_pointer`'s arena arm via a u64-typed parent
/// member and a `cast_lookup` hit. Also covers the cast analyzer
/// scenario where a u64 slot's recovered target id happens to
/// resolve to a Fwd at runtime (e.g. cross-BTF id mismatch under
/// libbpf BTF dedup).
#[test]
fn cast_chase_arena_target_fwd_resolves_to_complete_struct_sibling() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push(&mut strings, "u64");
    let n_t = push(&mut strings, "scx_task_map_val");
    let n_data = push(&mut strings, "data");
    let n_target = push(&mut strings, "task_ctx");
    let n_field = push(&mut strings, "field");
    // Layout:
    //   id=1: u64
    //   id=2: struct scx_task_map_val { u64 data @ 0 } size=8
    //   id=3: BTF_KIND_FWD task_ctx (struct)
    //   id=4: BTF_KIND_STRUCT task_ctx { u64 field @ 0 } size=8
    //
    // The CastMap entry points at id=3 (the Fwd). Without the
    // Fwd shortcut, the chase would skip; with it, it lands on
    // id=4 and renders the field.
    let types = vec![
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_data,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Fwd {
            name_off: n_target,
            is_union: false,
        },
        CastSynType::Struct {
            name_off: n_target,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_field,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let parent_id: u32 = 2;
    let fwd_target_id: u32 = 3;

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0xABCDu64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (parent_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: fwd_target_id,
            addr_space: AddrSpace::Arena,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, parent_id, &outer_bytes, &reader);
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
        panic!(
            "data field must render as cast-recovered Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→arena"),
        "cast intercept must annotate the arena chase"
    );
    assert!(
        deref_skipped_reason.is_none(),
        "Fwd-resolved cast chase must not skip; got: {deref_skipped_reason:?}"
    );
    let payload = deref
        .as_deref()
        .expect("Fwd-resolved cast chase must produce deref payload");
    let RenderedValue::Struct {
        ref type_name,
        members: ref inner_members,
    } = *payload
    else {
        panic!("deref must be Struct render; got {payload:?}");
    };
    assert_eq!(
        type_name.as_deref(),
        Some("task_ctx"),
        "deref must carry the resolved Struct name"
    );
    assert_eq!(inner_members[0].name, "field");
    let RenderedValue::Uint { value, .. } = inner_members[0].value else {
        panic!(
            "inner field must decode as Uint; got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(value, 0xABCD);
}

/// Cast intercept kernel arm: same scenario as the arena test
/// above but with `AddrSpace::Kernel`. The kernel arm of
/// `render_cast_pointer` also calls `peel_modifiers_resolving_fwd`
/// so the resolution shortcut applies symmetrically across
/// address spaces.
#[test]
fn cast_chase_kernel_target_fwd_resolves_to_complete_struct_sibling() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push(&mut strings, "u64");
    let n_t = push(&mut strings, "parent");
    let n_data = push(&mut strings, "data");
    let n_target = push(&mut strings, "kernel_target");
    let n_field = push(&mut strings, "field");
    let types = vec![
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_data,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Fwd {
            name_off: n_target,
            is_union: false,
        },
        CastSynType::Struct {
            name_off: n_target,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_field,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let parent_id: u32 = 2;
    let fwd_target_id: u32 = 3;

    // Use a non-arena KVA so the kernel arm fires.
    const KVA: u64 = 0xffff_8000_0000_3000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let inner_bytes = 0xDEADBEEFu64.to_le_bytes().to_vec();
    let mut kva_bytes = std::collections::HashMap::new();
    kva_bytes.insert(KVA, inner_bytes);
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (parent_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: fwd_target_id,
            addr_space: AddrSpace::Kernel,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        kva_bytes_at: kva_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, parent_id, &outer_bytes, &reader);
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
        panic!(
            "data field must render as cast-recovered Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert_eq!(cast_annotation.as_deref(), Some("cast→kernel"));
    assert!(
        deref_skipped_reason.is_none(),
        "Fwd-resolved kernel cast chase must not skip; got: {deref_skipped_reason:?}"
    );
    let payload = deref
        .as_deref()
        .expect("Fwd-resolved kernel cast chase must produce deref payload");
    let RenderedValue::Struct {
        ref type_name,
        members: ref inner_members,
    } = *payload
    else {
        panic!("deref must be Struct render; got {payload:?}");
    };
    assert_eq!(
        type_name.as_deref(),
        Some("kernel_target"),
        "deref must carry the resolved Struct name"
    );
    let RenderedValue::Uint { value, .. } = inner_members[0].value else {
        panic!(
            "inner field must decode as Uint; got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(value, 0xDEADBEEF);
}

/// Aggregate-kind mismatch: a [`Type::Fwd`] declared as
/// `struct foo` must NOT resolve to a [`Type::Union`] of the same
/// name in the same BTF. The wire format permits same-name
/// struct/union declarations (rare but legal); collapsing across
/// aggregate kinds would mis-render the wrong layout. Verifies
/// the Fwd shortcut respects [`btf_rs::Fwd::is_struct`] /
/// [`is_union`].
#[test]
fn fwd_shortcut_rejects_aggregate_kind_mismatch() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push(&mut strings, "u64");
    let n_wrap = push(&mut strings, "wrap");
    let n_data = push(&mut strings, "data");
    let n_foo = push(&mut strings, "foo");
    let n_x = push(&mut strings, "x");
    // Layout:
    //   id=1: u64
    //   id=2: BTF_KIND_FWD struct foo (kind_flag=0)
    //   id=3: BTF_KIND_PTR -> id=2
    //   id=4: struct wrap { struct foo *data @ 0 } size=8
    //   id=5: BTF_KIND_UNION foo { u64 x @ 0 } size=8
    //         — the same name as the Fwd, but the wrong aggregate
    //         kind (Union, not Struct). The shortcut must NOT
    //         resolve to it.
    let types = vec![
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Fwd {
            name_off: n_foo,
            is_union: false,
        },
        CastSynType::Ptr { type_id: 2 },
        CastSynType::Struct {
            name_off: n_wrap,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_data,
                type_id: 3,
                byte_offset: 0,
            }],
        },
        // Encode a Union via the synthetic builder — emit a
        // BTF_KIND_UNION (wire kind=5) by hand since the cast test
        // helper does not yet include a Union variant. Using the
        // raw byte writer keeps the helper API minimal.
        CastSynType::Struct {
            // intentionally still Struct here as a placeholder —
            // see below for the union manual emission.
            name_off: n_foo,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    // The synthetic builder above emits Struct foo at id=5, NOT a
    // Union. To verify aggregate-kind mismatch we need a Union.
    // Manually patch the kind nibble of id=5's `info` u32 from
    // BTF_KIND_STRUCT (4) to BTF_KIND_UNION (5) in the produced
    // blob. The synthetic types vector above is only for
    // documentation: the actual Union emission happens inline
    // below.
    let mut blob = cast_build_btf(&types, &strings);
    // id=5 starts after the header (24 bytes) plus id=1..=4. Compute
    // the byte offset of id=5's info u32 (`name_off` + 4 bytes).
    // id=1 (Int): 16 bytes total
    // id=2 (Fwd): 12 bytes
    // id=3 (Ptr): 12 bytes
    // id=4 (Struct, 1 member): 12 + 12 = 24 bytes
    // id=5 starts at offset 24 + 16 + 12 + 12 + 24 = 88.
    let id5_info_off: usize = 24 + 16 + 12 + 12 + 24 + 4;
    // Read existing info u32, mask out kind nibble (bits 24..28),
    // write BTF_KIND_UNION (5).
    let info = u32::from_le_bytes(blob[id5_info_off..id5_info_off + 4].try_into().unwrap());
    let new_info = (info & !(0x1f << 24)) | (5u32 << 24);
    blob[id5_info_off..id5_info_off + 4].copy_from_slice(&new_info.to_le_bytes());

    let btf = Btf::from_bytes(&blob).expect("synthetic BTF with union parses");
    let wrap_id: u32 = 4;
    let fwd_id: u32 = 2;

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, wrap_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!("data field must render as Ptr; got {:?}", members[0].value);
    };
    assert!(
        deref.is_none(),
        "aggregate-kind mismatch must NOT resolve the Fwd; chase must skip"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("aggregate-kind mismatch must populate skip reason (Fwd unresolved)");
    assert!(
        reason.contains("forward declaration"),
        "skip reason must report the Fwd; got: {reason}"
    );
    assert!(
        reason.contains("foo"),
        "skip reason must include the Fwd's name; got: {reason}"
    );
    assert!(
        reason.contains(&fwd_id.to_string()),
        "skip reason must include the Fwd's id; got: {reason}"
    );
}

/// Unit test: `peel_modifiers_resolving_fwd` returns the Fwd
/// unchanged when no complete sibling exists, so the
/// chase-pipeline skip path remains intact for the unrecoverable
/// case (e.g. `struct sdt_data` referenced by lavd: the body lives
/// only in the sdt_alloc library, never in the program's own
/// BTF).
#[test]
fn peel_modifiers_resolving_fwd_no_sibling_returns_fwd() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_fwd = push(&mut strings, "lonely_fwd");
    // BTF: id=1 u64, id=2 Fwd 'lonely_fwd' (struct, no sibling).
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: false,
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let (peeled, peeled_id) =
        peel_modifiers_resolving_fwd(&btf, 2).expect("Fwd resolves through helper");
    assert!(
        matches!(peeled, Type::Fwd(_)),
        "no-sibling lookup must return the original Fwd; got {peeled:?}"
    );
    assert_eq!(peeled_id, 2);
}

/// Unit test: anonymous Fwd (name_off=0) cannot be looked up by
/// name; helper returns the Fwd unchanged.
#[test]
fn peel_modifiers_resolving_fwd_anonymous_fwd_returns_fwd() {
    let strings: Vec<u8> = vec![0];
    let types = vec![
        CastSynType::Int {
            name_off: 0,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Fwd {
            name_off: 0,
            is_union: false,
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let (peeled, peeled_id) =
        peel_modifiers_resolving_fwd(&btf, 2).expect("anonymous Fwd resolves through helper");
    assert!(
        matches!(peeled, Type::Fwd(_)),
        "anonymous Fwd must remain Fwd; got {peeled:?}"
    );
    assert_eq!(peeled_id, 2);
}

/// Unit test: a Typedef chain ending in a Fwd is peeled through
/// the modifier chain AND the Fwd is then resolved to a complete
/// Struct sibling. Mirrors the typical `typedef struct foo foo;`
/// plus `struct foo;` pattern clang emits when a header
/// forward-declares a typedef and a separate compilation unit
/// defines the underlying struct.
#[test]
fn peel_modifiers_resolving_fwd_through_typedef_chain() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push(&mut strings, "u64");
    let n_alias = push(&mut strings, "alias");
    let n_target = push(&mut strings, "target");
    let n_field = push(&mut strings, "field");
    // BTF:
    //   id=1: u64
    //   id=2: BTF_KIND_TYPEDEF alias -> id=3
    //   id=3: BTF_KIND_FWD target (struct)
    //   id=4: BTF_KIND_STRUCT target { u64 field @ 0 } size=8
    //
    // Calling peel_modifiers_resolving_fwd(&btf, 2) must:
    //  1. Peel Typedef -> Fwd (peel_modifiers_with_id terminates
    //     at the Fwd).
    //  2. Resolve Fwd 'target' to the sibling Struct at id=4.
    let types = vec![
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Typedef {
            name_off: n_alias,
            type_id: 3,
        },
        CastSynType::Fwd {
            name_off: n_target,
            is_union: false,
        },
        CastSynType::Struct {
            name_off: n_target,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_field,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let (peeled, peeled_id) =
        peel_modifiers_resolving_fwd(&btf, 2).expect("Typedef→Fwd→Struct chain resolves");
    assert!(
        matches!(peeled, Type::Struct(_)),
        "Typedef→Fwd chain must land on the complete Struct; got {peeled:?}"
    );
    assert_eq!(
        peeled_id, 4,
        "resolved id must be the complete Struct's id, not the Typedef or Fwd id"
    );
}

/// Kernel cast hit where `read_kva` returns `None` (the page is
/// unmapped or the page-table walk failed). The renderer surfaces
/// `Ptr{ deref: None, deref_skipped_reason: Some("kernel read_kva
/// failed at 0x... (unmapped page or no PTE); needed N bytes") }`
/// from [`render_cast_pointer`]'s kernel-arm `read_kva`
/// failure branch. Without this gate, a `None` from
/// `read_kva` would propagate to the unwrap downstream and either
/// crash or render whatever default the inner type produces from
/// zero bytes — both worse than the labelled skip.
#[test]
fn cast_chase_kernel_read_kva_failure() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Q is an 8-byte struct (size=8), so `read_size` will be 8.
    // Reader carries NO `kva_bytes_at` entries, so `read_kva` returns
    // `None` for every address — the failure path under test.
    const KVA: u64 = 0xffff_8000_0000_2000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Kernel,
        }),
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
        ..
    } = members[0].value
    else {
        panic!(
            "read_kva failure must still surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref.is_none(),
        "read_kva failure must not produce a deref payload"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("read_kva failure must populate skip reason");
    assert!(
        reason.contains("read_kva failed"),
        "skip reason must say 'read_kva failed'; got: {reason}"
    );
    assert!(
        reason.contains(&format!("0x{KVA:x}")),
        "skip reason must include the failing address in hex; got: {reason}"
    );
    assert!(
        reason.contains("needed"),
        "skip reason must include the requested byte count; got: {reason}"
    );
}

/// Kernel cast hit where the requested size exceeds the bytes
/// remaining in the current 4 KiB page. `read_size = btf_size
/// .min(POINTER_CHASE_CAP).min(page_remaining)` clamp inside
/// [`render_cast_pointer`]'s kernel arm caps the read at the page
/// edge so `read_kva` never crosses an allocation boundary; the
/// resulting `truncated_at_cap` flag wraps the inner render in
/// `RenderedValue::Truncated{needed: btf_size, had: page_remaining,
/// partial: ...}` at the kernel arm's `truncated_at_cap` branch.
/// This
/// test pins the page-edge clipping AND the Truncated wrapper.
#[test]
fn cast_chase_kernel_page_edge_truncation() {
    // Build a target with size > page_remaining. Q's size is set to
    // 100 bytes (a single u64 at offset 0 plus 92 bytes of trailing
    // padding, recorded only in `size_type`). The kernel value KVA
    // = page_base + 4080 leaves 16 bytes in the current page
    // (4096 - (4080 % 4096) = 16). So:
    //   btf_size       = 100
    //   page_remaining = 16
    //   POINTER_CHASE_CAP = 4096
    //   read_size      = min(100, 4096, 16) = 16
    //   truncated_at_cap = (100 > 16) = true
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
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
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // Q size=100, single u64 member at offset 0. The struct's
        // declared size is what `type_size` reports — not the sum
        // of member sizes. Tail bytes 8..100 are unaccounted for in
        // BTF members, modelling a struct with padding or members
        // the test doesn't care about.
        CastSynType::Struct {
            name_off: n_q,
            size: 100,
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

    // KVA leaves exactly 16 bytes remaining in the page.
    const KVA: u64 = 0xffff_8000_0000_0ff0;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    // Provide 16 bytes at KVA so `read_kva(KVA, 16)` succeeds. First
    // 8 bytes carry Q.x = 0xCAFE (low value, top byte is 0x00 — the
    // plausibility gate (top-byte-0xff freelist heuristic) in
    // [`render_cast_pointer`]'s kernel arm accepts it).
    let mut target_bytes = vec![0u8; 16];
    target_bytes[0..8].copy_from_slice(&0xCAFEu64.to_le_bytes());
    let mut kva_bytes = std::collections::HashMap::new();
    kva_bytes.insert(KVA, target_bytes);
    // Use cast_map mode so only the outer T.f → Q intercept fires;
    // Q.x at (q_id, 0) has no entry, so the inner u64 surfaces as
    // a plain Uint render rather than recursing into another chase.
    let mut cast_map: crate::monitor::cast_analysis::CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Kernel,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        kva_bytes_at: kva_bytes,
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
        ..
    } = members[0].value
    else {
        panic!(
            "page-edge clipped chase must still surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref_skipped_reason.is_none(),
        "successful (clipped) read must carry no skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("read succeeded → deref must be Some");
    // Outer wrap: Truncated{needed: 100, had: 16, partial: ...}.
    let RenderedValue::Truncated {
        needed,
        had,
        ref partial,
    } = *inner
    else {
        panic!("btf_size > read_size must wrap deref payload in Truncated; got {inner:?}");
    };
    assert_eq!(needed, 100, "Truncated.needed must be the BTF size");
    assert_eq!(
        had, 16,
        "Truncated.had must be the page-edge-clipped read size"
    );
    // Partial render: the inner Struct{Q} also wraps as Truncated
    // because 16 bytes < Q.size=100, with its own partial: Struct.
    // Walk through both wrappers to reach Q's members.
    let inner_struct = match &**partial {
        RenderedValue::Struct { .. } => partial.as_ref(),
        RenderedValue::Truncated {
            partial: deeper, ..
        } => deeper.as_ref(),
        other => panic!(
            "partial render must reach a Q struct (possibly via inner Truncated); got {other:?}"
        ),
    };
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner_struct
    else {
        panic!("expected inner Struct render, got {inner_struct:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "inner struct must carry Q's name"
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "x");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0xCAFE, "first 8 bytes of clipped read must decode");
}

/// Kernel cast hit whose `read_kva` returns plausible bytes (top
/// byte != 0xff) and whose target peels + sizes correctly: the
/// chase succeeds and the rendered struct's members are surfaced
/// in the `deref` payload. Complement to
/// `cast_chase_kernel_plausibility_rejects_freed_slab` — same
/// path but the plausibility gate ALLOWS the read instead of
/// rejecting it. Without this test, a regression that flipped the
/// plausibility gate's polarity (`if first_qword >> 56 != 0xff`
/// rejects, instead of accepts) would only show up as a missing
/// deref on every kernel chase, with no test catching the inversion.
#[test]
fn cast_chase_kernel_successful_chase_top_byte_non_ff() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // KVA in the kernel direct-map range; the value passed to the
    // plausibility gate is the FIRST QWORD OF target_bytes (Q.x's
    // value), not KVA itself. Choose target bytes whose first qword
    // top byte is 0x00 (a plain counter) so the gate at
    // the plausibility gate in [`render_cast_pointer`]'s kernel
    // arm (top-byte-0xff freelist heuristic) accepts.
    const KVA: u64 = 0xffff_8000_dead_b000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    // Q is 8 bytes (the existing fixture). Provide exactly 8 bytes
    // at KVA whose first qword is a low counter value (0x42).
    let inner_bytes: Vec<u8> = 0x42u64.to_le_bytes().to_vec();
    let mut kva_bytes = std::collections::HashMap::new();
    kva_bytes.insert(KVA, inner_bytes);
    // Use cast_map mode so only the outer T.f → Q intercept fires;
    // Q.x at (q_id, 0) has no entry, so the inner u64 surfaces as
    // a plain Uint render rather than recursing into another chase.
    let mut cast_map: crate::monitor::cast_analysis::CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Kernel,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        kva_bytes_at: kva_bytes,
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
        ..
    } = members[0].value
    else {
        panic!(
            "kernel chase must surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref_skipped_reason.is_none(),
        "successful chase carries no skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("plausibility-allowed chase → deref must be Some");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the rendered Q struct; got {inner:?}");
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
    assert_eq!(value, 0x42, "Q.x must reflect the bytes read_kva returned");
}
