use super::*;

// ---- render_cast_pointer kernel-arm skip-reason paths ----------
//
// `render_cast_pointer`'s kernel arm walks four pre-read gates:
// (1) target type peels (via `peel_modifiers_resolving_fwd`), (2)
// `type_size` returns Some, (3) size != 0, (4) `read_kva` returns
// bytes. Each failure surfaces a distinctly
// worded `deref_skipped_reason`. A regression that collapsed two
// gates into one — or skipped a gate entirely — would produce a
// chase against an unresolved or zero-sized target, with the
// rendered output silently degrading to garbage. These tests pin
// each reason string so every failure path stays distinguishable.

/// Kernel cast target whose `target_type_id` does not resolve to any
/// type in the BTF — `peel_modifiers` returns `None`. The renderer
/// surfaces `Ptr{ deref: None, deref_skipped_reason: Some("kernel
/// cast target type id N unresolvable") }` from
/// [`resolve_chase_target`]'s step-1 peel (called by
/// [`render_cast_pointer`]'s kernel arm), whose
/// `peel_modifiers_resolving_fwd` call precedes the
/// `try_sdt_alloc_bridge` and size resolution.
/// Without this guard, a corrupt or stale CastMap entry pointing at
/// a freed type id would propagate `None` further down and surface
/// as the same "kernel read_kva failed" reason that genuine read
/// failures use — collapsing two distinct failure modes into one.
#[test]
fn cast_chase_kernel_target_type_id_unresolvable() {
    let (blob, t_id, _q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Use an id beyond every type emitted by `cast_btf_t_and_q` (it
    // produces ids 1..=3, so 9999 is safely out of range and
    // `btf.resolve_type_by_id` errors → peel_modifiers → None).
    const UNRESOLVABLE: u32 = 9999;
    const KVA: u64 = 0xffff_8000_0000_1000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit {
            alloc_size: None,
            target_type_id: UNRESOLVABLE,
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
            "unresolvable target must still surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref.is_none(),
        "unresolvable target must not produce a deref payload"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("peel_modifiers failure must populate skip reason");
    assert!(
        reason.contains("unresolvable"),
        "skip reason must mention 'unresolvable'; got: {reason}"
    );
    assert!(
        reason.contains(&UNRESOLVABLE.to_string()),
        "skip reason must include the offending type id; got: {reason}"
    );
}

/// Kernel cast target whose BTF size is 0 (a struct with `size_type
/// = 0` — represents an incomplete forward declaration the BPF
/// compiler emitted without a definition). The renderer surfaces
/// `Ptr{ deref: None, deref_skipped_reason: Some("...BTF size is 0
/// (incomplete type)") }` from [`resolve_chase_target`]'s
/// step-5 `if btf_size == 0` gate (reached from
/// [`render_cast_pointer`]'s kernel arm). This guard
/// prevents a zero-byte `read_kva` from succeeding spuriously and
/// rendering an empty struct as if the chase had landed.
#[test]
fn cast_chase_kernel_target_btf_size_zero() {
    // Build a BTF blob where the cast target is a zero-sized
    // struct. Layout:
    //   id=1: u64
    //   id=2: struct T { u64 f @ 0 }, size 8
    //   id=3: struct Q {}, size 0  (the zero-sized cast target)
    //
    // T_id=2, Q_id=3. The gate we exercise is
    // [`resolve_chase_target`]'s step-5 `if btf_size == 0`,
    // reached from [`render_cast_pointer`]'s kernel arm via its
    // `resolve_chase_target("kernel cast")` call; `type_size`
    // returns `Some(0)` for a zero-sized Struct so the prior guard
    // (None case) does not fire.
    let (strings, n_int, n_t, n_q, n_f, _n_x) = cast_strings_for_t_q();
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
        // Q with size 0 and no members: the BTF wire format permits
        // it (vlen=0, size_type=0), and `type_size` returns Some(0).
        CastSynType::Struct {
            name_off: n_q,
            size: 0,
            members: vec![],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;

    const KVA: u64 = 0xffff_8000_0000_1000;
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
            "zero-sized target must still surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(deref.is_none());
    let reason = deref_skipped_reason
        .as_deref()
        .expect("zero-sized target must populate skip reason");
    assert!(
        reason.contains("BTF size is 0"),
        "skip reason must say 'BTF size is 0'; got: {reason}"
    );
    assert!(
        reason.contains("incomplete type"),
        "skip reason must mention 'incomplete type'; got: {reason}"
    );
}

/// Kernel cast hit whose target peels to a `BTF_KIND_FWD` (forward
/// declaration). `type_size` returns `None` for `Type::Fwd` because
/// a forward declaration carries no body in this BTF, so the chase
/// has no BTF-declared size to bound the read. The renderer surfaces
/// a `Ptr{ deref: None, deref_skipped_reason: Some("kernel cast
/// target struct sdt_data (type id N) is a forward declaration;
/// body not in this BTF") }` via [`unsizable_chase_reason`].
///
/// Without this case-specific path, the `type_size` failure would
/// have surfaced as the generic "has unresolvable size" message
/// (the legacy fall-through), which gives no operator the cause —
/// they would not know whether the BTF was malformed, the analyzer
/// emitted a stale id, or the chase landed on a valid forward
/// declaration whose body lives in a sibling BTF.
#[test]
fn cast_chase_kernel_target_fwd_struct() {
    // Build a BTF blob where the cast target is a forward-declared
    // struct named "sdt_data". Layout:
    //   id=1: u64
    //   id=2: struct T { u64 f @ 0 }, size 8
    //   id=3: BTF_KIND_FWD struct sdt_data (no body)
    //
    // The cast analyzer's production output never hits this — it
    // only emits Struct/Union ids — but a future analyzer change or
    // a manual cast_map mutation should still surface a clear
    // diagnostic rather than the generic "unresolvable size".
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "T");
    let n_fwd = push(&mut strings, "sdt_data");
    let n_f = push(&mut strings, "f");
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
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: false,
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let fwd_id: u32 = 3;

    const KVA: u64 = 0xffff_8000_0000_3000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit {
            alloc_size: None,
            target_type_id: fwd_id,
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
            "Fwd target must still surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref.is_none(),
        "Fwd target must not produce a deref payload"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("Fwd target must populate skip reason");
    assert!(
        reason.contains("forward declaration"),
        "skip reason must mention 'forward declaration'; got: {reason}"
    );
    assert!(
        reason.contains("body not in this BTF"),
        "skip reason must mention body absence; got: {reason}"
    );
    assert!(
        reason.contains("sdt_data"),
        "skip reason must include the Fwd type's name; got: {reason}"
    );
    assert!(
        reason.contains("struct"),
        "skip reason must say 'struct' (not 'union') for is_struct() Fwd; got: {reason}"
    );
    assert!(
        reason.contains(&fwd_id.to_string()),
        "skip reason must include the type id; got: {reason}"
    );
    // The legacy fall-through message must NOT appear; if it does,
    // the dispatch in `unsizable_chase_reason` did not catch the
    // Type::Fwd arm and we regressed to the generic path.
    assert!(
        !reason.contains("has unresolvable size"),
        "Fwd targets must not surface the generic fall-through; got: {reason}"
    );
}

/// Same scenario as [`cast_chase_kernel_target_fwd_struct`] but the
/// forward declaration is a union (`is_union: true`). The Fwd
/// kind_flag bit selects struct vs union; the renderer surfaces
/// "union" in the reason so operators see the correct aggregate
/// kind.
#[test]
fn cast_chase_kernel_target_fwd_union() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "T");
    let n_fwd = push(&mut strings, "my_union");
    let n_f = push(&mut strings, "f");
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
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: true,
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let fwd_id: u32 = 3;

    const KVA: u64 = 0xffff_8000_0000_4000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit {
            alloc_size: None,
            target_type_id: fwd_id,
            addr_space: AddrSpace::Kernel,
        }),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "Fwd union target must surface as Ptr; got {:?}",
            members[0].value
        );
    };
    let reason = deref_skipped_reason
        .as_deref()
        .expect("Fwd union target must populate skip reason");
    assert!(
        reason.contains("union my_union"),
        "skip reason must surface 'union my_union'; got: {reason}"
    );
    assert!(
        !reason.contains("struct my_union"),
        "Fwd union must not be labelled 'struct'; got: {reason}"
    );
}

/// Arena chase whose pointee BTF type is a `BTF_KIND_FWD`. The
/// real-world trigger: a `struct sdt_chunk` union member declared
/// as `struct sdt_data __arena *`, where `struct sdt_data`'s body
/// lives in the sdt_alloc library's BTF and the using scheduler's
/// own program BTF carries only a forward declaration. The
/// [`btf_rs::Type::Ptr`] arm calls `chase_arena_pointer` with the
/// pointee type id; before this fix `type_size` returned `None`
/// and surfaced as "arena chase target type id N has unresolvable
/// size", giving the operator no signal that the cause was a
/// forward declaration.
///
/// The fix routes `type_size` failures through
/// [`unsizable_chase_reason`], which inspects the peeled type and
/// emits a Fwd-specific message. This test mirrors the production
/// trigger: outer struct holds a `Ptr` to a `Fwd`, the pointer
/// lands in the arena window, and the renderer surfaces the
/// renamed reason.
#[test]
fn arena_chase_pointee_fwd_surfaces_descriptive_reason() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "sdt_chunk");
    let n_fwd = push(&mut strings, "sdt_data");
    let n_data = push(&mut strings, "data");
    // BTF layout:
    //   id=1: u64
    //   id=2: BTF_KIND_FWD struct sdt_data (no body — emulates the
    //         scheduler-side view of the library struct)
    //   id=3: BTF_KIND_PTR -> id=2 (the `struct sdt_data *` field)
    //   id=4: struct sdt_chunk { struct sdt_data *data @ 0 }, size 8
    //
    // The Type::Ptr arm in render_value_inner reads the u64 at
    // offset 0, recognises the value as an arena address (via
    // `is_arena_addr`), and calls `chase_arena_pointer(btf,
    // pointee_type_id=2, ...)`. The pointee peels to Type::Fwd,
    // type_size returns None, and `unsizable_chase_reason`
    // composes the descriptive message under test.
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
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let chunk_id: u32 = 4;
    let fwd_id: u32 = 2;

    // Arena window 0x10_0000_0000 .. 0x10_0001_0000; the address
    // 0x10_0000_1000 lands inside.
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, chunk_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].name, "data");
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!(
            "data field must render as Ptr (BTF Type::Ptr arm); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        cast_annotation.is_none(),
        "BTF-typed pointers must leave cast_annotation None; got {cast_annotation:?}"
    );
    assert!(
        deref.is_none(),
        "Fwd pointee chase must not produce a deref payload"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("Fwd pointee must populate skip reason");
    assert!(
        reason.starts_with("arena chase"),
        "BTF Ptr arm must use 'arena chase' label; got: {reason}"
    );
    assert!(
        reason.contains("forward declaration"),
        "skip reason must mention 'forward declaration'; got: {reason}"
    );
    assert!(
        reason.contains("body not in this BTF"),
        "skip reason must mention body absence; got: {reason}"
    );
    assert!(
        reason.contains("sdt_data"),
        "skip reason must include the Fwd type's name; got: {reason}"
    );
    assert!(
        reason.contains("struct"),
        "skip reason must say 'struct' (kind_flag=0); got: {reason}"
    );
    assert!(
        reason.contains(&fwd_id.to_string()),
        "skip reason must include the Fwd type id; got: {reason}"
    );
    assert!(
        !reason.contains("has unresolvable size"),
        "Fwd targets must not surface the legacy generic message; got: {reason}"
    );
}

/// Anonymous Fwd: a forward declaration with `name_off = 0` (an
/// unnamed forward — uncommon but legal in BTF). The reason text
/// must indicate "anonymous" and still record the type id and
/// aggregate kind so an operator can correlate.
#[test]
fn arena_chase_pointee_fwd_anonymous() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "wrap");
    let n_data = push(&mut strings, "data");
    // BTF layout:
    //   id=1: u64
    //   id=2: BTF_KIND_FWD anonymous (name_off=0) struct
    //   id=3: BTF_KIND_PTR -> id=2
    //   id=4: struct wrap { void *data @ 0 }, size 8
    //
    // Anonymous Fwd nodes appear in BTF when a struct is forward-
    // declared inside a function or unnamed scope. The chase
    // reason path must still produce a useful message — names
    // shouldn't be load-bearing.
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Fwd {
            name_off: 0,
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
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let chunk_id: u32 = 4;

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, chunk_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!("data field must render as Ptr; got {:?}", members[0].value);
    };
    let reason = deref_skipped_reason
        .as_deref()
        .expect("anonymous Fwd must populate skip reason");
    assert!(
        reason.contains("anonymous"),
        "anonymous Fwd reason must say 'anonymous'; got: {reason}"
    );
    assert!(
        reason.contains("struct forward declaration"),
        "anonymous Fwd reason must mention the aggregate kind; got: {reason}"
    );
}
