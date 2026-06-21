use super::*;

// ---- sdt_alloc arena-type bridge -------------------------------
//
// `MemReader::resolve_arena_type(addr)` lets the renderer recover
// the BTF type id of a chased arena pointer's pointee when the
// program BTF carries only a `BTF_KIND_FWD` (forward declaration —
// body in another BTF). The dump path's `AccessorMemReader`
// populates the lookup from the sdt_alloc pre-pass's allocator
// snapshots; the renderer's [`chase_arena_pointer`] /
// [`render_cast_pointer`] paths consult it after the BTF-only Fwd
// resolve fails.
//
// The tests below cover:
//   - default `MemReader` impl returns `None` (no bridge wiring).
//   - custom `MemReader` impl override returns the configured id.
//   - `chase_arena_pointer` with a Fwd target + matching bridge
//     entry produces a successful chase whose deref renders the
//     resolved struct.
//   - `chase_arena_pointer` with no bridge entry skips with the
//     existing "forward declaration; body not in this BTF" reason.
//   - Type::Ptr arena arm sets `cast_annotation` to "sdt_alloc"
//     when the bridge fires.
//   - `render_cast_pointer` arena arm extends `cast_annotation`
//     to `cast→arena (sdt_alloc)` when the bridge fires.

/// `MemReader` trait default for `resolve_arena_type` returns
/// `None` for every address. Pin the behaviour so a future change
/// that flipped the default would surface here as a test
/// regression rather than silently activating the bridge for every
/// reader.
#[test]
fn mem_reader_default_resolve_arena_type_is_none() {
    struct DefaultReader;
    impl MemReader for DefaultReader {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
    }
    let r = DefaultReader;
    assert!(
        r.resolve_arena_type(0x10_0000_1000).is_none(),
        "default resolve_arena_type must return None for any address",
    );
    assert!(
        r.resolve_arena_type(0).is_none(),
        "default resolve_arena_type must return None for null too",
    );
    assert!(
        r.resolve_arena_type(u64::MAX).is_none(),
        "default resolve_arena_type must return None for u64::MAX too",
    );
}

/// Custom [`MemReader`] override returns the configured
/// [`ArenaResolveHit`] for known addresses and `None` for
/// everything else. Mirrors the production
/// [`crate::monitor::dump::render_map::AccessorMemReader::resolve_arena_type`]
/// shape. Two distinct seeded entries cover the two production
/// shapes — payload-start chase (`header_skip = 0`) and slot-start
/// chase (`header_skip = header_size`).
#[test]
fn mem_reader_resolve_arena_type_override_returns_configured_hit() {
    let mut arena_types = std::collections::HashMap::new();
    // Payload-start entry: header_skip = 0.
    arena_types.insert(
        0x10_0000_1008u64,
        ArenaResolveHit {
            target_type_id: 7,
            header_skip: 0,
        },
    );
    // Slot-start entry: header_skip = 8 (the size of `union sdt_id`).
    arena_types.insert(
        0x10_0000_2000u64,
        ArenaResolveHit {
            target_type_id: 11,
            header_skip: 8,
        },
    );
    let reader = CastStubReader {
        arena_type_at: arena_types,
        ..Default::default()
    };
    assert_eq!(
        reader.resolve_arena_type(0x10_0000_1008),
        Some(ArenaResolveHit {
            target_type_id: 7,
            header_skip: 0,
        }),
    );
    assert_eq!(
        reader.resolve_arena_type(0x10_0000_2000),
        Some(ArenaResolveHit {
            target_type_id: 11,
            header_skip: 8,
        }),
    );
    assert!(
        reader.resolve_arena_type(0x10_0000_3000).is_none(),
        "address not in index must return None",
    );
    assert!(
        reader.resolve_arena_type(0).is_none(),
        "null address must return None",
    );
}

/// Build a synthetic BTF blob for the sdt_alloc bridge tests.
///
/// Layout:
///   - id=1: u64 (size=8, plain unsigned)
///   - id=2: BTF_KIND_FWD struct sdt_data (no body — emulates the
///     scheduler-side forward declaration of the library struct)
///   - id=3: BTF_KIND_PTR -> id=2 (the `struct sdt_data *` field
///     type)
///   - id=4: struct outer { struct sdt_data *data @ 0 }, size=8
///     (the field through which the renderer chases an arena
///     pointer)
///   - id=5: struct task_ctx { u64 weight @ 0 }, size=8 (the real
///     payload type the bridge resolves to — distinct from
///     sdt_data so the renderer must consult the bridge to find it)
///
/// The `struct sdt_data *` field name and pointee Fwd are
/// stand-ins for the bridge's TRIGGER shape, not the production
/// trigger itself. In production the bridge fires on PAYLOAD-START
/// pointers — `cpu_ctx::cached_taskc_raw` is a `u64` storing the
/// return value of `scx_task_data(p)` (which dereferences a
/// `struct sdt_data __arena *` slot and returns `data->payload`,
/// past the 8-byte header). The cast analyzer promotes that `u64`
/// to a typed pointer whose pointee surfaces as a `BTF_KIND_FWD`
/// in the program BTF (the body lives in the sdt_alloc library's
/// BTF). The fixture compresses that into one shape: a Fwd-pointee
/// `Type::Ptr` field whose stored value is the payload-start
/// address used to populate the bridge index.
///
/// `CastStubReader::resolve_arena_type` (the test's MemReader) keys
/// the bridge map on the FULL 64-bit address rather than the low 32
/// bits — production
/// [`crate::monitor::dump::render_map::AccessorMemReader::resolve_arena_type`]
/// masks `addr & 0xFFFF_FFFF` and looks up in the per-pass index. Tests use
/// full-address keys to avoid the masking concern in the test setup;
/// the masking itself is exercised by the production `resolve_arena_type`
/// masking tests in `monitor/dump/tests.rs`.
///
/// Returns `(blob, outer_id, fwd_id, task_ctx_id)`.
fn bridge_btf_outer_fwd_taskctx() -> (Vec<u8>, u32, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_outer = push(&mut strings, "outer");
    let n_fwd = push(&mut strings, "sdt_data");
    let n_data = push(&mut strings, "data");
    let n_task = push(&mut strings, "task_ctx");
    let n_weight = push(&mut strings, "weight");
    let types = vec![
        // id 1: u64.
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: BTF_KIND_FWD struct sdt_data (no body).
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: false,
        },
        // id 3: struct sdt_data *.
        CastSynType::Ptr { type_id: 2 },
        // id 4: struct outer { struct sdt_data *data @ 0; } size=8.
        CastSynType::Struct {
            name_off: n_outer,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_data,
                type_id: 3,
                byte_offset: 0,
            }],
        },
        // id 5: struct task_ctx { u64 weight @ 0; } size=8.
        CastSynType::Struct {
            name_off: n_task,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_weight,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    (cast_build_btf(&types, &strings), 4, 2, 5)
}

/// Type::Ptr arena arm with a Fwd pointee: the renderer's BTF-only
/// resolve fails (no complete sibling for the Fwd), but the
/// [`MemReader::resolve_arena_type`] bridge returns the real
/// payload type id. The chase succeeds and renders the pointee
/// against the recovered type. The resulting `Ptr` carries
/// `cast_annotation: Some("sdt_alloc")` to flag the bridge resolve.
#[test]
fn arena_chase_fwd_target_resolved_via_bridge() {
    let (blob, outer_id, _fwd_id, task_ctx_id) = bridge_btf_outer_fwd_taskctx();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    // outer { data: TARGET_ADDR } — the pointer the renderer
    // chases.
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    // task_ctx at TARGET_ADDR: weight = 0x42 (u64 LE).
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    // Payload-start chase: header_skip = 0 — the renderer reads
    // `btf_size` bytes directly from the chased address.
    arena_types.insert(
        TARGET_ADDR,
        ArenaResolveHit {
            target_type_id: task_ctx_id,
            header_skip: 0,
        },
    );
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
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
        deref_skipped_reason.is_none(),
        "bridge resolve must not surface a skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("bridge resolve must produce a deref");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the resolved task_ctx struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("task_ctx"),
        "bridge must land on the resolved struct's name, not the Fwd's name"
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "weight");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!(
            "task_ctx.weight must render as Uint, got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x42);
    assert_eq!(
        cast_annotation.as_deref(),
        Some("sdt_alloc"),
        "Type::Ptr arm bridge resolve must surface 'sdt_alloc' annotation",
    );
}

/// Type::Ptr arena arm with a Fwd pointee resolved via the
/// bridge for a SLOT-START chase: the bridge returns
/// `header_skip = header_size`, the chase reads
/// `header_skip + btf_size` bytes from the chased address, slices
/// off the header, and renders the payload struct. Pins the bug
/// fix that surfaced the `data` field in `scx_task_map_val` (a
/// slot-start pointer that did not resolve under the previous
/// payload-start-only key shape).
///
/// Layout: 8-byte header (the `union sdt_id` shape — two
/// arbitrary u32s here, not interpreted by the bridge), followed
/// by the payload struct (`task_ctx { u64 weight }`, 8 bytes).
/// Total elem_size = 16. The chased address points at slot start;
/// the renderer must NOT decode the header bytes as the payload.
#[test]
fn arena_chase_fwd_target_resolved_via_bridge_slot_start_skips_header() {
    let (blob, outer_id, _fwd_id, task_ctx_id) = bridge_btf_outer_fwd_taskctx();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    // Chased address = slot start. The bridge must direct the
    // chase to skip the first 8 bytes of header before rendering.
    const SLOT_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = SLOT_ADDR.to_le_bytes().to_vec();
    // 16 bytes of slot contents at SLOT_ADDR:
    //   [0..8]   header bytes — sentinel pattern, NOT the payload.
    //            If the renderer decoded them as payload, weight
    //            would resolve to 0xDEADBEEFCAFEBABE.
    //   [8..16]  payload (task_ctx.weight = 0x42, LE u64).
    let header_sentinel = 0xDEAD_BEEF_CAFE_BABEu64.to_le_bytes();
    let payload_bytes = 0x42u64.to_le_bytes();
    let mut slot_bytes = Vec::with_capacity(16);
    slot_bytes.extend_from_slice(&header_sentinel);
    slot_bytes.extend_from_slice(&payload_bytes);

    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(SLOT_ADDR, slot_bytes);
    let mut arena_types = std::collections::HashMap::new();
    // Slot-start chase: header_skip = 8 — the renderer reads
    // `header_skip + btf_size` bytes from the chased address and
    // slices off the first 8 bytes (the header) before rendering.
    arena_types.insert(
        SLOT_ADDR,
        ArenaResolveHit {
            target_type_id: task_ctx_id,
            header_skip: 8,
        },
    );
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
    } = members[0].value
    else {
        panic!("data field must render as Ptr; got {:?}", members[0].value);
    };
    assert_eq!(value, SLOT_ADDR);
    assert!(
        deref_skipped_reason.is_none(),
        "slot-start bridge resolve must not surface a skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("slot-start bridge resolve must produce a deref");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the resolved task_ctx struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("task_ctx"),
        "bridge must land on the resolved struct's name even with slot-start skip",
    );
    let RenderedValue::Uint {
        bits,
        value: weight,
    } = inner_members[0].value
    else {
        panic!(
            "task_ctx.weight must render as Uint, got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(bits, 64);
    // Critical assertion: weight is the PAYLOAD value (0x42),
    // not the HEADER sentinel (0xDEADBEEFCAFEBABE). If the
    // renderer skipped the header_skip step it would decode the
    // header bytes as the payload struct.
    assert_eq!(
        weight, 0x42,
        "slot-start chase must skip header — weight \
         must be payload value 0x42, not header sentinel \
         0xDEADBEEFCAFEBABE",
    );
    assert_eq!(
        cast_annotation.as_deref(),
        Some("sdt_alloc"),
        "Type::Ptr arm slot-start bridge resolve must surface 'sdt_alloc' annotation",
    );
}

/// Type::Ptr arena arm with a Fwd pointee but no bridge entry for
/// the chased value: the renderer surfaces the existing
/// "forward declaration; body not in this BTF" skip reason. Pin
/// the no-op behaviour so a misconfigured bridge (empty
/// `arena_type_at` / unkeyed addresses) cannot accidentally render
/// against an unrelated type.
#[test]
fn arena_chase_fwd_target_no_bridge_entry_skips() {
    let (blob, outer_id, _fwd_id, _task_ctx_id) = bridge_btf_outer_fwd_taskctx();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    // Reader has the arena window configured but NO entry for
    // TARGET_ADDR in arena_type_at. The bridge call returns
    // `None`; the renderer must surface the standard Fwd skip.
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
        ..
    } = members[0].value
    else {
        panic!("data field must render as Ptr; got {:?}", members[0].value);
    };
    assert!(
        deref.is_none(),
        "no-bridge Fwd target must not produce a deref"
    );
    assert!(
        cast_annotation.is_none(),
        "no-bridge resolve must leave cast_annotation None on the Type::Ptr arm; got {cast_annotation:?}",
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("Fwd-no-bridge must populate skip reason");
    assert!(
        reason.contains("forward declaration"),
        "skip reason must surface the forward-declaration cause; got: {reason}",
    );
    assert!(
        reason.contains("sdt_data"),
        "skip reason must include the Fwd type name; got: {reason}",
    );
}

/// Cast intercept arena arm with a Fwd target: the cast analyzer
/// produced a hit for a `u64` field but the target type id is a
/// forward declaration. The bridge resolves it, the chase
/// succeeds, and the resulting `Ptr` carries
/// `cast_annotation: Some("cast→arena (sdt_alloc)")`.
#[test]
fn cast_chase_arena_fwd_target_resolved_via_bridge() {
    // The cast intercept fires on a plain u64 field whose cast
    // hit's target is a Fwd; the bridge resolves the Fwd to a
    // real struct. Build a synthetic BTF with that exact shape
    // (the shared `bridge_btf_outer_fwd_taskctx` fixture has a
    // `struct sdt_data *` field instead, which the cast
    // intercept does not exercise).
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
    let n_task = push(&mut strings, "task_ctx");
    let n_f = push(&mut strings, "f");
    let n_weight = push(&mut strings, "weight");
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct T { u64 f @ 0; } size=8.
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: BTF_KIND_FWD struct sdt_data (the cast hit's
        // target_type_id — body absent, body not in this BTF).
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: false,
        },
        // id 4: struct task_ctx { u64 weight @ 0; } size=8 (the
        // bridge's resolved id — distinct from sdt_data so the
        // cast_annotation must reflect that the bridge fired).
        CastSynType::Struct {
            name_off: n_task,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_weight,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let local_fwd_id: u32 = 3;
    let local_task_ctx_id: u32 = 4;

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x55u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    // Payload-start chase: header_skip = 0.
    arena_types.insert(
        TARGET_ADDR,
        ArenaResolveHit {
            target_type_id: local_task_ctx_id,
            header_skip: 0,
        },
    );
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: local_fwd_id,
            addr_space: AddrSpace::Arena,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
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
        panic!(
            "intercept must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref_skipped_reason.is_none(),
        "successful chase: no skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("bridge-resolved cast must produce a deref");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the resolved task_ctx Struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("task_ctx"),
        "bridge must land on the resolved struct, not the Fwd"
    );
    assert_eq!(inner_members.len(), 1);
    let RenderedValue::Uint { value, .. } = inner_members[0].value else {
        panic!(
            "task_ctx.weight must render as Uint, got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(value, 0x55);
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→arena (sdt_alloc)"),
        "cast intercept arena bridge must extend annotation with '(sdt_alloc)'",
    );
}

/// Cast intercept arena arm with a Fwd target but no bridge entry:
/// the bridge returns None, the chase falls through to the
/// existing "forward declaration" skip path. The resulting `Ptr`
/// carries `cast_annotation: Some("cast→arena")` (no `(sdt_alloc)`
/// suffix) — pinning the no-op annotation when the bridge does
/// not fire.
#[test]
fn cast_chase_arena_fwd_target_no_bridge_keeps_plain_annotation() {
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

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: fwd_id,
            addr_space: AddrSpace::Arena,
        },
    );
    // Arena window configured, NO arena_type_at entries → bridge
    // returns None for every chase.
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
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
        ..
    } = members[0].value
    else {
        panic!(
            "intercept must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert!(
        deref.is_none(),
        "no-bridge Fwd cast must not produce a deref"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("Fwd-no-bridge must populate skip reason");
    assert!(
        reason.contains("forward declaration"),
        "skip reason must surface forward-declaration cause; got: {reason}",
    );
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→arena"),
        "no-bridge cast annotation must NOT include '(sdt_alloc)'; got {cast_annotation:?}",
    );
}

/// Cast intercept kernel arm with a Fwd target + bridge entry:
/// the bridge fires (mirrors the arena arm) and the cast
/// annotation extends to `cast→kernel (sdt_alloc)`. The reader's
/// `is_arena_addr` returns false (kernel-shaped value), so the
/// renderer dispatches to the kernel arm; the bridge wiring
/// covers the symmetric resolve there.
#[test]
fn cast_chase_kernel_fwd_target_resolved_via_bridge() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "u64");
    let n_t = push(&mut strings, "T");
    let n_fwd = push(&mut strings, "kern_fwd");
    let n_real = push(&mut strings, "kern_real");
    let n_f = push(&mut strings, "f");
    let n_x = push(&mut strings, "x");
    let types = vec![
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct T { u64 f @ 0; }
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: BTF_KIND_FWD struct kern_fwd
        CastSynType::Fwd {
            name_off: n_fwd,
            is_union: false,
        },
        // id 4: struct kern_real { u64 x @ 0; }
        CastSynType::Struct {
            name_off: n_real,
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
    let fwd_id: u32 = 3;
    let real_id: u32 = 4;

    // KVA outside any arena window — the runtime dispatcher routes
    // to the kernel arm. Use 0xffff_8000_... pattern (the kernel
    // direct-map range) so plausibility makes sense.
    const KVA: u64 = 0xffff_8000_0000_4000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let inner_bytes = 0x77u64.to_le_bytes().to_vec();
    let mut kva_bytes = std::collections::HashMap::new();
    kva_bytes.insert(KVA, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    // Payload-start chase: header_skip = 0.
    arena_types.insert(
        KVA,
        ArenaResolveHit {
            target_type_id: real_id,
            header_skip: 0,
        },
    );
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: fwd_id,
            addr_space: AddrSpace::Kernel,
        },
    );
    // No arena_window — `is_arena_addr` returns false for KVA, so
    // the dispatcher routes to the kernel arm.
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        kva_bytes_at: kva_bytes,
        arena_type_at: arena_types,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
        ..
    } = members[0].value
    else {
        panic!("intercept must produce Ptr; got {:?}", members[0].value);
    };
    assert!(
        deref_skipped_reason.is_none(),
        "kernel arm bridge resolve must succeed; got skip reason {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("kernel arm bridge resolve must produce a deref");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be kern_real Struct, got {inner:?}");
    };
    assert_eq!(inner_name.as_deref(), Some("kern_real"));
    let RenderedValue::Uint { value, .. } = inner_members[0].value else {
        panic!(
            "kern_real.x must render as Uint, got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(value, 0x77);
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→kernel (sdt_alloc)"),
        "kernel arm bridge must extend annotation with '(sdt_alloc)'",
    );
}

/// Cast intercept arena arm with `target_type_id == 0` (the
/// STX-flow analyzer sentinel for "deferred resolve"): the
/// `chase_arena_pointer` special case at the head of the helper
/// consults [`MemReader::resolve_arena_type`] BEFORE the normal
/// peel + Fwd resolve, expecting the bridge to supply the real
/// payload type id. With a populated `arena_type_at` entry the
/// chase succeeds: the bridge returns the resolved struct id, the
/// renderer reads `btf_size` bytes from the chased address (no
/// header skip), renders the payload struct, and `cast_ptr` emits
/// `cast_annotation: "cast→arena (sdt_alloc)"` because
/// `outcome.sdt_alloc_resolved == true` for the deferred-resolve
/// path (line ~3031 in `mod.rs`).
///
/// Pins the new STX-flow renderer path: a regression that broke
/// the deferred-resolve special case would surface as either a
/// miss (skip reason) or a wrong-id chase (rendered struct name
/// reflects the unrelated u64 underlying type).
#[test]
fn cast_chase_arena_target_type_id_zero_resolves_via_resolve_arena_type() {
    // BTF: u64(1), T(2, u64@0 source field), Q(3, u64@0 payload).
    // The CastHit's target_type_id is 0 (deferred); the bridge
    // must supply Q's id at chase time so the rendered subtree
    // names "Q", not "T" or anything else.
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    // Payload-start chase: header_skip = 0 — the renderer reads
    // `btf_size` bytes starting at TARGET_ADDR.
    arena_types.insert(
        TARGET_ADDR,
        ArenaResolveHit {
            target_type_id: q_id,
            header_skip: 0,
        },
    );
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            // STX-flow sentinel: analyzer left the target id
            // unresolved, expecting the bridge to fill it in.
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
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
        panic!(
            "intercept must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref_skipped_reason.is_none(),
        "deferred-resolve bridge fire must not surface a skip reason; \
         got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("deferred-resolve bridge must produce a deref");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the resolved Q Struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "bridge must land on the resolved struct's name (Q), \
         not the analyzer's deferred sentinel",
    );
    let RenderedValue::Uint { value, .. } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(value, 0x42);
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→arena (sdt_alloc)"),
        "deferred-resolve bridge fire must extend annotation with \
         '(sdt_alloc)' since `outcome.sdt_alloc_resolved` is set; \
         got {cast_annotation:?}",
    );
}

/// Cast intercept arena arm with `target_type_id == 0` AND no
/// bridge entry: the `chase_arena_pointer` special case calls
/// [`MemReader::resolve_arena_type`], gets `None`, and surfaces a
/// skip reason mentioning that the analyzer's STX-flow path tagged
/// the slot as Arena with deferred resolve but the bridge had no
/// entry. Pin the skip reason text so an operator reading a
/// failure dump can correlate the analyzer's hint with the
/// missing bridge population.
///
/// Without this gate, a stale or absent allocator pre-pass would
/// fall through to the normal peel + Fwd resolve path with
/// target_type_id=0 — which would either fail or, worse, succeed
/// against an unrelated BTF id 0 if such existed.
#[test]
fn cast_chase_arena_target_type_id_zero_no_bridge_entry_skips() {
    let (blob, t_id, _q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        },
    );
    // arena_window configured (so `is_arena_addr` returns true and
    // the chase enters the arena arm), but `arena_type_at` is
    // empty — the bridge query for TARGET_ADDR returns None.
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
        panic!(
            "intercept must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref.is_none(),
        "no-bridge deferred-resolve must not produce a deref"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("no-bridge deferred-resolve must populate skip reason");
    assert!(
        reason.contains("STX-flow path tagged slot as Arena"),
        "skip reason must surface the analyzer's STX-flow tag cause; \
         got: {reason}",
    );
    // `outcome.sdt_alloc_resolved` is `false` on the no-bridge
    // path, so the annotation stays at the unprefixed `cast→arena`
    // form (no `(sdt_alloc)` suffix).
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→arena"),
        "no-bridge deferred-resolve must NOT include '(sdt_alloc)' suffix; \
         got {cast_annotation:?}",
    );
}

/// Dedup short-circuit. When `is_already_rendered` returns
/// true for the chased arena address, `chase_arena_pointer`
/// surfaces a `Ptr` with `deref: None` and the
/// `"already rendered in sdt_allocations"` skip reason — no
/// arena read, no bridge query, no recursive render. The dedup
/// fires BEFORE the deferred-resolve special case so an address
/// pointing at a slot the sdt_alloc pre-pass already rendered
/// short-circuits even when the analyzer would otherwise have
/// supplied a bridge hit.
#[test]
fn cast_chase_already_rendered_short_circuits() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    arena_types.insert(
        TARGET_ADDR,
        ArenaResolveHit {
            target_type_id: q_id,
            header_skip: 0,
        },
    );
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        },
    );
    // Seed the dedup set with TARGET_ADDR's low-32 bits — the
    // production [`AccessorMemReader::is_already_rendered`] keys
    // on `addr as u32` (low-32 windowed slot start). Even though
    // arena_type_at would have resolved the address, the dedup
    // takes precedence and skips the chase.
    let mut rendered = std::collections::HashSet::new();
    rendered.insert(TARGET_ADDR as u32);
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
        rendered_slot_addrs: rendered,
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
            "dedup must still produce Ptr (only the deref is suppressed); \
             got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref.is_none(),
        "dedup short-circuit must suppress the deref"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("dedup must populate the skip reason");
    assert_eq!(
        reason, "already rendered in sdt_allocations",
        "dedup skip reason is wire-stable (operator reads it from \
         RenderedValue::Ptr::deref_skipped_reason); the exact format \
         is part of the dump's machine-checkable contract: got '{reason}'"
    );
}

/// Dedup with miss falls through to normal chase. An
/// address NOT in `rendered_slot_addrs` proceeds with the
/// existing chase pipeline (bridge query, peel, read, render).
/// Pins that the dedup gate is per-address and does not blank
/// the chase wholesale when a different slot was rendered.
#[test]
fn cast_chase_already_rendered_miss_proceeds_with_normal_chase() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    const RENDERED_OTHER_ADDR: u64 = 0x10_0000_2000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    arena_types.insert(
        TARGET_ADDR,
        ArenaResolveHit {
            target_type_id: q_id,
            header_skip: 0,
        },
    );
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        },
    );
    // Dedup set has a different slot start. The chase target
    // (TARGET_ADDR) is NOT in the set, so dedup misses and the
    // chase proceeds normally through the bridge.
    let mut rendered = std::collections::HashSet::new();
    rendered.insert(RENDERED_OTHER_ADDR as u32);
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
        rendered_slot_addrs: rendered,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr { ref deref, .. } = members[0].value else {
        panic!("expected Ptr, got {:?}", members[0].value);
    };
    let inner = deref
        .as_deref()
        .expect("dedup-miss path must still produce a deref via the bridge");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        ..
    } = *inner
    else {
        panic!("deref payload must be the resolved Q Struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "dedup-miss path must land on Q via the normal chase pipeline",
    );
}

/// Default `is_already_rendered` returns false. Readers
/// without a rendered-slot index (the trait default impl)
/// proceed with the chase — pins the no-regression case for
/// every existing renderer that doesn't wire the dedup set.
#[test]
fn cast_chase_default_is_already_rendered_returns_false() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut arena_types = std::collections::HashMap::new();
    arena_types.insert(
        TARGET_ADDR,
        ArenaResolveHit {
            target_type_id: q_id,
            header_skip: 0,
        },
    );
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        },
    );
    // No rendered_slot_addrs — the field defaults to an empty
    // HashSet, so `is_already_rendered` returns false for every
    // address. The chase must proceed without the short-circuit.
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        arena_type_at: arena_types,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr { ref deref, .. } = members[0].value else {
        panic!("expected Ptr, got {:?}", members[0].value);
    };
    assert!(
        deref.is_some(),
        "empty rendered_slot_addrs must NOT short-circuit the chase",
    );
}

/// Cast intercept kernel arm with `target_type_id == 0`: the
/// analyzer hinted Arena (the STX-flow sentinel only emits with
/// `addr_space: Arena`) but the runtime value falls outside the
/// arena window so `is_arena_addr` returns false and the kernel
/// arm fires. The kernel arm's special case at line ~3390 of
/// `mod.rs` recognises `target_type_id == 0` as the cgx-bridge
/// sentinel and surfaces a skip reason explaining the
/// analyzer/runtime mismatch — without a BTF id there is no way
/// to size the kernel read.
///
/// Pins the kernel-arm fall-through behaviour: a regression that
/// stripped the special case would attempt to peel type id 0 in
/// the program BTF (which fails) and surface a less useful
/// "kernel cast target type id 0 unresolvable" message.
#[test]
fn cast_chase_kernel_target_type_id_zero_falls_through_with_mismatch_reason() {
    let (blob, t_id, _q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // KVA pattern (kernel direct-map range) outside ANY arena
    // window — the dispatcher routes to the kernel arm. The reader
    // has no `arena_window` configured, so `is_arena_addr` returns
    // `false` for every value and the kernel arm receives the
    // chase.
    const KVA: u64 = 0xffff_8000_0000_4000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            // STX-flow analyzer sentinel — intentionally `Arena`
            // since that is the only space the analyzer emits with
            // target_type_id=0. Runtime detection sees the value
            // outside the arena window and routes to the kernel
            // arm, exercising the kernel-arm `target_type_id == 0`
            // special case.
            target_type_id: 0,
            addr_space: AddrSpace::Arena,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
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
        panic!(
            "intercept must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref.is_none(),
        "kernel-arm `target_type_id == 0` special case must skip the chase",
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("kernel-arm `target_type_id == 0` must populate skip reason");
    assert!(
        reason.contains("kernel cast target unresolved"),
        "skip reason must mention `kernel cast target unresolved`; \
         got: {reason}",
    );
    assert!(
        reason.contains("analyzer hinted Arena with deferred resolve"),
        "skip reason must surface the analyzer-hint / runtime-window \
         mismatch; got: {reason}",
    );
    // Kernel-arm path: `cast_ptr` is called with `sdt_alloc_resolved = false`
    // (line ~3401 in mod.rs), so the annotation reflects the actual
    // path taken (kernel) without the sdt_alloc suffix.
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→kernel"),
        "kernel-arm fall-through must use `cast→kernel` annotation \
         (the path actually taken); got {cast_annotation:?}",
    );
}

/// `Type::Ptr` arena arm: an out-of-arena-window pointer must
/// NOT fire the sdt_alloc bridge, even when the
/// `MemReader::resolve_arena_type` table contains a stale entry
/// for that exact address.
///
/// The arm dispatches on `is_arena_addr(value)` BEFORE entering
/// `chase_arena_pointer`. The bridge lives inside the chase
/// helper, so an out-of-window value skips the helper entirely
/// and falls into the kernel-kptr branch (cpumask-name dispatch /
/// Fwd-no-body skip). This test pins the bridge's no-op behaviour
/// for the kptr branch by asserting the rendered Ptr has neither a
/// successful deref nor an `sdt_alloc` annotation.
///
/// The fixture's bridge map keys on the FULL 64-bit address
/// (`CastStubReader` does not implement the production
/// `addr & 0xFFFF_FFFF` mask). That is fine for this test — the
/// gating happens before the lookup runs.
#[test]
fn arena_chase_bridge_address_outside_window_is_no_op() {
    let (blob, outer_id, _fwd_id, task_ctx_id) = bridge_btf_outer_fwd_taskctx();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    // OUT_OF_WINDOW lies BELOW the configured arena window; the
    // BTF Type::Ptr arm dispatches on `is_arena_addr` so it never
    // reaches `chase_arena_pointer` for this value, even though
    // the bridge index has an entry for it. Verifies that an
    // out-of-window address with a stale bridge entry cannot
    // accidentally surface as a chased struct.
    const OUT_OF_WINDOW: u64 = 0x0F_0000_1000;
    let outer_bytes = OUT_OF_WINDOW.to_le_bytes().to_vec();
    let mut arena_types = std::collections::HashMap::new();
    // Stale entry mapped to a payload-start shape — the gate must
    // reject the address before this entry is consulted.
    arena_types.insert(
        OUT_OF_WINDOW,
        ArenaResolveHit {
            target_type_id: task_ctx_id,
            header_skip: 0,
        },
    );
    let reader = CastStubReader {
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_type_at: arena_types,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ref cast_annotation,
        ..
    } = members[0].value
    else {
        panic!("data field must render as Ptr; got {:?}", members[0].value);
    };
    // The Type::Ptr arm reaches `chase_arena_pointer` only when
    // `is_arena_addr` returned true. For an out-of-window value,
    // the kernel-kptr branch of the Type::Ptr arm runs; that
    // branch has its own peel/size resolve and bails on a Fwd
    // pointee whose body is missing.
    assert!(
        deref.is_none(),
        "out-of-window pointer must not chase via the bridge"
    );
    assert!(
        cast_annotation.is_none(),
        "BTF Type::Ptr arm must leave cast_annotation None on the kptr branch"
    );
    // Surface either the cpumask-name dispatch reason
    // ("size 0" or absent), the Fwd reason, or the BTF-resolution
    // failure — any of which is correct for the kptr branch with
    // a Fwd pointee. The test only asserts the bridge did NOT
    // fire (no successful deref, no annotation).
    let _ = deref_skipped_reason;
}

/// Cross-BTF Fwd resolution end-to-end: the entry BTF declares
/// `outer { u64 cgx_raw @ 0 }` plus `struct cgx_target;` (a
/// `BTF_KIND_FWD` only — no body). A sibling BTF defines
/// `struct cgx_target { u64 marker @ 0 }` (the body). The cast
/// analyzer recovered `(outer, 0) -> (cgx_target, Arena)` so the
/// chase enters [`render_cast_pointer`] → arena branch →
/// [`chase_arena_pointer`]. Local Fwd resolve fails (no sibling in
/// the entry BTF), the sdt_alloc bridge stays dormant, then
/// [`try_cross_btf_fwd_resolve`] consults
/// [`MemReader::cross_btf_resolve_fwd`] which returns the sibling
/// BTF's `cgx_target` body. The recursion renders against that
/// body and produces `cgx_target { marker = 0xCAFE }`.
///
/// Without the cross-BTF bridge, the chase would have skipped
/// with "forward declaration; body not in this BTF" and the
/// rendered output would be a bare `Ptr` carrying the chased
/// address.
#[test]
fn cross_btf_fwd_resolve_renders_cgx_body_through_sibling_btf() {
    use std::sync::Arc;

    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };

    // Entry BTF: outer (id 2) carrying a u64 cgx_raw @ 0; Fwd of
    // cgx_target (id 3, struct, no body).
    let mut s_a = vec![0u8];
    let n_a_u64 = push(&mut s_a, "u64");
    let n_a_outer = push(&mut s_a, "outer");
    let n_a_field = push(&mut s_a, "cgx_raw");
    let n_a_cgx = push(&mut s_a, "cgx_target");
    let types_a = vec![
        CastSynType::Int {
            name_off: n_a_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_a_outer,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_a_field,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Fwd {
            name_off: n_a_cgx,
            is_union: false,
        },
    ];
    let blob_a = cast_build_btf(&types_a, &s_a);
    let btf_entry = Btf::from_bytes(&blob_a).expect("entry BTF parses");

    // Sibling BTF: cgx_target as a complete struct (id 2) with
    // `u64 marker @ 0`.
    let mut s_b = vec![0u8];
    let n_b_u64 = push(&mut s_b, "u64");
    let n_b_cgx = push(&mut s_b, "cgx_target");
    let n_b_marker = push(&mut s_b, "marker");
    let types_b = vec![
        CastSynType::Int {
            name_off: n_b_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_b_cgx,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_b_marker,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob_b = cast_build_btf(&types_b, &s_b);
    let btf_sibling = Arc::new(Btf::from_bytes(&blob_b).expect("sibling BTF parses"));

    // Configure CastStubReader: outer.cgx_raw u64 maps to a
    // Pointer{cgx_target}; the chased value is an arena address;
    // arena bytes at that address carry the cgx_target body.
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_2000;
    let outer_id = 2u32;
    // Cast hint resolves the u64 slot at (outer, 0) to the entry
    // BTF's Fwd `cgx_target` (id 3). The chase then asks the
    // reader to bridge that Fwd via cross-BTF resolution.
    let cgx_fwd_id = 3u32;
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (outer_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: cgx_fwd_id,
            addr_space: AddrSpace::Arena,
        },
    );
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    // Sibling cgx_target body: marker = 0xCAFE.
    arena_bytes.insert(TARGET_ADDR, 0xCAFEu64.to_le_bytes().to_vec());
    let mut cross_btf_index = std::collections::HashMap::new();
    cross_btf_index.insert("cgx_target".to_string(), (0usize, 2u32, true));
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        cross_btf_btfs: vec![btf_sibling.clone()],
        cross_btf_index,
        ..Default::default()
    };

    // Render outer; cgx_raw must surface as a Ptr whose deref
    // renders against the SIBLING BTF's cgx_target body — the
    // marker field is 0xCAFE.
    let v = render_value_with_mem(&btf_entry, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr { ref deref, .. } = members[0].value else {
        panic!("cgx_raw must render as Ptr; got {:?}", members[0].value);
    };
    let inner = deref
        .as_ref()
        .expect("cross-BTF Fwd resolve must produce a deref (sibling BTF body), but got None");
    let RenderedValue::Struct {
        ref type_name,
        ref members,
    } = **inner
    else {
        panic!("inner must be Struct (cgx_target body); got {inner:?}");
    };
    assert_eq!(
        type_name.as_deref(),
        Some("cgx_target"),
        "rendered subtree must carry the sibling BTF's struct name"
    );
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].name, "marker");
    let RenderedValue::Uint { value: marker, .. } = members[0].value else {
        panic!("marker must render as Uint; got {:?}", members[0].value);
    };
    assert_eq!(
        marker, 0xCAFE,
        "rendered marker must come from the cross-BTF body's bytes"
    );

    // Sanity: drop the cross-BTF index and re-render — without
    // the bridge, the chase must skip with the Fwd reason and
    // the deref stays None.
    let reader_no_bridge = CastStubReader {
        cast_map: Some({
            let mut m = crate::monitor::cast_analysis::CastMap::new();
            m.insert(
                (outer_id, 0),
                CastHit {
                    alloc_size: None,
                    target_type_id: cgx_fwd_id,
                    addr_space: AddrSpace::Arena,
                },
            );
            m
        }),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: {
            let mut a = std::collections::HashMap::new();
            a.insert(TARGET_ADDR, 0xCAFEu64.to_le_bytes().to_vec());
            a
        },
        ..Default::default()
    };
    let v = render_value_with_mem(&btf_entry, outer_id, &outer_bytes, &reader_no_bridge);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!("cgx_raw must render as Ptr; got {:?}", members[0].value);
    };
    assert!(
        deref.is_none(),
        "without cross-BTF bridge, Fwd target must not chase"
    );
    let reason = deref_skipped_reason
        .as_ref()
        .expect("Fwd skip must populate deref_skipped_reason");
    assert!(
        reason.contains("cgx_target") && reason.contains("forward declaration"),
        "skip reason must name the Fwd target: {reason:?}"
    );
}

/// Kernel-arm cross-BTF Fwd resolution: a `CastHit` with
/// `addr_space: Kernel` whose target_type_id resolves to a
/// `BTF_KIND_FWD` in the entry BTF — the renderer's kernel arm
/// dispatches on `is_arena_addr(value)` returning false (no arena
/// window matches), then peels the Fwd target. With the
/// sdt_alloc bridge dormant (no `arena_type_at` entry for the
/// kernel value), `chase_arena_pointer` is NOT invoked but the
/// kernel arm shares the same `try_cross_btf_fwd_resolve` shortcut
/// (line ~3447 in `mod.rs`): a sibling BTF whose
/// [`MemReader::cross_btf_resolve_fwd`] override matches the Fwd
/// name surfaces the body, the kernel read fires against the
/// sibling-BTF's resolved type id, and the rendered subtree
/// names the sibling struct.
///
/// Pin the symmetric kernel-arm wiring against a regression that
/// stripped the cross-BTF probe from the kernel arm (only arena
/// arm honoured it). The shared
/// [`try_cross_btf_fwd_resolve`] call at the kernel-arm
/// fall-through is the only mechanism for kernel-targeted Fwd
/// resolves through a sibling BTF.
#[test]
fn cast_chase_kernel_cross_btf_fwd_resolve_succeeds() {
    use std::sync::Arc;

    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };

    // Entry BTF: T (id 2) with a u64 field at offset 0; Fwd of
    // kern_target (id 3, struct, no body in this BTF).
    let mut s_entry = vec![0u8];
    let n_e_u64 = push(&mut s_entry, "u64");
    let n_e_t = push(&mut s_entry, "T");
    let n_e_f = push(&mut s_entry, "f");
    let n_e_kern = push(&mut s_entry, "kern_target");
    let types_entry = vec![
        CastSynType::Int {
            name_off: n_e_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_e_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_e_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: BTF_KIND_FWD struct kern_target — body lives in
        // the sibling BTF, the renderer must consult the cross-BTF
        // index.
        CastSynType::Fwd {
            name_off: n_e_kern,
            is_union: false,
        },
    ];
    let blob_entry = cast_build_btf(&types_entry, &s_entry);
    let btf_entry = Btf::from_bytes(&blob_entry).expect("entry BTF parses");
    let t_id: u32 = 2;
    let kern_fwd_id: u32 = 3;

    // Sibling BTF: kern_target as a complete struct (id 2) with
    // `u64 marker @ 0`.
    let mut s_sib = vec![0u8];
    let n_s_u64 = push(&mut s_sib, "u64");
    let n_s_kern = push(&mut s_sib, "kern_target");
    let n_s_marker = push(&mut s_sib, "marker");
    let types_sib = vec![
        CastSynType::Int {
            name_off: n_s_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_s_kern,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_s_marker,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob_sib = cast_build_btf(&types_sib, &s_sib);
    let btf_sib = Arc::new(Btf::from_bytes(&blob_sib).expect("sibling BTF parses"));

    // Kernel value outside any arena window — the dispatcher
    // routes to the kernel arm. Use the kernel direct-map range
    // so the plausibility-gate sanity holds (top byte 0xff would
    // trigger the freed-slab heuristic).
    const KVA: u64 = 0xffff_8000_0001_2000;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    // Sibling kern_target body: marker = 0xBEEF.
    let inner_bytes = 0xBEEFu64.to_le_bytes().to_vec();
    let mut kva_bytes = std::collections::HashMap::new();
    kva_bytes.insert(KVA, inner_bytes);
    let mut cross_btf_index = std::collections::HashMap::new();
    // Cross-BTF index: kern_target -> (sibling BTF index 0, type
    // id 2, want_struct=true).
    cross_btf_index.insert("kern_target".to_string(), (0usize, 2u32, true));
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: kern_fwd_id,
            addr_space: AddrSpace::Kernel,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        kva_bytes_at: kva_bytes,
        // No arena_window — `is_arena_addr` returns false for KVA,
        // dispatcher routes to the kernel arm. No `arena_type_at`
        // — the sdt_alloc bridge stays dormant on the kernel arm
        // so the cross-BTF shortcut fires instead.
        cross_btf_btfs: vec![btf_sib.clone()],
        cross_btf_index,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf_entry, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "kernel cast intercept must surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert!(
        deref_skipped_reason.is_none(),
        "kernel-arm cross-BTF Fwd resolve must succeed; \
         got skip reason {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_ref()
        .expect("kernel-arm cross-BTF Fwd resolve must produce a deref");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = **inner
    else {
        panic!("deref payload must be the kern_target body Struct; got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("kern_target"),
        "rendered subtree must carry the sibling BTF's struct name",
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "marker");
    let RenderedValue::Uint { value: marker, .. } = inner_members[0].value else {
        panic!(
            "kern_target.marker must render as Uint; got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(
        marker, 0xBEEF,
        "rendered marker must come from the kva-side body bytes \
         decoded against the sibling BTF",
    );
}

