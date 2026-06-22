use super::*;

// ---- POINTER_CHASE_CAP and parent-byte-boundary edge cases -----
//
// These tests pin the cap-and-boundary behavior that protects the
// renderer from unbounded reads (POINTER_CHASE_CAP) and from
// out-of-range slicing (parent_bytes boundary). Both produce
// `RenderedValue::Truncated` wrappers when the rendered subtree is
// partial, so the consumer can tell the rendered output is
// incomplete.

/// Arena cast target whose BTF size exceeds [`POINTER_CHASE_CAP`]
/// (4096 bytes). [`chase_arena_pointer`]'s
/// `let read_size = btf_size.min(POINTER_CHASE_CAP)` clamp limits
/// the read to 4096 bytes and sets `truncated_at_cap = true`, then
/// wraps the rendered struct in `RenderedValue::Truncated{needed:
/// btf_size, had: 4096, partial: ...}` at its `truncated_at_cap`
/// payload-wrap branch. Without
/// the cap, an analyzer that emits a 1 MiB struct would force the
/// renderer to allocate (and read) a megabyte from the arena snapshot
/// for a single failure dump — pulling the dump through
/// O(num_recovered_pointers * pointee_size) memory pressure.
#[test]
fn cast_chase_arena_pointee_exceeds_cap_wraps_in_truncated() {
    // Q is sized 5000 (above POINTER_CHASE_CAP=4096) so the cap
    // clamps the read. Single u64 member at offset 0 covers the
    // first 8 bytes; the remaining 4992 bytes are unaccounted-for
    // padding in the BTF wire format.
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
        CastSynType::Struct {
            name_off: n_q,
            size: 5000,
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

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    // Provide exactly 4096 bytes (POINTER_CHASE_CAP) at TARGET_ADDR
    // so `read_arena(TARGET_ADDR, 4096)` succeeds. Bytes 0..8 carry
    // Q.x = 0x77.
    let mut target_bytes = vec![0u8; 4096];
    target_bytes[0..8].copy_from_slice(&0x77u64.to_le_bytes());
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, target_bytes);
    // Use cast_map mode (NOT the universal `hit` field) so only the
    // outer T.f → Q intercept fires; Q.x at (q_id, 0) has no entry,
    // so the inner u64 falls through to the plain Uint render and
    // doesn't recurse into a phantom kernel chase.
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
        arena_bytes_at: arena_bytes,
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
            "cap-clamped chase must still surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(
        deref_skipped_reason.is_none(),
        "cap-clamped read is a SUCCESS; no skip reason expected; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("read succeeded → deref must be Some");
    // Outer wrap: Truncated{needed: 5000, had: 4096, partial: ...}.
    let RenderedValue::Truncated {
        needed,
        had,
        ref partial,
    } = *inner
    else {
        panic!("btf_size > POINTER_CHASE_CAP must wrap deref in Truncated; got {inner:?}");
    };
    assert_eq!(needed, 5000, "Truncated.needed must be Q's BTF size");
    assert_eq!(
        had, 4096,
        "Truncated.had must equal POINTER_CHASE_CAP (4096)"
    );
    // `render_struct` itself emitted a Truncated wrapper because
    // 4096 < Q.size=5000; walk through it to reach the inner Struct.
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
    assert_eq!(inner_name.as_deref(), Some("Q"));
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(bits, 64);
    assert_eq!(
        value, 0x77,
        "first 8 bytes of cap-clamped read must decode correctly"
    );
}

/// Cast intercept on a u64 member where `byte_off + 8 >
/// parent_bytes.len()` — the `field_bytes.get(..8)?` boundary
/// guard in [`try_cast_intercept`] returns `None`, the intercept
/// does not fire, and execution falls through to the existing
/// partial-decode path in [`render_member`] (the `byte_off + size
/// <= parent_bytes.len()` check that wraps a short member in
/// `RenderedValue::Truncated`) which
/// emits a `RenderedValue::Truncated` wrapping whatever the inner
/// renderer salvaged. Without this guard, the intercept would slice
/// `parent_bytes[byte_off..byte_off+8]` past the slice's end and
/// either crash or read uninitialized memory off the end of the
/// allocation — a critical bug given the renderer runs over
/// guest-supplied bytes the analyzer already corrupted somewhere.
#[test]
fn cast_intercept_u64_at_parent_bytes_boundary_falls_through() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // T's u64 member is at offset 0 with size 8 — the full member
    // needs 8 bytes. Provide only 4 bytes so `byte_off + 8 = 8 > 4
    // = parent_bytes.len()` and the intercept short-circuits. The
    // configured cast hit + arena window + bytes are all set up so
    // a buggy intercept that ignored the boundary check would
    // produce a Ptr render — a successful test confirms only Truncated.
    let outer_bytes = vec![0xCA, 0xFE, 0xBA, 0xBE];
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(0xBEBA_FECAu64, vec![0u8; 8]);
    let reader = CastStubReader {
        hit: Some(CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        arena_window: Some((0, u64::MAX)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    // The outer T struct is 8 bytes but only 4 were supplied, so
    // `render_struct` wraps in `Truncated{needed:8, had:4, partial:
    // Struct{T}}`. The intercept's boundary guard at
    // [`try_cast_intercept`]'s `field_bytes.get(..8)?` boundary
    // guard short-circuits when `byte_off + 8 > parent_bytes.len()`,
    // so T.f also surfaces as a per-member `Truncated`, NOT a `Ptr`.
    let outer_struct = match &v {
        RenderedValue::Struct { .. } => &v,
        RenderedValue::Truncated { partial, .. } => partial.as_ref(),
        other => panic!("expected Struct or Truncated{{Struct}}; got {other:?}"),
    };
    let RenderedValue::Struct { ref members, .. } = *outer_struct else {
        panic!("expected Struct under outer Truncated; got {outer_struct:?}");
    };
    match &members[0].value {
        RenderedValue::Truncated { needed, had, .. } => {
            assert_eq!(*needed, 8, "u64 needs 8 bytes");
            assert_eq!(*had, 4, "supplied bytes for member is 4");
        }
        RenderedValue::Ptr { .. } => panic!(
            "boundary fall-through must NOT produce Ptr — intercept's \
             boundary guard (`field_bytes.get(..8)?` in \
             try_cast_intercept) short-circuits; got {:?}",
            members[0].value
        ),
        other => panic!("boundary fall-through must produce Truncated; got {other:?}"),
    }
}

// ---- Int-encoding gate paths -----------------------------------
//
// The intercept's `int.size() != 8 || int.is_signed() ||
// int.is_bool() || int.is_char()` encoding gate in
// [`try_cast_intercept`] rejects every non-plain-unsigned-u64
// member regardless of the
// reader's `cast_lookup` hit. The size-mismatch case
// (`cast_intercept_non_u64_field_not_intercepted`) is already
// covered above; these tests cover the encoding flags. BPF stores
// recovered typed pointers in plain-unsigned u64 slots only; a
// `_Bool`, `char`, or signed 8-byte field is structurally not the
// analyzer's output shape and must NOT be intercepted.

/// Cast intercept on a `_Bool`-encoded 8-byte field with a fixed
/// hit returned by `cast_lookup`: the intercept's gate at
/// the encoding gate in [`try_cast_intercept`] rejects
/// (int.is_bool() == true) before
/// `cast_lookup` is consulted, and the renderer falls through to
/// the normal Int render — `RenderedValue::Bool{value}` (an 8-byte
/// _Bool encodes as is_bool() at the BTF level; render_int
/// produces Bool when is_bool()).
#[test]
fn cast_intercept_bool_field_not_intercepted() {
    // Build T with a single 8-byte _Bool member at offset 0.
    // Encoding = BTF_INT_BOOL (= 4); size = 8; bits = 64. The Q
    // type is included so the cast hit can reference a real id, but
    // the gate rejects before the hit is consulted.
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let types = vec![
        // id 1: 8-byte _Bool. encoding=4 = BTF_INT_BOOL.
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 4,
            offset: 0,
            bits: 64,
        },
        // id 2: struct T { _Bool f @ 0 }, size 8.
        CastSynType::Struct {
            name_off: n_t,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: struct Q { _Bool x @ 0 }, size 8 (cast target;
        // unused since the gate rejects first, but a valid id).
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

    // Outer T bytes: a non-zero _Bool value (0x01). Reader returns
    // Some(hit) for any (parent, offset) — the bool gate must
    // reject before this is consulted.
    let outer_bytes = 1u64.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        arena_window: Some((0, u64::MAX)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    // The bool gate must reject the intercept. Render path produces
    // Bool — the cast intercept must NOT have fired.
    match &members[0].value {
        RenderedValue::Bool { value } => {
            assert!(*value, "bool value 0x01 must render as true");
        }
        RenderedValue::Ptr { .. } => panic!(
            "_Bool field must NOT be intercepted (int.is_bool() gate at \
             encoding gate in try_cast_intercept rejects); got {:?}",
            members[0].value
        ),
        other => panic!("_Bool field must render as Bool; got {other:?}"),
    }
}

/// Cast intercept on a SIGNED 8-byte int member with a fixed hit:
/// the encoding gate in [`try_cast_intercept`] rejects
/// (int.is_signed() == true) before `cast_lookup` is consulted, and
/// the renderer falls through to the normal Int render —
/// `RenderedValue::Int{bits: 64, value: <signed value>}`. Signed
/// 8-byte ints in BPF programs are typically counters / deltas, not
/// recovered pointers; the gate keeps the analyzer-emitted hits
/// from contaminating a counter slot.
#[test]
fn cast_intercept_signed_8byte_int_not_intercepted() {
    // Build T with a single 8-byte SIGNED int member at offset 0.
    // Encoding = BTF_INT_SIGNED (= 1); size = 8; bits = 64.
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let types = vec![
        // id 1: 8-byte signed int. encoding=1 = BTF_INT_SIGNED.
        CastSynType::Int {
            name_off: n_int,
            size: 8,
            encoding: 1,
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
        // id 3: struct Q with the same signed-int field. Cast target
        // is unused since the gate rejects first.
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

    // Outer bytes: signed -1 (all-ones u64). Without the gate, the
    // reader would interpret 0xFFFFFFFFFFFFFFFF as a pointer value
    // — the gate must reject the intercept.
    let outer_bytes = (-1i64).to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        arena_window: Some((0, u64::MAX)),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    match &members[0].value {
        RenderedValue::Int { bits, value } => {
            assert_eq!(
                *bits, 64,
                "signed int must render at its declared 64-bit width"
            );
            assert_eq!(*value, -1, "signed -1 must round-trip as Int{{value: -1}}");
        }
        RenderedValue::Ptr { .. } => panic!(
            "signed 8-byte int must NOT be intercepted (int.is_signed() \
             encoding gate in try_cast_intercept rejects); got {:?}",
            members[0].value
        ),
        other => panic!("signed 8-byte int must render as Int; got {other:?}"),
    }
}

// ---- parent_type_id == None path -------------------------------
//
// `render_member`'s `parent_type_id` parameter is `Option<u32>` at
// [`render_member`]'s parameter list. Through the public entry
// points (`render_value_with_mem` → `render_value_inner` →
// `render_struct`'s per-member loop) it is always
// `Some(parent_type_id)`. The `None` case is reachable only by
// calling `render_member` directly. [`render_member`]'s
// `parent_type_id.and_then(|parent| ...)` guard short-circuits
// when `None`, leaving the closure's
// `cast_intercept` as `None` and the renderer falling through to
// the unmodified path. This test pins the no-crash, no-intercept
// behavior.

/// Direct call to `render_member` with `parent_type_id = None` must
/// short-circuit the cast intercept ([`render_member`]'s
/// `parent_type_id.and_then(...)` guard) and produce the
/// same render as the no-cast case — for a u64 field, that is
/// `RenderedValue::Uint{bits: 64, value}`. Establishes the contract
/// for any future entry point that bypasses `render_struct` (e.g.
/// rendering a stand-alone member from a synthesized layout): such
/// callers must explicitly opt in to the cast intercept by passing
/// the parent struct id, never silently inheriting it.
#[test]
fn cast_intercept_parent_type_id_none_does_not_crash() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Resolve T's first member directly so we can hand it to
    // render_member without going through render_struct (which
    // always passes Some(parent_type_id)).
    let Type::Struct(t_struct) = btf.resolve_type_by_id(t_id).expect("T resolves") else {
        panic!("T_id resolves to non-Struct type");
    };
    let m = t_struct.members.first().expect("T has one member");

    // Configure the reader so a buggy renderer that ignored the
    // None short-circuit and called cast_lookup anyway WOULD chase
    // the value. The correct render must produce Uint regardless.
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, 0xAAu64.to_le_bytes().to_vec());
    let reader = CastStubReader {
        hit: Some(CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    // Direct call to render_member with parent_type_id = None.
    let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let v = render_member(
        &btf,
        m,
        None,
        &outer_bytes,
        0,
        Some(&reader as &dyn MemReader),
        &mut visited,
    );

    // None short-circuit MUST keep the intercept off; the u64
    // renders as Uint with the loaded value.
    let RenderedValue::Uint { bits, value } = v else {
        panic!(
            "parent_type_id=None must short-circuit the intercept and \
             render as Uint; got {v:?}. A failure here means \
             render_member's `let parent = parent_type_id?` guard at \
             `parent_type_id.and_then(...)` guard in render_member \
             was bypassed."
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, TARGET_ADDR);
}

// ---- Recursive cast chase + modifier-chain unit test -----------
//
// A cast target whose own struct contains a u64 field flagged by
// the cast analyzer drives the renderer into a recursive chase:
// the outer chase produces a `Ptr{ deref: Some(Struct{ ... }) }`,
// and the inner Struct's u64 field surfaces as another nested
// `Ptr{ deref: Some(Struct{ ... }) }`. The visited set carries the
// outer address through the recursion so genuine cycles still
// surface as `[cycle]` (covered by `cast_chase_cycle_detection`).
// This test pins the non-cycle recursive case where two distinct
// addresses chain.
//
// The modifier-chain unit test renders a `Typedef -> Const ->
// Struct(T)` surface type via `cast_map`-mode lookup keyed on
// `T_id` directly. The renderer's `peel_modifiers_with_id` MUST
// produce the underlying `T_id` so `cast_lookup(T_id, 0)` hits.
// This is a focused unit-only sibling of
// `cast_pipeline_modifier_chain_renderer_peels_to_analyzer_struct_id`
// (which also runs the analyzer); the pure-renderer test isolates
// the peel behavior so a regression in one half (renderer or
// analyzer) is distinguishable from a regression in the other.

/// Target struct whose body itself contains a cast-flagged u64
/// field. The renderer's recursion must:
///   - chase the outer Ptr (T.f → Q struct at TARGET_ADDR),
///   - render Q.x as a NESTED Ptr (cast hit at (Q, 0)),
///   - chase Q.x → R struct at TARGET_ADDR_2 → R.y = Uint(0xBB).
///
/// Both deref payloads must be present (deref: Some) and neither
/// should carry a skip reason.
#[test]
fn cast_chase_recursive_target_with_inner_cast_field() {
    // Build a 4-type BTF: u64, T (8 bytes, u64 @ 0), Q (8 bytes,
    // u64 @ 0), R (8 bytes, u64 @ 0). The CastMap entries are:
    //   (T_id, 0) → (Q_id, Arena)
    //   (Q_id, 0) → (R_id, Arena)
    // R has no flagged member, so the recursion terminates.
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let mut strings = strings;
    let n_r = push(&mut strings, "R");
    let n_y = push(&mut strings, "y");
    let types = vec![
        // id 1: u64
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
        // id 3: Q { u64 x @ 0 }
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 4: R { u64 y @ 0 }
        CastSynType::Struct {
            name_off: n_r,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_y,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let t_id: u32 = 2;
    let q_id: u32 = 3;
    let r_id: u32 = 4;

    let mut cast_map: crate::monitor::cast_analysis::CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );
    cast_map.insert(
        (q_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: r_id,
            addr_space: AddrSpace::Arena,
        },
    );

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    const TARGET_ADDR_2: u64 = 0x10_0000_2000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    // Q at TARGET_ADDR: Q.x = TARGET_ADDR_2 (the inner cast value).
    let q_bytes: Vec<u8> = TARGET_ADDR_2.to_le_bytes().to_vec();
    // R at TARGET_ADDR_2: R.y = 0xBB (terminal value).
    let r_bytes: Vec<u8> = 0xBBu64.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, q_bytes);
    arena_bytes.insert(TARGET_ADDR_2, r_bytes);
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
    // Outer T.f: Ptr → Q.
    let RenderedValue::Ptr {
        value: outer_value,
        deref: ref outer_deref,
        deref_skipped_reason: ref outer_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "outer cast intercept must surface as Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(outer_value, TARGET_ADDR);
    assert!(
        outer_reason.is_none(),
        "outer chase must succeed; got {outer_reason:?}"
    );
    let q_inner = outer_deref
        .as_deref()
        .expect("outer deref Some (chase succeeded)");
    let RenderedValue::Struct {
        type_name: ref q_name,
        members: ref q_members,
    } = *q_inner
    else {
        panic!("outer deref must be Q struct; got {q_inner:?}");
    };
    assert_eq!(q_name.as_deref(), Some("Q"));
    assert_eq!(q_members.len(), 1);
    assert_eq!(q_members[0].name, "x");
    // Inner Q.x: nested Ptr → R (the cast intercept fires
    // recursively because the renderer recurses with parent_type_id
    // = Q_id, and the CastMap has (Q, 0) → (R, Arena)).
    let RenderedValue::Ptr {
        value: inner_value,
        deref: ref inner_deref,
        deref_skipped_reason: ref inner_reason,
        ..
    } = q_members[0].value
    else {
        panic!(
            "inner Q.x must surface as Ptr (recursive cast hit); got {:?}. \
             A failure here means the renderer didn't pass Q_id as \
             parent_type_id when recursing through the deref payload.",
            q_members[0].value
        );
    };
    assert_eq!(inner_value, TARGET_ADDR_2);
    assert!(
        inner_reason.is_none(),
        "inner chase must succeed; got {inner_reason:?}"
    );
    let r_inner = inner_deref
        .as_deref()
        .expect("inner deref Some (chase succeeded)");
    let RenderedValue::Struct {
        type_name: ref r_name,
        members: ref r_members,
    } = *r_inner
    else {
        panic!("inner deref must be R struct; got {r_inner:?}");
    };
    assert_eq!(r_name.as_deref(), Some("R"));
    assert_eq!(r_members.len(), 1);
    assert_eq!(r_members[0].name, "y");
    // Terminal R.y: plain Uint (no cast entry at (R, 0)).
    let RenderedValue::Uint { bits, value } = r_members[0].value else {
        panic!(
            "R.y must terminate as Uint (no recursive cast entry at (R,0)); \
             got {:?}",
            r_members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0xBB);
}

/// Modifier-chain unit case: the parent type is rendered via a
/// `Typedef -> Const -> Struct(T)` chain, and the CastMap is keyed
/// on `(T_id, 0)` directly. The renderer's
/// `peel_modifiers_with_id` MUST collapse the wrapper chain to
/// `T_id` before threading it into `render_struct` as
/// `parent_type_id`; the cast intercept then queries
/// `cast_lookup(T_id, 0)` and hits. Pure-renderer test —
/// complementary to
/// `cast_pipeline_modifier_chain_renderer_peels_to_analyzer_struct_id`
/// which couples the analyzer in. Without this isolated test, a
/// regression in just the renderer's peel would only surface
/// indirectly through the integration test (where it'd be
/// indistinguishable from an analyzer regression).
#[test]
fn cast_intercept_modifier_chain_parent_uses_post_peel_id() {
    // Layout:
    //   id=1: u64
    //   id=2: T { u64 f @ 0 }
    //   id=3: Q { u64 x @ 0 } (cast target)
    //   id=4: const(T) — wraps id=2
    //   id=5: typedef T_alias = const(T) — wraps id=4
    let (strings, n_int, n_t, n_q, n_f, n_x) = cast_strings_for_t_q();
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let mut strings = strings;
    let n_typedef = push(&mut strings, "T_alias");
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
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        CastSynType::Const { type_id: 2 },
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

    // CastMap keyed on the POST-PEEL id (T_id, 0). If the renderer
    // forwarded the typedef_id (or any non-peeled id) as
    // parent_type_id, this lookup would miss and the field would
    // render as Uint — failing the assertion below.
    let mut cast_map: crate::monitor::cast_analysis::CastMap = std::collections::BTreeMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        },
    );

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, 0x55u64.to_le_bytes().to_vec());
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    // Render via the typedef wrapper id; the renderer's peel must
    // produce parent_type_id=T_id (the post-peel struct id) so the
    // cast lookup hits.
    let v = render_value_with_mem(&btf, typedef_id, &outer_bytes, &reader);
    let RenderedValue::Struct {
        type_name,
        ref members,
    } = v
    else {
        panic!("expected Struct render, got {v:?}");
    };
    assert_eq!(
        type_name.as_deref(),
        Some("T"),
        "renderer must collapse modifier wrappers to underlying T name"
    );
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "modifier-chain rendering must reach the cast intercept (peel \
             must produce T_id as parent_type_id); got {:?}. A failure here \
             means peel_modifiers_with_id forwarded the typedef wrapper id \
             instead of the post-peel struct id when calling \
             render_struct.",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR);
    assert!(deref_skipped_reason.is_none());
    let inner = deref.as_deref().expect("chase deref Some");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be Q struct, got {inner:?}");
    };
    assert_eq!(inner_name.as_deref(), Some("Q"));
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x55);
}
