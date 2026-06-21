use super::*;

// ---- Cast intercept (render_cast_pointer) ----------------------
//
// `render_member`'s cast intercept fires when:
//   - the parent BTF id is known (we are inside `render_struct`),
//   - a [`MemReader`] is plumbed,
//   - the member peels to a plain unsigned 8-byte Int (not signed,
//     bool, char, or any other size),
//   - [`MemReader::cast_lookup`] returns `Some(hit)` for
//     (parent_btf_id, member_byte_offset),
//   - the parent_bytes slice covers the full 8-byte u64 field.
//
// On hit, [`render_cast_pointer`] dispatches by [`AddrSpace`]:
// arena reads through `read_arena` after `is_arena_addr`; kernel
// reads through `read_kva` and applies the freed-slab plausibility
// gate (top-byte 0xff on the first qword).
//
// The tests below build minimal synthetic BTF blobs (mirroring
// `cast_analysis::tests::build_btf` but pared down to only the
// kinds the renderer's cast path uses: BTF_KIND_INT and
// BTF_KIND_STRUCT) and parse them via `Btf::from_bytes`. Each test
// supplies a stub `MemReader` that returns the canned `CastHit`
// the scenario exercises and observes the resulting
// `RenderedValue` tree directly. Synthetic BTF + stub reader
// keeps these tests independent of vmlinux availability and pins
// the exact intercept gate without any real kernel BTF noise.

/// Pull the sole member's value out of a rendered single-member
/// struct, transparently peeling the `Truncated` wrapper that
/// `render_struct` adds when `bytes.len() < struct size`.
fn sole_member_value(v: &RenderedValue) -> &RenderedValue {
    match v {
        RenderedValue::Struct { members, .. } => {
            assert_eq!(members.len(), 1, "expected exactly one member");
            &members[0].value
        }
        RenderedValue::Truncated { partial, .. } => sole_member_value(partial),
        other => panic!("expected Struct/Truncated, got {other:?}"),
    }
}

#[test]
fn render_bitfield_unsigned_multibit_at_nonzero_offset_decodes_exact() {
    // struct B { u64 f : 12; } with the field placed at bit offset 4.
    // 8 bytes little-endian = 0x0000_0000_0000_ABC0 -> (val>>4)&0xFFF = 0xABC.
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push(&mut strings, "u64");
    let n_b = push(&mut strings, "B");
    let n_f = push(&mut strings, "f");
    let types = vec![
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::BitfieldStruct {
            name_off: n_b,
            size: 8,
            members: vec![CastSynBitMember {
                name_off: n_f,
                type_id: 1,
                bit_offset: 4,
                bitfield_size: 12,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let bytes = 0xABC0u64.to_le_bytes();
    let v = render_value(&btf, 2, &bytes);
    assert_eq!(
        sole_member_value(&v),
        &RenderedValue::Uint {
            bits: 12,
            value: 0xABC,
        }
    );
}

#[test]
fn render_bitfield_signed_int_base_decodes_negative() {
    // struct B { int f : 4; } at bit offset 0; byte 0x0F -> raw 0xF ->
    // sign_extend(0xF, 4) = -1.
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_int = push(&mut strings, "int");
    let n_b = push(&mut strings, "B");
    let n_f = push(&mut strings, "f");
    let types = vec![
        // encoding=1 sets BTF_INT_SIGNED.
        CastSynType::Int {
            name_off: n_int,
            size: 4,
            encoding: 1,
            offset: 0,
            bits: 32,
        },
        CastSynType::BitfieldStruct {
            name_off: n_b,
            size: 4,
            members: vec![CastSynBitMember {
                name_off: n_f,
                type_id: 1,
                bit_offset: 0,
                bitfield_size: 4,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let v = render_value(&btf, 2, &[0x0F, 0x00, 0x00, 0x00]);
    assert_eq!(
        sole_member_value(&v),
        &RenderedValue::Int { bits: 4, value: -1 }
    );
}

#[test]
fn render_bitfield_signed_enum_base_decodes_negative_int() {
    // struct B { enum E f : 8; } where E is signed (a negative variant).
    // render_bitfield routes signed enum bases through the Int arm, so
    // the result is Int{-1}, NOT Enum.
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_e = push(&mut strings, "E");
    let n_neg = push(&mut strings, "NEG");
    let n_b = push(&mut strings, "B");
    let n_f = push(&mut strings, "f");
    let types = vec![
        // id 1: signed enum E { NEG = -1 } size 4.
        CastSynType::Enum {
            name_off: n_e,
            size: 4,
            signed: true,
            members: vec![(n_neg, 0xFFFF_FFFFu32)],
        },
        CastSynType::BitfieldStruct {
            name_off: n_b,
            size: 4,
            members: vec![CastSynBitMember {
                name_off: n_f,
                type_id: 1,
                bit_offset: 0,
                bitfield_size: 8,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let v = render_value(&btf, 2, &[0xFF, 0x00, 0x00, 0x00]);
    assert_eq!(
        sole_member_value(&v),
        &RenderedValue::Int { bits: 8, value: -1 }
    );
}

#[test]
fn render_sub_byte_signed_enum_resolves_negative_variant_name() {
    // A 1-byte signed enum E { NEG = -1 } rendered directly. `raw` reads
    // 1 byte (0xFF); the member's stored value is the full 0xFFFF_FFFF.
    // Width-truncating the member value to the enum's byte width before
    // compare resolves the variant name — without it the name was
    // silently dropped (the value rendered correctly as -1, but the
    // human-readable "NEG" was lost on sub-4-byte signed enums).
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_e = push(&mut strings, "E");
    let n_neg = push(&mut strings, "NEG");
    let types = vec![CastSynType::Enum {
        name_off: n_e,
        size: 1,
        signed: true,
        members: vec![(n_neg, 0xFFFF_FFFFu32)],
    }];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let v = render_value(&btf, 1, &[0xFF]);
    assert_eq!(v, enum_v(8, -1, Some("NEG"), true));
}

#[test]
fn render_sub_byte_signed_enum64_resolves_negative_variant_name() {
    // BTF_KIND_ENUM64 mirror of the Enum32 sub-byte case: a 4-byte
    // signed enum64 E { NEG = -1 } whose member is stored as the full
    // u64 (u64::MAX for -1). `raw` reads `needed`=4 bytes (0xFFFF_FFFF);
    // the Enum64 width-mask (mod.rs val_mask, needed*8=32 < 64 →
    // (1<<32)-1) truncates the member before compare so "NEG" resolves.
    // Without the mask the member u64::MAX != raw 0xFFFF_FFFF and the
    // variant name was silently dropped on sub-8-byte signed enum64s.
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_e = push(&mut strings, "E64");
    let n_neg = push(&mut strings, "NEG");
    let types = vec![CastSynType::Enum64 {
        name_off: n_e,
        size: 4,
        signed: true,
        members: vec![(n_neg, u64::MAX)],
    }];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let v = render_value(&btf, 1, &[0xFF, 0xFF, 0xFF, 0xFF]);
    assert_eq!(v, enum_v(32, -1, Some("NEG"), true));
}

#[test]
fn render_full_width_signed_enum64_resolves_negative_variant_name() {
    // size=8 enum64: `needed`=8 → needed*8 == 64 ≥ 64, so val_mask is
    // u64::MAX — the branch that avoids `1u64 << 64` UB. member u64::MAX,
    // raw 8×0xFF (u64::MAX) → value -1, (member & u64::MAX) == raw →
    // "NEG" resolves. Pins the needed=8 boundary distinct from the
    // sub-byte path; the only other size=8 enum64 test goes through the
    // bitfield path and asserts the Int value, never the variant name.
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_e = push(&mut strings, "E64");
    let n_neg = push(&mut strings, "NEG");
    let types = vec![CastSynType::Enum64 {
        name_off: n_e,
        size: 8,
        signed: true,
        members: vec![(n_neg, u64::MAX)],
    }];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let v = render_value(&btf, 1, &[0xFF; 8]);
    assert_eq!(v, enum_v(64, -1, Some("NEG"), true));
}

#[test]
fn render_bitfield_signed_enum64_base_decodes_negative_int() {
    // Same as the Enum case but with a BTF_KIND_ENUM64 (kind 19) base.
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_e = push(&mut strings, "E64");
    let n_neg = push(&mut strings, "NEG");
    let n_b = push(&mut strings, "B");
    let n_f = push(&mut strings, "f");
    let types = vec![
        CastSynType::Enum64 {
            name_off: n_e,
            size: 8,
            signed: true,
            members: vec![(n_neg, u64::MAX)],
        },
        CastSynType::BitfieldStruct {
            name_off: n_b,
            size: 8,
            members: vec![CastSynBitMember {
                name_off: n_f,
                type_id: 1,
                bit_offset: 0,
                bitfield_size: 8,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let v = render_value(&btf, 2, &[0xFF, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        sole_member_value(&v),
        &RenderedValue::Int { bits: 8, value: -1 }
    );
}

#[test]
fn render_bitfield_width_over_64_is_unsupported() {
    // width=65 reaches render_bitfield (render_member passes any width>0)
    // and is rejected by the width>64 guard.
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push(&mut strings, "u64");
    let n_b = push(&mut strings, "B");
    let n_f = push(&mut strings, "f");
    let types = vec![
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::BitfieldStruct {
            name_off: n_b,
            size: 16,
            members: vec![CastSynBitMember {
                name_off: n_f,
                type_id: 1,
                bit_offset: 0,
                bitfield_size: 65,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let v = render_value(&btf, 2, &[0u8; 16]);
    match sole_member_value(&v) {
        RenderedValue::Unsupported { reason } => {
            assert!(
                reason.contains("out of range"),
                "unexpected reason: {reason}"
            );
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn render_bitfield_width_zero_is_unsupported_direct() {
    // render_member filters width==0 before calling render_bitfield, so
    // exercise render_bitfield's own width==0 rejection directly.
    // render_bitfield is module-private; the tests submodule reaches it
    // via super::. A 1-type BTF (just a u64) supplies member_type_id.
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push(&mut strings, "u64");
    let types = vec![CastSynType::Int {
        name_off: n_u64,
        size: 8,
        encoding: 0,
        offset: 0,
        bits: 64,
    }];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let v = super::render_bitfield(&btf, 1, &[0u8; 8], 0, 0);
    match v {
        RenderedValue::Unsupported { reason } => {
            assert!(
                reason.contains("out of range"),
                "unexpected reason: {reason}"
            );
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn render_bitfield_straddling_end_of_bytes_is_truncated() {
    // struct B { u64 f : 64; } size 8, but only 1 byte supplied. The
    // member byte_off (0) is < bytes.len() (1) so render_struct does NOT
    // skip it; render_bitfield needs 8 bytes -> Truncated.
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push(&mut strings, "u64");
    let n_b = push(&mut strings, "B");
    let n_f = push(&mut strings, "f");
    let types = vec![
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::BitfieldStruct {
            name_off: n_b,
            size: 8,
            members: vec![CastSynBitMember {
                name_off: n_f,
                type_id: 1,
                bit_offset: 0,
                bitfield_size: 64,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let v = render_value(&btf, 2, &[0xAB]);
    match sole_member_value(&v) {
        RenderedValue::Truncated { needed, had, .. } => {
            assert_eq!((*needed, *had), (8, 1));
        }
        other => panic!("expected Truncated, got {other:?}"),
    }
}


/// Build a BTF blob where T's intercepted member is u32 (size=4)
/// instead of u64. Used to verify the intercept's size==8 gate
/// rejects sub-u64 fields. id=1: u32 (size=4,bits=32),
/// id=2: struct T { u32 f at offset 0; } size=4. T_id=2 — Q
/// (the cast target) is unused since the gate fires before
/// `cast_lookup`, but we still emit a valid u64 + Q so the
/// fixture covers a hit returned for a hypothetical reader that
/// returns Some despite the size mismatch.
fn cast_btf_t_with_u32() -> (Vec<u8>, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u32 = push(&mut strings, "u32");
    let n_u64 = push(&mut strings, "u64");
    let n_t = push(&mut strings, "T");
    let n_q = push(&mut strings, "Q");
    let n_f = push(&mut strings, "f");
    let n_x = push(&mut strings, "x");

    let types = vec![
        // id 1: u32 plain unsigned.
        CastSynType::Int {
            name_off: n_u32,
            size: 4,
            encoding: 0,
            offset: 0,
            bits: 32,
        },
        // id 2: u64 plain unsigned (Q's field type).
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 3: struct T { u32 f at offset 0; } size=4.
        CastSynType::Struct {
            name_off: n_t,
            size: 4,
            members: vec![CastSynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 4: struct Q { u64 x at offset 0; } size=8.
        CastSynType::Struct {
            name_off: n_q,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_x,
                type_id: 2,
                byte_offset: 0,
            }],
        },
    ];
    (cast_build_btf(&types, &strings), 3, 4)
}


/// Arena cast hit on a u64 member: render_cast_pointer chases
/// the value through `read_arena` and surfaces a Ptr whose
/// `deref` carries the rendered target struct. The outer
/// rendered Struct member is `Ptr{ value, deref: Some(...) }` —
/// not `Uint`, which is the no-intercept default.
#[test]
fn cast_intercept_u64_renders_as_ptr_with_chase() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Outer T bytes: u64 at offset 0 = 0x10_0000_1000 (an arena
    // address inside the configured window). Inner Q bytes at
    // that arena address: u64 at offset 0 = 0x42 (a plain
    // counter-shaped value — passes the kernel plausibility
    // gate, though that gate is irrelevant here since the
    // address space is Arena).
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();

    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
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
    let RenderedValue::Struct {
        type_name,
        ref members,
    } = v
    else {
        panic!("expected Struct render, got {v:?}");
    };
    assert_eq!(type_name.as_deref(), Some("T"));
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].name, "f");
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "intercept must produce Ptr (not Uint); got {:?}",
            members[0].value
        );
    };
    assert_eq!(
        value, TARGET_ADDR,
        "Ptr value must be the loaded u64 (arena address)"
    );
    assert!(
        deref_skipped_reason.is_none(),
        "successful chase: no skip reason; got {deref_skipped_reason:?}"
    );
    let inner = deref
        .as_deref()
        .expect("chase succeeded → deref must be Some");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!("deref payload must be the rendered Q Struct, got {inner:?}");
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "inner deref Struct must carry the target's name"
    );
    assert_eq!(inner_members.len(), 1);
    assert_eq!(inner_members[0].name, "x");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!("Q.x must render as Uint, got {:?}", inner_members[0].value);
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x42);
}

/// Null-pointer guard: `value == 0` short-circuits before any
/// `is_arena_addr` / `read_arena` call. Output is
/// `Ptr{ value: 0, deref: None, deref_skipped_reason: None }` —
/// no chase attempted, no skip reason emitted (matching the
/// `Type::Ptr` arm's null handling).
#[test]
fn cast_intercept_null_value_no_crash() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Outer T bytes: u64 at offset 0 = 0. The reader is
    // configured with no arena window and no canned bytes; if
    // the renderer ever called is_arena_addr or read_arena it
    // would short-circuit on `false` / `None`, but the null
    // guard fires first and neither is reached.
    let outer_bytes = 0u64.to_le_bytes().to_vec();
    let reader = CastStubReader {
        hit: Some(CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
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
            "null intercept must still surface as Ptr (matches Type::Ptr arm); got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, 0);
    assert!(deref.is_none(), "null Ptr has no deref");
    assert!(
        deref_skipped_reason.is_none(),
        "null Ptr must NOT carry a skip reason: a chase was never attempted"
    );
}

/// Size gate: a u32 (size=4) member with `cast_lookup` returning
/// `Some(hit)` is NOT intercepted. The gate
/// `int.size() != 8` fires before `cast_lookup` is called and
/// the renderer falls through to the normal Int render, producing
/// `RenderedValue::Uint{ bits: 32, .. }`. This is the structural
/// invariant the cast intercept relies on: BPF stores recovered
/// typed pointers in u64 slots only.
#[test]
fn cast_intercept_non_u64_field_not_intercepted() {
    let (blob, t_id, q_id) = cast_btf_t_with_u32();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    let outer_bytes = 0xCAFEu32.to_le_bytes().to_vec();
    // Reader returns Some(hit) for any (parent, offset) — the
    // gate must reject before this is consulted.
    let reader = CastStubReader {
        hit: Some(CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Arena,
        }),
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Uint { bits, value } = members[0].value else {
        panic!(
            "u32 field with size==4 must render as Uint, NOT Ptr; got {:?}",
            members[0].value
        );
    };
    assert_eq!(bits, 32, "u32 surfaces as 32-bit Uint");
    assert_eq!(value, 0xCAFE);
}

/// `MemReader::cast_lookup` returns `None` for every
/// `(parent, off)`; the cast intercept short-circuits on the
/// `cast_lookup` gate and the renderer takes the pre-existing
/// path. A u64 member surfaces as `Uint{ bits: 64, value }` —
/// same as before the cast intercept landed. A regression that
/// fired the intercept for None-returning readers would surface
/// here as a Ptr render, breaking every reader that doesn't
/// override `cast_lookup`.
#[test]
fn cast_intercept_no_hit_renders_uint() {
    let (blob, t_id, _q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    let outer_bytes = 0x12345678u64.to_le_bytes().to_vec();
    // CastStubReader::default() leaves `hit` as None — its
    // `cast_lookup` returns None for every (parent, offset). This
    // mirrors the "no override" default trait method behavior.
    let reader = CastStubReader::default();

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct render, got {v:?}");
    };
    let RenderedValue::Uint { bits, value } = members[0].value else {
        panic!(
            "no cast_lookup hit must yield plain Uint, got {:?}",
            members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x12345678);
}

/// Self-cycle through cast-recovered pointers: T at the outer
/// bytes contains a u64 whose cast hit chases another T at the
/// SAME arena address. `render_cast_pointer` inserts the value
/// into the visited set, recurses into the inner T render, and
/// the inner u64's cast intercept hits the visited check and
/// surfaces `deref_skipped_reason` containing "cycle".
///
/// Without the visited check, the chase would recurse until
/// `MAX_RENDER_DEPTH` (32), producing a deep nest of Ptr -> Ptr
/// -> ... in the failure dump.
#[test]
fn cast_chase_cycle_detection() {
    // Use T with a self-cycle: T contains a u64 at offset 0 that
    // points to a T-shaped instance whose own u64 at offset 0
    // points back to itself.
    let (blob, t_id, _q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const SELF_ADDR: u64 = 0x10_0000_1000;
    // Outer T bytes: u64 at offset 0 = SELF_ADDR.
    let outer_bytes = SELF_ADDR.to_le_bytes().to_vec();
    // Bytes at SELF_ADDR: u64 at offset 0 = SELF_ADDR (loop).
    let self_bytes = SELF_ADDR.to_le_bytes().to_vec();

    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(SELF_ADDR, self_bytes);
    let reader = CastStubReader {
        // Target T (id=2) so the inner render recurses into the
        // same shape and the inner u64 also gets the cast
        // intercept (cast_lookup returns the same hit
        // regardless of parent id).
        hit: Some(CastHit {
            alloc_size: None,
            target_type_id: t_id,
            addr_space: AddrSpace::Arena,
        }),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
    // (outer_bytes is little-endian per the renderer's wire format
    // assumption; SELF_ADDR.to_le_bytes() above produces those.)
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected outer Struct render, got {v:?}");
    };
    // Outer Ptr: chase succeeded once (visited was empty), so
    // deref is Some(inner T struct), no skip reason.
    let RenderedValue::Ptr {
        value: outer_value,
        deref: ref outer_deref,
        deref_skipped_reason: ref outer_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "outer chase must surface as Ptr, got {:?}",
            members[0].value
        );
    };
    assert_eq!(outer_value, SELF_ADDR);
    assert!(
        outer_reason.is_none(),
        "outer chase succeeded; no skip reason expected, got {outer_reason:?}"
    );
    let inner = outer_deref.as_deref().expect("outer chase deref Some");
    // Inner is the rendered T at SELF_ADDR. Its `f` member must
    // surface the cycle (visited contains SELF_ADDR by the time
    // the inner render reaches it).
    let RenderedValue::Struct {
        members: ref inner_members,
        ..
    } = *inner
    else {
        panic!("inner deref must be a Struct, got {inner:?}");
    };
    let RenderedValue::Ptr {
        value: inner_value,
        deref: ref inner_deref,
        deref_skipped_reason: ref inner_reason,
        ..
    } = inner_members[0].value
    else {
        panic!(
            "inner u64 cast intercept must surface as Ptr, got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(inner_value, SELF_ADDR);
    assert!(
        inner_deref.is_none(),
        "cycle detection must NOT recurse into the deref payload"
    );
    let reason = inner_reason
        .as_deref()
        .expect("cycle detection must populate deref_skipped_reason");
    assert!(
        reason.contains("cycle"),
        "skip reason must mention cycle, got: {reason}"
    );
}

/// Kernel cast hit whose `read_kva` returns 8 bytes whose first
/// qword has top byte 0xff — the freed-slab freelist-pointer
/// signature on x86_64 / aarch64. `render_cast_pointer`'s
/// plausibility gate rejects the read and surfaces
/// `deref_skipped_reason` mentioning "plausibility". The rendered
/// Ptr's `deref` is `None` (chase attempted but rejected).
#[test]
fn cast_chase_kernel_plausibility_rejects_freed_slab() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    const KVA: u64 = 0xffff_8000_dead_beef;
    let outer_bytes = KVA.to_le_bytes().to_vec();
    // Bytes at KVA: first qword has top byte 0xff (matches the
    // SLAB_FREELIST_HARDENED-defeating heuristic that the
    // plausibility gate enforces). Use a value whose top byte is
    // 0xff but isn't structurally a real kernel address; the gate
    // doesn't care about the lower bits, only the top byte.
    let stale_bytes: Vec<u8> = 0xff00_0000_0000_0001u64.to_le_bytes().to_vec();

    let mut kva_bytes = std::collections::HashMap::new();
    kva_bytes.insert(KVA, stale_bytes);
    let reader = CastStubReader {
        hit: Some(CastHit {
            alloc_size: None,
            target_type_id: q_id,
            addr_space: AddrSpace::Kernel,
        }),
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
            "kernel cast intercept must surface as Ptr, got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, KVA);
    assert!(
        deref.is_none(),
        "plausibility-rejected chase must NOT carry a deref payload"
    );
    let reason = deref_skipped_reason
        .as_deref()
        .expect("plausibility rejection must populate skip reason");
    assert!(
        reason.contains("plausibility"),
        "skip reason must mention plausibility, got: {reason}"
    );
}

/// Runtime dispatch wins over the analyzer's address-space hint:
/// a `CastHit` whose `addr_space` is `Kernel` but whose value
/// falls inside the arena window must still chase via the arena
/// reader (not `read_kva`), and the `cast_annotation` must
/// reflect the path actually taken (`"cast→arena"`).
///
/// Per [`render_cast_pointer`]'s runtime address-space dispatch:
/// "Address-space dispatch is RUNTIME-driven: `is_arena_addr` is
/// consulted on the actual pointer value to decide whether to
/// chase via `read_arena` (in-window) or `read_kva` (out-of-window).
/// The `CastHit::addr_space` tag from the analyzer is treated as
/// a hint only — runtime evidence from the pointer value is
/// authoritative." This test pins that contract: a Kernel-hinted
/// hit with an arena-window value goes through the arena path.
#[test]
fn cast_intercept_kernel_hint_arena_value_dispatches_to_arena_reader() {
    let (blob, t_id, q_id) = cast_btf_t_and_q();
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // Configure: arena window covers [ARENA_LO, ARENA_HI); the
    // pointer value (TARGET_ADDR) falls inside that window even
    // though the analyzer's hint says Kernel. The arena reader
    // has bytes for TARGET_ADDR; the kva reader is intentionally
    // empty so a wrongly-routed kernel chase would surface as a
    // skip reason rather than a successful chase.
    const ARENA_LO: u64 = 0x10_0000_0000;
    const ARENA_HI: u64 = 0x10_0001_0000;
    const TARGET_ADDR: u64 = 0x10_0000_1000;
    let outer_bytes = TARGET_ADDR.to_le_bytes().to_vec();
    let inner_bytes = 0x42u64.to_le_bytes().to_vec();

    let mut arena_bytes = std::collections::HashMap::new();
    arena_bytes.insert(TARGET_ADDR, inner_bytes);
    // CastMap mode (key-specific) so the inner Q.x render
    // does NOT re-trigger the cast intercept on a (Q, 0) lookup —
    // only the outer (T, 0) key has an entry. The hit-on-every-
    // query mode would chase Q.x through the kernel path because
    // 0x42 is not in the arena window, surfacing a spurious
    // failure that has nothing to do with the runtime-vs-hint
    // dispatch this test pins.
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (t_id, 0),
        CastHit {
            alloc_size: None,
            target_type_id: q_id,
            // Kernel hint — but value is an arena address. Runtime
            // detection is_arena_addr(value) returns true → arena
            // reader fires regardless of the hint.
            addr_space: AddrSpace::Kernel,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        arena_window: Some((ARENA_LO, ARENA_HI)),
        arena_bytes_at: arena_bytes,
        // kva_bytes_at is intentionally empty — read_kva would
        // return None for any address. If runtime dispatch ever
        // routed a Kernel-hinted but in-arena value through the
        // kernel path, this test would surface a skip reason.
        ..Default::default()
    };

    let v = render_value_with_mem(&btf, t_id, &outer_bytes, &reader);
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
        panic!(
            "Kernel-hint + arena-value cast must surface as Ptr, got {:?}",
            members[0].value
        );
    };
    assert_eq!(value, TARGET_ADDR, "Ptr value is the loaded u64");
    assert!(
        deref_skipped_reason.is_none(),
        "arena dispatch chose the arena reader → no skip reason; got {deref_skipped_reason:?}",
    );
    let inner = deref
        .as_deref()
        .expect("arena reader returned Some bytes → deref payload populated");
    let RenderedValue::Struct {
        type_name: ref inner_name,
        members: ref inner_members,
    } = *inner
    else {
        panic!(
            "deref payload must be the rendered Q struct (proves arena chase, \
             not kernel chase, did the read), got {inner:?}",
        );
    };
    assert_eq!(
        inner_name.as_deref(),
        Some("Q"),
        "inner deref carries Q's name → render_value_inner(target_type_id) succeeded",
    );
    assert_eq!(inner_members.len(), 1, "Q has one u64 member");
    let RenderedValue::Uint { bits, value } = inner_members[0].value else {
        panic!(
            "Q.x must render as Uint (was rendered through arena reader bytes), got {:?}",
            inner_members[0].value
        );
    };
    assert_eq!(bits, 64);
    assert_eq!(value, 0x42, "arena reader returned 0x42 at TARGET_ADDR");
    // The annotation must reflect the path actually taken (arena),
    // not the analyzer's hint (Kernel). A regression that emitted
    // "cast→kernel" here would mean the renderer fell back to the
    // kernel arm but somehow succeeded, which is impossible without
    // canned kva bytes — but the annotation pins the dispatch
    // outcome explicitly so the contract isn't merely inferred.
    assert_eq!(
        cast_annotation.as_deref(),
        Some("cast→arena"),
        "runtime dispatch chose arena → annotation is cast→arena, NOT cast→kernel; \
         got {cast_annotation:?}",
    );
}

