use super::*;

// ---- bpf_cpumask __kptr deref: positive-path render -------------
//
// The Type::Ptr arm chases a `struct bpf_cpumask`/`cpumask` kptr
// member to a CpuList by reading the bitmap via MemReader::read_kva
// (mod.rs:2309-2434). The existing tests only pin the no-op/skip
// paths (try_render_cpumask_bits byte-decode, and
// arena_chase_bridge_address_outside_window_is_no_op which asserts
// deref.is_none()). These pin the SUCCESS path end-to-end: a
// `cpumask` pointee + a mock read_kva returning a real bitmap must
// render `Ptr{ value, deref: Some(CpuList{cpus}) }`.

/// Build `outer { mask: *cpumask }` where `struct cpumask` has a u64
/// `bits` member (size > 0 so the kptr branch's incomplete-type gate
/// passes). `pointee_name` selects "cpumask" or "bpf_cpumask" — both
/// must match the name predicate. Returns `(btf, outer_type_id)`.
fn cpumask_kptr_outer_btf(pointee_name: &str) -> (Btf, u32) {
    let mut strings: Vec<u8> = vec![0];
    let mut push = |name: &str| -> u32 {
        let off = strings.len() as u32;
        strings.extend_from_slice(name.as_bytes());
        strings.push(0);
        off
    };
    let n_u64 = push("u64");
    let n_pointee = push(pointee_name);
    let n_bits = push("bits");
    let n_outer = push("outer");
    let n_mask = push("mask");
    let types = vec![
        // id 1: u64 (the cpumask's bits member element)
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct cpumask { u64 bits; } — size 8 > 0
        CastSynType::Struct {
            name_off: n_pointee,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_bits,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: *cpumask (the kptr)
        CastSynType::Ptr { type_id: 2 },
        // id 4: struct outer { *cpumask mask; }
        CastSynType::Struct {
            name_off: n_outer,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_mask,
                type_id: 3,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic cpumask BTF parses");
    (btf, 4)
}

/// A 1024-byte cpumask read (CPUMASK_READ_CAP) with the given low
/// 64-bit word; the rest of the slab read is zero.
fn cpumask_read_bytes(word0: u64) -> Vec<u8> {
    let mut b = vec![0u8; 1024];
    b[..8].copy_from_slice(&word0.to_le_bytes());
    b
}

#[test]
fn cpumask_kptr_member_chases_to_cpu_list() {
    for name in ["cpumask", "bpf_cpumask"] {
        let (btf, outer_id) = cpumask_kptr_outer_btf(name);
        let val: u64 = 0xFFFF_8000_0010_0000; // kernel kptr VA the slot holds
        let outer_bytes = val.to_le_bytes().to_vec();
        // bits 0, 1, 3 set → "0-1,3" after range collapse.
        let mut kva = std::collections::HashMap::new();
        kva.insert(val, cpumask_read_bytes(0b0000_1011));
        let reader = CastStubReader {
            kva_bytes_at: kva,
            ..Default::default()
        };

        let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
        let RenderedValue::Struct { ref members, .. } = v else {
            panic!("expected Struct render for {name}, got {v:?}");
        };
        let RenderedValue::Ptr {
            value,
            ref deref,
            ref deref_skipped_reason,
            ref cast_annotation,
            ..
        } = members[0].value
        else {
            panic!(
                "mask field must render as Ptr for {name}; got {:?}",
                members[0].value
            );
        };
        assert_eq!(value, val, "raw pointer retained alongside deref ({name})");
        assert!(
            deref_skipped_reason.is_none(),
            "valid cpumask must not skip ({name}): {deref_skipped_reason:?}"
        );
        assert!(
            cast_annotation.is_none(),
            "kptr branch leaves cast_annotation None ({name})"
        );
        match deref.as_deref() {
            Some(RenderedValue::CpuList { cpus }) => {
                assert_eq!(cpus, "0-1,3", "decoded cpu set ({name})")
            }
            other => panic!("expected deref Some(CpuList) for {name}, got {other:?}"),
        }
    }
}

#[test]
fn cpumask_kptr_null_pointer_no_chase() {
    let (btf, outer_id) = cpumask_kptr_outer_btf("bpf_cpumask");
    let outer_bytes = 0u64.to_le_bytes().to_vec(); // NULL kptr
    // read_kva should never be consulted; empty reader.
    let reader = CastStubReader::default();
    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct, got {v:?}");
    };
    let RenderedValue::Ptr {
        value,
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!("mask field must render as Ptr; got {:?}", members[0].value);
    };
    assert_eq!(value, 0);
    assert!(deref.is_none(), "NULL kptr must not chase");
    assert!(
        deref_skipped_reason.is_none(),
        "NULL kptr is a clean skip with no reason, got {deref_skipped_reason:?}"
    );
}

#[test]
fn cpumask_kptr_plausibility_gate_rejects_freed_slab_pattern() {
    let (btf, outer_id) = cpumask_kptr_outer_btf("cpumask");
    let val: u64 = 0xFFFF_8000_0010_0000;
    let outer_bytes = val.to_le_bytes().to_vec();
    // word0 top byte == 0xff: the freed-slab freelist-pointer pattern
    // the plausibility gate (mod.rs:2372) rejects.
    let mut kva = std::collections::HashMap::new();
    kva.insert(val, cpumask_read_bytes(0xFF00_0000_0000_0000));
    let reader = CastStubReader {
        kva_bytes_at: kva,
        ..Default::default()
    };
    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!("mask field must render as Ptr; got {:?}", members[0].value);
    };
    assert!(
        deref.is_none(),
        "freed-slab pattern must not decode to a CpuList"
    );
    assert!(
        deref_skipped_reason
            .as_deref()
            .is_some_and(|r| r.contains("plausibility")),
        "skip reason must cite the plausibility gate, got {deref_skipped_reason:?}"
    );
}

// ---- llc_cpumask inline-bitmap render -------------------------------
//
// `struct llc_cpumask { unsigned long bits[NR]; }` (scx_mitosis) is an
// inline NON-kptr bitmap. Its BTF name must be in render_struct's
// embedded cpumask-detection list so the bytes render as a CpuList
// rather than a raw u64 array. A synthetic BTF avoids a vmlinux
// dependency (the name is scheduler-specific, absent from vmlinux).

#[test]
fn llc_cpumask_struct_renders_as_cpu_list() {
    let mut strings: Vec<u8> = vec![0];
    let mut push = |name: &str| -> u32 {
        let off = strings.len() as u32;
        strings.extend_from_slice(name.as_bytes());
        strings.push(0);
        off
    };
    let n_u64 = push("u64");
    let n_llc = push("llc_cpumask");
    let n_bits = push("bits");
    // struct llc_cpumask { u64 bits; } — the renderer treats the whole
    // struct's bytes as the bitmap, so a single-word stand-in exercises
    // the name-detection arm (real scx_mitosis declares bits[128]).
    let types = vec![
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        CastSynType::Struct {
            name_off: n_llc,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_bits,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic llc_cpumask BTF parses");
    // bits 0, 2, 5 set.
    let bytes = 0b10_0101u64.to_le_bytes();
    match render_value(&btf, 2, &bytes) {
        RenderedValue::CpuList { cpus } => assert_eq!(cpus, "0,2,5"),
        other => panic!("expected CpuList for llc_cpumask, got {other:?}"),
    }
}

/// Plausibility gate must NOT reject a fully-online <=64-CPU mask: an
/// all-ones first word (0xFFFF..FFFF) is a valid mask, not a freed-slab
/// freelist pointer (a real next pointer carries address bits and is
/// never all-ones / non-canonical). Regression for the gate
/// false-positive — on the pre-fix top-byte-0xff check this read would
/// skip with a "plausibility" reason instead of decoding cpus 0-63.
#[test]
fn cpumask_kptr_all_ones_word_decodes_not_rejected() {
    let (btf, outer_id) = cpumask_kptr_outer_btf("bpf_cpumask");
    let val: u64 = 0xFFFF_8000_0010_0000;
    let outer_bytes = val.to_le_bytes().to_vec();
    // word0 all ones (CPUs 0-63 online), higher words zero.
    let mut kva = std::collections::HashMap::new();
    kva.insert(val, cpumask_read_bytes(u64::MAX));
    let reader = CastStubReader {
        kva_bytes_at: kva,
        ..Default::default()
    };
    let v = render_value_with_mem(&btf, outer_id, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!("mask field must render as Ptr; got {:?}", members[0].value);
    };
    match deref.as_deref() {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "0-63"),
        other => panic!(
            "all-ones mask must decode to cpus 0-63, got {other:?} \
             (skip reason {deref_skipped_reason:?})"
        ),
    }
}

/// Realistic kptr shape: `struct bpf_cpumask __kptr *` is a
/// btf_type_tag("kptr") wrapping a Ptr, NOT a bare Ptr (which the
/// other cpumask_kptr tests use). The render path must peel the
/// TypeTag to reach the Ptr cpumask arm. This is the shape a datasec
/// global like scx_mitosis's `all_cpumask` carries; render_datasec
/// routes each variable through the same render_value_inner exercised
/// here (after try_cast_intercept returns None for a non-u64 Ptr
/// type), so this pins the datasec-kptr value-decode link.
#[test]
fn cpumask_kptr_through_kptr_typetag_chases_to_cpu_list() {
    let mut strings: Vec<u8> = vec![0];
    let mut push = |name: &str| -> u32 {
        let off = strings.len() as u32;
        strings.extend_from_slice(name.as_bytes());
        strings.push(0);
        off
    };
    let n_u64 = push("u64");
    let n_cpumask = push("bpf_cpumask");
    let n_bits = push("bits");
    let n_kptr = push("kptr");
    let n_outer = push("outer");
    let n_mask = push("mask");
    let types = vec![
        // id 1: u64
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct bpf_cpumask { u64 bits; }
        CastSynType::Struct {
            name_off: n_cpumask,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_bits,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: *bpf_cpumask
        CastSynType::Ptr { type_id: 2 },
        // id 4: __kptr tag wrapping the pointer (the realistic shape)
        CastSynType::TypeTag {
            name_off: n_kptr,
            type_id: 3,
        },
        // id 5: struct outer { bpf_cpumask __kptr *mask; }
        CastSynType::Struct {
            name_off: n_outer,
            size: 8,
            members: vec![CastSynMember {
                name_off: n_mask,
                type_id: 4,
                byte_offset: 0,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic kptr-typetag BTF parses");
    let val: u64 = 0xFFFF_8000_0010_0000;
    let outer_bytes = val.to_le_bytes().to_vec();
    let mut kva = std::collections::HashMap::new();
    kva.insert(val, cpumask_read_bytes(0b0000_1011)); // bits 0,1,3
    let reader = CastStubReader {
        kva_bytes_at: kva,
        ..Default::default()
    };
    let v = render_value_with_mem(&btf, 5, &outer_bytes, &reader);
    let RenderedValue::Struct { ref members, .. } = v else {
        panic!("expected Struct, got {v:?}");
    };
    let RenderedValue::Ptr {
        ref deref,
        ref deref_skipped_reason,
        ..
    } = members[0].value
    else {
        panic!(
            "kptr-tagged mask field must render as Ptr; got {:?}",
            members[0].value
        );
    };
    match deref.as_deref() {
        Some(RenderedValue::CpuList { cpus }) => assert_eq!(cpus, "0-1,3"),
        other => panic!(
            "kptr TypeTag must peel to the Ptr cpumask chase; got {other:?} \
             (skip reason {deref_skipped_reason:?})"
        ),
    }
}

/// `RenderedValue::as_bool` coerces a `Ptr` to a non-null test, in
/// lockstep with `as_u64` (which treats a pointer as its numeric
/// address). Before the `Ptr` arm landed, a pointer fell through to
/// `None` here while the scalar `SnapshotField::as_bool` accepted it —
/// so `field.as_bool()` and `field.as_bool_array()` (which routes each
/// element through THIS method) disagreed on a pointer. Pin both the
/// non-null and null cases.
#[test]
fn rendered_value_as_bool_coerces_ptr_as_non_null() {
    let non_null = RenderedValue::Ptr {
        value: 0xffff_8000_0000_0000,
        deref: None,
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    let null = RenderedValue::Ptr {
        value: 0,
        deref: None,
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    assert_eq!(non_null.as_bool(), Some(true), "non-null pointer is true");
    assert_eq!(null.as_bool(), Some(false), "null pointer is false");
    // Agreement with as_u64: both surface the pointer's numeric value.
    assert_eq!(non_null.as_u64(), Some(0xffff_8000_0000_0000));
    assert_eq!(null.as_u64(), Some(0));
}

/// `as_bool_array` (which calls `as_bool` per element) must accept an
/// array of pointers now that the per-element coercion has a `Ptr`
/// arm — previously the first pointer element collapsed the whole
/// array to `None`, the array-path half of the scalar-vs-array
/// divergence.
#[test]
fn rendered_value_as_bool_array_accepts_pointer_elements() {
    let arr = RenderedValue::Array {
        len: 3,
        elements: vec![
            RenderedValue::Ptr {
                value: 0x1000,
                deref: None,
                deref_skipped_reason: None,
                cast_annotation: None,
            },
            RenderedValue::Ptr {
                value: 0,
                deref: None,
                deref_skipped_reason: None,
                cast_annotation: None,
            },
            RenderedValue::Ptr {
                value: 0xdead_beef,
                deref: None,
                deref_skipped_reason: None,
                cast_annotation: None,
            },
        ],
    };
    assert_eq!(
        arr.as_bool_array(),
        Some(vec![true, false, true]),
        "an array of pointers coerces to a per-element non-null mask",
    );
}
