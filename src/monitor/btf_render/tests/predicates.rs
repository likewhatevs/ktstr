use super::*;

// ---- is_text_byte -----------------------------------------------
//
// Module-private helper; pinned via direct call to defend the
// string-detection contract: NUL terminator, \n, and printable
// ASCII (0x20..=0x7e) are "text". \t and \r are explicitly NOT
// accepted (binary arrays starting with those bytes were
// misclassified as strings). Non-ASCII (>= 0x80) and other ASCII
// control chars are not text.

#[test]
fn is_text_byte_accepts_nul() {
    assert!(is_text_byte(0x00), "NUL is the C string terminator");
}

#[test]
fn is_text_byte_accepts_newline() {
    assert!(is_text_byte(b'\n'));
}

#[test]
fn is_text_byte_rejects_tab_and_cr() {
    // \t and \r in BPF data are almost always binary; accepting
    // them produced false-positive string classification.
    assert!(!is_text_byte(b'\t'));
    assert!(!is_text_byte(b'\r'));
}

#[test]
fn is_text_byte_accepts_printable_ascii() {
    // Boundaries of the printable range.
    assert!(is_text_byte(0x20)); // space
    assert!(is_text_byte(b'A'));
    assert!(is_text_byte(0x7e)); // ~
}

#[test]
fn is_text_byte_rejects_other_control_chars() {
    // Non-newline control chars.
    assert!(!is_text_byte(0x01));
    assert!(!is_text_byte(0x07)); // BEL
    assert!(!is_text_byte(0x1f));
}

#[test]
fn is_text_byte_rejects_high_bit_bytes() {
    assert!(!is_text_byte(0x7f)); // DEL is just past printable
    assert!(!is_text_byte(0x80));
    assert!(!is_text_byte(0xff));
}

// ---- is_string_value --------------------------------------------
//
// Detects whether a RenderedValue::Array represents a printable
// C-string-like byte array. Used by the Display path to suppress
// bpf_printk format strings nested inside Structs.

#[test]
fn is_string_value_accepts_8bit_int_array() {
    let v = RenderedValue::Array {
        len: 4,
        elements: vec![
            RenderedValue::Int {
                bits: 8,
                value: b'h' as i64,
            },
            RenderedValue::Int {
                bits: 8,
                value: b'i' as i64,
            },
            RenderedValue::Int {
                bits: 8,
                value: b'\n' as i64,
            },
            RenderedValue::Int { bits: 8, value: 0 },
        ],
    };
    assert!(is_string_value(&v));
}

#[test]
fn is_string_value_accepts_8bit_uint_array() {
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Uint {
                bits: 8,
                value: b'a' as u64,
            },
            RenderedValue::Uint {
                bits: 8,
                value: b'b' as u64,
            },
        ],
    };
    assert!(is_string_value(&v));
}

#[test]
fn is_string_value_accepts_char_array() {
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Char { value: b'X' },
            RenderedValue::Char { value: 0 },
        ],
    };
    assert!(is_string_value(&v));
}

#[test]
fn is_string_value_rejects_too_short_array() {
    // Single-element arrays don't qualify (length floor of 2).
    let v = RenderedValue::Array {
        len: 1,
        elements: vec![RenderedValue::Char { value: b'X' }],
    };
    assert!(!is_string_value(&v));
}

#[test]
fn is_string_value_rejects_non_text_byte() {
    // 0x80 is non-text → array doesn't qualify as string.
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Uint {
                bits: 8,
                value: b'a' as u64,
            },
            RenderedValue::Uint {
                bits: 8,
                value: 0x80,
            },
        ],
    };
    assert!(!is_string_value(&v));
}

#[test]
fn is_string_value_rejects_wider_int() {
    // u32 elements aren't bytes — even with text-looking values.
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Uint {
                bits: 32,
                value: b'a' as u64,
            },
            RenderedValue::Uint {
                bits: 32,
                value: b'b' as u64,
            },
        ],
    };
    assert!(!is_string_value(&v));
}

#[test]
fn is_string_value_rejects_non_array() {
    // Only Array is a candidate; everything else is not.
    assert!(!is_string_value(&RenderedValue::Bytes { hex: "00".into() }));
    assert!(!is_string_value(&RenderedValue::Uint {
        bits: 8,
        value: b'a' as u64
    }));
    assert!(!is_string_value(&RenderedValue::Struct {
        type_name: None,
        members: vec![],
    }));
}

// ---- is_zero ----------------------------------------------------
//
// Module-public: callers (dump/display.rs Display paths) suppress
// zeroed scalars in struct/array rendering. Pin every variant's
// zero-detection.

#[test]
fn is_zero_int_uint_bool_char() {
    assert!(is_zero(&RenderedValue::Int { bits: 32, value: 0 }));
    assert!(!is_zero(&RenderedValue::Int {
        bits: 32,
        value: -1
    }));
    assert!(is_zero(&RenderedValue::Uint { bits: 64, value: 0 }));
    assert!(!is_zero(&RenderedValue::Uint { bits: 64, value: 1 }));
    assert!(is_zero(&RenderedValue::Bool { value: false }));
    assert!(!is_zero(&RenderedValue::Bool { value: true }));
    assert!(is_zero(&RenderedValue::Char { value: 0 }));
    assert!(!is_zero(&RenderedValue::Char { value: b'a' }));
}

#[test]
fn is_zero_float() {
    assert!(is_zero(&RenderedValue::Float {
        bits: 64,
        value: 0.0
    }));
    assert!(!is_zero(&RenderedValue::Float {
        bits: 64,
        value: 1.0
    }));
    // -0.0 is bit-distinct from 0.0 but is_zero compares with ==,
    // so both register as zero per IEEE-754.
    assert!(is_zero(&RenderedValue::Float {
        bits: 64,
        value: -0.0
    }));
}

#[test]
fn is_zero_enum() {
    assert!(is_zero(&enum_v(32, 0, None, false)));
    assert!(!is_zero(&enum_v(32, 1, Some("RUNNING"), false)));
}

#[test]
fn is_zero_cpulist_empty_vs_populated() {
    assert!(
        is_zero(&RenderedValue::CpuList {
            cpus: String::new()
        }),
        "empty cpu list is zero"
    );
    assert!(
        !is_zero(&RenderedValue::CpuList { cpus: "0-7".into() }),
        "populated cpu list is non-zero"
    );
}

#[test]
fn is_zero_ptr() {
    assert!(is_zero(&RenderedValue::Ptr {
        value: 0,
        deref: None,
        deref_skipped_reason: None,
        cast_annotation: None,
    }));
    assert!(!is_zero(&RenderedValue::Ptr {
        value: 0xffff_8000_dead_beef,
        deref: None,
        deref_skipped_reason: None,
        cast_annotation: None,
    }));
    // A pointer with a deref but value=0 is still zero per the
    // is_zero contract — only `value` matters.
    assert!(is_zero(&RenderedValue::Ptr {
        value: 0,
        deref: Some(Box::new(RenderedValue::Uint { bits: 32, value: 5 })),
        deref_skipped_reason: None,
        cast_annotation: None,
    }));
}

#[test]
fn is_zero_compound_always_false() {
    // Per the impl, Struct / Array / Bytes / Truncated /
    // Unsupported short-circuit to false. Pin that to lock in the
    // "compounds are always rendered" decision.
    assert!(!is_zero(&RenderedValue::Struct {
        type_name: None,
        members: vec![],
    }));
    assert!(!is_zero(&RenderedValue::Array {
        len: 0,
        elements: vec![],
    }));
    assert!(!is_zero(&RenderedValue::Bytes { hex: "".into() }));
    assert!(!is_zero(&RenderedValue::Unsupported { reason: "x".into() }));
}

// ---- is_inline_scalar -------------------------------------------
//
// Determines whether an array element renders inline (single
// line) vs block-style. Scalars + Bytes + Unsupported are inline;
// composite types (Struct, Array, CpuList, Truncated) are block.

#[test]
fn is_inline_scalar_accepts_scalars() {
    assert!(is_inline_scalar(&RenderedValue::Int { bits: 32, value: 0 }));
    assert!(is_inline_scalar(&RenderedValue::Uint {
        bits: 64,
        value: 1
    }));
    assert!(is_inline_scalar(&RenderedValue::Bool { value: false }));
    assert!(is_inline_scalar(&RenderedValue::Char { value: b'x' }));
    assert!(is_inline_scalar(&RenderedValue::Float {
        bits: 64,
        value: 0.0
    }));
    assert!(is_inline_scalar(&enum_v(32, 0, None, false)));
    assert!(is_inline_scalar(&RenderedValue::Ptr {
        value: 0,
        deref: None,
        deref_skipped_reason: None,
        cast_annotation: None,
    }));
    assert!(is_inline_scalar(&RenderedValue::Bytes { hex: "00".into() }));
    assert!(is_inline_scalar(&RenderedValue::Unsupported {
        reason: "void".into(),
    }));
}

#[test]
fn is_inline_scalar_rejects_composites() {
    assert!(!is_inline_scalar(&RenderedValue::Struct {
        type_name: None,
        members: vec![],
    }));
    assert!(!is_inline_scalar(&RenderedValue::Array {
        len: 0,
        elements: vec![],
    }));
    assert!(!is_inline_scalar(&RenderedValue::CpuList {
        cpus: "0".into(),
    }));
    assert!(!is_inline_scalar(&RenderedValue::Truncated {
        needed: 4,
        had: 0,
        partial: Box::new(RenderedValue::Bytes { hex: "".into() }),
    }));
    assert!(!is_inline_scalar(&RenderedValue::Ptr {
        value: 0x1000,
        deref: Some(Box::new(RenderedValue::Struct {
            type_name: Some("scx_cgroup_llc_ctx".into()),
            members: vec![],
        })),
        deref_skipped_reason: None,
        cast_annotation: None,
    }));
}

// ---- anonymous-only struct rendering ----------------------------

#[test]
fn struct_with_only_anonymous_members_renders_inner_fields() {
    // After flatten: anonymous struct members are promoted to the parent.
    // render_struct produces this flattened form; verify Display handles it.
    let v = RenderedValue::Struct {
        type_name: Some("scx_cgroup_ctx".into()),
        members: vec![
            RenderedMember {
                name: "id".into(),
                value: RenderedValue::Uint {
                    bits: 64,
                    value: 49,
                },
            },
            RenderedMember {
                name: "quota".into(),
                value: RenderedValue::Uint {
                    bits: 64,
                    value: 5000000,
                },
            },
            RenderedMember {
                name: "is_throttled".into(),
                value: RenderedValue::Bool { value: true },
            },
            RenderedMember {
                name: "nr_throttled".into(),
                value: RenderedValue::Uint {
                    bits: 32,
                    value: 100,
                },
            },
        ],
    };
    let rendered = format!("{v}");
    assert!(
        !rendered.ends_with("{}"),
        "flattened struct must not render as empty: {rendered}"
    );
    assert!(
        rendered.contains("id"),
        "field 'id' must be visible: {rendered}"
    );
    assert!(
        rendered.contains("49"),
        "field value must be visible: {rendered}"
    );
    assert!(
        rendered.contains("is_throttled"),
        "field 'is_throttled' must be visible: {rendered}"
    );
}

#[test]
fn struct_with_nested_anonymous_members_flattens_on_display() {
    // Legacy/JSON form: anonymous members still nested. Display should
    // flatten them for rendering.
    let v = RenderedValue::Struct {
        type_name: Some("scx_cgroup_ctx".into()),
        members: vec![
            RenderedMember {
                name: String::new(),
                value: RenderedValue::Struct {
                    type_name: None,
                    members: vec![
                        RenderedMember {
                            name: "id".into(),
                            value: RenderedValue::Uint {
                                bits: 64,
                                value: 49,
                            },
                        },
                        RenderedMember {
                            name: "quota".into(),
                            value: RenderedValue::Uint {
                                bits: 64,
                                value: 5000000,
                            },
                        },
                    ],
                },
            },
            RenderedMember {
                name: String::new(),
                value: RenderedValue::Struct {
                    type_name: None,
                    members: vec![RenderedMember {
                        name: "is_throttled".into(),
                        value: RenderedValue::Bool { value: true },
                    }],
                },
            },
        ],
    };
    let rendered = format!("{v}");
    assert!(
        !rendered.ends_with("{}"),
        "nested anonymous struct must not render as empty: {rendered}"
    );
    assert!(
        rendered.contains("id"),
        "inner field 'id' must be visible: {rendered}"
    );
    assert!(
        rendered.contains("is_throttled"),
        "inner field 'is_throttled' must be visible: {rendered}"
    );
}

// ---- Ptr Display with deref → arrow notation -------------------
//
// A Ptr with `deref: Some(...)` renders as `0x<hex> → <inner>`.
// The arrow ("→", U+2192) is the resolved-pointer indicator.
// Tests cover: deref to a scalar, deref to a CpuList (the
// common cpumask kptr chase), deref to a struct.

#[test]
fn display_ptr_with_scalar_deref_uses_arrow() {
    let v = RenderedValue::Ptr {
        value: 0xffff_8000_1234_5678,
        deref: Some(Box::new(RenderedValue::Uint {
            bits: 32,
            value: 42,
        })),
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    let out = format!("{v}");
    assert!(
        out.contains(" → "),
        "Display must include arrow separator: {out}"
    );
    assert!(out.starts_with("0xffff800012345678"));
    assert!(out.ends_with("42"));
}

#[test]
fn display_ptr_with_cpulist_deref_renders_inline() {
    // Mirrors the cpumask-kptr chase: pointer hex, arrow, then
    // the rendered cpu list.
    let v = RenderedValue::Ptr {
        value: 0xffff_8888_aaaa_bbbb,
        deref: Some(Box::new(RenderedValue::CpuList { cpus: "0-3".into() })),
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    assert_eq!(format!("{v}"), "0xffff8888aaaabbbb → cpus={0-3}");
}

#[test]
fn display_ptr_with_struct_deref_indents_correctly() {
    // The deref payload is a Struct → render follows the arrow.
    // The inline form is `inner{v=7}` (no `struct` keyword,
    // no space before brace, `=` separator).
    let inner = RenderedValue::Struct {
        type_name: Some("inner".into()),
        members: vec![RenderedMember {
            name: "v".into(),
            value: RenderedValue::Uint { bits: 32, value: 7 },
        }],
    };
    let v = RenderedValue::Ptr {
        value: 0xdead_beef,
        deref: Some(Box::new(inner)),
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    let out = format!("{v}");
    assert!(out.contains("0xdeadbeef → inner{"));
    assert!(out.contains("v=7"));
}

#[test]
fn display_ptr_without_deref_no_arrow() {
    // Negative control: deref=None AND no skip reason must NOT
    // emit the arrow.
    let v = RenderedValue::Ptr {
        value: 0xff,
        deref: None,
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    let out = format!("{v}");
    assert!(
        !out.contains("→"),
        "no-deref Ptr must not have arrow: {out}"
    );
    assert_eq!(out, "0xff");
}

/// Ptr with deref_skipped_reason but no deref renders the
/// reason inline in `[chase: ...]` notation. Pins the
/// fix: a chase that failed for a known cause (cross-page,
/// 4 KiB cap, plausibility gate) must surface that cause in
/// the operator-visible Display, not silently look identical
/// to "no chase attempted."
#[test]
fn display_ptr_with_skip_reason_surfaces_inline() {
    let v = RenderedValue::Ptr {
        value: 0x7fff_aaaa_0000,
        deref: None,
        deref_skipped_reason: Some(
            "arena read failed (cross-page boundary or unmapped page)".to_string(),
        ),
        cast_annotation: None,
    };
    let out = format!("{v}");
    assert!(
        out.contains("[chase: arena read failed"),
        "skip reason must be surfaced in [chase: ...] form: {out}"
    );
    assert!(
        out.starts_with("0x7fffaaaa0000"),
        "pointer hex must come first: {out}"
    );
    assert!(
        !out.contains("→"),
        "skip reason render must NOT emit arrow (no actual deref): {out}"
    );
}
