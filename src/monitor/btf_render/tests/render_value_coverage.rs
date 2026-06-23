//! Coverage tests for RenderedValue scalar/array accessors, Display
//! layout edge cases, and the render decode paths (extracted from the
//! pre-decomposition flat suite).
use super::*;

// ---------------------------------------------------------------------------
// RenderedValue scalar coercion accessors: as_i64 / as_f64 / as_bool and the
// typed-array variants. The as_u64 surface is already pinned above; these
// cover the sibling accessors' per-variant arms (mod.rs as_i64 ~347, as_f64
// ~361, as_bool ~385, as_i64_array / as_f64_array, and array_elements'
// Ptr-deref peel).
// ---------------------------------------------------------------------------

/// `as_i64` accepts Int directly, Uint within i64::MAX, Bool, Char, and Enum
/// (signed or unsigned — both stored as the raw i64). A Uint above i64::MAX
/// and a Float reject. Pins every arm of the accessor.
#[test]
fn rendered_value_as_i64_per_variant_arms() {
    assert_eq!(
        RenderedValue::Int {
            bits: 32,
            value: -7,
        }
        .as_i64(),
        Some(-7),
        "Int passes through unchanged",
    );
    assert_eq!(uint(42).as_i64(), Some(42), "Uint <= i64::MAX coerces");
    assert_eq!(
        uint(i64::MAX as u64 + 1).as_i64(),
        None,
        "Uint above i64::MAX rejects (no silent wrap to negative)",
    );
    assert_eq!(RenderedValue::Bool { value: true }.as_i64(), Some(1));
    assert_eq!(RenderedValue::Bool { value: false }.as_i64(), Some(0));
    assert_eq!(RenderedValue::Char { value: b'Z' }.as_i64(), Some(90));
    assert_eq!(
        enum_v(32, -5, Some("X"), true).as_i64(),
        Some(-5),
        "Enum surfaces its stored i64 verbatim",
    );
    assert_eq!(
        RenderedValue::Float {
            bits: 64,
            value: 1.5,
        }
        .as_i64(),
        None,
        "Float has no lossless i64 coercion",
    );
}

/// `as_f64` is direct from Float and widens Int / Uint / Enum via `as f64`.
/// Bool, Char, and aggregate variants reject.
#[test]
fn rendered_value_as_f64_per_variant_arms() {
    assert_eq!(
        RenderedValue::Float {
            bits: 32,
            value: 2.5,
        }
        .as_f64(),
        Some(2.5),
    );
    assert_eq!(
        RenderedValue::Int {
            bits: 32,
            value: -3,
        }
        .as_f64(),
        Some(-3.0),
        "Int widens to f64",
    );
    assert_eq!(uint(8).as_f64(), Some(8.0), "Uint widens to f64");
    assert_eq!(
        enum_v(32, 4, None, true).as_f64(),
        Some(4.0),
        "Enum widens to f64",
    );
    assert_eq!(
        RenderedValue::Bool { value: true }.as_f64(),
        None,
        "Bool has no f64 arm",
    );
    assert_eq!(struct_with(vec![]).as_f64(), None);
}

/// `as_bool` is direct from Bool and treats Uint / Int / Char / Enum as a
/// `!= 0` test. Float and aggregates reject. The Ptr arm is pinned separately
/// in `rendered_value_as_bool_coerces_ptr_as_non_null`.
#[test]
fn rendered_value_as_bool_non_ptr_arms() {
    assert_eq!(RenderedValue::Bool { value: false }.as_bool(), Some(false));
    assert_eq!(uint(0).as_bool(), Some(false), "zero Uint is false");
    assert_eq!(uint(9).as_bool(), Some(true), "non-zero Uint is true");
    assert_eq!(
        RenderedValue::Int {
            bits: 32,
            value: -1,
        }
        .as_bool(),
        Some(true),
        "non-zero Int is true",
    );
    assert_eq!(RenderedValue::Char { value: 0 }.as_bool(), Some(false));
    assert_eq!(RenderedValue::Char { value: b'q' }.as_bool(), Some(true));
    assert_eq!(enum_v(32, 0, None, false).as_bool(), Some(false));
    assert_eq!(enum_v(32, 3, None, false).as_bool(), Some(true));
    assert_eq!(
        RenderedValue::Float {
            bits: 64,
            value: 0.0,
        }
        .as_bool(),
        None,
        "Float has no bool coercion",
    );
}

/// `as_i64_array` and `as_f64_array` collect every element through the scalar
/// accessor; a single element that fails the coercion drops the whole result.
#[test]
fn rendered_value_as_i64_and_f64_array_collect_and_reject() {
    let i64_arr = RenderedValue::Array {
        len: 3,
        elements: vec![
            RenderedValue::Int {
                bits: 32,
                value: -1,
            },
            RenderedValue::Int { bits: 32, value: 0 },
            RenderedValue::Int { bits: 32, value: 5 },
        ],
    };
    assert_eq!(i64_arr.as_i64_array(), Some(vec![-1, 0, 5]));

    let f64_arr = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Float {
                bits: 64,
                value: 1.0,
            },
            RenderedValue::Float {
                bits: 64,
                value: 2.0,
            },
        ],
    };
    assert_eq!(f64_arr.as_f64_array(), Some(vec![1.0, 2.0]));

    // One non-coercible element (a Bool can't be an f64) makes the whole
    // f64 collection reject rather than returning a partial vec.
    let mixed = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Float {
                bits: 64,
                value: 1.0,
            },
            RenderedValue::Bool { value: true },
        ],
    };
    assert_eq!(
        mixed.as_f64_array(),
        None,
        "a non-coercible element rejects the whole collection",
    );

    // Non-array self rejects.
    assert_eq!(uint(1).as_i64_array(), None);
}

/// `array_elements` (the shared peel behind every `as_*_array` accessor)
/// transparently peels a `Ptr{deref: Some(Array)}`, mirroring `index`'s
/// random-access peel so full-iteration and random-access agree.
#[test]
fn rendered_value_array_accessor_peels_ptr_deref() {
    let inner = RenderedValue::Array {
        len: 2,
        elements: vec![uint(11), uint(22)],
    };
    let ptr = RenderedValue::Ptr {
        value: 0x4000,
        deref: Some(Box::new(inner)),
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    assert_eq!(
        ptr.as_u64_array(),
        Some(vec![11, 22]),
        "as_u64_array peels Ptr{{deref: Some(Array)}}",
    );
    // index() peels the same Ptr deref for random access.
    assert!(matches!(
        ptr.index(1),
        Some(RenderedValue::Uint { value: 22, .. })
    ));
}

// ---------------------------------------------------------------------------
// Display layout engine: paths the existing golden-string tests do not reach
// — unsigned-Enum wire value, Ptr cast_annotation tag, the multi-line struct
// column `=` alignment, the inline-list wrap-at-budget path, 8-bit Int/Uint
// C-string detection, and the block-style group `[start-end]` / truncation
// markers.
// ---------------------------------------------------------------------------

/// An unsigned Enum stored as a negative i64 (the renderer keeps the raw bit
/// pattern) must Display its u64 wire value, not the signed `-1`. Mirrors the
/// `as_u64` round-trip — without the `is_signed` Display branch a u64::MAX
/// variant printed as `-1`.
#[test]
fn display_unsigned_enum_renders_wire_value_not_negative() {
    let unsigned_max = enum_v(64, -1_i64, None, false);
    assert_eq!(
        format!("{unsigned_max}"),
        "18446744073709551615",
        "unsigned enum displays its u64 wire value",
    );
    // A signed enum with the same storage is a true -1 and displays as such.
    let signed_neg = enum_v(64, -1_i64, None, true);
    assert_eq!(format!("{signed_neg}"), "-1");
    // Named unsigned variant: `name (wire_value)`.
    let named = enum_v(64, -1_i64, Some("ALL"), false);
    assert_eq!(format!("{named}"), "ALL (18446744073709551615)");
}

/// A Ptr carrying a `cast_annotation` surfaces it as a parenthesised tag
/// right after the hex value — the operator-visible marker distinguishing
/// cast-recovered pointers from native BTF pointers.
#[test]
fn display_ptr_with_cast_annotation_surfaces_tag() {
    let v = RenderedValue::Ptr {
        value: 0xdead_beef,
        deref: None,
        deref_skipped_reason: None,
        cast_annotation: Some(std::borrow::Cow::Borrowed("cast→arena")),
    };
    // `0x<hex>` followed by the parenthesised cast tag, no deref arrow.
    assert_eq!(format!("{v}"), "0xdeadbeef (cast→arena)");
}

/// A Ptr whose skip reason starts with `cycle ` collapses to the dense
/// `[cycle]` marker rather than the verbose `[chase: ...]` form — the cycle
/// address is already visible in the preceding hex.
#[test]
fn display_ptr_cycle_reason_collapses_to_dense_marker() {
    let v = RenderedValue::Ptr {
        value: 0xabcd,
        deref: None,
        deref_skipped_reason: Some("cycle → 0xabcd".to_string()),
        cast_annotation: None,
    };
    assert_eq!(format!("{v}"), "0xabcd [cycle]");
}

/// Multi-line struct column `=` alignment: a struct that overflows the inline
/// budget AND has 3+ rows of scalar cells with >=4-char field-name variation
/// in a column pads names so the equals signs line up (` = ` form). Pins the
/// padded-column branch (mod.rs ~1095) distinct from the compact `name=value`
/// branch.
#[test]
fn display_multiline_struct_column_alignment_pads_equals() {
    // 9 scalar fields (3 rows of 3). Column 0 carries names whose lengths
    // span >= 4 chars ("a" vs "a_long_name") so pad_eq[0] is true; the
    // padded form emits ` = ` with the short name padded out to the column
    // max. Large values force the inline form past the 120-char budget so
    // the breadcrumb/multi-line path is taken.
    let mk = |name: &str, value: u64| RenderedMember {
        name: name.into(),
        value: RenderedValue::Uint { bits: 64, value },
    };
    let v = RenderedValue::Struct {
        type_name: Some("wide".into()),
        members: vec![
            mk("a", 111_111_111_111),
            mk("bb", 222_222_222_222),
            mk("cc", 333_333_333_333),
            mk("a_long_name", 444_444_444_444),
            mk("dd", 555_555_555_555),
            mk("ee", 666_666_666_666),
            mk("ff", 777_777_777_777),
            mk("gg", 888_888_888_888),
            mk("hh", 999_999_999_999),
        ],
    };
    let out = format!("{v}");
    assert!(
        out.starts_with("wide:"),
        "multi-line breadcrumb form: {out}"
    );
    // Column 0 names "a" and "a_long_name" differ by >= 4 chars → the column
    // is padded so the equals signs align. The long name (column max width)
    // emits the aligned ` = ` form with no extra name padding.
    assert!(
        out.contains("a_long_name = 444444444444"),
        "long column-0 name uses the ` = ` aligned form: {out}",
    );
    // The short name "a" is space-padded out to the column's max name width
    // (11) before the aligned ` = ` separator. Locate the row and assert the
    // padded shape: `a` + spaces + ` = ` + value (distinct from the compact
    // `a=value` form taken when a column is not padded).
    let padded = format!("a{} = 111111111111", " ".repeat(11 - 1));
    assert!(
        out.contains(&padded),
        "short column-0 name must be padded to width 11 before the aligned \
         ` = `; looked for {padded:?} in: {out}",
    );
}

/// `write_inline_list_wrapped` wrap mode: a fully-populated inline scalar
/// array whose single-line `[v, v, ...]` form exceeds the 120-char budget
/// wraps at element boundaries (never mid-element) with continuation rows
/// indented one level. bits>=32 Uints render as hex.
#[test]
fn display_inline_array_wraps_past_budget() {
    // 12 u64 values each rendering as an 18-char hex literal
    // (0x____________________). Joined by ", " the single line is well past
    // 120 chars, forcing the wrap path.
    let elements: Vec<RenderedValue> = (0..12)
        .map(|_| RenderedValue::Uint {
            bits: 64,
            value: 0x1111_2222_3333_4444,
        })
        .collect();
    let v = RenderedValue::Array { len: 12, elements };
    let out = format!("{v}");
    assert!(
        out.contains('\n'),
        "over-budget contiguous array must wrap to multiple lines: {out}",
    );
    assert!(
        out.starts_with('['),
        "wrap output still opens with `[`: {out}"
    );
    assert!(
        out.ends_with(']'),
        "wrap output still closes with `]`: {out}"
    );
    // Every element value must survive the wrap (no mid-element break / drop).
    assert_eq!(
        out.matches("0x1111222233334444").count(),
        12,
        "all 12 elements present after wrap: {out}",
    );
    // The line break elides the separator's trailing space: a wrapped row
    // ends with `,` immediately before the newline.
    assert!(out.contains(",\n"), "wrap breaks after the comma: {out}");
}

/// 8-bit Int/Uint arrays (not Char) that hold printable ASCII are detected as
/// C strings and rendered quoted — the string-detection branch has distinct
/// arms for Int{bits:8}, Uint{bits:8}, and Char. The Char arm is covered
/// elsewhere; this pins the Int and Uint arms.
#[test]
fn display_8bit_int_uint_arrays_render_as_quoted_strings() {
    let int_str = RenderedValue::Array {
        len: 3,
        elements: vec![
            RenderedValue::Int {
                bits: 8,
                value: b'h' as i64,
            },
            RenderedValue::Int {
                bits: 8,
                value: b'i' as i64,
            },
            RenderedValue::Int { bits: 8, value: 0 },
        ],
    };
    assert_eq!(
        format!("{int_str}"),
        "\"hi\"",
        "8-bit Int array reads as a C string"
    );

    let uint_str = RenderedValue::Array {
        len: 3,
        elements: vec![
            RenderedValue::Uint {
                bits: 8,
                value: b'o' as u64,
            },
            RenderedValue::Uint {
                bits: 8,
                value: b'k' as u64,
            },
            RenderedValue::Uint { bits: 8, value: 0 },
        ],
    };
    assert_eq!(
        format!("{uint_str}"),
        "\"ok\"",
        "8-bit Uint array reads as a C string"
    );
}

/// A multi-line C string built from 8-bit Uint elements (not Char) exercises
/// the Uint arm of the newline-extraction loop inside the `|` block-scalar
/// render path.
#[test]
fn display_8bit_uint_multiline_string_uses_pipe_block() {
    let v = RenderedValue::Array {
        len: 5,
        elements: vec![
            RenderedValue::Uint {
                bits: 8,
                value: b'x' as u64,
            },
            RenderedValue::Uint {
                bits: 8,
                value: b'\n' as u64,
            },
            RenderedValue::Uint {
                bits: 8,
                value: b'y' as u64,
            },
            RenderedValue::Uint { bits: 8, value: 0 },
            RenderedValue::Uint { bits: 8, value: 0 },
        ],
    };
    let out = format!("{v}");
    assert!(
        out.starts_with("|\n"),
        "multiline string uses pipe block: {out}"
    );
    assert!(out.contains("x"), "first segment present: {out}");
    assert!(out.contains("y"), "second segment present: {out}");
}

/// Block-style array whose rendered element count is fewer than the declared
/// `len` appends a `/* N of M shown */` truncation note. Reached by an array
/// of non-inline (struct) elements where `elements.len() < len`.
#[test]
fn display_block_array_truncation_note_when_fewer_shown() {
    let mk = |x: u64| RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![RenderedMember {
            name: "x".into(),
            value: RenderedValue::Uint { bits: 32, value: x },
        }],
    };
    // len declares 5 but only 2 elements are present (e.g. a MAX_ARRAY_ELEMS
    // clamp in the real renderer). The block path emits the shown count.
    let v = RenderedValue::Array {
        len: 5,
        elements: vec![mk(1), mk(2)],
    };
    let out = format!("{v}");
    assert!(
        out.contains("/* 2 of 5 shown */"),
        "block render must note partial element coverage: {out}",
    );
}

/// Block-style struct-run scan breaks when a later group's member count
/// differs from the run's first struct: the differing-shape struct is
/// rendered as its own group rather than merged into a template. Exercises
/// the `next_m.len() != first_m.len()` break in the run scanner (mod.rs ~792).
#[test]
fn display_block_array_struct_run_breaks_on_member_count_mismatch() {
    // Distinct-content structs so the leading group walker keeps each as
    // its own single-element group (no run collapse). The run scanner then
    // tries to merge them: group[0] (2 fields) vs group[1] (2 fields) is a
    // candidate, but group[2] has a different member count — the
    // `next_m.len() != first_m.len()` break fires, ending the run before
    // the one-field struct. Distinct `a` values keep the structs ungrouped.
    let two_field = |a: u64| RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![
            RenderedMember {
                name: "a".into(),
                value: RenderedValue::Uint { bits: 32, value: a },
            },
            RenderedMember {
                name: "b".into(),
                value: RenderedValue::Uint { bits: 32, value: 2 },
            },
        ],
    };
    let one_field = RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![RenderedMember {
            name: "a".into(),
            value: RenderedValue::Uint { bits: 32, value: 9 },
        }],
    };
    let v = RenderedValue::Array {
        len: 3,
        elements: vec![two_field(1), two_field(7), one_field],
    };
    let out = format!("{v}");
    assert!(
        out.contains("[2] "),
        "differing-shape struct renders standalone: {out}"
    );
    assert!(
        out.contains("a=9"),
        "the one-field struct's value surfaces: {out}"
    );
    assert!(out.contains("a=1"), "first struct's value surfaces: {out}");
    assert!(out.contains("a=7"), "second struct's value surfaces: {out}");
}

/// A template-eligible run of anonymous structs (no type_name) emits the
/// `[start-end]:` header without a name, and a common all-zero field is
/// suppressed from the common-fields section. Exercises the `None` type_name
/// arm (mod.rs ~1517) and the `is_deeply_zero` common-field skip (~1533).
#[test]
fn display_anonymous_struct_template_omits_name_and_suppresses_zero_common_field() {
    // Anonymous structs (type_name None) with a constant zero `pad` field
    // and a varying `v` field. 3 elements → template merge fires.
    let mk = |v: u64| RenderedValue::Struct {
        type_name: None,
        members: vec![
            RenderedMember {
                name: "pad".into(),
                value: RenderedValue::Uint { bits: 32, value: 0 },
            },
            RenderedMember {
                name: "v".into(),
                value: RenderedValue::Uint { bits: 32, value: v },
            },
        ],
    };
    let arr = RenderedValue::Array {
        len: 3,
        elements: vec![mk(1), mk(2), mk(3)],
    };
    let out = format!("{arr}");
    // Anonymous template header: `[0-2]:` (no name after the range).
    assert!(
        out.contains("[0-2]:"),
        "anonymous template uses nameless `[range]:` header: {out}",
    );
    // The constant-zero `pad` common field is suppressed (is_deeply_zero).
    assert!(
        !out.contains("pad="),
        "all-zero common field must be suppressed from the template: {out}",
    );
    // The varying field still renders as a per-index list.
    assert!(out.contains("v: "), "varying field present: {out}");
    assert!(out.contains("[0]="), "per-index marker present: {out}");
}

// ---------------------------------------------------------------------------
// Decode-path arms reached through render_value with synthetic BTF: char int,
// >8-byte int hex fallback, Enum/Enum64 truncation, standalone Var forward,
// Fwd / Func / Void Unsupported arms, datasec malformed entries, the bitfield
// non-int/non-enum base, and type_size's Array / modifier arms.
// ---------------------------------------------------------------------------

/// A 1-byte char-encoded BTF int renders as `RenderedValue::Char`. The
/// `is_char()` arm (mod.rs ~2618) is reached only for `needed == 1`; encoding
/// bit 2 (`BTF_INT_CHAR`) sets it.
#[test]
fn render_char_int_renders_as_char_variant() {
    let mut strings: Vec<u8> = vec![0];
    let n = strings.len() as u32;
    strings.extend_from_slice(b"char\0");
    // encoding=2 == BTF_INT_CHAR; size 1, bits 8.
    let types = vec![CastSynType::Int {
        name_off: n,
        size: 1,
        encoding: 2,
        offset: 0,
        bits: 8,
    }];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let v = render_value(&btf, 1, b"Q");
    assert_eq!(v, RenderedValue::Char { value: b'Q' });
}

/// A BTF int wider than 8 bytes (e.g. `__int128`, size 16) cannot fit a u64,
/// so the renderer falls back to a `Bytes` hex dump rather than discarding the
/// upper bits (mod.rs ~2626).
#[test]
fn render_int_wider_than_8_bytes_falls_back_to_bytes() {
    let mut strings: Vec<u8> = vec![0];
    let n = strings.len() as u32;
    strings.extend_from_slice(b"__int128\0");
    let types = vec![CastSynType::Int {
        name_off: n,
        size: 16,
        encoding: 0,
        offset: 0,
        bits: 128,
    }];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let bytes: Vec<u8> = (0u8..16).collect();
    let v = render_value(&btf, 1, &bytes);
    match v {
        RenderedValue::Bytes { hex } => {
            assert_eq!(
                hex, "00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f",
                "wide int dumps every byte as lowercase hex",
            );
        }
        other => panic!("expected Bytes hex fallback for a 16-byte int, got {other:?}"),
    }
}

/// An Enum value with fewer bytes than the enum's declared size yields a
/// `Truncated` whose partial is the raw hex of what was available (mod.rs
/// ~2157). A full-width Enum resolves its variant name (~2186).
#[test]
fn render_enum_truncated_and_variant_resolution() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_e = push(&mut strings, "E");
    let n_a = push(&mut strings, "A");
    let n_b = push(&mut strings, "B");
    let types = vec![CastSynType::Enum {
        name_off: n_e,
        size: 4,
        signed: false,
        members: vec![(n_a, 1), (n_b, 2)],
    }];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // 2 bytes for a 4-byte enum -> Truncated{needed:4, had:2, Bytes}.
    let trunc = render_value(&btf, 1, &[0x01, 0x00]);
    match trunc {
        RenderedValue::Truncated {
            needed,
            had,
            partial,
        } => {
            assert_eq!(needed, 4);
            assert_eq!(had, 2);
            assert_eq!(
                *partial,
                RenderedValue::Bytes {
                    hex: "01 00".into()
                }
            );
        }
        other => panic!("expected Truncated enum, got {other:?}"),
    }

    // Full width, value 2 -> variant "B".
    let v = render_value(&btf, 1, &[0x02, 0x00, 0x00, 0x00]);
    assert_eq!(v, enum_v(32, 2, Some("B"), false));
}

/// Enum64 mirrors the Enum32 decode: truncation surfaces hex, full-width
/// resolves the variant. Exercises the Enum64 arm (mod.rs ~2198).
#[test]
fn render_enum64_truncated_and_variant_resolution() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_e = push(&mut strings, "E64");
    let n_hi = push(&mut strings, "HI");
    let types = vec![CastSynType::Enum64 {
        name_off: n_e,
        size: 8,
        signed: false,
        members: vec![(n_hi, 0x1_0000_0000u64)],
    }];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");

    // 4 bytes for an 8-byte enum64 -> Truncated.
    let trunc = render_value(&btf, 1, &[0, 0, 0, 0]);
    match trunc {
        RenderedValue::Truncated { needed, had, .. } => {
            assert_eq!(needed, 8);
            assert_eq!(had, 4);
        }
        other => panic!("expected Truncated enum64, got {other:?}"),
    }

    // Full width 0x1_0000_0000 -> variant "HI".
    let bytes = 0x1_0000_0000u64.to_le_bytes();
    let v = render_value(&btf, 1, &bytes);
    assert_eq!(v, enum_v(64, 0x1_0000_0000, Some("HI"), false));
}

/// Rendering a standalone `BTF_KIND_VAR` type id forwards to its underlying
/// type against the supplied bytes (mod.rs ~2563). libbpf wraps Var in a
/// Datasec normally; this hits the standalone path directly.
#[test]
fn render_standalone_var_forwards_to_underlying_type() {
    // Build via the raw wire format: Var (kind 14) is not in CastSynType,
    // so emit a minimal blob by hand. id1: u32 int; id2: VAR -> id1.
    const BTF_KIND_VAR: u32 = 14;
    let mut strings: Vec<u8> = vec![0];
    let n_u32 = strings.len() as u32;
    strings.extend_from_slice(b"u32\0");
    let n_g = strings.len() as u32;
    strings.extend_from_slice(b"g\0");

    let mut type_section = Vec::new();
    // id1: BTF_KIND_INT u32.
    type_section.extend_from_slice(&n_u32.to_le_bytes());
    type_section.extend_from_slice(&((1u32 << 24) & 0x1f00_0000).to_le_bytes());
    type_section.extend_from_slice(&4u32.to_le_bytes());
    type_section.extend_from_slice(&32u32.to_le_bytes());
    // id2: BTF_KIND_VAR "g" -> id1, plus linkage word (1).
    type_section.extend_from_slice(&n_g.to_le_bytes());
    type_section.extend_from_slice(&((BTF_KIND_VAR << 24) & 0x1f00_0000).to_le_bytes());
    type_section.extend_from_slice(&1u32.to_le_bytes()); // type = id1
    type_section.extend_from_slice(&1u32.to_le_bytes()); // linkage (global)

    let type_len = type_section.len() as u32;
    let str_len = strings.len() as u32;
    let mut blob = Vec::new();
    blob.extend_from_slice(&0xEB9Fu16.to_le_bytes());
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&24u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.extend_from_slice(&type_len.to_le_bytes());
    blob.extend_from_slice(&type_len.to_le_bytes());
    blob.extend_from_slice(&str_len.to_le_bytes());
    blob.extend_from_slice(&type_section);
    blob.extend_from_slice(&strings);

    let btf = Btf::from_bytes(&blob).expect("synthetic Var BTF parses");
    let v = render_value(&btf, 2, &0xDEADu32.to_le_bytes());
    assert_eq!(
        v,
        RenderedValue::Uint {
            bits: 32,
            value: 0xDEAD,
        },
        "standalone Var renders its underlying u32 against the bytes",
    );
}

/// A `BTF_KIND_FWD` rendered directly (its body is in another BTF) yields
/// `Unsupported` with a forward-declaration reason (mod.rs ~2546).
#[test]
fn render_fwd_type_is_unsupported_forward_declaration() {
    let mut strings: Vec<u8> = vec![0];
    let n = strings.len() as u32;
    strings.extend_from_slice(b"opaque\0");
    let types = vec![CastSynType::Fwd {
        name_off: n,
        is_union: false,
    }];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    let v = render_value(&btf, 1, &[0u8; 8]);
    match v {
        RenderedValue::Unsupported { reason } => {
            assert!(
                reason.contains("forward declaration"),
                "Fwd render reason names the forward declaration: {reason}",
            );
        }
        other => panic!("expected Unsupported for Fwd, got {other:?}"),
    }
}

/// A bitfield whose base type is neither Int nor Enum (here a Ptr) is treated
/// as unsigned by `render_bitfield` — the `_ => false` signedness arm (mod.rs
/// ~4570). The result is a `Uint` of the raw bits, never sign-extended.
#[test]
fn render_bitfield_non_int_non_enum_base_is_unsigned() {
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
        // id1: u64 (Ptr pointee).
        CastSynType::Int {
            name_off: n_u64,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id2: Ptr -> u64 (a non-Int, non-Enum base for the bitfield).
        CastSynType::Ptr { type_id: 1 },
        // id3: struct B { (Ptr) f : 4; } size 8, kind_flag bitfield.
        CastSynType::BitfieldStruct {
            name_off: n_b,
            size: 8,
            members: vec![CastSynBitMember {
                name_off: n_f,
                type_id: 2,
                bit_offset: 0,
                bitfield_size: 4,
            }],
        },
    ];
    let blob = cast_build_btf(&types, &strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic BTF parses");
    // Low nibble 0xF: an Int base would sign-extend to -1; a Ptr base stays
    // unsigned -> Uint{4, 0xF}.
    let mut bytes = [0u8; 8];
    bytes[0] = 0x0F;
    let v = render_value(&btf, 3, &bytes);
    assert_eq!(
        sole_member_value(&v),
        &RenderedValue::Uint {
            bits: 4,
            value: 0xF
        },
        "a non-Int / non-Enum bitfield base is rendered unsigned",
    );
}

/// `type_size` resolves an array member's size as `len * elem_size` and peels
/// a Const-wrapped member to its underlying size. Reached via a struct with a
/// `u32 arr[3]` member and a `const u32` member: rendering the struct sizes
/// each member through `type_size` (mod.rs Array arm ~4734, Const arm ~4740).
#[test]
fn render_struct_sizes_array_and_const_members_via_type_size() {
    let mut strings: Vec<u8> = vec![0];
    let push = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u32 = push(&mut strings, "u32");
    let n_s = push(&mut strings, "S");
    let n_arr = push(&mut strings, "arr");
    let n_c = push(&mut strings, "c");

    // CastSynType has no Array variant; emit the array type (kind 3) and the
    // struct by hand to keep the struct member layout aligned with type_size.
    const BTF_KIND_ARRAY: u32 = 3;
    let mut type_section = Vec::new();
    // id1: BTF_KIND_INT u32 (size 4).
    type_section.extend_from_slice(&n_u32.to_le_bytes());
    type_section.extend_from_slice(&((1u32 << 24) & 0x1f00_0000).to_le_bytes());
    type_section.extend_from_slice(&4u32.to_le_bytes());
    type_section.extend_from_slice(&32u32.to_le_bytes());
    // id2: BTF_KIND_ARRAY of id1, len 3. btf_array{ type(4), index_type(4),
    // nelems(4) }; the type record's size_type word is unused for arrays.
    type_section.extend_from_slice(&0u32.to_le_bytes()); // name_off
    type_section.extend_from_slice(&((BTF_KIND_ARRAY << 24) & 0x1f00_0000).to_le_bytes());
    type_section.extend_from_slice(&0u32.to_le_bytes()); // size_type (unused)
    type_section.extend_from_slice(&1u32.to_le_bytes()); // elem type id
    type_section.extend_from_slice(&1u32.to_le_bytes()); // index type id
    type_section.extend_from_slice(&3u32.to_le_bytes()); // nelems
    // id3: BTF_KIND_CONST -> id1.
    const BTF_KIND_CONST: u32 = 10;
    type_section.extend_from_slice(&0u32.to_le_bytes());
    type_section.extend_from_slice(&((BTF_KIND_CONST << 24) & 0x1f00_0000).to_le_bytes());
    type_section.extend_from_slice(&1u32.to_le_bytes());
    // id4: struct S { u32 arr[3] @0; const u32 c @12; } size 16.
    type_section.extend_from_slice(&n_s.to_le_bytes());
    let vlen = 2u32;
    type_section.extend_from_slice(&(((4u32 << 24) & 0x1f00_0000) | (vlen & 0xffff)).to_le_bytes());
    type_section.extend_from_slice(&16u32.to_le_bytes());
    // member arr -> id2 @ bit 0.
    type_section.extend_from_slice(&n_arr.to_le_bytes());
    type_section.extend_from_slice(&2u32.to_le_bytes());
    type_section.extend_from_slice(&0u32.to_le_bytes());
    // member c -> id3 @ bit 96 (byte 12).
    type_section.extend_from_slice(&n_c.to_le_bytes());
    type_section.extend_from_slice(&3u32.to_le_bytes());
    type_section.extend_from_slice(&(96u32).to_le_bytes());

    let type_len = type_section.len() as u32;
    let str_len = strings.len() as u32;
    let mut blob = Vec::new();
    blob.extend_from_slice(&0xEB9Fu16.to_le_bytes());
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&24u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.extend_from_slice(&type_len.to_le_bytes());
    blob.extend_from_slice(&type_len.to_le_bytes());
    blob.extend_from_slice(&str_len.to_le_bytes());
    blob.extend_from_slice(&type_section);
    blob.extend_from_slice(&strings);

    let btf = Btf::from_bytes(&blob).expect("synthetic array/const BTF parses");
    // 16 bytes: arr = [1, 2, 3], c = 7.
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&1u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
    bytes[8..12].copy_from_slice(&3u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&7u32.to_le_bytes());
    let v = render_value(&btf, 4, &bytes);
    let RenderedValue::Struct { members, .. } = v else {
        panic!("expected Struct, got {v:?}");
    };
    assert_eq!(members.len(), 2);
    assert_eq!(members[0].name, "arr");
    assert_eq!(
        members[0].value,
        RenderedValue::Array {
            len: 3,
            elements: vec![uint32(1), uint32(2), uint32(3)],
        },
        "array member sized as len*elem via type_size",
    );
    assert_eq!(members[1].name, "c");
    assert_eq!(
        members[1].value,
        uint32(7),
        "const-wrapped member peeled and sized via type_size",
    );
}

/// Helper: a 32-bit unsigned `RenderedValue` for the array/const member test.
fn uint32(value: u64) -> RenderedValue {
    RenderedValue::Uint { bits: 32, value }
}

// ---------------------------------------------------------------------------
// unsizable_chase_reason variants reached through a cast chase: a Func / Void
// pointee target produces a distinct, type-named skip reason instead of the
// generic "unresolvable size" message. Pins the per-kind diagnostic arms
// (mod.rs ~3224 Func, ~3241 Void).
// ---------------------------------------------------------------------------

/// A kernel cast whose recovered target type is a `BTF_KIND_FUNC` cannot be
/// sized; `unsizable_chase_reason` surfaces the function-specific message.
#[test]
fn cast_chase_kernel_func_target_surfaces_func_reason() {
    // id1: u64 (FuncProto return / the chased field type). id2: FuncProto.
    // id3: Func -> id2. The chase target_type_id points at the Func.
    const BTF_KIND_FUNC_PROTO: u32 = 13;
    const BTF_KIND_FUNC: u32 = 12;
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = strings.len() as u32;
    strings.extend_from_slice(b"u64\0");
    let n_fn = strings.len() as u32;
    strings.extend_from_slice(b"fn\0");

    let mut type_section = Vec::new();
    // id1: u64 int.
    type_section.extend_from_slice(&n_u64.to_le_bytes());
    type_section.extend_from_slice(&((1u32 << 24) & 0x1f00_0000).to_le_bytes());
    type_section.extend_from_slice(&8u32.to_le_bytes());
    type_section.extend_from_slice(&64u32.to_le_bytes());
    // id2: FuncProto (vlen 0) returning id1.
    type_section.extend_from_slice(&0u32.to_le_bytes());
    type_section.extend_from_slice(&((BTF_KIND_FUNC_PROTO << 24) & 0x1f00_0000).to_le_bytes());
    type_section.extend_from_slice(&1u32.to_le_bytes()); // return type id1
    // id3: Func "fn" -> id2.
    type_section.extend_from_slice(&n_fn.to_le_bytes());
    type_section.extend_from_slice(&((BTF_KIND_FUNC << 24) & 0x1f00_0000).to_le_bytes());
    type_section.extend_from_slice(&2u32.to_le_bytes()); // proto id2
    // id4: struct T { u64 f @0; } size 8 — the parent the cast keys on.
    let n_t = strings.len() as u32;
    strings.extend_from_slice(b"T\0");
    let n_f = strings.len() as u32;
    strings.extend_from_slice(b"f\0");
    type_section.extend_from_slice(&n_t.to_le_bytes());
    type_section.extend_from_slice(&(((4u32 << 24) & 0x1f00_0000) | 1).to_le_bytes());
    type_section.extend_from_slice(&8u32.to_le_bytes());
    type_section.extend_from_slice(&n_f.to_le_bytes());
    type_section.extend_from_slice(&1u32.to_le_bytes());
    type_section.extend_from_slice(&0u32.to_le_bytes());

    let type_len = type_section.len() as u32;
    let str_len = strings.len() as u32;
    let mut blob = Vec::new();
    blob.extend_from_slice(&0xEB9Fu16.to_le_bytes());
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&24u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.extend_from_slice(&type_len.to_le_bytes());
    blob.extend_from_slice(&type_len.to_le_bytes());
    blob.extend_from_slice(&str_len.to_le_bytes());
    blob.extend_from_slice(&type_section);
    blob.extend_from_slice(&strings);
    let btf = Btf::from_bytes(&blob).expect("synthetic Func BTF parses");

    // T is id4. Cast on (T_id, 0) -> target_type_id = Func id3, kernel space.
    // Runtime value is outside any arena window (kernel arm), and read_kva is
    // configured so the chase reaches the size resolution before any read.
    const KVA: u64 = 0xffff_9000_0000_1000;
    let mut cast_map = crate::monitor::cast_analysis::CastMap::new();
    cast_map.insert(
        (4, 0),
        CastHit {
            alloc_size: None,
            target_type_id: 3,
            addr_space: AddrSpace::Kernel,
        },
    );
    let reader = CastStubReader {
        cast_map: Some(cast_map),
        ..Default::default()
    };
    let outer = KVA.to_le_bytes().to_vec();
    let v = render_value_with_mem(&btf, 4, &outer, &reader);
    let RenderedValue::Struct { members, .. } = v else {
        panic!("expected Struct, got {v:?}");
    };
    let RenderedValue::Ptr {
        deref,
        deref_skipped_reason,
        ..
    } = &members[0].value
    else {
        panic!("cast member must render as Ptr, got {:?}", members[0].value);
    };
    assert!(deref.is_none(), "func target cannot be chased: deref None");
    let reason = deref_skipped_reason
        .as_deref()
        .expect("a Func chase target must record a skip reason");
    assert!(
        reason.contains("function") && reason.contains("no storage size"),
        "Func target surfaces the function-specific unsizable reason: {reason}",
    );
}
