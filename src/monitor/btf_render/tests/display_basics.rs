use super::*;

#[test]
fn read_uint_le_padding() {
    assert_eq!(read_uint_le(&[0x12, 0x34]), 0x3412);
    assert_eq!(read_uint_le(&[0xff]), 0xff);
    assert_eq!(read_uint_le(&[0xff; 8]), u64::MAX);
}

#[test]
fn sign_extend_basic() {
    // 8-bit -1 => 0xFF; sign-extend to 64-bit gives all-ones.
    assert_eq!(sign_extend(0xFF, 8), u64::MAX);
    // 16-bit -1 => 0xFFFF; sign-extend to 64-bit gives all-ones.
    assert_eq!(sign_extend(0xFFFF, 16), u64::MAX);
    // 8-bit 0x7F is positive (max signed); should stay 0x7F.
    assert_eq!(sign_extend(0x7F, 8), 0x7F);
    // 0-bit and 64-bit are no-ops.
    assert_eq!(sign_extend(123, 0), 123);
    assert_eq!(sign_extend(u64::MAX, 64), u64::MAX);
}

#[test]
fn render_int_truncated() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    // `int` is virtually guaranteed present in vmlinux BTF.
    let Ok(ids) = btf.resolve_ids_by_name("int") else {
        crate::report::test_skip("BTF missing 'int' type");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'int' to empty id list");
        return;
    };
    // Empty bytes for a 4-byte int -> Truncated.
    let v = render_value(&btf, id, &[]);
    assert!(matches!(
        v,
        RenderedValue::Truncated {
            needed: 4,
            had: 0,
            ..
        }
    ));
}

#[test]
fn render_truncated_unsigned_int() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    // `u32` is a Linux-side typedef; some BTFs may not expose it.
    // A `test_skip` here surfaces a visible reason rather than a
    // silent pass.
    let Ok(ids) = btf.resolve_ids_by_name("u32") else {
        crate::report::test_skip("BTF missing 'u32' typedef");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'u32' to empty id list");
        return;
    };
    // 2 bytes for a 4-byte u32 should yield Truncated.
    let v = render_value(&btf, id, &[0xff, 0xff]);
    assert!(matches!(
        v,
        RenderedValue::Truncated {
            needed: 4,
            had: 2,
            ..
        }
    ));
}

// ---- Display impl coverage --------------------------------------
//
// Display is the human-readable form used in test failure output.
// Variant matrix tests:
//   - scalars (Int / Uint / Bool / Char / Float / Enum / Ptr)
//   - Bytes / Unsupported
//   - Truncated (with various partial shapes)
//   - Array (inline scalar vs block-style nested)
//   - Struct (named, unnamed, empty, nested)

#[test]
fn display_int_uint_bool() {
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Int {
                bits: 32,
                value: -7
            }
        ),
        "-7"
    );
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Uint {
                bits: 64,
                value: 42
            }
        ),
        "42"
    );
    assert_eq!(format!("{}", RenderedValue::Bool { value: true }), "true");
    assert_eq!(format!("{}", RenderedValue::Bool { value: false }), "false");
}

#[test]
fn display_char_printable_and_nonprintable() {
    // Printable ASCII renders as 'x'.
    assert_eq!(format!("{}", RenderedValue::Char { value: b'A' }), "'A'");
    // Non-printable (NUL, control, high-bit) renders as 0xNN.
    assert_eq!(format!("{}", RenderedValue::Char { value: 0x00 }), "0x00");
    assert_eq!(format!("{}", RenderedValue::Char { value: 0x7f }), "0x7f");
    assert_eq!(format!("{}", RenderedValue::Char { value: 0xab }), "0xab");
}

#[test]
fn display_float() {
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Float {
                bits: 64,
                value: 1.5
            }
        ),
        "1.5"
    );
}

#[test]
fn display_enum_with_and_without_variant() {
    assert_eq!(
        format!("{}", enum_v(32, 1, Some("RUNNING"), false)),
        "RUNNING (1)"
    );
    assert_eq!(format!("{}", enum_v(32, 99, None, false)), "99");
}

#[test]
fn display_ptr_is_lowercase_hex() {
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Ptr {
                value: 0xffff_8000_1234_5678,
                deref: None,
                deref_skipped_reason: None,
                cast_annotation: None,
            }
        ),
        "0xffff800012345678"
    );
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Ptr {
                value: 0,
                deref: None,
                deref_skipped_reason: None,
                cast_annotation: None,
            }
        ),
        "0x0"
    );
    assert_eq!(
        format!(
            "{}",
            RenderedValue::CpuList {
                cpus: "0-7".to_string()
            }
        ),
        "cpus={0-7}"
    );
    assert_eq!(
        format!(
            "{}",
            RenderedValue::CpuList {
                cpus: String::new()
            }
        ),
        "cpus={}"
    );
}

#[test]
fn display_bytes_passes_through() {
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Bytes {
                hex: "12 34 ab".into()
            }
        ),
        "12 34 ab"
    );
}

#[test]
fn display_unsupported_includes_reason() {
    assert_eq!(
        format!(
            "{}",
            RenderedValue::Unsupported {
                reason: "void".into()
            }
        ),
        "<unsupported: void>"
    );
}

#[test]
fn display_truncated_with_bytes_partial() {
    let v = RenderedValue::Truncated {
        needed: 4,
        had: 2,
        partial: Box::new(RenderedValue::Bytes {
            hex: "12 34".into(),
        }),
    };
    assert_eq!(format!("{v}"), "<truncated needed=4 had=2> 12 34");
}

#[test]
fn display_struct_with_named_members() {
    // Display a typical task-context struct value. The inline
    // form is `TypeName{f=v, f=v}` — `=` separates field name
    // from value, no space before the opening brace, and the
    // `struct` keyword is dropped (the type name stands alone).
    let v = RenderedValue::Struct {
        type_name: Some("task_ctx".into()),
        members: vec![
            RenderedMember {
                name: "weight".into(),
                value: RenderedValue::Uint {
                    bits: 32,
                    value: 1024,
                },
            },
            RenderedMember {
                name: "last_runnable_at".into(),
                value: RenderedValue::Uint {
                    bits: 64,
                    value: 12_345_678_901_234,
                },
            },
        ],
    };
    assert_eq!(
        format!("{v}"),
        "task_ctx{weight=1024, last_runnable_at=12345678901234}"
    );
}

#[test]
fn display_struct_anonymous_uses_struct_brace() {
    // Anonymous struct: no type name → inline form is just
    // `{f=v}`.
    let v = RenderedValue::Struct {
        type_name: None,
        members: vec![RenderedMember {
            name: "x".into(),
            value: RenderedValue::Int { bits: 32, value: 7 },
        }],
    };
    assert_eq!(format!("{v}"), "{x=7}");
}

#[test]
fn display_empty_struct_is_one_line() {
    // Empty struct (no members at all) renders inline as
    // `Type{}` (no space between name and brace).
    let v = RenderedValue::Struct {
        type_name: Some("empty".into()),
        members: vec![],
    };
    assert_eq!(format!("{v}"), "empty{}");
}

#[test]
fn display_anonymous_member_uses_anon_marker() {
    // BTF anonymous union/struct members surface with empty
    // name; the inline form marks them with `<anon>=` so the
    // operator knows the position without seeing a bare `=`
    // with no preceding identifier.
    let v = RenderedValue::Struct {
        type_name: Some("u".into()),
        members: vec![RenderedMember {
            name: String::new(),
            value: RenderedValue::Uint { bits: 32, value: 5 },
        }],
    };
    assert_eq!(format!("{v}"), "u{<anon>=5}");
}

#[test]
fn display_nested_struct_renders_inline_when_small() {
    // Outer struct with one nested-Struct field — both small
    // enough to fit inline. Nested-struct value renders via
    // `try_render_inline_string`, packed into the outer's
    // inline form: `outer{child=inner{a=1}}`. No newlines.
    let inner = RenderedValue::Struct {
        type_name: Some("inner".into()),
        members: vec![RenderedMember {
            name: "a".into(),
            value: RenderedValue::Uint { bits: 32, value: 1 },
        }],
    };
    let outer = RenderedValue::Struct {
        type_name: Some("outer".into()),
        members: vec![RenderedMember {
            name: "child".into(),
            value: inner,
        }],
    };
    assert_eq!(format!("{outer}"), "outer{child=inner{a=1}}");
}

#[test]
fn display_nested_struct_breaks_to_multiline_past_inline_budget() {
    // Boundary partner of `display_nested_struct_renders_inline_when_small`:
    // when the inner struct's inline form exceeds
    // STRUCT_INLINE_WIDTH_BUDGET (120), `try_inline_from_rendered`
    // returns None and `write_struct` falls through to the
    // breadcrumb form. The outer wrapping then sees `\n` in the
    // child's pre-rendered string (line ~601 in mod.rs) and also
    // bails to multi-line, so the rendered output must contain
    // newlines.
    //
    // 20 u64 members named `field_NN` with value 3735928559
    // (0xdeadbeef as decimal — `RenderedValue::Uint` renders as
    // decimal regardless of magnitude) produces an inline form
    // around 400+ chars, well past the 120-char budget.
    let inner_members: Vec<RenderedMember> = (0..20)
        .map(|i| RenderedMember {
            name: format!("field_{i:02}"),
            value: RenderedValue::Uint {
                bits: 64,
                value: 0xdeadbeef,
            },
        })
        .collect();
    let inner = RenderedValue::Struct {
        type_name: Some("inner".into()),
        members: inner_members,
    };
    let outer = RenderedValue::Struct {
        type_name: Some("outer".into()),
        members: vec![RenderedMember {
            name: "child".into(),
            value: inner,
        }],
    };
    let rendered = format!("{outer}");
    assert!(
        rendered.contains('\n'),
        "over-budget nested struct must break to multi-line; got: {rendered:?}",
    );
    // The breadcrumb form starts with the outer type name followed
    // by `:` (not `{`) — pin that shape so a regression that
    // re-routes back through the inline path with a wider budget
    // would be caught.
    assert!(
        rendered.starts_with("outer:"),
        "multi-line form must lead with `outer:` breadcrumb, got: {rendered:?}",
    );
    // And the inner field's value must appear at least once so
    // the test isn't satisfied by the breadcrumb header alone.
    assert!(
        rendered.contains("3735928559"),
        "inner-member values must still surface in multi-line form: {rendered:?}",
    );
}

#[test]
fn display_array_scalars_inline() {
    // Use bits:32 to bypass the bits:8 string-detection branch
    // which would route through the C-string render path. With
    // every slot populated and starting at index 0, the array
    // collapses to plain `[v1, v2, v3]` (no run brackets) —
    // run brackets are reserved for sparse / gapped arrays.
    // bits>=32 Uints render as hex via write_array_element.
    let v = RenderedValue::Array {
        len: 3,
        elements: vec![
            RenderedValue::Uint { bits: 32, value: 1 },
            RenderedValue::Uint { bits: 32, value: 2 },
            RenderedValue::Uint { bits: 32, value: 3 },
        ],
    };
    assert_eq!(format!("{v}"), "[0x1, 0x2, 0x3]");
}

#[test]
fn display_array_empty() {
    let v = RenderedValue::Array {
        len: 0,
        elements: vec![],
    };
    assert_eq!(format!("{v}"), "[]");
}

#[test]
fn display_array_truncated_marker() {
    // Element list shorter than declared `len` surfaces the
    // truncation in a comment. The 2 elements form a contiguous
    // run from 0; since the run doesn't cover the full declared
    // length (5), the sparse render path applies: `[0..1]={v, v}`
    // with the trailing truncation comment. Use bits:32 to
    // bypass the bits:8 string-detection branch.
    let v = RenderedValue::Array {
        len: 5,
        elements: vec![
            RenderedValue::Uint { bits: 32, value: 1 },
            RenderedValue::Uint { bits: 32, value: 2 },
        ],
    };
    assert_eq!(format!("{v}"), "[[0..1]={0x1, 0x2}] /* 2 of 5 shown */");
}

#[test]
fn display_array_of_structs_block_style() {
    // Single-element array of struct: block-style render
    // prefixes the struct with `[0] ` to mark the index. The
    // inline-struct form is `Type{f=v}` (no `struct` keyword,
    // no space before brace, `=` separator).
    let elem = RenderedValue::Struct {
        type_name: Some("e".into()),
        members: vec![RenderedMember {
            name: "v".into(),
            value: RenderedValue::Uint {
                bits: 32,
                value: 10,
            },
        }],
    };
    let v = RenderedValue::Array {
        len: 1,
        elements: vec![elem],
    };
    assert_eq!(format!("{v}"), "[\n  [0] e{v=10}\n]");
}

#[test]
fn display_truncated_with_struct_partial_shows_decoded_members() {
    // The partial-render contract: decoded members survive when
    // the struct's byte slice was short. Display surfaces the
    // partial so test failure output points the operator at the
    // fields that DID decode.
    let partial = RenderedValue::Struct {
        type_name: Some("partial_struct".into()),
        members: vec![
            RenderedMember {
                name: "a".into(),
                value: RenderedValue::Uint { bits: 32, value: 7 },
            },
            RenderedMember {
                name: "b".into(),
                value: RenderedValue::Truncated {
                    needed: 4,
                    had: 0,
                    partial: Box::new(RenderedValue::Bytes { hex: "".into() }),
                },
            },
        ],
    };
    let v = RenderedValue::Truncated {
        needed: 8,
        had: 4,
        partial: Box::new(partial),
    };
    let out = format!("{v}");
    // Outer truncation marker then breadcrumb form (the
    // Truncated member `b` prevents inline collapsing).
    assert!(
        out.starts_with("<truncated needed=8 had=4> partial_struct:"),
        "expected breadcrumb form, got: {out}"
    );
    assert!(out.contains("a=7"));
    assert!(
        !out.contains("truncated needed=4 had=0"),
        "had=0 truncated fields must be suppressed: {out}"
    );
}

// ---- partial-render contract -------------------------------------
//
// Truncated must carry a `partial: Box<RenderedValue>` rather than
// discarding decoded members. Two fixtures:
//   1. struct truncation -> partial is Struct with decoded
//      members (one of which may itself be Truncated for the
//      member that overran).
//   2. scalar truncation -> partial is Bytes hex of available bytes.

#[test]
fn truncated_int_carries_bytes_partial() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("u32") else {
        crate::report::test_skip("BTF missing 'u32'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'u32' to empty id list");
        return;
    };
    let v = render_value(&btf, id, &[0x12, 0x34]);
    match v {
        RenderedValue::Truncated {
            needed,
            had,
            partial,
        } => {
            assert_eq!(needed, 4);
            assert_eq!(had, 2);
            match *partial {
                RenderedValue::Bytes { hex } => {
                    assert_eq!(hex, "12 34");
                }
                other => panic!("expected Bytes partial, got {other:?}"),
            }
        }
        other => panic!("expected Truncated, got {other:?}"),
    }
}

#[test]
fn truncated_struct_carries_struct_partial_with_decoded_members() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    // `task_struct` is the canonical large struct in vmlinux BTF.
    let Ok(ids) = btf.resolve_ids_by_name("task_struct") else {
        crate::report::test_skip("BTF missing 'task_struct'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'task_struct' to empty id list");
        return;
    };
    // Feed only 16 bytes — task_struct is multi-KB. Expect
    // Truncated with a Struct partial whose first members
    // decoded (or are themselves Truncated for any member that
    // straddled the cutoff).
    let v = render_value(&btf, id, &[0u8; 16]);
    match v {
        RenderedValue::Truncated {
            needed,
            had,
            partial,
        } => {
            assert!(needed > 16, "expected struct size > 16, got {needed}");
            assert_eq!(had, 16);
            match *partial {
                RenderedValue::Struct { type_name, members } => {
                    assert_eq!(type_name.as_deref(), Some("task_struct"));
                    assert!(
                        !members.is_empty(),
                        "partial struct must carry SOME decoded members"
                    );
                }
                other => panic!("expected Struct partial, got {other:?}"),
            }
        }
        other => panic!("expected Truncated, got {other:?}"),
    }
}

#[test]
fn truncated_array_element_carries_bytes_partial() {
    // Synthesize a struct containing an array whose backing
    // bytes are short. Use BTF if available; otherwise skip.
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    // Find a type that ends in `[]` of int — `cpumask_t.bits`
    // is unsigned long array; struct cpumask exists in vmlinux.
    let Ok(ids) = btf.resolve_ids_by_name("cpumask") else {
        crate::report::test_skip("BTF missing 'cpumask'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'cpumask' to empty id list");
        return;
    };
    // Render with a 1-byte buffer; `bits` is u64[NR_CPUS/64],
    // so the array's first element won't fit, producing a
    // Truncated element somewhere in the partial.
    let v = render_value(&btf, id, &[0u8]);
    // Either the outer struct is Truncated (size > 1) or, if
    // cpumask happens to be 0-byte (kernels with NR_CPUS=0 —
    // not realistic), it would render as Struct. Assert either
    // outcome carries a usable partial / member chain.
    match v {
        RenderedValue::Truncated { partial, .. } => {
            // Partial must be the outer Struct (cpumask), which
            // carries `bits` as either a Truncated array or an
            // array with Truncated elements. Either is correct
            // partial render.
            match *partial {
                RenderedValue::Struct { members, .. } => {
                    // At least one member surfaces partial info.
                    assert!(!members.is_empty());
                }
                other => panic!("expected Struct partial, got {other:?}"),
            }
        }
        // Acceptable fallback: cpumask happens to fit in 1 byte
        // somehow (unlikely in real kernels but not a renderer
        // failure if it does).
        RenderedValue::Struct { .. } => {}
        other => panic!("expected Truncated or Struct, got {other:?}"),
    }
}
