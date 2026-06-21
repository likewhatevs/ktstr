use super::*;

// ---- RenderedValue typed-array accessors (uncovered arms) -------
//
// `as_u32_array` / `as_i64_array` / `as_f64_array` / `as_bool_array`
// each route every element through the matching scalar accessor and
// collect, returning None the moment a single element fails to
// coerce or the receiver is not an Array (peeling Ptr-deref /
// Truncated{partial: Array}). The existing suite pins
// `as_u32_array` overflow-reject, `as_u64_array` collect, and two
// `as_bool_array` peel/coerce cases; these fill the remaining arms.

#[test]
fn as_u32_array_collects_in_range_values() {
    // Every element fits u32 → Some(Vec<u32>) preserving order.
    let arr = RenderedValue::Array {
        len: 4,
        elements: vec![uint(0), uint(1), uint(255), uint(u32::MAX as u64)],
    };
    assert_eq!(arr.as_u32_array(), Some(vec![0, 1, 255, u32::MAX]));
}

#[test]
fn as_u32_array_non_array_is_none() {
    // A scalar receiver (not Array / Ptr-deref-Array / Truncated-
    // Array) yields None — array_elements() returns None first.
    assert_eq!(uint(7).as_u32_array(), None);
}

#[test]
fn as_i64_array_collects_signed_and_negative() {
    // Int elements (including a negative) all coerce via as_i64.
    let arr = RenderedValue::Array {
        len: 3,
        elements: vec![
            RenderedValue::Int { bits: 64, value: -5 },
            RenderedValue::Int { bits: 32, value: 0 },
            RenderedValue::Int {
                bits: 64,
                value: i64::MAX,
            },
        ],
    };
    assert_eq!(arr.as_i64_array(), Some(vec![-5, 0, i64::MAX]));
}

#[test]
fn as_i64_array_rejects_uint_above_i64_max() {
    // as_i64 rejects a Uint > i64::MAX (sign-overflow), so the whole
    // array collapses to None — no partial result, no silent wrap.
    let arr = RenderedValue::Array {
        len: 2,
        elements: vec![
            uint(1),
            RenderedValue::Uint {
                bits: 64,
                value: u64::MAX,
            },
        ],
    };
    assert_eq!(arr.as_i64_array(), None);
}

#[test]
fn as_i64_array_rejects_float_element() {
    // A Float element is not coercible via as_i64 → None.
    let arr = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Int { bits: 64, value: 1 },
            RenderedValue::Float {
                bits: 64,
                value: 2.5,
            },
        ],
    };
    assert_eq!(arr.as_i64_array(), None);
}

#[test]
fn as_i64_array_non_array_is_none() {
    assert_eq!(uint(3).as_i64_array(), None);
}

#[test]
fn as_f64_array_collects_mixed_numeric() {
    // Float / Int / Uint / Enum all widen via as_f64.
    let arr = RenderedValue::Array {
        len: 4,
        elements: vec![
            RenderedValue::Float {
                bits: 64,
                value: 1.5,
            },
            RenderedValue::Int { bits: 64, value: -2 },
            uint(3),
            enum_v(32, 4, Some("FOUR"), true),
        ],
    };
    assert_eq!(arr.as_f64_array(), Some(vec![1.5, -2.0, 3.0, 4.0]));
}

#[test]
fn as_f64_array_rejects_bool_element() {
    // as_f64 has no Bool arm (only Float/Int/Uint/Enum), so a Bool
    // element forces None for the whole array.
    let arr = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Float {
                bits: 64,
                value: 1.0,
            },
            RenderedValue::Bool { value: true },
        ],
    };
    assert_eq!(arr.as_f64_array(), None);
}

#[test]
fn as_f64_array_non_array_is_none() {
    assert_eq!(
        RenderedValue::Float {
            bits: 64,
            value: 1.0,
        }
        .as_f64_array(),
        None,
    );
}

#[test]
fn as_bool_array_rejects_float_element() {
    // as_bool accepts Bool/Uint/Int/Char/Enum/Ptr but NOT Float —
    // a Float element collapses as_bool_array to None.
    let arr = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Bool { value: true },
            RenderedValue::Float {
                bits: 64,
                value: 0.0,
            },
        ],
    };
    assert_eq!(arr.as_bool_array(), None);
}

#[test]
fn as_bool_array_coerces_mixed_scalars_to_nonzero_mask() {
    // Uint/Int/Char/Enum elements each coerce to value != 0.
    let arr = RenderedValue::Array {
        len: 4,
        elements: vec![
            uint(0),
            RenderedValue::Int { bits: 32, value: -1 },
            RenderedValue::Char { value: 0 },
            enum_v(32, 7, None, false),
        ],
    };
    assert_eq!(
        arr.as_bool_array(),
        Some(vec![false, true, false, true]),
        "each element coerces via `value != 0`",
    );
}

// ---- is_deeply_zero: direct coverage --------------------------
//
// `is_deeply_zero` recurses into Struct/Array (all-members /
// all-elements deeply-zero, empty qualifies) and matches `is_zero`
// for scalars; Bytes / Truncated / Unsupported are NEVER deeply
// zero (they carry diagnostic content). The existing suite only
// exercises it indirectly through struct Display suppression.

#[test]
fn is_deeply_zero_all_zero_nested_struct_is_true() {
    let v = RenderedValue::Struct {
        type_name: Some("nest".into()),
        members: vec![
            RenderedMember {
                name: "a".into(),
                value: RenderedValue::Uint { bits: 32, value: 0 },
            },
            RenderedMember {
                name: "inner".into(),
                value: RenderedValue::Struct {
                    type_name: None,
                    members: vec![
                        RenderedMember {
                            name: "x".into(),
                            value: RenderedValue::Bool { value: false },
                        },
                        RenderedMember {
                            name: "arr".into(),
                            value: RenderedValue::Array {
                                len: 2,
                                elements: vec![uint(0), uint(0)],
                            },
                        },
                    ],
                },
            },
        ],
    };
    assert!(is_deeply_zero(&v));
}

#[test]
fn is_deeply_zero_one_nonzero_leaf_is_false() {
    let v = RenderedValue::Struct {
        type_name: None,
        members: vec![
            RenderedMember {
                name: "a".into(),
                value: RenderedValue::Uint { bits: 32, value: 0 },
            },
            RenderedMember {
                name: "b".into(),
                // single non-zero leaf deep in the tree
                value: RenderedValue::Array {
                    len: 2,
                    elements: vec![uint(0), uint(9)],
                },
            },
        ],
    };
    assert!(!is_deeply_zero(&v));
}

#[test]
fn is_deeply_zero_empty_aggregates_are_true() {
    // Empty struct / empty array: the `.all()` over zero items is
    // vacuously true.
    assert!(is_deeply_zero(&RenderedValue::Struct {
        type_name: None,
        members: vec![],
    }));
    assert!(is_deeply_zero(&RenderedValue::Array {
        len: 0,
        elements: vec![],
    }));
}

#[test]
fn is_deeply_zero_bytes_truncated_unsupported_never_zero() {
    // These three carry diagnostic content (hex / partial / reason)
    // the consumer must see even with all-zero numeric content.
    assert!(!is_deeply_zero(&RenderedValue::Bytes {
        hex: "00 00".into(),
    }));
    assert!(!is_deeply_zero(&RenderedValue::Truncated {
        needed: 4,
        had: 0,
        partial: Box::new(RenderedValue::Bytes { hex: String::new() }),
    }));
    assert!(!is_deeply_zero(&RenderedValue::Unsupported {
        reason: "x".into(),
    }));
}

// ---- anonymous-overlay union dedup -----------------------------
//
// A BTF union surfaces as anonymous Struct overlays whose scalar
// leaves often duplicate a named sibling's value (the classic
// `tid: struct sdt_id { val: u64 }` overlay). `write_struct`
// builds a sibling-value pool once via `build_sibling_scalar_pool`
// and suppresses each anonymous overlay whose every non-zero leaf
// is already in the pool via `anon_duplicates_pool`. Neither helper
// had a direct test.

#[test]
fn build_sibling_scalar_pool_collects_nonzero_and_descends_one_level() {
    // Named scalar (5) + a zero scalar (skipped) + a single-field
    // struct sibling whose inner scalar (7) is descended into.
    let members = vec![
        RenderedMember {
            name: "weight".into(),
            value: uint(5),
        },
        RenderedMember {
            name: "zero".into(),
            value: uint(0),
        },
        RenderedMember {
            name: "tid".into(),
            value: RenderedValue::Struct {
                type_name: Some("sdt_id".into()),
                members: vec![RenderedMember {
                    name: "val".into(),
                    value: uint(7),
                }],
            },
        },
    ];
    let pool = build_sibling_scalar_pool(&members);
    assert!(pool.contains(&5), "named scalar value collected");
    assert!(pool.contains(&7), "single-level struct descent collected inner scalar");
    assert!(!pool.contains(&0), "zero values are never pooled");
    assert_eq!(pool.len(), 2);
}

#[test]
fn build_sibling_scalar_pool_signed_int_pooled_as_bit_pattern() {
    // scalar_numeric_value reinterprets a signed Int as its u64 bit
    // pattern so an anonymous overlay storing the same wire bits via
    // an unsigned member dedups against it. -1 → 0xFFFF_FFFF_FFFF_FFFF.
    let members = vec![RenderedMember {
        name: "v".into(),
        value: RenderedValue::Int { bits: 64, value: -1 },
    }];
    let pool = build_sibling_scalar_pool(&members);
    assert!(pool.contains(&u64::MAX), "signed -1 pooled as u64::MAX bits");
}

#[test]
fn anon_duplicates_pool_true_when_all_nonzero_leaves_present() {
    // Overlay struct {x=5, y=0}: 5 is in the pool, 0 is the zero
    // half of a wider scalar (skipped) → duplicate, suppressible.
    let mut pool: std::collections::HashSet<u64> = std::collections::HashSet::new();
    pool.insert(5);
    let anon = RenderedValue::Struct {
        type_name: None,
        members: vec![
            RenderedMember {
                name: "x".into(),
                value: uint(5),
            },
            RenderedMember {
                name: "y".into(),
                value: uint(0),
            },
        ],
    };
    assert!(anon_duplicates_pool(&anon, &pool));
}

#[test]
fn anon_duplicates_pool_false_on_unmatched_value() {
    // 9 is not in the pool → the overlay carries unique content and
    // must NOT be suppressed.
    let mut pool: std::collections::HashSet<u64> = std::collections::HashSet::new();
    pool.insert(5);
    let anon = RenderedValue::Struct {
        type_name: None,
        members: vec![RenderedMember {
            name: "x".into(),
            value: uint(9),
        }],
    };
    assert!(!anon_duplicates_pool(&anon, &pool));
}

#[test]
fn anon_duplicates_pool_false_on_compound_submember() {
    // A nested compound sub-member (None from scalar_numeric_value)
    // can't be deduped → false even if the rest matches.
    let mut pool: std::collections::HashSet<u64> = std::collections::HashSet::new();
    pool.insert(5);
    let anon = RenderedValue::Struct {
        type_name: None,
        members: vec![RenderedMember {
            name: "nested".into(),
            value: RenderedValue::Array {
                len: 1,
                elements: vec![uint(5)],
            },
        }],
    };
    assert!(!anon_duplicates_pool(&anon, &pool));
}

#[test]
fn anon_duplicates_pool_false_for_non_struct_or_empty_pool() {
    // Non-Struct anon → false. Empty pool / empty members → false.
    let mut pool: std::collections::HashSet<u64> = std::collections::HashSet::new();
    pool.insert(5);
    // non-Struct
    assert!(!anon_duplicates_pool(&uint(5), &pool));
    // empty members
    let empty_struct = RenderedValue::Struct {
        type_name: None,
        members: vec![],
    };
    assert!(!anon_duplicates_pool(&empty_struct, &pool));
    // empty pool
    let empty_pool: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let anon = RenderedValue::Struct {
        type_name: None,
        members: vec![RenderedMember {
            name: "x".into(),
            value: uint(5),
        }],
    };
    assert!(!anon_duplicates_pool(&anon, &empty_pool));
}

#[test]
fn write_struct_anonymous_overlay_duplicating_sibling_is_suppressed() {
    // A named scalar `pid=42` plus an anonymous (empty-name) union
    // overlay Struct whose only leaf is also 42. The overlay merely
    // re-views the named field, so `write_struct` suppresses it (the
    // documented `anon_duplicates_pool` / `build_sibling_scalar_pool`
    // intent): the named-sibling pool is built BEFORE the flatten pass,
    // so the empty-name Struct is recognized as a pure duplicate and
    // dropped rather than flattened into a duplicate sibling column.
    // Only `pid=42` shows.
    let v = RenderedValue::Struct {
        type_name: Some("scx_task".into()),
        members: vec![
            RenderedMember {
                name: "pid".into(),
                value: uint(42),
            },
            RenderedMember {
                name: String::new(), // anonymous union overlay
                value: RenderedValue::Struct {
                    type_name: None,
                    members: vec![RenderedMember {
                        name: "overlay_alias".into(),
                        value: uint(42),
                    }],
                },
            },
        ],
    };
    let out = format!("{v}");
    assert!(out.contains("pid=42"), "named sibling visible: {out}");
    assert!(
        !out.contains("overlay_alias"),
        "duplicate overlay suppressed, not flattened to a sibling: {out}",
    );
    assert_eq!(out, "scx_task{pid=42}", "exact suppressed form");
}

#[test]
fn write_struct_anonymous_overlay_with_unique_field_is_flattened() {
    // The companion to the suppression case: an anonymous overlay whose
    // leaf does NOT duplicate any named sibling is flattened onto the
    // parent (shown), not suppressed — the dedup drops only overlays
    // that are pure re-views of named fields.
    let v = RenderedValue::Struct {
        type_name: Some("scx_task".into()),
        members: vec![
            RenderedMember {
                name: "pid".into(),
                value: uint(42),
            },
            RenderedMember {
                name: String::new(), // anonymous union overlay
                value: RenderedValue::Struct {
                    type_name: None,
                    members: vec![RenderedMember {
                        name: "unique_field".into(),
                        value: uint(99), // 99 not in the named-sibling pool {42}
                    }],
                },
            },
        ],
    };
    let out = format!("{v}");
    assert_eq!(
        out, "scx_task{pid=42, unique_field=99}",
        "non-duplicate overlay flattened in",
    );
}

// ---- write_struct multi-line column path -----------------------
//
// When the inline form exceeds STRUCT_INLINE_WIDTH_BUDGET (120),
// flat-scalar members pack 3-per-row under the `TypeName:`
// breadcrumb. Two sub-paths: padded (`name = value`, needs >=3
// rows AND >=4-char name-length variance in a column) and compact
// (`name=value`). The existing multi-line test exercises only the
// COMPOUND-member line path; these pin the scalar column grid.
// Exact strings captured from the renderer.

#[test]
fn write_struct_multiline_column_padded_alignment() {
    let mk = |n: &str, v: u64| RenderedMember {
        name: n.into(),
        value: RenderedValue::Uint { bits: 32, value: v },
    };
    // 9 cells (3 rows). Column 0 = {a, loooong0, mid0}: len 1..8 →
    // variance >=4 → padded ` = ` alignment fires for column 0
    // only. Columns 1,2 cluster within 3 chars → bare `=`.
    let v = RenderedValue::Struct {
        type_name: Some("topo".into()),
        members: vec![
            mk("a", 11111111),
            mk("bb", 22222222),
            mk("ccc", 33333333),
            mk("loooong0", 44444444),
            mk("eeee", 55555555),
            mk("ffff", 66666666),
            mk("mid0", 77777777),
            mk("gggg", 88888888),
            mk("hhhh", 99999999),
        ],
    };
    let out = format!("{v}");
    let expected = "topo:\n  \
        a        = 11111111   bb=22222222     ccc=33333333\n  \
        loooong0 = 44444444   eeee=55555555   ffff=66666666\n  \
        mid0     = 77777777   gggg=88888888   hhhh=99999999";
    assert_eq!(out, expected, "padded column grid mismatch");
    // Column 0 uses the padded ` = ` separator; the short-name cell
    // is padded to the column max width (8) before the separator.
    assert!(out.contains("a        = 11111111"));
    assert!(out.contains("loooong0 = 44444444"));
    // Columns 1/2 stay on the compact bare-`=` form (variance < 4).
    assert!(out.contains("bb=22222222"));
    assert!(out.contains("ccc=33333333"));
}

#[test]
fn write_struct_multiline_column_compact_single_row() {
    let mk = |n: &str, v: u64| RenderedMember {
        name: n.into(),
        value: RenderedValue::Uint { bits: 64, value: v },
    };
    // Exactly 3 cells → n_rows = 1 < 3, so pad_eq never fires; the
    // names/values are long enough to push the inline form past 120,
    // forcing the multi-line grid in its compact (bare-`=`) form.
    let v = RenderedValue::Struct {
        type_name: Some("counters".into()),
        members: vec![
            mk("aaaaaaaaaaaaaaaaaaa", 11111111111111111),
            mk("bbbbbbbbbbbbbbbbbbb", 22222222222222222),
            mk("ccccccccccccccccccc", 33333333333333333),
        ],
    };
    let out = format!("{v}");
    let expected = "counters:\n  \
        aaaaaaaaaaaaaaaaaaa=11111111111111111   \
        bbbbbbbbbbbbbbbbbbb=22222222222222222   \
        ccccccccccccccccccc=33333333333333333";
    assert_eq!(out, expected, "compact column grid mismatch");
    // No padded separator anywhere (single row never aligns `=`).
    assert!(
        !out.contains(" = "),
        "single-row multi-line struct must use bare `=`, got: {out}",
    );
    // Breadcrumb form, not inline braces.
    assert!(out.starts_with("counters:\n"));
    assert!(!out.contains('{'), "multi-line must not use inline braces: {out}");
}