use super::*;

// ---------------------------------------------------------------------------
// RenderedValue inherent navigators. These exercise the raw
// Option-returning surface used by tests that drop below SnapshotField
// via .raw() and by the failure-dump renderer pipelines.
// ---------------------------------------------------------------------------

#[test]
fn rendered_value_member_walks_struct() {
    let v = struct_with(vec![("foo", uint(7)), ("bar", uint(11))]);
    assert!(matches!(
        v.member("foo"),
        Some(RenderedValue::Uint { value: 7, .. })
    ));
    assert!(matches!(
        v.member("bar"),
        Some(RenderedValue::Uint { value: 11, .. })
    ));
    assert!(v.member("missing").is_none());
}

#[test]
fn rendered_value_member_returns_none_for_non_struct() {
    assert!(uint(0).member("anything").is_none());
    assert!(
        RenderedValue::Array {
            len: 0,
            elements: vec![],
        }
        .member("anything")
        .is_none()
    );
}

#[test]
fn rendered_value_member_peels_ptr_deref() {
    let target = struct_with(vec![("inner", uint(42))]);
    let ptr = RenderedValue::Ptr {
        value: 0x1000,
        deref: Some(Box::new(target)),
        deref_skipped_reason: None,
        cast_annotation: None,
    };
    let inner = ptr.member("inner").expect("ptr deref walks transparently");
    assert!(matches!(inner, RenderedValue::Uint { value: 42, .. }));
}

#[test]
fn rendered_value_member_peels_truncated_partial() {
    let partial = struct_with(vec![("kept", uint(1))]);
    let trunc = RenderedValue::Truncated {
        needed: 64,
        had: 8,
        partial: Box::new(partial),
    };
    assert!(matches!(
        trunc.member("kept"),
        Some(RenderedValue::Uint { value: 1, .. })
    ));
}

#[test]
fn rendered_value_get_walks_dotted_path() {
    let nested = struct_with(vec![(
        "outer",
        struct_with(vec![("middle", struct_with(vec![("leaf", uint(99))]))]),
    )]);
    assert!(matches!(
        nested.get("outer.middle.leaf"),
        Some(RenderedValue::Uint { value: 99, .. })
    ));
}

#[test]
fn rendered_value_get_empty_path_returns_self() {
    let v = uint(5);
    assert!(matches!(
        v.get(""),
        Some(RenderedValue::Uint { value: 5, .. })
    ));
}

#[test]
fn rendered_value_get_rejects_empty_component() {
    let v = struct_with(vec![("a", uint(1))]);
    assert!(v.get("a..").is_none());
    assert!(v.get("..a").is_none());
}

#[test]
fn rendered_value_index_walks_array_and_peels() {
    let arr = RenderedValue::Array {
        len: 3,
        elements: vec![uint(10), uint(20), uint(30)],
    };
    assert!(matches!(
        arr.index(1),
        Some(RenderedValue::Uint { value: 20, .. })
    ));
    assert!(arr.index(99).is_none());
    let trunc = RenderedValue::Truncated {
        needed: 12,
        had: 4,
        partial: Box::new(arr),
    };
    assert!(matches!(
        trunc.index(2),
        Some(RenderedValue::Uint { value: 30, .. })
    ));
}

#[test]
fn rendered_value_as_u64_accepts_scalar_variants() {
    assert_eq!(uint(7).as_u64(), Some(7));
    assert_eq!(RenderedValue::Int { bits: 32, value: 9 }.as_u64(), Some(9));
    assert_eq!(RenderedValue::Bool { value: true }.as_u64(), Some(1));
    assert_eq!(RenderedValue::Char { value: b'A' }.as_u64(), Some(65));
    assert_eq!(
        RenderedValue::Ptr {
            value: 0xdead_beef,
            deref: None,
            deref_skipped_reason: None,
            cast_annotation: None,
        }
        .as_u64(),
        Some(0xdead_beef)
    );
}

#[test]
fn rendered_value_as_u64_rejects_negative_int_and_aggregate() {
    assert_eq!(
        RenderedValue::Int {
            bits: 32,
            value: -1
        }
        .as_u64(),
        None,
        "negative Int does not silently wrap to u64::MAX"
    );
    assert_eq!(struct_with(vec![]).as_u64(), None);
    assert_eq!(
        RenderedValue::Array {
            len: 0,
            elements: vec![],
        }
        .as_u64(),
        None
    );
}

#[test]
fn rendered_value_as_u64_unsigned_enum_high_bit_round_trips() {
    // Unsigned 64-bit enum whose wire value has the high bit set is
    // stored as i64 = -1 by the renderer (raw bit pattern preserved
    // via `as i64`). as_u64() must reinterpret that bit pattern back
    // to its true unsigned wire value rather than rejecting it.
    let unsigned_max = enum_v(64, -1_i64, None, false);
    assert_eq!(unsigned_max.as_u64(), Some(u64::MAX));

    // The same negative storage on a signed enum is a true negative
    // variant and as_u64() must reject it — silent wrap to u64::MAX
    // is the sign-loss bug we are guarding against.
    let signed_negative = enum_v(64, -1_i64, None, true);
    assert_eq!(signed_negative.as_u64(), None);
}

#[test]
fn rendered_value_as_u64_signed_enum_positive_value() {
    // Positive value on a signed enum coerces cleanly to u64 — the
    // sign-loss rejection only kicks in for actually-negative values.
    // Without this pin a renderer regression that always returned
    // None for is_signed=true (treating all signed enums as
    // un-coercible) would silently pass the high-bit round-trip test
    // above.
    let positive_signed = enum_v(32, 42, None, true);
    assert_eq!(positive_signed.as_u64(), Some(42));
}

/// End-to-end pin that `render_value_inner` propagates the BTF
/// `is_signed` flag from the source type into [`RenderedValue::Enum`].
/// The renderer reads `e.is_signed()` from `btf_rs` and threads it
/// onto the variant — a refactor that hardcoded `is_signed: false`
/// (or `true`) would silently break the sign-loss rejection in
/// [`RenderedValue::as_u64`] and the signed-vs-unsigned Display
/// rendering. The unit-level `enum_v` fixture tests above pin the
/// downstream behaviour for a given flag; this test pins that the
/// flag arrives correctly from BTF in the first place.
///
/// Probes one unsigned kernel enum + one signed kernel enum (with
/// a fallback signed candidate if the primary is absent):
///   - `hrtimer_mode` (include/linux/hrtimer.h:43): all variants
///     non-negative (0..=0x10, HRTIMER_MODE_ABS through
///     HRTIMER_MODE_LAZY_REARM), expected `is_signed = false`.
///   - `perf_event_state` (include/linux/perf_event.h:680): includes
///     `PERF_EVENT_STATE_DEAD = -5`, expected `is_signed = true` so
///     the renderer sign-extends correctly.
///   - Fallback signed probe: `cpuhp_state` (include/linux/cpuhotplug.h:57)
///     declares `CPUHP_INVALID = -1`, present in any kernel BTF with
///     cpu hotplug support (effectively all production kernels).
///
/// Each enum probe is gated on `resolve_ids_by_name` returning a
/// match. The unsigned arm and at least ONE signed arm MUST land,
/// otherwise the test panics — a silent SKIP across both signedness
/// flavours would leave a hardcoded-is_signed regression undetected,
/// defeating the test's purpose.
#[test]
fn render_value_inner_propagates_btf_enum_signedness() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };

    let mut unsigned_probe_ran = false;
    let mut signed_probe_ran = false;

    // ── Unsigned probe: hrtimer_mode ────────────────────────────
    if let Ok(ids) = btf.resolve_ids_by_name("hrtimer_mode")
        && let Some(&id) = ids.first()
    {
        // HRTIMER_MODE_REL = 0x01; 4-byte enum.
        let v = render_value(&btf, id, &[0x01, 0x00, 0x00, 0x00]);
        match v {
            RenderedValue::Enum {
                is_signed, value, ..
            } => {
                assert!(
                    !is_signed,
                    "hrtimer_mode has no negative variants (HRTIMER_MODE_ABS..\
                     HRTIMER_MODE_LAZY_REARM = 0..=0x10 at \
                     include/linux/hrtimer.h:43) — BTF must mark it unsigned, \
                     got is_signed=true (renderer dropped the kind_flag bit)"
                );
                assert_eq!(
                    value, 1,
                    "wire 0x01 (HRTIMER_MODE_REL) must render as 1; got {value}"
                );
                unsigned_probe_ran = true;
            }
            other => panic!(
                "expected RenderedValue::Enum for hrtimer_mode, got {other:?} \
                 (renderer routed away from the Enum arm)"
            ),
        }
    } else {
        crate::report::test_skip("BTF missing 'hrtimer_mode' (unsigned-enum probe)");
    }

    // ── Signed probe: perf_event_state (primary), cpuhp_state (fallback) ──
    let signed_probe_candidates = &[
        // (name, negative-variant value, kernel-source citation).
        (
            "perf_event_state",
            -5_i32,
            "PERF_EVENT_STATE_DEAD = -5 at include/linux/perf_event.h:680",
        ),
        (
            "cpuhp_state",
            -1_i32,
            "CPUHP_INVALID = -1 at include/linux/cpuhotplug.h:57",
        ),
    ];
    for (name, neg_value, cite) in signed_probe_candidates {
        let Ok(ids) = btf.resolve_ids_by_name(name) else {
            continue;
        };
        let Some(&id) = ids.first() else {
            continue;
        };
        let neg_bytes = neg_value.to_le_bytes();
        let v = render_value(&btf, id, &neg_bytes);
        match v {
            RenderedValue::Enum {
                is_signed, value, ..
            } => {
                assert!(
                    is_signed,
                    "{name} has a negative variant ({cite}) — BTF must mark it \
                     signed, got is_signed=false (renderer dropped the kind_flag \
                     bit)"
                );
                assert_eq!(
                    value, *neg_value as i64,
                    "wire {neg_value:#x} (truncated to i32) must sign-extend to \
                     {neg_value}; got {value} (renderer dropped the sign \
                     extension on a 4-byte signed enum)"
                );
                signed_probe_ran = true;
                break;
            }
            other => panic!(
                "expected RenderedValue::Enum for {name}, got {other:?} (renderer \
                 routed away from the Enum arm)"
            ),
        }
    }
    if !signed_probe_ran {
        crate::report::test_skip(
            "no signed-enum probe present in BTF: tried perf_event_state \
             (CONFIG_PERF_EVENTS=y typical) and cpuhp_state (CPU hotplug — \
             present in any production kernel BTF)",
        );
    }

    // A silent SKIP across BOTH signedness flavours would leave a
    // hardcoded-is_signed regression undetected — the entire point
    // of this test is to catch a renderer that always returns false
    // or always returns true. Require at least the unsigned arm AND
    // at least one signed arm to land, OR fail hard. The granular
    // `test_skip` lines above keep the diagnostic informative when
    // only the fallback fires; this final gate rejects a wholly-
    // SKIPped run.
    assert!(
        unsigned_probe_ran && signed_probe_ran,
        "BTF coverage gap: unsigned_probe_ran={unsigned_probe_ran}, \
         signed_probe_ran={signed_probe_ran}. Both signedness paths must be \
         exercised to catch a hardcoded-is_signed regression. Use a kernel BTF \
         that includes hrtimer_mode AND either perf_event_state or cpuhp_state."
    );
}

#[test]
fn rendered_value_as_u32_array_rejects_overflow() {
    let arr = RenderedValue::Array {
        len: 2,
        elements: vec![uint(u32::MAX as u64), uint(u32::MAX as u64 + 1)],
    };
    assert_eq!(
        arr.as_u32_array(),
        None,
        "element exceeding u32::MAX errors rather than truncating"
    );
}

#[test]
fn rendered_value_as_u64_array_collects_struct_array() {
    let arr = RenderedValue::Array {
        len: 3,
        elements: vec![uint(1), uint(2), uint(3)],
    };
    assert_eq!(arr.as_u64_array(), Some(vec![1, 2, 3]));
}

#[test]
fn rendered_value_as_bool_array_via_truncated_peel() {
    let arr = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Bool { value: true },
            RenderedValue::Bool { value: false },
        ],
    };
    let trunc = RenderedValue::Truncated {
        needed: 8,
        had: 2,
        partial: Box::new(arr),
    };
    assert_eq!(trunc.as_bool_array(), Some(vec![true, false]));
}
