use super::*;

// ---- Struct template grouping (try_write_struct_template) -----
//
// Block-style array Display has a "template" optimization for
// arrays of similar structs: when 3+ consecutive single-element
// groups are structs of the same shape with < 8 differing
// fields, they collapse into one `[start-end] struct {}` block
// showing common fields once and varying fields in a per-index
// table. Below the threshold (< 3 structs OR 0 or > 3 varying
// fields), it falls back to the default block render.

#[test]
fn array_of_3_similar_structs_uses_template_block() {
    // Three structs differing only in one field → template
    // collapse: common field shown once, varying field in
    // per-index lines.
    let mk = |x: u64| RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![
            RenderedMember {
                name: "common".into(),
                value: RenderedValue::Uint {
                    bits: 32,
                    value: 100,
                },
            },
            RenderedMember {
                name: "x".into(),
                value: RenderedValue::Uint { bits: 32, value: x },
            },
        ],
    };
    let v = RenderedValue::Array {
        len: 3,
        elements: vec![mk(1), mk(2), mk(3)],
    };
    let out = format!("{v}");
    // Template indicator: `[start-end] TypeName:` breadcrumb form.
    assert!(
        out.contains("[0-2] s:"),
        "must surface template index range header: {out}"
    );
    // Common field shown once with `=` assignment.
    assert!(out.contains("common=100"), "common field once: {out}");
    // Varying field rendered as per-index list — `name:`
    // introduces the list (multiple values per row), distinct
    // from the `name=value` scalar form.
    assert!(out.contains("x: "), "varying field name present: {out}");
    assert!(out.contains("[0]="), "per-index marker for first: {out}");
    assert!(out.contains("[2]="), "per-index marker for last: {out}");
}

#[test]
fn array_of_2_similar_structs_renders_per_element() {
    // Regression: the inline-template path checks
    // `structs.len() < 3` and returns Ok(false) without writing
    // anything. The caller previously discarded that signal
    // (`let _ = try_write_struct_template(...)`) and skipped
    // past both groups, producing an empty `[\n]`. Fix surfaces
    // the false return and falls through to per-element render.
    // The inline form drops the `struct` keyword and uses
    // `Type{f=v}` notation.
    let mk = |x: u64| RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![RenderedMember {
            name: "x".into(),
            value: RenderedValue::Uint { bits: 32, value: x },
        }],
    };
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![mk(1), mk(2)],
    };
    let out = format!("{v}");
    // Template header NOT present (group merge requires >= 3).
    assert!(
        !out.contains("[0-1]"),
        "two-element array must not use template: {out}"
    );
    // Both elements must surface in per-element render; the
    // pre-fix code dropped them entirely.
    assert!(out.contains("[0] s{"), "missing [0]: {out}");
    assert!(out.contains("[1] s{"), "missing [1]: {out}");
    assert!(out.contains("x=1"), "missing x=1: {out}");
    assert!(out.contains("x=2"), "missing x=2: {out}");
}

#[test]
fn array_with_too_many_varying_fields_falls_back() {
    // > 3 varying fields → template not used; falls back to
    // per-element block render. Pre-fix the same `let _ =
    // try_write_struct_template` bug also dropped these three
    // elements silently; assert that they all surface.
    let mk = |a: u64, b: u64, c: u64, d: u64, e: u64| RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![
            RenderedMember {
                name: "a".into(),
                value: RenderedValue::Uint { bits: 32, value: a },
            },
            RenderedMember {
                name: "b".into(),
                value: RenderedValue::Uint { bits: 32, value: b },
            },
            RenderedMember {
                name: "c".into(),
                value: RenderedValue::Uint { bits: 32, value: c },
            },
            RenderedMember {
                name: "d".into(),
                value: RenderedValue::Uint { bits: 32, value: d },
            },
            RenderedMember {
                name: "e".into(),
                value: RenderedValue::Uint { bits: 32, value: e },
            },
        ],
    };
    // All 5 fields differ across the 3 elements → varying.len() > 3.
    let v = RenderedValue::Array {
        len: 3,
        elements: vec![mk(1, 1, 1, 1, 1), mk(2, 2, 2, 2, 2), mk(3, 3, 3, 3, 3)],
    };
    let out = format!("{v}");
    assert!(
        !out.contains("[0-2]"),
        ">3 varying fields must skip template, falls back to per-element: {out}",
    );
    // Per-element fallback must surface all three. Each
    // element renders inline as `[N] s{a=v, b=v, ...}` because
    // the rendered single-line form fits within the inline
    // width budget; per-element prefix `[N] ` is added by the
    // array's per-element block path.
    assert!(out.contains("[0] s{"), "missing [0]: {out}");
    assert!(out.contains("[1] s{"), "missing [1]: {out}");
    assert!(out.contains("[2] s{"), "missing [2]: {out}");
}

#[test]
fn array_of_identical_structs_groups_via_run() {
    // All-identical structs: the leading group walker collapses
    // them into one `[start-end]` group rather than the template
    // (template requires varying fields). Pins that the
    // ConsecutiveSimilar detection routes correctly. The
    // grouped struct itself renders inline so the marker is
    // `[0-2] s{x=5}`.
    let s = RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![RenderedMember {
            name: "x".into(),
            value: RenderedValue::Uint { bits: 32, value: 5 },
        }],
    };
    let v = RenderedValue::Array {
        len: 3,
        elements: vec![s.clone(), s.clone(), s],
    };
    let out = format!("{v}");
    // Identical-element grouping shows `[0-2]` followed by
    // the inline-struct render.
    assert!(out.contains("[0-2] s{"), "must group identical: {out}");
}

#[test]
fn array_inline_sparse_runs() {
    // Inline scalar arrays with gaps render each contiguous
    // non-zero run as `[idx]=value` (single) or
    // `[start..end]={v, v}` (multi-element). Zero gaps are
    // implicit from the run brackets — no `(N zero)` suffix.
    // Use bits:32 to avoid the bits:8 string-detection branch
    // which routes to the C-string render path.
    let v = RenderedValue::Array {
        len: 5,
        elements: vec![
            RenderedValue::Uint { bits: 32, value: 0 },
            RenderedValue::Uint { bits: 32, value: 1 },
            RenderedValue::Uint { bits: 32, value: 0 },
            RenderedValue::Uint { bits: 32, value: 0 },
            RenderedValue::Uint { bits: 32, value: 2 },
        ],
    };
    // Two single-element runs at indices 1 and 4. The gap
    // between them (indices 2, 3) is implicit from the index
    // brackets.
    assert_eq!(format!("{v}"), "[[1]=0x1  [4]=0x2]");
}

#[test]
fn array_inline_all_zero_collapses() {
    // All-zero inline array renders the special "all N zero"
    // collapse marker. Use bits:32 since bits:8 all-zero
    // arrays trip the string-detection NUL-string branch
    // (rendered as `""`) — see the "[all N zero]"
    // short-circuit at the head of the Array Display arm.
    let v = RenderedValue::Array {
        len: 3,
        elements: vec![
            RenderedValue::Uint { bits: 32, value: 0 },
            RenderedValue::Uint { bits: 32, value: 0 },
            RenderedValue::Uint { bits: 32, value: 0 },
        ],
    };
    assert_eq!(format!("{v}"), "[all 3 zero]");
}

#[test]
fn array_block_all_zero_collapses() {
    // Block-style (non-inline) all-zero collapse: when every
    // element is a zero-rendering compound proxy. Since the
    // is_zero check skips compound types, this only triggers
    // when elements are inline scalars wrapped in something
    // else — or, more reliably, when the elements pass the
    // inline check AND are zero. Use Ptr with value=0 (inline,
    // is_zero=true).
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Ptr {
                value: 0,
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
        ],
    };
    let out = format!("{v}");
    // Inline scalars (Ptr passes is_inline_scalar), so this hits
    // the inline branch with all-zero collapse.
    assert!(
        out.contains("all 2 zero"),
        "inline all-zero collapse: {out}"
    );
}

#[test]
fn struct_zero_field_suppression_drops_silently() {
    // Struct Display suppresses zero fields silently — no
    // `(N fields zero)` summary appears anywhere. The
    // operator infers from the rendered (and absent) fields
    // that the rest are zero; an explicit count line adds
    // overhead without insight.
    let v = RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![
            RenderedMember {
                name: "shown".into(),
                value: RenderedValue::Uint { bits: 32, value: 5 },
            },
            RenderedMember {
                name: "zero1".into(),
                value: RenderedValue::Uint { bits: 32, value: 0 },
            },
            RenderedMember {
                name: "zero2".into(),
                value: RenderedValue::Uint { bits: 32, value: 0 },
            },
        ],
    };
    let out = format!("{v}");
    assert!(out.contains("shown=5"), "non-zero field shown: {out}");
    assert!(!out.contains("zero1"), "zero fields suppressed: {out}");
    assert!(
        !out.contains("fields zero"),
        "no `(N fields zero)` summary in any form: {out}",
    );
}

#[test]
fn struct_all_zero_emits_empty_inline_form() {
    // All-zero struct: every field is suppressed by deeply-
    // zero collapse, leaving an empty visible set. The inline
    // form emits a bare brace pair `Type{}` — no count
    // summary. The empty body is self-explanatory.
    let v = RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![
            RenderedMember {
                name: "a".into(),
                value: RenderedValue::Uint { bits: 32, value: 0 },
            },
            RenderedMember {
                name: "b".into(),
                value: RenderedValue::Uint { bits: 32, value: 0 },
            },
        ],
    };
    let out = format!("{v}");
    assert_eq!(
        out, "s{}",
        "all-zero struct collapses to empty inline form: {out}",
    );
}

#[test]
fn struct_bpf_printk_format_strings_collapsed() {
    // Members whose names contain "___fmt" / "____fmt" AND whose
    // values are string-shaped Arrays get suppressed (compile-
    // time constants, not runtime state). With inline rendering
    // active for ≤ 3 visible fields, the summary line
    // "(N bpf_printk format strings)" is dropped — at this
    // density the suppression is implicit.
    let fmt_string_value = RenderedValue::Array {
        len: 3,
        elements: vec![
            RenderedValue::Char { value: b'h' },
            RenderedValue::Char { value: b'i' },
            RenderedValue::Char { value: 0 },
        ],
    };
    let v = RenderedValue::Struct {
        type_name: Some("s".into()),
        members: vec![
            RenderedMember {
                name: "real_field".into(),
                value: RenderedValue::Uint {
                    bits: 32,
                    value: 42,
                },
            },
            RenderedMember {
                name: "ktstr___fmt_blah".into(),
                value: fmt_string_value.clone(),
            },
            RenderedMember {
                name: "____fmt_other".into(),
                value: fmt_string_value,
            },
        ],
    };
    let out = format!("{v}");
    assert!(out.contains("real_field=42"));
    assert!(
        !out.contains("ktstr___fmt_blah"),
        "fmt string suppressed: {out}"
    );
    assert!(
        !out.contains("____fmt_other"),
        "fmt string suppressed: {out}"
    );
}

// ---- Array string-mode rendering -------------------------------
//
// 8-bit Int/Uint/Char arrays containing printable bytes render
// as a quoted string (single-line) or block (multi-line). Pins
// both branches.

#[test]
fn array_renders_as_quoted_string_when_printable() {
    let v = RenderedValue::Array {
        len: 6,
        elements: vec![
            RenderedValue::Char { value: b'h' },
            RenderedValue::Char { value: b'e' },
            RenderedValue::Char { value: b'l' },
            RenderedValue::Char { value: b'l' },
            RenderedValue::Char { value: b'o' },
            RenderedValue::Char { value: 0 },
        ],
    };
    let out = format!("{v}");
    assert_eq!(out, "\"hello\"");
}

#[test]
fn array_renders_multiline_string_with_pipe() {
    // A string with embedded newlines uses the `|` block scalar
    // marker and indents per line.
    let v = RenderedValue::Array {
        len: 8,
        elements: vec![
            RenderedValue::Char { value: b'a' },
            RenderedValue::Char { value: b'\n' },
            RenderedValue::Char { value: b'b' },
            RenderedValue::Char { value: b'\n' },
            RenderedValue::Char { value: b'c' },
            RenderedValue::Char { value: 0 },
            RenderedValue::Char { value: 0 },
            RenderedValue::Char { value: 0 },
        ],
    };
    let out = format!("{v}");
    assert!(
        out.starts_with("|\n"),
        "must start with pipe + newline: {out}"
    );
    assert!(out.contains("a"), "must contain first segment: {out}");
    assert!(out.contains("b"), "must contain second segment: {out}");
}

#[test]
fn write_array_element_uint_wide_renders_hex() {
    // write_array_element formats Uint with bits>=32 as hex,
    // bits<32 stays decimal. Indirect coverage via array
    // Display.
    let v = RenderedValue::Array {
        len: 2,
        elements: vec![
            RenderedValue::Uint {
                bits: 32,
                value: 255,
            },
            RenderedValue::Uint {
                bits: 64,
                value: 0xdead_beef,
            },
        ],
    };
    let out = format!("{v}");
    // 32+ bit Uints render as hex, separated and indexed.
    assert!(out.contains("0xff"), "32-bit uint hex: {out}");
    assert!(out.contains("0xdeadbeef"), "64-bit uint hex: {out}");
}

// ---- Cycle detection in pointer chase --------------------------
//
// A `Type::Ptr` whose deref contains a back-pointer to an
// already-visited address must not recurse through the cycle.
// Without the visited-set check, the renderer recurses until
// `MAX_RENDER_DEPTH` (32) fires, producing a wall of identical
// nested structs in the failure dump. With the check, the
// pointer surfaces a `[cycle]` marker after its hex value and
// stops.

/// Stub MemReader that returns canned bytes for specific arena
/// addresses. Used to construct synthetic cycles in pointer
/// chases. `bytes_by_addr` maps arena address → backing bytes;
/// `arena_range` defines `is_arena_addr` accept set.
struct CycleArenaReader {
    bytes_by_addr: std::collections::HashMap<u64, Vec<u8>>,
    arena_start: u64,
    arena_end: u64,
}
impl MemReader for CycleArenaReader {
    fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
        None
    }
    fn is_arena_addr(&self, addr: u64) -> bool {
        addr >= self.arena_start && addr < self.arena_end
    }
    fn read_arena(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
        let bytes = self.bytes_by_addr.get(&addr)?;
        if bytes.len() < len {
            return None;
        }
        Some(bytes[..len].to_vec())
    }
}

/// Self-pointing cycle: a `struct list_head` whose `next` field
/// points to its own arena address. The renderer must surface
/// the cycle on the inner pointer rather than recursing 32
/// levels deep.
#[test]
fn ptr_cycle_self_pointing_surfaces_cycle_reason() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("list_head") else {
        crate::report::test_skip("BTF missing 'list_head'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'list_head' to empty id list");
        return;
    };
    // Verify list_head is the expected shape (Struct with two
    // pointer fields). Skip if the BTF carries a different
    // type under the same name.
    let Some(ty) = peel_modifiers(&btf, id) else {
        crate::report::test_skip("could not peel list_head modifiers");
        return;
    };
    let Type::Struct(_) = ty else {
        crate::report::test_skip("BTF 'list_head' is not a Struct");
        return;
    };
    let Some(size) = type_size(&btf, &ty) else {
        crate::report::test_skip("list_head size unresolved");
        return;
    };
    // Arena addresses for the cycle.
    const ARENA_START: u64 = 0x10_0000_0000;
    const ARENA_END: u64 = 0x10_0001_0000;
    const NODE_A: u64 = 0x10_0000_1000;
    // Bytes: list_head { next = NODE_A, prev = NODE_A } —
    // both fields point back at this same node. The first
    // ptr-chase visits NODE_A; the recursive render of
    // NODE_A's content sees `next` pointing at NODE_A again
    // and the visited-set check fires.
    let mut node_bytes = vec![0u8; size];
    node_bytes[0..8].copy_from_slice(&NODE_A.to_le_bytes());
    node_bytes[8..16].copy_from_slice(&NODE_A.to_le_bytes());

    let mut bytes_by_addr = std::collections::HashMap::new();
    bytes_by_addr.insert(NODE_A, node_bytes);
    let reader = CycleArenaReader {
        bytes_by_addr,
        arena_start: ARENA_START,
        arena_end: ARENA_END,
    };

    // Wrap NODE_A in a parent ptr buffer (8 bytes pointing at
    // NODE_A) and render against `list_head *`. Find a
    // `list_head *` type id by scanning all types — if the
    // BTF doesn't expose a typed pointer to list_head as a
    // top-level type id, we exercise the cycle through Struct
    // member rendering instead.
    //
    // Simpler alternative: render the Struct directly with
    // bytes containing back-pointers to the arena. The
    // renderer recurses through the `next`/`prev` fields, both
    // arena-typed pointers, and exercises the cycle path.
    // Render the struct with bytes that point its `next` at
    // an arena address whose stored value points back at the
    // same address. Visit NODE_A from inside the rendered
    // struct.

    // Build outer buffer: a struct list_head whose next/prev
    // both point to NODE_A. The renderer will chase NODE_A
    // (visiting it the first time, inserting into visited),
    // recurse into the rendered struct, and on rendering the
    // inner `next` pointer, see NODE_A is already visited.
    let mut outer = vec![0u8; size];
    outer[0..8].copy_from_slice(&NODE_A.to_le_bytes());
    outer[8..16].copy_from_slice(&NODE_A.to_le_bytes());

    let v = render_value_with_mem(&btf, id, &outer, &reader);
    let out = format!("{v}");

    // The output must contain a `[cycle]` marker for at least
    // one pointer in the rendered tree. The exact placement
    // depends on traversal order but the marker must appear.
    assert!(
        out.contains("[cycle]"),
        "rendered output must surface cycle marker for a self-pointing list_head: {out}",
    );
    // The output must NOT recurse 32 levels deep — verify by
    // counting `0x{NODE_A:x}` occurrences. Without cycle
    // detection, the renderer would emit NODE_A's hex many
    // times (once per recursion frame). With detection, the
    // address appears once per pointer site (no duplicate hex
    // inside the `[cycle]` marker, which is now a bare
    // diagnostic with no embedded address). Outer's 2 fields
    // each chase NODE_A once and emit a `[cycle]` for each
    // inner field — total ≤ 6 NODE_A occurrences. Cap at 10
    // to leave a margin without admitting a 32-deep runaway.
    let node_hex = format!("0x{NODE_A:x}");
    let occurrences = out.matches(&node_hex).count();
    assert!(
        occurrences < 10,
        "cycle detection must bound recursion; saw {occurrences} \
         occurrences of {node_hex}: {out}",
    );
}

/// Two-node cycle: NODE_A's `next` → NODE_B; NODE_B's `next` →
/// NODE_A. The renderer chases A, then B, then sees A in the
/// visited set and surfaces the cycle.
#[test]
fn ptr_cycle_two_node_loop_surfaces_cycle_reason() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("list_head") else {
        crate::report::test_skip("BTF missing 'list_head'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'list_head' to empty id list");
        return;
    };
    let Some(ty) = peel_modifiers(&btf, id) else {
        crate::report::test_skip("could not peel list_head modifiers");
        return;
    };
    let Type::Struct(_) = ty else {
        crate::report::test_skip("BTF 'list_head' is not a Struct");
        return;
    };
    let Some(size) = type_size(&btf, &ty) else {
        crate::report::test_skip("list_head size unresolved");
        return;
    };

    const ARENA_START: u64 = 0x10_0000_0000;
    const ARENA_END: u64 = 0x10_0001_0000;
    const NODE_A: u64 = 0x10_0000_1000;
    const NODE_B: u64 = 0x10_0000_2000;

    // NODE_A: next=NODE_B, prev=NODE_B.
    let mut a_bytes = vec![0u8; size];
    a_bytes[0..8].copy_from_slice(&NODE_B.to_le_bytes());
    a_bytes[8..16].copy_from_slice(&NODE_B.to_le_bytes());
    // NODE_B: next=NODE_A, prev=NODE_A.
    let mut b_bytes = vec![0u8; size];
    b_bytes[0..8].copy_from_slice(&NODE_A.to_le_bytes());
    b_bytes[8..16].copy_from_slice(&NODE_A.to_le_bytes());

    let mut bytes_by_addr = std::collections::HashMap::new();
    bytes_by_addr.insert(NODE_A, a_bytes);
    bytes_by_addr.insert(NODE_B, b_bytes);
    let reader = CycleArenaReader {
        bytes_by_addr,
        arena_start: ARENA_START,
        arena_end: ARENA_END,
    };

    // Render starting with bytes matching NODE_A's content.
    let mut outer = vec![0u8; size];
    outer[0..8].copy_from_slice(&NODE_B.to_le_bytes());
    outer[8..16].copy_from_slice(&NODE_B.to_le_bytes());

    let v = render_value_with_mem(&btf, id, &outer, &reader);
    let out = format!("{v}");

    // Must surface the cycle marker.
    assert!(
        out.contains("[cycle]"),
        "two-node cycle must surface cycle marker: {out}",
    );
}

/// `render_value_with_mem` constructs a fresh empty visited set
/// for each call. Two independent renders with the same arena
/// reader must each detect their own cycle independently — a
/// stale visited entry from a prior call must not poison a
/// later one.
#[test]
fn ptr_cycle_visited_set_does_not_leak_across_calls() {
    let Some(btf) = test_btf() else {
        crate::report::test_skip("test_btf returned None");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("list_head") else {
        crate::report::test_skip("BTF missing 'list_head'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'list_head' to empty id list");
        return;
    };
    let Some(ty) = peel_modifiers(&btf, id) else {
        crate::report::test_skip("could not peel list_head modifiers");
        return;
    };
    let Type::Struct(_) = ty else {
        crate::report::test_skip("BTF 'list_head' is not a Struct");
        return;
    };
    let Some(size) = type_size(&btf, &ty) else {
        crate::report::test_skip("list_head size unresolved");
        return;
    };

    const ARENA_START: u64 = 0x10_0000_0000;
    const ARENA_END: u64 = 0x10_0001_0000;
    const NODE_A: u64 = 0x10_0000_1000;

    let mut node_bytes = vec![0u8; size];
    node_bytes[0..8].copy_from_slice(&NODE_A.to_le_bytes());
    node_bytes[8..16].copy_from_slice(&NODE_A.to_le_bytes());

    let mut bytes_by_addr = std::collections::HashMap::new();
    bytes_by_addr.insert(NODE_A, node_bytes);
    let reader = CycleArenaReader {
        bytes_by_addr,
        arena_start: ARENA_START,
        arena_end: ARENA_END,
    };

    let mut outer = vec![0u8; size];
    outer[0..8].copy_from_slice(&NODE_A.to_le_bytes());
    outer[8..16].copy_from_slice(&NODE_A.to_le_bytes());

    // Two back-to-back renders. Each must succeed and surface
    // a cycle marker. A leaking visited set from the first
    // call would prevent the second call from chasing NODE_A
    // at all (it would surface the cycle on the OUTER
    // pointer, not the inner one — wrong semantics).
    let v1 = render_value_with_mem(&btf, id, &outer, &reader);
    let out1 = format!("{v1}");
    assert!(out1.contains("[cycle]"), "call 1 cycle: {out1}");

    let v2 = render_value_with_mem(&btf, id, &outer, &reader);
    let out2 = format!("{v2}");
    assert!(out2.contains("[cycle]"), "call 2 cycle: {out2}");

    // Both renders must produce identical output (visited set
    // cleared between them).
    assert_eq!(out1, out2, "fresh visited set per call: outputs must match",);
}

// ---- cast_annotation_for static-string mapping ------------------
//
// `cast_annotation_for` is the single source of truth for the
// operator-visible cast tag emitted on `RenderedValue::Ptr`. A
// 2x2 match over `(AddrSpace, sdt_alloc_resolved)` returns one of
// four `&'static str` literals; the renderer borrows these via
// `Cow::Borrowed` so the annotation costs zero per-chase
// allocations. Because every cast-recovered `Ptr` consumer
// (Display, JSON serializer, downstream operator tooling) keys
// off these exact bytes, drift in any of the four cells is a
// silent operator-visible behavior change.
//
// This test pins all four mappings directly. It is co-located
// with the Cast intercept section below because the integration
// tests assert the SAME strings via `cast_annotation.as_deref()`
// — when one of those tests fails, this one localises the
// regression to the mapping table itself rather than the chase
// pipeline that calls it.

/// Direct-call coverage of every `(AddrSpace, sdt_alloc_resolved)`
/// pair handled by [`cast_annotation_for`]. Asserts the exact
/// `&'static str` returned for each of the four cells:
///
/// - `(Arena, false)` → `"cast→arena"`
/// - `(Arena, true)`  → `"cast→arena (sdt_alloc)"`
/// - `(Kernel, false)`→ `"cast→kernel"`
/// - `(Kernel, true)` → `"cast→kernel (sdt_alloc)"`
///
/// `AddrSpace` is `Copy` so the same enum value is reused for
/// the two `sdt_alloc_resolved` polarities. The match in
/// `cast_annotation_for` is exhaustive over `AddrSpace`, so a
/// new variant fails compilation here AND in production —
/// keeping the operator-visible tag set in lockstep with the
/// analyzer's address-space taxonomy.
#[test]
fn cast_annotation_for_all_four_cells() {
    assert_eq!(
        cast_annotation_for(AddrSpace::Arena, false),
        "cast→arena",
        "(Arena, false) annotation drift",
    );
    assert_eq!(
        cast_annotation_for(AddrSpace::Arena, true),
        "cast→arena (sdt_alloc)",
        "(Arena, true) annotation drift",
    );
    assert_eq!(
        cast_annotation_for(AddrSpace::Kernel, false),
        "cast→kernel",
        "(Kernel, false) annotation drift",
    );
    assert_eq!(
        cast_annotation_for(AddrSpace::Kernel, true),
        "cast→kernel (sdt_alloc)",
        "(Kernel, true) annotation drift",
    );
}
