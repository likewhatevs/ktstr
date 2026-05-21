//! `AssertResult::note_value` / `NoteValue` plus the
//! `any_of` / `all_of` short-circuit constructors. The note_value
//! tests pin the From-impl routing per scalar type, the
//! independent-buffer invariant against `details`, the merge
//! union with last-write-wins on key collision, and the wire
//! format's `skip_serializing_if = "is_empty"` softness.

use super::*;

/// Each `From` impl on [`NoteValue`] routes to the matching enum
/// variant. Pin every variant so a regression that swapped two
/// `From` arms (e.g. `i64` mistakenly producing `NoteValue::Uint`)
/// trips here, not at the consumer's run-time mismatch.
#[test]
fn note_value_from_impls_route_to_correct_variant() {
    assert_eq!(NoteValue::from(42i64), NoteValue::Int(42));
    assert_eq!(NoteValue::from(42u64), NoteValue::Uint(42));
    assert_eq!(NoteValue::from(0.5_f64), NoteValue::Float(0.5));
    assert_eq!(NoteValue::from(true), NoteValue::Bool(true));
    assert_eq!(
        NoteValue::from("hello".to_string()),
        NoteValue::Text("hello".to_string()),
    );
    assert_eq!(
        NoteValue::from("borrowed"),
        NoteValue::Text("borrowed".to_string()),
    );
}

/// `note_value` writes into [`AssertResult::measurements`] without
/// altering the verdict. Distinct from [`Self::note`] (which
/// pushes a [`crate::assert::InfoNote`] onto
/// [`AssertResult::info_notes`]) — a producer commonly calls
/// BOTH and they occupy independent buffers.
#[test]
fn note_value_records_without_altering_verdict() {
    let mut r = AssertResult::pass();
    let was_pass = r.is_pass();
    let was_skip = r.is_skip();
    let was_outcomes = r.outcomes.len();
    r.note_value("max_wchar", 12345i64);
    r.note_value("psi_available", true);
    assert_eq!(r.is_pass(), was_pass);
    assert_eq!(r.is_skip(), was_skip);
    assert_eq!(r.outcomes.len(), was_outcomes);
    assert_eq!(r.measurements.len(), 2);
    assert_eq!(r.measurements["max_wchar"], NoteValue::Int(12345));
    assert_eq!(r.measurements["psi_available"], NoteValue::Bool(true));
}

/// Duplicate-key write overwrites: producers that re-record under
/// the same key (a producer bug, but well-defined) get the latest
/// value. Pin this so a future "first write wins" refactor surfaces
/// here.
#[test]
fn note_value_overwrites_on_duplicate_key() {
    let mut r = AssertResult::pass();
    r.note_value("counter", 1i64);
    r.note_value("counter", 2i64);
    assert_eq!(r.measurements["counter"], NoteValue::Int(2));
    assert_eq!(r.measurements.len(), 1);
}

/// `merge` folds `other.measurements` into `self.measurements`
/// with last-write-wins on key collision (matching `note_value`).
/// Pins the shape so a regression that union-with-keep-first
/// (e.g. `entry.or_insert(v)`) trips here.
#[test]
fn merge_unions_measurements_last_write_wins() {
    let mut a = AssertResult::pass();
    a.note_value("a_only", 1i64);
    a.note_value("shared", 100i64);

    let mut b = AssertResult::pass();
    b.note_value("b_only", 2i64);
    b.note_value("shared", 200i64);

    a.merge(b);
    assert_eq!(a.measurements.len(), 3);
    assert_eq!(a.measurements["a_only"], NoteValue::Int(1));
    assert_eq!(a.measurements["b_only"], NoteValue::Int(2));
    assert_eq!(
        a.measurements["shared"],
        NoteValue::Int(200),
        "merge must adopt other's value on key collision (last write wins)",
    );
}

/// `measurements` survives serde round-trip with the
/// externally-tagged default representation flowing into the right
/// variant on deserialize. Pins the wire-format invariant so a
/// regression that re-adds `#[serde(untagged)]` — which postcard
/// cannot decode (returns `WontImplement`) — trips here at test
/// time rather than as a silent data drop at runtime. See
/// `assert_result_postcard_roundtrip` in `tests_serde.rs` for the
/// postcard-side pin.
#[test]
fn note_value_survives_serde_roundtrip() {
    let mut r = AssertResult::pass();
    r.note_value("answer", 42i64);
    r.note_value("ratio", 0.5_f64);
    r.note_value("name", "fio");
    let json = serde_json::to_string(&r).unwrap();
    assert!(
        json.contains("\"measurements\""),
        "measurements key must appear in JSON when populated: {json}",
    );
    let r2: AssertResult = serde_json::from_str(&json).unwrap();
    assert_eq!(r2.measurements["answer"], NoteValue::Int(42));
    assert_eq!(r2.measurements["ratio"], NoteValue::Float(0.5));
    assert_eq!(r2.measurements["name"], NoteValue::Text("fio".to_string()));
}

/// Empty `measurements` is present in the wire format as `{}`.
/// `skip_serializing_if` was removed because AssertResult is
/// serialized with postcard (positional) — skipping a field on
/// serialize misaligns the deserializer. `#[serde(default)]`
/// handles old sidecars that lack the key.
#[test]
fn empty_measurements_present_in_wire_format() {
    let r = AssertResult::pass();
    let json = serde_json::to_string(&r).unwrap();
    assert!(
        json.contains("\"measurements\":{}"),
        "empty measurements must be present in JSON for postcard compat: {json}",
    );
}

// -- AssertResult::any_of / AssertResult::all_of --------------------

/// `any_of` with at least one passing branch returns a passing
/// result and annotates which branch was chosen. Pin: failed-branch
/// details are dropped (they would only confuse the operator with
/// messages from not-taken paths).
#[test]
fn any_of_chooses_passing_branch() {
    let r = AssertResult::any_of([
        {
            let mut a = AssertResult::pass();
            a.record_fail(AssertDetail::new(DetailKind::Other, "boom"));
            a
        },
        AssertResult::pass(),
    ]);
    assert!(r.is_pass());
    // The "boom" detail from the failed branch must NOT appear —
    // the chosen branch's outcomes prevail.
    assert!(
        !r.failure_details().any(|d| d.message.contains("boom")),
        "failed-branch details must be dropped: {:?}",
        r.outcomes,
    );
    // The chosen-branch annotation MUST appear in info_notes with
    // branch index 1 — any_of moves its "branch N satisfied"
    // annotation to the structural info-notes stream, never the
    // failure stream.
    assert!(
        r.info_notes
            .iter()
            .any(|n| n.message.contains("any_of: branch 1 satisfied")),
        "chosen-branch annotation missing: {:?}",
        r.info_notes,
    );
}

/// `any_of` with all branches failing must prefix EVERY branch's
/// `info_notes` with the same `any_of[<idx>]:` stamp as `details`.
/// Symmetric structural treatment — the auto-repro renderer attributes
/// notes to the branch that emitted them. A regression that dropped
/// the info_notes prefix loop would silently strip attribution.
#[test]
fn any_of_all_fail_prefixes_info_notes_with_branch_index() {
    let r = AssertResult::any_of([
        {
            let mut a = AssertResult::fail(AssertDetail::new(DetailKind::Other, "boom_0"));
            a.note("context_from_branch_0");
            a
        },
        {
            let mut b = AssertResult::fail(AssertDetail::new(DetailKind::Other, "boom_1"));
            b.note("context_from_branch_1");
            b
        },
    ]);
    assert!(r.is_fail());
    let messages: Vec<&str> = r.info_notes.iter().map(|n| n.message.as_str()).collect();
    assert!(
        messages.contains(&"any_of[0]: context_from_branch_0"),
        "branch 0 note must carry index prefix: {:?}",
        r.info_notes,
    );
    assert!(
        messages.contains(&"any_of[1]: context_from_branch_1"),
        "branch 1 note must carry index prefix: {:?}",
        r.info_notes,
    );
}

/// `any_of` with at least one passing branch unions info_notes from
/// every passing branch (each prefix-stamped with `any_of[<idx>]:`)
/// plus the bare arbiter "branch N satisfied" annotation. Failed-
/// branch info_notes are dropped (matching failed-branch details
/// policy). Pinned so a regression that drops the pass-path union
/// loop or the prefix stamping silently loses operator-visible
/// provenance.
#[test]
fn any_of_pass_path_unions_passing_branch_info_notes_with_prefix() {
    let r = AssertResult::any_of([
        {
            let mut a = AssertResult::pass();
            a.note("context_from_branch_0");
            a
        },
        {
            let mut b = AssertResult::pass();
            b.note("context_from_branch_1");
            b
        },
        {
            let mut c = AssertResult::fail(AssertDetail::new(DetailKind::Other, "boom"));
            c.note("context_from_branch_2_failed");
            c
        },
    ]);
    assert!(r.is_pass());
    let messages: Vec<&str> = r.info_notes.iter().map(|n| n.message.as_str()).collect();
    // Branch 0 + branch 1 notes survive, both prefix-stamped.
    assert!(
        messages.contains(&"any_of[0]: context_from_branch_0"),
        "branch 0 passing note must survive with prefix: {messages:?}"
    );
    assert!(
        messages.contains(&"any_of[1]: context_from_branch_1"),
        "branch 1 passing note must survive with prefix: {messages:?}"
    );
    // Failed-branch note is dropped (matches details policy).
    assert!(
        !messages.iter().any(|m| m.contains("branch_2_failed")),
        "failed-branch note must be dropped: {messages:?}"
    );
    // Bare arbiter annotation appended last (no prefix).
    assert!(
        messages.contains(&"any_of: branch 0 satisfied the disjunction"),
        "bare arbiter annotation missing: {messages:?}"
    );
}

/// `any_of` with all branches failing returns a failing result and
/// concatenates every branch's details under a `any_of[<idx>]:`
/// prefix so the operator can identify which branch produced
/// which failure.
#[test]
fn any_of_concatenates_branch_failures_with_index_prefixes() {
    let r = AssertResult::any_of([
        AssertResult::fail(AssertDetail::new(DetailKind::Other, "first boom")),
        AssertResult::fail(AssertDetail::new(DetailKind::Other, "second boom")),
    ]);
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| d.message == "any_of[0]: first boom"),
        "branch 0 detail must carry index prefix: {:?}",
        r.outcomes,
    );
    assert!(
        r.failure_details()
            .any(|d| d.message == "any_of[1]: second boom"),
        "branch 1 detail must carry index prefix: {:?}",
        r.outcomes,
    );
    // A summary line names the per-disposition counts (failed /
    // inconclusive / skipped) so the operator sees the shape of
    // the disjunction, not just the failure count.
    assert!(
        r.failure_details()
            .any(|d| d.message.contains("2 failed") && d.message.contains("of 2 branches")),
        "summary line missing per-disposition counts: {:?}",
        r.outcomes,
    );
}

/// `any_of` with empty input fails — an empty disjunction is
/// logically false. Pinned to surface a producer bug as a
/// nameable failure rather than a vacuous pass.
#[test]
fn any_of_empty_input_fails() {
    let r = AssertResult::any_of(std::iter::empty());
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .any(|d| d.message.contains("empty branch list")),
        "empty disjunction must surface as named failure: {:?}",
        r.outcomes,
    );
}

/// `any_of` with all branches Inconclusive yields an Inconclusive
/// verdict (not Fail and not Pass). Pins the discriminant-aware
/// synthesis at the end of `any_of` — without this, an all-
/// Inconclusive disjunction would synthesize `Outcome::Fail` and
/// promote unevaluated branches to a real failure, lying to CI.
#[test]
fn any_of_all_inconclusive_branches_yields_inconclusive() {
    let mut b0 = AssertResult::pass();
    b0.record_inconclusive(AssertDetail::new(
        DetailKind::Migration,
        "zero denom branch 0",
    ));
    let mut b1 = AssertResult::pass();
    b1.record_inconclusive(AssertDetail::new(
        DetailKind::Benchmark,
        "zero denom branch 1",
    ));
    let r = AssertResult::any_of([b0, b1]);
    assert!(
        r.is_inconclusive(),
        "all-Inconclusive disjunction must yield Inconclusive, not Fail or Pass: {:?}",
        r.outcomes,
    );
    assert!(!r.is_fail(), "must not promote Inconclusive to Fail");
    assert!(
        !r.is_pass(),
        "must not silently pass on unevaluated branches"
    );
    // The synthesized summary line names the per-disposition counts.
    assert!(
        r.inconclusive_details()
            .any(|d| d.message.contains("2 inconclusive") && d.message.contains("of 2 branches")),
        "summary line missing inconclusive count: {:?}",
        r.outcomes,
    );
    // Every branch's payload is prefixed with `any_of[N]:` so the
    // operator can trace which branch was inconclusive.
    let inconc_messages: Vec<&str> = r
        .inconclusive_details()
        .map(|d| d.message.as_str())
        .collect();
    assert!(
        inconc_messages
            .iter()
            .any(|m| m.contains("any_of[0]: zero denom branch 0")),
        "branch 0 inconclusive missing prefix: {inconc_messages:?}",
    );
    assert!(
        inconc_messages
            .iter()
            .any(|m| m.contains("any_of[1]: zero denom branch 1")),
        "branch 1 inconclusive missing prefix: {inconc_messages:?}",
    );
}

/// `any_of` with Fail-plus-Inconclusive branches yields Fail
/// (real failure dominates an unevaluated check). Pins the
/// `if n_fail > 0` short-circuit in the synthesis — a regression
/// to "any Inconclusive demotes to Inconclusive" would silently
/// hide real failures behind a not-evaluated label.
#[test]
fn any_of_fail_plus_inconclusive_yields_fail() {
    let failing = AssertResult::fail(AssertDetail::new(DetailKind::Other, "real boom"));
    let mut inconc = AssertResult::pass();
    inconc.record_inconclusive(AssertDetail::new(DetailKind::Migration, "zero denom"));
    let r = AssertResult::any_of([failing, inconc]);
    assert!(
        r.is_fail(),
        "Fail+Inconclusive disjunction must yield Fail: {:?}",
        r.outcomes,
    );
    assert!(!r.is_inconclusive());
    // Summary names both counts (1 fail + 1 inconclusive of 2).
    assert!(
        r.failure_details()
            .any(|d| d.message.contains("1 failed") && d.message.contains("1 inconclusive")),
        "summary line missing per-disposition counts: {:?}",
        r.outcomes,
    );
}

/// `any_of` with Inconclusive-plus-Skip branches (no Fail, no
/// Pass) yields Inconclusive — Inconclusive dominates Skip per
/// `Fail > Inconclusive > Pass > Skip`. Pins that the synthesis
/// chain falls through to the `n_inc > 0` arm rather than the
/// all-Skip arm when any branch is Inconclusive.
#[test]
fn any_of_inconclusive_plus_skip_yields_inconclusive() {
    let mut inconc = AssertResult::pass();
    inconc.record_inconclusive(AssertDetail::new(DetailKind::Migration, "zero denom"));
    let skip = AssertResult::skip("topology missing");
    let r = AssertResult::any_of([inconc, skip]);
    assert!(
        r.is_inconclusive(),
        "Inconclusive+Skip disjunction must yield Inconclusive (Inconclusive dominates Skip): {:?}",
        r.outcomes,
    );
    assert!(!r.is_skip());
    assert!(!r.is_pass());
}

/// `all_of` is conjunction: passes iff every branch passes.
/// Empty input yields the passing identity (matches
/// `Iterator::all` semantics).
#[test]
fn all_of_passes_when_every_branch_passes() {
    let r = AssertResult::all_of([AssertResult::pass(), AssertResult::pass()]);
    assert!(r.is_pass());

    // One failing branch flips the verdict.
    let r = AssertResult::all_of([
        AssertResult::pass(),
        AssertResult::fail(AssertDetail::new(DetailKind::Other, "boom")),
    ]);
    assert!(r.is_fail());

    // Empty input is the passing identity.
    let r = AssertResult::all_of(std::iter::empty());
    assert!(r.is_pass());
    assert!(r.outcomes.is_empty());
}

/// `Verdict::note_value` mirrors [`AssertResult::note_value`] —
/// records under `measurements` without altering the verdict.
#[test]
fn verdict_note_value_records_into_underlying_result() {
    let mut v = Verdict::new();
    v.note_value("max_wchar", 12345i64);
    v.note_value("psi_available", false);
    let r = v.into_result();
    assert!(r.is_pass());
    assert_eq!(r.measurements.len(), 2);
    assert_eq!(r.measurements["max_wchar"], NoteValue::Int(12345));
    assert_eq!(r.measurements["psi_available"], NoteValue::Bool(false));
}
