//! Wire-format regression pin for the `PassDetail.comparator` vocabulary.
//!
//! Every comparator method in `src/assert/claim.rs` emits a specific
//! canonical token into `PassDetail.comparator`. A future refactor that
//! silently changes the emitted token (e.g. `set_len_eq` → `set_size_eq`
//! in claim.rs) would not break compilation and would not fail any
//! existing test — but it would silently change the wire format consumed
//! by the auto-repro renderer, sidecar parsers, and any tooling
//! that filters `result.passes` by comparator. This file makes that
//! drift loud.
//!
//! Coverage: each of the 23 tokens in [`ktstr::COMPARATOR_VOCABULARY`]
//! is exercised by exactly one passing claim; the test asserts that the
//! resulting `passes[0].comparator` equals the expected token literal.
//! If a token is added to the vocabulary, a matching test must be added
//! here.

use ktstr::prelude::*;
use std::collections::BTreeSet;

fn single_pass(v: Verdict) -> PassDetail {
    let r = v.into_result();
    assert_eq!(
        r.passes.len(),
        1,
        "expected exactly one PassDetail; got {} passes and {} details",
        r.passes.len(),
        r.details.len()
    );
    r.passes.into_iter().next().unwrap()
}

#[test]
fn eq_emits_eq() {
    let mut v = Verdict::new();
    v.claim("x", 42u64).eq(42);
    assert_eq!(single_pass(v).comparator, "eq");
}

#[test]
fn ne_emits_ne() {
    let mut v = Verdict::new();
    v.claim("x", 42u64).ne(0);
    assert_eq!(single_pass(v).comparator, "ne");
}

#[test]
fn at_least_emits_ge() {
    let mut v = Verdict::new();
    v.claim("x", 42u64).at_least(40);
    assert_eq!(single_pass(v).comparator, "ge");
}

#[test]
fn at_most_emits_le() {
    let mut v = Verdict::new();
    v.claim("x", 42u64).at_most(50);
    assert_eq!(single_pass(v).comparator, "le");
}

#[test]
fn lt_emits_lt() {
    let mut v = Verdict::new();
    v.claim("x", 42u64).lt(50);
    assert_eq!(single_pass(v).comparator, "lt");
}

#[test]
fn gt_emits_gt() {
    let mut v = Verdict::new();
    v.claim("x", 42u64).gt(10);
    assert_eq!(single_pass(v).comparator, "gt");
}

#[test]
fn between_emits_in_range() {
    let mut v = Verdict::new();
    v.claim("x", 42u64).between(40, 50);
    assert_eq!(single_pass(v).comparator, "in_range");
}

#[test]
fn is_finite_emits_is_finite() {
    let mut v = Verdict::new();
    v.claim("x", 1.5f64).is_finite();
    assert_eq!(single_pass(v).comparator, "is_finite");
}

#[test]
fn near_emits_near_within() {
    let mut v = Verdict::new();
    v.claim("x", 1.5f64).near(1.4, 0.2);
    assert_eq!(single_pass(v).comparator, "near_within");
}

#[test]
fn set_empty_emits_set_is_empty() {
    let mut v = Verdict::new();
    let s: BTreeSet<u64> = BTreeSet::new();
    v.claim_set("x", &s).empty();
    assert_eq!(single_pass(v).comparator, "set_is_empty");
}

#[test]
fn set_nonempty_emits_set_is_non_empty() {
    let mut v = Verdict::new();
    let s: BTreeSet<u64> = [1u64, 2, 3].into_iter().collect();
    v.claim_set("x", &s).nonempty();
    assert_eq!(single_pass(v).comparator, "set_is_non_empty");
}

#[test]
fn set_contains_emits_set_contains() {
    let mut v = Verdict::new();
    let s: BTreeSet<u64> = [1u64, 2, 3].into_iter().collect();
    v.claim_set("x", &s).contains(&2);
    assert_eq!(single_pass(v).comparator, "set_contains");
}

#[test]
fn set_len_eq_emits_set_len_eq() {
    let mut v = Verdict::new();
    let s: BTreeSet<u64> = [1u64, 2, 3].into_iter().collect();
    v.claim_set("x", &s).len_eq(3);
    assert_eq!(single_pass(v).comparator, "set_len_eq");
}

#[test]
fn set_len_at_most_emits_set_len_le() {
    let mut v = Verdict::new();
    let s: BTreeSet<u64> = [1u64, 2, 3].into_iter().collect();
    v.claim_set("x", &s).len_at_most(5);
    assert_eq!(single_pass(v).comparator, "set_len_le");
}

#[test]
fn set_len_at_least_emits_set_len_ge() {
    let mut v = Verdict::new();
    let s: BTreeSet<u64> = [1u64, 2, 3].into_iter().collect();
    v.claim_set("x", &s).len_at_least(2);
    assert_eq!(single_pass(v).comparator, "set_len_ge");
}

#[test]
fn set_subset_of_emits_subset_of() {
    let mut v = Verdict::new();
    let s: BTreeSet<u64> = [1u64, 2].into_iter().collect();
    let whitelist: BTreeSet<u64> = [1u64, 2, 3, 4].into_iter().collect();
    v.claim_set("x", &s).subset_of(&whitelist);
    assert_eq!(single_pass(v).comparator, "subset_of");
}

#[test]
fn set_disjoint_from_emits_disjoint_from() {
    let mut v = Verdict::new();
    let s: BTreeSet<u64> = [1u64, 2].into_iter().collect();
    let forbidden: BTreeSet<u64> = [10u64, 20].into_iter().collect();
    v.claim_set("x", &s).disjoint_from(&forbidden);
    assert_eq!(single_pass(v).comparator, "disjoint_from");
}

#[test]
fn sequence_empty_emits_sequence_is_empty() {
    let mut v = Verdict::new();
    let s: Vec<u64> = Vec::new();
    v.claim_seq("x", &s).empty();
    assert_eq!(single_pass(v).comparator, "sequence_is_empty");
}

#[test]
fn sequence_nonempty_emits_sequence_is_non_empty() {
    let mut v = Verdict::new();
    let s = vec![1u64, 2, 3];
    v.claim_seq("x", &s).nonempty();
    assert_eq!(single_pass(v).comparator, "sequence_is_non_empty");
}

#[test]
fn sequence_contains_emits_sequence_contains() {
    let mut v = Verdict::new();
    let s = vec![1u64, 2, 3];
    v.claim_seq("x", &s).contains(&2);
    assert_eq!(single_pass(v).comparator, "sequence_contains");
}

#[test]
fn sequence_len_eq_emits_sequence_len_eq() {
    let mut v = Verdict::new();
    let s = vec![1u64, 2, 3];
    v.claim_seq("x", &s).len_eq(3);
    assert_eq!(single_pass(v).comparator, "sequence_len_eq");
}

#[test]
fn sequence_len_at_most_emits_sequence_len_le() {
    let mut v = Verdict::new();
    let s = vec![1u64, 2, 3];
    v.claim_seq("x", &s).len_at_most(5);
    assert_eq!(single_pass(v).comparator, "sequence_len_le");
}

#[test]
fn sequence_len_at_least_emits_sequence_len_ge() {
    let mut v = Verdict::new();
    let s = vec![1u64, 2, 3];
    v.claim_seq("x", &s).len_at_least(2);
    assert_eq!(single_pass(v).comparator, "sequence_len_ge");
}

/// The vocabulary slice must enumerate exactly the 23 tokens above. A
/// new comparator method added to claim.rs requires both (a) a new
/// token in COMPARATOR_VOCABULARY and (b) a new test in this file.
/// This guard catches the (a)-without-(b) case at test time.
#[test]
fn vocabulary_count_matches_test_coverage() {
    assert_eq!(
        COMPARATOR_VOCABULARY.len(),
        23,
        "COMPARATOR_VOCABULARY size changed; add a matching #[test] above and update this count"
    );
}

/// Every token in COMPARATOR_VOCABULARY must be exercised by exactly
/// one #[test] above (see the manual mapping in the test names).
/// This guard checks the membership invariant — every test's expected
/// token is in the vocabulary slice (catches typos in either
/// direction).
#[test]
fn every_expected_token_is_in_vocabulary() {
    let expected_tokens = [
        "eq", "ne", "ge", "le", "lt", "gt", "in_range", "is_finite", "near_within",
        "set_is_empty", "set_is_non_empty", "set_contains", "set_len_eq", "set_len_le",
        "set_len_ge", "subset_of", "disjoint_from", "sequence_is_empty", "sequence_is_non_empty",
        "sequence_contains", "sequence_len_eq", "sequence_len_le", "sequence_len_ge",
    ];
    for tok in expected_tokens {
        assert!(
            COMPARATOR_VOCABULARY.contains(&tok),
            "expected token {tok:?} missing from COMPARATOR_VOCABULARY — vocabulary drift between test and pub const"
        );
    }
    assert_eq!(
        expected_tokens.len(),
        COMPARATOR_VOCABULARY.len(),
        "expected_tokens and COMPARATOR_VOCABULARY differ in size"
    );
}

// =====================================================================
// Coverage-class tests: merge-survival, chain-emits-N,
// sentinel-truncation. Guard against regressions in cross-cutting
// PassDetail handling that the per-comparator tests above don't catch.
// =====================================================================

/// Merging two Verdicts must preserve every PassDetail's comparator
/// — neither side gets dropped or has its comparator rewritten.
/// Catches a refactor that accidentally normalizes comparators
/// during Verdict::merge or filters by some category.
#[test]
fn merge_preserves_comparators_from_both_verdicts() {
    let mut a = Verdict::new();
    a.claim("x", 42u64).eq(42);

    let mut b = Verdict::new();
    b.claim("y", 1.5f64).is_finite();

    let result_b = b.into_result();
    a.merge(result_b);
    let result = a.into_result();

    assert_eq!(result.passes.len(), 2, "both passes must survive merge");
    let comparators: std::collections::HashSet<&str> = result
        .passes
        .iter()
        .map(|p| p.comparator.as_ref())
        .collect();
    assert!(comparators.contains("eq"), "eq from verdict A lost in merge");
    assert!(
        comparators.contains("is_finite"),
        "is_finite from verdict B lost in merge"
    );
}

/// A single Verdict with N chained claims of DIFFERENT comparators
/// emits N PassDetails — one per claim, each with its own
/// comparator. Catches accidental dedup or last-claim-wins logic
/// in the record-pass path.
#[test]
fn chain_emits_one_pass_per_comparator() {
    let mut v = Verdict::new();
    v.claim("a", 42u64).eq(42);
    v.claim("b", 1.5f64).is_finite();
    v.claim("c", 5u64).at_least(1);

    let r = v.into_result();
    assert_eq!(r.passes.len(), 3, "3 chained claims must emit 3 passes");
    assert_eq!(r.passes[0].comparator, "eq");
    assert_eq!(r.passes[1].comparator, "is_finite");
    assert_eq!(r.passes[2].comparator, "ge"); // at_least → "ge"
}

/// Merging two Verdicts whose passes carry the SAME comparator token
/// must NOT dedupe by token-equivalence — both passes survive as
/// distinct records. Pins that merge() is content-blind concat, not
/// dedup-by-comparator.
#[test]
fn merge_preserves_duplicate_comparators() {
    let mut a = Verdict::new();
    a.claim("x", 42u64).eq(42);

    let mut b = Verdict::new();
    b.claim("y", 7u64).eq(7);

    let result_b = b.into_result();
    a.merge(result_b);
    let result = a.into_result();

    assert_eq!(
        result.passes.len(),
        2,
        "merge must preserve both eq passes, not dedupe by token"
    );
    assert_eq!(result.passes[0].comparator, "eq");
    assert_eq!(result.passes[1].comparator, "eq");
    // Names distinguish the two records when comparator can't.
    assert_eq!(result.passes[0].name, "x");
    assert_eq!(result.passes[1].name, "y");
}

/// Empty-side merge (both directions) must be a no-op on the
/// passes vector. Pins the zero-length transit degenerate case.
#[test]
fn merge_with_empty_verdict_is_noop() {
    // non-empty.merge(empty)
    let mut a = Verdict::new();
    a.claim("x", 42u64).eq(42);
    let empty = Verdict::new().into_result();
    a.merge(empty);
    let result = a.into_result();
    assert_eq!(result.passes.len(), 1, "merging empty must not drop the existing pass");
    assert_eq!(result.passes[0].comparator, "eq");

    // empty.merge(non-empty)
    let mut a = Verdict::new();
    let mut b = Verdict::new();
    b.claim("x", 42u64).eq(42);
    a.merge(b.into_result());
    let result = a.into_result();
    assert_eq!(result.passes.len(), 1, "merging into empty must adopt the incoming pass");
    assert_eq!(result.passes[0].comparator, "eq");
}

/// Subsequent pushes BEYOND the cap-th sentinel must be no-ops —
/// the passes vec must stay at MAX_RECORDED_PASSES + 1 entries
/// regardless of how many over-cap claims fire. Catches a
/// regression that re-pushes the sentinel on every over-cap claim
/// (vec would balloon to MAX + over_cap_count).
#[test]
fn truncation_sentinel_idempotent_under_repeated_overflow() {
    let mut v = Verdict::new();
    // Fire MAX_RECORDED_PASSES + 100 claims so the over-cap branch
    // hits 100 times after the sentinel is published.
    for _ in 0..(MAX_RECORDED_PASSES + 100) {
        v.claim("loop", 42u64).eq(42);
    }
    let r = v.into_result();
    assert_eq!(
        r.passes.len(),
        MAX_RECORDED_PASSES + 1,
        "passes vec must stay at cap+1 — additional over-cap pushes must be no-ops, not re-push the sentinel"
    );
    let last = r.passes.last().expect("non-empty after cap-hit");
    assert_eq!(last.name, PASSES_TRUNCATION_SENTINEL_NAME);
    assert_eq!(last.comparator, PASSES_TRUNCATION_SENTINEL_COMPARATOR);
}

/// Sentinel record's `value` field must encode the cap (not be
/// empty and not leak a real claim's value). Pins the
/// `format!("cap={MAX_RECORDED_PASSES}")` convention from
/// record_pass_inner so an auto-repro renderer can grep for
/// the cap-encoded value to detect truncation.
#[test]
fn truncation_sentinel_value_encodes_cap() {
    let mut v = Verdict::new();
    for _ in 0..(MAX_RECORDED_PASSES + 1) {
        v.claim("loop", 42u64).eq(42);
    }
    let r = v.into_result();
    let last = r.passes.last().expect("non-empty after cap-hit");
    assert_eq!(last.name, PASSES_TRUNCATION_SENTINEL_NAME);
    let expected_value = format!("cap={MAX_RECORDED_PASSES}");
    assert_eq!(
        last.value, expected_value,
        "sentinel value must encode the cap so renderers can detect truncation"
    );
    assert!(
        last.expected.is_none(),
        "sentinel uses unary shape — no expected field"
    );
}

/// Exceeding MAX_RECORDED_PASSES must produce a truncation sentinel
/// as the cap-th entry: `name == PASSES_TRUNCATION_SENTINEL_NAME`
/// and `comparator == PASSES_TRUNCATION_SENTINEL_COMPARATOR`
/// ("truncated"). Pins the sentinel record's wire shape against
/// silent refactors that change the cap-hit behavior.
#[test]
fn truncation_sentinel_caps_at_max_recorded_passes() {
    let mut v = Verdict::new();
    // Fire MAX_RECORDED_PASSES+1 claims so the cap-th slot becomes
    // the sentinel + further pushes are no-ops.
    for _ in 0..(MAX_RECORDED_PASSES + 1) {
        v.claim("loop", 42u64).eq(42);
    }
    let r = v.into_result();
    assert_eq!(
        r.passes.len(),
        MAX_RECORDED_PASSES + 1,
        "passes vec grows to cap + 1 sentinel"
    );
    let last = r.passes.last().expect("non-empty after cap-hit");
    assert_eq!(
        last.name, PASSES_TRUNCATION_SENTINEL_NAME,
        "sentinel name pins the wire contract"
    );
    assert_eq!(
        last.comparator, PASSES_TRUNCATION_SENTINEL_COMPARATOR,
        "sentinel comparator is the out-of-vocabulary 'truncated' token"
    );
    // The first MAX_RECORDED_PASSES entries are real claim records.
    assert_eq!(r.passes[0].comparator, "eq");
    assert_eq!(r.passes[MAX_RECORDED_PASSES - 1].comparator, "eq");
}
