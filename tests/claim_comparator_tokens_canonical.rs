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
