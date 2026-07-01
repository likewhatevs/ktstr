//! `Verdict` / `ClaimBuilder` / `SetClaim` / `SeqClaim` and the
//! `claim!` macro. Coverage mirrors what the previous
//! `Expect` / `Checks` tests pinned, expressed in the new
//! claim-based shape: every comparator on every supported type
//! bound (PartialEq, PartialOrd, f64-special, container).

use super::*;
use crate::claim;
use crate::workload::{WorkerReport, WorkerReportClaim};

/// `AssertResult::note` records an [`InfoNote`] on `info_notes`
/// without flipping `passed` or `skipped`, and without touching
/// the `details` failure stream. The structural separation
/// (notes != failures) is the invariant sidecar consumers rely on
/// — iterating `details` counts real failures, iterating
/// `info_notes` surfaces diagnostic context.
#[test]
fn assert_result_note_does_not_flip_passed_or_skipped() {
    let mut r = AssertResult::pass();
    let was_pass = r.is_pass();
    let was_skip = r.is_skip();
    let was_outcomes = r.outcomes.len();
    r.note("observed worker.iterations=12345");
    assert_eq!(r.is_pass(), was_pass);
    assert_eq!(r.is_skip(), was_skip);
    assert_eq!(
        r.outcomes.len(),
        was_outcomes,
        "note must not pollute outcomes"
    );
    assert_eq!(r.info_notes.len(), 1);
    assert!(r.info_notes[0].message.contains("worker.iterations"));

    // Same on a skip-only result.
    let mut r = AssertResult::skip("topo missing");
    r.note("topo had 2 LLCs, test wants 4");
    // Skip is the "could not run" verdict — is_pass is FALSE for skip.
    assert!(!r.is_pass());
    assert!(r.is_skip());
    // skip pushed one Outcome::Skip; note pushed to info_notes.
    assert_eq!(r.outcomes.len(), 1);
    let d = r.skip_details().next().unwrap();
    assert_eq!(d.kind, DetailKind::Skip);
    assert_eq!(r.info_notes.len(), 1);
    assert!(r.info_notes[0].message.contains("LLCs"));
}

/// `AssertResult::with_note` is the builder-style sibling of
/// `note`. Same invariant: never flips verdict, appends to
/// `info_notes` (not `details`), returns the same owned shape.
#[test]
fn assert_result_with_note_preserves_verdict() {
    let r = AssertResult::pass().with_note("max_wchar=6543");
    assert!(r.is_pass());
    assert!(!r.is_skip());
    assert!(r.outcomes.is_empty(), "with_note must not pollute outcomes");
    assert_eq!(r.info_notes.len(), 1);
    assert!(r.info_notes[0].message.contains("max_wchar=6543"));
}

/// `info_notes` survive serde round-trip — sidecar consumers
/// match on the structural `info_notes` field, so the wire format
/// must preserve it across pass/fail/skip variants.
#[test]
fn info_note_survives_serde_roundtrip() {
    let r = AssertResult::pass().with_note("snapshot=disabled");
    let json = serde_json::to_string(&r).unwrap();
    let r2: AssertResult = serde_json::from_str(&json).unwrap();
    assert!(r2.outcomes.is_empty());
    assert_eq!(r2.info_notes.len(), 1);
    assert!(r2.info_notes[0].message.contains("snapshot=disabled"));
}

/// `AssertResult::merge` must extend `info_notes` from BOTH sides.
/// A regression that dropped the extend line (or biased to one side)
/// would silently lose diagnostic context from the merged-in result
/// — sidecar consumers would see only one of the two info-streams
/// that fed the merge.
#[test]
fn merge_extends_info_notes_from_both_sides() {
    let mut a = AssertResult::pass().with_note("note_from_a");
    let b = AssertResult::pass().with_note("note_from_b");
    a.merge(b);
    assert_eq!(
        a.info_notes.len(),
        2,
        "merge must concatenate both sides' notes"
    );
    let messages: Vec<&str> = a.info_notes.iter().map(|n| n.message.as_str()).collect();
    assert!(
        messages.contains(&"note_from_a"),
        "a's note dropped during merge"
    );
    assert!(
        messages.contains(&"note_from_b"),
        "b's note dropped during merge"
    );
}

/// `merge` preserves order: self's notes precede other's notes.
/// `extend(iter)` is the natural ordering — pinning it so a future
/// "sort by phase" refactor doesn't silently reorder.
#[test]
fn merge_preserves_info_notes_order_self_then_other() {
    let mut a = AssertResult::pass().with_note("first");
    let b = AssertResult::pass().with_note("second");
    a.merge(b);
    assert_eq!(a.info_notes[0].message, "first");
    assert_eq!(a.info_notes[1].message, "second");
}

// -- Verdict pointwise-claim API -------------------------------------

#[test]
fn verdict_empty_is_passing() {
    let r = Verdict::new().into_result();
    assert!(r.is_pass());
    assert!(r.outcomes.is_empty());
    assert_eq!(r.stats.total_workers, 0);
}

#[test]
fn verdict_default_matches_new() {
    let d = Verdict::default();
    let n = Verdict::new();
    assert_eq!(d.is_pass(), n.is_pass());
    assert_eq!(d.detail_count(), n.detail_count());
}

#[test]
fn verdict_assert_verdict_attaches_threshold_config() {
    let v = Assert::default_checks().verdict();
    assert!(v.is_pass());
    assert!(v.assert().is_some());

    let v = Verdict::new();
    assert!(v.assert().is_none());
}

/// Pins the lockstep contract documented at `Assert::NO_OVERRIDES`
/// (the `Assert::NO_OVERRIDES` const + `Assert::default_checks` const-fn
/// in `assert::builder`): the const value and the const-fn entry
/// point return the same `Assert`. `default_checks()`'s body is
/// `Self::NO_OVERRIDES`, so the equality is structurally guaranteed
/// today — this test catches a future regression where someone adds a
/// threshold default to `default_checks()` (e.g. flipping
/// `enforce_monitor_thresholds` to true) without realising the const
/// is the documented spread-composition source and must stay all-None.
/// `Assert` doesn't derive `PartialEq` (f64 fields preclude `Eq`); the
/// comparison goes via `Debug`, which prints EVERY field of the struct
/// (including the bool `enforce_monitor_thresholds` that
/// [`Assert::format_human`] doesn't render).
#[test]
fn no_overrides_matches_default_checks() {
    assert_eq!(
        format!("{:?}", Assert::NO_OVERRIDES),
        format!("{:?}", Assert::default_checks()),
        "Assert::default_checks() must remain field-by-field identical to Assert::NO_OVERRIDES",
    );
}

#[test]
fn verdict_passing_into_result_matches_assert_result_pass() {
    let r1 = Verdict::new().into_result();
    let r2 = AssertResult::pass();
    assert_eq!(r1.is_pass(), r2.is_pass());
    assert_eq!(r1.is_skip(), r2.is_skip());
    assert_eq!(r1.outcomes, r2.outcomes);
}

/// `to_result` terminates a FLUENT chain built on a temporary `Verdict`
/// — the one-liner `Verdict::new().claim(..).at_least(..).to_result()`
/// that `into_result(self)` cannot express (the comparators return
/// `&mut Verdict`, so the temporary can't be moved out from under the
/// borrow, E0507). Compiling this test IS the regression pin for that
/// shape; the assertions pin that it folds the recorded claims (a pass
/// and a fail outcome) identically to the bound-local `into_result`.
#[test]
fn verdict_to_result_terminates_fluent_chain() {
    // Fluent one-liner off a temporary: this is the shape that does not
    // compile with into_result. A passing claim.
    let r = Verdict::new()
        .claim("answer", 42u64)
        .at_least(40)
        .to_result();
    assert!(r.is_pass());

    // A failing claim flows through the same by-ref terminal.
    let r = Verdict::new()
        .claim("answer", 42u64)
        .at_most(10)
        .to_result();
    assert!(r.is_fail());

    // to_result (clone from &self) and into_result (move) yield the same
    // folded verdict for an equivalent bound local. Use a FAILING claim so
    // the compared outcome streams are NON-empty ([Fail] == [Fail]), not the
    // empty pass identity (which would compare vec![] == vec![] and pin
    // nothing about how outcomes fold).
    let mut bound = Verdict::new();
    bound.claim("answer", 42u64).at_most(10);
    let via_to = {
        let mut v = Verdict::new();
        v.claim("answer", 42u64).at_most(10);
        v.to_result()
    };
    let bound_outcomes = bound.into_result().outcomes;
    assert!(
        !bound_outcomes.is_empty(),
        "a failing claim must record an outcome so the equality is non-trivial",
    );
    assert_eq!(via_to.outcomes, bound_outcomes);
}

#[test]
fn claim_eq_pass_returns_passing_verdict() {
    let mut v = Verdict::new();
    let answer = 42u64;
    claim!(v, answer).eq(42);
    let r = v.into_result();
    assert!(r.is_pass());
    assert!(
        r.outcomes.is_empty(),
        "passing comparator must not push a detail",
    );
}

#[test]
fn claim_eq_fail_names_subject_and_values() {
    let mut v = Verdict::new();
    let answer = 42u64;
    claim!(v, answer).eq(7);
    let r = v.into_result();
    assert!(r.is_fail());
    assert_eq!(r.outcomes.len(), 1);
    let d = r.failure_details().next().unwrap();
    assert_eq!(d.kind, DetailKind::Other);
    assert!(d.message.contains("answer"), "msg: {}", d.message);
    assert!(d.message.contains("expected"), "msg: {}", d.message);
    assert!(d.message.contains("was"), "msg: {}", d.message);
    assert!(d.message.contains("42"), "msg: {}", d.message);
    assert!(d.message.contains('7'), "msg: {}", d.message);
}

#[test]
fn claim_ne_pass_and_fail() {
    let mut v = Verdict::new();
    let flag_pass = 0u64;
    claim!(v, flag_pass).ne(1);
    assert!(v.is_pass());

    let mut v = Verdict::new();
    let flag_fail = 1u64;
    claim!(v, flag_fail).ne(1);
    let r = v.into_result();
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .next()
            .unwrap()
            .message
            .contains("flag_fail")
    );
    assert!(r.failure_details().next().unwrap().message.contains("!="));
}

#[test]
fn claim_at_least_boundary_is_inclusive() {
    let mut v = Verdict::new();
    let counter = 100u64;
    claim!(v, counter).at_least(100);
    assert!(v.is_pass());

    let mut v = Verdict::new();
    let counter = 99u64;
    claim!(v, counter).at_least(100);
    let r = v.into_result();
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .next()
            .unwrap()
            .message
            .contains("at least 100")
    );
    assert!(
        r.failure_details()
            .next()
            .unwrap()
            .message
            .contains("counter")
    );
}

#[test]
fn claim_at_most_boundary_is_inclusive() {
    let mut v = Verdict::new();
    let counter = 100u64;
    claim!(v, counter).at_most(100);
    assert!(v.is_pass());

    let mut v = Verdict::new();
    let counter = 101u64;
    claim!(v, counter).at_most(100);
    let r = v.into_result();
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .next()
            .unwrap()
            .message
            .contains("at most 100")
    );
}

#[test]
fn claim_lt_strict_upper_bound() {
    let mut v = Verdict::new();
    let x = 99u64;
    claim!(v, x).lt(100);
    assert!(v.is_pass());

    let mut v = Verdict::new();
    let x = 100u64;
    claim!(v, x).lt(100);
    let r = v.into_result();
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .next()
            .unwrap()
            .message
            .contains("less than 100")
    );
}

#[test]
fn claim_gt_strict_lower_bound() {
    let mut v = Verdict::new();
    let x = 101u64;
    claim!(v, x).gt(100);
    assert!(v.is_pass());

    let mut v = Verdict::new();
    let x = 100u64;
    claim!(v, x).gt(100);
    let r = v.into_result();
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .next()
            .unwrap()
            .message
            .contains("greater than 100")
    );
}

#[test]
fn claim_between_inclusive_on_both_ends() {
    let mut v = Verdict::new();
    let lo = 10u64;
    let hi = 20u64;
    let mid = 15u64;
    claim!(v, lo).between(10, 20);
    claim!(v, hi).between(10, 20);
    claim!(v, mid).between(10, 20);
    assert!(v.is_pass());

    let mut v = Verdict::new();
    let below = 9u64;
    claim!(v, below).between(10, 20);
    let r = v.into_result();
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .next()
            .unwrap()
            .message
            .contains("[10, 20]")
    );
}

#[test]
fn claim_between_inverted_interval_fails_with_visible_typo() {
    let mut v = Verdict::new();
    let x = 15u64;
    claim!(v, x).between(20, 10);
    let r = v.into_result();
    assert!(r.is_fail());
    let msg = &*r.failure_details().next().unwrap().message;
    assert!(msg.contains("caller error"), "msg: {msg}");
    assert!(msg.contains("interval inverted"), "msg: {msg}");
    assert!(msg.contains("lo=20"), "msg: {msg}");
    assert!(msg.contains("hi=10"), "msg: {msg}");
}

#[test]
fn claim_kind_override_is_persisted_to_detail() {
    let mut v = Verdict::new();
    let p99 = 5000u64;
    claim!(v, p99).kind(DetailKind::Benchmark).at_most(1000);
    let r = v.into_result();
    assert!(r.is_fail());
    assert_eq!(
        r.failure_details().next().unwrap().kind,
        DetailKind::Benchmark
    );
}

#[test]
fn claim_default_kind_is_other() {
    let mut v = Verdict::new();
    let anything = 1u64;
    claim!(v, anything).eq(2);
    let r = v.into_result();
    assert!(r.is_fail());
    assert_eq!(r.failure_details().next().unwrap().kind, DetailKind::Other);
}

#[test]
fn claim_works_across_concrete_types() {
    let mut v = Verdict::new();
    let counter = 1u64;
    let pid = 42i32;
    let len = 3usize;
    claim!(v, counter).eq(1);
    claim!(v, pid).at_least(0);
    claim!(v, len).between(1, 5);
    assert!(v.is_pass());
}

#[test]
fn claim_is_finite_passes_for_normal_values() {
    for v_val in [0.0_f64, 1.0, -1.0, 1e308, -1e308] {
        let mut v = Verdict::new();
        let x = v_val;
        claim!(v, x).is_finite();
        assert!(v.is_pass(), "{v_val} should be finite");
    }
}

#[test]
fn claim_is_finite_fails_for_nan_and_infinities() {
    for v_val in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut v = Verdict::new();
        let x = v_val;
        claim!(v, x).is_finite();
        let r = v.into_result();
        assert!(!r.is_pass(), "{v_val} must fail is_finite");
        assert!(
            r.failure_details()
                .next()
                .unwrap()
                .message
                .contains("expected finite")
        );
    }
}

#[test]
fn claim_near_inclusive_at_tolerance_boundary() {
    let mut v = Verdict::new();
    let exact = 1.0_f64;
    let on_edge = 1.001_f64;
    claim!(v, exact).near(1.0, 0.001);
    claim!(v, on_edge).near(1.0, 0.001);
    assert!(v.is_pass());

    let mut v = Verdict::new();
    let outside = 1.002_f64;
    claim!(v, outside).near(1.0, 0.001);
    let r = v.into_result();
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .next()
            .unwrap()
            .message
            .contains("near 1")
    );
}

#[test]
fn claim_near_nan_input_fails() {
    let mut v = Verdict::new();
    let nan = f64::NAN;
    claim!(v, nan).near(1.0, 0.5);
    assert!(!v.is_pass());
}

#[test]
fn claim_near_negative_tolerance_is_caller_error() {
    let mut v = Verdict::new();
    let exact = 1.0_f64;
    claim!(v, exact).near(1.0, -0.001);
    let r = v.into_result();
    assert!(
        !r.is_pass(),
        "negative tolerance must surface as a caller-error fail",
    );
    let msg = &*r.failure_details().next().unwrap().message;
    assert!(msg.contains("caller error"), "msg: {msg}");
    assert!(msg.contains("tolerance negative"), "msg: {msg}");
    assert!(msg.contains("-0.001"), "msg: {msg}");
}

#[test]
fn claim_near_handles_infinity_equality() {
    let mut v = Verdict::new();
    let pos_inf = f64::INFINITY;
    let neg_inf = f64::NEG_INFINITY;
    claim!(v, pos_inf).near(f64::INFINITY, 0.001);
    claim!(v, neg_inf).near(f64::NEG_INFINITY, 0.001);
    assert!(
        v.is_pass(),
        "infinity == infinity must pass near() despite NaN diff",
    );

    let mut v = Verdict::new();
    let pos_inf = f64::INFINITY;
    claim!(v, pos_inf).near(f64::NEG_INFINITY, 0.001);
    assert!(!v.is_pass());
}

#[test]
fn verdict_continues_past_failure_and_accumulates_all_details() {
    let mut v = Verdict::new();
    let a = 5u64;
    let b = 200u64;
    let c = 42i32;
    let d = 7u64;
    claim!(v, a).at_least(50); // fail
    claim!(v, b).at_most(100); // fail
    claim!(v, c).eq(42); // pass
    claim!(v, d).between(10, 20); // fail
    let r = v.into_result();
    assert!(r.is_fail());
    assert_eq!(
        r.outcomes.len(),
        3,
        "exactly the 3 failing claims must record details: {:?}",
        r.outcomes,
    );
    assert!(r.failure_details().any(|d| d.message.contains("a:")));
    assert!(r.failure_details().any(|d| d.message.contains("b:")));
    assert!(r.failure_details().any(|d| d.message.contains("d:")));
    assert!(
        !r.failure_details().any(|d| d.message.contains("c:")),
        "passing claim must not push a detail: {:?}",
        r.outcomes,
    );
}

#[test]
fn verdict_per_claim_kind_override_routes_to_detail() {
    let mut v = Verdict::new();
    let p99 = 5000u64;
    let locality = 0.5_f64;
    claim!(v, p99).kind(DetailKind::Benchmark).at_most(1000);
    claim!(v, locality)
        .kind(DetailKind::PageLocality)
        .at_least(0.9);
    let r = v.into_result();
    assert!(r.is_fail());
    let bench = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::Benchmark))
        .expect("Benchmark kind must propagate");
    assert!(bench.message.contains("p99"));
    let loc = r
        .failure_details()
        .find(|d| matches!(d.kind, DetailKind::PageLocality))
        .expect("PageLocality kind must propagate");
    assert!(loc.message.contains("locality"));
}

#[test]
fn verdict_merge_folds_in_external_assert_result() {
    let mut v = Verdict::new();
    let a = 100u64;
    claim!(v, a).at_least(50); // pass

    let mut external = AssertResult::pass();

    external.record_fail(AssertDetail::new(DetailKind::Starved, "tid 7 starved"));
    v.merge(external);

    let b = 5u64;
    claim!(v, b).at_least(50); // fail

    let r = v.into_result();
    assert!(r.is_fail());
    assert_eq!(r.outcomes.len(), 2);
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::Starved))
    );
    assert!(
        r.failure_details()
            .any(|d| matches!(d.kind, DetailKind::Other))
    );
}

#[test]
fn verdict_is_pass_and_detail_count_are_non_consuming_reads() {
    let mut v = Verdict::new();
    assert!(v.is_pass());
    assert_eq!(v.detail_count(), 0);

    let x = 100u64;
    claim!(v, x).at_least(50);
    assert!(v.is_pass());
    assert_eq!(v.detail_count(), 0);

    let y = 5u64;
    claim!(v, y).at_least(50);
    assert!(!v.is_pass());
    assert_eq!(v.detail_count(), 1);

    // Re-read to confirm peeks are non-consuming.
    assert!(!v.is_pass());
    assert_eq!(v.detail_count(), 1);

    let z = 200u64;
    claim!(v, z).at_most(100);
    assert_eq!(v.detail_count(), 2);
}

#[test]
fn claim_against_cgroup_stats_via_derived_accessors() {
    let cg = CgroupStats {
        num_workers: 2,
        num_cpus: 2,
        max_gap_ms: 50,
        total_iterations: 1000,
        ..Default::default()
    };

    let mut v = Verdict::new();
    cg.claim_max_gap_ms(&mut v).at_most(100);
    cg.claim_num_workers(&mut v).between(1, 10);
    cg.claim_total_iterations(&mut v).at_least(100);
    let r = v.into_result();
    assert!(r.is_pass(), "details: {:?}", r.outcomes);

    // Failing claim still names the field.
    let mut v = Verdict::new();
    cg.claim_max_gap_ms(&mut v).at_most(10); // 50 > 10 → fail
    let r = v.into_result();
    assert!(r.is_fail());
    assert!(
        r.failure_details()
            .next()
            .unwrap()
            .message
            .contains("max_gap_ms")
    );
    assert!(
        r.failure_details()
            .next()
            .unwrap()
            .message
            .contains("at most 10")
    );
}

#[test]
fn claim_against_worker_report_via_derived_accessors() {
    let report = WorkerReport {
        tid: 4242,
        work_units: 1_000_000,
        cpu_time_ns: 2_500_000_000,
        wall_time_ns: 5_000_000_000,
        off_cpu_ns: 2_500_000_000,
        migration_count: 3,
        cpus_used: [0, 1].into_iter().collect(),
        migrations: vec![],
        max_gap_ms: 50,
        max_gap_cpu: 0,
        max_gap_at_ms: 1000,
        wake_latencies_ns: vec![100, 200, 300, 400, 500],
        wake_sample_total: 5,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        timer_latencies_ns: vec![],
        timer_sample_total: 0,
        iterations: 1000,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
        affinity_error: None,
        phase_slices: vec![],
        taobench_whole: None,
    };

    let mut v = Verdict::new();
    report.claim_tid(&mut v).eq(4242);
    report.claim_iterations(&mut v).at_least(100);
    report.claim_migration_count(&mut v).at_most(10);
    report.claim_completed(&mut v).eq(true);
    report.claim_cpus_used(&mut v).len_at_most(4);
    report.claim_wake_latencies_ns(&mut v).len_eq(5);
    let r = v.into_result();
    assert!(r.is_pass(), "details: {:?}", r.outcomes);
}

#[test]
fn claim_set_comparators_cover_membership_and_size() {
    let s: BTreeSet<usize> = [1, 2, 3].into_iter().collect();
    let mut v = Verdict::new();
    v.claim_set("s", &s).contains(&1);
    v.claim_set("s", &s).len_eq(3);
    v.claim_set("s", &s).len_at_most(5);
    v.claim_set("s", &s).len_at_least(1);
    v.claim_set("s", &s).nonempty();
    let allowed: BTreeSet<usize> = [1, 2, 3, 4].into_iter().collect();
    v.claim_set("s", &s).subset_of(&allowed);
    let forbidden: BTreeSet<usize> = [10, 11].into_iter().collect();
    v.claim_set("s", &s).disjoint_from(&forbidden);
    assert!(v.is_pass());

    let empty: BTreeSet<usize> = BTreeSet::new();
    let mut v = Verdict::new();
    v.claim_set("s", &empty).empty();
    assert!(v.is_pass());
}

#[test]
fn claim_seq_comparators_cover_membership_and_size() {
    let v_seq: Vec<u64> = vec![10, 20, 30];
    let mut verdict = Verdict::new();
    verdict.claim_seq("seq", &v_seq).contains(&20);
    verdict.claim_seq("seq", &v_seq).len_eq(3);
    verdict.claim_seq("seq", &v_seq).len_at_most(10);
    verdict.claim_seq("seq", &v_seq).len_at_least(1);
    verdict.claim_seq("seq", &v_seq).nonempty();
    assert!(verdict.is_pass());

    let empty: Vec<u64> = vec![];
    let mut verdict = Verdict::new();
    verdict.claim_seq("seq", &empty).empty();
    assert!(verdict.is_pass());
}

#[test]
fn verdict_skip_marks_skipped_without_failing() {
    let mut v = Verdict::new();
    v.skip("topology missing");
    let r = v.into_result();
    // A pure skip terminal verdict is is_skip()=true,
    // is_fail()=false, is_pass()=false: empty `outcomes` is the Pass
    // identity, but a Skip variant in the vec makes the all-Skip
    // stream report as Skip, not Pass.
    assert!(!r.is_fail(), "pure skip must not be a failure");
    assert!(r.is_skip());
    assert!(!r.is_pass(), "skip is not pass");
    assert!(
        r.skip_details()
            .any(|d| matches!(d.kind, DetailKind::Skip) && d.message.contains("topology missing"))
    );
}

/// `Verdict::skip` MUST NOT mask a prior failed claim. A claim that
/// failed produced real evidence — a later skip cannot retroactively
/// erase it. The verdict must surface as failed-and-skipped so gate
/// callers see both the failure and the reason the scenario stopped.
///
/// A prior implementation forced `passed = true` on skip, which let
/// a missing-precondition skip mask a real failure that had already
/// been recorded. That is silent data loss: the test would render as
/// a clean skip and the failure would never surface.
#[test]
fn verdict_skip_preserves_prior_failure() {
    let mut v = Verdict::new();
    claim!(v, 5u64).at_most(3);
    assert!(!v.is_pass(), "prior claim should fail (5 > 3)");
    v.skip("precondition missing");
    let r = v.into_result();
    // Skip-after-fail produces a Fail terminal verdict
    // (any-Fail-dominates), with the skip reason recorded as a
    // separate Outcome::Skip entry. The result is is_fail() = true
    // (failure not masked) AND skip_details() carries the skip
    // detail. is_skip() is false because the outcomes stream contains
    // both a Fail and a Skip (not all-Skip).
    assert!(
        r.is_fail(),
        "prior failure must NOT be masked by a later skip — got is_fail={}",
        r.is_fail(),
    );
    // The skip reason and the prior failure must both appear in the
    // appropriate per-variant streams.
    assert!(
        r.skip_details()
            .any(|d| matches!(d.kind, DetailKind::Skip)
                && d.message.contains("precondition missing")),
        "skip reason must be recorded: {:?}",
        r.outcomes,
    );
    assert!(
        r.failure_details().any(|d| d.message.contains("at most 3")),
        "prior claim failure must be retained: {:?}",
        r.outcomes,
    );
}

#[test]
fn verdict_skip_if_is_conditional() {
    let mut v = Verdict::new();
    v.skip_if(false, "nope");
    assert!(!v.into_result().is_skip());

    let mut v = Verdict::new();
    v.skip_if(true, "yep");
    assert!(v.into_result().is_skip());
}

#[test]
fn verdict_inconclusive_records_outcome_and_blocks_pass() {
    let mut v = Verdict::new();
    v.inconclusive(AssertDetail::new(DetailKind::Migration, "denominator zero"));
    let r = v.into_result();
    assert!(
        !r.is_pass(),
        "Inconclusive must NOT fold to Pass — got is_pass={}",
        r.is_pass(),
    );
    assert!(
        r.is_inconclusive(),
        "single Inconclusive outcome → is_inconclusive(): {:?}",
        r.outcomes,
    );
    assert!(
        r.inconclusive_details()
            .any(|d| d.message.contains("denominator zero")),
        "inconclusive detail must surface via inconclusive_details(): {:?}",
        r.outcomes,
    );
}

#[test]
fn verdict_inconclusive_if_is_conditional() {
    let mut v = Verdict::new();
    v.inconclusive_if(
        false,
        AssertDetail::new(DetailKind::Migration, "would-be inconclusive"),
    );
    assert!(!v.into_result().is_inconclusive());

    let mut v = Verdict::new();
    v.inconclusive_if(
        true,
        AssertDetail::new(DetailKind::Migration, "actually inconclusive"),
    );
    assert!(v.into_result().is_inconclusive());
}

#[test]
fn verdict_inconclusive_does_not_mask_prior_failure() {
    // Fail dominates Inconclusive per the
    // `Fail > Inconclusive > Pass > Skip` lattice — a verdict that
    // recorded a real failure before a later Inconclusive must
    // surface as Fail, not Inconclusive. Pins the symmetric
    // counterpart of `verdict_skip_preserves_prior_failure`.
    let mut v = Verdict::new();
    let counter = 5u64;
    claim!(v, counter).at_most(3); // records Fail
    v.inconclusive(AssertDetail::new(
        DetailKind::Migration,
        "later inconclusive",
    ));
    let r = v.into_result();
    assert!(
        r.is_fail(),
        "Fail dominates Inconclusive — later inconclusive must NOT mask the earlier fail"
    );
    assert!(
        !r.is_inconclusive(),
        "is_inconclusive is false when any Fail is recorded"
    );
    assert!(
        r.failure_details().any(|d| d.message.contains("at most 3")),
        "prior claim failure must be retained: {:?}",
        r.outcomes,
    );
    assert!(
        r.inconclusive_details()
            .any(|d| d.message.contains("later inconclusive")),
        "later inconclusive must still appear in inconclusive_details stream: {:?}",
        r.outcomes,
    );
}

#[test]
fn verdict_note_does_not_affect_verdict() {
    let mut v = Verdict::new();
    v.note("observed counter=12345");
    let r = v.into_result();
    assert!(r.is_pass());
    assert!(!r.is_skip());
    assert!(
        r.outcomes.is_empty(),
        "verdict.note must not pollute details"
    );
    assert_eq!(r.info_notes.len(), 1);
    assert!(r.info_notes[0].message.contains("counter=12345"));
}

#[test]
fn claim_eq_against_nan_follows_ieee_754() {
    let mut v = Verdict::new();
    let nan = f64::NAN;
    claim!(v, nan).eq(f64::NAN);
    assert!(
        !v.is_pass(),
        "NaN == NaN is false per IEEE 754; eq(NaN) must FAIL",
    );

    let mut v = Verdict::new();
    let nan = f64::NAN;
    claim!(v, nan).ne(f64::NAN);
    assert!(
        v.is_pass(),
        "NaN != NaN is true per IEEE 754; ne(NaN) must PASS",
    );

    // The recommended NaN test routes through bool-of-`is_nan`:
    let value = f64::NAN;
    let mut v = Verdict::new();
    let is_nan = value.is_nan();
    claim!(v, is_nan).eq(true);
    assert!(v.is_pass());
}

#[test]
fn claim_because_reason_appears_in_failure_message() {
    let mut v = Verdict::new();
    let counter = 5u64;
    claim!(v, counter)
        .because("scheduler should have produced more events")
        .at_least(50);
    let r = v.into_result();
    assert!(r.is_fail());
    let msg = &*r.failure_details().next().unwrap().message;
    assert!(
        msg.contains("scheduler should have produced more events"),
        "msg: {msg}"
    );
    assert!(msg.contains("counter"), "msg: {msg}");
    assert!(msg.contains("at least 50"), "msg: {msg}");
}

#[test]
fn verdict_clone_carries_state() {
    let mut original = Verdict::new();
    let counter = 5u64;
    claim!(original, counter).at_least(50); // fail → push detail
    let copy = original.clone();
    assert_eq!(original.is_pass(), copy.is_pass());
    assert_eq!(original.detail_count(), copy.detail_count());

    // Mutating one must not affect the other.
    let mut copy = copy;
    let more = 1u64;
    claim!(copy, more).eq(1); // pass — no detail
    assert_eq!(original.detail_count(), 1);
    assert_eq!(copy.detail_count(), 1);

    let yet = 0u64;
    claim!(copy, yet).eq(1); // fail
    assert_eq!(original.detail_count(), 1);
    assert_eq!(copy.detail_count(), 2);
}

#[test]
fn verdict_merge_skipped_does_not_fail_accumulator() {
    let mut v = Verdict::new();
    let counter = 100u64;
    claim!(v, counter).at_least(50); // pass — adds PassDetail to passes, outcomes stays empty
    v.merge(AssertResult::skip("optional probe")); // outcomes: [Skip(detail)]
    let r = v.into_result();
    // Merging a skip onto an accumulator with no failures yields an
    // all-Skip terminal verdict. The "does not fail" intent is
    // preserved: is_fail() stays false — no failure was introduced.
    // A skip is not a pass, it's a skip.
    assert!(!r.is_fail(), "merging a skip must not introduce a failure");
    assert!(r.is_skip(), "post-merge stream is all-Skip");
    assert!(!r.is_pass(), "all-Skip is not pass");
    assert!(
        r.skip_details()
            .any(|d| d.message.contains("optional probe")),
        "skip rationale must reach merged outcomes: {:?}",
        r.outcomes
    );
}

/// `AssertDetail::display_with_kind` renders `[<variant>] <message>`
/// without altering the bare `Display` path. Pins both surfaces so a
/// regression that conflates the two (e.g. injecting the kind prefix
/// into the default formatter and breaking every consumer that
/// expected `format!("{}", d)` to produce just the message) trips
/// here.
#[test]
fn assert_detail_display_with_kind_prefixes_variant_token() {
    let d = AssertDetail::new(DetailKind::Stuck, "tid 7 stuck 1500ms on cpu3");
    assert_eq!(
        d.to_string(),
        "tid 7 stuck 1500ms on cpu3",
        "bare Display must remain message-only",
    );
    assert_eq!(
        d.display_with_kind().to_string(),
        "[Stuck] tid 7 stuck 1500ms on cpu3",
        "display_with_kind must prepend [<variant>]",
    );
}

/// `display_with_kind` rendering uses the Debug form of the
/// variant (e.g. `SchedulerCrashed`), not the snake_case rename
/// from any future `serde::Serialize` impl. Pinning the spelling
/// for a multi-word variant catches a swap from `{:?}` to
/// `{:#?}` (which would line-break) or to a serde-driven
/// renderer (which could rename the token).
#[test]
fn assert_detail_display_with_kind_uses_debug_token_for_multiword_variant() {
    let d = AssertDetail::new(DetailKind::SchedulerCrashed, "scheduler process died");
    assert_eq!(
        d.display_with_kind().to_string(),
        "[SchedulerCrashed] scheduler process died",
    );
}

// -- Verdict::log_passes ---------------------------------------------

/// New verdicts default to log_passes=false, which suppresses the
/// positive-confirmation `tracing::info!` output on the pass path
/// (the format/log cost); it does not make the pass path
/// allocation-free — a `PassDetail` is pushed to `result.passes`
/// on every passing claim regardless of the flag. The default is
/// seeded from the `KTSTR_LOG_PASSES` env var, read once at
/// `Verdict::new` time.
#[test]
fn verdict_log_passes_default_off() {
    // Guard against test pollution: env-var-seeded tests below
    // mutate KTSTR_LOG_PASSES, and nextest serializes tests by
    // default within a binary, but a cross-binary parallel run
    // could leak. Snapshot + restore.
    let saved = std::env::var(crate::KTSTR_LOG_PASSES_ENV).ok();
    // SAFETY: std::env::set_var/remove_var became unsafe in Rust
    // 2024 because the libc environ pointer can race with
    // setenv's realloc across threads. In this binary the risk
    // is absent: KTSTR_LOG_PASSES has exactly one reader
    // (log_passes_default in claim.rs), and nextest's default
    // profile runs each test binary single-threaded and
    // serializes tests within the binary, so no concurrent
    // reader exists during the bracketed mutation. The
    // snapshot+restore pair brackets the mutation so
    // log_passes_default observes the unset state inside the
    // test and the original value is restored before the next
    // test runs.
    unsafe {
        std::env::remove_var(crate::KTSTR_LOG_PASSES_ENV);
    }
    let v = Verdict::new();
    assert!(
        !v.log_passes(),
        "Verdict::new must default to log_passes=false when KTSTR_LOG_PASSES is unset",
    );
    // Restore.
    if let Some(prior) = saved {
        // SAFETY: same as above.
        unsafe {
            std::env::set_var(crate::KTSTR_LOG_PASSES_ENV, prior);
        }
    }
}

/// `with_log_passes` is a chainable setter — flips the flag and
/// returns `Self` so the result-into-claim chain stays fluent.
#[test]
fn verdict_with_log_passes_toggles_flag() {
    let on = Verdict::new().with_log_passes(true);
    assert!(on.log_passes());
    let off = Verdict::new().with_log_passes(true).with_log_passes(false);
    assert!(
        !off.log_passes(),
        "with_log_passes(false) must turn the flag off"
    );
}

/// A passing scalar claim emits a `tracing::info!` event under
/// the `ktstr::assert::claim` target when log_passes is on. The
/// event names the claim subject and the values compared so a
/// `--nocapture` run shows positive confirmation rather than
/// silent acceptance.
#[tracing_test::traced_test]
#[test]
fn verdict_log_passes_emits_event_on_scalar_pass() {
    let mut v = Verdict::new().with_log_passes(true);
    claim!(v, 42u64).at_least(40);
    assert!(
        logs_contain("42 ge 40"),
        "positive-confirmation log must name the value and the comparator token (snake_case per the COMPARATOR_VOCABULARY contract — `ge`, not `>=`)",
    );
    let r = v.into_result();
    assert!(
        r.is_pass(),
        "claim must still pass — log_passes only adds output"
    );
    assert!(r.outcomes.is_empty(), "pass arm must add no details");
}

/// A passing scalar claim emits no `tracing::info!` event when
/// log_passes is off (the default) — the flag suppresses the
/// positive-confirmation tracing event (the format/log cost). It
/// does not avoid the `PassDetail` allocation, which happens on
/// every pass regardless of the flag.
#[tracing_test::traced_test]
#[test]
fn verdict_log_passes_silent_when_off() {
    let mut v = Verdict::new();
    claim!(v, 42u64).at_least(40);
    assert!(
        !logs_contain("42 ge 40"),
        "log_passes=false must suppress the positive-confirmation log",
    );
}

/// A failing claim records a detail and never emits the pass
/// log even when log_passes is on — the pass arm and the fail
/// arm are mutually exclusive at the comparator level.
#[tracing_test::traced_test]
#[test]
fn verdict_log_passes_silent_on_fail_arm() {
    let mut v = Verdict::new().with_log_passes(true);
    claim!(v, 5u64).at_least(40);
    assert!(
        !logs_contain("5 ge 40"),
        "fail arm must NOT emit the positive-confirmation log",
    );
    let r = v.into_result();
    assert!(r.is_fail());
    assert_eq!(r.outcomes.len(), 1);
}

// ---------- into_anyhow_or_log (AssertResult + Verdict wrapper) ----------

#[tracing_test::traced_test]
#[test]
fn into_anyhow_or_log_pass_arm_returns_ok_and_traces_notes() {
    let mut v = Verdict::new();
    v.note("observed cross_frac = 0.42");
    v.note("observed same_frac = 0.58");
    claim!(v, 10u64).at_least(5);
    assert!(
        v.result().is_pass(),
        "setup must produce a pure-pass outcome so this test exercises the pass arm specifically (not the skip arm, which also routes to Ok)",
    );
    v.into_anyhow_or_log().expect("pass arm returns Ok");
    assert!(
        logs_contain("observed cross_frac = 0.42"),
        "first info_note must be traced via tracing::info!",
    );
    assert!(
        logs_contain("observed same_frac = 0.58"),
        "second info_note must be traced via tracing::info!",
    );
}

#[test]
fn into_anyhow_or_log_single_failure_arm_returns_err_with_message() {
    let mut v = Verdict::new();
    claim!(v, 5u64).at_least(40);
    let err = v
        .into_anyhow_or_log()
        .expect_err("single failure surfaces as Err");
    let msg = err.to_string();
    assert!(
        msg.contains("5") && msg.contains("40") && msg.contains("expected at least"),
        "single-failure message passes through verbatim (no count prefix); got {msg}",
    );
    assert!(
        !msg.contains("assertion failures:"),
        "single-failure path should NOT emit the count prefix",
    );
}

#[test]
fn into_anyhow_or_log_multiple_failures_arm_concatenates_every_detail() {
    let mut v = Verdict::new();
    claim!(v, 5u64).at_least(40);
    claim!(v, 100u64).at_most(50);
    claim!(v, 7u64).at_least(8);
    let err = v
        .into_anyhow_or_log()
        .expect_err("multi-failure surfaces as Err");
    let msg = err.to_string();
    assert!(
        msg.starts_with("3 assertion failures:"),
        "multi-failure path must lead with count; got {msg:?}",
    );
    assert!(msg.contains("1."), "must enumerate detail 1");
    assert!(msg.contains("2."), "must enumerate detail 2");
    assert!(msg.contains("3."), "must enumerate detail 3");
    // Each individual claim's diagnostic survives the concatenation.
    for needle in ["40", "50", "8"] {
        assert!(
            msg.contains(needle),
            "every individual failure's message must appear in concatenation; \
             missing '{needle}' in {msg:?}",
        );
    }
}

#[tracing_test::traced_test]
#[test]
fn into_anyhow_or_log_notes_traced_before_bail_on_failure() {
    // The notes-before-bail ordering: even on a failed run the
    // operator sees the observed-value context that led to the
    // failure, alongside (not after) the bail message in the log.
    let mut v = Verdict::new();
    v.note("observed pre-bail context");
    claim!(v, 5u64).at_least(40);
    let _ = v.into_anyhow_or_log();
    assert!(
        logs_contain("observed pre-bail context"),
        "info_notes must surface via tracing::info! BEFORE the bail \
         path returns, so a failing run carries diagnostic context \
         alongside the failure message",
    );
}

#[test]
fn into_anyhow_or_log_via_verdict_wrapper_matches_assert_result_path() {
    // Verdict::into_anyhow_or_log chains into AssertResult — pin
    // they produce identical Err shapes for the same input.
    let mut a = Verdict::new();
    claim!(a, 5u64).at_least(40);
    let err_via_verdict = a.into_anyhow_or_log().expect_err("err");

    let mut b = Verdict::new();
    claim!(b, 5u64).at_least(40);
    let err_via_result = b.into_result().into_anyhow_or_log().expect_err("err");

    assert_eq!(
        err_via_verdict.to_string(),
        err_via_result.to_string(),
        "Verdict wrapper must produce the same Err as the underlying AssertResult path",
    );
}

#[test]
fn into_anyhow_or_log_skip_arm_returns_ok() {
    // Verdict with ONLY a skip outcome — no failures, no passes.
    // skip is not failure; helper returns Ok.
    let mut v = Verdict::new();
    v.result_mut()
        .record_skip("irrelevant for this test config");
    assert!(
        v.result().is_skip(),
        "setup must produce a skip-only outcome so this test exercises the skip arm specifically (not the pass arm, which also routes to Ok)",
    );
    v.into_anyhow_or_log()
        .expect("skip-only verdict returns Ok");
}

#[test]
fn into_anyhow_or_log_single_inconclusive_arm_returns_err_with_inconclusive_preamble() {
    // Pins the CI-gate invariant from `into_anyhow_or_log`'s
    // Inconclusive contract: an Inconclusive-only verdict must NOT
    // route to Ok (which would let a CI gate keying off
    // `is_ok()` treat a zero-denominator ratio as green). A
    // single Inconclusive bails with the `"1 inconclusive verdict:"`
    // preamble — distinct from the `"assertion failures:"` preamble
    // so an operator triaging the log can immediately tell whether
    // claims failed or merely lacked signal to evaluate.
    let mut v = Verdict::new();
    v.inconclusive(AssertDetail::new(
        DetailKind::Migration,
        "denominator zero blocks ratio evaluation",
    ));
    assert!(
        v.result().is_inconclusive(),
        "setup must produce an Inconclusive-only outcome so this test exercises the Inconclusive arm specifically (not the fail arm, which has its own preamble)",
    );
    let err = v
        .into_anyhow_or_log()
        .expect_err("inconclusive-only verdict must surface as Err, not Ok — silent-pass guard");
    let msg = err.to_string();
    assert!(
        msg.starts_with("1 inconclusive verdict:"),
        "single-inconclusive path must lead with the inconclusive-verdict preamble; got {msg:?}",
    );
    assert!(
        !msg.contains("assertion failures:"),
        "inconclusive path must NOT use the failure preamble — preambles must remain distinguishable for triage; got {msg:?}",
    );
    assert!(
        msg.contains("denominator zero blocks ratio evaluation"),
        "inconclusive payload must survive into the bail message; got {msg:?}",
    );
}

#[test]
fn into_anyhow_or_log_multiple_inconclusive_arm_concatenates_every_detail() {
    let mut v = Verdict::new();
    v.inconclusive(AssertDetail::new(DetailKind::Migration, "ratio A: 0/0"));
    v.inconclusive(AssertDetail::new(
        DetailKind::Migration,
        "ratio B: missing phase",
    ));
    v.inconclusive(AssertDetail::new(
        DetailKind::Migration,
        "ratio C: zero baseline",
    ));
    assert!(
        v.result().is_inconclusive(),
        "setup must produce a pure Inconclusive outcome (no failures) so this test exercises the Inconclusive multi-detail arm",
    );
    let err = v
        .into_anyhow_or_log()
        .expect_err("multi-inconclusive surfaces as Err");
    let msg = err.to_string();
    assert!(
        msg.starts_with("3 inconclusive verdicts:"),
        "multi-inconclusive path must lead with count; got {msg:?}",
    );
    assert!(msg.contains("1."), "must enumerate detail 1");
    assert!(msg.contains("2."), "must enumerate detail 2");
    assert!(msg.contains("3."), "must enumerate detail 3");
    for needle in [
        "ratio A: 0/0",
        "ratio B: missing phase",
        "ratio C: zero baseline",
    ] {
        assert!(
            msg.contains(needle),
            "every individual inconclusive's message must appear in concatenation; \
             missing '{needle}' in {msg:?}",
        );
    }
    assert!(
        !msg.contains("assertion failures:"),
        "multi-inconclusive path must not emit the failure preamble",
    );
}

#[test]
fn into_anyhow_or_log_fail_dominates_inconclusive_in_bail_preamble() {
    // Fail > Inconclusive in the merge lattice — pin that the
    // dominance is reflected at the bail surface too: when both
    // outcomes are recorded, the failure preamble wins and the
    // inconclusive bail arm is unreachable. Sister-test to
    // `verdict_inconclusive_does_not_mask_prior_failure`, which pins
    // the same dominance at the is_fail / is_inconclusive level.
    let mut v = Verdict::new();
    claim!(v, 5u64).at_least(40); // records Fail
    v.inconclusive(AssertDetail::new(
        DetailKind::Migration,
        "sibling inconclusive",
    ));
    assert!(
        v.result().is_fail(),
        "setup pre-condition: a Fail recorded alongside an Inconclusive must still report is_fail",
    );
    let err = v
        .into_anyhow_or_log()
        .expect_err("Fail+Inconclusive verdict surfaces as Err");
    let msg = err.to_string();
    assert!(
        msg.contains("expected at least"),
        "failure narrative must surface — Fail dominates Inconclusive at the bail surface; got {msg:?}",
    );
    assert!(
        !msg.contains("inconclusive verdict"),
        "inconclusive preamble must NOT appear when a Fail is also recorded — Fail dominates; got {msg:?}",
    );
}
