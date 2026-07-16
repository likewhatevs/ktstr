//! Integration test for the [`Verdict`] + [`claim!`] surface using
//! real [`WorkerReport`] data.
//!
//! `Verdict` is the primary user-facing pointwise-assertion surface,
//! built via [`Verdict::new`] / [`Assert::verdict`] and finished via
//! [`Verdict::into_result`]. The `#[derive(Claim)]` macro on
//! [`WorkerReport`] generates one `claim_<field>` accessor per public
//! field; the [`claim!`](ktstr::claim) macro covers local bindings and
//! arbitrary expressions. Both routes label the recorded claim via
//! `stringify!` over their input — the label cannot drift independent
//! of the source token a regression in the codegen would surface here.
//!
//! This test runs a real Thread-mode workload host-side (no VM
//! required), then exercises both label sources against the collected
//! `WorkerReport`s. It pins three properties:
//!
//! 1. The typed `claim_<field>` accessors exist for every WorkerReport
//!    field type the user is likely to assert against (scalar, set,
//!    sequence) and route through the Verdict accumulator without
//!    type-check ergonomics issues.
//! 2. The `claim!` macro accepts both local-binding labels (`claim!(v,
//!    iter_total)`) and expression labels
//!    (`claim!(v, total_iters as f64 / wall_secs)`), and the labels
//!    rendered into [`AssertDetail::message`] match `stringify!` of
//!    the input tokens.
//! 3. `Verdict::into_result` produces an `AssertResult` whose
//!    `outcomes` / `is_pass()` / `failure_details()` correctly reflect
//!    the claim outcomes — a
//!    failed verdict surfaces a detail naming the field via the
//!    derive-generated `stringify!(field)` label, not a hand-typed
//!    string.

use ktstr::assert::{Assert, AssertResult, DetailKind, Verdict};
use ktstr::claim;
use ktstr::workload::{
    AffinityIntent, CloneMode, SchedPolicy, WorkType, WorkerReport, WorkerReportClaim,
    WorkloadConfig, WorkloadHandle,
};

/// Spawn 2 Thread-mode SpinWait workers, let them run for 200ms, then
/// collect their `WorkerReport`s. Thread mode runs in-process via
/// `std::thread::spawn`, so this is a pure host-side test — no VM,
/// no `/dev/kvm`, no kernel build. Returns at least one report
/// (panics if the workload returns none, since downstream claims
/// would be vacuous).
fn collect_real_reports() -> Vec<WorkerReport> {
    let config = WorkloadConfig {
        num_workers: 2,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::SpinWait,
        affinity: AffinityIntent::Inherit,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config).expect("Thread-mode SpinWait must spawn");
    handle.start();
    // 200ms is well above the wake-cadence floor for a SpinWait worker
    // (1024-iter checkpoints clear in microseconds), so every report
    // records non-zero wall_time_ns and a non-zero work_units count
    // — the at_least(1) bounds below depend on real progress.
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reports = handle.stop_and_collect();
    assert!(
        !reports.is_empty(),
        "Thread-mode SpinWait must produce at least one WorkerReport",
    );
    reports
}

/// End-to-end Verdict pipeline: spawn workload, collect reports,
/// build Assert with defaults, open Verdict, claim through both the
/// typed `claim_<field>` accessors and the `claim!` macro, finish
/// with `into_result`.
///
/// Pins the contract that the entry point named in the user-facing
/// docs (`use ktstr::prelude::*; let mut v = Assert::default_checks().verdict();`)
/// composes with both label sources against real `WorkerReport` data
/// and produces a passing result with empty details.
#[test]
fn verdict_passing_claims_against_real_worker_report() {
    let reports = collect_real_reports();
    let report = &reports[0];

    // Build the Verdict via `Assert::default_checks().verdict()` —
    // the user-facing entry point referenced in the lib.rs Quick
    // Start. `Assert::default_checks()` carries the threshold
    // layer; `.verdict()` hands it forward so any merge of an
    // `assert_*` result later sees the same thresholds.
    let mut v = Assert::default_checks().verdict();
    assert!(
        v.assert().is_some(),
        "Assert::default_checks().verdict() must attach the threshold layer",
    );

    // Typed scalar accessor. wall_time_ns is u64; the workload ran
    // for 200ms, so wall_time_ns is multiple orders of magnitude
    // above 1ns. The `at_least(1)` floor would still fail on a
    // sentinel report (`wall_time_ns = 0`) — that is the intended
    // discrimination.
    report.claim_wall_time_ns(&mut v).at_least(1);
    // Typed scalar accessor with `eq` — `completed` is bool. Real
    // reports from a graceful stop_and_collect have `completed =
    // true`; sentinel reports have `false`. Pins the graceful-stop
    // path executed.
    report.claim_completed(&mut v).eq(true);
    // Typed set accessor. `cpus_used` is BTreeSet<usize>; SetClaim's
    // `nonempty` covers the "worker actually ran on at least one CPU"
    // baseline.
    report.claim_cpus_used(&mut v).nonempty();

    // Macro on a local binding. The label is `"work_units"` (the
    // binding name) — the failing-test below proves the binding
    // ident, not the field name, is what reaches the detail message.
    let work_units = report.work_units;
    claim!(v, work_units).at_least(1);

    // Macro on an expression. Label is the full token tree from
    // stringify!.
    claim!(v, report.iterations + report.work_units).at_least(1);

    let r = v.into_result();
    assert!(
        r.is_pass(),
        "every claim should pass against a real Thread-mode WorkerReport: {:?}",
        r.outcomes,
    );
    assert!(
        r.outcomes.is_empty(),
        "passing claims must not push details: {:?}",
        r.outcomes,
    );
}

/// Failing claims must label each detail with the `stringify!` token
/// from the call site -- the field name for typed accessors, the
/// source token for the macro. A regression that hard-codes a label
/// or drifts independent of the source surfaces here.
#[test]
fn verdict_failing_claims_label_with_stringify_tokens() {
    let reports = collect_real_reports();
    let report = &reports[0];

    let mut v = Verdict::new();

    // Force a failure on the typed accessor route: ceiling far below
    // any real wall_time_ns (a 200ms run is at least 100µs even on a
    // pathologically slow host). Detail message must contain
    // "wall_time_ns" -- the field name from `stringify!(wall_time_ns)`
    // in the derive-generated method body.
    report.claim_wall_time_ns(&mut v).at_most(0);

    // Force a failure on the macro local-binding route. Label must be
    // "iter_count" -- the binding name, not "iterations" (the field
    // name) -- because the macro stringifies the expression token,
    // not the path it traverses.
    let iter_count = report.iterations;
    claim!(v, iter_count).eq(u64::MAX);

    // Force a failure on the macro expression route. Label must be
    // the entire token tree "report.cpus_used.len()" verbatim, since
    // stringify! preserves all source tokens including dot
    // separators and parentheses.
    claim!(v, report.cpus_used.len()).at_least(usize::MAX);

    let r = v.into_result();
    assert!(
        !r.is_pass(),
        "deliberately-failing claims must produce a failed verdict",
    );
    assert_eq!(
        r.outcomes.len(),
        3,
        "one detail per failed claim; got: {:?}",
        r.outcomes,
    );

    // Every detail records DetailKind::Other (the ClaimBuilder default
    // when no `.kind(...)` was set) and surfaces the comparator's
    // formatted message.
    for d in r.failure_details() {
        assert_eq!(d.kind, DetailKind::Other);
    }

    // Detail 0: typed accessor on the `wall_time_ns` field. The
    // derive-generated method body calls
    // `verdict.claim(stringify!(wall_time_ns), ...)`, so the label
    // is the field name verbatim.
    assert!(
        r.failure_details()
            .next()
            .unwrap()
            .message
            .contains("wall_time_ns"),
        "typed accessor failure must label with the field name from \
         stringify!(field) in the derive expansion: {}",
        r.failure_details().next().unwrap().message,
    );
    assert!(
        r.failure_details()
            .next()
            .unwrap()
            .message
            .contains("at most 0"),
        "at_most failure must render the bound: {}",
        r.failure_details().next().unwrap().message,
    );

    // Detail 1: macro on local binding. Label is the binding ident
    // verbatim -- proves the macro reads the expression token, not
    // the variable's underlying source.
    assert!(
        r.failure_details()
            .nth(1)
            .unwrap()
            .message
            .contains("iter_count"),
        "claim! macro failure on a local binding must label with the \
         binding name from stringify!($value): {}",
        r.failure_details().nth(1).unwrap().message,
    );
    // Negative: the field name `iterations` (which the binding was
    // initialized from) MUST NOT appear -- the label tracks the
    // expression token, not the value's provenance.
    assert!(
        !r.failure_details()
            .nth(1)
            .unwrap()
            .message
            .contains("iterations"),
        "claim! must NOT leak the underlying field name when the \
         caller bound it to a different ident: {}",
        r.failure_details().nth(1).unwrap().message,
    );

    // Detail 2: macro on a multi-token expression. Label preserves
    // every token from the input, including method-call parens.
    assert!(
        r.failure_details()
            .nth(2)
            .unwrap()
            .message
            .contains("report.cpus_used.len()"),
        "claim! on an expression must stringify the full token tree: {}",
        r.failure_details().nth(2).unwrap().message,
    );
}

/// `Verdict` mixes upstream `AssertResult` values via [`Verdict::merge`]
/// alongside pointwise claims. Real production tests fold an
/// `assert_not_stuck` (or similar) result into a verdict that also
/// carries pointwise claims; this test pins that the merge path
/// preserves both passing pointwise records and a failing merged
/// upstream.
#[test]
fn verdict_merges_external_assert_result_into_pointwise_claims() {
    let reports = collect_real_reports();
    let report = &reports[0];

    let mut v = Verdict::new();
    // Passing pointwise claim against a real value.
    report.claim_completed(&mut v).eq(true);

    // Synthesize a failing upstream AssertResult (simulating an
    // `assert_*` returning a failure).
    let mut upstream = AssertResult::pass();

    upstream.record_fail(ktstr::assert::AssertDetail::new(
        DetailKind::Other,
        "synthetic upstream failure".to_string(),
    ));

    v.merge(upstream);

    let r = v.into_result();
    assert!(
        !r.is_pass(),
        "merging a failing upstream must conjoin into the verdict",
    );
    let messages: Vec<&str> = r.failure_details().map(|d| d.message.as_str()).collect();
    assert!(
        messages
            .iter()
            .any(|m| m.contains("synthetic upstream failure")),
        "merged upstream details must survive into the final result: {:?}",
        messages,
    );
}

/// Skip path. `Verdict::skip` records a skip reason and leaves
/// `passed` true (skips are not failures). Real tests use this when a
/// precondition (kernel feature, hardware) is missing; pin that the
/// skip detail kind and reason survive `into_result`.
#[test]
fn verdict_skip_records_skip_kind_with_reason() {
    let reports = collect_real_reports();
    let report = &reports[0];

    let mut v = Verdict::new();
    // Real claim against the report so the verdict has at least one
    // pointwise record alongside the skip.
    report.claim_completed(&mut v).eq(true);
    v.skip("integration test demonstrates skip path");

    let r = v.into_result();
    // skip is not a failure (is_fail() = false) but also not a pass
    // (is_pass() = false on an all-Skip stream — skipped means "didn't
    // run"). is_skip() carries the skip terminal verdict.
    assert!(
        !r.is_fail(),
        "skip must NOT mark the verdict failed: {:?}",
        r.outcomes,
    );
    assert!(r.is_skip(), "skip flag must be set on the result");
    let skip_detail = r
        .skip_details()
        .find(|d| d.kind == DetailKind::Skip)
        .expect("at least one Skip-kind detail must be present");
    assert!(
        skip_detail
            .message
            .contains("integration test demonstrates skip path"),
        "skip detail must carry the supplied reason verbatim: {}",
        skip_detail.message,
    );
}

/// `claim_better` orients "better" by the metric's registry polarity, and a
/// metric with NO polarity (an unregistered / typo'd name) is Inconclusive — never
/// a silent pass. Pins the wiring (polarity-via-`MetricId::def` + `better_outcome`)
/// and the headline guardrail through `claim_better` itself; the doctest only covers
/// the LowerBetter-pass path.
#[test]
fn claim_better_orients_by_polarity_and_quarantines_unregistered_metrics() {
    use ktstr::prelude::BuiltinMetric;

    // LowerBetter (latency): candidate WORSE than baseline -> Fail.
    let mut v = Verdict::new();
    v.claim_better(BuiltinMetric::WakeupP99LatencyUs, 60.0)
        .than(50.0);
    assert!(
        v.into_result().is_fail(),
        "LowerBetter: candidate 60 > baseline 50 is worse -> Fail",
    );

    // HigherBetter (loops_per_cpu_sec): candidate better -> Pass.
    let mut v = Verdict::new();
    v.claim_better(BuiltinMetric::SchbenchLoopsPerCpuSec, 200.0)
        .than(100.0);
    assert!(
        v.into_result().is_pass(),
        "HigherBetter: candidate 200 > baseline 100 is better -> Pass",
    );

    // THE GUARDRAIL: an unregistered / typo'd metric name has no polarity ->
    // Undirected -> Inconclusive, NOT a silent pass.
    let mut v = Verdict::new();
    v.claim_better("definitely_not_a_metric", 1.0).than(2.0);
    let r = v.into_result();
    assert!(
        r.is_inconclusive(),
        "an unregistered metric name has no 'better' direction -> Inconclusive, never a \
         silent pass: {:?}",
        r.outcomes,
    );

    // than_by (fractional margin): an exactly-10%-better LowerBetter candidate passes
    // the 10% bar; a 5%-better one fails it.
    let mut v = Verdict::new();
    v.claim_better(BuiltinMetric::WakeupP99LatencyUs, 90.0)
        .than_by(100.0, 0.10);
    assert!(
        v.into_result().is_pass(),
        "90 is exactly 10% better than 100 -> passes .than_by(., 0.10)",
    );
    let mut v = Verdict::new();
    v.claim_better(BuiltinMetric::WakeupP99LatencyUs, 95.0)
        .than_by(100.0, 0.10);
    assert!(
        v.into_result().is_fail(),
        "95 is only 5% better than 100 -> fails .than_by(., 0.10)",
    );
}

/// `claim_present` fails LOUDLY on an absent (None) metric for ALL comparators
/// (not just the at_least path the macro doctest covers), carries `.because` through
/// the absent-Fail, and on Some is behavior-identical to `claim`. Pins the
/// loud-on-absent guardrail + the modifier carry-through.
#[test]
fn claim_present_fails_loud_on_absent_across_comparators_and_carries_modifiers() {
    use ktstr::claim_present;

    // Absent (None) -> Fail on non-at_least comparators (the absent arm is uniform;
    // pin a representative spread).
    let mut v = Verdict::new();
    let missing: Option<f64> = None;
    claim_present!(v, missing).at_most(100.0);
    assert!(
        v.into_result().is_fail(),
        "None.at_most -> loud Fail, not a vacuous pass",
    );

    let mut v = Verdict::new();
    let missing: Option<f64> = None;
    claim_present!(v, missing).between(1.0, 2.0);
    assert!(v.into_result().is_fail(), "None.between -> loud Fail");

    let mut v = Verdict::new();
    let missing: Option<f64> = None;
    claim_present!(v, missing).is_finite();
    assert!(v.into_result().is_fail(), "None.is_finite -> loud Fail");

    // Some(v) is behavior-identical to claim(): a passing bound passes, a failing
    // bound fails.
    let mut v = Verdict::new();
    let present: Option<f64> = Some(1500.0);
    claim_present!(v, present).at_least(1000.0);
    assert!(
        v.into_result().is_pass(),
        "Some(1500).at_least(1000) -> Pass",
    );

    let mut v = Verdict::new();
    let present: Option<f64> = Some(10.0);
    claim_present!(v, present).at_least(1000.0);
    assert!(v.into_result().is_fail(), "Some(10).at_least(1000) -> Fail");

    // .because(reason) carries through the ABSENT arm into the failure message.
    let mut v = Verdict::new();
    let missing: Option<f64> = None;
    claim_present!(v, missing)
        .because("throughput probe")
        .at_least(1.0);
    let r = v.into_result();
    assert!(r.is_fail(), "None.because(..).at_least -> Fail");
    let msg = &r.failure_details().next().unwrap().message;
    assert!(
        msg.contains("metric absent") && msg.contains("(throughput probe)"),
        "the absent-Fail message must name the absence AND carry the .because reason: {msg}",
    );
}
