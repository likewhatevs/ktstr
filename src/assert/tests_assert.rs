//! `Assert` struct tests: `NO_OVERRIDES` / `default_checks`,
//! per-field merge precedence, builder setters, the
//! `worker_plan` / `monitor_thresholds` extractors,
//! `has_worker_checks` discriminator, and the
//! `gap_threshold_ms` debug-vs-release helper.

use super::tests_common::rpt;
use super::*;

#[test]
fn assert_no_overrides_has_no_checks() {
    let v = Assert::NO_OVERRIDES;
    assert!(v.not_starved.is_none());
    assert!(v.isolation.is_none());
    assert!(v.max_gap_ms.is_none());
    assert!(v.max_spread_pct.is_none());
    assert!(v.max_imbalance_ratio.is_none());
}

#[test]
fn assert_default_checks_is_no_overrides() {
    let v = Assert::default_checks();
    assert!(v.not_starved.is_none());
    assert!(v.isolation.is_none());
    assert!(v.max_imbalance_ratio.is_none());
    assert!(v.max_local_dsq_depth.is_none());
    assert!(v.fail_on_stall.is_none());
    assert!(v.sustained_samples.is_none());
    assert!(v.max_fallback_rate.is_none());
    assert!(v.max_keep_last_rate.is_none());
}

/// Wire-format pin: `Assert::with_monitor_defaults()` MUST set
/// `enforce_monitor_thresholds = true`. The propagation chain is:
/// builder.with_monitor_defaults() → enforce_monitor_thresholds=true
/// → monitor_thresholds().enforce=true → MonitorThresholds.enforce=true
/// → evaluate() respects enforce.
///
/// Per `Assert::with_monitor_defaults` doc: opt into pass/fail
/// enforcement for monitor thresholds. Without this call, monitor
/// violations are reported in the verdict's details but do not
/// fail the test. With it, any monitor threshold violation fails
/// the test.
///
/// A regression that breaks the propagation (e.g. forgets to set
/// enforce_monitor_thresholds in the auto-fill helper, OR changes
/// the field name without updating the threshold extractor) would
/// silently disable enforcement for every test using the builder
/// pattern.
#[test]
fn with_monitor_defaults_propagates_enforce_true_to_thresholds() {
    let assert = Assert::NO_OVERRIDES.with_monitor_defaults();
    let t = assert.monitor_thresholds();
    assert!(
        t.enforce,
        "with_monitor_defaults() must propagate enforce=true to MonitorThresholds"
    );
}

/// Wire-format pin: without `.with_monitor_defaults()` the builder
/// produces enforce=false. This is the post-enforce-default-flip
/// report-only default that broke 19 tests when the default flipped
/// — pin both polarities so a flip in either direction is caught.
#[test]
fn assert_default_without_monitor_defaults_yields_enforce_false() {
    let assert = Assert::NO_OVERRIDES;
    let t = assert.monitor_thresholds();
    assert!(
        !t.enforce,
        "default Assert (no .with_monitor_defaults()) must yield enforce=false"
    );
}

/// Direct-field pin: `Assert::NO_OVERRIDES.enforce_monitor_thresholds`
/// MUST be `false`. Parallel to the `monitor_thresholds()` canary above —
/// catches a regression where the helper is preserved but the underlying
/// field default flips. NO_OVERRIDES is the runtime composition base
/// (see the runtime composition documented on `Assert::merge`), so a
/// flipped default here would silently enable enforcement for every
/// scheduler that doesn't explicitly opt out.
#[test]
fn no_overrides_enforce_monitor_thresholds_field_is_false() {
    let v = Assert::NO_OVERRIDES;
    assert!(
        !v.enforce_monitor_thresholds,
        "NO_OVERRIDES.enforce_monitor_thresholds must be false (post-enforce-default-flip default)"
    );
}

/// Merge semantics pin: `enforce_monitor_thresholds` uses sticky-OR
/// (see the OR expression in `Assert::merge`:
/// `self.enforce_monitor_thresholds || other.enforce_monitor_thresholds`).
/// Distinct from every other field which uses last-Some-wins. The
/// OR rule means a child Assert that opted into enforcement cannot
/// be silently downgraded by merging over (or under) a non-enforcing
/// peer.
///
/// This canary covers the false||false=false case — the baseline that
/// guarantees a merge of two NO_OVERRIDES (the runtime no-op composition)
/// stays in report-only mode.
#[test]
fn merge_enforce_monitor_thresholds_or_semantics_false_false() {
    let merged = Assert::NO_OVERRIDES.merge(&Assert::NO_OVERRIDES);
    assert!(
        !merged.enforce_monitor_thresholds,
        "false || false must yield false"
    );
}

/// Merge semantics pin: true (self) || false (other) → true.
/// A test-level `Assert::NO_OVERRIDES.with_monitor_defaults()` on the
/// left merged with a NO_OVERRIDES override on the right keeps
/// enforce=true. Without OR-semantics, a regression could silently
/// flip back to last-Some-wins and drop enforcement on every test
/// that uses a non-enforcing override layer.
#[test]
fn merge_enforce_monitor_thresholds_or_semantics_true_left() {
    let left = Assert::NO_OVERRIDES.with_monitor_defaults();
    let right = Assert::NO_OVERRIDES;
    let merged = left.merge(&right);
    assert!(
        merged.enforce_monitor_thresholds,
        "true (self) || false (other) must yield true (sticky OR)"
    );
}

/// Merge semantics pin: false (self) || true (other) → true.
/// Symmetric to the previous canary — a non-enforcing base layer
/// merged with an enforcing override layer must yield enforce=true.
/// Both directions matter because the runtime composes via
/// `default_checks().merge(&scheduler.assert).merge(&test.assert)`
/// (see `Assert::merge`), so the enforcing assert can appear on either
/// side depending on which layer opted in.
#[test]
fn merge_enforce_monitor_thresholds_or_semantics_false_true() {
    let left = Assert::NO_OVERRIDES;
    let right = Assert::NO_OVERRIDES.with_monitor_defaults();
    let merged = left.merge(&right);
    assert!(
        merged.enforce_monitor_thresholds,
        "false (self) || true (other) must yield true (sticky OR)"
    );
}

/// Behavior pin: `with_monitor_defaults()` fills ALL six unset
/// monitor-threshold Option fields with `MonitorThresholds::new()`
/// values, not just `enforce_monitor_thresholds`. Documented on
/// `Assert::with_monitor_defaults` ("Also populates any unset
/// monitor-threshold field with the canonical default"). Without
/// this canary, a regression that drops the per-field auto-fill
/// loop in `with_monitor_defaults` would silently let enforced
/// tests run against `None` fields that
/// downstream `monitor_thresholds()` would still default — same end
/// behavior today, but a future API consumer reading the raw
/// `Assert` fields would see misleading `None`s.
#[test]
fn with_monitor_defaults_fills_all_unset_threshold_fields() {
    use crate::monitor::MonitorThresholds;
    let assert = Assert::NO_OVERRIDES.with_monitor_defaults();
    let d = MonitorThresholds::new();
    assert!(
        (assert.max_imbalance_ratio.unwrap() - d.max_imbalance_ratio).abs() < f64::EPSILON,
        "max_imbalance_ratio must be auto-filled with DEFAULT"
    );
    assert_eq!(
        assert.max_local_dsq_depth,
        Some(d.max_local_dsq_depth),
        "max_local_dsq_depth must be auto-filled with DEFAULT"
    );
    assert_eq!(
        assert.fail_on_stall,
        Some(d.fail_on_stall),
        "fail_on_stall must be auto-filled with DEFAULT"
    );
    assert_eq!(
        assert.sustained_samples,
        Some(d.sustained_samples),
        "sustained_samples must be auto-filled with DEFAULT"
    );
    assert!(
        (assert.max_fallback_rate.unwrap() - d.max_fallback_rate).abs() < f64::EPSILON,
        "max_fallback_rate must be auto-filled with DEFAULT"
    );
    assert!(
        (assert.max_keep_last_rate.unwrap() - d.max_keep_last_rate).abs() < f64::EPSILON,
        "max_keep_last_rate must be auto-filled with DEFAULT"
    );
    assert!(
        assert.enforce_monitor_thresholds,
        "enforce_monitor_thresholds must be set to true"
    );
}

/// Merge semantics pin: true (self) || true (other) → true.
/// Completes the OR truth-table coverage that the false_false /
/// true_left / false_true canaries above pin three of four cells
/// for. The TT cell is the case a future XOR-vs-OR regression would
/// catch only via this canary — false||false=F, false||true=T, and
/// true||false=T pass under XOR semantics too. true||true=T does
/// NOT pass under XOR (XOR would yield F here), so a regression
/// that swapped OR for XOR escapes the existing 3 canaries unless
/// this 4th cell pins it.
#[test]
fn merge_enforce_monitor_thresholds_or_semantics_true_true() {
    let left = Assert::NO_OVERRIDES.with_monitor_defaults();
    let right = Assert::NO_OVERRIDES.with_monitor_defaults();
    let merged = left.merge(&right);
    assert!(
        merged.enforce_monitor_thresholds,
        "true || true must yield true (OR, not XOR)"
    );
}

/// Behavior pin: `with_monitor_defaults()` MUST NOT clobber
/// user-set values for the 6 unset-Option monitor-threshold fields.
/// The auto-fill loop in `with_monitor_defaults` uses `if X.is_none()`
/// guards; a regression that drops the guard and unconditionally
/// stores `Some(default)` would silently overwrite a user's custom
/// thresholds (e.g. test author sets max_imbalance_ratio=2.0 to
/// catch a known schedule-balance issue, with_monitor_defaults
/// silently restores 4.0). The companion canary
/// `with_monitor_defaults_fills_all_unset_threshold_fields` above
/// tests the auto-fill SIDE of the contract; this canary tests
/// the inverse SIDE: user-set values survive.
#[test]
fn with_monitor_defaults_preserves_user_set_values() {
    use crate::monitor::MonitorThresholds;
    let d = MonitorThresholds::new();
    // Pick values DIFFERENT from each DEFAULT so the test fails if
    // the auto-fill loop drops its `is_none()` guard and silently
    // overwrites. Differs from DEFAULT by ~50% on each axis where
    // the type allows; bool / counter axes flip / shift respectively.
    let custom_imbalance = d.max_imbalance_ratio * 0.5;
    let custom_local_dsq = d.max_local_dsq_depth * 2;
    let custom_fail_on_stall = !d.fail_on_stall;
    let custom_sustained = d.sustained_samples * 2;
    let custom_fallback = d.max_fallback_rate * 0.5;
    let custom_keep_last = d.max_keep_last_rate * 0.5;
    // Zero-default guard: if a future DEFAULT value happens to be 0,
    // the `* 0.5` / `* 2` derivation lands at the same value
    // (0 * 0.5 == 0; 0 * 2 == 0) and the test passes vacuously
    // against a regression that drops the `is_none()` guard.
    // Pre-check that each custom value DIFFERS from DEFAULT so a
    // future DEFAULT-of-zero fails the test loudly here rather
    // than silently elsewhere.
    assert!(
        (custom_imbalance - d.max_imbalance_ratio).abs() > f64::EPSILON,
        "test derivation produced custom_imbalance == DEFAULT — DEFAULT.max_imbalance_ratio may have shifted to 0"
    );
    assert_ne!(
        custom_local_dsq, d.max_local_dsq_depth,
        "test derivation produced custom_local_dsq == DEFAULT — DEFAULT.max_local_dsq_depth may have shifted to 0"
    );
    assert_ne!(
        custom_fail_on_stall, d.fail_on_stall,
        "bool flip should always differ"
    );
    assert_ne!(
        custom_sustained, d.sustained_samples,
        "test derivation produced custom_sustained == DEFAULT — DEFAULT.sustained_samples may have shifted to 0"
    );
    assert!(
        (custom_fallback - d.max_fallback_rate).abs() > f64::EPSILON,
        "test derivation produced custom_fallback == DEFAULT — DEFAULT.max_fallback_rate may have shifted to 0"
    );
    assert!(
        (custom_keep_last - d.max_keep_last_rate).abs() > f64::EPSILON,
        "test derivation produced custom_keep_last == DEFAULT — DEFAULT.max_keep_last_rate may have shifted to 0"
    );
    let assert = Assert::NO_OVERRIDES
        .max_imbalance_ratio(custom_imbalance)
        .max_local_dsq_depth(custom_local_dsq)
        .fail_on_stall(custom_fail_on_stall)
        .sustained_samples(custom_sustained)
        .max_fallback_rate(custom_fallback)
        .max_keep_last_rate(custom_keep_last)
        .with_monitor_defaults();
    assert!(
        (assert.max_imbalance_ratio.unwrap() - custom_imbalance).abs() < f64::EPSILON,
        "max_imbalance_ratio user-set value must survive with_monitor_defaults"
    );
    assert_eq!(
        assert.max_local_dsq_depth,
        Some(custom_local_dsq),
        "max_local_dsq_depth user-set value must survive"
    );
    assert_eq!(
        assert.fail_on_stall,
        Some(custom_fail_on_stall),
        "fail_on_stall user-set value must survive"
    );
    assert_eq!(
        assert.sustained_samples,
        Some(custom_sustained),
        "sustained_samples user-set value must survive"
    );
    assert!(
        (assert.max_fallback_rate.unwrap() - custom_fallback).abs() < f64::EPSILON,
        "max_fallback_rate user-set value must survive"
    );
    assert!(
        (assert.max_keep_last_rate.unwrap() - custom_keep_last).abs() < f64::EPSILON,
        "max_keep_last_rate user-set value must survive"
    );
    assert!(
        assert.enforce_monitor_thresholds,
        "enforce_monitor_thresholds still set to true (the only field with_monitor_defaults unconditionally writes)"
    );
}

/// OR-chain coverage: `has_monitor_thresholds()` MUST return false
/// when every monitor-threshold Option is None. NO_OVERRIDES is
/// the canonical all-None starting point; this canary pins the
/// false-side of the OR-chain in `has_monitor_thresholds`.
#[test]
fn has_monitor_thresholds_false_when_all_none() {
    let v = Assert::NO_OVERRIDES;
    assert!(
        !v.has_monitor_thresholds(),
        "NO_OVERRIDES must report has_monitor_thresholds() == false"
    );
}

/// OR-chain coverage: `has_monitor_thresholds()` MUST return true
/// when ANY single monitor-threshold Option is Some. Parametric
/// over the 6 fields — each iteration sets exactly one field to a
/// non-None value via the builder, asserts has_monitor_thresholds
/// returns true. Catches a regression that drops a field from the
/// `has_monitor_thresholds` OR-chain (the same field-drift class
/// the FailureDumpReport round-trip test catches on the dump
/// side). A dropped field would silently skip monitor enforcement
/// for tests that set only the missed field.
#[test]
fn has_monitor_thresholds_true_when_any_set() {
    // Compile-time anchor for the maintenance
    // contract. Adding a new MonitorThresholds field requires
    // updating BOTH this struct-update literal (compile error
    // otherwise) AND the closure list below. Without this anchor,
    // the closure-list maintenance is doc-only and easy to miss
    // when adding a field — Rust would let the test pass silently
    // against the new field. The literal's _ binding discards the
    // value; only the compile-time field enumeration matters.
    use crate::monitor::MonitorThresholds;
    let _ = MonitorThresholds {
        max_imbalance_ratio: 0.0,
        max_local_dsq_depth: 0,
        fail_on_stall: false,
        sustained_samples: 0,
        max_fallback_rate: 0.0,
        max_keep_last_rate: 0.0,
        enforce: false,
    };
    // Each closure returns an Assert with exactly one threshold
    // field set. Adding a new MonitorThresholds field requires
    // adding a closure here — that's the maintenance contract,
    // anchored by the struct-update above.
    type FieldSetter = (&'static str, fn() -> Assert);
    let setters: &[FieldSetter] = &[
        ("max_imbalance_ratio", || {
            Assert::NO_OVERRIDES.max_imbalance_ratio(3.0)
        }),
        ("max_local_dsq_depth", || {
            Assert::NO_OVERRIDES.max_local_dsq_depth(64)
        }),
        ("fail_on_stall", || Assert::NO_OVERRIDES.fail_on_stall(true)),
        ("sustained_samples", || {
            Assert::NO_OVERRIDES.sustained_samples(7)
        }),
        ("max_fallback_rate", || {
            Assert::NO_OVERRIDES.max_fallback_rate(150.0)
        }),
        ("max_keep_last_rate", || {
            Assert::NO_OVERRIDES.max_keep_last_rate(75.0)
        }),
    ];
    for (field, build) in setters {
        let v = build();
        assert!(
            v.has_monitor_thresholds(),
            "has_monitor_thresholds() must return true when only `{field}` is set"
        );
    }
}

#[test]
fn assert_merge_other_overrides_self() {
    let base = Assert::NO_OVERRIDES;
    let other = Assert::NO_OVERRIDES
        .check_not_starved()
        .max_gap_ms(5000)
        .max_imbalance_ratio(2.0);
    let merged = base.merge(&other);
    assert_eq!(merged.not_starved, Some(true));
    assert_eq!(merged.max_gap_ms, Some(5000));
    assert_eq!(merged.max_imbalance_ratio, Some(2.0));
}

#[test]
fn assert_merge_preserves_self_when_other_is_none() {
    let base = Assert::NO_OVERRIDES
        .check_not_starved()
        .max_imbalance_ratio(4.0);
    let merged = base.merge(&Assert::NO_OVERRIDES);
    assert_eq!(merged.not_starved, Some(true));
    assert_eq!(merged.max_imbalance_ratio, Some(4.0));
}

#[test]
fn assert_merge_other_takes_precedence() {
    let base = Assert::NO_OVERRIDES.max_imbalance_ratio(4.0);
    let other = Assert::NO_OVERRIDES.max_imbalance_ratio(2.0);
    let merged = base.merge(&other);
    assert_eq!(merged.max_imbalance_ratio, Some(2.0));
}

#[test]
fn assert_merge_last_some_wins() {
    let base = Assert::NO_OVERRIDES.check_not_starved();
    let other = Assert::NO_OVERRIDES.check_isolation();
    let merged = base.merge(&other);
    assert_eq!(merged.not_starved, Some(true));
    assert_eq!(merged.isolation, Some(true));
}

#[test]
fn assert_merge_child_disables_not_starved() {
    let base = Assert::NO_OVERRIDES.check_not_starved();
    let other = Assert {
        not_starved: Some(false),
        ..Assert::NO_OVERRIDES
    };
    let merged = base.merge(&other);
    assert_eq!(merged.not_starved, Some(false));
    assert!(!merged.worker_plan().not_starved);
}

#[test]
fn assert_merge_child_disables_isolation() {
    let base = Assert::NO_OVERRIDES.check_isolation(); // isolation = Some(true)
    let other = Assert {
        isolation: Some(false),
        ..Assert::NO_OVERRIDES
    };
    let merged = base.merge(&other);
    assert_eq!(merged.isolation, Some(false));
    assert!(!merged.worker_plan().isolation);
}

#[test]
fn assert_worker_plan_extraction() {
    let v = Assert::NO_OVERRIDES
        .check_not_starved()
        .check_isolation()
        .max_gap_ms(3000)
        .max_spread_pct(25.0);
    assert_eq!(v.not_starved, Some(true));
    assert_eq!(v.isolation, Some(true));
    let plan = v.worker_plan();
    assert!(plan.not_starved);
    assert!(plan.isolation);
    assert_eq!(plan.max_gap_ms, Some(3000));
    assert_eq!(plan.max_spread_pct, Some(25.0));
}

#[test]
fn assert_cgroup_delegates_to_plan() {
    let v = Assert::NO_OVERRIDES.check_not_starved();
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
    let r = v.assert_cgroup(&reports, None);
    assert!(r.is_pass());
    assert_eq!(r.stats.total_workers, 1);
}

#[test]
fn assert_monitor_thresholds_extraction() {
    let v = Assert::NO_OVERRIDES
        .max_imbalance_ratio(2.5)
        .max_local_dsq_depth(100)
        .fail_on_stall(false)
        .sustained_samples(10)
        .max_fallback_rate(50.0)
        .max_keep_last_rate(25.0);
    let t = v.monitor_thresholds();
    assert!((t.max_imbalance_ratio - 2.5).abs() < f64::EPSILON);
    assert_eq!(t.max_local_dsq_depth, 100);
    assert!(!t.fail_on_stall);
    assert_eq!(t.sustained_samples, 10);
    assert!((t.max_fallback_rate - 50.0).abs() < f64::EPSILON);
    assert!((t.max_keep_last_rate - 25.0).abs() < f64::EPSILON);
}

#[test]
fn assert_monitor_thresholds_defaults_when_none() {
    let v = Assert::NO_OVERRIDES;
    let t = v.monitor_thresholds();
    let d = crate::monitor::MonitorThresholds::new();
    assert!((t.max_imbalance_ratio - d.max_imbalance_ratio).abs() < f64::EPSILON);
    assert_eq!(t.max_local_dsq_depth, d.max_local_dsq_depth);
}

#[test]
fn assert_chain_all_setters() {
    let v = Assert::NO_OVERRIDES
        .check_not_starved()
        .check_isolation()
        .max_gap_ms(1000)
        .max_spread_pct(5.0)
        .max_imbalance_ratio(3.0)
        .max_local_dsq_depth(20)
        .fail_on_stall(true)
        .sustained_samples(3)
        .max_fallback_rate(100.0)
        .max_keep_last_rate(50.0);
    assert_eq!(v.not_starved, Some(true));
    assert_eq!(v.isolation, Some(true));
    assert_eq!(v.max_gap_ms, Some(1000));
    assert_eq!(v.max_spread_pct, Some(5.0));
    assert_eq!(v.max_imbalance_ratio, Some(3.0));
    assert_eq!(v.max_local_dsq_depth, Some(20));
    assert_eq!(v.fail_on_stall, Some(true));
    assert_eq!(v.sustained_samples, Some(3));
    assert_eq!(v.max_fallback_rate, Some(100.0));
    assert_eq!(v.max_keep_last_rate, Some(50.0));
}

// -- gap_threshold_ms tests --

#[test]
fn gap_threshold_default() {
    let t = gap_threshold_ms();
    if cfg!(debug_assertions) {
        assert_eq!(t, 3000);
    } else {
        assert_eq!(t, 2000);
    }
}

// -- Assert::merge per-field tests --

#[test]
fn assert_merge_max_spread_pct() {
    let base = Assert::NO_OVERRIDES.max_spread_pct(10.0);
    let other = Assert::NO_OVERRIDES.max_spread_pct(5.0);
    assert_eq!(base.merge(&other).max_spread_pct, Some(5.0));
    assert_eq!(base.merge(&Assert::NO_OVERRIDES).max_spread_pct, Some(10.0));
}

#[test]
fn assert_merge_fail_on_stall() {
    let base = Assert::NO_OVERRIDES.fail_on_stall(true);
    let other = Assert::NO_OVERRIDES.fail_on_stall(false);
    assert_eq!(base.merge(&other).fail_on_stall, Some(false));
    assert_eq!(base.merge(&Assert::NO_OVERRIDES).fail_on_stall, Some(true));
}

#[test]
fn assert_merge_sustained_samples() {
    let base = Assert::NO_OVERRIDES.sustained_samples(5);
    let other = Assert::NO_OVERRIDES.sustained_samples(10);
    assert_eq!(base.merge(&other).sustained_samples, Some(10));
    assert_eq!(base.merge(&Assert::NO_OVERRIDES).sustained_samples, Some(5));
}

#[test]
fn assert_merge_max_fallback_rate() {
    let base = Assert::NO_OVERRIDES.max_fallback_rate(200.0);
    let other = Assert::NO_OVERRIDES.max_fallback_rate(50.0);
    assert_eq!(base.merge(&other).max_fallback_rate, Some(50.0));
    assert_eq!(
        base.merge(&Assert::NO_OVERRIDES).max_fallback_rate,
        Some(200.0)
    );
}

#[test]
fn assert_merge_max_keep_last_rate() {
    let base = Assert::NO_OVERRIDES.max_keep_last_rate(100.0);
    let other = Assert::NO_OVERRIDES.max_keep_last_rate(25.0);
    assert_eq!(base.merge(&other).max_keep_last_rate, Some(25.0));
    assert_eq!(
        base.merge(&Assert::NO_OVERRIDES).max_keep_last_rate,
        Some(100.0)
    );
}

#[test]
fn assert_merge_max_local_dsq_depth() {
    let base = Assert::NO_OVERRIDES.max_local_dsq_depth(50);
    let other = Assert::NO_OVERRIDES.max_local_dsq_depth(100);
    assert_eq!(base.merge(&other).max_local_dsq_depth, Some(100));
    assert_eq!(
        base.merge(&Assert::NO_OVERRIDES).max_local_dsq_depth,
        Some(50)
    );
}

#[test]
fn assert_merge_max_gap_ms() {
    let base = Assert::NO_OVERRIDES.max_gap_ms(2000);
    let other = Assert::NO_OVERRIDES.max_gap_ms(5000);
    assert_eq!(base.merge(&other).max_gap_ms, Some(5000));
    assert_eq!(base.merge(&Assert::NO_OVERRIDES).max_gap_ms, Some(2000));
}

#[test]
fn assert_merge_three_layers() {
    let defaults = Assert::default_checks();
    let sched = Assert::NO_OVERRIDES
        .max_imbalance_ratio(2.0)
        .max_fallback_rate(50.0);
    let test = Assert::NO_OVERRIDES.max_gap_ms(5000);
    let merged = defaults.merge(&sched).merge(&test);
    assert_eq!(merged.not_starved, None);
    assert_eq!(merged.max_imbalance_ratio, Some(2.0));
    assert_eq!(merged.max_fallback_rate, Some(50.0));
    assert_eq!(merged.max_gap_ms, Some(5000));
    assert_eq!(merged.sustained_samples, None);
}

#[test]
fn assert_merge_no_overrides_preserves_base() {
    let base = Assert::NO_OVERRIDES
        .check_not_starved()
        .max_imbalance_ratio(4.0)
        .fail_on_stall(true);
    let merged = base.merge(&Assert::NO_OVERRIDES);
    assert_eq!(merged.not_starved, Some(true));
    assert_eq!(merged.max_imbalance_ratio, Some(4.0));
    assert_eq!(merged.fail_on_stall, Some(true));
}

/// `default_checks()` is `NO_OVERRIDES`, so merging in either
/// direction is the identity.
#[test]
fn assert_merge_no_overrides_is_left_identity() {
    let merged = Assert::NO_OVERRIDES.merge(&Assert::default_checks());
    assert!(merged.not_starved.is_none());
    assert!(merged.max_imbalance_ratio.is_none());
    assert!(merged.max_gap_ms.is_none());
    assert!(merged.isolation.is_none());
}

/// The runtime three-layer chain
/// `default_checks -> scheduler -> test` collapses to
/// `NO_OVERRIDES` when both override layers are also `NO_OVERRIDES`.
#[test]
fn assert_merge_runtime_chain_with_no_overrides_yields_defaults() {
    let scheduler_assert = Assert::NO_OVERRIDES;
    let test_assert = Assert::NO_OVERRIDES;
    let merged = Assert::default_checks()
        .merge(&scheduler_assert)
        .merge(&test_assert);
    assert!(merged.not_starved.is_none());
    assert!(merged.max_imbalance_ratio.is_none());
    assert!(merged.max_local_dsq_depth.is_none());
    assert!(merged.fail_on_stall.is_none());
}

#[test]
fn assert_merge_overrides_fields() {
    let base = Assert::NO_OVERRIDES;
    let overrides = Assert::NO_OVERRIDES
        .max_imbalance_ratio(5.0)
        .max_gap_ms(1000)
        .check_not_starved();
    let merged = base.merge(&overrides);
    assert_eq!(merged.not_starved, Some(true));
    assert_eq!(merged.max_imbalance_ratio, Some(5.0));
    assert_eq!(merged.max_gap_ms, Some(1000));
}

#[test]
fn assert_merge_later_overrides_earlier() {
    let a = Assert::NO_OVERRIDES.max_imbalance_ratio(2.0);
    let b = Assert::NO_OVERRIDES.max_imbalance_ratio(10.0);
    let merged = a.merge(&b);
    assert_eq!(merged.max_imbalance_ratio, Some(10.0));
}

#[test]
fn assert_worker_plan_extracts_fields() {
    let v = Assert::NO_OVERRIDES
        .check_not_starved()
        .check_isolation()
        .max_gap_ms(500)
        .max_spread_pct(10.0);
    assert_eq!(v.not_starved, Some(true));
    assert_eq!(v.isolation, Some(true));
    let plan = v.worker_plan();
    assert!(plan.not_starved);
    assert!(plan.isolation);
    assert_eq!(plan.max_gap_ms, Some(500));
    assert_eq!(plan.max_spread_pct, Some(10.0));
}

#[test]
fn assert_monitor_thresholds_defaults() {
    let v = Assert::NO_OVERRIDES;
    let t = v.monitor_thresholds();
    // Should use MonitorThresholds::new() values.
    let d = crate::monitor::MonitorThresholds::new();
    assert_eq!(t.max_imbalance_ratio, d.max_imbalance_ratio);
    assert_eq!(t.max_local_dsq_depth, d.max_local_dsq_depth);
}

#[test]
fn assert_monitor_thresholds_overridden() {
    let v = Assert::NO_OVERRIDES
        .max_imbalance_ratio(99.0)
        .max_local_dsq_depth(42)
        .fail_on_stall(false)
        .sustained_samples(10)
        .max_fallback_rate(0.5)
        .max_keep_last_rate(0.3);
    let t = v.monitor_thresholds();
    assert_eq!(t.max_imbalance_ratio, 99.0);
    assert_eq!(t.max_local_dsq_depth, 42);
    assert!(!t.fail_on_stall);
    assert_eq!(t.sustained_samples, 10);
    assert_eq!(t.max_fallback_rate, 0.5);
    assert_eq!(t.max_keep_last_rate, 0.3);
}

#[test]
fn assert_max_spread_pct() {
    let v = Assert::NO_OVERRIDES.max_spread_pct(25.0);
    assert_eq!(v.max_spread_pct, Some(25.0));
}

// -- Assert::has_worker_checks --

#[test]
fn assert_no_overrides_has_no_worker_checks() {
    assert!(!Assert::NO_OVERRIDES.has_worker_checks());
}

#[test]
fn assert_default_checks_has_no_worker_checks() {
    assert!(!Assert::default_checks().has_worker_checks());
}

#[test]
fn assert_single_field_has_worker_checks() {
    assert!(Assert::NO_OVERRIDES.max_gap_ms(5000).has_worker_checks());
    assert!(Assert::NO_OVERRIDES.check_isolation().has_worker_checks());
    assert!(
        Assert::NO_OVERRIDES
            .max_spread_pct(10.0)
            .has_worker_checks()
    );
    assert!(
        Assert::NO_OVERRIDES
            .max_throughput_cv(0.5)
            .has_worker_checks()
    );
    assert!(
        Assert::NO_OVERRIDES
            .min_work_rate(100.0)
            .has_worker_checks()
    );
    assert!(
        Assert::NO_OVERRIDES
            .max_p99_wake_latency_ns(1000)
            .has_worker_checks()
    );
    assert!(
        Assert::NO_OVERRIDES
            .max_wake_latency_cv(0.5)
            .has_worker_checks()
    );
    assert!(
        Assert::NO_OVERRIDES
            .min_iteration_rate(10.0)
            .has_worker_checks()
    );
    assert!(
        Assert::NO_OVERRIDES
            .max_migration_ratio(0.5)
            .has_worker_checks()
    );
}

#[test]
fn assert_monitor_only_no_worker_checks() {
    let a = Assert::NO_OVERRIDES
        .max_imbalance_ratio(5.0)
        .fail_on_stall(true);
    assert!(!a.has_worker_checks());
}

// -- Assert::merge worker + benchmark + monitor fields --

#[test]
fn assert_merge_all_field_categories() {
    // Layer 1: defaults (all None now).
    let defaults = Assert::default_checks();

    // Layer 2: scheduler sets worker and benchmark fields.
    let sched = Assert::NO_OVERRIDES
        .max_spread_pct(50.0)
        .max_p99_wake_latency_ns(100_000)
        .max_migration_ratio(0.5)
        .fail_on_stall(true);

    // Layer 3: test overrides a worker field and sets isolation.
    let test = Assert::NO_OVERRIDES.check_isolation().max_spread_pct(80.0);

    let merged = defaults.merge(&sched).merge(&test);

    // test overrides sched's spread.
    assert_eq!(merged.max_spread_pct, Some(80.0));
    // sched's benchmark fields survive (test didn't set them).
    assert_eq!(merged.max_p99_wake_latency_ns, Some(100_000));
    assert_eq!(merged.max_migration_ratio, Some(0.5));
    // test sets isolation.
    assert_eq!(merged.isolation, Some(true));
    // sched's monitor fields survive (test didn't set them).
    assert_eq!(merged.fail_on_stall, Some(true));
}

/// `Assert::expect_scx_bpf_error_contains` builder is `const fn` so
/// it composes inside `Assert::NO_OVERRIDES.expect_..._contains(...)`
/// const-context chains. Empty literals panic at construction (an
/// empty literal would silently match every message via
/// `str::contains`).
#[test]
fn expect_scx_bpf_error_contains_builder_sets_field() {
    let a = Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("apply_cell_config");
    assert_eq!(a.expect_scx_bpf_error_contains, Some("apply_cell_config"));
    assert_eq!(a.expect_scx_bpf_error_matches, None);
}

#[test]
#[should_panic(expected = "must be non-empty")]
fn expect_scx_bpf_error_contains_builder_panics_on_empty() {
    let _ = Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("");
}

#[test]
fn expect_scx_bpf_error_matches_builder_sets_field() {
    let a = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"apply_cell_config.*-EINVAL");
    assert_eq!(
        a.expect_scx_bpf_error_matches,
        Some(r"apply_cell_config.*-EINVAL"),
    );
    assert_eq!(a.expect_scx_bpf_error_contains, None);
}

#[test]
#[should_panic(expected = "must be non-empty")]
fn expect_scx_bpf_error_matches_builder_panics_on_empty() {
    let _ = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches("");
}

/// No matchers set → evaluator returns empty Vec regardless of
/// captured text or expect_err state. Existing tests that never opt
/// into reproducer mode must see zero matcher-driven details.
#[test]
fn evaluate_scx_bpf_error_match_returns_empty_when_no_matcher_configured() {
    let a = Assert::NO_OVERRIDES;
    assert!(
        a.evaluate_scx_bpf_error_match("some scheduler text", true)
            .is_empty(),
    );
    assert!(
        a.evaluate_scx_bpf_error_match("some scheduler text", false)
            .is_empty(),
    );
    assert!(a.evaluate_scx_bpf_error_match("", true).is_empty());
}

/// Matcher set + expect_err = false → evaluator rejects with a
/// misuse diagnostic naming the expect_err requirement. Catches
/// runtime misuse even if the entry-time `KtstrTestEntry::validate`
/// gate is somehow bypassed.
#[test]
fn evaluate_scx_bpf_error_match_rejects_when_expect_err_unset() {
    let a = Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("anything");
    let details = a.evaluate_scx_bpf_error_match("text including anything substring", false);
    assert_eq!(details.len(), 1);
    let msg = &details[0].message;
    assert!(
        msg.contains("expect_err = true"),
        "misuse diagnostic must name expect_err: {msg}",
    );
}

/// Literal-substring matcher passes when the literal appears in the
/// captured corpus.
#[test]
fn evaluate_scx_bpf_error_match_contains_passes_on_substring_present() {
    let a =
        Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("apply_cell_config returned -EINVAL");
    let corpus = "scx_mitosis: apply_cell_config returned -EINVAL at line:42; cell_id=3\n";
    let details = a.evaluate_scx_bpf_error_match(corpus, true);
    assert!(
        details.is_empty(),
        "matcher must pass on substring match: {details:?}"
    );
}

/// Literal-substring matcher fails when the literal is absent, with
/// a diagnostic naming the expected literal AND quoting the corpus
/// so the operator can compare at a glance.
#[test]
fn evaluate_scx_bpf_error_match_contains_fails_on_absent_substring() {
    let a =
        Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("apply_cell_config returned -EINVAL");
    let corpus = "scx_mitosis: a DIFFERENT bug fired: stuck task X for 5000ms\n";
    let details = a.evaluate_scx_bpf_error_match(corpus, true);
    assert_eq!(details.len(), 1);
    let msg = &details[0].message;
    assert!(
        msg.contains("apply_cell_config returned -EINVAL"),
        "diagnostic must name expected literal: {msg}",
    );
    assert!(
        msg.contains("a DIFFERENT bug fired"),
        "diagnostic must include the actual captured corpus: {msg}",
    );
}

/// Regex matcher passes on match — anchors, character classes, and
/// captures all work via the `regex` crate.
#[test]
fn evaluate_scx_bpf_error_match_regex_passes_on_match() {
    let a = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"apply_cell_config.*-E[A-Z]+");
    let corpus = "scx_mitosis: apply_cell_config returned -EINVAL at line:42\n";
    let details = a.evaluate_scx_bpf_error_match(corpus, true);
    assert!(details.is_empty(), "regex matcher must pass: {details:?}");
}

#[test]
fn evaluate_scx_bpf_error_match_regex_fails_on_mismatch() {
    let a = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"apply_cell_config.*-E[A-Z]+");
    let corpus = "scx_mitosis: stuck task for 5000ms\n";
    let details = a.evaluate_scx_bpf_error_match(corpus, true);
    assert_eq!(details.len(), 1);
    let msg = &details[0].message;
    assert!(msg.contains("apply_cell_config"));
    assert!(msg.contains("stuck task"));
}

/// Invalid regex syntax panics at construction with a diagnostic
/// naming the pattern AND the compile error. Construction-time
/// rejection means a typo in the pattern surfaces immediately when
/// the test binary loads (or the `#[ktstr_test]` macro generates
/// the entry) instead of silently sitting in the entry until the
/// matcher actually evaluates a corpus.
#[test]
#[should_panic(expected = "is not valid regex")]
fn evaluate_scx_bpf_error_match_regex_invalid_pattern_panics_at_construction() {
    let _ = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches("[unclosed");
}

/// Both matchers set + both match → empty details.
#[test]
fn evaluate_scx_bpf_error_match_both_set_and_both_match_passes() {
    let a = Assert::NO_OVERRIDES
        .expect_scx_bpf_error_contains("apply_cell_config")
        .expect_scx_bpf_error_matches(r"returned -E[A-Z]+");
    let corpus = "scx_mitosis: apply_cell_config returned -EINVAL\n";
    assert!(a.evaluate_scx_bpf_error_match(corpus, true).is_empty());
}

/// Both matchers set + only one matches → matcher fails with a
/// diagnostic for the one that didn't.
#[test]
fn evaluate_scx_bpf_error_match_both_set_one_mismatches_fails() {
    let a = Assert::NO_OVERRIDES
        .expect_scx_bpf_error_contains("apply_cell_config")
        .expect_scx_bpf_error_matches(r"returned -E[A-Z]+");
    let corpus = "scx_mitosis: apply_cell_config but no errno here\n";
    let details = a.evaluate_scx_bpf_error_match(corpus, true);
    assert_eq!(details.len(), 1);
    assert!(details[0].message.contains("returned -E"));
}

/// Multi-event scenario: when the upstream pipeline concatenates
/// two or more `sched_ext: BPF scheduler "..." disabled (...)` exit
/// events into one corpus (rapid load+disable cycles per
/// `parse_kmsg_window`'s "back-to-back disable events" doc), the matcher must scan the WHOLE
/// concatenated string, not stop at the first event boundary.
///
/// The corpus below interleaves two distinct events; the matched
/// pattern lives ONLY in the second event. A regression that
/// scanned only the first event would treat the second as if it
/// didn't exist — the matcher would fail despite the bug actually
/// reproducing in the second cycle. Pin BOTH the literal and regex
/// matchers since their corpus-scanning paths are independent
/// (`str::contains` vs. `regex::Regex::is_match`).
#[test]
fn evaluate_scx_bpf_error_match_finds_pattern_in_later_event() {
    let corpus = "scx_mitosis: first different error at line:1\n\
                  sched_ext: BPF scheduler \"scx_mitosis\" disabled (Error)\n\
                  scx_mitosis: apply_cell_config returned -EINVAL at line:42\n\
                  sched_ext: BPF scheduler \"scx_mitosis\" disabled (Error)\n";

    // Literal-substring matcher walks the whole corpus.
    let a_lit =
        Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("apply_cell_config returned -EINVAL");
    let lit_details = a_lit.evaluate_scx_bpf_error_match(corpus, true);
    assert!(
        lit_details.is_empty(),
        "contains matcher must scan the WHOLE concatenated corpus, not stop \
         at the first event boundary: {lit_details:?}",
    );

    // Regex matcher walks the whole corpus.
    let a_re = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"apply_cell_config.*-E[A-Z]+");
    let re_details = a_re.evaluate_scx_bpf_error_match(corpus, true);
    assert!(
        re_details.is_empty(),
        "regex matcher must scan the WHOLE concatenated corpus: {re_details:?}",
    );
}

/// Empty corpus with a configured literal matcher fails loudly. The
/// diagnostic must name the matcher, quote the expected literal, and
/// surface the corpus length so an operator can tell at a glance
/// whether the scheduler produced any captured text at all. Pins that
/// "0 bytes" appears verbatim — that token is the operator's signal
/// that the upstream capture pipeline (bulk-port scheduler log +
/// sched_ext dump extract) produced nothing, distinct from "matcher
/// mismatched substantial output."
#[test]
fn evaluate_scx_bpf_error_match_contains_fails_on_empty_corpus() {
    let a = Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("apply_cell_config");
    let details = a.evaluate_scx_bpf_error_match("", true);
    assert_eq!(details.len(), 1);
    let msg = &details[0].message;
    assert!(
        msg.contains("expect_scx_bpf_error_contains"),
        "diagnostic must name the matcher: {msg}",
    );
    assert!(
        msg.contains("apply_cell_config"),
        "diagnostic must quote the expected literal: {msg}",
    );
    assert!(
        msg.contains("0 bytes"),
        "diagnostic must surface the corpus length so operators can distinguish \
         empty-capture from matcher-mismatch: {msg}",
    );
    assert!(
        msg.contains("up to 400 bytes follow:"),
        "diagnostic must include the standard excerpt suffix even when empty: {msg}",
    );
}

/// Empty corpus with a configured regex matcher fails loudly with
/// the same diagnostic shape as the literal-substring variant.
#[test]
fn evaluate_scx_bpf_error_match_regex_fails_on_empty_corpus() {
    let pattern = r"apply_cell_config.*-EINVAL";
    let a = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(pattern);
    let details = a.evaluate_scx_bpf_error_match("", true);
    assert_eq!(details.len(), 1);
    let msg = &details[0].message;
    assert!(
        msg.contains("expect_scx_bpf_error_matches"),
        "diagnostic must name the matcher: {msg}",
    );
    assert!(
        msg.contains(pattern),
        "diagnostic must quote the pattern: {msg}",
    );
    assert!(
        msg.contains("0 bytes"),
        "diagnostic must surface the corpus length: {msg}",
    );
    assert!(
        msg.contains("up to 400 bytes follow:"),
        "diagnostic must include the standard excerpt suffix: {msg}",
    );
}

/// Both matchers configured + empty corpus → each produces its own
/// failure detail (AND-semantics: every matcher must hit, every miss
/// surfaces independently). Pin both details land, but do not pin
/// their order — a future refactor that swaps order would still be
/// semantically correct.
#[test]
fn evaluate_scx_bpf_error_match_both_fail_on_empty_corpus() {
    let pattern = r"apply_cell_config.*-EINVAL";
    let a = Assert::NO_OVERRIDES
        .expect_scx_bpf_error_contains("apply_cell_config")
        .expect_scx_bpf_error_matches(pattern);
    let details = a.evaluate_scx_bpf_error_match("", true);
    assert_eq!(
        details.len(),
        2,
        "both matchers must produce independent failure details on empty corpus; got {details:?}",
    );
    assert!(
        details
            .iter()
            .any(|d| d.message.contains("expect_scx_bpf_error_contains")),
        "one detail must name the contains matcher: {details:?}",
    );
    assert!(
        details
            .iter()
            .any(|d| d.message.contains("expect_scx_bpf_error_matches")),
        "the other detail must name the regex matcher: {details:?}",
    );
    for d in &details {
        assert!(
            d.message.contains("0 bytes"),
            "every detail must include the corpus-size diagnostic: {}",
            d.message,
        );
    }
}

/// The failure-diagnostic excerpt is byte-truncated at a UTF-8
/// boundary so its size matches the "up to 400 bytes follow:" prose.
/// A future regression that switched back to `chars().take(400)` would
/// produce excerpts up to 4 × 400 bytes on multi-byte corpora; pin
/// here so the byte-budget contract holds. The fixture uses em-dashes
/// (3-byte UTF-8): byte 400 lands mid-codepoint, so the boundary walk
/// steps back to 399. This is the only 1-byte/2-byte/3-byte/4-byte
/// width that produces a non-no-op walk against a 400-byte budget;
/// 2-byte and 4-byte codepoints land cleanly on byte 400 and would
/// trivially pass the fast path.
///
/// Both matcher branches share the `excerpt` closure today, so this
/// test pins the contract via the literal branch AND a second
/// assertion via the regex branch — if a future refactor splits the
/// closure (e.g. matcher path adds context lines), this test catches
/// the regression rather than silently passing on the surviving
/// shared path.
#[test]
fn evaluate_scx_bpf_error_match_excerpt_byte_truncated_at_char_boundary() {
    // 200 em-dashes (each 3 bytes in UTF-8) = 600 bytes / 200 chars.
    // chars().take(400) would yield all 200 chars = 600 bytes (over budget);
    // byte-truncation stops at or below 400 bytes.
    let mut corpus = String::new();
    for _ in 0..200 {
        corpus.push('\u{2014}'); // em-dash, 3 UTF-8 bytes
    }
    assert_eq!(corpus.len(), 600, "test fixture: 200 em-dashes = 600 bytes");

    let assert_excerpt_within_budget = |msg: &str| {
        let after_marker = msg
            .split_once("up to 400 bytes follow:\n")
            .unwrap_or_else(|| panic!("diagnostic missing excerpt marker: {msg}"))
            .1;
        // 400 / 3 = 133 full em-dashes fit (399 bytes); the 401st byte
        // would split a codepoint, so the truncation steps back to 399.
        // Exact-length pin catches both "no truncation" regressions
        // (600 bytes) and forward-walk off-by-one (402 bytes) — and
        // since the slice is valid UTF-8 by construction, the codepoint
        // composition assertion below also pins that the excerpt is a
        // legitimate em-dash prefix rather than a lossy decode.
        assert_eq!(
            after_marker.len(),
            399,
            "200 em-dashes at 3 bytes each: 133 fit in 399 bytes; 134 would overflow at 402",
        );
        assert!(
            after_marker.chars().all(|c| c == '\u{2014}'),
            "excerpt must contain only em-dashes (no replacement char or partial codepoint): {after_marker:?}",
        );
    };

    // Literal matcher branch.
    let literal = Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("not present");
    let literal_details = literal.evaluate_scx_bpf_error_match(&corpus, true);
    assert_eq!(literal_details.len(), 1);
    assert_excerpt_within_budget(&literal_details[0].message);

    // Regex matcher branch — uses the same `excerpt` closure today,
    // but a future refactor that splits the two branches would let
    // a regression slip past a literal-only pin.
    let regex = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"not present anywhere");
    let regex_details = regex.evaluate_scx_bpf_error_match(&corpus, true);
    assert_eq!(regex_details.len(), 1);
    assert_excerpt_within_budget(&regex_details[0].message);
}

/// Literal-substring matcher does NOT trim or normalize whitespace.
/// A pattern with leading or trailing spaces only matches a corpus
/// with those bytes present verbatim. Pins this so a future "helpful"
/// trim regression (e.g. `captured_text.trim().contains(literal.trim())`)
/// fails here — operators rely on byte-for-byte matching to pin
/// exact kernel printk strings.
#[test]
fn evaluate_scx_bpf_error_match_contains_no_trim_on_literal() {
    let leading = Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("  leading spaces");
    // Corpus has "leading spaces" but NOT with the two-space prefix.
    let no_match = leading.evaluate_scx_bpf_error_match("the line: leading spaces here", true);
    assert_eq!(
        no_match.len(),
        1,
        "literal with two leading spaces must NOT match corpus where the only \
         instance lacks the leading spaces; got {no_match:?}",
    );
    // Same matcher against corpus that DOES have the two-space prefix → empty.
    let matches =
        leading.evaluate_scx_bpf_error_match("prefix\n  leading spaces appear here", true);
    assert!(
        matches.is_empty(),
        "literal with leading spaces must match when corpus has them byte-for-byte; got {matches:?}",
    );

    // Trailing-whitespace symmetry. The corpus must contain the BASE
    // token "trailing" so that a hypothetical `.trim_end()` regression
    // would make the literal match incorrectly — comparing against an
    // unrelated corpus would pass under either implementation and
    // wouldn't pin no-trim semantics.
    let trailing = Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("trailing  ");
    // Negative: corpus has "trailing" with only one trailing space, not two.
    // Under a `.trim_end()` regression both sides would collapse to "trailing"
    // and the matcher would silently succeed.
    let no_trail_match = trailing.evaluate_scx_bpf_error_match("some trailing text", true);
    assert_eq!(
        no_trail_match.len(),
        1,
        "literal with two trailing spaces must NOT match corpus where the base \
         token is followed by one space (catches hypothetical trim regression): \
         {no_trail_match:?}",
    );
    // Positive: corpus has the two trailing spaces byte-for-byte.
    let trail_match = trailing.evaluate_scx_bpf_error_match("some trailing  text", true);
    assert!(
        trail_match.is_empty(),
        "literal with two trailing spaces must match corpus with two spaces \
         byte-for-byte: {trail_match:?}",
    );
}

/// Literal-substring matcher is unaffected by embedded `\n` / `\t` in
/// the corpus around the literal token. Pin that the contains() walk
/// is byte-level — newlines and tabs surrounding the match are not
/// boundaries the matcher cares about.
#[test]
fn evaluate_scx_bpf_error_match_contains_unaffected_by_newlines_and_tabs() {
    let a = Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("apply_cell_config");
    let corpus = "first line\n\tindented tab here apply_cell_config more\n";
    let details = a.evaluate_scx_bpf_error_match(corpus, true);
    assert!(
        details.is_empty(),
        "literal match must succeed across embedded whitespace surroundings: {details:?}",
    );

    // A literal that itself contains a newline only matches when that
    // exact byte sequence is present — newlines aren't elided.
    let with_nl = Assert::NO_OVERRIDES.expect_scx_bpf_error_contains("apply_cell_conf\nig");
    let nl_details = with_nl.evaluate_scx_bpf_error_match("apply_cell_config", true);
    assert_eq!(
        nl_details.len(),
        1,
        "literal containing a newline must NOT match a corpus without that newline: {nl_details:?}",
    );
}

/// Regex `^` / `$` are STRING-boundary anchors by default — they
/// match the start / end of the WHOLE corpus, not individual lines.
/// Operators who want line-level anchoring must opt in with the
/// inline `(?m)` flag. Pin BOTH anchors:
///   `$` end-anchor:
///     (1) `apply_cell_config$` fails when the token is mid-corpus,
///     (2) `apply_cell_config$` matches when the token is at corpus-end,
///     (3) `(?m)apply_cell_config$` matches the token at any line-end.
///   `^` start-anchor:
///     (4) `^apply_cell_config` fails when the token is mid-corpus,
///     (5) `^apply_cell_config` matches when the token is at corpus-start,
///     (6) `(?m)^apply_cell_config` matches the token at any line-start.
#[test]
fn evaluate_scx_bpf_error_match_regex_anchors_are_string_boundaries_by_default() {
    // $ end-anchor — default mode: anchors to string-end.
    let dollar = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"apply_cell_config$");
    let dollar_mid = "apply_cell_config\nbut more after";
    let dollar_mid_details = dollar.evaluate_scx_bpf_error_match(dollar_mid, true);
    assert_eq!(
        dollar_mid_details.len(),
        1,
        "default-mode $ anchors to STRING-end; mid-corpus token must NOT match: {dollar_mid_details:?}",
    );

    // $ end-anchor — default mode positive: matches when token IS at string-end.
    let dollar_end = "prefix junk\napply_cell_config";
    let dollar_end_details = dollar.evaluate_scx_bpf_error_match(dollar_end, true);
    assert!(
        dollar_end_details.is_empty(),
        "default-mode $ matches when corpus actually ends with the token: {dollar_end_details:?}",
    );

    // $ end-anchor — (?m) opt-in: now matches line-end.
    let dollar_multiline =
        Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"(?m)apply_cell_config$");
    let dollar_multi_details = dollar_multiline.evaluate_scx_bpf_error_match(dollar_mid, true);
    assert!(
        dollar_multi_details.is_empty(),
        "(?m) opt-in makes $ match line-end; mid-corpus token at line-end matches: {dollar_multi_details:?}",
    );

    // ^ start-anchor — default mode: anchors to string-start.
    let caret = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"^apply_cell_config");
    let caret_mid = "prefix junk\napply_cell_config more";
    let caret_mid_details = caret.evaluate_scx_bpf_error_match(caret_mid, true);
    assert_eq!(
        caret_mid_details.len(),
        1,
        "default-mode ^ anchors to STRING-start; mid-corpus token must NOT match: {caret_mid_details:?}",
    );

    // ^ start-anchor — default mode positive: matches when token IS at string-start.
    let caret_start = "apply_cell_config\nbut more after";
    let caret_start_details = caret.evaluate_scx_bpf_error_match(caret_start, true);
    assert!(
        caret_start_details.is_empty(),
        "default-mode ^ matches when corpus actually starts with the token: {caret_start_details:?}",
    );

    // ^ start-anchor — (?m) opt-in: now matches line-start.
    let caret_multiline =
        Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"(?m)^apply_cell_config");
    let caret_multi_details = caret_multiline.evaluate_scx_bpf_error_match(caret_mid, true);
    assert!(
        caret_multi_details.is_empty(),
        "(?m) opt-in makes ^ match line-start; mid-corpus token at line-start matches: {caret_multi_details:?}",
    );
}

/// Regex `.` does NOT match `\n` by default. Operators who want a
/// pattern to span lines must opt in with the inline `(?s)` (DOTALL)
/// flag. Pin all three cases:
///   (1) `apply.*EINVAL` fails when the two tokens are on separate lines,
///   (2) `apply.*EINVAL` matches when both are on the same line,
///   (3) `(?s)apply.*EINVAL` matches across line boundaries.
#[test]
fn evaluate_scx_bpf_error_match_regex_dot_excludes_newline_by_default() {
    let a = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"apply.*EINVAL");

    let split = "apply_cell_config triggered\n-EINVAL returned";
    let split_details = a.evaluate_scx_bpf_error_match(split, true);
    assert_eq!(
        split_details.len(),
        1,
        "default-mode . excludes \\n; pattern spanning lines must NOT match: {split_details:?}",
    );

    let same_line = "apply_cell_config triggered -EINVAL returned";
    let same_details = a.evaluate_scx_bpf_error_match(same_line, true);
    assert!(
        same_details.is_empty(),
        "default-mode . matches non-newline bytes; same-line pattern matches: {same_details:?}",
    );

    let a_dotall = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"(?s)apply.*EINVAL");
    let dotall_details = a_dotall.evaluate_scx_bpf_error_match(split, true);
    assert!(
        dotall_details.is_empty(),
        "(?s) DOTALL opt-in makes . cross newlines; split-line pattern matches: {dotall_details:?}",
    );
}

/// Each pattern that satisfies `is_match("")` gets its own
/// `#[should_panic]` test. Per-pattern tests give each rejection a
/// distinct backtrace target — when the rejection gate ever
/// weakens, the test for the specific pattern that slipped through
/// fails by name, telling the operator which case regressed. The
/// four pinned cases cover both no-op classes the gate rejects:
/// patterns that match every position (`a?` optional, `.*`
/// zero-or-more, `(?:)` empty non-capturing group) trivially pass
/// against any corpus, and the start-immediately-end anchor (`^$`)
/// trivially fails against any non-empty corpus. All four share
/// the `is_match("")` predicate — one check rejects both no-op
/// classes.
#[test]
#[should_panic(expected = "matches the empty string")]
fn evaluate_scx_bpf_error_match_rejects_vacuous_optional() {
    let _ = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"a?");
}

#[test]
#[should_panic(expected = "matches the empty string")]
fn evaluate_scx_bpf_error_match_rejects_vacuous_star() {
    let _ = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r".*");
}

#[test]
#[should_panic(expected = "matches the empty string")]
fn evaluate_scx_bpf_error_match_rejects_vacuous_empty_group() {
    let _ = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"(?:)");
}

#[test]
#[should_panic(expected = "matches the empty string")]
fn evaluate_scx_bpf_error_match_rejects_vacuous_anchored_empty() {
    let _ = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"^$");
}

/// `(a|)` is the alternation form of vacuous — distinct syntactic
/// family from the quantifier (`a?`, `.*`), empty-group (`(?:)`),
/// and anchor-pair (`^$`) families. A real-world footgun: an
/// operator writes `expect_matches(r"(error|)")` intending "match
/// the word error or anything else" without realizing the empty
/// alternation branch makes the whole pattern vacuous. Pinning
/// this case ensures the construction gate's `is_match("")`
/// predicate continues to catch the alternation form too.
#[test]
#[should_panic(expected = "matches the empty string")]
fn evaluate_scx_bpf_error_match_rejects_vacuous_alternation_with_empty_branch() {
    let _ = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches(r"(a|)");
}

/// The builder's construction-time validation is bypassed when an
/// `Assert` is built via struct-literal syntax (the field is `pub`).
/// In that case the evaluator's defense-in-depth catches invalid
/// regex syntax at first use — pinning that path here ensures a
/// hypothetical bypass route (a deserializer, a test helper that
/// constructs `Assert` directly, or any future code that field-
/// inits the struct) still surfaces a loud diagnostic rather than
/// silently no-op'ing. The builder-path equivalent
/// (`evaluate_scx_bpf_error_match_regex_invalid_pattern_panics_at_construction`)
/// pins the construction-time panic; this pins the evaluator-side
/// fallback for the bypass case.
#[test]
fn evaluate_scx_bpf_error_match_regex_invalid_pattern_via_direct_construction_fails_loudly() {
    let a = Assert {
        expect_scx_bpf_error_matches: Some("[unclosed"),
        ..Assert::NO_OVERRIDES
    };
    let details = a.evaluate_scx_bpf_error_match("corpus text", true);
    assert_eq!(details.len(), 1);
    let msg = &details[0].message;
    assert!(
        msg.contains("[unclosed"),
        "invalid-pattern diagnostic must name the pattern: {msg}",
    );
    assert!(
        msg.contains("regex compilation failed"),
        "invalid-pattern diagnostic must say the pattern failed to compile: {msg}",
    );
}

/// Regex literal whitespace (space, tab) in the pattern matches the
/// same bytes in the corpus — no normalization. A pattern with a
/// literal tab `\t` only matches a corpus with a literal tab, not a
/// corpus with a space at that position.
#[test]
fn evaluate_scx_bpf_error_match_regex_literal_whitespace_matches_byte_for_byte() {
    let a_tab = Assert::NO_OVERRIDES.expect_scx_bpf_error_matches("apply\tcell");

    let with_tab = "apply\tcell_config triggered";
    let tab_match = a_tab.evaluate_scx_bpf_error_match(with_tab, true);
    assert!(
        tab_match.is_empty(),
        "regex with literal tab must match corpus with literal tab: {tab_match:?}",
    );

    let with_space = "apply cell_config triggered";
    let space_no_match = a_tab.evaluate_scx_bpf_error_match(with_space, true);
    assert_eq!(
        space_no_match.len(),
        1,
        "regex with literal tab must NOT match corpus with a space in that position: {space_no_match:?}",
    );
}

// -- serde derive on Assert -------------------------------------------

/// Compile-time pin that `Assert` implements `Serialize +
/// Deserialize`. A future regression that drops either trait would
/// break sidecar carrying of threshold config; pin it here so the
/// drop trips at type-check time, not at the sidecar consumer's
/// deserialize failure.
#[test]
fn assert_implements_serialize_and_deserialize() {
    fn requires<T: serde::Serialize + serde::de::DeserializeOwned>() {}
    requires::<Assert>();
}

/// Full round-trip: every numeric / bool field on `Assert` survives
/// serialize → JSON → deserialize. The two reproducer-string fields
/// (expect_scx_bpf_error_contains / _matches) are intentionally marked
/// `#[serde(skip)]` because their `&'static str` shape can't be
/// deserialized (no lifetime context in the deserializer); the
/// strings are test-author static literals carried in the test
/// definition, not runtime data the sidecar needs. See the dedicated
/// `assert_serde_skips_reproducer_string_fields` test below for the
/// skip contract.
#[test]
fn assert_serde_roundtrip_preserves_every_non_skipped_field() {
    let a = Assert {
        not_starved: Some(true),
        isolation: Some(true),
        max_gap_ms: Some(1234),
        max_spread_pct: Some(12.5),
        max_throughput_cv: Some(0.42),
        min_work_rate: Some(100.0),
        max_p99_wake_latency_ns: Some(5_000_000),
        max_wake_latency_cv: Some(0.7),
        min_iteration_rate: Some(500.0),
        max_migration_ratio: Some(0.05),
        max_imbalance_ratio: Some(2.5),
        max_local_dsq_depth: Some(64),
        fail_on_stall: Some(true),
        sustained_samples: Some(3),
        max_fallback_rate: Some(0.01),
        max_keep_last_rate: Some(0.005),
        enforce_monitor_thresholds: true,
        min_page_locality: Some(0.95),
        max_cross_node_migration_ratio: Some(0.02),
        max_slow_tier_ratio: Some(0.1),
        expect_scx_bpf_error_contains: None,
        expect_scx_bpf_error_matches: None,
    };
    let json = serde_json::to_string(&a).unwrap();
    let b: Assert = serde_json::from_str(&json).unwrap();
    assert_eq!(a.not_starved, b.not_starved);
    assert_eq!(a.isolation, b.isolation);
    assert_eq!(a.max_gap_ms, b.max_gap_ms);
    assert_eq!(a.max_spread_pct, b.max_spread_pct);
    assert_eq!(a.max_throughput_cv, b.max_throughput_cv);
    assert_eq!(a.min_work_rate, b.min_work_rate);
    assert_eq!(a.max_p99_wake_latency_ns, b.max_p99_wake_latency_ns);
    assert_eq!(a.max_wake_latency_cv, b.max_wake_latency_cv);
    assert_eq!(a.min_iteration_rate, b.min_iteration_rate);
    assert_eq!(a.max_migration_ratio, b.max_migration_ratio);
    assert_eq!(a.max_imbalance_ratio, b.max_imbalance_ratio);
    assert_eq!(a.max_local_dsq_depth, b.max_local_dsq_depth);
    assert_eq!(a.fail_on_stall, b.fail_on_stall);
    assert_eq!(a.sustained_samples, b.sustained_samples);
    assert_eq!(a.max_fallback_rate, b.max_fallback_rate);
    assert_eq!(a.max_keep_last_rate, b.max_keep_last_rate);
    assert_eq!(a.enforce_monitor_thresholds, b.enforce_monitor_thresholds);
    assert_eq!(a.min_page_locality, b.min_page_locality);
    assert_eq!(
        a.max_cross_node_migration_ratio,
        b.max_cross_node_migration_ratio
    );
    assert_eq!(a.max_slow_tier_ratio, b.max_slow_tier_ratio);
}

// -- Outcome enum -----------------------------------------------------

/// Compile-time pin that `Outcome` implements the documented derive
/// set. A regression that drops Serialize/Deserialize (breaking
/// sidecar wire-format) or PartialEq/Eq (breaking stats-comparison
/// tooling) would trip at type-check time.
#[test]
fn outcome_implements_expected_traits() {
    fn requires<
        T: Clone + std::fmt::Debug + PartialEq + Eq + serde::Serialize + serde::de::DeserializeOwned,
    >() {
    }
    requires::<Outcome>();
}

/// `Outcome::merge` precedence on the **3-variant (Fail / Pass /
/// Skip) subset** of the full `Fail > Inconclusive > Pass > Skip`
/// lattice. Full 9-case commutative truth table over these three
/// variants; any cell flipping would indicate the sub-lattice
/// reordered. The `Inconclusive` arm of the full lattice is pinned
/// separately by
/// [`outcome_merge_precedence_inconclusive_above_pass_below_fail`].
/// A "Fail > Skip > Pass" alternative would flip the existing
/// `merge_skip_plus_pass_demotes_skip` test invariant; the
/// `Fail > Pass > Skip` ordering is the one this implementation
/// actually realizes on the three-variant subset.
#[test]
fn outcome_merge_precedence_three_variant_subset_fail_pass_skip() {
    let d = AssertDetail::new(DetailKind::Other, "payload");
    let pass = || Outcome::Pass;
    let skip = || Outcome::Skip(d.clone());
    let fail = || Outcome::Fail(d.clone());
    // Skip ∪ Pass → Pass (both orderings)
    assert!(skip().merge(pass()).is_pass());
    assert!(pass().merge(skip()).is_pass());
    // Skip ∪ Fail → Fail (both orderings)
    assert!(skip().merge(fail()).is_fail());
    assert!(fail().merge(skip()).is_fail());
    // Pass ∪ Fail → Fail (both orderings — any failure wins)
    assert!(pass().merge(fail()).is_fail());
    assert!(fail().merge(pass()).is_fail());
    // Identity cases
    assert!(pass().merge(pass()).is_pass());
    assert!(skip().merge(skip()).is_skip());
    assert!(fail().merge(fail()).is_fail());
}

/// `Outcome::{is_pass, is_skip, is_fail, is_inconclusive}` 4×4 truth
/// table. A regression that returned wrong polarity (e.g. `is_skip`
/// silently returning false for `Skip(_)` because of a pattern-match
/// typo) would trip here.
#[test]
fn outcome_accessor_truth_table() {
    let d = AssertDetail::new(DetailKind::Other, "x");
    assert!(Outcome::Pass.is_pass());
    assert!(!Outcome::Pass.is_skip());
    assert!(!Outcome::Pass.is_fail());
    assert!(!Outcome::Pass.is_inconclusive());

    assert!(!Outcome::Skip(d.clone()).is_pass());
    assert!(Outcome::Skip(d.clone()).is_skip());
    assert!(!Outcome::Skip(d.clone()).is_fail());
    assert!(!Outcome::Skip(d.clone()).is_inconclusive());

    assert!(!Outcome::Fail(d.clone()).is_pass());
    assert!(!Outcome::Fail(d.clone()).is_skip());
    assert!(Outcome::Fail(d.clone()).is_fail());
    assert!(!Outcome::Fail(d.clone()).is_inconclusive());

    assert!(!Outcome::Inconclusive(d.clone()).is_pass());
    assert!(!Outcome::Inconclusive(d.clone()).is_skip());
    assert!(!Outcome::Inconclusive(d.clone()).is_fail());
    assert!(Outcome::Inconclusive(d.clone()).is_inconclusive());
}

/// `Outcome::merge` precedence around the `Inconclusive` variant.
/// Lattice: `Fail > Inconclusive > Pass > Skip`. An Inconclusive
/// dominates Pass and Skip (because "couldn't evaluate" is not a
/// real Pass), but Fail still dominates Inconclusive (a real
/// failure beats an unevaluated check). Pins the precedence rule
/// the verdict pipeline relies on for zero-denominator ratio
/// asserts.
#[test]
fn outcome_merge_precedence_inconclusive_above_pass_below_fail() {
    let d = AssertDetail::new(DetailKind::Other, "payload");
    let pass = || Outcome::Pass;
    let skip = || Outcome::Skip(d.clone());
    let fail = || Outcome::Fail(d.clone());
    let inconc = || Outcome::Inconclusive(d.clone());
    // Inconclusive ∪ Pass → Inconclusive (both orderings)
    assert!(inconc().merge(pass()).is_inconclusive());
    assert!(pass().merge(inconc()).is_inconclusive());
    // Inconclusive ∪ Skip → Inconclusive (both orderings)
    assert!(inconc().merge(skip()).is_inconclusive());
    assert!(skip().merge(inconc()).is_inconclusive());
    // Inconclusive ∪ Fail → Fail (both orderings — Fail dominates)
    assert!(inconc().merge(fail()).is_fail());
    assert!(fail().merge(inconc()).is_fail());
    // Identity (any payload survives — same-discriminant merge).
    assert!(inconc().merge(inconc()).is_inconclusive());

    // LEFT-payload-tie pin: same-discriminant Inconclusive merges
    // preserve the LEFT operand's payload (mirrors the Fail+Fail and
    // Skip+Skip semantic). A regression to RIGHT-wins would silently
    // change which Inconclusive reason surfaces in the terminal
    // verdict.
    let left = AssertDetail::new(DetailKind::Migration, "first-inconc");
    let right = AssertDetail::new(DetailKind::Benchmark, "second-inconc");
    let Outcome::Inconclusive(d) = Outcome::Inconclusive(left).merge(Outcome::Inconclusive(right))
    else {
        panic!("Inconclusive+Inconclusive must yield Inconclusive");
    };
    assert_eq!(d.kind, DetailKind::Migration, "LEFT-wins on payload tie");
    assert!(d.message.contains("first-inconc"));
}

/// `AssertResult::record_inconclusive` appends an `Outcome::Inconclusive`
/// and the payload surfaces via `inconclusive_details()`. Pins that
/// a Fail-plus-Inconclusive stream reads as Fail (Fail dominates) and
/// that `is_pass()` returns false for an Inconclusive-only stream so a
/// zero-denominator ratio doesn't slip past CI gates as Pass.
#[test]
fn record_inconclusive_appears_in_inconclusive_details_and_is_not_pass() {
    let mut r = AssertResult::pass();
    r.record_inconclusive(AssertDetail::new(
        DetailKind::Migration,
        "denominator was zero",
    ));
    assert!(!r.is_pass(), "inconclusive must not read as pass");
    assert!(!r.is_fail());
    assert!(!r.is_skip());
    assert!(r.is_inconclusive());
    let reasons: Vec<_> = r.inconclusive_details().collect();
    assert_eq!(reasons.len(), 1);
    assert_eq!(reasons[0].kind, DetailKind::Migration);
    assert!(reasons[0].message.contains("denominator was zero"));
    // failure_details must NOT include Inconclusive payloads — they
    // are a sibling iterator. A regression that folded them in would
    // misclassify the verdict as a Fail.
    assert_eq!(r.failure_details().count(), 0);

    // Fail dominates Inconclusive — adding a Fail to an Inconclusive
    // stream flips is_fail() true and clears is_inconclusive() per
    // the `Fail > Inconclusive > Pass > Skip` precedence.
    r.record_fail(AssertDetail::new(DetailKind::Other, "real failure"));
    assert!(r.is_fail());
    assert!(!r.is_inconclusive(), "Fail dominates Inconclusive");
}

/// `AssertResult::outcome()` folds the `outcomes: Vec<Outcome>` slot
/// into a single terminal verdict per `Outcome::merge`'s precedence
/// (`Fail > Inconclusive > Pass > Skip`; identity Pass). Pins the
/// three non-Inconclusive arms of the synthesis rules so consumers
/// that read `outcome()` see Pass / Skip / Fail without depending
/// on the inner vec structure. The Inconclusive arm is pinned by
/// [`record_inconclusive_appears_in_inconclusive_details_and_is_not_pass`].
#[test]
fn assert_result_outcome_folds_outcomes_vec() {
    // Pass result with no notable details → Outcome::Pass.
    assert!(AssertResult::pass().outcome().is_pass());

    // Skip result → Outcome::Skip carrying the Skip-kind detail.
    let skip_result = AssertResult::skip("topology missing");
    let Outcome::Skip(d) = skip_result.outcome() else {
        panic!("expected Outcome::Skip, got {:?}", skip_result.outcome());
    };
    assert_eq!(d.kind, DetailKind::Skip);
    assert!(d.message.contains("topology missing"));

    // Fail result → Outcome::Fail carrying the non-Skip detail.
    let fail_result = AssertResult::fail(AssertDetail::new(DetailKind::Starved, "boom"));
    let Outcome::Fail(d) = fail_result.outcome() else {
        panic!("expected Outcome::Fail, got {:?}", fail_result.outcome());
    };
    assert_eq!(d.kind, DetailKind::Starved);
    assert!(d.message.contains("boom"));
}

/// Mixed `Pass + Skip` stream: `outcome()` must return
/// `Outcome::Pass`, NOT Skip. Pins the inner-else branch in the
/// outcome() folder — a real Pass marker alongside any Skip beats
/// the Skip per the "Pass demotes Skip" semantic that
/// `merge_skip_plus_explicit_pass_demotes_skip` already pins on
/// `is_pass()`. A regression that returned `Outcome::Skip` here
/// would silently downgrade a passing scenario's terminal verdict.
#[test]
fn assert_result_outcome_pass_plus_skip_is_pass_not_skip() {
    let mut r = AssertResult::pass();
    r.record_pass();
    r.record_skip("optional probe");
    assert!(matches!(r.outcome(), Outcome::Pass));
    assert!(r.is_pass());
    assert!(!r.is_skip());
}

/// Multi-Skip stream: `outcome()` returns the FIRST Skip's payload
/// (mirrors `Outcome::merge`'s left-wins payload-tie semantic). Pins
/// the .next() iterator in the all-Skip branch — a regression to
/// .last() would silently change which reason surfaces in the
/// terminal verdict.
#[test]
fn assert_result_outcome_multi_skip_returns_first_payload() {
    let mut r = AssertResult::pass();
    r.record_skip("first");
    r.record_skip("second");
    r.record_skip("third");
    let Outcome::Skip(d) = r.outcome() else {
        panic!("expected Skip, got {:?}", r.outcome());
    };
    assert!(
        d.message.contains("first"),
        "first-Skip-wins; got: {}",
        d.message
    );
}

/// `Outcome::as_ref` projection preserves the discriminant +
/// borrows the payload in place. Pins the no-clone contract that
/// the verdict-read fast path ([`AssertResult::outcome_ref`])
/// relies on. (Borrow-in-place is enforced by the type system —
/// `OutcomeRef::Fail(&'a AssertDetail)` cannot construct without
/// borrowing from the source.)
#[test]
fn outcome_as_ref_preserves_discriminant_and_payload() {
    assert!(matches!(Outcome::Pass.as_ref(), OutcomeRef::Pass));
    let fail = Outcome::Fail(AssertDetail::new(DetailKind::Starved, "boom"));
    let OutcomeRef::Fail(d) = fail.as_ref() else {
        panic!("Fail as_ref should be Fail variant");
    };
    assert_eq!(d.kind, DetailKind::Starved);
    assert!(d.message.contains("boom"));
    let skip = Outcome::Skip(AssertDetail::new(DetailKind::Skip, "missing"));
    let OutcomeRef::Skip(d) = skip.as_ref() else {
        panic!("Skip as_ref should be Skip variant");
    };
    assert_eq!(d.kind, DetailKind::Skip);
    assert!(d.message.contains("missing"));
    let inconc =
        Outcome::Inconclusive(AssertDetail::new(DetailKind::Migration, "zero denominator"));
    let OutcomeRef::Inconclusive(d) = inconc.as_ref() else {
        panic!("Inconclusive as_ref should be Inconclusive variant");
    };
    assert_eq!(d.kind, DetailKind::Migration);
    assert!(d.message.contains("zero denominator"));
}

/// `AssertResult::outcome_ref` matches `outcome()` shape across
/// every branch (Pass identity / non-empty all-Skip / mixed-Pass-
/// plus-Skip / Fail wins) while borrowing the payload instead of
/// cloning. Pins the borrowed-fast-path semantic against the
/// owned accessor so a regression that drifted one but not the
/// other trips here.
#[test]
fn assert_result_outcome_ref_matches_owned_outcome_shape() {
    // Pass identity: empty stream.
    assert!(matches!(
        AssertResult::pass().outcome_ref(),
        OutcomeRef::Pass
    ));
    // Non-empty all-Skip → Skip with first payload.
    let mut all_skip = AssertResult::pass();
    all_skip.record_skip("only-skip");
    let OutcomeRef::Skip(d) = all_skip.outcome_ref() else {
        panic!("all-Skip stream should yield Skip");
    };
    assert!(d.message.contains("only-skip"));
    // Mixed Pass + Skip → Pass (Pass demotes Skip per outcome()
    // inner-else branch).
    let mut mixed = AssertResult::pass();
    mixed.record_pass();
    mixed.record_skip("demoted");
    assert!(matches!(mixed.outcome_ref(), OutcomeRef::Pass));
    // Any Fail → Fail with first Fail's payload.
    let mut fail = AssertResult::pass();
    fail.record_fail(AssertDetail::new(DetailKind::Stuck, "first-fail"));
    fail.record_fail(AssertDetail::new(DetailKind::Starved, "second-fail"));
    let OutcomeRef::Fail(d) = fail.outcome_ref() else {
        panic!("any-Fail stream should yield Fail");
    };
    assert_eq!(d.kind, DetailKind::Stuck);
    assert!(d.message.contains("first-fail"));
    // Non-empty all-Inconclusive → Inconclusive with first payload
    // (mirrors the all-Skip branch and the `Outcome::merge` LEFT-
    // wins payload-tie semantic).
    let mut all_inconc = AssertResult::pass();
    all_inconc.record_inconclusive(AssertDetail::new(DetailKind::Migration, "first-inconc"));
    all_inconc.record_inconclusive(AssertDetail::new(DetailKind::Benchmark, "second-inconc"));
    let OutcomeRef::Inconclusive(d) = all_inconc.outcome_ref() else {
        panic!("all-Inconclusive stream should yield Inconclusive");
    };
    assert_eq!(d.kind, DetailKind::Migration);
    assert!(d.message.contains("first-inconc"));
    // Fail dominates Inconclusive: a Fail-plus-Inconclusive stream
    // yields Fail (per `Fail > Inconclusive > Pass > Skip`).
    let mut fail_over_inconc = AssertResult::pass();
    fail_over_inconc
        .record_inconclusive(AssertDetail::new(DetailKind::Migration, "denominator-zero"));
    fail_over_inconc.record_fail(AssertDetail::new(DetailKind::Stuck, "real-fail"));
    let OutcomeRef::Fail(d) = fail_over_inconc.outcome_ref() else {
        panic!("Fail+Inconclusive stream should yield Fail");
    };
    assert_eq!(d.kind, DetailKind::Stuck);
    assert!(d.message.contains("real-fail"));
    // Inconclusive dominates Skip: an Inconclusive-plus-Skip stream
    // (no Fail, no Pass) yields Inconclusive (per
    // `Fail > Inconclusive > Pass > Skip`). LEFT-wins payload-tie
    // semantics carry the first Inconclusive's detail through, even
    // though the Skip arrived after.
    let mut inconc_over_skip = AssertResult::pass();
    inconc_over_skip.record_inconclusive(AssertDetail::new(DetailKind::Benchmark, "left-inconc"));
    inconc_over_skip.record_skip("right-skip");
    let OutcomeRef::Inconclusive(d) = inconc_over_skip.outcome_ref() else {
        panic!(
            "Inconclusive+Skip stream should yield Inconclusive (Inconclusive > Skip): got {:?}",
            inconc_over_skip.outcome_ref()
        );
    };
    assert_eq!(d.kind, DetailKind::Benchmark);
    assert!(d.message.contains("left-inconc"));
    // Reverse order (Skip first, Inconclusive second) must yield the
    // same verdict — merge is commutative on lattice precedence even
    // though LEFT-wins on payload ties. With distinct ranks the rank
    // alone decides the variant, and the Inconclusive payload wins
    // because it is the only Inconclusive in the stream.
    let mut skip_then_inconc = AssertResult::pass();
    skip_then_inconc.record_skip("left-skip");
    skip_then_inconc.record_inconclusive(AssertDetail::new(DetailKind::Benchmark, "right-inconc"));
    let OutcomeRef::Inconclusive(d) = skip_then_inconc.outcome_ref() else {
        panic!(
            "Skip+Inconclusive stream should still yield Inconclusive: got {:?}",
            skip_then_inconc.outcome_ref()
        );
    };
    assert_eq!(d.kind, DetailKind::Benchmark);
    assert!(d.message.contains("right-inconc"));
}

/// Mutator semantics: repeated `record_fail` calls append distinct
/// Fail outcomes onto the vec; `outcome()` folds them and the
/// result is Fail with the LEFT operand's payload winning per
/// `Outcome::merge`'s payload-tie semantics.
#[test]
fn assert_result_record_fail_appends_and_folds_fail() {
    let mut r = AssertResult::pass();
    assert!(r.is_pass(), "fresh AssertResult::pass is_pass");
    r.record_fail(AssertDetail::new(DetailKind::Starved, "first"));
    r.record_fail(AssertDetail::new(DetailKind::Stuck, "second"));
    assert_eq!(r.outcomes.len(), 2);
    assert!(r.is_fail());
    let Outcome::Fail(d) = r.outcome() else {
        panic!("expected Outcome::Fail, got {:?}", r.outcome());
    };
    // LEFT-wins on Fail+Fail ties: first record_fail's detail wins.
    assert_eq!(d.kind, DetailKind::Starved);
    assert!(d.message.contains("first"));
    // Iteration helper surfaces both fails.
    let collected: Vec<&AssertDetail> = r.failure_details().collect();
    assert_eq!(collected.len(), 2);
}

/// `record_skip` appends a Skip outcome carrying the reason; mixed
/// Skip + Pass = Pass (Pass dominates Skip in the merge fold).
#[test]
fn assert_result_record_skip_then_pass_marker_yields_pass() {
    let mut r = AssertResult::pass();
    r.record_skip("topology mismatch");
    assert!(r.is_skip(), "skip-only stream is is_skip");
    r.record_pass();
    // Skip + Pass = Pass (the Pass entry beats the Skip per merge precedence).
    assert!(r.is_pass(), "Skip + Pass folds to Pass");
    assert!(!r.is_skip(), "is_skip requires all-Skip + non-empty");
}

/// `is_skip()` returns FALSE on empty outcomes (Pass identity, not
/// vacuous skip). Pins the "empty = Pass" convention so a future
/// refactor flipping is_skip to vacuous-all-skip semantics trips
/// loudly.
#[test]
fn assert_result_empty_outcomes_is_pass_not_skip() {
    let r = AssertResult::pass();
    assert!(r.outcomes.is_empty());
    assert!(r.is_pass());
    assert!(!r.is_skip());
    assert!(!r.is_fail());
}

/// `record_outcome` escape hatch pushes a pre-folded [`Outcome`]
/// onto the stream. Pins the surface so the dead-code lint doesn't
/// strip it, and verifies it composes with the variant-specific
/// mutators (record_outcome of a Fail is observable via is_fail()
/// and failure_details() identically to a record_fail call).
#[test]
fn assert_result_record_outcome_pushes_and_observable() {
    let mut r = AssertResult::pass();
    let d = AssertDetail::new(DetailKind::Other, "external verdict");
    r.record_outcome(Outcome::Fail(d.clone()));
    assert_eq!(r.outcomes.len(), 1);
    assert!(r.is_fail());
    let collected: Vec<&AssertDetail> = r.failure_details().collect();
    assert_eq!(collected.len(), 1);
    assert!(collected[0].message.contains("external verdict"));
    // Composes with record_outcome of a Skip: Fail still dominates.
    r.record_outcome(Outcome::Skip(AssertDetail::new(DetailKind::Skip, "stop")));
    assert_eq!(r.outcomes.len(), 2);
    assert!(r.is_fail(), "Fail dominates the merged outcome");
    assert_eq!(r.skip_details().count(), 1);
}

/// `Outcome` serde uses the externally-tagged default (no
/// `#[serde(tag, content)]`). The adjacently-tagged style was
/// dropped because postcard — the wire format for the AssertResult
/// TLV in [`crate::test_support::output::parse_assert_result_from_drain`]
/// and [`crate::test_support::test_helpers::assert_result_tlv_entry`]
/// — is not self-describing and cannot decode adjacently-tagged
/// enums. Pin both the JSON shape and the postcard roundtrip so a
/// refactor that re-adds `#[serde(tag, content)]` trips loudly on
/// the wire format that consumers actually depend on.
#[test]
fn outcome_serde_externally_tagged_roundtrips_via_json_and_postcard() {
    let d = AssertDetail::new(DetailKind::Other, "msg");

    // JSON shape: unit variant serializes as the bare variant
    // name; data variants serialize as `{"<Variant>": {...}}`.
    let pass_json = serde_json::to_string(&Outcome::Pass).unwrap();
    assert_eq!(
        pass_json, "\"Pass\"",
        "Pass must serialize as bare variant name"
    );
    let fail_json = serde_json::to_string(&Outcome::Fail(d.clone())).unwrap();
    assert!(
        fail_json.starts_with("{\"Fail\":"),
        "Fail must serialize as externally-tagged object: {fail_json}"
    );
    // Absence-check for the dropped adjacently-tagged shape: the
    // outer object's first key must be the variant name, not the
    // prior `"kind"` discriminator, and there must be no outer
    // `"data"` field wrapping the payload. (The inner
    // `AssertDetail` still has a `kind: DetailKind` field — that's
    // unrelated to Outcome's tagging.)
    assert!(
        !fail_json.starts_with("{\"kind\":"),
        "Fail outer key must be the variant name, not the dropped \"kind\" tag: {fail_json}"
    );
    assert!(
        !fail_json.contains("\"data\":"),
        "Fail must not carry the dropped \"data\" wrapper: {fail_json}"
    );

    // JSON roundtrip.
    let recovered: Outcome = serde_json::from_str(&fail_json).unwrap();
    assert!(recovered.is_fail());

    // Inconclusive serializes with the same externally-tagged shape
    // — `{"Inconclusive":{...}}` — and roundtrips via JSON. Pin
    // both so a regression that drops Inconclusive from the wire
    // format or re-adds `#[serde(tag, content)]` trips loudly.
    let inconc_detail = AssertDetail::new(DetailKind::Migration, "zero denom");
    let inconc_json = serde_json::to_string(&Outcome::Inconclusive(inconc_detail.clone())).unwrap();
    assert!(
        inconc_json.starts_with("{\"Inconclusive\":"),
        "Inconclusive must serialize as externally-tagged object: {inconc_json}"
    );
    assert!(
        !inconc_json.contains("\"data\":"),
        "Inconclusive must not carry the dropped \"data\" wrapper: {inconc_json}"
    );
    let inconc_recovered: Outcome = serde_json::from_str(&inconc_json).unwrap();
    assert!(inconc_recovered.is_inconclusive());

    // Postcard roundtrip — the wire format that the TLV path uses.
    // A regression that re-adds `#[serde(tag, content)]` would
    // silently break this and surface only at runtime as
    // ERR_NO_TEST_FUNCTION_OUTPUT.
    let pc_pass = postcard::to_stdvec(&Outcome::Pass).unwrap();
    let pc_pass_recovered: Outcome = postcard::from_bytes(&pc_pass).unwrap();
    assert!(pc_pass_recovered.is_pass());
    let pc_fail = postcard::to_stdvec(&Outcome::Fail(d)).unwrap();
    let pc_fail_recovered: Outcome = postcard::from_bytes(&pc_fail).unwrap();
    assert!(pc_fail_recovered.is_fail());
    let pc_inconc = postcard::to_stdvec(&Outcome::Inconclusive(inconc_detail)).unwrap();
    let pc_inconc_recovered: Outcome = postcard::from_bytes(&pc_inconc).unwrap();
    assert!(pc_inconc_recovered.is_inconclusive());
}

/// Wire-format byte-sentinel: postcard encodes `Outcome` by
/// **variant index** (Pass=0, Skip=1, Inconclusive=2, Fail=3), and
/// that index is the FIRST byte of the encoded payload (varint,
/// single byte for indices < 128). A refactor that reorders,
/// removes, or inserts a variant ahead of one of these four would
/// shift the leading byte and silently reinterpret guest payloads
/// on the host — a Pass byte becoming Skip, for example, with no
/// decode error.
///
/// Pin the leading byte for every variant so the silent-shift
/// regression trips at test time rather than at production
/// runtime. The Outcome enum doc explicitly documents this
/// stability contract ("Wire-format stability (postcard variant
/// index)" section) — this test is the enforcement for that
/// contract.
#[test]
fn outcome_postcard_variant_index_byte_sentinel() {
    let d = AssertDetail::new(DetailKind::Other, "x");

    let pc_pass = postcard::to_stdvec(&Outcome::Pass).unwrap();
    assert_eq!(
        pc_pass.first().copied(),
        Some(0u8),
        "Outcome::Pass MUST encode with leading variant-index byte 0 (got bytes {pc_pass:?}); \
         a variant reorder would silently corrupt the TLV wire path",
    );

    let pc_skip = postcard::to_stdvec(&Outcome::Skip(d.clone())).unwrap();
    assert_eq!(
        pc_skip.first().copied(),
        Some(1u8),
        "Outcome::Skip MUST encode with leading variant-index byte 1 (got bytes {pc_skip:?})",
    );

    let pc_inconc = postcard::to_stdvec(&Outcome::Inconclusive(d.clone())).unwrap();
    assert_eq!(
        pc_inconc.first().copied(),
        Some(2u8),
        "Outcome::Inconclusive MUST encode with leading variant-index byte 2 \
         (got bytes {pc_inconc:?})",
    );

    let pc_fail = postcard::to_stdvec(&Outcome::Fail(d)).unwrap();
    assert_eq!(
        pc_fail.first().copied(),
        Some(3u8),
        "Outcome::Fail MUST encode with leading variant-index byte 3 (got bytes {pc_fail:?})",
    );
}

/// `expect_scx_bpf_error_contains` and `expect_scx_bpf_error_matches`
/// are `#[serde(skip)]` because `Option<&'static str>` cannot
/// round-trip through a borrowed deserializer (no source-string
/// lifetime to bind to). Pin both invariants:
///  1. Serialize omits the two reproducer fields from JSON output
///     entirely (no key, no null).
///  2. Deserialize of a previously-serialized Assert that HAD those
///     fields populated produces `None` for both — the skipped
///     fields default per `Option::default() == None`.
///
/// The sidecar use case for `Assert` serde is "threshold config
/// roundtrip for stats comparison." Reproducer matcher strings are
/// part of the test definition itself (not per-run data), so
/// skipping them keeps the wire format clean.
///
/// Switching the field type to `Option<Cow<'static, str>>` (which
/// can deserialize) was considered but rejected: `Cow` is not
/// `Copy`, which cascade-breaks `declare_scheduler!`'s const-fn
/// `Scheduler::assert` macro path used by tests that build `Assert`
/// values in `const` / `static` initializers. Keeping
/// `Option<&'static str>` + `#[serde(skip)]` preserves the const-
/// construction surface; the reproducer matcher strings are test-
/// author static literals carried in the test definition rather than
/// runtime data the sidecar needs to round-trip. A future
/// decomposition into a `ReproducerMatchers` sub-config may revisit
/// this if a use case (e.g. sidecar-loaded test definitions) needs
/// the strings to survive serialization end-to-end.
#[test]
fn assert_serde_skips_reproducer_string_fields() {
    let a = Assert::NO_OVERRIDES
        .expect_scx_bpf_error_contains("apply_cell_config")
        .expect_scx_bpf_error_matches(r"\bEINVAL\b");
    let json = serde_json::to_string(&a).unwrap();
    assert!(
        !json.contains("expect_scx_bpf_error_contains"),
        "expect_scx_bpf_error_contains must NOT appear in serialized JSON; got: {json}"
    );
    assert!(
        !json.contains("expect_scx_bpf_error_matches"),
        "expect_scx_bpf_error_matches must NOT appear in serialized JSON; got: {json}"
    );
    let b: Assert = serde_json::from_str(&json).unwrap();
    assert_eq!(
        b.expect_scx_bpf_error_contains, None,
        "deserialized reproducer-contains field must default to None"
    );
    assert_eq!(
        b.expect_scx_bpf_error_matches, None,
        "deserialized reproducer-matches field must default to None"
    );
}
