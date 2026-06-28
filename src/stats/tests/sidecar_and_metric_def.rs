use super::*;
use crate::assert::ScenarioStats;

// -- sidecar_to_row tests --

#[test]
fn sidecar_to_row_basic() {
    use crate::monitor;
    use crate::test_support;
    let sc = test_support::SidecarResult {
        test_name: "my_test".to_string(),
        topology: "1n2l4c2t".to_string(),
        scheduler: "scx_mitosis".to_string(),
        stats: ScenarioStats {
            cgroups: vec![],
            total_workers: 4,
            total_cpus: 8,
            total_migrations: 12,
            worst_spread: 15.0,
            worst_gap_ms: 200,
            worst_gap_cpu: 3,
            ..Default::default()
        },
        monitor: Some(monitor::MonitorSummary {
            total_samples: 10,
            max_imbalance_ratio: 2.5,
            max_local_dsq_depth: 4,
            stuck_count: 1,
            event_deltas: Some(monitor::ScxEventDeltas {
                total_fallback: 7,
                fallback_rate: 0.5,
                max_fallback_burst: 2,
                total_dispatch_offline: 0,
                total_dispatch_keep_last: 3,
                keep_last_rate: 0.2,
                total_enq_skip_exiting: 0,
                total_enq_skip_migration_disabled: 0,
                ..Default::default()
            }),
            schedstat_deltas: None,
            prog_stats_deltas: None,
            ..Default::default()
        }),
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(row.scenario, "my_test");
    assert_eq!(row.topology, "1n2l4c2t");
    assert!(row.is_pass());
    assert_eq!(row.spread, 15.0);
    assert_eq!(row.gap_ms, 200);
    assert_eq!(row.migrations, 12);
    assert_eq!(row.imbalance_ratio, 2.5);
    assert_eq!(row.max_dsq_depth, 4);
    assert_eq!(row.stuck_count, 1.0);
    assert_eq!(row.fallback_count, 7);
    assert_eq!(row.keep_last_count, 3);
}

/// `worst_iterations_per_cpu_sec` (the overcommit-invariant compare
/// metric the budget warning recommends) flows ScenarioStats ->
/// worst_iterations_per_cpu_sec is now a `MetricKind::WorstLowest` ext
/// metric (re-pooled post-merge by populate_run_distribution_metrics).
/// sidecar_to_row carries it through the ext_metrics copy, and
/// MetricDef::read surfaces it via the `|_| None` accessor's ext fallback;
/// an absent key (no cgroup reported a defined rate) reads None, distinct
/// from a measured 0.0. The metric stays registered so
/// `stats compare --metric worst_iterations_per_cpu_sec` resolves.
#[test]
fn sidecar_to_row_carries_worst_iterations_per_cpu_sec_via_ext() {
    use crate::test_support;
    let mut stats = ScenarioStats::default();
    stats
        .ext_metrics
        .insert("worst_iterations_per_cpu_sec".to_string(), 1234.5);
    let present = test_support::SidecarResult {
        stats,
        ..test_support::SidecarResult::test_fixture()
    };
    let def = metric_def("worst_iterations_per_cpu_sec")
        .expect("metric must be registered so `stats compare --metric` resolves it");
    assert_eq!(def.read(&sidecar_to_row(&present)), Some(1234.5));

    // Absent key (re-pool wrote nothing) → read None, NOT a measured 0.0.
    let absent = test_support::SidecarResult {
        stats: ScenarioStats::default(),
        ..test_support::SidecarResult::test_fixture()
    };
    assert_eq!(def.read(&sidecar_to_row(&absent)), None);
}

/// The host-side monitor schedstat aggregates flow into
/// `GauntletRow.ext_metrics` as the seven raw `Polarity::Informational`
/// counters; each is registered so `MetricDef::read` surfaces it via the
/// `|_| None` ext fallback and `classify_direction` returns `None` (never
/// gated). When `schedstat_deltas` is `None` (CONFIG_SCHEDSTATS off) the
/// keys are ABSENT, not 0 — a 0 would pollute the cross-run Counter sum and
/// the Rate denominators.
#[test]
fn sidecar_to_row_carries_monitor_schedstat_ext_counters() {
    use crate::monitor;
    use crate::test_support;
    let sc = test_support::SidecarResult {
        monitor: Some(monitor::MonitorSummary {
            schedstat_deltas: Some(monitor::SchedstatDeltas {
                total_run_delay: 6000,
                total_pcount: 3,
                total_sched_count: 100,
                total_yld_count: 5,
                total_sched_goidle: 20,
                total_ttwu_count: 200,
                total_ttwu_local: 150,
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    for (name, want) in [
        ("total_run_delay", 6000.0),
        ("total_pcount", 3.0),
        ("total_sched_count", 100.0),
        ("total_yld_count", 5.0),
        ("total_sched_goidle", 20.0),
        ("total_ttwu_count", 200.0),
        ("total_ttwu_local", 150.0),
    ] {
        assert_eq!(row.ext_metrics.get(name).copied(), Some(want), "{name} ext key");
        let def = metric_def(name).unwrap_or_else(|| panic!("{name} registered"));
        assert_eq!(def.read(&row), Some(want), "{name} surfaced via ext fallback");
        assert_eq!(def.classify_direction(), None, "{name} is informational (never gates)");
    }

    // schedstat_deltas == None => keys ABSENT (not 0).
    let no_sd = test_support::SidecarResult {
        monitor: Some(monitor::MonitorSummary {
            schedstat_deltas: None,
            ..Default::default()
        }),
        ..test_support::SidecarResult::test_fixture()
    };
    let row2 = sidecar_to_row(&no_sd);
    assert!(!row2.ext_metrics.contains_key("total_run_delay"));
    assert!(!row2.ext_metrics.contains_key("total_ttwu_count"));
}

/// `avg_nr_running` (whole-run mean runqueue occupancy) flows from
/// `MonitorSummary::avg_nr_running` into `GauntletRow.ext_metrics` as a
/// registered `Gauge(Avg)` / LowerBetter metric, surfaced via the `|_| None`
/// ext fallback. Absent (not 0.0) when the run has no monitor samples — a
/// 0-sample run carries no occupancy signal.
#[test]
fn sidecar_to_row_carries_avg_nr_running_when_sampled() {
    use crate::monitor;
    use crate::test_support;
    let sc = test_support::SidecarResult {
        monitor: Some(monitor::MonitorSummary {
            total_samples: 10,
            avg_nr_running: 2.5,
            ..Default::default()
        }),
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(row.ext_metrics.get("avg_nr_running").copied(), Some(2.5));
    let def = metric_def("avg_nr_running").expect("avg_nr_running registered");
    assert_eq!(def.read(&row), Some(2.5));
    // Directional (LowerBetter), NOT informational: classify_direction Some(true).
    assert_eq!(def.classify_direction(), Some(true));

    // No samples => ABSENT (MonitorSummary.avg_nr_running defaults to 0.0, but a
    // 0-sample run has no occupancy signal — absent, not a false 0.0).
    let no_samples = test_support::SidecarResult {
        monitor: Some(monitor::MonitorSummary {
            total_samples: 0,
            avg_nr_running: 0.0,
            ..Default::default()
        }),
        ..test_support::SidecarResult::test_fixture()
    };
    assert!(
        !sidecar_to_row(&no_samples)
            .ext_metrics
            .contains_key("avg_nr_running")
    );
}

#[test]
fn sidecar_to_row_no_monitor() {
    use crate::test_support;
    let sc = test_support::SidecarResult {
        test_name: "eevdf_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        passed: false,
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(row.scenario, "eevdf_test");
    assert!(row.is_fail());
    assert_eq!(row.imbalance_ratio, 0.0);
    assert_eq!(row.max_dsq_depth, 0);
    assert_eq!(row.stuck_count, 0.0);
    assert_eq!(row.fallback_count, 0);
    assert_eq!(row.keep_last_count, 0);
}

/// `sidecar_to_row` must copy `SidecarResult::project_commit`
/// into `GauntletRow::commit` verbatim so the typed
/// `--project-commit` filter (and any sibling slicers added
/// later) see the value the sidecar writer recorded. A
/// regression that left the field at the
/// `Option::default()` (`None`) would silently drop the
/// commit dimension from every comparison even when the
/// sidecar had a populated value. Pinned for `None`, clean
/// `Some` (no suffix), and dirty `Some` (`-dirty` suffix) to
/// catch a regression that special-cases one shape and not
/// the others — e.g. one that stripped the suffix when copying.
#[test]
fn sidecar_to_row_propagates_project_commit() {
    use crate::test_support;
    let sc_dirty = test_support::SidecarResult {
        test_name: "commit_dirty_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        project_commit: Some("abcdef1-dirty".to_string()),
        ..test_support::SidecarResult::test_fixture()
    };
    let row_dirty = sidecar_to_row(&sc_dirty);
    assert_eq!(
        row_dirty.commit.as_deref(),
        Some("abcdef1-dirty"),
        "populated dirty project_commit must propagate \
             verbatim, including the `-dirty` suffix",
    );

    let sc_clean = test_support::SidecarResult {
        test_name: "commit_clean_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        project_commit: Some("abcdef1".to_string()),
        ..test_support::SidecarResult::test_fixture()
    };
    let row_clean = sidecar_to_row(&sc_clean);
    assert_eq!(
        row_clean.commit.as_deref(),
        Some("abcdef1"),
        "populated clean project_commit (no `-dirty` suffix) \
             must propagate verbatim — a regression that always \
             appended `-dirty` or always stripped a tail would \
             surface here independently of the dirty case above",
    );

    let sc_none = test_support::SidecarResult {
        test_name: "no_commit_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        project_commit: None,
        ..test_support::SidecarResult::test_fixture()
    };
    let row_none = sidecar_to_row(&sc_none);
    assert!(
        row_none.commit.is_none(),
        "absent project_commit must propagate as None — a \
             regression substituting an empty string would dilute \
             every `--project-commit` filter into matching all None rows",
    );
}

/// `sidecar_to_row` must copy `SidecarResult::kernel_commit`
/// into `GauntletRow::kernel_commit` verbatim so the typed
/// `--kernel-commit` filter and per-side
/// `--a-kernel-commit` / `--b-kernel-commit` slicers see the
/// value the sidecar writer recorded. A regression that left
/// the field at the `Option::default()` (`None`) would
/// silently drop the kernel-commit dimension from every
/// comparison even when the sidecar had a populated value.
/// Mirrors `sidecar_to_row_propagates_project_commit` for
/// the kernel_commit field; pinned for `None`, clean `Some`
/// (no suffix), and dirty `Some` (`-dirty` suffix) to catch
/// a regression that special-cases one shape and not the
/// others.
#[test]
fn sidecar_to_row_propagates_kernel_commit() {
    use crate::test_support;
    let sc_dirty = test_support::SidecarResult {
        test_name: "kc_dirty_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        kernel_commit: Some("kabcde7-dirty".to_string()),
        ..test_support::SidecarResult::test_fixture()
    };
    let row_dirty = sidecar_to_row(&sc_dirty);
    assert_eq!(
        row_dirty.kernel_commit.as_deref(),
        Some("kabcde7-dirty"),
        "populated dirty kernel_commit must propagate \
             verbatim, including the `-dirty` suffix",
    );

    let sc_clean = test_support::SidecarResult {
        test_name: "kc_clean_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        kernel_commit: Some("kabcde7".to_string()),
        ..test_support::SidecarResult::test_fixture()
    };
    let row_clean = sidecar_to_row(&sc_clean);
    assert_eq!(
        row_clean.kernel_commit.as_deref(),
        Some("kabcde7"),
        "populated clean kernel_commit (no `-dirty` suffix) \
             must propagate verbatim — a regression that always \
             appended `-dirty` or always stripped a tail would \
             surface here independently of the dirty case above",
    );

    let sc_none = test_support::SidecarResult {
        test_name: "no_kc_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        kernel_commit: None,
        ..test_support::SidecarResult::test_fixture()
    };
    let row_none = sidecar_to_row(&sc_none);
    assert!(
        row_none.kernel_commit.is_none(),
        "absent kernel_commit must propagate as None — a \
             regression substituting an empty string would dilute \
             every `--kernel-commit` filter into matching all \
             None rows",
    );

    // Field non-aliasing pin: kernel_commit and commit must
    // route to distinct row fields. A regression that
    // accidentally cross-wired the two (e.g. `commit:
    // sc.kernel_commit.clone()` instead of
    // `sc.project_commit.clone()`) would hide behind the
    // populated tests above unless the values differ — which
    // they do here. Distinct tokens make the swap obvious.
    let sc_both = test_support::SidecarResult {
        test_name: "both_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        project_commit: Some("project1".to_string()),
        kernel_commit: Some("kernel1".to_string()),
        ..test_support::SidecarResult::test_fixture()
    };
    let row_both = sidecar_to_row(&sc_both);
    assert_eq!(
        row_both.commit.as_deref(),
        Some("project1"),
        "row.commit must come from project_commit, not kernel_commit",
    );
    assert_eq!(
        row_both.kernel_commit.as_deref(),
        Some("kernel1"),
        "row.kernel_commit must come from kernel_commit, not project_commit",
    );
}

/// `sidecar_to_row` must copy `SidecarResult::run_source` into
/// `GauntletRow::run_source` verbatim so the typed `--run-source`
/// filter and per-side `--a-run-source` / `--b-run-source` slicers
/// see the run-environment provenance tag the sidecar writer
/// recorded. A regression that left the field at the
/// `Option::default()` (`None`) would silently drop the
/// run-source dimension from every comparison even when the
/// sidecar had a populated value. Mirrors
/// `sidecar_to_row_propagates_kernel_commit` for the
/// `run_source` field; pinned for `None` and the canonical
/// `Some("local")` / `Some("ci")` / `Some("archive")`
/// values so a regression that special-cased one tag and
/// not the others surfaces here. A non-aliasing pin
/// confirms `run_source` reads from `sc.run_source` rather
/// than being cross-wired to the visually-similar
/// `kernel_commit` / `project_commit` fields.
#[test]
fn sidecar_to_row_propagates_run_source() {
    use crate::test_support;
    for tag in ["local", "ci", "archive"] {
        let sc = test_support::SidecarResult {
            test_name: format!("run_source_{tag}_test"),
            topology: "1n1l2c1t".to_string(),
            run_source: Some(tag.to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(
            row.run_source.as_deref(),
            Some(tag),
            "populated run_source `{tag}` must propagate verbatim",
        );
    }

    let sc_none = test_support::SidecarResult {
        test_name: "no_run_source_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        run_source: None,
        ..test_support::SidecarResult::test_fixture()
    };
    let row_none = sidecar_to_row(&sc_none);
    assert!(
        row_none.run_source.is_none(),
        "absent run_source must propagate as None — a regression \
             substituting an empty string would dilute every \
             `--run-source` filter into matching all None rows",
    );

    // Field non-aliasing pin: `run_source` must route to its
    // own row field. A regression that cross-wired
    // `run_source` to `kernel_commit` (or vice versa) would
    // hide behind the populated tests above unless the values
    // are visibly different. Distinct tokens make the swap
    // obvious.
    let sc_distinct = test_support::SidecarResult {
        test_name: "run_source_distinct_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        run_source: Some("local".to_string()),
        kernel_commit: Some("kabcde7".to_string()),
        project_commit: Some("pabcde7".to_string()),
        ..test_support::SidecarResult::test_fixture()
    };
    let row_distinct = sidecar_to_row(&sc_distinct);
    assert_eq!(
        row_distinct.run_source.as_deref(),
        Some("local"),
        "row.run_source must come from sc.run_source, not from \
             kernel_commit or project_commit",
    );
    assert_eq!(
        row_distinct.kernel_commit.as_deref(),
        Some("kabcde7"),
        "row.kernel_commit must remain sourced from sc.kernel_commit",
    );
    assert_eq!(
        row_distinct.commit.as_deref(),
        Some("pabcde7"),
        "row.commit must remain sourced from sc.project_commit",
    );
}

/// `sidecar_to_row` must copy `SidecarResult::resolve_source` into
/// `GauntletRow::resolve_source` verbatim so the `--resolve-source`
/// compare filter and `stats list-values` surface the discovery-path tag
/// the sidecar writer recorded. A regression leaving the row field at
/// `Option::default()` (`None`) would silently drop the resolve-source
/// dimension. Mirrors `sidecar_to_row_propagates_run_source`; pinned for
/// `None` and the canonical discovery-path tags.
#[test]
fn sidecar_to_row_propagates_resolve_source() {
    use crate::test_support;
    for tag in ["auto_built", "target_debug", "path"] {
        let sc = test_support::SidecarResult {
            test_name: format!("resolve_source_{tag}_test"),
            topology: "1n1l2c1t".to_string(),
            resolve_source: Some(tag.to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(
            row.resolve_source.as_deref(),
            Some(tag),
            "populated resolve_source `{tag}` must propagate verbatim",
        );
    }

    let sc_none = test_support::SidecarResult {
        test_name: "no_resolve_source_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        resolve_source: None,
        ..test_support::SidecarResult::test_fixture()
    };
    assert!(
        sidecar_to_row(&sc_none).resolve_source.is_none(),
        "absent resolve_source must propagate as None",
    );
}

#[test]
fn sidecar_to_row_no_stall() {
    use crate::monitor;
    use crate::test_support;
    let sc = test_support::SidecarResult {
        monitor: Some(monitor::MonitorSummary {
            prog_stats_deltas: None,
            total_samples: 5,
            max_imbalance_ratio: 1.0,
            max_local_dsq_depth: 0,
            stuck_count: 0,
            event_deltas: None,
            schedstat_deltas: None,
            ..Default::default()
        }),
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(row.stuck_count, 0.0);
    assert_eq!(row.fallback_count, 0);
    assert_eq!(row.keep_last_count, 0);
}

/// Drive every direct f64 field on [`GauntletRow`] through
/// `finite_or_zero` with `non_finite` planted in the source
/// [`SidecarResult`], then assert each lands as 0.0 on the row.
///
/// Covers the `finite_or_zero` call sites in `sidecar_to_row`: the
/// remaining direct [`ScenarioStats`] f64 fields (worst_spread,
/// worst_migration_ratio) plus `imbalance_ratio` from [`MonitorSummary`].
/// (The wake / run-delay and both NUMA roll-ups — `worst_page_locality`,
/// `worst_cross_node_migration_ratio` — are now ext_metrics-sourced; non-finite
/// ext entries are DROPPED, covered by
/// `sidecar_to_row_drops_non_finite_ext_metrics`.) A missed call site would
/// leave one assert comparing the non-finite input to 0.0 (NaN != 0.0,
/// ±Infinity != 0.0) and fail the test.
fn assert_all_direct_f64_fields_sanitized(non_finite: f64) {
    use crate::assert::ScenarioStats;
    use crate::monitor::MonitorSummary;
    use crate::test_support;
    let sc = test_support::SidecarResult {
        stats: ScenarioStats {
            worst_spread: non_finite,
            worst_migration_ratio: non_finite,
            ..Default::default()
        },
        monitor: Some(MonitorSummary {
            max_imbalance_ratio: non_finite,
            ..Default::default()
        }),
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    for (name, val) in [
        ("spread", row.spread),
        ("migration_ratio", row.migration_ratio),
        ("imbalance_ratio", row.imbalance_ratio),
    ] {
        assert_eq!(
            val, 0.0,
            "{name} must collapse to 0.0 for non-finite input {non_finite:?}",
        );
    }
    // Motivation check: the sanitized row serializes. Without the
    // `finite_or_zero` wraps, serde_json::to_string would return
    // Err because NaN / Infinity have no JSON representation.
    serde_json::to_string(&row).expect("sanitized row must serialize cleanly");
}

/// `sidecar_to_row` must sanitize NaN in every direct f64 field
/// (both [`ScenarioStats`]-sourced and the
/// [`MonitorSummary`]-sourced `imbalance_ratio`), not just a
/// representative sample — same `serde_json` rejects-NaN
/// motivation. Unlike `ext_metrics`, direct fields can't be
/// dropped (the row schema is fixed), so non-finite collapses to
/// 0.0 with a warn.
#[test]
fn sidecar_to_row_zeros_nan_in_every_direct_f64_field() {
    assert_all_direct_f64_fields_sanitized(f64::NAN);
}

/// Companion to `sidecar_to_row_zeros_nan_in_every_direct_f64_field`
/// pinning the `+Infinity` branch of `finite_or_zero` for every
/// direct f64 field on the row.
#[test]
fn sidecar_to_row_zeros_pos_infinity_in_every_direct_f64_field() {
    assert_all_direct_f64_fields_sanitized(f64::INFINITY);
}

/// Companion to `sidecar_to_row_zeros_nan_in_every_direct_f64_field`
/// pinning the `-Infinity` branch of `finite_or_zero` for every
/// direct f64 field on the row.
#[test]
fn sidecar_to_row_zeros_neg_infinity_in_every_direct_f64_field() {
    assert_all_direct_f64_fields_sanitized(f64::NEG_INFINITY);
}

/// Subnormal f64 values (IEEE 754 denormals) are finite —
/// `is_finite()` returns `true` for them — and must pass through
/// `finite_or_zero` unchanged. Guards against a future refactor
/// that reaches for `is_normal()` instead of `is_finite()`,
/// which would incorrectly collapse subnormals to 0.0 and erase
/// very-small legitimate measurements. `f64::MIN_POSITIVE` is the
/// smallest normal positive; `/ 2.0` lands in the subnormal
/// range.
#[test]
fn sidecar_to_row_preserves_subnormal_f64_in_direct_fields() {
    use crate::assert::ScenarioStats;
    use crate::test_support;
    let subnormal = f64::MIN_POSITIVE / 2.0;
    assert!(subnormal.is_finite(), "subnormal must still be finite");
    assert!(!subnormal.is_normal(), "subnormal must not be normal");
    assert!(subnormal > 0.0, "subnormal is positive");
    let sc = test_support::SidecarResult {
        stats: ScenarioStats {
            worst_spread: subnormal,
            worst_migration_ratio: -subnormal,
            ..Default::default()
        },
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(
        row.spread, subnormal,
        "positive subnormal must pass through finite_or_zero unchanged",
    );
    assert_eq!(
        row.migration_ratio, -subnormal,
        "negative subnormal on a second direct-f64 field must also pass through unchanged",
    );
    // Motivation check: subnormals serialize (unlike NaN / ±Inf,
    // serde_json emits them as standard decimal literals).
    serde_json::to_string(&row).expect("subnormals serialize cleanly");
}

/// Pins that the direct-field NaN sanitization in
/// `sidecar_to_row` does NOT reach into `ext_metrics`. Finite
/// `ext_metrics` entries must survive untouched even when every
/// direct f64 field collapses to 0.0, and the `ext_metrics` map
/// must not grow a sanitization-synthesized entry. Complements
/// [`sidecar_to_row_drops_non_finite_ext_metrics`] (which pins
/// that non-finite `ext_metrics` entries are DROPPED) by pinning
/// the orthogonal claim: direct-field sanitization never writes
/// into `ext_metrics` regardless of the direct values.
#[test]
fn sidecar_to_row_direct_field_nan_does_not_touch_ext_metrics() {
    use crate::assert::ScenarioStats;
    use crate::test_support;
    let mut ext = BTreeMap::new();
    ext.insert("finite_nonzero".to_string(), 2.5);
    ext.insert("finite_zero".to_string(), 0.0);
    ext.insert("finite_negative".to_string(), -7.25);
    let sc = test_support::SidecarResult {
        stats: ScenarioStats {
            // Every remaining direct f64 field non-finite.
            worst_spread: f64::NAN,
            worst_migration_ratio: f64::INFINITY,
            ext_metrics: ext.clone(),
            ..Default::default()
        },
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);

    // Direct-field collapse still works.
    assert_eq!(row.spread, 0.0);
    assert_eq!(row.migration_ratio, 0.0);

    // ext_metrics survives unchanged — same length, same keys,
    // same values.
    assert_eq!(
        row.ext_metrics.len(),
        ext.len(),
        "direct-field sanitization must not add or drop ext_metrics entries",
    );
    for (k, v) in &ext {
        assert_eq!(
            row.ext_metrics.get(k),
            Some(v),
            "ext_metrics entry {k:?} must pass through unchanged",
        );
    }

    // Motivation check: the full row still serializes.
    serde_json::to_string(&row).expect("sanitized row must serialize cleanly");
}

/// `sidecar_to_row` must drop NaN / +Infinity / -Infinity from
/// `ext_metrics` because `serde_json::to_string` rejects non-finite
/// f64 values — without this guard a single malformed scenario
/// metric would poison every sidecar write on its batch. Finite
/// entries must pass through unchanged. Also checks that the
/// post-filter row serializes cleanly (the motivation for the
/// filter).
#[test]
fn sidecar_to_row_drops_non_finite_ext_metrics() {
    use crate::assert::ScenarioStats;
    use crate::test_support;
    let mut ext = BTreeMap::new();
    ext.insert("good".to_string(), 1.0);
    ext.insert("nan".to_string(), f64::NAN);
    ext.insert("inf".to_string(), f64::INFINITY);
    ext.insert("neg_inf".to_string(), f64::NEG_INFINITY);
    let sc = test_support::SidecarResult {
        stats: ScenarioStats {
            ext_metrics: ext,
            ..Default::default()
        },
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(
        row.ext_metrics.len(),
        1,
        "only the finite entry should survive: {:?}",
        row.ext_metrics
    );
    assert_eq!(row.ext_metrics.get("good"), Some(&1.0));
    assert!(!row.ext_metrics.contains_key("nan"));
    assert!(!row.ext_metrics.contains_key("inf"));
    assert!(!row.ext_metrics.contains_key("neg_inf"));
    // Motivation check: the post-filter row serializes. Without the
    // filter, serde_json::to_string would return Err because NaN /
    // Infinity have no JSON representation.
    serde_json::to_string(&row).expect("filtered row must serialize cleanly");
}

/// `sidecar_to_row` must drop the JSON-walker depth-cap sentinel
/// [`crate::test_support::WALK_TRUNCATION_SENTINEL_NAME`] from
/// `ext_metrics`. The sentinel is diagnostic metadata about the
/// extraction pass (depth cap hit), not a scenario metric, so it
/// must not leak into A/B comparison output where it would be
/// mistaken for a real measurement and skew filter / aggregation
/// logic. Sibling finite entries must survive untouched.
#[test]
fn sidecar_to_row_drops_walk_truncation_sentinel() {
    use crate::assert::ScenarioStats;
    use crate::test_support;
    let mut ext = BTreeMap::new();
    ext.insert("good".to_string(), 1.0);
    ext.insert(
        test_support::WALK_TRUNCATION_SENTINEL_NAME.to_string(),
        72.0,
    );
    let sc = test_support::SidecarResult {
        stats: ScenarioStats {
            ext_metrics: ext,
            ..Default::default()
        },
        ..test_support::SidecarResult::test_fixture()
    };
    let row = sidecar_to_row(&sc);
    assert_eq!(
        row.ext_metrics.len(),
        1,
        "only the real metric should survive: {:?}",
        row.ext_metrics,
    );
    assert_eq!(row.ext_metrics.get("good"), Some(&1.0));
    assert!(
        !row.ext_metrics
            .contains_key(test_support::WALK_TRUNCATION_SENTINEL_NAME),
        "sentinel must not appear in the row's ext_metrics",
    );
}

// -- metric_def tests --

#[test]
fn metric_def_known() {
    let d = metric_def("worst_spread").unwrap();
    assert_eq!(d.name, "worst_spread");
    assert!(d.higher_is_worse());
    assert_eq!(d.display_unit, "%");
}

#[test]
fn metric_def_not_higher_is_worse() {
    let d = metric_def("total_iterations").unwrap();
    assert!(!d.higher_is_worse());
}

#[test]
fn metric_def_unknown() {
    assert!(metric_def("nonexistent").is_none());
}

#[test]
fn infer_higher_is_worse_latency_shaped() {
    // Latency / delay / time-unit names are higher-is-worse.
    // `AssertResult::merge` folds these by `max` (worst case).
    for name in &[
        "p99_wake_latency",
        "wake_latency_us",
        "scheduling_delay",
        "task_run_delay_ns",
        "io_completion_ms",
        "stall_count",
        "stuck_count",
        "schedule_jitter_cv",
        "max_gap_us",
        "spread",
        "page_drop_count",
        "error_rate",
        "fail_count",
        "migration_ratio",
        "imbalance_factor",
    ] {
        assert!(
            infer_higher_is_worse(name),
            "metric `{name}` must infer higher_is_worse=true \
                 (latency/error-shaped); a folded max keeps the \
                 worst case across cgroups"
        );
    }
}

#[test]
fn infer_higher_is_worse_throughput_shaped() {
    // Throughput / rate / iteration-shaped names are
    // higher-is-better. `AssertResult::merge` folds these by
    // `min` (the cgroup that fell behind is the worst case).
    for name in &[
        "read_iops",
        "write_iops",
        "throughput_mbps",
        "bandwidth_kb",
        "total_iterations",
        "iterations_per_worker",
        "ops_per_sec",
        "page_locality",
        "pass_score",
        "goodput",
    ] {
        assert!(
            !infer_higher_is_worse(name),
            "metric `{name}` must infer higher_is_worse=false \
                 (throughput-shaped); a folded min surfaces the \
                 cgroup that fell behind"
        );
    }
}

#[test]
fn infer_higher_is_worse_unknown_falls_back_to_higher_is_worse() {
    // Names that don't match either token list fall back to
    // higher-is-worse. The fold keeps the max — which surfaces
    // an unexpectedly high reading rather than masking it under
    // a min collapse.
    for name in &["unrelated_field", "random_thing", "metric", "x", "a.b.c"] {
        assert!(
            infer_higher_is_worse(name),
            "unknown metric `{name}` must fall back to \
                 higher_is_worse=true (conservative for regression \
                 detection)"
        );
    }
}

#[test]
fn infer_higher_is_worse_compound_names_resolve_to_latency() {
    // Compound names that contain BOTH a latency-shaped token
    // and a throughput-shaped token resolve to higher-is-worse
    // (the latency interpretation wins). Pin the order-of-
    // checks contract so a refactor that swaps the token lists
    // surfaces here.
    assert!(
        infer_higher_is_worse("read_iops_latency_us"),
        "compound name with `latency` and `iops` must resolve \
             to higher-is-worse (latency token checked first)"
    );
}

#[test]
fn metric_def_polarity_inverse_sense() {
    use crate::test_support::Polarity;
    // higher_is_worse=true means growing = regression; the
    // Polarity for "what do we want it to move toward?" is
    // LowerBetter.
    let d = metric_def("worst_spread").unwrap();
    assert!(d.higher_is_worse());
    assert_eq!(d.polarity, Polarity::LowerBetter);
    // higher_is_worse=false means growing = improvement; the
    // Polarity is HigherBetter.
    let d = metric_def("total_iterations").unwrap();
    assert!(!d.higher_is_worse());
    assert_eq!(d.polarity, Polarity::HigherBetter);
}

#[test]
fn metric_def_polarity_covers_all_entries() {
    use crate::test_support::Polarity;
    // Every METRICS entry must map cleanly to HigherBetter or
    // LowerBetter; no entry should produce TargetValue or Unknown
    // from the bool->Polarity adaptor.
    for m in METRICS.iter() {
        assert!(
            matches!(
                m.polarity,
                Polarity::HigherBetter | Polarity::LowerBetter | Polarity::Informational
            ),
            "metric {} produced unexpected polarity {:?} — only HigherBetter / \
             LowerBetter (directional, gated) and Informational (directionless, \
             never gated) are registered; TargetValue / Unknown are not used by \
             any METRICS entry (Unknown stays the conservative default for \
             UNclassified metrics, not a deliberate registry choice)",
            m.name,
            m.polarity
        );
    }
}

#[test]
fn metric_def_all_entries_unique() {
    let mut names: Vec<&str> = METRICS.iter().map(|m| m.name).collect();
    let len = names.len();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), len);
}

/// Registry integrity: every `MetricKind::Rate`'s numerator and
/// denominator MUST name a registered `Counter` metric. `derive_rate_metrics`
/// silently skips a Rate whose component key is absent from the map
/// (`derive_rate_metrics_from` `continue`s on a missing key), so a typo'd
/// component name would never derive and never fail a value test — pin the
/// names at the registry level instead. Counter (not Gauge/Peak) because the
/// re-pool needs Σnum/Σdenom (sum-fold), which only the Counter kind gives.
#[test]
fn every_rate_metric_has_registered_counter_components() {
    for m in METRICS.iter() {
        let MetricKind::Rate {
            numerator,
            denominator,
        } = m.kind
        else {
            continue;
        };
        for (role, comp) in [("numerator", numerator), ("denominator", denominator)] {
            let def = metric_def(comp).unwrap_or_else(|| {
                panic!("Rate {} {role} {comp:?} is not a registered metric", m.name)
            });
            assert!(
                matches!(def.kind, MetricKind::Counter),
                "Rate {} {role} {comp:?} must be a Counter for the Σ-fold re-pool",
                m.name,
            );
        }
    }
}

// -- list_metrics tests --

/// Text-mode [`list_metrics`] emits a table that names every
/// registered metric at least once. Uses substring contains
/// rather than column-exact equality so a future comfy-table
/// preset rename (NOTHING → other) that rewraps whitespace
/// does not false-fail — the surface contract is "every metric
/// name appears somewhere in the rendered output", not a
/// column-width pin.
#[test]
fn list_metrics_text_names_every_metric() {
    let out = list_metrics(false).expect("text render must succeed");
    assert!(!out.is_empty(), "text output must be non-empty");
    for m in METRICS {
        assert!(
            out.contains(m.name),
            "list_metrics(false) output missing metric name {}: {out}",
            m.name,
        );
    }
}

/// Text-mode [`list_metrics`] header row names every column. Pins
/// the header contract so a column rename in
/// `list_metrics` lands here instead of silently in downstream CI
/// scripts that grep the output.
#[test]
fn list_metrics_text_header_pins_column_names() {
    let out = list_metrics(false).expect("text render must succeed");
    for header in ["NAME", "POLARITY", "DEFAULT_ABS", "DEFAULT_REL", "UNIT"] {
        assert!(
            out.contains(header),
            "list_metrics(false) output missing column header {header}: {out}",
        );
    }
}

/// JSON-mode [`list_metrics`] parses back to a `Vec<MetricDef>`-
/// shaped structure with one entry per registry member. `MetricDef`
/// itself does not derive `Deserialize` (the `accessor` fn-pointer
/// is unserializable), so we deserialize into a minimal struct
/// that captures the fields the wire contract promises.
#[test]
fn list_metrics_json_round_trips_via_minimal_schema() {
    #[derive(serde::Deserialize)]
    struct MetricEntry {
        name: String,
        default_abs: f64,
        default_rel: f64,
        display_unit: String,
        // polarity is serialized as an enum tag string by serde
        // (Polarity derives Serialize with the default
        // externally-tagged representation). Deserialize into a
        // serde_json::Value to avoid a cross-crate enum
        // dependency in the test-private schema.
        polarity: serde_json::Value,
    }

    let out = list_metrics(true).expect("json render must succeed");
    let parsed: Vec<MetricEntry> = serde_json::from_str(&out).expect("json output must parse");
    assert_eq!(
        parsed.len(),
        METRICS.len(),
        "json entry count must match METRICS.len()",
    );
    for (parsed_m, registry_m) in parsed.iter().zip(METRICS.iter()) {
        assert_eq!(parsed_m.name, registry_m.name);
        assert_eq!(parsed_m.default_abs, registry_m.default_abs);
        assert_eq!(parsed_m.default_rel, registry_m.default_rel);
        assert_eq!(parsed_m.display_unit, registry_m.display_unit);
        assert!(
            !parsed_m.polarity.is_null(),
            "polarity for {} must serialize as a non-null value",
            registry_m.name,
        );
    }
}

/// JSON-mode [`list_metrics`] must NOT expose the `accessor`
/// fn-pointer field. The `#[serde(skip)]` attribute on
/// `MetricDef::accessor` carries that contract; a regression that
/// dropped the attribute would surface here as the emitted JSON
/// gaining an "accessor" key. Pins the wire surface.
#[test]
fn list_metrics_json_omits_accessor_field() {
    let out = list_metrics(true).expect("json render must succeed");
    assert!(
        !out.contains("\"accessor\""),
        "list_metrics(true) must not emit the accessor field — \
             fn-pointers are not serializable and the field carries \
             #[serde(skip)]: {out}",
    );
}

/// The Distribution/WorstLowest MetricKind serde shape is user-facing via
/// `cargo ktstr stats list-metrics --json` (list_metrics(true) serializes
/// the full MetricDef incl. `kind`). Pin the externally-tagged JSON variant
/// and helper-enum strings so a rename of MetricKind / SampleSource /
/// SampleReduction / WorstLowest* trips this test rather than silently
/// changing the CLI output. MetricKind is Serialize-only (output contract;
/// no deserialize symmetry to check).
#[test]
fn distribution_worstlowest_kind_json_shape_pinned() {
    let dist = serde_json::to_string(&MetricKind::Distribution {
        source: SampleSource::WakeLatencyNs,
        reduction: SampleReduction::P99,
    })
    .expect("MetricKind serializes");
    for tok in [
        "\"Distribution\"",
        "\"source\"",
        "\"WakeLatencyNs\"",
        "\"reduction\"",
        "\"P99\"",
    ] {
        assert!(dist.contains(tok), "{tok} missing from {dist}");
    }
    let wl = serde_json::to_string(&MetricKind::WorstLowest {
        numerator: WorstLowestNumerator::Iterations,
        denominator: WorstLowestDenominator::CpuTimeNs,
    })
    .expect("MetricKind serializes");
    for tok in [
        "\"WorstLowest\"",
        "\"numerator\"",
        "\"Iterations\"",
        "\"denominator\"",
        "\"CpuTimeNs\"",
    ] {
        assert!(wl.contains(tok), "{tok} missing from {wl}");
    }
    // The NUMA pairing (worst_page_locality's kind) — pins the NumaLocal /
    // NumaTotal variant strings so a rename trips here, not the CLI output.
    let numa = serde_json::to_string(&MetricKind::WorstLowest {
        numerator: WorstLowestNumerator::NumaLocal,
        denominator: WorstLowestDenominator::NumaTotal,
    })
    .expect("MetricKind serializes");
    for tok in ["\"NumaLocal\"", "\"NumaTotal\""] {
        assert!(numa.contains(tok), "{tok} missing from {numa}");
    }
    // worst_cross_node_migration_ratio's kind (a unit variant) — pin the variant
    // string so a rename trips here, not the CLI output.
    let xnode = serde_json::to_string(&MetricKind::WorstCrossNodeRatio)
        .expect("MetricKind serializes");
    assert!(
        xnode.contains("WorstCrossNodeRatio"),
        "WorstCrossNodeRatio missing from {xnode}",
    );
}

/// Iteration order of [`list_metrics`] matches [`METRICS`]
/// declaration order. Registry order is the canonical surface
/// order for sidecar / CI-gate consumers; a renderer that sorted
/// by name or polarity would silently break scripts that key on
/// the first row.
#[test]
fn list_metrics_text_preserves_registry_order() {
    let out = list_metrics(false).expect("text render must succeed");
    let mut last_pos = 0usize;
    for m in METRICS {
        // Match m.name as a WHOLE name, not a substring of a longer one: a bare
        // per-cgroup name (e.g. `migration_ratio`) is a substring of its
        // run-level `worst_*` sibling (`worst_migration_ratio`), so a plain
        // `find` would match the earlier `worst_` row and report a false
        // out-of-order. Accept only an occurrence whose preceding char is not a
        // name char (alphanumeric or `_`) — i.e. the start of a NAME-column cell.
        let pos = out
            .match_indices(m.name)
            .find(|(i, _)| {
                *i == 0
                    || !out[..*i]
                        .chars()
                        .next_back()
                        .is_some_and(|c| c.is_alphanumeric() || c == '_')
            })
            .map(|(i, _)| i)
            .unwrap_or_else(|| {
                panic!("metric {} must appear as a whole name in text output", m.name)
            });
        assert!(
            pos >= last_pos,
            "metric {} appears before a prior metric — text output must \
                 preserve METRICS declaration order",
            m.name,
        );
        last_pos = pos;
    }
}

// -- list_values --

/// Helper that writes N sidecars to `{root}/{run_key}/{run_key}.ktstr.json`.
/// Each sidecar overrides only the fields the test wants to vary;
/// the rest come from `SidecarResult::test_fixture()`. Used by the
/// `list_values_*` tests to build pool fixtures isolated from
/// `runs_root()`.
fn write_listvalues_fixture(
    root: &std::path::Path,
    sidecars: &[crate::test_support::SidecarResult],
) {
    for (i, sc) in sidecars.iter().enumerate() {
        let run_key = format!("__lv_fixture_{i}__");
        let run_dir = root.join(&run_key);
        std::fs::create_dir_all(&run_dir).expect("create run dir");
        let json = serde_json::to_string(sc).expect("serialize fixture sidecar");
        let path = run_dir.join(format!("{run_key}.ktstr.json"));
        std::fs::write(&path, json).expect("write fixture sidecar");
    }
}

/// Empty pool (no run subdirs) must produce a well-formed text
/// shape with the "(no sidecars in pool)" sentinel under each
/// dimension heading. The function does NOT bail — discovery on
/// an empty pool is a valid query that should answer "nothing"
/// rather than fail.
#[test]
fn list_values_empty_pool_text_has_sentinel_per_dim() {
    let alt = tempfile::TempDir::new().expect("tempdir");
    let out = list_values(false, Some(alt.path())).expect("text render must succeed");
    for dim in [
        "kernel:",
        "commit:",
        "kernel_commit:",
        "source:",
        "resolve_source:",
        "cpu_budget:",
        "scheduler:",
        "topology:",
        "work_type:",
    ] {
        assert!(
            out.contains(dim),
            "text output must include heading for {dim}: {out}",
        );
    }
    // Each dim should report the empty-pool sentinel exactly nine
    // times — one per dim — so a regression that dropped the
    // sentinel for one dim falls out as a count mismatch.
    let sentinel_count = out.matches("(no sidecars in pool)").count();
    assert_eq!(
        sentinel_count, 9,
        "empty pool must surface the no-sidecars sentinel under every \
             one of the 9 dims (kernel/commit/kernel_commit/source/\
             resolve_source/cpu_budget/scheduler/topology/work_type); got \
             {sentinel_count} occurrences in:\n{out}",
    );
}

/// Empty pool → JSON object with empty arrays for every dim.
/// Pins the JSON shape so a regression that dropped a key (e.g.
/// "scheduler") on the empty-pool branch surfaces here.
#[test]
fn list_values_empty_pool_json_emits_empty_arrays() {
    let alt = tempfile::TempDir::new().expect("tempdir");
    let out = list_values(true, Some(alt.path())).expect("json render must succeed");
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("json output must parse");
    for dim in [
        "kernel",
        "commit",
        "kernel_commit",
        "source",
        "resolve_source",
        "cpu_budget",
        "scheduler",
        "topology",
        "work_type",
    ] {
        let arr = parsed
            .get(dim)
            .unwrap_or_else(|| panic!("missing key {dim}"));
        assert!(arr.is_array(), "key {dim} must serialize as an array");
        assert_eq!(
            arr.as_array().unwrap().len(),
            0,
            "key {dim} must be an empty array on empty pool",
        );
    }
}

/// Populated pool: distinct values per dim are deduplicated and
/// sorted; `kernel_version: None` and `project_commit: None`
/// produce a `null` entry (JSON) and `unknown` line (text).
#[test]
fn list_values_text_dedupes_and_sorts_per_dim() {
    use crate::test_support::SidecarResult;

    let alt = tempfile::TempDir::new().expect("tempdir");
    let sidecars = vec![
        SidecarResult {
            test_name: "t_a".to_string(),
            topology: "1n2l4c1t".to_string(),
            scheduler: "scx_rusty".to_string(),
            work_type: "SpinWait".to_string(),
            kernel_version: Some("6.14.2".to_string()),
            project_commit: Some("abcdef1".to_string()),
            ..SidecarResult::test_fixture()
        },
        SidecarResult {
            test_name: "t_b".to_string(),
            topology: "1n4l2c1t".to_string(),
            scheduler: "eevdf".to_string(),
            work_type: "PageFaultChurn".to_string(),
            kernel_version: None,
            project_commit: None,
            ..SidecarResult::test_fixture()
        },
        // Duplicate of the first sidecar's identity-fields; the
        // BTreeSet must dedupe so each value lands once in the
        // rendered output.
        SidecarResult {
            test_name: "t_c".to_string(),
            topology: "1n2l4c1t".to_string(),
            scheduler: "scx_rusty".to_string(),
            work_type: "SpinWait".to_string(),
            kernel_version: Some("6.14.2".to_string()),
            project_commit: Some("abcdef1".to_string()),
            ..SidecarResult::test_fixture()
        },
    ];
    write_listvalues_fixture(alt.path(), &sidecars);

    let out = list_values(false, Some(alt.path())).expect("text render must succeed");

    // Dedupe: each distinct VALUE appears EXACTLY once per
    // dim (set semantics) even though "scx_rusty" / "1n2l4c1t"
    // / "SpinWait" / "abcdef1" / "6.14.2" come from two of the
    // three fixtures. Each value below is unique to its dim
    // so it should appear once across the rendered text. The
    // `unknown` sentinel is checked separately because both
    // `kernel` and `commit` are optional dims and each emits
    // its own `unknown` line.
    for value in [
        "6.14.2",
        "abcdef1",
        "scx_rusty",
        "eevdf",
        "1n2l4c1t",
        "1n4l2c1t",
        "SpinWait",
        "PageFaultChurn",
    ] {
        let count = out.matches(value).count();
        assert_eq!(
            count, 1,
            "value {value} must appear exactly once in text output (BTreeSet dedup); \
                 got {count} in:\n{out}",
        );
    }
    // `unknown` appears once per Optional dim that has a None
    // entry: kernel, commit, and kernel_commit. The second
    // fixture has `kernel_version: None` and `project_commit:
    // None`; every fixture in this test leaves `kernel_commit`
    // at its `test_fixture` default (None), so the
    // kernel_commit set's None bucket renders one `unknown`
    // line as well. Total: 3 occurrences.
    //
    // `run_source` is an optional dim but does NOT contribute an
    // `unknown` here: `list_values(_, Some(dir))` calls
    // `apply_archive_source_override` on the loaded pool (the `--dir`
    // flag treats the supplied root as an archive), which rewrites
    // every `run_source: None` to `Some("archive")` BEFORE the
    // dimension-set is built. Every fixture above leaves `run_source`
    // at its `test_fixture` default (None), but they all surface as
    // `archive` after the override — the run_source set never holds a
    // None entry on this code path, so no `unknown` line is emitted
    // for it. `resolve_source` IS the fourth `unknown`-contributing
    // optional dim: the archive override touches only `run_source`,
    // so `resolve_source` (None on every fixture here) keeps its None
    // entry and renders one `unknown` line. Total: kernel + commit +
    // kernel_commit + resolve_source = 4.
    let unknown_count = out.matches("unknown").count();
    assert_eq!(
        unknown_count, 4,
        "`unknown` must render once per optional dim with a None \
             entry (kernel + commit + kernel_commit + resolve_source = \
             4); got {unknown_count} in:\n{out}",
    );

    // Sort: both schedulers in ascending lex order means
    // "eevdf" appears BEFORE "scx_rusty" in the rendered text.
    let pos_eevdf = out.find("eevdf").expect("eevdf in output");
    let pos_rusty = out.find("scx_rusty").expect("scx_rusty in output");
    assert!(
        pos_eevdf < pos_rusty,
        "values within a dim must render sorted (BTreeSet iter order); \
             expected 'eevdf' before 'scx_rusty' in:\n{out}",
    );
}

/// The populated `cpu_budget` dim renders distinct budgets once
/// each, sorted ascending, in both the text block and as a numeric
/// JSON array (every other dim is a string array; cpu_budget is the
/// sole numeric one). Skip rows (budget 0) are excluded.
#[test]
fn list_values_cpu_budget_renders_distinct_sorted() {
    use crate::test_support::SidecarResult;

    let alt = tempfile::TempDir::new().expect("tempdir");
    let sidecars = vec![
        SidecarResult {
            test_name: "t_a".to_string(),
            cpu_budget: 32,
            vcpus: 32,
            ..SidecarResult::test_fixture()
        },
        SidecarResult {
            test_name: "t_b".to_string(),
            cpu_budget: 4,
            vcpus: 16,
            ..SidecarResult::test_fixture()
        },
        // Duplicate budget 4 — the BTreeSet must dedupe.
        SidecarResult {
            test_name: "t_c".to_string(),
            cpu_budget: 4,
            vcpus: 16,
            ..SidecarResult::test_fixture()
        },
        // Skip row (budget 0) — excluded from the budget set.
        SidecarResult {
            test_name: "t_d".to_string(),
            cpu_budget: 0,
            vcpus: 0,
            ..SidecarResult::test_fixture()
        },
    ];
    write_listvalues_fixture(alt.path(), &sidecars);

    let text = list_values(false, Some(alt.path())).expect("text render must succeed");
    // Distinct budgets appear once each. Scope the search to the
    // cpu_budget block so digits in other values can't false-match:
    // the block is the lines between "cpu_budget:\n" and the next
    // blank line.
    let block_start =
        text.find("cpu_budget:\n").expect("cpu_budget heading") + "cpu_budget:\n".len();
    let block = &text[block_start..];
    let block_end = block.find("\n\n").map(|i| i + 1).unwrap_or(block.len());
    let block = &block[..block_end];
    assert_eq!(
        block.matches("  4\n").count(),
        1,
        "budget 4 once: {block:?}"
    );
    assert_eq!(
        block.matches("  32\n").count(),
        1,
        "budget 32 once: {block:?}"
    );
    let pos_4 = block.find("  4\n").expect("4 present");
    let pos_32 = block.find("  32\n").expect("32 present");
    assert!(pos_4 < pos_32, "budgets must sort ascending: {block:?}");

    let json = list_values(true, Some(alt.path())).expect("json render must succeed");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("json must parse");
    let arr = parsed
        .get("cpu_budget")
        .and_then(|v| v.as_array())
        .expect("cpu_budget must be a JSON array");
    let nums: Vec<u64> = arr
        .iter()
        .map(|v| v.as_u64().expect("numeric budget"))
        .collect();
    assert_eq!(
        nums,
        vec![4, 32],
        "JSON cpu_budget must be a sorted numeric array"
    );
}

/// JSON shape: `kernel` and `commit` arrays carry `null` for
/// absent values, `Value::String` for present values; the other
/// dims are bare `String` arrays.
#[test]
fn list_values_json_carries_null_for_optional_dims() {
    use crate::test_support::SidecarResult;

    let alt = tempfile::TempDir::new().expect("tempdir");
    let sidecars = vec![
        SidecarResult {
            test_name: "t_known".to_string(),
            kernel_version: Some("6.14.2".to_string()),
            project_commit: Some("abcdef1".to_string()),
            resolve_source: Some("auto_built".to_string()),
            ..SidecarResult::test_fixture()
        },
        SidecarResult {
            test_name: "t_unknown".to_string(),
            kernel_version: None,
            project_commit: None,
            resolve_source: None,
            ..SidecarResult::test_fixture()
        },
    ];
    write_listvalues_fixture(alt.path(), &sidecars);

    let out = list_values(true, Some(alt.path())).expect("json render must succeed");
    let parsed: serde_json::Value = serde_json::from_str(&out).expect("json output must parse");

    let kernel = parsed
        .get("kernel")
        .expect("kernel key")
        .as_array()
        .unwrap();
    assert!(
        kernel.iter().any(|v| v.is_null()),
        "kernel array must include a literal null for the None entry; got {kernel:?}",
    );
    assert!(
        kernel.iter().any(|v| v.as_str() == Some("6.14.2")),
        "kernel array must include the populated value 6.14.2; got {kernel:?}",
    );

    let commit = parsed
        .get("commit")
        .expect("commit key")
        .as_array()
        .unwrap();
    assert!(
        commit.iter().any(|v| v.is_null()),
        "commit array must include a literal null for the None entry; got {commit:?}",
    );
    assert!(
        commit.iter().any(|v| v.as_str() == Some("abcdef1")),
        "commit array must include the populated value abcdef1; got {commit:?}",
    );

    // resolve_source is the ninth list-values dimension (JSON key
    // "resolve_source"); it carries null for the None fixture and the
    // populated discovery-path tag for the other. Unlike run_source, the
    // `--dir` archive override does NOT touch resolve_source, so the None
    // entry stays null here.
    let resolve = parsed
        .get("resolve_source")
        .expect("resolve_source key")
        .as_array()
        .unwrap();
    assert!(
        resolve.iter().any(|v| v.is_null()),
        "resolve_source array must include a literal null for the None entry; got {resolve:?}",
    );
    assert!(
        resolve.iter().any(|v| v.as_str() == Some("auto_built")),
        "resolve_source array must include the populated tag auto_built; got {resolve:?}",
    );
}

/// `dir = None` resolves against `runs_root()`; if `runs_root()`
/// does not exist, the function returns Ok with empty arrays /
/// per-dim sentinel rather than bailing. Pins the no-bail
/// contract on missing-root.
#[test]
fn list_values_none_dir_does_not_bail_on_missing_root() {
    // We cannot reliably wipe `runs_root()` from a unit test, but
    // we can pin the "Some(nonexistent_path)" branch which
    // exercises the same `collect_pool -> empty Vec` codepath
    // (`fs::read_dir` returns Err on a missing root, and
    // `collect_pool` swallows that into an empty pool).
    let alt = tempfile::TempDir::new().expect("tempdir");
    let nonexistent = alt.path().join("definitely_does_not_exist");
    let out = list_values(false, Some(&nonexistent)).expect("must not bail on missing root");
    assert!(
        out.contains("(no sidecars in pool)"),
        "missing root must render the no-sidecars sentinel: {out}",
    );
}

// -- MetricDef::read tests --

fn read_metric(row: &GauntletRow, name: &str) -> Option<f64> {
    metric_def(name).expect("metric name").read(row)
}

#[test]
fn metric_def_read_named_fields() {
    let mut row = make_row("a", "t", true, 42.0);
    row.gap_ms = 100;
    row.migrations = 7;
    row.migration_ratio = 0.3;
    row.imbalance_ratio = 2.0;
    row.max_dsq_depth = 5;
    row.stuck_count = 3.0;
    row.fallback_count = 11;
    row.keep_last_count = 4;
    row.total_iterations = 1000;
    // Distribution + WorstLowest + WorstCrossNodeRatio (worst_page_locality,
    // worst_cross_node_migration_ratio) roll-ups are ext_metrics-sourced now
    // (accessor |_| None); read_metric resolves them via MetricDef::read's ext
    // fallback.
    for (name, v) in [
        ("worst_p99_wake_latency_us", 99.0),
        ("worst_median_wake_latency_us", 50.0),
        ("worst_wake_latency_cv", 0.5),
        ("worst_mean_run_delay_us", 25.0),
        ("worst_run_delay_us", 200.0),
        ("worst_page_locality", 0.8),
        ("worst_cross_node_migration_ratio", 0.1),
    ] {
        row.ext_metrics.insert(name.to_string(), v);
    }
    assert_eq!(read_metric(&row, "worst_spread"), Some(42.0));
    assert_eq!(read_metric(&row, "worst_gap_ms"), Some(100.0));
    assert_eq!(read_metric(&row, "total_migrations"), Some(7.0));
    assert_eq!(read_metric(&row, "worst_migration_ratio"), Some(0.3));
    assert_eq!(read_metric(&row, "max_imbalance_ratio"), Some(2.0));
    assert_eq!(read_metric(&row, "max_dsq_depth"), Some(5.0));
    assert_eq!(read_metric(&row, "stuck_count"), Some(3.0));
    assert_eq!(read_metric(&row, "total_fallback"), Some(11.0));
    assert_eq!(read_metric(&row, "total_keep_last"), Some(4.0));
    assert_eq!(read_metric(&row, "worst_p99_wake_latency_us"), Some(99.0));
    assert_eq!(
        read_metric(&row, "worst_median_wake_latency_us"),
        Some(50.0)
    );
    assert_eq!(read_metric(&row, "worst_wake_latency_cv"), Some(0.5));
    assert_eq!(read_metric(&row, "total_iterations"), Some(1000.0));
    assert_eq!(read_metric(&row, "worst_mean_run_delay_us"), Some(25.0));
    assert_eq!(read_metric(&row, "worst_run_delay_us"), Some(200.0));
    assert_eq!(read_metric(&row, "worst_page_locality"), Some(0.8));
    assert_eq!(
        read_metric(&row, "worst_cross_node_migration_ratio"),
        Some(0.1)
    );
}

#[test]
fn metric_def_read_prefers_accessor_over_ext_metrics() {
    // When a name is in METRICS, the built-in accessor wins.
    // Even if ext_metrics carries a colliding entry for the
    // same name, MetricDef::read returns the accessor's value
    // — built-in fields are the authoritative source.
    let mut row = make_row("a", "t", true, 5.0);
    row.ext_metrics.insert("worst_spread".into(), 999.0);
    assert_eq!(read_metric(&row, "worst_spread"), Some(5.0));

    // User ext_metrics with no matching MetricDef are reachable
    // via the direct ext_metrics map; metric_def returns None
    // for unregistered names.
    row.ext_metrics.insert("custom_metric".into(), 77.0);
    assert!(metric_def("custom_metric").is_none());
    assert_eq!(row.ext_metrics.get("custom_metric").copied(), Some(77.0));
}
