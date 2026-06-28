use super::*;

// -- RowFilter / apply_row_filters --

/// Default `RowFilter` (every field None/empty) matches every
/// row — it's the identity filter. Pins the no-op contract so a
/// future regression that flipped the default to a "match
/// nothing" semantic lands here.
#[test]
fn row_filter_default_matches_every_row() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.14.2"));
    let filter = RowFilter::default();
    assert!(filter.matches(&row), "empty filter must match every row");
}

/// `--scheduler` is strict equality, NOT substring. A filter of
/// `"scx"` does not match a row with scheduler `"scx_rusty"`.
/// Pins the typed-vs-substring asymmetry: -E stays as the
/// substring knob; typed flags exact-match.
#[test]
fn row_filter_scheduler_strict_equality_rejects_prefix() {
    let row = make_filter_row("t", "scx_rusty", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        schedulers: vec!["scx".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "strict-equality scheduler filter must NOT match a prefix; \
             got match for scheduler=`scx_rusty` against filter=`scx`",
    );
}

/// Exact scheduler match passes; the strict-equality contract's
/// happy path.
#[test]
fn row_filter_scheduler_strict_equality_matches_exact() {
    let row = make_filter_row("t", "scx_rusty", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        schedulers: vec!["scx_rusty".to_string()],
        ..RowFilter::default()
    };
    assert!(filter.matches(&row));
}

/// `--kernel 6.14.2` against a row whose `kernel_version` is
/// `None` must NOT match — the operator opted in to a specific
/// kernel and a None-row would silently dilute the filtered set.
#[test]
fn row_filter_kernel_none_row_never_matches_populated_filter() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        kernels: vec!["6.14.2".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "None-row must not match populated filter; got dilution",
    );
}

/// `--kernel 6.14.2` against a row whose `kernel_version` is
/// `Some("6.14.2")` matches.
#[test]
fn row_filter_kernel_exact_match() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.14.2"));
    let filter = RowFilter {
        kernels: vec!["6.14.2".to_string()],
        ..RowFilter::default()
    };
    assert!(filter.matches(&row));
}

/// `--kernel 6.14.2` against a row whose `kernel_version` is
/// `Some("6.14.3")` rejects.
#[test]
fn row_filter_kernel_mismatch_rejects() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.14.3"));
    let filter = RowFilter {
        kernels: vec!["6.14.2".to_string()],
        ..RowFilter::default()
    };
    assert!(!filter.matches(&row));
}

/// Repeatable `--kernel A --kernel B` is OR-combined: a row
/// matches iff its `kernel_version` equals ANY listed entry.
/// Pins the multi-value semantic.
#[test]
fn row_filter_kernels_or_combined_matches_any_listed() {
    let row_a = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.14.2"));
    let row_b = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.15.0"));
    let row_c = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.16.0"));
    let filter = RowFilter {
        kernels: vec!["6.14.2".to_string(), "6.15.0".to_string()],
        ..RowFilter::default()
    };
    assert!(filter.matches(&row_a), "first listed kernel must match");
    assert!(filter.matches(&row_b), "second listed kernel must match");
    assert!(
        !filter.matches(&row_c),
        "kernel outside the listed set must reject",
    );
}

/// Repeatable `--scheduler A --scheduler B` is OR-combined:
/// a row matches iff its `scheduler` equals ANY listed entry.
/// Pins the multi-value semantic for the
/// post-Vec-promotion `schedulers` field; before promotion
/// `--scheduler` was a single-value `Option<String>` and the
/// OR semantic did not exist.
#[test]
fn row_filter_schedulers_or_combined_matches_any_listed() {
    let row_a = make_filter_row("t", "scx_alpha", "1n2l4c1t", "SpinWait", None);
    let row_b = make_filter_row("t", "scx_beta", "1n2l4c1t", "SpinWait", None);
    let row_c = make_filter_row("t", "scx_gamma", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        schedulers: vec!["scx_alpha".to_string(), "scx_beta".to_string()],
        ..RowFilter::default()
    };
    assert!(filter.matches(&row_a), "first listed scheduler must match",);
    assert!(filter.matches(&row_b), "second listed scheduler must match",);
    assert!(
        !filter.matches(&row_c),
        "scheduler outside the listed set must reject",
    );
}

/// Repeatable `--topology A --topology B` is OR-combined:
/// a row matches iff its `topology` equals ANY listed entry.
/// Mirror of
/// `row_filter_schedulers_or_combined_matches_any_listed`
/// for the topologies field.
#[test]
fn row_filter_topologies_or_combined_matches_any_listed() {
    let row_a = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let row_b = make_filter_row("t", "scx_a", "1n2l4c2t", "SpinWait", None);
    let row_c = make_filter_row("t", "scx_a", "1n4l8c1t", "SpinWait", None);
    let filter = RowFilter {
        topologies: vec!["1n2l4c1t".to_string(), "1n2l4c2t".to_string()],
        ..RowFilter::default()
    };
    assert!(filter.matches(&row_a), "first listed topology must match",);
    assert!(filter.matches(&row_b), "second listed topology must match",);
    assert!(
        !filter.matches(&row_c),
        "topology outside the listed set must reject",
    );
}

/// Repeatable `--work-type A --work-type B` is OR-combined:
/// a row matches iff its `work_type` equals ANY listed
/// entry. Mirror of
/// `row_filter_schedulers_or_combined_matches_any_listed`
/// for the work_types field.
#[test]
fn row_filter_work_types_or_combined_matches_any_listed() {
    let row_a = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let row_b = make_filter_row("t", "scx_a", "1n2l4c1t", "PageFaultChurn", None);
    let row_c = make_filter_row("t", "scx_a", "1n2l4c1t", "MutexContention", None);
    let filter = RowFilter {
        work_types: vec!["SpinWait".to_string(), "PageFaultChurn".to_string()],
        ..RowFilter::default()
    };
    assert!(filter.matches(&row_a), "first listed work_type must match",);
    assert!(filter.matches(&row_b), "second listed work_type must match",);
    assert!(
        !filter.matches(&row_c),
        "work_type outside the listed set must reject",
    );
}

/// `--project-commit abcdef1` against a row whose `commit` is `None`
/// must NOT match — same opt-in policy as `--kernel`. Mirror
/// of `row_filter_kernel_none_row_never_matches_populated_filter`
/// for the project-commit field.
#[test]
fn row_filter_commit_none_row_never_matches_populated_filter() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        project_commits: vec!["abcdef1".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "None-commit row must not match populated filter; \
             got dilution",
    );
}

/// `--project-commit abcdef1` against a row whose `commit` is
/// `Some("abcdef1")` matches; `Some("other")` rejects.
/// Pins the strict-equality contract for commit, including
/// the OR-combined multi-value semantic and the `-dirty`
/// suffix's contribution to identity (a clean and dirty run
/// of the same HEAD bucket separately).
#[test]
fn row_filter_commit_exact_match_and_or_combined() {
    let mut row_clean = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_clean.commit = Some("abcdef1".to_string());
    let mut row_dirty = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_dirty.commit = Some("abcdef1-dirty".to_string());
    let mut row_other = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_other.commit = Some("fedcba2".to_string());

    let filter_single = RowFilter {
        project_commits: vec!["abcdef1".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter_single.matches(&row_clean),
        "exact commit match must succeed",
    );
    assert!(
        !filter_single.matches(&row_dirty),
        "`abcdef1-dirty` must NOT match a filter for `abcdef1` — \
             the suffix is part of identity, so the dirty run buckets \
             separately from the clean run of the same HEAD",
    );
    assert!(
        !filter_single.matches(&row_other),
        "different commit must reject",
    );

    let filter_or = RowFilter {
        project_commits: vec!["abcdef1".to_string(), "fedcba2".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter_or.matches(&row_clean),
        "first listed commit must match in OR-combined filter",
    );
    assert!(
        filter_or.matches(&row_other),
        "second listed commit must match in OR-combined filter",
    );
    assert!(
        !filter_or.matches(&row_dirty),
        "`abcdef1-dirty` must still reject — the suffix-bearing \
             form is its own identity even in OR-combined mode",
    );
}

/// `--kernel-commit kabcde7` against a row whose
/// `kernel_commit` is `None` must NOT match — same opt-in
/// policy as `--project-commit` and `--kernel`. Mirror of
/// `row_filter_commit_none_row_never_matches_populated_filter`
/// for the kernel-commit field.
#[test]
fn row_filter_kernel_commit_none_row_never_matches_populated_filter() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        kernel_commits: vec!["kabcde7".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "None-kernel-commit row must not match populated filter; \
             got dilution",
    );
}

/// `--kernel-commit kabcde7` against a row whose
/// `kernel_commit` is `Some("kabcde7")` matches;
/// `Some("other")` rejects. Pins the strict-equality
/// contract for kernel_commit, including the OR-combined
/// multi-value semantic and the `-dirty` suffix's
/// contribution to identity (a clean and dirty run of the
/// same kernel HEAD bucket separately).
#[test]
fn row_filter_kernel_commit_exact_match_and_or_combined() {
    let mut row_clean = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_clean.kernel_commit = Some("kabcde7".to_string());
    let mut row_dirty = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_dirty.kernel_commit = Some("kabcde7-dirty".to_string());
    let mut row_other = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_other.kernel_commit = Some("fedcba2".to_string());

    let filter_single = RowFilter {
        kernel_commits: vec!["kabcde7".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter_single.matches(&row_clean),
        "exact kernel_commit match must succeed",
    );
    assert!(
        !filter_single.matches(&row_dirty),
        "`kabcde7-dirty` must NOT match a filter for `kabcde7` — \
             the suffix is part of identity, so the dirty run buckets \
             separately from the clean run of the same kernel HEAD",
    );
    assert!(
        !filter_single.matches(&row_other),
        "different kernel_commit must reject",
    );

    let filter_or = RowFilter {
        kernel_commits: vec!["kabcde7".to_string(), "fedcba2".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter_or.matches(&row_clean),
        "first listed kernel_commit must match in OR-combined filter",
    );
    assert!(
        filter_or.matches(&row_other),
        "second listed kernel_commit must match in OR-combined filter",
    );
    assert!(
        !filter_or.matches(&row_dirty),
        "`kabcde7-dirty` must still reject — the suffix-bearing \
             form is its own identity even in OR-combined mode",
    );
}

/// `--kernel-commit` and `--project-commit` filter on DISTINCT row
/// fields. Pins the field non-aliasing: a row whose
/// `kernel_commit` matches but whose `commit` does not (or
/// vice versa) must reject. A regression that cross-wired
/// the `matches()` arms (e.g. `kernel_commits` checked
/// against `row.commit`) would silently dilute filtered
/// sets.
#[test]
fn row_filter_kernel_commit_and_commit_filter_distinct_fields() {
    let mut row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row.commit = Some("project1".to_string());
    row.kernel_commit = Some("kernel1".to_string());

    // Filter on kernel_commit only — commit dimension is unconstrained.
    let kc_only = RowFilter {
        kernel_commits: vec!["kernel1".to_string()],
        ..RowFilter::default()
    };
    assert!(
        kc_only.matches(&row),
        "kernel_commit match with no commit filter must accept",
    );

    let kc_mismatch = RowFilter {
        kernel_commits: vec!["project1".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !kc_mismatch.matches(&row),
        "kernel_commits filter must check `kernel_commit` not `commit` — \
             a regression that cross-wired the fields would accept here",
    );

    // Filter on commit only — kernel_commit dimension is unconstrained.
    let commit_mismatch = RowFilter {
        project_commits: vec!["kernel1".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !commit_mismatch.matches(&row),
        "project_commits filter must check `commit` not `kernel_commit` — \
             a regression that cross-wired the fields would accept here",
    );
}

/// `--run-source local` against a row whose `run_source` is
/// `None` must NOT match — same opt-in policy as `--kernel`,
/// `--project-commit`, and `--kernel-commit`. The operator wrote
/// specific tags and a None-row would silently dilute the
/// filtered set. Mirror of
/// `row_filter_kernel_commit_none_row_never_matches_populated_filter`
/// for the `run_source` field.
#[test]
fn row_filter_run_source_none_row_never_matches_populated_filter() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        run_sources: vec!["local".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "None-run_source row must not match populated filter; \
             got dilution",
    );
}

/// Repeatable `--run-source A --run-source B` is OR-combined: a row
/// matches iff its `run_source` equals ANY listed entry.
/// Mirror of `row_filter_kernels_or_combined_matches_any_listed`
/// for the `run_source` dimension.
#[test]
fn row_filter_run_sources_or_combined_matches_any_listed() {
    let mut row_local = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_local.run_source = Some("local".to_string());
    let mut row_ci = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_ci.run_source = Some("ci".to_string());
    let mut row_archive = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_archive.run_source = Some("archive".to_string());
    let filter = RowFilter {
        run_sources: vec!["local".to_string(), "ci".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter.matches(&row_local),
        "first listed run_source must match",
    );
    assert!(
        filter.matches(&row_ci),
        "second listed run_source must match",
    );
    assert!(
        !filter.matches(&row_archive),
        "run_source outside the listed set must reject",
    );
}

/// Repeatable `--resolve-source A --resolve-source B` is OR-combined: a
/// row matches iff its `resolve_source` equals ANY listed entry; a row
/// whose `resolve_source` is `None` never matches a populated filter.
/// Mirror of `row_filter_run_sources_or_combined_matches_any_listed` for
/// the `resolve_source` dimension.
#[test]
fn row_filter_resolve_sources_or_combined_matches_any_listed() {
    let mut row_auto = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_auto.resolve_source = Some("auto_built".to_string());
    let mut row_debug = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_debug.resolve_source = Some("target_debug".to_string());
    let mut row_path = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row_path.resolve_source = Some("path".to_string());
    let row_none = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let filter = RowFilter {
        resolve_sources: vec!["auto_built".to_string(), "target_debug".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter.matches(&row_auto),
        "first listed resolve_source must match",
    );
    assert!(
        filter.matches(&row_debug),
        "second listed resolve_source must match",
    );
    assert!(
        !filter.matches(&row_path),
        "resolve_source outside the listed set must reject",
    );
    assert!(
        !filter.matches(&row_none),
        "None resolve_source must never match a populated filter (opt-in policy)",
    );
}

/// `--run-source` and `--kernel-commit` filter on DISTINCT row
/// fields. Pins the field non-aliasing: a row whose
/// `run_source` matches but whose `kernel_commit` does not
/// (or vice versa) must reject. A regression that cross-wired
/// the `matches()` arms (e.g. `run_sources` checked against
/// `row.kernel_commit`) would silently dilute filtered sets.
/// Mirror of
/// `row_filter_kernel_commit_and_commit_filter_distinct_fields`
/// for the `run_source` × `kernel_commit` cross-wire surface.
#[test]
fn row_filter_run_sources_and_kernel_commits_are_distinct_fields() {
    let mut row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row.run_source = Some("local".to_string());
    row.kernel_commit = None;
    let filter = RowFilter {
        run_sources: vec!["local".to_string()],
        kernel_commits: vec!["abc1234".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "AND composition must reject when kernel_commit gate \
             fails (row's kernel_commit is None) even though the \
             run_source gate matches; a regression that cross-wired \
             run_sources against `row.kernel_commit` would accept here",
    );

    // Symmetric arm: run_source mismatches but kernel_commit
    // matches. Whole filter must still reject.
    let mut row2 = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    row2.run_source = Some("ci".to_string());
    row2.kernel_commit = Some("abc1234".to_string());
    let filter2 = RowFilter {
        run_sources: vec!["local".to_string()],
        kernel_commits: vec!["abc1234".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter2.matches(&row2),
        "AND composition must reject when run_source gate \
             fails even though kernel_commit gate passes; a \
             regression that cross-wired kernel_commits against \
             `row.run_source` would accept here",
    );
}

/// `--project-commit` and `--kernel` compose with AND semantics: a
/// populated commit filter and a populated kernel filter must
/// BOTH match for the row to survive. Pins the cross-field
/// composition rule for the new commit field, mirroring the
/// existing multi-field test for scheduler+topology+kernel.
#[test]
fn row_filter_commit_and_kernel_compose_and() {
    let mut row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.14.2"));
    row.commit = Some("abcdef1".to_string());
    let filter_both_match = RowFilter {
        kernels: vec!["6.14.2".to_string()],
        project_commits: vec!["abcdef1".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter_both_match.matches(&row),
        "both filters matching must accept the row",
    );
    let filter_kernel_only_match = RowFilter {
        kernels: vec!["6.14.2".to_string()],
        project_commits: vec!["fedcba2".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter_kernel_only_match.matches(&row),
        "AND composition must reject when commit mismatches even \
             though kernel matches",
    );
}

/// `--topology 1n2l4c1t` strict-equal against the row's
/// rendered topology. The filter is the same string the
/// `Topology::Display` impl emits and `cargo ktstr stats list`
/// shows; passing the exact form that appears in the listing
/// is the operator's expected workflow.
#[test]
fn row_filter_topology_strict_equality() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", None);
    let filter_match = RowFilter {
        topologies: vec!["1n2l4c1t".to_string()],
        ..RowFilter::default()
    };
    assert!(filter_match.matches(&row));
    let filter_miss = RowFilter {
        topologies: vec!["1n2l4c2t".to_string()],
        ..RowFilter::default()
    };
    assert!(!filter_miss.matches(&row));
}

/// Multiple typed filters compose with AND semantics: every
/// populated field must match. A mismatch on any one field
/// rejects the whole match. Pinned via a row that matches 3
/// of 4 filter fields and assertion that it still rejects.
#[test]
fn row_filter_multi_field_and_composes() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.14.2"));
    // 3 of 4 typed fields match (scheduler, topology, kernels);
    // work_type mismatches. Whole filter must reject.
    let filter = RowFilter {
        schedulers: vec!["scx_a".to_string()],
        topologies: vec!["1n2l4c1t".to_string()],
        kernels: vec!["6.14.2".to_string()],
        work_types: vec!["YieldHeavy".to_string()],
        ..RowFilter::default()
    };
    assert!(
        !filter.matches(&row),
        "AND composition must reject when any single field mismatches; \
             got match despite work_type divergence",
    );
}

/// `apply_row_filters` preserves the original row order and
/// drops only non-matching rows. Pinned by feeding a 3-row
/// vec where row 1 of 3 matches; result must be a 1-element
/// vec with the original middle row.
#[test]
fn apply_row_filters_preserves_order_drops_mismatch() {
    let rows = vec![
        make_filter_row("t1", "scx_a", "1n2l4c1t", "SpinWait", None),
        make_filter_row("t2", "scx_b", "1n2l4c1t", "SpinWait", None),
        make_filter_row("t3", "scx_a", "1n2l4c1t", "SpinWait", None),
    ];
    let filter = RowFilter {
        schedulers: vec!["scx_b".to_string()],
        ..RowFilter::default()
    };
    let kept = apply_row_filters(&rows, &filter);
    assert_eq!(kept.len(), 1, "expected 1 surviving row, got {kept:?}");
    assert_eq!(kept[0].scenario, "t2");
}

/// `apply_row_filters` with the default filter is the identity
/// — every row survives in original order.
#[test]
fn apply_row_filters_default_is_identity() {
    let rows = vec![
        make_filter_row("t1", "scx_a", "1n2l4c1t", "SpinWait", None),
        make_filter_row("t2", "scx_b", "1n2l4c2t", "YieldHeavy", Some("6.14.2")),
    ];
    let kept = apply_row_filters(&rows, &RowFilter::default());
    assert_eq!(kept.len(), rows.len());
    for (a, b) in kept.iter().zip(rows.iter()) {
        assert_eq!(a.scenario, b.scenario);
    }
}

// -- group_and_average_by / AveragedGroup --

/// Mutate a row's metric fields away from defaults so
/// aggregation has a non-zero signal to average. Returns the
/// row reference for chaining.
fn paint_metrics(row: &mut GauntletRow, spread: f64, gap_ms: u64, migrations: u64, iters: u64) {
    row.spread = spread;
    row.gap_ms = gap_ms;
    row.migrations = migrations;
    row.migration_ratio = spread / 100.0;
    row.imbalance_ratio = spread / 10.0;
    row.max_dsq_depth = (gap_ms / 10) as u32;
    row.stuck_count = (migrations / 10) as f64;
    row.fallback_count = migrations as i64;
    row.keep_last_count = -(migrations as i64);
    row.total_iterations = iters;
    // The wake / run-delay / iteration-efficiency / NUMA roll-ups are now
    // ext_metrics-sourced (Distribution / WorstLowest / WorstCrossNodeRatio); paint
    // them there so the cross-RUN ext fold (group_and_average_by → aggregate_finite)
    // exercises them: the percentile / CV / mean reductions and the WorstLowest /
    // WorstCrossNodeRatio selectors (`worst_page_locality`, `worst_iterations_*`,
    // `worst_cross_node_migration_ratio`) MEAN-fold cross-RUN, worst_run_delay_us
    // (Worst) MAX-folds.
    for (name, v) in [
        ("worst_p99_wake_latency_us", spread * 2.0),
        ("worst_median_wake_latency_us", spread),
        ("worst_wake_latency_cv", spread / 50.0),
        ("worst_mean_run_delay_us", gap_ms as f64),
        ("worst_run_delay_us", (gap_ms * 2) as f64),
        ("worst_iterations_per_worker", iters as f64 / 10.0),
        ("worst_iterations_per_cpu_sec", iters as f64 / 5.0),
        ("worst_page_locality", 1.0 - spread / 100.0),
        ("worst_cross_node_migration_ratio", spread / 200.0),
    ] {
        row.ext_metrics.insert(name.to_string(), v);
    }
}

/// Empty input produces zero aggregated rows. Pins the empty-
/// vec edge case so callers iterating over the result vector
/// don't need to special-case the `--average` path on empty
/// run directories.
#[test]
fn group_and_average_empty_input_yields_empty_output() {
    let out = group_and_average_by(&[], LEGACY_PAIRING_DIMS);
    assert!(out.is_empty());
}

/// Single passing contributor: aggregate is a faithful copy
/// of the input, with `passes_observed = total_observed = 1`.
/// Pins the trivial pass-through path so a regression in the
/// `denom` math (e.g. division by `total_observed` instead of
/// `passes_observed`) lands here.
#[test]
fn group_and_average_single_pass_passes_through_metrics() {
    let mut row = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut row, 12.0, 200, 50, 1000);
    let out = group_and_average_by(std::slice::from_ref(&row), LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert_eq!(ar.passes_observed, 1);
    assert_eq!(ar.total_observed, 1);
    assert!(ar.row.passed);
    assert!(!ar.row.skipped);
    assert_eq!(ar.row.spread, 12.0);
    assert_eq!(ar.row.gap_ms, 200);
    assert_eq!(ar.row.migrations, 50);
    assert_eq!(ar.row.total_iterations, 1000);
    assert_eq!(ar.row.fallback_count, 50);
    assert_eq!(ar.row.keep_last_count, -50);
    // worst_p99_wake_latency_us is now ext_metrics-sourced (Distribution);
    // single-pass pass-through carries it verbatim (spread*2 = 24.0).
    assert_eq!(
        ar.row.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(24.0),
    );
}

/// `stuck_count` folds as an EXACT `f64` mean — no rounding.
/// Three passing contributors with stuck_count 1, 1, 2 average to
/// 4/3 = 1.333..., NOT a rounded 1. Pins the fix for the rounding
/// bug: rounding the cross-run mean to an integer let the
/// up-to-1.0 per-A/B-pair error defeat stuck_count's
/// `default_abs` of 1.0 and fabricate single-stall regressions
/// from sub-integer differences. Every other typed integer field
/// rounds (abs >= 5.0 absorbs the error); stuck_count alone is f64.
#[test]
fn group_and_average_stuck_count_is_exact_fractional_mean() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.stuck_count = 1.0;
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.stuck_count = 1.0;
    let mut c = make_row("t", "tiny-1llc", true, 0.0);
    c.stuck_count = 2.0;
    let out = group_and_average_by(&[a, b, c], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1, "same-key contributors fold to one aggregate");
    let got = out[0].row.stuck_count;
    assert!(
        (got - 4.0 / 3.0).abs() < 1e-9,
        "stuck_count must fold to the EXACT fractional mean 1.333..., \
         not a rounded integer; got {got}",
    );
}

/// Three passing contributors with the same key are folded
/// into a single aggregate. Per-MetricKind cross-RUN fold:
/// Counter / Gauge(Last) typed fields take the arithmetic
/// mean (operator-natural cohort comparison); Peak typed
/// fields take the MAX (kind-correct — averaging Peak
/// dilutes the worst-instant signal). f64 means are exact
/// modulo IEEE rounding; u64/i64 means are rounded.
#[test]
fn group_and_average_multi_pass_kind_aware_fold() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut a, 10.0, 100, 30, 900);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut b, 20.0, 200, 60, 1100);
    let mut c = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut c, 30.0, 300, 90, 1000);
    let out = group_and_average_by(&[a, b, c], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert_eq!(ar.passes_observed, 3);
    assert_eq!(ar.total_observed, 3);
    assert!(ar.row.passed);
    assert!(!ar.row.skipped);
    // Gauge(Last) f64 mean: (10 + 20 + 30) / 3 = 20.0.
    assert_eq!(ar.row.spread, 20.0);
    // Peak u64 MAX (NOT mean — kind-correct cross-RUN fold).
    // Was 200 under arithmetic-mean; now 300.
    assert_eq!(ar.row.gap_ms, 300);
    // Counter u64 mean: (30 + 60 + 90) / 3 = 60.
    assert_eq!(ar.row.migrations, 60);
    // Counter u64 mean: (900 + 1100 + 1000) / 3 = 1000.
    assert_eq!(ar.row.total_iterations, 1000);
    // Counter i64 mean for fallback_count: (30 + 60 + 90)/3 = 60.
    assert_eq!(ar.row.fallback_count, 60);
    // Counter i64 mean for keep_last_count: (-30 + -60 + -90)/3 = -60.
    assert_eq!(ar.row.keep_last_count, -60);
    // Distribution worst_p99_wake_latency_us cross-RUN MEAN (unweighted)
    // through the ext fold: spread*2 = 20/40/60; (20 + 40 + 60)/3 = 40.
    assert_eq!(
        ar.row.ext_metrics.get("worst_p99_wake_latency_us").copied(),
        Some(40.0),
    );
    // WorstLowest worst_iterations_per_cpu_sec cross-RUN MEAN through the
    // ext fold: iters/5 = 180/220/200; (180 + 220 + 200)/3 = 200.
    assert_eq!(
        ar.row
            .ext_metrics
            .get("worst_iterations_per_cpu_sec")
            .copied(),
        Some(200.0),
    );
    // worst_run_delay_us is the SOLE Worst Distribution: cross-RUN it
    // folds by MAX (the peak survives), NOT mean — gap_ms*2 = 200/400/600
    // → MAX 600 (a MEAN would give 400). Pins the Worst arm AND its
    // ordering before the general Distribution MEAN arm in aggregate_finite.
    assert_eq!(
        ar.row.ext_metrics.get("worst_run_delay_us").copied(),
        Some(600.0),
    );
    // The remaining Distribution + WorstLowest reductions cross-RUN MEAN
    // (unweighted), locking the full MAX-vs-MEAN split: worst_median =
    // spread 10/20/30 → 20; worst_mean_run_delay_us = gap_ms 100/200/300
    // → 200; worst_iterations_per_worker = iters/10 90/110/100 → 100.
    assert_eq!(
        ar.row
            .ext_metrics
            .get("worst_median_wake_latency_us")
            .copied(),
        Some(20.0),
    );
    assert_eq!(
        ar.row.ext_metrics.get("worst_mean_run_delay_us").copied(),
        Some(200.0),
    );
    assert_eq!(
        ar.row
            .ext_metrics
            .get("worst_iterations_per_worker")
            .copied(),
        Some(100.0),
    );
    // WorstLowest worst_page_locality cross-RUN MEAN through the ext fold:
    // 1.0 - spread/100 = 0.9/0.8/0.7; (0.9 + 0.8 + 0.7)/3 = 0.8 (float-approx).
    let pl = ar
        .row
        .ext_metrics
        .get("worst_page_locality")
        .copied()
        .expect("worst_page_locality present");
    assert!(
        (pl - 0.8).abs() < 1e-9,
        "worst_page_locality cross-RUN MEAN ~0.8, got {pl}",
    );
    // WorstCrossNodeRatio worst_cross_node_migration_ratio cross-RUN MEAN through
    // the ext fold: spread/200 = 0.05/0.10/0.15; (0.05 + 0.10 + 0.15)/3 = 0.10.
    let xnode = ar
        .row
        .ext_metrics
        .get("worst_cross_node_migration_ratio")
        .copied()
        .expect("worst_cross_node_migration_ratio present");
    assert!(
        (xnode - 0.10).abs() < 1e-9,
        "worst_cross_node_migration_ratio cross-RUN MEAN ~0.10, got {xnode}",
    );
    // CV mean (spread/50 = 0.2/0.4/0.6 → 0.4) is float-approximate.
    let cv = ar
        .row
        .ext_metrics
        .get("worst_wake_latency_cv")
        .copied()
        .expect("worst_wake_latency_cv present");
    assert!(
        (cv - 0.4).abs() < 1e-9,
        "worst_wake_latency_cv cross-RUN MEAN ~0.4, got {cv}",
    );
}

/// The two monitor schedstat Rates re-derive Σnumerator / Σdenominator
/// across runs (the `MetricKind::Rate` pooled fold), NOT a mean of per-run
/// ratios. Two runs with deliberately different per-run ratios make the two
/// estimators disagree, so this pins the pooled form:
///   A: run_delay 1000 / pcount 1 = 1000; ttwu 1/1   = 1.0
///   B: run_delay 1000 / pcount 9 = ~111; ttwu 1/99  = ~0.0101
///   mean-of-ratios run_delay = 555.5  vs  pooled Σ/Σ = 2000/10 = 200
///   mean-of-ratios ttwu      = 0.505  vs  pooled Σ/Σ = 2/100  = 0.02
#[test]
fn group_and_average_schedstat_rates_pool_sigma_over_sigma() {
    let mk = |run_delay: f64, pcount: f64, local: f64, count: f64| {
        let mut r = make_row("t", "tiny-1llc", true, 0.0);
        r.ext_metrics.insert("total_run_delay".into(), run_delay);
        r.ext_metrics.insert("total_pcount".into(), pcount);
        r.ext_metrics.insert("total_ttwu_local".into(), local);
        r.ext_metrics.insert("total_ttwu_count".into(), count);
        r
    };
    let out = group_and_average_by(
        &[mk(1000.0, 1.0, 1.0, 1.0), mk(1000.0, 9.0, 1.0, 99.0)],
        LEGACY_PAIRING_DIMS,
    );
    assert_eq!(out.len(), 1);
    let row = &out[0].row;
    // Counter ext components SUM-fold across runs.
    assert_eq!(row.ext_metrics.get("total_run_delay").copied(), Some(2000.0));
    assert_eq!(row.ext_metrics.get("total_pcount").copied(), Some(10.0));
    assert_eq!(row.ext_metrics.get("total_ttwu_count").copied(), Some(100.0));
    assert_eq!(row.ext_metrics.get("total_ttwu_local").copied(), Some(2.0));
    // Rates re-derive Σ/Σ (the pooled mean), NOT the per-run mean-of-ratios.
    let per_sched = metric_def("total_run_delay_ns_per_sched")
        .unwrap()
        .read(row)
        .expect("rate derived post-fold");
    assert!(
        (per_sched - 200.0).abs() < 1e-6,
        "Σrun_delay/Σpcount = 200, got {per_sched} (mean-of-ratios would be 555.5)",
    );
    let frac = metric_def("ttwu_local_fraction")
        .unwrap()
        .read(row)
        .expect("rate derived post-fold");
    assert!(
        (frac - 0.02).abs() < 1e-9,
        "Σlocal/Σcount = 0.02, got {frac} (mean-of-ratios would be 0.505)",
    );
}

/// `sched_goidle_fraction` = Σsched_goidle / Σsched_count — the load-normalized
/// go-idle rate added for --noise-adjust spread. Derives per-run from the raw
/// counter ext keys, and pools Σ/Σ across runs (the `MetricKind::Rate` fold),
/// NOT a mean of per-run ratios.
#[test]
fn sched_goidle_fraction_derives_and_pools_sigma_over_sigma() {
    let mk = |goidle: f64, count: f64| {
        let mut r = make_row("t", "tiny-1llc", true, 0.0);
        r.ext_metrics.insert("total_sched_goidle".into(), goidle);
        r.ext_metrics.insert("total_sched_count".into(), count);
        r
    };
    // Single run: 30 goidle / 100 schedules = 0.30.
    let one = group_and_average_by(&[mk(30.0, 100.0)], LEGACY_PAIRING_DIMS);
    let frac = metric_def("sched_goidle_fraction")
        .unwrap()
        .read(&one[0].row)
        .expect("rate derived");
    assert!((frac - 0.30).abs() < 1e-9, "30/100 = 0.30, got {frac}");

    // Two runs with different per-run ratios → pooled Σ/Σ, not mean-of-ratios.
    // A: 1/1 = 1.0; B: 1/99 ≈ 0.0101; mean-of-ratios ≈ 0.505; pooled = 2/100 = 0.02.
    let out = group_and_average_by(&[mk(1.0, 1.0), mk(1.0, 99.0)], LEGACY_PAIRING_DIMS);
    let pooled = metric_def("sched_goidle_fraction")
        .unwrap()
        .read(&out[0].row)
        .expect("rate derived post-fold");
    assert!(
        (pooled - 0.02).abs() < 1e-9,
        "Σgoidle/Σcount = 0.02, got {pooled} (mean-of-ratios would be 0.505)",
    );
}

/// Whole-run taobench qps + hit Rates derive per-run from their Counter ext
/// components and pool Σ/Σ across runs (the `MetricKind::Rate` cross-run fold),
/// NOT a mean of per-run qps. Unequal wall windows make the two disagree: run A
/// 900 fast + 100 slow over 1 s, run B 100 fast + 2900 slow over 99 s. Σfast
/// 1000, Σslow 3000, Σops 4000, Σwall 100 s → total 40/s, fast 10/s, slow 30/s,
/// hit 1000/4000 = 0.25; the mean-of-ratios would give ~515 / ~451 / ~65 / 0.47.
#[test]
fn taobench_whole_run_rates_derive_and_pool_sigma_over_sigma() {
    let mk = |fast: f64, slow: f64, wall: f64| {
        let mut r = make_row("t", "tiny-1llc", true, 0.0);
        r.ext_metrics.insert("total_taobench_ops".into(), fast + slow);
        r.ext_metrics.insert("total_taobench_fast_ops".into(), fast);
        r.ext_metrics.insert("total_taobench_slow_ops".into(), slow);
        r.ext_metrics.insert("total_taobench_wall_sec".into(), wall);
        r
    };
    let read = |row: &_, name: &str| metric_def(name).unwrap().read(row).expect("rate derived");

    // Single run: 900 fast + 100 slow over 1 s.
    let one = group_and_average_by(&[mk(900.0, 100.0, 1.0)], LEGACY_PAIRING_DIMS);
    assert_eq!(read(&one[0].row, "taobench_total_ops_per_sec"), 1000.0);
    assert_eq!(read(&one[0].row, "taobench_fast_ops_per_sec"), 900.0);
    assert_eq!(read(&one[0].row, "taobench_slow_ops_per_sec"), 100.0);
    assert!((read(&one[0].row, "taobench_hit_fraction") - 0.9).abs() < 1e-9);

    // Two runs, unequal walls → pooled Σ/Σ, not mean-of-ratios.
    let out = group_and_average_by(
        &[mk(900.0, 100.0, 1.0), mk(100.0, 2900.0, 99.0)],
        LEGACY_PAIRING_DIMS,
    );
    assert_eq!(out.len(), 1);
    let row = &out[0].row;
    // Counter components SUM-fold across runs.
    assert_eq!(row.ext_metrics.get("total_taobench_ops").copied(), Some(4000.0));
    assert_eq!(
        row.ext_metrics.get("total_taobench_fast_ops").copied(),
        Some(1000.0),
    );
    assert_eq!(
        row.ext_metrics.get("total_taobench_slow_ops").copied(),
        Some(3000.0),
    );
    assert_eq!(
        row.ext_metrics.get("total_taobench_wall_sec").copied(),
        Some(100.0),
    );
    // Rates re-derive Σ/Σ (the pooled cohort throughput), NOT mean-of-ratios.
    let total = read(row, "taobench_total_ops_per_sec");
    assert!(
        (total - 40.0).abs() < 1e-9,
        "Σops/Σwall = 4000/100 = 40, got {total} (mean-of-ratios ~515)",
    );
    assert!((read(row, "taobench_fast_ops_per_sec") - 10.0).abs() < 1e-9);
    assert!((read(row, "taobench_slow_ops_per_sec") - 30.0).abs() < 1e-9);
    let hit = read(row, "taobench_hit_fraction");
    assert!(
        (hit - 0.25).abs() < 1e-9,
        "Σfast/Σops = 1000/4000 = 0.25, got {hit} (mean-of-ratios ~0.47)",
    );
}

/// `avg_nr_running` (Gauge(Avg) ext key) folds cross-run as the
/// SAMPLE-WEIGHTED pooled mean — Σ(avg_i × samples_i) / Σ samples_i, weighted by
/// run_sample_count — NOT the unweighted arithmetic mean a typed field would
/// give. Two runs with very different sample counts make the two disagree.
#[test]
fn group_and_average_avg_nr_running_is_sample_weighted_mean() {
    // Run A: avg 2.0 over 10 samples; Run B: avg 4.0 over 90 samples.
    // weighted   = (2*10 + 4*90) / (10+90) = 380/100 = 3.8
    // unweighted = (2 + 4) / 2 = 3.0  (what a typed mean-fold would give)
    let mk = |avg: f64, samples: usize| {
        let mut r = make_row("t", "tiny-1llc", true, 0.0);
        r.run_sample_count = samples;
        r.ext_metrics.insert("avg_nr_running".into(), avg);
        r
    };
    let out = group_and_average_by(&[mk(2.0, 10), mk(4.0, 90)], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let v = metric_def("avg_nr_running")
        .unwrap()
        .read(&out[0].row)
        .expect("avg_nr_running present after fold");
    assert!(
        (v - 3.8).abs() < 1e-9,
        "sample-weighted pooled mean = 3.8, got {v} (unweighted would be 3.0)",
    );
}

/// The cross-RUN unweighted mean of a Distribution/WorstLowest metric
/// divides by the count of contributors that EMITTED the key
/// (`finite.len()`), NOT by `passes_observed`: a passing run that omits the
/// key is EXCLUDED from the mean, not folded in as 0.0. Three passing runs,
/// only TWO carry `worst_p99_wake_latency_us` (20, 40) → aggregate is their
/// mean 30.0, NOT the (20+40+0)/3 = 20.0 a passes_observed divisor gives.
#[test]
fn group_and_average_distribution_excludes_key_omitting_run_from_mean() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.ext_metrics
        .insert("worst_p99_wake_latency_us".to_string(), 20.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.ext_metrics
        .insert("worst_p99_wake_latency_us".to_string(), 40.0);
    // Third passing run omits the key entirely.
    let c = make_row("t", "tiny-1llc", true, 0.0);
    let out = group_and_average_by(&[a, b, c], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].passes_observed, 3);
    assert_eq!(
        out[0]
            .row
            .ext_metrics
            .get("worst_p99_wake_latency_us")
            .copied(),
        Some(30.0),
        "unweighted mean over the 2 emitting runs (20+40)/2=30, NOT \
             (20+40+0)/3=20 a passes_observed divisor would give",
    );
}

/// Same cross-RUN exclude-missing MEAN, pinned on the relocated metric:
/// `worst_wake_latency_tail_ratio` (`MetricKind::WakeLatencyTailRatio`), whose
/// deleted typed fold was `sum / passes_observed` (folding sub-threshold runs
/// in as 0.0). Three passing runs, only TWO emit the key (2.0, 8.0) → aggregate
/// is their mean 5.0, NOT (2+8+0)/3 = 3.33 a passes_observed divisor gives. The
/// two emitters carry UNEQUAL `run_sample_count` (1000 vs 1) to prove the fold
/// is UNWEIGHTED — a sample-count-weighted mean would be (2*1000+8*1)/1001 ≈
/// 2.006, far from 5.0.
#[test]
fn group_and_average_tail_ratio_excludes_omitting_run_and_is_unweighted() {
    let key = "worst_wake_latency_tail_ratio";
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.run_sample_count = 1000;
    a.ext_metrics.insert(key.to_string(), 2.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.run_sample_count = 1;
    b.ext_metrics.insert(key.to_string(), 8.0);
    // Third passing run omits the key entirely.
    let c = make_row("t", "tiny-1llc", true, 0.0);
    let out = group_and_average_by(&[a, b, c], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].passes_observed, 3);
    assert_eq!(
        out[0].row.ext_metrics.get(key).copied(),
        Some(5.0),
        "unweighted mean over the 2 emitting runs (2+8)/2=5.0 — NOT \
             (2+8+0)/3=3.33 (passes_observed divisor), and NOT a sample-count \
             weighted mean (the 1000-vs-1 weights would pull it toward 2.0)",
    );
}

/// PerPhaseDeltaSum (`system_time_ns` / `user_time_ns`) folds cross-RUN by the
/// UNWEIGHTED mean over the runs that emitted the key: each run contributes one
/// per-phase-summed total, NOT weighted by `run_sample_count` (the monitor
/// capture count, an unrelated population). With run_sample_counts of 1000 and
/// 1 the unweighted mean (6000) and a sample-weighted mean (~7996) are
/// numerically distinct, pinning the unweighted fold specifically.
#[test]
fn group_and_average_per_phase_delta_sum_is_unweighted_mean_cross_run() {
    let key = "system_time_ns";
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.run_sample_count = 1000;
    a.ext_metrics.insert(key.to_string(), 8000.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.run_sample_count = 1;
    b.ext_metrics.insert(key.to_string(), 4000.0);
    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].row.ext_metrics.get(key).copied(),
        Some(6000.0),
        "unweighted mean over the 2 runs (8000+4000)/2=6000 — NOT a \
             run_sample_count-weighted mean (1000 vs 1 → ~7996)",
    );
}

/// `group_and_average_by` propagates `run_sample_count` to the
/// aggregated row's `run_sample_count` as the SUM of
/// contributor weights so a downstream consumer that further
/// folds the aggregated rows can apply the same weighted
/// semantic to the next-level cohort.
#[test]
fn group_and_average_run_sample_count_sums_across_contributors() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.run_sample_count = 5;
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.run_sample_count = 15;
    let mut c = make_row("t", "tiny-1llc", true, 0.0);
    c.run_sample_count = 30;
    let out = group_and_average_by(&[a, b, c], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].row.run_sample_count, 50);
}

/// Cross-RUN ext_metrics fold dispatches by registered
/// MetricKind — a registered `Gauge(Avg)` metric with two
/// contributors carrying different `run_sample_count` weights
/// uses the weighted mean rather than arithmetic mean.
/// (10 * 5 + 30 * 15) / (5 + 15) = 25.0 vs unweighted 20.0.
/// Uses `avg_dsq_depth` (registered as `Gauge(Avg)` per the
/// METRICS table) so the dispatch path is exercised against a
/// real registry entry, not a synthetic fixture.
#[test]
fn group_and_average_ext_metrics_gauge_avg_weighted_by_run_sample_count() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.run_sample_count = 5;
    a.ext_metrics.insert("avg_dsq_depth".to_string(), 10.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.run_sample_count = 15;
    b.ext_metrics.insert("avg_dsq_depth".to_string(), 30.0);
    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let mean = out[0]
        .row
        .ext_metrics
        .get("avg_dsq_depth")
        .copied()
        .expect("ext_metrics propagates avg_dsq_depth aggregate");
    // Weighted mean: (10*5 + 30*15) / 20 = 25.0.
    // Unweighted would be (10 + 30) / 2 = 20.0.
    assert!(
        (mean - 25.0).abs() < f64::EPSILON,
        "expected weighted mean 25.0, got {mean}",
    );
}

/// A passing run that emitted a Gauge(Avg) ext key but recorded zero monitor
/// samples (run_sample_count == 0) must still contribute ONE observation to the
/// cross-RUN weighted mean — never be silently zero-weighted out of a mixed
/// cohort. Pins the .max(1) weight floor in `Accumulator::observe` (matching the
/// floors in run_metrics.rs `populate_run_ext_metrics_from_phases` and
/// stats_types.rs `merge_metric_values`). row a (weight 0 -> floored to 1,
/// value 10) + row b (weight 10, value 30): (10*1 + 30*10) / (1 + 10) = 28.18.
/// Without the floor, a is dropped (weight 0) and the mean collapses to b's 30.0.
#[test]
fn group_and_average_gauge_avg_floors_zero_sample_count_weight() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.run_sample_count = 0;
    a.ext_metrics.insert("avg_dsq_depth".to_string(), 10.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.run_sample_count = 10;
    b.ext_metrics.insert("avg_dsq_depth".to_string(), 30.0);
    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let mean = out[0]
        .row
        .ext_metrics
        .get("avg_dsq_depth")
        .copied()
        .expect("ext_metrics propagates avg_dsq_depth aggregate");
    assert!(
        (mean - (310.0 / 11.0)).abs() < 1e-9,
        "weight-0 run must be floored to weight 1, not dropped: \
         expected (10*1 + 30*10)/11 = 28.18, got {mean}",
    );
}

/// Cross-RUN re-pool of the pooled `iterations_per_cpu_sec` Rate:
/// `group_and_average_by` SKIPS folding the Rate itself and re-derives it
/// from the folded Counter components (`total_iterations_pooled`,
/// `total_cpu_time_sec`). Registered Counters fold cross-RUN as a SUM
/// (`aggregate_finite` Counter arm = `finite.iter().sum()`, weight
/// ignored), so the components sum to Σnum / Σdenom = 1010 / 10.0 = 101.0 —
/// the true pooled rate, count-invariant, NOT the mean-of-per-run-ratios
/// (~500.6). The two components are co-inserted both-or-neither, so they
/// always share a contributor set; the SUM fold makes the rate identical
/// regardless of contributor count. The folded-COMPONENT assertions below
/// discriminate the SUM fold from a (wrong) hypothetical mean fold — the
/// rate value alone cannot, because Σ/Σ equals mean/mean when the
/// contributor count is equal (the N cancels); the component assertions
/// below discriminate. A stale per-run rate value is discarded by the
/// skip-then-derive path.
#[test]
fn group_and_average_repools_iterations_per_cpu_sec_from_components() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.ext_metrics
        .insert("total_iterations_pooled".to_string(), 1000.0);
    a.ext_metrics.insert("total_cpu_time_sec".to_string(), 1.0);
    // A stale per-run rate must be DISCARDED (a Rate is derived, never
    // folded from its own samples).
    a.ext_metrics
        .insert("iterations_per_cpu_sec".to_string(), 999.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.ext_metrics
        .insert("total_iterations_pooled".to_string(), 10.0);
    b.ext_metrics.insert("total_cpu_time_sec".to_string(), 9.0);
    b.ext_metrics
        .insert("iterations_per_cpu_sec".to_string(), 999.0);
    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    // Components fold as SUM (1000+10, 1.0+9.0), NOT mean (505, 5.0) — this
    // is what discriminates the fold mechanism the derived rate cannot.
    assert_eq!(
        out[0]
            .row
            .ext_metrics
            .get("total_iterations_pooled")
            .copied(),
        Some(1010.0),
        "numerator component folds cross-run as SUM (1010), not mean (505)",
    );
    assert_eq!(
        out[0].row.ext_metrics.get("total_cpu_time_sec").copied(),
        Some(10.0),
        "denominator component folds cross-run as SUM (10.0), not mean (5.0)",
    );
    let rate = out[0]
        .row
        .ext_metrics
        .get("iterations_per_cpu_sec")
        .copied()
        .expect("re-derived pooled rate present");
    // Σnum / Σdenom = 1010 / 10.0 = 101.0.
    assert!(
        (rate - 101.0).abs() < 1e-9,
        "cross-run pooled rate must re-derive to Σ/Σ = 101.0, not the \
             mean-of-ratios (~500.6) or the stale 999.0; got {rate}",
    );
}

/// Cross-RUN count-invariance vs a key-ABSENT run: two key-bearing passing
/// runs PLUS a third passing run with NO pooled component keys (all its
/// cgroups unmeasured, so populate_run_pooled_iterations_per_cpu_sec
/// inserted neither). The components SUM over the runs that carry them
/// (aggregate_finite Counter arm folds the present (value, weight) pairs),
/// so the key-absent run contributes NOTHING — the folded components and
/// the derived rate are identical to the two-run cohort. Asserting the
/// components (1010, 10.0 — NOT a mean-over-all-three 336.7, 3.33) guards
/// against a future regression that diluted the fold by treating a
/// key-absent run as a contributor (which a mean-over-all-runs fold would).
#[test]
fn group_and_average_pooled_rate_unaffected_by_key_absent_run() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.ext_metrics
        .insert("total_iterations_pooled".to_string(), 1000.0);
    a.ext_metrics.insert("total_cpu_time_sec".to_string(), 1.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.ext_metrics
        .insert("total_iterations_pooled".to_string(), 10.0);
    b.ext_metrics.insert("total_cpu_time_sec".to_string(), 9.0);
    // Third PASSING run with NO pooled component keys (all cgroups
    // unmeasured — populate_run_pooled inserted neither key).
    let c = make_row("t", "tiny-1llc", true, 0.0);
    let out = group_and_average_by(&[a, b, c], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    // Components SUM over the two key-bearing runs only; the key-absent
    // run contributes nothing — NOT diluted to (1010)/3 or (10.0)/3.
    assert_eq!(
        out[0]
            .row
            .ext_metrics
            .get("total_iterations_pooled")
            .copied(),
        Some(1010.0),
        "key-absent run must not dilute the summed numerator",
    );
    assert_eq!(
        out[0].row.ext_metrics.get("total_cpu_time_sec").copied(),
        Some(10.0),
        "key-absent run must not dilute the summed denominator",
    );
    // Rate identical to the two-run cohort: Σ/Σ = 1010/10.0 = 101.0.
    assert_eq!(
        out[0]
            .row
            .ext_metrics
            .get("iterations_per_cpu_sec")
            .copied(),
        Some(101.0),
        "key-absent run must not change the pooled rate (count-invariant)",
    );
}

/// Unregistered ext_metric keys fall back to arithmetic mean
/// (same legacy semantic the (sum, count) accumulator
/// produced). Pins that the weighted dispatch only fires for
/// METRICS-known keys; unknown keys ignore the weights.
#[test]
fn group_and_average_ext_metrics_unregistered_falls_back_to_arithmetic_mean() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.run_sample_count = 5;
    a.ext_metrics
        .insert("custom.unregistered".to_string(), 10.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.run_sample_count = 15;
    b.ext_metrics
        .insert("custom.unregistered".to_string(), 30.0);
    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    let mean = out[0]
        .row
        .ext_metrics
        .get("custom.unregistered")
        .copied()
        .expect("ext_metrics propagates custom key");
    // Arithmetic mean (legacy semantic): (10 + 30) / 2 = 20.0.
    assert!(
        (mean - 20.0).abs() < f64::EPSILON,
        "expected arithmetic mean 20.0, got {mean}",
    );
}

/// Different (scenario, topology, work_type) groups produce
/// distinct aggregates — the tuple is the join key. Pins the
/// group-key contract so a regression that dropped a key
/// component would land here as a collision.
#[test]
fn group_and_average_distinct_groups_stay_separate() {
    let mut a = make_row("alpha", "tiny-1llc", true, 0.0);
    paint_metrics(&mut a, 10.0, 100, 30, 1000);
    let mut b = make_row("beta", "tiny-1llc", true, 0.0);
    paint_metrics(&mut b, 50.0, 500, 100, 2000);
    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 2);
    // First-seen iteration order preserved (alpha before beta).
    assert_eq!(out[0].row.scenario, "alpha");
    assert_eq!(out[1].row.scenario, "beta");
}

/// Failing contributors are excluded from the metric mean and
/// flip the aggregate's `passed` to false. The aggregate's
/// `total_observed` still counts every contributor;
/// `passes_observed` counts only the clean ones.
#[test]
fn group_and_average_failed_contributors_excluded_from_mean_and_flag_aggregate() {
    let mut pass1 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass1, 10.0, 100, 30, 1000);
    let mut fail = make_row("t", "tiny-1llc", false, 0.0);
    // The failing row's metrics are pathologically large —
    // if they leaked into the mean, the aggregate's `spread`
    // would explode upward.
    paint_metrics(&mut fail, 10000.0, 99999, 99999, 99999);
    let mut pass2 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass2, 30.0, 300, 90, 1000);
    let out = group_and_average_by(&[pass1, fail, pass2], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert_eq!(ar.passes_observed, 2);
    assert_eq!(ar.total_observed, 3);
    // ALL-must-pass: a single failure flips the aggregate.
    assert!(
        !ar.row.passed,
        "any failing contributor must flip the aggregate to passed=false",
    );
    // Mean of only the passing entries' spread (Gauge(Last)):
    // (10 + 30) / 2 = 20.0. If the failing row leaked in,
    // this would be ~3346.
    assert_eq!(ar.row.spread, 20.0);
    // MAX of only the passing entries' gap_ms (Peak): max(100, 300) = 300.
    // If the failing row leaked into the max, it'd be 99999.
    assert_eq!(ar.row.gap_ms, 300);
}

/// Skipped contributors are excluded from the metric mean
/// and flip the aggregate's `skipped` to true (any-skipped
/// OR rule). `passes_observed` does not count them; the
/// passing-only entries still feed the mean cleanly.
#[test]
fn group_and_average_skipped_contributors_excluded_from_mean_and_flag_aggregate() {
    let mut pass1 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass1, 10.0, 100, 30, 1000);
    let mut skip = make_row("t", "tiny-1llc", true, 0.0);
    skip.skipped = true;
    // Pathological metrics on the skipped row to prove the
    // exclusion is real.
    paint_metrics(&mut skip, 9999.0, 99999, 99999, 99999);
    let mut pass2 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass2, 50.0, 500, 70, 2000);
    let out = group_and_average_by(&[pass1, skip, pass2], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert_eq!(ar.passes_observed, 2);
    assert_eq!(ar.total_observed, 3);
    assert!(
        ar.row.skipped,
        "any skipped contributor must flip the aggregate to skipped=true",
    );
    assert!(
        !ar.row.passed,
        "skipped aggregate must collapse `passed` to false so compare_rows \
             routes the pair through the excluded_pairs gate",
    );
    // Gauge(Last) mean of (pass1, pass2): (10 + 50)/2 = 30.0.
    assert_eq!(ar.row.spread, 30.0);
    // Peak MAX of (pass1, pass2) gap_ms: max(100, 500) = 500.
    // Was 300 under arithmetic-mean.
    assert_eq!(ar.row.gap_ms, 500);
}

/// Inconclusive contributors are excluded from the metric
/// mean and flip the aggregate's `inconclusive` to true (per
/// the `Fail > Inconclusive > Pass > Skip` lattice, an
/// Inconclusive contributor in an otherwise-passing cohort
/// dominates the verdict). `passes_observed` does not count
/// them; pathological metrics on the inconclusive row stay
/// out of the cohort means. Pins that the inconclusive bit
/// surfaces on the aggregate so downstream stats tooling can
/// distinguish a cohort that ran-but-couldn't-evaluate from
/// one that truly passed.
#[test]
fn group_and_average_inconclusive_contributors_excluded_from_mean_and_flag_aggregate() {
    let mut pass1 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass1, 10.0, 100, 30, 1000);
    // Inconclusive row: passed=false, skipped=false,
    // inconclusive=true. Pathological metrics on this row
    // must NOT leak into the mean.
    let mut inc = make_row("t", "tiny-1llc", false, 0.0);
    inc.inconclusive = true;
    paint_metrics(&mut inc, 7777.0, 77777, 77777, 77777);
    let mut pass2 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass2, 30.0, 300, 90, 2000);
    let out = group_and_average_by(&[pass1, inc, pass2], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert_eq!(ar.passes_observed, 2);
    assert_eq!(ar.total_observed, 3);
    assert!(
        ar.row.inconclusive,
        "any inconclusive contributor must flip the aggregate to inconclusive=true",
    );
    assert!(
        !ar.row.passed,
        "an inconclusive contributor must flip the aggregate to passed=false \
             (Inconclusive dominates Pass per the lattice)",
    );
    // Mean of only the passing entries' spread (Gauge(Last)):
    // (10 + 30) / 2 = 20.0. If the inconclusive row leaked
    // in, this would be ~2605.
    assert_eq!(ar.row.spread, 20.0);
    // MAX of only the passing entries' gap_ms (Peak):
    // max(100, 300) = 300. Was 77777 under leaked semantics.
    assert_eq!(ar.row.gap_ms, 300);
}

/// Fail dominates Inconclusive: a cohort with both a Fail and
/// an Inconclusive contributor produces `passed=false,
/// inconclusive=false` (Fail wins per the
/// `Fail > Inconclusive > Pass > Skip` lattice). Pins the
/// `inconclusive: acc.any_inconclusive && !acc.any_failed`
/// guard so the aggregate verdict surfaces the dominant Fail
/// signal rather than the lesser Inconclusive one.
#[test]
fn group_and_average_fail_dominates_inconclusive_in_aggregate_verdict() {
    let mut pass = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut pass, 10.0, 100, 30, 1000);
    let mut inc = make_row("t", "tiny-1llc", false, 0.0);
    inc.inconclusive = true;
    paint_metrics(&mut inc, 7777.0, 77777, 77777, 77777);
    let mut fail = make_row("t", "tiny-1llc", false, 0.0);
    paint_metrics(&mut fail, 9999.0, 99999, 99999, 99999);
    let out = group_and_average_by(&[pass, inc, fail], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert!(!ar.row.passed, "any Fail must flip passed to false");
    assert!(
        !ar.row.inconclusive,
        "Fail dominates Inconclusive: aggregate must surface as Fail \
             (inconclusive=false), not Inconclusive",
    );
}

/// All contributors fail: aggregate has `passes_observed = 0`,
/// `passed = false`, and zero metric values (no contributor
/// fed the running sums). Pins the divide-by-zero guard:
/// `denom` must default to 1.0 when `passes_observed = 0`.
#[test]
fn group_and_average_all_failed_collapses_to_default_zero_metrics_and_failed_flag() {
    let mut fail1 = make_row("t", "tiny-1llc", false, 0.0);
    paint_metrics(&mut fail1, 99.0, 999, 99, 999);
    let mut fail2 = make_row("t", "tiny-1llc", false, 0.0);
    paint_metrics(&mut fail2, 88.0, 888, 88, 888);
    let out = group_and_average_by(&[fail1, fail2], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    assert_eq!(ar.passes_observed, 0);
    assert_eq!(ar.total_observed, 2);
    assert!(!ar.row.passed);
    // Failed-only group: every metric collapses to its zero
    // default. The aggregate's `passed=false` then routes the
    // pair through compare_rows' excluded_pairs gate.
    assert_eq!(ar.row.spread, 0.0);
    assert_eq!(ar.row.gap_ms, 0);
    assert_eq!(ar.row.migrations, 0);
}

/// `ext_metrics` keys are unioned across passing
/// contributors; each key averages over the contributors
/// that carried it. A key absent on some passing rows is
/// NOT treated as a stored zero — its denominator is the
/// present-only count.
#[test]
fn group_and_average_ext_metrics_average_per_key_present_count() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.ext_metrics.insert("shared".into(), 10.0);
    a.ext_metrics.insert("a_only".into(), 100.0);
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.ext_metrics.insert("shared".into(), 30.0);
    b.ext_metrics.insert("b_only".into(), 200.0);
    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    let ar = &out[0];
    // shared: (10 + 30) / 2 = 20.
    assert_eq!(ar.row.ext_metrics.get("shared"), Some(&20.0));
    // a_only: present only in a → mean over 1 entry = 100.
    assert_eq!(ar.row.ext_metrics.get("a_only"), Some(&100.0));
    // b_only: present only in b → mean over 1 entry = 200.
    assert_eq!(ar.row.ext_metrics.get("b_only"), Some(&200.0));
}

/// `group_and_average_by` preserves first-seen iteration order so
/// downstream tests against the result remain deterministic
/// even though the internal map uses BTreeMap (key-sorted)
/// for storage. Pinned by feeding keys in z→a order and
/// asserting the output keeps that order.
#[test]
fn group_and_average_preserves_first_seen_order() {
    let zebra = make_row("zebra", "tiny-1llc", true, 0.0);
    let alpha = make_row("alpha", "tiny-1llc", true, 0.0);
    let mango = make_row("mango", "tiny-1llc", true, 0.0);
    let out = group_and_average_by(&[zebra, alpha, mango], LEGACY_PAIRING_DIMS);
    let names: Vec<&str> = out.iter().map(|r| r.row.scenario.as_str()).collect();
    assert_eq!(
        names,
        vec!["zebra", "alpha", "mango"],
        "output must follow first-seen iteration order, not key sort",
    );
}

/// Cohort with mixed clean/dirty `commit` values (same hex)
/// renders with `+mixed` appended to the canonical
/// un-suffixed hex. First contributor is dirty; the second
/// is clean. Pinning the rendered form catches a regression
/// where averaging silently kept first-seen behaviour and
/// hid the WIP-vs-committed disagreement.
#[test]
fn group_and_average_mixed_dirty_project_commit_renders_plus_mixed() {
    let mut dirty = make_row("t", "tiny-1llc", true, 0.0);
    dirty.commit = Some("abc1234-dirty".to_string());
    let mut clean = make_row("t", "tiny-1llc", true, 0.0);
    clean.commit = Some("abc1234".to_string());

    let out = group_and_average_by(&[dirty, clean], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].row.commit.as_deref(),
        Some("abc1234+mixed"),
        "mixed clean+dirty must render as `{{hex}}+mixed`, not first-seen",
    );
}

/// Same shape on `kernel_commit`. Pins the second commit
/// dimension separately because the production code uses
/// two parallel accumulator-state pairs and a regression
/// could miss one.
#[test]
fn group_and_average_mixed_dirty_kernel_commit_renders_plus_mixed() {
    let mut clean = make_row("t", "tiny-1llc", true, 0.0);
    clean.kernel_commit = Some("def5678".to_string());
    let mut dirty = make_row("t", "tiny-1llc", true, 0.0);
    dirty.kernel_commit = Some("def5678-dirty".to_string());

    let out = group_and_average_by(&[clean, dirty], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].row.kernel_commit.as_deref(),
        Some("def5678+mixed"),
        "mixed clean+dirty kernel_commit must render as `{{hex}}+mixed`",
    );
}

/// Homogeneous-dirty cohort (every contributor has `-dirty`)
/// must NOT receive the `+mixed` marker — the cohort agrees
/// on the working-tree state. Pinning this guards against a
/// regression where the marker fires on every dirty value
/// regardless of clean siblings.
#[test]
fn group_and_average_all_dirty_keeps_dirty_suffix_no_mixed() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.commit = Some("abc1234-dirty".to_string());
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.commit = Some("abc1234-dirty".to_string());

    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(
        out[0].row.commit.as_deref(),
        Some("abc1234-dirty"),
        "homogeneous-dirty cohort must keep first-seen `-dirty`, no `+mixed`",
    );
}

/// Homogeneous-clean cohort (every contributor lacks
/// `-dirty`) keeps the un-suffixed first-seen value, no
/// marker.
#[test]
fn group_and_average_all_clean_keeps_value_no_mixed() {
    let mut a = make_row("t", "tiny-1llc", true, 0.0);
    a.commit = Some("abc1234".to_string());
    let mut b = make_row("t", "tiny-1llc", true, 0.0);
    b.commit = Some("abc1234".to_string());

    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert_eq!(
        out[0].row.commit.as_deref(),
        Some("abc1234"),
        "homogeneous-clean cohort must keep first-seen value, no `+mixed`",
    );
}

/// Skipped contributors participate in mixed-dirty tracking.
/// The cohort's WIP state is metadata independent of metric
/// outcome — a skipped sidecar from a dirty tree still
/// counts toward the dirty-flag because it records the
/// producer's working-tree state at run time. Pin: one
/// passing-clean + one skipped-dirty contributor renders
/// `+mixed`.
///
/// Tests the SKIPPED arm only (`passed=true, skipped=true`).
/// The failed arm (`passed=false, skipped=false`) is pinned
/// separately by
/// `group_and_average_mixed_dirty_tracking_includes_failed_contributors`
/// — the two arms exit through distinct `continue` statements
/// in `group_and_average_by` and a regression in either is
/// independent of the other.
#[test]
fn group_and_average_mixed_dirty_tracking_includes_skipped() {
    let mut clean_pass = make_row("t", "tiny-1llc", true, 0.0);
    clean_pass.commit = Some("abc1234".to_string());
    let mut dirty_skip = make_row("t", "tiny-1llc", true, 0.0);
    dirty_skip.skipped = true;
    dirty_skip.commit = Some("abc1234-dirty".to_string());

    let out = group_and_average_by(&[clean_pass, dirty_skip], LEGACY_PAIRING_DIMS);
    assert_eq!(
        out[0].row.commit.as_deref(),
        Some("abc1234+mixed"),
        "skipped contributors still flip the dirty flag — \
             cohort metadata is independent of metric outcome",
    );
}

/// Failed contributor pin: a passing-clean row paired with a
/// FAILING-dirty row (`passed=false`, `skipped=false`) must
/// still flip the cohort's mixed-dirty flag and render
/// `+mixed` on the aggregate's commit field. The
/// `update_dirty_tracking` call site executes BEFORE the
/// `if !row.passed { continue; }` short-circuit, which is
/// the load-bearing ordering: dirty-status is per-row
/// metadata about the producer's working tree, NOT a metric
/// outcome, so failed contributors must carry their dirty
/// flag forward even though their metrics are excluded from
/// the mean. A regression that moved `update_dirty_tracking`
/// below the failed-skip continue would silently drop the
/// failed row's dirty status and the cohort would render the
/// clean form — hiding WIP-vs-committed disagreement that
/// the operator needs to see.
///
/// Distinct from
/// `group_and_average_mixed_dirty_tracking_includes_skipped`
/// which exercises the SKIPPED arm only (`passed=true,
/// skipped=true`). The two arms have separate `continue`
/// statements and one could regress without the other; this
/// test pins the FAILED arm specifically.
#[test]
fn group_and_average_mixed_dirty_tracking_includes_failed_contributors() {
    let mut clean_pass = make_row("t", "tiny-1llc", true, 0.0);
    clean_pass.commit = Some("abc1234".to_string());
    let mut dirty_fail = make_row("t", "tiny-1llc", false, 0.0);
    dirty_fail.commit = Some("abc1234-dirty".to_string());

    let out = group_and_average_by(&[clean_pass, dirty_fail], LEGACY_PAIRING_DIMS);
    assert_eq!(out.len(), 1, "single cohort key must produce one aggregate");
    assert_eq!(
        out[0].row.commit.as_deref(),
        Some("abc1234+mixed"),
        "failed contributor's `-dirty` flag must still flip the \
             cohort's dirty-tracking — cohort metadata is independent \
             of metric outcome. A regression moving update_dirty_tracking \
             below the `if !row.passed` continue would drop the failed \
             row's dirty status and render `abc1234` instead",
    );
    // Symmetric arm: passing-dirty + failing-clean. The
    // dirty-tracking flip on the failing contributor's clean
    // form must register as well — `any_clean` is the
    // counterpart flag, and the same code path executes for
    // both `Some(hex)` and `Some(hex-dirty)` values.
    let mut dirty_pass = make_row("t", "tiny-1llc", true, 0.0);
    dirty_pass.commit = Some("def5678-dirty".to_string());
    let mut clean_fail = make_row("t", "tiny-1llc", false, 0.0);
    clean_fail.commit = Some("def5678".to_string());

    let out = group_and_average_by(&[dirty_pass, clean_fail], LEGACY_PAIRING_DIMS);
    assert_eq!(
        out[0].row.commit.as_deref(),
        Some("def5678+mixed"),
        "failed contributor's CLEAN form must also flip the \
             cohort's any_clean flag — symmetric to the dirty arm",
    );
    // Failed contributor's `passed=false` still flips the
    // aggregate's `passed` flag (logical-AND across all
    // contributors). This sanity-checks that the new test
    // doesn't accidentally exercise an aggregate-passes path
    // — failed rows are correctly being excluded from the
    // metric mean while contributing to dirty tracking.
    assert!(
        !out[0].row.passed,
        "any failing contributor must flip the aggregate to \
             passed=false, regardless of dirty-tracking semantics",
    );
}

/// Mixed-dirty marker uses canonical un-suffixed hex even
/// when `acc.first` is the dirty form. Pin: first contributor
/// is `abc1234-dirty`, second is `abc1234`; rendered form is
/// `abc1234+mixed`, NOT `abc1234-dirty+mixed`. Guards against
/// a stripping bug in `render_mixed_dirty`.
#[test]
fn group_and_average_mixed_dirty_strips_dirty_from_first_seen() {
    let mut dirty_first = make_row("t", "tiny-1llc", true, 0.0);
    dirty_first.commit = Some("abc1234-dirty".to_string());
    let mut clean_second = make_row("t", "tiny-1llc", true, 0.0);
    clean_second.commit = Some("abc1234".to_string());

    let out = group_and_average_by(&[dirty_first, clean_second], LEGACY_PAIRING_DIMS);
    let rendered = out[0].row.commit.as_deref().expect("commit must render");
    assert_eq!(rendered, "abc1234+mixed");
    assert!(
        !rendered.contains("-dirty"),
        "rendered form must drop `-dirty` even when first contributor was dirty; got: {rendered}",
    );
}

/// `None`-only cohort keeps `None`. Sanity check that the
/// dirty-tracking does not synthesize a marker when no
/// contributor has a commit value.
#[test]
fn group_and_average_all_none_commits_keeps_none_no_mixed() {
    let a = make_row("t", "tiny-1llc", true, 0.0);
    let b = make_row("t", "tiny-1llc", true, 0.0);

    let out = group_and_average_by(&[a, b], LEGACY_PAIRING_DIMS);
    assert!(
        out[0].row.commit.is_none(),
        "None-only cohort must keep None — no synthesized `+mixed`",
    );
}

/// End-to-end: aggregated rows feed `compare_rows` cleanly.
/// Side A has [10, 12, 14] (mean 12); side B has [28, 30, 32]
/// (mean 30). The 18-unit delta on `worst_spread`
/// (default_abs=5.0, default_rel=0.25) clears both gates,
/// producing a regression. Pins the full averaging pipeline.
#[test]
fn group_and_average_then_compare_rows_yields_regression_on_means() {
    let mut a1 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut a1, 10.0, 100, 30, 1000);
    let mut a2 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut a2, 12.0, 120, 35, 1000);
    let mut a3 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut a3, 14.0, 140, 40, 1000);
    let mut b1 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut b1, 28.0, 280, 70, 1000);
    let mut b2 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut b2, 30.0, 300, 75, 1000);
    let mut b3 = make_row("t", "tiny-1llc", true, 0.0);
    paint_metrics(&mut b3, 32.0, 320, 80, 1000);

    let agg_a = group_and_average_by(&[a1, a2, a3], LEGACY_PAIRING_DIMS);
    let agg_b = group_and_average_by(&[b1, b2, b3], LEGACY_PAIRING_DIMS);
    let rows_a: Vec<GauntletRow> = agg_a.iter().map(|r| r.row.clone()).collect();
    let rows_b: Vec<GauntletRow> = agg_b.iter().map(|r| r.row.clone()).collect();
    let res = compare_rows_by(
        &rows_a,
        &rows_b,
        LEGACY_PAIRING_DIMS,
        None,
        &ComparisonPolicy::default(),
    );
    let spread = res
        .findings
        .iter()
        .find(|f| f.metric.name == "worst_spread")
        .expect("worst_spread must regress on aggregated means");
    assert!(spread.kind == FindingKind::Regression);
    assert_eq!(spread.val_a, 12.0, "mean of [10, 12, 14] = 12");
    assert_eq!(spread.val_b, 30.0, "mean of [28, 30, 32] = 30");
    assert_eq!(spread.delta, 18.0);
}

/// `compare_partitions` with the default (averaging-on)
/// path must aggregate every matching sidecar within each
/// side and detect regressions on the aggregated means.
/// End-to-end pin against on-disk fixtures so a regression
/// in the aggregation → compare wiring lands here.
///
/// Fixture: two runs each carrying three sidecars that
/// differ on `scheduler` (the slicing dim). Side A's three
/// trials cluster around `worst_spread = 10` (mean 12);
/// side B's three cluster around `worst_spread = 30` (mean
/// 30). The 18-unit delta clears the default dual gate, so
/// `compare_partitions` returns exit code 1 (regressions
/// detected).
#[test]
fn compare_partitions_with_average_default_produces_regression_on_aggregated_means() {
    use crate::test_support::SidecarResult;

    let alt_root = tempfile::TempDir::new().expect("create alt-root tempdir");
    let run_a = "__avg_thread_a__";
    let run_b = "__avg_thread_b__";

    // Three trials per side, same (scenario, topology,
    // work_type) so they aggregate into a single key. Vary
    // the per-trial spread so the mean is non-degenerate
    // (regression flags would also fire if the values were
    // identical, but the average path is exercised either way).
    let trials_a = [(10.0, 100), (12.0, 120), (14.0, 140)];
    let trials_b = [(28.0, 280), (30.0, 300), (32.0, 320)];

    // Scheduler is the slicing dim: side A's three trials
    // run under "scx_alpha", side B's under "scx_beta". The
    // pairing dims are everything else (kernel/topology/
    // work_type/commit) which match across both runs,
    // so the three trials on each side aggregate into one
    // mean row keyed by `(scenario, topology, work_type)`
    // plus the matching kernel/commit values.
    for (run_key, trials, sched) in [
        (run_a, &trials_a, "scx_alpha"),
        (run_b, &trials_b, "scx_beta"),
    ] {
        let run_dir = alt_root.path().join(run_key);
        std::fs::create_dir_all(&run_dir).expect("create run dir");
        for (i, (spread, gap_ms)) in trials.iter().enumerate() {
            let trial_name = format!("avg_trial_{run_key}_{i}");
            let mut sidecar = SidecarResult {
                test_name: "avg_test".to_string(),
                topology: "1n2l4c1t".to_string(),
                scheduler: sched.to_string(),
                work_type: "SpinWait".to_string(),
                ..SidecarResult::test_fixture()
            };
            sidecar.stats.worst_spread = *spread;
            sidecar.stats.worst_gap_ms = *gap_ms;
            sidecar.passed = true;
            sidecar.skipped = false;
            let json = serde_json::to_string(&sidecar).expect("serialize fixture sidecar");
            let sidecar_path = run_dir.join(format!("{trial_name}.ktstr.json"));
            std::fs::write(&sidecar_path, json).expect("write fixture sidecar");
        }
    }

    let filter_a = RowFilter {
        schedulers: vec!["scx_alpha".to_string()],
        ..RowFilter::default()
    };
    let filter_b = RowFilter {
        schedulers: vec!["scx_beta".to_string()],
        ..RowFilter::default()
    };

    // Default (averaging-on) path: three sidecars per side
    // share one pairing key, so each side aggregates to a
    // single mean row. The 18-unit worst_spread delta on
    // those means (12 vs 30) clears the default dual gate
    // and surfaces exit code 1.
    let exit = compare_partitions(
        &filter_a,
        &filter_b,
        None,
        &ComparisonPolicy::default(),
        Some(alt_root.path()),
        false, // no_average=false → averaging is ON
        &PhaseDisplayOptions::default(),
    )
    .expect("compare_partitions must succeed against valid fixtures");
    assert_eq!(
        exit, 1,
        "an 18-unit worst_spread regression on the aggregated mean \
             (a=12 → b=30) must clear the default dual gate and surface \
             exit code 1; got {exit}",
    );
}
