use super::*;

/// `PhaseDisplayOptions::rel_threshold` returns the
/// `phase_threshold` percent divided by 100 (the override
/// branch) when the flag is set, regardless of what
/// `ComparisonPolicy` says. Confirms `--phase-threshold X`
/// is the sole determinant of per-phase relative-gate
/// resolution at the override branch.
#[test]
fn phase_display_options_rel_threshold_override_branch_takes_precedence() {
    let opts = PhaseDisplayOptions {
        phase_threshold: Some(25.0), // 25%
        ..PhaseDisplayOptions::default()
    };
    let mut policy = ComparisonPolicy::default();
    policy
        .per_metric_percent
        .insert("max_dsq_depth".into(), 99.0);
    let resolved = opts.rel_threshold(&policy, "max_dsq_depth", 0.50);
    assert_eq!(
        resolved, 0.25,
        "--phase-threshold 25 → 0.25 fraction regardless of policy override"
    );
}

/// `PhaseDisplayOptions::rel_threshold` falls through to
/// `ComparisonPolicy::rel_threshold` when `phase_threshold`
/// is absent. The policy's per-metric or default-percent
/// override applies (the operator's CLI surface). Pins the
/// fallback chain so a `--policy file.json` invocation with
/// no `--phase-threshold` continues to feed the per-metric
/// thresholds into the per-phase pass.
#[test]
fn phase_display_options_rel_threshold_falls_through_to_policy_when_unset() {
    let opts = PhaseDisplayOptions::default(); // phase_threshold = None
    let mut policy = ComparisonPolicy::default();
    policy
        .per_metric_percent
        .insert("max_dsq_depth".into(), 30.0); // 30%
    let resolved = opts.rel_threshold(&policy, "max_dsq_depth", 0.50);
    assert_eq!(
        resolved, 0.30,
        "absent --phase-threshold → policy.rel_threshold = 30% / 100 = 0.30"
    );
}

/// When neither `PhaseDisplayOptions::phase_threshold` nor
/// any `ComparisonPolicy` override applies, fall through to
/// the registry default (e.g. `MetricDef.default_rel = 0.50`
/// for `max_dsq_depth`). Pins the "no flag at all" path so
/// the dual-gate semantic at the data layer continues to
/// use the metric registry as the source of truth.
#[test]
fn phase_display_options_rel_threshold_falls_through_to_registry_default() {
    let opts = PhaseDisplayOptions::default();
    let policy = ComparisonPolicy::default(); // no overrides
    let resolved = opts.rel_threshold(&policy, "max_dsq_depth", 0.50);
    assert_eq!(
        resolved, 0.50,
        "no flag, no policy → registry default_rel = 0.50"
    );
}

// -- PhaseDisplayOptions::matches_phase (step-axis filter) --

/// Default-shape opts (`--phase` None, `--steps-only` false)
/// match every step_index. Pins the "no flag, every step
/// renders" default path that the earlier per-phase pass tests
/// implicitly assume but never assert directly on the
/// extracted helper.
#[test]
fn matches_phase_default_opts_pass_all_steps() {
    let opts = PhaseDisplayOptions::default();
    for step in [0u16, 1, 2, 7, 65535] {
        assert!(
            opts.matches_phase(step),
            "default opts must match step_index = {step}"
        );
    }
}

/// `--steps-only` suppresses BASELINE (step_index = 0) and
/// admits every other step. Pins the single-step exclusion
/// behavior and confirms that no other steps are
/// accidentally caught by the gate.
#[test]
fn matches_phase_steps_only_suppresses_only_baseline() {
    let opts = PhaseDisplayOptions {
        steps_only: true,
        ..PhaseDisplayOptions::default()
    };
    assert!(
        !opts.matches_phase(0),
        "--steps-only: step_index = 0 (BASELINE) must be suppressed"
    );
    for step in [1u16, 2, 3, 7, 65535] {
        assert!(
            opts.matches_phase(step),
            "--steps-only: step_index = {step} must NOT be suppressed"
        );
    }
}

/// `--phase N` keeps only step_index == N and rejects every
/// other step (including BASELINE when N != 0). Pins the
/// single-phase filter.
#[test]
fn matches_phase_phase_filter_keeps_only_target_step() {
    let opts = PhaseDisplayOptions {
        phase: Some(2),
        ..PhaseDisplayOptions::default()
    };
    assert!(opts.matches_phase(2), "--phase 2 must keep step_index = 2");
    for step in [0u16, 1, 3, 7, 65535] {
        assert!(
            !opts.matches_phase(step),
            "--phase 2: step_index = {step} must be suppressed"
        );
    }
}

/// `--phase 0` keeps BASELINE only — the
/// `--phase`-via-step-zero arm. Confirms an operator can
/// explicitly request BASELINE through the `--phase` flag
/// rather than only through "no flag at all" default.
/// (Mutually exclusive with `--steps-only` at the CLI parse
/// layer; here we test the method in isolation.)
#[test]
fn matches_phase_phase_zero_keeps_only_baseline() {
    let opts = PhaseDisplayOptions {
        phase: Some(0),
        ..PhaseDisplayOptions::default()
    };
    assert!(opts.matches_phase(0), "--phase 0 must keep step_index = 0");
    for step in [1u16, 2, 7, 65535] {
        assert!(
            !opts.matches_phase(step),
            "--phase 0: step_index = {step} must be suppressed"
        );
    }
}

// -- format_average_header / format_per_group_pass_counts --

/// `format_average_header` renders the exact header line that
/// `compare_partitions` prints above the comparison table.
/// Pins the operator-visible surface
/// (the "averaged across N runs (A) and M runs (B)" string)
/// so a regression that reworded the header without
/// updating downstream parsers / scripts lands here.
#[test]
fn format_average_header_exact_string() {
    let out = format_average_header(5, 3, "kernel-6.14", "kernel-6.15");
    assert_eq!(
        out,
        "averaged across 5 runs (kernel-6.14) and 3 runs (kernel-6.15)",
    );
}

/// Zero-contributor sides are surfaced verbatim — operator
/// will see `0 runs` for an empty side. Pins the empty-side
/// edge case so a regression that special-cased `pre_agg = 0`
/// (e.g. omitted the side, said "no contributors") would
/// fail here. The companion empty-rows path is already
/// guarded upstream by `compare_partitions`' `sidecars_*.is_empty()`
/// bail; this test guards the formatter itself in case it's
/// reused outside the compare path.
#[test]
fn format_average_header_zero_contributor_sides_render_verbatim() {
    assert_eq!(
        format_average_header(0, 0, "a", "b"),
        "averaged across 0 runs (a) and 0 runs (b)",
    );
}

/// Helper for the per-group-block tests: build an
/// `AveragedGroup` with the named identity and pass counters
/// while leaving every metric field at zero. Metrics aren't
/// observed by [`format_per_group_pass_counts`] — only the
/// identity tuple and pass counters drive the output.
fn group(
    scenario: &str,
    topology: &str,
    work_type: &str,
    passes_observed: u32,
    total_observed: u32,
) -> AveragedGroup {
    let mut row = make_row(scenario, topology, true, 0.0);
    row.work_type = work_type.into();
    AveragedGroup {
        row,
        passes_observed,
        skips_observed: 0,
        inconclusives_observed: 0,
        failures_observed: 0,
        total_observed,
    }
}

/// Build an [`AveragedGroup`] with non-zero skip / inc / fail
/// counts for the per-group breakdown test below.
fn group_with_breakdown(
    scenario: &str,
    topology: &str,
    work_type: &str,
    passes: u32,
    skips: u32,
    incs: u32,
    fails: u32,
) -> AveragedGroup {
    let mut row = make_row(scenario, topology, true, 0.0);
    row.work_type = work_type.into();
    AveragedGroup {
        row,
        passes_observed: passes,
        skips_observed: skips,
        inconclusives_observed: incs,
        failures_observed: fails,
        total_observed: passes + skips + incs + fails,
    }
}

/// Regression guard for the per-group 4-state breakdown:
/// `format_per_group_pass_counts` must append `(N skip, N inc,
/// N fail)` when any non-pass bucket is non-zero, mirroring
/// `format_dimension_summary`'s contract. A future refactor that
/// drops the breakdown suffix or stops populating the new
/// AveragedGroup counters would silently regress the operator's
/// per-group view back to the pre-fix `passes_observed /
/// total_observed` shape that hid skip / inconc / fail
/// distinctions in the M-N gap.
#[test]
fn format_per_group_pass_counts_renders_skip_inc_fail_breakdown() {
    let g_a = group_with_breakdown("scn", "topo", "wt", 1, 1, 1, 1);
    let g_b = group_with_breakdown("scn", "topo", "wt", 4, 0, 0, 0);
    let out = format_per_group_pass_counts(&[g_a], &[g_b], "A", "B");
    assert!(
        out.contains("A=1/4 (1 skip, 1 inc, 1 fail)"),
        "non-zero buckets must surface in the per-group breakdown for A: {out}"
    );
    assert!(
        out.contains("B=4/4"),
        "all-pass side must stay terse (no breakdown suffix): {out}"
    );
    assert!(
        !out.contains("B=4/4 ("),
        "B has zero non-pass buckets — must NOT append empty breakdown: {out}"
    );
}

/// Empty input: no groups on either side. The formatter
/// returns an empty string so the caller can suppress the
/// block entirely (no header, no body, no separator).
#[test]
fn format_per_group_pass_counts_empty_returns_empty_string() {
    let out = format_per_group_pass_counts(&[], &[], "a", "b");
    assert!(
        out.is_empty(),
        "empty input must yield empty output, got: {out:?}",
    );
}

/// Both-sides-present: every (scenario, topology, work_type)
/// group renders one line. Healthy 5/5 groups appear
/// alongside unhealthy 3/5 groups — the spec is "show every
/// group", not "show only the broken ones".
#[test]
fn format_per_group_pass_counts_renders_every_group_with_n_over_m() {
    let avg_a = vec![
        group("alpha", "tiny-1llc", "SpinWait", 5, 5),
        group("beta", "tiny-1llc", "SpinWait", 3, 5),
    ];
    let avg_b = vec![
        group("alpha", "tiny-1llc", "SpinWait", 4, 5),
        group("beta", "tiny-1llc", "SpinWait", 5, 5),
    ];
    let out = format_per_group_pass_counts(&avg_a, &avg_b, "a", "b");
    // Header line present.
    assert!(
        out.contains("per-group pass counts"),
        "header line must appear, got: {out:?}",
    );
    // Both groups render with their per-side N/M counters.
    assert!(
        out.contains("alpha/tiny-1llc/SpinWait: a=5/5 b=4/5"),
        "alpha group line missing; got: {out:?}",
    );
    assert!(
        out.contains("beta/tiny-1llc/SpinWait: a=3/5 b=5/5"),
        "beta group line missing; got: {out:?}",
    );
    // Trailing newline so the next section reads cleanly.
    assert!(
        out.ends_with('\n'),
        "block must end with newline, got: {out:?}",
    );
}

/// One-side-only group renders `-` for the missing side.
/// Pins the asymmetric-key path: a B-side row that has no
/// A-side match gets `a=-`; symmetric for A-only / B-side.
/// The block surfaces the asymmetry by name so the operator
/// doesn't have to cross-reference the summary's `new_in_b`
/// / `removed_from_a` counters to know which groups went
/// missing.
#[test]
fn format_per_group_pass_counts_one_side_missing_renders_dash() {
    let avg_a = vec![group("only_a", "tiny-1llc", "SpinWait", 5, 5)];
    let avg_b = vec![group("only_b", "tiny-1llc", "SpinWait", 3, 5)];
    let out = format_per_group_pass_counts(&avg_a, &avg_b, "a", "b");
    assert!(
        out.contains("only_a/tiny-1llc/SpinWait: a=5/5 b=-"),
        "A-only group must render b=-; got: {out:?}",
    );
    assert!(
        out.contains("only_b/tiny-1llc/SpinWait: a=- b=3/5"),
        "B-only group must render a=-; got: {out:?}",
    );
}

// -- ComparePartition::as_str --

/// `ComparePartition::as_str` maps each variant to the one-letter
/// label the scalar table headers use, so the noise per-phase coverage rows and
/// the scalar findings table share the same operator-facing side
/// identifier. Both arms pinned: a future swap (`A => "B"`) would
/// silently mislabel every noise per-phase coverage row's SIDE column.
#[test]
fn compare_partition_as_str_maps_each_variant_to_its_letter() {
    assert_eq!(ComparePartition::A.as_str(), "A");
    assert_eq!(ComparePartition::B.as_str(), "B");
}
