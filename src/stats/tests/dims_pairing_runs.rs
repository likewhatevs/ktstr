use super::*;

// -- Dimension / derive_slicing_dims / pairing dims --

/// `Dimension::ALL` lists all nine dims in canonical order.
/// Order matters for [`PairingKey::from_row`] and for header
/// rendering — a regression that reordered the slice would
/// silently shift every dynamic key, splitting previously-
/// paired rows. Pin the literal order.
#[test]
fn dimension_all_canonical_order() {
    assert_eq!(
        Dimension::ALL,
        &[
            Dimension::Kernel,
            Dimension::Scheduler,
            Dimension::Topology,
            Dimension::WorkType,
            Dimension::ProjectCommit,
            Dimension::KernelCommit,
            Dimension::RunSource,
            Dimension::ResolveSource,
            Dimension::CpuBudget,
        ],
    );
}

/// `Dimension::pairing_dims` returns every dim NOT in the
/// slicing set, preserving canonical order. Two slicing
/// orderings produce the same pairing-dim list (the function
/// iterates `ALL`, not `slicing`).
#[test]
fn dimension_pairing_dims_complements_slicing() {
    let pair = Dimension::pairing_dims(&[Dimension::Kernel, Dimension::ProjectCommit]);
    assert_eq!(
        pair,
        vec![
            Dimension::Scheduler,
            Dimension::Topology,
            Dimension::WorkType,
            Dimension::KernelCommit,
            Dimension::RunSource,
            Dimension::ResolveSource,
            Dimension::CpuBudget,
        ],
    );
    // Order of slicing input doesn't change the output —
    // the function iterates ALL and filters.
    let pair_reversed = Dimension::pairing_dims(&[Dimension::ProjectCommit, Dimension::Kernel]);
    assert_eq!(pair, pair_reversed);
}

/// Empty slicing set → every dim is a pairing dim.
#[test]
fn dimension_pairing_dims_empty_slicing_yields_all() {
    let pair = Dimension::pairing_dims(&[]);
    assert_eq!(pair, Dimension::ALL.to_vec());
}

/// `derive_slicing_dims` returns every dimension on which
/// filter_a and filter_b differ. Equal filters → empty
/// slicing.
#[test]
fn derive_slicing_dims_identical_filters_yields_empty() {
    let f = RowFilter {
        schedulers: vec!["scx_alpha".to_string()],
        ..RowFilter::default()
    };
    assert!(derive_slicing_dims(&f, &f).is_empty());
}

/// One-dim diff on a SLICEABLE (version) axis: only that dimension is reported.
#[test]
fn derive_slicing_dims_single_dim_diff() {
    let f_a = RowFilter {
        kernels: vec!["6.14".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        kernels: vec!["6.15".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(derive_slicing_dims(&f_a, &f_b), vec![Dimension::Kernel]);
}

/// Vec dims (kernels/commits) compare as sorted-deduped sets —
/// order and duplicates inside the filter don't shift the
/// slicing-dim derivation.
#[test]
fn derive_slicing_dims_vec_compares_as_set() {
    let f_a = RowFilter {
        kernels: vec!["6.14".to_string(), "6.15".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        kernels: vec!["6.15".to_string(), "6.14".to_string(), "6.14".to_string()],
        ..RowFilter::default()
    };
    assert!(
        derive_slicing_dims(&f_a, &f_b).is_empty(),
        "same set in different order/multiplicity must NOT slice",
    );
}

/// Multi-dim diff across SLICEABLE axes: every differing sliceable
/// dimension is reported, in canonical [`Dimension::ALL`] order. A
/// differing NON-sliceable dim (here `schedulers`) is NOT reported — it
/// is filter + pairing only (see [`Dimension::SLICEABLE`]).
#[test]
fn derive_slicing_dims_multi_dim_diff_in_canonical_order() {
    let f_a = RowFilter {
        kernels: vec!["6.14".to_string()],
        project_commits: vec!["aaaaaaa".to_string()],
        schedulers: vec!["scx_alpha".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        kernels: vec!["6.15".to_string()],
        project_commits: vec!["bbbbbbb".to_string()],
        schedulers: vec!["scx_beta".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        derive_slicing_dims(&f_a, &f_b),
        vec![Dimension::Kernel, Dimension::ProjectCommit],
        "sliceable dims slice in canonical order; the differing scheduler \
         (non-sliceable) is filter+pairing only and never slices",
    );
}

/// Run-source is FILTER + PAIRING only — NOT a sliceable axis (see
/// [`Dimension::SLICEABLE`]). A `run_sources` difference must NOT form an A/B
/// contrast: only the version axes (kernel / project-commit / kernel-commit)
/// slice; contrasting across run-source would bulk-compare heterogeneous runs
/// the significance math cannot soundly attribute. It still narrows the cohort
/// and joins A to B as a pairing dim.
#[test]
fn derive_slicing_dims_run_source_is_filter_pairing_only() {
    let f_a = RowFilter {
        run_sources: vec!["local".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        run_sources: vec!["ci".to_string()],
        ..RowFilter::default()
    };
    assert!(
        derive_slicing_dims(&f_a, &f_b).is_empty(),
        "run_source is not sliceable — a difference must not slice",
    );
}

/// Resolve-source is FILTER + PAIRING only — NOT sliceable (see
/// [`Dimension::SLICEABLE`]). A `resolve_sources` difference must NOT slice;
/// mirror of `derive_slicing_dims_run_source_is_filter_pairing_only`.
#[test]
fn derive_slicing_dims_resolve_source_is_filter_pairing_only() {
    let f_a = RowFilter {
        resolve_sources: vec!["auto_built".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        resolve_sources: vec!["target_debug".to_string()],
        ..RowFilter::default()
    };
    assert!(
        derive_slicing_dims(&f_a, &f_b).is_empty(),
        "resolve_source is not sliceable — a difference must not slice",
    );
}

/// Topology is FILTER + PAIRING only — NOT sliceable (see
/// [`Dimension::SLICEABLE`]). A `topologies` difference must NOT slice:
/// contrasting across topology compares physically different machines, which
/// the significance math cannot attribute to a code change. It joins A to B as
/// a pairing dim.
#[test]
fn derive_slicing_dims_topology_is_filter_pairing_only() {
    let f_a = RowFilter {
        topologies: vec!["1n2l4c1t".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        topologies: vec!["1n2l4c2t".to_string()],
        ..RowFilter::default()
    };
    assert!(
        derive_slicing_dims(&f_a, &f_b).is_empty(),
        "topology is not sliceable — a difference must not slice",
    );
}

/// WorkType is FILTER + PAIRING only — NOT sliceable (see
/// [`Dimension::SLICEABLE`]). A `work_types` difference must NOT slice; mirror
/// of `derive_slicing_dims_topology_is_filter_pairing_only`.
#[test]
fn derive_slicing_dims_work_type_is_filter_pairing_only() {
    let f_a = RowFilter {
        work_types: vec!["SpinWait".to_string()],
        ..RowFilter::default()
    };
    let f_b = RowFilter {
        work_types: vec!["PageFaultChurn".to_string()],
        ..RowFilter::default()
    };
    assert!(
        derive_slicing_dims(&f_a, &f_b).is_empty(),
        "work_type is not sliceable — a difference must not slice",
    );
}

/// `kernel_filter_matches`: major.minor (`6.12`) prefix
/// matches every patch in the series via the
/// `starts_with("6.12.")` arm, and ALSO matches `6.12`
/// exactly. Three-segment-or-longer filters are strict.
#[test]
fn kernel_filter_matches_major_minor_prefix() {
    // Two-segment filter: prefix matches.
    assert!(kernel_filter_matches("6.12", "6.12"));
    assert!(kernel_filter_matches("6.12", "6.12.0"));
    assert!(kernel_filter_matches("6.12", "6.12.5"));
    assert!(!kernel_filter_matches("6.12", "6.13.0"));
    // Critically: `6.1` must not match `6.10.0` — the
    // trailing-dot in the prefix path prevents the
    // accidental wildcard.
    assert!(!kernel_filter_matches("6.1", "6.10.0"));
}

/// `kernel_filter_matches`: major.minor prefix admits the
/// `MAJOR.MINOR-rcN` pre-release shape via the
/// `starts_with("MAJOR.MINOR-")` arm. The rcN kernel shares
/// the `(major, minor, patch=0)` tuple with the eventual
/// release per `kernel_path::decompose_version_for_compare`,
/// and the operator filtering on `6.14` wants the whole
/// series — release AND pre-releases. This complements the
/// `6.14.0-rc3` (kernel-banner) shape which already matched
/// via the trailing-dot prefix.
#[test]
fn kernel_filter_matches_major_minor_admits_rc_pre_release() {
    // No-patch pre-release shape (kernel_path KernelId::Version
    // doc cites `6.15-rc3` as a valid version string).
    assert!(kernel_filter_matches("6.14", "6.14-rc3"));
    assert!(kernel_filter_matches("6.14", "6.14-rc1"));
    // Patch+rc shape (kernel banner from a kernel.org
    // `v6.14-rc3` tag is `Linux version 6.14.0-rc3+`).
    assert!(kernel_filter_matches("6.14", "6.14.0-rc3"));
    assert!(kernel_filter_matches("6.14", "6.14.0-rc3+"));
    // The dash-prefix arm must NOT wildcard across series:
    // `6.1` filtering must reject `6.14-rc3` for the same
    // reason `6.1` rejects `6.10.0`.
    assert!(!kernel_filter_matches("6.1", "6.14-rc3"));
    // Cross-minor rc rejection.
    assert!(!kernel_filter_matches("6.14", "6.15-rc3"));
}

/// `kernel_filter_matches`: three-segment+ filters are strict
/// equality.
#[test]
fn kernel_filter_matches_strict_for_three_plus_segments() {
    assert!(kernel_filter_matches("6.14.2", "6.14.2"));
    // Critically: `6.14.2` must NOT match `6.14.20` — the
    // strict-equality arm prevents the patch-level prefix
    // wildcarding.
    assert!(!kernel_filter_matches("6.14.2", "6.14.20"));
    assert!(!kernel_filter_matches("6.14.2", "6.14.21"));
    // RC suffixes are also strict.
    assert!(kernel_filter_matches("6.15-rc3", "6.15-rc3"));
    assert!(!kernel_filter_matches("6.15-rc3", "6.15-rc30"));
}

/// `RowFilter::matches` with a major.minor `--kernel` filter
/// admits the row whose `kernel_version` is a patch in that
/// series.
#[test]
fn row_filter_kernel_major_minor_prefix_admits_patch_version() {
    let row = make_filter_row("t", "scx_a", "1n2l4c1t", "SpinWait", Some("6.12.5"));
    let filter = RowFilter {
        kernels: vec!["6.12".to_string()],
        ..RowFilter::default()
    };
    assert!(
        filter.matches(&row),
        "major.minor filter `6.12` must admit row with kernel_version `6.12.5`",
    );
}

// -- PairingKey --

/// `PairingKey::from_row` always puts `scenario` first, then
/// the requested dims in canonical order. Two rows with the
/// same scenario+dims agree; one with a different topology
/// (when topology IS a pairing dim) does not.
#[test]
fn pairing_key_from_row_basic() {
    let row_a = make_filter_row("scenA", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    let row_b = make_filter_row("scenA", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    let row_c = make_filter_row("scenA", "scx_a", "2n2l", "SpinWait", Some("6.14"));
    let dims = &[Dimension::Topology, Dimension::WorkType];
    assert_eq!(
        PairingKey::from_row(&row_a, dims),
        PairingKey::from_row(&row_b, dims),
    );
    assert_ne!(
        PairingKey::from_row(&row_a, dims),
        PairingKey::from_row(&row_c, dims),
        "different topology must distinguish the keys when topology is a pairing dim",
    );
}

/// Slicing on topology means topology is NOT in the pairing
/// dim set — so two rows that differ ONLY on topology pair
/// to the same key, allowing the comparison to contrast
/// them across A/B sides.
#[test]
fn pairing_key_excludes_slicing_dim() {
    let row_a = make_filter_row("scenA", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    let row_b = make_filter_row("scenA", "scx_a", "2n2l", "SpinWait", Some("6.14"));
    // Pairing dims = ALL minus Topology. So these two rows
    // pair iff they agree on everything BUT topology.
    let pair_dims = Dimension::pairing_dims(&[Dimension::Topology]);
    assert_eq!(
        PairingKey::from_row(&row_a, &pair_dims),
        PairingKey::from_row(&row_b, &pair_dims),
        "rows differing only on a slicing dim must produce equal pairing keys",
    );
}

/// `PairingKey::from_row` first slot is always scenario;
/// rendering via `parts.join("/")` reproduces the
/// `scenario/topology/work_type` shape when those dims are
/// pairing dims.
#[test]
fn pairing_key_join_renders_legacy_shape() {
    let row = make_filter_row("test_a", "scx_a", "1n2l", "SpinWait", Some("6.14"));
    let key = PairingKey::from_row(&row, LEGACY_PAIRING_DIMS);
    assert_eq!(
        key.0.join("/"),
        "test_a/1n2l/SpinWait",
        "legacy-shape join must render the three-segment label",
    );
}

/// `PairingKey::from_row` includes the row's `kernel_commit`
/// when `KernelCommit` is in the pairing-dim list, and
/// excludes it when `KernelCommit` is the slicing dim. Pins
/// the [`Dimension::KernelCommit`] arm of the from_row match
/// — a regression that omitted the arm or substituted the
/// wrong row field would surface here as either a missing
/// key slot or a slot carrying the wrong value.
///
/// `None` kernel_commit renders as the empty string slot per
/// the `unwrap_or_default()` policy on Option dims; that
/// shape is shared across every Option-typed dim arm.
#[test]
fn pairing_key_from_row_includes_kernel_commit_when_pairing() {
    let mut row_some = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_some.kernel_commit = Some("kabcde7".to_string());
    let mut row_none = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_none.kernel_commit = None;

    // KernelCommit in pairing dims → key carries the commit
    // value (or the empty slot for None). The two rows
    // therefore produce DIFFERENT keys because their
    // kernel_commit values disagree.
    let pair_dims = &[Dimension::KernelCommit];
    let key_some = PairingKey::from_row(&row_some, pair_dims);
    let key_none = PairingKey::from_row(&row_none, pair_dims);
    assert_eq!(
        key_some.0,
        vec!["scn".to_string(), "kabcde7".to_string()],
        "Some(kernel_commit) must occupy the second slot verbatim",
    );
    assert_eq!(
        key_none.0,
        vec!["scn".to_string(), String::new()],
        "None kernel_commit must collapse to an empty slot per \
             unwrap_or_default policy",
    );
    assert_ne!(
        key_some, key_none,
        "two rows differing on kernel_commit must produce \
             distinct pairing keys when KernelCommit is a pairing dim",
    );

    // KernelCommit excluded (slicing) → the two rows pair to
    // the same key because the dim is dropped. Pins the
    // dimensional-slicing semantic for the new arm.
    let slice_dims = Dimension::pairing_dims(&[Dimension::KernelCommit]);
    assert_eq!(
        PairingKey::from_row(&row_some, &slice_dims),
        PairingKey::from_row(&row_none, &slice_dims),
        "rows differing only on the slicing dim (KernelCommit) \
             must produce equal pairing keys",
    );
}

/// `PairingKey::from_row` includes the row's `run_source`
/// when `RunSource` is in the pairing-dim list, and excludes it
/// when `RunSource` is the slicing dim. Pins the
/// [`Dimension::RunSource`] arm of the from_row match — same
/// shape and motivation as
/// `pairing_key_from_row_includes_kernel_commit_when_pairing`
/// but for the run_source arm. A regression that omitted the
/// arm or substituted `row.kernel_commit` for
/// `row.run_source` would surface here.
#[test]
fn pairing_key_from_row_includes_run_source_when_pairing() {
    let mut row_local = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_local.run_source = Some("local".to_string());
    let mut row_ci = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_ci.run_source = Some("ci".to_string());
    let mut row_none = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_none.run_source = None;

    let pair_dims = &[Dimension::RunSource];
    let key_local = PairingKey::from_row(&row_local, pair_dims);
    let key_ci = PairingKey::from_row(&row_ci, pair_dims);
    let key_none = PairingKey::from_row(&row_none, pair_dims);
    assert_eq!(
        key_local.0,
        vec!["scn".to_string(), "local".to_string()],
        "Some(run_source) must occupy the second slot verbatim",
    );
    assert_eq!(key_ci.0, vec!["scn".to_string(), "ci".to_string()]);
    assert_eq!(
        key_none.0,
        vec!["scn".to_string(), String::new()],
        "None run_source must collapse to an empty slot per \
             unwrap_or_default policy",
    );
    assert_ne!(
        key_local, key_ci,
        "two rows differing on run_source must produce \
             distinct pairing keys when Source is a pairing dim",
    );

    // Source excluded (slicing) → the differing-run_source
    // rows pair to the same key.
    let slice_dims = Dimension::pairing_dims(&[Dimension::RunSource]);
    assert_eq!(
        PairingKey::from_row(&row_local, &slice_dims),
        PairingKey::from_row(&row_ci, &slice_dims),
        "rows differing only on the slicing dim (Source) must \
             produce equal pairing keys",
    );
}

/// `PairingKey::from_row` includes the row's `resolve_source` when
/// ResolveSource is a pairing dim, and excludes it when it is the
/// slicing dim. Mirror of
/// `pairing_key_from_row_includes_run_source_when_pairing` — a
/// regression substituting `row.run_source` for `row.resolve_source`
/// in the from_row arm would surface here.
#[test]
fn pairing_key_from_row_includes_resolve_source_when_pairing() {
    let mut row_auto = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_auto.resolve_source = Some("auto_built".to_string());
    let mut row_debug = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_debug.resolve_source = Some("target_debug".to_string());
    let mut row_none = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_none.resolve_source = None;

    let pair_dims = &[Dimension::ResolveSource];
    let key_auto = PairingKey::from_row(&row_auto, pair_dims);
    let key_debug = PairingKey::from_row(&row_debug, pair_dims);
    let key_none = PairingKey::from_row(&row_none, pair_dims);
    assert_eq!(
        key_auto.0,
        vec!["scn".to_string(), "auto_built".to_string()],
        "Some(resolve_source) must occupy the second slot verbatim",
    );
    assert_eq!(
        key_debug.0,
        vec!["scn".to_string(), "target_debug".to_string()]
    );
    assert_eq!(
        key_none.0,
        vec!["scn".to_string(), String::new()],
        "None resolve_source must collapse to an empty slot",
    );
    assert_ne!(
        key_auto, key_debug,
        "two rows differing on resolve_source must produce distinct \
             pairing keys when ResolveSource is a pairing dim",
    );
    let slice_dims = Dimension::pairing_dims(&[Dimension::ResolveSource]);
    assert_eq!(
        PairingKey::from_row(&row_auto, &slice_dims),
        PairingKey::from_row(&row_debug, &slice_dims),
        "rows differing only on the slicing dim (ResolveSource) must \
             produce equal pairing keys",
    );
}

/// `PairingKey::from_row` includes the row's cpu_budget when CpuBudget
/// is a pairing dim — so cross-budget rows NEVER pair — and excludes
/// it when CpuBudget is the slicing dim. This is deliberate: a
/// 4-CPU-budget run and a 32-CPU-budget run measure different
/// things and must not be silently compared.
#[test]
fn pairing_key_from_row_includes_cpu_budget_when_pairing() {
    let mut row_4 = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_4.cpu_budget = Some(4);
    let mut row_32 = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_32.cpu_budget = Some(32);
    let mut row_none = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_none.cpu_budget = None;

    let pair_dims = &[Dimension::CpuBudget];
    let key_4 = PairingKey::from_row(&row_4, pair_dims);
    let key_32 = PairingKey::from_row(&row_32, pair_dims);
    let key_none = PairingKey::from_row(&row_none, pair_dims);
    assert_eq!(key_4.0, vec!["scn".to_string(), "4".to_string()]);
    assert_eq!(key_32.0, vec!["scn".to_string(), "32".to_string()]);
    assert_eq!(
        key_none.0,
        vec!["scn".to_string(), String::new()],
        "None cpu_budget (a skip) collapses to an empty slot",
    );
    assert_ne!(
        key_4, key_32,
        "rows of different cpu_budget must NOT pair when CpuBudget pairs",
    );
    assert_ne!(
        key_4, key_none,
        "a budgeted row must not pair with a skip (None budget)",
    );

    // CpuBudget sliced → the dim is dropped → the two budgets pair.
    let slice_dims = Dimension::pairing_dims(&[Dimension::CpuBudget]);
    assert_eq!(
        PairingKey::from_row(&row_4, &slice_dims),
        PairingKey::from_row(&row_32, &slice_dims),
        "rows differing only on the sliced dim (CpuBudget) must pair",
    );
}

/// Clean and dirty contributors at the same canonical hex
/// must land in the same pairing bucket. Without the
/// `-dirty` strip in `commit_pairing_key_part`, `abc1234`
/// and `abc1234-dirty` shatter into separate groups,
/// defeating `group_and_average_by`'s `+mixed` cohort
/// detection (which can only fire when the two contributors
/// land in ONE group).
#[test]
fn pairing_key_from_row_strips_dirty_suffix_on_commit() {
    let mut row_clean = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_clean.commit = Some("abc1234".to_string());
    let mut row_dirty = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_dirty.commit = Some("abc1234-dirty".to_string());

    let pair_dims = &[Dimension::ProjectCommit];
    let key_clean = PairingKey::from_row(&row_clean, pair_dims);
    let key_dirty = PairingKey::from_row(&row_dirty, pair_dims);

    assert_eq!(
        key_clean, key_dirty,
        "clean `abc1234` and dirty `abc1234-dirty` must produce \
             EQUAL pairing keys so the +mixed cohort machinery in \
             group_and_average_by can surface their disagreement",
    );
    assert_eq!(
        key_clean.0,
        vec!["scn".to_string(), "abc1234".to_string()],
        "key part must be the canonical un-suffixed hex",
    );
}

/// Same shape on the kernel_commit dimension. Pins the
/// second commit dim's strip independently because
/// `from_row` uses two parallel arms; a regression could
/// strip one but not the other.
#[test]
fn pairing_key_from_row_strips_dirty_suffix_on_kernel_commit() {
    let mut row_clean = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_clean.kernel_commit = Some("def5678".to_string());
    let mut row_dirty = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_dirty.kernel_commit = Some("def5678-dirty".to_string());

    let pair_dims = &[Dimension::KernelCommit];
    let key_clean = PairingKey::from_row(&row_clean, pair_dims);
    let key_dirty = PairingKey::from_row(&row_dirty, pair_dims);

    assert_eq!(
        key_clean, key_dirty,
        "clean and dirty kernel_commit at the same canonical \
             hex must pair together",
    );
    assert_eq!(key_clean.0, vec!["scn".to_string(), "def5678".to_string()],);
}

/// Distinct hexes still differentiate even when one carries
/// `-dirty`. Pins that the strip operates ONLY on the
/// suffix, not on the entire value — `aaa1111-dirty` and
/// `bbb2222` remain distinct.
#[test]
fn pairing_key_from_row_distinct_hexes_remain_distinct_under_strip() {
    let mut row_a = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_a.commit = Some("aaa1111-dirty".to_string());
    let mut row_b = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row_b.commit = Some("bbb2222".to_string());

    let pair_dims = &[Dimension::ProjectCommit];
    let key_a = PairingKey::from_row(&row_a, pair_dims);
    let key_b = PairingKey::from_row(&row_b, pair_dims);

    assert_ne!(
        key_a, key_b,
        "distinct canonical hexes must remain distinct after the \
             -dirty strip — only the suffix is stripped",
    );
    assert_eq!(key_a.0[1], "aaa1111");
    assert_eq!(key_b.0[1], "bbb2222");
}

/// `None` commit values still collapse to the empty slot
/// (the strip is a no-op on `None`). Pins the absence path
/// against a regression that special-cased the strip and
/// inadvertently changed the unwrap_or_default behavior.
#[test]
fn pairing_key_from_row_none_commit_unchanged_under_strip() {
    let mut row = make_filter_row("scn", "scx_a", "1n1l", "SpinWait", Some("6.14"));
    row.commit = None;
    row.kernel_commit = None;
    let pair_dims = &[Dimension::ProjectCommit, Dimension::KernelCommit];
    let key = PairingKey::from_row(&row, pair_dims);
    assert_eq!(
        key.0,
        vec!["scn".to_string(), String::new(), String::new()],
        "None commit and None kernel_commit must collapse to empty slots",
    );
}

// -- render_side_label --

/// Empty slicing dims → the bare label is returned.
#[test]
fn render_side_label_empty_dims_yields_bare() {
    let f = RowFilter::default();
    assert_eq!(render_side_label(&f, &[], "A"), "A");
}

/// Single-dim single-value scheduler renders the value
/// verbatim. After the Vec promotion of `--scheduler` the
/// scheduler arm goes through `render_vec_dim` like every
/// other Vec dim; a single entry still surfaces the bare
/// string.
#[test]
fn render_side_label_single_value_dim() {
    let f = RowFilter {
        schedulers: vec!["scx_rusty".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f, &[Dimension::Scheduler], "A"),
        "scx_rusty",
    );
}

/// Vec dim with ≤3 entries joins with `|` (sorted).
#[test]
fn render_side_label_vec_dim_short_joins_with_pipe() {
    let f = RowFilter {
        kernels: vec!["6.15".to_string(), "6.14".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f, &[Dimension::Kernel], "A"),
        "6.14|6.15",
        "≤3 values must join sorted with `|`",
    );
}

/// Vec dim with >3 entries collapses to the bare label.
#[test]
fn render_side_label_vec_dim_long_collapses_to_bare() {
    let f = RowFilter {
        kernels: vec![
            "6.10".to_string(),
            "6.11".to_string(),
            "6.12".to_string(),
            "6.13".to_string(),
            "6.14".to_string(),
        ],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f, &[Dimension::Kernel], "A"),
        "A",
        ">3 values must collapse to the bare letter so the \
             column header stays readable",
    );
}

/// Multi-dim slicing joins per-dim parts with `:`.
#[test]
fn render_side_label_multi_dim_joins_with_colon() {
    let f = RowFilter {
        kernels: vec!["6.14".to_string()],
        schedulers: vec!["scx_rusty".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f, &[Dimension::Kernel, Dimension::Scheduler], "A"),
        "6.14:scx_rusty",
    );
}

/// Empty per-side filter on a slicing dim falls back to the
/// bare label (the slice exists because the OTHER side
/// populated the dim).
#[test]
fn render_side_label_empty_dim_value_uses_bare() {
    let f = RowFilter::default();
    assert_eq!(
        render_side_label(&f, &[Dimension::Kernel], "B"),
        "B",
        "empty Vec dim must fall back to the bare letter",
    );
    assert_eq!(
        render_side_label(&f, &[Dimension::Scheduler], "B"),
        "B",
        "None Option dim must fall back to the bare letter",
    );
}

/// `Dimension::KernelCommit` arm of [`render_side_label`] reads
/// `filter.kernel_commits` (a Vec) and routes through the same
/// `render_vec_dim` path as `Kernel` / `ProjectCommit`. Pins
/// the arm so a regression that omitted it (or substituted the
/// wrong field, e.g. `filter.project_commits`) surfaces here
/// instead of silently rendering the bare label even when the
/// filter is populated.
///
/// Single-value: emits the value verbatim. Two-value: joins
/// sorted with `|` per `render_vec_dim`'s ≤3 rule. >3 values:
/// collapse to bare. Empty Vec: bare. Same shape as the
/// `Kernel` arm pinned above; a regression in the
/// `KernelCommit` arm specifically would NOT be caught by the
/// existing `render_side_label_vec_dim_*` tests because those
/// only exercise the `Kernel` field.
#[test]
fn render_side_label_kernel_commit_arm_renders_filter_value() {
    let f_one = RowFilter {
        kernel_commits: vec!["kabcde7".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_one, &[Dimension::KernelCommit], "A"),
        "kabcde7",
        "single kernel_commit value must render verbatim — \
             a regression that read `filter.project_commits` instead of \
             `filter.kernel_commits` would render `A` here because \
             the project-commit field is empty",
    );

    let f_two = RowFilter {
        kernel_commits: vec!["kbbb222".to_string(), "kaaa111".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_two, &[Dimension::KernelCommit], "A"),
        "kaaa111|kbbb222",
        "≤3 kernel_commit values must join sorted with `|`",
    );

    let f_long = RowFilter {
        kernel_commits: vec![
            "k111".to_string(),
            "k222".to_string(),
            "k333".to_string(),
            "k444".to_string(),
        ],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_long, &[Dimension::KernelCommit], "A"),
        "A",
        ">3 kernel_commit values must collapse to the bare letter",
    );

    let f_empty = RowFilter::default();
    assert_eq!(
        render_side_label(&f_empty, &[Dimension::KernelCommit], "B"),
        "B",
        "empty kernel_commits Vec must fall back to the bare letter",
    );
}

/// `Dimension::RunSource` arm of [`render_side_label`] reads
/// `filter.run_sources` (a Vec) and routes through the same
/// `render_vec_dim` path as the other Vec dims. Mirror of
/// `render_side_label_kernel_commit_arm_renders_filter_value`
/// for the Source arm. A regression that omitted the Source
/// arm or substituted the wrong field would surface here
/// instead of silently rendering the bare label even when
/// the filter is populated.
#[test]
fn render_side_label_source_arm_renders_filter_value() {
    let f_one = RowFilter {
        run_sources: vec!["local".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_one, &[Dimension::RunSource], "A"),
        "local",
        "single run_source value must render verbatim — a \
             regression that read another field would render `A` here",
    );

    let f_two = RowFilter {
        run_sources: vec!["local".to_string(), "ci".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_two, &[Dimension::RunSource], "A"),
        "ci|local",
        "≤3 run_source values must join sorted with `|`",
    );

    let f_long = RowFilter {
        run_sources: vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_long, &[Dimension::RunSource], "A"),
        "A",
        ">3 run_source values must collapse to the bare letter",
    );

    let f_empty = RowFilter::default();
    assert_eq!(
        render_side_label(&f_empty, &[Dimension::RunSource], "B"),
        "B",
        "empty run_sources Vec must fall back to the bare letter",
    );
}

/// `render_side_label` for the ResolveSource arm renders
/// `filter.resolve_sources` via render_vec_dim. Mirror of
/// `render_side_label_source_arm_renders_filter_value` — a regression
/// substituting another field would surface here instead of silently
/// rendering the bare label even when the filter is populated.
#[test]
fn render_side_label_resolve_source_arm_renders_filter_value() {
    let f_one = RowFilter {
        resolve_sources: vec!["auto_built".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_one, &[Dimension::ResolveSource], "A"),
        "auto_built",
        "single resolve_source value must render verbatim",
    );
    let f_two = RowFilter {
        resolve_sources: vec!["target_debug".to_string(), "auto_built".to_string()],
        ..RowFilter::default()
    };
    assert_eq!(
        render_side_label(&f_two, &[Dimension::ResolveSource], "A"),
        "auto_built|target_debug",
        "≤3 resolve_source values must join sorted with `|`",
    );
    let f_empty = RowFilter::default();
    assert_eq!(
        render_side_label(&f_empty, &[Dimension::ResolveSource], "B"),
        "B",
        "empty resolve_sources Vec must fall back to the bare letter",
    );
}

/// `zero_match_diagnostic` flags a `--run-source` value that is
/// not present in the pool, naming the unknown value AND the
/// distinct values actually seen. Guards against the
/// typo-class miss (e.g. `--run-source loca` for `local`,
/// `--run-source CI` for `ci`) that produces a silent
/// zero-match in `compare_partitions`.
#[test]
fn zero_match_diagnostic_unknown_run_source_lists_present_values() {
    let mut row_local = make_row("scn", "1n1l1c1t", true, 1.0);
    row_local.run_source = Some("local".to_string());
    let mut row_ci = make_row("scn", "1n1l1c1t", true, 1.0);
    row_ci.run_source = Some("ci".to_string());
    let rows = vec![row_local, row_ci];
    let filter = RowFilter {
        run_sources: vec!["loca".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        msg.contains("--run-source `loca` not found"),
        "must name the unknown value verbatim; got:\n{msg}",
    );
    assert!(
        msg.contains("`ci`") && msg.contains("`local`"),
        "must list distinct values present in the pool so the \
             operator can correct the typo; got:\n{msg}",
    );
    assert!(
        msg.contains("case-sensitive"),
        "must mention case sensitivity (`ci` ≠ `CI`); got:\n{msg}",
    );
}

/// resolve_source equivalent of
/// `zero_match_diagnostic_unknown_run_source_lists_present_values`:
/// a `--resolve-source` typo names the unknown value + lists the
/// discovery-path tags actually present.
#[test]
fn zero_match_diagnostic_unknown_resolve_source_lists_present_values() {
    let mut row_auto = make_row("scn", "1n1l1c1t", true, 1.0);
    row_auto.resolve_source = Some("auto_built".to_string());
    let mut row_debug = make_row("scn", "1n1l1c1t", true, 1.0);
    row_debug.resolve_source = Some("target_debug".to_string());
    let rows = vec![row_auto, row_debug];
    let filter = RowFilter {
        resolve_sources: vec!["auto_bui".to_string()],
        ..Default::default()
    };
    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());
    assert!(
        msg.contains("--resolve-source `auto_bui` not found"),
        "must name the unknown value verbatim; got:\n{msg}",
    );
    assert!(
        msg.contains("`auto_built`") && msg.contains("`target_debug`"),
        "must list distinct resolve_source tags present in the pool; got:\n{msg}",
    );
    assert!(
        msg.contains("case-sensitive"),
        "must mention case sensitivity; got:\n{msg}",
    );
}

/// resolve_source equivalent of the empty-pool absence-explainer:
/// when every row has `resolve_source: None`, the hint surfaces the
/// "(none — every row has `resolve_source: null`)" form.
#[test]
fn zero_match_diagnostic_unknown_resolve_source_with_empty_pool_explains_absence() {
    let row = make_row("scn", "1n1l1c1t", true, 1.0);
    let rows = vec![row];
    let filter = RowFilter {
        resolve_sources: vec!["auto_built".to_string()],
        ..Default::default()
    };
    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());
    assert!(
        msg.contains("--resolve-source `auto_built` not found"),
        "must name the unknown value; got:\n{msg}",
    );
    assert!(
        msg.contains("none — every row has `resolve_source: null`"),
        "must explain the empty-distinct-values case; got:\n{msg}",
    );
}

/// A `--resolve-source` value that DOES match a row must NOT fire the
/// unknown hint. Mirror of the known_run_source variant.
#[test]
fn zero_match_diagnostic_known_resolve_source_does_not_fire_unknown_hint() {
    let mut row = make_row("scn", "1n1l1c1t", true, 1.0);
    row.resolve_source = Some("auto_built".to_string());
    let rows = vec![row];
    let filter = RowFilter {
        resolve_sources: vec!["auto_built".to_string()],
        ..Default::default()
    };
    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());
    assert!(
        !msg.contains("--resolve-source `auto_built` not found"),
        "a present resolve_source must not fire the unknown hint; got:\n{msg}",
    );
}

/// `zero_match_diagnostic` flags a `--cpu-budget` value not present
/// in the pool, naming the unknown value AND the distinct budgets
/// actually seen — the numeric mirror of the run_source hint. Skip
/// rows (`cpu_budget: None`) contribute no budget to the list.
#[test]
fn zero_match_diagnostic_unknown_cpu_budget_lists_present_values() {
    let mut row4 = make_row("scn", "1n1l1c1t", true, 1.0);
    row4.cpu_budget = Some(4);
    let mut row32 = make_row("scn", "1n1l1c1t", true, 1.0);
    row32.cpu_budget = Some(32);
    let skip = make_row("scn", "1n1l1c1t", true, 1.0); // cpu_budget None
    let rows = vec![row4, row32, skip];
    let filter = RowFilter {
        cpu_budgets: vec!["64".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        msg.contains("--cpu-budget `64` not found"),
        "must name the unknown budget verbatim; got:\n{msg}",
    );
    assert!(
        msg.contains("`4`") && msg.contains("`32`"),
        "must list distinct budgets present in the pool; got:\n{msg}",
    );
}

/// A `--cpu-budget` value that DOES match a row must NOT trigger
/// the unknown-budget hint (guards against the hint firing for
/// every populated `--cpu-budget` regardless of pool membership).
#[test]
fn zero_match_diagnostic_known_cpu_budget_does_not_fire_unknown_hint() {
    let mut row = make_row("scn", "1n1l1c1t", true, 1.0);
    row.cpu_budget = Some(4);
    let rows = vec![row];
    let filter = RowFilter {
        cpu_budgets: vec!["4".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        !msg.contains("--cpu-budget `4` not found"),
        "a present budget must not fire the unknown-budget hint; got:\n{msg}",
    );
}

/// When every row has `run_source: None`, the hint surfaces the
/// "(none — every row has `run_source: null`)" form rather than
/// an empty list. This is the post-`apply_archive_source_override`
/// path with a pool that pre-dates the run_source field, so
/// distinguishing "unknown value, no values present" from
/// "unknown value, here's what's there" is operator-actionable.
#[test]
fn zero_match_diagnostic_unknown_run_source_with_empty_pool_explains_absence() {
    let row = make_row("scn", "1n1l1c1t", true, 1.0);
    let rows = vec![row];
    let filter = RowFilter {
        run_sources: vec!["ci".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        msg.contains("--run-source `ci` not found"),
        "must name the unknown value; got:\n{msg}",
    );
    assert!(
        msg.contains("none — every row has `run_source: null`"),
        "must explain the empty-distinct-values case rather than \
             listing nothing; got:\n{msg}",
    );
}

/// A `--run-source` value that DOES match a row in the pool
/// must NOT trigger the unknown-value hint, even when the
/// filter still matches zero rows due to other dimension
/// mismatches (e.g. scenario filter zeroes the set first).
/// Pinning this guards against a regression where the hint
/// fires for every populated `--run-source` regardless of
/// pool membership.
#[test]
fn zero_match_diagnostic_known_run_source_does_not_fire_unknown_hint() {
    let mut row = make_row("scn", "1n1l1c1t", true, 1.0);
    row.run_source = Some("local".to_string());
    let rows = vec![row];
    let filter = RowFilter {
        run_sources: vec!["local".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        !msg.contains("--run-source") || !msg.contains("not found"),
        "must NOT fire the unknown-source hint when the value is \
             present in the pool; got:\n{msg}",
    );
}

/// `zero_match_diagnostic` fires the dirty-form hint for a
/// `--project-commit X` filter when the pool contains a
/// matching `X-dirty` row — pointing the operator at the
/// dirty form so they don't have to manually scan
/// `stats list-values`. The hint must name the original
/// value, the dirty form, and the suggested replacement
/// flag form.
#[test]
fn zero_match_diagnostic_project_commit_dirty_hint_fires() {
    let mut row = make_row("scn", "1n1l1c1t", true, 1.0);
    row.commit = Some("abcdef1-dirty".to_string());
    let rows = vec![row];
    let filter = RowFilter {
        project_commits: vec!["abcdef1".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        msg.contains("no rows match `--project-commit abcdef1`"),
        "hint must name the unmatched filter value verbatim; \
             got:\n{msg}",
    );
    assert!(
        msg.contains("`abcdef1-dirty` exists in the pool"),
        "hint must surface the dirty form found in the pool; \
             got:\n{msg}",
    );
    assert!(
        msg.contains("did you mean `--project-commit abcdef1-dirty`"),
        "hint must propose the dirty form as the corrected flag; \
             got:\n{msg}",
    );
}

/// Companion to
/// `zero_match_diagnostic_project_commit_dirty_hint_fires`
/// for the `kernel_commits` arm. Same shape: hint names the
/// unmatched value, the matching `-dirty` form found in the
/// pool, and the suggested `--kernel-commit` replacement.
/// A regression that wired the kernel_commits arm to scan
/// `row.commit` (or never wired it at all) would surface
/// here as a missing hint.
#[test]
fn zero_match_diagnostic_kernel_commit_dirty_hint_fires() {
    let mut row = make_row("scn", "1n1l1c1t", true, 1.0);
    row.kernel_commit = Some("kabcde7-dirty".to_string());
    let rows = vec![row];
    let filter = RowFilter {
        kernel_commits: vec!["kabcde7".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        msg.contains("no rows match `--kernel-commit kabcde7`"),
        "hint must name the unmatched kernel_commit value verbatim; \
             got:\n{msg}",
    );
    assert!(
        msg.contains("`kabcde7-dirty` exists in the pool"),
        "hint must surface the dirty form found in the pool; \
             got:\n{msg}",
    );
    assert!(
        msg.contains("did you mean `--kernel-commit kabcde7-dirty`"),
        "hint must propose the dirty form as the corrected flag; \
             got:\n{msg}",
    );
}

/// `zero_match_diagnostic` appends the `stats list-values`
/// redirect when the operator narrowed on a commit
/// dimension (project_commits OR kernel_commits populated)
/// — that redirect points at the per-dimension dump where
/// the commit values can be cross-referenced. Without a
/// commit-dim filter the redirect is suppressed because
/// `list-values` would dump every dimension, which is no
/// more actionable than the existing `stats list` redirect
/// at the top of the message for a kernel / scheduler /
/// topology miss.
#[test]
fn zero_match_diagnostic_list_values_redirect_when_commit_dim_populated() {
    let row = make_row("scn", "1n1l1c1t", true, 1.0);
    let rows = vec![row];
    let filter = RowFilter {
        project_commits: vec!["abcdef1".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        msg.contains("cargo ktstr stats list-values"),
        "must include the list-values redirect when commit \
             dim filter is populated; got:\n{msg}",
    );

    // Same redirect when only kernel_commits is populated.
    let filter_kc = RowFilter {
        kernel_commits: vec!["kabcde7".to_string()],
        ..Default::default()
    };
    let msg_kc = zero_match_diagnostic("A", &filter_kc, &rows, rows.len());
    assert!(
        msg_kc.contains("cargo ktstr stats list-values"),
        "list-values redirect must also fire on the \
             kernel_commits arm; got:\n{msg_kc}",
    );
}

/// Without a commit-dim filter populated, the list-values
/// redirect must NOT fire — generic kernel / scheduler /
/// topology / work-type misses already get the `stats list`
/// redirect, and a list-values dump would be noise rather
/// than signal. Pins the suppression so a regression that
/// always emitted the redirect (or omitted the touched-
/// commit-dim guard) surfaces here.
#[test]
fn zero_match_diagnostic_no_list_values_redirect_when_no_commit_dim() {
    let row = make_row("scn", "1n1l1c1t", true, 1.0);
    let rows = vec![row];
    // Filter narrowed on a non-commit dim only — the
    // redirect must stay quiet.
    let filter = RowFilter {
        schedulers: vec!["scx_alpha".to_string()],
        ..Default::default()
    };

    let msg = zero_match_diagnostic("A", &filter, &rows, rows.len());

    assert!(
        !msg.contains("cargo ktstr stats list-values"),
        "list-values redirect must NOT fire when no commit-dim \
             filter is populated; got:\n{msg}",
    );
}

// -- sorted_run_entries (testable extraction of list_runs sort logic) --

/// `sorted_run_entries` orders subdirectories under `root` by
/// directory mtime DESCENDING — newest first. Pins the contract
/// so a regression that flips the sort direction (e.g. drops
/// `Reverse`) or removes the mtime probe (reverting to
/// `file_name`-only sort) surfaces here as the order shift.
///
/// Three subdirs are created with `std::thread::sleep` between
/// `create_dir` calls so each directory's mtime is captured at
/// a strictly later instant than the previous one. 100 ms is
/// generous: ext4/btrfs/xfs nsec resolution + monotonic
/// CLOCK_REALTIME advancement guarantee distinct mtimes per
/// dir at this granularity.
///
/// The OLDEST directory is named `aaa_oldest` and the NEWEST
/// is named `zzz_newest` — paired so the lexical-ascending
/// order (aaa, mmm, zzz) is the OPPOSITE of the mtime-descending
/// order (zzz_newest, mmm_middle, aaa_oldest). Without this
/// pairing, lexical-ascending and mtime-descending would
/// produce the same output and a regression to filename-only
/// sort would not be detectable. With this pairing, any
/// regression that drops `Reverse` (mtime-ASCENDING) OR
/// reverts to filename-only sort (lexical-ASCENDING) yields
/// `aaa, mmm, zzz` — the WRONG order — and the test fails
/// loud.
#[test]
fn sorted_run_entries_orders_by_mtime_descending() {
    use std::thread::sleep;
    use std::time::Duration;

    let root = tempfile::TempDir::new().expect("tempdir");
    let oldest = root.path().join("aaa_oldest");
    let middle = root.path().join("mmm_middle");
    let newest = root.path().join("zzz_newest");
    std::fs::create_dir(&oldest).expect("mkdir oldest");
    sleep(Duration::from_millis(100));
    std::fs::create_dir(&middle).expect("mkdir middle");
    sleep(Duration::from_millis(100));
    std::fs::create_dir(&newest).expect("mkdir newest");

    let rows = super::sorted_run_entries(root.path()).expect("sorted_run_entries must succeed");
    let names: Vec<String> = rows
        .iter()
        .map(|(p, _, _, _)| {
            p.file_name()
                .expect("path must have a file_name")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "zzz_newest".to_string(),
            "mmm_middle".to_string(),
            "aaa_oldest".to_string(),
        ],
        "rows must be sorted by mtime descending: newest dir \
             (`zzz_newest`) first, oldest dir (`aaa_oldest`) last. \
             A regression that drops Reverse (mtime-ascending) or \
             reverts to filename-only sort (lexical-ascending) \
             would yield aaa, mmm, zzz — the OPPOSITE of the \
             expected mtime-descending order — and would fail this \
             assertion.",
    );
}

/// Empty root: `sorted_run_entries` returns an empty vec
/// rather than erroring. Pins the no-runs path that the
/// `list_runs` caller short-circuits with the
/// "no runs found" eprintln.
#[test]
fn sorted_run_entries_empty_root_yields_empty_vec() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let rows = super::sorted_run_entries(root.path()).expect("sorted_run_entries must succeed");
    assert!(
        rows.is_empty(),
        "empty root must yield empty vec; got {rows:?}",
    );
}

/// `sorted_run_entries` skips files (only subdirectories
/// become rows). Pins the `is_dir()` filter — a regression
/// that included file entries would surface here as a row
/// for the file.
#[test]
fn sorted_run_entries_skips_non_directory_entries() {
    let root = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir(root.path().join("a_dir")).expect("mkdir");
    std::fs::write(root.path().join("a_file"), b"not a run dir").expect("write file");

    let rows = super::sorted_run_entries(root.path()).expect("sorted_run_entries must succeed");
    let names: Vec<String> = rows
        .iter()
        .map(|(p, _, _, _)| {
            p.file_name()
                .expect("path must have a file_name")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        names,
        vec!["a_dir".to_string()],
        "only the subdirectory must be returned; file entries are skipped",
    );
}

/// `sorted_run_entries` skips dotfile-prefixed subdirectories.
/// Pins the filter that excludes the flock sentinel
/// subdirectory `.locks/` from `cargo ktstr stats list` —
/// a regression that dropped the dotfile filter would
/// surface here as a `.locks` row in the listing, polluting
/// the operator-facing run table with internal coordination
/// state. Other dotfile directories (`.git`, `.cache`, etc.)
/// are filtered uniformly by the same predicate so the test
/// uses two different dotfile names to pin the rule rather
/// than the specific `.locks` instance.
#[test]
fn sorted_run_entries_skips_dotfile_subdirectories() {
    let root = tempfile::TempDir::new().expect("tempdir");
    std::fs::create_dir(root.path().join("real-run")).expect("mkdir");
    std::fs::create_dir(root.path().join(".locks")).expect("mkdir .locks");
    std::fs::create_dir(root.path().join(".cache")).expect("mkdir .cache");

    let rows = super::sorted_run_entries(root.path()).expect("sorted_run_entries must succeed");
    let names: Vec<String> = rows
        .iter()
        .map(|(p, _, _, _)| {
            p.file_name()
                .expect("path must have a file_name")
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        names,
        vec!["real-run".to_string()],
        "dotfile-prefixed subdirs (.locks, .cache) must be filtered \
             out of the run listing; only `real-run` may surface",
    );
}

/// `sorted_run_entries` extracts the arch from the first
/// sidecar that carries `host.arch`. Pins the arch-extraction
/// contract so a regression that drops the field, scans the
/// wrong option leg, or stops short on a host-None sidecar
/// surfaces here.
#[test]
fn sorted_run_entries_extracts_arch_from_first_sidecar() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let run_dir = root.path().join("run-with-arch");
    std::fs::create_dir(&run_dir).expect("mkdir run dir");
    // First sidecar: host populated → arch surfaces.
    let mut sc = crate::test_support::SidecarResult::test_fixture();
    sc.host = Some(crate::host_context::HostContext::test_fixture());
    std::fs::write(
        run_dir.join("t-0000000000000000.ktstr.json"),
        serde_json::to_string(&sc).expect("serialize fixture"),
    )
    .expect("write sidecar");

    let rows = super::sorted_run_entries(root.path()).expect("sorted_run_entries must succeed");
    assert_eq!(rows.len(), 1, "one run dir must yield one row");
    let (_, _, _, arch) = &rows[0];
    assert_eq!(
        arch.as_deref(),
        Some("x86_64"),
        "arch must come from host.arch on the first sidecar — \
             test_fixture populates `Some(\"x86_64\")`",
    );
}

/// A run with no host-populated sidecars yields `None` for
/// arch. Pins the absent-host fallback so the caller's
/// display-sentinel substitution (the `"-"` cell in
/// `list_runs`) is reached.
#[test]
fn sorted_run_entries_arch_none_when_no_host() {
    let root = tempfile::TempDir::new().expect("tempdir");
    let run_dir = root.path().join("run-no-host");
    std::fs::create_dir(&run_dir).expect("mkdir run dir");
    // SidecarResult::test_fixture defaults host: None.
    let sc = crate::test_support::SidecarResult::test_fixture();
    std::fs::write(
        run_dir.join("t-0000000000000000.ktstr.json"),
        serde_json::to_string(&sc).expect("serialize fixture"),
    )
    .expect("write sidecar");

    let rows = super::sorted_run_entries(root.path()).expect("sorted_run_entries must succeed");
    assert_eq!(rows.len(), 1, "one run dir must yield one row");
    let (_, _, _, arch) = &rows[0];
    assert!(
        arch.is_none(),
        "no host-populated sidecar must yield None arch; got {arch:?}",
    );
}
