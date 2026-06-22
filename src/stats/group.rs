use super::*;

/// One of the eight dimensions that compose a `GauntletRow`'s
/// identity in the comparison pipeline: `kernel`, `scheduler`,
/// `topology`, `work-type`, `project-commit`, `kernel-commit`,
/// `run-source`, `cpu-budget`. Each maps to the corresponding
/// `RowFilter` field and `GauntletRow` field; the dimension
/// model lets `compare_partitions` derive its slicing dims and
/// dynamic pairing key without hardcoding the dimension list at
/// every call site. Variant names match the CLI flag suffix
/// (e.g. `Dimension::ProjectCommit` ↔ `--project-commit`,
/// `Dimension::RunSource` ↔ `--run-source`,
/// `Dimension::CpuBudget` ↔ `--cpu-budget`) so a reader can map
/// from operator surface to internal enum without a translation
/// table.
///
/// `scenario` is NOT a dimension — it is the test name and is
/// always part of the pairing key (you can't compare scenario A
/// against scenario B; that would compare unrelated tests).
///
/// Iteration order via [`Dimension::ALL`] is deterministic and
/// matches the order operators read in the CLI flags
/// (`--kernel` / `--scheduler` / `--topology` / `--work-type` /
/// `--project-commit` / `--kernel-commit` / `--run-source` /
/// `--cpu-budget`), so generated labels and error messages list
/// dims in a stable, predictable order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dimension {
    Kernel,
    Scheduler,
    Topology,
    WorkType,
    ProjectCommit,
    KernelCommit,
    RunSource,
    CpuBudget,
}

impl Dimension {
    /// Every dimension in CLI-flag order. Used by
    /// [`derive_slicing_dims`] to walk the dimension space and by
    /// `compare_partitions` to compute the pairing-dim
    /// complement set (all dims minus slicing dims).
    pub const ALL: &'static [Dimension] = &[
        Dimension::Kernel,
        Dimension::Scheduler,
        Dimension::Topology,
        Dimension::WorkType,
        Dimension::ProjectCommit,
        Dimension::KernelCommit,
        Dimension::RunSource,
        Dimension::CpuBudget,
    ];

    /// Compute pairing dims from a slicing-dim set: every
    /// dimension in [`Dimension::ALL`] that is NOT in `slicing`,
    /// in canonical order. This is the dynamic key derivation the
    /// comparison pipeline uses everywhere — slicing dims define
    /// the contrast (different on A vs B), pairing dims define
    /// the join (same across A and B).
    pub fn pairing_dims(slicing: &[Dimension]) -> Vec<Dimension> {
        Self::ALL
            .iter()
            .copied()
            .filter(|d| !slicing.contains(d))
            .collect()
    }

    /// Operator-readable name for diagnostic and table output.
    /// Matches the CLI flag suffix (e.g. `--kernel` →
    /// `"kernel"`, `--work-type` → `"work-type"`). Used in the
    /// "slicing dimensions: ..." / "pairing on: ..." header
    /// lines and in the "A and B select identical rows" error.
    pub fn name(self) -> &'static str {
        match self {
            Dimension::Kernel => "kernel",
            Dimension::Scheduler => "scheduler",
            Dimension::Topology => "topology",
            Dimension::WorkType => "work-type",
            Dimension::ProjectCommit => "project-commit",
            Dimension::KernelCommit => "kernel-commit",
            Dimension::RunSource => "run-source",
            Dimension::CpuBudget => "cpu-budget",
        }
    }
}

/// Legacy pairing-dim set used by tests that pre-date the
/// dimensional-slicing refactor. Equivalent to the historical
/// hardcoded tuple `(scenario, topology, work_type)` — scenario
/// is always implicit in [`PairingKey::from_row`] and the
/// remaining two dimensions are listed here. Production
/// callers (`compare_partitions`) compute pairing dims via
/// [`Dimension::pairing_dims`] from the slicing-dim derivation;
/// only test fixtures use this constant directly, so it is gated
/// behind `#[cfg(test)]`.
#[cfg(test)]
pub(crate) const LEGACY_PAIRING_DIMS: &[Dimension] = &[Dimension::Topology, Dimension::WorkType];

/// Derive the set of dimensions on which `filter_a` and
/// `filter_b` differ. These are the SLICING dimensions —
/// dimensions on which the two sides select disjoint cohorts and
/// therefore form the A/B contrast. The complement (every other
/// dimension) is the PAIRING-key dimension set used by
/// `compare_rows` to join A-side rows against B-side rows.
///
/// Comparison shape per dimension: every dim uses the same
/// SORTED-DEDUPED `Vec<&str>` comparison — order and multiplicity
/// don't matter (`--a-kernel 6.14 --a-kernel 6.15` and
/// `--b-kernel 6.15 --b-kernel 6.14` are NOT a slice). All eight
/// dimensions are repeatable Vec filters; the previously
/// `Option<String>`-typed `scheduler` / `topology` / `work_type`
/// dims were promoted to `Vec<String>` so the operator-visible
/// shape is uniform across every dimension.
///
/// Returns dimensions in [`Dimension::ALL`] order so callers
/// (header lines, error messages, side labels) get a stable
/// presentation.
pub fn derive_slicing_dims(filter_a: &RowFilter, filter_b: &RowFilter) -> Vec<Dimension> {
    let mut out = Vec::new();
    for &dim in Dimension::ALL {
        let differs = match dim {
            Dimension::Kernel => sorted_dedup(&filter_a.kernels) != sorted_dedup(&filter_b.kernels),
            Dimension::Scheduler => {
                sorted_dedup(&filter_a.schedulers) != sorted_dedup(&filter_b.schedulers)
            }
            Dimension::Topology => {
                sorted_dedup(&filter_a.topologies) != sorted_dedup(&filter_b.topologies)
            }
            Dimension::WorkType => {
                sorted_dedup(&filter_a.work_types) != sorted_dedup(&filter_b.work_types)
            }
            Dimension::ProjectCommit => {
                sorted_dedup(&filter_a.project_commits) != sorted_dedup(&filter_b.project_commits)
            }
            Dimension::KernelCommit => {
                sorted_dedup(&filter_a.kernel_commits) != sorted_dedup(&filter_b.kernel_commits)
            }
            Dimension::RunSource => {
                sorted_dedup(&filter_a.run_sources) != sorted_dedup(&filter_b.run_sources)
            }
            Dimension::CpuBudget => {
                sorted_dedup(&filter_a.cpu_budgets) != sorted_dedup(&filter_b.cpu_budgets)
            }
        };
        if differs {
            out.push(dim);
        }
    }
    out
}

fn sorted_dedup(v: &[String]) -> Vec<&str> {
    let mut s: Vec<&str> = v.iter().map(String::as_str).collect();
    s.sort_unstable();
    s.dedup();
    s
}

/// Render a side's filter values into a column-header label for
/// the comparison table. `dims` is the slicing-dimension set —
/// the only dims whose values vary between A and B. The label
/// concatenates each dim's per-side filter value(s) with `:`
/// between dim values (e.g. `"6.14.2:scx_rusty"` when both
/// `kernel` and `scheduler` slice). For multi-value Vec filters
/// (kernels, commits) the values join with `|` when there
/// are ≤3; longer lists collapse to `"A"` or `"B"` (the bare
/// side label) to keep the column header readable.
///
/// `bare_label` is `"A"` / `"B"`, used as the fallback when a
/// slicing dim's filter has more than 3 values OR the slicing
/// dim's filter is empty on this side (the slice exists because
/// the OTHER side populated the filter — the empty-side label is
/// the bare letter).
pub(crate) fn render_side_label(
    filter: &RowFilter,
    dims: &[Dimension],
    bare_label: &str,
) -> String {
    if dims.is_empty() {
        return bare_label.to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    for &dim in dims {
        let part = match dim {
            Dimension::Kernel => render_vec_dim(&filter.kernels, bare_label),
            Dimension::Scheduler => render_vec_dim(&filter.schedulers, bare_label),
            Dimension::Topology => render_vec_dim(&filter.topologies, bare_label),
            Dimension::WorkType => render_vec_dim(&filter.work_types, bare_label),
            Dimension::ProjectCommit => render_vec_dim(&filter.project_commits, bare_label),
            Dimension::KernelCommit => render_vec_dim(&filter.kernel_commits, bare_label),
            Dimension::RunSource => render_vec_dim(&filter.run_sources, bare_label),
            Dimension::CpuBudget => render_vec_dim(&filter.cpu_budgets, bare_label),
        };
        parts.push(part);
    }
    parts.join(":")
}

/// `≤3` values: join with `|`. `>3` values: collapse to
/// `bare_label`. Empty Vec: also bare label (slicing exists
/// because the OTHER side populated the same dim).
fn render_vec_dim(values: &[String], bare_label: &str) -> String {
    if values.is_empty() || values.len() > 3 {
        bare_label.to_string()
    } else {
        let mut sorted: Vec<&str> = values.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        sorted.join("|")
    }
}

/// Dynamic pairing key for [`compare_rows_by`] — the tuple of
/// values on every NON-slicing dimension, plus the always-pinned
/// `scenario`. Two rows pair iff their dynamic keys match.
///
/// Stored as a `Vec<String>` so the same struct shape works for
/// any `pairing_dims` slice (the alternative — a tuple of
/// `Option<&str>` per dim — would force every consumer to know
/// the dim list at compile time, defeating the point of
/// dimension-set parametrisation).
///
/// First element is always `scenario`; subsequent elements
/// follow `pairing_dims` order (which is itself
/// [`Dimension::ALL`] order minus the slicing dims).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
pub(crate) struct PairingKey(pub Vec<String>);

impl PairingKey {
    /// Extract the pairing key for `row` given the list of
    /// dimensions to include. The scenario is ALWAYS the first
    /// component; the `pairing_dims` list controls the rest.
    /// Each non-scenario dim contributes a single string slot:
    /// `Option<String>` fields render `None` as the empty
    /// string, `Vec<String>` fields render as a sorted-deduped
    /// `|`-joined string so the same set produces the same key
    /// regardless of input order.
    ///
    /// Commit dimensions (`ProjectCommit`, `KernelCommit`) strip the
    /// trailing `-dirty` suffix before contributing to the key.
    /// Without the strip, a clean run at HEAD `abc1234` and a
    /// dirty run at the same HEAD (`abc1234-dirty`) would shatter
    /// into two separate pairing buckets, defeating
    /// [`group_and_average_by`]'s `+mixed` cohort detection — that
    /// helper can only surface "this aggregate has both clean and
    /// dirty contributors" when the two contributors actually land
    /// in the same group. Stripping at the key level pairs them by
    /// canonical hex; the per-row `-dirty` distinction is preserved
    /// downstream in the aggregate's `commit` / `kernel_commit`
    /// field via the `+mixed` marker in
    /// `group_and_average_by`'s `render_mixed_dirty` helper.
    pub fn from_row(row: &GauntletRow, pairing_dims: &[Dimension]) -> Self {
        let mut parts = Vec::with_capacity(1 + pairing_dims.len());
        parts.push(row.scenario.clone());
        for &dim in pairing_dims {
            parts.push(match dim {
                Dimension::Kernel => row.kernel_version.clone().unwrap_or_default(),
                Dimension::Scheduler => row.scheduler.clone(),
                Dimension::Topology => row.topology.clone(),
                Dimension::WorkType => row.work_type.clone(),
                Dimension::ProjectCommit => commit_pairing_key_part(&row.commit),
                Dimension::KernelCommit => commit_pairing_key_part(&row.kernel_commit),
                Dimension::RunSource => row.run_source.clone().unwrap_or_default(),
                // Cross-budget rows never pair: a row's budget value
                // becomes part of its pairing key (None -> empty, distinct
                // from any real budget). A skip (None) only pairs with
                // another skip.
                Dimension::CpuBudget => row.cpu_budget.map(|n| n.to_string()).unwrap_or_default(),
            });
        }
        PairingKey(parts)
    }
}

/// Strip the trailing `-dirty` suffix from a commit dimension's
/// value before it contributes to a [`PairingKey`]. `None` and
/// already-clean values pass through unchanged (`None` → empty
/// string; `Some("abc1234")` → `"abc1234"`); a dirty value
/// (`Some("abc1234-dirty")`) is canonicalized to `"abc1234"` so
/// it pairs with its clean sibling.
///
/// Used by [`PairingKey::from_row`] for both the `ProjectCommit`
/// and `KernelCommit` arms; the per-row `-dirty` distinction is
/// preserved separately by [`group_and_average_by`] via its
/// dirty-tracking accumulator and `+mixed` marker.
fn commit_pairing_key_part(value: &Option<String>) -> String {
    let Some(s) = value.as_deref() else {
        return String::new();
    };
    s.strip_suffix("-dirty").unwrap_or(s).to_string()
}

/// One aggregated `GauntletRow` produced by `group_and_average_by`,
/// plus the pass-bookkeeping needed to render the per-group summary
/// block (`N/M passed` + the `(S skip, I inc, F fail)` breakdown).
///
/// `row` carries arithmetic-mean metric values across every real
/// Pass contributor in the group; the (`scenario`, `topology`,
/// `work_type`, `scheduler`, `kernel_version`) identity is taken
/// verbatim from the first contributor in iteration order — every
/// contributor in the group shares the identity tuple by
/// construction (`scenario`, `topology`, and `work_type` ARE the
/// group key, and `scheduler` / `kernel_version` are
/// typed-filter-narrowed at the call site so they can only vary if
/// the operator passed no `--scheduler` / `--kernel` filter).
///
/// The verdict bits on `row` (`passed`, `skipped`, `inconclusive`)
/// fold under the strict 4-state
/// `Fail > Inconclusive > Pass > Skip` lattice: any failing
/// contributor sets the aggregate to Fail (`passed=false`,
/// `inconclusive=false`, `skipped=false`); else any inconclusive
/// contributor sets `inconclusive=true`; else any skipped
/// contributor sets `skipped=true`; only an all-pass cohort yields
/// `passed=true`. The lattice mechanics match
/// `GauntletRow::is_pass`'s triple-conjunct, so the aggregated
/// row's accessor reads honestly. Aggregate rows that are not real
/// Pass route the pair through `compare_rows_by`'s
/// `excluded_pairs` gate.
///
/// `passes_observed`, `skips_observed`, `inconclusives_observed`,
/// `failures_observed` and `total_observed` count contributors per
/// the strict 4-state mutex: the four bucket counters sum to
/// `total_observed` because every contributor falls into exactly
/// one bucket. Only real Pass contributors feed the per-row sums —
/// failing, inconclusive, and skipped contributors all carry no
/// comparable per-run signal (failure-mode telemetry; "couldn't
/// evaluate" non-signal; "didn't run" non-signal). When no
/// contributor passed cleanly the running sum is zero and the
/// aggregate `row` carries default-zero metric values plus
/// `passed = false` — the downstream `excluded_pairs` gate then
/// drops the pair from the regression math.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct AveragedGroup {
    /// Aggregated row carrying arithmetic-mean metric values plus
    /// the lattice-folded `(passed, skipped, inconclusive)` bits
    /// matching the `Fail > Inconclusive > Pass > Skip`
    /// dominance. `passed` is true only when every contributor was
    /// a real pass; `inconclusive` fires when at least one
    /// contributor was Inconclusive and none failed; `skipped`
    /// fires when at least one contributor was Skip and none
    /// failed or was Inconclusive. Fed directly into
    /// `compare_rows` when `--average` is active.
    pub row: GauntletRow,
    /// Number of contributors that were a real pass
    /// (`is_pass() == true`). Renders as the numerator of the
    /// per-group `N/M` summary.
    pub passes_observed: u32,
    /// Number of contributors that were Skip (`is_skip() == true`).
    /// Surfaced in the per-group rendering as the "S skipped"
    /// breakdown so an operator can distinguish "scenario didn't
    /// run" from real failures.
    pub skips_observed: u32,
    /// Number of contributors that were Inconclusive
    /// (`is_inconclusive() == true`). Surfaced in the per-group
    /// rendering as the "I inconclusive" breakdown so an operator
    /// can distinguish "couldn't evaluate" from real failures —
    /// same defense-in-depth pattern as
    /// `format_dimension_summary`'s inconc bucket.
    pub inconclusives_observed: u32,
    /// Number of contributors that were a real Fail
    /// (`is_fail() == true`). Surfaced in the per-group rendering
    /// as the "F failed" breakdown.
    pub failures_observed: u32,
    /// Total contributors in the group (`= group.len()`). Renders
    /// as the denominator of the per-group `N/M` summary.
    /// Mechanically:
    /// `total_observed == passes_observed + skips_observed +
    /// inconclusives_observed + failures_observed`
    /// under the strict 4-state mutex.
    pub total_observed: u32,
}

/// Per-row dirty-status update used by [`group_and_average_by`] to
/// detect when a group's contributors disagree on the `-dirty`
/// suffix for a commit dimension. `value` is `Some(hex)` /
/// `Some(hex-dirty)` / `None`; the function flips `any_clean` if
/// the value lacks the `-dirty` suffix and `any_dirty` if it
/// carries one. `first_base` records the first un-suffixed form
/// seen (used to render the `+mixed` marker against a canonical
/// hex even when `acc.first` happens to be the dirty form).
///
/// Per-row scope spans EVERY contributor (passing, failing,
/// skipped). Mixed-dirty is metadata about the cohort's working-
/// tree state, not about which contributors succeeded — surfacing
/// it only across passes would hide WIP-vs-committed disagreement
/// that the operator needs to know about. `None` values do not
/// flip either flag and do not seed `first_base`.
fn update_dirty_tracking(
    value: &Option<String>,
    any_clean: &mut bool,
    any_dirty: &mut bool,
    first_base: &mut Option<String>,
) {
    let Some(s) = value.as_deref() else { return };
    let (base, is_dirty) = match s.strip_suffix("-dirty") {
        Some(base) => (base, true),
        None => (s, false),
    };
    if is_dirty {
        *any_dirty = true;
    } else {
        *any_clean = true;
    }
    if first_base.is_none() {
        *first_base = Some(base.to_string());
    }
}

/// Render the aggregate's commit string for one dimension
/// (project_commit or kernel_commit) given the cohort-wide
/// dirty/clean tracking state. When `any_clean && any_dirty` for
/// the same un-suffixed hex, the rendered form is
/// `Some("{first_base}+mixed")`; otherwise the function returns
/// `acc.first.commit` (or `acc.first.kernel_commit`) verbatim,
/// preserving the existing first-seen behaviour for homogeneous
/// cohorts (every contributor clean, every contributor dirty, or
/// every contributor `None`).
///
/// `first_base` is the canonical un-suffixed hex captured by
/// [`update_dirty_tracking`]; using it (rather than stripping
/// `acc.first.commit`) ensures the rendered form is `abc1234+mixed`
/// regardless of whether the first contributor was clean or dirty.
fn render_mixed_dirty(
    any_clean: bool,
    any_dirty: bool,
    first_base: &Option<String>,
    first_commit: &Option<String>,
) -> Option<String> {
    if any_clean
        && any_dirty
        && let Some(base) = first_base
    {
        return Some(format!("{base}+mixed"));
    }
    first_commit.clone()
}

/// Per-pairing-group fold accumulator for [`group_and_average_by`].
/// Built via [`Accumulator::new`] from the group's first contributor,
/// fed one contributor at a time via [`Accumulator::observe`], and
/// folded into the emitted [`AveragedGroup`] via
/// [`Accumulator::into_averaged_group`]. Split out of
/// `group_and_average_by` only to satisfy the source-function size
/// guard — the field set and fold math are unchanged from the
/// in-function definition.
struct Accumulator<'a> {
    first: &'a GauntletRow,
    total_observed: u32,
    passes_observed: u32,
    skips_observed: u32,
    inconclusives_observed: u32,
    failures_observed: u32,
    any_skipped: bool,
    any_failed: bool,
    any_inconclusive: bool,
    // Tracks whether contributors disagree on the `-dirty`
    // suffix for the project_commit / kernel_commit dimensions.
    // `any_*_clean` is true if any contributor's value is the
    // un-suffixed form; `any_*_dirty` is true if any contributor
    // ends in `-dirty`. When BOTH are true the aggregate is
    // mixed-dirty and the rendered `commit` / `kernel_commit`
    // gets a `+mixed` marker so downstream readers don't see a
    // single arbitrary contributor's status. Tracked across
    // EVERY contributor (passing, failing, skipped) — a mixed
    // working-tree state is metadata about the cohort, not
    // about the metric mean. Empty / `None` values are ignored
    // and do not flip either flag.
    any_project_clean: bool,
    any_project_dirty: bool,
    any_kernel_clean: bool,
    any_kernel_dirty: bool,
    // First-seen un-suffixed (clean-form) project / kernel
    // commit string. Held separately from `first` because
    // `first.commit` may be `Some("abc1234-dirty")` when the
    // first contributor was dirty but later contributors carry
    // the clean form — the rendered `+mixed` marker should
    // still attach to the canonical un-suffixed hex so the
    // operator sees `abc1234+mixed` not `abc1234-dirty+mixed`.
    first_project_base: Option<String>,
    first_kernel_base: Option<String>,
    // Sums across passing+non-skipped contributors only.
    // Counts are tracked per ext_metric key separately because
    // a key may be absent from some contributors.
    // Per-row sum for mean-fold fields (Counter / Gauge(Last) /
    // Gauge(Avg) — though no typed Gauge(Avg) field exists
    // today). Arithmetic mean across runs is the operator-
    // facing cohort-comparison default; per-RUN totals are
    // averaged to produce a comparable per-run quantity
    // across cohorts of different run counts.
    sum_spread: f64,
    sum_migrations: u64,
    sum_migration_ratio: f64,
    sum_stuck_count: f64,
    sum_fallback_count: i64,
    sum_keep_last_count: i64,
    sum_total_iterations: u64,
    sum_page_locality: f64,
    sum_cross_node_mig: f64,
    // Per-row MAX-fold for Peak-kind fields. Per
    // `MetricKind::Peak` contract, cross-RUN aggregation
    // surfaces the worst-instant observed across the cohort —
    // averaging Peak across runs dilutes the high-water signal
    // (a 1-run spike at 100 averaged with 4 runs at 0 reports
    // 20, hiding the actual peak). MAX preserves "did this
    // peak ever fire in this cohort".
    max_gap_ms: u64,
    max_imbalance_ratio: f64,
    max_max_dsq_depth: u32,
    // Per-ext-metric (value, weight) pairs, accumulated across
    // contributors. At emit time the kind-aware fold dispatches
    // each key through `aggregate_samples` with `Some(&weights)`
    // so Gauge(Avg) metrics get a weighted mean (per the F-C
    // fix on aggregate_samples) and other kinds fold by their
    // own semantics. Unregistered metric names (no MetricDef)
    // fall back to arithmetic mean — same legacy semantic the
    // previous (sum, u32) shape produced.
    ext_pairs: BTreeMap<String, Vec<(f64, usize)>>,
    // Sum of `run_sample_count` across contributors. Carries
    // through to the aggregated row's `run_sample_count` so a
    // downstream cross-RUN consumer that further folds these
    // already-aggregated rows can apply the same weighted
    // semantic. Currently no typed Gauge(Avg) field exists
    // (imbalance_ratio is registered as `max_imbalance_ratio`
    // kind=Peak, NOT Gauge(Avg) — the Gauge(Avg) sibling
    // `avg_imbalance_ratio` lands in ext_metrics where the
    // weighted-mean dispatch already fires); the sum is
    // preserved here for future typed-field Gauge(Avg)
    // additions and for downstream cohort-of-cohort
    // aggregation that wants a meaningful weight.
    sum_run_sample_count: usize,
}

impl<'a> Accumulator<'a> {
    /// Seed an accumulator from the group's first contributor.
    /// Identity is taken from `first`; every counter / sum / max
    /// starts at its zero value. `observe` (called once per
    /// contributor, including `first`) performs the per-row fold.
    fn new(first: &'a GauntletRow) -> Self {
        Accumulator {
            first,
            total_observed: 0,
            passes_observed: 0,
            skips_observed: 0,
            inconclusives_observed: 0,
            failures_observed: 0,
            any_skipped: false,
            any_failed: false,
            any_inconclusive: false,
            any_project_clean: false,
            any_project_dirty: false,
            any_kernel_clean: false,
            any_kernel_dirty: false,
            first_project_base: None,
            first_kernel_base: None,
            sum_spread: 0.0,
            sum_migrations: 0,
            sum_migration_ratio: 0.0,
            sum_stuck_count: 0.0,
            sum_fallback_count: 0,
            sum_keep_last_count: 0,
            sum_total_iterations: 0,
            sum_page_locality: 0.0,
            sum_cross_node_mig: 0.0,
            max_gap_ms: 0,
            max_imbalance_ratio: 0.0,
            max_max_dsq_depth: 0,
            ext_pairs: BTreeMap::new(),
            sum_run_sample_count: 0,
        }
    }

    /// Fold one contributor into the accumulator. Called once per
    /// row in the group (including the group's first contributor).
    /// Skip / fail / inconclusive contributors flip their verdict
    /// bits and return early without feeding the metric sums; only
    /// real passes contribute to the per-row sums and maxes.
    fn observe(&mut self, row: &GauntletRow) {
        self.total_observed += 1;
        // Dirty-status tracking spans ALL contributors. Same hex
        // with mixed dirty/clean across the cohort is the case the
        // `+mixed` marker exists to surface — the per-row scope
        // (passing, failing, skipped) is irrelevant since the
        // marker describes WIP-vs-committed disagreement among the
        // contributors, not their metric outcomes.
        update_dirty_tracking(
            &row.commit,
            &mut self.any_project_clean,
            &mut self.any_project_dirty,
            &mut self.first_project_base,
        );
        update_dirty_tracking(
            &row.kernel_commit,
            &mut self.any_kernel_clean,
            &mut self.any_kernel_dirty,
            &mut self.first_kernel_base,
        );
        if row.is_skip() {
            self.any_skipped = true;
            self.skips_observed += 1;
            return;
        }
        if row.is_fail() {
            self.any_failed = true;
            self.failures_observed += 1;
            return;
        }
        if row.is_inconclusive() {
            // Inconclusive contributors are not passes (the gate
            // could not be evaluated) and carry no measured signal
            // worth folding into the cohort means. Track the bit
            // for the aggregated verdict's `inconclusive` field
            // (so the aggregate row reads Inconclusive in the
            // `Fail > Inconclusive > Pass > Skip` lattice when no
            // contributor failed) and skip the per-row sums.
            self.any_inconclusive = true;
            self.inconclusives_observed += 1;
            return;
        }
        self.passes_observed += 1;
        self.sum_spread += row.spread;
        self.sum_migrations = self.sum_migrations.saturating_add(row.migrations);
        self.sum_migration_ratio += row.migration_ratio;
        self.sum_stuck_count += row.stuck_count;
        self.sum_fallback_count = self.sum_fallback_count.saturating_add(row.fallback_count);
        self.sum_keep_last_count = self.sum_keep_last_count.saturating_add(row.keep_last_count);
        self.sum_total_iterations = self
            .sum_total_iterations
            .saturating_add(row.total_iterations);
        self.sum_page_locality += row.page_locality;
        self.sum_cross_node_mig += row.cross_node_migration_ratio;
        // Peak-kind typed fields: cross-RUN aggregation surfaces
        // the worst-instant observed across the cohort, NOT the
        // arithmetic mean (which dilutes a single peak across
        // many quiet runs and hides the high-water signal).
        self.max_gap_ms = self.max_gap_ms.max(row.gap_ms);
        if row.imbalance_ratio > self.max_imbalance_ratio {
            self.max_imbalance_ratio = row.imbalance_ratio;
        }
        self.max_max_dsq_depth = self.max_max_dsq_depth.max(row.max_dsq_depth);
        self.sum_run_sample_count = self
            .sum_run_sample_count
            .saturating_add(row.run_sample_count);
        for (k, v) in &row.ext_metrics {
            self.ext_pairs
                .entry(k.clone())
                .or_default()
                .push((*v, row.run_sample_count));
        }
    }

    /// Emit the folded [`AveragedGroup`] for this group. Identity
    /// fields are first-seen; metric fields are the kind-correct
    /// cross-RUN fold (mean for Counter / mean-fold, MAX for Peak,
    /// rounded mean for integer-typed fields); the verdict bits
    /// fold under the `Fail > Inconclusive > Pass > Skip` lattice.
    fn into_averaged_group(self) -> AveragedGroup {
        let acc = self;
        let n = acc.passes_observed;
        let denom = if n == 0 { 1.0 } else { f64::from(n) };
        // Rounded mean for integer-typed Counter / mean-fold
        // fields. When n == 0 the sums are all zero, so dividing
        // by 1.0 still yields 0 — the aggregate's passed=false
        // routes the pair through excluded_pairs downstream and
        // the metrics are never consulted. Peak-kind integer
        // fields (max_dsq_depth) take the MAX-fold path directly
        // and don't need a rounding helper.
        let round_u64 = |sum: u64| -> u64 { (sum as f64 / denom).round() as u64 };
        let round_i64 = |sum: i64| -> i64 { (sum as f64 / denom).round() as i64 };

        // Mixed-dirty markers. When the cohort contains both a
        // clean-form and dirty-form contributor for the same hex
        // (e.g. some sidecars from a clean tree, others from a
        // -dirty WIP), the rendered commit field carries `+mixed`
        // appended to the canonical un-suffixed hex. The
        // alternative — taking `acc.first.commit` verbatim — would
        // hide WIP-vs-committed disagreement, presenting `abc1234`
        // when half the contributors actually came from a dirty
        // tree (or `abc1234-dirty` when half came from a clean
        // tree). Operators reading averaged stats need to know the
        // cohort spanned a working-tree state change, since that
        // changes the meaning of the metric mean. `+mixed` is the
        // chosen separator (not `-mixed`) so it cannot be confused
        // with the existing `-dirty` suffix grammar — `dirty` is a
        // per-record property, `mixed` is a cohort-level property.
        let project_commit_rendered = render_mixed_dirty(
            acc.any_project_clean,
            acc.any_project_dirty,
            &acc.first_project_base,
            &acc.first.commit,
        );
        let kernel_commit_rendered = render_mixed_dirty(
            acc.any_kernel_clean,
            acc.any_kernel_dirty,
            &acc.first_kernel_base,
            &acc.first.kernel_commit,
        );
        // ext_metrics is built BEFORE the struct so Rate keys can be
        // re-derived from the folded components as a post-pass. ONLY Rate is
        // skipped here: its components survive cross-RUN as their own ext keys
        // so it re-derives Σnum/Σdenom (folding two ready-made ratios would
        // lose the re-pool, and routing a Rate through
        // aggregate_samples_weighted would hit the aggregate_finite guard).
        // Distribution / WorstLowest are NOT skipped — their raw components do
        // NOT survive cross-RUN (phases are dropped), so there is no pooled set
        // to re-derive; they fall through to aggregate_samples_weighted and
        // fold by kind (MEAN for the percentile / CV / mean reductions and
        // every WorstLowest, MAX for SampleReduction::Worst — the
        // aggregate_finite arms). Dispatch by registered MetricKind so
        // Gauge(Avg) gets the weighted-mean fold (matches the per-phase merge
        // contract); unregistered names (no metric_def) fall back to
        // arithmetic mean, the legacy (sum, count) semantic. Skip a key whose
        // reduction is None (every value NaN — defensive post sidecar_to_row
        // sanitize).
        let ext_metrics = fold_ext_metrics(acc.ext_pairs);
        let aggregated = GauntletRow {
            scenario: acc.first.scenario.clone(),
            topology: acc.first.topology.clone(),
            work_type: acc.first.work_type.clone(),
            scheduler: acc.first.scheduler.clone(),
            kernel_version: acc.first.kernel_version.clone(),
            commit: project_commit_rendered,
            kernel_commit: kernel_commit_rendered,
            run_source: acc.first.run_source.clone(),
            // First-seen budget metadata, like scheduler/kernel_version
            // above. When CpuBudget is a PAIRING dim it is part of the
            // group key, so every contributor shares one budget and the
            // first row's value is the group's. When the operator slices
            // on budget (e.g. an asymmetric `--a-cpu-budget`), CpuBudget
            // is a SLICING dim and is dropped from the pairing key, so a
            // group's contributors may carry heterogeneous budgets — the
            // first-seen value is then representative metadata, not a join
            // key, and `render_overcommit_warning` surfaces the cross-budget
            // mix on the compared sides. vcpus is likewise first-seen
            // metadata — and is NOT a Dimension, so a TOPOLOGY-sliced group
            // (vcpus = topology.total_cpus()) can mix vcpus too. No
            // post-aggregation consumer reads the aggregated vcpus (the
            // overcommit checks run pre-aggregation on the raw rows), so the
            // first-seen value is metadata only.
            cpu_budget: acc.first.cpu_budget,
            vcpus: acc.first.vcpus,
            // ALL must pass: any failed, inconclusive, or skipped
            // contributor flips the aggregate. A group with zero
            // passes_observed (every contributor failed, was
            // inconclusive, or was skipped) collapses to
            // passed=false here. The four-bit verdict is
            // strict 4-state (exactly one of pass/skip/inconc/fail
            // set per row); the lattice
            // `Fail > Inconclusive > Pass > Skip` determines which
            // bit dominates when a cohort has mixed contributors.
            // Skip is the lowest-precedence bit — it fires only
            // when no contributor failed AND no contributor was
            // inconclusive AND at least one was skipped. Fail
            // (all-false) dominates Inconclusive dominates Skip;
            // exactly one of the four states is encoded per row.
            passed: !acc.any_failed && !acc.any_inconclusive && !acc.any_skipped && n > 0,
            skipped: !acc.any_failed && !acc.any_inconclusive && acc.any_skipped,
            inconclusive: !acc.any_failed && acc.any_inconclusive,
            // Sum across contributors so the aggregated row's
            // weight is the cohort's total sample population. A
            // downstream consumer that further folds these
            // aggregated rows can apply the same weighted semantic
            // (a 5-RUN cohort of 50-sample runs weighs 250 vs a
            // 1-RUN cohort of 10 samples weighting 10).
            run_sample_count: acc.sum_run_sample_count,
            spread: acc.sum_spread / denom,
            // Peak-kind typed fields: MAX across runs (kind-correct
            // cross-RUN fold; arithmetic mean dilutes the
            // worst-instant signal).
            gap_ms: acc.max_gap_ms,
            imbalance_ratio: acc.max_imbalance_ratio,
            max_dsq_depth: acc.max_max_dsq_depth,
            migrations: round_u64(acc.sum_migrations),
            migration_ratio: acc.sum_migration_ratio / denom,
            stuck_count: acc.sum_stuck_count / denom,
            fallback_count: round_i64(acc.sum_fallback_count),
            keep_last_count: round_i64(acc.sum_keep_last_count),
            total_iterations: round_u64(acc.sum_total_iterations),
            page_locality: acc.sum_page_locality / denom,
            cross_node_migration_ratio: acc.sum_cross_node_mig / denom,
            ext_metrics,
            // Phase buckets do not aggregate cleanly across an
            // averaged group: two contributors might run different
            // scenarios with different phase counts, and per-phase
            // averaging across mismatched step_index sets would
            // invent rows neither side carried. Surface the empty
            // slice so downstream consumers fall back to the flat
            // bucket. A future MergeKind::Phase aware merge will
            // revisit this once compare_partitions' cross-cardinality
            // (per-step_index intersection + unpaired surfacing)
            // lands and gives us a tested intersection semantic to
            // reuse here.
            phases: Vec::new(),
        };
        AveragedGroup {
            row: aggregated,
            passes_observed: acc.passes_observed,
            skips_observed: acc.skips_observed,
            inconclusives_observed: acc.inconclusives_observed,
            failures_observed: acc.failures_observed,
            total_observed: acc.total_observed,
        }
    }
}

/// Fold one group's accumulated per-ext-metric (value, weight) pairs
/// into the aggregated row's `ext_metrics` map. ONLY Rate is skipped
/// in the kind dispatch: its components survive cross-RUN as their own
/// ext keys so it re-derives Σnum/Σdenom (folding two ready-made
/// ratios would lose the re-pool, and routing a Rate through
/// aggregate_samples_weighted would hit the aggregate_finite guard).
/// Distribution / WorstLowest are NOT skipped — their raw components do
/// NOT survive cross-RUN (phases are dropped), so there is no pooled set
/// to re-derive; they fall through to aggregate_samples_weighted and
/// fold by kind (MEAN for the percentile / CV / mean reductions and
/// every WorstLowest, MAX for SampleReduction::Worst — the
/// aggregate_finite arms). Dispatch by registered MetricKind so
/// Gauge(Avg) gets the weighted-mean fold (matches the per-phase merge
/// contract); unregistered names (no metric_def) fall back to
/// arithmetic mean, the legacy (sum, count) semantic. Skip a key whose
/// reduction is None (every value NaN — defensive post sidecar_to_row
/// sanitize). Rate metrics are then re-derived from the folded
/// components (Σnum/Σdenom) as a post-pass.
fn fold_ext_metrics(
    ext_pairs: BTreeMap<String, Vec<(f64, usize)>>,
) -> BTreeMap<String, f64> {
    let mut ext_metrics: std::collections::BTreeMap<String, f64> = ext_pairs
        .into_iter()
        .filter_map(|(k, pairs)| {
            if let Some(def) = metric_def(&k) {
                if matches!(def.kind, MetricKind::Rate { .. }) {
                    return None;
                }
                aggregate_samples_weighted(&pairs, def.kind).map(|v| (k, v))
            } else {
                let n = pairs.len();
                if n == 0 {
                    None
                } else {
                    let sum: f64 = pairs.iter().map(|(v, _)| *v).sum();
                    Some((k, sum / n as f64))
                }
            }
        })
        .collect();
    // Re-derive Rate metrics from the folded components (Σnum/Σdenom).
    derive_rate_metrics(&mut ext_metrics);
    ext_metrics
}

/// Group `rows` by the dynamic pairing key (`scenario` plus every
/// dimension in `pairing_dims`) and arithmetic-mean their metric
/// fields, returning one [`AveragedGroup`] per distinct key.
/// Slicing dims are EXCLUDED from `pairing_dims` (rows on the A/B
/// sides differ on them by design); pairing dims are INCLUDED.
///
/// Group key matches [`compare_rows_by`]' pairing key so the post-
/// aggregation row vec joins cleanly across A/B sides under the
/// same identity contract.
///
/// Aggregation rules:
/// - The verdict bits `(passed, skipped, inconclusive)` aggregate
///   under the strict 4-state mutex per the
///   `Fail > Inconclusive > Pass > Skip` lattice. Fail (all-false)
///   dominates: any failed contributor flips the aggregate's
///   `passed` to `false` and leaves `skipped`/`inconclusive` clear,
///   yielding Fail at the aggregate level. Otherwise Inconclusive
///   dominates: any inconclusive contributor sets the aggregate's
///   `inconclusive = true`. Otherwise Skip dominates: any skipped
///   contributor sets `skipped = true`. Only when every contributor
///   was a real Pass does the aggregate carry `passed = true`. This
///   matches [`GauntletRow::is_pass`]'s triple-conjunct semantics
///   so the aggregate's accessor reads honestly.
/// - Metrics (`f64` / `u64` / `i64` fields, plus `ext_metrics`
///   entries) are summed only across contributors where
///   `passed && !skipped`, then divided by that count to yield an
///   arithmetic mean. Failing/skipped contributors carry telemetry
///   dominated by the failure mode, NOT scheduler behaviour, and
///   are therefore excluded from the mean. When no contributor
///   passed cleanly, every metric defaults to zero and the
///   aggregate's `passed = false` routes the pair to
///   [`compare_rows_by`]' `excluded_pairs` gate.
/// - `u64` / `i64` fields take the rounded mean
///   (`(sum / count).round() as u64`). The up-to-0.5-unit rounding
///   error is well below each such field's `default_abs` gate (the
///   smallest is `total_fallback` / `total_keep_last` at 5.0).
/// - `stuck_count` is the exception: it is `f64` and carries the
///   EXACT mean (`sum / count`, no rounding). Its `default_abs` is
///   1.0 — tight enough that a rounded mean's up-to-1.0 per-A/B-pair
///   error would fabricate single-stall regressions from sub-integer
///   differences (an A-side mean of 1.4 vs a B-side 1.6 rounds to
///   1 vs 2, a spurious delta of 1).
/// - `ext_metrics` keys are unioned across passing contributors;
///   each key's mean is computed only across contributors that
///   carried it. A key present in some passing rows and absent
///   from others uses the present-only count as its denominator —
///   absent-and-zero are not equivalent (the `BTreeMap<String,
///   f64>` shape cannot represent "absent" with a stored zero).
/// - Identity fields (`scenario`, `topology`, `work_type`,
///   `scheduler`, `kernel_version`) come from the first contributor
///   in iteration order. Every contributor in the group shares the
///   first three by construction (group key); `scheduler` and
///   `kernel_version` may vary across the group if the operator did
///   not narrow via typed filters first, but the aggregated row
///   carries the first contributor's value in any case — the join
///   downstream uses the three-tuple, so scheduler/version on the
///   aggregate is metadata, not a join key.
/// - Commit dimensions (`commit`, `kernel_commit`) follow a
///   first-seen rule with one exception: when contributors disagree
///   on the `-dirty` suffix for the same canonical hex (some clean,
///   some dirty), the rendered form becomes `{hex}+mixed` so the
///   working-tree disagreement is surfaced rather than hidden by
///   first-seen. `+mixed` (not `-mixed`) is intentional —
///   `-dirty` is a per-record property of one sidecar, `+mixed`
///   is a cohort-level property of the average. Mixed-dirty
///   tracking spans EVERY contributor (passing, failing, skipped)
///   because the cohort's WIP state is metadata, not a metric.
///
/// Group iteration order matches the order of FIRST appearance of
/// each key in `rows`; `BTreeMap` ordering is by key (not iteration
/// order) so we maintain a parallel `Vec<key>` to preserve
/// first-seen ordering. Stable order keeps test fixtures
/// deterministic across runs.
pub fn group_and_average_by(
    rows: &[GauntletRow],
    pairing_dims: &[Dimension],
) -> Vec<AveragedGroup> {
    // Dynamic pairing key — scenario + every NON-slicing
    // dimension's value, in [`Dimension::ALL`] order. The
    // `PairingKey` newtype is owned (`Vec<String>`) so the
    // BTreeMap can hold keys without lifetime gymnastics; the
    // alternative — borrowing slices into `rows` — would force
    // every consumer to keep `rows` alive for the duration of
    // the map.
    type Key = PairingKey;

    let mut order: Vec<Key> = Vec::new();
    let mut groups: BTreeMap<Key, Accumulator<'_>> = BTreeMap::new();

    for row in rows {
        let key = PairingKey::from_row(row, pairing_dims);
        let acc = groups.entry(key.clone()).or_insert_with(|| {
            order.push(key);
            Accumulator::new(row)
        });
        acc.observe(row);
    }

    let mut out = Vec::with_capacity(order.len());
    for key in order {
        let acc = groups
            .remove(&key)
            .expect("first-seen key must still be in groups map");
        out.push(acc.into_averaged_group());
    }
    out
}

/// Convert a SidecarResult to a GauntletRow for run-to-run comparison.
///
/// Non-finite f64 values (NaN, ±Infinity) are sanitized to 0.0 with a
/// warn before they reach the row. `serde_json::to_string` rejects
/// non-finite, so a single poisoned metric would otherwise halt every
/// downstream JSON write. Sanitizing at the ingress boundary keeps the
/// serializer happy without silencing the upstream data quality issue.
///
/// # NaN → 0.0 ambiguity for zero-meaningful metrics
///
/// The 0.0 substitution is indistinguishable from a legitimate 0.0
/// measurement for metrics whose natural zero carries its own signal.
/// Two direct f64 fields are especially affected — note in-tree producers
/// already guard the typical divide-by-zero path (`assert.rs` emits
/// `0.0` for migration_ratio when `total_iters == 0` and `1.0` for
/// page_locality when `total == 0`), so a NaN reaching this boundary
/// indicates an upstream producer outside those guards (e.g. an
/// external `ext_metrics` contributor, or a schedstat arithmetic
/// edge that slipped past a guard):
///
/// - `migration_ratio`: lower-better. A real 0.0 means "no task was
///   migrated" (ideal locality). A sanitized NaN collapses to the
///   same value and reads as *falsely good* — a downstream regression
///   gate sees "perfect locality" where the truth is "no data".
/// - `page_locality`: higher-better. A real 0.0 means "no local-node
///   accesses". A sanitized NaN collapses to the same value and
///   reads as *falsely bad* — a downstream regression gate sees
///   "everything cross-node" where the truth is "no data". The
///   polarity is opposite to `migration_ratio`: the two failure
///   modes push the comparison in opposite directions.
///
/// The reclassified wake-latency / run-delay distributions (e.g.
/// `worst_wake_latency_cv`) are NO LONGER direct f64 fields — they flow
/// through `ext_metrics`, where a non-finite value is DROPPED (the entry is
/// absent), NOT substituted with 0.0. That is the opposite, no-false-zero
/// contract: an absent key reads as no-data, distinct from a measured 0.0.
///
/// The accompanying `tracing::warn!` is the only signal that
/// separates a sanitized NaN from a real 0.0; downstream aggregation
/// by value alone cannot distinguish them.
pub fn sidecar_to_row(sc: &crate::test_support::SidecarResult) -> GauntletRow {
    // Local closure so the warn can carry the scenario name as
    // context — keyed by field so the operator can pinpoint which
    // metric produced the bad value.
    let finite_or_zero = |field: &str, v: f64| -> f64 {
        if v.is_finite() {
            v
        } else {
            tracing::warn!(
                test = %sc.test_name,
                field,
                value = v,
                "non-finite f64 in GauntletRow field; substituting 0.0",
            );
            0.0
        }
    };

    GauntletRow {
        scenario: sc.test_name.clone(),
        topology: sc.topology.clone(),
        work_type: sc.work_type.clone(),
        scheduler: sc.scheduler.clone(),
        kernel_version: sc.kernel_version.clone(),
        commit: sc.project_commit.clone(),
        kernel_commit: sc.kernel_commit.clone(),
        run_source: sc.run_source.clone(),
        // 0 = skip rows (never booted) -> None: skips carry no budget
        // identity, so they don't pair into a "budget 0" bucket.
        cpu_budget: (sc.cpu_budget != 0).then_some(sc.cpu_budget),
        vcpus: (sc.vcpus != 0).then_some(sc.vcpus),
        passed: sc.is_pass(),
        skipped: sc.is_skip(),
        inconclusive: sc.is_inconclusive(),
        run_sample_count: sc.monitor.as_ref().map(|m| m.total_samples).unwrap_or(0),
        spread: finite_or_zero("spread", sc.stats.worst_spread),
        gap_ms: sc.stats.worst_gap_ms,
        migrations: sc.stats.total_migrations,
        migration_ratio: finite_or_zero("migration_ratio", sc.stats.worst_migration_ratio),
        imbalance_ratio: finite_or_zero(
            "imbalance_ratio",
            sc.monitor
                .as_ref()
                .map(|m| m.max_imbalance_ratio)
                .unwrap_or(0.0),
        ),
        max_dsq_depth: sc
            .monitor
            .as_ref()
            .map(|m| m.max_local_dsq_depth)
            .unwrap_or(0),
        stuck_count: sc.monitor.as_ref().map(|m| m.stuck_count).unwrap_or(0) as f64,
        fallback_count: sc
            .monitor
            .as_ref()
            .and_then(|m| m.event_deltas.as_ref())
            .map(|e| e.total_fallback)
            .unwrap_or(0),
        keep_last_count: sc
            .monitor
            .as_ref()
            .and_then(|m| m.event_deltas.as_ref())
            .map(|e| e.total_dispatch_keep_last)
            .unwrap_or(0),
        total_iterations: sc.stats.total_iterations,
        page_locality: finite_or_zero("page_locality", sc.stats.worst_page_locality),
        cross_node_migration_ratio: finite_or_zero(
            "cross_node_migration_ratio",
            sc.stats.worst_cross_node_migration_ratio,
        ),
        // Non-finite entries would also break `serde_json::to_string`,
        // but the map shape makes "substitute 0.0" ambiguous (the entry
        // might legitimately be 0.0 for a different scenario). Drop the
        // entry entirely so the non-finite value can't be confused with
        // a real zero datapoint.
        //
        // Also drop the walk-depth truncation sentinel
        // [`crate::test_support::WALK_TRUNCATION_SENTINEL_NAME`]:
        // it is diagnostic metadata from the JSON-walker depth cap,
        // not a scenario metric, and must not participate in A/B
        // comparison output.
        ext_metrics: sc
            .stats
            .ext_metrics
            .iter()
            .filter_map(|(k, &v)| {
                if crate::test_support::is_truncation_sentinel_name(k) {
                    return None;
                }
                if v.is_finite() {
                    Some((k.clone(), v))
                } else {
                    tracing::warn!(
                        test = %sc.test_name,
                        metric = %k,
                        value = v,
                        "dropping non-finite ext_metric; serde_json rejects NaN/Infinity",
                    );
                    None
                }
            })
            .collect(),
        // Carry per-phase buckets verbatim from the source
        // ScenarioStats. The bucket structure has already been
        // reduced by the host-side phase aggregator (Counter via
        // `phase_counter_delta`, Gauge/Peak/Timestamp via
        // `aggregate_samples`), so the sidecar -> row step just
        // forwards the prebuilt slice. An empty `phases` slot on
        // the source sidecar (single-phase scenario or legacy
        // file) flows through as an empty slice.
        phases: sc.stats.phases.clone(),
    }
}
