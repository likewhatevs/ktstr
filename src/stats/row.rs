use super::*;

/// Per-scenario result row for gauntlet analysis and run-to-run comparison.
///
/// Populated by [`sidecar_to_row`] from on-disk [`SidecarResult`](crate::test_support::SidecarResult)s. The
/// comparison pipeline reads metric values through [`MetricDef::read`]
/// / [`METRICS`] rather than dereferencing fields directly so new
/// metrics can land through the registry without touching every
/// reader.
///
/// # NaN-ambiguity on direct f64 fields
///
/// All direct f64 fields on this struct are sanitized via
/// `finite_or_zero` at [`sidecar_to_row`] ingress. A `0.0` on any
/// direct f64 field may represent either a genuine zero measurement
/// or a sanitized non-finite upstream value (NaN / ±Infinity). See
/// [`sidecar_to_row`]'s NaN-ambiguity doc for the full policy;
/// `tracing::warn!` is the disambiguation channel — the sanitizer
/// warns on every non-finite it rewrites to zero, so the log
/// timeline tells you which run's zeroes were real. Consumers that
/// cannot accept the ambiguity should prefer metric paths that
/// flow through `ext_metrics` (a `BTreeMap<String, f64>` — see the
/// field definition below): non-finite entries are DROPPED at
/// [`sidecar_to_row`] ingress rather than stored. A subsequent
/// `ext_metrics.get(name)` returns `None` because the key is
/// absent, not because an `Option::None` sentinel is stored — the
/// map's value type is `f64`, which cannot represent "missing".
/// Absent-key and zero-valued metrics therefore remain distinguishable
/// for downstream consumers.
///
/// # `#[non_exhaustive]` migration note
///
/// Downstream code that pattern-matches a `GauntletRow` must end
/// the match with `..`; future fields added alongside new metrics
/// otherwise break every matcher. Prefer reading values via
/// [`MetricDef::read`] / the registry — the point of the
/// registry indirection is that new metrics do not touch
/// existing readers.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct GauntletRow {
    pub scenario: String,
    pub topology: String,
    pub work_type: String,
    /// Effective host-CPU budget the run's vCPU threads ran on
    /// (`SidecarResult::cpu_budget`); `None` for skip rows (budget 0).
    /// Drives the [`Dimension::CpuBudget`] pairing so cross-budget runs
    /// are never compared (confining 32 vCPUs to 4 host CPUs measures
    /// something different), and (with [`vcpus`](Self::vcpus)) feeds the
    /// compare-path overcommit warning in [`render_overcommit_warning`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_budget: Option<u32>,
    /// Guest vCPU count (`SidecarResult::vcpus`); `None` for skip rows.
    /// NOT a Dimension — it rides alongside [`cpu_budget`](Self::cpu_budget)
    /// so [`render_overcommit_warning`] can flag a compared run whose host
    /// time-sliced its vCPUs (`cpu_budget < vcpus`), whose guest-scheduler
    /// timing metrics are then host-contention-confounded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vcpus: Option<u32>,
    /// Denomination of this row's workload-throughput rate metrics,
    /// carried from `SidecarResult::throughput_denomination`
    /// (`#[serde(default)]` = `Wall` for pre-marker rows). NOT an
    /// operator-facing Dimension — it is an unconditional part of every
    /// `PairingKey` (like `scenario`; see `super::group`), so a
    /// wall-era row never pairs with, or group-averages into, a
    /// cpu-era row: the throughput keys kept their names across the
    /// denomination change and their values are not comparable across it.
    /// Cross-era near-misses are surfaced by
    /// `CompareReport::denomination_mismatches`.
    #[serde(default)]
    pub throughput_denomination: crate::test_support::ThroughputDenomination,
    /// Scheduler binary name carried from the source sidecar
    /// (`SidecarResult::scheduler`). Surfaced through the substring
    /// filter in [`compare_rows_by`] and the typed
    /// `RowFilter::schedulers` so users can narrow A/B comparisons
    /// by scheduler name.
    pub scheduler: String,
    /// Kernel version carried from the source sidecar
    /// (`SidecarResult::kernel_version`). `None` when the sidecar
    /// writer could not extract a version (e.g. a raw kernel image
    /// path with no metadata.json sibling, or a dirty source tree
    /// where HEAD does not describe the build). Surfaced via the
    /// typed [`RowFilter::kernels`] for narrowing — when the user
    /// passes `--kernel 6.14.2` (repeatable), rows with `None` are
    /// dropped to preserve the operator's intent ("only these
    /// kernels"); a `None`-as-wildcard would silently dilute the
    /// filtered set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_version: Option<String>,
    /// ktstr project git commit carried from the source sidecar
    /// (`SidecarResult::project_commit`). Short hex with optional
    /// `-dirty` suffix (e.g. `"abcdef1"` or `"abcdef1-dirty"`).
    /// `None` when the sidecar writer could not probe a git repo
    /// at write time (cwd not inside a checkout, or
    /// `crate::test_support::sidecar::detect_project_commit`
    /// failed for any reason). Surfaced via the typed
    /// [`RowFilter::project_commits`] for narrowing — when the
    /// user passes `--project-commit abcdef1` (repeatable), rows
    /// with `None` are dropped to preserve the operator's intent
    /// ("only these commits"); a `None`-as-wildcard would silently
    /// dilute the filtered set, mirroring the [`RowFilter::kernels`]
    /// policy.
    ///
    /// Sourced from `SidecarResult::project_commit`; shortened to
    /// `commit` on the row because the project commit is the
    /// most-frequently-narrowed-on of the three commit dimensions
    /// on [`SidecarResult`](crate::test_support::SidecarResult). The other two commit fields —
    /// `SidecarResult::scheduler_commit` and
    /// `SidecarResult::kernel_commit` — get fully-qualified names
    /// here (`scheduler_commit` is reserved and not yet exposed,
    /// `kernel_commit` is the typed filter `RowFilter::kernel_commits`
    /// applies). The bare `commit` shortening is internal to
    /// `GauntletRow`; the CLI flag is the disambiguated
    /// `--project-commit` form so an operator never has to guess
    /// which "commit" dimension a bare `--commit` would have meant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// Kernel SOURCE TREE git commit carried from the source
    /// sidecar (`SidecarResult::kernel_commit`). Short hex with
    /// optional `-dirty` suffix (e.g. `"abcdef1"` or
    /// `"abcdef1-dirty"`). `None` when the sidecar writer could
    /// not probe a git repo for the kernel directory at write
    /// time (KTSTR_KERNEL points at a non-git path, the
    /// underlying source is `Tarball` / `Git` rather than
    /// `Local`, or
    /// `crate::test_support::sidecar::detect_kernel_commit`
    /// failed for any reason).
    ///
    /// Distinct from [`GauntletRow::commit`]: that field tracks
    /// the ktstr framework HEAD ("which version of the harness
    /// produced this sidecar?"); this field tracks the kernel
    /// tree HEAD ("which kernel commit did this run boot?"). Two
    /// runs with the same `commit` but different `kernel_commit`
    /// values are typical when the kernel under test is updated
    /// without re-checking out the harness; two runs with the
    /// same `kernel_commit` but different `commit` values are
    /// typical when the harness is bumped without rebuilding the
    /// kernel.
    ///
    /// Surfaced via the typed [`RowFilter::kernel_commits`] for
    /// narrowing — same opt-in policy as [`RowFilter::project_commits`]:
    /// rows with `None` never match a populated filter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_commit: Option<String>,
    /// Run-environment provenance tag carried from
    /// `SidecarResult::run_source` (`"local"` for developer runs,
    /// `"ci"` when `crate::test_support::sidecar::KTSTR_CI_ENV`
    /// was set at write time, `"archive"` when the consumer pulled
    /// the pool from a non-default `--dir`). `None` for sidecars
    /// produced before the field existed (pre-1.0 disposable
    /// schema; re-running the test regenerates the entry).
    /// Surfaced via the typed [`RowFilter::run_sources`] for
    /// narrowing — when the user passes `--run-source local`
    /// (repeatable), rows with `None` are dropped to preserve the
    /// operator's intent ("only these environments"); a
    /// `None`-as-wildcard would silently dilute the filtered set,
    /// mirroring the [`RowFilter::kernels`] /
    /// [`RowFilter::project_commits`] / [`RowFilter::kernel_commits`]
    /// policy.
    ///
    /// Field name `run_source` (renamed from `source`) disambiguates
    /// from [`crate::cache::KernelSource`] / `KernelMetadata.source`
    /// — those describe the kernel build's input (tarball / git /
    /// local), this describes the run-environment provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_source: Option<String>,
    /// Scheduler-resolution provenance carried from
    /// [`crate::test_support::SidecarResult::resolve_source`] —
    /// the snake_case resolution-path tag (`"auto_built"`,
    /// `"env_var"`, `"path"`, ...; historical sidecars may carry retired
    /// tags such as `"target_debug"`). `None` for sidecars produced
    /// before the field existed (pre-1.0 disposable schema) and for skip
    /// rows (no binary resolved). Surfaced via the typed
    /// [`RowFilter::resolve_sources`] (`--resolve-source`) for narrowing +
    /// pairing — same opt-in policy as `run_source` — and listed by
    /// `stats list-values`. Provenance, not identity: distinct
    /// from `scheduler` / `kernel_commit` — it records HOW the scheduler
    /// binary was found, so a reader can tell an auto-built-from-HEAD run
    /// from a possibly-stale `$PATH` binary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolve_source: Option<String>,
    /// True when the underlying [`crate::assert::AssertResult::is_pass`] returned
    /// true at sidecar emission time — a real pass with at least one
    /// observed outcome and no Fail/Inconclusive/Skip. Mutually
    /// exclusive with [`Self::skipped`] and [`Self::inconclusive`]:
    /// the three bits encode a strict 4-state verdict where exactly
    /// one of (Pass, Skip, Inconclusive, Fail) is set per row.
    pub passed: bool,
    /// True when the run was skipped (topology mismatch, missing
    /// resource). Mutually exclusive with [`Self::passed`] (Skip is
    /// not Pass; the empty / all-Skip outcomes vec maps to Skip,
    /// not Pass) and with [`Self::inconclusive`]. Lets stats tooling
    /// exclude these from pass counts so skipped runs don't inflate
    /// the apparent pass rate.
    pub skipped: bool,
    /// True when at least one assertion recorded
    /// [`crate::assert::Outcome::Inconclusive`] — the run ran but a
    /// zero-denominator ratio gate could not be evaluated. Mutually
    /// exclusive with [`Self::passed`] and [`Self::skipped`]; in the
    /// `Fail > Inconclusive > Pass > Skip` lattice, Inconclusive
    /// dominates Pass/Skip but loses to Fail, so a row with both
    /// Inconclusive and Fail outcomes records `inconclusive = false,
    /// passed = false` (Fail wins). Surfaced as a distinct bit so
    /// `is_fail` can exclude these from hard-fail counts and
    /// dashboards can triage zero-denominator runs separately from
    /// real regressions.
    #[serde(default)]
    pub inconclusive: bool,
    /// True when this row's source run was an `expect_err` /
    /// `expect_auto_repro` test whose actual scenario failure was
    /// inverted to a pass — its telemetry is failure-mode-dominated
    /// (short / stalled run), so [`compare_rows_by`] excludes it from
    /// the regression math even though [`passed`](Self::passed) is now
    /// `true`. Carried from `SidecarResult::expected_failure`;
    /// OR-folded across a cohort in `group_and_average_by`.
    #[serde(default)]
    pub expected_failure: bool,
    /// Number of monitor samples this run was averaged over —
    /// the natural per-RUN weight for `Gauge(Avg)` metrics when
    /// folded across multiple runs at cross-RUN comparison time
    /// (`group_and_average_by`). Sourced from
    /// `MonitorSummary::total_samples` at sidecar-write time;
    /// `0` when the monitor did not run for this scenario
    /// (host-only test, early VM failure). A `0` weight
    /// degenerates to unweighted mean per the fallback at
    /// `aggregate_samples`'s zero-total-weight branch.
    ///
    /// The field exists because the cross-RUN aggregator
    /// previously computed unweighted arithmetic mean for every
    /// metric — biased for `Gauge(Avg)` when runs in a cohort
    /// had different sample populations (a 5-sample run and a
    /// 50-sample run contributing equally to the cohort mean).
    /// Carrying the per-RUN count here lets the aggregator dispatch
    /// per-`MetricKind` weighted folds via the helper.
    #[serde(default)]
    pub run_sample_count: usize,
    /// Worst-case per-cgroup spread across the run. Four names
    /// describe the same quantity across the pipeline:
    /// - [`ScenarioStats::worst_spread`](crate::assert::ScenarioStats::worst_spread)
    ///   — the upstream source. `sidecar_to_row` reads it and
    ///   writes the value into this field via `finite_or_zero`.
    /// - `GauntletRow.spread` (this field) — the Rust-side
    ///   struct access path inside the comparison pipeline.
    /// - `MetricDef.name == "worst_spread"` — the [`METRICS`]
    ///   registry key, which is the domain-level name that appears
    ///   in sidecars, CI gates, and `cargo ktstr perf-delta`
    ///   output.
    /// - DataFrame column `"spread"` — the polars column name used
    ///   when the rows are projected into a DataFrame for group /
    ///   aggregate operations.
    ///
    /// The registry name is not renamed to match the field name
    /// because existing sidecars and CI regression gates reference
    /// `"worst_spread"` by string and a rename would silently
    /// invalidate them. The DataFrame column stays `"spread"` for
    /// terseness and to match the field; consumers that cross
    /// the registry / DataFrame boundary translate via
    /// [`MetricDef::read`] rather than by string comparison.
    pub spread: f64,
    /// Worst-case per-cgroup scheduling gap (ms). Surfaced in
    /// [`METRICS`] under registry name `worst_gap_ms`; the
    /// field / registry / DataFrame-column divergence is catalogued
    /// in the triples table on [`METRICS`].
    pub gap_ms: u64,
    /// Total CPU migrations across the run. Surfaced in [`METRICS`]
    /// under registry name `total_migrations`; see the triples
    /// table on [`METRICS`] for the rationale behind the
    /// field / registry / DataFrame-column divergence.
    pub migrations: u64,
    /// Worst-case per-cgroup migrations-per-iteration ratio.
    /// Surfaced in [`METRICS`] under registry name
    /// `worst_migration_ratio`; see the triples table on
    /// [`METRICS`] for the field / registry / DataFrame-column
    /// divergence.
    pub migration_ratio: f64,
    // Monitor fields (host-side telemetry from guest memory reads).
    /// Worst per-sample cgroup imbalance ratio. Surfaced in
    /// [`METRICS`] under registry name `max_imbalance_ratio`
    /// (DataFrame column `imbalance`); see the triples table on
    /// [`METRICS`] for the registry/field/column rationale.
    pub imbalance_ratio: f64,
    /// Worst observed DSQ queue depth. Registry and field names
    /// match (`max_dsq_depth`) but the DataFrame column is
    /// `dsq_depth`; see the triples table on [`METRICS`] for the
    /// column-level rename rationale.
    pub max_dsq_depth: u32,
    /// Stuck-sample count across the run (CPUs whose `rq_clock`
    /// failed to advance between consecutive samples). Distinct from
    /// the sched_ext watchdog stall (`SCX_EXIT_ERROR_STALL`):
    /// "stuck" tracks rq_clock not advancing on a CPU, while a
    /// watchdog stall describes a runnable task that hasn't been
    /// scheduled within the watchdog timeout. Registry and field
    /// names match (`stuck_count`) but the DataFrame column is
    /// `stuck`; see the triples table on [`METRICS`] for the
    /// column-level rename rationale.
    ///
    /// `f64`, not an integer: a per-run row carries an exact integer
    /// count, but the cross-run average fold (`super::group`'s
    /// `group_and_average_by`) stores the fractional mean here so the
    /// A/B comparison reads the true delta. Rounding the mean to an
    /// integer would let the up-to-1.0 rounding error defeat the
    /// `stuck_count` metric's `default_abs` of 1.0 and fabricate
    /// single-stall regressions from sub-integer differences (e.g. an
    /// A-side mean of 1.4 vs a B-side mean of 1.6 rounds to 1 vs 2).
    pub stuck_count: f64,
    /// Fallback-dispatch count across the run. Carried as-is from
    /// `MonitorSummary::event_deltas.total_fallback` — an integer
    /// event count, NOT a rate. Surfaced in [`METRICS`] under
    /// registry name `total_fallback` (DataFrame column `fallback`);
    /// see the triples table on [`METRICS`] for the registry / field /
    /// column rationale.
    pub fallback_count: i64,
    /// Keep-last dispatch count across the run. Carried as-is from
    /// `MonitorSummary::event_deltas.total_dispatch_keep_last` — an
    /// integer event count, NOT a rate. Surfaced in [`METRICS`] under
    /// registry name `total_keep_last` (DataFrame column `keep_last`);
    /// see the triples table on [`METRICS`] for the registry / field /
    /// column rationale.
    pub keep_last_count: i64,
    // Benchmarking fields. The wake-latency (p99 / median / CV) and
    // run-delay (mean / worst) roll-ups are NO LONGER typed fields: they are
    // `MetricKind::Distribution`, re-pooled into `ext_metrics` post-merge by
    // `crate::assert::populate_run_distribution_metrics`; `MetricDef::read`
    // surfaces them via the ext fallback (their accessors are `|_| None`).
    pub total_iterations: u64,
    // worst_wake_latency_tail_ratio is NO LONGER a typed field: it is
    // `MetricKind::WakeLatencyTailRatio`, re-selected into `ext_metrics`
    // post-merge by `crate::assert::populate_run_distribution_metrics` (max
    // over the per-cgroup p99/median ratios, floor-gated below
    // [`WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`]); `MetricDef::read` surfaces
    // it via the ext fallback (accessor `|_| None`).
    // worst_iterations_per_worker / worst_iterations_per_cpu_sec are NO
    // LONGER typed fields: they are `MetricKind::WorstLowest`, re-selected
    // into `ext_metrics` post-merge by
    // `crate::assert::populate_run_distribution_metrics` (lowest-wins over
    // the per-cgroup counters); `MetricDef::read` surfaces them via the ext
    // fallback. The `worst_` naming convention is documented on [`METRICS`].
    // NUMA fields.
    // Neither NUMA roll-up is a typed GauntletRow column any longer:
    // worst_page_locality is `MetricKind::WorstLowest` (NumaLocal/NumaTotal) and
    // worst_cross_node_migration_ratio is `MetricKind::WorstCrossNodeRatio`, both
    // ext-sourced (re-pooled from the per-phase NUMA carriers by
    // populate_run_distribution_metrics); `MetricDef::read` surfaces them via the
    // ext fallback, so no typed columns.
    /// Extensible metrics populated by scenarios and processed by the
    /// comparison pipeline. Keyed by metric name; looked up via
    /// [`metric_def`] when a matching entry exists in [`METRICS`].
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ext_metrics: BTreeMap<String, f64>,
    /// The subset of [`Self::ext_metrics`] keys that are Dynamic monotonic
    /// counters (the level-suffixed `lb_*`/`alb_*` schedstat deltas + any
    /// `ScalarCounter` watched bpf field). These keys are not in the static
    /// `METRICS` registry, so the cross-run aggregator (`fold_ext_metrics`)
    /// consults this set to SUM-fold them instead of averaging — matching the
    /// registered-Counter cross-run convention. Empty when the run has no
    /// monitor or no Dynamic counter keys.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub ext_counter_keys: BTreeSet<String>,
    /// Per-phase metric buckets carried verbatim from the source
    /// sidecar's [`crate::assert::ScenarioStats::phases`]. Each
    /// [`crate::assert::PhaseBucket`] surfaces the metric values
    /// reduced over one scenario phase (BASELINE at
    /// `step_index = 0`, Step ordinals at `step_index = 1..=N`
    /// per the 1-indexed phase convention) so the per-phase
    /// comparison renderer at [`compare_partitions_noise`] can pair
    /// matched phases across two sidecars by `step_index` and
    /// emit per-phase spread rows without re-deriving phase
    /// boundaries.
    ///
    /// Empty when the source sidecar had no phase data — single-
    /// phase scenarios that didn't drive an explicit Step vec,
    /// or legacy sidecars from before the phase-aware pipeline
    /// shipped. Per the pre-1.0 disposability contract, the
    /// expected response to a legacy sidecar is to re-run the
    /// test and regenerate, NOT to back-fill the field on read.
    /// `serde(default, skip_serializing_if = "Vec::is_empty")`
    /// keeps the serialized shape compact: a row with no phase
    /// data omits the field entirely on the wire rather than
    /// carrying an empty array.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub phases: Vec<crate::assert::PhaseBucket>,
    /// Per-test [`crate::test_support::PerfDeltaAssertion`] declarations (owned
    /// records) carried from the sidecar so the perf-delta noise compare can
    /// enforce the declared gate. Empty for rows with none; `serde(default)`
    /// tolerates legacy sidecars (pre-1.0 disposability). NOT a pairing/slicing
    /// dimension — per-test gate metadata only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub perf_delta_assertions: Vec<crate::test_support::PerfDeltaAssertionRecord>,
}

impl GauntletRow {
    /// Convenience accessor mirroring [`crate::assert::AssertResult::is_pass`]
    /// so the is_pass / is_fail vocabulary applies uniformly across both
    /// verdict surfaces. GauntletRow is the sidecar-wire shape; its
    /// `passed` bool is populated from `AssertResult::is_pass()` at
    /// sidecar emission time.
    ///
    /// Returns true only when the row reached a real Pass — neither
    /// skipped, inconclusive, nor failed. The triple-conjunct guard
    /// matches the strict 4-state mutex encoded with three stored
    /// bits `(passed, skipped, inconclusive)` (Fail is the all-false
    /// derived state, no dedicated bit), so a manually-constructed
    /// row that sets `passed = true, skipped = true` (which would
    /// violate the mutex) still reads as not-pass here.
    ///
    /// Part of the `is_pass` / `is_fail` / `is_inconclusive` /
    /// `is_skip` vocabulary uniform across the verdict surfaces:
    /// [`crate::assert::AssertResult::is_pass`] /
    /// [`crate::test_support::SidecarResult::is_pass`] /
    /// [`crate::assert::Outcome::is_pass`] / `MonitorVerdict::is_pass`
    /// (in the `monitor` module, which is `pub(crate)`) /
    /// `Verdict::is_pass` (re-exported at [`crate::assert::Verdict`])
    /// / `Self::is_pass`.
    pub fn is_pass(&self) -> bool {
        self.passed && !self.skipped && !self.inconclusive
    }
    /// Convenience accessor mirroring
    /// [`crate::assert::AssertResult::is_fail`]. True only when the
    /// row is a real failure — not a skip, not an inconclusive
    /// (zero-denominator) run. Excludes `inconclusive` so a stats
    /// gate that counts "real regressions" does not conflate
    /// inconclusive runs with hard failures.
    pub fn is_fail(&self) -> bool {
        !self.passed && !self.skipped && !self.inconclusive
    }
    /// Convenience accessor mirroring
    /// [`crate::assert::AssertResult::is_skip`].
    pub fn is_skip(&self) -> bool {
        self.skipped
    }
    /// Convenience accessor mirroring
    /// [`crate::assert::AssertResult::is_inconclusive`]. True when
    /// the row reflects a zero-denominator ratio gate that could
    /// not be evaluated.
    pub fn is_inconclusive(&self) -> bool {
        self.inconclusive
    }
}

/// Typed-field filter set for narrowing `GauntletRow` sets in the
/// `cargo ktstr perf-delta` pipeline. Every field is `None` /
/// empty by default; populated fields are AND-combined ACROSS
/// fields, with field-internal OR/AND semantics described per-field
/// below. Applied via `apply_row_filters` in `compare_partitions`
/// before the rows reach `compare_rows_by`.
///
/// Match semantics:
/// - `scheduler` / `topology` / `work_type` — STRICT EQUALITY against
///   the row's corresponding field. The sibling substring filter on
///   `compare_rows_by` (`-E`) stays as the only fuzzy-match knob;
///   typed fields are exact so a `--scheduler scx_rusty` filter does
///   NOT spuriously match `scx_rusty_alt`.
/// - `kernels` — repeatable, OR-combined: a row matches iff its
///   `kernel_version` equals ANY entry in `kernels`. Mirrors the
///   `--kernel` flag on `cargo ktstr test`/`coverage`/`llvm-cov`
///   so the same flag name carries the same multi-value semantic
///   across every subcommand.
/// - `project_commits` — repeatable, OR-combined: a row matches
///   iff its `commit` equals ANY entry in `project_commits`. Same
///   multi-value semantic as `kernels`, applied to the ktstr
///   project commit recorded by `detect_project_commit` at
///   sidecar-write time. Surfaced as the `--project-commit` CLI
///   flag.
/// - `kernel_commits` — repeatable, OR-combined: a row matches
///   iff its `kernel_commit` equals ANY entry in `kernel_commits`.
///   Same multi-value semantic as `project_commits`, applied to
///   the kernel source-tree commit recorded by
///   `crate::test_support::sidecar::detect_kernel_commit` at
///   sidecar-write time. Filters on the kernel HEAD, NOT on the
///   kernel release version (`kernels` is the version filter).
/// - `run_sources` — repeatable, OR-combined: a row matches iff
///   its `run_source` equals ANY entry in `run_sources`. Same
///   multi-value semantic as `kernels` / `project_commits` /
///   `kernel_commits`, applied to the run-environment provenance
///   tag (`"local"`, `"ci"`, `"archive"`) recorded by
///   `crate::test_support::sidecar::detect_run_source` at
///   sidecar-write time, or rewritten to `"archive"` at load
///   time when the consumer pulled the pool from a non-default
///   `--dir`. Surfaced as the `--run-source` CLI flag.
/// - A `kernels`-populated filter against a row whose
///   `kernel_version` is `None` ALWAYS fails (no wildcard semantic)
///   — the operator wrote specific versions and a `None`-row would
///   silently dilute the set. The same opt-in policy applies to
///   `project_commits` against rows with `commit == None`, to
///   `kernel_commits` against rows with `kernel_commit == None`,
///   and to `run_sources` against rows with `run_source == None`.
///
/// Empty `RowFilter` (every field `None`/empty) is the no-op default
/// and matches every row. Use [`RowFilter::default()`] to build it.
#[derive(Clone, Debug, Default)]
pub struct RowFilter {
    /// Repeatable kernel-version filter, OR-combined: a row matches
    /// iff its `GauntletRow::kernel_version` equals ANY entry. Empty
    /// vec disables the filter ("do not filter on kernel"). A row
    /// whose `kernel_version` is itself `None` never matches a
    /// non-empty filter.
    pub kernels: Vec<String>,
    /// Repeatable project-commit filter, OR-combined: a row matches
    /// iff its `GauntletRow::commit` equals ANY entry. Empty vec
    /// disables the filter ("do not filter on commit"). A row whose
    /// `commit` is itself `None` never matches a non-empty filter
    /// — same opt-in semantic as `kernels`.
    ///
    /// Field name `project_commits` (renamed from `commits`)
    /// disambiguates from the sibling `kernel_commits` field — both
    /// describe commit dimensions, so the prefix makes "which
    /// repository's commit?" obvious at every call site.
    pub project_commits: Vec<String>,
    /// Repeatable kernel-source-commit filter, OR-combined: a row
    /// matches iff its `GauntletRow::kernel_commit` equals ANY
    /// entry. Empty vec disables the filter ("do not filter on
    /// kernel commit"). A row whose `kernel_commit` is itself
    /// `None` never matches a non-empty filter — same opt-in
    /// semantic as `project_commits`.
    ///
    /// Distinct from `project_commits` (the ktstr framework commit)
    /// and from `kernels` (the kernel release version): two runs
    /// with the same `kernel_version` but different `kernel_commit`
    /// values represent the same release rebuilt from different
    /// trees (e.g. WIP patches on top, a different remote ref).
    pub kernel_commits: Vec<String>,
    /// Repeatable run-environment-source filter, OR-combined: a row
    /// matches iff its `GauntletRow::run_source` equals ANY entry.
    /// Empty vec disables the filter ("do not filter on
    /// run_source"). A row whose `run_source` is itself `None`
    /// (sidecar pre-dates the field) never matches a non-empty
    /// filter — same opt-in semantic as `kernels` /
    /// `project_commits` / `kernel_commits`.
    /// Typical values: `"local"`, `"ci"`, `"archive"`. The schema
    /// is open: any string is acceptable so a future producer can
    /// introduce a new tag without a version bump.
    ///
    /// Field name `run_sources` (renamed from `sources`)
    /// disambiguates from `KernelMetadata.source` /
    /// [`crate::cache::KernelSource`] — those describe the kernel
    /// build's input, this describes the run-environment provenance.
    pub run_sources: Vec<String>,
    /// Repeatable scheduler-resolution-source filter, OR-combined: a
    /// row matches iff its `GauntletRow::resolve_source` equals ANY
    /// entry. Empty vec disables the filter. A row whose
    /// `resolve_source` is `None` (sidecar pre-dates the field, or a
    /// skip resolved no binary) never matches a non-empty filter —
    /// same opt-in semantic as `run_sources`. Values are the
    /// [`crate::test_support::ResolveSource::as_str`] tags
    /// (`"auto_built"`, `"env_var"`, `"path"`, ...; historical sidecars
    /// may also carry retired tags such as `"target_debug"`). Distinct from
    /// `run_sources` (the run ENVIRONMENT): this is HOW the scheduler
    /// binary was found. Backs the [`Dimension::ResolveSource`] filter +
    /// pairing dim (`--resolve-source`).
    pub resolve_sources: Vec<String>,
    /// Repeatable cpu-budget filter, OR-combined: a row matches iff its
    /// `GauntletRow::cpu_budget` (the effective host-CPU budget, as a
    /// decimal string) equals ANY entry. Empty vec disables the filter.
    /// Rows with `cpu_budget == None` (skips) are dropped when this filter
    /// is non-empty, mirroring `kernels` / `run_sources`. Backs the
    /// [`Dimension::CpuBudget`] filter + pairing dim (`--cpu-budget`).
    pub cpu_budgets: Vec<String>,
    /// Repeatable scheduler-name filter, OR-combined: a row matches
    /// iff its `GauntletRow::scheduler` equals ANY entry. Empty vec
    /// disables the filter ("do not filter on scheduler"). Strict
    /// equality on each entry — the substring `-E` filter is the
    /// only fuzzy-match knob; typed flags exact-match. Mirrors the
    /// shape of `kernels` / `project_commits` / `kernel_commits` /
    /// `run_sources` so every typed dimension supports the same
    /// repeatable OR-combined idiom.
    pub schedulers: Vec<String>,
    /// Repeatable topology filter, OR-combined: a row matches iff
    /// its `GauntletRow::topology` equals ANY entry. The filter
    /// values are the rendered form (e.g. `"1n2l4c2t"`) that
    /// `Topology::Display` emits and `cargo ktstr stats list`
    /// shows. Empty vec disables the filter.
    pub topologies: Vec<String>,
    /// Repeatable work-type filter, OR-combined: a row matches iff
    /// its `GauntletRow::work_type` equals ANY entry. Valid names
    /// are the PascalCase variants of `WorkType::ALL_NAMES`. Empty
    /// vec disables the filter.
    pub work_types: Vec<String>,
}

impl RowFilter {
    /// Returns true when every populated filter field matches the
    /// row. The empty `RowFilter` (default) returns true for every
    /// row — it's the identity filter.
    pub fn matches(&self, row: &GauntletRow) -> bool {
        if !self.kernels.is_empty() {
            // OR-combined: the row matches iff its kernel version
            // matches ANY listed kernel. A row with `None`
            // kernel_version never satisfies a non-empty filter —
            // same opt-in semantic the original `Option<String>`
            // field carried.
            //
            // Match shape: a filter value with two dot-separated
            // digit segments (e.g. `6.12`) is a major.minor PREFIX —
            // the row matches if its `kernel_version` equals
            // `6.12` exactly, starts with `6.12.` (patch releases
            // including the `6.12.0-rcN+` kernel banner shape),
            // or starts with `6.12-` (the no-patch `6.12-rcN`
            // shape `kernel_path::KernelId::Version` admits). A
            // filter with three or more segments (e.g. `6.14.2`,
            // `6.15-rc3`) is strict equality. The two-segment
            // cutoff matches the shape of `MAJOR.MINOR` versus
            // `MAJOR.MINOR.PATCH` / `MAJOR.MINOR-rcN` — there is
            // no shorter form on the sidecar producer side worth
            // treating as a prefix (`6` alone would match every
            // 6.x release, which is a less useful cohort than the
            // per-stable-series narrowing the operator usually
            // wants).
            let row_kernel = row.kernel_version.as_deref();
            let any = self.kernels.iter().any(|want| match row_kernel {
                Some(rk) => kernel_filter_matches(want, rk),
                None => false,
            });
            if !any {
                return false;
            }
        }
        if !self.project_commits.is_empty() {
            // OR-combined match against `GauntletRow::commit`,
            // mirroring the `kernels` policy: a row whose `commit`
            // is `None` (the sidecar writer's gix probe failed or
            // cwd was outside any git repo) never matches a
            // populated filter, so a `--project-commit` argument is opt-in
            // to "only rows with this commit" rather than a wildcard.
            let row_commit = row.commit.as_deref();
            let any = self
                .project_commits
                .iter()
                .any(|want| row_commit == Some(want.as_str()));
            if !any {
                return false;
            }
        }
        if !self.kernel_commits.is_empty() {
            // OR-combined match against `GauntletRow::kernel_commit`,
            // mirroring the `project_commits` policy: a row whose
            // `kernel_commit` is `None` (the sidecar writer's
            // `detect_kernel_commit` probe failed, or `KTSTR_KERNEL`
            // pointed at a non-git source) never matches a populated
            // filter — same opt-in semantic as `--project-commit` /
            // `--kernel`.
            let row_kc = row.kernel_commit.as_deref();
            let any = self
                .kernel_commits
                .iter()
                .any(|want| row_kc == Some(want.as_str()));
            if !any {
                return false;
            }
        }
        if !self.run_sources.is_empty() {
            // OR-combined match against `GauntletRow::run_source`,
            // mirroring the `kernels` / `project_commits` /
            // `kernel_commits` opt-in policy: a row whose
            // `run_source` is `None` (sidecar pre-dates the field)
            // never matches a populated filter, so a `--run-source`
            // argument demands a tagged row rather than acting as a
            // wildcard.
            let row_run_source = row.run_source.as_deref();
            let any = self
                .run_sources
                .iter()
                .any(|want| row_run_source == Some(want.as_str()));
            if !any {
                return false;
            }
        }
        if !self.resolve_sources.is_empty() {
            // OR-combined match against `GauntletRow::resolve_source`,
            // mirroring the `run_sources` opt-in policy: a row whose
            // `resolve_source` is `None` (sidecar pre-dates the field,
            // or a skip resolved no binary) never matches a populated
            // filter, so a `--resolve-source` argument demands a tagged
            // row rather than acting as a wildcard.
            let row_resolve_source = row.resolve_source.as_deref();
            let any = self
                .resolve_sources
                .iter()
                .any(|want| row_resolve_source == Some(want.as_str()));
            if !any {
                return false;
            }
        }
        if !self.cpu_budgets.is_empty() {
            // OR-combined match against `GauntletRow::cpu_budget` rendered
            // as a decimal string. A row with `cpu_budget == None` (skip)
            // never matches a populated filter — same opt-in policy as
            // `run_sources` / `kernels`.
            let row_budget = row.cpu_budget.map(|n| n.to_string());
            let any = self
                .cpu_budgets
                .iter()
                .any(|want| row_budget.as_deref() == Some(want.as_str()));
            if !any {
                return false;
            }
        }
        if !self.schedulers.is_empty() {
            // OR-combined match against `GauntletRow::scheduler`
            // (a `String`, never `None`). Strict equality on each
            // entry — same shape as the other repeatable typed
            // filters above.
            let any = self.schedulers.contains(&row.scheduler);
            if !any {
                return false;
            }
        }
        if !self.topologies.is_empty() {
            // OR-combined match against `GauntletRow::topology`.
            let any = self.topologies.contains(&row.topology);
            if !any {
                return false;
            }
        }
        if !self.work_types.is_empty() {
            // OR-combined match against `GauntletRow::work_type`.
            let any = self.work_types.contains(&row.work_type);
            if !any {
                return false;
            }
        }
        true
    }
}

/// Drop rows from `rows` that do not match every populated filter
/// field on `filter`. Returns the surviving rows in their original
/// order. The caller is responsible for any further dedup or
/// aggregation; this helper preserves duplicates as written.
///
/// Used by [`compare_partitions`] before the surviving rows reach
/// [`compare_rows_by`], so the substring-`-E` filter and the typed
/// filters compose: typed narrows happen first, substring runs over
/// the surviving set.
pub fn apply_row_filters(rows: &[GauntletRow], filter: &RowFilter) -> Vec<GauntletRow> {
    rows.iter().filter(|r| filter.matches(r)).cloned().collect()
}

/// Match a single `--kernel` filter value against a row's
/// `kernel_version`. Major.minor (two-segment) filter values match
/// any patch release in that series via prefix; longer filter
/// values use strict equality.
///
/// `want` is the user-supplied filter value (e.g. `6.12`,
/// `6.14.2`, `6.15-rc3`). `row_kernel` is the sidecar-recorded
/// kernel version (e.g. `6.12.5`). The two-segment cutoff matches
/// the natural shape of `MAJOR.MINOR` versus
/// `MAJOR.MINOR.PATCH` / `MAJOR.MINOR-rcN` — `6.12.` is a
/// stable-series prefix; `6.14.2` is one specific release.
///
/// Examples:
/// - `kernel_filter_matches("6.12", "6.12.5")` → true (prefix)
/// - `kernel_filter_matches("6.12", "6.12")` → true (exact equal)
/// - `kernel_filter_matches("6.12", "6.13.0")` → false
/// - `kernel_filter_matches("6.14", "6.14-rc3")` → true (prefix
///   admits the `-rcN` pre-release of the same series; per
///   `kernel_path::decompose_version_for_compare`, `6.14-rc3`
///   shares the `(major=6, minor=14, patch=0)` tuple with the
///   `6.14` release, and the operator filtering on the series
///   wants both)
/// - `kernel_filter_matches("6.14", "6.14.0-rc3+")` → true
///   (kernel banner shape — patch=0 plus `-rcN` plus `EXTRAVERSION`)
/// - `kernel_filter_matches("6.14.2", "6.14.2")` → true
/// - `kernel_filter_matches("6.14.2", "6.14.20")` → false
///   (strict equality on three-segment filter — without the
///   strict path, `6.14.2` would also match `6.14.20`,
///   `6.14.21`, ..., which is not what the operator asked for)
pub(crate) fn kernel_filter_matches(want: &str, row_kernel: &str) -> bool {
    if is_major_minor_prefix(want) {
        // Three accepted shapes for a major.minor (`MAJOR.MINOR`)
        // prefix filter, all designed so the prefix is bounded by
        // a non-digit separator that disambiguates the series:
        //
        //   1. Exact equal: `row_kernel == "6.14"`. The row's
        //      recorded version IS the major.minor string itself
        //      (no patch, no rc).
        //   2. Trailing-dot prefix: `row_kernel.starts_with("6.14.")`.
        //      Covers patch releases (`6.14.0`, `6.14.5`) and
        //      kernel banner shapes (`6.14.0-rc3+`).
        //   3. Trailing-dash prefix: `row_kernel.starts_with("6.14-")`.
        //      Covers the no-patch pre-release shape (`6.14-rc3`).
        //      Per `kernel_path` (KernelId::Version doc), this is a
        //      valid emitted shape; per
        //      `decompose_version_for_compare` it shares the
        //      `(major, minor, patch=0)` triple with the `6.14`
        //      release and the operator filtering on the series
        //      wants both.
        //
        // The non-digit separator after `want` (`.` or `-`)
        // prevents `6.1` from spuriously matching `6.10.0` or
        // `6.10-rc3` — both fail because the next character after
        // `6.1` is `0`, which is neither separator. The `6.140`
        // case is also rejected for the same reason.
        row_kernel == want
            || row_kernel.starts_with(&format!("{want}."))
            || row_kernel.starts_with(&format!("{want}-"))
    } else {
        row_kernel == want
    }
}

/// Whether a filter value looks like a major.minor PREFIX. Two
/// non-empty dot-separated digit segments and nothing else
/// (no `-rcN`, no third dot). Conservative: anything outside the
/// `MAJOR.MINOR` shape falls through to strict equality so a typo
/// like `6.14.2.` or `6.14-something` does not silently turn into
/// a wildcard.
fn is_major_minor_prefix(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    parts.len() == 2
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
}
