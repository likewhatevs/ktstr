use super::*;

/// Aggregated statistics across all cgroups in a scenario.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize, crate::Claim)]
pub struct ScenarioStats {
    /// Per-cgroup stats, one entry per cgroup.
    pub cgroups: Vec<CgroupStats>,
    /// Sum of workers across all cgroups.
    pub total_workers: usize,
    /// Sum of per-cgroup distinct CPU counts (not deduplicated across cgroups).
    pub total_cpus: usize,
    /// Sum of migration counts across all cgroups.
    pub total_migrations: u64,
    /// Worst spread across any cgroup (highest).
    pub worst_spread: f64,
    /// Worst gap across any cgroup (highest, ms). Paired with
    /// `worst_gap_cpu` — both come from the same cgroup.
    pub worst_gap_ms: u64,
    /// CPU where the worst gap occurred across all cgroups. Paired
    /// with `worst_gap_ms` — both come from the same cgroup.
    pub worst_gap_cpu: usize,
    /// Worst migration ratio across any cgroup (highest).
    pub worst_migration_ratio: f64,
    /// Sum of iteration counts across all cgroups.
    pub total_iterations: u64,
    /// Worst page locality fraction across cgroups (lowest non-zero).
    pub worst_page_locality: f64,
    /// Worst cross-node migration ratio across cgroups (highest).
    pub worst_cross_node_migration_ratio: f64,
    // worst_wake_latency_tail_ratio is NO LONGER a typed field: it is
    // `crate::stats::MetricKind::WakeLatencyTailRatio`, re-selected into
    // `ext_metrics` post-merge by `populate_run_distribution_metrics` (max
    // over the per-cgroup `CgroupStats::wake_latency_tail_ratio` values,
    // floor-gated below WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS); `MetricDef::read`
    // surfaces it via the ext fallback.
    /// Extensible metrics for the generic comparison pipeline.
    /// Populated from per-cgroup ext_metrics (worst value across cgroups).
    pub ext_metrics: BTreeMap<String, f64>,
    /// Per-phase metric buckets in step-index order. A scenario
    /// with N Steps populates `N + 1` entries: phase 0 is the
    /// BASELINE settle window before Step 0 fires, phases
    /// 1..=N align with Step 0..Step N-1 in scenario order
    /// (1-indexed Steps so the BASELINE encoding doesn't collide
    /// with first-Step's index).
    ///
    /// Empty when the scenario produced no periodic captures
    /// (Default::default() yields `vec![]`). The existing
    /// flat-bucket scalars on this struct are independent of the
    /// per-phase view — they remain the "all phases merged"
    /// reading, unchanged in semantics by the introduction of
    /// `phases`.
    ///
    /// **Auto-populated by the framework**: scenarios that fire
    /// periodic captures (via
    /// [`crate::test_support::KtstrTestEntry::num_snapshots`] or
    /// [`crate::scenario::ops::Op::CaptureSnapshot`]) have this
    /// field populated automatically inside
    /// `crate::test_support::eval`'s `evaluate_vm_result` —
    /// test code never needs to call
    /// [`crate::assert::build_phase_buckets`] manually. The auto-
    /// populate path drains the snapshot bridge from the
    /// [`crate::vmm::VmResult`] returned by the framework and folds
    /// the per-sample readings through
    /// `crate::stats::aggregate_samples_for_phase` per metric.
    /// Single-phase scenarios that fire no captures leave this
    /// `vec![]`; the flat-bucket scalars on this struct cover the
    /// single-phase case.
    ///
    /// See [`PhaseBucket`] for the per-phase shape.
    #[serde(default)]
    pub phases: Vec<PhaseBucket>,
}

impl ScenarioStats {
    /// Look up the phase bucket for a phase index.
    ///
    /// **Heads up:** `step_index = 0` returns the pre-Step BASELINE
    /// settle window, NOT the first Step. The first Step the
    /// scenario author wrote lives at `step_index = 1` per the
    /// 1-indexed Step encoding. To look up the test author's "Step
    /// N", pass `N + 1` — or use [`Self::step`] for an accessor
    /// that takes the 0-indexed scenario Step number directly.
    ///
    /// Returns `None` when no bucket with that index exists
    /// (single-phase scenario, scenario didn't reach the step, or
    /// `step_index` past the last phase).
    pub fn phase(&self, step_index: u16) -> Option<&PhaseBucket> {
        self.phases.iter().find(|p| p.step_index == step_index)
    }

    /// Look up the phase bucket for a 0-indexed scenario Step
    /// number — the natural index the test author used when
    /// constructing `vec![step_a, step_b, step_c]` (Step A is
    /// `scenario_step_idx = 0`, Step B is `1`, etc.).
    ///
    /// Internally translates to `step_index = scenario_step_idx + 1`
    /// per the 1-indexed phase encoding (phase 0 is reserved for
    /// BASELINE). Use this for the common "I want metrics for the
    /// N-th Step I wrote" case; use [`Self::phase`] when you need
    /// to address BASELINE explicitly or work in phase-index space.
    ///
    /// Returns `None` when the scenario didn't reach that Step or
    /// `phases` is empty.
    pub fn step(&self, scenario_step_idx: u16) -> Option<&PhaseBucket> {
        scenario_step_idx
            .checked_add(1)
            .and_then(|phase_idx| self.phase(phase_idx))
    }

    /// Shortcut: look up a single metric value in a specific
    /// phase by phase-index. Returns `None` when:
    /// (a) the phase is absent (no bucket with `step_index` in
    ///     [`Self::phases`]),
    /// (b) the phase exists but had no finite samples for that
    ///     metric, OR
    /// (c) `metric` is not a registered metric name (typo case —
    ///     [`Self::is_known_metric`] surfaces it).
    ///
    /// Sentinel-free: `Some(0.0)` means the reducer produced a
    /// real zero from finite samples, NOT "missing data". See
    /// [`PhaseBucket::metrics`] for the registry source. When
    /// debugging an unexpected `None`, gate the lookup on
    /// [`Self::is_known_metric`] to distinguish typos from absent
    /// data.
    ///
    /// **Heads up:** same 1-indexed Step encoding as
    /// [`Self::phase`] — `step_index = 0` is BASELINE, not the
    /// first Step. Use [`Self::step_metric`] for the 0-indexed
    /// scenario-Step lookup.
    pub fn phase_metric(&self, step_index: u16, metric: &str) -> Option<f64> {
        self.phase(step_index).and_then(|p| p.get(metric))
    }

    /// Cross-cgroup balance: the ratio of the busiest cell's per-worker
    /// throughput to the quietest's — `max / min` over each cgroup's
    /// [`CgroupStats::iterations_per_worker`]. The bread-and-butter
    /// scheduler-fairness assertion (every balance test hand-rolls this
    /// `max/min` over `self.cgroups` today).
    ///
    /// No-worker cgroups (`iterations_per_worker() == None`) are SKIPPED: a
    /// 0-worker cell is a config condition, not a balance signal. Returns
    /// `None` when fewer than two cgroups have workers (a ratio needs two);
    /// check the cgroup count separately if every declared cell must have
    /// workers. A cell that ran workers but completed zero iterations
    /// (measured `Some(0.0)`) drives the ratio to `f64::INFINITY` so
    /// starvation SURFACES rather than vanishing — matching the
    /// `None`-vs-`Some(0.0)` discipline of
    /// [`CgroupStats::iterations_per_worker`]. For an explicit starvation
    /// gate, check `min > 0` over the same cgroups separately.
    ///
    /// Whole-run aggregate: this reads `self.cgroups`, which sums over all
    /// phases. For a single phase's per-cgroup balance in a multi-phase
    /// scenario, read each cgroup's per-phase throughput (`total_iterations`
    /// / `num_workers`) from [`crate::vmm::VmResult::phase_cgroup`] — the
    /// per-phase per-cgroup carriers now landed in
    /// [`PhaseBucket::per_cgroup`].
    pub fn cgroup_balance_ratio(&self) -> Option<f64> {
        let mut min = f64::INFINITY;
        let mut max = 0.0_f64;
        let mut n = 0usize;
        for cg in &self.cgroups {
            if let Some(rate) = cg.iterations_per_worker() {
                min = min.min(rate);
                max = max.max(rate);
                n += 1;
            }
        }
        if n < 2 {
            return None;
        }
        if min == 0.0 {
            // A with-worker cell did zero work: starvation. Surface it as an
            // infinite ratio rather than a NaN (0/0) or a hidden None.
            return Some(f64::INFINITY);
        }
        Some(max / min)
    }

    /// Shortcut: look up a single metric value in a 0-indexed
    /// scenario Step. Sibling of [`Self::step`]. See [`Self::phase_metric`]
    /// for the None-cause taxonomy and
    /// [`Self::is_known_metric`] for typo-debugging.
    pub fn step_metric(&self, scenario_step_idx: u16, metric: &str) -> Option<f64> {
        self.step(scenario_step_idx).and_then(|p| p.get(metric))
    }

    /// True when `name` matches a registered metric (see
    /// [`PhaseBucket::metrics`] for the registry source). Use to
    /// disambiguate the typo None-cause from [`Self::phase_metric`]
    /// / [`Self::step_metric`]: if the lookup returns `None` and
    /// `is_known_metric(name) == false`, the metric name is a typo
    /// (caller mistake), not missing data (legitimately-absent
    /// samples).
    pub fn is_known_metric(name: &str) -> bool {
        crate::stats::METRICS.iter().any(|m| m.name == name)
    }

    /// Iterate the canonical metric names a test author may pass
    /// to [`Self::phase_metric`] / [`Self::step_metric`]. Sourced
    /// from the registry referenced by [`PhaseBucket::metrics`].
    ///
    /// Sample usage for an A/B scheduler-swap assertion that
    /// compares every registered metric across two scenario Steps:
    /// ```ignore
    /// for metric in ScenarioStats::known_metrics() {
    ///     let baseline = r.stats.step_metric(0, metric);
    ///     let after_swap = r.stats.step_metric(2, metric);
    ///     // ... compare per metric ...
    /// }
    /// ```
    ///
    /// Heads up: not every known name is phase-readable. The
    /// `MetricKind::Distribution` / `MetricKind::WorstLowest` family
    /// (`worst_*_wake_latency_*` / `worst_*_run_delay_*` /
    /// `worst_iterations_per_*`) is RUN-LEVEL only — it never appears
    /// in [`PhaseBucket::metrics`], so [`Self::phase_metric`] /
    /// [`Self::step_metric`] return `None` for those names. Read them
    /// via [`Self::run_metric`] instead. Iterating `known_metrics()`
    /// through `step_metric` (as above) silently skips that family.
    pub fn known_metrics() -> impl Iterator<Item = &'static str> {
        crate::stats::METRICS.iter().map(|m| m.name)
    }

    /// True iff the scenario produced at least one Step-phase
    /// bucket (any phase with `step_index >= 1`). False when
    /// `phases` is empty OR contains only `BASELINE` (the
    /// pre-first-Step settle window).
    ///
    /// Use this to fail a phase-aware assertion BEFORE calling
    /// [`Self::step`] / [`Self::step_metric`] on a scenario that
    /// silently never advanced past BASELINE: a test that declared
    /// no `Step`s, OR a scenario that bailed in setup before any
    /// `Step` ran, would otherwise see [`Self::step`] return
    /// `None` for every index and the test would either panic on
    /// `.expect(...)` or pass vacuously.
    ///
    /// ```ignore
    /// anyhow::ensure!(
    ///     r.stats.has_steps(),
    ///     "scenario produced no Step-phase buckets — \
    ///      declare a Step or use Self::phase(0) for BASELINE",
    /// );
    /// let throughput = r.stats.step_metric(0, "throughput");
    /// ```
    pub fn has_steps(&self) -> bool {
        self.phases.iter().any(|p| p.step_index >= 1)
    }

    /// Run-level value for a metric by registry name, for the
    /// ext-sourced metric family that carries no typed
    /// `ScenarioStats` field.
    ///
    /// Resolves [`Self::ext_metrics`] — the run-level map the
    /// framework fills post-merge with every metric whose value has no
    /// typed struct field: the pooled wake-latency / run-delay
    /// distributions and worst-cgroup iteration efficiencies
    /// (the `MetricKind::Distribution` / `MetricKind::WorstLowest`
    /// registry kinds — `worst_p99_wake_latency_us`, `worst_run_delay_us`,
    /// `worst_iterations_per_cpu_sec`, …), the derived rates
    /// (`iteration_rate`, and the pooled `iterations_per_cpu_sec` —
    /// distinct from the `worst_iterations_per_cpu_sec` selector above),
    /// the per-thread-group `system_time_ns` / `user_time_ns`, and
    /// `avg_imbalance_ratio` / `avg_dsq_depth`. This is the
    /// run-level analogue of [`Self::phase_metric`] for that family:
    /// code holding the run's [`AssertResult`] reads
    /// `r.stats.run_metric("worst_run_delay_us")` instead of reaching
    /// into the raw `ext_metrics` map by string key (`ScenarioStats` is
    /// the [`AssertResult::stats`] field — the value a test body, or a
    /// callback that builds an `AssertResult` via `collect_all` /
    /// `execute_scenario`, holds). A `post_vm` callback instead receives
    /// a `VmResult`, which has NO `stats` field and no run-level
    /// Distribution surface — compare those cross-run via `cargo ktstr
    /// stats compare`.
    ///
    /// The ext family is populated only by the `#[ktstr_test]` eval
    /// flow's post-merge producer
    /// ([`populate_run_distribution_metrics`]). An `AssertResult` built
    /// by a DIRECT host assertion (`assert_not_starved` /
    /// `AssertPlan::assert_cgroup`, which never run that producer)
    /// carries the per-cgroup values on [`Self::cgroups`] but none of
    /// these run-level roll-ups, so `run_metric` returns `None` for them
    /// on that path — read the per-cgroup `CgroupStats` field directly
    /// (e.g. `r.stats.cgroups[i].p99_wake_latency_us`) there.
    ///
    /// Sentinel-free, matching [`Self::phase_metric`]: `None` means
    /// the metric is absent from this run (no contributing cgroup or
    /// carrier, or a name not present in the map); `Some(0.0)` is a
    /// real measured zero. Gate on [`Self::is_known_metric`] to tell a
    /// typo from genuinely-absent data. (The map also carries any
    /// user-defined extensible-metric keys, plus the framework-internal
    /// Rate-component Counters — `total_phase_iterations` /
    /// `total_phase_duration_sec` / `total_iterations_pooled` /
    /// `total_cpu_time_sec`, the numerator/denominator plumbing behind
    /// `iteration_rate` / `iterations_per_cpu_sec` — all of which resolve
    /// here too; prefer the derived rate over its raw components.)
    ///
    /// NOT resolved here (these are not in `ext_metrics`):
    /// - the typed cross-cgroup fields — read them via their named
    ///   struct fields ([`Self::worst_spread`],
    ///   [`Self::worst_migration_ratio`], [`Self::worst_gap_ms`],
    ///   [`Self::total_migrations`], [`Self::total_iterations`],
    ///   [`Self::worst_page_locality`],
    ///   [`Self::worst_cross_node_migration_ratio`]). They are
    ///   `0.0`-sentinel f64 (no not-measured state), so exposing them
    ///   here would split this method's sentinel-free contract.
    ///   (`worst_wake_latency_tail_ratio` is NO LONGER in this group —
    ///   it is now the `WakeLatencyTailRatio` ext key and IS resolved
    ///   here via the ext lookup.)
    /// - the monitor-sourced run-level metrics (`max_imbalance_ratio`,
    ///   `max_dsq_depth`, `stuck_count`, `total_fallback`,
    ///   `total_keep_last`), which `ScenarioStats` does not hold
    ///   run-level — read those per-phase via [`Self::phase_metric`] /
    ///   [`Self::step_metric`].
    ///
    /// So this does NOT cover the full registry: iterating
    /// [`Self::known_metrics`] through it yields `None` for those typed
    /// and monitor names. There is no single run-level by-name accessor
    /// over the whole registry (the typed fields live on `ScenarioStats`
    /// directly, the monitor metrics only per-phase); this resolves the
    /// ext-sourced family, the one with no typed field.
    pub fn run_metric(&self, name: &str) -> Option<f64> {
        self.ext_metrics.get(name).copied()
    }
}

/// Registry metric names that already have a typed `GauntletRow` field — the
/// typed accessor populates them at `sidecar_to_row` time and
/// `MetricDef::read` prefers the accessor over `ext_metrics`, so writing the
/// same key into `ext_metrics` would create unread sidecar bloat AND
/// double-source the run-level value. For `stuck_count` the typed whole-run
/// count (`MonitorSummary::stuck_count`, windowed over the full sample
/// stream) is authoritative; the per-phase fold sum shares the
/// `is_cpu_stuck` predicate but is a lower-or-equal (`<=`),
/// partition-dependent quantity (it drops cross-boundary + out-of-phase
/// windows, so it falls strictly below once any of those is stuck), so
/// injecting the ext copy would shadow the authoritative typed value with
/// a redundant — and, once a dropped window is stuck, divergent —
/// number. Both run-level ext-metrics populators consult this — the
/// SampleSeries
/// path ([`populate_run_ext_metrics`]) and the phase-fold path
/// ([`populate_run_ext_metrics_from_phases`]) — so only ext-metrics-only
/// registry entries are written and a typed-backed metric's run-level value
/// always comes from its accessor. `max_imbalance_ratio` is included because
/// its accessor reads the typed `GauntletRow.imbalance_ratio` (whole-run
/// MonitorSummary); its per-phase monitor fold feeds rendering only.
const TYPED_FIELD_NAMES: &[&str] = &[
    "max_dsq_depth",
    "max_imbalance_ratio",
    "total_fallback",
    "total_keep_last",
    "stuck_count",
    "total_iterations",
    "total_migrations",
];

/// Sibling of [`populate_run_ext_metrics`] that mines per-phase
/// metrics back into the run-level `ext_metrics` map. Closes the
/// gap for registered metrics whose values live in
/// `PhaseBucket.metrics` but never reach `ext_metrics` via the
/// SampleSeries path (their `read_sample` returns `None`):
/// `avg_imbalance_ratio` (sourced from MonitorSample windowing
/// inside [`build_phase_buckets`]), `iteration_rate` (sourced from
/// stimulus event totals inside [`build_phase_buckets_with_stimulus`]),
/// and `system_time_ns` / `user_time_ns` (per-thread-group CPU-time
/// deltas injected by `phase_group_cpu_delta` inside
/// `buckets_from_grouped`). The fold is generic over every key
/// present on any phase, so it carries any such phase-only metric (the
/// ext-metrics-only set whose `read_sample` returns `None`). Keys with a
/// typed `GauntletRow` field (`TYPED_FIELD_NAMES`) are SKIPPED: their
/// run-level value comes from the typed accessor (which wins on read), so
/// re-injecting them here would double-source the run aggregate — the
/// hazard the const's doc describes. Their per-phase `PhaseBucket` value
/// still feeds per-phase rendering.
///
/// Per-phase reduction dispatch is described on [`PhaseBucket`];
/// the cross-phase fold here uses `sample_count` as the weight so
/// Gauge(Avg) keys get the weighted mean (the correct cross-phase
/// semantic for typical-load metrics) while other kinds fold per
/// their natural reduction. Existing keys in `target` are not
/// overwritten — `read_sample` path values win when both produced
/// an entry.
///
/// Without this fill, `cargo ktstr stats compare` silently misses
/// these phase-only metrics (avg_imbalance_ratio, iteration_rate,
/// system_time_ns, user_time_ns) in flat-row output because
/// `MetricDef::read` falls back to ext_metrics and finds nothing.
pub fn populate_run_ext_metrics_from_phases(
    phases: &[PhaseBucket],
    target: &mut std::collections::BTreeMap<String, f64>,
) {
    // No early-return on empty `phases`: the derive_rate_metrics post-pass
    // below must still run over whatever components populate_run_ext_metrics
    // already inserted into `target` (the empty-phases case), so a run-level
    // Rate is re-derived rather than silently dropped. The loops below are
    // no-ops when `phases` is empty.
    // Collect every metric key that appears on any phase.
    let mut keys: std::collections::BTreeSet<&String> = std::collections::BTreeSet::new();
    for phase in phases {
        for key in phase.metrics.keys() {
            keys.insert(key);
        }
    }
    for key in keys {
        if target.contains_key(key) {
            continue;
        }
        let Some(def) = crate::stats::metric_def(key) else {
            continue;
        };
        // Derived metrics (Rate / Distribution / WorstLowest) are produced
        // from their pooled components, not folded as per-phase values: skip
        // here. A Rate re-derives after the loop (Σnum/Σdenom over the folded
        // components); Distribution / WorstLowest are re-pooled run-level by
        // `populate_run_distribution_metrics` (and never appear in
        // phase.metrics anyway). Folding a ready-made derived value would lose
        // the re-pool, and routing one into aggregate_samples_weighted within
        // a run is not its producer path.
        if def.kind.is_derived() {
            continue;
        }
        // Typed-backed keys (those in TYPED_FIELD_NAMES — a typed GauntletRow
        // accessor that wins on read) must NOT be re-injected into ext_metrics
        // from the phase fold: the ext copy would be unread bloat and, for
        // stuck_count (whose per-phase fold sum is `<=` the typed whole-run
        // count, strictly below once a cross-boundary/out-of-phase window is
        // stuck — they share the is_cpu_stuck predicate but the run-level
        // count windows the full stream), a redundant-or-divergent value,
        // not a guaranteed duplicate. Their per-phase
        // PhaseBucket value still feeds rendering; the run-level value stays
        // the typed path. Mirrors the sibling populate_run_ext_metrics.
        // (Without this, folding max_imbalance_ratio + stuck_count onto
        // captured buckets would leak both into ext_metrics on the common
        // path.)
        if TYPED_FIELD_NAMES.contains(&key.as_str()) {
            continue;
        }
        // Per-phase (value, sample_count) for the kind-aware fold.
        // A phase that doesn't carry the key contributes nothing.
        // Lock-step shape enforced by the (f64, usize) pair type.
        // `sample_count.max(1)` is load-bearing for Gauge(Avg) keys: a
        // synthesized zero-capture phase (the
        // build_phase_buckets_with_stimulus seam) carrying a
        // capture-independent Gauge(Avg) value at sample_count==0 gets
        // weight 1 (one phase observation) rather than being zero-weighted
        // out of the run-level mean. The floor is a no-op for
        // Counter/DeltaSum keys, which sum with weights ignored (see
        // aggregate_finite): iteration_rate's components
        // total_phase_iterations / total_phase_duration_sec are such
        // Counters, so a synthesized step's iterations are INCLUDED in the
        // re-pooled iteration_rate via the sum — the run-aggregate
        // completion of the per-step rate handling (iteration_rate itself is a
        // Rate, skipped above and re-derived below). A regression dropping
        // the floor would silently re-drop a zero-capture step's Gauge(Avg)
        // value from the sidecar aggregate.
        let pairs: Vec<(f64, usize)> = phases
            .iter()
            .filter_map(|phase| {
                phase
                    .metrics
                    .get(key)
                    .copied()
                    .map(|v| (v, phase.sample_count.max(1)))
            })
            .collect();
        if pairs.is_empty() {
            continue;
        }
        if let Some(reduced) = crate::stats::aggregate_samples_weighted(&pairs, def.kind) {
            target.insert(key.clone(), reduced);
        }
    }
    // Re-derive Rate metrics from the now-folded components so the run
    // rate is Σnumerator / Σdenominator (the components folded by their
    // own kinds above — a Counter numerator summed across phases).
    crate::stats::derive_rate_metrics(target);
}

/// Inject the run-level POOLED `iterations_per_cpu_sec` Rate's two Counter
/// components into `stats.ext_metrics`, summed across the cgroups that have
/// measured on-CPU time — the cross-cgroup re-pool axis. Rather than routing
/// the per-cgroup efficiency through `AssertResult::merge`'s worst-by-polarity
/// `ext_metrics` fold (which picks the WORST cgroup's value, not Σ, and has
/// no derive post-pass), this reads the already-merged `stats.cgroups` vec
/// directly: `iterations_per_cpu_sec` = Σ`total_iterations` /
/// Σ(`total_cpu_time_ns`/1e9) over cgroups with `total_cpu_time_ns > 0` — the
/// per-cgroup [`CgroupStats::iterations_per_cpu_sec`] re-pooled, NOT a mean of
/// per-cgroup ratios, NOT the worst single cgroup.
///
/// MUST run at the eval layer AFTER the cgroup-bearing merges (every merge that
/// contributes a [`CgroupStats`], so `stats.cgroups` holds every per-cgroup
/// entry) and BEFORE the sidecar write. The trailing monitor-verdict merge at
/// the eval layer merges an `inconclusive()` carrying empty `stats` (no cgroups,
/// no ext keys), so it is safe to run after this. If component injection ever
/// moved BEFORE a cgroup-bearing merge, that worst-by-polarity fold would
/// min/max these Counter keys into single-cgroup scalars, silently corrupting
/// the pooled sum.
///
/// A cgroup with `total_cpu_time_ns == 0` (schedstat unavailable, or
/// `num_workers == 0`) is EXCLUDED from BOTH sums — mirroring the per-cgroup
/// [`CgroupStats::iterations_per_cpu_sec`] None-on-zero (`total_cpu_time_ns >
/// 0` implies `num_workers > 0`, so the one predicate covers both). Crediting
/// an unmeasured cgroup's iterations against the measured cgroups' CPU-seconds
/// would overstate cohort efficiency — the silent-wrong-answer this gate
/// prevents. Both components are inserted both-or-neither (the
/// `derive_rate_metrics` co-location invariant), only when the summed MEASURED
/// on-CPU time is > 0 (every cgroup unmeasured ⇒ no rate). The ns→s `/1e9` is
/// applied ONCE here on the summed ns (not per-cgroup, to avoid repeated float
/// rounding), since `derive_rate_metrics` is a bare num/den.
/// `total_iterations_pooled` is a DISTINCT ext-only key, not the typed
/// `total_iterations` (skipped from ext_metrics; it folds cross-RUN as a MEAN
/// — a display average — while a Rate numerator must SUM-fold so Σnum/Σdenom
/// re-pools, so one shared key cannot carry both folds). Because it sums only
/// MEASURED cgroups, it is ≤ the merge-summed typed `total_iterations` (which
/// includes any zero-cpu-time cgroups), and equals it unless an excluded
/// zero-cpu-time cgroup carried iterations>0.
pub fn populate_run_pooled_iterations_per_cpu_sec(stats: &mut ScenarioStats) {
    // Exclude cgroups with no measured on-CPU time from BOTH sums (mirrors the
    // per-cgroup None-on-zero): crediting an unmeasured cgroup's iterations
    // against the measured cgroups' CPU-seconds would overstate efficiency.
    let summed_ns: u64 = stats
        .cgroups
        .iter()
        .filter(|c| c.total_cpu_time_ns > 0)
        .map(|c| c.total_cpu_time_ns)
        .sum();
    if summed_ns == 0 {
        return;
    }
    let summed_iters: u64 = stats
        .cgroups
        .iter()
        .filter(|c| c.total_cpu_time_ns > 0)
        .map(|c| c.total_iterations)
        .sum();
    stats
        .ext_metrics
        .insert("total_iterations_pooled".to_string(), summed_iters as f64);
    stats
        .ext_metrics
        .insert("total_cpu_time_sec".to_string(), summed_ns as f64 / 1e9);
    crate::stats::derive_rate_metrics(&mut stats.ext_metrics);
}

/// Populate run-level DERIVED distributional metrics into
/// `stats.ext_metrics`: every registered `MetricKind::Distribution`
/// and `MetricKind::WorstLowest`. This is the SOLE
/// within-run producer of those metrics' values — they carry no per-phase
/// sample slice and no cross-cgroup merge fold, and their registry accessors
/// are `|_| None`, so `MetricDef::read` reads the value
/// written here from `ext_metrics`.
///
/// DISTRIBUTION (the 5 wake / run-delay aggregates): pools the RAW sample
/// vectors held in `stats.phases[].per_cgroup` across EVERY phase and EVERY
/// cgroup into one combined set, then recomputes the percentile / CV / mean
/// / extreme over it — the statistic of the union, NOT a max or mean of
/// per-cgroup reductions (the percentile of a union is not the max of
/// per-source percentiles). The ns→µs scale is applied ONCE here (the
/// carriers store raw ns, per [`PhaseCgroupStats::run_delays_ns`]). The wake
/// pool is population-WEIGHTED: each phase carrier's samples carry weight
/// `wake_sample_total / wake_latencies_ns.len()`, so a phase whose reservoir
/// hit the cap contributes by true population, not capped length (the
/// cross-PHASE de-skew) — reduced via the weighted percentile / moments.
/// The run-delay pool is unweighted (per-worker, never reservoir-capped, so
/// length IS population). Below the wake cap every weight is 1.0, so the
/// weighted P99 / median / mean / worst are byte-identical to the unweighted
/// concat; the weighted CV matches only within ~1e-9 (it sums the mean in f64
/// where the unweighted path sums in u64 — a weighted variance cannot keep the
/// u64 sum).
///
/// CARRIER-LESS FOLD (graceful degradation): a cgroup whose raw samples are
/// NOT in the pool — a backdrop epoch that fell on BASELINE or the
/// inter-step gap (no paired host bucket, so no carrier) or a cgroup whose
/// carrier was stripped/empty (`strip_phase_cgroup_samples`) — is NOT
/// dropped. Its
/// surviving per-cgroup [`CgroupStats`] reduction folds worst-wins (max — every
/// Distribution metric is `LowerBetter`, registry-gated) into the pooled value.
/// The CgroupStats reductions are never stripped — `stats.cgroups[]` is the
/// already-reduced `cgroup_stats(reports)` output, a SEPARATE reduction path
/// from the per-phase carriers — so a carrier-less cgroup always has a source.
/// When EVERY carrier is empty (a fully-stripped run) the pool is empty and the
/// result degenerates to the max over every cgroup's reduction — the pre-Item-7
/// cross-cgroup max. NOTE the value CLASS of a folded cgroup differs from a
/// pooled one for the P99 / Median / Mean / CV reductions: a pooled cgroup
/// contributes to the percentile of the union; a carrier-less cgroup
/// contributes its per-cgroup reduction worst-wins (a worst-cgroup proxy, not
/// pooled). For the `SampleReduction::Worst` reduction the two COINCIDE
/// (max-of-union == max-of-per-cgroup-maxes), so the carrier-less fold is exact
/// there, not a proxy. A second asymmetry specific to CV (from the population
/// weighting): the POOLED CV divides variance/mean by Σ per-sample weights (the
/// reconstructed population), while a carrier-less cgroup's folded CV is
/// [`cgroup_stats`]'s UNWEIGHTED CV (`n = all_latencies.len()`). The two
/// coincide below the cap (all weights 1.0) and diverge above it; the mix is
/// sound — a carrier-less cgroup has no per-phase weight data to
/// population-weight (its carrier is absent by definition), and both feed the
/// same LowerBetter worst-wins max. Backdrop step-phase carriers now join
/// the pool directly (per-epoch expansion in `collect_handles`); only the
/// carrier-less cases above fold worst-wins.
///
/// WORSTLOWEST (the 2 iteration efficiencies): the lowest (worst) cgroup's
/// efficiency, computed per-cgroup from the `stats.cgroups[]` COUNTERS via
/// [`CgroupStats::iterations_per_worker`] / [`CgroupStats::iterations_per_cpu_sec`]
/// and the None-aware lowest-wins fold (a measured `Some(0.0)` — starvation
/// — wins; a no-data `None` is skipped; an all-`None` cohort writes no key,
/// preserving absence as a missing ext entry rather than a `0.0`). The
/// counters survive stripping, so WorstLowest needs no fallback branch.
///
/// Runs post-merge at the eval layer beside
/// [`populate_run_pooled_iterations_per_cpu_sec`], AFTER the per-cgroup
/// carriers are folded into `stats.phases` and BEFORE the sidecar write, so
/// `stats.phases[].per_cgroup` is fully merged and `stats.cgroups` is the
/// final per-cgroup roll-up.
pub fn populate_run_distribution_metrics(stats: &mut ScenarioStats) {
    // Pool the per-phase per-cgroup raw sample vectors across every phase and
    // cgroup ONCE for the Distribution PRIMARY path, then sort so the
    // percentile reductions can index directly. `wake_latencies_ns` is
    // per-WAKEUP (reservoir-capped at MAX_WAKE_SAMPLES on the carrier because
    // it can reach 100k); `run_delays_ns` is per-WORKER (one sample/worker, not
    // capped), so the run-delay pool is total-workers × phases — genuinely
    // small. The wake pool is NOT intrinsically small: it is the union of the
    // per-carrier wake vectors, num_carriers × MAX_WAKE_SAMPLES worst case, so
    // its size is bounded by the upstream 16 MiB bulk-frame cap on the arriving
    // carriers (strip_phase_cgroup_samples is the overflow lever) rather than by
    // being tiny — no OOM risk, no cap needed here. Both are transient: reduced
    // to scalars here, never re-serialized.
    // Wake samples carry a per-sample population WEIGHT (`wake_sample_total /
    // reservoir len`) so a >cap phase contributes in proportion to its true
    // population, not its guest-capped length (the cross-PHASE de-skew). Run-delay
    // samples are per-worker and never reservoir-capped (no `*_sample_total`), so
    // their length IS their population — pooled unweighted.
    let mut wake_pool: Vec<(u64, f64)> = Vec::new();
    let mut run_delay_pool: Vec<u64> = Vec::new();
    // Names of cgroups that contributed NON-EMPTY samples to each pool. A
    // cgroup absent here — a backdrop epoch that fell on BASELINE / the
    // inter-step gap (no paired host bucket, so no carrier) or a
    // stripped/empty carrier — is NOT dropped from the run-level
    // Distribution: the re-pool folds its surviving per-cgroup CgroupStats
    // reduction worst-wins (see `populate_run_distribution_metrics_from`).
    // Backdrop step-phase carriers now join the pool directly (per-epoch
    // expansion in collect_handles), so a step-matched backdrop epoch pools
    // rather than worst-wins-folds.
    //
    // The fallback dedup keys on cgroup NAME (a `stats.cgroups` entry whose
    // name is in `*_carriers` is pooled, not reduction-folded), which assumes
    // carrier-bearing and carrier-less cgroup names are DISJOINT. That holds
    // WITHIN one step's collect (cgroupfs path uniqueness — two live cgroups
    // cannot share a name, mkdir would EEXIST — and a single collect_handles
    // call attaches carriers to all its handles or none). It does NOT hold
    // across STEPS: `AssertResult::merge` extends `stats.cgroups` per
    // (handle, step), so a name that carried samples at step k recurs at step
    // k+1, and the step-(k+1) entry is skipped by this dedup (its name is in
    // `*_carriers`). That only OMITS a contribution, never vanishes the metric
    // (the step-k pool still produces it). A skipped step-(k+1) entry whose
    // carrier is merely EMPTY (collected no samples) is harmless: its per-cgroup
    // reduction is the trivial zero a worst-wins f64::max ignores. The only
    // LOSSY case is a step-(k+1) entry STRIPPED of live samples while step k
    // survives, and that cannot arise today: `strip_phase_cgroup_samples` strips
    // RUN-WIDE (every phase at once), so a run is never partially stripped per
    // step. A backdrop name now enters `*_carriers` (pooled once via its
    // per-epoch expansion) so it is skipped from the reduction-fold — and a
    // backdrop and a step-local cgroup cannot share a live name (cgroupfs
    // mkdir EEXIST; a backdrop is live the whole scenario), so each
    // stats.cgroups entry still contributes via exactly one of {pool,
    // reduction-fold} — no double count.
    let mut wake_carriers: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    let mut run_delay_carriers: std::collections::BTreeSet<&str> =
        std::collections::BTreeSet::new();
    for phase in &stats.phases {
        for (cgname, pcg) in &phase.per_cgroup {
            if !pcg.wake_latencies_ns.is_empty() {
                // Per-sample weight = true population / surviving reservoir size.
                // A ≤cap carrier has len == wake_sample_total → weight 1.0, so the
                // pool is value-for-value with the unweighted concat; a >cap
                // carrier's capped samples each stand for `total/len > 1` true
                // wakes, restoring the cross-phase population proportion.
                //
                // INVARIANT: `reservoir_push` bumps wake_sample_total on EVERY
                // wakeup but pushes into the reservoir only up to MAX_WAKE_SAMPLES,
                // and both the carrier merge and `phase_cgroup_stats` SUM the two,
                // so wake_sample_total >= len always (== len below the cap). A
                // carrier violating that — samples present but a zeroed/under-count
                // total — would yield weight < 1 and silently UNDER-weight (at
                // weight 0, DROP) its samples. Clamp the numerator to len so a
                // malformed carrier degrades to unit weight (reservoir treated as
                // its own population) instead of dropping data; debug_assert the
                // invariant so a real counting bug surfaces in dev.
                let len = pcg.wake_latencies_ns.len() as u64;
                debug_assert!(
                    pcg.wake_sample_total >= len,
                    "wake_sample_total ({}) < reservoir len ({}): malformed carrier",
                    pcg.wake_sample_total,
                    len,
                );
                let w = pcg.wake_sample_total.max(len) as f64 / len as f64;
                wake_pool.extend(pcg.wake_latencies_ns.iter().map(|&v| (v, w)));
                wake_carriers.insert(cgname.as_str());
            }
            if !pcg.run_delays_ns.is_empty() {
                run_delay_pool.extend_from_slice(&pcg.run_delays_ns);
                run_delay_carriers.insert(cgname.as_str());
            }
        }
    }
    wake_pool.sort_unstable_by_key(|&(v, _)| v);
    run_delay_pool.sort_unstable();
    populate_run_distribution_metrics_from(
        &mut stats.ext_metrics,
        crate::stats::METRICS.iter().filter_map(|m| {
            matches!(
                m.kind,
                crate::stats::MetricKind::Distribution { .. }
                    | crate::stats::MetricKind::WorstLowest { .. }
                    | crate::stats::MetricKind::WakeLatencyTailRatio
            )
            .then_some((m.name, m.kind))
        }),
        &wake_pool,
        &wake_carriers,
        &run_delay_pool,
        &run_delay_carriers,
        &stats.cgroups,
        stats.total_iterations,
    );
}

/// Inner of [`populate_run_distribution_metrics`] taking the metric specs
/// `(name, kind)` and the pre-pooled+SORTED sample sets explicitly, so the
/// re-pool math is unit-testable without registered metrics (the
/// `derive_rate_metrics_from` precedent). `wake_pool` / `run_delay_pool` are
/// the cross-phase+cross-cgroup raw-ns unions (ascending); `*_carriers` name
/// the cgroups that contributed samples to each pool; `cgroups` supplies the
/// WorstLowest counters and the per-cgroup reductions that carrier-less
/// cgroups (backdrop / stripped) fold into the Distribution result.
#[allow(clippy::too_many_arguments)]
pub(crate) fn populate_run_distribution_metrics_from<'a>(
    target: &mut std::collections::BTreeMap<String, f64>,
    metrics: impl Iterator<Item = (&'a str, crate::stats::MetricKind)>,
    wake_pool: &[(u64, f64)],
    wake_carriers: &std::collections::BTreeSet<&str>,
    run_delay_pool: &[u64],
    run_delay_carriers: &std::collections::BTreeSet<&str>,
    cgroups: &[CgroupStats],
    run_total_iterations: u64,
) {
    use crate::stats::{MetricKind, SampleSource, WorstLowestDenominator};
    for (name, kind) in metrics {
        let value: Option<f64> = match kind {
            MetricKind::Distribution { source, reduction } => {
                // Pool the carried samples (the thesis: percentile of the
                // UNION), then fold worst-wins (max — Distribution is
                // LowerBetter, registry-gated) the surviving per-cgroup
                // reduction of every cgroup WITHOUT a carrier-with-samples for
                // this source (a backdrop, or a stripped/empty carrier), so no
                // cgroup is dropped from the run-level distribution. When EVERY
                // carrier is empty (fully stripped) the pool is empty and this
                // degenerates to the max over every cgroup — the pre-Item-7
                // cross-cgroup max.
                //
                // Pool reduction is per-source: WakeLatencyNs is population-WEIGHTED
                // (each phase's guest-capped samples carry weight
                // wake_sample_total/len, so a >cap phase contributes by true
                // population not capped length — the cross-PHASE de-skew, via
                // reduce_weighted_sorted_distribution); RunDelayNs is unweighted
                // (per-worker, never reservoir-capped, so length IS population, via
                // reduce_sorted_distribution).
                //
                // CONTRACT (differs from WorstLowest and WakeLatencyTailRatio
                // below, by design): a cohort with cgroups present but NO carrier
                // samples whose per-cgroup reductions are all 0.0 (e.g. phases
                // empty / no wake samples anywhere) folds to Some(0.0) — a
                // measured zero, matching the deleted 0.0-sentinel typed field
                // this replaced. The absent-vs-0.0 boundary is NOT purely
                // source-type-driven: WorstLowest yields ABSENCE (None) for its
                // all-None cohort because iterations_per_worker() /
                // iterations_per_cpu_sec() return Option; and WakeLatencyTailRatio
                // ALSO yields None when no cgroup has a tail, even though
                // wake_latency_tail_ratio() is a 0.0-sentinel f64 like the
                // Distribution reductions here — because a 0.0 ratio means "no
                // measurable tail" (median <= 0, i.e. NOT measured), not a
                // measured-zero percentile. So: Distribution emits Some(0.0) for a
                // no-sample run (a real measured zero of the percentile);
                // WorstLowest and WakeLatencyTailRatio emit None (no measurement).
                let (mut v, carriers): (Option<f64>, &std::collections::BTreeSet<&str>) =
                    match source {
                        SampleSource::WakeLatencyNs => (
                            (!wake_pool.is_empty())
                                .then(|| reduce_weighted_sorted_distribution(wake_pool, reduction)),
                            wake_carriers,
                        ),
                        SampleSource::RunDelayNs => (
                            (!run_delay_pool.is_empty())
                                .then(|| reduce_sorted_distribution(run_delay_pool, reduction)),
                            run_delay_carriers,
                        ),
                    };
                for cg in cgroups {
                    if !carriers.contains(cg.cgroup_name.as_str()) {
                        let r = distribution_cgroup_reduction(cg, source, reduction);
                        v = Some(v.map_or(r, |acc| acc.max(r)));
                    }
                }
                v
            }
            // numerator is always Iterations (the only variant); the
            // denominator picks the per-cgroup efficiency method.
            //
            // In a MULTI-STEP scenario `AssertResult::merge` extends
            // `stats.cgroups` per (handle, step), so the same cgroup name
            // appears once per step; this selects the lowest single
            // (handle, step) entry, NOT a per-name whole-run efficiency. That
            // preserves the deleted `fold_lowest_some` granularity exactly and
            // mirrors `populate_run_pooled_iterations_per_cpu_sec`, which sums
            // over the same per-(handle, step) entries.
            MetricKind::WorstLowest { denominator, .. } => {
                let mut worst: Option<f64> = None;
                for cg in cgroups {
                    let per_cg = match denominator {
                        WorstLowestDenominator::NumWorkers => cg.iterations_per_worker(),
                        WorstLowestDenominator::CpuTimeNs => cg.iterations_per_cpu_sec(),
                    };
                    // Lowest-wins, None-aware (the semantic the deleted
                    // `fold_lowest_some` carried in `AssertResult::merge`): a
                    // measured `Some(0.0)` (starvation) wins the worst bucket;
                    // a `None` is skipped.
                    if let Some(v) = per_cg
                        && worst.is_none_or(|w| v < w)
                    {
                        worst = Some(v);
                    }
                }
                worst
            }
            // Worst-cgroup wake-latency tail amplification: the MAX over each
            // cgroup's own p99/median ratio (`CgroupStats::wake_latency_tail_ratio`).
            // Emit NO key below the min-iterations noise floor (low-N ratios are
            // single-outlier noise, not a distributional signal — gated HERE at
            // the producer, NOT via a meaned-iteration accessor on the
            // aggregated row), and none when no cgroup carried a measurable tail
            // (every per-cgroup ratio 0.0, i.e. no median wake latency anywhere).
            // Absence then stays distinct from a measured value and no
            // sub-threshold run enters the cross-RUN mean. `wake_latency_tail_ratio`
            // returns 0.0 for a cgroup with no wake samples (median <= 0), which
            // a max-wins fold over the r > 0.0 reals correctly skips.
            MetricKind::WakeLatencyTailRatio => {
                if run_total_iterations < crate::stats::WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS {
                    None
                } else {
                    let mut worst: Option<f64> = None;
                    for cg in cgroups {
                        let r = cg.wake_latency_tail_ratio();
                        if r > 0.0 {
                            worst = Some(worst.map_or(r, |w| w.max(r)));
                        }
                    }
                    worst
                }
            }
            _ => None,
        };
        // Insert only a real, FINITE value: an absent key (all-None
        // WorstLowest cohort, or no cgroups at all) stays distinct from a
        // measured 0.0, matching the None-vs-Some(0.0) contract the typed
        // Option carried. The is_finite guard is a no-op for every
        // registry-valid metric (reduce_sorted_distribution reduces non-empty
        // pools with CV guarded to 0.0 on zero mean; WorstLowest reuses
        // iterations_per_worker()/iterations_per_cpu_sec() which return None on
        // a zero denominator), but it MATTERS for the registry-impossible
        // cross-source arm of distribution_cgroup_reduction: that arm returns
        // NaN, and when a Distribution has no pool (every carrier stripped) the
        // carrier-less fold can carry that NaN to `v`. An inserted NaN would
        // fail the ENTIRE serde_json sidecar write (serde_json rejects
        // non-finite), losing ALL run telemetry — so the guard degrades a
        // misauthored metric to ABSENCE here rather than risking that write
        // failure downstream.
        if let Some(v) = value.filter(|v| v.is_finite()) {
            target.insert(name.to_string(), v);
        }
    }
}

/// Reduce a NON-EMPTY ascending-sorted raw-ns sample pool to one
/// [`crate::stats::SampleReduction`] value, ns→µs once. Mirrors the
/// per-cgroup reductions [`cgroup_stats`] computes (p99 / median via
/// [`percentile`], CV with `n = pool.len()`, mean, max) so the run-level
/// re-pool reproduces them over the COMBINED cross-cgroup set — to within
/// FP tolerance for CV / mean, not bit-exactly: this sums over the
/// ASCENDING-sorted pool while `cgroup_stats` sums over the unsorted
/// arrival order, so the float results differ by ~1e-15 (the parity test
/// `repool_distribution_value_for_value_with_cgroup_stats` uses a 1e-9
/// bound). Same "distribution-equivalent, not byte-identical" framing as
/// the `wake_latencies_ns` carrier doc.
pub(crate) fn reduce_sorted_distribution(
    sorted: &[u64],
    reduction: crate::stats::SampleReduction,
) -> f64 {
    use crate::stats::SampleReduction;
    match reduction {
        SampleReduction::P99 => percentile(sorted, 0.99) as f64 / 1000.0,
        SampleReduction::Median => percentile(sorted, 0.5) as f64 / 1000.0,
        SampleReduction::Cv => {
            let n = sorted.len() as f64;
            let mean_ns = sorted.iter().sum::<u64>() as f64 / n;
            if mean_ns > 0.0 {
                let variance = sorted
                    .iter()
                    .map(|&v| (v as f64 - mean_ns).powi(2))
                    .sum::<f64>()
                    / n;
                variance.sqrt() / mean_ns
            } else {
                0.0
            }
        }
        // Divide ONCE on the summed/maxed ns (the carriers store raw ns):
        // mean(ns)/1000 == mean(ns/1000) and max(ns)/1000 == max(ns/1000).
        // Sum in f64 (not u64-then-cast) to match cgroup_stats's f64 run-delay
        // accumulation and PhaseCgroupStats::run_delay_summary — overflow-safe
        // (an f64 sum saturates toward +inf; a u64 sum would panic in debug /
        // silently wrap in release on a pathological pool), value identical
        // within the 1e-9 parity bound. (The Cv arm's mean_ns above keeps the u64 sum
        // because cgroup_stats's CV also u64-sums all_latencies — matching it is
        // exact-parity-preserving there.)
        SampleReduction::Mean => {
            sorted.iter().map(|&v| v as f64).sum::<f64>() / sorted.len() as f64 / 1000.0
        }
        // Sorted ascending, so the last element is the max.
        SampleReduction::Worst => *sorted.last().expect("non-empty by caller") as f64 / 1000.0,
    }
}

/// Weighted nearest-rank percentile over a value-sorted `(value, weight)` pool —
/// the weighted sibling of [`percentile`]. Matches `percentile`'s convention
/// (the value at 1-indexed rank `ceil(W * p)`, `W` = total weight, floored at
/// rank 1) so with UNIT weights (every weight `1.0`) it returns byte-identically:
/// cumulative weight after `k` elements is `k`, `ceil(W*p) == ceil(n*p)`, and the
/// `.max(1.0)` floor mirrors `percentile`'s `saturating_sub(1)`, so the crossing
/// element is `percentile`'s `sorted[ceil(n*p)-1]` for p>0 and `sorted[0]` at
/// p=0. Used by the run-level wake re-pool to weight each phase's samples by
/// true population.
pub(crate) fn weighted_percentile(sorted: &[(u64, f64)], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    debug_assert!(
        sorted.windows(2).all(|w| w[0].0 <= w[1].0),
        "weighted_percentile() requires value-sorted input",
    );
    let total: f64 = sorted.iter().map(|&(_, w)| w).sum();
    // Nearest-rank target, floored at 1 so `p == 0.0` maps to the first element
    // (mirrors percentile's saturating_sub(1) flooring rank 0 to index 0).
    let target = (total * p).ceil().max(1.0);
    let mut cum = 0.0;
    for &(v, w) in sorted {
        cum += w;
        if cum >= target {
            return v;
        }
    }
    sorted.last().map(|&(v, _)| v).unwrap_or(0)
}

/// Weighted sibling of [`reduce_sorted_distribution`] for the wake-latency
/// re-pool: each `(value, weight)` carries a per-sample weight of
/// `wake_sample_total / reservoir_len`, so a >cap phase (reservoir-capped on the
/// guest) contributes in proportion to its TRUE population, not its capped
/// length — removing the cross-PHASE length-skew. With UNIT weights (every phase
/// ≤cap, so `len == wake_sample_total`) it reduces byte-identically to
/// [`reduce_sorted_distribution`] for P99 / Median / Mean / Worst; the Cv arm
/// differs only by the f64-vs-u64 mean sum. For the small fixed pool the parity
/// test uses, that gap is ~1e-15 (within its 1e-9 bound), but it grows ~n·ε with
/// pool size — a cross-phase pool can reach millions of samples (~1e-9–1e-8 on a
/// high-CV pool), so a LARGE-pool parity test must not assume a universal 1e-15.
/// A weighted variance cannot keep the u64 sum.  Exhaustive over SampleReduction,
/// mirroring [`reduce_sorted_distribution`], so a new variant fails the build.
///
/// The Cv / Mean `total_w <= 0.0` guards and [`weighted_percentile`]'s
/// all-weight-zero fall-through are degenerate-input belts: the capture-path
/// caller [`populate_run_distribution_metrics`] clamps every per-sample weight to
/// a floor of 1.0, so `total_w >= len >= 1` there and those branches are
/// unreachable on the production path.
pub(crate) fn reduce_weighted_sorted_distribution(
    sorted: &[(u64, f64)],
    reduction: crate::stats::SampleReduction,
) -> f64 {
    use crate::stats::SampleReduction;
    match reduction {
        SampleReduction::P99 => weighted_percentile(sorted, 0.99) as f64 / 1000.0,
        SampleReduction::Median => weighted_percentile(sorted, 0.5) as f64 / 1000.0,
        SampleReduction::Cv => {
            let total_w: f64 = sorted.iter().map(|&(_, w)| w).sum();
            if total_w <= 0.0 {
                return 0.0;
            }
            let mean_ns = sorted.iter().map(|&(v, w)| v as f64 * w).sum::<f64>() / total_w;
            if mean_ns > 0.0 {
                let variance = sorted
                    .iter()
                    .map(|&(v, w)| w * (v as f64 - mean_ns).powi(2))
                    .sum::<f64>()
                    / total_w;
                variance.sqrt() / mean_ns
            } else {
                0.0
            }
        }
        SampleReduction::Mean => {
            let total_w: f64 = sorted.iter().map(|&(_, w)| w).sum();
            if total_w <= 0.0 {
                return 0.0;
            }
            sorted.iter().map(|&(v, w)| v as f64 * w).sum::<f64>() / total_w / 1000.0
        }
        // Max value present, weight-invariant — last element of the value-sorted pool.
        SampleReduction::Worst => sorted.last().map(|&(v, _)| v).unwrap_or(0) as f64 / 1000.0,
    }
}

/// One cgroup's surviving [`CgroupStats`] reduction for a
/// [`crate::stats::MetricKind::Distribution`] (source, reduction) pair — the
/// value folded worst-wins into the run-level distribution for a cgroup whose
/// raw samples are NOT in the pool (a backdrop, or a stripped/empty carrier).
/// Worst-wins is `f64::max` (every Distribution metric is `LowerBetter`,
/// enforced by `every_metric_has_kind_consistent_with_naming`).
///
/// Per-source match, EXHAUSTIVE over SampleReduction (no `_` catch-all,
/// mirroring reduce_sorted_distribution) so a new SampleSource or
/// SampleReduction variant fails the build until a reduction field is wired.
/// The cross-source reductions (a wake source asking for a run-delay reduction,
/// or vice versa) are registry-impossible (no CgroupStats field exists), so
/// they debug_assert in tests and, in release, return `f64::NAN` rather than
/// 0.0 — NaN is IGNORED by the caller's `f64::max` worst-wins fold, and if it
/// still reaches `populate_run_distribution_metrics`'s insert (a pool-less
/// Distribution whose every carrier-less cgroup hits this arm) the is_finite
/// insert guard drops it to absence. Either way a registry-authoring mistake
/// drops the bogus contribution instead of folding a 0.0 that a LowerBetter
/// metric would read as "perfect".
fn distribution_cgroup_reduction(
    cg: &CgroupStats,
    source: crate::stats::SampleSource,
    reduction: crate::stats::SampleReduction,
) -> f64 {
    use crate::stats::{SampleReduction, SampleSource};
    match source {
        SampleSource::WakeLatencyNs => match reduction {
            SampleReduction::P99 => cg.p99_wake_latency_us,
            SampleReduction::Median => cg.median_wake_latency_us,
            SampleReduction::Cv => cg.wake_latency_cv,
            SampleReduction::Mean | SampleReduction::Worst => {
                debug_assert!(false, "no CgroupStats wake reduction for {reduction:?}");
                f64::NAN
            }
        },
        SampleSource::RunDelayNs => match reduction {
            SampleReduction::Mean => cg.mean_run_delay_us,
            SampleReduction::Worst => cg.worst_run_delay_us,
            SampleReduction::P99 | SampleReduction::Median | SampleReduction::Cv => {
                debug_assert!(
                    false,
                    "no CgroupStats run-delay reduction for {reduction:?}"
                );
                f64::NAN
            }
        },
    }
}

/// Populate cross-RUN aggregate entries for every registered
/// `crate::stats::MetricDef` whose `read_sample` returns finite
/// values across the entire sample series. Writes into
/// `target` (typically `ScenarioStats::ext_metrics`) under the
/// metric's registry name — the same key the per-phase
/// [`PhaseBucket::metrics`] uses, so cross-RUN and per-phase
/// consumers reference the same name.
///
/// Existing keys are NOT overwritten — a typed GauntletRow field's
/// value (populated via the MetricDef accessor at sidecar-write
/// time) wins on the read path, and this fn fills the gap for
/// registered metrics that have a `read_sample` wire but no typed
/// GauntletRow field. Without this fill, `cargo ktstr stats compare`
/// silently skips the metric (read returns None on both sides;
/// the EPSILON guard drops the row).
///
/// Per-phase reduction dispatch is described on [`PhaseBucket`];
/// the cross-RUN fold here uses `crate::stats::aggregate_samples_for_phase`
/// over the full sample series, with TYPED_FIELD_NAMES gating to
/// avoid duplicating typed-accessor sources.
pub fn populate_run_ext_metrics(
    samples: &crate::scenario::sample::SampleSeries,
    target: &mut std::collections::BTreeMap<String, f64>,
) {
    // Typed-backed keys are skipped via the module-level TYPED_FIELD_NAMES
    // (shared with populate_run_ext_metrics_from_phases) so only
    // ext-metrics-only registry entries are populated here.
    for metric_def in crate::stats::METRICS {
        if target.contains_key(metric_def.name) {
            continue;
        }
        if TYPED_FIELD_NAMES.contains(&metric_def.name) {
            continue;
        }
        let readings: Vec<f64> = samples
            .iter_samples()
            .filter_map(|s| metric_def.read_sample(&s))
            .collect();
        if readings.is_empty() {
            continue;
        }
        if let Some(reduced) = crate::stats::aggregate_samples_for_phase(metric_def, &readings) {
            target.insert(metric_def.name.to_string(), reduced);
        }
    }
    // Re-derive Rate metrics from the read_sample components just folded
    // in. populate_run_ext_metrics is pub and called standalone (tests,
    // and not only ahead of populate_run_ext_metrics_from_phases), so it
    // derives its own rates to stay self-contained.
    crate::stats::derive_rate_metrics(target);
}
