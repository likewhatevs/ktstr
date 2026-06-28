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
    // worst_page_locality is NO LONGER a typed field: it is
    // `crate::stats::MetricKind::WorstLowest` (NumaLocal/NumaTotal), re-pooled
    // None-aware post-merge by `populate_run_distribution_metrics` from the
    // per-phase NUMA carriers (the lowest per-cgroup page_locality, a measured
    // 0.0 winning) — the `worst_page_locality` method shared with
    // `run_metric`. The deleted typed field folded via `fold_lowest_nonzero`,
    // which SKIPPED a measured 0.0 and reported a better-than-worst cross-run
    // value; the ext re-pool fixes both the field and the read surfaces at once.
    // worst_cross_node_migration_ratio is NO LONGER a typed field: it is
    // `crate::stats::MetricKind::WorstCrossNodeRatio`, re-pooled post-merge by
    // `populate_run_distribution_metrics` from the per-phase NUMA carriers (the
    // MAX per-cgroup cross-node churn ratio over the latest residency total) via
    // the `worst_cross_node_migration_ratio` method shared with `run_metric`. The
    // deleted typed `Gauge(Last)` field/GauntletRow column was merge-max-folded
    // within-run but cross-run averaged each run's value over `passes_observed`
    // (folding a NUMA-less passing run's 0.0 sentinel into the mean), AND
    // diverged from `run_metric` (which already re-derived from the per-phase
    // carriers) — so the sidecar and the in-test read gave different values on
    // multi-phase runs. The ext re-pool writes no key for a never-measured run,
    // so the MEAN divides only by runs that measured NUMA. A CHURN ratio
    // (cumulative migration EVENTS / final-snapshot resident pages), so it can
    // exceed 1.0; NOT a bounded `[0,1]` fraction.
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
    /// Look up the phase bucket for a [`Phase`] — `Phase::BASELINE` for the
    /// pre-first-Step settle window, `Phase::step(k)` for the test author's
    /// 0-indexed Step k. The typed `Phase` keeps the 1-indexed encoding (Step 0
    /// lives at the underlying `step_index = 1`) invisible at the call site.
    ///
    /// Returns `None` when no bucket with that phase exists (single-phase
    /// scenario, the scenario didn't reach that Step, or a phase past the last).
    pub fn phase(&self, phase: Phase) -> Option<&PhaseBucket> {
        self.phases.iter().find(|p| p.step_index == phase.as_u16())
    }

    /// Shortcut: look up a single metric value in a specific
    /// phase by phase-index. Returns `None` when:
    /// (a) the phase is absent (no bucket with `step_index` in
    ///     [`Self::phases`]),
    /// (b) the phase exists but had no finite samples for that
    ///     metric, OR
    /// (c) `metric` is a dynamic [`crate::stats::MetricId`] key (a
    ///     scheduler-runtime / payload string) not present in this phase's
    ///     stores. A built-in [`crate::stats::BuiltinMetric`] cannot be a
    ///     typo — that is a compile error — so this case needs a dynamic key.
    ///
    /// Two stores are checked: [`PhaseBucket::metrics`] (via
    /// [`PhaseBucket::get`]) and, failing that, the cross-cgroup phase
    /// SUM of a per-cgroup Counter ([`PhaseBucket::cgroup_counter_total`])
    /// for `"total_migrations"` / `"total_iterations"` / `"total_cpu_time_ns"` —
    /// registered `Counter`s with no per-sample source, so they live ONLY in the
    /// per-cgroup carriers; without this fallback the pooled lookup
    /// returned a silent `None` while the per-cgroup
    /// [`Self::phase_cgroup_metric`] surfaced the value. Symmetric with
    /// [`crate::vmm::VmResult::phase_metric`].
    ///
    /// Sentinel-free: `Some(0.0)` means the reducer produced a
    /// real zero from finite samples, NOT "missing data". See
    /// [`PhaseBucket::metrics`] for the registry source. When
    /// debugging an unexpected `None` on a dynamic key, check
    /// [`crate::stats::MetricId::def`]`().is_some()` to tell an unregistered
    /// key from absent data (built-in ids always resolve).
    ///
    /// Pass `Phase::BASELINE` for the settle window or `Phase::step(k)` for the
    /// test author's 0-indexed Step k — the typed `Phase` hides the 1-indexed
    /// encoding (see [`Self::phase`]).
    pub fn phase_metric(
        &self,
        phase: Phase,
        metric: impl Into<crate::stats::MetricId>,
    ) -> Option<f64> {
        let metric = metric.into();
        self.phase(phase).and_then(|p| {
            p.get(metric.as_str())
                .or_else(|| p.cgroup_counter_total(metric.as_str()))
        })
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

    /// Per-cgroup analog of [`Self::phase_metric`]: look up `metric` for a named
    /// `cgroup` in a [`Phase`] (`Phase::BASELINE` / `Phase::step(k)`), via that
    /// cgroup's per-phase carrier ([`PhaseCgroupStats::get`]), falling back to
    /// [`PhaseCgroupStats::cgroup_counter`] for the per-cgroup Counters
    /// `total_migrations`/`total_iterations`/`total_cpu_time_ns`. `None` when the
    /// phase has no bucket, no carrier for `cgroup`, the carrier carried no finite
    /// value for the metric, OR the metric is an unregistered dynamic key (a
    /// built-in id is typo-proof; an unregistered [`crate::stats::MetricId`] has no
    /// [`crate::stats::MetricId::def`], same as [`Self::phase_metric`]). The
    /// N-cgroups-to-N-queryable-sets surface on the AssertResult-holding path (the
    /// in-VM `post_vm` path uses [`crate::vmm::VmResult::phase_cgroup_metric`]).
    pub fn phase_cgroup_metric(
        &self,
        phase: Phase,
        cgroup: &str,
        metric: impl Into<crate::stats::MetricId>,
    ) -> Option<f64> {
        let metric = metric.into();
        self.phase(phase)
            .and_then(|p| p.per_cgroup.get(cgroup))
            .and_then(|pc| {
                pc.get(metric.as_str())
                    .or_else(|| pc.cgroup_counter(metric.as_str()))
            })
    }

    /// True iff the scenario produced at least one Step-phase
    /// bucket (any phase with `step_index >= 1`). False when
    /// `phases` is empty OR contains only `BASELINE` (the
    /// pre-first-Step settle window).
    ///
    /// Use this to fail a phase-aware assertion BEFORE calling
    /// [`Self::phase`] with a `Phase::step(k)` on a scenario that
    /// silently never advanced past BASELINE: a test that declared
    /// no `Step`s, OR a scenario that bailed in setup before any
    /// `Step` ran, would otherwise see [`Self::phase`] return
    /// `None` for every Step and the test would either panic on
    /// `.expect(...)` or pass vacuously.
    ///
    /// ```ignore
    /// anyhow::ensure!(
    ///     r.stats.has_steps(),
    ///     "scenario produced no Step-phase buckets — \
    ///      declare a Step or read Phase::BASELINE",
    /// );
    /// let throughput = r.stats.phase_metric(Phase::step(0), "throughput");
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
    /// real measured zero. Check [`crate::stats::MetricId::def`] on a dynamic
    /// key to tell an unregistered key from genuinely-absent data (built-in
    /// ids always resolve). (The map also carries any
    /// user-defined extensible-metric keys, plus the framework-internal
    /// Rate-component Counters — `total_phase_iterations` /
    /// `total_phase_duration_sec` / `total_iterations_pooled` /
    /// `total_cpu_time_sec`, the numerator/denominator plumbing behind
    /// `iteration_rate` / `iterations_per_cpu_sec` — all of which resolve
    /// here too; prefer the derived rate over its raw components.)
    ///
    /// RESOLVED here None-aware, ahead of the ext lookup, via a typed dispatch
    /// (`typed_sentinel_metric`): the cross-cgroup metrics `worst_spread`,
    /// `worst_migration_ratio`, `worst_gap_ms`, `total_migrations`,
    /// `total_iterations`, `worst_page_locality`,
    /// `worst_cross_node_migration_ratio`. The dispatch RE-DERIVES each from the
    /// per-cgroup (`self.cgroups`) / per-phase (`self.phases[].per_cgroup`, the
    /// NUMA metrics) carriers — `None` when no carrier measured it, `Some(0.0)`
    /// for a measured zero — preserving the sentinel-free contract. The 5
    /// non-NUMA metrics are 0.0-sentinel typed struct fields (a not-measured
    /// carrier coerces to 0.0, indistinguishable from a measured zero, so the
    /// re-derivation recovers the distinction); the two NUMA roll-ups
    /// (`worst_page_locality`, `worst_cross_node_migration_ratio`) have no struct
    /// field and re-pool purely from the per-phase NUMA carriers.
    /// (`worst_wake_latency_tail_ratio` resolves via the ext lookup as the
    /// `WakeLatencyTailRatio` key.)
    ///
    /// NOT resolved here (not in `ext_metrics`, no typed dispatch):
    /// - the monitor-sourced run-level metrics (`max_imbalance_ratio`,
    ///   `max_dsq_depth`, `stuck_count`, `total_fallback`,
    ///   `total_keep_last`), which `ScenarioStats` does not hold
    ///   run-level — read those per-phase via [`Self::phase_metric`].
    ///
    /// So this resolves the ext-sourced family AND the typed cross-cgroup fields
    /// (via the dispatch); only the monitor-sourced run-level metrics remain
    /// unresolved here (`ScenarioStats` holds them only per-phase).
    pub fn run_metric(&self, metric: impl Into<crate::stats::MetricId>) -> Option<f64> {
        let id = metric.into();
        // The sentinel-prone cross-cgroup run-level metrics re-derive None-aware
        // from the per-cgroup / per-phase carriers, recovering the never-measured
        // vs measured-zero distinction the 0.0-coerced struct fields (or, for
        // worst_page_locality, the absent field) cannot carry. A non-sentinel id
        // returns `None` from the dispatch and falls through to the ext-metrics map.
        if let crate::stats::MetricId::Builtin(b) = &id
            && let Some(resolved) = self.typed_sentinel_metric(*b)
        {
            return resolved;
        }
        self.ext_metrics.get(id.as_str()).copied()
    }

    /// Re-derive a typed run-level metric None-aware from the per-cgroup /
    /// per-phase carriers — recovering the never-measured (`None`) vs
    /// measured-zero (`Some(0.0)`) distinction a bare 0.0-sentinel `f64` cannot
    /// carry. The 0.0-sentinel typed struct fields (`worst_spread`,
    /// `worst_migration_ratio`, ...) coerce a not-measured carrier to 0.0 in
    /// `scenario_stats_for_cgroup` (indistinguishable from a measured zero);
    /// `worst_page_locality` is no longer a struct field at all — it re-pools
    /// purely from the per-phase NUMA carriers (see below).
    ///
    /// Returns `None` when `m` is NOT one of the typed sentinel ids — the caller
    /// then falls through to the ext-metrics lookup. `Some(None)` when `m` IS a
    /// sentinel id but no contributing carrier exists (loud-absent). `Some(Some(v))`
    /// for the re-derived value (a measured zero stays `Some(0.0)`).
    ///
    /// The 5 non-NUMA metrics source from `self.cgroups`, whose `Option` / counter
    /// carriers preserve the measured-vs-unmeasured state. The 2 NUMA roll-ups
    /// (`worst_page_locality`, `worst_cross_node_migration_ratio` — neither a
    /// struct field) source from `self.phases[].per_cgroup`, for different
    /// reasons: `page_locality` is structurally 0.0 on the `cgroup_stats` path
    /// (it needs an expected-node set the reports-only builder lacks), so
    /// `self.cgroups` cannot source it; `cross_node_migration_ratio` IS populated
    /// on `cgroup_stats`, but only as a single pre-folded scalar that cannot carry
    /// the cross-phase fold (SUM the per-phase migration-counter deltas over the
    /// LATEST residency total), so it too re-pools from the per-phase
    /// `PhaseCgroupStats` NUMA counters — which also single-sources it with the
    /// ext/sidecar value. On a phase-less direct-assertion `AssertResult` (no NUMA
    /// capture) both NUMA roll-ups read `None` (correct loud-absent).
    fn typed_sentinel_metric(&self, m: crate::stats::BuiltinMetric) -> Option<Option<f64>> {
        use crate::stats::BuiltinMetric as B;
        Some(match m {
            // worst_spread: highest spread over cgroups that measured it
            // (CgroupStats::spread is already Option — None when no worker had
            // measurable wall time). None iff no cgroup measured spread.
            B::WorstSpread => self.cgroups.iter().filter_map(|c| c.spread).reduce(f64::max),
            // worst_migration_ratio: highest migration ratio over cgroups that
            // ran iterations (a 0.0 with total_iterations>0 is measured; with
            // ==0 it is the rate-over-zero sentinel — never-measured).
            B::WorstMigrationRatio => self
                .cgroups
                .iter()
                .filter(|c| c.total_iterations > 0)
                .map(|c| c.migration_ratio)
                .reduce(f64::max),
            // total_migrations / total_iterations: cross-cgroup SUMs. 0 is a real
            // measured sum (cgroups ran, zero events); never-measured = no cgroups.
            B::TotalMigrations => (!self.cgroups.is_empty()).then(|| {
                self.cgroups
                    .iter()
                    .map(|c| c.total_migrations)
                    .fold(0u64, u64::saturating_add) as f64
            }),
            B::TotalIterations => (!self.cgroups.is_empty()).then(|| {
                self.cgroups
                    .iter()
                    .map(|c| c.total_iterations)
                    .fold(0u64, u64::saturating_add) as f64
            }),
            // worst_gap_ms: longest gap over cgroups with workers (gap 0 with
            // workers = measured "no gap observed"; never-measured = no workers).
            B::WorstGapMs => self
                .cgroups
                .iter()
                .filter(|c| c.num_workers > 0)
                .map(|c| c.max_gap_ms as f64)
                .reduce(f64::max),
            // worst_page_locality: LOWEST per-cgroup locality (the worst cell) over
            // cgroups that measured NUMA residency — a measured 0.0 (all pages
            // off-node) is the worst and WINS the lowest fold. Shared with the
            // ext/sidecar re-pool via worst_page_locality() so the typed read and
            // the cross-run comparison value agree (the same fix replaces the
            // fold_lowest_nonzero sentinel-skip on BOTH surfaces).
            B::WorstPageLocality => self.worst_page_locality(),
            // worst_cross_node_migration_ratio: highest cross-node churn ratio over
            // cgroups that measured NUMA. Shared with the ext/sidecar re-pool via
            // worst_cross_node_migration_ratio() so the typed read and the cross-run
            // comparison value agree (the divergence the typed field had).
            B::WorstCrossNodeMigrationRatio => self.worst_cross_node_migration_ratio(),
            _ => return None,
        })
    }

    /// Aggregate each cgroup's NUMA carriers across `self.phases` into
    /// `(latest_local, latest_total, summed_migrated)`. The cross-phase fold is
    /// ASYMMETRIC by counter class: `numa_pages_local` / `numa_pages_total` are
    /// current-residency GAUGE snapshots (the live `/proc/.../numa_maps`
    /// residency, recomputed each read), so the LATEST MEASURED phase's value is the run-end placement
    /// — SUMMING across phases would multiply residency by the phase count (a
    /// silent over-count); `cross_node_migrated` is a per-phase migration-counter
    /// DELTA over disjoint intervals, so it SUMS to the run total. The shared
    /// `latest_total` denominator serves both ratios (page_locality and
    /// cross_node share one page total), matching `cgroup_stats`, which reads the
    /// final-snapshot residency. One entry per cgroup that appeared in any phase.
    ///
    /// LATEST is None-aware: a later phase carrier that measured NO NUMA pages
    /// (`numa_pages_total == 0` — an empty backdrop slice, a non-NUMA `WorkType`
    /// phase, or a zero-worker carrier) does NOT overwrite an earlier MEASURED
    /// residency. A bare overwrite would zero out the earlier snapshot, and the
    /// `total > 0` filter in `worst_page_locality()` would then drop the cgroup
    /// from the worst-pool — silently reporting a better-than-worst run-level
    /// locality (the same measured-vs-unmeasured discipline the cross-cgroup fold
    /// applies, extended to the cross-phase axis).
    fn numa_agg_per_cgroup(&self) -> Vec<(u64, u64, u64)> {
        let mut by_cg: std::collections::BTreeMap<&str, (u64, u64, u64)> =
            std::collections::BTreeMap::new();
        // self.phases is in step (chronological) order, so a later MEASURED
        // phase's residency snapshot overwrites the earlier (LATEST wins); a
        // not-measured (total == 0) phase leaves the prior snapshot intact. The
        // migration deltas accumulate unconditionally (a 0 delta adds nothing).
        for phase in &self.phases {
            for (name, pc) in &phase.per_cgroup {
                let e = by_cg.entry(name.as_str()).or_insert((0, 0, 0));
                if pc.numa_pages_total > 0 {
                    e.0 = pc.numa_pages_local;
                    e.1 = pc.numa_pages_total;
                }
                // saturating: guest-runtime migration counter pooled across
                // phases; never wrap the derived cross-node-migration ratio.
                e.2 = e.2.saturating_add(pc.cross_node_migrated);
            }
        }
        by_cg.into_values().collect()
    }

    /// The run-level WORST (lowest) per-cgroup page-locality fraction, re-pooled
    /// None-aware from the per-phase NUMA carriers: the lowest
    /// `page_locality_of(latest_local, latest_total)` over cgroups that measured
    /// NUMA residency (`numa_pages_total > 0`) — a measured 0.0 (all pages
    /// off-node) WINS the lowest; an all-unmeasured cohort yields `None`. Shared
    /// by `run_metric`'s `WorstPageLocality` dispatch AND
    /// `populate_run_distribution_metrics`'s ext/sidecar re-pool so the typed
    /// read and the cross-run comparison value are byte-identical.
    fn worst_page_locality(&self) -> Option<f64> {
        self.numa_agg_per_cgroup()
            .into_iter()
            .filter(|&(_, total, _)| total > 0)
            .map(|(local, total, _)| super::reductions::page_locality_of(local, total))
            .reduce(f64::min)
    }

    /// The run-level WORST (highest) per-cgroup cross-node migration-churn ratio,
    /// re-pooled from the per-phase NUMA carriers: the MAX
    /// `cross_node_migration_ratio_of(summed_migrated, latest_total)` over cgroups
    /// that measured NUMA residency (`numa_pages_total > 0`) — an all-unmeasured
    /// cohort yields `None`. The polarity twin of `worst_page_locality` (LowerBetter
    /// → highest-wins). Shared by `run_metric`'s `WorstCrossNodeMigrationRatio`
    /// dispatch AND `populate_run_distribution_metrics`'s ext/sidecar re-pool so the
    /// typed read and the cross-run comparison value are byte-identical (the typed
    /// `Gauge(Last)` field diverged from this on multi-phase runs).
    fn worst_cross_node_migration_ratio(&self) -> Option<f64> {
        self.numa_agg_per_cgroup()
            .into_iter()
            .filter(|&(_, total, _)| total > 0)
            .map(|(_, total, migrated)| {
                super::reductions::cross_node_migration_ratio_of(migrated, total)
            })
            .reduce(f64::max)
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
        // Derived metrics (every `is_derived()`: Rate / Distribution / WorstLowest /
        // WakeLatencyTailRatio / WorstCrossNodeRatio / PerPhase) are produced
        // from their pooled components, not folded as per-phase values: skip
        // here. A Rate re-derives after the loop (Σnum/Σdenom over the folded
        // components); the distributional kinds (Distribution / WorstLowest /
        // WakeLatencyTailRatio / WorstCrossNodeRatio) are re-pooled run-level by
        // `populate_run_distribution_metrics` (and never appear in
        // phase.metrics anyway); PerPhase is re-derived by `derive_phase_metrics`.
        // Folding a ready-made derived value would lose
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
        // PerPhaseDeltaSum folds cross-PHASE by SUM: the run-level value is the
        // sum of the disjoint per-phase OBSERVED CPU-time deltas (a lower bound
        // on total run CPU time — excludes head / tail / inter-phase-gap
        // windows; see the MetricKind::PerPhaseDeltaSum doc). It is NOT routed
        // through aggregate_samples_weighted, whose PerPhaseDeltaSum arm
        // implements the CROSS-RUN unweighted mean — mean-folding the phases
        // here would double-count duration (the bug this kind fixes). Weights
        // (phase sample_count) are irrelevant to a sum.
        if def.kind == crate::stats::MetricKind::PerPhaseDeltaSum {
            let sum: f64 = pairs.iter().map(|(v, _)| v).sum();
            target.insert(key.clone(), sum);
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

/// Run the FULL run-level `ext_metrics` population sequence into `stats` — the
/// single source of truth shared by the eval layer (`evaluate_vm_result`) and
/// [`crate::vmm::VmResult::run_metric`], so the two produce byte-identical
/// run-level ext maps for the same run. Reads `samples` + `stats.phases` +
/// `stats.cgroups`, writes `stats.ext_metrics`, in the canonical order:
/// 1. [`populate_run_ext_metrics`] — the `read_sample`-wired registry family
///    (over every freeze in `samples`), then the whole-run wall + IRQ rates.
/// 2. [`populate_run_ext_metrics_from_phases`] — the phase-only ext metrics
///    whose `read_sample` is `None` (`avg_imbalance_ratio`, `iteration_rate`,
///    `system_time_ns`, `user_time_ns`, the per-CPU IRQ spatial maxes, and the
///    per-cgroup PSI-irq spatial maxes).
/// 3. [`populate_run_pooled_iterations_per_cpu_sec`] — the pooled cross-cgroup
///    `iterations_per_cpu_sec` Rate (from `stats.cgroups`).
/// 4. [`populate_run_pooled_taobench`] — the whole-run taobench qps + hit Rates
///    pooled cross-cgroup (from `stats.cgroups[].taobench_whole`).
/// 5. [`populate_run_pooled_schbench`] — the schbench whole-run Class-3 loop
///    Counter + role-separate run-delay gate-Rates (from
///    `stats.phases[].per_cgroup[].schbench` raw pairs, summed over phases+cgroups).
/// 6. [`populate_run_pooled_schbench_distribution`] — the schbench whole-run
///    latency/rps `*_whole` percentiles (union of the per-phase per-cgroup
///    `PlatStats` histograms, percentile re-derived over the union).
/// 7. [`populate_run_distribution_metrics`] — the `Distribution` / `WorstLowest`
///    / `WakeLatencyTailRatio` / `WorstCrossNodeRatio` re-pools (from
///    `stats.phases[].per_cgroup` raw samples + `stats.cgroups`).
///
/// ORDER IS LOAD-BEARING: step 1 must precede step 2 so the whole-run wall +
/// whole-run IRQ-counter deltas land before step 2's `contains_key` skip (the
/// multi-phase-rate fix); steps 3-7 fold the per-cgroup roll-up after the
/// per-phase families.
///
/// `stats.phases` SHOULD be the PRE-`derive_phase_metrics` fold (the host buckets
/// with the guest per-cgroup carriers folded in, before the per-phase scalar
/// derivation) — the exact phase shape the eval layer feeds to step 2, since the
/// eval path runs `derive_phase_metrics` AFTER this call. Feeding the pre-derive
/// fold reproduces the eval sequence by construction (eval-faithful). Post-derive
/// phases (e.g. [`crate::vmm::VmResult::phase_buckets`]) yield the SAME map today —
/// step 2 skips `is_derived` keys, and every pooled scalar `derive_phase_metrics`
/// writes to `bucket.metrics` (the schbench / taobench scalars) is
/// `MetricKind::PerPhase` (which `is_derived`), so step 2 drops them either way —
/// but the pre-derive fold avoids DEPENDING on that skip: a pooled key ever
/// registered as non-derived would be folded run-level under post-derive,
/// diverging from the eval map, but not under pre-derive. Pass the pre-derive
/// fold ([`crate::vmm::VmResult::run_metric`] uses `phase_buckets_pre_derive`).
pub fn populate_run_ext_all(
    stats: &mut ScenarioStats,
    samples: &crate::scenario::sample::SampleSeries,
) {
    populate_run_ext_metrics(samples, &mut stats.ext_metrics);
    populate_run_ext_metrics_from_phases(&stats.phases, &mut stats.ext_metrics);
    populate_run_pooled_iterations_per_cpu_sec(stats);
    populate_run_pooled_taobench(stats);
    populate_run_pooled_schbench(stats);
    populate_run_pooled_schbench_distribution(stats);
    populate_run_distribution_metrics(stats);
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
    // saturating fold: pool the guest-runtime cpu-time-ns / iteration counters
    // across cgroups; a plain `.sum()` would debug-panic / release-wrap on a
    // corrupt/hostile component, silently corrupting iterations_per_cpu_sec.
    let summed_ns: u64 = stats
        .cgroups
        .iter()
        .filter(|c| c.total_cpu_time_ns > 0)
        .map(|c| c.total_cpu_time_ns)
        .fold(0u64, u64::saturating_add);
    if summed_ns == 0 {
        return;
    }
    let summed_iters: u64 = stats
        .cgroups
        .iter()
        .filter(|c| c.total_cpu_time_ns > 0)
        .map(|c| c.total_iterations)
        .fold(0u64, u64::saturating_add);
    stats
        .ext_metrics
        .insert("total_iterations_pooled".to_string(), summed_iters as f64);
    stats
        .ext_metrics
        .insert("total_cpu_time_sec".to_string(), summed_ns as f64 / 1e9);
    crate::stats::derive_rate_metrics(&mut stats.ext_metrics);
}

/// Inject the whole-run taobench engine's qps + hit Rate components into
/// `stats.ext_metrics`, pooled across the run's `WorkType::Taobench` cgroups.
/// Each cgroup carries its workers' merged whole-run aggregate
/// ([`crate::assert::CgroupStats::taobench_whole`]); this folds those across
/// cgroups (Σ ops, MAX wall window — the window is shared by the concurrent
/// cohorts, per `TaobenchStats::merge`) and writes the four
/// `total_taobench_*` Counter components, from which
/// `crate::stats::derive_rate_metrics` derives `taobench_total_ops_per_sec`,
/// `taobench_fast_ops_per_sec`, `taobench_slow_ops_per_sec`, and
/// `taobench_hit_fraction`. The whole-run Rate keys are registered METRICS, so —
/// unlike the per-phase `taobench_*_qps` (`MetricKind::PerPhase`, invisible to
/// the whole-run cross-run fold) — they reach the perf-delta `--noise-adjust`
/// spread analysis.
///
/// MUST run post-`merge` (after every cgroup-bearing merge has populated
/// `stats.cgroups`), exactly like [`populate_run_pooled_iterations_per_cpu_sec`]:
/// an earlier run would pool over an incomplete cgroup set. A run with no
/// Taobench cgroup writes nothing (the pool is `None`) — the keys stay absent,
/// keeping a non-taobench run distinct from a measured zero.
///
/// Both-or-neither (the `derive_rate_metrics` co-location invariant): the four
/// components are inserted together, gated on a measured wall window
/// (`elapsed_ns > 0`) — the qps denominator. `taobench_hit_fraction` =
/// `total_taobench_fast_ops` / `total_taobench_ops` then derives iff ops
/// completed (`total_taobench_ops > 0`); `derive_rate_metrics` skips a
/// zero-denominator rate, so a window-but-no-ops run gets qps=0 keys but no
/// false hit fraction. The ns→s `/1e9` is applied ONCE here (not in
/// `derive_rate_metrics`, a bare num/den), mirroring `total_cpu_time_sec`.
/// Cross-RUN the components SUM-fold (Counter), so each Rate re-pools as
/// Σnumerator / Σdenominator over the cohort — the aggregate throughput, not a
/// mean of per-run qps.
pub fn populate_run_pooled_taobench(stats: &mut ScenarioStats) {
    use crate::stats::{
        TOTAL_TAOBENCH_FAST_OPS, TOTAL_TAOBENCH_OPS, TOTAL_TAOBENCH_SLOW_OPS,
        TOTAL_TAOBENCH_WALL_SEC,
    };
    // Pool the per-cgroup whole-run aggregates across the run's Taobench cgroups:
    // Σ ops, MAX wall window (shared by the concurrent cohorts). `None` when no
    // cgroup ran a Taobench worker.
    let pooled = stats.cgroups.iter().filter_map(|c| c.taobench_whole.as_ref()).fold(
        None,
        |acc: Option<crate::workload::taobench::run::TaobenchStats>, t| {
            Some(match acc {
                Some(mut a) => {
                    a.merge(t);
                    a
                }
                None => *t,
            })
        },
    );
    let Some(w) = pooled else {
        return;
    };
    // qps is undefined without a measured wall window; write no components (so
    // hit_fraction stays absent too) rather than a 0/0 rate.
    if w.elapsed_ns == 0 {
        return;
    }
    stats
        .ext_metrics
        .insert(TOTAL_TAOBENCH_OPS.to_string(), w.total_ops() as f64);
    stats
        .ext_metrics
        .insert(TOTAL_TAOBENCH_FAST_OPS.to_string(), w.fast_ops as f64);
    stats
        .ext_metrics
        .insert(TOTAL_TAOBENCH_SLOW_OPS.to_string(), w.slow_ops as f64);
    stats
        .ext_metrics
        .insert(TOTAL_TAOBENCH_WALL_SEC.to_string(), w.elapsed_ns as f64 / 1e9);
    crate::stats::derive_rate_metrics(&mut stats.ext_metrics);
}

/// Inject the schbench whole-run Class-3 metrics — the loop Counter and the
/// role-separate run-delay gate-Rate components — into `stats.ext_metrics`,
/// summed across EVERY phase and EVERY cgroup from the per-phase
/// `SchbenchPhaseStats` raw pairs (`stats.phases[].per_cgroup[].schbench`). The
/// raw `(run_delay_ns, pcount)` pairs and `loop_count` are integer and
/// associative, so summing across phases+cgroups gives the run-level totals; the
/// two `*_run_delay_ns_per_sched` Rates then re-derive Σrun_delay/Σpcount (the
/// sample-weighted per-schedule mean — NOT a mean of per-run means). The MESSAGE
/// and WORKER thread roles pool SEPARATELY (different per-schedule wait
/// populations — never cross-pool).
///
/// Runs in [`populate_run_ext_all`] (post-merge, after
/// [`populate_run_pooled_taobench`]); reads the per-phase carriers (a disjoint
/// source from the iterations/taobench pools) and writes distinct
/// `total_schbench_*` / `schbench_*_run_delay_ns_per_sched` keys, so it is
/// order-independent. A run with no schbench carrier writes nothing (keys stay
/// absent — a non-schbench run is distinct from a measured zero).
///
/// Both-or-neither PER ROLE: each role's two Counter components are inserted only
/// when that role was scheduled (`pcount > 0`), so `derive_rate_metrics` yields
/// the role's gate-Rate iff it ran (never a 0/0); the two roles are independent
/// (a worker-only run emits only the worker Rate). `total_schbench_loops` is
/// always written when any schbench carrier ran (0 is a measured zero). Distinct
/// from the per-phase `sched_delay_msg/worker_us` (mean-of-means parity,
/// PerPhase display-only) — these Rates gate; no double-count. Cross-RUN the
/// components SUM-fold (Counter), so each Rate re-pools Σrun_delay/Σpcount.
pub fn populate_run_pooled_schbench(stats: &mut ScenarioStats) {
    use crate::stats::{
        TOTAL_SCHBENCH_LOOPS, TOTAL_SCHBENCH_MSG_PCOUNT, TOTAL_SCHBENCH_MSG_RUN_DELAY_NS,
        TOTAL_SCHBENCH_WORKER_PCOUNT, TOTAL_SCHBENCH_WORKER_RUN_DELAY_NS,
    };
    let mut msg_run_delay_ns: u64 = 0;
    let mut msg_pcount: u64 = 0;
    let mut worker_run_delay_ns: u64 = 0;
    let mut worker_pcount: u64 = 0;
    let mut loops: u64 = 0;
    let mut any = false;
    for phase in &stats.phases {
        for pc in phase.per_cgroup.values() {
            if let Some(s) = pc.schbench.as_ref() {
                any = true;
                // saturating: guest-runtime run-delay-ns / pcount / loop
                // counters pooled across phases+cgroups (matches the already-
                // saturating SchbenchPhaseStats::merge); never wrap a gate-Rate.
                msg_run_delay_ns = msg_run_delay_ns.saturating_add(s.msg_run_delay_ns);
                msg_pcount = msg_pcount.saturating_add(s.msg_pcount);
                worker_run_delay_ns = worker_run_delay_ns.saturating_add(s.worker_run_delay_ns);
                worker_pcount = worker_pcount.saturating_add(s.worker_pcount);
                loops = loops.saturating_add(s.loop_count);
            }
        }
    }
    if !any {
        return;
    }
    stats
        .ext_metrics
        .insert(TOTAL_SCHBENCH_LOOPS.to_string(), loops as f64);
    if msg_pcount > 0 {
        stats
            .ext_metrics
            .insert(TOTAL_SCHBENCH_MSG_RUN_DELAY_NS.to_string(), msg_run_delay_ns as f64);
        stats
            .ext_metrics
            .insert(TOTAL_SCHBENCH_MSG_PCOUNT.to_string(), msg_pcount as f64);
    }
    if worker_pcount > 0 {
        stats.ext_metrics.insert(
            TOTAL_SCHBENCH_WORKER_RUN_DELAY_NS.to_string(),
            worker_run_delay_ns as f64,
        );
        stats
            .ext_metrics
            .insert(TOTAL_SCHBENCH_WORKER_PCOUNT.to_string(), worker_pcount as f64);
    }
    crate::stats::derive_rate_metrics(&mut stats.ext_metrics);
}

/// Inject the schbench whole-run DISTRIBUTIONAL metrics (the wakeup /
/// request latency percentiles + min/max and the achieved-rps percentiles) into
/// `stats.ext_metrics` as the `*_whole` keys, re-pooled run-level by UNIONING the
/// per-phase per-cgroup `PlatStats` histograms
/// (`stats.phases[].per_cgroup[].schbench.{wakeup,request,rps}`) across EVERY
/// phase and EVERY cgroup, then re-deriving each percentile / min / max over the
/// merged histogram. `PlatStats::combine` is an associative bucket-count add, so
/// the merged histogram is the FAITHFUL union and the re-derived percentile is
/// the percentile OF the pooled sample set — NOT a mean of per-phase / per-cgroup
/// percentiles (the percentile operator is non-linear). This is the schbench
/// histogram analog of [`populate_run_distribution_metrics`]'s raw-sample union.
///
/// Runs in [`populate_run_ext_all`] (post-merge, after
/// [`populate_run_pooled_schbench`]); reads the per-phase carriers (disjoint from
/// the iterations / taobench / schbench loop/run-delay pools) and writes distinct
/// `*_whole` keys, so it is order-independent. Each stream's keys are written
/// only when its merged histogram has samples (`sample_count() > 0`) — a stream
/// with no samples (e.g. a sub-1s run with no rps samples) reads ABSENT, never a
/// false 0 (mirrors the per-phase `write_schbench_scalars` gating and the
/// carrier-less graceful degradation of [`populate_run_distribution_metrics`]).
///
/// The `*_whole` keys are `crate::stats::MetricKind::PerRunDistribution`:
/// noise-compared per-run by `crate::stats::noise_findings` (each run's own p99),
/// NEVER cross-RUN folded (a percentile of a union is not a mean of per-run
/// percentiles, and the per-phase histograms are dropped at the cross-run
/// boundary), so they are gated out of the cross-RUN ext fold and the within-run
/// reducers (`is_derived`). Distinct names from the per-phase percentile keys
/// (one registry name = one kind), produced solely here.
pub fn populate_run_pooled_schbench_distribution(stats: &mut ScenarioStats) {
    use crate::stats::{
        SCHBENCH_REQUEST_MAX_US_WHOLE, SCHBENCH_REQUEST_MIN_US_WHOLE,
        SCHBENCH_REQUEST_P50_US_WHOLE, SCHBENCH_REQUEST_P90_US_WHOLE,
        SCHBENCH_REQUEST_P99_US_WHOLE, SCHBENCH_REQUEST_P999_US_WHOLE, SCHBENCH_RPS_MAX_WHOLE,
        SCHBENCH_RPS_MIN_WHOLE, SCHBENCH_RPS_P20_WHOLE, SCHBENCH_RPS_P50_WHOLE,
        SCHBENCH_RPS_P90_WHOLE, SCHBENCH_WAKEUP_MAX_US_WHOLE, SCHBENCH_WAKEUP_MIN_US_WHOLE,
        SCHBENCH_WAKEUP_P50_US_WHOLE, SCHBENCH_WAKEUP_P90_US_WHOLE, SCHBENCH_WAKEUP_P99_US_WHOLE,
        SCHBENCH_WAKEUP_P999_US_WHOLE,
    };
    use crate::workload::schbench::plat::{PlatStats, Pct};

    // Union the per-stream histograms across ALL phases+cgroups (combine =
    // associative bucket-count add → the faithful pooled histogram).
    let mut wakeup = PlatStats::default();
    let mut request = PlatStats::default();
    let mut rps = PlatStats::default();
    for phase in &stats.phases {
        for pc in phase.per_cgroup.values() {
            if let Some(s) = pc.schbench.as_ref() {
                wakeup.combine(&s.wakeup);
                request.combine(&s.request);
                rps.combine(&s.rps);
            }
        }
    }
    // Latency streams: 4 percentiles + min/max, re-derived over the union, µs.
    if wakeup.sample_count() > 0 {
        let q = wakeup.percentiles();
        stats
            .ext_metrics
            .insert(SCHBENCH_WAKEUP_P50_US_WHOLE.to_string(), q.value_at(Pct::P50) as f64);
        stats
            .ext_metrics
            .insert(SCHBENCH_WAKEUP_P90_US_WHOLE.to_string(), q.value_at(Pct::P90) as f64);
        stats
            .ext_metrics
            .insert(SCHBENCH_WAKEUP_P99_US_WHOLE.to_string(), q.value_at(Pct::P99) as f64);
        stats.ext_metrics.insert(
            SCHBENCH_WAKEUP_P999_US_WHOLE.to_string(),
            q.value_at(Pct::P999) as f64,
        );
        stats
            .ext_metrics
            .insert(SCHBENCH_WAKEUP_MIN_US_WHOLE.to_string(), q.min as f64);
        stats
            .ext_metrics
            .insert(SCHBENCH_WAKEUP_MAX_US_WHOLE.to_string(), q.max as f64);
    }
    if request.sample_count() > 0 {
        let q = request.percentiles();
        stats.ext_metrics.insert(
            SCHBENCH_REQUEST_P50_US_WHOLE.to_string(),
            q.value_at(Pct::P50) as f64,
        );
        stats.ext_metrics.insert(
            SCHBENCH_REQUEST_P90_US_WHOLE.to_string(),
            q.value_at(Pct::P90) as f64,
        );
        stats.ext_metrics.insert(
            SCHBENCH_REQUEST_P99_US_WHOLE.to_string(),
            q.value_at(Pct::P99) as f64,
        );
        stats.ext_metrics.insert(
            SCHBENCH_REQUEST_P999_US_WHOLE.to_string(),
            q.value_at(Pct::P999) as f64,
        );
        stats
            .ext_metrics
            .insert(SCHBENCH_REQUEST_MIN_US_WHOLE.to_string(), q.min as f64);
        stats
            .ext_metrics
            .insert(SCHBENCH_REQUEST_MAX_US_WHOLE.to_string(), q.max as f64);
    }
    // RPS stream: PLIST_FOR_RPS = 20/50/90 + min/max (the schbench rps table).
    if rps.sample_count() > 0 {
        let r = rps.percentiles();
        stats
            .ext_metrics
            .insert(SCHBENCH_RPS_P20_WHOLE.to_string(), r.value_at(Pct::P20) as f64);
        stats
            .ext_metrics
            .insert(SCHBENCH_RPS_P50_WHOLE.to_string(), r.value_at(Pct::P50) as f64);
        stats
            .ext_metrics
            .insert(SCHBENCH_RPS_P90_WHOLE.to_string(), r.value_at(Pct::P90) as f64);
        stats
            .ext_metrics
            .insert(SCHBENCH_RPS_MIN_WHOLE.to_string(), r.min as f64);
        stats
            .ext_metrics
            .insert(SCHBENCH_RPS_MAX_WHOLE.to_string(), r.max as f64);
    }
}

/// Populate run-level DERIVED distributional metrics into
/// `stats.ext_metrics`: every registered `MetricKind::Distribution`,
/// `MetricKind::WorstLowest`, `MetricKind::WakeLatencyTailRatio`, and
/// `MetricKind::WorstCrossNodeRatio`. This is the SOLE
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
    // Distinct timer-latency pool, population-WEIGHTED like
    // wake_pool (reservoir-capped, so a >cap phase carries weight
    // timer_sample_total/len for the cross-phase de-skew).
    let mut timer_pool: Vec<(u64, f64)> = Vec::new();
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
    let mut timer_carriers: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
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
            if !pcg.timer_latencies_ns.is_empty() {
                // Population-weighted exactly like the wake pool: a >cap phase's
                // capped samples each stand for timer_sample_total/len true
                // wakes, restoring the cross-phase population proportion.
                let len = pcg.timer_latencies_ns.len() as u64;
                debug_assert!(
                    pcg.timer_sample_total >= len,
                    "timer_sample_total ({}) < reservoir len ({}): malformed carrier",
                    pcg.timer_sample_total,
                    len,
                );
                let w = pcg.timer_sample_total.max(len) as f64 / len as f64;
                timer_pool.extend(pcg.timer_latencies_ns.iter().map(|&v| (v, w)));
                timer_carriers.insert(cgname.as_str());
            }
            if !pcg.run_delays_ns.is_empty() {
                run_delay_pool.extend_from_slice(&pcg.run_delays_ns);
                run_delay_carriers.insert(cgname.as_str());
            }
        }
    }
    wake_pool.sort_unstable_by_key(|&(v, _)| v);
    timer_pool.sort_unstable_by_key(|&(v, _)| v);
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
        &timer_pool,
        &timer_carriers,
        &run_delay_pool,
        &run_delay_carriers,
        &stats.cgroups,
        stats.total_iterations,
    );
    // worst_page_locality (WorstLowest{NumaLocal,NumaTotal}) re-pools from the
    // per-phase NUMA carriers, which the _from helper above cannot reach (it
    // takes only stats.cgroups, and the reports-only CgroupStats hardcodes
    // page_locality 0.0). Single-sourced with run_metric's WorstPageLocality via
    // worst_page_locality(): the lowest per-cgroup page_locality over cgroups
    // that measured NUMA residency (a measured 0.0 — all off-node — winning); an
    // all-unmeasured cohort writes no key (absence preserved as a missing ext
    // entry, never a 0.0 sentinel). numa_agg_per_cgroup()'s borrow ends before
    // the insert.
    if let Some(v) = stats.worst_page_locality() {
        stats
            .ext_metrics
            .insert("worst_page_locality".to_string(), v);
    }
    // worst_cross_node_migration_ratio (WorstCrossNodeRatio) re-pools from the same
    // per-phase NUMA carriers — the MAX per-cgroup churn ratio over the latest
    // residency total — single-sourced with run_metric's WorstCrossNodeMigrationRatio
    // via worst_cross_node_migration_ratio(); an all-unmeasured cohort writes no key
    // (absence preserved). Not handled by the _from helper above (it has only
    // stats.cgroups, and the metric is excluded from that filter).
    if let Some(v) = stats.worst_cross_node_migration_ratio() {
        stats
            .ext_metrics
            .insert("worst_cross_node_migration_ratio".to_string(), v);
    }
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
    timer_pool: &[(u64, f64)],
    timer_carriers: &std::collections::BTreeSet<&str>,
    run_delay_pool: &[u64],
    run_delay_carriers: &std::collections::BTreeSet<&str>,
    cgroups: &[CgroupStats],
    run_total_iterations: u64,
) {
    use crate::stats::{MetricKind, SampleSource, WorstLowestDenominator, WorstLowestNumerator};
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
                // CONTRACT (now uniform with WorstLowest and WakeLatencyTailRatio
                // below): a cohort with NO measurement for this source anywhere
                // (empty carrier pool AND every non-carrier cgroup not-measured)
                // yields ABSENCE (None), not Some(0.0). A percentile / mean over
                // zero samples is undefined, not a measured zero — folding its
                // 0.0 sentinel into the cross-run mean would, for the LowerBetter
                // wake/timer/run-delay metrics, falsely drag the mean toward
                // "perfect". The non-carrier fold below is gated on
                // `cg.measured_for(source)` so an unmeasured cgroup contributes
                // nothing; a non-carrier cgroup that DID measure (a genuine
                // measured zero, e.g. workers that queued for no time) still
                // contributes its real value. This matches WorstLowest (None when
                // every iterations_per_*() is None) and WakeLatencyTailRatio (None
                // when no cgroup has a tail) — every distributional kind now emits
                // None for a no-measurement run.
                let (mut v, carriers): (Option<f64>, &std::collections::BTreeSet<&str>) =
                    match source {
                        SampleSource::WakeLatencyNs => (
                            (!wake_pool.is_empty())
                                .then(|| reduce_weighted_sorted_distribution(wake_pool, reduction)),
                            wake_carriers,
                        ),
                        SampleSource::TimerLatencyNs => (
                            (!timer_pool.is_empty()).then(|| {
                                reduce_weighted_sorted_distribution(timer_pool, reduction)
                            }),
                            timer_carriers,
                        ),
                        SampleSource::RunDelayNs => (
                            (!run_delay_pool.is_empty())
                                .then(|| reduce_sorted_distribution(run_delay_pool, reduction)),
                            run_delay_carriers,
                        ),
                    };
                for cg in cgroups {
                    if !carriers.contains(cg.cgroup_name.as_str()) && cg.measured_for(source) {
                        let r = distribution_cgroup_reduction(cg, source, reduction);
                        v = Some(v.map_or(r, |acc| acc.max(r)));
                    }
                }
                v
            }
            // The iterations-efficiency WorstLowest selectors (numerator
            // Iterations) re-pool from the `stats.cgroups` counters here; the
            // denominator picks the per-cgroup efficiency method. The
            // page-locality selector (numerator NumaLocal) re-pools instead from
            // the per-phase NUMA carriers in populate_run_distribution_metrics
            // (this fn has only stats.cgroups, not stats.phases), so it is
            // skipped here (the NumaLocal arm below returns None).
            //
            // In a MULTI-STEP scenario `AssertResult::merge` extends
            // `stats.cgroups` per (handle, step), so the same cgroup name
            // appears once per step; this selects the lowest single
            // (handle, step) entry, NOT a per-name whole-run efficiency. That
            // preserves the deleted `fold_lowest_some` granularity exactly and
            // mirrors `populate_run_pooled_iterations_per_cpu_sec`, which sums
            // over the same per-(handle, step) entries.
            MetricKind::WorstLowest {
                numerator: WorstLowestNumerator::Iterations,
                denominator,
            } => {
                let mut worst: Option<f64> = None;
                for cg in cgroups {
                    let per_cg = match denominator {
                        WorstLowestDenominator::NumWorkers => cg.iterations_per_worker(),
                        WorstLowestDenominator::CpuTimeNs => cg.iterations_per_cpu_sec(),
                        // NumaTotal pairs only with the NumaLocal numerator
                        // (handled in the NumaLocal arm below) — never with
                        // Iterations.
                        WorstLowestDenominator::NumaTotal => None,
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
            // page-locality (numerator NumaLocal): re-pooled from the per-phase
            // NUMA carriers in populate_run_distribution_metrics, not here.
            MetricKind::WorstLowest {
                numerator: WorstLowestNumerator::NumaLocal,
                ..
            } => None,
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
        SampleReduction::P999 => percentile(sorted, 0.999) as f64 / 1000.0,
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
        SampleReduction::P999 => weighted_percentile(sorted, 0.999) as f64 / 1000.0,
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
            SampleReduction::P999 | SampleReduction::Mean | SampleReduction::Worst => {
                debug_assert!(false, "no CgroupStats wake reduction for {reduction:?}");
                f64::NAN
            }
        },
        SampleSource::RunDelayNs => match reduction {
            SampleReduction::Mean => cg.mean_run_delay_us,
            SampleReduction::Worst => cg.worst_run_delay_us,
            SampleReduction::P99
            | SampleReduction::P999
            | SampleReduction::Median
            | SampleReduction::Cv => {
                debug_assert!(
                    false,
                    "no CgroupStats run-delay reduction for {reduction:?}"
                );
                f64::NAN
            }
        },
        SampleSource::TimerLatencyNs => match reduction {
            SampleReduction::Median => cg.median_timer_latency_us,
            SampleReduction::P99 => cg.p99_timer_latency_us,
            SampleReduction::P999 => cg.p999_timer_latency_us,
            SampleReduction::Worst => cg.worst_timer_latency_us,
            SampleReduction::Cv | SampleReduction::Mean => {
                debug_assert!(false, "no CgroupStats timer reduction for {reduction:?}");
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
    // Run-level capture-window wall for the IRQ rates: the elapsed span of the
    // per_cpu_time-bearing freezes — the SAME freezes the IRQ-counter numerators
    // above were read_sampled from (whole-run last-minus-first via
    // phase_counter_delta). Inserting it HERE (the direct/whole-run path) makes
    // the run-level rate's numerator AND denominator share the whole-run span.
    // Without it, derive_rate_metrics below finds the numerator but no
    // denominator (total_phase_wall_sec has no read_sample arm), and the rate
    // would instead be derived by populate_run_ext_metrics_from_phases over a
    // Σ-per-phase-capture denominator — a NARROWER time base than this whole-run
    // numerator (it excludes the inter-phase / cross-capture gaps the numerator
    // counts), inflating the rate on MULTI-phase runs. Run-level analog of the
    // per-phase assert::phase_build. Gated on total_hardirqs
    // (irqtime-independent), so the count-rates derive even when
    // CONFIG_IRQ_TIME_ACCOUNTING is off and total_irq_time_ns is absent. Min/max
    // over the elapsed-bearing freezes: chronological periodic samples put the
    // numerator's first/last reading at min/max elapsed, so num and den span the
    // identical interval.
    if target.contains_key("total_hardirqs") {
        let irq_elapsed_ms: Vec<u64> = samples
            .iter_samples()
            .filter(|s| !s.snapshot.per_cpu_time().is_empty())
            .filter_map(|s| s.elapsed_ms)
            .collect();
        if let (Some(first), Some(last)) = (
            irq_elapsed_ms.iter().min().copied(),
            irq_elapsed_ms.iter().max().copied(),
        ) && last > first
        {
            let wall_ms = (last - first) as f64;
            target
                .entry("total_phase_wall_ns".to_string())
                .or_insert(wall_ms * 1_000_000.0);
            target
                .entry("total_phase_wall_sec".to_string())
                .or_insert(wall_ms / 1000.0);
        }
    }
    // Re-derive Rate metrics from the read_sample components just folded
    // in. populate_run_ext_metrics is pub and called standalone (tests,
    // and not only ahead of populate_run_ext_metrics_from_phases), so it
    // derives its own rates to stay self-contained.
    crate::stats::derive_rate_metrics(target);
}

/// Derive the per-phase scalar metrics ([`crate::stats::MetricKind::PerPhase`])
/// for every cgroup carrier in each phase. Two families:
///
/// 1. NON-schbench carrier scalars (via [`write_carrier_scalars`], for EVERY
///    cgroup regardless of work type): the wake/run-delay/off-cpu distributions
///    + the migration/iterations/locality ratios + the carrier counters,
///    written ONLY into each carrier's `PhaseCgroupStats::metrics` (per-cgroup,
///    read via `phase_cgroup_metric`) — NO pooled [`PhaseBucket::metrics`]
///    entry; their run-level aggregate is the `worst_*` ext-metrics key.
/// 2. schbench scalars (via [`write_schbench_scalars`], only for cgroups
///    carrying a `SchbenchPhaseStats`): written per-cgroup into
///    `PhaseCgroupStats::metrics` AND, pooled across the phase's schbench
///    carriers, into [`PhaseBucket::metrics`] (read via `phase_metric`).
///    Percentiles MUST come from the POOLED histogram (combine the latency
///    histograms + integer-add the run-delay raw pairs), never an average of
///    per-cgroup percentiles; the run-delay means are ABSENT when `pcount == 0`
///    so a never-scheduled class is not a false `0`.
///
/// MUST run POST-merge: the buckets are final and keyed by `step_index`, so a
/// per-phase A/B claim reads the value via `phase_metric` /
/// `phase_cgroup_metric`. It runs after the per-cgroup carriers are folded
/// ([`fold_guest_per_cgroup_into_host_buckets`]) and after any
/// [`merge_matched_phase_buckets`] — both SKIP is_derived keys (the `continue`
/// in the merge loop), so deriving earlier would DROP the keys. Idempotent
/// (overwrites the keys), so callers that rebuild buckets per call (e.g.
/// `VmResult::phase_buckets`) may re-run it freely.
// doc_lazy_continuation: pre-existing numbered-list wording surfaced by the clippy
// 1.94 bump; renders fine. Suppress rather than reflow the prose.
#[allow(clippy::doc_lazy_continuation)]
pub(crate) fn derive_phase_metrics(phases: &mut [PhaseBucket]) {
    use crate::workload::schbench::run::SchbenchPhaseStats;

    for bucket in phases.iter_mut() {
        // Derive the schbench scalars BOTH per cgroup (into pc.metrics — the
        // per-cgroup set: N cgroups -> N queryable sets) AND pooled across the
        // phase's cgroups (into bucket.metrics — the aggregate). Both go through the
        // SAME reducer (write_schbench_scalars) from the SAME carriers, so the
        // pooled set is the cross-cgroup re-pool of the per-cgroup carriers (a
        // percentile re-derives from the pooled histogram via PlatStats::combine =
        // bucket-count add, never averaged), and pooled == cross-cgroup re-pool.
        let mut pooled: Option<SchbenchPhaseStats> = None;
        let mut pooled_taobench: Option<crate::workload::taobench::run::TaobenchStats> = None;
        for pc in bucket.per_cgroup.values_mut() {
            // Non-schbench carrier-derived metrics (wake/run-delay/off-cpu
            // distributions + the migration/iterations/locality ratios + the
            // carrier counters), emitted for EVERY cgroup regardless of work
            // type. Runs BEFORE the schbench block: `write_carrier_scalars`
            // takes `&mut pc` (its summaries read many `pc` fields, then it
            // writes `pc.metrics`), so it cannot share the iteration with the
            // disjoint-field borrow the schbench block relies on.
            write_carrier_scalars(pc);
            // Disjoint field borrows: `s` reads pc.schbench, the reducer writes
            // pc.metrics — different fields, so the immutable + mutable borrows of
            // `*pc` coexist.
            if let Some(s) = pc.schbench.as_ref() {
                write_schbench_scalars(s, &mut pc.metrics);
                // `take()` completes the borrow before the reassignment in both arms.
                match pooled.take() {
                    Some(mut acc) => {
                        acc.merge(s);
                        pooled = Some(acc);
                    }
                    None => pooled = Some(s.clone()),
                }
            }
            // taobench per-phase: same disjoint-field-borrow pattern (`t` reads
            // pc.taobench, the reducer writes pc.metrics) + pool across cgroups.
            if let Some(t) = pc.taobench.as_ref() {
                write_taobench_scalars(t, &mut pc.metrics);
                match pooled_taobench.take() {
                    Some(mut acc) => {
                        acc.merge(t);
                        pooled_taobench = Some(acc);
                    }
                    None => pooled_taobench = Some(*t),
                }
            }
        }
        if let Some(p) = pooled {
            write_schbench_scalars(&p, &mut bucket.metrics);
        }
        if let Some(t) = pooled_taobench {
            write_taobench_scalars(&t, &mut bucket.metrics);
        }
    }
}

/// Write the per-phase per-cgroup NON-schbench DERIVED scalars from ONE
/// carrier into `pc.metrics`, keyed by registry [`crate::stats::MetricDef`]
/// name — the per-cgroup analog of the run-level reductions, single-sourced
/// through the shared ratio helpers in [`crate::assert::reductions`] (so the
/// per-cgroup value and the run-level `worst_*` fold agree by construction)
/// and the carrier summary methods. Each gate mirrors the carrier's ABSENT
/// discipline: an empty wake pool, a not-measured off-CPU state, a
/// zero-worker cgroup, or a no-NUMA-pages cgroup emits no key (reads as
/// missing, never a false 0); `migration_ratio` is ALWAYS present (a measured
/// 0.0 when no iterations ran).
///
/// Unlike the schbench families, these families have NO pooled
/// `PhaseBucket::metrics` entry at all: as PerPhase / is_derived metrics whose
/// `read_sample` is `None`, the per-phase fold never writes them to
/// `bucket.metrics`. Their run-level aggregate is the `worst_*` key
/// `populate_run_distribution_metrics` writes into the run-level
/// `ScenarioStats::ext_metrics` map; per-phase they are queryable ONLY per-cgroup
/// via `pc.metrics`. So this writes ONLY `pc.metrics`, never `bucket.metrics` —
/// the per-cgroup carrier value is authoritative for the per-cgroup question.
///
/// Takes `&mut PhaseCgroupStats` (not `(&pc, &mut pc.metrics)`): it reads many
/// `pc` fields AND writes `pc.metrics`, so the summaries/ratios are bound to
/// locals from `&*pc` first, then the `&mut pc.metrics` borrow is taken once.
fn write_carrier_scalars(pc: &mut PhaseCgroupStats) {
    use crate::assert::reductions::{
        cross_node_migration_ratio_of, iterations_per_cpu_sec_of, iterations_per_worker_of,
        migration_ratio_of, page_locality_of,
    };
    // Read everything from &*pc into locals BEFORE the &mut pc.metrics borrow.
    let wake = pc.wake_summary().zip(pc.wake_cv());
    let timer = pc.timer_summary();
    let run_delay = pc.run_delay_summary();
    let off_cpu = pc.off_cpu_summary();
    let migration_ratio = migration_ratio_of(pc.total_migrations, pc.total_iterations);
    let ipw = iterations_per_worker_of(pc.num_workers, pc.total_iterations);
    let ipcs = iterations_per_cpu_sec_of(pc.num_workers, pc.total_cpu_time_ns, pc.total_iterations);
    let numa = if pc.numa_pages_total > 0 {
        Some((
            page_locality_of(pc.numa_pages_local, pc.numa_pages_total),
            cross_node_migration_ratio_of(pc.cross_node_migrated, pc.numa_pages_total),
        ))
    } else {
        None
    };

    let m = &mut pc.metrics;
    // WAKE — all three or none (wake_summary + wake_cv share the empty-pool gate).
    if let Some(((p99, median), cv)) = wake {
        m.insert("p99_wake_latency_us".to_string(), p99);
        m.insert("median_wake_latency_us".to_string(), median);
        m.insert("wake_latency_cv".to_string(), cv);
    }
    // TIMER — all three or none (timer_summary shares the empty-pool gate).
    if let Some((median, p99, p999)) = timer {
        m.insert("median_timer_latency_us".to_string(), median);
        m.insert("p99_timer_latency_us".to_string(), p99);
        m.insert("p999_timer_latency_us".to_string(), p999);
    }
    // RUN-DELAY — both or none.
    if let Some((mean, worst)) = run_delay {
        m.insert("mean_run_delay_us".to_string(), mean);
        m.insert("max_run_delay_us".to_string(), worst);
    }
    // OFF-CPU — four or none (None == not-measured).
    if let Some((avg, min, max, spread)) = off_cpu {
        m.insert("avg_off_cpu_pct".to_string(), avg);
        m.insert("min_off_cpu_pct".to_string(), min);
        m.insert("max_off_cpu_pct".to_string(), max);
        m.insert("off_cpu_spread_pct".to_string(), spread);
    }
    // RATIOS — migration_ratio ALWAYS (measured 0.0 when no iterations ran).
    m.insert("migration_ratio".to_string(), migration_ratio);
    if let Some(v) = ipw {
        m.insert("iterations_per_worker".to_string(), v);
    }
    if let Some(v) = ipcs {
        m.insert("iterations_per_cpu_sec".to_string(), v);
    }
    // NUMA ratios only when pages were observed (mirrors the carrier's
    // numa_pages_total>0 gate; absent is more honest than a 0.0 default).
    if let Some((locality, cross_node)) = numa {
        m.insert("page_locality".to_string(), locality);
        m.insert("cross_node_migration_ratio".to_string(), cross_node);
    }
}

/// Write the per-phase schbench scalar metrics derived from ONE
/// `SchbenchPhaseStats` (a single cgroup's carrier, or the cross-cgroup pool)
/// into `out`, keyed by registry [`crate::stats::MetricDef`] name. The sole
/// producer of these keys for both the per-cgroup ([`PhaseCgroupStats::metrics`])
/// and pooled ([`PhaseBucket::metrics`]) maps — one derivation, no duplicated
/// percentile / min/max / rps / sched-delay math. Each gate mirrors the carrier's
/// ABSENT discipline: an empty histogram or a zero pcount emits no key (reads as
/// missing, never a false 0); loop_count is always present (0 = a real measured
/// no-cycles outcome).
fn write_schbench_scalars(
    p: &crate::workload::schbench::run::SchbenchPhaseStats,
    out: &mut std::collections::BTreeMap<String, f64>,
) {
    use crate::stats::{
        SCHBENCH_LOOP_COUNT, SCHBENCH_REQUEST_MAX_US, SCHBENCH_REQUEST_MIN_US,
        SCHBENCH_REQUEST_P50_US, SCHBENCH_REQUEST_P90_US, SCHBENCH_REQUEST_P99_US,
        SCHBENCH_REQUEST_P999_US, SCHBENCH_RPS_MAX, SCHBENCH_RPS_MIN, SCHBENCH_RPS_P20,
        SCHBENCH_RPS_P50, SCHBENCH_RPS_P90, SCHBENCH_SCHED_DELAY_MSG_US,
        SCHBENCH_SCHED_DELAY_WORKER_US, SCHBENCH_WAKEUP_MAX_US, SCHBENCH_WAKEUP_MIN_US,
        SCHBENCH_WAKEUP_P50_US, SCHBENCH_WAKEUP_P90_US, SCHBENCH_WAKEUP_P99_US,
        SCHBENCH_WAKEUP_P999_US,
    };
    use crate::workload::schbench::plat::Pct;

    if p.wakeup.sample_count() > 0 {
        let q = p.wakeup.percentiles();
        out.insert(
            SCHBENCH_WAKEUP_P50_US.to_string(),
            q.value_at(Pct::P50) as f64,
        );
        out.insert(
            SCHBENCH_WAKEUP_P90_US.to_string(),
            q.value_at(Pct::P90) as f64,
        );
        out.insert(
            SCHBENCH_WAKEUP_P99_US.to_string(),
            q.value_at(Pct::P99) as f64,
        );
        out.insert(
            SCHBENCH_WAKEUP_P999_US.to_string(),
            q.value_at(Pct::P999) as f64,
        );
        out.insert(SCHBENCH_WAKEUP_MIN_US.to_string(), q.min as f64);
        out.insert(SCHBENCH_WAKEUP_MAX_US.to_string(), q.max as f64);
    }
    if p.request.sample_count() > 0 {
        let q = p.request.percentiles();
        out.insert(
            SCHBENCH_REQUEST_P50_US.to_string(),
            q.value_at(Pct::P50) as f64,
        );
        out.insert(
            SCHBENCH_REQUEST_P90_US.to_string(),
            q.value_at(Pct::P90) as f64,
        );
        out.insert(
            SCHBENCH_REQUEST_P99_US.to_string(),
            q.value_at(Pct::P99) as f64,
        );
        out.insert(
            SCHBENCH_REQUEST_P999_US.to_string(),
            q.value_at(Pct::P999) as f64,
        );
        out.insert(SCHBENCH_REQUEST_MIN_US.to_string(), q.min as f64);
        out.insert(SCHBENCH_REQUEST_MAX_US.to_string(), q.max as f64);
    }
    // Per-phase achieved-RPS distribution (the control thread's per-second samples
    // attributed to this epoch). Gated on sample_count()>0 so a phase shorter than
    // the ~1s control cadence reads ABSENT, never rps=0. schbench's RPS table is
    // PLIST_FOR_RPS = 20/50/90 (schbench.c:130) + min/max.
    if p.rps.sample_count() > 0 {
        let r = p.rps.percentiles();
        out.insert(SCHBENCH_RPS_P20.to_string(), r.value_at(Pct::P20) as f64);
        out.insert(SCHBENCH_RPS_P50.to_string(), r.value_at(Pct::P50) as f64);
        out.insert(SCHBENCH_RPS_P90.to_string(), r.value_at(Pct::P90) as f64);
        out.insert(SCHBENCH_RPS_MIN.to_string(), r.min as f64);
        out.insert(SCHBENCH_RPS_MAX.to_string(), r.max as f64);
    }
    // Sample-weighted run-delay mean (Σrun_delay / Σpcount), ns→µs; ABSENT when
    // pcount==0 (a never-scheduled class) so it reads as missing, not 0.
    if p.msg_pcount > 0 {
        let mean_us = p.msg_run_delay_ns as f64 / p.msg_pcount as f64 / 1000.0;
        out.insert(SCHBENCH_SCHED_DELAY_MSG_US.to_string(), mean_us);
    }
    if p.worker_pcount > 0 {
        let mean_us = p.worker_run_delay_ns as f64 / p.worker_pcount as f64 / 1000.0;
        out.insert(SCHBENCH_SCHED_DELAY_WORKER_US.to_string(), mean_us);
    }
    // loop_count is always present for a schbench carrier: 0 = no cycles ran (a real
    // measured value; HigherBetter → worst), distinct from a non-schbench carrier
    // which has no schbench data at all (the caller skips it).
    out.insert(SCHBENCH_LOOP_COUNT.to_string(), p.loop_count as f64);
}

/// Write the per-phase taobench scalar metrics derived from ONE
/// `TaobenchStats` (a single cgroup's carrier, or the cross-cgroup pool)
/// into `out`, keyed by registry [`crate::stats::MetricDef`] name. The sole
/// producer of these keys for both the per-cgroup ([`PhaseCgroupStats::metrics`])
/// and pooled ([`PhaseBucket::metrics`]) maps. ABSENT discipline: the qps keys
/// only when the wall window was measured (`elapsed_ns > 0`); hit_ratio only when
/// ops completed (`total > 0`); hit_rate only when lookups were issued
/// (`get_cmds > 0`) — a not-measured value reads as missing, never a false 0.
fn write_taobench_scalars(
    p: &crate::workload::taobench::run::TaobenchStats,
    out: &mut std::collections::BTreeMap<String, f64>,
) {
    use crate::stats::{
        TAOBENCH_FAST_QPS, TAOBENCH_HIT_RATE, TAOBENCH_HIT_RATIO, TAOBENCH_SLOW_QPS,
        TAOBENCH_TOTAL_QPS,
    };
    let total = p.total_ops();
    if p.elapsed_ns > 0 {
        let secs = p.elapsed_ns as f64 / 1e9;
        out.insert(TAOBENCH_TOTAL_QPS.to_string(), total as f64 / secs);
        out.insert(TAOBENCH_FAST_QPS.to_string(), p.fast_ops as f64 / secs);
        out.insert(TAOBENCH_SLOW_QPS.to_string(), p.slow_ops as f64 / secs);
    }
    if total > 0 {
        out.insert(
            TAOBENCH_HIT_RATIO.to_string(),
            p.fast_ops as f64 / total as f64,
        );
    }
    if p.get_cmds > 0 {
        out.insert(
            TAOBENCH_HIT_RATE.to_string(),
            1.0 - (p.get_misses as f64 / p.get_cmds as f64),
        );
    }
}
