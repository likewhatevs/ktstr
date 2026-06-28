use super::*;

/// Definition of a metric for the comparison pipeline.
///
/// Each entry describes polarity (`higher_is_worse`), dual-gate
/// significance thresholds (`default_abs`, `default_rel`), a
/// display unit string for formatted output, and a row accessor
/// (`accessor`) that returns the metric's value from a
/// [`GauntletRow`] without a hand-maintained name→field match.
///
/// The `accessor` field is skipped in serde output because `fn`
/// pointers are not serializable. A future `Deserialize` impl
/// would need callers to re-hydrate the accessor by looking up
/// `name` via [`metric_def`] — the static [`METRICS`] table is
/// the authoritative source of the function identity. No such
/// impl exists today; the note is a forward-conditional so that
/// if one is added, the migration path is spelled out rather
/// than reinvented per site.
///
/// # Registered vs unregistered metrics
///
/// The static [`METRICS`] registry is the "core metric" set with
/// hand-authored accessors, hand-tuned dual-gate thresholds
/// (`default_abs` / `default_rel`), and display units. Each
/// registered `MetricDef.accessor` reads a typed field on
/// `GauntletRow` directly (e.g. `r.spread`, `r.gap_ms`).
///
/// Metrics that fall OUTSIDE this registry are carried on
/// `GauntletRow.ext_metrics: BTreeMap<String, f64>`. Registered
/// metrics never flow through `ext_metrics`; unregistered metrics
/// never flow through the typed fields. [`MetricDef::read`] and
/// `read_metric` check the registered-field accessor first and
/// fall back to an `ext_metrics.get(name)` lookup — a name that
/// matches neither returns `None`. Consumers that want to
/// distinguish "registered-but-null" from "unregistered-and-
/// absent" must inspect the registry directly rather than rely
/// on the fallback.
///
/// # `#[non_exhaustive]` migration note
///
/// Downstream code that pattern-matches an instance of `MetricDef`
/// must end the match with `..` so a future field addition does
/// not become a breaking change. Prefer reading values through
/// the static [`METRICS`] registry and [`metric_def`] lookup
/// rather than constructing `MetricDef` values by hand.
#[derive(Debug, Clone, serde::Serialize)]
#[non_exhaustive]
pub struct MetricDef {
    pub name: &'static str,
    /// Regression direction for this metric. A metric that
    /// previously used `higher_is_worse: true` maps to
    /// [`Polarity::LowerBetter`](crate::test_support::Polarity::LowerBetter)
    /// (bigger values are regressions, so smaller is better);
    /// `false` maps to
    /// [`Polarity::HigherBetter`](crate::test_support::Polarity::HigherBetter).
    /// The sense is INVERSE: the old bool answered "does growing
    /// this value mean worse?" while the enum answers "what
    /// direction do we want this to move?".
    pub polarity: crate::test_support::Polarity,
    /// Temporal aggregation kind. Drives how
    /// [`aggregate_samples`] collapses N readings of the same
    /// metric across multiple capture samples (e.g. periodic
    /// monitor ticks within one run, or two `cargo ktstr stats
    /// compare` snapshot subdirectories) into one comparable
    /// value. Distinct from [`Self::polarity`], which is the
    /// "good direction" of the FINAL value: kind tells us HOW to
    /// reduce a vec of samples; polarity tells us how to interpret
    /// the reduced number.
    ///
    /// Default `Counter` matches the most common shape — every
    /// kernel monotonic counter (SCX_EV_*, ttwu_count, run_delay,
    /// cpustat[]) collapses by sum-of-deltas. ~80% of ktstr fields
    /// are counters; the field exists so the remaining peaks and
    /// gauges can opt out of sum-aggregation explicitly.
    pub kind: MetricKind,
    pub default_abs: f64,
    pub default_rel: f64,
    pub display_unit: &'static str,
    #[serde(skip)]
    pub accessor: fn(&GauntletRow) -> Option<f64>,
}

/// Temporal aggregation classification for a metric.
///
/// Kernel-source-grounded per the metric-semantics taxonomy.
/// Drives [`aggregate_samples`] — the function that collapses a
/// slice of per-sample readings of the SAME metric into one
/// representative value for downstream regression / display.
///
/// Reduction semantics by variant:
///   - [`MetricKind::Counter`] — kernel monotonic counter; the
///     temporal aggregate is the SUM of consecutive deltas across
///     the sample window. For pre-deltaed inputs (each sample
///     carries its own window's count) this is `samples.iter().sum()`.
///   - [`MetricKind::Gauge`] — instantaneous value; the
///     [`GaugeAgg`] subkind picks Avg / Last / Max.
///   - [`MetricKind::Peak`] — kernel-side max-of-window (e.g.
///     `max_run_delay`, `max_newidle_lb_cost`); temporal aggregate
///     is max-of-max so a window-wise high-water never gets
///     diluted.
///   - [`MetricKind::Timestamp`] — wall/rq clock; the temporal
///     aggregate is the LAST sample's value (a snapshot of "where
///     the clock is now"). Diffing two captures gives elapsed
///     time, but a single window's reduction picks the latest
///     reading — averaging timestamps is meaningless.
// Serialize only: MetricKind is serialized as part of MetricDef (which is
// Serialize-only) but is never deserialized. A `Deserialize` derive here
// would narrow to `Deserialize<'static>` because the Rate variant carries
// `&'static str` fields (serde treats `&str` as borrowed), so it would not
// satisfy `DeserializeOwned` and would break any future container that
// deserializes an embedded MetricKind. Drop it rather than carry a fragile,
// unused impl.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[non_exhaustive]
pub enum MetricKind {
    /// Monotonic counter (SCX_EV_* event counters,
    /// `cpustat[CPUTIME_*]`, `bpf_prog_stats.cnt`, `ttwu_count`,
    /// `nr_migrations`, …). Aggregate by sum.
    Counter,
    /// Instantaneous value (`nr_running`, `local_dsq.nr`, current
    /// `policy`, current `comm`). The [`GaugeAgg`] tag picks the
    /// reduction: Avg for typical-load, Last for "what's happening
    /// now", Max for worst-instant.
    Gauge(GaugeAgg),
    /// Kernel max-of-window (`max_run_delay`,
    /// `max_newidle_lb_cost`, the per-CPU preempt-off peak).
    /// Aggregate by max — a peak that ever fired must survive the
    /// reduction.
    Peak,
    /// Clock or wall-time reading (`rq.clock`,
    /// CLOCK_REALTIME-stamped capture timestamps). Aggregate by
    /// Last — averaging timestamps loses meaning.
    Timestamp,
    /// PRE-DELTAED counter: each sample is already a delta-since-the-
    /// previous-read, not a cumulative-since-boot total. Schedulers
    /// that delta their scx_stats Metrics server-side per reader
    /// request (e.g. scx_mitosis) produce this — one ktstr snapshot =
    /// one reader request = one delta. The per-phase reduction is the
    /// SUM of the in-phase deltas (NOT the `Counter` last-minus-first,
    /// which would difference two deltas into nonsense); the flat-run
    /// reduction is likewise the sum. Boundary: the first in-phase
    /// delta straddles the phase boundary (it spans from the last
    /// pre-phase read to the first in-phase read, so it includes a
    /// little pre-phase activity); it is attributed to the phase its
    /// read lands in — a slight left-edge over-attribution, the
    /// deliberate semantic since a per-read delta cannot be split.
    DeltaSum,
    /// An INJECTED per-phase delta that SUMS across phases within a run but
    /// folds by UNWEIGHTED MEAN across runs. Each per-phase value is a
    /// thread-group CPU-time delta (`system_time_ns` / `user_time_ns`,
    /// written directly into `PhaseBucket.metrics` by `phase_group_cpu_delta`
    /// — there is no `read_sample` arm). The cross-PHASE fold is the SUM of the
    /// disjoint per-phase deltas (like `Counter`), NOT a sample-count-weighted
    /// mean: a per-phase delta is already proportional to the phase's
    /// wall-clock duration, and the freeze `sample_count` is too, so a weighted
    /// mean would double-count duration and collapse the run to a meaningless
    /// duration-weighted average per-phase delta. The sum is the run's total
    /// OBSERVED CPU time — first-to-last freeze WITHIN each phase — a lower
    /// bound that excludes pre-first-freeze, post-last-freeze, and
    /// inter-phase-gap windows (and single-freeze phases, which contribute
    /// nothing), per `phase_group_cpu_delta`'s observed-window semantic; that
    /// is sufficient for an A/B regression signal, where the same
    /// observed-window sum is compared on both sides. Across RUNS each run
    /// contributes exactly one total, folded by the UNWEIGHTED arithmetic mean
    /// (`Σ / contributors`) over the runs that emitted the key — NOT weighted
    /// by `run_sample_count` (the monitor capture count, an unrelated
    /// population that would also silently zero-weight a monitor-off run). This
    /// SUM-cross-phase / MEAN-cross-run pair is exactly why these deltas are
    /// NOT `Gauge(Avg)` (weighted mean at BOTH levels — the bug this kind
    /// fixes) and NOT `DeltaSum` (sum at both levels, which would inflate the
    /// cross-run value by the run count). The value is run-wide POOLED (one
    /// scalar per phase across all tgids), NOT per-cgroup; the cross-cgroup
    /// same-step merge is `Commutative` (`a + b`, like `Counter`) but is the
    /// defensive same-step-index path only — dead for these keys, which are
    /// injected exactly once into the pooled host `PhaseBucket`. NOT
    /// [`Self::is_derived`] — it carries a real per-phase value, unlike Rate /
    /// Distribution.
    PerPhaseDeltaSum,
    /// Derived ratio of two component metrics — a RATE that must be
    /// recomputed from its components at every in-map aggregation level, never
    /// averaged as a ready-made ratio. The variant carries the registry
    /// names of its `numerator` and `denominator` component metrics, each
    /// itself registered with its own kind (e.g. a `Counter` numerator).
    ///
    /// A Rate has NO samples of its own. Its value is DERIVED from the
    /// already-reduced component values as `map[numerator] /
    /// map[denominator]` by the [`derive_rate_metrics`] post-pass. An
    /// aggregation level that pools the components FIRST (each by its own
    /// kind — a `Counter` numerator sums, a `Gauge(Avg)` averages) and
    /// then re-derives the rate RE-POOLS correctly: for the common
    /// `Counter / Counter` case the result is `Σnumerator / Σdenominator`,
    /// NOT a mean of two phases' ready-made ratios `(r₁ + r₂) / 2` (which
    /// is WRONG whenever the phases carry different denominator weight,
    /// e.g. iterations-per-cpu-second across phases of unequal CPU time).
    /// The numerator and denominator must already be expressed in units
    /// whose quotient is the intended rate unit (the component
    /// registration owns the unit choice; this variant does not scale).
    ///
    /// `derive_rate_metrics` runs as a post-pass at the seven aggregation
    /// sites where the components co-locate in one map: the two per-phase
    /// builds (`buckets_from_grouped`, `build_phase_buckets_with_stimulus`),
    /// the cross-phase bucket merge (`merge_matched_phase_buckets`), the
    /// three cross-RUN ext-metrics reducers (`populate_run_ext_metrics`,
    /// `populate_run_ext_metrics_from_phases`, and `group_and_average_by`),
    /// and the cross-CGROUP pooled re-pool
    /// (`crate::assert::populate_run_pooled_iterations_per_cpu_sec`).
    /// The cross-CGROUP `AssertResult::merge` ext-metrics fold itself uses
    /// worst-case polarity (min/max) and is NOT a re-pool site; the pooled
    /// re-pool runs separately after it, at the eval layer, reading
    /// `stats.cgroups` directly. `iteration_rate` does not exercise the merge
    /// fold either: it and its components are host-injected by
    /// `populate_run_ext_metrics_from_phases` AFTER the cross-cgroup `merge`,
    /// so the fold never sees them. The pooled `iterations_per_cpu_sec` is the
    /// rate whose components ARE per-cgroup, and
    /// `populate_run_pooled_iterations_per_cpu_sec` re-pools it post-merge.
    ///
    /// Because a single sample slice cannot express the re-pool, a Rate is
    /// FORBIDDEN from the single-slice reducers ([`aggregate_finite`]
    /// panics on it); the post-pass is its only producer.
    Rate {
        /// Registry name of the numerator component metric.
        numerator: &'static str,
        /// Registry name of the denominator component metric.
        denominator: &'static str,
    },
    /// Derived DISTRIBUTIONAL aggregate re-pooled from a raw per-cgroup
    /// sample set, never folded from ready-made per-cgroup reductions. The
    /// variant names the [`SampleSource`] (which
    /// [`crate::assert::PhaseCgroupStats`] sample vector feeds it) and the
    /// [`SampleReduction`] (which statistic to compute over the pooled set).
    ///
    /// Like [`MetricKind::Rate`], a Distribution has NO value of its own at
    /// the WITHIN-RUN levels: its run-level value is DERIVED post-merge by
    /// `crate::assert::populate_run_distribution_metrics`, which pools the
    /// raw samples from `stats.phases[].per_cgroup` across every phase and
    /// cgroup and recomputes the statistic over the COMBINED set — the
    /// percentile / CV / mean / extreme of the pooled distribution, NOT a
    /// max or mean of per-cgroup reductions (the percentile of a union is
    /// not the max of per-source percentiles). It is therefore FORBIDDEN
    /// from the per-phase single-slice reducers
    /// ([`aggregate_samples_for_phase`] returns None via
    /// [`MetricKind::is_derived`]); the post-pass is its only within-run
    /// producer. When the size-limited bulk frame strips the sample pools
    /// (`crate::assert::strip_phase_cgroup_samples`), the producer falls
    /// back to a worst-wins fold over the surviving per-cgroup `CgroupStats`
    /// reductions so the metric degrades rather than vanishing.
    ///
    /// CROSS-RUN it is a HYBRID, unlike Rate: a run's components (the raw
    /// sample vectors) do not survive into the cross-RUN ext-metrics map
    /// (phases are dropped at the cross-RUN fold), so there is no combined
    /// sample SET to re-pool across runs. The cross-RUN value is instead a
    /// plain fold of the per-run derived values — an UNWEIGHTED mean (over the
    /// runs that emitted the key, `sum / finite.len()`) for the percentile /
    /// CV / mean reductions and a MAX for [`SampleReduction::Worst`] (the
    /// peak run-delay) — applied by [`aggregate_finite`] over the per-run ext
    /// values. So `is_derived`
    /// skips it at the within-run sites, but the cross-RUN ext fold does
    /// NOT skip it (only Rate, whose components DO survive cross-RUN, is
    /// skipped there).
    Distribution {
        /// Which raw sample vector on
        /// [`crate::assert::PhaseCgroupStats`] feeds this aggregate.
        source: SampleSource,
        /// Which statistic to recompute over the pooled sample set.
        reduction: SampleReduction,
    },
    /// Derived LOWEST-WINS per-cgroup selector — the worst (lowest) cgroup's
    /// `numerator / denominator` ratio across the run, re-pooled from per-cgroup
    /// carriers rather than folded from ready-made ratios. None-aware lowest-wins
    /// (the semantic the deleted `fold_lowest_some` carried in
    /// [`crate::assert::AssertResult::merge`], now in
    /// `crate::assert::populate_run_distribution_metrics`): a measured
    /// `Some(0.0)` — a cgroup that ran zero iterations (real starvation) for the
    /// efficiency selectors, or one with all pages off-node for
    /// `worst_page_locality` — wins the worst bucket; a not-measured `None` (no
    /// workers / no on-CPU time / no NUMA pages) is skipped; and an all-`None`
    /// cohort produces no key (absence preserved as a missing ext entry, never a
    /// `0.0`).
    ///
    /// Derived post-merge by
    /// `crate::assert::populate_run_distribution_metrics`. The SOURCE depends on
    /// the numerator: the iteration-efficiency selectors (`Iterations`) re-pool
    /// from the `stats.cgroups[]` counters (which survive bulk-frame stripping,
    /// so they need no degraded fallback); `worst_page_locality` (`NumaLocal`)
    /// re-pools from the per-phase `stats.phases[].per_cgroup` NUMA carriers (the
    /// reports-only `CgroupStats` hardcodes `page_locality` 0.0, so it cannot
    /// source from `stats.cgroups[]`). Like Distribution it is `is_derived`
    /// (skipped at the within-run reducers) and CROSS-RUN it MEAN-folds the
    /// per-run derived values through [`aggregate_finite`].
    WorstLowest {
        /// The per-cgroup numerator (`Iterations` or `NumaLocal`).
        numerator: WorstLowestNumerator,
        /// The per-cgroup denominator the numerator is divided by.
        denominator: WorstLowestDenominator,
    },
    /// Derived WORST-CGROUP wake-latency tail-amplification selector — the
    /// highest per-cgroup `p99 / median` wake-latency ratio across the run.
    /// Higher-is-worse (a stretched long tail), so "worst" is the MAX over
    /// cgroups — the polarity-opposite of [`MetricKind::WorstLowest`]'s
    /// lowest-wins. Re-selected post-merge by
    /// `crate::assert::populate_run_distribution_metrics` from the
    /// `stats.cgroups[]` entries via `CgroupStats::wake_latency_tail_ratio`
    /// (deliberately NOT `pooled_p99 / pooled_median` of the cross-cgroup
    /// union — that is the distinct `worst_p99_wake_latency_us` /
    /// `worst_median_wake_latency_us` Distribution pair). Like Distribution /
    /// WorstLowest it is [`MetricKind::is_derived`] (skipped at the within-run
    /// reducers); the producer emits NO key when the run is below the
    /// [`WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`] noise floor or no cgroup
    /// carried a measurable tail (absence preserved as a missing ext entry,
    /// never a `0.0` sentinel — the no-false-zero contract the deleted typed
    /// field could not express).
    ///
    /// CROSS-RUN it folds, like every WorstLowest selector, by the UNWEIGHTED
    /// exclude-missing MEAN through [`aggregate_finite`] (`sum / finite.len()`
    /// over the runs that emitted the key) — the cohort's TYPICAL worst-cgroup
    /// tail amplification, deliberately NOT a MAX: peak-of-peaks is reserved
    /// for [`SampleReduction::Worst`] (a peak detector answering "did this ever
    /// fire"), whereas this answers "what is this cohort's characteristic
    /// worst-cgroup tail". A run below the floor never enters the mean, so no
    /// sub-threshold run dilutes the cohort (the bug the ext relocation fixed:
    /// the deleted typed cross-RUN fold summed every passing run's raw ratio
    /// over `passes_observed`, folding noisy low-N runs in as real values).
    WakeLatencyTailRatio,
    /// Derived WORST-CGROUP cross-node migration-churn selector — the highest
    /// per-cgroup `cross_node_migrated / numa_pages_total` ratio across the run.
    /// LowerBetter (more cross-node migration is worse), so "worst" is the MAX
    /// over cgroups — the polarity twin of `worst_page_locality`
    /// ([`MetricKind::WorstLowest`] `NumaLocal`/`NumaTotal`, lowest-wins), sharing
    /// the same per-phase NUMA carriers and the `numa_agg_per_cgroup` helper.
    /// Re-pooled post-merge by
    /// `crate::assert::populate_run_distribution_metrics` from
    /// `stats.phases[].per_cgroup`: the cross-phase fold SUMs the per-phase
    /// migration-counter deltas over the LATEST residency total
    /// (`cross_node_migration_ratio_of(summed_migrated, latest_total)`), then
    /// MAXes across cgroups that measured NUMA residency (`numa_pages_total > 0`);
    /// a never-measured cohort yields no key (absence preserved as a missing ext
    /// entry, never a `0.0`). A CHURN ratio (cumulative migration EVENTS over a
    /// residency SNAPSHOT) — can legitimately exceed 1.0, NOT a bounded `[0,1]`
    /// fraction. A dedicated max-selector like [`MetricKind::WakeLatencyTailRatio`]
    /// (no generic numerator/denominator), since cross_node is the sole max-wins
    /// phase-carrier ratio. Like the other derived kinds it is
    /// [`MetricKind::is_derived`] (skipped at the within-run reducers) and
    /// CROSS-RUN MEAN-folds through [`aggregate_finite`] over the runs that
    /// EMITTED the key — the cohort's typical worst-cgroup churn. The deleted
    /// typed `Gauge(Last)` field instead averaged its value over
    /// `passes_observed`, folding a NUMA-less passing run's `0.0` sentinel into
    /// the mean (dilution); the ext re-pool writes no key for a never-measured
    /// run (absence preserved), so it never enters the divisor.
    WorstCrossNodeRatio,
    /// Per-phase-only scalar derived ONCE per phase from a phase-scoped
    /// carrier, NOT from monitor samples and NOT re-pooled run-level. The sole
    /// producer is [`crate::assert::derive_phase_metrics`], which derives two
    /// families per phase:
    /// - the schbench scalars: it pools each phase's
    ///   `crate::assert::PhaseCgroupStats` schbench carriers and re-derives a
    ///   percentile (from the merged latency histogram), a sample-weighted mean
    ///   (run-delay), or a count (loop_count), writing them into BOTH each
    ///   carrier's `PhaseCgroupStats::metrics` (per-cgroup, read via
    ///   `phase_cgroup_metric`) AND the pooled
    ///   `crate::assert::PhaseBucket::metrics` (read via `phase_metric`).
    /// - the NON-schbench carrier scalars (wake/run-delay/off-cpu distributions
    ///   + migration/iterations/locality ratios, via `write_carrier_scalars`):
    ///   written ONLY into each `PhaseCgroupStats::metrics` (per-cgroup, read
    ///   via `phase_cgroup_metric`) — these have NO pooled
    ///   `crate::assert::PhaseBucket::metrics` entry; their run-level aggregate
    ///   is the `worst_*` ext-metrics key.
    /// It is [`MetricKind::is_derived`] (so the within-run reducers —
    /// [`aggregate_samples_for_phase`] and the phase-bucket merge loop — skip
    /// it) and has NO run-level producer; it is ADDITIONALLY gated out of the
    /// cross-RUN ext fold (`fold_ext_metrics`), since a per-phase scalar has no
    /// meaningful cross-run aggregate. A unit marker (no payload): the
    /// derivation owns the metric-name→computation mapping, so the kind need not
    /// carry a percentile selector (which would leak `plat::Pct` through this
    /// public enum).
    // doc_lazy_continuation: pre-existing list-item wording surfaced by the clippy
    // 1.94 bump; renders fine. Suppress rather than reflow the prose.
    #[allow(clippy::doc_lazy_continuation)]
    PerPhase,
    /// A WHOLE-RUN distributional value (a percentile / min / max) re-pooled
    /// run-level by UNIONING the per-phase per-cgroup raw distribution carriers
    /// and re-deriving the statistic over the union — the schbench engine's
    /// `*_whole` wakeup / request / rps keys, written by
    /// `crate::assert::populate_run_pooled_schbench_distribution` from the
    /// `PhaseCgroupStats::schbench` `PlatStats` histograms (`PlatStats::combine`
    /// is an associative bucket-count add, so the merged histogram is the faithful
    /// union and the re-derived percentile is the percentile OF the pooled sample
    /// set, NOT a mean of per-source percentiles). UNLIKE
    /// [`MetricKind::Distribution`] it is NOT cross-RUN folded: a percentile of a
    /// union is not a mean of per-run percentiles, and the per-phase histograms
    /// are dropped at the cross-run boundary (no pooled set survives to
    /// re-derive), so the only honest cross-run treatment is the per-run
    /// noise-compare — `crate::stats::noise_findings` reads each run's own
    /// `*_whole` scalar and compares the spread, never averaging them. So it is
    /// [`MetricKind::is_derived`] (the within-run reducers skip it; the value is
    /// produced solely by the run-level union populate) AND gated out of the
    /// cross-RUN ext fold (`fold_ext_metrics`); `noise_findings` is its only
    /// consumer. ext-only (`accessor |_| None`); the `*_whole` names are distinct
    /// from the per-phase [`MetricKind::PerPhase`] percentile keys (one registry
    /// name = one kind) — the established per-phase-vs-whole-run coexistence (as
    /// the taobench and schbench loop/run-delay whole-run keys also do).
    PerRunDistribution,
}

/// Sub-classification for [`MetricKind::Gauge`] picking the
/// per-window reduction. Most ktstr gauges are Avg ("typical-load
/// over the window"); Last fits "current state" snapshots like
/// `comm` / `policy`; Max fits worst-instant queue-depth probes.
// Serialize-only, matching its container MetricKind (which is Serialize-only)
// and the sibling MetricKind sub-enums (SampleSource / SampleReduction /
// WorstLowestNumerator / WorstLowestDenominator). Nothing deserializes a
// MetricKind / GaugeAgg, so the prior Deserialize derive was dead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[non_exhaustive]
pub enum GaugeAgg {
    /// Reduce by arithmetic mean. Default for `nr_running`-style
    /// gauges where the question is "what was the typical load".
    Avg,
    /// Take the latest sample. Default for `comm` / `policy` /
    /// `cgroup_path`-style snapshots where the value is "what is
    /// it RIGHT NOW".
    Last,
    /// Take the max sample. Useful when a gauge is being used to
    /// detect a worst-case regression (e.g. queue-depth probe
    /// where any spike is the signal of interest).
    Max,
}

/// The raw per-cgroup sample vector on
/// [`crate::assert::PhaseCgroupStats`] that a [`MetricKind::Distribution`]
/// re-pools over. Each variant maps to exactly one un-reduced sample
/// vector the per-phase per-cgroup carrier holds (stored RAW in
/// nanoseconds; the [`SampleReduction`] applies the ns→µs scale once).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[non_exhaustive]
// The `Ns` suffix on every variant documents the unit (all sources are RAW
// nanoseconds) at each use site, not just in the enum doc; clippy's
// enum_variant_names is a style heuristic that misfires on a meaningful shared
// unit suffix — renaming would drop the unit, so the suffix is kept.
#[allow(clippy::enum_variant_names)]
pub enum SampleSource {
    /// Per-wakeup latency samples in ns
    /// (`crate::assert::PhaseCgroupStats::wake_latencies_ns`). One sample per
    /// observed wakeup (reservoir-capped per cgroup), so the pooled set is the
    /// cross-cgroup union of those capped per-wakeup samples.
    WakeLatencyNs,
    /// Per-worker schedstat run-delay samples in ns
    /// (`crate::assert::PhaseCgroupStats::run_delays_ns`). One sample per worker
    /// — each is that worker's `sched_info.run_delay` delta over the carrier's
    /// window (whole-run last-minus-first for the step-local carrier; the
    /// per-phase delta for the backdrop slice carrier), so the pool size is the
    /// worker count, NOT a per-wakeup stream like `WakeLatencyNs`.
    RunDelayNs,
    /// Per-timer-cycle latency samples in ns
    /// (`crate::assert::PhaseCgroupStats::timer_latencies_ns`). One sample per
    /// `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)` wake — the observed
    /// wake time minus the absolute deadline, floored at 0 — recorded by
    /// [`crate::workload::WorkType::TimerLatency`] (reservoir-capped per cgroup
    /// like `WakeLatencyNs`). Distinct from `WakeLatencyNs` so cyclictest-style
    /// timer latency does not blur with the blocking variants' wake latency in a
    /// shared sidecar.
    TimerLatencyNs,
}

/// The statistic a [`MetricKind::Distribution`] computes over its pooled
/// [`SampleSource`] set. Each maps to the matching reduction
/// `crate::assert::cgroup_stats` computes per cgroup, so the run-level
/// re-pool reproduces that reduction over the COMBINED cross-cgroup set
/// rather than folding ready-made per-cgroup reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[non_exhaustive]
pub enum SampleReduction {
    /// 99th percentile (nearest-rank), ns→µs.
    P99,
    /// 99.9th percentile (nearest-rank), ns→µs — the deep tail an RT /
    /// cyclictest-style latency probe turns on (a single max is one sample;
    /// p99.9 is the robust deep-tail percentile between p99 and max).
    P999,
    /// Median (50th percentile, nearest-rank), ns→µs.
    Median,
    /// Coefficient of variation (stddev / mean) over the pooled set,
    /// `n = pool.len()`. Unitless.
    Cv,
    /// Arithmetic mean over the pooled set, ns→µs.
    Mean,
    /// Maximum (worst) sample over the pooled set, ns→µs. CROSS-RUN this is
    /// the one reduction [`aggregate_finite`] folds by MAX (peak survives),
    /// not MEAN — see [`MetricKind::Distribution`].
    Worst,
}

/// The per-cgroup numerator of a [`MetricKind::WorstLowest`] lowest-wins
/// selector. `#[non_exhaustive]`, mirroring [`MetricKind::Rate`]'s `numerator`.
/// The producer branches on the numerator to pick the per-cgroup SOURCE:
/// `Iterations` reads the `crate::assert::CgroupStats` counters; `NumaLocal`
/// reads the per-phase NUMA carriers (`page_locality` is structurally 0.0 in the
/// reports-only `CgroupStats`, so it must come from the phase carriers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[non_exhaustive]
pub enum WorstLowestNumerator {
    /// Per-cgroup total iteration count
    /// (`crate::assert::CgroupStats::total_iterations`).
    Iterations,
    /// Per-cgroup pages resident on the expected NUMA node(s) — the
    /// page-locality numerator, the LATEST per-phase residency snapshot summed
    /// across the cgroup's workers (`crate::assert::PhaseCgroupStats::numa_pages_local`),
    /// NOT a `CgroupStats` counter.
    NumaLocal,
}

/// The per-cgroup denominator a [`MetricKind::WorstLowest`] numerator is divided
/// by to form the lowest-wins ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[non_exhaustive]
pub enum WorstLowestDenominator {
    /// Worker count (`crate::assert::CgroupStats::num_workers`) — yields
    /// iterations-per-worker (raw throughput, scales with the CPU budget).
    NumWorkers,
    /// On-CPU nanoseconds (`crate::assert::CgroupStats::total_cpu_time_ns`),
    /// converted ns→s ONCE on the summed counter — yields the
    /// overcommit-invariant iterations-per-CPU-second efficiency.
    CpuTimeNs,
    /// Total resident pages — the page-locality denominator, the LATEST
    /// per-phase residency snapshot (`crate::assert::PhaseCgroupStats::numa_pages_total`),
    /// shared with the cross-node ratio. Paired with `NumaLocal` to yield the
    /// page-locality fraction; the per-cgroup ratio is absent (None) when the
    /// cgroup measured no NUMA pages.
    NumaTotal,
}

/// How a per-phase metric reduction merges across two
/// [`crate::assert::AssertResult`]s that both carry a
/// [`crate::assert::PhaseBucket`] at the same `step_index`.
///
/// Driven by [`MetricKind::merge_kind`] so a future
/// [`MetricKind`] addition is forced to declare its merge
/// semantic explicitly (the match is `#[non_exhaustive]`-aware
/// via the helper rather than a bare `match` in every caller).
///
/// The split mirrors the rolling-aggregation contract in
/// [`AssertResult::merge`](crate::assert::AssertResult::merge): the
/// per-phase fold must commute so the accumulator pattern
/// `AssertResult::pass().merge(real_a).merge(real_b)` yields the
/// same result whether merges arrive in `a→b` or `b→a` order
/// — EXCEPT for kinds whose reduction is intrinsically the LAST
/// sample (`Gauge(Last)`, `Timestamp`), where the merge must
/// resolve to the bucket whose `end_ms` is later.
///
/// Counter, Peak, and Gauge(Max/Avg) are commutative because their
/// reductions are sum / max / weighted-mean respectively — all
/// associative, commutative folds over reduced values. Gauge(Last)
/// and Timestamp are NOT commutative under a per-merge cumulative
/// fold (the "later" sample wins) so the merge uses `end_ms` as
/// the tiebreaker rather than the operand order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MergeKind {
    /// The reduction commutes: `merge(a, b) == merge(b, a)`. The
    /// merge folds the two reduced values via the kind's natural
    /// commutative operation (Counter → sum, Peak / Gauge(Max) →
    /// max, Gauge(Avg) → weighted mean by `sample_count`).
    Commutative,
    /// The reduction is "the LATEST sample's value" (Gauge(Last),
    /// Timestamp). The merge resolves to the value from whichever
    /// bucket has the later `end_ms`; ties keep `self`.
    NonCommutative,
    /// The value is DERIVED post-merge from pooled components, never folded
    /// from two already-reduced values. Covers every
    /// [`MetricKind::is_derived`] kind:
    /// - [`MetricKind::Rate`]: re-derived as `Σnumerator / Σdenominator` from
    ///   its component keys by [`derive_rate_metrics`];
    /// - [`MetricKind::Distribution`] / [`MetricKind::WorstLowest`] /
    ///   [`MetricKind::WakeLatencyTailRatio`] / [`MetricKind::WorstCrossNodeRatio`]:
    ///   re-pooled post-merge by
    ///   `crate::assert::populate_run_distribution_metrics`;
    /// - [`MetricKind::PerPhase`]: re-derived per phase by
    ///   `crate::assert::derive_phase_metrics`.
    ///
    /// The per-metric merge loop skips these derived keys entirely and the
    /// post-pass produces them, so this variant is classification metadata: no
    /// merge dispatches on it.
    Recompute,
}

impl MetricKind {
    /// Map each [`MetricKind`] variant to the corresponding
    /// [`MergeKind`] used by per-phase
    /// [`AssertResult::merge`](crate::assert::AssertResult::merge).
    /// Centralising the mapping here means a future kind
    /// addition fails the build until the new variant is wired
    /// (the inner `match` is exhaustive even though `MetricKind`
    /// is `#[non_exhaustive]` because this fn lives in the same
    /// crate).
    pub fn merge_kind(self) -> MergeKind {
        match self {
            MetricKind::Counter => MergeKind::Commutative,
            MetricKind::Peak => MergeKind::Commutative,
            MetricKind::Gauge(GaugeAgg::Avg) => MergeKind::Commutative,
            MetricKind::Gauge(GaugeAgg::Max) => MergeKind::Commutative,
            MetricKind::Gauge(GaugeAgg::Last) => MergeKind::NonCommutative,
            MetricKind::Timestamp => MergeKind::NonCommutative,
            // Per-phase reduction is a sum of in-phase deltas — an
            // associative, commutative fold, so cross-AssertResult merge
            // sums the two reduced values (same as Counter).
            MetricKind::DeltaSum => MergeKind::Commutative,
            // PerPhaseDeltaSum: the cross-cgroup same-step merge is a
            // commutative sum (`a + b`, like Counter), but for these run-wide
            // POOLED keys (one scalar per phase, injected once into the host
            // bucket) it is the defensive same-step-index path only — two real
            // values are never summed. (The SUM-cross-phase / MEAN-cross-run
            // split is handled at the fold sites, not in this merge.)
            MetricKind::PerPhaseDeltaSum => MergeKind::Commutative,
            // A Rate is re-derived from its pooled components, never
            // folded from two ready-made ratios.
            MetricKind::Rate { .. } => MergeKind::Recompute,
            // Distribution and WorstLowest are derived post-merge by
            // `populate_run_distribution_metrics` (re-pooled from the
            // per-cgroup raw samples / counters), so the per-phase merge
            // loop skips them and re-derives — classification-only, like
            // Rate. See [`MetricKind::is_derived`].
            MetricKind::Distribution { .. } => MergeKind::Recompute,
            MetricKind::WorstLowest { .. } => MergeKind::Recompute,
            // Worst-cgroup wake-latency tail ratio: derived post-merge by
            // `populate_run_distribution_metrics` (max over the merged
            // `stats.cgroups` per-cgroup ratios), so the per-phase merge loop
            // skips and re-derives it — classification-only, like the other
            // derived kinds.
            MetricKind::WakeLatencyTailRatio => MergeKind::Recompute,
            // Worst-cgroup cross-node migration churn: derived post-merge by
            // `populate_run_distribution_metrics` (max over the per-phase NUMA
            // carriers' per-cgroup churn ratio), so the per-phase merge loop skips
            // and re-derives it — classification-only, like the other derived kinds.
            MetricKind::WorstCrossNodeRatio => MergeKind::Recompute,
            // PerPhase is derived post-merge by `derive_phase_metrics` (schbench
            // scalars into PhaseBucket.metrics + PhaseCgroupStats::metrics;
            // non-schbench carrier scalars into PhaseCgroupStats::metrics only);
            // the per-phase merge loop skips it (is_derived) and never re-derives
            // via a kind — classification-only, like the other derived kinds.
            MetricKind::PerPhase => MergeKind::Recompute,
            // PerRunDistribution is derived run-level by
            // `populate_run_pooled_schbench_distribution` (union of the per-phase
            // per-cgroup PlatStats histograms, percentile re-derived over the
            // union); the per-phase merge loop skips it (is_derived) and never
            // re-derives via a kind — classification-only, like the other derived
            // kinds.
            MetricKind::PerRunDistribution => MergeKind::Recompute,
        }
    }

    /// Whether this kind is DERIVED post-merge from other data rather than
    /// reduced from its own per-phase sample slice: [`MetricKind::Rate`]
    /// (from numerator/denominator components), [`MetricKind::Distribution`]
    /// (re-pooled from the per-cgroup raw sample sets), [`MetricKind::WorstLowest`]
    /// (lowest-wins over per-cgroup counters),
    /// [`MetricKind::WakeLatencyTailRatio`] (max over the per-cgroup p99/median
    /// wake-latency ratios, floor-gated), and [`MetricKind::WorstCrossNodeRatio`]
    /// (max over the per-cgroup cross-node migration-churn ratios).
    ///
    /// Drives the WITHIN-RUN skip-sites that must not reduce a derived kind
    /// from a slice: [`aggregate_samples_for_phase`] returns None, and the
    /// per-phase build, the cross-phase
    /// `crate::assert::merge_matched_phase_buckets` key-loop, and
    /// [`crate::assert::populate_run_ext_metrics_from_phases`] all skip the
    /// key then re-derive.
    ///
    /// NOT a uniform cross-RUN skip: at the cross-RUN ext fold
    /// ([`group_and_average_by`]) [`MetricKind::Rate`] AND
    /// [`MetricKind::PerPhase`] are skipped — Rate's components survive
    /// cross-RUN so it re-derives there, and PerPhase is a per-phase-only
    /// scalar with no meaningful cross-RUN aggregate (its skip also keeps
    /// [`aggregate_finite`]'s `PerPhase => unreachable!` unreachable) — while
    /// Distribution / WorstLowest / WakeLatencyTailRatio / WorstCrossNodeRatio,
    /// whose components do NOT survive cross-RUN, fall through to be plainly
    /// folded (MEAN, or MAX for [`SampleReduction::Worst`]) by
    /// [`aggregate_finite`]. So callers
    /// gate on `is_derived` for the within-run sites and on
    /// `matches!(.., Rate { .. } | PerPhase)` for the cross-RUN ext fold.
    pub fn is_derived(self) -> bool {
        matches!(
            self,
            MetricKind::Rate { .. }
                | MetricKind::Distribution { .. }
                | MetricKind::WorstLowest { .. }
                | MetricKind::WakeLatencyTailRatio
                | MetricKind::WorstCrossNodeRatio
                | MetricKind::PerPhase
                | MetricKind::PerRunDistribution
        )
    }
}

/// Reduce a slice of per-sample readings of the same metric into
/// one representative value, dispatching on [`MetricKind`]. Used
/// by sample-windowed comparison paths (e.g. multi-tick monitor
/// captures, stats compare across multiple snapshot
/// subdirectories) to collapse a sample vec into the value the
/// existing scalar-comparison pipeline already understands.
///
/// Returns `None` when `samples` is empty — the caller decides
/// whether absence is a missing-data condition or a benign
/// "no samples in window" result. NaN samples are dropped from
/// the reduction (same semantics as the existing percentile()
/// helper); a final all-NaN input also returns `None`.
///
/// Semantics by kind:
///   - `Counter` → sum of finite samples — the flat-run reduction
///     for cross-RUN aggregation. NOT the right semantic for
///     per-phase reduction of a cumulative-since-boot Counter
///     (which would over-count). Callers wanting per-phase
///     Counter reduction use [`aggregate_samples_for_phase`],
///     which routes Counter through a dedicated last-minus-first
///     branch instead of dispatching through here.
///   - `Gauge(Avg)` → arithmetic mean of finite samples.
///   - `Gauge(Last)` → last finite sample.
///   - `Gauge(Max)` → max of finite samples.
///   - `Peak` → max of finite samples.
///   - `Timestamp` → last finite sample.
///
/// Live caller: [`aggregate_samples_for_phase`] dispatches every
/// non-Counter kind through this entry point so the per-phase
/// reduction inherits the flat-run semantic for Gauge / Peak /
/// Timestamp without restating it. That fn is itself folded by
/// [`crate::assert::build_phase_buckets`] whose live caller is
/// the host-side `evaluate_vm_result` AssertResult-population
/// site at `src/test_support/eval/mod.rs`.
pub fn aggregate_samples(samples: &[f64], kind: MetricKind) -> Option<f64> {
    let finite: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();
    aggregate_finite(&finite, |_| 1, kind)
}

/// Weighted variant of [`aggregate_samples`]. Takes a slice of
/// `(value, weight)` pairs so the lock-step shape is enforced by
/// the type — there is no length-mismatch class for the caller to
/// trigger. Weight is consulted for [`MetricKind::Gauge`] with
/// [`GaugeAgg::Avg`] (weighted mean); other kinds fold by their
/// natural reduction and ignore weight.
///
/// NaN-valued pairs drop along with their weight (filter operates
/// on the value field — no risk of weights misaligning to other
/// samples after filtering, unlike the previous parallel-slice
/// shape).
///
/// Zero total weight degenerates to the unweighted mean per the
/// `merge_metric_values` precedent. Weight sum uses `checked_add`
/// with fallback to unweighted on overflow so a pathological
/// caller can't crash the aggregator.
pub fn aggregate_samples_weighted(pairs: &[(f64, usize)], kind: MetricKind) -> Option<f64> {
    let finite: Vec<(f64, usize)> = pairs
        .iter()
        .copied()
        .filter(|(x, _)| x.is_finite())
        .collect();
    if finite.is_empty() {
        return None;
    }
    let values: Vec<f64> = finite.iter().map(|(x, _)| *x).collect();
    aggregate_finite(&values, |i| finite[i].1, kind)
}

/// Inner fold shared by [`aggregate_samples`] (uniform weights)
/// and [`aggregate_samples_weighted`] (caller-supplied weights).
/// `weight_for(i)` returns the weight for the i-th element of
/// `finite`; callers either pass `|_| 1` (unweighted) or a
/// closure that reads from their pair vec (weighted). Pre-filtered
/// `finite` carries only NaN-free values so the closure indexes
/// into a known-good vec without risking shape drift.
fn aggregate_finite(
    finite: &[f64],
    weight_for: impl Fn(usize) -> usize,
    kind: MetricKind,
) -> Option<f64> {
    if finite.is_empty() {
        return None;
    }
    Some(match kind {
        // Counter (cumulative-since-boot, cross-RUN flat sum) and
        // DeltaSum (each sample already a per-read delta) both reduce to
        // a plain sum of the finite samples here; they differ only in
        // the PER-PHASE path (Counter last-minus-first vs DeltaSum sum —
        // see aggregate_samples_for_phase).
        MetricKind::Counter | MetricKind::DeltaSum => finite.iter().sum(),
        // PerPhaseDeltaSum at the CROSS-RUN fold: each contributor is one run's
        // already-summed per-phase total, so runs fold by the UNWEIGHTED
        // arithmetic mean over the runs that emitted the key — `Σ / len`, the
        // same shape as the Distribution / WorstLowest cross-run mean below and
        // deliberately NOT weighted by `run_sample_count`. The CROSS-PHASE sum
        // that builds each run's total is done directly in
        // `crate::assert::populate_run_ext_metrics_from_phases`, which does NOT
        // route this kind through `aggregate_finite`, so this arm only ever runs
        // at the cross-RUN ext fold.
        MetricKind::PerPhaseDeltaSum => finite.iter().sum::<f64>() / (finite.len() as f64),
        // Distribution Worst (peak run-delay): the cross-RUN fold is MAX
        // so the high-water peak survives, distinct from the MEAN-folded
        // percentile / CV / mean reductions below. (WITHIN-RUN no
        // derived kind reaches here — `is_derived` skips every one at
        // the per-phase reducers; this arm only fires at the cross-RUN ext
        // fold in `group_and_average_by`.) Matched before the general
        // `Distribution { .. }` mean arm so Worst takes MAX, not MEAN.
        MetricKind::Distribution {
            reduction: SampleReduction::Worst,
            ..
        } => finite.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        // Cross-RUN MEAN fold of the remaining Distribution reductions (p99 /
        // median / CV / mean run-delay) and every WorstLowest selector: each
        // per-run value is itself a within-run pooled reduction or a
        // lowest-wins selector, NOT a monitor-sampled gauge, so the cross-RUN
        // fold is an UNWEIGHTED arithmetic mean — `sum / finite.len()`, i.e.
        // over the runs that EMITTED a finite value for the key. This matches
        // the unweighted-mean SHAPE of the surviving typed siblings
        // (spread, migration_ratio), but its
        // divisor is the present-finite-contributor count, NOT the typed path's
        // `sum / passes_observed`: a passing run that omitted the key (absent /
        // dropped-non-finite ext entry) is EXCLUDED from the mean rather than
        // folded in as 0.0 — the deliberate no-false-zero improvement the ext
        // relocation buys (the old typed field defaulted a no-data run to 0.0).
        // Weighting by `run_sample_count` (the MONITOR capture count) would
        // weight by an unrelated population AND silently zero-weight a
        // monitor-off run, so it is deliberately NOT used here. (WITHIN-RUN
        // these never reach here — `is_derived` skips them at the per-phase
        // reducers; this arm only fires at the cross-RUN ext fold in
        // `group_and_average_by`.)
        //
        // EXTREMUM ASYMMETRY (on the record, ratified): every WorstLowest
        // selector is a within-run lowest-wins ("worst cgroup") value yet folds
        // cross-RUN by this MEAN, NOT by an extremum — UNLIKE worst_run_delay_us
        // (SampleReduction::Worst), whose dedicated MAX arm above preserves the
        // peak-of-peaks. Both reproduce the deleted typed cross-RUN folds
        // exactly: run-delay is a peak detector (MAX), the iteration
        // efficiencies are a starvation-floor cohort statistic (MEAN). Aligning
        // WorstLowest to an extremum (a MIN arm gated on HigherBetter) would be
        // a future product decision, tracked separately, not a Stage-1 fix.
        //
        // HYBRID caveat (sharpest for CV): a cross-RUN value here is a
        // mean-of-per-run-reductions, NOT a reduction recomputed over the
        // combined raw set — the raw samples do not survive cross-RUN (phases
        // are dropped), so there is no union to re-pool. For p99 / median /
        // mean run-delay this mean-of-summaries is a defensible cohort
        // statistic; for worst_wake_latency_cv it is a mean-of-ratios (the
        // fold-of-ready-made-ratios shape the Rate kind exists to avoid), not a
        // pooled CV — accepted here only because no combined set exists to
        // recompute over, and it reproduces the deleted typed path's shape
        // exactly. See [`MetricKind::Distribution`].
        MetricKind::Distribution { .. }
        | MetricKind::WorstLowest { .. }
        | MetricKind::WakeLatencyTailRatio
        | MetricKind::WorstCrossNodeRatio => finite.iter().sum::<f64>() / (finite.len() as f64),
        MetricKind::Gauge(GaugeAgg::Avg) => {
            // Weighted mean: sum(v * w) / sum(w). Uniform-weight
            // callers (aggregate_samples) reduce to arithmetic
            // mean per weight_for == |_| 1. Zero total weight
            // degenerates to the unweighted mean rather than
            // dividing by zero; mirrors `merge_metric_values` at
            // `crate::assert::merge_matched_phase_buckets` per
            // single-source-of-truth.
            //
            // `checked_add` on the running weight sum so a
            // pathological caller (huge per-RUN sample counts
            // across many runs) saturates to MAX rather than
            // wrapping silently in release. On overflow we
            // collapse to the unweighted-mean fallback so the
            // returned value stays plausible.
            let total_weight: usize = finite
                .iter()
                .enumerate()
                .try_fold(0usize, |acc, (i, _)| acc.checked_add(weight_for(i)))
                .unwrap_or(0);
            if total_weight == 0 {
                finite.iter().sum::<f64>() / (finite.len() as f64)
            } else {
                finite
                    .iter()
                    .enumerate()
                    .map(|(i, x)| *x * (weight_for(i) as f64))
                    .sum::<f64>()
                    / (total_weight as f64)
            }
        }
        MetricKind::Gauge(GaugeAgg::Last) | MetricKind::Timestamp => {
            *finite.last().expect("non-empty by check above")
        }
        MetricKind::Gauge(GaugeAgg::Max) | MetricKind::Peak => {
            finite.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        }
        // A Rate is derived from its components by `derive_rate_metrics`,
        // never reduced from a single sample slice (one slice cannot
        // express Σnum/Σdenom). EVERY aggregation path skips Rate before
        // reaching the reducers: `aggregate_samples_for_phase` returns
        // None, and the per-phase build, the cross-phase merge, and both
        // cross-RUN reducers skip Rate keys then re-derive via
        // `derive_rate_metrics`. So reaching here is a routing bug.
        MetricKind::Rate { .. } => unreachable!(
            "MetricKind::Rate must be derived via derive_rate_metrics, \
             not reduced from a sample slice"
        ),
        // PerPhase is derived post-merge by derive_phase_metrics (into
        // PhaseBucket.metrics and/or PhaseCgroupStats::metrics) and is gated out
        // of the cross-RUN ext fold (fold_ext_metrics) + the within-run reducers
        // (is_derived), so it never reaches a sample-slice reduction. Reaching
        // here is a routing bug (the gate or is_derived was bypassed).
        MetricKind::PerPhase => unreachable!(
            "MetricKind::PerPhase is derived by derive_phase_metrics, \
             not reduced from a sample slice"
        ),
        // PerRunDistribution is produced run-level by
        // populate_run_pooled_schbench_distribution (union of the per-phase
        // PlatStats histograms) and is gated out of BOTH the within-run reducers
        // (is_derived) AND the cross-RUN ext fold (fold_ext_metrics skip) — its
        // only consumer is noise_findings reading the per-run scalar. So it never
        // reaches a sample-slice reduction; reaching here is a routing bug.
        MetricKind::PerRunDistribution => unreachable!(
            "MetricKind::PerRunDistribution is produced by \
             populate_run_pooled_schbench_distribution and noise-compared per-run, \
             not reduced from a sample slice"
        ),
    })
}

/// Per-phase metric reduction with the correct semantic per
/// [`MetricKind`].
///
/// Counter kinds bypass [`aggregate_samples`]'s flat-run `sum`
/// (which is correct for cross-RUN aggregation, but wrong for
/// cumulative-since-boot per-phase data — summing 10 samples at
/// `[100, 150, 175, ...]` yields ~425 instead of the per-phase
/// delta `175 - 100 = 75`) and route through
/// [`phase_counter_delta`] instead. All other kinds use
/// [`aggregate_samples`] verbatim, which is correct for them
/// (Gauge avg/last/max, Peak max, Timestamp last, and DeltaSum — whose
/// samples are ALREADY per-read deltas, so the per-phase reduction is
/// the sum of the in-phase deltas, NOT a last-minus-first that would
/// difference two deltas into nonsense).
///
/// `samples` are the per-Sample readings of `metric` collected
/// over one phase's window of
/// [`crate::scenario::sample::Sample`]s via `MetricDef::read_sample`
/// once that helper is wired through.
/// Returns `None` when every reading was `None` / `NaN`.
///
/// Live caller: [`crate::assert::build_phase_buckets`] folds
/// per-phase sample slices through this entry point and the
/// result lands on [`crate::assert::PhaseBucket::metrics`]; the
/// host-side `evaluate_vm_result` at `src/test_support/eval/mod.rs`
/// is the consumer that drives the call.
pub fn aggregate_samples_for_phase(metric: &MetricDef, samples: &[f64]) -> Option<f64> {
    match metric.kind {
        MetricKind::Counter => phase_counter_delta(samples),
        // Derived kinds (every `is_derived()`: Rate / Distribution / WorstLowest /
        // WakeLatencyTailRatio / WorstCrossNodeRatio / PerPhase / PerRunDistribution)
        // have no samples
        // of their own: their value is produced by a post-pass
        // (`derive_rate_metrics` / `crate::assert::populate_run_distribution_metrics`)
        // from pooled components, not reduced from a per-phase slice. Return
        // None so the build loop inserts no key here.
        k if k.is_derived() => None,
        _ => aggregate_samples(samples, metric.kind),
    }
}

/// Per-phase reduction for [`MetricKind::Counter`]: compute the
/// last finite sample minus the first finite sample, clamping
/// negative results (counter reset across a scheduler restart)
/// to 0 and emitting a `tracing::warn!` so the reset is visible
/// in stderr. Mirrors the existing
/// `crate::monitor`-side counter-delta clamp pattern used
/// when reducing cumulative kernel counters across boundaries
/// for the same reset-detection reason.
///
/// Edge cases (sentinel-free absent-vs-measured-zero):
///   - 0 or 1 finite samples -> `None`. A delta is UNMEASURABLE from
///     fewer than two points; absence here is distinct from a measured
///     zero. The renderer's has-data signal is `PhaseBucket::sample_count`
///     (see `expect_metric`), NOT this value, so absence loses no
///     diagnostic. (Previously a 1-sample phase returned a phantom
///     `Some(0.0)` that made a per-phase Counter claim read 0 even when
///     the phase fired plenty — only one freeze landed.)
///   - 2+ finite samples -> `Some(max(0.0, last - first))` (equal
///     endpoints give a REAL `Some(0.0)`: the counter did not advance).
///
/// Live caller: [`aggregate_samples_for_phase`] dispatches the
/// Counter variant through this entry point.
pub fn phase_counter_delta(samples: &[f64]) -> Option<f64> {
    let finite: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();
    match finite.as_slice() {
        // 0 or 1 finite samples: a delta is unmeasurable from fewer than two
        // points -> None (sentinel-free contract). The 2+-equal case below
        // still yields a real Some(0.0).
        [] | [_] => None,
        [first, .., last] => {
            let delta = *last - *first;
            if delta < 0.0 {
                tracing::warn!(
                    first = *first,
                    last = *last,
                    "phase_counter_delta: counter reset detected (last < first); clamping to 0"
                );
                Some(0.0)
            } else {
                Some(delta)
            }
        }
    }
}

/// Derive every registered [`MetricKind::Rate`] metric in `metrics`
/// from its already-present numerator / denominator component values:
/// `metrics[rate] = metrics[numerator] / metrics[denominator]`.
///
/// This is the SOLE producer of a Rate metric's value. It runs as a
/// post-pass at seven aggregation sites where the components co-locate in
/// one map: the two per-phase builds, the cross-phase bucket merge, the
/// three cross-RUN ext-metrics reducers (`populate_run_ext_metrics`,
/// `populate_run_ext_metrics_from_phases`, `group_and_average_by`), and the
/// cross-CGROUP pooled re-pool
/// (`crate::assert::populate_run_pooled_iterations_per_cpu_sec`, run
/// post-`merge` at the eval layer to re-pool `iterations_per_cpu_sec` across a
/// run's cgroups). At each, the components are
/// pooled FIRST by their own kinds (a `Counter` numerator summed), then
/// the rate is re-derived — so for `Counter / Counter` the result is
/// `Σnumerator / Σdenominator`, the correct re-pool rather than a mean of
/// ready-made ratios. (The cross-CGROUP `AssertResult::merge` ext-metrics
/// fold itself uses worst-case polarity and is NOT a derive site — the
/// pooled re-pool above runs separately after it; see [`MetricKind::Rate`].)
///
/// A rate is skipped (its key left absent) when either component key is
/// missing, the denominator is zero, or either component is non-finite —
/// keeping an absent rate distinct from a real `0.0`.
///
/// INVARIANT: the producers must co-insert both components from the same
/// observation (both-or-neither per map) — e.g.
/// `build_phase_buckets_with_stimulus` inserts `total_phase_iterations` and
/// `total_phase_duration_sec` together under one `rate_components` guard. A
/// partial pair (numerator from one source, denominator from another) is
/// never produced today but would derive a cross-paired rate; any second
/// Rate must keep the co-insertion contract.
pub(crate) fn derive_rate_metrics(metrics: &mut std::collections::BTreeMap<String, f64>) {
    derive_rate_metrics_from(
        metrics,
        METRICS.iter().filter_map(|m| match m.kind {
            MetricKind::Rate {
                numerator,
                denominator,
            } => Some((m.name, numerator, denominator)),
            _ => None,
        }),
    );
}

/// Inner of [`derive_rate_metrics`] taking the rate specs explicitly as
/// `(name, numerator, denominator)` so the derivation math is
/// unit-testable without a registered Rate metric in [`METRICS`].
pub(crate) fn derive_rate_metrics_from<'a>(
    metrics: &mut std::collections::BTreeMap<String, f64>,
    rates: impl Iterator<Item = (&'a str, &'a str, &'a str)>,
) {
    for (name, numerator, denominator) in rates {
        let (Some(num), Some(den)) = (
            metrics.get(numerator).copied(),
            metrics.get(denominator).copied(),
        ) else {
            continue;
        };
        if num.is_finite() && den.is_finite() && den != 0.0 {
            // Guard the QUOTIENT too: a finite num / finite tiny den can
            // overflow to +/-inf. Insert only a finite rate so an absent
            // rate stays distinct from a real value (no inf in the map).
            let rate = num / den;
            if rate.is_finite() {
                metrics.insert(name.to_string(), rate);
            }
        }
    }
}

impl MetricDef {
    /// Read this metric's value from `row`. Consults the
    /// accessor first (for built-in `GauntletRow` fields) and
    /// falls back to `row.ext_metrics[self.name]` when the
    /// accessor returns `None`.
    pub fn read(&self, row: &GauntletRow) -> Option<f64> {
        (self.accessor)(row).or_else(|| row.ext_metrics.get(self.name).copied())
    }

    /// Read this metric's value from a single
    /// [`crate::scenario::sample::Sample`] — the per-sample
    /// analogue of [`Self::read`] used by the per-phase
    /// aggregator to fold a window of samples into one
    /// [`crate::assert::PhaseBucket`] value per metric.
    ///
    /// Returns `None` for metrics that cannot be derived from a
    /// single-sample shape: most ktstr metrics are computed host-side
    /// (cross-CPU / cross-cgroup folds, run-level distributional
    /// re-pools, or monitor-axis windowing), not from one sample —
    /// `worst_spread`, `worst_gap_ms`, `worst_migration_ratio`,
    /// `max_imbalance_ratio`, the `worst_*_wake_latency_*` /
    /// `worst_mean_run_delay_us` / `worst_run_delay_us` distributions,
    /// `worst_iterations_per_worker` / `worst_iterations_per_cpu_sec`,
    /// `worst_page_locality`, `worst_cross_node_migration_ratio`,
    /// `worst_wake_latency_tail_ratio` — and have no single-sample
    /// reading.
    ///
    /// Wired per-sample arms (return `Some`): `max_dsq_depth` /
    /// `avg_dsq_depth` from `sample.snapshot`'s DSQ-walker and
    /// `total_fallback` / `total_keep_last` from its SCX events
    /// region. Every other registered metric falls to `_ => None`
    /// here, for one of three reasons: (1) it is a MONITOR-axis
    /// signal with no guest-`Snapshot` shape (`stuck_count`,
    /// `max_imbalance_ratio`, `avg_imbalance_ratio`) — folded
    /// per-phase from `MonitorSample` windowing in
    /// [`crate::assert::build_phase_buckets`], NOT from read_sample;
    /// (2) it has no per-sample source yet (`total_migrations`,
    /// `total_iterations` — per-task guest counters not captured per
    /// tick); or (3) it is a run-level metric with no single-sample
    /// reading (the `worst_*` family above).
    /// [`crate::stats::aggregate_samples_for_phase`] surfaces an
    /// all-None reduction as a `None` bucket entry — distinct from
    /// `Some(0.0)` (a real zero) — so the bucket renderer can paint
    /// "no data" vs "real zero" distinctly without losing information.
    ///
    /// Live caller: [`crate::assert::build_phase_buckets`] calls
    /// `read_sample` once per [`crate::stats::METRICS`] entry per
    /// sample to collect the per-sample readings the per-phase
    /// aggregator folds. The host-side `evaluate_vm_result` at
    /// `src/test_support/eval/mod.rs` drives the chain.
    pub fn read_sample(&self, sample: &crate::scenario::sample::Sample<'_>) -> Option<f64> {
        // Per-metric dispatch by registry name. Only the metrics
        // whose value is genuinely a per-sample reading are wired;
        // the remaining 16 entries in the METRICS registry are
        // cross-cgroup folds or run-level distributional re-pools
        // computed host-side at `evaluate_vm_result` time
        // (worst-spread / worst-gap-ms fold; the
        // `worst_*_wake_latency_*` distributions + worst-iterations-per-
        // worker efficiencies re-pool) and have no single-sample
        // equivalent —
        // they fall through to None below and the phase
        // aggregator paints them as absent bucket entries
        // (distinct from a real zero — sentinel-free contract).
        match self.name {
            // BPF dsq-state walker captures per-DSQ depth at the
            // freeze instant. `local_dsq_depth` is the per-CPU
            // local DSQ; take max across CPUs because the metric
            // is Peak-kind ("worst depth this instant"). DsqState
            // sets `origin = "local cpu N"` for local DSQs (see
            // src/monitor/scx_walker.rs `DsqState::origin`); the
            // filter pins the metric to the local-DSQ class so
            // global / bypass / user DSQs do not pollute the
            // reading.
            "max_dsq_depth" => sample
                .snapshot
                .dsq_states()
                .iter()
                .filter(|d| d.origin.starts_with("local cpu "))
                .map(|d| u64::from(d.nr))
                .max()
                .map(|v| v as f64),
            // Per-sample arithmetic mean of the same local-CPU
            // DSQ depth readings `max_dsq_depth` walks. Returns
            // `None` when no local DSQs are present so the bucket
            // renderer can distinguish "no data" from "real zero"
            // (sentinel-free contract); a zero-population set
            // never enters the mean.
            "avg_dsq_depth" => {
                let locals: Vec<f64> = sample
                    .snapshot
                    .dsq_states()
                    .iter()
                    .filter(|d| d.origin.starts_with("local cpu "))
                    .map(|d| u64::from(d.nr) as f64)
                    .collect();
                if locals.is_empty() {
                    None
                } else {
                    Some(locals.iter().sum::<f64>() / locals.len() as f64)
                }
            }
            // Cumulative `select_cpu_fallback` counter at the
            // freeze instant. The host's event-counter walker
            // builds a per-tick timeline of CPU-summed counters
            // (`EventCounterSample` at src/monitor/dump/mod.rs:442);
            // `.last()` gives the cumulative reading at the most
            // recent tick within this freeze's capture window.
            // Counter-kind reduction folds `last - first` across
            // the phase's sample window, yielding the per-phase
            // delta (the genuine "how many fallbacks fired during
            // THIS phase").
            "total_fallback" => sample
                .snapshot
                .event_counter_timeline()
                .last()
                .map(|e| e.select_cpu_fallback as f64),
            // Cumulative `dispatch_keep_last` counter; same
            // per-tick timeline source as `total_fallback`. Same
            // Counter-kind reduction semantic; per-phase delta
            // surfaces the keep-last count for THIS phase.
            "total_keep_last" => sample
                .snapshot
                .event_counter_timeline()
                .last()
                .map(|e| e.dispatch_keep_last as f64),
            // IRQ observability: cross-CPU SUM of the cumulative per-CPU
            // counter at this freeze. Counter kind takes the per-phase
            // last-minus-first (phase_counter_delta) over these per-freeze
            // totals. The per-CPU set is fixed across freezes (every CPU is
            // present every freeze), so the cross-CPU sum has NO task-set-change
            // inflation — the exact reason system_time_ns below is NOT a
            // read_sample arm but these are. Empty per_cpu_time -> None
            // (loud-absent, never a false zero). Softirq vectors index against
            // the compile-pinned named consts, never bare literals.
            "total_hardirqs" => {
                let cpus = sample.snapshot.per_cpu_time();
                (!cpus.is_empty()).then(|| cpus.iter().map(|c| c.irqs_sum).sum::<u64>() as f64)
            }
            "total_softirq_net_rx" => {
                let cpus = sample.snapshot.per_cpu_time();
                (!cpus.is_empty()).then(|| {
                    cpus.iter()
                        .map(|c| c.softirqs[crate::monitor::btf_offsets::SOFTIRQ_NET_RX])
                        .sum::<u64>() as f64
                })
            }
            "total_softirq_net_tx" => {
                let cpus = sample.snapshot.per_cpu_time();
                (!cpus.is_empty()).then(|| {
                    cpus.iter()
                        .map(|c| c.softirqs[crate::monitor::btf_offsets::SOFTIRQ_NET_TX])
                        .sum::<u64>() as f64
                })
            }
            "total_softirq_timer" => {
                let cpus = sample.snapshot.per_cpu_time();
                (!cpus.is_empty()).then(|| {
                    cpus.iter()
                        .map(|c| c.softirqs[crate::monitor::btf_offsets::SOFTIRQ_TIMER])
                        .sum::<u64>() as f64
                })
            }
            "total_softirq_sched" => {
                let cpus = sample.snapshot.per_cpu_time();
                (!cpus.is_empty()).then(|| {
                    cpus.iter()
                        .map(|c| c.softirqs[crate::monitor::btf_offsets::SOFTIRQ_SCHED])
                        .sum::<u64>() as f64
                })
            }
            "total_irq_time_ns" => {
                let cpus = sample.snapshot.per_cpu_time();
                (!cpus.is_empty()).then(|| cpus.iter().map(|c| c.cpustat_irq_ns).sum::<u64>() as f64)
            }
            "total_softirq_time_ns" => {
                let cpus = sample.snapshot.per_cpu_time();
                (!cpus.is_empty())
                    .then(|| cpus.iter().map(|c| c.cpustat_softirq_ns).sum::<u64>() as f64)
            }
            "total_steal_time_ns" => {
                let cpus = sample.snapshot.per_cpu_time();
                (!cpus.is_empty())
                    .then(|| cpus.iter().map(|c| c.cpustat_steal_ns).sum::<u64>() as f64)
            }
            // `system_time_ns` / `user_time_ns` are deliberately absent
            // here: they are NOT read per-sample. A per-sample
            // cross-thread SUM followed by a Counter `last - first`
            // inflates whenever the captured task set changes between
            // freezes — a task carrying a large cumulative counter that
            // appears only in a LATER sample dumps its entire pre-phase
            // history into the delta. They are injected post-hoc as a
            // per-thread-GROUP delta (each tgid's first-seen-to-last-seen
            // `thread_group_cputime`) by
            // [`crate::assert::phase_group_cpu_delta`], which subtracts
            // each group's own first-seen total and so bounds the result
            // by wall-clock × cores. Still observer-free — that injector
            // reads the same frozen `task_struct` enrichments.
            //
            // Every other metric stays None. The 16 host-only
            // names (full list in the doc comment above) compute
            // cross-cgroup folds at `evaluate_vm_result` time and
            // have no per-sample equivalent until a per-cgroup
            // per-sample capture path lands; surfacing them via a
            // synthetic single-sample value would falsify the
            // per-phase trajectory the bucket renderer paints.
            _ => None,
        }
    }

    /// Returns `true` when a metric INCREASING is the bad direction:
    /// [`LowerBetter`](crate::test_support::Polarity::LowerBetter),
    /// [`TargetValue`](crate::test_support::Polarity::TargetValue), and the
    /// conservative [`Unknown`](crate::test_support::Polarity::Unknown)
    /// default (an unclassified metric is treated as higher-is-worse so a
    /// real regression in it is still caught). `false` for
    /// [`HigherBetter`](crate::test_support::Polarity::HigherBetter) and
    /// [`Informational`](crate::test_support::Polarity::Informational).
    ///
    /// This is the cross-cgroup FOLD direction (max-vs-min when merging
    /// per-cgroup `ext_metrics`, see [`crate::assert::AssertResult::merge`])
    /// and the timeline-narrative direction. The PERF-DELTA VERDICT path
    /// uses [`classify_direction`](Self::classify_direction) instead, which
    /// returns `None` for `Informational` so it never gates. `Informational`
    /// folding as `false` (min) here is harmless: the system-wide monitor
    /// counters that carry it are row-level and never hit the per-cgroup
    /// merge.
    pub const fn higher_is_worse(&self) -> bool {
        use crate::test_support::Polarity;
        matches!(
            self.polarity,
            Polarity::LowerBetter | Polarity::TargetValue(_) | Polarity::Unknown
        )
    }

    /// Verdict direction for the perf-delta comparison:
    /// - `Some(true)`  — an INCREASE is a regression (`LowerBetter` /
    ///   `TargetValue` / the conservative `Unknown` default),
    /// - `Some(false)` — a DECREASE is a regression (`HigherBetter`),
    /// - `None`        — [`Informational`](crate::test_support::Polarity::Informational):
    ///   directionless; the comparison records and displays it but NEVER
    ///   classifies it as regression/improvement and it NEVER affects the
    ///   exit code.
    ///
    /// The verdict sites in [`compare_partitions`] branch on the `Option`:
    /// `None` => informational, `Some(hiw)` => the dual-gated
    /// regression/improvement split. This matches
    /// [`higher_is_worse`](Self::higher_is_worse) for every variant EXCEPT
    /// `Informational` (there `false`/min-fold, here `None`/never-gated) —
    /// the deliberate split between "fold needs a direction" and "verdict
    /// must stay neutral".
    pub const fn classify_direction(&self) -> Option<bool> {
        use crate::test_support::Polarity;
        match self.polarity {
            Polarity::Informational => None,
            Polarity::HigherBetter => Some(false),
            Polarity::LowerBetter | Polarity::TargetValue(_) | Polarity::Unknown => Some(true),
        }
    }
}

/// Unified metric registry covering all built-in and extensible metrics.
///
/// The comparison pipeline uses `higher_is_worse` to determine regression
/// direction, `default_abs`/`default_rel` for dual-gate significance
/// thresholds, and `display_unit` for formatted output. Per-test
/// assertion overrides can still use their own thresholds; this registry
/// is the source of truth for polarity and display.
///
/// `AssertResult::merge` consults `higher_is_worse` via [`metric_def`]
/// when folding per-cgroup `ext_metrics` into the scenario-level worst
/// case: `true` takes max, `false` takes min. Unknown names (not in
/// this registry) default to max; register a `MetricDef` here before
/// relying on min-polarity merge. The comparison system
/// ([`compare_partitions`]) uses `higher_is_worse` for delta direction.
///
/// # Metric-name triples (registry / field / DataFrame column)
///
/// Each metric is referenced by three names across the pipeline.
/// The registry name is the stable surface — sidecars, CI gates,
/// and `cargo ktstr stats compare` output all quote it verbatim —
/// and cannot be renamed without silently invalidating downstream
/// consumers. The field name on [`GauntletRow`] and the polars
/// DataFrame column name are internal; they are kept terse and
/// match each other, but diverge from the registry name where
/// the domain-level wording adds context (`worst_*`, `total_*`,
/// `max_*`) that would be noise on an already-qualified field.
/// Nine divergent triples:
///
/// | Registry (`MetricDef.name`) | `GauntletRow` field | DataFrame column |
/// |---|---|---|
/// | `worst_spread` | `spread` | `spread` |
/// | `worst_gap_ms` | `gap_ms` | `gap_ms` |
/// | `total_migrations` | `migrations` | `migrations` |
/// | `worst_migration_ratio` | `migration_ratio` | `migration_ratio` |
/// | `max_imbalance_ratio` | `imbalance_ratio` | `imbalance` |
/// | `max_dsq_depth` | `max_dsq_depth` | `dsq_depth` |
/// | `stuck_count` | `stuck_count` | `stuck` |
/// | `total_fallback` | `fallback_count` | `fallback` |
/// | `total_keep_last` | `keep_last_count` | `keep_last` |
///
/// One of the remaining metrics in [`METRICS`] has matching
/// registry / field / DataFrame column names backed by a typed
/// `GauntletRow` field (`total_iterations`) and is not listed — no
/// translation to document.
///
/// The ten wake-latency / run-delay / iteration-efficiency / NUMA roll-ups
/// (`worst_p99_wake_latency_us`, `worst_median_wake_latency_us`,
/// `worst_wake_latency_cv`, `worst_mean_run_delay_us`,
/// `worst_run_delay_us`, `worst_iterations_per_worker`,
/// `worst_iterations_per_cpu_sec`, `worst_wake_latency_tail_ratio`,
/// `worst_page_locality`, `worst_cross_node_migration_ratio`) are
/// DERIVED kinds ([`MetricKind::Distribution`] / [`MetricKind::WorstLowest`]
/// / [`MetricKind::WakeLatencyTailRatio`] / [`MetricKind::WorstCrossNodeRatio`])
/// with NO typed `GauntletRow`
/// field: their accessors are `|_| None` and
/// `crate::assert::populate_run_distribution_metrics` re-pools their value
/// into `ext_metrics` post-merge, so [`MetricDef::read`] reads them through
/// the ext fallback.
///
/// `worst_` naming convention: it is the codebase-wide prefix for a
/// cross-cgroup roll-up, independent of polarity and of HOW the roll-up is
/// formed. Polarity-directional selectors (`worst_spread`, and the derived
/// `worst_cross_node_migration_ratio`, both LowerBetter → max) and
/// [`MetricKind::WorstLowest`] (`worst_page_locality` +
/// `worst_iterations_per_*`, None-aware lowest-wins where a measured 0.0
/// wins) both surface the most problematic cgroup; whereas
/// [`MetricKind::Distribution`] (`worst_p99_wake_latency_us` etc.) is the
/// POOLED cross-cgroup distribution over the combined sample set, NOT a
/// per-cgroup selection — here `worst_` is retained for sidecar /
/// DataFrame / CI-gate name stability rather than literal accuracy. A
/// `lowest_*` rename of the HigherBetter selectors was weighed and
/// rejected as a high-churn rename across sidecars / DataFrames / CI gates
/// for no readability gain.
///
/// Quoting the matching list instead of a bare count avoids
/// silent drift on rename: a metric whose registry / field /
/// column names diverge belongs in the table above, while a
/// matching triple belongs in this paragraph; a future rename
/// that forgets to migrate the metric across the boundary
/// surfaces here as a stale list rather than a wrong count.
///
/// Consumers that cross the registry / DataFrame boundary should
/// go through [`MetricDef::read`] / the accessor closure rather
/// than hand-translating by string. The four-name mapping for
/// `worst_spread` specifically is documented in detail on the
/// [`GauntletRow::spread`] field (adds the
/// [`ScenarioStats::worst_spread`](crate::assert::ScenarioStats::worst_spread)
/// upstream source as a fourth name).
/// Registry names for the schbench per-phase metrics ([`MetricKind::PerPhase`]).
/// Shared by the [`METRICS`] entries below and the schbench per-phase derivation
/// (`crate::assert::derive_phase_metrics`) so the registered name and
/// the key the derivation writes into `crate::assert::PhaseBucket::metrics` are
/// one source of truth. Latency keys are µs (the unit `plat` buckets in);
/// sched-delay keys are µs (converted from ns at derivation); loop_count is a
/// bare count.
pub(crate) const SCHBENCH_WAKEUP_P50_US: &str = "wakeup_p50_latency_us";
pub(crate) const SCHBENCH_WAKEUP_P90_US: &str = "wakeup_p90_latency_us";
pub(crate) const SCHBENCH_WAKEUP_P99_US: &str = "wakeup_p99_latency_us";
pub(crate) const SCHBENCH_WAKEUP_P999_US: &str = "wakeup_p999_latency_us";
pub(crate) const SCHBENCH_REQUEST_P50_US: &str = "request_p50_latency_us";
pub(crate) const SCHBENCH_REQUEST_P90_US: &str = "request_p90_latency_us";
pub(crate) const SCHBENCH_REQUEST_P99_US: &str = "request_p99_latency_us";
pub(crate) const SCHBENCH_REQUEST_P999_US: &str = "request_p999_latency_us";
pub(crate) const SCHBENCH_SCHED_DELAY_MSG_US: &str = "sched_delay_msg_us";
pub(crate) const SCHBENCH_SCHED_DELAY_WORKER_US: &str = "sched_delay_worker_us";
pub(crate) const SCHBENCH_LOOP_COUNT: &str = "schbench_loop_count";
// schbench WHOLE-RUN Class-3 keys (loop count + role-separate run-delay gate
// Rates) for perf-delta --noise-adjust. Re-pooled run-level by
// `populate_run_pooled_schbench` from the per-phase per-cgroup
// `SchbenchPhaseStats` raw pairs (Σ over all phases+cgroups). The four
// `total_schbench_*` run-delay/pcount Counters are the rate components
// (RENDER_SUPPRESSED) for the two sample-weighted Σrun_delay/Σpcount gate Rates
// (workload-scoped siblings of the system-wide `total_run_delay_ns_per_sched`);
// the message and worker thread ROLES pool separately (different per-schedule
// wait populations). Distinct from the per-phase `sched_delay_msg/worker_us`
// (mean-of-means, schbench.c parity) which stay PerPhase display-only — the
// Rates gate, no double-count. `total_schbench_loops` is the whole-run loop
// Counter (distinct from the per-phase `schbench_loop_count`).
pub(crate) const TOTAL_SCHBENCH_MSG_RUN_DELAY_NS: &str = "total_schbench_msg_run_delay_ns";
pub(crate) const TOTAL_SCHBENCH_MSG_PCOUNT: &str = "total_schbench_msg_pcount";
pub(crate) const TOTAL_SCHBENCH_WORKER_RUN_DELAY_NS: &str = "total_schbench_worker_run_delay_ns";
pub(crate) const TOTAL_SCHBENCH_WORKER_PCOUNT: &str = "total_schbench_worker_pcount";
pub(crate) const TOTAL_SCHBENCH_LOOPS: &str = "total_schbench_loops";
pub(crate) const SCHBENCH_MSG_RUN_DELAY_NS_PER_SCHED: &str = "schbench_msg_run_delay_ns_per_sched";
pub(crate) const SCHBENCH_WORKER_RUN_DELAY_NS_PER_SCHED: &str =
    "schbench_worker_run_delay_ns_per_sched";
// schbench WHOLE-RUN distributional keys for perf-delta --noise-adjust:
// each per-phase percentile/min/max re-pooled run-level by
// `populate_run_pooled_schbench_distribution` (union of the per-phase per-cgroup
// PlatStats histograms, percentile re-derived over the union). MetricKind::
// PerRunDistribution: noise-compared per-run, NEVER cross-run folded. `*_whole`
// names keep them registry-distinct from the per-phase PerPhase keys above (one
// name = one kind). Latency LowerBetter; rps HigherBetter.
pub(crate) const SCHBENCH_WAKEUP_P50_US_WHOLE: &str = "wakeup_p50_latency_us_whole";
pub(crate) const SCHBENCH_WAKEUP_P90_US_WHOLE: &str = "wakeup_p90_latency_us_whole";
pub(crate) const SCHBENCH_WAKEUP_P99_US_WHOLE: &str = "wakeup_p99_latency_us_whole";
pub(crate) const SCHBENCH_WAKEUP_P999_US_WHOLE: &str = "wakeup_p999_latency_us_whole";
pub(crate) const SCHBENCH_WAKEUP_MIN_US_WHOLE: &str = "wakeup_min_latency_us_whole";
pub(crate) const SCHBENCH_WAKEUP_MAX_US_WHOLE: &str = "wakeup_max_latency_us_whole";
pub(crate) const SCHBENCH_REQUEST_P50_US_WHOLE: &str = "request_p50_latency_us_whole";
pub(crate) const SCHBENCH_REQUEST_P90_US_WHOLE: &str = "request_p90_latency_us_whole";
pub(crate) const SCHBENCH_REQUEST_P99_US_WHOLE: &str = "request_p99_latency_us_whole";
pub(crate) const SCHBENCH_REQUEST_P999_US_WHOLE: &str = "request_p999_latency_us_whole";
pub(crate) const SCHBENCH_REQUEST_MIN_US_WHOLE: &str = "request_min_latency_us_whole";
pub(crate) const SCHBENCH_REQUEST_MAX_US_WHOLE: &str = "request_max_latency_us_whole";
pub(crate) const SCHBENCH_RPS_P20_WHOLE: &str = "rps_p20_whole";
pub(crate) const SCHBENCH_RPS_P50_WHOLE: &str = "rps_p50_whole";
pub(crate) const SCHBENCH_RPS_P90_WHOLE: &str = "rps_p90_whole";
pub(crate) const SCHBENCH_RPS_MIN_WHOLE: &str = "rps_min_whole";
pub(crate) const SCHBENCH_RPS_MAX_WHOLE: &str = "rps_max_whole";
// taobench per-phase metric keys (the WorkType::Taobench engine's qps + hit
// ratios, derived per-phase by write_taobench_scalars; MetricKind::PerPhase).
// total/fast qps are HigherBetter; slow_qps + hit_ratio + hit_rate are
// Informational (slow_qps is a component, not a direction; the hit numbers are
// run-validity signals, not regression directions).
pub(crate) const TAOBENCH_TOTAL_QPS: &str = "taobench_total_qps";
pub(crate) const TAOBENCH_FAST_QPS: &str = "taobench_fast_qps";
pub(crate) const TAOBENCH_SLOW_QPS: &str = "taobench_slow_qps";
pub(crate) const TAOBENCH_HIT_RATIO: &str = "taobench_hit_ratio";
pub(crate) const TAOBENCH_HIT_RATE: &str = "taobench_hit_rate";
// taobench WHOLE-RUN Rate component + Rate keys (the run-level qps + hit
// fraction, pooled cross-cgroup by `populate_run_pooled_taobench` and derived by
// `derive_rate_metrics`). Distinct from the per-phase `taobench_*_qps` above
// (`MetricKind::PerPhase`, invisible to the whole-run cross-run fold): these are
// registered `Rate`/`Counter` METRICS so they reach perf-delta `--noise-adjust`
// spread. The four `total_taobench_*` Counters are the rate components (their
// `total_` prefix satisfies the Counter naming gate); they are
// `RENDER_SUPPRESSED_COMPONENTS` so the compare output shows the rates, not the
// raw counts.
pub(crate) const TOTAL_TAOBENCH_OPS: &str = "total_taobench_ops";
pub(crate) const TOTAL_TAOBENCH_FAST_OPS: &str = "total_taobench_fast_ops";
pub(crate) const TOTAL_TAOBENCH_SLOW_OPS: &str = "total_taobench_slow_ops";
pub(crate) const TOTAL_TAOBENCH_WALL_SEC: &str = "total_taobench_wall_sec";
pub(crate) const TAOBENCH_TOTAL_OPS_PER_SEC: &str = "taobench_total_ops_per_sec";
pub(crate) const TAOBENCH_FAST_OPS_PER_SEC: &str = "taobench_fast_ops_per_sec";
pub(crate) const TAOBENCH_SLOW_OPS_PER_SEC: &str = "taobench_slow_ops_per_sec";
pub(crate) const TAOBENCH_HIT_FRACTION: &str = "taobench_hit_fraction";
// Per-phase latency min/max (schbench's `min=`/`max=` table footer,
// `schbench.c:579`): the per-phase PlatStats already carries them, so these are
// emitted from `q.min`/`q.max`. LowerBetter (a higher min/max latency is worse).
pub(crate) const SCHBENCH_WAKEUP_MIN_US: &str = "wakeup_min_latency_us";
pub(crate) const SCHBENCH_WAKEUP_MAX_US: &str = "wakeup_max_latency_us";
pub(crate) const SCHBENCH_REQUEST_MIN_US: &str = "request_min_latency_us";
pub(crate) const SCHBENCH_REQUEST_MAX_US: &str = "request_max_latency_us";
// Per-phase achieved-RPS distribution (schbench's RPS table, PLIST_FOR_RPS =
// 20/50/90, `schbench.c:130`) + its min/max. HigherBetter (more requests/sec =
// more throughput); the min/max INVERT the latency polarity (a higher worst-
// second rate is better). A per-second RATE, so no `_us` suffix.
pub(crate) const SCHBENCH_RPS_P20: &str = "rps_p20";
pub(crate) const SCHBENCH_RPS_P50: &str = "rps_p50";
pub(crate) const SCHBENCH_RPS_P90: &str = "rps_p90";
pub(crate) const SCHBENCH_RPS_MIN: &str = "rps_min";
pub(crate) const SCHBENCH_RPS_MAX: &str = "rps_max";

pub static METRICS: &[MetricDef] = &[
    MetricDef {
        // `"worst_spread"` is the wire/surface name — emitted in
        // sidecars, referenced by CI gates, and printed by
        // `cargo ktstr stats compare`. Internally the field on
        // `GauntletRow` is named `spread` and the polars DataFrame
        // column keeps that shorter name; see the doc on
        // `GauntletRow.spread` for the rationale (rename-of-
        // registry-name is not safe because existing gate configs
        // match this string by value).
        name: "worst_spread",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Last),
        default_abs: 5.0,
        default_rel: 0.25,
        display_unit: "%",
        accessor: |r| Some(r.spread),
    },
    MetricDef {
        name: "worst_gap_ms",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Peak,
        default_abs: 500.0,
        default_rel: 0.50,
        display_unit: "ms",
        accessor: |r| Some(r.gap_ms as f64),
    },
    MetricDef {
        name: "total_migrations",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Counter,
        default_abs: 10.0,
        default_rel: 0.30,
        display_unit: "",
        accessor: |r| Some(r.migrations as f64),
    },
    MetricDef {
        name: "worst_migration_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Last),
        default_abs: 0.05,
        default_rel: 0.20,
        display_unit: "",
        accessor: |r| Some(r.migration_ratio),
    },
    MetricDef {
        name: "max_imbalance_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Peak,
        default_abs: 1.0,
        default_rel: 0.25,
        display_unit: "x",
        accessor: |r| Some(r.imbalance_ratio),
    },
    MetricDef {
        // Per-phase mean of per-tick imbalance_ratio observations
        // (max(nr_running) / max(1, min(nr_running)) per CPU; full-
        // class count). Sourced from MonitorSample (not Snapshot)
        // because Snapshot exposes only scx_rq.nr_running (SCX-
        // only) while imbalance is meaningful only across the
        // full per-CPU runqueue. Populated by build_phase_buckets
        // via per-phase MonitorSample windowing — bypasses
        // MetricDef::read_sample (which dispatches off
        // sample.snapshot only) per the data-axis split. Kind
        // Gauge(Avg) folds across cgroups via weighted-mean per
        // sample_count; Polarity::LowerBetter mirrors the Peak
        // sibling.
        name: "avg_imbalance_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Avg),
        default_abs: 0.5,
        default_rel: 0.25,
        display_unit: "x",
        accessor: |_| None,
    },
    MetricDef {
        name: "max_dsq_depth",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Peak,
        default_abs: 10.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |r| Some(r.max_dsq_depth as f64),
    },
    MetricDef {
        // Per-sample mean of local-CPU DSQ depths sourced from
        // the BPF DSQ walker (Snapshot::dsq_states() filtered by
        // `origin.starts_with("local cpu ")`), reduced per phase
        // via the Gauge(Avg) path. The DSQ-walker axis is the
        // authoritative source — it reads the actual scheduler
        // dispatch queues. The legacy Timeline::build path
        // computed avg_dsq_depth from MonitorSample.CpuSnapshot.
        // local_dsq_depth (a per-CPU rq-level metric); the new
        // DSQ-walker axis is more accurate for an scx scheduler
        // because it observes the dispatch queue directly rather
        // than the rq-level reflection.
        //
        // Truncation caveat: when scx_walker hits MAX_NODES_PER_LIST
        // (per src/monitor/scx_walker.rs), the captured DSQs are a
        // prefix of the full set. The mean then shifts toward the
        // captured prefix's central tendency; a 64-CPU box capturing
        // only 20 DSQs reports the mean of those 20, not the mean
        // over 64. max_dsq_depth (the Peak sibling) is robust to
        // this (max-of-captured surfaces the deepest queue ever
        // captured); avg_dsq_depth has no such monotonicity. If
        // walker truncation becomes routine, add a denom-aware
        // version that sums-then-divides by the topology's
        // expected local-CPU count.
        //
        // Accessor falls back to ext_metrics (no typed GauntletRow
        // field; promoting to typed is gated on cross-RUN
        // aggregation needs surfacing).
        name: "avg_dsq_depth",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Avg),
        default_abs: 5.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        name: "stuck_count",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Counter,
        // abs=1.0 (vs 5.0 for the event counters below): one additional
        // scheduler stall is high-signal, so a delta of a single whole
        // stall — gated by the 0.50 rel threshold — is worth flagging.
        default_abs: 1.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |r| Some(r.stuck_count),
    },
    MetricDef {
        name: "total_fallback",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Counter,
        default_abs: 5.0,
        default_rel: 0.30,
        // Integer event count, not a rate — the source field on
        // `MonitorSummary::event_deltas.total_fallback` is a cumulative
        // delta across the run, not per-second. Empty unit matches the
        // other counter metrics (`stuck_count`, `total_iterations`,
        // `total_migrations`).
        display_unit: "",
        accessor: |r| Some(r.fallback_count as f64),
    },
    MetricDef {
        name: "total_keep_last",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Counter,
        default_abs: 5.0,
        default_rel: 0.30,
        // Integer event count, not a rate — see `total_fallback`
        // rationale above. Source field is
        // `MonitorSummary::event_deltas.total_dispatch_keep_last`.
        display_unit: "",
        accessor: |r| Some(r.keep_last_count as f64),
    },
    // -- System-wide schedstat aggregates. Read host-side from guest memory at
    // -- freeze (zero observer effect) via `MonitorSummary::schedstat_deltas`
    // -- (per-rq `struct rq` schedstat fields summed across CPUs over the run);
    // -- `sidecar_to_row` inserts them into `GauntletRow::ext_metrics` so the
    // -- `|_| None` accessors surface them through the ext fallback. The seven
    // -- raw counters are `Polarity::Informational` — directionless (more
    // -- wakeups / context-switches / yields is neither inherently better nor
    // -- worse), so they are SHOWN but NEVER gated. They are also
    // -- WINDOW-DURATION- and LOAD-CONFOUNDED raw sums (a longer monitor window
    // -- or more offered runnable work inflates them independent of the
    // -- scheduler) — a second reason they are Informational, not LowerBetter:
    // -- a large raw delta is not a regression. The duration- and load-robust
    // -- GATED signals are the per-schedule mean (`total_run_delay_ns_per_sched`)
    // -- and the locality ratio (`ttwu_local_fraction`) derived below; four of
    // -- the raw counters double as those Rates' Counter components.
    MetricDef {
        // Numerator of `total_run_delay_ns_per_sched`. Cumulative runqueue-wait
        // delay (ns) across all tasks + all CPUs (`rq.rq_sched_info.run_delay`).
        // `total_` prefix satisfies the Counter naming gate.
        name: "total_run_delay",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 1_000_000.0,
        default_rel: 0.10,
        display_unit: "ns",
        accessor: |_| None,
    },
    MetricDef {
        // Denominator of `total_run_delay_ns_per_sched`. Count of non-idle task
        // arrivals (`rq.rq_sched_info.pcount`) — the number of schedules the
        // run-delay accrued over.
        name: "total_pcount",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Context-switch count (`rq.sched_count`). Informational: more
        // context-switches can mean responsiveness OR thrashing — no direction.
        name: "total_sched_count",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // `sched_yield()` call count (`rq.yld_count`). Informational (workload
        // behavior, not a scheduler-quality signal).
        name: "total_yld_count",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Go-idle count (`rq.sched_goidle`): times a CPU picked the idle task.
        // Informational (good utilization vs wasted idle — ambiguous).
        name: "total_sched_goidle",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Denominator of `ttwu_local_fraction`. Total wakeups (`rq.ttwu_count`)
        // — workload activity. Informational.
        name: "total_ttwu_count",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Numerator of `ttwu_local_fraction`. Wakeups kept on the waking CPU
        // (`rq.ttwu_local`). Informational on its own; the locality RATIO below
        // carries the direction.
        name: "total_ttwu_local",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // GATED. System-wide per-schedule MEAN runqueue-wait delay =
        // Σrun_delay / Σpcount, re-derived across CPUs/runs by
        // `derive_rate_metrics` (the `Rate` kind's `MergeKind::Recompute` pools
        // the components — never a mean-of-ratios). Duration- and load-robust
        // (per-EVENT, not per-time): the system-wide analog of the
        // workload-scoped per-task `mean_run_delay_us` (schbench's
        // `mean_sched_delay = run_delay/pcount`). LowerBetter. Absent when
        // `total_pcount` is 0 (no schedules) or CONFIG_SCHEDSTATS is off
        // (components absent).
        name: "total_run_delay_ns_per_sched",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Rate {
            numerator: "total_run_delay",
            denominator: "total_pcount",
        },
        default_abs: 1000.0,
        default_rel: 0.15,
        display_unit: "ns",
        accessor: |_| None,
    },
    MetricDef {
        // GATED. Wakeup LOCALITY = Σttwu_local / Σttwu_count, re-derived by
        // `derive_rate_metrics`. A fraction in [0, 1]: the share of wakeups kept
        // on the waking CPU (better cache locality, fewer cross-CPU hops on
        // wakeup). HigherBetter. Absent when `total_ttwu_count` is 0 or
        // CONFIG_SCHEDSTATS is off.
        name: "ttwu_local_fraction",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Rate {
            numerator: "total_ttwu_local",
            denominator: "total_ttwu_count",
        },
        default_abs: 0.05,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // GATED. Go-idle FRACTION = Σsched_goidle / Σsched_count, re-derived by
        // `derive_rate_metrics`. A fraction in [0, 1]: the share of `schedule()`
        // calls that picked the idle task (the CPU found nothing runnable).
        // Load-normalized (per-schedule, not per-time), so it is duration- AND
        // arrival-rate-stable — the genuinely-useful-for-spread schedstat rate
        // (a bare per-second rate carries the same spread as the raw total when
        // cohort runs share a duration, so it adds nothing at equal duration).
        // Informational: a high idle fraction is ambiguous — efficient when no
        // runnable work exists, but a starvation symptom when runnable work is
        // not dispatched — so it surfaces in `--noise-adjust` spread but does
        // not gate a regression verdict. Absent when `total_sched_count` is 0
        // (no schedules) or CONFIG_SCHEDSTATS is off (components absent).
        name: "sched_goidle_fraction",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Rate {
            numerator: "total_sched_goidle",
            denominator: "total_sched_count",
        },
        default_abs: 0.05,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Whole-run mean per-CPU runqueue depth (`rq.nr_running`, ALL scheduling
        // classes), read host-side from guest memory via
        // `MonitorSummary::avg_nr_running`. The occupancy LEVEL — distinct from
        // `avg_dsq_depth` (scx DSQ only) and `avg_imbalance_ratio` (cross-CPU
        // SKEW, not level). `MetricKind::Gauge(Avg)`: the cross-run fold is the
        // sample-weighted pooled mean (Σ avg×samples / Σ samples via
        // `aggregate_samples_weighted`, weight = `run_sample_count`). The weight
        // is sample count, not samples×CPUs — EXACT under same-topology pairing
        // (CPU count is a pairing dim, so cross-folded runs share it; the same
        // basis `avg_imbalance_ratio` uses). LowerBetter — higher mean depth =
        // more tasks waiting, but load-confounded (more offered runnable tasks
        // raises it independent of the scheduler), the same caveat
        // `avg_dsq_depth` carries; meaningful for same-offered-load A/B.
        // ext_metrics-only (accessor `|_| None`, surfaced via the ext fallback);
        // absent when the run has no monitor samples.
        name: "avg_nr_running",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Avg),
        default_abs: 0.5,
        default_rel: 0.20,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Wake-latency p99, re-pooled over the COMBINED wake-latency sample
        // set across every cgroup (and phase), NOT a max of per-cgroup p99s.
        // Distribution kind: derived post-merge by
        // `crate::assert::populate_run_distribution_metrics`; accessor is
        // |_| None so `MetricDef::read` takes the ext_metrics value the
        // re-pool writes. (The `worst_` name is retained for sidecar /
        // DataFrame / CI-gate stability — see the `worst_` naming
        // convention on [`METRICS`].)
        name: "worst_p99_wake_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Distribution {
            source: SampleSource::WakeLatencyNs,
            reduction: SampleReduction::P99,
        },
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        // Wake-latency median (50th pct), re-pooled over the combined wake
        // set — see `worst_p99_wake_latency_us`.
        name: "worst_median_wake_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Distribution {
            source: SampleSource::WakeLatencyNs,
            reduction: SampleReduction::Median,
        },
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        // Wake-latency coefficient of variation (stddev/mean), re-pooled
        // over the combined wake set with a population-WEIGHTED variance and
        // mean (denominator = Σ per-sample population weights, i.e. the
        // reconstructed true wakeup population; == `pool.len()` only below the
        // reservoir cap, where every weight is 1.0) — see
        // `worst_p99_wake_latency_us`.
        name: "worst_wake_latency_cv",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Distribution {
            source: SampleSource::WakeLatencyNs,
            reduction: SampleReduction::Cv,
        },
        default_abs: 0.10,
        default_rel: 0.25,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Run-level timer-latency p99 (WorkType::TimerLatency cyclictest probe),
        // re-pooled over the combined timer-latency sample set across every
        // cgroup and phase (NOT a max of per-cgroup p99s). Distribution: derived
        // post-merge by populate_run_distribution_metrics; accessor |_| None
        // reads the ext_metrics value the re-pool writes.
        name: "worst_p99_timer_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Distribution {
            source: SampleSource::TimerLatencyNs,
            reduction: SampleReduction::P99,
        },
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        // Run-level timer-latency median — see worst_p99_timer_latency_us.
        name: "worst_median_timer_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Distribution {
            source: SampleSource::TimerLatencyNs,
            reduction: SampleReduction::Median,
        },
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        // Run-level timer-latency p99.9 (the deep RT tail) — see
        // worst_p99_timer_latency_us.
        name: "worst_p999_timer_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Distribution {
            source: SampleSource::TimerLatencyNs,
            reduction: SampleReduction::P999,
        },
        default_abs: 100.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        // Run-level WORST (max) timer-latency — the cyclictest headline.
        // MAX-folds cross-RUN (SampleReduction::Worst, the peak survives) via
        // aggregate_finite, distinct from the MEAN-folded percentiles above.
        // Named worst_* with no pNN exactly like worst_run_delay_us
        // (Distribution{RunDelayNs, Worst}).
        name: "worst_timer_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Distribution {
            source: SampleSource::TimerLatencyNs,
            reduction: SampleReduction::Worst,
        },
        default_abs: 200.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        // Per-phase worker iterations per second. MetricKind::Rate with
        // Counter components total_phase_iterations / total_phase_duration_sec:
        // build_phase_buckets_with_stimulus emits those two components (the
        // iteration delta + the window seconds) from adjacent stimulus events'
        // total_iterations / elapsed_ms deltas — NOT a ready ratio — and
        // derive_rate_metrics re-derives iteration_rate = Σiterations /
        // Σseconds, so it re-pools correctly across phases/runs rather than
        // averaging per-phase ratios. Higher-is-better (more throughput). The
        // registry entry exists so MetricDef::read on a
        // GauntletRow.ext_metrics fallback surfaces it through cargo ktstr
        // stats compare like any other metric, and so
        // Timeline::from_phase_buckets reads it by the canonical name from
        // PhaseBucket.metrics. No typed GauntletRow field; accessor is the
        // ext_metrics fallback.
        name: "iteration_rate",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Rate {
            numerator: "total_phase_iterations",
            denominator: "total_phase_duration_sec",
        },
        default_abs: 1.0,
        default_rel: 0.30,
        display_unit: "iter/s",
        accessor: |_| None,
    },
    MetricDef {
        name: "total_iterations",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |r| Some(r.total_iterations as f64),
    },
    MetricDef {
        // Per-phase iteration delta — the NUMERATOR component of the
        // `iteration_rate` Rate. ext_metrics-only (no GauntletRow field):
        // inserted per phase as the last-minus-first delta of the cumulative
        // iteration counter, alongside `total_phase_duration_sec`, so
        // `derive_rate_metrics` yields `iteration_rate` = Σ(iter delta) /
        // Σ(phase seconds). `total_` prefix satisfies the Counter naming gate.
        name: "total_phase_iterations",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Per-phase WALL-clock duration in SECONDS — the DENOMINATOR
        // component of the `iteration_rate` Rate. ext_metrics-only. The
        // ms→s conversion is applied at the component-insertion site (NOT in
        // `derive_rate_metrics`, which does a bare num/den with no scaling),
        // so the stored value is already seconds and the derived rate is
        // iterations/second. `total_` prefix satisfies the Counter naming gate.
        name: "total_phase_duration_sec",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Counter,
        default_abs: 1.0,
        default_rel: 0.30,
        display_unit: "s",
        accessor: |_| None,
    },
    MetricDef {
        // Run-level POOLED CPU-seconds — the DENOMINATOR component of the
        // pooled `iterations_per_cpu_sec` Rate. ext_metrics-only (accessor
        // |_| None): populate_run_pooled_iterations_per_cpu_sec sums the
        // MEASURED cgroups' CgroupStats.total_cpu_time_ns (total_cpu_time_ns >
        // 0) and inserts the ns→s value (= Σns / 1e9) at the post-merge eval
        // site. The measured-only filter leaves this denominator unchanged
        // (excluded cgroups contribute 0 ns) — it matters for the numerator,
        // whose excluded cgroups carry nonzero iterations. The /1e9 lives
        // there (NOT in derive_rate_metrics, which does a bare num/den),
        // applied ONCE on the summed ns. `total_` prefix satisfies the Counter
        // gate.
        name: "total_cpu_time_sec",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Counter,
        default_abs: 1.0,
        default_rel: 0.30,
        display_unit: "s",
        accessor: |_| None,
    },
    MetricDef {
        // Run-level POOLED iteration count — the NUMERATOR component of the
        // pooled `iterations_per_cpu_sec` Rate, summed over cgroups with
        // MEASURED cpu-time (total_cpu_time_ns > 0). ext_metrics-only,
        // DISTINCT from the typed `total_iterations` Counter on purpose: the
        // typed field is skipped from ext_metrics (TYPED_FIELD_NAMES) and folds
        // cross-RUN as a MEAN (group_and_average_by's round_u64 divides the
        // accumulated sum by the contributor count — a display average), while
        // a Rate numerator must fold cross-RUN as a SUM (aggregate_finite
        // Counter arm, no divide) so Σnum/Σdenom re-pools. One shared key
        // cannot carry both folds, so the numerator gets its own ext key. It
        // also sums only MEASURED cgroups, where the typed field's per-RUN
        // cross-cgroup merge sums ALL cgroups — so it equals the merge-summed
        // typed total_iterations unless an excluded (zero-cpu-time) cgroup
        // carried iterations>0, in which case it is LESS.
        // `total_` prefix satisfies the Counter naming gate.
        name: "total_iterations_pooled",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Run-level cohort CPU-time EFFICIENCY pooled across cgroups (and
        // re-pooled across runs): Σiterations / Σcpu-seconds. MetricKind::Rate
        // over the two Counter components above; derive_rate_metrics re-derives
        // it = Σtotal_iterations_pooled / Σtotal_cpu_time_sec at every level.
        // Distinct from the per-cgroup `worst_iterations_per_cpu_sec`
        // WorstLowest metric (the lowest-wins min-fold starvation selector):
        // this is the POOLED cohort rate, overcommit-invariant. _per_cpu_sec name + Rate kind passes the
        // reverse naming gate; ext_metrics-only (accessor |_| None).
        //
        // SAME physical quantity as worst_iterations_per_cpu_sec (iter/CPU-s
        // efficiency), so it shares that sibling's compare thresholds:
        // default_rel=0.10 (a 10% efficiency change is the regression signal)
        // and default_abs=10.0 (near-zero anti-jitter floor — a real busy
        // workload's rate is orders of magnitude larger). NOT the looser
        // iteration_rate throughput gate (rel=0.30), which would silently
        // swallow a 10-29% efficiency regression the per-cgroup row flags.
        name: "iterations_per_cpu_sec",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Rate {
            numerator: "total_iterations_pooled",
            denominator: "total_cpu_time_sec",
        },
        default_abs: 10.0,
        default_rel: 0.10,
        display_unit: "iter/cpu-s",
        accessor: |_| None,
    },
    MetricDef {
        // Per-phase SYSTEM (in-kernel) CPU time in nanoseconds. Read
        // host-side from frozen task_struct.stime + the thread-group
        // signal_struct.stime accumulator (zero guest work). Injected
        // post-hoc — NOT a read_sample metric — as a per-thread-GROUP
        // delta over the phase: `crate::assert::phase_group_cpu_delta`
        // sums each tgid's `thread_group_cputime` (signal + live-thread
        // stime) at its first and last appearance among the phase's
        // freeze samples and takes `last - first` = system CPU time the
        // group spent during the phase. PerPhaseDeltaSum: the per-phase value
        // is already a delta, so the disjoint per-phase deltas SUM across the
        // run (the run's total OBSERVED system CPU time — a lower bound
        // excluding head / tail / inter-phase-gap windows; see the kind doc),
        // and the per-run totals fold by UNWEIGHTED MEAN cross-RUN (NOT
        // sample-count-weighted), like user_time_ns. LowerBetter — the DSQ-spinlock
        // regression surfaces as rising system time (CPUs spinning in
        // the kernel). No typed GauntletRow field; the ext_metrics
        // fallback carries it through cargo ktstr stats compare.
        name: "system_time_ns",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhaseDeltaSum,
        default_abs: 1_000_000.0,
        default_rel: 0.30,
        display_unit: "ns",
        accessor: |_| None,
    },
    MetricDef {
        // Per-phase USER-mode CPU time in nanoseconds. Same host-side /
        // injected / PerPhaseDeltaSum shape as `system_time_ns` (task_struct
        // .utime + the thread-group signal_struct.utime accumulator,
        // per-tgid delta via `crate::assert::phase_group_cpu_delta`; SUM
        // cross-phase, unweighted MEAN cross-run).
        // Pairs with it so a test can distinguish "system time rose,
        // user work flat" (the lock-contention signature) from "both
        // rose" (genuine extra work). LowerBetter — less CPU consumed
        // for the same work is the efficiency win; utime already
        // includes gtime so the two are never summed.
        name: "user_time_ns",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhaseDeltaSum,
        default_abs: 1_000_000.0,
        default_rel: 0.30,
        display_unit: "ns",
        accessor: |_| None,
    },
    // ---- IRQ observability ----
    // Host-side observer-free IRQ signals from PerCpuTimeStats (freeze
    // Snapshot, src/monitor/dump/mod.rs), cross-CPU folded at
    // read_sample and carried through ext_metrics (accessor |_| None) like
    // system_time_ns. The time signals require CONFIG_IRQ_TIME_ACCOUNTING;
    // loud-absent (None), never false-zero, when off. Per-phase
    // reduction is the Counter last-minus-first over the bucket's freeze
    // captures (needs num_snapshots >= 2). The per-CPU SPATIAL axis
    // (max_cpu_hardirqs + max_cpu_hardirq_concentration, the busiest-CPU
    // dimension vs this cross-CPU SUM) is registered below; per-softirq
    // spatial-max is a follow-up.
    MetricDef {
        // Sum of kernel_stat.irqs_sum across CPUs — total hardirqs fired
        // (per-CPU monotonic count, __kstat_incr_irqs_this_cpu,
        // kernel/irq/internals.h). NOT gated on irqtime (always populates).
        name: "total_hardirqs",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 1000.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Sum of kernel_stat.softirqs[NET_RX] across CPUs (index via
        // SOFTIRQ_NAMES; kstat_incr_softirqs_this_cpu, kernel/softirq.c). The
        // load-bearing softirq for NetTraffic RX.
        name: "total_softirq_net_rx",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 500.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Sum of kernel_stat.softirqs[NET_TX] across CPUs.
        name: "total_softirq_net_tx",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 500.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Sum of kernel_stat.softirqs[TIMER] across CPUs.
        name: "total_softirq_timer",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 1000.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Sum of kernel_stat.softirqs[SCHED] across CPUs.
        name: "total_softirq_sched",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 1000.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Sum of kernel_cpustat.cpustat[CPUTIME_IRQ] across CPUs — raw ns in
        // hardirq (irqtime_account_delta, kernel/sched/cputime.c). Read from
        // guest memory as ns (NOT /proc/stat jiffies — no nsec_to_clock_t).
        // Requires CONFIG_IRQ_TIME_ACCOUNTING; Counter/ns like system_time_ns.
        name: "total_irq_time_ns",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 1_000_000.0,
        default_rel: 0.50,
        display_unit: "ns",
        accessor: |_| None,
    },
    MetricDef {
        // Sum of kernel_cpustat.cpustat[CPUTIME_SOFTIRQ] across CPUs — raw ns
        // in softirq. Requires CONFIG_IRQ_TIME_ACCOUNTING.
        name: "total_softirq_time_ns",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 1_000_000.0,
        default_rel: 0.50,
        display_unit: "ns",
        accessor: |_| None,
    },
    MetricDef {
        // Sum of kernel_cpustat.cpustat[CPUTIME_STEAL] across CPUs — raw ns the
        // hypervisor stole (account_steal_time; needs CONFIG_PARAVIRT_TIME_
        // ACCOUNTING + kvm-clock steal-time). CPUTIME_STEAL is an unconditional
        // enum member (enum cpu_usage_stat, include/linux/kernel_stat.h), so
        // steal-accounting-off reads a constant 0 — a measured Some(0.0), NOT
        // loud-absent like the BTF-gated avg_irq gauge.
        name: "total_steal_time_ns",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 1_000_000.0,
        default_rel: 0.50,
        display_unit: "ns",
        accessor: |_| None,
    },
    MetricDef {
        // Mean across CPUs of rq->avg_irq.util_avg — the PELT IRQ load average
        // (struct sched_avg, kernel/sched/sched.h; range [0, 1024] =
        // SCHED_CAPACITY_SCALE). INSTANTANEOUS gauge (decaying PELT), NEVER
        // deltaed. Requires CONFIG_HAVE_SCHED_AVG_IRQ (def_bool y when
        // (IRQ_TIME_ACCOUNTING || PARAVIRT_TIME_ACCOUNTING) && SMP — init/Kconfig).
        // Distinct from taskstats avg_irq_delay_ns (irq-DELAY accounting); this
        // is PELT util.
        name: "avg_irq_util",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Avg),
        default_abs: 20.0,
        default_rel: 0.30,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Host spatial-max ACROSS CPUs of the INSTANTANEOUS rq->avg_irq.util_avg
        // gauge (worst-CPU IRQ load at the freeze) — NOT a kernel max-of-window.
        // Peak because both the spatial and temporal reduces are max over
        // instantaneous values (no cumulative-delta hazard, unlike a counter's
        // spatial-max; per-CPU axis is a follow-up). Range [0, 1024].
        name: "max_avg_irq_util",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Peak,
        default_abs: 50.0,
        default_rel: 0.30,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // DERIVED rate: total_hardirqs / total_phase_wall_sec — hardirqs per
        // second over the CAPTURE WINDOW (first->last freeze span, NOT the full
        // phase; see total_phase_wall_sec). For A/B compare the cadence cancels.
        name: "hardirq_rate",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Rate {
            numerator: "total_hardirqs",
            denominator: "total_phase_wall_sec",
        },
        default_abs: 10.0,
        default_rel: 0.30,
        display_unit: "irq/s",
        accessor: |_| None,
    },
    MetricDef {
        // DERIVED rate: total_softirq_net_rx / total_phase_wall_sec — NET_RX
        // softirqs per second over the capture window. The NetTraffic
        // softirq-pressure signal.
        name: "net_rx_softirq_rate",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Rate {
            numerator: "total_softirq_net_rx",
            denominator: "total_phase_wall_sec",
        },
        default_abs: 10.0,
        default_rel: 0.30,
        display_unit: "softirq/s",
        accessor: |_| None,
    },
    MetricDef {
        // DERIVED rate: total_irq_time_ns / total_phase_wall_ns — the
        // dimensionless [0,1] fraction of the capture window spent in hardirq.
        // ns/ns (both over the SAME first->last freeze span) so the span-vs-
        // phase gap cancels. The exact-integral companion to avg_irq_util's
        // smoothed PELT gauge.
        name: "irq_time_fraction",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Rate {
            numerator: "total_irq_time_ns",
            denominator: "total_phase_wall_ns",
        },
        default_abs: 0.02,
        default_rel: 0.30,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Hidden rate-denominator component (NOT user-facing): the CAPTURE-
        // WINDOW duration in seconds = (bucket end_ms - start_ms)/1000, co-
        // inserted in buckets_from_grouped both-or-neither with the IRQ
        // counters (the /1000 lives at the insertion site; derive_rate_metrics
        // does bare num/den). Backs hardirq_rate / net_rx_softirq_rate. Counter
        // so it survives the cross-RUN Sum-fold (Sum count / Sum sec re-derives).
        name: "total_phase_wall_sec",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 0.1,
        default_rel: 0.30,
        display_unit: "s",
        accessor: |_| None,
    },
    MetricDef {
        // Hidden rate-denominator component (NOT user-facing): the capture-
        // window duration in NANOSECONDS = (bucket end_ms - start_ms) * 1e6,
        // co-inserted with the IRQ counters. Backs irq_time_fraction (ns/ns).
        name: "total_phase_wall_ns",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 1_000_000.0,
        default_rel: 0.30,
        display_unit: "ns",
        accessor: |_| None,
    },
    MetricDef {
        // Per-CPU IRQ spatial axis: the BUSIEST CPU's hardirq delta over the
        // phase — max over CPUs of each CPU's (last - first freeze) irqs_sum,
        // correlated by the per_cpu_time cpu field (NOT the cross-CPU sum, which
        // is total_hardirqs). Custom per-CPU-delta fold in assert::phase_build,
        // NOT a read_sample arm (read_sample yields one f64 per freeze, no
        // per-CPU vector). Peak = spatial-max of a per-CPU cumulative-counter
        // delta. Informational: a high busiest-CPU count is ambiguous (high
        // traffic vs concentration) — the concentration ratio below is the
        // balance signal, mirroring the raw-counts-Informational split.
        name: "max_cpu_hardirqs",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Peak,
        default_abs: 500.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // IRQ-concentration ratio: max_cpu_hardirqs / mean per-CPU hardirq delta
        // over the SAME reporting-CPU set — the busiest CPU's share of the
        // average. Range [1, num_cpus]: 1.0 = perfectly even, higher = IRQs
        // concentrated on one CPU. Peak (worst per-phase concentration),
        // LowerBetter. Computed in the same per-CPU-delta fold, NOT a Rate (a
        // Peak numerator fails every_rate_metric_has_registered_counter_components,
        // and max/mean is not Σ-poolable). DELIBERATELY max/MEAN, distinct from
        // the sibling max_imbalance_ratio's max/MIN: max/min explodes when any
        // CPU takes ~0 IRQs, whereas max/mean measures disproportionate SHARE
        // (the IRQ-steering question). Absent (None) when < 2 reporting CPUs or
        // mean == 0.
        name: "max_cpu_hardirq_concentration",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Peak,
        default_abs: 1.0,
        default_rel: 0.25,
        display_unit: "x",
        accessor: |_| None,
    },
    MetricDef {
        // Per-CPU NET_RX softirq spatial axis: the BUSIEST CPU's NET_RX softirq
        // delta over the phase — max over CPUs of each CPU's (last - first freeze)
        // kstat.softirqs[NET_RX] delta, correlated by the per_cpu_time cpu field
        // (NOT the cross-CPU sum, which is total_softirq_net_rx). Counts softirq
        // RUNS/invocations (handle_softirqs increments once per pending NET_RX bit
        // per dispatch via kstat_incr_softirqs_this_cpu), NOT packets — a
        // softirq-frequency / affinity-concentration signal. Custom per-CPU-delta
        // fold in assert::phase_build (fold_per_cpu_spatial_max), NOT a read_sample
        // arm. Peak = spatial-max of a per-CPU cumulative-counter delta. The
        // softirq sibling of max_cpu_hardirqs; Informational for the same reason —
        // a high busiest-CPU count is ambiguous (high RX traffic vs concentration),
        // the concentration ratio below is the balance signal.
        name: "max_cpu_softirq_net_rx",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Peak,
        default_abs: 500.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // NET_RX-softirq-concentration ratio: max_cpu_softirq_net_rx / mean
        // per-CPU NET_RX softirq delta over the SAME reporting-CPU set — the
        // busiest CPU's share of the average. Range [1, num_cpus]: 1.0 = even,
        // higher = NET_RX softirqs concentrated on one CPU (the single-queue-NIC
        // vs RPS/RSS-spread signal). Peak, LowerBetter, max/mean — the softirq
        // sibling of max_cpu_hardirq_concentration (same NOT-a-Rate, max/MEAN-not-
        // max/MIN, >=2-reporting-CPU + mean>0 discipline). Absent (None) when
        // < 2 reporting CPUs or mean == 0.
        name: "max_cpu_softirq_net_rx_concentration",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Peak,
        default_abs: 1.0,
        default_rel: 0.25,
        display_unit: "x",
        accessor: |_| None,
    },
    MetricDef {
        // Per-cgroup IRQ-pressure spatial axis: the busiest workload-leaf cgroup's
        // PSI-irq `full` stall DELTA over the phase (decoded µs) — max over the
        // workload-root leaf cgroups of each leaf's (last - first freeze)
        // cgroup->psi total[PSI_AVGS][PSI_IRQ_FULL], correlated across freezes by
        // (cgroup_kva, serial_nr) — the serial disambiguates a freed slab KVA
        // reused by a new cgroup. The per-cgroup analog of max_cpu_hardirqs (the
        // busiest-CELL dimension);
        // attributes IRQ-servicing stall to the workload cell that induced it,
        // which the system-wide total_irq_pressure_us cannot. Custom per-cgroup
        // delta fold (assert::phase_build fold_per_cgroup_psi), NOT a read_sample
        // arm. Peak = spatial-max of a per-cgroup cumulative-counter delta.
        // Informational: an absolute per-cell stall is workload-confounded (more
        // work → more stall) — the concentration ratio below is the isolation
        // signal, mirroring the max_cpu_hardirqs raw-counts split.
        name: "max_cgroup_irq_pressure",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Peak,
        default_abs: 1000.0,
        default_rel: 0.50,
        display_unit: "µs",
        accessor: |_| None,
    },
    MetricDef {
        // Per-cgroup IRQ-pressure concentration: max_cgroup_irq_pressure / the
        // mean per-leaf IRQ-full stall delta over the SAME reporting-leaf set —
        // the busiest cell's share of the average. Range [1, num_leaves]: 1.0 =
        // evenly spread, higher = IRQ-servicing stall concentrated on one workload
        // cell (the cgroup-isolation / cell-steering signal). Peak, LowerBetter,
        // max/MEAN — the per-cgroup sibling of max_cpu_hardirq_concentration (same
        // NOT-a-Rate, max/MEAN-not-max/MIN, >=2-reporting-leaf + mean>0 discipline;
        // disjoint leaves — cgroup2's no-internal-process rule — so no
        // double-count). Absent (None) when < 2 reporting leaves or mean == 0.
        name: "max_cgroup_irq_pressure_concentration",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Peak,
        default_abs: 1.0,
        default_rel: 0.25,
        display_unit: "x",
        accessor: |_| None,
    },
    MetricDef {
        // Per-cgroup IRQ-pressure GAUGE: the worst workload-leaf cgroup's PSI-irq
        // `full` avg10 (decoded 10s-EWMA percent, 0..=100) — per freeze the max
        // across the leaves, then the max across the phase's freezes. The
        // instantaneous-pressure companion to max_cgroup_irq_pressure (a gauge, so
        // a spatial-max with no delta — the max_avg_irq_util shape on the cgroup
        // axis). Peak; LowerBetter (less IRQ pressure on the worst cell is better).
        // Custom fold (fold_per_cgroup_psi), NOT a read_sample arm. Loud-absent
        // when no leaf reported PSI (psi_cgroups off / absent workload root).
        name: "max_cgroup_psi_irq_avg10",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Peak,
        default_abs: 5.0,
        default_rel: 0.30,
        display_unit: "%",
        accessor: |_| None,
    },
    MetricDef {
        // System-wide PSI-irq `full` avg10: the mean over monitor samples of the
        // decoded 10s-EWMA full IRQ pressure (percent, 0..=100), host-walked from
        // the global `psi_system` (NOT a guest /proc read). Gauge(Avg) like
        // `avg_irq_util` — an instantaneous smoothed gauge, never deltaed; the
        // cross-run fold sample-weights it. LowerBetter (less IRQ pressure is
        // better). ext-only (accessor |_| None), folded from MonitorSummary in
        // group::sidecar_to_row. Loud-absent (None) when CONFIG_PSI /
        // CONFIG_IRQ_TIME_ACCOUNTING is off (no PSI_IRQ_FULL in BTF), never 0.0.
        name: "psi_irq_full_avg10",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Avg),
        default_abs: 5.0,
        default_rel: 0.30,
        display_unit: "%",
        accessor: |_| None,
    },
    MetricDef {
        // Cumulative system-wide PSI-irq `full` stall over the monitoring window
        // (µs): the end-start delta of `total[PSI_AVGS][PSI_IRQ_FULL]` (decoded
        // ns→µs), host-walked from `psi_system`. Counter (a monotonic cumulative
        // total, end-start deltaed; saturating on reset) like `total_irq_time_ns`,
        // so the cross-run fold Σ-pools it. Informational: an absolute stall time
        // is workload-confounded (longer run → more stall) — the avg10 gauge is
        // the magnitude-normalized signal; mirrors the total_irq_time_ns split.
        // ext-only, same loud-absent gate as `psi_irq_full_avg10`.
        name: "total_irq_pressure_us",
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 1000.0,
        default_rel: 0.50,
        display_unit: "µs",
        accessor: |_| None,
    },
    MetricDef {
        // Mean schedstat run-delay, re-pooled as the mean over the COMBINED
        // run-delay sample set across every cgroup (and phase), RAW ns→µs
        // once — see `worst_p99_wake_latency_us`. Each sample is one per-WORKER
        // cumulative sched_info.run_delay total (NOT per-dispatch), so the pool
        // size is the worker count — see
        // [`crate::assert::PhaseCgroupStats::run_delays_ns`]. Distribution kind;
        // accessor |_| None (ext_metrics-sourced from the re-pool).
        name: "worst_mean_run_delay_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Distribution {
            source: SampleSource::RunDelayNs,
            reduction: SampleReduction::Mean,
        },
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        // Worst (max) schedstat run-delay over the combined run-delay sample
        // set, RAW ns→µs once. Distribution kind with the Worst reduction:
        // the one Distribution reduction whose cross-RUN fold is MAX (the
        // peak survives), not MEAN — see [`crate::stats::SampleReduction::Worst`].
        name: "worst_run_delay_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Distribution {
            source: SampleSource::RunDelayNs,
            reduction: SampleReduction::Worst,
        },
        default_abs: 100.0,
        default_rel: 0.50,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        // Ratio of p99 / median wake latency, worst-case across
        // cgroups. `LowerBetter` because a higher ratio signals a
        // stretched long tail. Unitless; baseline is 1.0 (p99 == median
        // is the perfect-uniform floor set by order-statistic
        // ordering). `default_abs = 0.5` guards against trivially
        // small deltas that percent-only gates would flag; `default_rel
        // = 0.25` matches the wake-latency metrics' percent gate.
        //
        // BASIS: the per-cgroup worst — the MAX over each cgroup's own
        // p99/median ratio (`CgroupStats::wake_latency_tail_ratio`), selected
        // post-merge over `stats.cgroups`. Deliberately NOT
        // `pooled_p99 / pooled_median` of the `worst_p99_wake_latency_us` /
        // `worst_median_wake_latency_us` Distributions (those re-pool the
        // cross-cgroup union), so the two do not satisfy
        // `tail_ratio == pooled_p99/pooled_median`.
        //
        // CROSS-RUN FOLD = unweighted exclude-missing MEAN (NOT MAX), by
        // design. `MetricKind::WakeLatencyTailRatio` is a WITHIN-RUN
        // worst-across-cgroups selector; cross-RUN `aggregate_finite`
        // MEAN-folds the per-run worst values over ONLY the runs that cleared
        // the floor (divisor = present-finite-contributor count), so a cohort
        // of repeated runs reports its TYPICAL worst-cgroup tail amplification
        // — the operator-facing cohort-comparison default shared with every
        // WorstLowest selector. It deliberately does NOT fold by MAX: MAX
        // (peak-of-peaks) is reserved for `SampleReduction::Worst`
        // (worst_run_delay_us), a peak DETECTOR; this answers "what is this
        // cohort's characteristic worst-cgroup tail". Aligning worst-across
        // selectors to a cross-RUN extremum is a tracked product decision (see
        // the EXTREMUM ASYMMETRY note in `aggregate_finite`), not this fix.
        //
        // Samples-required noise gate, enforced at the PRODUCER (not an
        // accessor): `crate::assert::populate_run_distribution_metrics` emits
        // NO ext key when the run completed fewer than
        // [`WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`] iterations (with few
        // samples the p99 estimate is effectively the observed maximum and the
        // ratio is dominated by a single outlier, not a distributional signal),
        // and none when no cgroup carried a measurable tail. An absent key is
        // EXCLUDED from the cross-RUN mean (no sub-threshold run dilutes the
        // cohort) and read as `None` by `compare_rows`, which `unwrap_or(0.0)`s
        // both sides into the `abs() < EPSILON` skip. This REPLACES the deleted
        // typed field's accessor gate, which (a) summed every passing run's raw
        // ratio over `passes_observed` cross-RUN — folding noisy low-N runs in
        // as real values — and (b) re-gated the AGGREGATED row against a MEANED
        // iteration count. See [`WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`] for
        // the threshold-value rationale.
        //
        // accessor |_| None: ext_metrics-sourced from the post-merge producer.
        name: "worst_wake_latency_tail_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::WakeLatencyTailRatio,
        default_abs: 0.5,
        default_rel: 0.25,
        display_unit: "x",
        accessor: |_| None,
    },
    MetricDef {
        // Per-worker iteration throughput, worst (lowest) cgroup.
        // `HigherBetter` mirrors [`total_iterations`]: a cgroup that
        // fell behind regresses this downward, and a cross-variant
        // improvement raises it. `default_abs = 10.0` is the absolute
        // iteration-count floor below which deltas are noise;
        // `default_rel = 0.10` mirrors the `total_iterations` gate.
        //
        // Derivation of `abs = 10`: this metric is PER-WORKER. In-tree
        // fixtures span `workers_per_cgroup` from 1 through 8 (see
        // the KtstrTestEntry declarations under src/scenario/*.rs and
        // tests/*.rs); `KtstrTestEntry::DEFAULT.workers_per_cgroup`
        // is 2, with scenario-level overrides commonly picking 4 or
        // 8. A per-worker floor of 10 therefore corresponds to
        // aggregate regressions of 10-80 total iterations across the
        // supported worker counts — high enough that a lightly-
        // loaded scheduler's jitter does not flag a regression, low
        // enough that a genuine drop (e.g. a cgroup that fell behind
        // by 10 iterations at workers=1, or 80 at workers=8) still
        // trips the gate. Going below 10 would flag normal cross-run
        // jitter on single-worker configs; going above 10 would mask
        // regressions on low-worker-count tests. The `rel=0.10`
        // companion gate handles larger throughputs proportionally,
        // so the `abs=10` floor only binds in the small-count regime
        // where rel-only would let single-digit losses slip through.
        //
        // WorstLowest kind: the lowest (worst) cgroup's
        // total_iterations / num_workers, re-pooled post-merge by
        // `crate::assert::populate_run_distribution_metrics` from the
        // per-cgroup counters via the None-aware lowest-wins fold (a
        // measured Some(0.0) wins; a no-workers None is skipped). Accessor
        // |_| None — ext_metrics-sourced; an all-None cohort writes no key.
        name: "worst_iterations_per_worker",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::WorstLowest {
            numerator: WorstLowestNumerator::Iterations,
            denominator: WorstLowestDenominator::NumWorkers,
        },
        default_abs: 10.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Overcommit-INVARIANT per-cgroup efficiency (iterations per
        // CPU-second). `HigherBetter`: a cgroup that lost efficiency
        // regresses this downward. Unlike worst_iterations_per_worker
        // (raw work, scales with the host-CPU budget), this is the metric
        // to compare across `cpu_budget` settings — the overcommit marker
        // and compare-path warning point operators here.
        //
        // `default_rel = 0.10` is the binding proportional gate (a 10%
        // efficiency change is the regression signal), mirroring the
        // per-worker sibling. `default_abs = 10.0` (iterations/CPU-second)
        // is a near-zero noise floor: for any real busy workload the rate
        // is orders of magnitude larger, so the floor only binds for a
        // near-idle cgroup, where it stops a large rel% on a tiny rate
        // from flagging jitter. Distinct from the per-worker metric's
        // floor (which scales with worker count) — this is a per-second
        // rate, so the floor is a flat anti-noise guard, not a per-worker
        // derivation.
        //
        // WorstLowest kind: the lowest (worst) cgroup's
        // total_iterations / (total_cpu_time_ns / 1e9), re-pooled post-merge
        // by `crate::assert::populate_run_distribution_metrics` (None when a
        // cgroup has no workers or no on-CPU time; lowest measured wins).
        // Accessor |_| None — ext_metrics-sourced.
        name: "worst_iterations_per_cpu_sec",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::WorstLowest {
            numerator: WorstLowestNumerator::Iterations,
            denominator: WorstLowestDenominator::CpuTimeNs,
        },
        default_abs: 10.0,
        default_rel: 0.10,
        // Same physical quantity as the pooled iterations_per_cpu_sec Rate;
        // share its unit string rather than leaving this one under-specified.
        display_unit: "iter/cpu-s",
        accessor: |_| None,
    },
    MetricDef {
        // The WORST (lowest) per-cgroup page-locality fraction across the run.
        // HigherBetter, so lowest-wins = worst — a WorstLowest selector
        // (None-aware: a measured 0.0, all pages off-node, WINS the lowest; a
        // cgroup that measured no NUMA pages is skipped, not a 0.0 sentinel).
        // Re-pooled post-merge from the per-phase NUMA carriers
        // (assert::populate_run_distribution_metrics, numa_agg_per_cgroup) — NOT
        // a typed field: the reports-only CgroupStats hardcodes page_locality 0.0
        // (no expected-node set), and the prior typed Gauge field folded via
        // fold_lowest_nonzero, which SKIPPED a measured 0.0 and reported a
        // better-than-worst cross-run value. accessor None: ext-sourced.
        name: "worst_page_locality",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::WorstLowest {
            numerator: WorstLowestNumerator::NumaLocal,
            denominator: WorstLowestDenominator::NumaTotal,
        },
        default_abs: 0.05,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // The WORST (highest) per-cgroup cross-node migration-churn ratio across
        // the run. LowerBetter, so highest-wins = worst — a WorstCrossNodeRatio
        // max-selector re-pooled post-merge from the per-phase NUMA carriers
        // (assert::populate_run_distribution_metrics, numa_agg_per_cgroup) — NOT a
        // typed field: the prior typed Gauge(Last) field/GauntletRow column was
        // merge-max-folded within-run but cross-run averaged each run's value over
        // passes_observed (folding a NUMA-less run's 0.0 sentinel in), AND diverged
        // from run_metric (which already re-derived from the per-phase carriers), so
        // the sidecar and the in-test read gave different values on multi-phase
        // runs. accessor None: ext-sourced.
        name: "worst_cross_node_migration_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::WorstCrossNodeRatio,
        default_abs: 0.05,
        default_rel: 0.20,
        display_unit: "",
        accessor: |_| None,
    },
    // -- schbench per-phase metrics (MetricKind::PerPhase) --
    // Derived ONCE per phase by `crate::assert::derive_phase_metrics`
    // from the phase's pooled schbench histograms / run-delay raw pairs, written
    // directly into `PhaseBucket::metrics`. is_derived (skipped by the within-run
    // reducers + the phase-bucket merge) with no run-level producer; a per-phase
    // A/B claim reads them via `phase_metric`. `accessor: |_| None` — they never
    // live on a `GauntletRow`. Latency p50/p90 mirror worst_median_wake_latency_us
    // (abs 20), p99/p999 + sched-delay mirror worst_p99/mean (abs 50); all rel 0.25.
    MetricDef {
        name: SCHBENCH_WAKEUP_P50_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_WAKEUP_P90_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_WAKEUP_P99_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_WAKEUP_P999_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_REQUEST_P50_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_REQUEST_P90_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_REQUEST_P99_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_REQUEST_P999_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_SCHED_DELAY_MSG_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_SCHED_DELAY_WORKER_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        // Completed work cycles in the phase — more is better (throughput).
        name: SCHBENCH_LOOP_COUNT,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerPhase,
        default_abs: 10.0,
        default_rel: 0.30,
        display_unit: "",
        accessor: |_| None,
    },
    // taobench per-phase qps + hit ratios (WorkType::Taobench engine, derived by
    // write_taobench_scalars). total/fast qps HigherBetter (throughput); slow_qps
    // + hit_ratio + hit_rate Informational (a component / run-validity signals,
    // never a regression direction — see classify_direction).
    MetricDef {
        name: TAOBENCH_TOTAL_QPS,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerPhase,
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "ops/s",
        accessor: |_| None,
    },
    MetricDef {
        name: TAOBENCH_FAST_QPS,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerPhase,
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "ops/s",
        accessor: |_| None,
    },
    MetricDef {
        name: TAOBENCH_SLOW_QPS,
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::PerPhase,
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "ops/s",
        accessor: |_| None,
    },
    MetricDef {
        name: TAOBENCH_HIT_RATIO,
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::PerPhase,
        default_abs: 0.02,
        default_rel: 0.05,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        name: TAOBENCH_HIT_RATE,
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::PerPhase,
        default_abs: 0.02,
        default_rel: 0.05,
        display_unit: "",
        accessor: |_| None,
    },
    // taobench WHOLE-RUN qps + hit Rates and their Counter components, pooled
    // cross-cgroup by `crate::assert::populate_run_pooled_taobench` and derived
    // by `derive_rate_metrics`. The four `total_taobench_*` Counters are the rate
    // components (ext_metrics-only, accessor |_| None; `total_` prefix satisfies
    // the Counter naming gate) and are `RENDER_SUPPRESSED_COMPONENTS`, so their
    // default_abs/default_rel are inert at the compare layer — the entries exist
    // for the re-pool (`name` is the component key, `kind` drives the Counter
    // SUM-fold). Cross-RUN each component SUMs, so the Rates re-pool as
    // Σnumerator / Σdenominator (aggregate throughput, not a mean of per-run qps).
    // HIT exposed whole-run is the RESPONSE-time taobench_hit_fraction
    // (Σfast/Σcompleted); the COMMAND-time hit (get-based, 1-Σmisses/Σcmds, the
    // whole-run analog of the per-phase taobench_hit_rate) is intentionally NOT
    // a run-level key: in the current closed-loop engine every issued lookup
    // completes, so command-time and response-time hit converge over a whole run
    // (run.rs module docs) and a second run-level hit would carry no distinct
    // --noise-adjust spread. They diverge only under open-loop arrival (a future
    // taobench mode), which is when the command-time whole-run hit becomes
    // meaningful and should be added (its get_cmds/get_misses components are
    // already pooled in the carrier).
    MetricDef {
        name: TOTAL_TAOBENCH_OPS,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Counter,
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        name: TOTAL_TAOBENCH_FAST_OPS,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Counter,
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        name: TOTAL_TAOBENCH_SLOW_OPS,
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Whole-run wall window (ns→s applied once at the producer), the qps
        // DENOMINATOR. Counter — cross-RUN SUM, mirroring `total_cpu_time_sec`,
        // so Σops/Σwall re-pools the cohort throughput. `total_` prefix satisfies
        // the Counter naming gate.
        name: TOTAL_TAOBENCH_WALL_SEC,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Counter,
        default_abs: 1.0,
        default_rel: 0.30,
        display_unit: "s",
        accessor: |_| None,
    },
    MetricDef {
        // Whole-run total throughput = Σcompleted ops / Σwall-seconds.
        // HigherBetter (throughput). Shares the per-phase `taobench_total_qps`
        // thresholds. Absent when no Taobench cgroup ran or the wall window was
        // unmeasured (components absent).
        name: TAOBENCH_TOTAL_OPS_PER_SEC,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Rate {
            numerator: TOTAL_TAOBENCH_OPS,
            denominator: TOTAL_TAOBENCH_WALL_SEC,
        },
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "ops/s",
        accessor: |_| None,
    },
    MetricDef {
        // Whole-run hit (fast-path) throughput = Σfast ops / Σwall-seconds.
        // HigherBetter.
        name: TAOBENCH_FAST_OPS_PER_SEC,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Rate {
            numerator: TOTAL_TAOBENCH_FAST_OPS,
            denominator: TOTAL_TAOBENCH_WALL_SEC,
        },
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "ops/s",
        accessor: |_| None,
    },
    MetricDef {
        // Whole-run slow-path throughput = Σslow ops / Σwall-seconds.
        // Informational — the slow path is a component of total throughput, not a
        // standalone regression direction (mirrors the per-phase
        // `taobench_slow_qps`).
        name: TAOBENCH_SLOW_OPS_PER_SEC,
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Rate {
            numerator: TOTAL_TAOBENCH_SLOW_OPS,
            denominator: TOTAL_TAOBENCH_WALL_SEC,
        },
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "ops/s",
        accessor: |_| None,
    },
    MetricDef {
        // Whole-run cache hit FRACTION = Σfast ops / Σcompleted ops — the SAME
        // response-time hit measurement as the per-phase `taobench_hit_ratio`, at
        // whole-run scope (its run-level Σ/Σ pool). The name differs only by axis:
        // per-phase `_ratio` vs whole-run `_fraction` — a distinct registry key is
        // required because a Rate cannot share a name with the per-phase PerPhase
        // entry, and `_fraction` reads as the pooled [0, 1] ratio-of-counters (the
        // qps siblings diverge the same way: per-phase `_qps` vs whole-run
        // `_ops_per_sec`). DISTINCT from the command-time `taobench_hit_rate`
        // (1 - misses/cmds), which is request-time, not response-time — see the
        // block comment above. A fraction in [0, 1]. Informational — a hit-rate
        // change is a workload-shape signal, not a scheduler regression direction.
        // Absent when no ops completed (`total_taobench_ops` is 0 →
        // `derive_rate_metrics` skips the zero denominator).
        name: TAOBENCH_HIT_FRACTION,
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Rate {
            numerator: TOTAL_TAOBENCH_FAST_OPS,
            denominator: TOTAL_TAOBENCH_OPS,
        },
        default_abs: 0.02,
        default_rel: 0.05,
        display_unit: "",
        accessor: |_| None,
    },
    // schbench WHOLE-RUN Class-3: role-separate run-delay gate Rates + their
    // Counter components + the whole-run loop Counter, re-pooled run-level by
    // `crate::assert::populate_run_pooled_schbench` from the per-phase per-cgroup
    // SchbenchPhaseStats raw pairs (Σ over phases+cgroups). The four run-delay /
    // pcount Counters are ext-only rate components (accessor |_| None;
    // RENDER_SUPPRESSED; `total_` prefix → Counter gate). The two Rates are the
    // sample-weighted Σrun_delay/Σpcount per-schedule means (the workload-scoped
    // siblings of the system-wide `total_run_delay_ns_per_sched`); message and
    // worker roles pool separately. Distinct from the per-phase
    // `sched_delay_msg/worker_us` (mean-of-means parity, PerPhase display-only) —
    // the Rates gate, no double-count.
    MetricDef {
        name: TOTAL_SCHBENCH_MSG_RUN_DELAY_NS,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Counter,
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "ns",
        accessor: |_| None,
    },
    MetricDef {
        name: TOTAL_SCHBENCH_MSG_PCOUNT,
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        name: TOTAL_SCHBENCH_WORKER_RUN_DELAY_NS,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Counter,
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "ns",
        accessor: |_| None,
    },
    MetricDef {
        name: TOTAL_SCHBENCH_WORKER_PCOUNT,
        polarity: crate::test_support::Polarity::Informational,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Whole-run completed work cycles (Σ over phases+cgroups). HigherBetter
        // (throughput). NOT a rate component, so NOT suppressed. Distinct from the
        // per-phase `schbench_loop_count` (PerPhase timeline).
        name: TOTAL_SCHBENCH_LOOPS,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Message-thread per-schedule run-delay mean = Σrun_delay_ns / Σpcount
        // (sample-weighted, NOT mean-of-per-run-means). LowerBetter (higher
        // scheduling wait is worse). Absent when no message thread was scheduled
        // (Σpcount == 0).
        name: SCHBENCH_MSG_RUN_DELAY_NS_PER_SCHED,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Rate {
            numerator: TOTAL_SCHBENCH_MSG_RUN_DELAY_NS,
            denominator: TOTAL_SCHBENCH_MSG_PCOUNT,
        },
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "ns",
        accessor: |_| None,
    },
    MetricDef {
        // Worker per-schedule run-delay mean = Σrun_delay_ns / Σpcount. LowerBetter.
        // Pooled SEPARATELY from the message role (different per-schedule wait
        // populations). Absent when no worker was scheduled (Σpcount == 0).
        name: SCHBENCH_WORKER_RUN_DELAY_NS_PER_SCHED,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Rate {
            numerator: TOTAL_SCHBENCH_WORKER_RUN_DELAY_NS,
            denominator: TOTAL_SCHBENCH_WORKER_PCOUNT,
        },
        default_abs: 1000.0,
        default_rel: 0.10,
        display_unit: "ns",
        accessor: |_| None,
    },
    // Per-phase latency min/max. LowerBetter (a higher min/max latency is worse).
    // min is a low-tail value → p50/p90 abs tier (20). max is a PEAK (a single
    // extreme sample, the flakiest latency stat) → the peak rel tolerance (0.50,
    // matching worst_gap_ms) so one outlier spike does not fabricate a regression.
    MetricDef {
        name: SCHBENCH_WAKEUP_MIN_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_WAKEUP_MAX_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 50.0,
        default_rel: 0.50,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_REQUEST_MIN_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_REQUEST_MAX_US,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 50.0,
        default_rel: 0.50,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    // Per-phase achieved-RPS distribution (PLIST_FOR_RPS = 20/50/90, schbench.c:130)
    // + min/max (schbench.c:579 stderr footer + :713-714/:1963 JSON — parity with
    // what schbench emits, not an extension). HigherBetter (more requests/sec = more
    // throughput) — note min/max INVERT the latency polarity (a higher worst-second
    // rate is better). A per-second RATE spanning tens..tens-of-thousands, so
    // rel-dominant (rel 0.10) with a small abs floor (100) — NOT loop_count's
    // count-style abs 10/rel 0.30.
    //
    // rps min/max keep the percentile-tier rel (0.10), NOT the loosened latency-max
    // tier (0.50): each rps sample is a 1-second-AVERAGED rate (cycles completed that
    // second), not a single event like a latency sample, so an rps extreme is the
    // worst/best SECOND — far less flaky than a latency per-request peak, and the
    // worst-second is a meaningful scheduler-tail signal worth a tight gate. (The
    // latency-max 0.50 loosening guards single-request spikes that do not exist in the
    // 1s-averaged rps series.)
    //
    // rps_min is UNRELIABLE when any 0-rps (starvation) second occurs: TWO independent
    // paths drop a real 0 from the min. (1) Within a histogram, add_lat's min==0
    // sentinel (plat.rs `if min==0 || us<min`) treats 0 as "unset" — a 0 sets min=0 but
    // the next sample replaces it (e.g. [100,0,200] -> min=200), so min reads 0 only
    // when a 0 is the last min-lowering sample. (2) Across cgroups, PlatStats::combine's
    // `other.min != 0` guard (plat.rs:227 — correct for latency, where 0 means empty)
    // skips a starved cgroup's min=0 when pooling, so a 0-rps cgroup pooled with a
    // nonzero one leaves rps_min nonzero. rps_min is thus a trustworthy worst-second
    // floor only absent 0-seconds. Sustained starvation (0-seconds >= 20% of the
    // window) still shows in rps_p20, which reads the pooled histogram's bucket 0
    // (folded unconditionally, plat.rs:221) in both cases; a single 0-second in a longer
    // window is below p20 and lost from rps_min — invisible to both. Faithful to
    // schbench's add_lat min sentinel.
    MetricDef {
        name: SCHBENCH_RPS_P20,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerPhase,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "req/s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_RPS_P50,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerPhase,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "req/s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_RPS_P90,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerPhase,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "req/s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_RPS_MIN,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerPhase,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "req/s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_RPS_MAX,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerPhase,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "req/s",
        accessor: |_| None,
    },
    // schbench WHOLE-RUN distributional metrics (MetricKind::
    // PerRunDistribution): each per-phase percentile/min/max above, re-pooled
    // run-level by populate_run_pooled_schbench_distribution (union of the
    // per-phase per-cgroup PlatStats histograms, percentile re-derived over the
    // union — the faithful percentile-of-union). Noise-compared per-run (never
    // cross-run folded). Thresholds + polarity + unit mirror the per-phase
    // sibling. accessor |_| None (ext-only, written by the union populate).
    MetricDef {
        name: SCHBENCH_WAKEUP_P50_US_WHOLE,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_WAKEUP_P90_US_WHOLE,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_WAKEUP_P99_US_WHOLE,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_WAKEUP_P999_US_WHOLE,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_WAKEUP_MIN_US_WHOLE,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_WAKEUP_MAX_US_WHOLE,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 50.0,
        default_rel: 0.50,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_REQUEST_P50_US_WHOLE,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_REQUEST_P90_US_WHOLE,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_REQUEST_P99_US_WHOLE,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_REQUEST_P999_US_WHOLE,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_REQUEST_MIN_US_WHOLE,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_REQUEST_MAX_US_WHOLE,
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 50.0,
        default_rel: 0.50,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_RPS_P20_WHOLE,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "req/s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_RPS_P50_WHOLE,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "req/s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_RPS_P90_WHOLE,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "req/s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_RPS_MIN_WHOLE,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "req/s",
        accessor: |_| None,
    },
    MetricDef {
        name: SCHBENCH_RPS_MAX_WHOLE,
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerRunDistribution,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "req/s",
        accessor: |_| None,
    },
    // -- Per-cgroup per-phase NON-schbench families. PerPhase, `accessor:
    // |_| None`: read from `PhaseCgroupStats::metrics` by name (written by
    // `write_carrier_scalars`), never from a `GauntletRow`. BARE per-cgroup
    // names (NOT the run-level `worst_*`): a single cgroup's value is not a
    // "worst across cgroups", and reusing `worst_*` would collide
    // `is_known_metric` with the run-level selector. Thresholds mirror the
    // analogous `worst_*` entries. (`iterations_per_cpu_sec` is intentionally
    // absent — it is already a Rate entry above; the per-cgroup value resolves
    // through that name without a second registration.)
    MetricDef {
        name: "p99_wake_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: "median_wake_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: "wake_latency_cv",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 0.10,
        default_rel: 0.25,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        // Per-cgroup per-phase timer-latency (WorkType::TimerLatency). PerPhase,
        // accessor |_| None: read from PhaseCgroupStats::metrics by name
        // (written by write_carrier_scalars). Bare name (not worst_*) — a single
        // cgroup's value, not a worst-across-cgroups.
        name: "p99_timer_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: "median_timer_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: "p999_timer_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 100.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: "mean_run_delay_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: "max_run_delay_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 100.0,
        default_rel: 0.50,
        display_unit: "\u{00b5}s",
        accessor: |_| None,
    },
    MetricDef {
        name: "avg_off_cpu_pct",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 5.0,
        default_rel: 0.25,
        display_unit: "%",
        accessor: |_| None,
    },
    MetricDef {
        name: "min_off_cpu_pct",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 5.0,
        default_rel: 0.25,
        display_unit: "%",
        accessor: |_| None,
    },
    MetricDef {
        name: "max_off_cpu_pct",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 5.0,
        default_rel: 0.25,
        display_unit: "%",
        accessor: |_| None,
    },
    MetricDef {
        name: "off_cpu_spread_pct",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 5.0,
        default_rel: 0.25,
        display_unit: "%",
        accessor: |_| None,
    },
    MetricDef {
        name: "migration_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 0.05,
        default_rel: 0.20,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        name: "iterations_per_worker",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerPhase,
        default_abs: 10.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        name: "page_locality",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::PerPhase,
        default_abs: 0.05,
        default_rel: 0.10,
        display_unit: "",
        accessor: |_| None,
    },
    MetricDef {
        name: "cross_node_migration_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::PerPhase,
        default_abs: 0.05,
        default_rel: 0.20,
        display_unit: "",
        accessor: |_| None,
    },
    // Per-cgroup carrier counter (read via `cgroup_counter` /
    // `cgroup_counter_total`), Counter kind, `accessor: |_| None` (no
    // GauntletRow field). Mirrors `total_iterations`' HigherBetter polarity.
    MetricDef {
        name: "total_cpu_time_ns",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Counter,
        default_abs: 100.0,
        default_rel: 0.10,
        display_unit: "ns",
        accessor: |_| None,
    },
];

/// Minimum total iterations a run must have accumulated before the
/// `worst_wake_latency_tail_ratio` metric participates in regression
/// math.
///
/// Below this threshold the p99 / median ratio is dominated by a
/// handful of outlier samples rather than a distributional signal:
/// p99 on an N-sample set where `N < 100` collapses to approximately
/// `samples.max()` (the empirical p99 sits at the Nth item of a
/// sorted set, rounded down, so with N=10 every "p99" is in fact the
/// maximum), and the ratio `max/median` swings by order of magnitude
/// across runs that differ only in which worker happened to hit a
/// scheduling stall. `compare_rows` would report those swings as
/// regressions / improvements, burying real signal under low-N noise.
///
/// 100 is the threshold of interest because percentile estimation
/// stabilizes when the sample count crosses `1 / (1 - target_p)` —
/// i.e. 100 samples for a p99 — which is the point at which at least
/// one sample is expected in the 99th-percentile tail by pigeonhole.
/// Below this floor the p99 estimator degenerates to the observed
/// maximum (`samples[99]` when N is exactly 100, and a still-sparse
/// tail at N just above 100). Above 100 the ratio begins to reflect
/// actual tail behavior rather than single-sample extrema.
///
/// The gate uses `total_iterations` (scenario-wide sum across every
/// cgroup in the run) as a coarse floor, not an exact per-cgroup
/// sample count. That sum OVERESTIMATES the per-cgroup iteration
/// count when the scenario has multiple cgroups sharing load, so a
/// scenario whose total just clears the floor may still have
/// individual cgroups with fewer than 100 iterations and therefore
/// noisy per-cgroup tail ratios. The floor is a minimum-viable
/// filter against the lowest-N degeneracy, not a guarantee that
/// every cgroup in a passing row has a stable p99.
///
/// The gate is applied at the PRODUCER, not an accessor:
/// `crate::assert::populate_run_distribution_metrics` emits no
/// `worst_wake_latency_tail_ratio` ext key for a run with
/// `total_iterations < WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`. The absent key
/// is excluded from the cross-RUN mean and read as `None` by `compare_rows`,
/// which `unwrap_or(0.0)`s both A- and B-side rows into the
/// `abs() < EPSILON` "unchanged" guard, emitting no finding.
pub const WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS: u64 = 100;

/// Look up a metric definition by name.
pub fn metric_def(name: &str) -> Option<&'static MetricDef> {
    METRICS.iter().find(|m| m.name == name)
}

/// Rate-COMPONENT metric names suppressed from compare OUTPUT (scalar findings,
/// per-phase deltas, and unpaired-phase rows). These are the internal
/// numerator/denominator Counters of the derived rates — `iteration_rate`
/// (`total_phase_iterations` / `total_phase_duration_sec`) and the pooled
/// `iterations_per_cpu_sec` (`total_iterations_pooled` / `total_cpu_time_sec`) —
/// and emitting them alongside their rate is redundant: three rows for one
/// user-facing concept.
///
/// They are suppressed ONLY at the compare-render layer. They REMAIN in the
/// persisted sidecar, in `GauntletRow::ext_metrics`, and in
/// `PhaseBucket::metrics`, because the cross-RUN re-pool
/// ([`group_and_average_by`]) re-derives the rates as `Σnum / Σdenom` from these
/// components read out of the rows — stripping them from storage would break
/// rate aggregation. The two user-facing rates and the typed `total_iterations`
/// are NOT suppressed. (Their `default_abs`/`default_rel` thresholds are inert
/// while suppressed — the compare significance gate never reads them — but the
/// entries keep their registry slot: `name` is the re-pool component key and
/// `kind` drives the fold dispatch.)
const RENDER_SUPPRESSED_COMPONENTS: &[&str] = &[
    "total_phase_iterations",
    "total_phase_duration_sec",
    "total_iterations_pooled",
    "total_cpu_time_sec",
    // taobench whole-run qps / hit_fraction Rate components (the raw op counts +
    // wall window). Suppressed so compare shows the four `taobench_*` Rates, not
    // the redundant raw counts. Remain in the sidecar / row for the cross-RUN
    // Σnum/Σdenom re-pool, like the iterations components above.
    TOTAL_TAOBENCH_OPS,
    TOTAL_TAOBENCH_FAST_OPS,
    TOTAL_TAOBENCH_SLOW_OPS,
    TOTAL_TAOBENCH_WALL_SEC,
    // schbench role-separate run-delay gate-Rate components (raw run_delay_ns +
    // pcount per role). Suppressed so compare shows the two
    // `schbench_*_run_delay_ns_per_sched` Rates, not the raw Σ pairs. Remain in
    // the row for the cross-RUN Σ/Σ re-pool. (total_schbench_loops is NOT here —
    // it is a standalone throughput Counter, not a rate component.)
    TOTAL_SCHBENCH_MSG_RUN_DELAY_NS,
    TOTAL_SCHBENCH_MSG_PCOUNT,
    TOTAL_SCHBENCH_WORKER_RUN_DELAY_NS,
    TOTAL_SCHBENCH_WORKER_PCOUNT,
];

/// True when `name` is a Rate component suppressed from compare output (see
/// [`RENDER_SUPPRESSED_COMPONENTS`]).
pub(crate) fn is_render_suppressed_component(name: &str) -> bool {
    RENDER_SUPPRESSED_COMPONENTS.contains(&name)
}

/// Clone a per-phase metrics map with the suppressed Rate components removed —
/// used for the unpaired-phase compare rows so a side-only phase does not render
/// the component plumbing (see [`RENDER_SUPPRESSED_COMPONENTS`]).
pub(crate) fn metrics_without_suppressed(
    metrics: &std::collections::BTreeMap<String, f64>,
) -> std::collections::BTreeMap<String, f64> {
    metrics
        .iter()
        .filter(|(k, _)| !is_render_suppressed_component(k.as_str()))
        .map(|(k, v)| (k.clone(), *v))
        .collect()
}

/// Infer the regression polarity (`higher_is_worse`) of a metric
/// not present in [`METRICS`].
///
/// Used by [`crate::assert::AssertResult::merge`] when it folds an
/// `ext_metrics` value whose name is not registered. Returning the
/// wrong polarity here surfaces as a silent merge bug: a
/// throughput-shaped metric (`*_iops`, `*_throughput`) folded with
/// `max` keeps the BETTER value across cgroups instead of the
/// worst, masking the cgroup that fell behind. The previous
/// fallback (`unwrap_or(true)` — always max) had this exact bug
/// for any payload-author metric whose name was not pre-registered
/// in the static `METRICS` table.
///
/// The inference is name-substring based, in the style of the
/// `Polarity::Unknown` fallback used by `MetricHint`. The token
/// list mirrors the polarity choices in [`METRICS`] for the
/// metrics already registered there:
///
/// - Tokens that signal HigherBetter (returned `false`):
///   `iops`, `throughput`, `bandwidth`, `iterations`, `ops_per_sec`,
///   `locality`, `_score`, `goodput`. The scheduler-test fixture's
///   `total_iterations` and `worst_iterations_per_worker` already
///   carry this polarity in the registry; a payload-author metric
///   like `jobs.0.read.iops` from the schbench LlmExtract path
///   should fold the same way.
/// - Tokens that signal LowerBetter (returned `true`):
///   `latency`, `delay`, `gap`, `stall`, `stuck`, `cv`, `error`,
///   `fail`, `drop`, `spread`, `_us`, `_ms`, `_ns`, `migration_ratio`,
///   `imbalance`. These are the polarity signals from the existing
///   registered LowerBetter entries (`worst_p99_wake_latency_us`,
///   `worst_run_delay_us`, `worst_gap_ms`, `stuck_count`,
///   `worst_wake_latency_cv`, `worst_spread`, `worst_migration_ratio`,
///   `max_imbalance_ratio`). `stall` covers payload-author metrics
///   that surface the sched_ext watchdog stall (`SCX_EXIT_ERROR_STALL`)
///   while `stuck` covers `stuck_count` (CPU's `rq_clock` not
///   advancing) — distinct conditions but both higher-is-worse.
///
/// When a name matches no token (e.g. `bogo_ops`, `read_kb`,
/// `jobs.0.runtime`), returns `true` (LowerBetter). The fallback
/// is conservative for regression detection: a payload that emits
/// a not-yet-classifiable metric and then folds an unexpectedly
/// high value across cgroups is more useful surfaced than silently
/// kept at the minimum (which would mask the high reading
/// entirely). Authors who need a different default should register
/// a [`MetricDef`] in [`METRICS`] or tag the metric via
/// [`crate::test_support::MetricHint`].
///
/// Token order matters when names contain both signals (e.g. the
/// hypothetical `low_iops_latency_ms` would match `latency` first
/// and be classified as higher-is-worse). The token lists above
/// are tested by `infer_higher_is_worse_*` in this module's tests.
pub fn infer_higher_is_worse(name: &str) -> bool {
    // First-pass: explicit "higher value is the regression" signals
    // (latency, delay, error, etc.). Checked first so a name
    // carrying both kinds of token (rare; e.g. `*_iops_latency_us`)
    // resolves to the latency interpretation, which matches the
    // semantics of compound counters/timers.
    const HIGHER_IS_WORSE_TOKENS: &[&str] = &[
        "latency",
        "delay",
        "_gap",
        "stall",
        "stuck",
        "_cv",
        "error",
        "fail",
        "drop",
        "spread",
        "_us",
        "_ms",
        "_ns",
        "migration_ratio",
        "imbalance",
        // DSQ depth is "lower is better" — a shallower queue
        // means the scheduler is keeping up. `_depth` and `dsq`
        // are independently meaningful: `_depth` catches names
        // like `max_dsq_depth` / `avg_dsq_depth` whose source is
        // the scheduler's local dispatch queue; `dsq` is the
        // defensive fallback for future DSQ-related metric names
        // that don't carry `_depth` (e.g. `dsq_overflow_count`).
        // Without these tokens, a future refactor that drops a
        // DSQ metric from the METRICS registry would fall through
        // to the conservative `true` default — correct by luck
        // for DSQ depth (higher = worse) but not by reasoning;
        // these tokens make the inference grounded.
        //
        // False-positive caveat: a future metric named
        // `dsq_throughput` / `dsq_iops` / `cache_depth` /
        // `tree_depth` would be classified higher-is-worse
        // here when the truth is the opposite. The fallback
        // path matters only when METRICS doesn't register the
        // name explicitly — register every new dsq-or-depth
        // metric so the token-based inference never runs.
        "_depth",
        "dsq",
    ];
    if HIGHER_IS_WORSE_TOKENS.iter().any(|t| name.contains(t)) {
        return true;
    }
    // Second-pass: "higher value is the improvement" signals
    // (throughput, iops, etc.). Matching here returns `false`
    // (LowerBetter inverted into HigherBetter, i.e. min is the
    // worst-case fold).
    const HIGHER_IS_BETTER_TOKENS: &[&str] = &[
        "iops",
        "throughput",
        "bandwidth",
        "iterations",
        "ops_per_sec",
        "locality",
        "_score",
        "goodput",
    ];
    if HIGHER_IS_BETTER_TOKENS.iter().any(|t| name.contains(t)) {
        return false;
    }
    // Conservative fallback: treat as higher-is-worse so a folded
    // value is the maximum across cgroups. Surfacing a maximum is
    // safer than masking it; payload authors who disagree should
    // register the metric.
    true
}

/// Render the [`METRICS`] registry for `cargo ktstr stats list-metrics`.
///
/// `json=false` renders a comfy-table with one row per registered
/// metric and columns NAME / POLARITY / DEFAULT_ABS / DEFAULT_REL
/// / UNIT. `json=true` emits `serde_json::to_string_pretty`
/// on the whole [`METRICS`] slice — the `accessor` fn-pointer is
/// `#[serde(skip)]` so the array carries only wire-stable fields.
///
/// Iteration order equals [`METRICS`] declaration order (the
/// canonical surface order for sidecar / CI-gate consumers).
///
/// The return is owned `String` rather than a print-direct helper so
/// callers can pin output via `assert_eq!` in tests; the cargo-ktstr
/// dispatch arm at `run_stats` writes it to stdout verbatim.
pub fn list_metrics(json: bool) -> anyhow::Result<String> {
    if json {
        return serde_json::to_string_pretty(METRICS)
            .map_err(|e| anyhow::anyhow!("serialize METRICS to JSON: {e}"));
    }

    let mut table = crate::cli::new_table();
    table.set_header(vec![
        "NAME",
        "POLARITY",
        "DEFAULT_ABS",
        "DEFAULT_REL",
        "UNIT",
    ]);
    for m in METRICS {
        table.add_row(vec![
            m.name.to_string(),
            polarity_label(m.polarity),
            format!("{}", m.default_abs),
            format!("{}", m.default_rel),
            m.display_unit.to_string(),
        ]);
    }
    Ok(format!("{table}\n"))
}

/// Short human label for a [`Polarity`](crate::test_support::Polarity)
/// variant in the list-metrics table.
///
/// `HigherBetter` → `higher`, `LowerBetter` → `lower`,
/// `TargetValue(t)` → `target(t)`, `Unknown` → `unknown`. Match is
/// total; adding a new `Polarity` variant without extending this
/// rendering surfaces as a compile error.
fn polarity_label(p: crate::test_support::Polarity) -> String {
    use crate::test_support::Polarity;
    match p {
        Polarity::HigherBetter => "higher".to_string(),
        Polarity::LowerBetter => "lower".to_string(),
        Polarity::TargetValue(t) => format!("target({t})"),
        Polarity::Unknown => "unknown".to_string(),
        Polarity::Informational => "informational".to_string(),
    }
}
