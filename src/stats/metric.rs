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
    /// Derived LOWEST-WINS per-cgroup efficiency selector — the worst
    /// (lowest) cgroup's `numerator / denominator` rate across the run,
    /// re-pooled from per-cgroup counters rather than folded from
    /// ready-made rates. None-aware lowest-wins (the semantic the deleted
    /// `fold_lowest_some` carried in [`crate::assert::AssertResult::merge`],
    /// now in `crate::assert::populate_run_distribution_metrics`): a measured
    /// `Some(0.0)` (a cgroup that ran zero iterations — real starvation)
    /// wins the worst bucket, a not-measured `None` (no workers / no
    /// on-CPU time) is skipped, and an all-`None` cohort produces no key
    /// (absence preserved as a missing ext entry, never a `0.0`).
    ///
    /// Derived post-merge by
    /// `crate::assert::populate_run_distribution_metrics` from the
    /// `stats.cgroups[]` counters (which survive bulk-frame stripping, so
    /// WorstLowest needs no degraded fallback). Like Distribution it is
    /// `is_derived` (skipped at the within-run reducers) and CROSS-RUN it
    /// MEAN-folds the per-run derived values through [`aggregate_finite`].
    WorstLowest {
        /// The per-cgroup iteration-count numerator.
        numerator: WorstLowestNumerator,
        /// The per-cgroup denominator the iteration count is divided by.
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
pub enum SampleSource {
    /// Per-wakeup latency samples in ns
    /// (`crate::assert::PhaseCgroupStats::wake_latencies_ns`). One sample per
    /// observed wakeup (reservoir-capped per cgroup), so the pooled set is the
    /// cross-cgroup union of those capped per-wakeup samples.
    WakeLatencyNs,
    /// Per-worker schedstat run-delay samples in ns
    /// (`crate::assert::PhaseCgroupStats::run_delays_ns`). One sample per worker
    /// — each is that worker's whole-run cumulative `sched_info.run_delay`
    /// delta (last-minus-first), so the pool size is the worker count, NOT a
    /// per-wakeup stream like `WakeLatencyNs`.
    RunDelayNs,
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

/// The per-cgroup iteration-count numerator of a
/// [`MetricKind::WorstLowest`] efficiency selector. Single variant today
/// (`Iterations`); the slot mirrors [`MetricKind::Rate`]'s `numerator` and is
/// `#[non_exhaustive]` so a future numerator (e.g. a work-unit count) can be
/// added without a breaking change. The producer matches only on the
/// `denominator`, treating the numerator as always-iterations for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[non_exhaustive]
pub enum WorstLowestNumerator {
    /// Per-cgroup total iteration count
    /// (`crate::assert::CgroupStats::total_iterations`).
    Iterations,
}

/// The per-cgroup denominator a [`MetricKind::WorstLowest`] iteration
/// count is divided by to form the efficiency rate.
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
    /// from two already-reduced values. Covers all three
    /// [`MetricKind::is_derived`] kinds:
    /// - [`MetricKind::Rate`]: re-derived as `Σnumerator / Σdenominator` from
    ///   its component keys by [`derive_rate_metrics`];
    /// - [`MetricKind::Distribution`] / [`MetricKind::WorstLowest`]: re-pooled
    ///   from the raw per-cgroup samples / counters by
    ///   `crate::assert::populate_run_distribution_metrics`.
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
        }
    }

    /// Whether this kind is DERIVED post-merge from other data rather than
    /// reduced from its own per-phase sample slice: [`MetricKind::Rate`]
    /// (from numerator/denominator components), [`MetricKind::Distribution`]
    /// (re-pooled from the per-cgroup raw sample sets), [`MetricKind::WorstLowest`]
    /// (lowest-wins over per-cgroup counters), and
    /// [`MetricKind::WakeLatencyTailRatio`] (max over the per-cgroup p99/median
    /// wake-latency ratios, floor-gated).
    ///
    /// Drives the WITHIN-RUN skip-sites that must not reduce a derived kind
    /// from a slice: [`aggregate_samples_for_phase`] returns None, and the
    /// per-phase build, the cross-phase
    /// `crate::assert::merge_matched_phase_buckets` key-loop, and
    /// [`crate::assert::populate_run_ext_metrics_from_phases`] all skip the
    /// key then re-derive.
    ///
    /// NOT a uniform cross-RUN skip: at the cross-RUN ext fold
    /// ([`group_and_average_by`]) ONLY [`MetricKind::Rate`] is skipped —
    /// its components survive cross-RUN so it re-derives there — while
    /// Distribution / WorstLowest / WakeLatencyTailRatio, whose components do
    /// NOT survive cross-RUN, fall through to be plainly folded (MEAN, or MAX for
    /// [`SampleReduction::Worst`]) by [`aggregate_finite`]. So callers
    /// gate on `is_derived` for the within-run sites and on
    /// `matches!(.., Rate { .. })` for the cross-RUN ext fold.
    pub fn is_derived(self) -> bool {
        matches!(
            self,
            MetricKind::Rate { .. }
                | MetricKind::Distribution { .. }
                | MetricKind::WorstLowest { .. }
                | MetricKind::WakeLatencyTailRatio
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
        // Distribution Worst (peak run-delay): the cross-RUN fold is MAX
        // so the high-water peak survives, distinct from the MEAN-folded
        // percentile / CV / mean reductions below. (WITHIN-RUN no
        // Distribution/WorstLowest reaches here — `is_derived` skips them at
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
        | MetricKind::WakeLatencyTailRatio => finite.iter().sum::<f64>() / (finite.len() as f64),
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
        // Derived kinds (Rate / Distribution / WorstLowest) have no samples
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
/// Edge cases:
///   - 0 finite samples -> `None`.
///   - 1 finite sample -> `Some(0.0)` (self-delta; the metric
///     was observed but no per-phase change can be computed).
///   - 2+ finite samples -> `Some(max(0.0, last - first))`.
///
/// Live caller: [`aggregate_samples_for_phase`] dispatches the
/// Counter variant through this entry point.
pub fn phase_counter_delta(samples: &[f64]) -> Option<f64> {
    let finite: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();
    match finite.as_slice() {
        [] => None,
        [_only] => Some(0.0),
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

    /// Returns `true` for [`crate::test_support::Polarity::LowerBetter`], `false` for
    /// [`crate::test_support::Polarity::HigherBetter`]. [`crate::test_support::Polarity::TargetValue`] and
    /// [`crate::test_support::Polarity::Unknown`] branches keep the match total; they
    /// are unreachable for the current [`METRICS`] entries (guarded
    /// by the `metric_def_polarity_covers_all_entries` test).
    pub const fn higher_is_worse(&self) -> bool {
        use crate::test_support::Polarity;
        matches!(
            self.polarity,
            Polarity::LowerBetter | Polarity::TargetValue(_) | Polarity::Unknown
        )
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
/// Eleven divergent triples:
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
/// | `worst_page_locality` | `page_locality` | `page_locality` |
/// | `worst_cross_node_migration_ratio` | `cross_node_migration_ratio` | `cross_node_migration_ratio` |
///
/// One of the remaining metrics in [`METRICS`] has matching
/// registry / field / DataFrame column names backed by a typed
/// `GauntletRow` field (`total_iterations`) and is not listed — no
/// translation to document.
///
/// The eight wake-latency / run-delay / iteration-efficiency roll-ups
/// (`worst_p99_wake_latency_us`, `worst_median_wake_latency_us`,
/// `worst_wake_latency_cv`, `worst_mean_run_delay_us`,
/// `worst_run_delay_us`, `worst_iterations_per_worker`,
/// `worst_iterations_per_cpu_sec`, `worst_wake_latency_tail_ratio`) are
/// DERIVED kinds ([`MetricKind::Distribution`] / [`MetricKind::WorstLowest`]
/// / [`MetricKind::WakeLatencyTailRatio`]) with NO typed `GauntletRow`
/// field: their accessors are `|_| None` and
/// `crate::assert::populate_run_distribution_metrics` re-pools their value
/// into `ext_metrics` post-merge, so [`MetricDef::read`] reads them through
/// the ext fallback.
///
/// `worst_` naming convention: it is the codebase-wide prefix for a
/// cross-cgroup roll-up, independent of polarity and of HOW the roll-up is
/// formed. Polarity-directional selectors (`worst_spread` LowerBetter →
/// max; `worst_page_locality` HigherBetter → lowest-non-zero) and
/// [`MetricKind::WorstLowest`] (`worst_iterations_per_*`, None-aware
/// lowest-wins) both surface the most problematic cgroup; whereas
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
        default_abs: 1.0,
        default_rel: 0.50,
        display_unit: "",
        accessor: |r| Some(r.stuck_count as f64),
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
        // group spent during the phase. Gauge(Avg): the per-phase value
        // is already a delta (one per phase; cross-RUN folds by mean,
        // like user_time_ns). LowerBetter — the DSQ-spinlock
        // regression surfaces as rising system time (CPUs spinning in
        // the kernel). No typed GauntletRow field; the ext_metrics
        // fallback carries it through cargo ktstr stats compare.
        name: "system_time_ns",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Avg),
        default_abs: 1_000_000.0,
        default_rel: 0.30,
        display_unit: "ns",
        accessor: |_| None,
    },
    MetricDef {
        // Per-phase USER-mode CPU time in nanoseconds. Same host-side /
        // injected / Gauge(Avg) shape as `system_time_ns` (task_struct
        // .utime + the thread-group signal_struct.utime accumulator,
        // per-tgid delta via `crate::assert::phase_group_cpu_delta`).
        // Pairs with it so a test can distinguish "system time rose,
        // user work flat" (the lock-contention signature) from "both
        // rose" (genuine extra work). LowerBetter — less CPU consumed
        // for the same work is the efficiency win; utime already
        // includes gtime so the two are never summed.
        name: "user_time_ns",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Avg),
        default_abs: 1_000_000.0,
        default_rel: 0.30,
        display_unit: "ns",
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
        name: "worst_page_locality",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Gauge(GaugeAgg::Last),
        default_abs: 0.05,
        default_rel: 0.10,
        display_unit: "",
        accessor: |r| Some(r.page_locality),
    },
    MetricDef {
        name: "worst_cross_node_migration_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Last),
        default_abs: 0.05,
        default_rel: 0.20,
        display_unit: "",
        accessor: |r| Some(r.cross_node_migration_ratio),
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
    }
}
