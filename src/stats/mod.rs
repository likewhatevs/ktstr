//! Gauntlet analysis and run-to-run comparison.
//!
//! Collects per-scenario results into hand-rolled aggregation passes
//! (group-by via `BTreeMap`, mean / std via plain iterators) for
//! statistical analysis, regression detection, and run-to-run compare
//! workflows. The earlier polars-backed implementation paid for a
//! columnar engine that was overkill for the dozens-to-low-thousands
//! of rows a gauntlet produces; the hand-rolled form gives the same
//! per-scenario means, outlier rankings, and dimension summaries
//! without polars's ~30-40 transitive crates.

use std::collections::{BTreeMap, HashMap, HashSet};

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
fn derive_rate_metrics_from<'a>(
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
fn is_render_suppressed_component(name: &str) -> bool {
    RENDER_SUPPRESSED_COMPONENTS.contains(&name)
}

/// Clone a per-phase metrics map with the suppressed Rate components removed —
/// used for the unpaired-phase compare rows so a side-only phase does not render
/// the component plumbing (see [`RENDER_SUPPRESSED_COMPONENTS`]).
fn metrics_without_suppressed(
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
    /// Scheduler binary name carried from the source sidecar
    /// (`SidecarResult::scheduler`). Surfaced through the substring
    /// filter in [`compare_rows_by`] and the typed
    /// `RowFilter::scheduler` so users can narrow A/B comparisons
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
    ///   in sidecars, CI gates, and `cargo ktstr stats compare`
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
    pub stuck_count: usize,
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
    /// Worst-case per-cgroup NUMA page-locality fraction (lowest
    /// non-zero). Surfaced in [`METRICS`] under registry name
    /// `worst_page_locality`; see the triples table on
    /// [`METRICS`] for the registry/field/column rationale.
    pub page_locality: f64,
    /// Worst-case cross-node migration ratio. Surfaced in
    /// [`METRICS`] under registry name
    /// `worst_cross_node_migration_ratio`; see the triples table
    /// on [`METRICS`] for the registry/field/column rationale.
    pub cross_node_migration_ratio: f64,
    /// Extensible metrics populated by scenarios and processed by the
    /// comparison pipeline. Keyed by metric name; looked up via
    /// [`metric_def`] when a matching entry exists in [`METRICS`].
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ext_metrics: BTreeMap<String, f64>,
    /// Per-phase metric buckets carried verbatim from the source
    /// sidecar's [`crate::assert::ScenarioStats::phases`]. Each
    /// [`crate::assert::PhaseBucket`] surfaces the metric values
    /// reduced over one scenario phase (BASELINE at
    /// `step_index = 0`, Step ordinals at `step_index = 1..=N`
    /// per the 1-indexed phase convention) so the per-phase
    /// comparison renderer at [`compare_partitions`] can pair
    /// matched phases across two sidecars by `step_index` and
    /// emit per-phase delta rows without re-deriving phase
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
/// `cargo ktstr stats compare` pipeline. Every field is `None` /
/// empty by default; populated fields are AND-combined ACROSS
/// fields, with field-internal OR/AND semantics described per-field
/// below. Applied via `apply_row_filters` in `compare_partitions`
/// before the rows reach `compare_rows`.
///
/// Match semantics:
/// - `scheduler` / `topology` / `work_type` — STRICT EQUALITY against
///   the row's corresponding field. The sibling substring filter on
///   `compare_rows` (`-E`) stays as the only fuzzy-match knob;
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
    /// Repeatable cpu-budget filter, OR-combined: a row matches iff its
    /// `GauntletRow::cpu_budget` (the effective host-CPU budget, as a
    /// decimal string) equals ANY entry. Empty vec disables the filter.
    /// Rows with `cpu_budget == None` (skips) are dropped when this filter
    /// is non-empty, mirroring `kernels` / `run_sources`. Backs the
    /// [`Dimension::CpuBudget`] slice (`--cpu-budget` / `--a-cpu-budget` /
    /// `--b-cpu-budget`).
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
///   (`(sum / count).round() as u64`). The 0.5-unit rounding error
///   is well below every integer metric's `default_abs` gate (the
///   smallest is `stuck_count = 1.0`).
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
        sum_stuck_count: usize,
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

    let mut order: Vec<Key> = Vec::new();
    let mut groups: BTreeMap<Key, Accumulator<'_>> = BTreeMap::new();

    for row in rows {
        let key = PairingKey::from_row(row, pairing_dims);
        let acc = groups.entry(key.clone()).or_insert_with(|| {
            order.push(key);
            Accumulator {
                first: row,
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
                sum_stuck_count: 0,
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
        });
        acc.total_observed += 1;
        // Dirty-status tracking spans ALL contributors. Same hex
        // with mixed dirty/clean across the cohort is the case the
        // `+mixed` marker exists to surface — the per-row scope
        // (passing, failing, skipped) is irrelevant since the
        // marker describes WIP-vs-committed disagreement among the
        // contributors, not their metric outcomes.
        update_dirty_tracking(
            &row.commit,
            &mut acc.any_project_clean,
            &mut acc.any_project_dirty,
            &mut acc.first_project_base,
        );
        update_dirty_tracking(
            &row.kernel_commit,
            &mut acc.any_kernel_clean,
            &mut acc.any_kernel_dirty,
            &mut acc.first_kernel_base,
        );
        if row.is_skip() {
            acc.any_skipped = true;
            acc.skips_observed += 1;
            continue;
        }
        if row.is_fail() {
            acc.any_failed = true;
            acc.failures_observed += 1;
            continue;
        }
        if row.is_inconclusive() {
            // Inconclusive contributors are not passes (the gate
            // could not be evaluated) and carry no measured signal
            // worth folding into the cohort means. Track the bit
            // for the aggregated verdict's `inconclusive` field
            // (so the aggregate row reads Inconclusive in the
            // `Fail > Inconclusive > Pass > Skip` lattice when no
            // contributor failed) and skip the per-row sums.
            acc.any_inconclusive = true;
            acc.inconclusives_observed += 1;
            continue;
        }
        acc.passes_observed += 1;
        acc.sum_spread += row.spread;
        acc.sum_migrations = acc.sum_migrations.saturating_add(row.migrations);
        acc.sum_migration_ratio += row.migration_ratio;
        acc.sum_stuck_count = acc.sum_stuck_count.saturating_add(row.stuck_count);
        acc.sum_fallback_count = acc.sum_fallback_count.saturating_add(row.fallback_count);
        acc.sum_keep_last_count = acc.sum_keep_last_count.saturating_add(row.keep_last_count);
        acc.sum_total_iterations = acc
            .sum_total_iterations
            .saturating_add(row.total_iterations);
        acc.sum_page_locality += row.page_locality;
        acc.sum_cross_node_mig += row.cross_node_migration_ratio;
        // Peak-kind typed fields: cross-RUN aggregation surfaces
        // the worst-instant observed across the cohort, NOT the
        // arithmetic mean (which dilutes a single peak across
        // many quiet runs and hides the high-water signal).
        acc.max_gap_ms = acc.max_gap_ms.max(row.gap_ms);
        if row.imbalance_ratio > acc.max_imbalance_ratio {
            acc.max_imbalance_ratio = row.imbalance_ratio;
        }
        acc.max_max_dsq_depth = acc.max_max_dsq_depth.max(row.max_dsq_depth);
        acc.sum_run_sample_count = acc
            .sum_run_sample_count
            .saturating_add(row.run_sample_count);
        for (k, v) in &row.ext_metrics {
            acc.ext_pairs
                .entry(k.clone())
                .or_default()
                .push((*v, row.run_sample_count));
        }
    }

    let mut out = Vec::with_capacity(order.len());
    for key in order {
        let acc = groups
            .remove(&key)
            .expect("first-seen key must still be in groups map");
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
        let round_usize = |sum: usize| -> usize { (sum as f64 / denom).round() as usize };

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
        let mut ext_metrics: std::collections::BTreeMap<String, f64> = acc
            .ext_pairs
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
            stuck_count: round_usize(acc.sum_stuck_count),
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
        out.push(AveragedGroup {
            row: aggregated,
            passes_observed: acc.passes_observed,
            skips_observed: acc.skips_observed,
            inconclusives_observed: acc.inconclusives_observed,
            failures_observed: acc.failures_observed,
            total_observed: acc.total_observed,
        });
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
        stuck_count: if sc.monitor.as_ref().is_some_and(|m| m.stuck_detected) {
            1
        } else {
            0
        },
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

/// Detected outlier helper: extract one numeric metric off a
/// [`GauntletRow`]. The same accessor is used to compute the overall
/// (cohort-wide) mean/std and the per-scenario mean — keeping the
/// "what counts as `imbalance`?" decision in one place avoids
/// drift between the two passes that polars-side handled implicitly
/// via the `df!` column name.
type MetricAccessor = fn(&GauntletRow) -> f64;

/// Pinned list of `(display_name, accessor)` for every metric that
/// outlier detection considers. The display name appears in
/// [`Outlier`] output verbatim ("scenario: imbalance 4.5 ..."); the
/// accessor pulls the f64 value off a `GauntletRow`. Mirrors the
/// `metrics` slice the old polars code keyed off DataFrame column
/// names, so the outlier set surfaces the same metrics under the same
/// names.
const OUTLIER_METRICS: &[(&str, MetricAccessor)] = &[
    ("spread", |r| r.spread),
    ("gap_ms", |r| r.gap_ms as f64),
    ("migrations", |r| r.migrations as f64),
    ("migration_ratio", |r| r.migration_ratio),
    ("imbalance", |r| r.imbalance_ratio),
    ("dsq_depth", |r| r.max_dsq_depth as f64),
    ("stuck", |r| r.stuck_count as f64),
    ("fallback", |r| r.fallback_count as f64),
    ("keep_last", |r| r.keep_last_count as f64),
    // Distribution-kind roll-ups are ext_metrics-sourced (no typed field):
    // read them through the ext map, 0.0 when absent (the prior typed-field
    // default), mirroring the deleted `worst_*` accessors. The 0.0-on-absent
    // here is INTENTIONALLY distinct from the cross-RUN `group_and_average_by`
    // fold (via `aggregate_finite`'s Distribution arm), which EXCLUDES an absent
    // key from the mean (no-false-zero). The key is present whenever any cgroup
    // or carrier contributed; the 0.0-on-absent path is still reachable for a
    // telemetry-free pass (no phases AND no cgroups, e.g. host_only). A 0.0
    // here not only escapes being flagged as its own scenario's outlier — it
    // also ENTERS the cross-scenario overall_mean/overall_std baseline
    // `find_outliers` builds over all is_pass() rows, so it can shift the
    // 2-sigma threshold for the whole cohort. Both effects are benign ONLY
    // because every OUTLIER_METRICS Distribution entry is LowerBetter (a 0.0
    // reads as best, never the high tail outlier detection targets, and a
    // telemetry-free row carries no measured signal to begin with) — a coupling
    // the Distribution=>LowerBetter registry gate enforces. A HigherBetter ext
    // metric added here would NOT be benign (a 0.0 would depress the baseline
    // AND could itself read as a low outlier). So the two consumers diverge by
    // design, not by accident.
    ("worst_p99_wake_latency_us", |r| {
        r.ext_metrics
            .get("worst_p99_wake_latency_us")
            .copied()
            .unwrap_or(0.0)
    }),
    ("worst_wake_latency_cv", |r| {
        r.ext_metrics
            .get("worst_wake_latency_cv")
            .copied()
            .unwrap_or(0.0)
    }),
    ("worst_mean_run_delay_us", |r| {
        r.ext_metrics
            .get("worst_mean_run_delay_us")
            .copied()
            .unwrap_or(0.0)
    }),
    ("worst_run_delay_us", |r| {
        r.ext_metrics
            .get("worst_run_delay_us")
            .copied()
            .unwrap_or(0.0)
    }),
];

/// Arithmetic mean over the finite values produced by `iter`.
/// Non-finite values (NaN, ±inf) are excluded so a single outlier
/// or sentinel can't poison the mean. Returns 0.0 on an empty
/// (post-filter) input — matches what polars's `.mean()` does on a
/// chunked array of length zero.
fn mean<I: Iterator<Item = f64>>(iter: I) -> f64 {
    let (sum, count) = iter
        .filter(|x| x.is_finite())
        .fold((0.0_f64, 0usize), |(s, c), x| (s + x, c + 1));
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Sample standard deviation (Bessel-corrected, ddof = 1) over the
/// finite values produced by `iter`. Returns 0.0 when fewer than two
/// finite values remain — matches polars's `.std(1)` semantics on a
/// 0- or 1-element chunked array. Requires `Iterator + Clone` because
/// the computation needs two passes (mean, then squared deviations).
fn std_dev<I: Iterator<Item = f64> + Clone>(iter: I) -> f64 {
    let m = mean(iter.clone());
    let (sum_sq, count) = iter
        .filter(|x| x.is_finite())
        .fold((0.0_f64, 0usize), |(s, c), x| {
            let d = x - m;
            (s + d * d, c + 1)
        });
    if count < 2 {
        0.0
    } else {
        (sum_sq / (count - 1) as f64).sqrt()
    }
}

/// Extract a grouping dimension's `&str` field off a [`GauntletRow`].
/// Replaces the polars `col(group_col)` lookup with a fn-pointer
/// dispatch over the three accepted dimension names. Returns `None`
/// for any other column name — `analyze_rows` and
/// `format_dimension_summary` both restrict the dimension to one of
/// the three documented columns, so the `None` arm is unreachable in
/// production but kept as defense-in-depth against a stray call site.
fn group_field<'a>(row: &'a GauntletRow, col: &str) -> Option<&'a str> {
    match col {
        "scenario" => Some(row.scenario.as_str()),
        "topology" => Some(row.topology.as_str()),
        "work_type" => Some(row.work_type.as_str()),
        _ => None,
    }
}

/// Detected outlier: a scenario with an anomalous stat.
struct Outlier {
    scenario: String,
    metric: &'static str,
    value: f64,
    overall_mean: f64,
    sigma: f64,
    worst_topos: Vec<String>,
}

impl std::fmt::Display for Outlier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} {:.1} (overall avg {:.1}, +{:.1}\u{03c3})",
            self.scenario, self.metric, self.value, self.overall_mean, self.sigma
        )?;
        if !self.worst_topos.is_empty() {
            write!(f, "\n    worst on: {}", self.worst_topos.join(", "))?;
        }
        Ok(())
    }
}

// `col_f64`, `col_u32`, `col_str`, `col_mean_std`: removed alongside
// the polars dep. The aggregation paths now read metrics directly off
// `&GauntletRow` via the [`MetricAccessor`] / [`group_field`] dispatch
// at the top of this section, and per-iterator mean / std go through
// the standalone [`mean`] / [`std_dev`] helpers.

/// Find outlier scenarios where a metric exceeds 2 sigma.
///
/// For every metric in [`OUTLIER_METRICS`]: compute the
/// cohort-wide (overall_mean, overall_std), set the threshold at
/// `overall_mean + 2 * overall_std`, then check every per-scenario
/// mean against that threshold. Scenarios whose mean exceeds the
/// threshold get an [`Outlier`] entry annotated with the topology
/// rows (via [`find_worst_topos`]) that drove the excursion.
///
/// 4-state lattice filtering: only real-pass rows
/// (`passed && !skipped && !inconclusive`, matching
/// `GauntletRow::is_pass`) contribute to the per-scenario mean
/// AND the overall mean/std baseline. Skipped / inconclusive /
/// failed rows carry default-zero metric values (for typed fields
/// `sidecar_to_row` substitutes zero for non-finite + missing
/// fields; for the ext-sourced Distribution entries the
/// `OUTLIER_METRICS` accessor's own `.unwrap_or(0.0)` supplies the
/// zero, since `sidecar_to_row` copies only present finite ext keys
/// and never zero-fills a missing one; see the block comment on
/// those entries), and
/// including them would silently depress every measured mean — a
/// scenario with 1 real-pass run (value=100) and 9 inconclusive
/// runs (value=0) would otherwise report a per-scenario mean of
/// 10.0 and the same pollution would deflate the cohort-wide
/// baseline used for the 2-sigma threshold. Filtering on
/// `is_pass` matches the same defense-in-depth as
/// `format_dimension_summary`'s pass_count + the
/// `compare_rows_by` regression-math gate.
fn find_outliers(rows: &[GauntletRow]) -> Vec<Outlier> {
    let pass_rows: Vec<&GauntletRow> = rows.iter().filter(|r| r.is_pass()).collect();
    if pass_rows.is_empty() {
        return Vec::new();
    }

    // Bucket pass rows by scenario name. BTreeMap iterates in sorted
    // order so the outlier vector before the final sigma-sort is
    // already alphabetic per (metric, scenario) — deterministic across
    // runs even when multiple scenarios tie on sigma. Borrowing `&str`
    // out of the row avoids cloning the scenario names just to bucket.
    let mut by_scenario: BTreeMap<&str, Vec<&GauntletRow>> = BTreeMap::new();
    for r in &pass_rows {
        by_scenario.entry(r.scenario.as_str()).or_default().push(r);
    }

    let mut outliers = Vec::new();
    for &(name, accessor) in OUTLIER_METRICS {
        let overall_mean = mean(pass_rows.iter().map(|r| accessor(r)));
        let overall_std = std_dev(pass_rows.iter().map(|r| accessor(r)));
        // Drop metrics with std below epsilon. The cohort produced no
        // measurable spread on this metric, so flagging "outliers"
        // against a near-zero baseline would surface noise. Mirrors the
        // pre-polars `active.filter_map` short-circuit.
        if overall_std < f64::EPSILON {
            continue;
        }
        let threshold = overall_mean + 2.0 * overall_std;

        for (&scenario, rows_in_scenario) in &by_scenario {
            let scenario_mean = mean(rows_in_scenario.iter().map(|r| accessor(r)));
            if scenario_mean <= threshold {
                continue;
            }
            let sigma = (scenario_mean - overall_mean) / overall_std;
            // Worst topologies are computed against the full row set
            // (not the pass-only subset) so a failure cluster on a
            // particular topology still surfaces even when its rows
            // failed the is_pass gate — the outlier line is the place
            // where the operator first sees that topology, and gating
            // it on is_pass would hide the worst offenders.
            let worst = find_worst_topos(rows, scenario, accessor, threshold);
            outliers.push(Outlier {
                scenario: scenario.to_string(),
                metric: name,
                value: scenario_mean,
                overall_mean,
                sigma,
                worst_topos: worst,
            });
        }
    }

    outliers.sort_by(|a, b| {
        b.sigma
            .partial_cmp(&a.sigma)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    outliers
}

/// Topology names of rows in `scenario` whose metric value exceeds
/// `threshold`. Used by [`find_outliers`] to attribute a per-scenario
/// outlier to the specific topologies driving the excursion.
fn find_worst_topos(
    rows: &[GauntletRow],
    scenario: &str,
    accessor: MetricAccessor,
    threshold: f64,
) -> Vec<String> {
    // `filter` passes `&Self::Item`, so inside the closure `r` is
    // `&&GauntletRow`. Destructure with `|&r|` so `r: &GauntletRow`
    // and `accessor(r)` matches its `fn(&GauntletRow) -> f64`
    // signature without an explicit deref.
    rows.iter()
        .filter(|&r| r.scenario == scenario && accessor(r) > threshold)
        .map(|r| r.topology.clone())
        .collect()
}

/// Format a group-by summary for one dimension (`scenario`,
/// `topology`, or `work_type`). For each value of the dimension,
/// renders one line carrying:
///
/// `{name:<25} {pass}/{total} passed ({skip} skipped, {inconc} inconclusive, {fail} failed)
///  avg_spread={spread:.1}%  avg_gap={gap:.0}ms[  imbal=…][  dsq=…][  stuck=…][  fallback=…]`
///
/// Sorted by `avg_spread` descending so the worst dimension values
/// land at the top. Pass / skip / inconclusive / fail follow the
/// 4-state mutex documented in detail on the prior polars-side
/// pass_count aggregation: `fail` is the residual after subtracting
/// pass / skip / inconc from total, NOT a separate count. Splitting
/// out `inconclusive` from `failed` is what keeps a zero-denominator
/// inconclusive run from silently rendering as "failed."
///
/// `imbal` / `dsq` / `stuck` / `fallback` tail tokens render only
/// when their value crosses a per-metric threshold (`> 1.0` for
/// imbalance, `> 0.0` for the others) so a healthy dimension's line
/// stays terse.
///
/// Returns an empty string when `group_col` is not one of the three
/// accepted dimension names — matches the prior behavior of bailing
/// without panicking on a stray column name.
fn format_dimension_summary(rows: &[GauntletRow], group_col: &str) -> String {
    // Reject unknown dimension names up-front via `group_field`'s
    // None arm. The match exhausts at the first row — every row
    // resolves identically given the same `group_col` — so the
    // probe is O(1).
    if rows.is_empty()
        || rows
            .first()
            .and_then(|r| group_field(r, group_col))
            .is_none()
    {
        return String::new();
    }

    let mut by_dim: BTreeMap<&str, Vec<&GauntletRow>> = BTreeMap::new();
    for r in rows {
        if let Some(key) = group_field(r, group_col) {
            by_dim.entry(key).or_default().push(r);
        }
    }

    struct GroupStats<'a> {
        name: &'a str,
        pass_count: usize,
        skip_count: usize,
        inconc_count: usize,
        total: usize,
        avg_spread: f64,
        avg_gap_ms: f64,
        avg_imbalance: f64,
        avg_dsq_depth: f64,
        total_stuck: f64,
        avg_fallback: f64,
    }

    let mut groups: Vec<GroupStats> = by_dim
        .iter()
        .map(|(name, group_rows)| GroupStats {
            name,
            pass_count: group_rows.iter().filter(|r| r.is_pass()).count(),
            skip_count: group_rows.iter().filter(|r| r.is_skip()).count(),
            inconc_count: group_rows.iter().filter(|r| r.is_inconclusive()).count(),
            total: group_rows.len(),
            avg_spread: mean(group_rows.iter().map(|r| r.spread)),
            avg_gap_ms: mean(group_rows.iter().map(|r| r.gap_ms as f64)),
            avg_imbalance: mean(group_rows.iter().map(|r| r.imbalance_ratio)),
            avg_dsq_depth: mean(group_rows.iter().map(|r| r.max_dsq_depth as f64)),
            total_stuck: group_rows
                .iter()
                .map(|r| r.stuck_count as f64)
                .filter(|x| x.is_finite())
                .sum(),
            avg_fallback: mean(group_rows.iter().map(|r| r.fallback_count as f64)),
        })
        .collect();

    // Descending sort by avg_spread so the dimension value with the
    // worst spread reads first. Tie-breaker is the input order from
    // BTreeMap (alphabetic) — same deterministic shape as the polars
    // `sort([avg_spread], descending=true)` form.
    groups.sort_by(|a, b| {
        b.avg_spread
            .partial_cmp(&a.avg_spread)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = String::new();
    for g in &groups {
        let fail = g
            .total
            .saturating_sub(g.pass_count)
            .saturating_sub(g.skip_count)
            .saturating_sub(g.inconc_count);
        let mut line = format!(
            "  {:<25} {}/{} passed ({} skipped, {} inconclusive, {} failed)  avg_spread={:.1}%  avg_gap={:.0}ms",
            g.name,
            g.pass_count,
            g.total,
            g.skip_count,
            g.inconc_count,
            fail,
            g.avg_spread,
            g.avg_gap_ms,
        );
        if g.avg_imbalance > 1.0 {
            line.push_str(&format!("  imbal={:.1}", g.avg_imbalance));
        }
        if g.avg_dsq_depth > 0.0 {
            line.push_str(&format!("  dsq={:.0}", g.avg_dsq_depth));
        }
        if g.total_stuck > 0.0 {
            line.push_str(&format!("  stuck={}", g.total_stuck as u64));
        }
        if g.avg_fallback > 0.0 {
            line.push_str(&format!("  fallback={:.0}", g.avg_fallback));
        }
        line.push('\n');
        out.push_str(&line);
    }
    out
}

/// Analyze pre-built gauntlet rows and return a formatted report.
pub fn analyze_rows(rows: &[GauntletRow]) -> String {
    if rows.is_empty() {
        return String::new();
    }

    let mut report = String::from("\n=== GAUNTLET ANALYSIS ===\n\n");

    let outliers = find_outliers(rows);
    if outliers.is_empty() {
        report.push_str("No outliers detected.\n");
    } else {
        report.push_str("Outliers detected:\n");
        for o in &outliers {
            report.push_str(&format!("  {o}\n"));
        }
    }

    report.push_str("\nBy scenario (worst first):\n");
    report.push_str(&format_dimension_summary(rows, "scenario"));

    report.push_str("\nBy topology:\n");
    report.push_str(&format_dimension_summary(rows, "topology"));

    // Surface a "By work_type" pane only when the input carries
    // more than one work_type value; a single-work_type cohort
    // would render an identical "all rows" pane to the scenario
    // pane and add visual noise.
    let work_types: std::collections::BTreeSet<&str> =
        rows.iter().map(|r| r.work_type.as_str()).collect();
    if work_types.len() > 1 {
        report.push_str("\nBy work_type:\n");
        report.push_str(&format_dimension_summary(rows, "work_type"));
    }

    report
}

// ---------------------------------------------------------------------------
// Test-run enumeration and A/B comparison
// ---------------------------------------------------------------------------

/// List the test-run directories under
/// `{CARGO_TARGET_DIR or "target"}/ktstr/`.
///
/// Each subdirectory is one run keyed `{kernel}-{project_commit}`
/// where `{project_commit}` is the project HEAD short hex with
/// `-dirty` suffix when the worktree differs. Two runs sharing the
/// same key reuse the same directory: the second run pre-clears
/// prior `*.ktstr.json` files at first write so the directory is a
/// last-writer-wins snapshot of (kernel, project commit) rather
/// than an append-only archive of every invocation.
///
/// The rendered table carries four columns:
/// - `RUN`: the run-directory leaf name
///   (`{kernel}-{project_commit}` per the keying above).
/// - `TESTS`: number of `*.ktstr.json` sidecars in the directory
///   (and one level of subdirectories for per-job gauntlet
///   layouts that `collect_sidecars` walks).
/// - `DATE`: the earliest sidecar timestamp in the directory.
///   Under last-writer-wins this equals the most recent run's
///   first sidecar timestamp because the prior run's sidecars
///   were pre-cleared at the new run's first write.
/// - `ARCH`: the `host.arch` value (`x86_64`, `aarch64`, …) from
///   the run's first sidecar that carries a populated host
///   field. Renders as `-` when no sidecar carries a host
///   (pre-host-context-landing archives, host-only test stubs
///   that never populate host) so the column reads consistently
///   with the `DATE` sentinel.
///
/// Rows are sorted by directory mtime, **most recent first**, so
/// the latest run lands at the top of the table — the operator's
/// usual interest. Sorting by `file_name()` would produce
/// alphabetical-by-hex output (the `{project_commit}` half is a
/// hex short-hash with no temporal ordering), which scatters
/// chronologically-adjacent runs across the listing. `file_name`
/// is the tiebreaker on the secondary axis: pairs with equal
/// mtimes (or both unreadable) collapse onto a deterministic
/// alphabetical order so the listing stays stable across
/// re-invocations. Entries whose mtime cannot be read at all
/// (filesystem error, permission issue) sort to the END of the
/// listing — `Reverse(None) > Reverse(Some(_))` lands them after
/// every dated entry, with the file_name tiebreaker keeping the
/// undated group itself stable.
pub fn list_runs() -> anyhow::Result<()> {
    let root = crate::test_support::runs_root();
    // Both the missing-root case (target/ktstr/ never created) and
    // the empty-rows case (directory exists but no run
    // subdirectories pass `is_run_directory`) reduce to the same
    // operator state: no test data is on disk yet. Surface the
    // same actionable hint in both branches so a fresh checkout
    // sees consistent guidance regardless of whether
    // `target/ktstr/` was ever created.
    let hint = "Run `cargo ktstr test` to generate sidecar data.";
    if !root.exists() {
        eprintln!("no runs found at {}. {hint}", root.display());
        return Ok(());
    }
    let rows = sorted_run_entries(&root)?;
    if rows.is_empty() {
        eprintln!("no runs found at {}. {hint}", root.display());
        return Ok(());
    }
    let mut table = crate::cli::new_table();
    table.set_header(vec!["RUN", "TESTS", "DATE", "ARCH"]);
    for (path, count, date, arch) in rows {
        let key = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let date_cell = date.unwrap_or_else(|| "-".to_string());
        // ARCH is sourced from `host.arch` on the run's first
        // sidecar; renders as `-` when no sidecar carries a host
        // (pre-host-context-landing archives, host-only test stubs
        // that never populate host) so the column reads consistently
        // with the DATE sentinel.
        let arch_cell = arch.unwrap_or_else(|| "-".to_string());
        table.add_row(vec![key, count.to_string(), date_cell, arch_cell]);
    }
    println!("{table}");
    Ok(())
}

/// One row produced by [`sorted_run_entries`] — see its doc for
/// the field-position contract. Aliased to keep the production
/// caller's `for (path, count, date, arch) in rows` destructure
/// readable while satisfying clippy's `type_complexity` lint
/// (a bare `Vec<(.., .., .., ..)>` over 4 elements crosses the
/// default complexity threshold).
type RunEntryRow = (std::path::PathBuf, usize, Option<String>, Option<String>);

/// Pure-IO sort + collection step of [`list_runs`]. Reads `root`,
/// filters to subdirectories, computes each entry's mtime + sidecar
/// pool, and returns the rows sorted by mtime DESCENDING (most
/// recent first), with `file_name` as a tiebreaker for equal mtimes
/// or both-unreadable mtime pairs. Entries whose mtime cannot be
/// read sort to the END of the returned vec.
///
/// Each row is a [`RunEntryRow`] = `(PathBuf, usize, Option<String>, Option<String>)`:
/// - `PathBuf`: the run-directory path (caller derives `file_name`
///   for display).
/// - `usize`: number of sidecars under the run dir (one level deep
///   per [`crate::test_support::collect_sidecars`]).
/// - `Option<String>`: earliest non-empty sidecar timestamp present
///   in the directory, or `None` when no sidecar carries a
///   non-empty timestamp (caller substitutes a display sentinel
///   like `"-"`).
/// - `Option<String>`: arch (e.g. `"x86_64"`, `"aarch64"`) from the
///   first sidecar that carries `host.arch`; `None` when no
///   sidecar in the dir has a `host`-populated arch (pre-
///   host-context-landing archives, host-only-stub paths that
///   never populate host). Caller substitutes a display sentinel
///   for the missing case the same way it does for date.
///
/// Factored out of [`list_runs`] so unit tests can exercise the
/// sort + row-shape contract without tee'ing stdout. Called once
/// per `cargo ktstr stats list` invocation; takes `&Path` so the
/// test harness can drive it against a tempdir-backed fixture
/// without mutating env vars.
fn sorted_run_entries(root: &std::path::Path) -> std::io::Result<Vec<RunEntryRow>> {
    use std::fs;
    use std::time::SystemTime;
    // Collect (entry, mtime) pairs so the sort key is computed once
    // per entry rather than per pairwise comparison. Entries whose
    // metadata or mtime cannot be read fall through with `None` and
    // sort to the end of the listing — the secondary `file_name`
    // tiebreaker keeps that group stable.
    let mut entries: Vec<(fs::DirEntry, Option<SystemTime>)> = fs::read_dir(root)?
        .filter_map(|e| e.ok())
        .filter(crate::test_support::is_run_directory)
        .map(|e| {
            let mtime = e.metadata().ok().and_then(|m| m.modified().ok());
            (e, mtime)
        })
        .collect();
    // Sort by mtime DESCENDING (newest first), with file_name as a
    // deterministic tiebreaker. `None` mtimes (metadata read
    // failure) sort to the END so good rows lead the listing —
    // `Reverse(Some(t))` orders larger times first, and `None`
    // wraps as `Reverse(None)` which compares as greater than
    // any `Reverse(Some(_))`, landing it at the end.
    entries.sort_by(|(a, a_mtime), (b, b_mtime)| {
        use std::cmp::Reverse;
        Reverse(*a_mtime)
            .cmp(&Reverse(*b_mtime))
            .then_with(|| a.file_name().cmp(&b.file_name()))
    });

    let rows = entries
        .into_iter()
        .map(|(entry, _)| {
            let path = entry.path();
            let sidecars = crate::test_support::collect_sidecars(&path);
            let count = sidecars.len();
            let date = sidecars
                .iter()
                .map(|s| s.timestamp.as_str())
                .filter(|t| !t.is_empty())
                .min()
                .map(|s| s.to_string());
            // Arch from the first sidecar that carries
            // `host.arch`. A run is on one machine so every
            // sidecar in the dir agrees on arch; taking the first
            // non-None reading keeps the lookup O(1) under the
            // common case (host-populated sidecar at the head
            // of `collect_sidecars`'s walk).
            let arch = sidecars
                .iter()
                .find_map(|s| s.host.as_ref().and_then(|h| h.arch.clone()));
            (path, count, date, arch)
        })
        .collect();
    Ok(rows)
}

/// Pool every sidecar under the runs root (or `dir` when set) and
/// emit the distinct values present on each filterable dimension.
///
/// Eight dimensions are reported: `kernel` (from
/// `SidecarResult::kernel_version`), `scheduler`, `topology`,
/// `work_type`, `commit` (from `SidecarResult::project_commit`),
/// `kernel_commit` (from `SidecarResult::kernel_commit`), `source`
/// (from `SidecarResult::run_source`), and `cpu_budget` (from
/// `SidecarResult::cpu_budget`). The dimension catalogue here matches
/// what `cargo ktstr stats compare` accepts as `--X` and `--a-X` /
/// `--b-X` filter flags — the command exists so an operator can answer
/// "what kernel versions are in the pool?" before crafting a compare
/// invocation. The JSON keys `commit` and `source` are the wire
/// contract; the corresponding per-side filter flags spell
/// `--project-commit` and `--run-source`.
///
/// `cpu_budget` is the sole NUMERIC dimension: its JSON value is an
/// array of integers (every other dimension is a string array), and
/// budget-0 skip rows (never-booted) are excluded — a non-empty pool
/// of only skips renders the `(all runs skipped — no budget recorded)`
/// sentinel rather than `null` / `unknown`.
///
/// `kernel_version`, `project_commit`, `kernel_commit`, and
/// `run_source` are `Option<String>` on the source sidecar;
/// absence is reported as a literal JSON `null` in the JSON
/// shape and the textual sentinel `unknown` in the table shape.
/// The set is sorted by the type's natural ordering (`BTreeSet`);
/// `None` collates before any populated value in `Option<String>`
/// ordering, so `null` / `unknown` always lands at the top of the
/// per-dimension listing.
///
/// `json=true` emits a JSON object keyed by dimension name with
/// arrays of values (with `null` interleaved for absent
/// `kernel`, `commit`, `kernel_commit`, or `source` entries —
/// the four optional dimensions); `json=false` emits a
/// per-dimension human-readable block with the values one per
/// line.
///
/// `dir` mirrors `compare_partitions` / `show_run_host` semantics:
/// when `Some(d)`, `d` replaces `runs_root()` as the pool source;
/// when `None`, `runs_root()` is used.
pub fn list_values(json: bool, dir: Option<&std::path::Path>) -> anyhow::Result<String> {
    use std::collections::BTreeSet;

    let (root, override_archive) = match dir {
        Some(d) => (d.to_path_buf(), true),
        None => (crate::test_support::runs_root(), false),
    };
    let mut pool = crate::test_support::collect_pool(&root);
    if override_archive {
        // `--dir` points at a non-default pool root. Stats tooling
        // treats those sidecars as `"archive"` regardless of the
        // tag they were written with — see
        // `apply_archive_source_override` for the rewrite contract.
        crate::test_support::apply_archive_source_override(&mut pool);
    }

    let mut kernels: BTreeSet<Option<String>> = BTreeSet::new();
    let mut project_commits: BTreeSet<Option<String>> = BTreeSet::new();
    let mut kernel_commits: BTreeSet<Option<String>> = BTreeSet::new();
    let mut run_sources: BTreeSet<Option<String>> = BTreeSet::new();
    let mut cpu_budgets: BTreeSet<u32> = BTreeSet::new();
    let mut schedulers: BTreeSet<String> = BTreeSet::new();
    let mut topologies: BTreeSet<String> = BTreeSet::new();
    let mut work_types: BTreeSet<String> = BTreeSet::new();

    for sc in &pool {
        kernels.insert(sc.kernel_version.clone());
        project_commits.insert(sc.project_commit.clone());
        kernel_commits.insert(sc.kernel_commit.clone());
        run_sources.insert(sc.run_source.clone());
        // 0 = skip rows (never booted); exclude — they carry no budget.
        if sc.cpu_budget != 0 {
            cpu_budgets.insert(sc.cpu_budget);
        }
        schedulers.insert(sc.scheduler.clone());
        topologies.insert(sc.topology.clone());
        work_types.insert(sc.work_type.clone());
    }

    if json {
        let kernels_json: Vec<serde_json::Value> = kernels
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        let project_commits_json: Vec<serde_json::Value> = project_commits
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        let kernel_commits_json: Vec<serde_json::Value> = kernel_commits
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        let run_sources_json: Vec<serde_json::Value> = run_sources
            .iter()
            .map(|opt| match opt {
                Some(s) => serde_json::Value::String(s.clone()),
                None => serde_json::Value::Null,
            })
            .collect();
        // JSON keys stay as `commit` / `source` — operator-visible
        // wire contract for `cargo ktstr stats list-values --json`
        // does not rename when the internal field/variable does.
        // Note: the per-side filter flags on `compare` spell as
        // `--project-commit` / `--run-source` (longer-form
        // disambiguating names), so the JSON keys here intentionally
        // diverge from the CLI flag names. The wire contract is the
        // shorter form because that's what every external consumer
        // (CI scripts, archive readers) has been parsing since the
        // sidecar format was first introduced.
        let payload = serde_json::json!({
            "kernel": kernels_json,
            "commit": project_commits_json,
            "kernel_commit": kernel_commits_json,
            "source": run_sources_json,
            "cpu_budget": cpu_budgets.iter().collect::<Vec<_>>(),
            "scheduler": schedulers.iter().collect::<Vec<_>>(),
            "topology": topologies.iter().collect::<Vec<_>>(),
            "work_type": work_types.iter().collect::<Vec<_>>(),
        });
        return serde_json::to_string_pretty(&payload)
            .map(|mut s| {
                s.push('\n');
                s
            })
            .map_err(|e| anyhow::anyhow!("serialize list-values JSON: {e}"));
    }

    let mut out = String::new();
    let render_opt_set = |out: &mut String, label: &str, set: &BTreeSet<Option<String>>| {
        out.push_str(label);
        out.push('\n');
        if set.is_empty() {
            out.push_str("  (no sidecars in pool)\n");
        } else {
            for opt in set {
                match opt {
                    Some(s) => {
                        out.push_str("  ");
                        out.push_str(s);
                        out.push('\n');
                    }
                    None => out.push_str("  unknown\n"),
                }
            }
        }
        out.push('\n');
    };
    let render_str_set = |out: &mut String, label: &str, set: &BTreeSet<String>| {
        out.push_str(label);
        out.push('\n');
        if set.is_empty() {
            out.push_str("  (no sidecars in pool)\n");
        } else {
            for s in set {
                out.push_str("  ");
                out.push_str(s);
                out.push('\n');
            }
        }
        out.push('\n');
    };
    render_opt_set(&mut out, "kernel:", &kernels);
    render_opt_set(&mut out, "commit:", &project_commits);
    render_opt_set(&mut out, "kernel_commit:", &kernel_commits);
    render_opt_set(&mut out, "source:", &run_sources);
    out.push_str("cpu_budget:\n");
    if cpu_budgets.is_empty() {
        // cpu_budgets excludes budget-0 skip rows, so an empty set on a
        // NON-empty pool means every sidecar was a skip — distinguish
        // that from a genuinely empty pool (the other dims always insert
        // a value per sidecar, so they never hit this).
        if pool.is_empty() {
            out.push_str("  (no sidecars in pool)\n");
        } else {
            out.push_str("  (all runs skipped — no budget recorded)\n");
        }
    } else {
        for b in &cpu_budgets {
            out.push_str(&format!("  {b}\n"));
        }
    }
    out.push('\n');
    render_str_set(&mut out, "scheduler:", &schedulers);
    render_str_set(&mut out, "topology:", &topologies);
    render_str_set(&mut out, "work_type:", &work_types);
    Ok(out)
}

/// One significant per-metric finding produced by [`compare_rows_by`].
///
/// `pairing_key` carries the dynamic identity the row pair joined
/// on — `scenario` plus every NON-slicing dimension's value. The
/// table renderer in [`compare_partitions`] decodes the key against
/// the slicing-dim list to produce a label like
/// `scenario/topology/work_type` (when topology + work_type are
/// pairing dims) or just `scenario` (when every other dim slices).
///
/// The `scenario` / `topology` / `work_type` fields carry the
/// matched row's values verbatim for legacy-shape consumers and
/// test fixtures that pre-date the dimensional-slicing refactor.
/// New code should read [`Finding::pairing_key`] directly so the
/// slicing-dim variation stays visible.
///
/// `metric` is the registry entry the comparison ran against;
/// consumers read polarity, display unit, and name through it
/// directly without re-looking up [`metric_def`].
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct Finding {
    pub pairing_key: PairingKey,
    pub scenario: String,
    pub topology: String,
    pub work_type: String,
    pub metric: &'static MetricDef,
    pub val_a: f64,
    pub val_b: f64,
    pub delta: f64,
    pub is_regression: bool,
}

/// Aggregate result of comparing two row sets via [`compare_rows_by`].
///
/// `regressions` and `improvements` count significant entries in
/// `findings`; `unchanged` counts metrics that fell below the dual
/// gate; `excluded_pairs` counts paired (scenario, topology, work_type)
/// row pairs where either side is not a real pass — `fail`,
/// `inconclusive`, and `skip` rows all route here. The field name
/// captures "excluded from regression math" rather than encoding any
/// of the three excluded states, because the per-side disposition
/// (which side, which state) is recoverable from the individual
/// `GauntletRow::is_*` accessors when the operator drills in.
/// `new_in_b`
/// counts B-side rows whose key has no match on the A side; the
/// converse is `removed_from_a`. The filter (when set) applies to
/// every counter, so excluded rows do not contribute.
///
/// `phase_deltas` and `unpaired_phases` carry the per-phase
/// comparison shape derived from
/// [`crate::assert::ScenarioStats::phases`] on each row pair. The
/// phase pass runs after the scalar-row pass via the same pairing
/// key; rows whose `phases` slice is empty on either side
/// contribute nothing here (single-phase scenarios skip the
/// per-phase view, falling back to the scalar findings already in
/// `findings`).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub(crate) struct CompareReport {
    pub regressions: u32,
    pub improvements: u32,
    pub unchanged: u32,
    pub excluded_pairs: u32,
    pub new_in_b: u32,
    pub removed_from_a: u32,
    pub findings: Vec<Finding>,
    pub phase_deltas: Vec<PhaseDeltaRow>,
    pub unpaired_phases: Vec<UnpairedPhaseRow>,
}

/// Which side of an A/B comparison a row belongs to. Typed surface
/// for the per-phase rows so new code does not propagate the
/// `"A"` / `"B"` string-literal pattern the scalar-finding path
/// uses (string siting cited at L4011-4012 etc., kept as-is for
/// the existing call sites).
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(crate) enum ComparePartition {
    A,
    B,
}

impl ComparePartition {
    /// Render the side as the same one-letter label
    /// `render_side_label` produces for the scalar table headers,
    /// so the new per-phase tables and the existing scalar table
    /// share the same operator-facing identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
        }
    }
}

/// One per-phase metric delta between matched A/B rows. Populated
/// when both rows carry a [`crate::assert::PhaseBucket`] at the
/// same `step_index` AND both buckets carry a value for the same
/// metric name. Generated by the parallel-pass after the
/// scalar-finding compare runs.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct PhaseDeltaRow {
    /// Same pairing key the scalar [`Finding`] uses, so the per-
    /// phase delta is unambiguously scoped to the matched row pair.
    pub pairing_key: PairingKey,
    /// 1-indexed phase encoding per the framework convention:
    /// `0` is BASELINE, `1..=N` are scenario Step ordinals.
    pub step_index: u16,
    /// Human-readable label mirroring
    /// [`crate::assert::PhaseBucket::label`] (`"BASELINE"` or
    /// `"Step[k-1]"`). The renderer prints this in the PHASE
    /// column.
    pub label: String,
    /// Registry entry the delta was computed against. Carries the
    /// `MetricKind` (Counter / Peak / Gauge / Timestamp) the
    /// phase aggregator used to fold the per-sample readings into
    /// the per-phase value, plus the `Polarity` the renderer uses
    /// to classify the delta direction.
    pub metric: &'static MetricDef,
    /// A-side phase-aggregated value (from
    /// `phase_bucket.metrics[metric.name]`).
    pub a: f64,
    /// B-side phase-aggregated value.
    pub b: f64,
    /// `b - a` for higher-is-worse metrics, `a - b` for
    /// lower-is-worse; matches the scalar [`Finding::delta`]
    /// polarity convention.
    pub delta: f64,
    /// `true` when the delta exceeds the dual-gate threshold in
    /// the regression direction (per metric polarity).
    pub is_regression: bool,
}

/// One per-phase bucket present on exactly one side of the A/B
/// comparison. Generated when a matched row pair has phase
/// coverage asymmetry — e.g. A ran a 3-Step scenario and B ran a
/// 4-Step version, so B's Step\[3\] has no A counterpart. The
/// renderer surfaces these in a dedicated "Phase Coverage
/// Asymmetry" section so the operator sees explicitly which side
/// is missing data; silently dropping them would mask the
/// scenario-shape difference.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct UnpairedPhaseRow {
    /// Which side carries the orphan bucket.
    pub side: ComparePartition,
    /// Same pairing key the matched [`PhaseDeltaRow`]s use.
    pub pairing_key: PairingKey,
    pub step_index: u16,
    pub label: String,
    /// Per-metric values carried verbatim from the orphan bucket's
    /// [`crate::assert::PhaseBucket::metrics`] map; the renderer
    /// prints them one-sided with `—` (em dash) in the absent
    /// column.
    pub metrics: std::collections::BTreeMap<String, f64>,
}

/// Per-metric threshold policy driving `compare_rows` /
/// `compare_partitions`.
///
/// Resolution priority for a given metric's relative significance
/// threshold, highest first:
///
/// 1. `per_metric_percent[metric_name]` — explicit override for
///    this metric.
/// 2. `default_percent` — uniform override across every metric
///    not listed in the map (equivalent to the old `--threshold N`
///    CLI flag).
/// 3. The metric's built-in `default_rel` from the `METRICS`
///    registry — the "no policy" fallback.
///
/// Values in the struct are stored as PERCENT (e.g. `10.0` meaning
/// 10%), NOT fractions. [`Self::rel_threshold`] does the `/100.0`
/// conversion so every caller inside `compare_rows` reads a
/// fraction without re-deriving the division.
///
/// Note on the registry-fallback branch: the `default_rel` field
/// on `MetricDef` is already a FRACTION (e.g. `0.25` for 25%),
/// not a percent. `rel_threshold` returns it verbatim — it
/// does NOT divide by 100. Only the override branches
/// (per-metric map, `default_percent`) do the percent-to-fraction
/// conversion because their inputs are percents. This asymmetry
/// is deliberate so callers supplying CLI/file-based overrides
/// work in human-intuitive percent units while the registry
/// defaults (which already ship in fraction form) pass through
/// unchanged.
///
/// The struct is `serde::Serialize` / `serde::Deserialize` so
/// `cargo ktstr stats compare --policy <path>` can load a
/// JSON-persisted policy file. Default construction produces an
/// empty policy that uses every registry default; [`Self::uniform`]
/// reproduces the old `--threshold N` behaviour without any
/// per-metric override plumbing at the call site.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ComparisonPolicy {
    /// Uniform override: when `Some(p)`, every metric whose name is
    /// NOT in [`Self::per_metric_percent`] uses `p / 100.0` as its
    /// relative threshold. `None` falls through to the registry
    /// `default_rel`. Stored as percent (e.g. `10.0` for 10%).
    pub default_percent: Option<f64>,
    /// Per-metric overrides keyed by metric name. Each value is a
    /// percent (e.g. `15.0` → 15%). An entry here takes precedence
    /// over both [`Self::default_percent`] and the registry
    /// `default_rel`.
    pub per_metric_percent: BTreeMap<String, f64>,
}

/// CLI-controlled rendering of the per-phase delta block in
/// `cargo ktstr stats compare`. Bundled as a struct so the
/// 5-flag clap surface threads through `compare_partitions` as
/// a single positional rather than five. Default value renders
/// every phase / every metric / every paired row — equivalent
/// to passing no phase flags.
///
/// The flags compose via AND on independent axes (block-level
/// suppression × phase-id × row-significance), with three
/// mutex constraints enforced at CLI parse time:
///
/// - `--no-phases` excludes every other phase flag (the whole
///   block is suppressed; refining what to render is a
///   contradiction).
/// - `--phases-only` excludes `--no-phases` (same reason).
/// - `--steps-only` excludes `--phase` (one of them collapses
///   to a single bucket; the other suppresses BASELINE — both
///   together are confused phrasing).
///
/// The 5 flags trigger renderer behaviour ONLY — the data
/// layer in `compare_rows_by` always emits the full set of
/// matched `PhaseDeltaRow`s and `UnpairedPhaseRow`s so
/// programmatic consumers of `CompareReport` see the
/// unfiltered surface. Filtering is render-time projection.
#[derive(Debug, Default, Clone)]
pub struct PhaseDisplayOptions {
    /// `--no-phases`: suppress the per-phase delta + unpaired
    /// tables entirely. The scalar findings table and footer
    /// render unchanged; the only effect is hiding the phase
    /// block (and the phase footer hint). Mutually exclusive
    /// with every other phase flag at CLI parse time.
    pub no_phases: bool,
    /// `--phases-only`: suppress the scalar findings table and
    /// the host-context delta; render ONLY the per-phase block.
    /// Useful for narrowing investigation to a phase regression
    /// when the scalar rollup is noise. Composes with
    /// `--steps-only`, `--phase`, and `--phase-threshold`.
    pub phases_only: bool,
    /// `--steps-only`: within the per-phase block, suppress
    /// the BASELINE bucket (`step_index == 0`); render only
    /// scenario Step buckets. Useful when the BASELINE settle
    /// window is dominated by scheduler startup transients.
    /// Mutually exclusive with `--phase`.
    pub steps_only: bool,
    /// `--phase <N>`: within the per-phase block, render only
    /// rows whose `step_index == N`. `0` selects BASELINE;
    /// `1..=N` selects scenario Step ordinals (1 → Step\[0\],
    /// 2 → Step\[1\], ...). Integer chosen over label so a label
    /// rename (`"Step[0]"` → `"Step:0"`) doesn't break operator
    /// CI invocations. Mutually exclusive with `--steps-only`.
    pub phase: Option<u16>,
    /// `--phase-threshold <PCT>`: render-side relative-delta
    /// gate for the per-phase pass. Suppresses paired rows
    /// where `|delta| / max(|a|, 1.0) < PCT / 100.0`. `0.0`
    /// shows every paired row; absence falls through to the
    /// registry's per-metric `default_rel`. Independent from
    /// the scalar `--threshold` — the two passes have separate
    /// filters so an operator can widen the phase view without
    /// widening the scalar view (the diagnostic "show me every
    /// per-phase delta but only load-bearing scalar findings"
    /// use case).
    pub phase_threshold: Option<f64>,
}

impl PhaseDisplayOptions {
    /// Resolve the per-phase relative threshold for a given
    /// metric. Returns the override fraction when
    /// `phase_threshold` is set, else falls through to the
    /// `ComparisonPolicy` resolution the scalar pass uses. The
    /// `metric_name` + `default_rel` shape mirrors
    /// [`ComparisonPolicy::rel_threshold`] so the two surfaces
    /// stay symmetric.
    pub fn rel_threshold(
        &self,
        policy: &ComparisonPolicy,
        metric_name: &str,
        default_rel: f64,
    ) -> f64 {
        match self.phase_threshold {
            Some(pct) => pct / 100.0,
            None => policy.rel_threshold(metric_name, default_rel),
        }
    }

    /// True when a phase row at the given `step_index` should
    /// render under the current display flags. Combines the two
    /// step-axis predicates (`--phase <N>` filter and
    /// `--steps-only` BASELINE-suppressor) into a single
    /// row-level decision the renderer can apply uniformly
    /// across `PhaseDeltaRow` and `UnpairedPhaseRow` vecs.
    /// Returns `true` when no relevant flag is set (default
    /// path: every step renders).
    pub fn matches_phase(&self, step_index: u16) -> bool {
        if let Some(want) = self.phase
            && step_index != want
        {
            return false;
        }
        if self.steps_only && step_index == 0 {
            return false;
        }
        true
    }

    /// True when a [`PhaseDeltaRow`] passes the
    /// `--phase-threshold` relative-significance gate. Computes
    /// `|delta| / max(|a|, 1.0) >= phase_threshold / 100.0` —
    /// the `max(|a|, 1.0)` denominator floor prevents NaN from
    /// `a == 0.0` (the row that pairs a zero against any
    /// non-zero produces a delta of finite magnitude that
    /// should not divide by zero). Returns `true` when no
    /// flag is set (default path: every row passes; per the
    /// `--phase-threshold` clap doc — absence keeps every
    /// paired row in the rendered output).
    ///
    /// `pub(crate)` rather than `pub` because [`PhaseDeltaRow`]
    /// is `pub(crate)` — the row type is an internal renderer
    /// detail, not a public surface. External consumers reach
    /// per-row decisions through the rendered output, not by
    /// instantiating a `PhaseDeltaRow` themselves.
    pub(crate) fn passes_delta_threshold(&self, delta: &PhaseDeltaRow) -> bool {
        let Some(pct) = self.phase_threshold else {
            return true;
        };
        let denom = delta.a.abs().max(1.0);
        let rel = delta.delta.abs() / denom;
        rel >= pct / 100.0
    }
}

impl ComparisonPolicy {
    /// Empty policy — every metric uses its `METRICS` registry
    /// default. Equivalent to the old `--threshold None` CLI path.
    pub fn new() -> Self {
        Self::default()
    }

    /// Uniform override: every metric uses `percent / 100.0`.
    /// Mirrors the old `--threshold N` CLI behaviour; the CLI
    /// dispatch at `cargo-ktstr stats compare --threshold N`
    /// constructs a policy via this constructor.
    pub fn uniform(percent: f64) -> Self {
        Self {
            default_percent: Some(percent),
            per_metric_percent: BTreeMap::new(),
        }
    }

    /// Load a JSON-persisted policy from a file. Errors propagate
    /// the read / parse reason as an `anyhow::Error` with the file
    /// path in the context chain so a malformed `--policy path.json`
    /// surfaces an actionable message rather than a generic
    /// "invalid JSON."
    ///
    /// Validates after parsing via [`Self::validate`]: rejects
    /// negative thresholds (a misconfigured 10 vs -10 would
    /// invert the dual-gate logic at the `.abs() >= rel_thresh`
    /// check and silently classify every metric as significant)
    /// and rejects per-metric keys not registered in `METRICS`
    /// (a typo like `"wrost_spread"` would otherwise be silently
    /// ignored — the key simply never matches during resolution
    /// and the metric falls through to `default_percent`).
    pub fn load_json(path: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("read comparison policy from {}", path.display()))?;
        let policy: ComparisonPolicy = serde_json::from_str(&data)
            .with_context(|| format!("parse comparison policy from {}", path.display()))?;
        policy
            .validate()
            .with_context(|| format!("validate comparison policy from {}", path.display()))?;
        Ok(policy)
    }

    /// Structural validation separate from parsing so both the
    /// `load_json` path and programmatic constructors (after
    /// [`Self::uniform`] with a user-supplied percent) can share
    /// one set of invariants without re-implementing checks at
    /// each call site. Called automatically by [`Self::load_json`];
    /// CLI dispatch should call it after constructing via
    /// [`Self::uniform`] to catch `--threshold -10` at the
    /// entry point rather than deep inside `compare_rows` where
    /// the dual-gate math silently misbehaves.
    ///
    /// Rejects:
    /// - Negative `default_percent` (nonsensical — thresholds are
    ///   absolute-value comparisons).
    /// - Negative entries in `per_metric_percent`.
    /// - Per-metric keys not in the `METRICS` registry (silent
    ///   typos would otherwise fall through to `default_percent`
    ///   unnoticed).
    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(p) = self.default_percent
            && p < 0.0
        {
            anyhow::bail!(
                "ComparisonPolicy: default_percent must be non-negative; got {p}. \
                 Thresholds are absolute-value comparisons — a negative value \
                 would invert the dual-gate logic and silently classify every \
                 delta as significant."
            );
        }
        for (name, p) in &self.per_metric_percent {
            if !METRICS.iter().any(|m| m.name == name) {
                let known: Vec<&str> = METRICS.iter().map(|m| m.name).collect();
                anyhow::bail!(
                    "ComparisonPolicy: per_metric_percent contains unknown \
                     metric `{name}`. A typo in the key would silently fall \
                     through to default_percent. Registered metrics: {}",
                    known.join(", "),
                );
            }
            if *p < 0.0 {
                anyhow::bail!(
                    "ComparisonPolicy: per_metric_percent[{name:?}] must be \
                     non-negative; got {p}",
                );
            }
        }
        Ok(())
    }

    /// Resolve the mutually-exclusive `--threshold` / `--policy` CLI
    /// pair into a policy: `--threshold N` is sugar for a uniform N%
    /// default (validated for sign); `--policy PATH` loads a
    /// per-metric JSON policy; neither falls through to the registry
    /// defaults. Shared by every subcommand that accepts the pair
    /// (`stats compare`, `perf-delta`) so the resolution rules — and
    /// the "exactly one of the two" contract — live in one place.
    ///
    /// Both flags set is rejected with an error. At the CLI call
    /// sites clap `conflicts_with` makes that unreachable, but this is
    /// a library entry point and must not panic on its inputs; the
    /// error is the defence-in-depth backstop.
    pub fn from_cli_flags(
        threshold: Option<f64>,
        policy: Option<&std::path::Path>,
    ) -> anyhow::Result<Self> {
        match (threshold, policy) {
            (Some(t), None) => {
                let p = Self::uniform(t);
                p.validate()?;
                Ok(p)
            }
            (None, Some(path)) => Self::load_json(path),
            (None, None) => Ok(Self::default()),
            (Some(_), Some(_)) => anyhow::bail!(
                "--threshold and --policy are mutually exclusive; use --policy \
                 for per-metric overrides"
            ),
        }
    }

    /// Resolve the relative threshold (as a fraction, e.g. `0.10`
    /// for 10%) for `metric_name` with `default_rel` as the
    /// registry-level fallback. Handles the percent→fraction
    /// conversion so `compare_rows_by` does not need to re-derive
    /// `p / 100.0` at every call site.
    pub fn rel_threshold(&self, metric_name: &str, default_rel: f64) -> f64 {
        if let Some(p) = self.per_metric_percent.get(metric_name) {
            p / 100.0
        } else if let Some(p) = self.default_percent {
            p / 100.0
        } else {
            default_rel
        }
    }
}

/// Compare two row sets metric-by-metric, parametrised on
/// `pairing_dims`.
///
/// Pure function: no I/O, no globals. Two rows pair iff their
/// [`PairingKey`] (scenario + every value for each dimension in
/// `pairing_dims`) is equal — this is the dimensional-slicing
/// pipeline's join primitive, with slicing dims EXCLUDED from
/// `pairing_dims` so rows on the A/B sides that differ on those
/// dims still pair as long as they agree on every non-slicing
/// dim. When `filter` is `Some(s)`, a row is included only if
/// `s` appears as a substring of the joined `"scenario topology
/// scheduler work_type"` string. The scheduler is
/// searchable via the substring filter but is not part of the
/// pairing key by default (only when `Dimension::Scheduler` is
/// in `pairing_dims`), so the same scenario+topology+work_type
/// pair compares correctly across different scheduler binaries
/// when the filter does not constrain it.
///
/// Row-pair accounting:
/// - B-side rows with no A-side match are counted in `new_in_b`.
/// - A-side rows with no B-side match are counted in `removed_from_a`
///   (a separate pass over `rows_a`).
/// - Paired rows where either side has `passed=false` are dropped
///   from the regression math and counted in `excluded_pairs`: a
///   failed scenario's metrics reflect the failure mode (short run,
///   stalled workload, missing samples), not the scheduler's
///   behavior.
///
/// The filter (when set) applies to every counter -- excluded rows
/// never reach the matching, pass, or metric stages.
///
/// `policy` carries the comparison thresholds. See
/// [`ComparisonPolicy`] for the resolution rules — per-metric
/// override → `default_percent` → registry `default_rel`. The
/// absolute gate always uses the metric's `default_abs`. A delta
/// must clear both gates to count as significant.
pub(crate) fn compare_rows_by(
    rows_a: &[GauntletRow],
    rows_b: &[GauntletRow],
    pairing_dims: &[Dimension],
    filter: Option<&str>,
    policy: &ComparisonPolicy,
) -> CompareReport {
    let mut report = CompareReport::default();

    // Build a HashMap<PairingKey, &GauntletRow> from rows_a once so
    // each row_b lookup is O(1) instead of O(rows_a). `or_insert_with`
    // preserves first-match semantics from the prior `rows_a.iter().find()`
    // call: on the rare path where two A-side rows share a key (the
    // averaging path produces unique keys; the `--no-average` path
    // bails earlier via `check_no_duplicate_pairing_keys`), the
    // earlier-iterated row wins.
    let mut a_by_key: HashMap<PairingKey, &GauntletRow> = HashMap::with_capacity(rows_a.len());
    for row_a in rows_a {
        let key = PairingKey::from_row(row_a, pairing_dims);
        a_by_key.entry(key).or_insert(row_a);
    }

    // Hoist the per-metric relative threshold out of the row×metric
    // loop. `policy.rel_threshold(m.name, m.default_rel)` is a pure
    // function of the metric — recomputing it for every row pair was
    // O(rows_b × METRICS) BTreeMap probes for nothing.
    let rel_thresholds: Vec<f64> = METRICS
        .iter()
        .map(|m| policy.rel_threshold(m.name, m.default_rel))
        .collect();

    for row_b in rows_b {
        // Dynamic pairing key: scenario + every NON-slicing
        // dimension's value. Two rows pair iff their dynamic keys
        // match.
        let key_b = PairingKey::from_row(row_b, pairing_dims);
        if let Some(f) = filter {
            // Substring filter joins all identity-bearing fields —
            // including the SLICING dim values — so an operator
            // can narrow by any visible field via `-E`.
            let joined = format!(
                "{} {} {} {}",
                row_b.scenario, row_b.topology, row_b.scheduler, row_b.work_type,
            );
            if !joined.contains(f) {
                continue;
            }
        }
        let Some(&row_a) = a_by_key.get(&key_b) else {
            report.new_in_b += 1;
            continue;
        };

        // Drop from regression math when either side is a skip,
        // inconclusive, or failure. Skips carry no executed metrics
        // (the run didn't happen); inconclusive runs ran but lacked
        // signal to evaluate (zero-denominator ratio gate); failures
        // carry telemetry dominated by the failure mode (short run,
        // stalled workload), not the scheduler's behavior —
        // comparing any of these against a real run produces
        // meaningless deltas.
        if row_a.is_fail()
            || row_b.is_fail()
            || row_a.is_inconclusive()
            || row_b.is_inconclusive()
            || row_a.is_skip()
            || row_b.is_skip()
        {
            report.excluded_pairs += 1;
            continue;
        }

        for (i, m) in METRICS.iter().enumerate() {
            // Rate components are internal plumbing — suppressed from compare
            // output (they remain in storage for the cross-run re-pool).
            if is_render_suppressed_component(m.name) {
                continue;
            }
            let val_a = m.read(row_a).unwrap_or(0.0);
            let val_b = m.read(row_b).unwrap_or(0.0);
            if val_a.abs() < f64::EPSILON && val_b.abs() < f64::EPSILON {
                continue;
            }

            let rel_thresh = rel_thresholds[i];

            let delta = val_b - val_a;
            let rel_delta = if val_a.abs() > f64::EPSILON {
                (delta / val_a).abs()
            } else {
                0.0
            };

            if delta.abs() < m.default_abs || rel_delta < rel_thresh {
                report.unchanged += 1;
                continue;
            }

            let is_regression = if m.higher_is_worse() {
                delta > 0.0
            } else {
                delta < 0.0
            };
            if is_regression {
                report.regressions += 1;
            } else {
                report.improvements += 1;
            }
            report.findings.push(Finding {
                pairing_key: key_b.clone(),
                scenario: row_b.scenario.clone(),
                topology: row_b.topology.clone(),
                work_type: row_b.work_type.clone(),
                metric: m,
                val_a,
                val_b,
                delta,
                is_regression,
            });
        }

        // Per-phase delta pass — runs on every paired row pair
        // alongside the scalar findings above. Walks the union of
        // step_index keys from `row_a.phases` and `row_b.phases`
        // and emits one [`PhaseDeltaRow`] per matched
        // (step_index, metric_name) pair where both sides carry
        // a value, or one [`UnpairedPhaseRow`] per side-only
        // step_index. Rows whose `phases` slice is empty on
        // either side contribute nothing here — single-phase
        // scenarios skip the per-phase view entirely without
        // emitting orphan UnpairedPhaseRows (an empty A-side
        // against a populated B-side would otherwise flood the
        // unpaired section for every B phase). The early-skip
        // matches the "Empty when scenario produced no periodic
        // captures" semantic on `ScenarioStats.phases`.
        if !row_a.phases.is_empty() && !row_b.phases.is_empty() {
            let a_by_step: std::collections::BTreeMap<u16, &crate::assert::PhaseBucket> =
                row_a.phases.iter().map(|p| (p.step_index, p)).collect();
            let b_by_step: std::collections::BTreeMap<u16, &crate::assert::PhaseBucket> =
                row_b.phases.iter().map(|p| (p.step_index, p)).collect();
            let union: std::collections::BTreeSet<u16> =
                a_by_step.keys().chain(b_by_step.keys()).copied().collect();
            for step_index in union {
                match (a_by_step.get(&step_index), b_by_step.get(&step_index)) {
                    (Some(pa), Some(pb)) => {
                        // Matched phase on both sides — emit a
                        // PhaseDeltaRow per metric_name present on
                        // BOTH sides. A name on only one side
                        // surfaces as an absent entry via the
                        // sentinel-free `PhaseBucket::get` contract;
                        // the renderer does not invent a synthetic
                        // delta for it.
                        //
                        // `is_regression` honors the same dual-gate
                        // the scalar pass applies inside its
                        // per-metric loop (search for `default_abs <`
                        // in `compare_rows_by` above): a row whose
                        // `|delta| < default_abs` OR whose
                        // `rel_delta < policy.rel_threshold` is
                        // classified `is_regression = false` even
                        // when the direction matches `polarity`.
                        // This mirrors the scalar `unchanged`
                        // semantic so a sub-threshold per-phase
                        // delta (e.g. `+0.1 ms` on a 10-ms-default
                        // gate) does not produce a false-positive
                        // REGRESSION verdict in the rendered table.
                        // The row is still emitted into
                        // `phase_deltas` so programmatic consumers
                        // of `CompareReport.phase_deltas` see every
                        // paired comparison; the filter is on the
                        // classification only.
                        for (metric_name, &val_a) in &pa.metrics {
                            // Suppress Rate components from the per-phase view
                            // too (they stay in PhaseBucket.metrics for the
                            // re-pool; only the rendered delta is dropped).
                            if is_render_suppressed_component(metric_name) {
                                continue;
                            }
                            let Some(&val_b) = pb.metrics.get(metric_name) else {
                                continue;
                            };
                            let Some(metric_def) = metric_def(metric_name) else {
                                continue;
                            };
                            let delta = val_b - val_a;
                            let rel_thresh =
                                policy.rel_threshold(metric_def.name, metric_def.default_rel);
                            let rel_delta = if val_a.abs() > f64::EPSILON {
                                (delta / val_a).abs()
                            } else {
                                0.0
                            };
                            let below_dual_gate =
                                delta.abs() < metric_def.default_abs || rel_delta < rel_thresh;
                            let is_regression = if below_dual_gate {
                                false
                            } else if metric_def.higher_is_worse() {
                                delta > 0.0
                            } else {
                                delta < 0.0
                            };
                            report.phase_deltas.push(PhaseDeltaRow {
                                pairing_key: key_b.clone(),
                                step_index,
                                label: pa.label.clone(),
                                metric: metric_def,
                                a: val_a,
                                b: val_b,
                                delta,
                                is_regression,
                            });
                        }
                    }
                    (Some(orphan), None) => {
                        report.unpaired_phases.push(UnpairedPhaseRow {
                            side: ComparePartition::A,
                            pairing_key: key_b.clone(),
                            step_index,
                            label: orphan.label.clone(),
                            metrics: metrics_without_suppressed(&orphan.metrics),
                        });
                    }
                    (None, Some(orphan)) => {
                        report.unpaired_phases.push(UnpairedPhaseRow {
                            side: ComparePartition::B,
                            pairing_key: key_b.clone(),
                            step_index,
                            label: orphan.label.clone(),
                            metrics: metrics_without_suppressed(&orphan.metrics),
                        });
                    }
                    (None, None) => {
                        unreachable!("step_index taken from union of a_by_step / b_by_step keys")
                    }
                }
            }
        }
    }

    // Second pass: A-side rows whose key has no match on the B side.
    // Filter applies here too, so rows excluded by the filter never
    // count as removed. Build a HashSet<PairingKey> from rows_b once
    // so the existence check is O(1) per row_a; rows_b are inserted
    // unfiltered to preserve prior behaviour where a row_b that fails
    // the substring filter still suppresses a same-key row_a's
    // removed_from_a increment (the substring filter compares against
    // identity-bearing fields including slicing dims, so two rows
    // sharing a pairing key can disagree on filter membership).
    let b_keys: HashSet<PairingKey> = rows_b
        .iter()
        .map(|r| PairingKey::from_row(r, pairing_dims))
        .collect();
    for row_a in rows_a {
        let key_a = PairingKey::from_row(row_a, pairing_dims);
        if let Some(f) = filter {
            let joined = format!(
                "{} {} {} {}",
                row_a.scenario, row_a.topology, row_a.scheduler, row_a.work_type,
            );
            if !joined.contains(f) {
                continue;
            }
        }
        if !b_keys.contains(&key_a) {
            report.removed_from_a += 1;
        }
    }

    report
}

/// Emit a stderr warning naming any `-dirty` commit values present
/// in the partitioned rows so the operator knows the comparison
/// includes builds whose source tree may not match the recorded
/// HEAD.
///
/// Scans `commit` (project HEAD) and `kernel_commit` (kernel source
/// tree HEAD) on both sides' rows, dedupes the surviving values,
/// and emits one warning block listing each distinct dirty value
/// per dimension. Emits at most one block — silent when no row
/// carries a `-dirty` suffix on either dimension.
///
/// Dirty runs reuse the same sidecar filename as their clean HEAD
/// (the variant hash excludes `commit` / `kernel_commit` per
/// `crate::test_support::sidecar`), so re-running the same test
/// from a dirty tree overwrites the previous record. The warning
/// surfaces this so an operator can decide whether to commit the
/// working tree before re-running for a reproducible comparison.
///
/// Splits collection from emission via [`render_dirty_warning`] so
/// unit tests can pin the rendered text without trapping `stderr`.
fn warn_on_dirty_builds(rows_a: &[GauntletRow], rows_b: &[GauntletRow]) {
    if let Some(text) = render_dirty_warning(rows_a, rows_b) {
        eprint!("{text}");
    }
}

/// Emit the CPU-budget hazard warning for a comparison, if any.
/// Pure-render half is [`render_overcommit_warning`]; this only
/// `eprint!`s it, mirroring [`warn_on_dirty_builds`].
fn warn_on_overcommit(rows_a: &[GauntletRow], rows_b: &[GauntletRow], pairing_dims: &[Dimension]) {
    if let Some(text) = render_overcommit_warning(rows_a, rows_b, pairing_dims) {
        eprint!("{text}");
    }
}

/// Build the CPU-budget hazard warning from the filtered compare
/// sides, or `None` when neither hazard is present.
///
/// Two independent hazards, both read from [`GauntletRow::cpu_budget`]
/// / [`GauntletRow::vcpus`] — the consumers that make those fields
/// load-bearing on the compare path:
///
/// - OVERCOMMIT (`cpu_budget < vcpus`): the host time-sliced that
///   run's vCPU threads, so its wake-latency / off-CPU / run-delay
///   timing metrics are host-contention artifacts, not scheduler
///   signal (see [`crate::vmm::host_topology::overcommit_warning`]).
///   Always flagged when present on either side: comparing raw timing
///   from an overcommitted run is the silent-wrong-answer the budget
///   stamp exists to surface.
/// - MIXED BUDGET: a single pairing group on a side holds more than
///   one distinct non-skip budget. [`group_and_average_by`] folds rows
///   that share a full [`PairingKey`], so this is exactly the set
///   `--average` would average together across budgets. It only arises
///   when [`Dimension::CpuBudget`] is NOT a pairing dim (the operator
///   sliced on cpu-budget, dropping it from the key); when it IS a
///   pairing dim, each budget keys its own group and is never folded.
///   Detection is per pairing group, NOT side-wide: two rows of
///   different scenarios (or any differing pairing dim) carry different
///   keys and never average, so a side merely spanning budgets across
///   distinct groups is not flagged.
///
/// Skip rows (budget 0 -> `None` in [`sidecar_to_row`]) carry no
/// budget identity and are ignored by both checks. Split from
/// emission so a unit test pins the text and the `None`-when-clean
/// polarity without trapping stderr, mirroring [`render_dirty_warning`].
fn render_overcommit_warning(
    rows_a: &[GauntletRow],
    rows_b: &[GauntletRow],
    pairing_dims: &[Dimension],
) -> Option<String> {
    use std::collections::BTreeSet;
    use std::fmt::Write;

    // Side-wide: the distinct overcommitted (budget, vcpus) pairs.
    let overcommitted = |rows: &[GauntletRow]| -> BTreeSet<(u32, u32)> {
        let mut over = BTreeSet::new();
        for r in rows {
            if let (Some(b), Some(v)) = (r.cpu_budget, r.vcpus)
                && b < v
            {
                over.insert((b, v));
            }
        }
        over
    };

    // Per pairing group: the union of budgets across groups that hold
    // >1 distinct budget — exactly the budgets `--average` folds into
    // one mean. Empty when CpuBudget is a pairing dim (each budget keys
    // its own group, so no group ever holds two).
    let cpu_budget_is_pairing = pairing_dims.contains(&Dimension::CpuBudget);
    let mixed_folded = |rows: &[GauntletRow]| -> BTreeSet<u32> {
        let mut folded = BTreeSet::new();
        if cpu_budget_is_pairing {
            return folded;
        }
        let mut by_key: std::collections::HashMap<PairingKey, BTreeSet<u32>> =
            std::collections::HashMap::new();
        for r in rows {
            if let Some(b) = r.cpu_budget {
                by_key
                    .entry(PairingKey::from_row(r, pairing_dims))
                    .or_default()
                    .insert(b);
            }
        }
        for budgets in by_key.values() {
            if budgets.len() > 1 {
                folded.extend(budgets.iter().copied());
            }
        }
        folded
    };

    let over_a = overcommitted(rows_a);
    let over_b = overcommitted(rows_b);
    let mixed_a = mixed_folded(rows_a);
    let mixed_b = mixed_folded(rows_b);

    if over_a.is_empty() && over_b.is_empty() && mixed_a.is_empty() && mixed_b.is_empty() {
        return None;
    }

    let any_overcommit = !over_a.is_empty() || !over_b.is_empty();
    let mut out = String::new();
    if any_overcommit {
        // Host time-slicing actually occurred -> raw timing is confounded.
        let _ = writeln!(
            out,
            "ktstr: WARNING: CPU-budget hazard in this comparison — a run was \
             host-overcommitted, so its guest-scheduler timing metrics \
             (wake-latency / off-CPU / run-delay) are host-contention-confounded. \
             Compare the overcommit-invariant worst_iterations_per_cpu_sec metric \
             (`stats compare --metric worst_iterations_per_cpu_sec`) instead of raw \
             timing."
        );
    } else {
        // Mixed budgets with NO overcommit: no host contention, the hazard is
        // collapsing two different measurement conditions into one number.
        let _ = writeln!(
            out,
            "ktstr: WARNING: CPU-budget hazard in this comparison — runs of \
             different CPU budgets share a pairing group, mixing two measurement \
             conditions. Slice with --cpu-budget, or compare the budget-invariant \
             worst_iterations_per_cpu_sec metric."
        );
    }
    let mut emit_side = |label: &str, over: &BTreeSet<(u32, u32)>, mixed: &BTreeSet<u32>| {
        if !over.is_empty() {
            let list = over
                .iter()
                .map(|(b, v)| format!("{b}/{v}"))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "  side {label}: host-overcommitted run(s) [budget/vcpus]: {list}"
            );
        }
        if !mixed.is_empty() {
            let list = mixed
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "  side {label}: CPU budgets [{list}] share a pairing group — \
                 --average folds them into one mean (--no-average rejects them as \
                 duplicate keys); slice with --cpu-budget so cross-budget runs are \
                 not compared under one key"
            );
        }
    };
    emit_side("A", &over_a, &mixed_a);
    emit_side("B", &over_b, &mixed_b);
    Some(out)
}

/// Build the dirty-builds warning block from row data.
///
/// Returns `None` when no row on either side carries a `-dirty`
/// suffix on either `commit` or `kernel_commit`. Otherwise returns
/// the full multi-line warning text — the body emitted to stderr by
/// [`warn_on_dirty_builds`] — terminated with a trailing newline so
/// the caller can `eprint!` it without further formatting.
///
/// Dimensions render in fixed order ("kernel source" before
/// "project") so the same dirty hashes always produce byte-identical
/// output across runs; values within each dimension are
/// `BTreeSet`-deduped so multiple rows sharing one dirty hash list
/// it once, and multiple distinct dirty hashes on one dimension list
/// in lex order.
fn render_dirty_warning(rows_a: &[GauntletRow], rows_b: &[GauntletRow]) -> Option<String> {
    use std::collections::BTreeSet;
    use std::fmt::Write;

    let mut dirty_kernel: BTreeSet<&str> = BTreeSet::new();
    let mut dirty_project: BTreeSet<&str> = BTreeSet::new();
    for row in rows_a.iter().chain(rows_b.iter()) {
        // `ends_with` matches the producer contract: `detect_kernel_commit`
        // and `detect_project_commit` (sidecar.rs:851, :983) append
        // `-dirty` as a SUFFIX to the 7-char hex via
        // `format!("{short_hash}-dirty")`, so the dirty marker is
        // always tail-positioned. `contains` would also match a
        // hex hash that legitimately contains the substring `-dirty`
        // somewhere in the middle (impossible for the current
        // 7-char hex prefix, but a future commit-ish format change
        // would let a non-dirty value flag itself dirty under
        // `contains`).
        if let Some(c) = row.kernel_commit.as_deref()
            && c.ends_with("-dirty")
        {
            dirty_kernel.insert(c);
        }
        if let Some(c) = row.commit.as_deref()
            && c.ends_with("-dirty")
        {
            dirty_project.insert(c);
        }
    }

    if dirty_kernel.is_empty() && dirty_project.is_empty() {
        return None;
    }

    let mut out = String::new();
    writeln!(out, "warning: comparison includes dirty builds:").unwrap();
    for v in &dirty_kernel {
        writeln!(
            out,
            "  - kernel source: {v} (working tree may have changed since this run)"
        )
        .unwrap();
    }
    for v in &dirty_project {
        writeln!(
            out,
            "  - project: {v} (working tree may have changed since this run)"
        )
        .unwrap();
    }
    writeln!(
        out,
        "  Dirty runs overwrite previous results with the same HEAD."
    )
    .unwrap();
    writeln!(out, "  Commit changes for reproducible-ish comparisons.").unwrap();
    Some(out)
}

/// Render the actionable bail message emitted when one side's filter
/// matches zero sidecars in the pool.
///
/// Beyond the generic "check filters / run `cargo ktstr stats list`"
/// redirect, this helper inspects WHY the filter matched nothing and
/// adds three operator-actionable hints when applicable:
///
/// 1. **Dirty-form hint**: when the user passed
///    `--project-commit X` (or per-side / kernel-commit equivalent)
///    and the pool contains a row whose `commit` (or `kernel_commit`)
///    is `X-dirty`, append "Did you mean `--project-commit X-dirty`?".
///    A clean-vs-dirty mismatch is the single most common cause of a
///    false-zero on the commit dims — `detect_project_commit` /
///    `detect_kernel_commit` append `-dirty` whenever HEAD-vs-index
///    or index-vs-worktree changes are observed, so an operator who
///    expected `abcdef1` but the recorded value is `abcdef1-dirty`
///    sees no rows match without realizing why.
///
/// 2. **Unknown run-source hint**: when the user passed
///    `--run-source X` (or per-side equivalent) and `X` is NOT
///    among the distinct `run_source` values present in the pool,
///    append a hint listing the actual values seen. The schema is
///    deliberately extensible (`"benchmark"` and other future tags
///    are valid), so this is a hint rather than a hard validator —
///    but a typo (`--run-source loca` for `local`, or `--run-source CI`
///    for `ci` since the values are case-sensitive) is the most
///    common cause of a false-zero on the source dim, and listing
///    the distinct values present is more actionable than asking
///    the operator to consult the schema doc.
///
/// 3. **list-values redirect for commit dims**: when the user
///    populated any commit dimension (`project_commits` /
///    `kernel_commits`), suggest `cargo ktstr stats list-values`
///    specifically — that command emits the exact distinct values
///    present per dimension, which is more actionable than the
///    generic `stats list` which only shows top-level run keys.
///
/// `side` is `"A"` or `"B"` for diagnostic context. `filter` is the
/// per-side `RowFilter`. `rows` is the sidecar-derived row vec
/// (post-`sidecar_to_row` mapping, pre-filtering). `pool_len` is
/// the raw pool count for the "(N pooled)" diagnostic context.
fn zero_match_diagnostic(
    side: &str,
    filter: &RowFilter,
    rows: &[GauntletRow],
    pool_len: usize,
) -> String {
    let mut msg = format!(
        "stats compare: {side} side filter matched 0 sidecars in \
         pool ({pool_len} pooled). Check the per-side filters or \
         confirm the runs exist with `cargo ktstr stats list`."
    );

    // Dirty-form hint per commit dimension. Only fires when a
    // populated filter value's `-dirty` form is in the pool.
    let mut dirty_hints: Vec<String> = Vec::new();
    for want in &filter.project_commits {
        let dirty = format!("{want}-dirty");
        let found = rows
            .iter()
            .any(|r| r.commit.as_deref() == Some(dirty.as_str()));
        if found {
            dirty_hints.push(format!(
                "no rows match `--project-commit {want}` but `{dirty}` exists in the pool — \
                 did you mean `--project-commit {dirty}`?"
            ));
        }
    }
    for want in &filter.kernel_commits {
        let dirty = format!("{want}-dirty");
        let found = rows
            .iter()
            .any(|r| r.kernel_commit.as_deref() == Some(dirty.as_str()));
        if found {
            dirty_hints.push(format!(
                "no rows match `--kernel-commit {want}` but `{dirty}` exists in the pool — \
                 did you mean `--kernel-commit {dirty}`?"
            ));
        }
    }
    for hint in dirty_hints {
        msg.push_str("\nhint: ");
        msg.push_str(&hint);
    }

    // Unknown-run-source hint. Fires when a `--run-source X` value
    // is not present in the pool — typo / wrong casing is the most
    // common cause. Schema is intentionally extensible (operators
    // can write `"benchmark"` etc.), so this is a hint not a hard
    // validator: the bail still fires, the operator still sees the
    // distinct values present, and the producer side is free to
    // emit any tag.
    if !filter.run_sources.is_empty() {
        let pool_run_sources: std::collections::BTreeSet<&str> = rows
            .iter()
            .filter_map(|r| r.run_source.as_deref())
            .collect();
        let unknowns: Vec<&str> = filter
            .run_sources
            .iter()
            .map(String::as_str)
            .filter(|want| !pool_run_sources.contains(*want))
            .collect();
        if !unknowns.is_empty() {
            let mut present: Vec<&str> = pool_run_sources.iter().copied().collect();
            present.sort_unstable();
            let unknown_list = unknowns
                .iter()
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(", ");
            let present_list = if present.is_empty() {
                "(none — every row has `run_source: null`)".to_string()
            } else {
                present
                    .iter()
                    .map(|s| format!("`{s}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            msg.push_str(&format!(
                "\nhint: --run-source {unknown_list} not found in pool; \
                 distinct values present: {present_list}. Values are \
                 case-sensitive (`ci` ≠ `CI`)."
            ));
        }
    }

    // Unknown-cpu-budget hint. Mirrors the run_sources hint for the
    // numeric budget dimension: fires when a `--cpu-budget` value is
    // not among the budgets present in the pool (the budgets render
    // canonically as decimal via `cpu_budget.to_string()`, so a
    // non-canonical input like `032` lists as not-found against the
    // canonical present set). Skip rows (`cpu_budget == None`) carry no
    // budget and are excluded.
    if !filter.cpu_budgets.is_empty() {
        let pool_budgets: std::collections::BTreeSet<u32> =
            rows.iter().filter_map(|r| r.cpu_budget).collect();
        let present_strs: std::collections::BTreeSet<String> =
            pool_budgets.iter().map(|b| b.to_string()).collect();
        let unknowns: Vec<&str> = filter
            .cpu_budgets
            .iter()
            .map(String::as_str)
            .filter(|want| !present_strs.contains(*want))
            .collect();
        if !unknowns.is_empty() {
            let unknown_list = unknowns
                .iter()
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(", ");
            let present_list = if pool_budgets.is_empty() {
                "(none — every row is a skip with no recorded budget)".to_string()
            } else {
                pool_budgets
                    .iter()
                    .map(|b| format!("`{b}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            msg.push_str(&format!(
                "\nhint: --cpu-budget {unknown_list} not found in pool; \
                 distinct budgets present: {present_list}."
            ));
        }
    }

    // list-values redirect: only fires when the operator narrowed
    // on a commit dimension. Generic case (no commit filter) keeps
    // the existing `stats list` redirect at the top of the message
    // — `list-values` would emit a long per-dimension dump that
    // isn't more actionable than `stats list` for a kernel/scheduler
    // /topology miss.
    let touched_commit_dim =
        !filter.project_commits.is_empty() || !filter.kernel_commits.is_empty();
    if touched_commit_dim {
        msg.push_str(
            "\nhint: run `cargo ktstr stats list-values` to see every \
             distinct commit value present in the pool — the specific \
             value the filter expected may not have a sidecar yet, or \
             may differ from what was recorded by \
             `detect_project_commit` / `detect_kernel_commit`.",
        );
    }
    msg
}

/// Compare two filter-defined partitions of the sidecar pool and
/// report regressions across slicing dimensions.
///
/// `filter_a` and `filter_b` are the per-side row filters that
/// define the A/B contrast. The dimensions on which the two
/// filters DIFFER are the SLICING dimensions; the dimensions on
/// which they AGREE (or on which both are unconstrained) are the
/// PAIRING dimensions. Two rows pair across the A/B sides iff
/// their dynamic [`PairingKey`] (scenario plus every pairing-dim
/// value) is equal — so the comparison naturally ignores
/// differences on the slicing axes (those ARE the contrast) and
/// joins on everything else.
///
/// `dir` overrides the default `runs_root()` for pool collection.
/// Pass `Some(path)` to compare archived sidecar trees copied off
/// a CI host; pass `None` to walk `target/ktstr/` (or
/// `CARGO_TARGET_DIR/ktstr/`).
///
/// Validation:
/// - Empty slicing-dim set (every dimension is identical between
///   A and B): bail with "specify at least one --a-X / --b-X to
///   define what to compare". This includes the no-flags-at-all
///   case (both filters are the empty default).
/// - Identical effective filters with at least one slicing dim is
///   a contradiction caught by clap-level construction; the
///   downstream check is "every value in filter_a appears in
///   filter_b on the same dim and vice versa." We catch that as
///   "A and B select identical rows" — symmetric to the empty
///   case.
/// - More than one slicing dimension prints a warning to stderr
///   ("warning: slicing on N dimensions; results compress
///   multiple axes into a single A/B contrast") but does NOT
///   bail — multi-dim slicing is a deliberate feature for
///   comparing e.g. (kernel A + scheduler A) against (kernel B +
///   scheduler B).
///
/// `no_average = false` (the default) groups every matching
/// sidecar within each side by pairing key and averages the
/// metrics across the group. `no_average = true` keeps each
/// sidecar row distinct; if multiple rows on one side share the
/// same pairing key the function bails with an actionable
/// "duplicate pairing keys" error rather than picking one
/// arbitrarily.
///
/// Returns 0 on no regressions, 1 if regressions detected.
pub fn compare_partitions(
    filter_a: &RowFilter,
    filter_b: &RowFilter,
    filter: Option<&str>,
    policy: &ComparisonPolicy,
    dir: Option<&std::path::Path>,
    no_average: bool,
    phase_opts: &PhaseDisplayOptions,
) -> anyhow::Result<i32> {
    // Validation gate 1: there must be at least one dimension
    // on which filter_a differs from filter_b — otherwise the
    // operator hasn't expressed a contrast and the function has
    // nothing to compare. Empty slicing dims OR identical filters
    // are both rejected here with actionable diagnostics so the
    // user knows which knob to turn.
    let slicing_dims = derive_slicing_dims(filter_a, filter_b);
    if slicing_dims.is_empty() {
        anyhow::bail!(
            "stats compare: A and B select identical rows. \
             Specify at least one per-side filter (e.g. \
             --a-kernel 6.14 --b-kernel 6.15) to define what \
             dimension separates the two sides."
        );
    }

    // Validation gate 2: warn (not error) when slicing on
    // multiple dimensions. The result is still well-defined —
    // the comparison joins on remaining pairing dims and
    // collapses the slicing-dim cross-product into a single
    // A/B contrast — but the operator is asking for a multi-axis
    // delta which is harder to interpret. The warning surfaces
    // the dim list so they can confirm the cohort shape.
    if slicing_dims.len() > 1 {
        let dim_names: Vec<&str> = slicing_dims.iter().map(|d| d.name()).collect();
        eprintln!(
            "warning: stats compare: slicing on {n} dimensions [{dims}]; \
             results compress multiple axes into a single A/B contrast.",
            n = slicing_dims.len(),
            dims = dim_names.join(", "),
        );
    }

    // Pairing dims = every dimension NOT in the slicing-dim set,
    // in canonical [`Dimension::ALL`] order. The dynamic key
    // shape `(scenario, *pairing_dims)` matches whatever
    // dimensions are currently NOT being contrasted across A
    // and B.
    let pairing_dims = Dimension::pairing_dims(&slicing_dims);

    // Pool every sidecar under the runs root (or the operator's
    // --dir override) and convert to rows. The full-scan cost
    // is acceptable for the single-comparison-per-session
    // workflow.
    //
    // `--dir`-loaded sidecars get their `source` field rewritten
    // to `"archive"` via `apply_archive_source_override` before
    // row conversion. The producer-side `"local"` / `"ci"`
    // distinction is meaningful on the host that wrote the
    // sidecars; once the files have been copied off, the only
    // useful classification is "this came from elsewhere", which
    // is what `--run-source archive` queries for. Operators who need
    // to retain the producer-side distinction read from the
    // default root (no `--dir`) so values pass through untouched.
    let (root, override_archive) = match dir {
        Some(d) => (d.to_path_buf(), true),
        None => (crate::test_support::runs_root(), false),
    };
    let mut pool = crate::test_support::collect_pool(&root);
    if override_archive {
        crate::test_support::apply_archive_source_override(&mut pool);
    }
    if pool.is_empty() {
        anyhow::bail!(
            "stats compare: no sidecar data found under {}. \
             Run `cargo ktstr test` to generate runs, or pass \
             --dir to point at an archived sidecar tree.",
            root.display(),
        );
    }
    let rows: Vec<GauntletRow> = pool.iter().map(sidecar_to_row).collect();

    // Partition: apply each side's filter to the same pool. A
    // row may match both sides (e.g. when scheduler is the
    // slicing dim and kernel is unconstrained on both, a row
    // whose `scheduler` is in `filter_a.schedulers` matches A
    // but NOT B unless `filter_b.schedulers` also contains it —
    // typically not when scheduler is the slicing axis).
    let rows_a = apply_row_filters(&rows, filter_a);
    let rows_b = apply_row_filters(&rows, filter_b);
    if rows_a.is_empty() {
        anyhow::bail!(
            "{}",
            zero_match_diagnostic("A", filter_a, &rows, pool.len()),
        );
    }
    if rows_b.is_empty() {
        anyhow::bail!(
            "{}",
            zero_match_diagnostic("B", filter_b, &rows, pool.len()),
        );
    }

    warn_on_dirty_builds(&rows_a, &rows_b);
    warn_on_overcommit(&rows_a, &rows_b, &pairing_dims);

    let pre_agg_a = rows_a.len();
    let pre_agg_b = rows_b.len();

    // Average by default: fold same-pairing-key rows on each
    // side into one mean row. `--no-average` keeps every
    // sidecar distinct but still rejects duplicate pairing keys
    // because compare_rows can't pair an A-row against multiple
    // B-rows with the same key.
    let (rows_a_for_compare, rows_b_for_compare, avg_a, avg_b) = if !no_average {
        let avg_a = group_and_average_by(&rows_a, &pairing_dims);
        let avg_b = group_and_average_by(&rows_b, &pairing_dims);
        let a_rows: Vec<GauntletRow> = avg_a.iter().map(|r| r.row.clone()).collect();
        let b_rows: Vec<GauntletRow> = avg_b.iter().map(|r| r.row.clone()).collect();
        (a_rows, b_rows, Some(avg_a), Some(avg_b))
    } else {
        // Detect duplicates manually so the error names the key
        // rather than letting compare_rows silently latch onto
        // the first match.
        check_no_duplicate_pairing_keys(&rows_a, &pairing_dims, "A")?;
        check_no_duplicate_pairing_keys(&rows_b, &pairing_dims, "B")?;
        (rows_a, rows_b, None, None)
    };

    let report = compare_rows_by(
        &rows_a_for_compare,
        &rows_b_for_compare,
        &pairing_dims,
        filter,
        policy,
    );

    // Side labels derive from the slicing dims' filter values.
    // Single slicing dim: e.g. "6.14.2" / "6.15.0". Multi: e.g.
    // "6.14.2:scx_rusty" / "6.15.0:scx_alpha". >3 values per dim:
    // collapse to "A"/"B" to keep column headers readable.
    let label_a = render_side_label(filter_a, &slicing_dims, "A");
    let label_b = render_side_label(filter_b, &slicing_dims, "B");

    // Header lines: name the slicing and pairing axes so the
    // operator can confirm the comparison shape at a glance.
    let slice_names: Vec<&str> = slicing_dims.iter().map(|d| d.name()).collect();
    let pair_names: Vec<&str> = pairing_dims.iter().map(|d| d.name()).collect();
    println!("slicing dimensions: {}", slice_names.join(", "));
    println!(
        "pairing on: scenario{}{}",
        if pair_names.is_empty() { "" } else { ", " },
        pair_names.join(", "),
    );

    if !no_average {
        println!(
            "{}",
            format_average_header(pre_agg_a, pre_agg_b, &label_a, &label_b)
        );
    }

    use comfy_table::{Cell, Color};
    // Scalar findings table — suppressed when the operator
    // passed `--phases-only` (they want the per-phase block
    // only). The scalar pre-aggregation already ran; this just
    // hides its render.
    if !phase_opts.phases_only {
        let mut table = crate::cli::new_table();
        table.set_header(vec![
            "TEST", "METRIC", &label_a, &label_b, "DELTA", "VERDICT",
        ]);
        for f in &report.findings {
            let (verdict_text, verdict_color) = if f.is_regression {
                ("REGRESSION", Color::Red)
            } else {
                ("improvement", Color::Green)
            };
            // PairingKey's first slot is scenario; subsequent slots
            // are the pairing-dim values in canonical order. Joining
            // with `/` produces a label whose shape mirrors the
            // pairing-dim count — so a comparison that pairs on
            // (topology, work_type) renders a `scenario/topology/work_type`
            // label, while a comparison that slices on most dims
            // renders a shorter identifier. The operator can always
            // cross-reference the "pairing on:" header line above to
            // see what each segment means.
            let label = f.pairing_key.0.join("/");
            table.add_row(vec![
                Cell::new(label),
                Cell::new(f.metric.name),
                Cell::new(format!("{:.2}", f.val_a)),
                Cell::new(format!("{:.2}", f.val_b)),
                Cell::new(format!("{:+.2}{}", f.delta, f.metric.display_unit)),
                Cell::new(verdict_text).fg(verdict_color),
            ]);
        }
        println!("{table}");
    }

    // Per-phase delta render. Activated when the parallel pass
    // populated either phase_deltas or unpaired_phases for the
    // current row-pair set AND `--no-phases` was not passed.
    // Single-phase scenarios (no periodic captures) leave both
    // vecs empty and the phase block is suppressed entirely.
    //
    // CLI filters compose by AND on independent axes:
    // - `--phase <N>` keeps only the named step_index
    // - `--steps-only` suppresses BASELINE (step_index == 0)
    // - `--phase-threshold <PCT>` filters paired rows whose
    //   `|delta| / max(|a|, 1.0)` is below `PCT / 100.0`
    //
    // Filtering is render-time projection — the underlying
    // CompareReport.phase_deltas / unpaired_phases vecs hold
    // the unfiltered data so programmatic consumers see every
    // paired row regardless of CLI flags.
    let render_phase_block = !phase_opts.no_phases
        && (!report.phase_deltas.is_empty() || !report.unpaired_phases.is_empty());
    if render_phase_block {
        let filtered_deltas: Vec<&PhaseDeltaRow> = report
            .phase_deltas
            .iter()
            .filter(|d| phase_opts.matches_phase(d.step_index))
            .filter(|d| phase_opts.passes_delta_threshold(d))
            .collect();
        let filtered_unpaired: Vec<&UnpairedPhaseRow> = report
            .unpaired_phases
            .iter()
            .filter(|u| phase_opts.matches_phase(u.step_index))
            .collect();
        // Capture filtered counts BEFORE moving `filtered_deltas`
        // into `sorted_deltas` below — the footer hint reads them
        // after the table rendering consumes the Vec.
        let filtered_delta_total = filtered_deltas.len();
        let filtered_delta_regressions = filtered_deltas.iter().filter(|d| d.is_regression).count();
        if !filtered_deltas.is_empty() || !filtered_unpaired.is_empty() {
            println!();
            println!("phase coverage:");
            if !filtered_deltas.is_empty() {
                let mut phase_table = crate::cli::new_table();
                phase_table.set_header(vec![
                    "PHASE", "TEST", "METRIC", &label_a, &label_b, "DELTA", "VERDICT",
                ]);
                // Sort by step_index ascending, then pairing key,
                // then metric name. step_index-first ordering matches
                // the operator-facing time order from BASELINE
                // through Step[N] so the reader scans top-down by
                // phase boundary; ties within a phase sort by row
                // pair then metric so the table is stable across
                // runs with identical input.
                let mut sorted_deltas = filtered_deltas;
                sorted_deltas.sort_by(|a, b| {
                    a.step_index
                        .cmp(&b.step_index)
                        .then_with(|| a.pairing_key.0.cmp(&b.pairing_key.0))
                        .then_with(|| a.metric.name.cmp(b.metric.name))
                });
                for d in sorted_deltas {
                    let (verdict_text, verdict_color) = if d.is_regression {
                        ("REGRESSION", Color::Red)
                    } else {
                        ("improvement", Color::Green)
                    };
                    let test_label = d.pairing_key.0.join("/");
                    let phase_cell = format!("{}: {}", d.step_index, d.label);
                    phase_table.add_row(vec![
                        Cell::new(phase_cell),
                        Cell::new(test_label),
                        Cell::new(d.metric.name),
                        Cell::new(format!("{:.2}", d.a)),
                        Cell::new(format!("{:.2}", d.b)),
                        Cell::new(format!("{:+.2}{}", d.delta, d.metric.display_unit)),
                        Cell::new(verdict_text).fg(verdict_color),
                    ]);
                }
                println!("{phase_table}");
            }
            if !filtered_unpaired.is_empty() {
                println!();
                println!("phase coverage asymmetry (one-sided phases):");
                let mut unpaired_table = crate::cli::new_table();
                unpaired_table.set_header(vec!["SIDE", "TEST", "PHASE", "METRIC", "VALUE"]);
                // Sort by step_index then side then pairing key then
                // metric name. Time-order (step_index first) reads
                // most naturally — the reader sees missing data in
                // the order it would have appeared during the
                // scenario, not grouped by which side is missing
                // (side grouping would force a mental flip-flop
                // across the paired rows above).
                let mut sorted_unpaired = filtered_unpaired;
                sorted_unpaired.sort_by(|a, b| {
                    a.step_index
                        .cmp(&b.step_index)
                        .then_with(|| a.side.as_str().cmp(b.side.as_str()))
                        .then_with(|| a.pairing_key.0.cmp(&b.pairing_key.0))
                });
                for u in sorted_unpaired {
                    let test_label = u.pairing_key.0.join("/");
                    let phase_cell = format!("{}: {}", u.step_index, u.label);
                    if u.metrics.is_empty() {
                        // Bucket present but no metrics — surface
                        // the empty shape rather than hiding it. The
                        // operator sees that the phase fired but
                        // produced no readable metric data on the
                        // single side it ran on, which is itself a
                        // signal. Two paths reach here: (1) capture
                        // landed but MetricDef::read_sample returned
                        // None for every registered metric on these
                        // samples; (2) the phase's only metrics were
                        // suppressed Rate components, which
                        // metrics_without_suppressed drops from the
                        // unpaired row.
                        unpaired_table.add_row(vec![
                            Cell::new(u.side.as_str()),
                            Cell::new(test_label),
                            Cell::new(phase_cell),
                            Cell::new("—"),
                            Cell::new("—"),
                        ]);
                    } else {
                        for (metric_name, &value) in &u.metrics {
                            unpaired_table.add_row(vec![
                                Cell::new(u.side.as_str()),
                                Cell::new(&test_label),
                                Cell::new(&phase_cell),
                                Cell::new(metric_name),
                                Cell::new(format!("{value:.2}")),
                            ]);
                        }
                    }
                }
                println!("{unpaired_table}");
            }
            // Operator hint surfaces only when the default-on
            // path is producing rows AND no filter flag was set —
            // a user who already passed `--phase`, `--steps-only`,
            // `--phase-threshold`, or `--phases-only` doesn't need
            // the discovery hint. `--no-phases` already
            // short-circuited the entire block above so it can't
            // reach here.
            let any_flag_set = phase_opts.phases_only
                || phase_opts.steps_only
                || phase_opts.phase.is_some()
                || phase_opts.phase_threshold.is_some();
            if !any_flag_set {
                println!(
                    "  phases: {filtered_delta_total} delta row(s) shown \
                     ({filtered_delta_regressions} regression{plural}). \
                     Filter with --phase N / --phases-only / --steps-only / \
                     --phase-threshold P / --no-phases.",
                    plural = if filtered_delta_regressions == 1 {
                        ""
                    } else {
                        "s"
                    },
                );
            }
        }
    }

    // Scalar summary block — regressions / improvements /
    // unchanged + skipped-failed + per-group pass counts +
    // new_in_b / removed_from_a. All four lines describe the
    // scalar findings table; suppress them under `--phases-only`
    // so the operator's "phase-block only" projection stays
    // pure (the phase block has its own footer hint above).
    if !phase_opts.phases_only {
        println!();
        println!(
            "summary: {} regressions, {} improvements, {} unchanged",
            report.regressions, report.improvements, report.unchanged,
        );
        if report.excluded_pairs > 0 {
            println!(
                "  {} pairing-key row pair(s) excluded from regression math because one \
                 or both sides did not pass (failed, inconclusive, or skipped)",
                report.excluded_pairs,
            );
        }
        if let (Some(avg_a), Some(avg_b)) = (&avg_a, &avg_b) {
            let block = format_per_group_pass_counts(avg_a, avg_b, &label_a, &label_b);
            if !block.is_empty() {
                print!("{block}");
            }
        }
        if report.new_in_b > 0 {
            println!(
                "  {} row(s) new in '{}' (no matching key in '{}')",
                report.new_in_b, label_b, label_a,
            );
        }
        if report.removed_from_a > 0 {
            println!(
                "  {} row(s) removed from '{}' (no matching key in '{}')",
                report.removed_from_a, label_a, label_b,
            );
        }
    }

    // Host-context delta. Same first-Some(host) baseline
    // `compare_partitions` uses — picking representative hosts
    // off the partitioned sidecars rather than the full pool so
    // the delta reflects what actually fed the comparison.
    //
    // Zip the pool with the pre-computed `rows` (built once above
    // via `pool.iter().map(sidecar_to_row).collect()`) so the
    // per-side filter reuses the existing row instead of calling
    // `sidecar_to_row` a second and third time. `pool` and `rows`
    // are the same length and same iteration order by construction.
    let sidecars_a: Vec<&crate::test_support::SidecarResult> = pool
        .iter()
        .zip(rows.iter())
        .filter(|(_, r)| filter_a.matches(r))
        .map(|(s, _)| s)
        .collect();
    let sidecars_b: Vec<&crate::test_support::SidecarResult> = pool
        .iter()
        .zip(rows.iter())
        .filter(|(_, r)| filter_b.matches(r))
        .map(|(s, _)| s)
        .collect();
    let host_a = sidecars_a.iter().find_map(|s| s.host.as_ref());
    let host_b = sidecars_b.iter().find_map(|s| s.host.as_ref());
    print!("{}", format_host_delta(host_a, host_b, &label_a, &label_b));

    Ok(if report.regressions > 0 { 1 } else { 0 })
}

/// Bail when `rows` contains two or more entries with the same
/// pairing key — only relevant under `--no-average`, where each
/// sidecar row stays distinct and `compare_rows_by` would
/// silently latch onto whichever entry happened to be first in
/// iteration order. Names the offending key in the diagnostic
/// so the operator can choose to either drop `--no-average` or
/// add another per-side filter to disambiguate.
fn check_no_duplicate_pairing_keys(
    rows: &[GauntletRow],
    pairing_dims: &[Dimension],
    side_label: &str,
) -> anyhow::Result<()> {
    let mut seen: BTreeMap<PairingKey, usize> = BTreeMap::new();
    for row in rows {
        let key = PairingKey::from_row(row, pairing_dims);
        *seen.entry(key).or_insert(0) += 1;
    }
    if let Some((dup_key, count)) = seen.iter().find(|&(_, &c)| c > 1) {
        anyhow::bail!(
            "stats compare --no-average: side {side_label} has {count} \
             sidecars with the same pairing key {key:?}. Either drop \
             --no-average to average them, or add another --{side}-X \
             filter to disambiguate.",
            key = dup_key.0,
            side = side_label.to_lowercase(),
        );
    }
    Ok(())
}

/// Render the host-context delta section of `stats compare --runs`
/// as a block of text ready to `print!`. Extracted as a pure
/// function of `(Option<&HostContext>, Option<&HostContext>, &str,
/// &str)` so the five match arms can be unit-tested without
/// fixturing a real run directory.
///
/// The returned string is either empty (when both sides have no
/// host data — nothing to print) or ends with a newline so callers
/// can chain further output. Single-side cases print a clear
/// "captured in X only, delta unavailable" message rather than
/// silently suppressing the section — a mixed-tooling-version run
/// comparison should surface the asymmetry.
/// Format the one-line averaging-mode header that prints above
/// the comparison table when `--average` is active.
///
/// Pure function of (`pre_agg_a`, `pre_agg_b`, `a`, `b`) so the
/// exact-string contract — the operator-visible "averaged across
/// N runs (A) and M runs (B)" surface — can be unit-tested
/// without capturing stdout from `compare_partitions`.
///
/// `pre_agg_a` / `pre_agg_b` are the post-typed-filter contributor
/// row counts (i.e. the number of sidecar rows that fed
/// [`group_and_average_by`]), NOT the post-aggregation unique-key
/// counts. The two answer different operator questions; the
/// header surfaces the contributor count because that's the
/// "how many trials got folded?" intuition the `--average` flag
/// is actually delivering.
pub(crate) fn format_average_header(
    pre_agg_a: usize,
    pre_agg_b: usize,
    a: &str,
    b: &str,
) -> String {
    format!("averaged across {pre_agg_a} runs ({a}) and {pre_agg_b} runs ({b})")
}

/// Format the per-group `passes_observed/total_observed` block
/// that prints below the summary line when `--average` is active.
///
/// Pure function of (`avg_a`, `avg_b`, `a`, `b`) so the rendered
/// surface — one line per (scenario, topology, work_type) group
/// present on either side, with `N/M` per side and `-` for any
/// side that lacks the group — can be unit-tested without
/// capturing stdout. Returns the trailing-newline-terminated
/// block, or empty string when neither side has groups.
///
/// Line shape:
/// `  scenario/topology/work_type: {a}=N/M {b}=N/M`
///
/// The leading two-space indent matches the sibling
/// `summary:` block's continuation lines (e.g.
/// `"  N (scenario, topology, work_type) row pair(s) skipped..."`)
/// so the per-group block reads as a continuation of the same
/// summary section. A blank line separates this block from the
/// preceding `summary:` line for readability.
///
/// Groups present on only one side render `-` for the missing
/// side (also counted in `compare_rows`' `new_in_b` /
/// `removed_from_a` upstream — the per-group block surfaces the
/// asymmetry by name so the operator can see *which* groups went
/// missing without cross-referencing the summary counters).
pub(crate) fn format_per_group_pass_counts(
    avg_a: &[AveragedGroup],
    avg_b: &[AveragedGroup],
    a: &str,
    b: &str,
) -> String {
    type SummaryKey<'a> = (&'a str, &'a str, &'a str);
    type SummaryValue<'a> = (Option<&'a AveragedGroup>, Option<&'a AveragedGroup>);
    let mut keys: BTreeMap<SummaryKey<'_>, SummaryValue<'_>> = BTreeMap::new();
    for ar in avg_a {
        let k = (
            ar.row.scenario.as_str(),
            ar.row.topology.as_str(),
            ar.row.work_type.as_str(),
        );
        keys.entry(k).or_insert((None, None)).0 = Some(ar);
    }
    for br in avg_b {
        let k = (
            br.row.scenario.as_str(),
            br.row.topology.as_str(),
            br.row.work_type.as_str(),
        );
        keys.entry(k).or_insert((None, None)).1 = Some(br);
    }
    if keys.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push('\n');
    out.push_str(
        "per-group pass counts (passes/total + skip/inconc/fail breakdown when non-zero):\n",
    );
    for ((scn, topo, wt), (ka, kb)) in keys.into_iter() {
        let fmt_side = |r: Option<&AveragedGroup>| -> String {
            let Some(x) = r else {
                return "-".to_string();
            };
            // Mirror format_dimension_summary's 4-state breakdown —
            // operators reading per-group lines must be able to
            // distinguish skip / inconclusive / fail buckets, not
            // see them collapsed into the (total - pass) denominator
            // gap. Skip silently rendering buckets that are zero so
            // the common-case "all passed" line stays terse.
            let mut s = format!("{}/{}", x.passes_observed, x.total_observed);
            let mut extras: Vec<String> = Vec::with_capacity(3);
            if x.skips_observed > 0 {
                extras.push(format!("{} skip", x.skips_observed));
            }
            if x.inconclusives_observed > 0 {
                extras.push(format!("{} inc", x.inconclusives_observed));
            }
            if x.failures_observed > 0 {
                extras.push(format!("{} fail", x.failures_observed));
            }
            if !extras.is_empty() {
                s.push_str(&format!(" ({})", extras.join(", ")));
            }
            s
        };
        out.push_str(&format!(
            "  {scn}/{topo}/{wt}: {a}={pa} {b}={pb}\n",
            pa = fmt_side(ka),
            pb = fmt_side(kb),
        ));
    }
    out
}

pub(crate) fn format_host_delta(
    host_a: Option<&crate::host_context::HostContext>,
    host_b: Option<&crate::host_context::HostContext>,
    a: &str,
    b: &str,
) -> String {
    match (host_a, host_b) {
        (Some(ha), Some(hb)) => {
            let delta = ha.diff(hb);
            if delta.is_empty() {
                // Identical hosts: surface arch when both sides
                // carry it so the operator sees WHAT is identical
                // (the two runs share x86_64 vs both being aarch64
                // is the operator's question). When
                // either side leaves arch as `None` (pre-host-
                // context-landing archive, or arch probe failed
                // on at least one side), fall through to the
                // bare "identical" message — emitting a partial
                // hint would mislead the reader into thinking
                // the silent side disagreed.
                match (ha.arch.as_deref(), hb.arch.as_deref()) {
                    (Some(arch_a), Some(arch_b)) if arch_a == arch_b => {
                        format!("\nhost: identical between '{a}' and '{b}' (arch: {arch_a})\n",)
                    }
                    _ => format!("\nhost: identical between '{a}' and '{b}'\n"),
                }
            } else {
                format!("\nhost delta ('{a}' → '{b}'):\n{delta}")
            }
        }
        (Some(_), None) => {
            format!("\nhost: captured in '{a}' only, delta unavailable\n")
        }
        (None, Some(_)) => {
            format!("\nhost: captured in '{b}' only, delta unavailable\n")
        }
        (None, None) => String::new(),
    }
}

#[cfg(test)]
mod tests;
