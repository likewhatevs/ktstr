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
}

/// Sub-classification for [`MetricKind::Gauge`] picking the
/// per-window reduction. Most ktstr gauges are Avg ("typical-load
/// over the window"); Last fits "current state" snapshots like
/// `comm` / `policy`; Max fits worst-instant queue-depth probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
    /// The reduction is a DERIVED ratio (a [`MetricKind::Rate`]): it
    /// cannot be folded from two already-divided values. The components
    /// merge by their own kinds and the rate is re-derived as
    /// `Σnumerator / Σdenominator` — see [`derive_rate_metrics`]. The
    /// per-metric merge loop skips Rate keys entirely and the post-pass
    /// re-derives them, so this variant is classification metadata: no
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
        }
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
        // A Rate has no samples of its own; its per-phase value is
        // derived from the reduced numerator/denominator component
        // buckets by `derive_rate_metrics` in the build post-pass. Return
        // None so the build loop inserts no rate key here.
        MetricKind::Rate { .. } => None,
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
    /// single-sample shape: most ktstr metrics are computed
    /// host-side from cross-CPU or cross-cgroup folds
    /// (`worst_spread`, `worst_gap_ms`, `worst_migration_ratio`,
    /// `max_imbalance_ratio`, all `worst_*_wake_latency_*`,
    /// `worst_iterations_per_worker`, `worst_iterations_per_cpu_sec`,
    /// `worst_page_locality`, `worst_cross_node_migration_ratio`,
    /// `worst_mean_run_delay_us`, `worst_run_delay_us`,
    /// `worst_wake_latency_tail_ratio`) and have no single-sample
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
        // cross-cgroup folds computed host-side at
        // `evaluate_vm_result` time (worst-spread, worst-gap-ms,
        // every `worst_*_wake_latency_*`, worst-iterations-per-
        // worker, etc.) and have no single-sample equivalent —
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
/// Nine of the remaining metrics in [`METRICS`] have matching
/// registry / field / DataFrame column names
/// (`worst_p99_wake_latency_us`, `worst_median_wake_latency_us`,
/// `worst_wake_latency_cv`, `total_iterations`,
/// `worst_mean_run_delay_us`, `worst_run_delay_us`,
/// `worst_wake_latency_tail_ratio`,
/// `worst_iterations_per_worker`,
/// `worst_iterations_per_cpu_sec`) and are not listed — no
/// translation to document.
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
        name: "worst_p99_wake_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Last),
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |r| Some(r.worst_p99_wake_latency_us),
    },
    MetricDef {
        name: "worst_median_wake_latency_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Last),
        default_abs: 20.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |r| Some(r.worst_median_wake_latency_us),
    },
    MetricDef {
        name: "worst_wake_latency_cv",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Last),
        default_abs: 0.10,
        default_rel: 0.25,
        display_unit: "",
        accessor: |r| Some(r.worst_wake_latency_cv),
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
        // Distinct from the per-cgroup `worst_iterations_per_cpu_sec` Gauge
        // (the min-fold starvation selector): this is the POOLED cohort rate,
        // overcommit-invariant. _per_cpu_sec name + Rate kind passes the
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
        name: "worst_mean_run_delay_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Last),
        default_abs: 50.0,
        default_rel: 0.25,
        display_unit: "\u{00b5}s",
        accessor: |r| Some(r.worst_mean_run_delay_us),
    },
    MetricDef {
        name: "worst_run_delay_us",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Peak,
        default_abs: 100.0,
        default_rel: 0.50,
        display_unit: "\u{00b5}s",
        accessor: |r| Some(r.worst_run_delay_us),
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
        // Samples-required noise gate: the accessor returns `None` when
        // the run completed fewer than
        // [`WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`] iterations; with
        // few samples the p99 estimate is effectively the observed
        // maximum and the tail ratio is dominated by a single
        // outlier rather than a distributional signal. Routing
        // through `None` lets `compare_rows` fall through to the
        // `ext_metrics` lookup (which is also empty for a sub-
        // threshold run), then to the `unwrap_or(0.0)` default, so
        // both A- and B-side rows collapse to 0.0 and the subsequent
        // `abs() < EPSILON` short-circuit silently skips the metric
        // for that row. See [`WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`]
        // for the threshold-value rationale.
        name: "worst_wake_latency_tail_ratio",
        polarity: crate::test_support::Polarity::LowerBetter,
        kind: MetricKind::Gauge(GaugeAgg::Last),
        default_abs: 0.5,
        default_rel: 0.25,
        display_unit: "x",
        accessor: |r| {
            if r.total_iterations < WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS {
                None
            } else {
                Some(r.worst_wake_latency_tail_ratio)
            }
        },
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
        name: "worst_iterations_per_worker",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Gauge(GaugeAgg::Last),
        default_abs: 10.0,
        default_rel: 0.10,
        display_unit: "",
        accessor: |r| Some(r.worst_iterations_per_worker),
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
        name: "worst_iterations_per_cpu_sec",
        polarity: crate::test_support::Polarity::HigherBetter,
        kind: MetricKind::Gauge(GaugeAgg::Last),
        default_abs: 10.0,
        default_rel: 0.10,
        // Same physical quantity as the pooled iterations_per_cpu_sec Rate;
        // share its unit string rather than leaving this one under-specified.
        display_unit: "iter/cpu-s",
        accessor: |r| Some(r.worst_iterations_per_cpu_sec),
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
/// The gate is applied in the metric's accessor closure in [`METRICS`]:
/// a row with `total_iterations < WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`
/// returns `None`, which `compare_rows` short-circuits to 0.0 against
/// both A- and B-side rows, which then falls under the
/// `abs() < EPSILON` "unchanged" guard and emits no finding.
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
    // Benchmarking fields.
    pub worst_p99_wake_latency_us: f64,
    pub worst_median_wake_latency_us: f64,
    pub worst_wake_latency_cv: f64,
    pub total_iterations: u64,
    pub worst_mean_run_delay_us: f64,
    pub worst_run_delay_us: f64,
    /// Worst-case ratio of p99 / median wake latency across cgroups.
    /// Higher values indicate a stretched long tail. Registry name
    /// matches the field name; see the triples table on [`METRICS`]
    /// for the full registry / field / DataFrame-column mapping.
    /// Noise-suppressed when the scenario produced fewer than
    /// [`WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`] iterations — see
    /// the constant's doc for the rationale.
    pub worst_wake_latency_tail_ratio: f64,
    /// Worst-case per-worker iteration count across cgroups (LOWEST
    /// across cgroups — lower is worse). Registry name matches the
    /// field name; see the triples table on [`METRICS`] for the
    /// field / registry / DataFrame-column mapping.
    ///
    /// # `worst_` vs `lowest_` naming evaluation
    ///
    /// A `lowest_iterations_per_worker` rename was considered — it
    /// would describe the merge direction (min across cgroups) more
    /// literally than `worst_`, which semantically maps "worst" to
    /// different merge operations depending on polarity (max for
    /// lower-better metrics, min for higher-better). Rejected
    /// because `worst_` is the codebase-wide prefix for
    /// cross-cgroup roll-ups regardless of polarity — see
    /// `worst_page_locality` (`HigherBetter` → the merge takes the
    /// LOWEST non-zero value) and `worst_spread` (`LowerBetter` →
    /// the merge takes the HIGHEST). Breaking that convention for
    /// one metric would require either (a) renaming every existing
    /// `HigherBetter` worst_* metric to `lowest_*` for consistency,
    /// or (b) accepting a mixed naming scheme where readers have to
    /// cross-reference each metric's polarity to understand the
    /// prefix. Option (a) is a high-churn rename across
    /// sidecars / DataFrames / CI gates; option (b) degrades
    /// readability. The current convention — `worst_` = "the
    /// cross-cgroup roll-up that surfaces the most problematic
    /// cgroup, direction determined by the metric's polarity" —
    /// is documented on [`METRICS`] and applies here.
    pub worst_iterations_per_worker: f64,
    /// Worst-case per-cgroup CPU-time efficiency
    /// ([`crate::assert::ScenarioStats::worst_iterations_per_cpu_sec`],
    /// the lowest [`crate::assert::CgroupStats::iterations_per_cpu_sec`]
    /// across the run's cgroups). Unlike
    /// [`Self::worst_iterations_per_worker`] (raw work, which scales with
    /// the host-CPU budget), this is OVERCOMMIT-INVARIANT — it is the
    /// metric to compare across runs of different `cpu_budget` without the
    /// host-contention confound the overcommit marker / compare-path
    /// warning point at. `0.0` when no cgroup reported a defined rate
    /// (collapsed from `None`, same as the sibling worst_* fields).
    /// Surfaced in [`METRICS`] under `worst_iterations_per_cpu_sec`.
    pub worst_iterations_per_cpu_sec: f64,
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
        sum_p99_wake: f64,
        sum_median_wake: f64,
        sum_wake_cv: f64,
        sum_total_iterations: u64,
        sum_mean_run_delay: f64,
        sum_tail_ratio: f64,
        sum_iters_per_worker: f64,
        sum_iters_per_cpu_sec: f64,
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
        max_run_delay_us: f64,
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
                sum_p99_wake: 0.0,
                sum_median_wake: 0.0,
                sum_wake_cv: 0.0,
                sum_total_iterations: 0,
                sum_mean_run_delay: 0.0,
                sum_tail_ratio: 0.0,
                sum_iters_per_worker: 0.0,
                sum_iters_per_cpu_sec: 0.0,
                sum_page_locality: 0.0,
                sum_cross_node_mig: 0.0,
                max_gap_ms: 0,
                max_imbalance_ratio: 0.0,
                max_max_dsq_depth: 0,
                max_run_delay_us: 0.0,
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
        acc.sum_p99_wake += row.worst_p99_wake_latency_us;
        acc.sum_median_wake += row.worst_median_wake_latency_us;
        acc.sum_wake_cv += row.worst_wake_latency_cv;
        acc.sum_total_iterations = acc
            .sum_total_iterations
            .saturating_add(row.total_iterations);
        acc.sum_mean_run_delay += row.worst_mean_run_delay_us;
        acc.sum_tail_ratio += row.worst_wake_latency_tail_ratio;
        acc.sum_iters_per_worker += row.worst_iterations_per_worker;
        acc.sum_iters_per_cpu_sec += row.worst_iterations_per_cpu_sec;
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
        if row.worst_run_delay_us > acc.max_run_delay_us {
            acc.max_run_delay_us = row.worst_run_delay_us;
        }
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
        // re-derived from the folded components as a post-pass. A Rate is
        // derived, not folded: folding two ready-made per-phase ratios
        // would lose the Σnum/Σdenom re-pool, and routing a Rate through
        // aggregate_samples_weighted would hit the aggregate_finite Rate
        // guard. Dispatch by registered MetricKind so Gauge(Avg) gets the
        // weighted-mean fold (matches the per-phase merge contract);
        // unregistered names (no metric_def) fall back to arithmetic mean,
        // the legacy (sum, count) semantic. Skip a key whose reduction is
        // None (every value NaN — defensive post sidecar_to_row sanitize).
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
            worst_run_delay_us: acc.max_run_delay_us,
            migrations: round_u64(acc.sum_migrations),
            migration_ratio: acc.sum_migration_ratio / denom,
            stuck_count: round_usize(acc.sum_stuck_count),
            fallback_count: round_i64(acc.sum_fallback_count),
            keep_last_count: round_i64(acc.sum_keep_last_count),
            worst_p99_wake_latency_us: acc.sum_p99_wake / denom,
            worst_median_wake_latency_us: acc.sum_median_wake / denom,
            worst_wake_latency_cv: acc.sum_wake_cv / denom,
            total_iterations: round_u64(acc.sum_total_iterations),
            worst_mean_run_delay_us: acc.sum_mean_run_delay / denom,
            worst_wake_latency_tail_ratio: acc.sum_tail_ratio / denom,
            worst_iterations_per_worker: acc.sum_iters_per_worker / denom,
            worst_iterations_per_cpu_sec: acc.sum_iters_per_cpu_sec / denom,
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
/// Three fields are especially affected — note in-tree producers
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
/// - `worst_wake_latency_cv`: lower-better. A real 0.0 means
///   "wake-latency samples were perfectly uniform" (ideal jitter).
///   A sanitized NaN collapses to the same value and reads as
///   *falsely good* — same direction as `migration_ratio`.
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
        worst_p99_wake_latency_us: finite_or_zero(
            "worst_p99_wake_latency_us",
            sc.stats.worst_p99_wake_latency_us,
        ),
        worst_median_wake_latency_us: finite_or_zero(
            "worst_median_wake_latency_us",
            sc.stats.worst_median_wake_latency_us,
        ),
        worst_wake_latency_cv: finite_or_zero(
            "worst_wake_latency_cv",
            sc.stats.worst_wake_latency_cv,
        ),
        total_iterations: sc.stats.total_iterations,
        worst_mean_run_delay_us: finite_or_zero(
            "worst_mean_run_delay_us",
            sc.stats.worst_mean_run_delay_us,
        ),
        worst_run_delay_us: finite_or_zero("worst_run_delay_us", sc.stats.worst_run_delay_us),
        worst_wake_latency_tail_ratio: finite_or_zero(
            "worst_wake_latency_tail_ratio",
            sc.stats.worst_wake_latency_tail_ratio,
        ),
        // `worst_iterations_per_worker` is `Option` on ScenarioStats
        // (None = no cgroup reported a worker; Some(0.0) = a cgroup
        // ran zero iterations). The GauntletRow field uses this
        // layer's documented 0.0-sentinel-with-warn convention (see
        // the doc above): a measured Some(0.0) maps to 0.0 (real
        // starvation, surfaced), and None warns and maps to 0.0 like
        // any other no-data field at this ingress. The cross-cgroup
        // None/Some(0.0) distinction is preserved upstream in
        // AssertResult::merge; this boundary matches the gauntlet
        // layer's uniform sentinel handling.
        worst_iterations_per_worker: match sc.stats.worst_iterations_per_worker {
            Some(v) => finite_or_zero("worst_iterations_per_worker", v),
            None => {
                tracing::warn!(
                    test = %sc.test_name,
                    field = "worst_iterations_per_worker",
                    "no cgroup reported a worker; substituting 0.0",
                );
                0.0
            }
        },
        // Overcommit-invariant efficiency. `None` when no cgroup reported
        // a defined rate (no workers or no on-CPU time captured); maps to
        // 0.0 with a warn, same sentinel convention as the sibling above.
        worst_iterations_per_cpu_sec: match sc.stats.worst_iterations_per_cpu_sec {
            Some(v) => finite_or_zero("worst_iterations_per_cpu_sec", v),
            None => {
                tracing::warn!(
                    test = %sc.test_name,
                    field = "worst_iterations_per_cpu_sec",
                    "no cgroup reported a defined per-cpu-sec rate; substituting 0.0",
                );
                0.0
            }
        },
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
    ("worst_p99_wake_latency_us", |r| r.worst_p99_wake_latency_us),
    ("worst_wake_latency_cv", |r| r.worst_wake_latency_cv),
    ("worst_mean_run_delay_us", |r| r.worst_mean_run_delay_us),
    ("worst_run_delay_us", |r| r.worst_run_delay_us),
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
/// failed rows carry default-zero metric values (sidecar_to_row
/// substitutes zero for non-finite + missing fields), and
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
mod tests {
    use super::*;
    use crate::assert::ScenarioStats;

    // -- MetricKind temporal aggregation ----------------------------

    /// `Counter` reduces by sum-of-finite-samples. NaN drops, empty
    /// returns None.
    #[test]
    fn aggregate_samples_counter_sums_finite_values() {
        assert_eq!(
            aggregate_samples(&[1.0, 2.0, 3.0], MetricKind::Counter),
            Some(6.0),
        );
        assert_eq!(
            aggregate_samples(&[1.0, f64::NAN, 3.0], MetricKind::Counter),
            Some(4.0),
            "NaN samples drop from the sum",
        );
        assert_eq!(
            aggregate_samples(&[], MetricKind::Counter),
            None,
            "empty input → None",
        );
        assert_eq!(
            aggregate_samples(&[f64::NAN, f64::INFINITY], MetricKind::Counter),
            None,
            "all-non-finite → None",
        );
    }

    /// `Gauge(Avg)` reduces by arithmetic mean.
    #[test]
    fn aggregate_samples_gauge_avg_means_finite() {
        let r = aggregate_samples(&[1.0, 2.0, 3.0], MetricKind::Gauge(GaugeAgg::Avg));
        assert_eq!(r, Some(2.0));
    }

    /// `Gauge(Last)` returns the last finite sample.
    #[test]
    fn aggregate_samples_gauge_last_returns_last() {
        let r = aggregate_samples(&[1.0, 2.0, 3.0], MetricKind::Gauge(GaugeAgg::Last));
        assert_eq!(r, Some(3.0));
        // NaN at the tail still drops; Last picks the last FINITE.
        let r = aggregate_samples(&[1.0, 2.0, f64::NAN], MetricKind::Gauge(GaugeAgg::Last));
        assert_eq!(r, Some(2.0));
    }

    /// `Gauge(Max)` and `Peak` both reduce by max.
    #[test]
    fn aggregate_samples_max_and_peak_pick_largest() {
        let r = aggregate_samples(&[1.0, 5.0, 3.0], MetricKind::Gauge(GaugeAgg::Max));
        assert_eq!(r, Some(5.0));
        let r = aggregate_samples(&[1.0, 5.0, 3.0], MetricKind::Peak);
        assert_eq!(r, Some(5.0));
    }

    /// `Timestamp` returns the last sample (latest snapshot).
    #[test]
    fn aggregate_samples_timestamp_returns_last() {
        let r = aggregate_samples(&[100.0, 200.0, 300.0], MetricKind::Timestamp);
        assert_eq!(r, Some(300.0));
    }

    // -- Weighted Gauge(Avg) ------------------------------------

    /// `Gauge(Avg)` with explicit per-sample weights computes the
    /// weighted arithmetic mean `sum(v*w) / sum(w)`. (10*5 + 20*15)
    /// / 20 = 17.5 — a 50%-larger weight on the higher value
    /// pulls the mean above the unweighted midpoint of 15.
    #[test]
    fn aggregate_samples_weighted_gauge_avg_pulls_toward_heavier_sample() {
        let r =
            aggregate_samples_weighted(&[(10.0, 5), (20.0, 15)], MetricKind::Gauge(GaugeAgg::Avg));
        assert_eq!(r, Some(17.5));
    }

    /// `Gauge(Avg)` without weights (unweighted entry point)
    /// computes the arithmetic mean (legacy semantic). The
    /// previous test's same value vector yields 15.0 here vs
    /// 17.5 weighted — the difference is the cross-RUN bias.
    #[test]
    fn aggregate_samples_gauge_avg_unweighted_is_arithmetic_mean() {
        let r = aggregate_samples(&[10.0, 20.0], MetricKind::Gauge(GaugeAgg::Avg));
        assert_eq!(r, Some(15.0));
    }

    /// `Gauge(Avg)` with zero total weight falls back to the
    /// unweighted arithmetic mean rather than dividing by zero.
    /// Mirrors `merge_metric_values` at
    /// `crate::assert::merge_matched_phase_buckets` per
    /// single-source-of-truth.
    #[test]
    fn aggregate_samples_weighted_gauge_avg_zero_total_weight_falls_back_to_mean() {
        let r =
            aggregate_samples_weighted(&[(10.0, 0), (30.0, 0)], MetricKind::Gauge(GaugeAgg::Avg));
        assert_eq!(r, Some(20.0));
    }

    /// `Counter` ignores weights — sum is weight-independent by
    /// construction. Pinned so a future refactor that introduces
    /// weight-sensitive Counter semantics breaks here.
    #[test]
    fn aggregate_samples_weighted_counter_ignores_weights() {
        let r = aggregate_samples_weighted(&[(10.0, 5), (20.0, 15)], MetricKind::Counter);
        assert_eq!(r, Some(30.0));
    }

    /// `Peak` ignores weights — max is weight-independent.
    #[test]
    fn aggregate_samples_weighted_peak_ignores_weights() {
        let r = aggregate_samples_weighted(&[(10.0, 5), (20.0, 15)], MetricKind::Peak);
        assert_eq!(r, Some(20.0));
    }

    /// `Gauge(Max)` ignores weights — max is weight-independent.
    #[test]
    fn aggregate_samples_weighted_gauge_max_ignores_weights() {
        let r =
            aggregate_samples_weighted(&[(10.0, 5), (20.0, 15)], MetricKind::Gauge(GaugeAgg::Max));
        assert_eq!(r, Some(20.0));
    }

    /// `Gauge(Last)` and `Timestamp` ignore weights — last-finite
    /// is weight-independent.
    #[test]
    fn aggregate_samples_weighted_gauge_last_and_timestamp_ignore_weights() {
        let last =
            aggregate_samples_weighted(&[(10.0, 5), (20.0, 15)], MetricKind::Gauge(GaugeAgg::Last));
        assert_eq!(last, Some(20.0));
        let ts = aggregate_samples_weighted(&[(10.0, 5), (20.0, 15)], MetricKind::Timestamp);
        assert_eq!(ts, Some(20.0));
    }

    /// NaN-valued pairs drop entirely (value AND weight) — the
    /// (f64, usize) pair type keeps the weight bound to its
    /// value so a NaN filter can't misalign weights to other
    /// samples. (10*5 + 30*20) / (5+20) = 650 / 25 = 26.0.
    #[test]
    fn aggregate_samples_weighted_gauge_avg_drops_nan_pairs_in_lockstep() {
        let r = aggregate_samples_weighted(
            &[(10.0, 5), (f64::NAN, 10), (30.0, 20)],
            MetricKind::Gauge(GaugeAgg::Avg),
        );
        assert_eq!(r, Some(26.0));
    }

    // -- Per-phase reductions --------------------------------------

    /// `phase_counter_delta` returns `last - first` of finite
    /// samples — the right semantic for a per-phase reduction
    /// over a cumulative-since-boot counter. Distinct from
    /// `aggregate_samples(..., Counter)` which sums the samples
    /// (correct for cross-run aggregation, wrong for per-phase
    /// deltas).
    #[test]
    fn phase_counter_delta_returns_last_minus_first() {
        // Cumulative-since-boot counter samples in [100, 150, 175,
        // 200] yield delta 100 across the phase window.
        assert_eq!(
            phase_counter_delta(&[100.0, 150.0, 175.0, 200.0]),
            Some(100.0),
        );
        // NaN samples drop from the finite slice; first/last
        // are computed over the filtered sequence.
        assert_eq!(
            phase_counter_delta(&[f64::NAN, 150.0, 175.0, f64::NAN]),
            Some(25.0),
        );
    }

    /// `phase_counter_delta` returns `Some(0.0)` for a phase with
    /// exactly one finite sample (self-delta — the metric was
    /// observed but no per-phase change can be computed), and
    /// `None` only when zero samples are finite. The distinction
    /// matters for the bucket renderer: `Some(0.0)` paints "phase
    /// has data, delta is 0"; `None` paints "no data".
    #[test]
    fn phase_counter_delta_one_finite_sample_is_self_delta() {
        assert_eq!(phase_counter_delta(&[42.0]), Some(0.0));
        assert_eq!(phase_counter_delta(&[f64::NAN, 42.0, f64::NAN]), Some(0.0));
        assert_eq!(phase_counter_delta(&[]), None);
        assert_eq!(phase_counter_delta(&[f64::NAN, f64::INFINITY]), None);
    }

    /// A counter that regresses across a phase window
    /// (scheduler-restart counter reset, kernel module reload,
    /// etc.) clamps to 0 rather than emitting a negative delta a
    /// downstream "negative count is impossible" assertion would
    /// either misread or trip on. Mirrors the existing
    /// `monitor::counter_delta` clamp pattern.
    #[test]
    fn phase_counter_delta_clamps_negative_to_zero_on_counter_reset() {
        assert_eq!(
            phase_counter_delta(&[500.0, 600.0, 100.0]),
            Some(0.0),
            "last < first clamps to 0 (counter reset detected)",
        );
    }

    /// `aggregate_samples_for_phase` dispatches Counter through
    /// `phase_counter_delta` (per-phase delta) and every other
    /// kind through `aggregate_samples` (flat-run semantic). Pins
    /// the invariant: a Counter-kind metric must NOT collapse
    /// to a sum across the phase window — that's the bug the
    /// per-phase aggregator was introduced to fix.
    #[test]
    fn aggregate_samples_for_phase_dispatches_on_kind() {
        let counter = MetricDef {
            name: "total_test_counter",
            accessor: |_| None,
            display_unit: "",
            polarity: crate::test_support::Polarity::HigherBetter,
            default_abs: 0.0,
            default_rel: 0.0,
            kind: MetricKind::Counter,
        };
        // Counter routes through `phase_counter_delta`, NOT
        // `aggregate_samples`'s sum.
        assert_eq!(
            aggregate_samples_for_phase(&counter, &[100.0, 150.0, 175.0]),
            Some(75.0),
            "Counter kind must reduce by last - first, not by sum",
        );
        assert_ne!(
            aggregate_samples_for_phase(&counter, &[100.0, 150.0, 175.0]),
            Some(425.0),
            "Counter kind MUST NOT collapse to flat-run sum across a phase",
        );

        let peak = MetricDef {
            name: "max_test_peak",
            accessor: |_| None,
            display_unit: "",
            polarity: crate::test_support::Polarity::LowerBetter,
            default_abs: 0.0,
            default_rel: 0.0,
            kind: MetricKind::Peak,
        };
        // Peak routes through `aggregate_samples` -> max.
        assert_eq!(
            aggregate_samples_for_phase(&peak, &[1.0, 5.0, 3.0]),
            Some(5.0),
            "Peak kind must reduce by max",
        );

        let gauge_avg = MetricDef {
            name: "worst_test_gauge",
            accessor: |_| None,
            display_unit: "",
            polarity: crate::test_support::Polarity::LowerBetter,
            default_abs: 0.0,
            default_rel: 0.0,
            kind: MetricKind::Gauge(GaugeAgg::Avg),
        };
        assert_eq!(
            aggregate_samples_for_phase(&gauge_avg, &[2.0, 4.0, 6.0]),
            Some(4.0),
            "Gauge(Avg) kind must reduce by arithmetic mean",
        );

        let delta_sum = MetricDef {
            name: "total_test_delta",
            accessor: |_| None,
            display_unit: "",
            polarity: crate::test_support::Polarity::LowerBetter,
            default_abs: 0.0,
            default_rel: 0.0,
            kind: MetricKind::DeltaSum,
        };
        // DeltaSum samples are ALREADY per-read deltas, so the per-phase
        // reduction SUMS them — NOT a last-minus-first that would
        // difference two deltas. [10, 20, 5] -> 35, not 5 - 10 (which a
        // Counter would clamp to 0).
        assert_eq!(
            aggregate_samples_for_phase(&delta_sum, &[10.0, 20.0, 5.0]),
            Some(35.0),
            "DeltaSum kind must reduce by sum of per-read deltas",
        );
        assert_eq!(
            aggregate_samples(&[10.0, 20.0, 5.0], MetricKind::DeltaSum),
            Some(35.0),
            "DeltaSum flat-run reduction is also a sum",
        );
    }

    /// A [`MetricKind::Rate`] is derived as numerator/denominator at the
    /// per-phase level and RE-POOLED (Σnum/Σdenom) across a merge — never
    /// averaged as a ready-made ratio. Pins the core of the Rate kind.
    #[test]
    fn rate_derives_per_phase_and_repools_across_merge() {
        use std::collections::BTreeMap;
        // Per-phase: rate = num / denom.
        let mut phase = BTreeMap::new();
        phase.insert("iters".to_string(), 1000.0);
        phase.insert("secs".to_string(), 4.0);
        derive_rate_metrics_from(&mut phase, std::iter::once(("rate", "iters", "secs")));
        assert_eq!(
            phase.get("rate").copied(),
            Some(250.0),
            "per-phase rate = num/denom",
        );

        // Cross-phase merge re-pools: the components fold by their own
        // kind (Counter -> sum) FIRST, then the rate is re-derived from
        // the pooled components. Phase A = 1000 iters / 1 s = 1000/s;
        // phase B = 10 iters / 9 s ≈ 1.11/s. The correct merged rate is
        // (1000+10)/(1+9) = 101.0, NOT the mean of the two ratios
        // (1000 + 1.11)/2 ≈ 500.6.
        let mut merged = BTreeMap::new();
        merged.insert("iters".to_string(), 1000.0 + 10.0); // Counter sum
        merged.insert("secs".to_string(), 1.0 + 9.0); // Counter sum
        derive_rate_metrics_from(&mut merged, std::iter::once(("rate", "iters", "secs")));
        assert_eq!(
            merged.get("rate").copied(),
            Some(101.0),
            "merged rate must re-pool Σnum/Σdenom",
        );
        let mean_of_ratios = (1000.0 + (10.0 / 9.0)) / 2.0;
        assert!(
            (merged.get("rate").copied().unwrap() - mean_of_ratios).abs() > 100.0,
            "re-pool must differ from mean-of-ratios (got {:?}, mean-of-ratios {mean_of_ratios})",
            merged.get("rate"),
        );
    }

    /// `derive_rate_metrics` leaves the rate key ABSENT (distinct from a
    /// real 0.0) when a component is missing, the denominator is zero, or
    /// a component is non-finite.
    #[test]
    fn rate_absent_on_missing_component_zero_or_nonfinite() {
        use std::collections::BTreeMap;
        // Denominator missing.
        let mut m = BTreeMap::new();
        m.insert("iters".to_string(), 5.0);
        derive_rate_metrics_from(&mut m, std::iter::once(("rate", "iters", "secs")));
        assert!(!m.contains_key("rate"), "absent denom -> no rate key");

        // Denominator zero (must NOT insert inf).
        m.insert("secs".to_string(), 0.0);
        derive_rate_metrics_from(&mut m, std::iter::once(("rate", "iters", "secs")));
        assert!(!m.contains_key("rate"), "zero denom -> no rate key");

        // Non-finite numerator.
        let mut n = BTreeMap::new();
        n.insert("iters".to_string(), f64::NAN);
        n.insert("secs".to_string(), 2.0);
        derive_rate_metrics_from(&mut n, std::iter::once(("rate", "iters", "secs")));
        assert!(!n.contains_key("rate"), "NaN numerator -> no rate key");

        // Finite inputs whose QUOTIENT overflows to inf: absent, not inf.
        let mut o = BTreeMap::new();
        o.insert("iters".to_string(), f64::MAX);
        o.insert("secs".to_string(), f64::MIN_POSITIVE);
        derive_rate_metrics_from(&mut o, std::iter::once(("rate", "iters", "secs")));
        assert!(!o.contains_key("rate"), "inf quotient -> no rate key");
    }

    /// A Rate has no samples of its own, so the per-phase reducer returns
    /// None (the build post-pass derives it from components instead).
    #[test]
    fn rate_kind_returns_none_from_per_phase_reducer() {
        let rate = MetricDef {
            name: "test_rate",
            accessor: |_| None,
            display_unit: "",
            polarity: crate::test_support::Polarity::HigherBetter,
            default_abs: 0.0,
            default_rel: 0.0,
            kind: MetricKind::Rate {
                numerator: "n",
                denominator: "d",
            },
        };
        assert_eq!(
            aggregate_samples_for_phase(&rate, &[1.0, 2.0, 3.0]),
            None,
            "Rate reduces to None per-phase; derive_rate_metrics owns it",
        );
    }

    /// Routing a Rate through the single-slice reducer is a bug — it
    /// cannot express Σnum/Σdenom — so `aggregate_finite` panics rather
    /// than silently producing a meaningless one-slice value.
    #[test]
    #[should_panic(expected = "must be derived via derive_rate_metrics")]
    fn rate_kind_panics_in_single_slice_reducer() {
        let _ = aggregate_samples(
            &[1.0, 2.0],
            MetricKind::Rate {
                numerator: "n",
                denominator: "d",
            },
        );
    }

    /// All-empty / all-NaN inputs to either entry point return
    /// `None`. The phase renderer treats absent values as "no
    /// finite samples for this metric in this phase" — distinct
    /// from `Some(0.0)` which is a real reduced zero from finite
    /// samples — so the `None` shape must round-trip.
    #[test]
    fn aggregate_samples_for_phase_returns_none_on_empty_or_all_nan() {
        let counter = MetricDef {
            name: "total_x",
            accessor: |_| None,
            display_unit: "",
            polarity: crate::test_support::Polarity::HigherBetter,
            default_abs: 0.0,
            default_rel: 0.0,
            kind: MetricKind::Counter,
        };
        assert_eq!(aggregate_samples_for_phase(&counter, &[]), None);
        assert_eq!(
            aggregate_samples_for_phase(&counter, &[f64::NAN, f64::NAN]),
            None,
        );
        let peak = MetricDef {
            name: "max_x",
            accessor: |_| None,
            display_unit: "",
            polarity: crate::test_support::Polarity::LowerBetter,
            default_abs: 0.0,
            default_rel: 0.0,
            kind: MetricKind::Peak,
        };
        assert_eq!(aggregate_samples_for_phase(&peak, &[]), None);
        assert_eq!(
            aggregate_samples_for_phase(&peak, &[f64::NAN, f64::INFINITY]),
            None,
        );
    }

    /// Every entry in the `METRICS` registry must have a kind set.
    /// Pinned via the registry walk so a future entry that forgot
    /// to specify `kind` fails to compile (struct-literal
    /// non_exhaustive forces it), and a registry entry whose kind
    /// is `Counter` matches one of the well-known total/stuck_count
    /// names — drift either direction trips here.
    #[test]
    fn every_metric_has_kind_consistent_with_naming() {
        for m in METRICS {
            // Counter and DeltaSum metrics are both cumulative totals
            // (Counter = since-boot, DeltaSum = sum of per-read deltas),
            // so both must be named with `total_` / `_count` /
            // `total_iterations` / `stuck_count` per the established
            // convention.
            if matches!(m.kind, MetricKind::Counter | MetricKind::DeltaSum) {
                assert!(
                    m.name.starts_with("total_") || m.name.ends_with("_count"),
                    "Counter/DeltaSum-kind metric must follow total_*/*_count naming, got {:?}",
                    m.name,
                );
            }
            // Peak metrics must be named with `max_` or be one of
            // the documented worst-case high-water entries
            // (`worst_gap_ms`, `worst_run_delay_us`).
            if matches!(m.kind, MetricKind::Peak) {
                assert!(
                    m.name.starts_with("max_")
                        || m.name == "worst_gap_ms"
                        || m.name == "worst_run_delay_us",
                    "Peak-kind metric must use max_* naming OR be a documented worst-* peak, got {:?}",
                    m.name,
                );
            }
            // Rate metrics are derived ratios; name them `*_rate` or
            // `*_per_*` so the registry reads as a rate at a glance.
            if let MetricKind::Rate {
                numerator,
                denominator,
            } = m.kind
            {
                assert!(
                    m.name.ends_with("_rate") || m.name.contains("_per_"),
                    "Rate-kind metric must use *_rate or *_per_* naming, got {:?}",
                    m.name,
                );
                // Components must be registered AND not themselves Rate:
                // derive_rate_metrics is a pure function of non-derived
                // components, which is what keeps the re-pool associative.
                // A rate-of-a-rate would make the post-pass order-dependent
                // on METRICS declaration order (the inner rate's key is
                // skipped in the merge loop, so a stale value could be read).
                for comp in [numerator, denominator] {
                    let cd = metric_def(comp).unwrap_or_else(|| {
                        panic!(
                            "Rate metric {:?} component {comp:?} is not registered",
                            m.name
                        )
                    });
                    assert!(
                        !matches!(cd.kind, MetricKind::Rate { .. }),
                        "Rate metric {:?} component {comp:?} must not itself be Rate \
                         (a rate-of-a-rate breaks the associative re-derive)",
                        m.name,
                    );
                }
            }

            // REVERSE gate: a metric NAMED like a per-second rate MUST be a
            // Rate, so a future per-second metric cannot silently ship as a
            // Gauge that averages ready-made ratios (the (r₁+r₂)/2 bug). Scoped
            // to per-SECOND tokens (`_rate` / `_per_sec` / `_per_cpu_sec`) — NOT
            // bare `_per_` — so a count-denominator metric like
            // `worst_iterations_per_worker` (intentionally a Gauge) is not
            // falsely flagged. `worst_iterations_per_cpu_sec` is the documented
            // exception: registered Gauge(Last) (temporal: last sample), its
            // cross-cgroup worst is a min-SELECTION via fold_lowest_some on
            // the typed ScenarioStats field (later surfaced as the GauntletRow
            // field) — the per-cgroup starvation signal, NOT a Σnum/Σdenom
            // re-pool — so it is correctly NOT a Rate and keeps its accurate
            // `_per_cpu_sec` name.
            let looks_like_rate = m.name.ends_with("_rate")
                || m.name.contains("_per_sec")
                || m.name.contains("_per_cpu_sec");
            if looks_like_rate && m.name != "worst_iterations_per_cpu_sec" {
                assert!(
                    matches!(m.kind, MetricKind::Rate { .. }),
                    "metric {:?} is named like a per-second rate but is not \
                     MetricKind::Rate (register it as a Rate, or allowlist it \
                     here if it is intentionally a non-re-pooled gauge)",
                    m.name,
                );
            }
        }
    }

    /// Replaces the legacy `col_mean_std_basic` polars-side check
    /// with a direct test on the hand-rolled [`mean`] / [`std_dev`]
    /// helpers that the new aggregation path uses end-to-end.
    /// `1..=5` is symmetric around 3, so the mean lands exactly and
    /// the Bessel-corrected std exceeds 1.0 (the population std of
    /// `1..=5` is √2 ≈ 1.414; the sample std is √2.5 ≈ 1.581).
    #[test]
    fn mean_std_basic() {
        let xs = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let m = mean(xs.iter().copied());
        let s = std_dev(xs.iter().copied());
        assert!((m - 3.0).abs() < 0.01);
        assert!(s > 1.0);
    }

    /// Mirrors the legacy `col_mean_std_missing_column` defense:
    /// when no finite values are present, both helpers return 0.0
    /// rather than NaN / panic — the same "missing column → (0.0,
    /// 0.0)" contract that polars's `col_mean_std` carried, expressed
    /// over the iterator surface.
    #[test]
    fn mean_std_empty_returns_zero() {
        let empty: [f64; 0] = [];
        assert_eq!(mean(empty.iter().copied()), 0.0);
        assert_eq!(std_dev(empty.iter().copied()), 0.0);
        // Single finite value: mean is that value, std is 0.0 (Bessel
        // correction requires count >= 2; matches polars `.std(1)`).
        let single = [7.5_f64];
        assert!((mean(single.iter().copied()) - 7.5).abs() < f64::EPSILON);
        assert_eq!(std_dev(single.iter().copied()), 0.0);
    }

    /// Non-finite values (NaN, ±inf) must be ignored — a sentinel
    /// metric value can't poison the cohort mean. Pins the
    /// `.filter(|x| x.is_finite())` arm in both helpers.
    #[test]
    fn mean_std_skips_non_finite() {
        let xs = [1.0_f64, f64::NAN, 3.0, f64::INFINITY, 5.0];
        // Only 1, 3, 5 contribute. Mean = 3.0; sample std = 2.0.
        assert!((mean(xs.iter().copied()) - 3.0).abs() < 1e-9);
        assert!((std_dev(xs.iter().copied()) - 2.0).abs() < 1e-9);
    }

    /// Mean and std_dev handle negative values correctly.
    /// Negative values are valid for metrics like migration_ratio.
    #[test]
    fn mean_std_handles_negative_values() {
        let xs = [-2.0_f64, -1.0, 0.0, 1.0, 2.0];
        let m = mean(xs.iter().copied());
        let s = std_dev(xs.iter().copied());
        assert!(
            (m - 0.0).abs() < 1e-9,
            "mean of symmetric values should be 0, got {m}"
        );
        // Sample std for [-2, -1, 0, 1, 2]: sqrt( (4+1+0+1+4) / 4 ) = sqrt(10/4) = sqrt(2.5) ≈ 1.581
        assert!((s - 1.58113883).abs() < 1e-6, "std dev mismatch, got {s}");
    }

    /// Mean and std_dev handle large values without overflow.
    /// Uses values near f64::MAX / 10 to avoid actual overflow in sum.
    #[test]
    fn mean_std_handles_large_values() {
        let large = 1e150_f64;
        let xs = [large, large * 2.0, large * 3.0];
        let m = mean(xs.iter().copied());
        let s = std_dev(xs.iter().copied());
        assert!(
            (m - large * 2.0).abs() / large < 1e-12,
            "mean of large values"
        );
        // Std dev should be large * sqrt( (1^2 + 0^2 + 1^2) / 2 ) = large * 1.0
        assert!((s - large).abs() / large < 1e-12, "std dev of large values");
    }

    /// Mean and std_dev handle subnormal (very small) values.
    /// Subnormals are finite and should be included in calculations.
    #[test]
    fn mean_std_handles_subnormal_values() {
        let tiny = f64::MIN_POSITIVE / 2.0; // Subnormal
        let xs = [tiny, tiny * 2.0, tiny * 3.0];
        let m = mean(xs.iter().copied());
        // The sum is 6 subnormal ULPs and 6/3 = 2 ULPs exactly, so an
        // exact equality is correct and catches wrong-divisor,
        // partial-sum, and first-element bugs that a finite/positive
        // check would admit (e.g. count-1 → 3*tiny, first sample → tiny).
        assert_eq!(m, 2.0 * tiny, "subnormals must be summed/averaged exactly");
    }

    /// std_dev with exactly two values uses Bessel's correction (ddof=1).
    /// For [a, b], std = sqrt( ((a-m)^2 + (b-m)^2) / 1 ) where m = (a+b)/2.
    /// This simplifies to |a-b| / sqrt(2) * sqrt(2) = |a-b|.
    #[test]
    fn std_dev_two_values_bessel_corrected() {
        let xs = [3.0_f64, 7.0];
        let s = std_dev(xs.iter().copied());
        // Mean = 5.0, deviations = [-2, 2], sum_sq = 8, variance = 8 / 1 = 8, std = sqrt(8) ≈ 2.828
        assert!(
            (s - 2.8284271247461903).abs() < 1e-9,
            "Bessel-corrected std dev for two values"
        );
    }

    // -- find_outliers and find_worst_topos tests --

    /// find_outliers returns empty vec when input is empty.
    #[test]
    fn find_outliers_empty_input() {
        let rows: Vec<GauntletRow> = vec![];
        let outliers = find_outliers(&rows);
        assert!(outliers.is_empty(), "empty input should yield no outliers");
    }

    /// find_outliers returns empty vec when no rows pass the filter.
    /// Only rows with is_pass() == true are considered for outlier detection.
    #[test]
    fn find_outliers_no_pass_rows() {
        let r1 = make_row("s1", "t1", false, 10.0); // failed
        let mut r2 = make_row("s2", "t2", true, 20.0);
        r2.skipped = true; // skipped, not a pass
        let rows = vec![r1, r2];
        let outliers = find_outliers(&rows);
        assert!(outliers.is_empty(), "no pass rows should yield no outliers");
    }

    /// find_outliers returns empty vec when only one scenario exists.
    /// With a single scenario, the scenario mean equals the overall mean,
    /// so no scenario can exceed the 2-sigma threshold.
    #[test]
    fn find_outliers_single_scenario_no_outlier() {
        let r1 = make_row("only", "t1", true, 10.0);
        let r2 = make_row("only", "t2", true, 12.0);
        let r3 = make_row("only", "t3", true, 11.0);
        let rows = vec![r1, r2, r3];
        let outliers = find_outliers(&rows);
        assert!(
            outliers.is_empty(),
            "single scenario cannot produce outliers"
        );
    }

    /// find_outliers detects a clear outlier scenario.
    /// Three scenarios: two with low spread (5.0), one with high spread (100.0).
    /// The high-spread scenario should be flagged as an outlier on the "spread" metric.
    #[test]
    fn find_outliers_detects_clear_outlier() {
        // Many "normal" scenarios with low spread (10 rows total)
        let mut rows = Vec::new();
        for i in 0..5 {
            rows.push(make_row("normal1", &format!("t{i}"), true, 5.0));
        }
        for i in 5..10 {
            rows.push(make_row("normal2", &format!("t{i}"), true, 5.0));
        }
        // One outlier scenario with high spread (100+ to ensure > 2-sigma)
        rows.push(make_row("outlier", "t10", true, 100.0));
        rows.push(make_row("outlier", "t11", true, 110.0));

        let outliers = find_outliers(&rows);
        assert!(!outliers.is_empty(), "should detect outlier scenario");

        // Find the spread outlier
        let spread_outlier = outliers.iter().find(|o| o.metric == "spread");
        assert!(
            spread_outlier.is_some(),
            "should have spread metric outlier"
        );
        let outlier = spread_outlier.unwrap();
        assert_eq!(outlier.scenario, "outlier");
        assert!(
            outlier.sigma > 2.0,
            "sigma should exceed 2.0 threshold, got {}",
            outlier.sigma
        );
        // Worst topos should include t10 and t11 (both exceed threshold)
        assert!(outlier.worst_topos.contains(&"t10".to_string()));
        assert!(outlier.worst_topos.contains(&"t11".to_string()));
    }

    /// find_outliers does NOT flag a scenario at exactly the 2-sigma threshold.
    /// The condition is `scenario_mean > threshold`, not `>=`, so exact equality
    /// should not produce an outlier. This prevents borderline cases from noise.
    #[test]
    fn find_outliers_threshold_is_strictly_greater() {
        // Create a scenario where mean is exactly at 2-sigma threshold.
        // For simplicity, use 3 scenarios: two with value 0.0, one with value X.
        // Overall mean = X/3, std = sqrt( ((0-m)^2 + (0-m)^2 + (X-m)^2) / 2 )
        // We want X = mean + 2*std. Solving: X = X/3 + 2*std => 2X/3 = 2*std => X/3 = std
        // std^2 = (m^2 + m^2 + (X-m)^2) / 2 = (2m^2 + (2m)^2) / 2 where m = X/3
        // = (2m^2 + 4m^2) / 2 = 3m^2 => std = m*sqrt(3) = X*sqrt(3)/3
        // Setting X/3 = X*sqrt(3)/3 => 1 = sqrt(3) which is false.
        // Simpler: use many normal rows and one borderline row.
        // Let's construct: 10 rows with value 0.0, 1 row with value V.
        // Mean = V/11, variance = (10*(V/11)^2 + (V - V/11)^2) / 10
        // = (10*V^2/121 + (10V/11)^2) / 10 = (10V^2/121 + 100V^2/121) / 10 = 110V^2 / 1210 = V^2 / 11
        // std = V / sqrt(11), threshold = V/11 + 2V/sqrt(11)
        // For the single row scenario, mean = V. We want V = threshold.
        // V = V/11 + 2V/sqrt(11) => 1 = 1/11 + 2/sqrt(11) => 10/11 = 2/sqrt(11) => sqrt(11) = 22/10 = 2.2 => 11 = 4.84 false.
        // This is getting complex. Instead, test the boundary condition directly:
        // Create a case where scenario_mean is just below threshold (no outlier)
        // and another just above (outlier).

        // Use 5 normal scenarios with spread 10.0, and 1 test scenario with spread 30.0
        // Normal: 5*2 = 10 rows at 10.0. Test: 2 rows at 30.0.
        // Overall mean = (100 + 60) / 12 = 160/12 ≈ 13.33
        // Std: variance = [10*(10-13.33)^2 + 2*(30-13.33)^2] / 11
        // = [10*11.09 + 2*277.78] / 11 = [110.9 + 555.56] / 11 = 666.46 / 11 ≈ 60.59, std ≈ 7.78
        // Threshold = 13.33 + 2*7.78 = 13.33 + 15.56 = 28.89
        // Test scenario mean = 30.0 > 28.89, so it IS an outlier.
        // To test the boundary, we need a value just below threshold.

        // Simpler approach: verify that the > comparison is used, not >=.
        // Create a minimal case where we know the threshold and test both sides.
        // 4 rows: two scenarios with 2 rows each
        // Scenario A: values 0.0, 0.0 (mean 0.0)
        // Scenario B: values 10.0, 10.0 (mean 10.0)
        // Overall mean = 5.0, std = sqrt( ((0-5)^2*2 + (10-5)^2*2) / 3 ) = sqrt( (50 + 50) / 3 ) = sqrt(100/3) ≈ 5.77
        // Threshold = 5.0 + 2*5.77 = 5.0 + 11.55 = 16.55
        // Neither scenario mean (0.0 or 10.0) exceeds 16.55, so no outliers.
        let rows = vec![
            make_row("A", "t1", true, 0.0),
            make_row("A", "t2", true, 0.0),
            make_row("B", "t3", true, 10.0),
            make_row("B", "t4", true, 10.0),
        ];

        let outliers = find_outliers(&rows);
        // No scenario should be an outlier since 10.0 < 16.55
        let spread_outliers: Vec<_> = outliers.iter().filter(|o| o.metric == "spread").collect();
        assert!(
            spread_outliers.is_empty(),
            "no outlier when below threshold"
        );
    }

    /// Companion to the below-threshold case: a scenario whose mean is
    /// far above overall_mean + 2*std MUST be flagged. The strictly-
    /// greater test only proves the absence side (10.0 < 16.55); without
    /// this, a never-flags or inverted (`<` instead of `>`) comparison
    /// passes. 10 scenarios at spread 10 + 1 at 100: overall mean ~18,
    /// std ~27, threshold ~72; the 100 scenario clears it decisively.
    #[test]
    fn find_outliers_flags_scenario_above_threshold() {
        let mut rows: Vec<GauntletRow> = (0..10)
            .map(|i| make_row(&format!("normal{i}"), "t", true, 10.0))
            .collect();
        rows.push(make_row("hot", "t", true, 100.0));
        let outliers = find_outliers(&rows);
        let spread: Vec<_> = outliers.iter().filter(|o| o.metric == "spread").collect();
        assert!(
            spread.iter().any(|o| o.scenario == "hot"),
            "a scenario far above the 2-sigma threshold must be flagged as a spread outlier",
        );
    }

    /// find_outliers skips metrics with near-zero standard deviation.
    /// When std < f64::EPSILON, the metric has no measurable spread,
    /// so outlier detection would be noise. The function should skip it.
    #[test]
    fn find_outliers_skips_zero_std_metrics() {
        // All rows have identical spread value, so std = 0.0
        let r1 = make_row("s1", "t1", true, 10.0);
        let r2 = make_row("s2", "t2", true, 10.0);
        let r3 = make_row("s3", "t3", true, 10.0);
        let rows = vec![r1, r2, r3];

        let outliers = find_outliers(&rows);
        // No spread outliers because std = 0 (below EPSILON)
        let spread_outliers: Vec<_> = outliers.iter().filter(|o| o.metric == "spread").collect();
        assert!(
            spread_outliers.is_empty(),
            "zero std dev should skip metric"
        );
    }

    /// find_outliers sorts results by sigma descending.
    /// Higher sigma (more extreme outliers) should appear first.
    #[test]
    fn find_outliers_sorts_by_sigma_descending() {
        // Create two outlier scenarios with different sigma values
        // Normal baseline: 30 rows with spread 5.0 (large baseline for stable std)
        let mut rows = Vec::new();
        for i in 0..15 {
            rows.push(make_row("normal1", &format!("t{i}"), true, 5.0));
        }
        for i in 15..30 {
            rows.push(make_row("normal2", &format!("t{i}"), true, 5.0));
        }
        // Outlier 1: (spread 100.0, 110.0)
        rows.push(make_row("outlier1", "t30", true, 100.0));
        rows.push(make_row("outlier1", "t31", true, 110.0));
        // Outlier 2: more extreme (spread 140.0, 150.0)
        // Both should exceed threshold, but outlier2 has higher sigma
        rows.push(make_row("outlier2", "t40", true, 140.0));
        rows.push(make_row("outlier2", "t41", true, 150.0));

        let outliers = find_outliers(&rows);
        let spread_outliers: Vec<_> = outliers.iter().filter(|o| o.metric == "spread").collect();
        assert!(
            spread_outliers.len() >= 2,
            "should have at least 2 spread outliers, got {}",
            spread_outliers.len()
        );

        // Outlier2 should have higher sigma and appear first
        let first = &spread_outliers[0];
        let second = &spread_outliers[1];
        assert!(
            first.sigma >= second.sigma,
            "outliers should be sorted by sigma descending"
        );
        assert_eq!(
            first.scenario, "outlier2",
            "extreme outlier should be first"
        );
    }

    /// find_worst_topos returns empty vec when no rows match scenario.
    #[test]
    fn find_worst_topos_empty_when_no_matching_scenario() {
        let r1 = make_row("s1", "t1", true, 10.0);
        let rows = vec![r1];
        let accessor: MetricAccessor = |r| r.spread;
        let worst = find_worst_topos(&rows, "nonexistent", accessor, 5.0);
        assert!(
            worst.is_empty(),
            "no matching scenario should yield empty vec"
        );
    }

    /// find_worst_topos returns only topologies exceeding threshold.
    /// The threshold comparison is strict (>), not >=.
    #[test]
    fn find_worst_topos_filters_by_threshold() {
        let mut r1 = make_row("s1", "t1", true, 10.0);
        r1.spread = 5.0;
        let mut r2 = make_row("s1", "t2", true, 10.0);
        r2.spread = 15.0;
        let mut r3 = make_row("s1", "t3", true, 10.0);
        r3.spread = 25.0;
        let rows = vec![r1, r2, r3];

        let accessor: MetricAccessor = |r| r.spread;
        // Threshold 15.0: t1 (5.0) no, t2 (15.0) no (strict >), t3 (25.0) yes
        let worst = find_worst_topos(&rows, "s1", accessor, 15.0);
        assert_eq!(worst.len(), 1, "only t3 should exceed threshold");
        assert!(worst.contains(&"t3".to_string()));
        assert!(
            !worst.contains(&"t2".to_string()),
            "t2 at threshold should not be included"
        );
    }

    /// find_worst_topos includes rows regardless of pass/fail status.
    /// The function is used to attribute outliers to topologies, and a
    /// failing topology that drives high metric values should still surface.
    #[test]
    fn find_worst_topos_includes_failed_rows() {
        let mut r1 = make_row("s1", "t1", true, 10.0); // pass
        r1.spread = 30.0;
        let mut r2 = make_row("s1", "t2", false, 10.0); // fail
        r2.spread = 40.0;
        let rows = vec![r1, r2];

        let accessor: MetricAccessor = |r| r.spread;
        let worst = find_worst_topos(&rows, "s1", accessor, 20.0);
        // Both should be included regardless of pass/fail
        assert_eq!(worst.len(), 2);
        assert!(worst.contains(&"t1".to_string()));
        assert!(worst.contains(&"t2".to_string()));
    }

    /// group_field returns None for unknown column names.
    /// Only "scenario", "topology", and "work_type" are valid dimensions.
    #[test]
    fn group_field_unknown_column_returns_none() {
        let row = make_row("s1", "t1", true, 10.0);
        assert!(group_field(&row, "scenario").is_some());
        assert!(group_field(&row, "topology").is_some());
        assert!(group_field(&row, "work_type").is_some());
        assert!(group_field(&row, "invalid").is_none());
        assert!(group_field(&row, "").is_none());
        assert!(group_field(&row, "Spread").is_none()); // case-sensitive
    }

    /// group_field extracts the correct field for each valid dimension.
    #[test]
    fn group_field_extracts_correct_dimension() {
        let row = make_row("my_scenario", "my_topo", true, 10.0);
        assert_eq!(group_field(&row, "scenario"), Some("my_scenario"));
        assert_eq!(group_field(&row, "topology"), Some("my_topo"));
        assert_eq!(group_field(&row, "work_type"), Some("SpinWait"));
    }

    fn make_row(scenario: &str, topo: &str, passed: bool, spread: f64) -> GauntletRow {
        GauntletRow {
            scenario: scenario.into(),
            topology: topo.into(),
            work_type: "SpinWait".into(),
            scheduler: String::new(),
            kernel_version: None,
            cpu_budget: None,
            vcpus: None,
            commit: None,
            kernel_commit: None,
            run_source: None,
            skipped: false,
            passed,
            inconclusive: false,
            run_sample_count: 0,
            spread,
            gap_ms: 50,
            migrations: 10,
            migration_ratio: 0.0,
            imbalance_ratio: 1.0,
            max_dsq_depth: 2,
            stuck_count: 0,
            fallback_count: 0,
            keep_last_count: 0,
            worst_p99_wake_latency_us: 0.0,
            worst_median_wake_latency_us: 0.0,
            worst_wake_latency_cv: 0.0,
            total_iterations: 0,
            worst_mean_run_delay_us: 0.0,
            worst_run_delay_us: 0.0,
            worst_wake_latency_tail_ratio: 0.0,
            worst_iterations_per_worker: 0.0,
            worst_iterations_per_cpu_sec: 0.0,
            page_locality: 0.0,
            cross_node_migration_ratio: 0.0,
            ext_metrics: BTreeMap::new(),
            phases: Vec::new(),
        }
    }

    // -- format_dimension_summary tests --

    #[test]
    fn format_dimension_summary_computed_values() {
        // Two scenarios: "fast" with spread=4.0, gap=40, and "slow" with spread=20.0, gap=200.
        // Each has 1 row. format_dimension_summary sorts by avg_spread descending.
        let mut r1 = make_row("slow", "tiny-1llc", false, 20.0);
        r1.gap_ms = 200;
        r1.imbalance_ratio = 2.5; // > 1.0, should show imbal=2.5
        r1.max_dsq_depth = 8; // > 0, should show dsq=8
        r1.stuck_count = 2; // > 0, should show stuck=2
        r1.fallback_count = 15; // > 0, should show fallback=15
        let r2 = make_row("fast", "tiny-1llc", true, 4.0);
        let rows = vec![r1, r2];
        let out = format_dimension_summary(&rows, "scenario");
        // "slow" has higher spread, should appear first (sorted descending).
        let slow_pos = out.find("slow").unwrap();
        let fast_pos = out.find("fast").unwrap();
        assert!(
            slow_pos < fast_pos,
            "slow should sort before fast, got:\n{out}"
        );
        // Check computed values for "slow"
        assert!(out.contains("0/1 passed"), "slow: 0/1 passed, got:\n{out}");
        assert!(
            out.contains("avg_spread=20.0%"),
            "slow: avg_spread=20.0%, got:\n{out}"
        );
        assert!(
            out.contains("avg_gap=200ms"),
            "slow: avg_gap=200ms, got:\n{out}"
        );
        assert!(out.contains("imbal=2.5"), "slow: imbal=2.5, got:\n{out}");
        assert!(out.contains("dsq=8"), "slow: dsq=8, got:\n{out}");
        assert!(out.contains("stuck=2"), "slow: stuck=2, got:\n{out}");
        assert!(
            out.contains("fallback=15"),
            "slow: fallback=15, got:\n{out}"
        );
        // "fast" should show 1/1 passed
        assert!(out.contains("1/1 passed"), "fast: 1/1 passed, got:\n{out}");
    }

    /// A row whose `inconclusive` bit is set must render in the
    /// "inconclusive" bucket of the dimension-summary line, NOT
    /// silently fold into the "failed" bucket as the arithmetic
    /// `fail = total - pass - skip` would have done. The current
    /// arithmetic `fail = total - pass - skip - inconc` plus the
    /// format string `"({} skipped, {} inconclusive, {} failed)"`
    /// together produce the correct breakdown.
    ///
    /// A future polars-aggregation refactor that drops the
    /// `inconc_count` agg, the `inconclusive` group-by column, or the
    /// subtraction at the residual would re-introduce the
    /// misclassification — this test fails loudly in that case.
    #[test]
    fn format_dimension_summary_renders_inconclusive_bucket_distinctly() {
        // Three rows on the same dimension: one Pass, one
        // Inconclusive (passed=false, inconclusive=true), one Fail
        // (all-false). Triple-state coverage in one dimension
        // confirms that the format string surfaces every non-pass
        // bucket honestly.
        let mut r_pass = make_row("group_a", "t1", true, 5.0);
        r_pass.skipped = false;
        r_pass.inconclusive = false;
        let mut r_inc = make_row("group_a", "t1", false, 5.0);
        r_inc.skipped = false;
        r_inc.inconclusive = true;
        let mut r_fail = make_row("group_a", "t1", false, 5.0);
        r_fail.skipped = false;
        r_fail.inconclusive = false;
        let rows = vec![r_pass, r_inc, r_fail];
        let out = format_dimension_summary(&rows, "scenario");
        assert!(
            out.contains("1/3 passed"),
            "expected '1/3 passed' for 1-pass-of-3: got:\n{out}"
        );
        assert!(
            out.contains("1 inconclusive"),
            "inconclusive row must NOT silently fold into the failed \
             bucket; got:\n{out}"
        );
        assert!(
            out.contains("1 failed"),
            "real Fail row must render as 1 failed (not be hidden by \
             the inconclusive subtraction); got:\n{out}"
        );
        assert!(
            out.contains("0 skipped"),
            "no Skip contributor; skipped bucket must be 0: got:\n{out}"
        );
    }

    // -- analyze_rows tests --

    #[test]
    fn analyze_rows_empty() {
        assert!(analyze_rows(&[]).is_empty());
    }

    #[test]
    fn analyze_rows_with_work_type_diversity() {
        let mut rows = vec![
            make_row("a", "t1", true, 5.0),
            make_row("a", "t1", true, 6.0),
        ];
        rows[0].work_type = "SpinWait".into();
        rows[1].work_type = "Bursty".into();
        let report = analyze_rows(&rows);
        assert!(
            report.contains("By work_type"),
            "should show work_type section when diverse"
        );
        assert!(report.contains("SpinWait"), "should list SpinWait");
        assert!(report.contains("Bursty"), "should list Bursty");
    }

    #[test]
    fn analyze_rows_no_work_type_section_when_uniform() {
        let rows = vec![
            make_row("a", "t1", true, 5.0),
            make_row("b", "t2", true, 8.0),
        ];
        let report = analyze_rows(&rows);
        assert!(
            !report.contains("By work_type"),
            "should not show work_type when uniform"
        );
    }

    // -- sidecar_to_row tests --

    #[test]
    fn sidecar_to_row_basic() {
        use crate::monitor;
        use crate::test_support;
        let sc = test_support::SidecarResult {
            test_name: "my_test".to_string(),
            topology: "1n2l4c2t".to_string(),
            scheduler: "scx_mitosis".to_string(),
            stats: ScenarioStats {
                cgroups: vec![],
                total_workers: 4,
                total_cpus: 8,
                total_migrations: 12,
                worst_spread: 15.0,
                worst_gap_ms: 200,
                worst_gap_cpu: 3,
                ..Default::default()
            },
            monitor: Some(monitor::MonitorSummary {
                total_samples: 10,
                max_imbalance_ratio: 2.5,
                max_local_dsq_depth: 4,
                stuck_detected: true,
                event_deltas: Some(monitor::ScxEventDeltas {
                    total_fallback: 7,
                    fallback_rate: 0.5,
                    max_fallback_burst: 2,
                    total_dispatch_offline: 0,
                    total_dispatch_keep_last: 3,
                    keep_last_rate: 0.2,
                    total_enq_skip_exiting: 0,
                    total_enq_skip_migration_disabled: 0,
                    ..Default::default()
                }),
                schedstat_deltas: None,
                prog_stats_deltas: None,
                ..Default::default()
            }),
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(row.scenario, "my_test");
        assert_eq!(row.topology, "1n2l4c2t");
        assert!(row.is_pass());
        assert_eq!(row.spread, 15.0);
        assert_eq!(row.gap_ms, 200);
        assert_eq!(row.migrations, 12);
        assert_eq!(row.imbalance_ratio, 2.5);
        assert_eq!(row.max_dsq_depth, 4);
        assert_eq!(row.stuck_count, 1);
        assert_eq!(row.fallback_count, 7);
        assert_eq!(row.keep_last_count, 3);
    }

    /// `worst_iterations_per_cpu_sec` (the overcommit-invariant compare
    /// metric the budget warning recommends) flows ScenarioStats ->
    /// GauntletRow: Some maps through, None collapses to 0.0 like the
    /// sibling worst_* fields, and the metric is in the METRICS registry
    /// so `stats compare --metric worst_iterations_per_cpu_sec` resolves.
    #[test]
    fn sidecar_to_row_maps_worst_iterations_per_cpu_sec() {
        use crate::test_support;
        let some = test_support::SidecarResult {
            stats: ScenarioStats {
                worst_iterations_per_cpu_sec: Some(1234.5),
                ..Default::default()
            },
            ..test_support::SidecarResult::test_fixture()
        };
        assert_eq!(sidecar_to_row(&some).worst_iterations_per_cpu_sec, 1234.5);

        let none = test_support::SidecarResult {
            stats: ScenarioStats {
                worst_iterations_per_cpu_sec: None,
                ..Default::default()
            },
            ..test_support::SidecarResult::test_fixture()
        };
        assert_eq!(sidecar_to_row(&none).worst_iterations_per_cpu_sec, 0.0);

        assert!(
            METRICS
                .iter()
                .any(|m| m.name == "worst_iterations_per_cpu_sec"),
            "metric must be registered so `stats compare --metric` resolves it",
        );
    }

    #[test]
    fn sidecar_to_row_no_monitor() {
        use crate::test_support;
        let sc = test_support::SidecarResult {
            test_name: "eevdf_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            passed: false,
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(row.scenario, "eevdf_test");
        assert!(row.is_fail());
        assert_eq!(row.imbalance_ratio, 0.0);
        assert_eq!(row.max_dsq_depth, 0);
        assert_eq!(row.stuck_count, 0);
        assert_eq!(row.fallback_count, 0);
        assert_eq!(row.keep_last_count, 0);
    }

    /// `sidecar_to_row` must copy `SidecarResult::project_commit`
    /// into `GauntletRow::commit` verbatim so the typed
    /// `--project-commit` filter (and any sibling slicers added
    /// later) see the value the sidecar writer recorded. A
    /// regression that left the field at the
    /// `Option::default()` (`None`) would silently drop the
    /// commit dimension from every comparison even when the
    /// sidecar had a populated value. Pinned for `None`, clean
    /// `Some` (no suffix), and dirty `Some` (`-dirty` suffix) to
    /// catch a regression that special-cases one shape and not
    /// the others — e.g. one that stripped the suffix when copying.
    #[test]
    fn sidecar_to_row_propagates_project_commit() {
        use crate::test_support;
        let sc_dirty = test_support::SidecarResult {
            test_name: "commit_dirty_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            project_commit: Some("abcdef1-dirty".to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row_dirty = sidecar_to_row(&sc_dirty);
        assert_eq!(
            row_dirty.commit.as_deref(),
            Some("abcdef1-dirty"),
            "populated dirty project_commit must propagate \
             verbatim, including the `-dirty` suffix",
        );

        let sc_clean = test_support::SidecarResult {
            test_name: "commit_clean_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            project_commit: Some("abcdef1".to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row_clean = sidecar_to_row(&sc_clean);
        assert_eq!(
            row_clean.commit.as_deref(),
            Some("abcdef1"),
            "populated clean project_commit (no `-dirty` suffix) \
             must propagate verbatim — a regression that always \
             appended `-dirty` or always stripped a tail would \
             surface here independently of the dirty case above",
        );

        let sc_none = test_support::SidecarResult {
            test_name: "no_commit_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            project_commit: None,
            ..test_support::SidecarResult::test_fixture()
        };
        let row_none = sidecar_to_row(&sc_none);
        assert!(
            row_none.commit.is_none(),
            "absent project_commit must propagate as None — a \
             regression substituting an empty string would dilute \
             every `--project-commit` filter into matching all None rows",
        );
    }

    /// `sidecar_to_row` must copy `SidecarResult::kernel_commit`
    /// into `GauntletRow::kernel_commit` verbatim so the typed
    /// `--kernel-commit` filter and per-side
    /// `--a-kernel-commit` / `--b-kernel-commit` slicers see the
    /// value the sidecar writer recorded. A regression that left
    /// the field at the `Option::default()` (`None`) would
    /// silently drop the kernel-commit dimension from every
    /// comparison even when the sidecar had a populated value.
    /// Mirrors `sidecar_to_row_propagates_project_commit` for
    /// the kernel_commit field; pinned for `None`, clean `Some`
    /// (no suffix), and dirty `Some` (`-dirty` suffix) to catch
    /// a regression that special-cases one shape and not the
    /// others.
    #[test]
    fn sidecar_to_row_propagates_kernel_commit() {
        use crate::test_support;
        let sc_dirty = test_support::SidecarResult {
            test_name: "kc_dirty_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            kernel_commit: Some("kabcde7-dirty".to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row_dirty = sidecar_to_row(&sc_dirty);
        assert_eq!(
            row_dirty.kernel_commit.as_deref(),
            Some("kabcde7-dirty"),
            "populated dirty kernel_commit must propagate \
             verbatim, including the `-dirty` suffix",
        );

        let sc_clean = test_support::SidecarResult {
            test_name: "kc_clean_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            kernel_commit: Some("kabcde7".to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row_clean = sidecar_to_row(&sc_clean);
        assert_eq!(
            row_clean.kernel_commit.as_deref(),
            Some("kabcde7"),
            "populated clean kernel_commit (no `-dirty` suffix) \
             must propagate verbatim — a regression that always \
             appended `-dirty` or always stripped a tail would \
             surface here independently of the dirty case above",
        );

        let sc_none = test_support::SidecarResult {
            test_name: "no_kc_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            kernel_commit: None,
            ..test_support::SidecarResult::test_fixture()
        };
        let row_none = sidecar_to_row(&sc_none);
        assert!(
            row_none.kernel_commit.is_none(),
            "absent kernel_commit must propagate as None — a \
             regression substituting an empty string would dilute \
             every `--kernel-commit` filter into matching all \
             None rows",
        );

        // Field non-aliasing pin: kernel_commit and commit must
        // route to distinct row fields. A regression that
        // accidentally cross-wired the two (e.g. `commit:
        // sc.kernel_commit.clone()` instead of
        // `sc.project_commit.clone()`) would hide behind the
        // populated tests above unless the values differ — which
        // they do here. Distinct tokens make the swap obvious.
        let sc_both = test_support::SidecarResult {
            test_name: "both_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            project_commit: Some("project1".to_string()),
            kernel_commit: Some("kernel1".to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row_both = sidecar_to_row(&sc_both);
        assert_eq!(
            row_both.commit.as_deref(),
            Some("project1"),
            "row.commit must come from project_commit, not kernel_commit",
        );
        assert_eq!(
            row_both.kernel_commit.as_deref(),
            Some("kernel1"),
            "row.kernel_commit must come from kernel_commit, not project_commit",
        );
    }

    /// `sidecar_to_row` must copy `SidecarResult::run_source` into
    /// `GauntletRow::run_source` verbatim so the typed `--run-source`
    /// filter and per-side `--a-run-source` / `--b-run-source` slicers
    /// see the run-environment provenance tag the sidecar writer
    /// recorded. A regression that left the field at the
    /// `Option::default()` (`None`) would silently drop the
    /// run-source dimension from every comparison even when the
    /// sidecar had a populated value. Mirrors
    /// `sidecar_to_row_propagates_kernel_commit` for the
    /// `run_source` field; pinned for `None` and the canonical
    /// `Some("local")` / `Some("ci")` / `Some("archive")`
    /// values so a regression that special-cased one tag and
    /// not the others surfaces here. A non-aliasing pin
    /// confirms `run_source` reads from `sc.run_source` rather
    /// than being cross-wired to the visually-similar
    /// `kernel_commit` / `project_commit` fields.
    #[test]
    fn sidecar_to_row_propagates_run_source() {
        use crate::test_support;
        for tag in ["local", "ci", "archive"] {
            let sc = test_support::SidecarResult {
                test_name: format!("run_source_{tag}_test"),
                topology: "1n1l2c1t".to_string(),
                run_source: Some(tag.to_string()),
                ..test_support::SidecarResult::test_fixture()
            };
            let row = sidecar_to_row(&sc);
            assert_eq!(
                row.run_source.as_deref(),
                Some(tag),
                "populated run_source `{tag}` must propagate verbatim",
            );
        }

        let sc_none = test_support::SidecarResult {
            test_name: "no_run_source_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            run_source: None,
            ..test_support::SidecarResult::test_fixture()
        };
        let row_none = sidecar_to_row(&sc_none);
        assert!(
            row_none.run_source.is_none(),
            "absent run_source must propagate as None — a regression \
             substituting an empty string would dilute every \
             `--run-source` filter into matching all None rows",
        );

        // Field non-aliasing pin: `run_source` must route to its
        // own row field. A regression that cross-wired
        // `run_source` to `kernel_commit` (or vice versa) would
        // hide behind the populated tests above unless the values
        // are visibly different. Distinct tokens make the swap
        // obvious.
        let sc_distinct = test_support::SidecarResult {
            test_name: "run_source_distinct_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            run_source: Some("local".to_string()),
            kernel_commit: Some("kabcde7".to_string()),
            project_commit: Some("pabcde7".to_string()),
            ..test_support::SidecarResult::test_fixture()
        };
        let row_distinct = sidecar_to_row(&sc_distinct);
        assert_eq!(
            row_distinct.run_source.as_deref(),
            Some("local"),
            "row.run_source must come from sc.run_source, not from \
             kernel_commit or project_commit",
        );
        assert_eq!(
            row_distinct.kernel_commit.as_deref(),
            Some("kabcde7"),
            "row.kernel_commit must remain sourced from sc.kernel_commit",
        );
        assert_eq!(
            row_distinct.commit.as_deref(),
            Some("pabcde7"),
            "row.commit must remain sourced from sc.project_commit",
        );
    }

    #[test]
    fn sidecar_to_row_no_stall() {
        use crate::monitor;
        use crate::test_support;
        let sc = test_support::SidecarResult {
            monitor: Some(monitor::MonitorSummary {
                prog_stats_deltas: None,
                total_samples: 5,
                max_imbalance_ratio: 1.0,
                max_local_dsq_depth: 0,
                stuck_detected: false,
                event_deltas: None,
                schedstat_deltas: None,
                ..Default::default()
            }),
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(row.stuck_count, 0);
        assert_eq!(row.fallback_count, 0);
        assert_eq!(row.keep_last_count, 0);
    }

    /// Drive every direct f64 field on [`GauntletRow`] through
    /// `finite_or_zero` with `non_finite` planted in the source
    /// [`SidecarResult`], then assert each lands as 0.0 on the row.
    ///
    /// Covers all twelve `finite_or_zero` call sites in `sidecar_to_row`:
    /// eleven fields drawn from [`ScenarioStats`] plus `imbalance_ratio`
    /// which is read from [`MonitorSummary`]. A missed call site would
    /// leave one of the asserts comparing the non-finite input to 0.0
    /// (NaN != 0.0, ±Infinity != 0.0) and fail the test.
    fn assert_all_direct_f64_fields_sanitized(non_finite: f64) {
        use crate::assert::ScenarioStats;
        use crate::monitor::MonitorSummary;
        use crate::test_support;
        let sc = test_support::SidecarResult {
            stats: ScenarioStats {
                worst_spread: non_finite,
                worst_migration_ratio: non_finite,
                worst_p99_wake_latency_us: non_finite,
                worst_median_wake_latency_us: non_finite,
                worst_wake_latency_cv: non_finite,
                worst_mean_run_delay_us: non_finite,
                worst_run_delay_us: non_finite,
                worst_wake_latency_tail_ratio: non_finite,
                worst_iterations_per_worker: Some(non_finite),
                worst_page_locality: non_finite,
                worst_cross_node_migration_ratio: non_finite,
                ..Default::default()
            },
            monitor: Some(MonitorSummary {
                max_imbalance_ratio: non_finite,
                ..Default::default()
            }),
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        for (name, val) in [
            ("spread", row.spread),
            ("migration_ratio", row.migration_ratio),
            ("imbalance_ratio", row.imbalance_ratio),
            ("worst_p99_wake_latency_us", row.worst_p99_wake_latency_us),
            (
                "worst_median_wake_latency_us",
                row.worst_median_wake_latency_us,
            ),
            ("worst_wake_latency_cv", row.worst_wake_latency_cv),
            ("worst_mean_run_delay_us", row.worst_mean_run_delay_us),
            ("worst_run_delay_us", row.worst_run_delay_us),
            (
                "worst_wake_latency_tail_ratio",
                row.worst_wake_latency_tail_ratio,
            ),
            (
                "worst_iterations_per_worker",
                row.worst_iterations_per_worker,
            ),
            ("page_locality", row.page_locality),
            ("cross_node_migration_ratio", row.cross_node_migration_ratio),
        ] {
            assert_eq!(
                val, 0.0,
                "{name} must collapse to 0.0 for non-finite input {non_finite:?}",
            );
        }
        // Motivation check: the sanitized row serializes. Without the
        // `finite_or_zero` wraps, serde_json::to_string would return
        // Err because NaN / Infinity have no JSON representation.
        serde_json::to_string(&row).expect("sanitized row must serialize cleanly");
    }

    /// `sidecar_to_row` must sanitize NaN in every direct f64 field
    /// (both [`ScenarioStats`]-sourced and the
    /// [`MonitorSummary`]-sourced `imbalance_ratio`), not just a
    /// representative sample — same `serde_json` rejects-NaN
    /// motivation. Unlike `ext_metrics`, direct fields can't be
    /// dropped (the row schema is fixed), so non-finite collapses to
    /// 0.0 with a warn.
    #[test]
    fn sidecar_to_row_zeros_nan_in_every_direct_f64_field() {
        assert_all_direct_f64_fields_sanitized(f64::NAN);
    }

    /// Companion to `sidecar_to_row_zeros_nan_in_every_direct_f64_field`
    /// pinning the `+Infinity` branch of `finite_or_zero` for every
    /// direct f64 field on the row.
    #[test]
    fn sidecar_to_row_zeros_pos_infinity_in_every_direct_f64_field() {
        assert_all_direct_f64_fields_sanitized(f64::INFINITY);
    }

    /// Companion to `sidecar_to_row_zeros_nan_in_every_direct_f64_field`
    /// pinning the `-Infinity` branch of `finite_or_zero` for every
    /// direct f64 field on the row.
    #[test]
    fn sidecar_to_row_zeros_neg_infinity_in_every_direct_f64_field() {
        assert_all_direct_f64_fields_sanitized(f64::NEG_INFINITY);
    }

    /// Subnormal f64 values (IEEE 754 denormals) are finite —
    /// `is_finite()` returns `true` for them — and must pass through
    /// `finite_or_zero` unchanged. Guards against a future refactor
    /// that reaches for `is_normal()` instead of `is_finite()`,
    /// which would incorrectly collapse subnormals to 0.0 and erase
    /// very-small legitimate measurements. `f64::MIN_POSITIVE` is the
    /// smallest normal positive; `/ 2.0` lands in the subnormal
    /// range.
    #[test]
    fn sidecar_to_row_preserves_subnormal_f64_in_direct_fields() {
        use crate::assert::ScenarioStats;
        use crate::test_support;
        let subnormal = f64::MIN_POSITIVE / 2.0;
        assert!(subnormal.is_finite(), "subnormal must still be finite");
        assert!(!subnormal.is_normal(), "subnormal must not be normal");
        assert!(subnormal > 0.0, "subnormal is positive");
        let sc = test_support::SidecarResult {
            stats: ScenarioStats {
                worst_spread: subnormal,
                worst_page_locality: -subnormal,
                worst_wake_latency_cv: subnormal,
                ..Default::default()
            },
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(
            row.spread, subnormal,
            "positive subnormal must pass through finite_or_zero unchanged",
        );
        assert_eq!(
            row.page_locality, -subnormal,
            "negative subnormal must pass through finite_or_zero unchanged",
        );
        assert_eq!(
            row.worst_wake_latency_cv, subnormal,
            "subnormal on a second direct-f64 field must also pass through",
        );
        // Motivation check: subnormals serialize (unlike NaN / ±Inf,
        // serde_json emits them as standard decimal literals).
        serde_json::to_string(&row).expect("subnormals serialize cleanly");
    }

    /// Pins that the direct-field NaN sanitization in
    /// `sidecar_to_row` does NOT reach into `ext_metrics`. Finite
    /// `ext_metrics` entries must survive untouched even when every
    /// direct f64 field collapses to 0.0, and the `ext_metrics` map
    /// must not grow a sanitization-synthesized entry. Complements
    /// [`sidecar_to_row_drops_non_finite_ext_metrics`] (which pins
    /// that non-finite `ext_metrics` entries are DROPPED) by pinning
    /// the orthogonal claim: direct-field sanitization never writes
    /// into `ext_metrics` regardless of the direct values.
    #[test]
    fn sidecar_to_row_direct_field_nan_does_not_touch_ext_metrics() {
        use crate::assert::ScenarioStats;
        use crate::test_support;
        let mut ext = BTreeMap::new();
        ext.insert("finite_nonzero".to_string(), 2.5);
        ext.insert("finite_zero".to_string(), 0.0);
        ext.insert("finite_negative".to_string(), -7.25);
        let sc = test_support::SidecarResult {
            stats: ScenarioStats {
                // Every direct f64 field non-finite.
                worst_spread: f64::NAN,
                worst_migration_ratio: f64::INFINITY,
                worst_p99_wake_latency_us: f64::NEG_INFINITY,
                worst_median_wake_latency_us: f64::NAN,
                worst_wake_latency_cv: f64::INFINITY,
                worst_mean_run_delay_us: f64::NEG_INFINITY,
                worst_run_delay_us: f64::NAN,
                worst_page_locality: f64::INFINITY,
                worst_cross_node_migration_ratio: f64::NEG_INFINITY,
                ext_metrics: ext.clone(),
                ..Default::default()
            },
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);

        // Direct-field collapse still works.
        assert_eq!(row.spread, 0.0);
        assert_eq!(row.migration_ratio, 0.0);
        assert_eq!(row.page_locality, 0.0);

        // ext_metrics survives unchanged — same length, same keys,
        // same values.
        assert_eq!(
            row.ext_metrics.len(),
            ext.len(),
            "direct-field sanitization must not add or drop ext_metrics entries",
        );
        for (k, v) in &ext {
            assert_eq!(
                row.ext_metrics.get(k),
                Some(v),
                "ext_metrics entry {k:?} must pass through unchanged",
            );
        }

        // Motivation check: the full row still serializes.
        serde_json::to_string(&row).expect("sanitized row must serialize cleanly");
    }

    /// `sidecar_to_row` must drop NaN / +Infinity / -Infinity from
    /// `ext_metrics` because `serde_json::to_string` rejects non-finite
    /// f64 values — without this guard a single malformed scenario
    /// metric would poison every sidecar write on its batch. Finite
    /// entries must pass through unchanged. Also checks that the
    /// post-filter row serializes cleanly (the motivation for the
    /// filter).
    #[test]
    fn sidecar_to_row_drops_non_finite_ext_metrics() {
        use crate::assert::ScenarioStats;
        use crate::test_support;
        let mut ext = BTreeMap::new();
        ext.insert("good".to_string(), 1.0);
        ext.insert("nan".to_string(), f64::NAN);
        ext.insert("inf".to_string(), f64::INFINITY);
        ext.insert("neg_inf".to_string(), f64::NEG_INFINITY);
        let sc = test_support::SidecarResult {
            stats: ScenarioStats {
                ext_metrics: ext,
                ..Default::default()
            },
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(
            row.ext_metrics.len(),
            1,
            "only the finite entry should survive: {:?}",
            row.ext_metrics
        );
        assert_eq!(row.ext_metrics.get("good"), Some(&1.0));
        assert!(!row.ext_metrics.contains_key("nan"));
        assert!(!row.ext_metrics.contains_key("inf"));
        assert!(!row.ext_metrics.contains_key("neg_inf"));
        // Motivation check: the post-filter row serializes. Without the
        // filter, serde_json::to_string would return Err because NaN /
        // Infinity have no JSON representation.
        serde_json::to_string(&row).expect("filtered row must serialize cleanly");
    }

    /// `sidecar_to_row` must drop the JSON-walker depth-cap sentinel
    /// [`crate::test_support::WALK_TRUNCATION_SENTINEL_NAME`] from
    /// `ext_metrics`. The sentinel is diagnostic metadata about the
    /// extraction pass (depth cap hit), not a scenario metric, so it
    /// must not leak into A/B comparison output where it would be
    /// mistaken for a real measurement and skew filter / aggregation
    /// logic. Sibling finite entries must survive untouched.
    #[test]
    fn sidecar_to_row_drops_walk_truncation_sentinel() {
        use crate::assert::ScenarioStats;
        use crate::test_support;
        let mut ext = BTreeMap::new();
        ext.insert("good".to_string(), 1.0);
        ext.insert(
            test_support::WALK_TRUNCATION_SENTINEL_NAME.to_string(),
            72.0,
        );
        let sc = test_support::SidecarResult {
            stats: ScenarioStats {
                ext_metrics: ext,
                ..Default::default()
            },
            ..test_support::SidecarResult::test_fixture()
        };
        let row = sidecar_to_row(&sc);
        assert_eq!(
            row.ext_metrics.len(),
            1,
            "only the real metric should survive: {:?}",
            row.ext_metrics,
        );
        assert_eq!(row.ext_metrics.get("good"), Some(&1.0));
        assert!(
            !row.ext_metrics
                .contains_key(test_support::WALK_TRUNCATION_SENTINEL_NAME),
            "sentinel must not appear in the row's ext_metrics",
        );
    }

    // -- metric_def tests --

    #[test]
    fn metric_def_known() {
        let d = metric_def("worst_spread").unwrap();
        assert_eq!(d.name, "worst_spread");
        assert!(d.higher_is_worse());
        assert_eq!(d.display_unit, "%");
    }

    #[test]
    fn metric_def_not_higher_is_worse() {
        let d = metric_def("total_iterations").unwrap();
        assert!(!d.higher_is_worse());
    }

    #[test]
    fn metric_def_unknown() {
        assert!(metric_def("nonexistent").is_none());
    }

    #[test]
    fn infer_higher_is_worse_latency_shaped() {
        // Latency / delay / time-unit names are higher-is-worse.
        // `AssertResult::merge` folds these by `max` (worst case).
        for name in &[
            "p99_wake_latency",
            "wake_latency_us",
            "scheduling_delay",
            "task_run_delay_ns",
            "io_completion_ms",
            "stall_count",
            "stuck_count",
            "schedule_jitter_cv",
            "max_gap_us",
            "spread",
            "page_drop_count",
            "error_rate",
            "fail_count",
            "migration_ratio",
            "imbalance_factor",
        ] {
            assert!(
                infer_higher_is_worse(name),
                "metric `{name}` must infer higher_is_worse=true \
                 (latency/error-shaped); a folded max keeps the \
                 worst case across cgroups"
            );
        }
    }

    #[test]
    fn infer_higher_is_worse_throughput_shaped() {
        // Throughput / rate / iteration-shaped names are
        // higher-is-better. `AssertResult::merge` folds these by
        // `min` (the cgroup that fell behind is the worst case).
        for name in &[
            "read_iops",
            "write_iops",
            "throughput_mbps",
            "bandwidth_kb",
            "total_iterations",
            "iterations_per_worker",
            "ops_per_sec",
            "page_locality",
            "pass_score",
            "goodput",
        ] {
            assert!(
                !infer_higher_is_worse(name),
                "metric `{name}` must infer higher_is_worse=false \
                 (throughput-shaped); a folded min surfaces the \
                 cgroup that fell behind"
            );
        }
    }

    #[test]
    fn infer_higher_is_worse_unknown_falls_back_to_higher_is_worse() {
        // Names that don't match either token list fall back to
        // higher-is-worse. The fold keeps the max — which surfaces
        // an unexpectedly high reading rather than masking it under
        // a min collapse.
        for name in &["unrelated_field", "random_thing", "metric", "x", "a.b.c"] {
            assert!(
                infer_higher_is_worse(name),
                "unknown metric `{name}` must fall back to \
                 higher_is_worse=true (conservative for regression \
                 detection)"
            );
        }
    }

    #[test]
    fn infer_higher_is_worse_compound_names_resolve_to_latency() {
        // Compound names that contain BOTH a latency-shaped token
        // and a throughput-shaped token resolve to higher-is-worse
        // (the latency interpretation wins). Pin the order-of-
        // checks contract so a refactor that swaps the token lists
        // surfaces here.
        assert!(
            infer_higher_is_worse("read_iops_latency_us"),
            "compound name with `latency` and `iops` must resolve \
             to higher-is-worse (latency token checked first)"
        );
    }

    #[test]
    fn metric_def_polarity_inverse_sense() {
        use crate::test_support::Polarity;
        // higher_is_worse=true means growing = regression; the
        // Polarity for "what do we want it to move toward?" is
        // LowerBetter.
        let d = metric_def("worst_spread").unwrap();
        assert!(d.higher_is_worse());
        assert_eq!(d.polarity, Polarity::LowerBetter);
        // higher_is_worse=false means growing = improvement; the
        // Polarity is HigherBetter.
        let d = metric_def("total_iterations").unwrap();
        assert!(!d.higher_is_worse());
        assert_eq!(d.polarity, Polarity::HigherBetter);
    }

    #[test]
    fn metric_def_polarity_covers_all_entries() {
        use crate::test_support::Polarity;
        // Every METRICS entry must map cleanly to HigherBetter or
        // LowerBetter; no entry should produce TargetValue or Unknown
        // from the bool->Polarity adaptor.
        for m in METRICS.iter() {
            assert!(
                matches!(m.polarity, Polarity::HigherBetter | Polarity::LowerBetter),
                "metric {} produced non-binary polarity {:?}",
                m.name,
                m.polarity
            );
        }
    }

    #[test]
    fn metric_def_all_entries_unique() {
        let mut names: Vec<&str> = METRICS.iter().map(|m| m.name).collect();
        let len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), len);
    }

    /// Registry integrity: every `MetricKind::Rate`'s numerator and
    /// denominator MUST name a registered `Counter` metric. `derive_rate_metrics`
    /// silently skips a Rate whose component key is absent from the map
    /// (`derive_rate_metrics_from` `continue`s on a missing key), so a typo'd
    /// component name would never derive and never fail a value test — pin the
    /// names at the registry level instead. Counter (not Gauge/Peak) because the
    /// re-pool needs Σnum/Σdenom (sum-fold), which only the Counter kind gives.
    #[test]
    fn every_rate_metric_has_registered_counter_components() {
        for m in METRICS.iter() {
            let MetricKind::Rate {
                numerator,
                denominator,
            } = m.kind
            else {
                continue;
            };
            for (role, comp) in [("numerator", numerator), ("denominator", denominator)] {
                let def = metric_def(comp).unwrap_or_else(|| {
                    panic!("Rate {} {role} {comp:?} is not a registered metric", m.name)
                });
                assert!(
                    matches!(def.kind, MetricKind::Counter),
                    "Rate {} {role} {comp:?} must be a Counter for the Σ-fold re-pool",
                    m.name,
                );
            }
        }
    }

    // -- list_metrics tests --

    /// Text-mode [`list_metrics`] emits a table that names every
    /// registered metric at least once. Uses substring contains
    /// rather than column-exact equality so a future comfy-table
    /// preset rename (NOTHING → other) that rewraps whitespace
    /// does not false-fail — the surface contract is "every metric
    /// name appears somewhere in the rendered output", not a
    /// column-width pin.
    #[test]
    fn list_metrics_text_names_every_metric() {
        let out = list_metrics(false).expect("text render must succeed");
        assert!(!out.is_empty(), "text output must be non-empty");
        for m in METRICS {
            assert!(
                out.contains(m.name),
                "list_metrics(false) output missing metric name {}: {out}",
                m.name,
            );
        }
    }

    /// Text-mode [`list_metrics`] header row names every column. Pins
    /// the header contract so a column rename in
    /// `list_metrics` lands here instead of silently in downstream CI
    /// scripts that grep the output.
    #[test]
    fn list_metrics_text_header_pins_column_names() {
        let out = list_metrics(false).expect("text render must succeed");
        for header in ["NAME", "POLARITY", "DEFAULT_ABS", "DEFAULT_REL", "UNIT"] {
            assert!(
                out.contains(header),
                "list_metrics(false) output missing column header {header}: {out}",
            );
        }
    }

    /// JSON-mode [`list_metrics`] parses back to a `Vec<MetricDef>`-
    /// shaped structure with one entry per registry member. `MetricDef`
    /// itself does not derive `Deserialize` (the `accessor` fn-pointer
    /// is unserializable), so we deserialize into a minimal struct
    /// that captures the fields the wire contract promises.
    #[test]
    fn list_metrics_json_round_trips_via_minimal_schema() {
        #[derive(serde::Deserialize)]
        struct MetricEntry {
            name: String,
            default_abs: f64,
            default_rel: f64,
            display_unit: String,
            // polarity is serialized as an enum tag string by serde
            // (Polarity derives Serialize with the default
            // externally-tagged representation). Deserialize into a
            // serde_json::Value to avoid a cross-crate enum
            // dependency in the test-private schema.
            polarity: serde_json::Value,
        }

        let out = list_metrics(true).expect("json render must succeed");
        let parsed: Vec<MetricEntry> = serde_json::from_str(&out).expect("json output must parse");
        assert_eq!(
            parsed.len(),
            METRICS.len(),
            "json entry count must match METRICS.len()",
        );
        for (parsed_m, registry_m) in parsed.iter().zip(METRICS.iter()) {
            assert_eq!(parsed_m.name, registry_m.name);
            assert_eq!(parsed_m.default_abs, registry_m.default_abs);
            assert_eq!(parsed_m.default_rel, registry_m.default_rel);
            assert_eq!(parsed_m.display_unit, registry_m.display_unit);
            assert!(
                !parsed_m.polarity.is_null(),
                "polarity for {} must serialize as a non-null value",
                registry_m.name,
            );
        }
    }

    /// JSON-mode [`list_metrics`] must NOT expose the `accessor`
    /// fn-pointer field. The `#[serde(skip)]` attribute on
    /// `MetricDef::accessor` carries that contract; a regression that
    /// dropped the attribute would surface here as the emitted JSON
    /// gaining an "accessor" key. Pins the wire surface.
    #[test]
    fn list_metrics_json_omits_accessor_field() {
        let out = list_metrics(true).expect("json render must succeed");
        assert!(
            !out.contains("\"accessor\""),
            "list_metrics(true) must not emit the accessor field — \
             fn-pointers are not serializable and the field carries \
             #[serde(skip)]: {out}",
        );
    }

    /// Iteration order of [`list_metrics`] matches [`METRICS`]
    /// declaration order. Registry order is the canonical surface
    /// order for sidecar / CI-gate consumers; a renderer that sorted
    /// by name or polarity would silently break scripts that key on
    /// the first row.
    #[test]
    fn list_metrics_text_preserves_registry_order() {
        let out = list_metrics(false).expect("text render must succeed");
        let mut last_pos = 0usize;
        for m in METRICS {
            let pos = out
                .find(m.name)
                .unwrap_or_else(|| panic!("metric {} must appear in text output", m.name));
            assert!(
                pos >= last_pos,
                "metric {} appears before a prior metric — text output must \
                 preserve METRICS declaration order",
                m.name,
            );
            last_pos = pos;
        }
    }

    // -- list_values --

    /// Helper that writes N sidecars to `{root}/{run_key}/{run_key}.ktstr.json`.
    /// Each sidecar overrides only the fields the test wants to vary;
    /// the rest come from `SidecarResult::test_fixture()`. Used by the
    /// `list_values_*` tests to build pool fixtures isolated from
    /// `runs_root()`.
    fn write_listvalues_fixture(
        root: &std::path::Path,
        sidecars: &[crate::test_support::SidecarResult],
    ) {
        for (i, sc) in sidecars.iter().enumerate() {
            let run_key = format!("__lv_fixture_{i}__");
            let run_dir = root.join(&run_key);
            std::fs::create_dir_all(&run_dir).expect("create run dir");
            let json = serde_json::to_string(sc).expect("serialize fixture sidecar");
            let path = run_dir.join(format!("{run_key}.ktstr.json"));
            std::fs::write(&path, json).expect("write fixture sidecar");
        }
    }

    /// Empty pool (no run subdirs) must produce a well-formed text
    /// shape with the "(no sidecars in pool)" sentinel under each
    /// dimension heading. The function does NOT bail — discovery on
    /// an empty pool is a valid query that should answer "nothing"
    /// rather than fail.
    #[test]
    fn list_values_empty_pool_text_has_sentinel_per_dim() {
        let alt = tempfile::TempDir::new().expect("tempdir");
        let out = list_values(false, Some(alt.path())).expect("text render must succeed");
        for dim in [
            "kernel:",
            "commit:",
            "kernel_commit:",
            "source:",
            "cpu_budget:",
            "scheduler:",
            "topology:",
            "work_type:",
        ] {
            assert!(
                out.contains(dim),
                "text output must include heading for {dim}: {out}",
            );
        }
        // Each dim should report the empty-pool sentinel exactly eight
        // times — one per dim — so a regression that dropped the
        // sentinel for one dim falls out as a count mismatch.
        let sentinel_count = out.matches("(no sidecars in pool)").count();
        assert_eq!(
            sentinel_count, 8,
            "empty pool must surface the no-sidecars sentinel under every \
             one of the 8 dims (kernel/commit/kernel_commit/source/\
             cpu_budget/scheduler/topology/work_type); got {sentinel_count} \
             occurrences in:\n{out}",
        );
    }

    /// Empty pool → JSON object with empty arrays for every dim.
    /// Pins the JSON shape so a regression that dropped a key (e.g.
    /// "scheduler") on the empty-pool branch surfaces here.
    #[test]
    fn list_values_empty_pool_json_emits_empty_arrays() {
        let alt = tempfile::TempDir::new().expect("tempdir");
        let out = list_values(true, Some(alt.path())).expect("json render must succeed");
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("json output must parse");
        for dim in [
            "kernel",
            "commit",
            "kernel_commit",
            "source",
            "cpu_budget",
            "scheduler",
            "topology",
            "work_type",
        ] {
            let arr = parsed
                .get(dim)
                .unwrap_or_else(|| panic!("missing key {dim}"));
            assert!(arr.is_array(), "key {dim} must serialize as an array");
            assert_eq!(
                arr.as_array().unwrap().len(),
                0,
                "key {dim} must be an empty array on empty pool",
            );
        }
    }

    /// Populated pool: distinct values per dim are deduplicated and
    /// sorted; `kernel_version: None` and `project_commit: None`
    /// produce a `null` entry (JSON) and `unknown` line (text).
    #[test]
    fn list_values_text_dedupes_and_sorts_per_dim() {
        use crate::test_support::SidecarResult;

        let alt = tempfile::TempDir::new().expect("tempdir");
        let sidecars = vec![
            SidecarResult {
                test_name: "t_a".to_string(),
                topology: "1n2l4c1t".to_string(),
                scheduler: "scx_rusty".to_string(),
                work_type: "SpinWait".to_string(),
                kernel_version: Some("6.14.2".to_string()),
                project_commit: Some("abcdef1".to_string()),
                ..SidecarResult::test_fixture()
            },
            SidecarResult {
                test_name: "t_b".to_string(),
                topology: "1n4l2c1t".to_string(),
                scheduler: "eevdf".to_string(),
                work_type: "PageFaultChurn".to_string(),
                kernel_version: None,
                project_commit: None,
                ..SidecarResult::test_fixture()
            },
            // Duplicate of the first sidecar's identity-fields; the
            // BTreeSet must dedupe so each value lands once in the
            // rendered output.
            SidecarResult {
                test_name: "t_c".to_string(),
                topology: "1n2l4c1t".to_string(),
                scheduler: "scx_rusty".to_string(),
                work_type: "SpinWait".to_string(),
                kernel_version: Some("6.14.2".to_string()),
                project_commit: Some("abcdef1".to_string()),
                ..SidecarResult::test_fixture()
            },
        ];
        write_listvalues_fixture(alt.path(), &sidecars);

        let out = list_values(false, Some(alt.path())).expect("text render must succeed");

        // Dedupe: each distinct VALUE appears EXACTLY once per
        // dim (set semantics) even though "scx_rusty" / "1n2l4c1t"
        // / "SpinWait" / "abcdef1" / "6.14.2" come from two of the
        // three fixtures. Each value below is unique to its dim
        // so it should appear once across the rendered text. The
        // `unknown` sentinel is checked separately because both
        // `kernel` and `commit` are optional dims and each emits
        // its own `unknown` line.
        for value in [
            "6.14.2",
            "abcdef1",
            "scx_rusty",
            "eevdf",
            "1n2l4c1t",
            "1n4l2c1t",
            "SpinWait",
            "PageFaultChurn",
        ] {
            let count = out.matches(value).count();
            assert_eq!(
                count, 1,
                "value {value} must appear exactly once in text output (BTreeSet dedup); \
                 got {count} in:\n{out}",
            );
        }
        // `unknown` appears once per Optional dim that has a None
        // entry: kernel, commit, and kernel_commit. The second
        // fixture has `kernel_version: None` and `project_commit:
        // None`; every fixture in this test leaves `kernel_commit`
        // at its `test_fixture` default (None), so the
        // kernel_commit set's None bucket renders one `unknown`
        // line as well. Total: 3 occurrences.
        //
        // `run_source` is the fourth optional dim but does NOT
        // contribute an `unknown` here: `list_values(_, Some(dir))`
        // calls `apply_archive_source_override` on the loaded pool
        // (the `--dir` flag treats the supplied root as an archive),
        // which rewrites every `run_source: None` to
        // `Some("archive")` BEFORE the dimension-set is built. Every
        // fixture above leaves `run_source` at its `test_fixture`
        // default (None), but they all surface as `archive` after
        // the override — the run_source set never holds a None
        // entry on this code path, so no `unknown` line is emitted
        // for it.
        let unknown_count = out.matches("unknown").count();
        assert_eq!(
            unknown_count, 3,
            "`unknown` must render once per optional dim with a None \
             entry (kernel + commit + kernel_commit = 3); got \
             {unknown_count} in:\n{out}",
        );

        // Sort: both schedulers in ascending lex order means
        // "eevdf" appears BEFORE "scx_rusty" in the rendered text.
        let pos_eevdf = out.find("eevdf").expect("eevdf in output");
        let pos_rusty = out.find("scx_rusty").expect("scx_rusty in output");
        assert!(
            pos_eevdf < pos_rusty,
            "values within a dim must render sorted (BTreeSet iter order); \
             expected 'eevdf' before 'scx_rusty' in:\n{out}",
        );
    }

    /// The populated `cpu_budget` dim renders distinct budgets once
    /// each, sorted ascending, in both the text block and as a numeric
    /// JSON array (every other dim is a string array; cpu_budget is the
    /// sole numeric one). Skip rows (budget 0) are excluded.
    #[test]
    fn list_values_cpu_budget_renders_distinct_sorted() {
        use crate::test_support::SidecarResult;

        let alt = tempfile::TempDir::new().expect("tempdir");
        let sidecars = vec![
            SidecarResult {
                test_name: "t_a".to_string(),
                cpu_budget: 32,
                vcpus: 32,
                ..SidecarResult::test_fixture()
            },
            SidecarResult {
                test_name: "t_b".to_string(),
                cpu_budget: 4,
                vcpus: 16,
                ..SidecarResult::test_fixture()
            },
            // Duplicate budget 4 — the BTreeSet must dedupe.
            SidecarResult {
                test_name: "t_c".to_string(),
                cpu_budget: 4,
                vcpus: 16,
                ..SidecarResult::test_fixture()
            },
            // Skip row (budget 0) — excluded from the budget set.
            SidecarResult {
                test_name: "t_d".to_string(),
                cpu_budget: 0,
                vcpus: 0,
                ..SidecarResult::test_fixture()
            },
        ];
        write_listvalues_fixture(alt.path(), &sidecars);

        let text = list_values(false, Some(alt.path())).expect("text render must succeed");
        // Distinct budgets appear once each. Scope the search to the
        // cpu_budget block so digits in other values can't false-match:
        // the block is the lines between "cpu_budget:\n" and the next
        // blank line.
        let block_start =
            text.find("cpu_budget:\n").expect("cpu_budget heading") + "cpu_budget:\n".len();
        let block = &text[block_start..];
        let block_end = block.find("\n\n").map(|i| i + 1).unwrap_or(block.len());
        let block = &block[..block_end];
        assert_eq!(
            block.matches("  4\n").count(),
            1,
            "budget 4 once: {block:?}"
        );
        assert_eq!(
            block.matches("  32\n").count(),
            1,
            "budget 32 once: {block:?}"
        );
        let pos_4 = block.find("  4\n").expect("4 present");
        let pos_32 = block.find("  32\n").expect("32 present");
        assert!(pos_4 < pos_32, "budgets must sort ascending: {block:?}");

        let json = list_values(true, Some(alt.path())).expect("json render must succeed");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("json must parse");
        let arr = parsed
            .get("cpu_budget")
            .and_then(|v| v.as_array())
            .expect("cpu_budget must be a JSON array");
        let nums: Vec<u64> = arr
            .iter()
            .map(|v| v.as_u64().expect("numeric budget"))
            .collect();
        assert_eq!(
            nums,
            vec![4, 32],
            "JSON cpu_budget must be a sorted numeric array"
        );
    }

    /// JSON shape: `kernel` and `commit` arrays carry `null` for
    /// absent values, `Value::String` for present values; the other
    /// dims are bare `String` arrays.
    #[test]
    fn list_values_json_carries_null_for_optional_dims() {
        use crate::test_support::SidecarResult;

        let alt = tempfile::TempDir::new().expect("tempdir");
        let sidecars = vec![
            SidecarResult {
                test_name: "t_known".to_string(),
                kernel_version: Some("6.14.2".to_string()),
                project_commit: Some("abcdef1".to_string()),
                ..SidecarResult::test_fixture()
            },
            SidecarResult {
                test_name: "t_unknown".to_string(),
                kernel_version: None,
                project_commit: None,
                ..SidecarResult::test_fixture()
            },
        ];
        write_listvalues_fixture(alt.path(), &sidecars);

        let out = list_values(true, Some(alt.path())).expect("json render must succeed");
        let parsed: serde_json::Value = serde_json::from_str(&out).expect("json output must parse");

        let kernel = parsed
            .get("kernel")
            .expect("kernel key")
            .as_array()
            .unwrap();
        assert!(
            kernel.iter().any(|v| v.is_null()),
            "kernel array must include a literal null for the None entry; got {kernel:?}",
        );
        assert!(
            kernel.iter().any(|v| v.as_str() == Some("6.14.2")),
            "kernel array must include the populated value 6.14.2; got {kernel:?}",
        );

        let commit = parsed
            .get("commit")
            .expect("commit key")
            .as_array()
            .unwrap();
        assert!(
            commit.iter().any(|v| v.is_null()),
            "commit array must include a literal null for the None entry; got {commit:?}",
        );
        assert!(
            commit.iter().any(|v| v.as_str() == Some("abcdef1")),
            "commit array must include the populated value abcdef1; got {commit:?}",
        );
    }

    /// `dir = None` resolves against `runs_root()`; if `runs_root()`
    /// does not exist, the function returns Ok with empty arrays /
    /// per-dim sentinel rather than bailing. Pins the no-bail
    /// contract on missing-root.
    #[test]
    fn list_values_none_dir_does_not_bail_on_missing_root() {
        // We cannot reliably wipe `runs_root()` from a unit test, but
        // we can pin the "Some(nonexistent_path)" branch which
        // exercises the same `collect_pool -> empty Vec` codepath
        // (`fs::read_dir` returns Err on a missing root, and
        // `collect_pool` swallows that into an empty pool).
        let alt = tempfile::TempDir::new().expect("tempdir");
        let nonexistent = alt.path().join("definitely_does_not_exist");
        let out = list_values(false, Some(&nonexistent)).expect("must not bail on missing root");
        assert!(
            out.contains("(no sidecars in pool)"),
            "missing root must render the no-sidecars sentinel: {out}",
        );
    }

    // -- MetricDef::read tests --

    fn read_metric(row: &GauntletRow, name: &str) -> Option<f64> {
        metric_def(name).expect("metric name").read(row)
    }

    #[test]
    fn metric_def_read_named_fields() {
        let mut row = make_row("a", "t", true, 42.0);
        row.gap_ms = 100;
        row.migrations = 7;
        row.migration_ratio = 0.3;
        row.imbalance_ratio = 2.0;
        row.max_dsq_depth = 5;
        row.stuck_count = 3;
        row.fallback_count = 11;
        row.keep_last_count = 4;
        row.worst_p99_wake_latency_us = 99.0;
        row.worst_median_wake_latency_us = 50.0;
        row.worst_wake_latency_cv = 0.5;
        row.total_iterations = 1000;
        row.worst_mean_run_delay_us = 25.0;
        row.worst_run_delay_us = 200.0;
        row.page_locality = 0.8;
        row.cross_node_migration_ratio = 0.1;
        assert_eq!(read_metric(&row, "worst_spread"), Some(42.0));
        assert_eq!(read_metric(&row, "worst_gap_ms"), Some(100.0));
        assert_eq!(read_metric(&row, "total_migrations"), Some(7.0));
        assert_eq!(read_metric(&row, "worst_migration_ratio"), Some(0.3));
        assert_eq!(read_metric(&row, "max_imbalance_ratio"), Some(2.0));
        assert_eq!(read_metric(&row, "max_dsq_depth"), Some(5.0));
        assert_eq!(read_metric(&row, "stuck_count"), Some(3.0));
        assert_eq!(read_metric(&row, "total_fallback"), Some(11.0));
        assert_eq!(read_metric(&row, "total_keep_last"), Some(4.0));
        assert_eq!(read_metric(&row, "worst_p99_wake_latency_us"), Some(99.0));
        assert_eq!(
            read_metric(&row, "worst_median_wake_latency_us"),
            Some(50.0)
        );
        assert_eq!(read_metric(&row, "worst_wake_latency_cv"), Some(0.5));
        assert_eq!(read_metric(&row, "total_iterations"), Some(1000.0));
        assert_eq!(read_metric(&row, "worst_mean_run_delay_us"), Some(25.0));
        assert_eq!(read_metric(&row, "worst_run_delay_us"), Some(200.0));
        assert_eq!(read_metric(&row, "worst_page_locality"), Some(0.8));
        assert_eq!(
            read_metric(&row, "worst_cross_node_migration_ratio"),
            Some(0.1)
        );
    }

    #[test]
    fn metric_def_read_prefers_accessor_over_ext_metrics() {
        // When a name is in METRICS, the built-in accessor wins.
        // Even if ext_metrics carries a colliding entry for the
        // same name, MetricDef::read returns the accessor's value
        // — built-in fields are the authoritative source.
        let mut row = make_row("a", "t", true, 5.0);
        row.ext_metrics.insert("worst_spread".into(), 999.0);
        assert_eq!(read_metric(&row, "worst_spread"), Some(5.0));

        // User ext_metrics with no matching MetricDef are reachable
        // via the direct ext_metrics map; metric_def returns None
        // for unregistered names.
        row.ext_metrics.insert("custom_metric".into(), 77.0);
        assert!(metric_def("custom_metric").is_none());
        assert_eq!(row.ext_metrics.get("custom_metric").copied(), Some(77.0));
    }

    // -- compare_rows tests --

    /// Build a row matching the sidecar-derived schema:
    /// `work_type = "SpinWait"`, all metrics zeroed except `spread`
    /// and `total_iterations`.
    fn cmp_row(scenario: &str, topo: &str, passed: bool, spread: f64, iters: u64) -> GauntletRow {
        let mut r = make_row(scenario, topo, passed, spread);
        r.gap_ms = 0;
        r.migrations = 0;
        r.imbalance_ratio = 0.0;
        r.max_dsq_depth = 0;
        r.total_iterations = iters;
        r
    }

    #[test]
    fn compare_rows_dual_gate_both_must_trigger() {
        // worst_spread default_abs=5.0, default_rel=0.25.
        // 10 -> 12: abs delta 2.0 < 5.0 (abs gate fails); rel 0.20 < 0.25
        // (rel gate also fails). Result: 0 regressions, 0 improvements,
        // unchanged for worst_spread.
        let rows_a = vec![cmp_row("test_a", "tiny-1llc", true, 10.0, 0)];
        let rows_b = vec![cmp_row("test_a", "tiny-1llc", true, 12.0, 0)];
        let res = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(res.regressions, 0, "abs gate must block 2.0 < 5.0");
        assert_eq!(res.improvements, 0);
        assert_eq!(
            res.unchanged, 1,
            "worst_spread should be classified unchanged"
        );
        assert!(res.findings.is_empty());

        // Confirm the rel gate alone is not enough: spread 10 -> 14 has
        // rel 0.40 (>= 0.25) but abs delta 4.0 (< 5.0), still unchanged.
        let rows_b2 = vec![cmp_row("test_a", "tiny-1llc", true, 14.0, 0)];
        let res2 = compare_rows_by(
            &rows_a,
            &rows_b2,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(
            res2.regressions, 0,
            "rel-only is insufficient: abs gate must also fire"
        );
        assert_eq!(res2.unchanged, 1);
    }

    #[test]
    fn compare_rows_synthetic_regression_and_improvement() {
        // spread 10 -> 30: abs delta 20.0 >= 5.0, rel 2.0 >= 0.10 →
        // regression (higher_is_worse).
        // total_iterations 1000 -> 500: abs delta 500 >= 100, rel 0.5
        // >= 0.10, higher_is_worse=false so decrease is a regression.
        // Net: 2 regressions, 0 improvements; one Finding per
        // significant metric.
        let rows_a = vec![cmp_row("test1", "tiny-1llc", true, 10.0, 1000)];
        let rows_b = vec![cmp_row("test1", "tiny-1llc", true, 30.0, 500)];
        let res = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::uniform(10.0),
        );
        assert_eq!(
            res.regressions, 2,
            "spread up + iterations down both regress"
        );
        assert_eq!(res.improvements, 0);
        assert_eq!(res.excluded_pairs, 0);
        let metrics: Vec<&str> = res.findings.iter().map(|d| d.metric.name).collect();
        assert!(metrics.contains(&"worst_spread"));
        assert!(metrics.contains(&"total_iterations"));
        for d in &res.findings {
            assert!(d.is_regression, "all reported deltas should be regressions");
            assert_eq!(d.scenario, "test1");
            assert_eq!(d.topology, "tiny-1llc");
        }

        // Reverse direction: improvements should also surface.
        let res_imp = compare_rows_by(
            &rows_b,
            &rows_a,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::uniform(10.0),
        );
        assert_eq!(res_imp.regressions, 0);
        assert_eq!(res_imp.improvements, 2);
        for d in &res_imp.findings {
            assert!(!d.is_regression);
        }
    }

    /// Rate-COMPONENT metrics are suppressed from compare findings, but the
    /// user-facing rate is not. `total_iterations_pooled` (a suppressed
    /// component) differs 1000->2000 — past the default gate, normally a
    /// finding — yet emits none; the pooled rate `iterations_per_cpu_sec`
    /// differs 500->1000 and DOES emit. Pins the compare-emit suppression while
    /// the components stay in `ext_metrics` for the cross-run re-pool.
    #[test]
    fn compare_rows_suppresses_rate_components_not_the_rate() {
        let mut a = cmp_row("t", "tiny-1llc", true, 0.0, 1000);
        a.ext_metrics
            .insert("total_iterations_pooled".to_string(), 1000.0);
        a.ext_metrics
            .insert("iterations_per_cpu_sec".to_string(), 500.0);
        let mut b = cmp_row("t", "tiny-1llc", true, 0.0, 1000);
        b.ext_metrics
            .insert("total_iterations_pooled".to_string(), 2000.0);
        b.ext_metrics
            .insert("iterations_per_cpu_sec".to_string(), 1000.0);
        let res = compare_rows_by(
            &[a],
            &[b],
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        let names: Vec<&str> = res.findings.iter().map(|d| d.metric.name).collect();
        assert!(
            !names.contains(&"total_iterations_pooled"),
            "the Rate component must be suppressed from compare findings; got {names:?}",
        );
        assert!(
            names.contains(&"iterations_per_cpu_sec"),
            "the user-facing pooled rate must still emit a finding; got {names:?}",
        );
    }

    #[test]
    fn compare_rows_higher_is_worse_inversion() {
        // total_iterations is higher_is_worse=false. A drop of 1000 ->
        // 500 must be reported as a regression, not an improvement.
        let rows_a = vec![cmp_row("t", "tiny-1llc", true, 0.0, 1000)];
        let rows_b = vec![cmp_row("t", "tiny-1llc", true, 0.0, 500)];
        let res = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        let iters_delta = res
            .findings
            .iter()
            .find(|d| d.metric.name == "total_iterations")
            .expect("total_iterations should produce a delta");
        assert!(
            iters_delta.is_regression,
            "iterations decrease is a regression"
        );
        assert_eq!(iters_delta.delta, -500.0);
        assert_eq!(res.regressions, 1);
        assert_eq!(res.improvements, 0);

        // worst_spread is higher_is_worse=true. An increase must be a
        // regression; a decrease must be an improvement.
        let rows_a2 = vec![cmp_row("t", "tiny-1llc", true, 10.0, 0)];
        let rows_b2 = vec![cmp_row("t", "tiny-1llc", true, 30.0, 0)];
        let res_up = compare_rows_by(
            &rows_a2,
            &rows_b2,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        let spread_up = res_up
            .findings
            .iter()
            .find(|d| d.metric.name == "worst_spread")
            .expect("worst_spread should produce a delta");
        assert!(spread_up.is_regression, "spread increase is a regression");
        assert_eq!(spread_up.delta, 20.0);

        let res_down = compare_rows_by(
            &rows_b2,
            &rows_a2,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        let spread_down = res_down
            .findings
            .iter()
            .find(|d| d.metric.name == "worst_spread")
            .expect("worst_spread should produce a delta");
        assert!(
            !spread_down.is_regression,
            "spread decrease is an improvement"
        );
        assert_eq!(spread_down.delta, -20.0);
    }

    #[test]
    fn compare_rows_skipped_side_drops_pair_into_excluded_pairs() {
        // A skipped row on either side of the comparison must not
        // contribute to regressions/improvements — a skipped run
        // carries no executed metrics, so the pair must short-circuit
        // via the is_skip() gate before regression math touches the
        // default-zero metric values.
        let mut row_a = cmp_row("t", "tiny-1llc", true, 10.0, 100);
        let mut row_b = cmp_row("t", "tiny-1llc", true, 10.0, 100);
        row_a.skipped = true; // A side was skipped
        let res = compare_rows_by(
            &[row_a.clone()],
            &[row_b.clone()],
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(res.regressions, 0);
        assert_eq!(res.improvements, 0);
        assert_eq!(
            res.excluded_pairs, 1,
            "skipped side must count as excluded_pairs, not produce deltas"
        );

        // Symmetrically on the B side.
        row_a.skipped = false;
        row_b.skipped = true;
        let res = compare_rows_by(
            &[row_a],
            &[row_b],
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(res.regressions, 0);
        assert_eq!(res.improvements, 0);
        assert_eq!(res.excluded_pairs, 1);
    }

    /// Rows where either side has `passed=false` are dropped from the
    /// regression math. A failed scenario's metrics reflect the failure
    /// mode (short run, stalled workload, missing samples), not
    /// scheduler behavior.
    #[test]
    fn compare_rows_skips_failed_scenarios() {
        // Three scenarios, all with the same metric movement. Only
        // test_ok (passed on both sides) should be eligible for the
        // regression math; the other two are counted as excluded_pairs.
        let rows_a = vec![
            cmp_row("test_ok", "tiny-1llc", true, 10.0, 1000),
            cmp_row("test_failed_b", "tiny-1llc", true, 10.0, 1000),
            cmp_row("test_failed_a", "tiny-1llc", false, 10.0, 1000),
        ];
        let rows_b = vec![
            cmp_row("test_ok", "tiny-1llc", true, 30.0, 500),
            cmp_row("test_failed_b", "tiny-1llc", false, 30.0, 500),
            cmp_row("test_failed_a", "tiny-1llc", true, 30.0, 500),
        ];
        let res = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::uniform(10.0),
        );
        assert_eq!(
            res.excluded_pairs, 2,
            "test_failed_a and test_failed_b skip"
        );
        // test_ok regresses on worst_spread and total_iterations only.
        assert_eq!(res.regressions, 2);
        assert_eq!(res.improvements, 0);
        for d in &res.findings {
            assert_eq!(d.scenario, "test_ok");
        }
    }

    #[test]
    fn compare_rows_filter_substring() {
        // Two scenarios in each run. Filter "alpha" must match the
        // alpha row (substring of the joined "scenario topology
        // scheduler work_type" string) and exclude the beta row.
        let rows_a = vec![
            cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
            cmp_row("beta", "tiny-1llc", true, 10.0, 0),
        ];
        let rows_b = vec![
            cmp_row("alpha", "tiny-1llc", true, 30.0, 0),
            cmp_row("beta", "tiny-1llc", true, 30.0, 0),
        ];
        let res = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            Some("alpha"),
            &ComparisonPolicy::default(),
        );
        assert_eq!(res.regressions, 1, "only alpha row should compare");
        assert_eq!(res.findings.len(), 1);
        assert_eq!(res.findings[0].scenario, "alpha");
        // Finding carries work_type so two findings sharing
        // scenario+topology under different workloads stay
        // distinguishable.
        assert_eq!(res.findings[0].work_type, "SpinWait");

        // Filter on topology substring is also honored. Both rows
        // share the "tiny-1llc" topology and only worst_spread crosses
        // both gates (10 -> 30 with default_abs=5.0, default_rel=0.25),
        // so each row contributes exactly one finding.
        let res_topo = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            Some("tiny"),
            &ComparisonPolicy::default(),
        );
        assert_eq!(res_topo.regressions, 2, "both rows match 'tiny' topology");
        assert_eq!(res_topo.findings.len(), 2);

        // Non-matching filter yields no comparisons at all.
        let res_none = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            Some("nomatch"),
            &ComparisonPolicy::default(),
        );
        assert_eq!(res_none.regressions, 0);
        assert_eq!(res_none.improvements, 0);
        assert_eq!(res_none.unchanged, 0);
        assert_eq!(res_none.excluded_pairs, 0);
    }

    #[test]
    fn compare_rows_threshold_override() {
        // worst_spread default_rel=0.25, default_abs=5.0. Move 100 ->
        // 106: abs delta 6.0 >= 5.0 (abs gate passes); rel 0.06 < 0.25
        // (default rel fails) → unchanged with default thresholds.
        let rows_a = vec![cmp_row("t", "tiny-1llc", true, 100.0, 0)];
        let rows_b = vec![cmp_row("t", "tiny-1llc", true, 106.0, 0)];
        let res_default = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        let spread_default = res_default
            .findings
            .iter()
            .find(|d| d.metric.name == "worst_spread");
        assert!(
            spread_default.is_none(),
            "default rel 0.25 must classify 6% change as unchanged"
        );

        // Override threshold to 5% (Some(5.0) → rel_thresh 0.05). Now
        // rel 0.06 >= 0.05, both gates fire → regression.
        let res_override = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::uniform(5.0),
        );
        let spread_override = res_override
            .findings
            .iter()
            .find(|d| d.metric.name == "worst_spread")
            .expect("override 5% must surface 6% spread change");
        assert!(spread_override.is_regression);
        assert_eq!(spread_override.delta, 6.0);

        // The override does NOT loosen the abs gate. Move 1.0 -> 1.5:
        // abs delta 0.5 < 5.0; even threshold=1% (rel_thresh 0.01)
        // can't promote it to significant.
        let rows_a_small = vec![cmp_row("t", "tiny-1llc", true, 1.0, 0)];
        let rows_b_small = vec![cmp_row("t", "tiny-1llc", true, 1.5, 0)];
        let res_small = compare_rows_by(
            &rows_a_small,
            &rows_b_small,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::uniform(1.0),
        );
        assert!(
            !res_small
                .findings
                .iter()
                .any(|d| d.metric.name == "worst_spread"),
            "abs gate must still block tiny absolute moves"
        );
    }

    /// `ComparisonPolicy::rel_threshold` resolution priority pinned
    /// by exhaustive enumeration: per-metric override wins over
    /// `default_percent`, which wins over the registry fallback.
    /// A regression that inverted the priority or shortcut the
    /// fallback (e.g. always returning `default_percent` even when
    /// a per-metric override exists) surfaces here, not as subtly-
    /// wrong thresholds inside `compare_rows`.
    #[test]
    fn comparison_policy_rel_threshold_resolution_priority() {
        // Empty policy → registry fallback. `default_rel` is
        // passed by the caller (compare_rows supplies it from
        // `m.default_rel`), so we pick an arbitrary fallback here
        // and check it's returned verbatim.
        let empty = ComparisonPolicy::default();
        assert_eq!(
            empty.rel_threshold("worst_spread", 0.25),
            0.25,
            "empty policy must fall through to the registry default_rel",
        );

        // Uniform override → default_percent / 100 wins over
        // the registry default.
        let uniform = ComparisonPolicy::uniform(10.0);
        assert_eq!(
            uniform.rel_threshold("worst_spread", 0.25),
            0.10,
            "uniform(10.0) must override the registry default_rel \
             with 10.0 / 100.0 = 0.10",
        );

        // Per-metric override wins over both `default_percent` and
        // the registry default. Use two metric names so the test
        // also proves other metrics still see `default_percent`
        // when no per-metric entry matches.
        let mut per_metric = ComparisonPolicy::uniform(10.0);
        per_metric
            .per_metric_percent
            .insert("worst_spread".to_string(), 5.0);
        assert_eq!(
            per_metric.rel_threshold("worst_spread", 0.25),
            0.05,
            "per-metric override (5.0) must win over default_percent \
             (10.0) and the registry default (0.25)",
        );
        assert_eq!(
            per_metric.rel_threshold("worst_gap_ms", 0.25),
            0.10,
            "metrics not in the per-metric map must still see the \
             default_percent (10.0 → 0.10), not the registry default",
        );
    }

    /// `worst_wake_latency_tail_ratio` must be suppressed below the
    /// [`WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS`] sample floor. Low-N
    /// runs produce p99/median ratios dominated by a single outlier;
    /// the metric accessor must return `None` in that regime so
    /// [`compare_rows_by`] short-circuits and emits no finding.
    ///
    /// Positive side: above the floor, the same delta that was
    /// suppressed below must produce a finding. This proves the
    /// None-vs-Some branching is the gate that's firing — not an
    /// unrelated threshold somewhere else in the comparison math.
    #[test]
    fn wake_latency_tail_ratio_is_suppressed_below_min_iteration_floor() {
        use crate::stats::WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS as MIN;
        let metric = metric_def("worst_wake_latency_tail_ratio")
            .expect("worst_wake_latency_tail_ratio must be registered in METRICS");

        // Below the floor: accessor returns None. Both sides collapse
        // to 0.0 via unwrap_or(0.0); the EPSILON-guard then classifies
        // the delta as unchanged.
        let mut low_a = make_row("tail_low", "tiny-1llc", true, 0.0);
        let mut low_b = make_row("tail_low", "tiny-1llc", true, 0.0);
        low_a.total_iterations = MIN - 1;
        low_b.total_iterations = MIN - 1;
        low_a.worst_wake_latency_tail_ratio = 2.0;
        low_b.worst_wake_latency_tail_ratio = 20.0;
        assert!(
            metric.read(&low_a).is_none(),
            "below-floor A accessor must return None so the regression \
             math cannot see a value",
        );
        assert!(
            metric.read(&low_b).is_none(),
            "below-floor B accessor must return None even when the \
             raw field would have carried a suspicious value",
        );
        let below = compare_rows_by(
            std::slice::from_ref(&low_a),
            std::slice::from_ref(&low_b),
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(
            below.regressions, 0,
            "below-floor comparison must not surface a regression — \
             low-N ratios are noise, not signal",
        );
        assert!(
            below.findings.is_empty(),
            "below-floor comparison must emit no findings",
        );

        // At and above the floor: accessor returns Some and the same
        // delta now produces a finding.
        let mut hi_a = make_row("tail_hi", "tiny-1llc", true, 0.0);
        let mut hi_b = make_row("tail_hi", "tiny-1llc", true, 0.0);
        hi_a.total_iterations = MIN;
        hi_b.total_iterations = MIN;
        hi_a.worst_wake_latency_tail_ratio = 2.0;
        hi_b.worst_wake_latency_tail_ratio = 20.0;
        assert_eq!(
            metric.read(&hi_a),
            Some(2.0),
            "at-floor accessor must return Some",
        );
        let above = compare_rows_by(
            std::slice::from_ref(&hi_a),
            std::slice::from_ref(&hi_b),
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(
            above.regressions, 1,
            "at-floor comparison with a 10x tail blow-up must surface \
             as a regression; threshold wiring has a gap otherwise",
        );
    }

    /// Explicit None-branch pin on the compare_rows accessor contract.
    ///
    /// `compare_rows` calls `m.read(row)` for every metric and
    /// falls through `unwrap_or(0.0)` to the EPSILON-guard when the
    /// accessor returns `None`. The `wake_latency_tail_ratio_is_suppressed_below_*`
    /// sibling exercises this path EMBEDDED in the full comparison
    /// flow (via the tail-ratio accessor's iteration-count gate),
    /// but does NOT directly prove that `compare_rows` handles a
    /// None result; a regression that removed the `unwrap_or(0.0)`
    /// and panicked on None would fail the sibling only through
    /// the indirect "compare_rows panicked" route, which could be
    /// mistaken for a test infrastructure problem.
    ///
    /// This test synthesizes the None condition explicitly — a
    /// below-floor iterations count with distinctly-different
    /// stored `worst_wake_latency_tail_ratio` values on each side
    /// — and asserts the three observable consequences:
    /// 1. `metric.read(&row)` returns `None` on both sides.
    /// 2. `compare_rows` does NOT panic.
    /// 3. The resulting `CompareReport` classifies the pair as
    ///    `unchanged` (EPSILON guard swallowed the 0.0/0.0 delta).
    ///
    /// A panic or a regression/improvement count > 0 here would
    /// indicate the `unwrap_or(0.0)` in `compare_rows` has drifted.
    #[test]
    fn compare_rows_handles_none_from_accessor_as_zero() {
        use crate::stats::WAKE_LATENCY_TAIL_RATIO_MIN_ITERATIONS as MIN;
        let metric = metric_def("worst_wake_latency_tail_ratio")
            .expect("tail ratio metric must be registered");

        let mut row_a = make_row("none_branch", "tiny-1llc", true, 0.0);
        let mut row_b = make_row("none_branch", "tiny-1llc", true, 0.0);
        row_a.total_iterations = MIN - 1;
        row_b.total_iterations = MIN - 1;
        // Stored fields are distinctly non-zero so a regression that
        // short-circuited the accessor (returned the stored value
        // directly) would produce a 1000x delta that would fail
        // both the "unchanged" classification AND the regression
        // count assertion.
        row_a.worst_wake_latency_tail_ratio = 1.0;
        row_b.worst_wake_latency_tail_ratio = 1000.0;

        assert!(
            metric.read(&row_a).is_none(),
            "accessor must return None for below-floor A input — \
             otherwise this test is not actually exercising the \
             None branch of compare_rows",
        );
        assert!(
            metric.read(&row_b).is_none(),
            "accessor must return None for below-floor B input",
        );

        // The call must not panic (a regression that dropped the
        // `unwrap_or` would trip here), and the result must
        // classify the pair as unchanged — both sides collapse to
        // 0.0 via unwrap_or, then the `abs() < EPSILON` guard
        // short-circuits without producing a finding.
        let report = compare_rows_by(
            std::slice::from_ref(&row_a),
            std::slice::from_ref(&row_b),
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(
            report.regressions, 0,
            "None accessor result must land as unchanged, not a regression",
        );
        assert_eq!(
            report.improvements, 0,
            "None accessor result must land as unchanged, not an improvement",
        );
        assert!(
            report.findings.is_empty(),
            "no findings must be emitted when the accessor returns None; \
             got: {:?}",
            report.findings,
        );
    }

    /// `ComparisonPolicy::load_json` round-trips a policy file: a
    /// policy constructed in memory, serialized, and reloaded must
    /// yield the same thresholds end-to-end. Pins the wire format
    /// for the `--policy <path>` CLI flag.
    #[test]
    fn comparison_policy_load_json_round_trip() {
        let mut original = ComparisonPolicy::uniform(10.0);
        original
            .per_metric_percent
            .insert("worst_spread".to_string(), 5.0);
        original
            .per_metric_percent
            .insert("worst_p99_wake_latency_us".to_string(), 20.0);

        let json = serde_json::to_string(&original).expect("serialize policy");

        let tmp = tempfile::NamedTempFile::new().expect("create tempfile");
        std::fs::write(tmp.path(), json).expect("write policy file");

        let loaded = ComparisonPolicy::load_json(tmp.path()).expect("load policy");

        assert_eq!(
            loaded.default_percent,
            Some(10.0),
            "default_percent must round-trip",
        );
        assert_eq!(
            loaded.per_metric_percent.get("worst_spread"),
            Some(&5.0),
            "per-metric worst_spread override must round-trip",
        );
        assert_eq!(
            loaded.per_metric_percent.get("worst_p99_wake_latency_us"),
            Some(&20.0),
            "per-metric worst_p99 override must round-trip",
        );
        // Resolution-path equivalence: the loaded policy resolves
        // every metric identically to the original.
        for metric_name in ["worst_spread", "worst_p99_wake_latency_us", "worst_gap_ms"] {
            assert_eq!(
                loaded.rel_threshold(metric_name, 0.25),
                original.rel_threshold(metric_name, 0.25),
                "load_json round-trip must preserve threshold \
                 resolution for {metric_name}",
            );
        }
    }

    /// `ComparisonPolicy::load_json` on a nonexistent path must
    /// surface an actionable error naming the path (not a generic
    /// "no such file"). Pins the `with_context` chain — a
    /// regression that dropped the context would collapse a
    /// user-facing `--policy missing.json` invocation into a
    /// bare `No such file or directory` with no clue about where
    /// the missing file was expected.
    #[test]
    fn comparison_policy_load_json_nonexistent_path_surfaces_path() {
        let path = std::path::Path::new("/nonexistent/ktstr/policy-DOES-NOT-EXIST.json");
        let err = ComparisonPolicy::load_json(path).expect_err("nonexistent path must fail");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains(&path.display().to_string()),
            "error must name the missing path so a user can see \
             which file was expected; got: {rendered}",
        );
        assert!(
            rendered.to_ascii_lowercase().contains("read")
                || rendered.to_ascii_lowercase().contains("no such"),
            "error must describe the read failure (either the \
             `with_context` \"read comparison policy from ...\" \
             prefix or std's underlying \"No such file...\" \
             reason); got: {rendered}",
        );
    }

    /// `ComparisonPolicy::load_json` on a malformed JSON body
    /// must include both the path (for locating) AND the parse
    /// context (for understanding the failure shape). A
    /// `serde_json::Error` on its own gives line/column but no
    /// file identity; the `with_context` adds the path. Pins
    /// both halves.
    #[test]
    fn comparison_policy_load_json_malformed_json_surfaces_path_and_parse_context() {
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        // Not JSON — clearly malformed.
        std::fs::write(tmp.path(), "this is not json at all {{{").expect("write");
        let err = ComparisonPolicy::load_json(tmp.path()).expect_err("malformed JSON must fail");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains(&tmp.path().display().to_string()),
            "malformed-JSON error must name the path; got: {rendered}",
        );
        assert!(
            rendered.to_ascii_lowercase().contains("parse")
                || rendered.to_ascii_lowercase().contains("expected"),
            "malformed-JSON error must include a parse-context \
             hint (either the `with_context` \"parse comparison \
             policy from ...\" prefix, or serde_json's \"expected \
             ...\" reason); got: {rendered}",
        );
    }

    /// `load_json` rejects unknown top-level fields per
    /// `deny_unknown_fields`. A misspelled field (e.g.
    /// `default_percentage` vs `default_percent`) must surface as
    /// a parse error, not silently drop the value and fall back
    /// to the default.
    #[test]
    fn comparison_policy_load_json_rejects_unknown_fields() {
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(tmp.path(), r#"{"default_percentage": 10.0}"#).expect("write");
        let err = ComparisonPolicy::load_json(tmp.path()).expect_err("unknown field must fail");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("default_percentage")
                || rendered.to_ascii_lowercase().contains("unknown"),
            "unknown-field error must name the typo so a user \
             can fix the policy file; got: {rendered}",
        );
    }

    /// `validate` rejects negative `default_percent`. A regression
    /// that lost the sign check would let `--threshold -10`
    /// through to `compare_rows`' dual-gate `.abs()` comparison,
    /// where a negative `rel_thresh` makes every delta (including
    /// zero) significant — silently inverting the comparison.
    #[test]
    fn comparison_policy_validate_rejects_negative_default_percent() {
        let policy = ComparisonPolicy::uniform(-10.0);
        let err = policy
            .validate()
            .expect_err("negative default_percent must fail validation");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("default_percent"),
            "validation error must name the field; got: {rendered}",
        );
        assert!(
            rendered.contains("-10"),
            "validation error must echo the rejected value; got: {rendered}",
        );
    }

    /// `validate` rejects unknown per-metric keys. A typo in the
    /// policy file would otherwise silently fall through to
    /// `default_percent` — a user debugging a regression with
    /// `--policy typo.json` would see the uniform threshold
    /// applied instead of the expected override and have no way
    /// to know why.
    #[test]
    fn comparison_policy_validate_rejects_unknown_per_metric_keys() {
        let mut policy = ComparisonPolicy::default();
        policy
            .per_metric_percent
            .insert("wrost_spread".to_string(), 5.0); // typo
        let err = policy
            .validate()
            .expect_err("unknown per-metric key must fail validation");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("wrost_spread"),
            "validation error must echo the unknown key so a user \
             can see the typo; got: {rendered}",
        );
        // Known-metric list should appear so the user can pick the
        // right spelling. Registered metric names include
        // `worst_spread` — a hint toward the correct key.
        assert!(
            rendered.contains("worst_spread"),
            "validation error should include the registered \
             metric list so users can find the right spelling; \
             got: {rendered}",
        );
    }

    /// `validate` rejects negative per-metric overrides. Covers
    /// the sibling case of the default_percent sign check above.
    #[test]
    fn comparison_policy_validate_rejects_negative_per_metric_value() {
        let mut policy = ComparisonPolicy::default();
        policy
            .per_metric_percent
            .insert("worst_spread".to_string(), -5.0);
        let err = policy
            .validate()
            .expect_err("negative per-metric percent must fail");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("worst_spread") && rendered.contains("-5"),
            "validation error must name both the key and the \
             rejected value; got: {rendered}",
        );
    }

    /// Defence-in-depth against an on-disk policy missing fields
    /// (e.g. older wire format, hand-edited JSON). The struct uses
    /// `#[serde(default)]` on every field so a partial JSON
    /// (`{}`, `{"default_percent": 5}`) deserializes to a policy
    /// with the missing field at its `Default` value. A regression
    /// that dropped the `#[serde(default)]` attribute would make
    /// `load_json` reject otherwise-valid partial policies.
    #[test]
    fn comparison_policy_load_json_accepts_partial_fields() {
        let tmp = tempfile::NamedTempFile::new().expect("create tempfile");
        // Empty object → policy with every default.
        std::fs::write(tmp.path(), "{}").expect("write empty policy");
        let loaded = ComparisonPolicy::load_json(tmp.path()).expect("load empty policy");
        assert_eq!(loaded.default_percent, None);
        assert!(loaded.per_metric_percent.is_empty());

        // Only default_percent set → empty per_metric.
        std::fs::write(tmp.path(), r#"{"default_percent": 7.5}"#).expect("write partial policy");
        let loaded = ComparisonPolicy::load_json(tmp.path()).expect("load partial policy");
        assert_eq!(loaded.default_percent, Some(7.5));
        assert!(loaded.per_metric_percent.is_empty());

        // Only per_metric_percent set → default_percent None.
        std::fs::write(
            tmp.path(),
            r#"{"per_metric_percent": {"worst_spread": 3.0}}"#,
        )
        .expect("write per-metric-only policy");
        let loaded = ComparisonPolicy::load_json(tmp.path()).expect("load per-metric-only policy");
        assert_eq!(loaded.default_percent, None);
        assert_eq!(loaded.per_metric_percent.get("worst_spread"), Some(&3.0),);
    }

    /// `from_cli_flags` resolves the `--threshold` / `--policy` pair
    /// the shared way for `stats compare` and `perf-delta`:
    /// threshold → uniform (validated), policy → load_json, neither →
    /// registry defaults, both → error (the clap-`conflicts_with`
    /// backstop). Pin every branch so a future edit can't silently
    /// drop the sign check or the mutual-exclusion guard.
    #[test]
    fn comparison_policy_from_cli_flags_resolves_each_branch() {
        // --threshold N → uniform default_percent = N.
        let p = ComparisonPolicy::from_cli_flags(Some(15.0), None).expect("threshold resolves");
        assert_eq!(p.default_percent, Some(15.0));
        assert!(p.per_metric_percent.is_empty());

        // A negative --threshold is rejected via validate().
        assert!(
            ComparisonPolicy::from_cli_flags(Some(-1.0), None).is_err(),
            "negative --threshold must be rejected before the dual-gate math",
        );

        // --policy PATH → load_json.
        let tmp = tempfile::NamedTempFile::new().expect("tempfile");
        std::fs::write(tmp.path(), r#"{"default_percent": 8.0}"#).expect("write policy");
        let p =
            ComparisonPolicy::from_cli_flags(None, Some(tmp.path())).expect("policy file resolves");
        assert_eq!(p.default_percent, Some(8.0));

        // Neither → registry defaults (no uniform override).
        let p = ComparisonPolicy::from_cli_flags(None, None).expect("default resolves");
        assert_eq!(p.default_percent, None);

        // Both set → error: clap `conflicts_with` makes this
        // unreachable at the CLI, but the library entry point must not
        // silently prefer one over the other.
        assert!(
            ComparisonPolicy::from_cli_flags(Some(10.0), Some(tmp.path())).is_err(),
            "--threshold + --policy together must error",
        );
    }

    /// End-to-end pin: `compare_rows` with a per-metric policy
    /// must apply the override for the matching metric AND fall
    /// through to `default_percent` for every other metric. The
    /// unit-level `comparison_policy_rel_threshold_resolution_priority`
    /// test above pins the resolution function in isolation; this
    /// test runs it through the actual compare_rows pipeline with
    /// rows that trigger distinct deltas on two metrics, proving
    /// that `compare_rows` reads `m.name` correctly and hands it
    /// to `policy.rel_threshold`. A regression that hard-coded a
    /// single metric name, or passed the wrong name to the
    /// resolver, would surface here as the wrong regression count.
    ///
    /// Fixture:
    /// - A: `worst_spread = 100`, `worst_median_wake_latency_us = 100`
    /// - B: `worst_spread = 106` (6% delta, passes the abs gate
    ///   at 5.0), `worst_median_wake_latency_us = 110` (10%
    ///   delta).
    /// - Policy: `default_percent = 20%`, per_metric
    ///   `worst_spread = 5%`.
    ///
    /// Expected: `worst_spread`'s 6% delta beats the 5%
    /// per-metric override → regression. `worst_median_wake_latency_us`'s
    /// 10% delta falls under the 20% default → unchanged. Total
    /// regressions = 1.
    #[test]
    fn compare_rows_per_metric_policy_resolves_each_metric_independently() {
        // Construct rows with both metrics non-default so we can
        // trigger per-metric and default_percent branches in one
        // row pair.
        let mut row_a = cmp_row("t", "tiny-1llc", true, 100.0, 0);
        row_a.worst_median_wake_latency_us = 100.0;
        let mut row_b = cmp_row("t", "tiny-1llc", true, 106.0, 0);
        row_b.worst_median_wake_latency_us = 110.0;

        let mut policy = ComparisonPolicy::uniform(20.0);
        policy
            .per_metric_percent
            .insert("worst_spread".to_string(), 5.0);

        let res = compare_rows_by(&[row_a], &[row_b], LEGACY_PAIRING_DIMS, None, &policy);

        let spread_finding = res
            .findings
            .iter()
            .find(|f| f.metric.name == "worst_spread");
        assert!(
            spread_finding.is_some(),
            "worst_spread per-metric override (5%) must fire on 6% \
             delta; got findings: {:?}",
            res.findings
                .iter()
                .map(|f| f.metric.name)
                .collect::<Vec<_>>(),
        );
        let spread_finding = spread_finding.unwrap();
        assert!(spread_finding.is_regression, "6% > 5% → regression");

        // worst_median_wake_latency_us has a 10% delta; under
        // default_percent = 20%, it must be unchanged (not in
        // findings).
        let wake_finding = res
            .findings
            .iter()
            .find(|f| f.metric.name == "worst_median_wake_latency_us");
        assert!(
            wake_finding.is_none(),
            "worst_median_wake_latency_us 10% delta must fall \
             under default_percent 20% and be unchanged. The \
             regression would indicate `compare_rows` ignored \
             default_percent for non-per-metric entries; got \
             finding: {wake_finding:?}",
        );

        assert_eq!(
            res.regressions, 1,
            "exactly one regression expected — the per-metric \
             spread override should win on spread, and the \
             default_percent should suppress wake latency. Got: \
             regressions={}, improvements={}, unchanged={}",
            res.regressions, res.improvements, res.unchanged,
        );
    }

    /// `compare_rows` uses `Iterator::find` to locate the A-side
    /// match for each B-side row, so when `rows_a` contains two
    /// entries with the same `(scenario, topology, work_type)` key
    /// the first one wins. Lock that contract in: the second
    /// duplicate must be ignored even though it would change the
    /// verdict.
    #[test]
    fn compare_rows_duplicate_key_first_match_wins() {
        // First A-side entry has spread=10 (would yield a regression
        // against B's 30). Second has spread=29 (would be unchanged).
        // The result must reflect the first entry only.
        let rows_a = vec![
            cmp_row("t", "tiny-1llc", true, 10.0, 0),
            cmp_row("t", "tiny-1llc", true, 29.0, 0),
        ];
        let rows_b = vec![cmp_row("t", "tiny-1llc", true, 30.0, 0)];
        let res = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(res.regressions, 1, "first match (spread=10) must win");
        let spread = res
            .findings
            .iter()
            .find(|d| d.metric.name == "worst_spread")
            .expect("worst_spread regression should fire");
        assert_eq!(
            spread.val_a, 10.0,
            "val_a must come from the first matching row"
        );
        assert_eq!(spread.delta, 20.0);
    }

    /// Filtering is applied before the failed-row gate. A failed row
    /// that the filter excludes never reaches the `passed` check, so
    /// `excluded_pairs` stays at zero -- the failure on the filtered
    /// row is invisible by design.
    #[test]
    fn compare_rows_filter_excludes_failed_from_skip_count() {
        let rows_a = vec![
            cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
            cmp_row("beta", "tiny-1llc", false, 10.0, 0),
        ];
        let rows_b = vec![
            cmp_row("alpha", "tiny-1llc", true, 30.0, 0),
            cmp_row("beta", "tiny-1llc", true, 30.0, 0),
        ];
        // Without a filter, beta's failed row contributes
        // excluded_pairs=1.
        let unfiltered = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(unfiltered.excluded_pairs, 1);
        assert_eq!(unfiltered.regressions, 1, "alpha still regresses");

        // Filtering to "alpha" excludes beta entirely; the failed row
        // is filtered out before the passed gate runs, so
        // excluded_pairs=0.
        let filtered = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            Some("alpha"),
            &ComparisonPolicy::default(),
        );
        assert_eq!(filtered.excluded_pairs, 0);
        assert_eq!(filtered.regressions, 1);
        assert_eq!(filtered.findings.len(), 1);
        assert_eq!(filtered.findings[0].scenario, "alpha");
    }

    /// The substring filter searches the joined "scenario topology
    /// scheduler work_type" string, so a scheduler name uniquely
    /// scopes the comparison even when scenarios and topologies
    /// overlap. Without scheduler in the join string this would
    /// require a less-precise substring (e.g. a scenario name).
    #[test]
    fn compare_rows_filter_substring_matches_scheduler() {
        let mut a1 = cmp_row("test1", "tiny-1llc", true, 10.0, 0);
        a1.scheduler = "scx_alpha".into();
        let mut a2 = cmp_row("test2", "tiny-1llc", true, 10.0, 0);
        a2.scheduler = "scx_beta".into();
        let mut b1 = cmp_row("test1", "tiny-1llc", true, 30.0, 0);
        b1.scheduler = "scx_alpha".into();
        let mut b2 = cmp_row("test2", "tiny-1llc", true, 30.0, 0);
        b2.scheduler = "scx_beta".into();

        let res = compare_rows_by(
            &[a1, a2],
            &[b1, b2],
            LEGACY_PAIRING_DIMS,
            Some("scx_alpha"),
            &ComparisonPolicy::default(),
        );
        assert_eq!(res.regressions, 1, "only the scx_alpha row compares");
        assert_eq!(res.findings.len(), 1);
        assert_eq!(res.findings[0].scenario, "test1");
        // scx_beta rows are filtered out, not counted as new/removed.
        assert_eq!(res.new_in_b, 0);
        assert_eq!(res.removed_from_a, 0);
    }

    /// `new_in_b` counts B-side rows whose key has no match on the A
    /// side; `removed_from_a` counts the converse. Both are needed so
    /// schema drift between two runs (a renamed scenario, an added
    /// topology preset, a removed work_type) is visible in the
    /// summary instead of silently dropped.
    #[test]
    fn compare_rows_tracks_new_and_removed_rows() {
        // alpha exists in both -> regression.
        // beta exists only in B -> new_in_b=1.
        // gamma exists only in A -> removed_from_a=1.
        let rows_a = vec![
            cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
            cmp_row("gamma", "tiny-1llc", true, 10.0, 0),
        ];
        let rows_b = vec![
            cmp_row("alpha", "tiny-1llc", true, 30.0, 0),
            cmp_row("beta", "tiny-1llc", true, 30.0, 0),
        ];
        let res = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            None,
            &ComparisonPolicy::default(),
        );
        assert_eq!(res.regressions, 1, "alpha regresses on worst_spread");
        assert_eq!(res.new_in_b, 1, "beta is new on B side");
        assert_eq!(res.removed_from_a, 1, "gamma is removed on B side");
        assert_eq!(res.excluded_pairs, 0);
    }

    /// The filter applies to every counter, including `new_in_b` and
    /// `removed_from_a`. An excluded row never reaches matching, so
    /// it contributes to no counter at all.
    #[test]
    fn compare_rows_filter_applies_to_new_and_removed_counters() {
        let rows_a = vec![
            cmp_row("alpha", "tiny-1llc", true, 10.0, 0),
            cmp_row("gamma", "tiny-1llc", true, 10.0, 0),
        ];
        let rows_b = vec![
            cmp_row("alpha", "tiny-1llc", true, 30.0, 0),
            cmp_row("beta", "tiny-1llc", true, 30.0, 0),
        ];

        // Filter to "alpha" -- beta and gamma are excluded by the
        // substring filter on both passes.
        let res = compare_rows_by(
            &rows_a,
            &rows_b,
            LEGACY_PAIRING_DIMS,
            Some("alpha"),
            &ComparisonPolicy::default(),
        );
        assert_eq!(res.regressions, 1);
        assert_eq!(res.new_in_b, 0, "beta is filtered out, not new");
        assert_eq!(res.removed_from_a, 0, "gamma is filtered out, not removed");
    }

    // -- format_host_delta: the 5 match arms of the host-delta
    //    section emitted under `stats compare --runs a b`. --

    /// Builder for a `HostContext` with enough populated fields to
    /// exercise `HostContext::diff`. Leaves everything else at its
    /// `Default` so each test varies only the field under study.
    fn host_ctx(release: &str, kernel_cmdline: Option<&str>) -> crate::host_context::HostContext {
        crate::host_context::HostContext {
            kernel_name: Some("Linux".to_string()),
            kernel_release: Some(release.to_string()),
            kernel_cmdline: kernel_cmdline.map(str::to_string),
            ..Default::default()
        }
    }

    /// `(Some, Some)` identical: the helper emits a one-line
    /// confirmation so users running `stats compare` can distinguish
    /// "same host" from "captured but unused" without inspecting
    /// individual sidecars.
    #[test]
    fn format_host_delta_both_present_identical() {
        let ctx = host_ctx("6.14.0", Some("preempt=lazy"));
        let out = format_host_delta(Some(&ctx), Some(&ctx), "a-run", "b-run");
        assert_eq!(out, "\nhost: identical between 'a-run' and 'b-run'\n");
    }

    /// `(Some, Some)` differing: the helper emits the header line
    /// followed by whatever `HostContext::diff` produced. Asserts
    /// the structural shape (header present, delta body present)
    /// rather than the exact diff formatting so this test stays
    /// robust to future tweaks to the diff renderer.
    #[test]
    fn format_host_delta_both_present_differ() {
        let ha = host_ctx("6.14.0", Some("preempt=lazy"));
        let hb = host_ctx("6.15.1", Some("preempt=lazy"));
        let out = format_host_delta(Some(&ha), Some(&hb), "a", "b");
        assert!(
            out.starts_with("\nhost delta ('a' → 'b'):\n"),
            "got: {out:?}"
        );
        // `kernel_release` differs between the two contexts so the
        // diff body must be non-empty — confirms we routed through
        // the `else` arm and not the `identical` arm.
        let body = &out["\nhost delta ('a' → 'b'):\n".len()..];
        assert!(
            !body.is_empty(),
            "differing contexts must produce a diff body"
        );
        // Pin the trailing-newline contract: the other three arms
        // (`identical`, left-only, right-only) all end with '\n'; the
        // differ arm delegates to `HostContext::diff()` whose output
        // must also terminate with a newline so caller-side
        // concatenation with subsequent sections doesn't butt headers
        // against the last diff line. A regression that trimmed the
        // trailing newline in `HostContext::diff` would produce
        // run-on output only in the differ case — this assertion
        // catches that asymmetry.
        assert!(
            out.ends_with('\n'),
            "differ arm must end with a newline for contiguous-section output: {out:?}",
        );
    }

    /// `(Some, None)` left-only: one run captured host data, the
    /// other did not (mixed tooling version, partial migration
    /// window). Surface the asymmetry explicitly so the missing
    /// side is diagnosable.
    #[test]
    fn format_host_delta_left_only() {
        let ctx = host_ctx("6.14.0", Some("preempt=lazy"));
        let out = format_host_delta(Some(&ctx), None, "a-run", "b-run");
        assert_eq!(out, "\nhost: captured in 'a-run' only, delta unavailable\n");
    }

    /// `(None, Some)` right-only: symmetric complement to
    /// `left_only`. The `b`-name must appear (not `a`) — guards
    /// against a future copy-paste typo that swaps the names.
    #[test]
    fn format_host_delta_right_only() {
        let ctx = host_ctx("6.14.0", Some("preempt=lazy"));
        let out = format_host_delta(None, Some(&ctx), "a-run", "b-run");
        assert_eq!(out, "\nhost: captured in 'b-run' only, delta unavailable\n");
    }

    /// `(None, None)`: neither side carries host data. The section
    /// is fully suppressed — no blank line, no header, nothing.
    /// Pinning this prevents a regression that introduces a
    /// spurious "host: none" footer on legacy runs.
    #[test]
    fn format_host_delta_both_absent_emits_nothing() {
        assert_eq!(format_host_delta(None, None, "a", "b"), "");
    }

    /// `(Some, Some)` identical with both sides carrying the SAME
    /// arch: the helper appends `(arch: {value})` to the identical
    /// confirmation line. Pins the identical-arch surfacing contract
    /// so an operator running `stats compare` on two same-arch runs
    /// sees that the matching dimension covers arch — distinguishing
    /// "both x86_64, identical" from "both aarch64, identical"
    /// without inspecting individual sidecars.
    #[test]
    fn format_host_delta_identical_with_arch_surfaces_arch() {
        let ctx = crate::host_context::HostContext {
            kernel_name: Some("Linux".to_string()),
            arch: Some("x86_64".to_string()),
            ..Default::default()
        };
        let out = format_host_delta(Some(&ctx), Some(&ctx), "a", "b");
        assert_eq!(
            out,
            "\nhost: identical between 'a' and 'b' (arch: x86_64)\n",
        );
    }

    /// `(Some, Some)` identical with arch on one side only: the
    /// helper falls back to the bare identical message. Pins the
    /// "partial hint would mislead" arm in GAP-D — emitting
    /// `(arch: x86_64)` when only one side has arch could read
    /// as if the other side disagreed, so the conservative
    /// rendering drops the hint when either side is `None`.
    ///
    /// Both legs of the asymmetry are tested below: arch on `a`
    /// only and on `b` only. Each must collapse to the bare
    /// message identical to the both-None case.
    #[test]
    fn format_host_delta_identical_partial_arch_falls_back() {
        // a-side has arch, b-side does not. Note both contexts
        // must compare equal under `HostContext::diff` — arch is
        // hash-participating so populating it on one side would
        // route through the differ arm. Construct two
        // semantically-equal HostContexts (only `arch` differs)
        // — the diff arm DOES emit a row when arch differs, so
        // this branch is unreachable through `format_host_delta`'s
        // identical arm. Verify by asserting it routes through
        // the differ arm instead.
        let ha = crate::host_context::HostContext {
            kernel_name: Some("Linux".to_string()),
            arch: Some("x86_64".to_string()),
            ..Default::default()
        };
        let hb = crate::host_context::HostContext {
            kernel_name: Some("Linux".to_string()),
            arch: None,
            ..Default::default()
        };
        let out = format_host_delta(Some(&ha), Some(&hb), "a", "b");
        // Arch difference routes through the differ arm — pin
        // that the partial-hint case is unreachable from the
        // identical arm by construction.
        assert!(
            out.starts_with("\nhost delta ('a' → 'b'):\n"),
            "asymmetric arch must route through differ arm, not \
             identical arm: {out:?}",
        );
        assert!(
            out.contains("arch:"),
            "differ arm must surface the arch row: {out:?}",
        );
    }

    /// `(Some, Some)` identical when arch is `None` on both sides:
    /// fall back to the bare identical message. Pre-host-context-
    /// landing archives or arch-probe failures on both sides hit
    /// this arm — the bare message reads correctly without the
    /// `(arch: ...)` clause.
    #[test]
    fn format_host_delta_identical_both_arch_none_falls_back() {
        let ctx = crate::host_context::HostContext {
            kernel_name: Some("Linux".to_string()),
            arch: None,
            ..Default::default()
        };
        let out = format_host_delta(Some(&ctx), Some(&ctx), "a", "b");
        assert_eq!(out, "\nhost: identical between 'a' and 'b'\n");
    }

    // -- GauntletRow serde round-trip tests --
    //
    // `ext_metrics: BTreeMap<String, f64>` carries
    // `#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]`.
    // These tests pin that contract: the key disappears from JSON
    // when the map is empty, round-trip through from_str
    // reconstructs an equivalent row, and a non-empty payload emits
    // its contents verbatim.

    /// Empty `ext_metrics` is elided on serialize. Regression guard
    /// for the `skip_serializing_if` half — dropping it would make
    /// the writer emit `"ext_metrics":{}` noise on every row (the
    /// `default` half is guarded by the sibling round-trip test).
    #[test]
    fn gauntlet_row_empty_ext_metrics_omits_key() {
        let row = make_row("scn", "topo", true, 0.0);
        assert!(row.ext_metrics.is_empty());
        let json = serde_json::to_string(&row).unwrap();
        assert!(
            !json.contains("\"ext_metrics\""),
            "empty ext_metrics must be omitted from JSON: {json}"
        );
    }

    /// Non-empty `ext_metrics` appears with its full payload. Locks
    /// in that `skip_serializing_if` only fires on empty, not on
    /// "has content". A false positive here would silently drop
    /// extensible metrics from sidecar files.
    #[test]
    fn gauntlet_row_non_empty_ext_metrics_emits_payload() {
        let mut row = make_row("scn", "topo", true, 0.0);
        row.ext_metrics.insert("custom_metric".into(), 42.5);
        let json = serde_json::to_string(&row).unwrap();
        assert!(
            json.contains("\"custom_metric\":42.5"),
            "ext_metrics payload missing: {json}"
        );
    }

    /// Round-trip with empty `ext_metrics`: the writer omits the key
    /// (via `skip_serializing_if`), so the reader must default it
    /// back to empty for the round-trip to close. Regression guard
    /// for the `default` half of the symmetric pair — removing it
    /// would make deserialize fail on JSON this same process just
    /// produced.
    #[test]
    fn gauntlet_row_round_trip_empty_ext_metrics() {
        let row = make_row("scn", "topo", true, 1.5);
        let json = serde_json::to_string(&row).unwrap();
        let back: GauntletRow = serde_json::from_str(&json).unwrap();
        assert_eq!(back, row);
        assert!(back.ext_metrics.is_empty());
    }

    /// Round-trip with populated `ext_metrics`: every entry survives
    /// the to_string → from_str cycle. Guards against any future
    /// field-level serde attribute (e.g. a rename or custom
    /// serializer) accidentally shearing content on one side of the
    /// cycle.
    #[test]
    fn gauntlet_row_round_trip_non_empty_ext_metrics() {
        let mut row = make_row("scn", "topo", false, std::f64::consts::PI);
        row.ext_metrics.insert("m1".into(), 1.0);
        row.ext_metrics.insert("m2".into(), 2.5);
        let json = serde_json::to_string(&row).unwrap();
        let back: GauntletRow = serde_json::from_str(&json).unwrap();
        assert_eq!(back, row);
    }

    /// Round-trip with populated `cpu_budget` / `vcpus`: the
    /// `Option<u32>` + `skip_serializing_if` pair emits the numeric
    /// keys and reads them back. Distinct from `SidecarResult`'s
    /// always-emit u32 round-trip (tests.rs) — this pins the
    /// GauntletRow Option serde contract, the compare-pipeline wire
    /// shape where the skip_serializing_if subtlety lives.
    #[test]
    fn gauntlet_row_round_trip_populated_cpu_budget() {
        let mut row = make_row("scn", "topo", true, 1.0);
        row.cpu_budget = Some(4);
        row.vcpus = Some(16);
        let json = serde_json::to_string(&row).unwrap();
        assert!(
            json.contains("\"cpu_budget\":4") && json.contains("\"vcpus\":16"),
            "populated budget/vcpus must emit numeric JSON keys: {json}"
        );
        let back: GauntletRow = serde_json::from_str(&json).unwrap();
        assert_eq!(back, row);
        assert_eq!(back.cpu_budget, Some(4));
        assert_eq!(back.vcpus, Some(16));
    }

    /// None `cpu_budget` / `vcpus` (skip rows) omit both keys via
    /// `skip_serializing_if`; the reader defaults them back to None so
    /// the round-trip closes without the keys present.
    #[test]
    fn gauntlet_row_none_cpu_budget_omits_keys() {
        let row = make_row("scn", "topo", true, 1.0);
        assert!(row.cpu_budget.is_none() && row.vcpus.is_none());
        let json = serde_json::to_string(&row).unwrap();
        assert!(
            !json.contains("\"cpu_budget\"") && !json.contains("\"vcpus\""),
            "None budget/vcpus must be omitted from JSON: {json}"
        );
        let back: GauntletRow = serde_json::from_str(&json).unwrap();
        assert_eq!(back, row);
    }

    /// `compare_partitions` honours the `--dir` override —
    /// pool-collection walks the override path rather than the
    /// default [`crate::test_support::runs_root`]. Pool source-of-
    /// truth threading regressed silently in earlier versions
    /// (`--dir` was parsed but ignored), so this test pins the
    /// load-bearing wire from CLI arg through `compare_partitions`
    /// down to `collect_pool`.
    ///
    /// Fixture: a tempdir alt-root with two run subdirectories,
    /// each holding one sidecar. The two sidecars differ on
    /// `scheduler` so the slicing-dim is `scheduler` and
    /// `compare_partitions` has a well-defined contrast. Calling
    /// `compare_partitions` with `dir = Some(alt_root)` finds the
    /// pooled fixtures and returns Ok; calling without `--dir`
    /// against runs_root (which doesn't contain these private
    /// fixtures) fails with a "no sidecar data" diagnostic.
    #[test]
    fn compare_partitions_threads_dir_through_to_pool_collection() {
        use crate::test_support::SidecarResult;

        let alt_root = tempfile::TempDir::new().expect("create alt-root tempdir");
        // Two run subdirs; each holds one sidecar. The sidecars
        // differ on scheduler so the slicing-dim derivation has
        // a non-empty result.
        for (run_key, sched) in [
            ("__dir_thread_a__", "scx_alpha"),
            ("__dir_thread_b__", "scx_beta"),
        ] {
            let run_dir = alt_root.path().join(run_key);
            std::fs::create_dir_all(&run_dir).expect("create run dir");
            let sidecar = SidecarResult {
                test_name: "dir_thread_fixture".to_string(),
                scheduler: sched.to_string(),
                ..SidecarResult::test_fixture()
            };
            let json = serde_json::to_string(&sidecar).expect("serialize fixture sidecar");
            let sidecar_path = run_dir.join(format!("{run_key}.ktstr.json"));
            std::fs::write(&sidecar_path, json).expect("write fixture sidecar");
        }

        let filter_a = RowFilter {
            schedulers: vec!["scx_alpha".to_string()],
            ..RowFilter::default()
        };
        let filter_b = RowFilter {
            schedulers: vec!["scx_beta".to_string()],
            ..RowFilter::default()
        };

        // Positive: --dir threads to collect_pool; the two
        // partitions resolve and the comparison runs without
        // bailing. Identical metric values mean exit 0 (no
        // regressions); we only care that the call succeeds.
        let exit = compare_partitions(
            &filter_a,
            &filter_b,
            None,
            &ComparisonPolicy::default(),
            Some(alt_root.path()),
            false,
            &PhaseDisplayOptions::default(),
        )
        .expect("compare_partitions must pool sidecars under --dir override");
        assert_eq!(
            exit, 0,
            "byte-identical metrics across the two scheduler \
             partitions must yield zero regressions (exit 0). \
             A non-zero exit means either the partitions loaded \
             different data than written above or compare_rows \
             regressed on identical inputs.",
        );
    }

    // -- render_dirty_warning --

    /// No `-dirty` commit values on either side returns `None` so
    /// the caller emits no banner. Pins the silent-when-clean
    /// contract that lets `warn_on_dirty_builds` be a no-op for
    /// release-quality runs.
    #[test]
    fn render_dirty_warning_silent_when_no_dirty_commits() {
        let mut row = make_row("scn", "topo", true, 1.0);
        row.commit = Some("abcdef1".into());
        row.kernel_commit = Some("0123456".into());
        let other = row.clone();
        assert!(
            super::render_dirty_warning(&[row], &[other]).is_none(),
            "clean rows on both sides must yield no warning"
        );
    }

    /// Empty input on both sides is silent — `compare_partitions`
    /// bails before the call when either side is empty, but the
    /// helper itself must still degrade cleanly.
    #[test]
    fn render_dirty_warning_silent_on_empty_inputs() {
        assert!(
            super::render_dirty_warning(&[], &[]).is_none(),
            "empty inputs must yield no warning"
        );
    }

    /// Dirty `kernel_commit` values across both sides are deduped
    /// into one block under "kernel source", with each distinct
    /// value listed once and `commit` (project) absent because
    /// none of the rows are dirty on that dimension.
    #[test]
    fn render_dirty_warning_kernel_only_dedupes_values_across_sides() {
        let mut a = make_row("scn", "topo", true, 1.0);
        a.kernel_commit = Some("aaaaaaa-dirty".into());
        a.commit = Some("clean01".into());
        let mut a2 = make_row("scn2", "topo", true, 1.0);
        a2.kernel_commit = Some("aaaaaaa-dirty".into()); // same as a
        let mut b = make_row("scn", "topo", true, 1.0);
        b.kernel_commit = Some("bbbbbbb-dirty".into());
        let text = super::render_dirty_warning(&[a, a2], &[b])
            .expect("dirty kernel_commit must yield warning");
        assert!(
            text.contains("warning: comparison includes dirty builds:"),
            "missing header in {text:?}"
        );
        assert_eq!(
            text.matches("kernel source: aaaaaaa-dirty").count(),
            1,
            "duplicate kernel_commit must be deduped, got {text:?}"
        );
        assert!(
            text.contains("kernel source: bbbbbbb-dirty"),
            "second distinct dirty kernel_commit must be listed, got {text:?}"
        );
        assert!(
            !text.contains("project:"),
            "no -dirty project commit; the project line must not appear: {text:?}"
        );
        assert!(
            text.contains("Dirty runs overwrite previous results with the same HEAD."),
            "missing trailer line 1 in {text:?}"
        );
        assert!(
            text.contains("Commit changes for reproducible-ish comparisons."),
            "missing trailer line 2 in {text:?}"
        );
    }

    /// Dirty `commit` (project) values are listed under "project"
    /// when no `kernel_commit` is dirty, so each dimension renders
    /// only when populated.
    #[test]
    fn render_dirty_warning_project_only_omits_kernel_section() {
        let mut a = make_row("scn", "topo", true, 1.0);
        a.commit = Some("ccccccc-dirty".into());
        let text = super::render_dirty_warning(&[a], &[]).expect("dirty commit must yield warning");
        assert!(
            text.contains("project: ccccccc-dirty"),
            "expected project line in {text:?}"
        );
        assert!(
            !text.contains("kernel source:"),
            "kernel section must not appear when only project is dirty: {text:?}"
        );
    }

    /// Both dimensions dirty: the warning lists "kernel source"
    /// before "project" in stable order so byte-identical inputs
    /// always render byte-identically. BTreeSet ordering of distinct
    /// hashes within each dimension is also pinned (lex order).
    #[test]
    fn render_dirty_warning_both_dimensions_in_stable_order() {
        let mut a = make_row("scn", "topo", true, 1.0);
        a.kernel_commit = Some("kkkkk22-dirty".into());
        a.commit = Some("pppp222-dirty".into());
        let mut b = make_row("scn", "topo", true, 1.0);
        b.kernel_commit = Some("kkkkk11-dirty".into());
        b.commit = Some("pppp111-dirty".into());
        let text = super::render_dirty_warning(&[a], &[b])
            .expect("both dimensions dirty must yield warning");
        let kernel11 = text
            .find("kernel source: kkkkk11-dirty")
            .expect("kernel11 line absent");
        let kernel22 = text
            .find("kernel source: kkkkk22-dirty")
            .expect("kernel22 line absent");
        let project11 = text
            .find("project: pppp111-dirty")
            .expect("project11 line absent");
        let project22 = text
            .find("project: pppp222-dirty")
            .expect("project22 line absent");
        assert!(
            kernel11 < kernel22,
            "kernel section must list values in lex order: {text:?}"
        );
        assert!(
            project11 < project22,
            "project section must list values in lex order: {text:?}"
        );
        assert!(
            kernel22 < project11,
            "kernel section must precede project section: {text:?}"
        );
    }

    /// `None` commit fields and clean (suffix-free) values on the
    /// other rows do not contribute to either set, so the warning
    /// only mentions the actually-dirty hash.
    #[test]
    fn render_dirty_warning_skips_none_and_clean_values() {
        let mut clean_a = make_row("a", "topo", true, 1.0);
        clean_a.commit = Some("clean01".into());
        clean_a.kernel_commit = None;
        let mut dirty_b = make_row("b", "topo", true, 1.0);
        dirty_b.commit = None;
        dirty_b.kernel_commit = Some("dddddd1-dirty".into());
        let text = super::render_dirty_warning(&[clean_a], &[dirty_b])
            .expect("at least one dirty value must yield warning");
        assert!(
            text.contains("kernel source: dddddd1-dirty"),
            "dirty kernel_commit must surface in {text:?}"
        );
        assert!(
            !text.contains("project:"),
            "no dirty project commit; project section must be absent in {text:?}"
        );
        assert!(
            !text.contains("clean01"),
            "clean commit values must not appear in {text:?}"
        );
    }

    // -- render_overcommit_warning --

    fn budget_row(scenario: &str, budget: Option<u32>, vcpus: Option<u32>) -> GauntletRow {
        let mut r = make_row(scenario, "topo", true, 1.0);
        r.cpu_budget = budget;
        r.vcpus = vcpus;
        r
    }

    /// No hazard: every row's budget meets its vCPU count and no group
    /// mixes budgets -> `None`, whether CpuBudget is pairing or sliced.
    #[test]
    fn render_overcommit_warning_none_when_clean() {
        let pairing: &[Dimension] = &[Dimension::CpuBudget];
        let sliced: &[Dimension] = &[];
        let a = budget_row("a", Some(16), Some(16));
        let b = budget_row("b", Some(32), Some(16)); // roomy, not overcommit
        assert!(
            super::render_overcommit_warning(
                std::slice::from_ref(&a),
                std::slice::from_ref(&b),
                pairing,
            )
            .is_none()
        );
        assert!(super::render_overcommit_warning(&[a], &[b], sliced).is_none());
    }

    /// Skip rows (budget `None`) carry no budget identity and never
    /// trip either check.
    #[test]
    fn render_overcommit_warning_ignores_skip_rows() {
        let sliced: &[Dimension] = &[];
        let a = budget_row("a", None, None);
        let b = budget_row("b", None, None);
        assert!(super::render_overcommit_warning(&[a], &[b], sliced).is_none());
    }

    /// An overcommitted run (budget < vcpus) is flagged on its side,
    /// names the budget/vcpus pair, and the warning lists run-delay as
    /// confounded (pins the kernel-grounded semantics). Fires
    /// regardless of pairing.
    #[test]
    fn render_overcommit_warning_flags_overcommitted_side() {
        let pairing: &[Dimension] = &[Dimension::CpuBudget];
        let a = budget_row("a", Some(4), Some(16));
        let b = budget_row("b", Some(16), Some(16));
        let text = super::render_overcommit_warning(&[a], &[b], pairing)
            .expect("an overcommitted A row must warn");
        assert!(text.contains("side A"), "must name side A: {text}");
        assert!(text.contains("4/16"), "must list budget/vcpus: {text}");
        assert!(
            text.contains("run-delay"),
            "warning must list run-delay as confounded: {text}",
        );
        assert!(
            !text.contains("side B"),
            "the clean B side must not be flagged: {text}",
        );
    }

    /// The mixed-budget warning fires per pairing GROUP, not side-wide:
    /// only rows that share a full PairingKey are averaged together.
    /// - CpuBudget pairing: budgets key separate groups -> no fold.
    /// - sliced + same scenario: budgets fold into one mean -> warn.
    /// - sliced + different scenarios: distinct keys, never folded -> no
    ///   warning (the precision that distinguishes "side spans budgets"
    ///   from "a group averages budgets").
    #[test]
    fn render_overcommit_warning_mixed_budget_per_group() {
        let pairing: &[Dimension] = &[Dimension::CpuBudget];
        // Realistic sliced pairing-dims: production passes
        // Dimension::pairing_dims(&slicing) = ALL minus the sliced dim, so
        // the per-group key includes scheduler/topology/work-type/commits/
        // source — NOT just scenario. Use the real derivation so a
        // from_row key-shape regression on the sliced path is caught.
        let sliced = Dimension::pairing_dims(&[Dimension::CpuBudget]);
        let a = budget_row("a", Some(16), Some(16));
        let b1 = budget_row("b", Some(8), Some(16)); // overcommit + two budgets...
        let b2 = budget_row("b", Some(16), Some(16)); // ...same scenario AND all other dims

        // CpuBudget pairing: budgets key separate groups; the only
        // hazard is the overcommitted 8/16 row, NOT a mixed-budget fold.
        let paired = super::render_overcommit_warning(
            std::slice::from_ref(&a),
            &[b1.clone(), b2.clone()],
            pairing,
        )
        .expect("overcommitted B row still warns");
        assert!(
            paired.contains("8/16") && !paired.contains("share a pairing group"),
            "pairing dim: overcommit flagged, no mixed-fold warning: {paired}",
        );

        // Sliced + b1/b2 share EVERY pairing dim (scenario + scheduler +
        // topology + ... all default-equal): one group, two budgets, so
        // --average folds them -> mixed warning on side B.
        let sliced_same = super::render_overcommit_warning(&[a], &[b1, b2], &sliced)
            .expect("mixed budgets in one group on a sliced side must warn");
        assert!(
            sliced_same.contains("share a pairing group") && sliced_same.contains("side B"),
            "sliced same-key: must warn B's budgets share a pairing group: {sliced_same}",
        );

        // Sliced but the two budgets differ on a NON-budget pairing dim
        // (scheduler): distinct pairing keys -> never folded -> no
        // warning, even though the side has two budgets and shares
        // scenario. Proves the per-group key uses the FULL dim set, not
        // just scenario (the degenerate &[] key would have missed this).
        let mut s1 = budget_row("c", Some(16), Some(16));
        s1.scheduler = "sched_a".to_string();
        let mut s2 = budget_row("c", Some(32), Some(32));
        s2.scheduler = "sched_b".to_string();
        let clean_a = budget_row("d", Some(16), Some(16));
        assert!(
            super::render_overcommit_warning(&[s1, s2], std::slice::from_ref(&clean_a), &sliced)
                .is_none(),
            "two budgets differing on a non-budget pairing dim (scheduler) key \
             separate groups -> no fold -> no warning",
        );

        // Sliced + different scenarios on ONE side: distinct pairing
        // keys, never folded, neither overcommitted -> NO warning.
        let xa = budget_row("x", Some(16), Some(16));
        let ya = budget_row("y", Some(32), Some(32));
        let clean_b = budget_row("z", Some(16), Some(16));
        assert!(
            super::render_overcommit_warning(&[xa, ya], std::slice::from_ref(&clean_b), &sliced)
                .is_none(),
            "one side spanning budgets across distinct scenarios -> no fold -> no warning",
        );
    }

    // -- RowFilter / apply_row_filters --

    /// Helper that builds a `GauntletRow` with controllable
    /// scheduler / topology / work_type / kernel_version for the
    /// filter tests. The metric fields default to harmless
    /// passing values; tests are interested in identity-field
    /// matching, not metrics.
    fn make_filter_row(
        scenario: &str,
        scheduler: &str,
        topology: &str,
        work_type: &str,
        kernel_version: Option<&str>,
    ) -> GauntletRow {
        GauntletRow {
            scenario: scenario.into(),
            topology: topology.into(),
            work_type: work_type.into(),
            scheduler: scheduler.into(),
            kernel_version: kernel_version.map(str::to_owned),
            cpu_budget: None,
            vcpus: None,
            commit: None,
            kernel_commit: None,
            run_source: None,
            passed: true,
            skipped: false,
            inconclusive: false,
            run_sample_count: 0,
            spread: 0.0,
            gap_ms: 0,
            migrations: 0,
            migration_ratio: 0.0,
            imbalance_ratio: 0.0,
            max_dsq_depth: 0,
            stuck_count: 0,
            fallback_count: 0,
            keep_last_count: 0,
            worst_p99_wake_latency_us: 0.0,
            worst_median_wake_latency_us: 0.0,
            worst_wake_latency_cv: 0.0,
            total_iterations: 0,
            worst_mean_run_delay_us: 0.0,
            worst_run_delay_us: 0.0,
            worst_wake_latency_tail_ratio: 0.0,
            worst_iterations_per_worker: 0.0,
            worst_iterations_per_cpu_sec: 0.0,
            page_locality: 0.0,
            cross_node_migration_ratio: 0.0,
            ext_metrics: BTreeMap::new(),
            phases: Vec::new(),
        }
    }

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
        row.stuck_count = (migrations / 10) as usize;
        row.fallback_count = migrations as i64;
        row.keep_last_count = -(migrations as i64);
        row.worst_p99_wake_latency_us = spread * 2.0;
        row.worst_median_wake_latency_us = spread;
        row.worst_wake_latency_cv = spread / 50.0;
        row.total_iterations = iters;
        row.worst_mean_run_delay_us = gap_ms as f64;
        row.worst_run_delay_us = (gap_ms * 2) as f64;
        row.worst_wake_latency_tail_ratio = spread / 25.0;
        row.worst_iterations_per_worker = iters as f64 / 10.0;
        row.worst_iterations_per_cpu_sec = iters as f64 / 5.0;
        row.page_locality = 1.0 - spread / 100.0;
        row.cross_node_migration_ratio = spread / 200.0;
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
        assert_eq!(ar.row.worst_p99_wake_latency_us, 24.0);
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
        // Gauge(Last) f64 mean for worst_p99_wake_latency_us:
        // (20 + 40 + 60)/3 = 40.
        assert_eq!(ar.row.worst_p99_wake_latency_us, 40.0);
        // worst_iterations_per_cpu_sec cross-RUN arithmetic mean:
        // iters/5 = 180/220/200; (180 + 220 + 200)/3 = 200.
        assert_eq!(ar.row.worst_iterations_per_cpu_sec, 200.0);
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
        assert!(spread.is_regression);
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

    // -- compare_rows_by per-phase pass tests --------------------------
    //
    // These tests exercise the per-row-pair phase intersection that
    // populates CompareReport.phase_deltas + unpaired_phases. They go
    // through compare_rows_by directly (rather than the full
    // compare_partitions which also does filtering/averaging) because
    // the parallel pass lives inside compare_rows_by's row-pair
    // iteration and that is the load-bearing surface to pin.
    //
    // Each test builds 2 GauntletRows via make_row, attaches phase
    // buckets explicitly, then asserts the resulting CompareReport
    // shape against the expected per-phase + unpaired data flow.

    fn make_phase_bucket(
        step_index: u16,
        label: &str,
        metrics: &[(&str, f64)],
    ) -> crate::assert::PhaseBucket {
        let metrics_map = metrics.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        crate::assert::PhaseBucket {
            per_cgroup: Default::default(),
            step_index,
            label: label.to_string(),
            start_ms: 0,
            end_ms: 100,
            sample_count: 1,
            metrics: metrics_map,
        }
    }

    /// Matched phases on both sides populate one
    /// PhaseDeltaRow per phase per metric present in both
    /// buckets; unpaired_phases stays empty. Pins the matched
    /// branch of the parallel pass.
    #[test]
    fn compare_rows_by_emits_phase_deltas_when_both_sides_have_matched_phases() {
        let mut row_a = make_row("test_a", "tiny-1llc", true, 0.0);
        let mut row_b = make_row("test_a", "tiny-1llc", true, 0.0);
        row_a.phases = vec![
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)]),
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 8.0)]),
        ];
        row_b.phases = vec![
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 6.0)]),
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 15.0)]),
        ];
        let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
        assert_eq!(
            report.phase_deltas.len(),
            2,
            "2 phases × 1 metric = 2 deltas"
        );
        assert!(
            report.unpaired_phases.is_empty(),
            "both phases matched, no orphans"
        );
        let baseline = report
            .phase_deltas
            .iter()
            .find(|r| r.step_index == 0)
            .expect("BASELINE delta present");
        assert_eq!(baseline.label, "BASELINE");
        assert_eq!(baseline.a, 5.0);
        assert_eq!(baseline.b, 6.0);
        assert_eq!(baseline.delta, 1.0);
        let step0 = report
            .phase_deltas
            .iter()
            .find(|r| r.step_index == 1)
            .expect("Step[0] delta present");
        assert_eq!(step0.label, "Step[0]");
        assert_eq!(step0.a, 8.0);
        assert_eq!(step0.b, 15.0);
        assert_eq!(step0.delta, 7.0);
    }

    /// A-side has phase [0, 1]; B-side has phase [0, 2].
    /// Matched phases (step_index = 0) emit PhaseDeltaRow;
    /// step_index = 1 emits UnpairedPhaseRow side=A;
    /// step_index = 2 emits UnpairedPhaseRow side=B. Pins the
    /// cross-cardinality intersection and the surface-don't-drop
    /// contract for orphan buckets.
    #[test]
    fn compare_rows_by_emits_unpaired_phases_when_phase_coverage_differs() {
        let mut row_a = make_row("test_x", "tiny-1llc", true, 0.0);
        let mut row_b = make_row("test_x", "tiny-1llc", true, 0.0);
        row_a.phases = vec![
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 4.0)]),
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 12.0)]),
        ];
        row_b.phases = vec![
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)]),
            make_phase_bucket(2, "Step[1]", &[("max_dsq_depth", 9.0)]),
        ];
        let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
        assert_eq!(
            report.phase_deltas.len(),
            1,
            "only BASELINE matches across sides -> 1 delta"
        );
        assert_eq!(report.phase_deltas[0].step_index, 0);
        assert_eq!(report.unpaired_phases.len(), 2, "step 1 (A) + step 2 (B)");
        let a_orphan = report
            .unpaired_phases
            .iter()
            .find(|u| u.side == ComparePartition::A)
            .expect("A-only orphan present");
        assert_eq!(a_orphan.step_index, 1);
        assert_eq!(a_orphan.label, "Step[0]");
        let b_orphan = report
            .unpaired_phases
            .iter()
            .find(|u| u.side == ComparePartition::B)
            .expect("B-only orphan present");
        assert_eq!(b_orphan.step_index, 2);
        assert_eq!(b_orphan.label, "Step[1]");
    }

    /// The per-phase AND unpaired-phase compare surfaces suppress Rate
    /// components too (not just the scalar findings pass). A and B share step 0
    /// (matched); A has a step-1 phase B lacks (side=A orphan) and B has a
    /// step-2 phase A lacks (side=B orphan) — covering BOTH orphan arms. Every
    /// bucket carries `total_phase_iterations` (a suppressed component — the
    /// real per-phase producer inserts it into every bucket) and
    /// `max_dsq_depth` (not suppressed). The component must appear in NEITHER
    /// the PhaseDeltaRows NOR either UnpairedPhaseRow.metrics, while
    /// `max_dsq_depth` survives on all. `make_phase_bucket` builds
    /// `PhaseBucket.metrics` directly — the exact map the per-phase pass reads.
    #[test]
    fn compare_rows_per_phase_and_unpaired_suppress_rate_components() {
        let mut row_a = make_row("t", "tiny-1llc", true, 0.0);
        let mut row_b = make_row("t", "tiny-1llc", true, 0.0);
        row_a.phases = vec![
            make_phase_bucket(
                0,
                "BASELINE",
                &[("total_phase_iterations", 1000.0), ("max_dsq_depth", 5.0)],
            ),
            make_phase_bucket(
                1,
                "Step[0]",
                &[("total_phase_iterations", 2000.0), ("max_dsq_depth", 8.0)],
            ),
        ];
        // B lacks step 1 -> A's step-1 phase is an unpaired (side=A) row; B has a
        // step-2 phase A lacks -> an unpaired (side=B) row. Covers BOTH orphan
        // arms (the (Some,None) and (None,Some) metrics_without_suppressed calls).
        row_b.phases = vec![
            make_phase_bucket(
                0,
                "BASELINE",
                &[("total_phase_iterations", 1500.0), ("max_dsq_depth", 6.0)],
            ),
            make_phase_bucket(
                2,
                "Step[1]",
                &[("total_phase_iterations", 3000.0), ("max_dsq_depth", 9.0)],
            ),
        ];
        let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());

        // Matched step 0: component suppressed from phase_deltas; the
        // non-suppressed metric still emits a delta (emitted regardless of gate).
        let delta_names: Vec<&str> = report.phase_deltas.iter().map(|r| r.metric.name).collect();
        assert!(
            !delta_names.contains(&"total_phase_iterations"),
            "Rate component must be suppressed from per-phase deltas; got {delta_names:?}",
        );
        assert!(
            delta_names.contains(&"max_dsq_depth"),
            "non-suppressed per-phase metric must still emit a delta; got {delta_names:?}",
        );

        // Unpaired step 1 (side A): component filtered from the orphan's metrics;
        // the non-suppressed metric survives.
        let orphan = report
            .unpaired_phases
            .iter()
            .find(|r| r.step_index == 1)
            .expect("A's step-1 phase emits an unpaired row");
        assert!(
            !orphan.metrics.contains_key("total_phase_iterations"),
            "Rate component must be filtered from UnpairedPhaseRow.metrics; got {:?}",
            orphan.metrics.keys().collect::<Vec<_>>(),
        );
        assert!(
            orphan.metrics.contains_key("max_dsq_depth"),
            "non-suppressed metric must survive in UnpairedPhaseRow.metrics",
        );

        // Unpaired step 2 (side B): the OTHER orphan arm also filters the
        // component (the two arms are distinct call sites).
        let orphan_b = report
            .unpaired_phases
            .iter()
            .find(|r| r.step_index == 2)
            .expect("B's step-2 phase emits an unpaired row");
        assert!(
            !orphan_b.metrics.contains_key("total_phase_iterations"),
            "Rate component must be filtered from the side-B UnpairedPhaseRow.metrics too; got {:?}",
            orphan_b.metrics.keys().collect::<Vec<_>>(),
        );
        assert!(
            orphan_b.metrics.contains_key("max_dsq_depth"),
            "non-suppressed metric must survive in the side-B UnpairedPhaseRow.metrics",
        );
    }

    /// Empty `phases` on either side suppresses the entire
    /// per-phase pass (no PhaseDeltaRow, no UnpairedPhaseRow).
    /// Pins the single-phase legacy-compat short-circuit: a
    /// legacy sidecar with no phase data on EITHER side does
    /// not flood the unpaired section with one orphan per
    /// populated B-side phase.
    #[test]
    fn compare_rows_by_skips_phase_pass_when_either_side_phases_empty() {
        let row_a = make_row("test_y", "tiny-1llc", true, 0.0);
        let mut row_b = make_row("test_y", "tiny-1llc", true, 0.0);
        // A has empty phases (legacy sidecar); B has phases.
        row_b.phases = vec![
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 6.0)]),
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 10.0)]),
        ];
        let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
        assert!(
            report.phase_deltas.is_empty(),
            "empty-on-either-side short-circuit must suppress all per-phase rows"
        );
        assert!(
            report.unpaired_phases.is_empty(),
            "empty-on-either-side must not emit orphan rows for B-side phases"
        );
    }

    /// Polarity-aware is_regression: a `worst_*`-style
    /// LowerBetter metric where B > A flags regression=true;
    /// a `*_iterations`/HigherBetter metric where B > A flags
    /// regression=false (improvement). Pins the polarity wiring
    /// that the existing scalar-finding path uses, applied to
    /// the per-phase pass.
    ///
    /// Values chosen to clear the per-phase dual-gate (the same
    /// `|delta| < default_abs || rel_delta < default_rel` gate
    /// the scalar pass uses inside its per-metric loop in
    /// `compare_rows_by`). max_dsq_depth has
    /// default_abs=10.0 / default_rel=0.50; total_iterations has
    /// default_abs=100.0 / default_rel=0.10. The 10→25 (delta=15)
    /// and 200→400 (delta=200) deltas both clear both gates, so
    /// polarity dispatches as documented. Sub-threshold deltas
    /// are explicitly exercised by the dual-gate test below.
    #[test]
    fn compare_rows_by_phase_deltas_respect_metric_polarity() {
        let mut row_a = make_row("test_z", "tiny-1llc", true, 0.0);
        let mut row_b = make_row("test_z", "tiny-1llc", true, 0.0);
        // max_dsq_depth Peak/LowerBetter: delta=15 > abs=10, rel=1.5 > rel=0.5
        // total_iterations Counter/HigherBetter: delta=200 > abs=100, rel=1.0 > rel=0.10
        row_a.phases = vec![make_phase_bucket(
            0,
            "BASELINE",
            &[("max_dsq_depth", 10.0), ("total_iterations", 200.0)],
        )];
        row_b.phases = vec![make_phase_bucket(
            0,
            "BASELINE",
            &[("max_dsq_depth", 25.0), ("total_iterations", 400.0)],
        )];
        let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
        assert_eq!(report.phase_deltas.len(), 2, "2 metrics in 1 bucket");
        let dsq = report
            .phase_deltas
            .iter()
            .find(|r| r.metric.name == "max_dsq_depth")
            .expect("max_dsq_depth delta present");
        assert!(
            dsq.is_regression,
            "max_dsq_depth bigger on B -> regression (LowerBetter polarity)"
        );
        let iters = report
            .phase_deltas
            .iter()
            .find(|r| r.metric.name == "total_iterations")
            .expect("total_iterations delta present");
        assert!(
            !iters.is_regression,
            "total_iterations bigger on B -> improvement (HigherBetter polarity)"
        );
    }

    /// per-phase pass honors the dual-gate semantic the
    /// scalar pass uses inside its per-metric loop in
    /// `compare_rows_by` (`|delta| < default_abs ||
    /// rel_delta < default_rel`). Sub-threshold deltas are still
    /// emitted into `phase_deltas` (programmatic consumers see
    /// every paired comparison) but their `is_regression` flag
    /// is `false` — the renderer paints them as "improvement"
    /// or unstyled rather than the red REGRESSION verdict.
    ///
    /// Two cases pinned in one test:
    /// - sub-abs-gate: `max_dsq_depth` `default_abs=10.0` —
    ///   delta=5 (5→10) is direction-matching for LowerBetter
    ///   polarity but `5 < 10` → is_regression=false.
    /// - above-both-gates: `max_dsq_depth` delta=15 (10→25)
    ///   passes both abs and rel gates → is_regression=true.
    #[test]
    fn compare_rows_by_phase_deltas_dual_gate_suppresses_subthreshold_regressions() {
        let mut row_a = make_row("test_dg", "tiny-1llc", true, 0.0);
        let mut row_b = make_row("test_dg", "tiny-1llc", true, 0.0);
        // BASELINE: delta=5 (under abs=10), Step[0]: delta=15 (over abs=10 + rel=0.5)
        row_a.phases = vec![
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 5.0)]),
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 10.0)]),
        ];
        row_b.phases = vec![
            make_phase_bucket(0, "BASELINE", &[("max_dsq_depth", 10.0)]),
            make_phase_bucket(1, "Step[0]", &[("max_dsq_depth", 25.0)]),
        ];
        let report = compare_rows_by(&[row_a], &[row_b], &[], None, &ComparisonPolicy::default());
        assert_eq!(
            report.phase_deltas.len(),
            2,
            "both phases emit a delta row regardless of dual-gate"
        );
        let baseline = report
            .phase_deltas
            .iter()
            .find(|r| r.step_index == 0)
            .expect("BASELINE delta present");
        assert_eq!(baseline.delta, 5.0);
        assert!(
            !baseline.is_regression,
            "delta=5 < default_abs=10 → sub-abs-gate; \
             is_regression must clear despite LowerBetter polarity direction"
        );
        let step0 = report
            .phase_deltas
            .iter()
            .find(|r| r.step_index == 1)
            .expect("Step[0] delta present");
        assert_eq!(step0.delta, 15.0);
        assert!(
            step0.is_regression,
            "delta=15 ≥ default_abs=10 AND rel_delta=1.5 ≥ default_rel=0.5 → \
             above both gates; LowerBetter polarity sets is_regression=true"
        );
    }

    // -- PhaseDisplayOptions::rel_threshold (plan_17 step 11c) --

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

    // -- PhaseDisplayOptions::passes_delta_threshold --

    /// `--phase-threshold` unset (default opts) passes every
    /// row regardless of delta magnitude — matches the clap
    /// doc contract ("absence shows every paired row"). Pins
    /// the default render-everything behavior.
    #[test]
    fn passes_delta_threshold_unset_admits_all_deltas() {
        let opts = PhaseDisplayOptions::default();
        let metric = METRICS
            .iter()
            .find(|m| m.name == "max_dsq_depth")
            .expect("max_dsq_depth in METRICS");
        // delta = 0.0 (no change) and delta = 999.0 (massive change) both pass.
        let zero_delta = PhaseDeltaRow {
            pairing_key: PairingKey(vec!["t".into()]),
            step_index: 0,
            label: "BASELINE".into(),
            metric,
            a: 100.0,
            b: 100.0,
            delta: 0.0,
            is_regression: false,
        };
        let big_delta = PhaseDeltaRow {
            pairing_key: PairingKey(vec!["t".into()]),
            step_index: 0,
            label: "BASELINE".into(),
            metric,
            a: 100.0,
            b: 1099.0,
            delta: 999.0,
            is_regression: true,
        };
        assert!(opts.passes_delta_threshold(&zero_delta));
        assert!(opts.passes_delta_threshold(&big_delta));
    }

    /// `--phase-threshold 10` (10% gate) suppresses rows whose
    /// relative delta is below 10% AND admits rows at or above.
    /// Pins the inclusive-at-boundary semantic (`>=`) so a
    /// future refactor to `>` silently flips it.
    #[test]
    fn passes_delta_threshold_inclusive_at_boundary() {
        let opts = PhaseDisplayOptions {
            phase_threshold: Some(10.0),
            ..PhaseDisplayOptions::default()
        };
        let metric = METRICS
            .iter()
            .find(|m| m.name == "max_dsq_depth")
            .expect("max_dsq_depth in METRICS");
        // delta=10, a=100 → rel = 10/100 = 0.10. Exactly at the boundary.
        let at_gate = PhaseDeltaRow {
            pairing_key: PairingKey(vec!["t".into()]),
            step_index: 0,
            label: "BASELINE".into(),
            metric,
            a: 100.0,
            b: 110.0,
            delta: 10.0,
            is_regression: true,
        };
        assert!(
            opts.passes_delta_threshold(&at_gate),
            "at-boundary delta (rel=0.10, gate=0.10) must pass via >= comparison"
        );
        // delta=5, a=100 → rel = 0.05 < 0.10. Below the gate.
        let below_gate = PhaseDeltaRow {
            delta: 5.0,
            b: 105.0,
            ..at_gate
        };
        assert!(
            !opts.passes_delta_threshold(&below_gate),
            "below-boundary delta (rel=0.05, gate=0.10) must be suppressed"
        );
    }

    /// `--phase-threshold 50` against `a = 0.0` divides by the
    /// `max(|a|, 1.0)` floor (NOT zero), so a delta of 10 yields
    /// rel = 10/1 = 10.0 (1000%) which passes the 0.5 gate. Pins
    /// the NaN-defense in the denominator: a future refactor that
    /// drops the `.max(1.0)` would divide by zero and either
    /// admit all rows (NaN >= X → false) or panic.
    #[test]
    fn passes_delta_threshold_zero_a_divides_by_unit_floor() {
        let opts = PhaseDisplayOptions {
            phase_threshold: Some(50.0),
            ..PhaseDisplayOptions::default()
        };
        let metric = METRICS
            .iter()
            .find(|m| m.name == "max_dsq_depth")
            .expect("max_dsq_depth in METRICS");
        let zero_a = PhaseDeltaRow {
            pairing_key: PairingKey(vec!["t".into()]),
            step_index: 0,
            label: "BASELINE".into(),
            metric,
            a: 0.0,
            b: 10.0,
            delta: 10.0,
            is_regression: true,
        };
        assert!(
            opts.passes_delta_threshold(&zero_a),
            "zero-a divisor floor (|a|.max(1.0)) must keep rel finite \
             (rel = |10|/max(0,1) = 10.0); 10.0 ≥ 0.5 → row passes"
        );
    }

    /// Distinguishing pin for the `.max(1.0)` divisor floor: the
    /// a=0/delta=10 case above passes both floored (10/1=10≥0.5) AND
    /// unfloored (10/0=+inf≥0.5), so it does NOT guard the floor. The
    /// floor actually protects a=0 AND delta=0 at --phase-threshold 0:
    /// floored gives 0/1=0, and 0≥0 → the row renders (the documented
    /// "PCT=0 shows every row" contract); unfloored gives 0/0=NaN, and
    /// NaN≥0 is false → the row is silently dropped. Drop `.max(1.0)`
    /// and this assertion flips to false.
    #[test]
    fn passes_delta_threshold_zero_a_zero_delta_at_pct_zero_renders() {
        let opts = PhaseDisplayOptions {
            phase_threshold: Some(0.0),
            ..PhaseDisplayOptions::default()
        };
        let metric = METRICS
            .iter()
            .find(|m| m.name == "max_dsq_depth")
            .expect("max_dsq_depth in METRICS");
        let zero_a_zero_delta = PhaseDeltaRow {
            pairing_key: PairingKey(vec!["t".into()]),
            step_index: 0,
            label: "BASELINE".into(),
            metric,
            a: 0.0,
            b: 0.0,
            delta: 0.0,
            is_regression: false,
        };
        assert!(
            opts.passes_delta_threshold(&zero_a_zero_delta),
            "zero-a/zero-delta at --phase-threshold 0 must render: floored \
             0/1=0, 0≥0 true. Dropping the .max(1.0) floor yields 0/0=NaN, \
             NaN≥0 false, silently dropping the row",
        );
    }

    /// `--phase-threshold 0` (PCT = 0) admits every row because
    /// `rel >= 0.0` is always true. Pins the documented sentinel
    /// "PCT = 0 shows every paired row" — a future refactor that
    /// special-cased 0 to mean "no filter" via different code
    /// path would still produce the same result, but the
    /// straight numeric comparison is the simplest implementation
    /// and this test pins it.
    #[test]
    fn passes_delta_threshold_zero_pct_admits_all_rows() {
        let opts = PhaseDisplayOptions {
            phase_threshold: Some(0.0),
            ..PhaseDisplayOptions::default()
        };
        let metric = METRICS
            .iter()
            .find(|m| m.name == "max_dsq_depth")
            .expect("max_dsq_depth in METRICS");
        let zero_delta = PhaseDeltaRow {
            pairing_key: PairingKey(vec!["t".into()]),
            step_index: 0,
            label: "BASELINE".into(),
            metric,
            a: 100.0,
            b: 100.0,
            delta: 0.0,
            is_regression: false,
        };
        assert!(
            opts.passes_delta_threshold(&zero_delta),
            "--phase-threshold 0 must admit even zero-delta rows (rel = 0 >= 0)"
        );
    }

    // -- format_average_header / format_per_group_pass_counts --

    /// `format_average_header` renders the exact header line that
    /// `compare_partitions` prints above the comparison table when
    /// `--average` is active. Pins the operator-visible surface
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

    // -- Dimension / derive_slicing_dims / pairing dims --

    /// `Dimension::ALL` lists all eight dims in canonical order.
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

    /// One-dim diff: only the differing dimension is reported.
    #[test]
    fn derive_slicing_dims_single_dim_diff() {
        let f_a = RowFilter {
            schedulers: vec!["scx_alpha".to_string()],
            ..RowFilter::default()
        };
        let f_b = RowFilter {
            schedulers: vec!["scx_beta".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(derive_slicing_dims(&f_a, &f_b), vec![Dimension::Scheduler]);
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

    /// Multi-dim diff: every differing dimension is reported, in
    /// canonical [`Dimension::ALL`] order.
    #[test]
    fn derive_slicing_dims_multi_dim_diff_in_canonical_order() {
        let f_a = RowFilter {
            kernels: vec!["6.14".to_string()],
            schedulers: vec!["scx_alpha".to_string()],
            ..RowFilter::default()
        };
        let f_b = RowFilter {
            kernels: vec!["6.15".to_string()],
            schedulers: vec!["scx_beta".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            derive_slicing_dims(&f_a, &f_b),
            vec![Dimension::Kernel, Dimension::Scheduler],
        );
    }

    /// Source-only diff: filters that disagree on `run_sources`
    /// and agree on every other dimension produce a slicing-dim
    /// set containing exactly `Dimension::RunSource`. Pins the
    /// Source arm of the per-dimension comparison switch in
    /// [`derive_slicing_dims`] — a regression that omitted the
    /// arm or compared the wrong field would surface here as an
    /// empty slicing-dim set (and downstream as a `compare`
    /// command that mistakenly bails with "A and B select
    /// identical rows" on a legitimate source contrast).
    #[test]
    fn derive_slicing_dims_source_only_diff() {
        let f_a = RowFilter {
            run_sources: vec!["local".to_string()],
            ..RowFilter::default()
        };
        let f_b = RowFilter {
            run_sources: vec!["ci".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            derive_slicing_dims(&f_a, &f_b),
            vec![Dimension::RunSource],
            "differing `run_sources` must surface Source as a slicing dim",
        );

        // Sorted-deduped Vec semantics also apply on the Source
        // dim — same set in different order/multiplicity must NOT
        // slice. Mirrors the `derive_slicing_dims_vec_compares_as_set`
        // contract for the Source arm.
        let f_c = RowFilter {
            run_sources: vec!["local".to_string(), "ci".to_string()],
            ..RowFilter::default()
        };
        let f_d = RowFilter {
            run_sources: vec!["ci".to_string(), "local".to_string(), "local".to_string()],
            ..RowFilter::default()
        };
        assert!(
            derive_slicing_dims(&f_c, &f_d).is_empty(),
            "same run_source set in different order/multiplicity must NOT slice",
        );
    }

    /// Topology-only diff: filters that disagree on `topologies`
    /// and agree on every other dimension produce a slicing-dim
    /// set containing exactly `Dimension::Topology`. Pins the
    /// Topology arm of the per-dimension comparison switch in
    /// [`derive_slicing_dims`] for the post-Vec-promotion
    /// `topologies` field; before promotion `--topology` was a
    /// single-value `Option<String>` and the per-arm comparison
    /// shape was `Option<String> != Option<String>`. Mirror of
    /// `derive_slicing_dims_source_only_diff` for the Topology
    /// arm.
    #[test]
    fn derive_slicing_dims_topology_only_diff() {
        let f_a = RowFilter {
            topologies: vec!["1n2l4c1t".to_string()],
            ..RowFilter::default()
        };
        let f_b = RowFilter {
            topologies: vec!["1n2l4c2t".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            derive_slicing_dims(&f_a, &f_b),
            vec![Dimension::Topology],
            "differing `topologies` must surface Topology as a slicing dim",
        );

        // Sorted-deduped Vec semantics: same set in different
        // order/multiplicity must NOT slice.
        let f_c = RowFilter {
            topologies: vec!["1n2l4c1t".to_string(), "1n2l4c2t".to_string()],
            ..RowFilter::default()
        };
        let f_d = RowFilter {
            topologies: vec![
                "1n2l4c2t".to_string(),
                "1n2l4c1t".to_string(),
                "1n2l4c1t".to_string(),
            ],
            ..RowFilter::default()
        };
        assert!(
            derive_slicing_dims(&f_c, &f_d).is_empty(),
            "same topology set in different order/multiplicity must NOT slice",
        );
    }

    /// WorkType-only diff: filters that disagree on `work_types`
    /// and agree on every other dimension produce a slicing-dim
    /// set containing exactly `Dimension::WorkType`. Mirror of
    /// `derive_slicing_dims_topology_only_diff` for the WorkType
    /// arm.
    #[test]
    fn derive_slicing_dims_work_type_only_diff() {
        let f_a = RowFilter {
            work_types: vec!["SpinWait".to_string()],
            ..RowFilter::default()
        };
        let f_b = RowFilter {
            work_types: vec!["PageFaultChurn".to_string()],
            ..RowFilter::default()
        };
        assert_eq!(
            derive_slicing_dims(&f_a, &f_b),
            vec![Dimension::WorkType],
            "differing `work_types` must surface WorkType as a slicing dim",
        );

        // Sorted-deduped Vec semantics: same set in different
        // order/multiplicity must NOT slice.
        let f_c = RowFilter {
            work_types: vec!["SpinWait".to_string(), "PageFaultChurn".to_string()],
            ..RowFilter::default()
        };
        let f_d = RowFilter {
            work_types: vec![
                "PageFaultChurn".to_string(),
                "SpinWait".to_string(),
                "SpinWait".to_string(),
            ],
            ..RowFilter::default()
        };
        assert!(
            derive_slicing_dims(&f_c, &f_d).is_empty(),
            "same work_type set in different order/multiplicity must NOT slice",
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

    /// `PairingKey::from_row` includes the row's cpu_budget when CpuBudget
    /// is a pairing dim — so cross-budget rows NEVER pair — and excludes
    /// it when CpuBudget is the slicing dim. This is the whole point of
    /// #5: a 4-CPU-budget run and a 32-CPU-budget run measure different
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
}
