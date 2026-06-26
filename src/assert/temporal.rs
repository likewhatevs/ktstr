//! Temporal-assertion patterns over a periodic
//! [`SampleSeries`](crate::scenario::sample::SampleSeries).
//!
//! `SeriesField<T>` is a per-sample column extracted from the
//! series via `SampleSeries::bpf` or `SampleSeries::stats` (or
//! the typed `bpf_map` / `stats_path` projectors). It carries a
//! parallel `(tag, elapsed_ms, SnapshotResult<T>)` triple per
//! sample so any failure-path message can name the offending tag
//! and timestamp without re-walking the source data.
//!
//! The seven built-in patterns are:
//!   1. `nondecreasing` — counter monotonicity (`v[i] <= v[i+1]`).
//!   2. `strictly_increasing` — strict counter monotonicity
//!      (`v[i] < v[i+1]`).
//!   3. `rate_within(lo, hi)` — bounded delta-per-millisecond
//!      between consecutive samples.
//!   4. `steady_within(warmup, tolerance)` — post-warmup samples
//!      stay inside `[mean·(1-tol), mean·(1+tol)]`.
//!   5. `converges_to(target, tol, deadline_ms)` — three
//!      consecutive samples land inside `[target-tol, target+tol]`
//!      before `deadline_ms`.
//!   6. `always_true` — boolean invariant at every sample
//!      (`SeriesField<bool>` only).
//!   7. `ratio_within(other, lo, hi)` — cross-field correlation
//!      between two same-length series.
//!
//! Per-sample scalar checks bypass the temporal patterns via
//! [`SeriesField::each`], which yields an [`EachClaim`] supporting
//! `at_least` / `at_most` / `between`. All patterns route through
//! [`Verdict`] and tag failures with [`DetailKind::Temporal`].

use crate::scenario::snapshot::{SnapshotError, SnapshotResult};

use super::{AssertDetail, DetailKind, Outcome, Verdict};

/// Per-sample column extracted from a
/// [`SampleSeries`](crate::scenario::sample::SampleSeries). Each
/// slot is a [`SnapshotResult<T>`] so a missing or
/// type-mismatched field does NOT abort the whole projection — it
/// surfaces at the temporal-assertion site as a per-sample error
/// the caller decides how to handle.
///
/// The label, tags, and per-sample timestamps are carried so
/// failure-path messages name the offending sample without the
/// caller re-threading the series. Tags and elapsed-ms vectors
/// are always the same length as `values`.
/// Per-sample triple `(tag, elapsed_ms, &value)` yielded by
/// [`SeriesField::iter_full`] and stored in the per-phase buckets
/// returned by [`SeriesField::by_phase`].
pub type SampleTriple<'a, T> = (&'a str, Option<u64>, &'a SnapshotResult<T>);

/// Return shape of [`SeriesField::by_phase`]: a `BTreeMap` keyed by
/// `Phase` carrying the samples whose source row had a stamped
/// step_index, plus a separate `none_bucket` for unstamped /
/// fixture samples.
pub type ByPhasePartition<'a, T> = (
    std::collections::BTreeMap<crate::assert::Phase, Vec<SampleTriple<'a, T>>>,
    Vec<SampleTriple<'a, T>>,
);

/// Render the numeric part of an optional sample timestamp for the
/// `(+{n}ms)` failure-/skip-message convention: `Some(ms)` -> `"123"`,
/// `None` (the bridge recorded no timestamp for this sample) -> `"?"`,
/// so a not-measured sample reads as `+?ms` — visibly distinct from a
/// measured `+0ms`. Keeping it numeric lets the existing
/// `"(+{elapsed_ms}ms)"` format strings stay unchanged.
fn fmt_elapsed_num(elapsed_ms: Option<u64>) -> String {
    match elapsed_ms {
        Some(ms) => ms.to_string(),
        None => "?".to_string(),
    }
}

#[derive(Debug, Clone)]
#[must_use = "SeriesField records nothing until a temporal pattern is invoked"]
pub struct SeriesField<T> {
    label: String,
    tags: Vec<String>,
    elapsed_ms: Vec<Option<u64>>,
    values: Vec<SnapshotResult<T>>,
    /// Per-sample scenario phase, mirrored from the
    /// [`crate::scenario::sample::Sample::step_index`] each value
    /// was projected from. `None` for unstamped fixture samples
    /// (no bridge phase context); `Some(phase)` for production
    /// captures whose source row carried a step_index. Same length
    /// as `values` (or empty by the from_parts contract — the
    /// 4-arg constructor fills with all-`None` for backward-compat
    /// callers that didn't have the phase column yet).
    phases: Vec<Option<crate::assert::Phase>>,
}

impl<T> SeriesField<T> {
    /// Build a new field. Internal — projection helpers in
    /// [`crate::scenario::sample`] call this on the series side.
    /// 4-arg backward-compat constructor: defaults `phases` to all
    /// `None`. New consumers that have phase context per sample
    /// should call [`Self::from_parts_with_phases`] instead.
    pub fn from_parts(
        label: impl Into<String>,
        tags: Vec<String>,
        elapsed_ms: Vec<u64>,
        values: Vec<SnapshotResult<T>>,
    ) -> Self {
        let phases = vec![None; values.len()];
        Self::from_parts_with_phases(label, tags, elapsed_ms, values, phases)
    }

    /// Build a new field with explicit per-sample phase tags.
    /// `phases.len()` MUST equal `values.len()` — the four parallel
    /// vecs (tags / elapsed_ms / values / phases) are addressed by
    /// the same index throughout. Like [`Self::from_parts`], this
    /// is intended for projection helpers in
    /// [`crate::scenario::sample`] that already know each sample's
    /// step_index from the drained bridge tuple.
    /// 5-arg convenience taking MEASURED `Vec<u64>` timestamps; wraps
    /// each in `Some` and delegates to [`Self::from_parts_with_phases_opt`].
    /// Test fixtures (which always model measured samples) and any
    /// all-measured caller use this. The production projection funnel
    /// — which threads `Option<u64>` from the bridge so "not measured"
    /// stays distinct from "measured 0" — calls
    /// [`Self::from_parts_with_phases_opt`] directly.
    pub fn from_parts_with_phases(
        label: impl Into<String>,
        tags: Vec<String>,
        elapsed_ms: Vec<u64>,
        values: Vec<SnapshotResult<T>>,
        phases: Vec<Option<crate::assert::Phase>>,
    ) -> Self {
        Self::from_parts_with_phases_opt(
            label,
            tags,
            elapsed_ms.into_iter().map(Some).collect(),
            values,
            phases,
        )
    }

    /// None-aware constructor: `elapsed_ms[i] == None` means the bridge
    /// recorded no timestamp for that sample (not measured), kept
    /// distinct from a measured `Some(0)`. Temporal patterns that do
    /// timestamp math ([`EachClaim`]-free `rate_within` dt,
    /// `steady_within` warmup gate, `converges_to` deadline gate) SKIP
    /// a `None`-anchored sample rather than fabricating a `0` (the
    /// silent-wrong-answer this distinction prevents).
    pub fn from_parts_with_phases_opt(
        label: impl Into<String>,
        tags: Vec<String>,
        elapsed_ms: Vec<Option<u64>>,
        values: Vec<SnapshotResult<T>>,
        phases: Vec<Option<crate::assert::Phase>>,
    ) -> Self {
        // Hard runtime check (not debug_assert_eq!) so the equal-
        // length guarantee documented on iter_full() holds in
        // release builds. A length mismatch would otherwise surface
        // as either a silent truncation in iter_full() (zip stops
        // at the shortest input) or an out-of-bounds panic from
        // the direct `tags[i]` / `elapsed_ms[i]` field access in
        // EachClaim failure-message rendering — both harder to
        // diagnose than a panic at the construction site.
        assert_eq!(tags.len(), values.len());
        assert_eq!(elapsed_ms.len(), values.len());
        assert_eq!(phases.len(), values.len());
        Self {
            label: label.into(),
            tags,
            elapsed_ms,
            values,
            phases,
        }
    }

    /// Per-sample phase tag, parallel to `values`. `None` for
    /// fixture / unstamped samples; `Some(phase)` for production
    /// captures whose source row carried a `step_index`.
    pub fn phases_iter(&self) -> impl Iterator<Item = Option<crate::assert::Phase>> + '_ {
        self.phases.iter().copied()
    }

    /// Per-phase folded reduction: extract the `f64` value from
    /// each Ok-sample (skipping per-sample errors) and route the
    /// per-phase slice through `crate::stats::aggregate_samples_for_phase`
    /// so Counter kinds get the cumulative-delta semantic while
    /// other kinds inherit the flat-run aggregator. Returns one
    /// entry per phase that has at least one Ok-sample with a
    /// finite value; phases with all-err / all-NaN samples are
    /// absent from the map (consistent with
    /// `aggregate_samples_for_phase` returning `None` on that
    /// input). Skips `None`-phase samples — fixture / unstamped
    /// data does not have a phase key to bucket against.
    ///
    /// The API entry point a test author uses to ask "what was the
    /// per-phase reduction of metric X?" without having to thread
    /// `by_phase` + the kind-aware reducer manually.
    pub fn aggregate_by_phase(
        &self,
        metric: &crate::stats::MetricDef,
    ) -> std::collections::BTreeMap<crate::assert::Phase, f64>
    where
        T: Copy + Into<f64>,
    {
        let (by_phase, _none_bucket) = self.by_phase();
        let mut out: std::collections::BTreeMap<crate::assert::Phase, f64> =
            std::collections::BTreeMap::new();
        for (phase, samples) in by_phase {
            let finite_values: Vec<f64> = samples
                .iter()
                .filter_map(|(_tag, _elapsed, value)| match value {
                    Ok(v) => Some((*v).into()),
                    Err(_) => None,
                })
                .collect();
            if let Some(reduced) = crate::stats::aggregate_samples_for_phase(metric, &finite_values)
            {
                out.insert(phase, reduced);
            }
        }
        out
    }

    /// Per-phase SUM of the Ok-sample values — the per-phase total for a
    /// metric whose samples are already per-read DELTAS, e.g. a
    /// scheduler-defined scx_stats metric the scheduler deltas
    /// server-side per reader request (one ktstr snapshot = one request
    /// = one delta), read via `series.stats(name)`. Returns one entry
    /// per phase with at least one finite Ok-sample; all-err / all-NaN
    /// phases and `None`-phase (fixture / unstamped) samples are
    /// excluded, matching [`Self::aggregate_by_phase`].
    ///
    /// This is the `crate::stats::MetricKind::DeltaSum` per-phase
    /// reduction WITHOUT requiring a `crate::stats::MetricDef` — the
    /// ergonomic accessor for the common case "I read a delta-reported
    /// scx_stats metric; give me each phase's total." For a registered
    /// metric, `aggregate_by_phase(&def)` with `def.kind == DeltaSum`
    /// gives the identical result.
    ///
    /// Boundary: the first in-phase delta straddles the phase boundary
    /// (it spans from the last pre-phase read to the first in-phase
    /// read, so it carries a little pre-phase activity); it is
    /// attributed to the phase its read lands in — a slight left-edge
    /// over-attribution, the deliberate semantic since a per-read delta
    /// cannot be split.
    pub fn sum_by_phase(&self) -> std::collections::BTreeMap<crate::assert::Phase, f64>
    where
        T: Copy + Into<f64>,
    {
        let (by_phase, _none_bucket) = self.by_phase();
        let mut out: std::collections::BTreeMap<crate::assert::Phase, f64> =
            std::collections::BTreeMap::new();
        for (phase, samples) in by_phase {
            let values: Vec<f64> = samples
                .iter()
                .filter_map(|(_tag, _elapsed, value)| match value {
                    Ok(v) => Some((*v).into()),
                    Err(_) => None,
                })
                .collect();
            if let Some(reduced) =
                crate::stats::aggregate_samples(&values, crate::stats::MetricKind::DeltaSum)
            {
                out.insert(phase, reduced);
            }
        }
        out
    }

    /// Apply `f` once per phase, with the per-phase slice of
    /// [`SampleTriple<T>`]s. `None`-phase samples (fixture /
    /// unstamped) are skipped — callers wanting them call
    /// [`Self::by_phase`] directly and handle the second-tuple
    /// `none_bucket` themselves. Phases iterate in `Phase` order
    /// (BASELINE first, then Step\[0\], Step\[1\], ...) per the
    /// `BTreeMap` key ordering, which lets temporal-pattern
    /// consumers apply a per-phase reduction without restating
    /// the `by_phase` unpacking at every call site.
    ///
    /// ```ignore
    /// field.for_each_phase(|phase, samples| {
    ///     // apply pattern X to samples, scoped to `phase`
    /// });
    /// ```
    pub fn for_each_phase(&self, mut f: impl FnMut(crate::assert::Phase, &[SampleTriple<'_, T>])) {
        let (by_phase, _none_bucket) = self.by_phase();
        for (phase, samples) in by_phase {
            f(phase, &samples);
        }
    }

    /// Partition samples by phase. `None`-phase samples bucket
    /// into the returned `none_bucket` outside the BTreeMap; phase
    /// values bucket by their `Phase` key. Each bucket retains the
    /// per-sample [`SampleTriple<T>`] the standard [`Self::iter_full`]
    /// yields.
    pub fn by_phase(&self) -> ByPhasePartition<'_, T> {
        let mut buckets: std::collections::BTreeMap<
            crate::assert::Phase,
            Vec<SampleTriple<'_, T>>,
        > = std::collections::BTreeMap::new();
        let mut none_bucket: Vec<SampleTriple<'_, T>> = Vec::new();
        for (((tag, elapsed_ms), value), phase) in self
            .tags
            .iter()
            .zip(self.elapsed_ms.iter())
            .zip(self.values.iter())
            .zip(self.phases.iter())
        {
            let triple = (tag.as_str(), *elapsed_ms, value);
            match phase {
                Some(p) => buckets.entry(*p).or_default().push(triple),
                None => none_bucket.push(triple),
            }
        }
        (buckets, none_bucket)
    }

    /// Label for failure-message rendering.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Number of samples in the field.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when no samples are present.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterate over per-sample values (each a [`SnapshotResult<T>`]).
    pub fn values_iter(&self) -> impl Iterator<Item = &SnapshotResult<T>> {
        self.values.iter()
    }

    /// Iterate over the full per-sample triple — `(tag,
    /// elapsed_ms, &SnapshotResult<T>)`. Lets a caller consume the
    /// projected column alongside its sample identity without
    /// re-threading the source [`SampleSeries`](crate::scenario::sample::SampleSeries).
    /// Yields entries in the same order as the underlying
    /// `Vec<SnapshotResult<T>>` storage; tags and elapsed-ms
    /// vectors are guaranteed equal-length to `values` by
    /// [`Self::from_parts`]'s `assert_eq!` checks (which run in
    /// both debug and release builds).
    pub fn iter_full(&self) -> impl Iterator<Item = (&str, Option<u64>, &SnapshotResult<T>)> {
        self.tags
            .iter()
            .zip(self.elapsed_ms.iter())
            .zip(self.values.iter())
            .map(|((tag, elapsed_ms), value)| (tag.as_str(), *elapsed_ms, value))
    }

    /// Open a per-sample claim builder for scalar comparators
    /// (`at_least`, `at_most`, `between`). Each successful sample
    /// runs the comparator independently; the first failure
    /// records a detail; subsequent failures pile on so the
    /// timeline shows every offending sample, not just the first.
    /// Borrows the verdict mutably for the duration of the
    /// comparator chain.
    pub fn each<'v>(&self, verdict: &'v mut Verdict) -> EachClaim<'_, 'v, T> {
        EachClaim {
            field: self,
            verdict,
        }
    }

    /// Iterate the [`SampleTriple`]s for one specific phase. Sugar
    /// for [`Self::by_phase`]`().0.get(&phase).map(...)` that drops
    /// the tuple-destructure noise the user otherwise repeats at
    /// every per-phase site. Returns an empty iterator when the
    /// phase had no samples; callers that need to distinguish
    /// "empty bucket" from "phase never observed" can use
    /// [`Self::by_phase`] directly.
    pub fn phase(&self, phase: crate::assert::Phase) -> Vec<SampleTriple<'_, T>> {
        self.iter_full()
            .zip(self.phases.iter())
            .filter_map(|(triple, p)| {
                if *p == Some(phase) {
                    Some(triple)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Single-value reduction for a cumulative counter or any
    /// metric where "the last sample of the phase" is the
    /// load-bearing value: returns the last Ok-sample's value
    /// for `phase`, or `None` when the phase had zero Ok-samples.
    /// Per-sample errors are skipped so a one-off projection
    /// hiccup doesn't drop the whole phase. The "value at end of
    /// phase" semantic is what cross-phase counter comparisons
    /// (e.g. "dispatch count of Step\[1\] ≤ 0.85 × dispatch count
    /// of Step\[0\]") almost always want, so this primitive
    /// removes the closure-over-by_phase-triples boilerplate the
    /// user otherwise writes at every site.
    pub fn value_at_phase(&self, phase: crate::assert::Phase) -> Option<T>
    where
        T: Copy,
    {
        self.iter_full()
            .zip(self.phases.iter())
            .filter_map(|((_, _, value), p)| {
                if *p == Some(phase) {
                    value.as_ref().ok().copied()
                } else {
                    None
                }
            })
            .last()
    }

    /// Reduce to a per-phase last-Ok-value map. The companion to
    /// [`Self::value_at_phase`] for callers that want every phase
    /// at once: each present phase maps to its last successful
    /// sample's value. Phases with all-Err samples are absent from
    /// the map. Matches the "cumulative-counter-at-end-of-phase"
    /// semantic [`Self::aggregate_by_phase`] applies for Counter
    /// metrics, but skips the kind-aware fold so callers reaching
    /// for the raw last value don't pay the projection cost.
    pub fn last_per_phase(&self) -> std::collections::BTreeMap<crate::assert::Phase, T>
    where
        T: Copy,
    {
        let mut out: std::collections::BTreeMap<crate::assert::Phase, T> =
            std::collections::BTreeMap::new();
        for ((_, _, value), p) in self.iter_full().zip(self.phases.iter()) {
            if let (Some(phase), Ok(v)) = (*p, value) {
                out.insert(phase, *v);
            }
        }
        out
    }

    /// Reduce to a per-phase first-Ok-value map. The symmetric
    /// companion to [`Self::last_per_phase`]: each present phase
    /// maps to its first successful sample's value. Per-sample
    /// errors are skipped so a one-off projection hiccup at the
    /// start of a phase doesn't drop the whole phase. Phases with
    /// all-Err samples are absent from the map.
    ///
    /// The load-bearing pairing is
    /// `last_per_phase - first_per_phase` — the work done WITHIN
    /// a phase, with no contribution from prior phases — surfaced
    /// directly via [`Self::counter_delta_per_phase`].
    pub fn first_per_phase(&self) -> std::collections::BTreeMap<crate::assert::Phase, T>
    where
        T: Copy,
    {
        let mut out: std::collections::BTreeMap<crate::assert::Phase, T> =
            std::collections::BTreeMap::new();
        for ((_, _, value), p) in self.iter_full().zip(self.phases.iter()) {
            if let (Some(phase), Ok(v)) = (*p, value) {
                out.entry(phase).or_insert(*v);
            }
        }
        out
    }

    /// Per-phase cumulative-counter delta: `last_per_phase(p) -
    /// first_per_phase(p)` for every phase with at least one Ok
    /// sample. The reducer A/B-compare tests reach for when a
    /// metric is a cumulative counter still accruing from prior
    /// phases — `value_at_phase`'s last-sample reading carries
    /// the whole-run accumulation, which lets prior-phase activity
    /// muddy the cross-phase ratio. The delta isolates the work
    /// performed WITHIN each phase so
    /// `ratio(phase_delta(later) / phase_delta(earlier))` answers
    /// the question the operator actually asked: "did `later`'s
    /// activity drop relative to `earlier`'s?"
    ///
    /// Single-Ok-sample phases yield a delta of zero (first ==
    /// last). Phases with all-Err samples are absent. Phases that
    /// appear in `first_per_phase` but not `last_per_phase` (the
    /// out-of-order edge case) are also absent — the delta is
    /// well-defined only when both endpoints are present.
    ///
    /// The reducer is intentionally generic: the test author owns
    /// `(same_delta, cross_delta)`-style compositions (e.g. fold
    /// two counter-delta maps into a per-phase fraction) without
    /// the framework needing a registered `MetricDef`. For
    /// Counter-typed registered metrics, the equivalent
    /// MetricDef-aware path is [`Self::aggregate_by_phase`].
    ///
    /// **Counter regression**: when a phase's last sample reads
    /// LOWER than its first, the reducer emits a `tracing::warn!`
    /// naming the field label + phase + (first, last) and stores
    /// `T::default()` for that phase (e.g. `0` for `u64`). The
    /// underlying counter is assumed non-decreasing within a phase
    /// (the common case for BPF per-event counters). A regression
    /// can mean either an upstream-signal bug (the counter source
    /// itself rolled back) OR a framework picker drift mid-phase
    /// (e.g. [`crate::scenario::sample::SampleSeries::bpf_live_u64`]
    /// resolved to different bss copies across same-phase
    /// snapshots in a post-`Op::ReplaceScheduler` swap window).
    /// Either way the regression is reported as zero progress
    /// rather than panicking, so a single bad sample does not
    /// crash the assertion engine.
    pub fn counter_delta_per_phase(&self) -> std::collections::BTreeMap<crate::assert::Phase, T>
    where
        T: Copy + PartialOrd + std::ops::Sub<Output = T> + Default + std::fmt::Debug,
    {
        let firsts = self.first_per_phase();
        let lasts = self.last_per_phase();
        let label = self.label();
        firsts
            .into_iter()
            .filter_map(|(phase, first)| {
                lasts.get(&phase).map(|last| {
                    if *last >= first {
                        (phase, *last - first)
                    } else {
                        tracing::warn!(
                            label = %label,
                            ?phase,
                            ?first,
                            last = ?*last,
                            "counter_delta_per_phase: phase counter regressed \
                             (last < first); reporting zero progress for this phase",
                        );
                        (phase, T::default())
                    }
                })
            })
            .collect()
    }

    /// Cross-phase comparator: pin `value_at_phase(later) /
    /// value_at_phase(earlier)` against a ceiling AND record the
    /// computed ratio + both phase values in the verdict so
    /// `--nocapture` runs surface the actual numbers without a
    /// per-test `println!` boilerplate. Records a failure when
    /// either phase had no Ok-samples, when the earlier value is
    /// zero (no baseline), or when the ratio exceeds `ceiling`.
    /// On success records an informational note carrying the
    /// observed ratio + both values so the operator can see the
    /// margin against the threshold.
    ///
    /// Returns `&mut Verdict` for chaining.
    pub fn ratio_across_phases<'v>(
        &self,
        verdict: &'v mut Verdict,
        earlier: crate::assert::Phase,
        later: crate::assert::Phase,
    ) -> CrossPhaseRatio<'v, T>
    where
        T: Copy + Into<f64> + std::fmt::Display,
    {
        let e = self.value_at_phase(earlier);
        let l = self.value_at_phase(later);
        CrossPhaseRatio {
            label: self.label().to_string(),
            verdict,
            earlier,
            later,
            earlier_value: e,
            later_value: l,
        }
    }
}

/// Cross-phase ratio builder returned by
/// [`SeriesField::ratio_across_phases`] and
/// [`PhaseMapExt::ratio_across_phases`]. Carries the resolved
/// `(earlier, later)` values and a caller-supplied label so the
/// terminal comparator chain (`at_most`) can format both values
/// and the ratio into a single failure-or-note message. Mirrors
/// the [`EachClaim`] shape (mutable verdict borrow held through
/// the chain).
///
/// The `label` is origin-neutral: SeriesField's entry point fills
/// it from the field's `.label()`, the PhaseMap entry point takes
/// it from the caller. An empty label suppresses the leading
/// `label:` / `[label]` prefix in the rendered message so the
/// rest of the diagnostic stays readable.
#[must_use = "CrossPhaseRatio records nothing until at_most is invoked"]
pub struct CrossPhaseRatio<'v, T> {
    label: String,
    verdict: &'v mut Verdict,
    earlier: crate::assert::Phase,
    later: crate::assert::Phase,
    earlier_value: Option<T>,
    later_value: Option<T>,
}

impl<'v, T> CrossPhaseRatio<'v, T>
where
    T: Copy + Into<f64> + std::fmt::Display,
{
    /// Pass when `later_value / earlier_value <= ceiling`. On
    /// failure records a [`DetailKind::Temporal`] detail naming
    /// the field label, both phase values, the computed ratio,
    /// and the ceiling so the failure message is self-contained.
    /// On success records an info note with the same trio so a
    /// `--nocapture` run surfaces the headroom without a separate
    /// per-metric `println!`.
    pub fn at_most(self, ceiling: f64) -> &'v mut Verdict {
        let label_prefix = if self.label.is_empty() {
            String::new()
        } else {
            format!("{}: ", self.label)
        };
        let note_prefix = if self.label.is_empty() {
            String::new()
        } else {
            format!("[{}] ", self.label)
        };
        let earlier_str = match self.earlier_value {
            Some(v) => format!("{v}"),
            None => "<no-samples>".to_string(),
        };
        let later_str = match self.later_value {
            Some(v) => format!("{v}"),
            None => "<no-samples>".to_string(),
        };
        let (Some(earlier), Some(later)) = (self.earlier_value, self.later_value) else {
            // INSTRUMENT-derived: one or both phases produced no
            // samples, so there is no value to ratio against. The
            // ratio cannot be computed; record Inconclusive rather
            // than Fail to distinguish "no signal to evaluate" from
            // "evaluated and exceeded ceiling."
            push_inconclusive(
                self.verdict,
                format!(
                    "{label_prefix}ratio_across_phases({:?}→{:?}) inconclusive: \
                     needs both phases — earlier={earlier_str}, later={later_str}",
                    self.earlier, self.later,
                ),
            );
            return self.verdict;
        };
        let earlier_f: f64 = earlier.into();
        let later_f: f64 = later.into();
        if earlier_f == 0.0 {
            // INSTRUMENT-derived: earlier baseline measured 0, so
            // later/earlier is undefined. Record Inconclusive — the
            // ceiling check has no signal to evaluate, neither pass
            // (would silently green a phase pair with no baseline)
            // nor fail (no actual ratio violation observed) is
            // truthful. POLICY-derived zero baselines (a policy
            // decision to compare against an intentional 0) are
            // out of scope for this gate.
            push_inconclusive(
                self.verdict,
                format!(
                    "{label_prefix}ratio_across_phases({:?}→{:?}) inconclusive: \
                     earlier value is 0 (no baseline to ratio against)",
                    self.earlier, self.later,
                ),
            );
            return self.verdict;
        }
        let ratio = later_f / earlier_f;
        if !ratio.is_finite() {
            // A non-finite ratio: the `earlier_f == 0.0` guard above
            // misses a NaN baseline (NaN != 0.0), and a NaN later_f
            // or an inf-producing quotient also lands here. Raw
            // `ratio > ceiling` is always false for NaN, so without
            // this guard the phase pair would silently PASS. Treat a
            // corrupt endpoint as a Fail, mirroring rate_within /
            // ratio_within's non-finite handling (distinct from the
            // zero-baseline Inconclusive above, which is "no signal").
            push_detail(
                self.verdict,
                format!(
                    "{label_prefix}ratio_across_phases({:?}→{:?}) = \
                     {later_str}/{earlier_str} = {ratio} is non-finite \
                     (corrupt endpoint) — cannot evaluate ceiling {ceiling:.4}",
                    self.earlier, self.later,
                ),
            );
        } else if ratio > ceiling {
            push_detail(
                self.verdict,
                format!(
                    "{label_prefix}ratio_across_phases({:?}→{:?}) = \
                     {later_str}/{earlier_str} = {ratio:.4} exceeds ceiling \
                     {ceiling:.4}",
                    self.earlier, self.later,
                ),
            );
        } else {
            // Pass — emit a note that surfaces in the sidecar
            // info_notes (visible under --nocapture and on the
            // failure render of any sibling claim) so the operator
            // sees the headroom against the ceiling without a
            // separate per-metric println.
            self.verdict.note(format!(
                "{note_prefix}ratio_across_phases({:?}→{:?}) = \
                 {later_str}/{earlier_str} = {ratio:.4} (ceiling {ceiling:.4})",
                self.earlier, self.later,
            ));
        }
        self.verdict
    }
}

/// The polarity-resolved outcome of a [`BetterThanPhase`] comparison, factored
/// out as a PURE decision so it is exhaustively unit-testable without a
/// `VmResult` or `Verdict`. The builder maps each variant to a verdict record +
/// message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BetterOutcome {
    /// Candidate is better than baseline (strictly, or by the required margin).
    Pass,
    /// Candidate is NOT better — worse, or short of the margin.
    Fail,
    /// A non-finite endpoint — a corrupt value the comparison can't trust
    /// (Fail, not Inconclusive: a `<` on NaN is silently false, so an unguarded
    /// corrupt value would falsely pass).
    Corrupt,
    /// One or both phases carried no value for the metric (no signal).
    Missing,
    /// The metric has no LowerBetter/HigherBetter polarity (TargetValue /
    /// Unknown / unregistered), so "better" has no direction.
    Undirected,
    /// A fractional margin was requested against a zero baseline — nothing to
    /// scale the margin against.
    ZeroBaseline,
}

/// Pure better-than decision: is `candidate` better than `baseline` for a metric
/// of the given `polarity`, by the optional `margin` (a FRACTION of the
/// baseline; `None` = strictly better, any improvement)? The whole point is the
/// SAME call works for a LowerBetter metric (latency) and a HigherBetter one
/// (throughput) with no caller-specified direction — the direction comes from
/// the registry-declared polarity.
fn better_outcome(
    baseline: Option<f64>,
    candidate: Option<f64>,
    polarity: Option<crate::test_support::Polarity>,
    margin: Option<f64>,
) -> BetterOutcome {
    use crate::test_support::Polarity;
    let (Some(b), Some(c)) = (baseline, candidate) else {
        return BetterOutcome::Missing;
    };
    if !b.is_finite() || !c.is_finite() {
        return BetterOutcome::Corrupt;
    }
    let lower_better = match polarity {
        Some(Polarity::LowerBetter) => true,
        Some(Polarity::HigherBetter) => false,
        // TargetValue / Unknown / unregistered: no "better" direction.
        _ => return BetterOutcome::Undirected,
    };
    let pass = match margin {
        // Strictly better (any improvement).
        None => {
            if lower_better {
                c < b
            } else {
                c > b
            }
        }
        // Better by at least `m` (a fraction of the baseline). A zero baseline
        // has nothing to scale the fractional margin against.
        Some(m) => {
            if b == 0.0 {
                return BetterOutcome::ZeroBaseline;
            }
            // Relative improvement as a fraction of the baseline, the DIVISION
            // form `(improvement)/baseline >= m` rather than the
            // algebraically-equivalent multiplicative `c <= b*(1-m)`: the
            // `b*(1-m)` intermediate (e.g. `100.0*(1.0-0.1)` = 89.999…) rejects
            // an EXACTLY-`m`-better candidate by an f64 epsilon, whereas
            // `(b-c)/b` is exact at round-number boundaries (`10.0/100.0` is the
            // same f64 bit pattern as the `0.1` literal). The
            // percentile-f64-threshold-rounding footgun.
            if lower_better {
                (b - c) / b >= m
            } else {
                (c - b) / b >= m
            }
        }
    };
    if pass {
        BetterOutcome::Pass
    } else {
        BetterOutcome::Fail
    }
}

/// Cross-phase "is the candidate phase better than the baseline phase on this
/// metric?" comparator, returned by
/// [`crate::vmm::VmResult::better_across_phases`]. The polarity-aware sibling of
/// [`CrossPhaseRatio`]: it reads two PER-PHASE scalars (via `phase_metric`, not
/// a sampled series) and orients "better" from the metric's registry-declared
/// polarity, so the same call expresses "scheduler beats EEVDF" for a
/// LowerBetter latency AND a HigherBetter throughput without the test author
/// naming a direction. A missing/undirected/zero-baseline comparison is
/// Inconclusive (never a silent pass); a non-finite value is a Fail.
#[must_use = "BetterThanPhase records nothing until better_than / by_at_least is invoked"]
pub struct BetterThanPhase<'v> {
    metric: String,
    verdict: &'v mut Verdict,
    baseline: crate::assert::Phase,
    candidate: crate::assert::Phase,
    baseline_value: Option<f64>,
    candidate_value: Option<f64>,
    polarity: Option<crate::test_support::Polarity>,
}

impl<'v> BetterThanPhase<'v> {
    /// Build from already-resolved per-phase values + polarity. Only
    /// [`crate::vmm::VmResult::better_across_phases`] constructs this — it
    /// resolves the values via `phase_metric` and the polarity via
    /// `crate::stats::metric_def`.
    pub(crate) fn new(
        metric: String,
        verdict: &'v mut Verdict,
        baseline: crate::assert::Phase,
        candidate: crate::assert::Phase,
        baseline_value: Option<f64>,
        candidate_value: Option<f64>,
        polarity: Option<crate::test_support::Polarity>,
    ) -> Self {
        Self {
            metric,
            verdict,
            baseline,
            candidate,
            baseline_value,
            candidate_value,
            polarity,
        }
    }

    /// Pass iff the candidate is STRICTLY better than the baseline per the
    /// metric's polarity (LowerBetter: candidate < baseline; HigherBetter:
    /// candidate > baseline) — any improvement, no margin.
    pub fn better_than(self) -> &'v mut Verdict {
        self.evaluate(None)
    }

    /// Pass iff the candidate improves on the baseline by at least `margin`, a
    /// FRACTION of the baseline (`0.10` = 10% better): LowerBetter requires
    /// `candidate <= baseline * (1 - margin)`, HigherBetter requires
    /// `candidate >= baseline * (1 + margin)`. `by_at_least(0.0)` means "no
    /// regression" (at least as good). A zero baseline is Inconclusive (nothing
    /// to scale the fractional margin against).
    pub fn by_at_least(self, margin: f64) -> &'v mut Verdict {
        self.evaluate(Some(margin))
    }

    fn evaluate(self, margin: Option<f64>) -> &'v mut Verdict {
        let outcome = better_outcome(
            self.baseline_value,
            self.candidate_value,
            self.polarity,
            margin,
        );
        let dir = match self.polarity {
            Some(crate::test_support::Polarity::LowerBetter) => "lower-is-better",
            Some(crate::test_support::Polarity::HigherBetter) => "higher-is-better",
            _ => "no-better-direction",
        };
        let b_str = self
            .baseline_value
            .map(|v| format!("{v}"))
            .unwrap_or_else(|| "<no-value>".to_string());
        let c_str = self
            .candidate_value
            .map(|v| format!("{v}"))
            .unwrap_or_else(|| "<no-value>".to_string());
        let req = match margin {
            None => "strictly better".to_string(),
            Some(m) => format!("better by >= {m:.4} fraction"),
        };
        let metric = &self.metric;
        let base = self.baseline;
        let cand = self.candidate;
        match outcome {
            BetterOutcome::Pass => {
                self.verdict.note(format!(
                    "[{metric}] candidate {cand}={c_str} {req} than baseline {base}={b_str} ({dir})"
                ));
            }
            BetterOutcome::Fail => {
                push_detail(
                    self.verdict,
                    format!(
                        "{metric}: candidate {cand}={c_str} is NOT {req} than baseline \
                         {base}={b_str} ({dir})"
                    ),
                );
            }
            BetterOutcome::Corrupt => {
                push_detail(
                    self.verdict,
                    format!(
                        "{metric}: non-finite value (baseline {base}={b_str}, candidate \
                         {cand}={c_str}) — cannot compare"
                    ),
                );
            }
            BetterOutcome::Missing => {
                push_inconclusive(
                    self.verdict,
                    format!(
                        "{metric}: better_across_phases({base}->{cand}) inconclusive: \
                         needs both phases — baseline={b_str}, candidate={c_str}"
                    ),
                );
            }
            BetterOutcome::Undirected => {
                push_inconclusive(
                    self.verdict,
                    format!(
                        "{metric}: better_across_phases inconclusive: metric has no \
                         lower/higher-is-better polarity — cannot orient 'better'"
                    ),
                );
            }
            BetterOutcome::ZeroBaseline => {
                push_inconclusive(
                    self.verdict,
                    format!(
                        "{metric}: better_across_phases inconclusive: baseline {base}=0, no \
                         baseline to scale the fractional margin ({req})"
                    ),
                );
            }
        }
        self.verdict
    }
}

/// Extension trait that lets a pre-reduced per-phase map
/// (typically the output of [`SeriesField::counter_delta_per_phase`],
/// [`SeriesField::last_per_phase`], or
/// [`SeriesField::first_per_phase`]) compose with the
/// cross-phase comparator chain [`SeriesField::ratio_across_phases`]
/// exposes — without re-projecting the per-phase values back
/// through a synthetic [`SeriesField`].
///
/// Also surfaces [`Self::zip_per_phase`] so two per-phase maps fold
/// element-wise into a derived per-phase map (e.g. a cross-LLC
/// dispatch fraction from two counter-delta maps).
pub trait PhaseMapExt<T> {
    /// Fold two per-phase maps element-wise on phase intersection.
    /// For every phase present in BOTH `self` AND `other`, invoke
    /// `f(self_value, other_value)` and collect the result keyed
    /// by phase. Phases present in only one input are absent from
    /// the result.
    ///
    /// **Intersection-only — NOT [`Iterator::zip`] semantics.** This
    /// pairs values by phase key, not by position; a missing phase
    /// on either side surfaces as an absence in the result, never
    /// as a synthesized zero or default. Callers that want to act
    /// on coverage gaps compare the result map's length against
    /// either input's length.
    ///
    /// Both values are passed BY VALUE — the trait constrains
    /// `T: Copy` and `U: Copy` to keep the closure body free of
    /// `*s` / `*c` deref noise that would otherwise dominate every
    /// composed-metric expression. Non-Copy element types are out
    /// of scope; per-phase reducers in this crate already return
    /// scalar `Copy` values (`u64`, `f64`, `i64`).
    fn zip_per_phase<U, R>(
        &self,
        other: &std::collections::BTreeMap<crate::assert::Phase, U>,
        f: impl FnMut(T, U) -> R,
    ) -> std::collections::BTreeMap<crate::assert::Phase, R>
    where
        T: Copy,
        U: Copy;

    /// Cross-phase ratio comparator on a pre-reduced per-phase
    /// map. Mirrors [`SeriesField::ratio_across_phases`]'s
    /// chain shape — `.at_most(ceiling)` records a failure detail
    /// or pass info note via the supplied verdict — but operates
    /// on the map directly so caller-derived per-phase values
    /// (e.g. a fraction of two counter deltas) skip a synthetic-
    /// SeriesField intermediate.
    ///
    /// Three load-bearing differences from the SeriesField entry:
    ///
    /// 1. **No implicit label.** SeriesField pulls its `.label()`
    ///    for the failure message; the map has no label, so the
    ///    caller names the metric being compared at the call site.
    /// 2. **Pre-reduced values.** SeriesField reduces by
    ///    last-Ok-sample at each comparator call; this trait
    ///    operates on values already reduced by any compatible
    ///    upstream reducer ([`SeriesField::counter_delta_per_phase`],
    ///    [`SeriesField::last_per_phase`], or a caller-defined fold).
    /// 3. **`T: Copy`** — the map's per-phase value is copied out
    ///    into the [`CrossPhaseRatio`] carrier's `Option<T>`
    ///    fields. Matches [`SeriesField::value_at_phase`]'s bound
    ///    for the same reason.
    fn ratio_across_phases<'v>(
        &self,
        verdict: &'v mut Verdict,
        label: impl Into<String>,
        earlier: crate::assert::Phase,
        later: crate::assert::Phase,
    ) -> CrossPhaseRatio<'v, T>
    where
        T: Copy + Into<f64> + std::fmt::Display;
}

/// Per-phase "share of total" reducer. Specialized for the dominant
/// counter shape (`BTreeMap<Phase, u64>`) because `u64: Into<f64>`
/// is intentionally absent (cast is lossy) — a generic
/// [`PhaseMapExt`] method with an `Into<f64>` bound would reject
/// every counter-delta map test authors actually reach for.
///
/// For every phase present in BOTH `self` AND `other`, computes
/// `self_value as f64 / (self_value + other_value) as f64`. When
/// both values are zero (sum is zero), the phase is **dropped from
/// the result** — there is no signal to share, and synthesizing
/// `0.0` would let downstream `at_most` / `ratio_within` gates
/// silently pass on a zero-event phase pair. Returning no entry
/// surfaces the absence so the consumer can treat it as
/// Inconclusive (the same shape as a phase only present in one
/// input). Phases present in only one input also drop from the
/// result, mirroring [`PhaseMapExt::zip_per_phase`]'s
/// intersection-only semantics; both drop conditions surface
/// identically as "no entry for this phase."
///
/// Targets the "cross-LLC dispatch fraction" idiom (`nr_cross /
/// (nr_cross + nr_same)`) and similar share-of-total patterns.
/// The general fold via [`PhaseMapExt::zip_per_phase`] requires
/// the caller to spell the safe-divide branch inline at every
/// call site; this trait owns the branch so test code expresses
/// the metric in one chain.
pub trait FracPair {
    /// See trait-level doc for the zero-total drop and
    /// intersection-only semantics.
    fn frac_pair(&self, other: &Self) -> std::collections::BTreeMap<crate::assert::Phase, f64>;
}

impl FracPair for std::collections::BTreeMap<crate::assert::Phase, u64> {
    fn frac_pair(&self, other: &Self) -> std::collections::BTreeMap<crate::assert::Phase, f64> {
        self.iter()
            .filter_map(|(p, n)| {
                other.get(p).and_then(|m| {
                    // `saturating_add` guards against u64 overflow on
                    // long-running counter pairs (the realistic
                    // failure is two near-u64::MAX counter deltas;
                    // wrap would produce a misleading fraction).
                    // Saturation to u64::MAX still yields a sensible
                    // fraction `n / u64::MAX` ≈ 0.0 for non-MAX `n`
                    // and 1.0 for the saturating case, with no NaN.
                    let total = n.saturating_add(*m);
                    if total == 0 {
                        // Zero-total phase: no events observed in
                        // either input. Drop the entry rather than
                        // synthesize 0.0 so a downstream `at_most`
                        // sees absence (= Inconclusive shape) instead
                        // of a silent pass against any positive
                        // threshold.
                        None
                    } else {
                        Some((*p, *n as f64 / total as f64))
                    }
                })
            })
            .collect()
    }
}

impl<T> PhaseMapExt<T> for std::collections::BTreeMap<crate::assert::Phase, T> {
    fn zip_per_phase<U, R>(
        &self,
        other: &std::collections::BTreeMap<crate::assert::Phase, U>,
        mut f: impl FnMut(T, U) -> R,
    ) -> std::collections::BTreeMap<crate::assert::Phase, R>
    where
        T: Copy,
        U: Copy,
    {
        self.iter()
            .filter_map(|(p, t)| other.get(p).map(|u| (*p, f(*t, *u))))
            .collect()
    }

    fn ratio_across_phases<'v>(
        &self,
        verdict: &'v mut Verdict,
        label: impl Into<String>,
        earlier: crate::assert::Phase,
        later: crate::assert::Phase,
    ) -> CrossPhaseRatio<'v, T>
    where
        T: Copy + Into<f64> + std::fmt::Display,
    {
        CrossPhaseRatio {
            label: label.into(),
            verdict,
            earlier,
            later,
            earlier_value: self.get(&earlier).copied(),
            later_value: self.get(&later).copied(),
        }
    }
}

/// Per-sample scalar claim builder returned by
/// [`SeriesField::each`]. Provides `at_least` / `at_most` /
/// `between` — comparators that apply to every (successfully
/// projected) sample independently. Per-sample errors from the
/// projection (missing field, type mismatch) are routed through
/// the verdict as failures so coverage gaps are never silent.
#[must_use = "EachClaim records nothing until a comparator is invoked"]
pub struct EachClaim<'f, 'v, T> {
    field: &'f SeriesField<T>,
    verdict: &'v mut Verdict,
}

impl<'f, 'v, T> EachClaim<'f, 'v, T>
where
    T: PartialOrd + std::fmt::Display + Copy,
{
    /// Pass when every sample's value satisfies `value >= floor`.
    /// Per-sample errors and per-sample violations both record a
    /// [`DetailKind::Temporal`] detail and flip the verdict to
    /// failed; the chain returns the verdict so further claims
    /// can stack.
    ///
    /// On `T = f64`, an incomparable value (NaN) is a failure: a
    /// NaN sample silently passing `value < floor`/`value > ceiling`
    /// (which IEEE-754 semantics give you on raw `<`/`>`) would
    /// hide a coverage gap, so the pattern uses `partial_cmp` and
    /// reports the offending sample distinctly.
    pub fn at_least(self, floor: T) -> &'v mut Verdict {
        let pre_outcomes = temporal_outcome_count(self.verdict);
        let label = self.field.label.as_str();
        let n = self.field.values.len();
        for (i, slot) in self.field.values.iter().enumerate() {
            match slot {
                Ok(v) => match v.partial_cmp(&floor) {
                    Some(std::cmp::Ordering::Less) => push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.at_least {floor}): sample {tag} (+{elapsed_ms}ms): \
                             value {v}",
                            tag = self.field.tags[i],
                            elapsed_ms = fmt_elapsed_num(self.field.elapsed_ms[i]),
                        ),
                    ),
                    None => push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.at_least {floor}): sample {tag} (+{elapsed_ms}ms): \
                             value {v} is incomparable (NaN)",
                            tag = self.field.tags[i],
                            elapsed_ms = fmt_elapsed_num(self.field.elapsed_ms[i]),
                        ),
                    ),
                    Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Greater) => {}
                },
                Err(e) => {
                    push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.at_least {floor}): sample {tag} (+{elapsed_ms}ms): \
                             projection error: {e}",
                            tag = self.field.tags[i],
                            elapsed_ms = fmt_elapsed_num(self.field.elapsed_ms[i]),
                        ),
                    );
                }
            }
        }
        maybe_log_pass_temporal(self.verdict, pre_outcomes, || {
            format!("{label} (each.at_least {floor}): all {n} samples passed")
        });
        self.verdict
    }

    /// Pass when every sample's value satisfies `value <= ceiling`.
    /// NaN samples (on `T = f64`) report an incomparable failure
    /// for the same reason documented on [`Self::at_least`].
    pub fn at_most(self, ceiling: T) -> &'v mut Verdict {
        let pre_outcomes = temporal_outcome_count(self.verdict);
        let label = self.field.label.as_str();
        let n = self.field.values.len();
        for (i, slot) in self.field.values.iter().enumerate() {
            match slot {
                Ok(v) => match v.partial_cmp(&ceiling) {
                    Some(std::cmp::Ordering::Greater) => push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.at_most {ceiling}): sample {tag} (+{elapsed_ms}ms): \
                             value {v}",
                            tag = self.field.tags[i],
                            elapsed_ms = fmt_elapsed_num(self.field.elapsed_ms[i]),
                        ),
                    ),
                    None => push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.at_most {ceiling}): sample {tag} (+{elapsed_ms}ms): \
                             value {v} is incomparable (NaN)",
                            tag = self.field.tags[i],
                            elapsed_ms = fmt_elapsed_num(self.field.elapsed_ms[i]),
                        ),
                    ),
                    Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Less) => {}
                },
                Err(e) => {
                    push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.at_most {ceiling}): sample {tag} (+{elapsed_ms}ms): \
                             projection error: {e}",
                            tag = self.field.tags[i],
                            elapsed_ms = fmt_elapsed_num(self.field.elapsed_ms[i]),
                        ),
                    );
                }
            }
        }
        maybe_log_pass_temporal(self.verdict, pre_outcomes, || {
            format!("{label} (each.at_most {ceiling}): all {n} samples passed")
        });
        self.verdict
    }

    /// Pass when every sample's value satisfies `lo <= value <= hi`.
    /// Caller error (`lo > hi`) lands as a single
    /// [`DetailKind::Temporal`] detail rather than evaluating each
    /// sample against an inverted range. NaN samples report an
    /// incomparable failure (see [`Self::at_least`]).
    pub fn between(self, lo: T, hi: T) -> &'v mut Verdict {
        let label = self.field.label.as_str();
        if lo > hi {
            push_detail(
                self.verdict,
                format!("{label} (each.between): caller error: lo={lo} > hi={hi}"),
            );
            return self.verdict;
        }
        let pre_outcomes = temporal_outcome_count(self.verdict);
        let n = self.field.values.len();
        for (i, slot) in self.field.values.iter().enumerate() {
            match slot {
                Ok(v) => {
                    let lo_cmp = v.partial_cmp(&lo);
                    let hi_cmp = v.partial_cmp(&hi);
                    if lo_cmp.is_none() || hi_cmp.is_none() {
                        push_detail(
                            self.verdict,
                            format!(
                                "{label} (each.between [{lo}, {hi}]): sample {tag} \
                                 (+{elapsed_ms}ms): value {v} is incomparable (NaN)",
                                tag = self.field.tags[i],
                                elapsed_ms = fmt_elapsed_num(self.field.elapsed_ms[i]),
                            ),
                        );
                    } else if matches!(lo_cmp, Some(std::cmp::Ordering::Less))
                        || matches!(hi_cmp, Some(std::cmp::Ordering::Greater))
                    {
                        push_detail(
                            self.verdict,
                            format!(
                                "{label} (each.between [{lo}, {hi}]): sample {tag} \
                                 (+{elapsed_ms}ms): value {v}",
                                tag = self.field.tags[i],
                                elapsed_ms = fmt_elapsed_num(self.field.elapsed_ms[i]),
                            ),
                        );
                    }
                }
                Err(e) => {
                    push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.between [{lo}, {hi}]): sample {tag} \
                             (+{elapsed_ms}ms): projection error: {e}",
                            tag = self.field.tags[i],
                            elapsed_ms = fmt_elapsed_num(self.field.elapsed_ms[i]),
                        ),
                    );
                }
            }
        }
        maybe_log_pass_temporal(self.verdict, pre_outcomes, || {
            format!("{label} (each.between [{lo}, {hi}]): all {n} samples passed")
        });
        self.verdict
    }
}

// ----- Seven temporal patterns -----

impl<T> SeriesField<T>
where
    T: PartialOrd + std::fmt::Display + Copy,
{
    /// Pass when every consecutive pair satisfies
    /// `values[i] <= values[i+1]`. A common shape for kernel
    /// counters whose only legal direction is up. Per-sample
    /// projection errors are SKIPPED — the affected pair-comparison
    /// is dropped, the skip count is logged as a verdict Note so
    /// coverage gaps stay visible, and the verdict is NOT flipped
    /// on a missing-sample condition (which is structurally
    /// missing data, not a regression). Adjacent samples on
    /// either side of a gap are still checked against each other.
    pub fn nondecreasing<'v>(&self, verdict: &'v mut Verdict) -> &'v mut Verdict {
        self.monotonicity_check(verdict, false)
    }

    /// Pass when every consecutive pair satisfies
    /// `values[i] < values[i+1]`. Useful for counters that MUST
    /// advance every period (e.g. a heartbeat tick). Same skip-on-
    /// projection-error semantics as [`Self::nondecreasing`].
    pub fn strictly_increasing<'v>(&self, verdict: &'v mut Verdict) -> &'v mut Verdict {
        self.monotonicity_check(verdict, true)
    }

    fn monotonicity_check<'v>(&self, verdict: &'v mut Verdict, strict: bool) -> &'v mut Verdict {
        let pat = if strict {
            "strictly_increasing"
        } else {
            "nondecreasing"
        };
        if self.values.len() < 2 {
            // Vacuous: 0 or 1 samples cannot violate monotonicity.
            // Surface an informational note via the verdict's
            // notes path so the timeline summary records that the
            // pattern was opened against an under-populated
            // series — without this, a bug that drops every
            // periodic capture would silently pass every
            // monotonicity claim.
            verdict.note(format!(
                "{label} ({pat}): only {n} samples — pattern vacuously holds; \
                 ensure num_snapshots >= 2 for meaningful coverage",
                label = self.label,
                n = self.values.len(),
            ));
            return verdict;
        }
        let pre_outcomes = temporal_outcome_count(verdict);
        // Per-sample projection errors are NOT temporal failures —
        // they indicate the underlying field was missing on that
        // sample (e.g. placeholder report from a freeze-rendezvous
        // timeout). Skip the affected pair-comparisons and surface
        // the skip count as a Note on the verdict so a coverage
        // gap is visible without flipping a temporal pattern that
        // is structurally about value monotonicity. The compare
        // proceeds across the rest of the series without bridging
        // the gap (a gap means we cannot conclude anything about
        // monotonicity ACROSS the missing sample, only on either
        // side of it).
        let mut skipped: Vec<String> = Vec::new();
        for i in 0..self.values.len() - 1 {
            let left = match &self.values[i] {
                Ok(v) => v,
                Err(e) => {
                    // Surface the underlying SnapshotError variant
                    // (PlaceholderSample, MissingStats, FieldNotFound,
                    // VarNotFound, TypeMismatch, …) in the Note so
                    // the operator can distinguish "freeze rendezvous
                    // timed out" from "field name typo" from
                    // "stats relay had no scheduler" without
                    // re-running the test under a debugger. The
                    // Display impl on SnapshotError gives the
                    // human-readable variant text plus context
                    // (available keys, requested path).
                    skipped.push(format!(
                        "{tag}(+{elapsed_ms}ms): {e}",
                        tag = self.tags[i],
                        elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                    ));
                    continue;
                }
            };
            let right = match &self.values[i + 1] {
                Ok(v) => v,
                Err(_) => {
                    // Skip recorded when the (i+1) slot becomes
                    // the `i` slot of the next iteration; avoid
                    // double-counting by only logging on the
                    // forward-edge here.
                    continue;
                }
            };
            let ok = if strict { right > left } else { right >= left };
            if !ok {
                push_detail(
                    verdict,
                    format!(
                        "{label} ({pat}): regression at sample {tag} (+{elapsed_ms}ms): \
                         value {right} after prior value {left} at sample {prev_tag} \
                         (+{prev_elapsed}ms)",
                        label = self.label,
                        tag = self.tags[i + 1],
                        elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i + 1]),
                        prev_tag = self.tags[i],
                        prev_elapsed = fmt_elapsed_num(self.elapsed_ms[i]),
                    ),
                );
            }
        }
        // The final sample's err state was not visited by the
        // loop's left-arm; check it explicitly so the skip count
        // is exhaustive. Carry the same `: {e}` rendering used
        // above so the trailing skip entry exposes the error
        // variant just like every other entry.
        if let Some(last) = self.values.last()
            && let Err(e) = last
        {
            let i = self.values.len() - 1;
            skipped.push(format!(
                "{tag}(+{elapsed_ms}ms): {e}",
                tag = self.tags[i],
                elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
            ));
        }
        if !skipped.is_empty() {
            verdict.note(format!(
                "{label} ({pat}): skipped {n} sample(s) with projection errors: \
                 {samples}",
                label = self.label,
                n = skipped.len(),
                samples = skipped.join(", "),
            ));
        }
        maybe_log_pass_temporal(verdict, pre_outcomes, || {
            format!(
                "{label} ({pat}): all {n} samples passed",
                label = self.label,
                n = self.values.len(),
            )
        });
        verdict
    }
}

impl SeriesField<f64> {
    /// Pass when every consecutive (delta_value / delta_ms) lies
    /// in `[lo, hi]`. The rate is computed with millisecond
    /// resolution from the per-sample elapsed-ms timestamps, so
    /// a counter that should advance at ~1 unit/ms reads cleanly
    /// as `rate_within(0.5, 2.0)`. A zero-time delta between
    /// adjacent samples lands as a caller-side or framework
    /// failure (samples too close to compute a rate); the detail
    /// names the offending pair.
    pub fn rate_within<'v>(&self, verdict: &'v mut Verdict, lo: f64, hi: f64) -> &'v mut Verdict {
        if lo > hi {
            push_detail(
                verdict,
                format!(
                    "{label} (rate_within): caller error: lo={lo} > hi={hi}",
                    label = self.label,
                ),
            );
            return verdict;
        }
        if self.values.len() < 2 {
            verdict.note(format!(
                "{label} (rate_within): only {n} samples — pattern vacuously holds",
                label = self.label,
                n = self.values.len(),
            ));
            return verdict;
        }
        let pre_outcomes = temporal_outcome_count(verdict);
        // Per-sample projection errors are treated as GAPS — no
        // rate is computed across the gap. Log every gap with the
        // underlying error variant via a Note so a coverage
        // problem is visible (with WHICH error) without flipping
        // the verdict on what is structurally a missing-data
        // condition, not a rate violation. When BOTH endpoints of
        // a pair errored, both errors are surfaced so the operator
        // can tell whether the projection has a per-sample
        // coverage hole on one side or a systemic problem on
        // both.
        let mut gaps: Vec<String> = Vec::new();
        for i in 0..self.values.len() - 1 {
            let (left, right) = match (&self.values[i], &self.values[i + 1]) {
                (Ok(l), Ok(r)) => (*l, *r),
                (lhs_slot, rhs_slot) => {
                    let mut endpoints: Vec<String> = Vec::with_capacity(2);
                    if let Err(e) = lhs_slot {
                        endpoints.push(format!(
                            "{tag}(+{elapsed_ms}ms): {e}",
                            tag = self.tags[i],
                            elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                        ));
                    }
                    if let Err(e) = rhs_slot {
                        endpoints.push(format!(
                            "{tag}(+{elapsed_ms}ms): {e}",
                            tag = self.tags[i + 1],
                            elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i + 1]),
                        ));
                    }
                    gaps.push(endpoints.join(" | "));
                    continue;
                }
            };
            // A None elapsed endpoint = the bridge recorded no timestamp
            // for that sample: the interval's duration is
            // UNDEFINED, so the rate over it cannot be computed. Skip the
            // pair (record it in `gaps`) rather than fabricating a dt from
            // a `0` — this runs BEFORE the dt<=0 guard, which applies only
            // once both endpoints are measured.
            let (Some(prev_ms), Some(next_ms)) = (self.elapsed_ms[i], self.elapsed_ms[i + 1])
            else {
                gaps.push(format!(
                    "{prev_tag}(+{prev_elapsed}ms)..{tag}(+{elapsed_ms}ms): elapsed not measured",
                    prev_tag = self.tags[i],
                    prev_elapsed = fmt_elapsed_num(self.elapsed_ms[i]),
                    tag = self.tags[i + 1],
                    elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i + 1]),
                ));
                continue;
            };
            let dt_ms = next_ms.saturating_sub(prev_ms) as f64;
            if dt_ms <= 0.0 {
                push_inconclusive(
                    verdict,
                    format!(
                        "{label} (rate_within): zero-time delta between sample {prev_tag} \
                         (+{prev_elapsed}ms) and {tag} (+{elapsed_ms}ms) — denominator is \
                         INSTRUMENT-derived; rate is neither pass nor fail",
                        label = self.label,
                        prev_tag = self.tags[i],
                        prev_elapsed = fmt_elapsed_num(self.elapsed_ms[i]),
                        tag = self.tags[i + 1],
                        elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i + 1]),
                    ),
                );
                continue;
            }
            let rate = (right - left) / dt_ms;
            // NaN can arise from inf-inf or NaN endpoints; raw `<`/`>`
            // against NaN is always false, so a NaN rate would
            // silently slip past the band check. Infinite rates
            // (inf endpoint, or finite endpoints whose difference
            // overflows f64) are also an upstream data corruption
            // signal — caller has no use for the band comparison
            // when the value is non-finite. Both cases get a
            // structured detail naming the sample pair so the
            // operator sees the offending span.
            if !rate.is_finite() {
                push_detail(
                    verdict,
                    format!(
                        "{label} (rate_within [{lo}, {hi}]): non-finite rate between \
                         samples {prev_tag} (+{prev_elapsed}ms, value {left}) and \
                         {tag} (+{elapsed_ms}ms, value {right}) — endpoint is NaN \
                         or produced inf in the delta",
                        label = self.label,
                        prev_tag = self.tags[i],
                        prev_elapsed = fmt_elapsed_num(self.elapsed_ms[i]),
                        tag = self.tags[i + 1],
                        elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i + 1]),
                    ),
                );
            } else if rate < lo || rate > hi {
                push_detail(
                    verdict,
                    format!(
                        "{label} (rate_within [{lo}, {hi}]): rate {rate:.4}/ms between \
                         samples {prev_tag} (+{prev_elapsed}ms, value {left}) and \
                         {tag} (+{elapsed_ms}ms, value {right})",
                        label = self.label,
                        prev_tag = self.tags[i],
                        prev_elapsed = fmt_elapsed_num(self.elapsed_ms[i]),
                        tag = self.tags[i + 1],
                        elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i + 1]),
                    ),
                );
            }
        }
        if !gaps.is_empty() {
            verdict.note(format!(
                "{label} (rate_within): {n} consecutive-pair gap(s) skipped \
                 due to a projection error or an unmeasured elapsed timestamp \
                 on at least one endpoint: {samples}",
                label = self.label,
                n = gaps.len(),
                samples = gaps.join(", "),
            ));
        }
        maybe_log_pass_temporal(verdict, pre_outcomes, || {
            format!(
                "{label} (rate_within [{lo}, {hi}]): all {n} consecutive-pair rates within band",
                label = self.label,
                n = self.values.len().saturating_sub(1),
            )
        });
        verdict
    }

    /// Pass when every post-warmup sample (`elapsed_ms >=
    /// warmup_ms`) lies inside `mean·(1-tolerance), mean·(1+tolerance)`.
    /// `tolerance` is a fraction (0.10 = ±10%). The mean is
    /// computed over the post-warmup samples only — the warmup
    /// region is excluded so ramp-up does not bias the steady-
    /// state baseline. Per-sample projection errors are SKIPPED
    /// (with a verdict Note logging the count and tags); they are
    /// treated as gaps in coverage, not band violations, so a
    /// missing post-warmup sample does not flip the verdict.
    pub fn steady_within<'v>(
        &self,
        verdict: &'v mut Verdict,
        warmup_ms: u64,
        tolerance: f64,
    ) -> &'v mut Verdict {
        if tolerance < 0.0 {
            push_detail(
                verdict,
                format!(
                    "{label} (steady_within): caller error: tolerance {tolerance} negative",
                    label = self.label,
                ),
            );
            return verdict;
        }
        let pre_outcomes = temporal_outcome_count(verdict);
        let mut active: Vec<(usize, f64)> = Vec::new();
        let mut skipped: Vec<String> = Vec::new();
        // Track whether any sample's elapsed_ms reached or exceeded
        // warmup_ms — distinguishes "warmup window absorbed every
        // sample" (genuine vacuous-pass) from "post-warmup samples
        // existed but all errored" (skip-Note already covers it).
        let mut any_post_warmup = false;
        for (i, slot) in self.values.iter().enumerate() {
            // A None timestamp cannot be placed relative to
            // the warmup window: skip with a Note rather than treating it
            // as 0 (< warmup, silently dropped) or admitting an
            // untimestamped value into the post-warmup band.
            let Some(ms) = self.elapsed_ms[i] else {
                skipped.push(format!(
                    "{tag}(+?ms): elapsed not measured (cannot place vs warmup)",
                    tag = self.tags[i],
                ));
                continue;
            };
            if ms < warmup_ms {
                continue;
            }
            any_post_warmup = true;
            match slot {
                // A non-finite value (NaN/inf) cannot be band-checked
                // — `v < lo` is always false for NaN — and a single
                // NaN poisons the mean (1320), making `lo`/`hi` NaN so
                // EVERY sample slips past the band and the assertion
                // silently PASSES. Treat a non-finite value as a gap,
                // like a projection error: drop it from the band
                // population (so it can neither poison the mean nor
                // slip the band) and surface it in the skip Note.
                Ok(v) if v.is_finite() => active.push((i, *v)),
                Ok(v) => skipped.push(format!(
                    "{tag}(+{elapsed_ms}ms): non-finite value {v}",
                    tag = self.tags[i],
                    elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                )),
                // Per-sample projection errors are treated as
                // gaps: a missing post-warmup sample cannot
                // violate the steady-state band (we have no value
                // to compare). Surface the skip count and the
                // underlying SnapshotError variant for each
                // skipped sample via a Note so the operator can
                // tell PlaceholderSample / MissingStats /
                // FieldNotFound / TypeMismatch apart instead of
                // collapsing every gap into "projection error" —
                // a coverage hole is visible WITH the failure
                // reason without flipping the verdict on what is
                // structurally missing data, not a band violation.
                Err(e) => skipped.push(format!(
                    "{tag}(+{elapsed_ms}ms): {e}",
                    tag = self.tags[i],
                    elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                )),
            }
        }
        if !skipped.is_empty() {
            verdict.note(format!(
                "{label} (steady_within): skipped {n} sample(s) with a projection \
                 error or an unmeasured elapsed timestamp: {samples}",
                label = self.label,
                n = skipped.len(),
                samples = skipped.join(", "),
            ));
        }
        if active.is_empty() {
            // Only emit the vacuous-warmup Note when the warmup
            // window genuinely absorbed every sample (no
            // post-warmup samples existed). When post-warmup
            // samples existed but all errored, the
            // skipped-with-projection-errors Note above already
            // explained the empty active set; emitting a second
            // Note here would falsely claim "no samples beyond
            // warmup".
            if !any_post_warmup {
                verdict.note(format!(
                    "{label} (steady_within): no samples beyond warmup_ms={warmup_ms} — \
                     pattern vacuously holds",
                    label = self.label,
                ));
            }
            return verdict;
        }
        let mean: f64 = active.iter().map(|(_, v)| *v).sum::<f64>() / (active.len() as f64);
        let lo = mean * (1.0 - tolerance);
        let hi = mean * (1.0 + tolerance);
        // For negative means (pathological), the multiplication
        // flips the band; protect by sorting.
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        let active_count = active.len();
        for (i, v) in active {
            if v < lo || v > hi {
                push_detail(
                    verdict,
                    format!(
                        "{label} (steady_within mean {mean:.4} ±{pct:.1}%): \
                         sample {tag} (+{elapsed_ms}ms): value {v} outside [{lo:.4}, {hi:.4}]",
                        label = self.label,
                        pct = tolerance * 100.0,
                        tag = self.tags[i],
                        elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                    ),
                );
            }
        }
        maybe_log_pass_temporal(verdict, pre_outcomes, || {
            format!(
                "{label} (steady_within mean {mean:.4} ±{pct:.1}%): all {n} post-warmup samples in band",
                label = self.label,
                pct = tolerance * 100.0,
                n = active_count,
            )
        });
        verdict
    }

    /// Pass when three consecutive samples land inside
    /// `[target-tolerance, target+tolerance]` AT OR BEFORE
    /// `deadline_ms`. The intent is "the system stabilizes near
    /// `target` by the deadline" — three consecutive in-band
    /// samples are the convergence-witness shape. Failures fire
    /// when the deadline passes without a witness.
    pub fn converges_to<'v>(
        &self,
        verdict: &'v mut Verdict,
        target: f64,
        tolerance: f64,
        deadline_ms: u64,
    ) -> &'v mut Verdict {
        if tolerance < 0.0 {
            push_detail(
                verdict,
                format!(
                    "{label} (converges_to): caller error: tolerance {tolerance} negative",
                    label = self.label,
                ),
            );
            return verdict;
        }
        let pre_outcomes = temporal_outcome_count(verdict);
        // Pre-check: counting all successfully-projected samples
        // (within the deadline window) do we have enough evidence
        // to even attempt a 3-consecutive witness? When fewer
        // than 3 successfully-projected samples exist before the
        // deadline, record an explicit Note (NOT a verdict
        // failure) and return — absence of data is a coverage gap
        // surfaced for the operator, not a negative finding the
        // verdict should fail on. Distinguishes "did not collect
        // enough samples" (Note here) from "collected enough
        // samples but never converged" (the no-witness Temporal
        // detail emitted below by the witness loop). The Note
        // names every errored in-window sample with its underlying
        // SnapshotError variant so the operator can tell
        // PlaceholderSample / MissingStats / FieldNotFound apart
        // when diagnosing a coverage hole — a count alone hides
        // which kind of failure produced the gap.
        let mut projected_count: usize = 0;
        let mut error_samples: Vec<String> = Vec::new();
        for (i, slot) in self.values.iter().enumerate() {
            // A None timestamp cannot be placed before/after
            // the deadline: skip it from the projected-sample count
            // rather than counting it as 0 <= deadline, which would
            // falsely admit an untimestamped sample into the window.
            if self.elapsed_ms[i].is_none_or(|ms| ms > deadline_ms) {
                continue;
            }
            match slot {
                Ok(_) => projected_count += 1,
                Err(e) => error_samples.push(format!(
                    "{tag}(+{elapsed_ms}ms): {e}",
                    tag = self.tags[i],
                    elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                )),
            }
        }
        if projected_count < 3 {
            let suffix = if error_samples.is_empty() {
                String::new()
            } else {
                format!("; errored sample(s): {}", error_samples.join(", "))
            };
            verdict.note(format!(
                "{label} (converges_to {target} ±{tolerance}, deadline_ms={deadline_ms}): \
                 insufficient samples for converges_to (need ≥3, have {projected_count}){suffix}",
                label = self.label,
            ));
            return verdict;
        }
        let lo = target - tolerance;
        let hi = target + tolerance;
        let mut consecutive: usize = 0;
        let mut witness_idx: Option<usize> = None;
        // Errored in-window samples that interrupted the
        // 3-consecutive witness search. Recorded here even when
        // the projected_count >= 3 pre-check passed so a verdict
        // failure ("no witness") still names the error variants
        // that broke each attempted run — the operator can see
        // whether the missing-witness was caused by genuine
        // out-of-band values or by a coverage hole resetting the
        // consecutive counter mid-run.
        let mut interrupting_errors: Vec<String> = Vec::new();
        for (i, slot) in self.values.iter().enumerate() {
            // A None timestamp cannot be placed before/after
            // the deadline: treat it as out-of-window (skip / reset the
            // witness run) rather than as 0 <= deadline, which would
            // falsely admit an untimestamped sample into the window.
            if self.elapsed_ms[i].is_none_or(|ms| ms > deadline_ms) {
                consecutive = 0;
                continue;
            }
            match slot {
                Ok(v) => {
                    if *v >= lo && *v <= hi {
                        consecutive += 1;
                        if consecutive >= 3 {
                            witness_idx = Some(i);
                            break;
                        }
                    } else {
                        consecutive = 0;
                    }
                }
                Err(e) => {
                    if consecutive > 0 {
                        // Only record the error when it actually
                        // interrupted an in-progress run — a
                        // string of out-of-band errors before any
                        // in-band samples is irrelevant to
                        // witness coverage.
                        interrupting_errors.push(format!(
                            "{tag}(+{elapsed_ms}ms): {e}",
                            tag = self.tags[i],
                            elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                        ));
                    }
                    consecutive = 0;
                }
            }
        }
        if witness_idx.is_none() {
            let suffix = if interrupting_errors.is_empty() {
                String::new()
            } else {
                format!(
                    "; in-progress runs interrupted by errored sample(s): {}",
                    interrupting_errors.join(", ")
                )
            };
            push_detail(
                verdict,
                format!(
                    "{label} (converges_to {target} ±{tolerance}, deadline_ms={deadline_ms}): \
                     no 3-consecutive-in-band witness before deadline ({n} samples evaluated){suffix}",
                    label = self.label,
                    n = self.values.len(),
                ),
            );
        }
        maybe_log_pass_temporal(verdict, pre_outcomes, || {
            let where_at = witness_idx
                .map(|i| {
                    format!(
                        "{tag} (+{elapsed_ms}ms)",
                        tag = self.tags[i],
                        elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                    )
                })
                .unwrap_or_else(|| "<unreached>".to_string());
            format!(
                "{label} (converges_to {target} ±{tolerance}, deadline_ms={deadline_ms}): \
                 3-consecutive-in-band witness reached at {where_at}",
                label = self.label,
            )
        });
        verdict
    }

    /// Pass when every consecutive `(self_value / other_value)`
    /// lies in `[lo, hi]`. Cross-field correlation: e.g. ensure a
    /// per-cgroup utilization always tracks a per-cgroup runtime
    /// within a fixed band. The two series MUST have matching
    /// length and tags; mismatches fire a single caller-error
    /// detail. Per-sample projection errors on EITHER lhs or rhs
    /// are SKIPPED — the affected pair is dropped, the skip count
    /// is logged as a verdict Note, and the verdict is NOT flipped
    /// on missing-data conditions.
    pub fn ratio_within<'v>(
        &self,
        verdict: &'v mut Verdict,
        other: &SeriesField<f64>,
        lo: f64,
        hi: f64,
    ) -> &'v mut Verdict {
        if lo > hi {
            push_detail(
                verdict,
                format!(
                    "{label} (ratio_within): caller error: lo={lo} > hi={hi}",
                    label = self.label,
                ),
            );
            return verdict;
        }
        if self.values.len() != other.values.len() {
            push_detail(
                verdict,
                format!(
                    "{label} (ratio_within {other}): caller error: length mismatch \
                     (this {n}, other {m})",
                    label = self.label,
                    other = other.label,
                    n = self.values.len(),
                    m = other.values.len(),
                ),
            );
            return verdict;
        }
        let pre_outcomes = temporal_outcome_count(verdict);
        // Per-sample projection errors on either lhs or rhs are
        // treated as gaps — no ratio is computed across the pair.
        // Surface every gap with the underlying error variant
        // (and which side errored: lhs / rhs / both) via a Note
        // so a coverage hole is visible WITH the failure reason
        // without flipping the verdict on what is structurally
        // missing data. The Display impl on SnapshotError gives
        // the variant text plus context (FieldNotFound's
        // available keys, TypeMismatch's expected/actual,
        // PlaceholderSample's reason) so the operator can tell
        // failure modes apart instead of collapsing every gap
        // into "projection error on one side".
        let mut gaps: Vec<String> = Vec::new();
        for (i, (lhs_slot, rhs_slot)) in self.values.iter().zip(other.values.iter()).enumerate() {
            let (lhs, rhs) = match (lhs_slot, rhs_slot) {
                (Ok(l), Ok(r)) => (*l, *r),
                _ => {
                    // Each side carries its own tag + elapsed_ms —
                    // the two SeriesFields can be projected from
                    // different rows of the same SampleSeries with
                    // distinct tags at index `i`, so a single outer
                    // tag would mislabel the RHS endpoint. Fold the
                    // per-side identity into each entry instead.
                    let mut endpoints: Vec<String> = Vec::with_capacity(2);
                    if let Err(e) = lhs_slot {
                        endpoints.push(format!(
                            "lhs {tag}(+{elapsed_ms}ms): {e}",
                            tag = self.tags[i],
                            elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                        ));
                    }
                    if let Err(e) = rhs_slot {
                        endpoints.push(format!(
                            "rhs {tag}(+{elapsed_ms}ms): {e}",
                            tag = other.tags[i],
                            elapsed_ms = fmt_elapsed_num(other.elapsed_ms[i]),
                        ));
                    }
                    gaps.push(endpoints.join(" | "));
                    continue;
                }
            };
            if rhs == 0.0 {
                push_inconclusive(
                    verdict,
                    format!(
                        "{label} (ratio_within): rhs == 0 at sample {tag} (+{elapsed_ms}ms) — \
                         denominator is INSTRUMENT-derived; ratio is neither pass nor fail",
                        label = self.label,
                        tag = self.tags[i],
                        elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                    ),
                );
                continue;
            }
            let ratio = lhs / rhs;
            // A NaN lhs/rhs (or finite endpoints whose quotient
            // overflows to inf) yields a non-finite ratio; the
            // `rhs == 0.0` guard above misses it (NaN != 0.0). Raw
            // `<`/`>` against NaN is always false, so a non-finite
            // ratio would silently slip past the band check and PASS.
            // Surface it as a detail naming the pair, mirroring
            // rate_within's non-finite-rate guard.
            if !ratio.is_finite() {
                push_detail(
                    verdict,
                    format!(
                        "{label} (ratio_within {other_label} [{lo}, {hi}]): non-finite \
                         ratio at sample {tag} (+{elapsed_ms}ms) — lhs={lhs} rhs={rhs}",
                        label = self.label,
                        other_label = other.label,
                        tag = self.tags[i],
                        elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                    ),
                );
            } else if ratio < lo || ratio > hi {
                push_detail(
                    verdict,
                    format!(
                        "{label} (ratio_within {other_label} [{lo}, {hi}]): \
                         ratio {ratio:.4} at sample {tag} (+{elapsed_ms}ms) — \
                         lhs={lhs} rhs={rhs}",
                        label = self.label,
                        other_label = other.label,
                        tag = self.tags[i],
                        elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                    ),
                );
            }
        }
        if !gaps.is_empty() {
            verdict.note(format!(
                "{label} (ratio_within): {n} pair(s) skipped due to projection \
                 errors on lhs or rhs: {samples}",
                label = self.label,
                n = gaps.len(),
                samples = gaps.join(", "),
            ));
        }
        maybe_log_pass_temporal(verdict, pre_outcomes, || {
            format!(
                "{label} (ratio_within {other} [{lo}, {hi}]): all {n} pair ratios in band",
                label = self.label,
                other = other.label,
                n = self.values.len(),
            )
        });
        verdict
    }
}

impl SeriesField<bool> {
    /// Pass when every sample's value is `true`. Per-sample
    /// projection errors fail the assertion. Use for boolean
    /// invariants — e.g. "scheduler is alive at every periodic
    /// boundary" projected as `snap.var("scheduler_alive").as_bool()`.
    pub fn always_true<'v>(&self, verdict: &'v mut Verdict) -> &'v mut Verdict {
        let pre_outcomes = temporal_outcome_count(verdict);
        for (i, slot) in self.values.iter().enumerate() {
            match slot {
                Ok(v) => {
                    if !*v {
                        push_detail(
                            verdict,
                            format!(
                                "{label} (always_true): sample {tag} (+{elapsed_ms}ms): \
                                 value false",
                                label = self.label,
                                tag = self.tags[i],
                                elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                            ),
                        );
                    }
                }
                Err(e) => {
                    push_detail(
                        verdict,
                        format!(
                            "{label} (always_true): sample {tag} (+{elapsed_ms}ms): \
                             projection error: {e}",
                            label = self.label,
                            tag = self.tags[i],
                            elapsed_ms = fmt_elapsed_num(self.elapsed_ms[i]),
                        ),
                    );
                }
            }
        }
        maybe_log_pass_temporal(verdict, pre_outcomes, || {
            format!(
                "{label} (always_true): all {n} samples true",
                label = self.label,
                n = self.values.len(),
            )
        });
        verdict
    }
}

fn push_detail(verdict: &mut Verdict, message: String) {
    verdict
        .result_mut()
        .record_fail(AssertDetail::new(DetailKind::Temporal, message));
}

/// Inconclusive-arm sibling of [`push_detail`]. Records one
/// `Outcome::Inconclusive` with a [`DetailKind::Temporal`] detail.
/// Use for INSTRUMENT-derived zero-denominator paths — a
/// zero-time-delta between two consecutive samples in
/// [`SeriesField::rate_within`] or a zero rhs in
/// [`SeriesField::ratio_within`] cannot be computed, so the
/// verdict is neither pass nor fail (see [`Outcome`] doc's
/// INSTRUMENT vs POLICY carve-out).
fn push_inconclusive(verdict: &mut Verdict, message: String) {
    verdict
        .result_mut()
        .record_inconclusive(AssertDetail::new(DetailKind::Temporal, message));
}

/// Count `DetailKind::Temporal` Fail + Inconclusive outcomes in
/// `verdict`'s underlying result. Used by
/// [`maybe_log_pass_temporal`] to gate the positive-confirmation
/// log on "this pattern added zero Temporal Fail or Inconclusive
/// outcomes." Inconclusives count because a pattern that emitted
/// only Inconclusives is not in a state where logging "passed"
/// would be truthful. Vacuous-pattern and projection-error skip
/// notes live on `AssertResult::info_notes` (a structurally-separate
/// field from outcomes) and are therefore naturally excluded from
/// this count, so a pattern that emits notes but no Fail or
/// Inconclusive outcomes still trips the positive log.
fn temporal_outcome_count(verdict: &Verdict) -> usize {
    verdict
        .result()
        .outcomes
        .iter()
        .filter(|o| {
            matches!(
                o,
                Outcome::Fail(d) | Outcome::Inconclusive(d) if matches!(d.kind, DetailKind::Temporal)
            )
        })
        .count()
}

/// Positive-confirmation mirror of [`push_detail`] /
/// [`push_inconclusive`]. Emits a `tracing::info!` event naming
/// the temporal pattern and its sample count IFF
/// [`Verdict::log_passes`] is on AND the calling pattern added no
/// `DetailKind::Temporal` Fail or Inconclusive outcomes over its
/// run (compared via `pre_outcomes` captured at pattern entry via
/// [`temporal_outcome_count`]).
///
/// The pre/post gate is what makes this a positive confirmation —
/// a pattern that emitted a [`push_detail`] or [`push_inconclusive`]
/// mid-run stays silent here so a partial failure or inconclusive
/// does not log a misleading "passed" event. The closure
/// constructs the message only when both gates pass, so the
/// `format!` cost is paid only on the explicit opt-in + a clean
/// pattern run.
fn maybe_log_pass_temporal<F: FnOnce() -> String>(
    verdict: &Verdict,
    pre_outcomes: usize,
    message: F,
) {
    if verdict.log_passes() && temporal_outcome_count(verdict) == pre_outcomes {
        let m = message();
        tracing::info!(target: "ktstr::assert::temporal", "{m}");
    }
}

// Bridge into Verdict's internal AssertResult — added below as an
// associated method on Verdict so the temporal module does not
// reach into internals from a sibling.

#[allow(dead_code)]
fn _silence_snapshot_error_import(_: SnapshotError) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::sample::SampleSeries;
    use crate::scenario::snapshot::{SnapshotError, SnapshotResult};
    use crate::test_support::Polarity;

    // -- BetterThanPhase polarity decision (the pure better_outcome core) --

    #[test]
    fn better_outcome_lower_is_better_strict() {
        let p = Some(Polarity::LowerBetter);
        assert_eq!(
            better_outcome(Some(100.0), Some(50.0), p, None),
            BetterOutcome::Pass
        );
        assert_eq!(
            better_outcome(Some(50.0), Some(100.0), p, None),
            BetterOutcome::Fail
        );
        assert_eq!(
            better_outcome(Some(50.0), Some(50.0), p, None),
            BetterOutcome::Fail,
            "equal is not STRICTLY better"
        );
    }

    #[test]
    fn better_outcome_higher_is_better_strict() {
        let p = Some(Polarity::HigherBetter);
        assert_eq!(
            better_outcome(Some(50.0), Some(100.0), p, None),
            BetterOutcome::Pass
        );
        assert_eq!(
            better_outcome(Some(100.0), Some(50.0), p, None),
            BetterOutcome::Fail
        );
        assert_eq!(
            better_outcome(Some(50.0), Some(50.0), p, None),
            BetterOutcome::Fail
        );
    }

    #[test]
    fn better_outcome_margin_is_a_baseline_fraction() {
        let lb = Some(Polarity::LowerBetter);
        let hb = Some(Polarity::HigherBetter);
        // LOWER: 10% better = candidate <= 90.
        assert_eq!(
            better_outcome(Some(100.0), Some(90.0), lb, Some(0.1)),
            BetterOutcome::Pass
        );
        assert_eq!(
            better_outcome(Some(100.0), Some(91.0), lb, Some(0.1)),
            BetterOutcome::Fail,
            "9% short of the required 10%"
        );
        // HIGHER: 10% better = candidate >= 110.
        assert_eq!(
            better_outcome(Some(100.0), Some(110.0), hb, Some(0.1)),
            BetterOutcome::Pass
        );
        assert_eq!(
            better_outcome(Some(100.0), Some(109.0), hb, Some(0.1)),
            BetterOutcome::Fail
        );
        // margin 0.0 = "no regression": equal passes, worse fails.
        assert_eq!(
            better_outcome(Some(100.0), Some(100.0), lb, Some(0.0)),
            BetterOutcome::Pass
        );
        assert_eq!(
            better_outcome(Some(100.0), Some(101.0), lb, Some(0.0)),
            BetterOutcome::Fail
        );
    }

    #[test]
    fn better_outcome_inconclusive_variants() {
        let lb = Some(Polarity::LowerBetter);
        // Missing: either value None (no signal in a phase).
        assert_eq!(
            better_outcome(None, Some(1.0), lb, None),
            BetterOutcome::Missing
        );
        assert_eq!(
            better_outcome(Some(1.0), None, lb, None),
            BetterOutcome::Missing
        );
        // Undirected: TargetValue / Unknown / unregistered (None) polarity.
        assert_eq!(
            better_outcome(Some(1.0), Some(2.0), Some(Polarity::Unknown), None),
            BetterOutcome::Undirected
        );
        assert_eq!(
            better_outcome(Some(1.0), Some(2.0), Some(Polarity::TargetValue(5.0)), None),
            BetterOutcome::Undirected
        );
        assert_eq!(
            better_outcome(Some(1.0), Some(2.0), None, None),
            BetterOutcome::Undirected
        );
        // ZeroBaseline: a fractional margin against a 0 baseline.
        assert_eq!(
            better_outcome(Some(0.0), Some(1.0), lb, Some(0.1)),
            BetterOutcome::ZeroBaseline
        );
        // ...but a STRICT (None-margin) compare against a 0 baseline is a plain
        // compare, not ZeroBaseline (1 < 0 is just false → Fail).
        assert_eq!(
            better_outcome(Some(0.0), Some(1.0), lb, None),
            BetterOutcome::Fail
        );
    }

    #[test]
    fn better_outcome_corrupt_nonfinite_is_fail_not_silent_pass() {
        let lb = Some(Polarity::LowerBetter);
        assert_eq!(
            better_outcome(Some(f64::NAN), Some(1.0), lb, None),
            BetterOutcome::Corrupt
        );
        assert_eq!(
            better_outcome(Some(1.0), Some(f64::INFINITY), lb, None),
            BetterOutcome::Corrupt
        );
        // Precedence: non-finite is checked before polarity, so a NaN with an
        // undirected polarity is Corrupt (not Undirected) — a `<` on NaN is
        // silently false, so this must NOT collapse to a silent pass.
        assert_eq!(
            better_outcome(Some(f64::NAN), Some(1.0), None, None),
            BetterOutcome::Corrupt
        );
        // Missing (None) still wins over Corrupt — no value at all short-circuits.
        assert_eq!(
            better_outcome(None, Some(f64::NAN), lb, None),
            BetterOutcome::Missing
        );
    }

    fn synthetic_field<T: Copy>(label: &'static str, values: Vec<(u64, T)>) -> SeriesField<T> {
        let tags: Vec<String> = (0..values.len())
            .map(|i| format!("periodic_{i:03}"))
            .collect();
        let elapsed: Vec<u64> = values.iter().map(|(t, _)| *t).collect();
        let vals: Vec<SnapshotResult<T>> = values.into_iter().map(|(_, v)| Ok(v)).collect();
        SeriesField::from_parts(label, tags, elapsed, vals)
    }

    #[test]
    fn nondecreasing_passes_on_monotonic_series() {
        let f = synthetic_field("counter", vec![(100, 1u64), (200, 2u64), (300, 3u64)]);
        let mut v = Verdict::new();
        f.nondecreasing(&mut v);
        assert!(v.is_pass());
    }

    #[test]
    fn nondecreasing_fails_on_regression() {
        let f = synthetic_field("counter", vec![(100, 5u64), (200, 3u64)]);
        let mut v = Verdict::new();
        f.nondecreasing(&mut v);
        let r = v.into_result();
        assert!(r.is_fail());
        assert!(r.failure_details().any(|d| d.kind == DetailKind::Temporal));
        assert!(r.failure_details().any(|d| d.message.contains("counter")));
    }

    #[test]
    fn strictly_increasing_fails_on_plateau() {
        let f = synthetic_field("counter", vec![(100, 5u64), (200, 5u64)]);
        let mut v = Verdict::new();
        f.strictly_increasing(&mut v);
        let r = v.into_result();
        assert!(r.is_fail());
    }

    #[test]
    fn rate_within_in_band_passes() {
        // Counter advances 1 unit per 100ms = 0.01/ms.
        let f = synthetic_field("ticks", vec![(100, 1.0f64), (200, 2.0f64), (300, 3.0f64)]);
        let mut v = Verdict::new();
        f.rate_within(&mut v, 0.005, 0.02);
        assert!(v.is_pass());
    }

    #[test]
    fn rate_within_out_of_band_fails() {
        let f = synthetic_field("ticks", vec![(100, 1.0f64), (200, 100.0f64)]);
        let mut v = Verdict::new();
        f.rate_within(&mut v, 0.0, 0.5);
        assert!(!v.is_pass());
    }

    /// A zero-time delta between two consecutive samples is
    /// INSTRUMENT-derived (the periodic monitor happened to emit
    /// two samples with the same elapsed_ms — typically a
    /// missed-tick coalescence). rate_within must record this as
    /// Inconclusive, NOT Fail, so a non-measurable interval cannot
    /// silently flip a real-pass workload into a false-fail. Pins
    /// the zero-denominator → Inconclusive contract on the rate
    /// pattern.
    #[test]
    fn rate_within_zero_dt_records_inconclusive() {
        let f = synthetic_field("ticks", vec![(100, 1.0f64), (100, 5.0f64)]);
        let mut v = Verdict::new();
        f.rate_within(&mut v, 0.0, 100.0);
        let r = v.into_result();
        assert!(
            r.is_inconclusive(),
            "zero-dt rate must record Inconclusive: {:?}",
            r.outcomes,
        );
        assert!(
            !r.is_fail(),
            "zero-dt is INSTRUMENT-derived; must NOT record Fail: {:?}",
            r.outcomes,
        );
        assert!(
            r.inconclusive_details()
                .any(|d| d.kind == DetailKind::Temporal
                    && d.message.contains("INSTRUMENT-derived")),
            "inconclusive detail must surface with Temporal kind and \
             INSTRUMENT-derived wording: {:?}",
            r.outcomes,
        );
    }

    #[test]
    fn steady_within_skips_warmup_and_passes() {
        // Warmup at +0..200ms; steady at 10.0 from +300..500.
        let f = synthetic_field(
            "util",
            vec![
                (100, 100.0f64),
                (200, 50.0f64),
                (300, 10.0f64),
                (400, 10.0f64),
                (500, 10.0f64),
            ],
        );
        let mut v = Verdict::new();
        f.steady_within(&mut v, 250, 0.01);
        assert!(v.is_pass(), "{:?}", v.into_result().outcomes);
    }

    #[test]
    fn steady_within_post_warmup_outlier_fails() {
        let f = synthetic_field("util", vec![(300, 10.0f64), (400, 10.0f64), (500, 50.0f64)]);
        let mut v = Verdict::new();
        f.steady_within(&mut v, 0, 0.10);
        assert!(!v.is_pass());
    }

    #[test]
    fn converges_to_finds_witness() {
        let f = synthetic_field(
            "load",
            vec![
                (100, 10.0f64),
                (200, 5.0f64),
                (300, 1.0f64),
                (400, 1.0f64),
                (500, 1.0f64),
            ],
        );
        let mut v = Verdict::new();
        f.converges_to(&mut v, 1.0, 0.5, 1000);
        assert!(v.is_pass());
    }

    #[test]
    fn converges_to_no_witness_fails() {
        let f = synthetic_field("load", vec![(100, 10.0f64), (200, 10.0f64), (300, 10.0f64)]);
        let mut v = Verdict::new();
        f.converges_to(&mut v, 1.0, 0.5, 500);
        assert!(!v.is_pass());
    }

    /// REGRESSION: steady_within SKIPS a sample whose elapsed
    /// timestamp is None — it cannot be placed relative to warmup_ms, so
    /// it must be Note-skipped, never treated as 0 (< warmup, silently
    /// dropped) nor admitted into the steady-state band. The None sample
    /// here carries a wild value that would blow the band if admitted.
    #[test]
    fn steady_within_skips_none_elapsed_with_note() {
        let f: SeriesField<f64> = SeriesField::from_parts_with_phases_opt(
            "util",
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![Some(300), None, Some(400)],
            vec![Ok(10.0), Ok(9999.0), Ok(10.0)],
            vec![None; 3],
        );
        let mut v = Verdict::new();
        f.steady_within(&mut v, 0, 0.05);
        let r = v.into_result();
        assert!(
            r.is_pass(),
            "None-elapsed sample must be skipped, not admitted into the band: {:?}",
            r.outcomes,
        );
        assert!(
            r.info_notes
                .iter()
                .any(|n| n.message.contains("not measured")),
            "the skipped None-elapsed sample must surface a Note: {:?}",
            r.info_notes,
        );
    }

    /// REGRESSION: converges_to treats a None-elapsed sample
    /// as out-of-window — it RESETS the 3-consecutive witness run rather
    /// than bridging it. Here a,b,d are all in-band (1.0) but the None at
    /// index 2 breaks the run, so the longest streak is 2 → no witness →
    /// fail. If None were coerced to 0 (in-window, in-band), c would
    /// complete a 3-run and the assertion would falsely PASS.
    #[test]
    fn converges_to_none_elapsed_breaks_witness() {
        let f: SeriesField<f64> = SeriesField::from_parts_with_phases_opt(
            "load",
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            vec![Some(100), Some(200), None, Some(300)],
            vec![Ok(1.0), Ok(1.0), Ok(1.0), Ok(1.0)],
            vec![None; 4],
        );
        let mut v = Verdict::new();
        f.converges_to(&mut v, 1.0, 0.1, 1000);
        assert!(
            !v.is_pass(),
            "a None-elapsed sample must break the witness run, not bridge it",
        );
    }

    #[test]
    fn always_true_passes_on_all_true() {
        let f = synthetic_field("alive", vec![(100, true), (200, true)]);
        let mut v = Verdict::new();
        f.always_true(&mut v);
        assert!(v.is_pass());
    }

    #[test]
    fn always_true_fails_on_false() {
        let f = synthetic_field("alive", vec![(100, true), (200, false)]);
        let mut v = Verdict::new();
        f.always_true(&mut v);
        assert!(!v.is_pass());
    }

    #[test]
    fn ratio_within_in_band_passes() {
        let lhs = synthetic_field("lhs", vec![(100, 10.0f64), (200, 20.0f64), (300, 30.0f64)]);
        let rhs = synthetic_field("rhs", vec![(100, 5.0f64), (200, 10.0f64), (300, 15.0f64)]);
        let mut v = Verdict::new();
        lhs.ratio_within(&mut v, &rhs, 1.5, 2.5);
        assert!(v.is_pass());
    }

    #[test]
    fn ratio_within_length_mismatch_fails_caller_error() {
        let lhs = synthetic_field("lhs", vec![(100, 10.0f64)]);
        let rhs = synthetic_field("rhs", vec![(100, 5.0f64), (200, 10.0f64)]);
        let mut v = Verdict::new();
        lhs.ratio_within(&mut v, &rhs, 1.5, 2.5);
        assert!(!v.is_pass());
    }

    /// A zero rhs at any sample is INSTRUMENT-derived (the
    /// projected value happened to be zero — a guest counter that
    /// reset, an aggregator that produced a zero bucket). The
    /// ratio_within pattern must record this as Inconclusive, NOT
    /// Fail, so a non-evaluable ratio cannot silently flip a
    /// real-pass workload into a false-fail. Pins the
    /// zero-denominator → Inconclusive contract on the ratio
    /// pattern.
    #[test]
    fn ratio_within_zero_rhs_records_inconclusive() {
        let lhs = synthetic_field("lhs", vec![(100, 10.0f64)]);
        let rhs = synthetic_field("rhs", vec![(100, 0.0f64)]);
        let mut v = Verdict::new();
        lhs.ratio_within(&mut v, &rhs, 1.5, 2.5);
        let r = v.into_result();
        assert!(
            r.is_inconclusive(),
            "zero-rhs ratio must record Inconclusive: {:?}",
            r.outcomes,
        );
        assert!(
            !r.is_fail(),
            "zero rhs is INSTRUMENT-derived; must NOT record Fail: {:?}",
            r.outcomes,
        );
        assert!(
            r.inconclusive_details()
                .any(|d| d.kind == DetailKind::Temporal
                    && d.message.contains("INSTRUMENT-derived")),
            "inconclusive detail must surface with Temporal kind and \
             INSTRUMENT-derived wording: {:?}",
            r.outcomes,
        );
    }

    #[test]
    fn each_at_least_passes() {
        let f = synthetic_field("counter", vec![(100, 5u64), (200, 7u64)]);
        let mut v = Verdict::new();
        f.each(&mut v).at_least(3u64);
        assert!(v.is_pass());
    }

    #[test]
    fn each_at_most_fails_on_outlier() {
        let f = synthetic_field("counter", vec![(100, 5u64), (200, 99u64)]);
        let mut v = Verdict::new();
        f.each(&mut v).at_most(10u64);
        assert!(!v.is_pass());
    }

    #[test]
    fn each_propagates_per_sample_projection_error() {
        let tags = vec!["periodic_000".to_string(), "periodic_001".to_string()];
        let elapsed = vec![100u64, 200u64];
        let values: Vec<SnapshotResult<u64>> = vec![
            Ok(5u64),
            Err(SnapshotError::VarNotFound {
                requested: "missing".to_string(),
                available: vec!["a".to_string()],
            }),
        ];
        let f = SeriesField::from_parts("x", tags, elapsed, values);
        let mut v = Verdict::new();
        f.each(&mut v).at_least(1u64);
        let r = v.into_result();
        assert!(r.is_fail());
        assert!(
            r.failure_details()
                .any(|d| d.message.contains("projection error"))
        );
    }

    // ---- iter_full() ----

    /// iter_full on an empty SeriesField yields no items. Guards
    /// the trivial case so a caller threading the iterator into a
    /// for-loop never triggers a phantom first iteration on a
    /// freshly-constructed empty field.
    #[test]
    fn iter_full_empty_yields_no_items() {
        let f: SeriesField<u64> =
            SeriesField::from_parts("empty", Vec::new(), Vec::new(), Vec::new());
        let collected: Vec<(&str, Option<u64>, &SnapshotResult<u64>)> = f.iter_full().collect();
        assert!(collected.is_empty());
        assert_eq!(f.iter_full().count(), 0);
    }

    /// iter_full on a populated SeriesField yields each
    /// (tag, elapsed_ms, &SnapshotResult<T>) triple in the same
    /// order as the underlying storage — both Ok and Err slots
    /// flow through unchanged. Mixes a successfully-projected
    /// sample with a SnapshotError variant so the test guards both
    /// branches of the per-sample SnapshotResult.
    #[test]
    fn iter_full_yields_triples_in_storage_order() {
        let tags = vec![
            "periodic_000".to_string(),
            "periodic_001".to_string(),
            "periodic_002".to_string(),
        ];
        let elapsed = vec![100u64, 200u64, 300u64];
        let values: Vec<SnapshotResult<u64>> = vec![
            Ok(7u64),
            Err(SnapshotError::VarNotFound {
                requested: "missing".to_string(),
                available: vec!["a".to_string()],
            }),
            Ok(42u64),
        ];
        let f = SeriesField::from_parts("counter", tags, elapsed, values);
        let collected: Vec<(&str, Option<u64>, &SnapshotResult<u64>)> = f.iter_full().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].0, "periodic_000");
        assert_eq!(collected[0].1, Some(100u64));
        assert_eq!(collected[0].2.as_ref().ok().copied(), Some(7u64));
        assert_eq!(collected[1].0, "periodic_001");
        assert_eq!(collected[1].1, Some(200u64));
        assert!(collected[1].2.is_err());
        assert_eq!(collected[2].0, "periodic_002");
        assert_eq!(collected[2].1, Some(300u64));
        assert_eq!(collected[2].2.as_ref().ok().copied(), Some(42u64));
    }

    /// iter_full's item count matches len(). Guards the
    /// equal-length invariant enforced at construction time
    /// (from_parts' assert_eq! checks): if any of the three
    /// vectors drifts, zip's shortest-input behavior would silently
    /// truncate the iterator, so a count mismatch would manifest
    /// here even when no slot is dereferenced.
    #[test]
    fn iter_full_count_matches_len() {
        let f = synthetic_field(
            "counter",
            vec![(100, 1u64), (200, 2u64), (300, 3u64), (400, 4u64)],
        );
        assert_eq!(f.iter_full().count(), f.len());
    }

    /// Vacuous holding when num_snapshots < 2 records a Note, not a
    /// failure.
    #[test]
    fn nondecreasing_with_one_sample_records_note() {
        let f = synthetic_field("counter", vec![(100, 1u64)]);
        let mut v = Verdict::new();
        f.nondecreasing(&mut v);
        let r = v.into_result();
        assert!(r.is_pass());
        assert!(!r.info_notes.is_empty());
    }

    /// End-to-end sample: sanity-check that a series projection
    /// flowing through a temporal pattern produces a coherent
    /// verdict. The `SampleSeries` shape exercise lives in
    /// `src/scenario/sample.rs`; this test only confirms the
    /// integration handshake works.
    #[test]
    fn series_projection_into_temporal_pattern_smoke_check() {
        // Empty series — every pattern should be vacuously ok.
        let series = SampleSeries::empty();
        let field = series.bpf("x", |snap| snap.var("missing").as_u64());
        let mut v = Verdict::new();
        field.nondecreasing(&mut v);
        let r = v.into_result();
        assert!(r.is_pass());
    }

    // ---- Skip-on-projection-error semantics ----

    /// nondecreasing skips errored samples, logs skip count, does
    /// NOT flip the verdict on missing data.
    #[test]
    fn nondecreasing_skips_projection_errors_with_note() {
        let tags = vec![
            "periodic_000".to_string(),
            "periodic_001".to_string(),
            "periodic_002".to_string(),
        ];
        let elapsed = vec![100u64, 200u64, 300u64];
        let values: Vec<SnapshotResult<u64>> = vec![
            Ok(1u64),
            Err(SnapshotError::VarNotFound {
                requested: "x".to_string(),
                available: vec![],
            }),
            Ok(2u64),
        ];
        let f = SeriesField::from_parts("counter", tags, elapsed, values);
        let mut v = Verdict::new();
        f.nondecreasing(&mut v);
        let r = v.into_result();
        assert!(
            r.is_pass(),
            "nondecreasing must NOT flip on projection error: {:?}",
            r.outcomes
        );
        assert!(
            r.info_notes
                .iter()
                .any(|n| n.message.contains("skipped 1 sample")
                    && n.message.contains("periodic_001")),
            "expected skip note: {:?}",
            r.info_notes
        );
    }

    /// rate_within treats errored samples as gaps (no rate
    /// computed across the gap), records skip count via a Note.
    #[test]
    fn rate_within_skips_gaps_with_note() {
        let tags = vec![
            "periodic_000".to_string(),
            "periodic_001".to_string(),
            "periodic_002".to_string(),
        ];
        let elapsed = vec![100u64, 200u64, 300u64];
        let values: Vec<SnapshotResult<f64>> = vec![
            Ok(1.0f64),
            Err(SnapshotError::VarNotFound {
                requested: "x".to_string(),
                available: vec![],
            }),
            Ok(2.0f64),
        ];
        let f = SeriesField::from_parts("ticks", tags, elapsed, values);
        let mut v = Verdict::new();
        f.rate_within(&mut v, 0.0, 1.0);
        let r = v.into_result();
        assert!(
            r.is_pass(),
            "rate_within must NOT flip on gap: {:?}",
            r.outcomes
        );
        assert!(
            r.info_notes.iter().any(|n| n.message.contains("gap")),
            "expected gap note: {:?}",
            r.info_notes
        );
    }

    /// REGRESSION: rate_within SKIPS an interval whose
    /// endpoint has no measured elapsed timestamp (`None`) — the dt is
    /// undefined, so the rate cannot be computed and must be skipped
    /// with a Note, NEVER fabricated from a `0`-coerced dt. The band
    /// `[0, 0.001]` is chosen so the bug-shape (coerce `None`→`0`, giving
    /// the b→c interval dt = 400 and rate (9-2)/400 = 0.0175) would
    /// FAIL; the correct skip leaves no computable interval and passes.
    #[test]
    fn rate_within_skips_none_elapsed_endpoint() {
        let tags = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        // Sample `b` carries no recorded timestamp.
        let elapsed = vec![Some(100u64), None, Some(400u64)];
        let values: Vec<SnapshotResult<f64>> = vec![Ok(1.0), Ok(2.0), Ok(9.0)];
        let f =
            SeriesField::from_parts_with_phases_opt("ticks", tags, elapsed, values, vec![None; 3]);
        let mut v = Verdict::new();
        f.rate_within(&mut v, 0.0, 0.001);
        let r = v.into_result();
        assert!(
            r.is_pass(),
            "None-endpoint intervals must be skipped, not coerced into a band failure: {:?}",
            r.outcomes,
        );
        assert!(
            r.info_notes
                .iter()
                .any(|n| n.message.contains("unmeasured elapsed")),
            "the skipped None-endpoint interval(s) must surface a Note: {:?}",
            r.info_notes,
        );
    }

    /// steady_within skips errored post-warmup samples, records a
    /// Note, does NOT flip the verdict on missing data.
    #[test]
    fn steady_within_skips_projection_errors_with_note() {
        let tags = vec![
            "periodic_000".to_string(),
            "periodic_001".to_string(),
            "periodic_002".to_string(),
        ];
        let elapsed = vec![300u64, 400u64, 500u64];
        let values: Vec<SnapshotResult<f64>> = vec![
            Ok(10.0f64),
            Err(SnapshotError::VarNotFound {
                requested: "x".to_string(),
                available: vec![],
            }),
            Ok(10.0f64),
        ];
        let f = SeriesField::from_parts("util", tags, elapsed, values);
        let mut v = Verdict::new();
        f.steady_within(&mut v, 0, 0.10);
        let r = v.into_result();
        assert!(r.is_pass(), "{:?}", r.outcomes);
        assert!(
            r.info_notes
                .iter()
                .any(|n| n.message.contains("skipped") && n.message.contains("periodic_001")),
            "expected skip note: {:?}",
            r.info_notes
        );
    }

    /// ratio_within skips pairs where either side errored, records
    /// gap count, does NOT flip on missing data.
    #[test]
    fn ratio_within_skips_gaps_with_note() {
        let lhs_values: Vec<SnapshotResult<f64>> = vec![
            Ok(10.0f64),
            Err(SnapshotError::VarNotFound {
                requested: "x".to_string(),
                available: vec![],
            }),
            Ok(20.0f64),
        ];
        let rhs_values: Vec<SnapshotResult<f64>> = vec![Ok(5.0f64), Ok(7.0f64), Ok(10.0f64)];
        let tags = vec![
            "periodic_000".to_string(),
            "periodic_001".to_string(),
            "periodic_002".to_string(),
        ];
        let elapsed = vec![100u64, 200u64, 300u64];
        let lhs = SeriesField::from_parts("lhs", tags.clone(), elapsed.clone(), lhs_values);
        let rhs = SeriesField::from_parts("rhs", tags, elapsed, rhs_values);
        let mut v = Verdict::new();
        lhs.ratio_within(&mut v, &rhs, 1.5, 2.5);
        let r = v.into_result();
        assert!(r.is_pass(), "{:?}", r.outcomes);
        assert!(
            r.info_notes.iter().any(|n| n.message.contains("1 pair")),
            "expected gap note: {:?}",
            r.info_notes
        );
    }

    /// A non-finite (NaN) post-warmup sample must not poison the mean
    /// into a silent PASS. With the NaN dropped from the band
    /// population (and noted), the mean is computed over the finite
    /// samples [10, 100] = 55, so both fall outside ±10% and the
    /// verdict FAILS — whereas the pre-fix code let the NaN make the
    /// mean NaN, the band NaN, and every band check vacuously pass.
    #[test]
    fn steady_within_nan_sample_does_not_silently_pass() {
        let tags = vec![
            "periodic_000".to_string(),
            "periodic_001".to_string(),
            "periodic_002".to_string(),
        ];
        let elapsed = vec![300u64, 400u64, 500u64];
        let values: Vec<SnapshotResult<f64>> = vec![Ok(10.0f64), Ok(f64::NAN), Ok(100.0f64)];
        let f = SeriesField::from_parts("util", tags, elapsed, values);
        let mut v = Verdict::new();
        f.steady_within(&mut v, 0, 0.10);
        let r = v.into_result();
        assert!(
            !r.is_pass(),
            "a NaN sample must not poison the band into a silent pass: {:?}",
            r.outcomes
        );
        assert!(
            r.info_notes
                .iter()
                .any(|n| n.message.contains("non-finite") && n.message.contains("periodic_001")),
            "expected non-finite skip note naming the NaN sample: {:?}",
            r.info_notes
        );
    }

    /// A NaN lhs or rhs yields a non-finite ratio that the `rhs == 0`
    /// guard misses (NaN != 0.0); it must flip the verdict (a detail),
    /// not slip past the band check (NaN comparisons are always false)
    /// into a silent pass.
    #[test]
    fn ratio_within_non_finite_ratio_does_not_silently_pass() {
        let tags = vec!["periodic_000".to_string()];
        let elapsed = vec![100u64];
        // NaN numerator.
        {
            let lhs =
                SeriesField::from_parts("lhs", tags.clone(), elapsed.clone(), vec![Ok(f64::NAN)]);
            let rhs =
                SeriesField::from_parts("rhs", tags.clone(), elapsed.clone(), vec![Ok(5.0f64)]);
            let mut v = Verdict::new();
            lhs.ratio_within(&mut v, &rhs, 0.0, 1.0);
            let r = v.into_result();
            assert!(
                !r.is_pass(),
                "NaN lhs must not silently pass: {:?}",
                r.outcomes
            );
        }
        // NaN denominator (passes the rhs==0 guard, since NaN != 0.0).
        {
            let lhs =
                SeriesField::from_parts("lhs", tags.clone(), elapsed.clone(), vec![Ok(5.0f64)]);
            let rhs =
                SeriesField::from_parts("rhs", tags.clone(), elapsed.clone(), vec![Ok(f64::NAN)]);
            let mut v = Verdict::new();
            lhs.ratio_within(&mut v, &rhs, 0.0, 1.0);
            let r = v.into_result();
            assert!(
                !r.is_pass(),
                "NaN rhs must not silently pass: {:?}",
                r.outcomes
            );
        }
    }

    /// converges_to with fewer than 3 successfully-projected
    /// samples in window records an explicit Note (not a verdict
    /// failure) — absence of data is a coverage gap, not a
    /// negative finding. The Note message names the count and the
    /// requirement so an operator can distinguish "did not collect
    /// enough samples" from "collected enough samples but never
    /// converged".
    #[test]
    fn converges_to_insufficient_samples_records_note() {
        let f = synthetic_field("load", vec![(100, 1.0f64), (200, 1.0f64)]);
        let mut v = Verdict::new();
        f.converges_to(&mut v, 1.0, 0.5, 1000);
        let r = v.into_result();
        assert!(
            r.is_pass(),
            "insufficient-samples must NOT flip the verdict: {:?}",
            r.outcomes
        );
        assert!(
            r.info_notes
                .iter()
                .any(|n| n.message.contains("insufficient samples")
                    && n.message.contains("need ≥3, have 2")),
            "expected insufficient-samples note with count: {:?}",
            r.info_notes
        );
    }

    /// converges_to with 3+ samples in window but none in band
    /// produces the "no witness" structured failure (the
    /// pre-existing code path), distinct from the
    /// insufficient-samples message.
    #[test]
    fn converges_to_no_witness_distinct_from_insufficient() {
        let f = synthetic_field(
            "load",
            vec![
                (100, 10.0f64),
                (200, 10.0f64),
                (300, 10.0f64),
                (400, 10.0f64),
            ],
        );
        let mut v = Verdict::new();
        f.converges_to(&mut v, 1.0, 0.5, 1000);
        let r = v.into_result();
        assert!(r.is_fail());
        assert!(
            r.failure_details()
                .any(|d| d.message.contains("no 3-consecutive-in-band witness")),
            "expected no-witness message: {:?}",
            r.outcomes
        );
        assert!(
            !r.failure_details()
                .any(|d| d.message.contains("insufficient samples")),
            "must NOT report insufficient-samples when there ARE enough samples: {:?}",
            r.outcomes
        );
    }

    // ---- NaN handling ----

    /// each.at_least on NaN sample reports an incomparable
    /// failure rather than silently passing the comparison.
    /// Without the partial_cmp fix, IEEE-754 `<` against NaN
    /// is always false, so a NaN sample would silently pass
    /// `at_least(0.0)`.
    #[test]
    fn each_at_least_flags_nan_sample() {
        let f = synthetic_field("util", vec![(100, 50.0f64), (200, f64::NAN)]);
        let mut v = Verdict::new();
        f.each(&mut v).at_least(0.0f64);
        let r = v.into_result();
        assert!(r.is_fail());
        assert!(
            r.failure_details()
                .any(|d| d.message.contains("NaN") && d.message.contains("periodic_001")),
            "expected NaN failure naming the sample: {:?}",
            r.outcomes
        );
    }

    /// each.at_most on NaN sample reports an incomparable failure.
    #[test]
    fn each_at_most_flags_nan_sample() {
        let f = synthetic_field("util", vec![(100, 50.0f64), (200, f64::NAN)]);
        let mut v = Verdict::new();
        f.each(&mut v).at_most(100.0f64);
        let r = v.into_result();
        assert!(r.is_fail());
        assert!(
            r.failure_details()
                .any(|d| d.message.contains("NaN") && d.message.contains("periodic_001")),
            "expected NaN failure naming the sample: {:?}",
            r.outcomes
        );
    }

    /// each.between on NaN sample reports an incomparable failure.
    #[test]
    fn each_between_flags_nan_sample() {
        let f = synthetic_field("util", vec![(100, 50.0f64), (200, f64::NAN)]);
        let mut v = Verdict::new();
        f.each(&mut v).between(0.0f64, 100.0f64);
        let r = v.into_result();
        assert!(r.is_fail());
        assert!(
            r.failure_details()
                .any(|d| d.message.contains("NaN") && d.message.contains("periodic_001")),
            "expected NaN failure naming the sample: {:?}",
            r.outcomes
        );
    }

    /// rate_within reports a non-finite-rate failure when the
    /// computed rate is NaN or Infinity (e.g. inf-inf endpoints,
    /// NaN in either endpoint, or a finite endpoint difference
    /// that overflows f64). Without the `rate.is_finite()` check,
    /// IEEE-754 `<` against NaN is always false and `<` against
    /// Inf trivially passes any finite ceiling, so non-finite
    /// rates would silently slip past the band check.
    #[test]
    fn rate_within_flags_non_finite_rate() {
        let f = synthetic_field("ticks", vec![(100, f64::INFINITY), (200, f64::INFINITY)]);
        let mut v = Verdict::new();
        f.rate_within(&mut v, 0.0, 1.0);
        let r = v.into_result();
        assert!(r.is_fail());
        assert!(
            r.failure_details()
                .any(|d| d.kind == DetailKind::Temporal && d.message.contains("non-finite rate")),
            "expected non-finite-rate failure: {:?}",
            r.outcomes
        );
    }

    /// nondecreasing skips placeholder samples (is_placeholder=true)
    /// with a Note rather than treating them as monotonicity
    /// regressions or generic projection errors. Placeholder
    /// reports must NOT silently register as zero progress on a
    /// counter.
    #[test]
    fn nondecreasing_skips_placeholder_samples() {
        use crate::monitor::dump::FailureDumpReport;
        let report_a = FailureDumpReport::default(); // not a placeholder; will yield VarNotFound
        let placeholder = FailureDumpReport::placeholder("rendezvous timeout");
        let report_b = FailureDumpReport::default();
        let drained = vec![
            ("periodic_000".to_string(), report_a, None, Some(100u64)),
            ("periodic_001".to_string(), placeholder, None, Some(200u64)),
            ("periodic_002".to_string(), report_b, None, Some(300u64)),
        ];
        let series = SampleSeries::from_drained(drained, None);
        // Project a missing var so non-placeholder samples also
        // produce errors — but the placeholder sample's Err must
        // be the dedicated PlaceholderSample variant. The skip-
        // with-Note path collects all skipped samples; we verify
        // the placeholder tag appears in the skip list.
        let field: SeriesField<u64> = series.bpf("counter", |snap| snap.var("missing").as_u64());
        let mut v = Verdict::new();
        field.nondecreasing(&mut v);
        let r = v.into_result();
        // Verdict passes (nondecreasing skips errored samples).
        assert!(r.is_pass(), "{:?}", r.outcomes);
        // The note message names the placeholder sample.
        assert!(
            r.info_notes
                .iter()
                .any(|n| n.message.contains("periodic_001")),
            "expected skip note naming placeholder sample: {:?}",
            r.info_notes
        );
    }

    /// nondecreasing skips MissingStats samples (stats=None at the
    /// row, surfaced through `series.stats(...)` as the dedicated
    /// `SnapshotError::MissingStats` variant) with a Note rather
    /// than treating them as monotonicity regressions. Mirrors
    /// `nondecreasing_skips_placeholder_samples` for the stats-
    /// coverage gap dimension: a per-sample missing-stats slot must
    /// NOT silently register as zero progress on a counter, and the
    /// skip-with-Note path must name the offending sample so the
    /// operator sees WHICH sample lacked stats.
    #[test]
    fn nondecreasing_skips_missing_stats_samples() {
        use crate::monitor::dump::FailureDumpReport;
        // Build three rows where sample[1]'s stats Option is None.
        // `series.stats(...)` projection will produce a per-sample
        // `Err(SnapshotError::MissingStats { tag: "periodic_001" })`
        // for that row (see SampleSeries::stats at
        // src/scenario/sample.rs lines 275-280) — the analogue of
        // the placeholder path producing PlaceholderSample. The
        // outer rows carry concrete JSON so their projection slot
        // is Ok; only the middle row exercises the MissingStats
        // skip path.
        let stats_a: serde_json::Value = serde_json::json!({"counter": 1u64});
        let stats_b: serde_json::Value = serde_json::json!({"counter": 2u64});
        let drained = vec![
            (
                "periodic_000".to_string(),
                FailureDumpReport::default(),
                Some(stats_a),
                Some(100u64),
            ),
            (
                "periodic_001".to_string(),
                FailureDumpReport::default(),
                None,
                Some(200u64),
            ),
            (
                "periodic_002".to_string(),
                FailureDumpReport::default(),
                Some(stats_b),
                Some(300u64),
            ),
        ];
        let series = SampleSeries::from_drained(drained, None);
        let field: SeriesField<u64> = series.stats("counter", |sv| sv.get("counter").as_u64());
        // Sanity-check the constructed field's middle slot is
        // exactly the MissingStats variant the spec calls out, so a
        // future refactor that drops or renames the variant fails
        // here at the construction site rather than as an opaque
        // verdict mismatch.
        let middle = field.values_iter().nth(1).expect("3 samples");
        assert!(
            matches!(
                middle,
                Err(SnapshotError::MissingStats { tag, .. }) if tag == "periodic_001"
            ),
            "middle slot must be MissingStats('periodic_001'), got {middle:?}"
        );
        let mut v = Verdict::new();
        field.nondecreasing(&mut v);
        let r = v.into_result();
        // Verdict passes — MissingStats is structurally missing
        // data, not a monotonicity regression.
        assert!(
            r.is_pass(),
            "nondecreasing must NOT flip on MissingStats: {:?}",
            r.outcomes
        );
        // The note message names the MissingStats sample so the
        // operator sees the stats-coverage gap without re-walking
        // the source.
        assert!(
            r.info_notes
                .iter()
                .any(|n| n.message.contains("periodic_001")),
            "expected skip note naming MissingStats sample: {:?}",
            r.info_notes
        );
    }

    /// `always_true` emits a `tracing::info!` event under the
    /// `ktstr::assert::temporal` target when log_passes is on AND
    /// the pattern adds no failure details. Mirrors the scalar
    /// claim's positive-confirmation contract at the temporal-
    /// pattern level: a passing run names the label and the
    /// sample count so a `--nocapture` operator sees the
    /// confirmation rather than silent acceptance.
    #[tracing_test::traced_test]
    #[test]
    fn always_true_emits_pass_log_when_log_passes_on() {
        let f = synthetic_field("alive", vec![(100, true), (200, true), (300, true)]);
        let mut v = Verdict::new().with_log_passes(true);
        f.always_true(&mut v);
        assert!(v.is_pass());
        assert!(
            logs_contain("alive (always_true): all 3 samples true"),
            "positive-confirmation log must name the label, pattern, and sample count",
        );
    }

    /// A failed temporal pattern stays silent on the positive
    /// log even when log_passes is on — the pre/post
    /// `temporal_outcome_count` gate ensures a partial-failure
    /// run does not log a misleading "all passed" event.
    #[tracing_test::traced_test]
    #[test]
    fn always_true_silent_on_fail_arm_even_with_log_passes() {
        let f = synthetic_field("alive", vec![(100, true), (200, false)]);
        let mut v = Verdict::new().with_log_passes(true);
        f.always_true(&mut v);
        assert!(!v.is_pass());
        assert!(
            !logs_contain("samples true"),
            "fail arm must NOT emit the positive-confirmation log",
        );
    }

    // ---------- for_each_phase + aggregate_by_phase ----------

    #[test]
    fn series_field_for_each_phase_invokes_closure_per_phase_in_phase_order() {
        let f = SeriesField::<f64>::from_parts_with_phases(
            "x",
            vec!["t0".into(), "t1".into(), "t2".into(), "t3".into()],
            vec![100, 200, 300, 400],
            vec![Ok(10.0), Ok(20.0), Ok(30.0), Ok(40.0)],
            vec![
                Some(crate::assert::Phase::step(1)),
                Some(crate::assert::Phase::BASELINE),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
            ],
        );
        let mut visited: Vec<(crate::assert::Phase, usize)> = Vec::new();
        f.for_each_phase(|phase, samples| {
            visited.push((phase, samples.len()));
        });
        // BTreeMap key order: BASELINE (0) < Step[0] (1) < Step[1] (2)
        assert_eq!(
            visited,
            vec![
                (crate::assert::Phase::BASELINE, 1),
                (crate::assert::Phase::step(0), 2),
                (crate::assert::Phase::step(1), 1),
            ],
            "for_each_phase must iterate phases in BTreeMap (Phase) order",
        );
    }

    #[test]
    fn series_field_for_each_phase_skips_none_phase_samples() {
        let f = SeriesField::<f64>::from_parts_with_phases(
            "x",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![Ok(1.0), Ok(2.0)],
            vec![None, Some(crate::assert::Phase::step(0))],
        );
        let mut visited: Vec<crate::assert::Phase> = Vec::new();
        f.for_each_phase(|phase, _| visited.push(phase));
        assert_eq!(visited, vec![crate::assert::Phase::step(0)]);
    }

    /// Helper: find a registered MetricDef by name from `crate::stats::METRICS`.
    fn metric_by_name(name: &str) -> &'static crate::stats::MetricDef {
        crate::stats::METRICS
            .iter()
            .find(|m| m.name == name)
            .unwrap_or_else(|| panic!("no MetricDef named '{}' in METRICS", name))
    }

    #[test]
    fn series_field_aggregate_by_phase_routes_counter_through_last_minus_first() {
        // Use a registered Counter metric. `total_migrations` is
        // MetricKind::Counter per stats.rs METRICS.
        let metric = metric_by_name("total_migrations");
        assert!(matches!(metric.kind, crate::stats::MetricKind::Counter));
        let f = SeriesField::<f64>::from_parts_with_phases(
            "x",
            vec!["t0".into(), "t1".into(), "t2".into()],
            vec![100, 200, 300],
            vec![Ok(100.0), Ok(150.0), Ok(175.0)],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
            ],
        );
        let agg = f.aggregate_by_phase(metric);
        assert_eq!(agg.len(), 1, "1 distinct phase");
        // Counter last-minus-first: 175 - 100 = 75 (NOT the flat-run sum 425)
        assert_eq!(
            agg[&crate::assert::Phase::step(0)],
            75.0,
            "Counter routes through phase_counter_delta (last-first), not the flat-run sum aggregate_samples",
        );
    }

    /// `sum_by_phase` totals per-read DELTAS per phase — the
    /// delta-reported scx_stats reduction. Each sample is its own
    /// window's count, so the per-phase total is their SUM, NOT a
    /// Counter last-minus-first (which would difference two deltas).
    #[test]
    fn series_field_sum_by_phase_sums_per_read_deltas_per_phase() {
        let f = SeriesField::<f64>::from_parts_with_phases(
            "x",
            vec![
                "t0".into(),
                "t1".into(),
                "t2".into(),
                "t3".into(),
                "t4".into(),
            ],
            vec![100, 200, 300, 400, 500],
            vec![Ok(10.0), Ok(20.0), Ok(5.0), Ok(100.0), Ok(50.0)],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let sums = f.sum_by_phase();
        assert_eq!(sums.len(), 2, "two distinct phases");
        assert_eq!(
            sums[&crate::assert::Phase::step(0)],
            35.0,
            "Step[0] is the SUM of its per-read deltas (10+20+5=35), NOT a \
             last-minus-first (5-10)",
        );
        assert_eq!(
            sums[&crate::assert::Phase::step(1)],
            150.0,
            "Step[1] = 100 + 50",
        );
    }

    #[test]
    fn series_field_aggregate_by_phase_routes_gauge_through_flat_run_aggregator() {
        // `worst_spread` is MetricKind::Gauge(GaugeAgg::Last) per stats.rs METRICS.
        let metric = metric_by_name("worst_spread");
        assert!(matches!(metric.kind, crate::stats::MetricKind::Gauge(_)));
        let f = SeriesField::<f64>::from_parts_with_phases(
            "x",
            vec!["t0".into(), "t1".into(), "t2".into()],
            vec![100, 200, 300],
            vec![Ok(2.0), Ok(4.0), Ok(6.0)],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
            ],
        );
        let agg = f.aggregate_by_phase(metric);
        // Gauge(Last) returns the last finite sample.
        assert_eq!(agg[&crate::assert::Phase::step(0)], 6.0);
    }

    #[test]
    fn series_field_aggregate_by_phase_skips_phases_with_no_finite_samples() {
        let metric = metric_by_name("worst_spread");
        let f = SeriesField::<f64>::from_parts_with_phases(
            "x",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![
                Err(crate::scenario::snapshot::SnapshotError::MissingStats {
                    tag: "t0".into(),
                    reason: crate::scenario::snapshot::MissingStatsReason::NoSchedulerBinary,
                }),
                Ok(5.0),
            ],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let agg = f.aggregate_by_phase(metric);
        assert!(
            !agg.contains_key(&crate::assert::Phase::step(0)),
            "phase with all-Err samples absent",
        );
        assert_eq!(agg[&crate::assert::Phase::step(1)], 5.0);
    }

    // ---------- SeriesField phase column ----------

    #[test]
    fn series_field_from_parts_defaults_phases_to_all_none() {
        let f = SeriesField::<f64>::from_parts(
            "x",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![Ok(1.0), Ok(2.0)],
        );
        let phases: Vec<_> = f.phases_iter().collect();
        assert_eq!(phases, vec![None, None]);
    }

    #[test]
    fn series_field_from_parts_with_phases_preserves_per_sample_phase() {
        let f = SeriesField::<f64>::from_parts_with_phases(
            "x",
            vec!["t0".into(), "t1".into(), "t2".into()],
            vec![100, 200, 300],
            vec![Ok(1.0), Ok(2.0), Ok(3.0)],
            vec![
                Some(crate::assert::Phase::BASELINE),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let phases: Vec<_> = f.phases_iter().collect();
        assert_eq!(
            phases,
            vec![
                Some(crate::assert::Phase::BASELINE),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn series_field_from_parts_with_phases_rejects_length_mismatch() {
        let _ = SeriesField::<f64>::from_parts_with_phases(
            "x",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![Ok(1.0), Ok(2.0)],
            vec![Some(crate::assert::Phase::BASELINE)], // 1 != 2 values
        );
    }

    #[test]
    fn series_field_by_phase_partitions_into_per_phase_buckets() {
        let f = SeriesField::<f64>::from_parts_with_phases(
            "x",
            vec!["t0".into(), "t1".into(), "t2".into(), "t3".into()],
            vec![100, 200, 300, 400],
            vec![Ok(10.0), Ok(20.0), Ok(30.0), Ok(40.0)],
            vec![
                Some(crate::assert::Phase::BASELINE),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let (by_phase, none_bucket) = f.by_phase();
        assert!(
            none_bucket.is_empty(),
            "no None-phase samples in this fixture"
        );
        assert_eq!(
            by_phase.len(),
            3,
            "3 distinct phases: BASELINE, Step[0], Step[1]"
        );
        assert_eq!(by_phase[&crate::assert::Phase::BASELINE].len(), 1);
        assert_eq!(by_phase[&crate::assert::Phase::step(0)].len(), 2);
        assert_eq!(by_phase[&crate::assert::Phase::step(1)].len(), 1);
    }

    #[test]
    fn series_field_by_phase_collects_none_samples_in_separate_bucket() {
        let f = SeriesField::<f64>::from_parts_with_phases(
            "x",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![Ok(1.0), Ok(2.0)],
            vec![None, Some(crate::assert::Phase::step(0))],
        );
        let (by_phase, none_bucket) = f.by_phase();
        assert_eq!(none_bucket.len(), 1, "1 None-phase sample");
        assert_eq!(by_phase.len(), 1, "1 phase bucket");
        assert_eq!(by_phase[&crate::assert::Phase::step(0)].len(), 1);
    }

    // ---------- phase() / value_at_phase() / last_per_phase() / ratio_across_phases() ----------

    #[test]
    fn phase_returns_only_samples_in_named_phase() {
        let f = SeriesField::<u64>::from_parts_with_phases(
            "ticks",
            vec!["t0".into(), "t1".into(), "t2".into(), "t3".into()],
            vec![100, 200, 300, 400],
            vec![Ok(1), Ok(2), Ok(3), Ok(4)],
            vec![
                Some(crate::assert::Phase::BASELINE),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let step0: Vec<_> = f
            .phase(crate::assert::Phase::step(0))
            .into_iter()
            .map(|(_, _, v)| v.as_ref().copied().ok())
            .collect();
        assert_eq!(step0, vec![Some(2), Some(3)]);
        let step2 = f.phase(crate::assert::Phase::step(2));
        assert!(
            step2.is_empty(),
            "phase with no samples must return empty Vec, got {step2:?}",
        );
    }

    #[test]
    fn value_at_phase_returns_last_ok_for_phase() {
        let f = SeriesField::<u64>::from_parts_with_phases(
            "ticks",
            vec!["t0".into(), "t1".into(), "t2".into()],
            vec![100, 200, 300],
            vec![Ok(10), Ok(20), Ok(30)],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        assert_eq!(
            f.value_at_phase(crate::assert::Phase::step(0)),
            Some(20),
            "value_at_phase returns the LAST Ok-sample for the phase",
        );
        assert_eq!(f.value_at_phase(crate::assert::Phase::step(1)), Some(30),);
        assert_eq!(
            f.value_at_phase(crate::assert::Phase::step(2)),
            None,
            "phase with no samples returns None",
        );
    }

    #[test]
    fn value_at_phase_skips_err_samples_within_phase() {
        let f = SeriesField::<u64>::from_parts_with_phases(
            "ticks",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![
                Ok(7),
                Err(SnapshotError::VarNotFound {
                    requested: "x".into(),
                    available: vec![],
                }),
            ],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
            ],
        );
        assert_eq!(
            f.value_at_phase(crate::assert::Phase::step(0)),
            Some(7),
            "Err in same phase must be skipped; last Ok wins",
        );
    }

    #[test]
    fn last_per_phase_returns_last_ok_per_present_phase() {
        let f = SeriesField::<u64>::from_parts_with_phases(
            "ticks",
            vec!["t0".into(), "t1".into(), "t2".into(), "t3".into()],
            vec![100, 200, 300, 400],
            vec![Ok(1), Ok(2), Ok(3), Ok(4)],
            vec![
                Some(crate::assert::Phase::BASELINE),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let m = f.last_per_phase();
        assert_eq!(m.len(), 3, "BASELINE + Step[0] + Step[1]");
        assert_eq!(m[&crate::assert::Phase::BASELINE], 1);
        assert_eq!(m[&crate::assert::Phase::step(0)], 3);
        assert_eq!(m[&crate::assert::Phase::step(1)], 4);
    }

    #[test]
    fn last_per_phase_omits_phases_with_only_err_samples() {
        let f = SeriesField::<u64>::from_parts_with_phases(
            "ticks",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![
                Err(SnapshotError::VarNotFound {
                    requested: "x".into(),
                    available: vec![],
                }),
                Ok(9),
            ],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let m = f.last_per_phase();
        assert!(
            !m.contains_key(&crate::assert::Phase::step(0)),
            "all-Err phase omitted from last_per_phase, got keys {:?}",
            m.keys().collect::<Vec<_>>(),
        );
        assert_eq!(m[&crate::assert::Phase::step(1)], 9);
    }

    #[test]
    fn first_per_phase_returns_first_ok_per_present_phase() {
        let f = SeriesField::<u64>::from_parts_with_phases(
            "ticks",
            vec!["t0".into(), "t1".into(), "t2".into(), "t3".into()],
            vec![100, 200, 300, 400],
            vec![Ok(10), Ok(20), Ok(30), Ok(40)],
            vec![
                Some(crate::assert::Phase::BASELINE),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let m = f.first_per_phase();
        assert_eq!(m.len(), 3);
        assert_eq!(m[&crate::assert::Phase::BASELINE], 10);
        assert_eq!(
            m[&crate::assert::Phase::step(0)],
            20,
            "first Ok in Step[0] is t1=20, NOT t2=30",
        );
        assert_eq!(m[&crate::assert::Phase::step(1)], 40);
    }

    #[test]
    fn first_per_phase_skips_leading_err_samples_within_phase() {
        let f = SeriesField::<u64>::from_parts_with_phases(
            "ticks",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![
                Err(SnapshotError::VarNotFound {
                    requested: "x".into(),
                    available: vec![],
                }),
                Ok(7),
            ],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
            ],
        );
        assert_eq!(
            f.first_per_phase()[&crate::assert::Phase::step(0)],
            7,
            "leading Err in phase must be skipped; first Ok wins",
        );
    }

    #[test]
    fn counter_delta_per_phase_regressed_phase_reports_zero_instead_of_panicking() {
        // Picker drift (or upstream-signal rollback) within a phase
        // produces `last < first`. The reducer must not panic; it
        // reports zero progress for that phase and emits a tracing
        // warn (verified separately via test_log subscribers — here we
        // pin the no-panic + zero contract).
        let f = SeriesField::<u64>::from_parts_with_phases(
            "ticks",
            vec!["t0".into(), "t1".into(), "t2".into(), "t3".into()],
            vec![100, 200, 300, 400],
            // Phase 0: monotonic 100 -> 150 (delta 50).
            // Phase 1: regressed 1000 -> 200 (would panic in debug
            // pre-fix, wrap in release).
            vec![Ok(100), Ok(150), Ok(1000), Ok(200)],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let m = f.counter_delta_per_phase();
        assert_eq!(m.len(), 2);
        assert_eq!(m[&crate::assert::Phase::step(0)], 50);
        assert_eq!(
            m[&crate::assert::Phase::step(1)],
            0,
            "regressed phase must yield 0 (no progress measurable) \
             rather than panicking on the underflowed subtraction",
        );
    }

    #[test]
    fn counter_delta_per_phase_subtracts_first_from_last() {
        let f = SeriesField::<u64>::from_parts_with_phases(
            "ticks",
            vec!["t0".into(), "t1".into(), "t2".into(), "t3".into()],
            vec![100, 200, 300, 400],
            vec![Ok(100), Ok(150), Ok(180), Ok(200)],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let m = f.counter_delta_per_phase();
        assert_eq!(m.len(), 2);
        assert_eq!(
            m[&crate::assert::Phase::step(0)],
            50,
            "Step[0]: last(150) - first(100) = 50",
        );
        assert_eq!(
            m[&crate::assert::Phase::step(1)],
            20,
            "Step[1]: last(200) - first(180) = 20 (NOT 200 - 100 = 100; \
             prior-phase accumulation excluded)",
        );
    }

    #[test]
    fn counter_delta_per_phase_single_sample_phase_yields_zero() {
        let f = SeriesField::<u64>::from_parts_with_phases(
            "ticks",
            vec!["t0".into()],
            vec![100],
            vec![Ok(42)],
            vec![Some(crate::assert::Phase::step(0))],
        );
        let m = f.counter_delta_per_phase();
        assert_eq!(
            m[&crate::assert::Phase::step(0)],
            0,
            "single Ok sample: first == last → delta of zero",
        );
    }

    #[test]
    fn counter_delta_per_phase_omits_phases_with_only_err_samples() {
        let f = SeriesField::<u64>::from_parts_with_phases(
            "ticks",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![
                Err(SnapshotError::VarNotFound {
                    requested: "x".into(),
                    available: vec![],
                }),
                Ok(5),
            ],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let m = f.counter_delta_per_phase();
        assert!(
            !m.contains_key(&crate::assert::Phase::step(0)),
            "all-Err phase omitted from counter_delta_per_phase",
        );
        assert_eq!(
            m[&crate::assert::Phase::step(1)],
            0,
            "single-Ok phase: first(5) == last(5) → delta of zero",
        );
    }

    #[test]
    fn counter_delta_per_phase_composes_with_a_b_ratio() {
        // Real-world composition exercise: two cumulative counters
        // (same / cross) projected separately, folded per-phase
        // into a fraction, then compared across phases. Pins the
        // load-bearing usage pattern that motivated the reducer.
        let tags: Vec<String> = (0..4).map(|i| format!("p{i}")).collect();
        let elapsed = vec![100, 200, 300, 400];
        let phases = vec![
            Some(crate::assert::Phase::step(0)),
            Some(crate::assert::Phase::step(0)),
            Some(crate::assert::Phase::step(1)),
            Some(crate::assert::Phase::step(1)),
        ];
        // same: 1000 → 1200 in step(0), 1200 → 1800 in step(1)
        let same = SeriesField::<u64>::from_parts_with_phases(
            "same",
            tags.clone(),
            elapsed.clone(),
            vec![Ok(1000), Ok(1200), Ok(1200), Ok(1800)],
            phases.clone(),
        );
        // cross: 100 → 200 in step(0), 200 → 300 in step(1)
        let cross = SeriesField::<u64>::from_parts_with_phases(
            "cross",
            tags,
            elapsed,
            vec![Ok(100), Ok(200), Ok(200), Ok(300)],
            phases,
        );
        let same_d = same.counter_delta_per_phase();
        let cross_d = cross.counter_delta_per_phase();
        let cross_frac = |p: crate::assert::Phase| -> f64 {
            let s = same_d[&p] as f64;
            let c = cross_d[&p] as f64;
            c / (s + c)
        };
        // Step[0]: 100 / (200 + 100) = 0.333...
        // Step[1]: 100 / (600 + 100) = 0.143
        let f0 = cross_frac(crate::assert::Phase::step(0));
        let f1 = cross_frac(crate::assert::Phase::step(1));
        assert!((f0 - 0.333333).abs() < 1e-4, "Step[0] cross_frac = {f0}");
        assert!((f1 - 0.142857).abs() < 1e-4, "Step[1] cross_frac = {f1}");
        let ratio = f1 / f0;
        assert!(
            ratio < 0.5,
            "phase-delta cross_frac ratio {ratio} should be well below 0.5 — \
             prior-phase accumulation would have inflated phase 1's reading",
        );
    }

    #[test]
    fn ratio_across_phases_pass_records_info_note() {
        // Step[0] = 100, Step[1] = 50 → ratio 0.5, ceiling 0.85 ⇒ pass.
        let f = SeriesField::<f64>::from_parts_with_phases(
            "dispatches",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![Ok(100.0), Ok(50.0)],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let mut v = Verdict::new();
        f.ratio_across_phases(
            &mut v,
            crate::assert::Phase::step(0),
            crate::assert::Phase::step(1),
        )
        .at_most(0.85);
        let r = v.into_result();
        assert!(
            r.is_pass(),
            "expected pass, got outcomes={:?} details={:?}",
            r.outcomes,
            r.failure_details().collect::<Vec<_>>(),
        );
        assert!(
            r.info_notes.iter().any(|n| n.message.contains("dispatches")
                && n.message.contains("50/100")
                && n.message.contains("0.5000")
                && n.message.contains("ceiling 0.8500")),
            "expected pass info note carrying ratio + ceiling, got {:?}",
            r.info_notes,
        );
    }

    /// A non-finite phase value must not let `ratio_across_phases().at_most()`
    /// silently pass: a NaN `later` makes later/earlier = NaN, the
    /// `earlier == 0.0` guard misses it (NaN != 0.0), and raw
    /// `ratio > ceiling` is false for NaN — so without the non-finite
    /// guard the pair would PASS. Same class as the ratio_within /
    /// steady_within NaN fixes.
    #[test]
    fn ratio_across_phases_non_finite_does_not_silently_pass() {
        let f = SeriesField::<f64>::from_parts_with_phases(
            "dispatches",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![Ok(100.0f64), Ok(f64::NAN)],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let mut v = Verdict::new();
        f.ratio_across_phases(
            &mut v,
            crate::assert::Phase::step(0),
            crate::assert::Phase::step(1),
        )
        .at_most(0.85);
        let r = v.into_result();
        assert!(
            !r.is_pass(),
            "a non-finite phase ratio must not silently pass: outcomes={:?}",
            r.outcomes,
        );
        assert!(
            r.failure_details()
                .any(|d| d.message.contains("non-finite")),
            "expected a non-finite failure detail: {:?}",
            r.failure_details().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn ratio_across_phases_failure_records_detail_with_ratio() {
        // Step[0] = 10, Step[1] = 20 → ratio 2.0, ceiling 0.85 ⇒ fail.
        let f = SeriesField::<f64>::from_parts_with_phases(
            "dispatches",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![Ok(10.0), Ok(20.0)],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let mut v = Verdict::new();
        f.ratio_across_phases(
            &mut v,
            crate::assert::Phase::step(0),
            crate::assert::Phase::step(1),
        )
        .at_most(0.85);
        let r = v.into_result();
        assert!(r.is_fail(), "expected fail, got outcomes={:?}", r.outcomes);
        assert!(
            r.failure_details().any(|d| d.kind == DetailKind::Temporal
                && d.message.contains("dispatches")
                && d.message.contains("20/10")
                && d.message.contains("2.0000")
                && d.message.contains("ceiling 0.8500")),
            "expected fail detail carrying ratio + ceiling, got {:?}",
            r.failure_details().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn ratio_across_phases_missing_phase_is_inconclusive_with_clear_detail() {
        // Step[1] has no samples → Inconclusive with "needs both
        // phases" (the ratio cannot be computed; neither pass nor
        // fail is truthful per the INSTRUMENT-derived zero-signal
        // contract).
        let f = SeriesField::<f64>::from_parts_with_phases(
            "dispatches",
            vec!["t0".into()],
            vec![100],
            vec![Ok(10.0)],
            vec![Some(crate::assert::Phase::step(0))],
        );
        let mut v = Verdict::new();
        f.ratio_across_phases(
            &mut v,
            crate::assert::Phase::step(0),
            crate::assert::Phase::step(1),
        )
        .at_most(0.85);
        let r = v.into_result();
        assert!(
            r.is_inconclusive(),
            "expected Inconclusive, got {:?}",
            r.outcomes
        );
        assert!(
            r.inconclusive_details()
                .any(|d| d.message.contains("needs both phases")
                    && d.message.contains("later=<no-samples>")),
            "expected `needs both phases` Inconclusive reason naming the missing side, got {:?}",
            r.inconclusive_details().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn ratio_across_phases_zero_baseline_is_inconclusive_with_clear_detail() {
        // earlier=0 → Inconclusive with "earlier value is 0".
        // earlier_f == 0 means the baseline measured zero; the
        // ratio later/earlier is undefined (INSTRUMENT-derived
        // zero denominator). Pre-#434 this was Fail; post-#434
        // the gate cannot evaluate so the verdict is neither
        // pass nor fail.
        let f = SeriesField::<f64>::from_parts_with_phases(
            "dispatches",
            vec!["t0".into(), "t1".into()],
            vec![100, 200],
            vec![Ok(0.0), Ok(5.0)],
            vec![
                Some(crate::assert::Phase::step(0)),
                Some(crate::assert::Phase::step(1)),
            ],
        );
        let mut v = Verdict::new();
        f.ratio_across_phases(
            &mut v,
            crate::assert::Phase::step(0),
            crate::assert::Phase::step(1),
        )
        .at_most(0.85);
        let r = v.into_result();
        assert!(
            r.is_inconclusive(),
            "expected Inconclusive, got {:?}",
            r.outcomes
        );
        assert!(
            r.inconclusive_details()
                .any(|d| d.message.contains("earlier value is 0")
                    && d.message.contains("no baseline")),
            "expected `earlier value is 0` Inconclusive reason, got {:?}",
            r.inconclusive_details().collect::<Vec<_>>(),
        );
    }

    // ---------- PhaseMapExt::ratio_across_phases (BTreeMap<Phase, T> entry) ----------

    #[test]
    fn phasemap_ratio_across_phases_pass_records_info_note() {
        let mut m: std::collections::BTreeMap<crate::assert::Phase, f64> =
            std::collections::BTreeMap::new();
        m.insert(crate::assert::Phase::step(0), 10.0);
        m.insert(crate::assert::Phase::step(1), 5.0);
        let mut v = Verdict::new();
        m.ratio_across_phases(
            &mut v,
            "cross_frac",
            crate::assert::Phase::step(0),
            crate::assert::Phase::step(1),
        )
        .at_most(0.85);
        let r = v.into_result();
        assert!(r.is_pass(), "expected pass, got {:?}", r.outcomes);
        assert!(
            r.info_notes.iter().any(|n| n.message.contains("cross_frac")
                && n.message.contains("5/10")
                && n.message.contains("0.5000")
                && n.message.contains("ceiling 0.8500")),
            "expected pass info note with caller-supplied label, got {:?}",
            r.info_notes,
        );
    }

    #[test]
    fn phasemap_ratio_across_phases_failure_records_detail_with_ratio() {
        let mut m: std::collections::BTreeMap<crate::assert::Phase, f64> =
            std::collections::BTreeMap::new();
        m.insert(crate::assert::Phase::step(0), 10.0);
        m.insert(crate::assert::Phase::step(1), 20.0);
        let mut v = Verdict::new();
        m.ratio_across_phases(
            &mut v,
            "cross_frac",
            crate::assert::Phase::step(0),
            crate::assert::Phase::step(1),
        )
        .at_most(0.85);
        let r = v.into_result();
        assert!(r.is_fail());
        assert!(
            r.failure_details().any(|d| d.kind == DetailKind::Temporal
                && d.message.contains("cross_frac")
                && d.message.contains("20/10")
                && d.message.contains("2.0000")
                && d.message.contains("ceiling 0.8500")),
            "expected fail detail with caller-supplied label + ratio, got {:?}",
            r.failure_details().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn phasemap_ratio_across_phases_missing_phase_is_inconclusive_with_clear_detail() {
        let mut m: std::collections::BTreeMap<crate::assert::Phase, f64> =
            std::collections::BTreeMap::new();
        m.insert(crate::assert::Phase::step(0), 10.0);
        // Phase 1 absent
        let mut v = Verdict::new();
        m.ratio_across_phases(
            &mut v,
            "cross_frac",
            crate::assert::Phase::step(0),
            crate::assert::Phase::step(1),
        )
        .at_most(0.85);
        let r = v.into_result();
        assert!(
            r.is_inconclusive(),
            "expected Inconclusive, got {:?}",
            r.outcomes
        );
        assert!(
            r.inconclusive_details()
                .any(|d| d.message.contains("needs both phases")
                    && d.message.contains("later=<no-samples>")),
            "expected needs-both-phases Inconclusive reason, got {:?}",
            r.inconclusive_details().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn phasemap_ratio_across_phases_zero_baseline_is_inconclusive_with_clear_detail() {
        let mut m: std::collections::BTreeMap<crate::assert::Phase, f64> =
            std::collections::BTreeMap::new();
        m.insert(crate::assert::Phase::step(0), 0.0);
        m.insert(crate::assert::Phase::step(1), 5.0);
        let mut v = Verdict::new();
        m.ratio_across_phases(
            &mut v,
            "cross_frac",
            crate::assert::Phase::step(0),
            crate::assert::Phase::step(1),
        )
        .at_most(0.85);
        let r = v.into_result();
        assert!(
            r.is_inconclusive(),
            "expected Inconclusive, got {:?}",
            r.outcomes
        );
        assert!(
            r.inconclusive_details()
                .any(|d| d.message.contains("earlier value is 0")
                    && d.message.contains("no baseline")),
            "expected zero-baseline Inconclusive reason, got {:?}",
            r.inconclusive_details().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn phasemap_ratio_across_phases_disjoint_phase_keys_is_inconclusive_cleanly() {
        // BTreeMap is non-empty but neither queried phase exists.
        // Both sides yield <no-samples> → Inconclusive (neither
        // pass nor fail can be evaluated when both inputs are
        // missing).
        let mut m: std::collections::BTreeMap<crate::assert::Phase, f64> =
            std::collections::BTreeMap::new();
        m.insert(crate::assert::Phase::BASELINE, 7.0);
        m.insert(crate::assert::Phase::step(5), 8.0);
        let mut v = Verdict::new();
        m.ratio_across_phases(
            &mut v,
            "cross_frac",
            crate::assert::Phase::step(0),
            crate::assert::Phase::step(1),
        )
        .at_most(0.85);
        let r = v.into_result();
        assert!(
            r.is_inconclusive(),
            "expected Inconclusive, got {:?}",
            r.outcomes
        );
        assert!(
            r.inconclusive_details()
                .any(|d| d.message.contains("needs both phases")
                    && d.message.contains("earlier=<no-samples>")
                    && d.message.contains("later=<no-samples>")),
            "both phases absent must surface in Inconclusive reason, got {:?}",
            r.inconclusive_details().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn cross_phase_ratio_empty_label_omits_label_prefix() {
        // Regression pin: with label="", the failure detail must NOT
        // have a leading ": " from a stale "{label}: " concatenation.
        let mut m: std::collections::BTreeMap<crate::assert::Phase, f64> =
            std::collections::BTreeMap::new();
        m.insert(crate::assert::Phase::step(0), 10.0);
        m.insert(crate::assert::Phase::step(1), 20.0);
        let mut v = Verdict::new();
        m.ratio_across_phases(
            &mut v,
            "",
            crate::assert::Phase::step(0),
            crate::assert::Phase::step(1),
        )
        .at_most(0.85);
        let r = v.into_result();
        let first = r
            .failure_details()
            .next()
            .expect("empty label still produces a detail when comparator fails");
        assert!(
            first.message.starts_with("ratio_across_phases("),
            "empty label must omit leading prefix; got {:?}",
            first.message,
        );
    }

    // ---------- PhaseMapExt::zip_per_phase ----------

    #[test]
    fn zip_per_phase_intersects_phase_keys() {
        let mut a: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        a.insert(crate::assert::Phase::step(0), 10);
        a.insert(crate::assert::Phase::step(1), 20);
        a.insert(crate::assert::Phase::step(2), 30);
        let mut b: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        b.insert(crate::assert::Phase::step(1), 100);
        b.insert(crate::assert::Phase::step(2), 200);
        b.insert(crate::assert::Phase::step(3), 300);
        let z = a.zip_per_phase(&b, |s, t| s + t);
        assert_eq!(z.len(), 2);
        assert_eq!(z[&crate::assert::Phase::step(1)], 120);
        assert_eq!(z[&crate::assert::Phase::step(2)], 230);
        assert!(!z.contains_key(&crate::assert::Phase::step(0)));
        assert!(!z.contains_key(&crate::assert::Phase::step(3)));
    }

    #[test]
    fn zip_per_phase_empty_intersection_yields_empty() {
        let mut a: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        a.insert(crate::assert::Phase::step(0), 1);
        let mut b: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        b.insert(crate::assert::Phase::step(1), 2);
        let z = a.zip_per_phase(&b, |s, t| s + t);
        assert!(z.is_empty());
    }

    #[test]
    fn zip_per_phase_both_empty_yields_empty() {
        let a: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        let b: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        let z = a.zip_per_phase(&b, |s, t| s + t);
        assert!(z.is_empty());
    }

    #[test]
    fn zip_per_phase_heterogeneous_t_u_types() {
        // Pins T and U can differ — trait isn't accidentally T=U-bound.
        let mut a: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        a.insert(crate::assert::Phase::step(0), 100);
        a.insert(crate::assert::Phase::step(1), 200);
        let mut b: std::collections::BTreeMap<crate::assert::Phase, f64> =
            std::collections::BTreeMap::new();
        b.insert(crate::assert::Phase::step(0), 0.5);
        b.insert(crate::assert::Phase::step(1), 2.0);
        let z = a.zip_per_phase(&b, |s, t| s as f64 * t);
        assert_eq!(z[&crate::assert::Phase::step(0)], 50.0);
        assert_eq!(z[&crate::assert::Phase::step(1)], 400.0);
    }

    #[test]
    fn zip_per_phase_takes_values_by_value_no_deref_noise() {
        // The composition body operates on owned T/U directly.
        // No `*s` / `*c` syntax — pins the bound is `T: Copy + U: Copy`
        // by-value, not by-reference.
        let mut a: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        a.insert(crate::assert::Phase::step(0), 1000);
        a.insert(crate::assert::Phase::step(1), 1200);
        let mut b: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        b.insert(crate::assert::Phase::step(0), 100);
        b.insert(crate::assert::Phase::step(1), 200);
        let frac = a.zip_per_phase(&b, |s, c| {
            let total = (s + c) as f64;
            if total == 0.0 { 0.0 } else { c as f64 / total }
        });
        assert!((frac[&crate::assert::Phase::step(0)] - (100.0 / 1100.0)).abs() < 1e-9);
        assert!((frac[&crate::assert::Phase::step(1)] - (200.0 / 1400.0)).abs() < 1e-9);
    }

    #[test]
    fn zip_then_ratio_across_phases_composes_a_b_test() {
        // End-to-end composition the scx_mitosis test reaches for:
        // two counter-delta maps → zip into cross_frac → ratio across
        // phases → verdict mutation.
        let mut same_d: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        same_d.insert(crate::assert::Phase::step(0), 200);
        same_d.insert(crate::assert::Phase::step(1), 600);
        let mut cross_d: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        cross_d.insert(crate::assert::Phase::step(0), 100);
        cross_d.insert(crate::assert::Phase::step(1), 100);
        let frac = same_d.zip_per_phase(&cross_d, |s, c| {
            let total = (s + c) as f64;
            if total == 0.0 { 0.0 } else { c as f64 / total }
        });
        let mut v = Verdict::new();
        frac.ratio_across_phases(
            &mut v,
            "cross_frac",
            crate::assert::Phase::step(0),
            crate::assert::Phase::step(1),
        )
        .at_most(0.85);
        let r = v.into_result();
        assert!(
            r.is_pass(),
            "Step[0] cross_frac = 100/300 ≈ 0.333, Step[1] = 100/700 ≈ 0.143; \
             ratio 0.143/0.333 ≈ 0.43 well below 0.85 ceiling. \
             Got outcomes={:?}, details={:?}",
            r.outcomes,
            r.failure_details().collect::<Vec<_>>(),
        );
        assert!(
            r.info_notes
                .iter()
                .any(|n| n.message.contains("cross_frac")
                    && n.message.contains("ratio_across_phases")),
            "expected pass info note carrying the composed-metric label, \
             got {:?}",
            r.info_notes,
        );
    }

    /// `frac_pair` collapses the `n / (n + m)` safe-divide closure
    /// that `zip_per_phase` callers spell inline. Two phase maps
    /// with overlapping phase keys produce per-phase fractions of
    /// `self / (self + other)`.
    #[test]
    fn frac_pair_computes_share_of_total_per_phase() {
        let mut cross: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        cross.insert(crate::assert::Phase::step(0), 100);
        cross.insert(crate::assert::Phase::step(1), 100);
        let mut same: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        same.insert(crate::assert::Phase::step(0), 200);
        same.insert(crate::assert::Phase::step(1), 600);
        let frac = cross.frac_pair(&same);
        assert!(
            (frac[&crate::assert::Phase::step(0)] - (100.0 / 300.0)).abs() < 1e-9,
            "phase 0: 100/(100+200) = 1/3; got {}",
            frac[&crate::assert::Phase::step(0)],
        );
        assert!(
            (frac[&crate::assert::Phase::step(1)] - (100.0 / 700.0)).abs() < 1e-9,
            "phase 1: 100/(100+600) = 1/7; got {}",
            frac[&crate::assert::Phase::step(1)],
        );
    }

    /// When both inputs sum to zero for a phase, `frac_pair` MUST
    /// drop the entry rather than synthesize `0.0`. A synthesized
    /// `0.0` would slip past any downstream `at_most(thr > 0)` gate
    /// without ever observing the phase pair carries no signal —
    /// the silent-pass class of bug `Outcome::Inconclusive` was
    /// introduced to prevent. Dropping the entry surfaces the
    /// absence so the consumer (typically a `ratio_within` /
    /// `at_most` chain over the resulting map) treats the missing
    /// phase the same as one only present on a single side —
    /// Inconclusive at the comparator boundary, never a silent pass.
    #[test]
    fn frac_pair_zero_total_drops_entry_no_silent_pass() {
        let mut a: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        a.insert(crate::assert::Phase::step(0), 0);
        let mut b: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        b.insert(crate::assert::Phase::step(0), 0);
        let frac = a.frac_pair(&b);
        assert!(
            !frac.contains_key(&crate::assert::Phase::step(0)),
            "zero/(zero+zero) must drop entry, not synthesize 0.0; got {frac:?}",
        );
        assert!(
            frac.is_empty(),
            "no phases survived → empty map; got {frac:?}"
        );
    }

    /// One zero side plus a positive other side is NOT a zero-total
    /// pair — the total is positive and the fraction is `0/(0+m) =
    /// 0.0` (a real measurement of "self has zero share"). The
    /// entry MUST be retained because the signal is real even
    /// though the value is zero. Pins the boundary between
    /// "real-zero" (kept) and "no-signal" (dropped).
    #[test]
    fn frac_pair_zero_self_positive_other_keeps_real_zero() {
        let mut a: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        a.insert(crate::assert::Phase::step(0), 0);
        let mut b: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        b.insert(crate::assert::Phase::step(0), 100);
        let frac = a.frac_pair(&b);
        let v = frac[&crate::assert::Phase::step(0)];
        assert_eq!(
            v, 0.0,
            "0/(0+100) is a real-zero measurement, not no-signal; got {v}",
        );
        assert!(!v.is_nan(), "frac_pair must never produce NaN");
    }

    /// `frac_pair` MUST saturate on u64 overflow rather than wrap.
    /// Two near-`u64::MAX` counter deltas wrapping silently would
    /// produce a wrong fraction. With `saturating_add`, the total
    /// caps at `u64::MAX` so the fraction is bounded and finite.
    #[test]
    fn frac_pair_saturates_on_u64_overflow_no_wrap() {
        let mut a: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        a.insert(crate::assert::Phase::step(0), u64::MAX);
        let mut b: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        b.insert(crate::assert::Phase::step(0), 1);
        let frac = a.frac_pair(&b);
        let v = frac[&crate::assert::Phase::step(0)];
        // u64::MAX.saturating_add(1) == u64::MAX. The fraction is
        // `u64::MAX as f64 / u64::MAX as f64` — both sides land on
        // the same f64 value (lossy cast collapses the bottom
        // ~11 bits) so the ratio is exactly 1.0.
        assert_eq!(
            v, 1.0,
            "saturating_add caps total at u64::MAX; both sides cast to same f64 → 1.0; got {v}",
        );
        assert!(!v.is_nan(), "must never produce NaN even at saturation");
        assert!(
            v.is_finite(),
            "must produce a finite value even at saturation"
        );
    }

    /// Intersection-only semantics: phases present in only one
    /// input drop from the result. Mirrors `zip_per_phase`.
    #[test]
    fn frac_pair_intersects_phase_keys_only() {
        let mut a: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        a.insert(crate::assert::Phase::step(0), 100);
        a.insert(crate::assert::Phase::step(1), 200);
        let mut b: std::collections::BTreeMap<crate::assert::Phase, u64> =
            std::collections::BTreeMap::new();
        b.insert(crate::assert::Phase::step(1), 100);
        b.insert(crate::assert::Phase::step(2), 50);
        let frac = a.frac_pair(&b);
        assert!(
            !frac.contains_key(&crate::assert::Phase::step(0)),
            "phase 0 absent from b — must drop from result",
        );
        assert!(
            !frac.contains_key(&crate::assert::Phase::step(2)),
            "phase 2 absent from a — must drop from result",
        );
        assert_eq!(frac.len(), 1, "only phase 1 is in the intersection");
        assert!(
            (frac[&crate::assert::Phase::step(1)] - (200.0 / 300.0)).abs() < 1e-9,
            "phase 1: 200/(200+100); got {}",
            frac[&crate::assert::Phase::step(1)],
        );
    }
}
