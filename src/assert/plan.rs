use super::*;

impl AssertResult {
    /// Empty passing result with no outcomes and default stats. Use
    /// when a scenario completed successfully with nothing interesting
    /// to report. Zero-allocation: `outcomes` is an empty `Vec` and
    /// [`Self::outcome`] folds it to [`Outcome::Pass`] via the
    /// merge identity.
    pub fn pass() -> Self {
        Self {
            outcomes: vec![],
            passes: vec![],
            stats: Default::default(),
            measurements: std::collections::BTreeMap::new(),
            info_notes: vec![],
        }
    }
    /// Pass result with a skip reason. Used when a scenario cannot run
    /// under the current topology or flag combination but is not a failure.
    /// Seeds [`Self::outcomes`] with a single [`Outcome::Skip`] carrying
    /// the reason.
    ///
    /// **Skip is not Pass**: a skipped result reports `is_pass() == false`
    /// (the outcomes vec contains a non-Pass entry). Callers that want
    /// "not a failure" gate semantics must test
    /// `r.is_pass() || r.is_skip()` rather than bare `r.is_pass()` —
    /// otherwise skipped runs count as failures.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self {
            outcomes: vec![Outcome::Skip(AssertDetail::new(DetailKind::Skip, reason))],
            ..Self::pass()
        }
    }
    /// Failing result carrying a single [`AssertDetail`]. Mirrors
    /// [`Self::pass`] / [`Self::skip`] for the failure axis so callers
    /// don't hand-roll the struct-literal shape at every diagnostic-only
    /// failure site. Seeds [`Self::outcomes`] with a single
    /// [`Outcome::Fail`] carrying the detail.
    pub fn fail(detail: AssertDetail) -> Self {
        Self {
            outcomes: vec![Outcome::Fail(detail)],
            ..Self::pass()
        }
    }
    /// Failing result carrying a single diagnostic message with
    /// [`DetailKind::Other`]. Shortcut for the common nesting
    /// `AssertResult::fail(AssertDetail::new(DetailKind::Other, msg))`
    /// at call sites where the failure is a diagnostic message and
    /// the kind is always `Other`. Named `fail_msg` rather than
    /// `fail_other` so the call site reads "failing result with a
    /// message" without leaking the [`DetailKind`] variant name into
    /// the API surface; external callers that do want a specific
    /// `kind` still reach for `AssertResult::fail` +
    /// `AssertDetail::new(kind, msg)`.
    pub fn fail_msg(msg: impl Into<String>) -> Self {
        Self::fail(AssertDetail::new(DetailKind::Other, msg))
    }

    /// Inconclusive result carrying a single [`AssertDetail`].
    /// Mirrors [`Self::pass`] / [`Self::skip`] / [`Self::fail`] for
    /// the inconclusive axis so callers don't hand-roll the struct-
    /// literal shape at sites that need to construct a fresh
    /// "couldn't evaluate" envelope (the symmetric peer of
    /// [`Self::fail`] for INSTRUMENT-derived zero-denominator
    /// gates). Seeds [`Self::outcomes`] with a single
    /// [`Outcome::Inconclusive`] carrying the detail. For mutating
    /// an existing accumulator in place, use
    /// [`Self::record_inconclusive`].
    pub fn inconclusive(detail: AssertDetail) -> Self {
        Self {
            outcomes: vec![Outcome::Inconclusive(detail)],
            ..Self::pass()
        }
    }

    /// Inconclusive result carrying a single message-only diagnostic.
    /// Shorthand for `AssertResult::inconclusive(AssertDetail::new(
    /// DetailKind::Other, msg))` — mirrors [`Self::fail_msg`] for the
    /// inconclusive axis at call sites where the operator hint is a
    /// flat string and the structured [`DetailKind`] would always be
    /// `Other`. Callers that need a specific kind still reach for
    /// `AssertResult::inconclusive` + `AssertDetail::new(kind, msg)`.
    pub fn inconclusive_msg(msg: impl Into<String>) -> Self {
        Self::inconclusive(AssertDetail::new(DetailKind::Other, msg))
    }

    /// Atomically record a Fail outcome carrying `detail`. Replaces
    /// the legacy two-step pattern `r.passed = false; r.details.push(d);`
    /// — collapses the producer-defect window where the discriminant
    /// flipped without a corresponding diagnostic. Returns `&mut Self`
    /// for chaining.
    ///
    /// See [`Self::record_inconclusive`] for ratio gates whose
    /// denominator legitimately reached zero — neither Pass nor
    /// Fail is truthful there, and Fail-coding a "couldn't evaluate"
    /// run loses signal in CI triage.
    pub fn record_fail(&mut self, detail: AssertDetail) -> &mut Self {
        self.outcomes.push(Outcome::Fail(detail));
        self
    }

    /// Atomically record a Skip outcome carrying `reason`. Replaces
    /// the legacy two-step pattern `r.skipped = true;
    /// r.details.push(AssertDetail::new(DetailKind::Skip, reason));`.
    /// Returns `&mut Self` for chaining.
    ///
    /// Boundary with [`Self::record_inconclusive`]: Skip = scenario
    /// precondition unmet (the check doesn't apply — e.g. host lacks
    /// the topology the test needs); Inconclusive = precondition met
    /// and the check applied, but the signal was absent and the gate
    /// couldn't conclude (e.g. a ratio with a zero denominator).
    /// Mis-coding an Inconclusive case as Skip drops it from the
    /// "ran but couldn't evaluate" bucket CI gates need for triage.
    pub fn record_skip(&mut self, reason: impl Into<String>) -> &mut Self {
        self.outcomes
            .push(Outcome::Skip(AssertDetail::new(DetailKind::Skip, reason)));
        self
    }

    /// Atomically record an Inconclusive outcome carrying `detail`.
    /// Signature mirrors [`Self::record_fail`] (takes the full
    /// [`AssertDetail`] so the producer's [`DetailKind`] flows into
    /// the inconclusive record for filterable diagnostics — a
    /// zero-iteration `max_migration_ratio` site emits
    /// `DetailKind::Migration`, not a flat string). Use for ratio
    /// gates whose INSTRUMENT-derived denominator (iteration count,
    /// sample count, wall-clock interval) reached zero: the gate
    /// has no signal to evaluate, neither Pass nor Fail is a
    /// truthful verdict. Returns `&mut Self` for chaining.
    ///
    /// Boundary with [`Self::record_skip`]: Inconclusive = the gate
    /// applied (preconditions met) but the signal was absent;
    /// Skip = the gate's precondition was unmet (e.g. host lacks
    /// the required topology) so the check did NOT apply. Boundary
    /// with [`Self::record_fail`]: Inconclusive = denominator is
    /// INSTRUMENT-derived (a measurement count that happened to be
    /// zero); Fail = denominator is POLICY-derived (a configured
    /// expectation that must hold — see the [`Outcome`] doc's
    /// `MemPolicy::Bind` carve-out for the canonical example).
    pub fn record_inconclusive(&mut self, detail: AssertDetail) -> &mut Self {
        self.outcomes.push(Outcome::Inconclusive(detail));
        self
    }

    /// Explicitly record a Pass marker. Rare — the zero-state
    /// `AssertResult::pass()` already folds to [`Outcome::Pass`] via
    /// the merge identity over an empty vec. Use when a test helper
    /// wants the outcome stream to carry an explicit pass record for
    /// per-check accounting (e.g. "this specific check ran and
    /// passed" vs "no check ran"). Returns `&mut Self` for chaining.
    pub fn record_pass(&mut self) -> &mut Self {
        self.outcomes.push(Outcome::Pass);
        self
    }

    /// Escape hatch: push a pre-folded [`Outcome`] onto the stream.
    /// Used by helpers that compute a verdict externally (e.g.
    /// "this branch returned `Outcome::Fail(d)`") and want to fold
    /// it into the running [`Self::outcomes`] without re-deriving
    /// the variant. Returns `&mut Self` for chaining.
    pub fn record_outcome(&mut self, outcome: Outcome) -> &mut Self {
        self.outcomes.push(outcome);
        self
    }

    /// True iff the scenario completed without failure or
    /// inconclusive verdict and actually ran (i.e. wasn't all-Skip).
    /// An empty outcomes stream (the [`Self::pass`] zero-state,
    /// which is the merge identity element) satisfies this; any
    /// stream containing at least one real Pass marker alongside no
    /// Fail / Inconclusive also satisfies it; an all-Skip stream
    /// returns false (a skipped scenario didn't pass, it didn't
    /// run); any Inconclusive returns false (a zero-denominator
    /// ratio gate didn't pass, it couldn't evaluate).
    ///
    /// Mechanically: `!self.is_fail() && !self.is_inconclusive() &&
    /// !self.is_skip()`. The three conjuncts capture "no failure
    /// recorded", "no inconclusive verdict recorded", AND "not
    /// vacuously satisfied by all-skip".
    ///
    /// Part of the `is_pass` / `is_fail` / `is_inconclusive` /
    /// `is_skip` vocabulary uniform across the verdict surfaces:
    /// [`AssertResult::is_pass`] /
    /// [`crate::test_support::SidecarResult::is_pass`] /
    /// [`Outcome::is_pass`] / `MonitorVerdict::is_pass` (in the
    /// `monitor` module, which is `pub(crate)`) / `Verdict::is_pass`
    /// (re-exported at [`crate::assert::Verdict`]) /
    /// `GauntletRow::is_pass` (in the `stats` module, which is
    /// `pub(crate)`).
    pub fn is_pass(&self) -> bool {
        !self.is_fail() && !self.is_inconclusive() && !self.is_skip()
    }

    /// True iff any recorded outcome is [`Outcome::Fail`]. Any fail
    /// in the stream dominates per `Fail > Inconclusive > Pass > Skip`
    /// precedence.
    pub fn is_fail(&self) -> bool {
        self.outcomes.iter().any(Outcome::is_fail)
    }

    /// True iff `outcomes` is non-empty AND every entry is
    /// [`Outcome::Skip`]. Empty `outcomes` is the Pass identity,
    /// NOT a vacuous Skip — `is_skip()` returns false on empty.
    pub fn is_skip(&self) -> bool {
        !self.outcomes.is_empty() && self.outcomes.iter().all(Outcome::is_skip)
    }

    /// True iff any recorded outcome is [`Outcome::Inconclusive`]
    /// AND no [`Outcome::Fail`] dominates it. Mirrors the precedence
    /// `Fail > Inconclusive > Pass > Skip`: a Fail-plus-Inconclusive
    /// stream is `is_fail() == true` and `is_inconclusive() == false`
    /// (the Fail wins; Inconclusive is dominated). Used by CI gates
    /// that want to surface "couldn't evaluate" verdicts distinctly
    /// from passes and failures.
    pub fn is_inconclusive(&self) -> bool {
        !self.is_fail() && self.outcomes.iter().any(Outcome::is_inconclusive)
    }

    /// Iterate every [`Outcome::Fail`]'s payload. Use to extract
    /// failure diagnostics for rendering or stats roll-up. Does NOT
    /// include [`Outcome::Inconclusive`] payloads —
    /// [`Self::inconclusive_details`] is the sibling iterator for
    /// those, and [`Self::into_anyhow_or_log`] bails only on Fail
    /// so folding Inconclusive into this iterator would break the
    /// "couldn't evaluate doesn't fail the run" semantic.
    pub fn failure_details(&self) -> impl Iterator<Item = &AssertDetail> {
        self.outcomes.iter().filter_map(|o| match o {
            Outcome::Fail(d) => Some(d),
            _ => None,
        })
    }

    /// Iterate every [`Outcome::Skip`]'s payload. Use to extract
    /// skip reasons when triaging "scenario didn't run" outcomes.
    /// The `_details` suffix mirrors [`Self::failure_details`] /
    /// [`Self::inconclusive_details`] — all three yield
    /// `&AssertDetail` payloads.
    pub fn skip_details(&self) -> impl Iterator<Item = &AssertDetail> {
        self.outcomes.iter().filter_map(|o| match o {
            Outcome::Skip(d) => Some(d),
            _ => None,
        })
    }

    /// Iterate every [`Outcome::Inconclusive`]'s payload. Use to
    /// extract diagnostic context for zero-denominator ratio gates
    /// or other "couldn't evaluate" verdicts when triaging.
    /// Symmetric with [`Self::failure_details`] /
    /// [`Self::skip_details`]; not folded into either so the
    /// failure / skip / inconclusive surfaces remain separately
    /// addressable. The `_details` suffix mirrors
    /// [`Self::failure_details`] — both yield `&AssertDetail`
    /// payloads that drive triage of material verdicts.
    pub fn inconclusive_details(&self) -> impl Iterator<Item = &AssertDetail> {
        self.outcomes.iter().filter_map(|o| match o {
            Outcome::Inconclusive(d) => Some(d),
            _ => None,
        })
    }

    /// Terminal post_vm-callback helper: route every
    /// [`Self::info_notes`] entry through `tracing::info!` (so
    /// `--nocapture` + `RUST_LOG=ktstr=info` users see them, but
    /// default-noise-level runs stay quiet) and bail on any
    /// accumulated failure OR inconclusive verdict. Returns `Ok(())`
    /// only on the pass / pure-skip path — idiomatic post_vm usage
    /// chains `?` to propagate the verdict or continue.
    ///
    /// # Failure behavior
    ///
    /// Per the precedence `Fail > Inconclusive > Pass > Skip`, Fail
    /// dominates Inconclusive: when any [`Outcome::Fail`] is recorded
    /// the helper bails with the failure narrative, regardless of
    /// any sibling Inconclusive outcomes. Every entry from
    /// [`Self::failure_details`] is concatenated into the returned
    /// `anyhow::Error` message — all failures surface, the helper
    /// does NOT drop N-1 details when multiple claims failed.
    ///
    /// # Inconclusive behavior
    ///
    /// When no failure is present but at least one Inconclusive is
    /// recorded (a zero-denominator ratio gate that couldn't
    /// evaluate), the helper bails with a distinct preamble
    /// `"N inconclusive verdict(s):"` carrying every
    /// [`Self::inconclusive_details`] payload. This prevents the
    /// silent-pass class of bug where a CI gate keying off
    /// `into_anyhow_or_log().is_ok()` would treat an Inconclusive
    /// run as green (the `is_pass()`-keyed invariant fails on
    /// Inconclusive, so the bail surface must match). The
    /// `"inconclusive verdict(s)"` preamble distinguishes the bail
    /// narrative from the failure preamble `"N assertion failures:"`
    /// so an operator triaging the log can immediately tell whether
    /// the run failed claims or merely lacked signal to evaluate them.
    ///
    /// # Note ordering
    ///
    /// Info notes are logged BEFORE the verdict check fires, so on
    /// a failed or inconclusive run the operator sees the
    /// diagnostic observations that led to the verdict ALONGSIDE
    /// the bail message in their log feed (rather than the bail
    /// terminating before the notes surface).
    ///
    /// # `tracing` vs `println!`
    ///
    /// Notes are emitted via `tracing::info!` with target
    /// `"ktstr::assert"` — matches the comparator pass-arm logging
    /// convention at [`crate::assert::claim`]. Operators set
    /// `RUST_LOG=ktstr::assert=info` (or broader) to surface them;
    /// `println!` would bypass the tracing subscriber and bake in
    /// stdout-only visibility.
    ///
    /// # Composability
    ///
    /// [`crate::assert::Verdict::into_anyhow_or_log`] is a thin
    /// wrapper for callers that hold a `Verdict` directly.
    pub fn into_anyhow_or_log(self) -> anyhow::Result<()> {
        for note in &self.info_notes {
            tracing::info!(target: "ktstr::assert", "{}", note.message);
        }
        let failures: Vec<String> = self.failure_details().map(|d| d.message.clone()).collect();
        if !failures.is_empty() {
            let combined = if failures.len() == 1 {
                failures.into_iter().next().unwrap()
            } else {
                let mut out = format!("{} assertion failures:\n", failures.len());
                for (i, msg) in failures.iter().enumerate() {
                    out.push_str(&format!("  {}. {}\n", i + 1, msg));
                }
                out.trim_end().to_string()
            };
            anyhow::bail!("{}", combined);
        }
        let inconclusives: Vec<String> = self
            .inconclusive_details()
            .map(|d| d.message.clone())
            .collect();
        if !inconclusives.is_empty() {
            let combined = if inconclusives.len() == 1 {
                format!("1 inconclusive verdict: {}", inconclusives[0])
            } else {
                let mut out = format!("{} inconclusive verdicts:\n", inconclusives.len());
                for (i, msg) in inconclusives.iter().enumerate() {
                    out.push_str(&format!("  {}. {}\n", i + 1, msg));
                }
                out.trim_end().to_string()
            };
            anyhow::bail!("{}", combined);
        }
        Ok(())
    }
    /// Append an informational annotation to [`Self::info_notes`].
    /// Does NOT alter the terminal verdict ([`Self::outcome`] is unaffected) — a note
    /// is context, not a verdict. Use to surface observed values
    /// alongside a passing or failing result so the sidecar carries
    /// the diagnostic context an operator needs without forcing every
    /// test to hand-format a `format!` and push onto `details`
    /// directly. Notes live on the structurally-separate
    /// [`Self::info_notes`] field — sidecar consumers iterating
    /// `details` see only failures, eliminating the prior
    /// "forgot to filter `kind == Note`" silent-miscount class of bug.
    pub fn note(&mut self, msg: impl Into<String>) -> &mut Self {
        self.info_notes.push(InfoNote::new(msg));
        self
    }
    /// Drop the best-effort RAW per-phase per-cgroup sample vectors
    /// (`wake_latencies_ns`, `run_delays_ns`, `off_cpu_pcts`) from every phase
    /// bucket, keeping the reduced counters (`num_workers`, `total_*`,
    /// `wake_sample_total`, the NUMA counts, the coupled gap) and the rest of the
    /// verdict. Returns the number of samples dropped.
    ///
    /// Graceful-degradation lever for [`crate::vmm::guest_comms::send_test_result`]:
    /// This is the first path to put raw per-cgroup sample vectors on the
    /// size-limited guest bulk port, and a scenario with many step-local cgroups
    /// over many steps accumulates one (non-merging) carrier per cgroup-step, so
    /// the AGGREGATE can exceed the frame even though each carrier is reservoir-
    /// capped. Rather than replace the whole AssertResult with a synthetic FAIL
    /// (a PASS→FAIL flip + total telemetry loss), the sender drops only these
    /// best-effort sample pools — the verdict, outcomes, and every scalar/counter
    /// (including `wake_sample_total`, the NUMA counts, the coupled gap, and the
    /// counter-derived metrics) survive. The wake p99 / median / CV (from
    /// `wake_latencies_ns`) and mean / worst run-delay (from `run_delays_ns`)
    /// re-pools DEGRADE rather than vanish: with their carrier samples gone
    /// every carrier is empty, so `populate_run_distribution_metrics` folds the
    /// surviving per-cgroup [`CgroupStats`] reductions worst-wins (the pre-Item-7
    /// cross-cgroup max — a worst-cgroup proxy, not the pooled distribution).
    /// `off_cpu_pcts` has no run-level re-pool consumer (worst_spread / off-CPU%
    /// come from the typed `CgroupStats` merge fold, not the carrier — off-CPU%
    /// is intrinsically per-cgroup, out of the run-level re-pool), so dropping it
    /// loses only the per-phase off-CPU% render, not any run-level value.
    pub(crate) fn strip_phase_cgroup_samples(&mut self) -> usize {
        let mut dropped = 0usize;
        for bucket in &mut self.stats.phases {
            for pc in bucket.per_cgroup.values_mut() {
                // Mark stripped only when this carrier ACTUALLY had samples to
                // drop, so the render distinguishes "stripped for size" from a
                // carrier that genuinely measured nothing (already-empty vecs
                // stay not-stripped).
                let had_samples = !pc.wake_latencies_ns.is_empty()
                    || !pc.run_delays_ns.is_empty()
                    || !pc.off_cpu_pcts.is_empty();
                dropped +=
                    pc.wake_latencies_ns.len() + pc.run_delays_ns.len() + pc.off_cpu_pcts.len();
                pc.wake_latencies_ns = Vec::new();
                pc.run_delays_ns = Vec::new();
                pc.off_cpu_pcts = Vec::new();
                if had_samples {
                    pc.stripped = true;
                }
            }
        }
        dropped
    }
    /// Builder-style sibling of [`Self::note`] returning the
    /// owned result so a scenario can chain
    /// `AssertResult::pass().with_note("max_wchar=12345")` at
    /// the return site. Equivalent to calling
    /// [`Self::note`] on a mutable binding.
    pub fn with_note(mut self, msg: impl Into<String>) -> Self {
        self.note(msg);
        self
    }
    /// Terminal verdict as a single [`Outcome`] value, aligned with
    /// [`Self::is_pass`] / [`Self::is_fail`] /
    /// [`Self::is_inconclusive`] / [`Self::is_skip`]:
    ///
    /// - any [`Outcome::Fail`] in the stream → [`Outcome::Fail`]
    ///   carrying the first Fail's payload (the LEFT operand wins
    ///   per [`Outcome::merge`]'s payload-tie semantics).
    /// - else any [`Outcome::Inconclusive`] in the stream →
    ///   [`Outcome::Inconclusive`] carrying the first
    ///   Inconclusive's payload (the gate ran but couldn't
    ///   evaluate; per `Fail > Inconclusive > Pass > Skip` this
    ///   sits below Fail but above Pass and Skip).
    /// - else non-empty all-[`Outcome::Skip`] → [`Outcome::Skip`]
    ///   carrying the first Skip's payload (a scenario whose only
    ///   recorded gates were skips didn't run — the terminal
    ///   verdict is Skip, not Pass).
    /// - else (empty stream OR at least one explicit
    ///   [`Outcome::Pass`] marker alongside no Fail / Inconclusive)
    ///   → [`Outcome::Pass`] (the zero-allocation pass identity
    ///   also lands here).
    ///
    /// Diverges from the naive `Outcome::merge` fold over the
    /// identity element [`Outcome::Pass`]: that fold would treat
    /// `[Skip(d)]` as `Pass.merge(Skip(d)) = Pass` per the
    /// `Fail > Inconclusive > Pass > Skip` precedence, contradicting
    /// the all-Skip branch of [`Self::is_skip`]. This accessor
    /// encodes the "empty Pass identity" / "real Pass beats Skip" /
    /// "all-Skip is Skip terminal" distinctions the boolean
    /// accessors enforce.
    ///
    /// Use [`Self::outcome_ref`] when the caller only needs to
    /// inspect the verdict shape/payload without taking ownership —
    /// avoids the per-call `AssertDetail::clone` this accessor
    /// performs on the `Skip` / `Fail` arms.
    pub fn outcome(&self) -> Outcome {
        // Delegates to [`Self::outcome_ref`] + [`OutcomeRef::to_owned`]
        // so the fold rule (Fail > Inconclusive > Pass > Skip with
        // the empty-vec / all-Skip / mixed-Pass-plus-Skip branch
        // resolution) lives in ONE place. A future change to the
        // fold lands at `outcome_ref` and propagates here for free;
        // the drift-guard test
        // `assert_result_outcome_ref_matches_owned_outcome_shape`
        // in `tests_assert.rs` was originally written to catch
        // divergence between two parallel implementations — after
        // this delegation it instead catches a single-source bug
        // (e.g. fold-rule + `as_ref`/`to_owned` mapping drift) but
        // remains load-bearing.
        self.outcome_ref().to_owned()
    }

    /// Borrow the terminal verdict as an [`OutcomeRef`]. Same fold
    /// semantics as [`Self::outcome`] —
    /// `Fail > Inconclusive > Pass > Skip` precedence, empty-vec /
    /// non-empty-all-Skip / mixed-Pass-plus-Skip branches all match
    /// — but the `Skip(_)` / `Inconclusive(_)` / `Fail(_)` arms
    /// borrow the source [`AssertDetail`] from `self.outcomes`
    /// instead of cloning. Use when the caller holds the source
    /// `AssertResult` and wants the verdict payload without the
    /// per-call clone (formatter / sidecar emit / debug-render paths).
    ///
    /// Drift guard: `assert_result_outcome_ref_matches_owned_outcome_shape`
    /// in `tests_assert.rs` pins the lockstep with [`Self::outcome`];
    /// any divergence (e.g. a future refactor that adds a new
    /// terminal arm here but forgets the owned accessor, or vice
    /// versa) trips the test.
    pub fn outcome_ref(&self) -> OutcomeRef<'_> {
        if let Some(d) = self.failure_details().next() {
            OutcomeRef::Fail(d)
        } else if let Some(d) = self.inconclusive_details().next() {
            OutcomeRef::Inconclusive(d)
        } else if let Some(d) = self.skip_details().next() {
            if self.outcomes.iter().all(Outcome::is_skip) {
                OutcomeRef::Skip(d)
            } else {
                OutcomeRef::Pass
            }
        } else {
            OutcomeRef::Pass
        }
    }
    /// Fold `other` into `self`. The four parallel vecs/maps —
    /// [`Self::outcomes`], [`Self::passes`], [`Self::info_notes`],
    /// [`Self::measurements`] — all extend with `other`'s contents
    /// (the three vecs concatenate; `measurements` is a `BTreeMap`
    /// merged with plain last-write-wins on key collision, i.e.
    /// `other`'s value overwrites `self`'s for shared keys).
    /// Aggregate `stats` adopt the worst-case value per dimension
    /// so the merged result represents the union of all checks
    /// applied. The polarity-aware per-key min/max selection for
    /// extensible scheduler metrics is a separate mechanism that
    /// applies inside `stats.ext_metrics` only — see the loop at
    /// the bottom of this body for the polarity registry path; the
    /// result-level `measurements` map deliberately does not consult
    /// the registry (it is a producer-attached typed annotation map,
    /// not a roll-up aggregation surface).
    ///
    /// Terminal-verdict semantics fall out automatically per the
    /// precedence `Fail > Inconclusive > Pass > Skip`: appending
    /// `other.outcomes` keeps every Fail in the stream so
    /// [`Self::outcome`]'s fold surfaces them; absent any Fail, any
    /// Inconclusive in either side dominates Pass/Skip so a
    /// zero-denominator gate in one branch survives the fold;
    /// Skip survives only when both inputs were Skip-only because a
    /// Pass or Inconclusive entry in either side beats Skip.
    pub fn merge(&mut self, mut other: AssertResult) {
        /// Lowest-non-zero fold: `*self_field` becomes `other_field`
        /// when `other_field` is strictly positive AND either
        /// `*self_field` is zero (uninitialized sentinel) or
        /// `other_field` is strictly smaller than `*self_field`.
        ///
        /// This is NOT `f64::min` — a plain min would let an
        /// unreported cgroup (`0.0` sentinel) clobber a real
        /// reading from another cgroup, treating "no data yet" as
        /// "worst possible." The accumulator pattern
        /// `AssertResult::pass().merge(real)` starts with 0.0 from
        /// `Default`, and a plain min would destroy any positive
        /// reading folded in — so every lowest-is-worse rollup
        /// uses this fold to treat 0.0 as a sentinel rather than a
        /// real measurement.
        fn fold_lowest_nonzero(self_field: &mut f64, other_field: f64) {
            if other_field > 0.0 && (*self_field == 0.0 || other_field < *self_field) {
                *self_field = other_field;
            }
        }

        self.outcomes.extend(other.outcomes);
        self.passes.extend(other.passes);
        self.info_notes.extend(other.info_notes);
        let s = &mut self.stats;
        let o = &other.stats;
        s.total_workers += o.total_workers;
        s.total_cpus += o.total_cpus;
        s.total_migrations += o.total_migrations;
        s.total_iterations += o.total_iterations;
        s.worst_spread = s.worst_spread.max(o.worst_spread);
        s.worst_migration_ratio = s.worst_migration_ratio.max(o.worst_migration_ratio);
        // worst_p99/median/cv wake-latency + mean/worst run-delay are no
        // longer folded here: they are `MetricKind::Distribution`, re-pooled
        // post-merge from the per-cgroup raw samples by
        // `populate_run_distribution_metrics` (the pooled-distribution
        // percentile, not a max of per-cgroup reductions).
        s.worst_cross_node_migration_ratio = s
            .worst_cross_node_migration_ratio
            .max(o.worst_cross_node_migration_ratio);
        // worst_wake_latency_tail_ratio is no longer folded here: it is
        // `MetricKind::WakeLatencyTailRatio`, re-selected post-merge by
        // `populate_run_distribution_metrics` (the max over the merged
        // `stats.cgroups` per-cgroup p99/median ratios, floor-gated).
        // worst_iterations_per_worker / worst_iterations_per_cpu_sec are no
        // longer folded here: they are `MetricKind::WorstLowest`, re-selected
        // post-merge by `populate_run_distribution_metrics` (the lowest-wins,
        // None-aware min over the merged `stats.cgroups` counters — the same
        // starvation semantic the deleted `fold_lowest_some` carried).
        // Coupled fields: `worst_gap_cpu` must come from the same
        // cgroup that posted the new worst `worst_gap_ms`.
        if o.worst_gap_ms > s.worst_gap_ms {
            s.worst_gap_ms = o.worst_gap_ms;
            s.worst_gap_cpu = o.worst_gap_cpu;
        }
        // NUMA page locality: lowest-non-zero fold — see
        // `fold_lowest_nonzero` above for the sentinel convention.
        fold_lowest_nonzero(&mut s.worst_page_locality, o.worst_page_locality);
        // Merge extensible metrics: take worst per key according to
        // each metric's polarity in the MetricDef registry. For
        // `higher_is_worse: true` the worst is max; for
        // `higher_is_worse: false` the worst is min.
        //
        // Unregistered metric names fall through to
        // [`crate::stats::infer_higher_is_worse`], which derives the
        // polarity from name substrings (e.g. `*_iops`,
        // `*_latency_us`). Without the inference, a payload-author
        // throughput metric — e.g. `jobs.0.read.iops` from
        // `OutputFormat::LlmExtract` — would fold with `max`,
        // keeping the BETTER (higher) value across cgroups and
        // masking a cgroup that fell behind. The inference returns a
        // higher-is-worse default when no token matches, so genuinely
        // unknown names still surface their max (the safer side of
        // the regression-vs-improvement misclassification).
        //
        // `or_insert(*v)` rather than `or_insert(0.0)`: the old sentinel
        // clobbered real-but-small values for min-polarity metrics on
        // first merge, making the subsequent min comparison meaningless.
        for (k, v) in &other.stats.ext_metrics {
            let higher_is_worse = crate::stats::metric_def(k)
                .map(|m| m.higher_is_worse())
                .unwrap_or_else(|| crate::stats::infer_higher_is_worse(k));
            let entry = self.stats.ext_metrics.entry(k.clone()).or_insert(*v);
            *entry = if higher_is_worse {
                entry.max(*v)
            } else {
                entry.min(*v)
            };
        }
        // Merge `phases` per `step_index`. For matched phases on
        // both sides, fold per-metric using `MetricKind::merge_kind`
        // to pick the commutative or NonCommutative path. Unpaired
        // phases (one side only) carry through verbatim — never
        // silently dropped, per the no-silent-drops contract. The
        // result is sorted by `step_index` for a deterministic
        // observable order regardless of merge-arrival order.
        //
        // Move `other.stats.phases` out before the per-cgroup
        // extend below (which moves the sibling `cgroups` field).
        // After the take, `other.stats.phases` is an empty Vec —
        // never read again because the rest of this fn references
        // only `other.stats.cgroups` and `other.measurements`.
        let other_phases = std::mem::take(&mut other.stats.phases);
        if !self.stats.phases.is_empty() || !other_phases.is_empty() {
            let other_len = other_phases.len();
            // No-silent-drops: phase buckets have unique step_index
            // (build_phase_buckets_with_stimulus / collect_step emit one per
            // step_index), but fold same-step_index duplicates via merge rather
            // than a last-wins collect so a future producer that violated the
            // invariant DEGRADES to a merge, never a silent release-mode drop. The
            // debug_assert still trips loudly in test/debug.
            let mut other_by_idx: std::collections::BTreeMap<u16, PhaseBucket> =
                std::collections::BTreeMap::new();
            for b in other_phases {
                match other_by_idx.remove(&b.step_index) {
                    Some(existing) => {
                        other_by_idx.insert(b.step_index, merge_matched_phase_buckets(existing, b));
                    }
                    None => {
                        other_by_idx.insert(b.step_index, b);
                    }
                }
            }
            debug_assert_eq!(
                other_by_idx.len(),
                other_len,
                "merged phase buckets must have unique step_index; a collision merged (not dropped)",
            );
            let self_buckets = std::mem::take(&mut self.stats.phases);
            let mut merged: Vec<PhaseBucket> =
                Vec::with_capacity(self_buckets.len() + other_by_idx.len());
            for s_bucket in self_buckets {
                if let Some(o_bucket) = other_by_idx.remove(&s_bucket.step_index) {
                    merged.push(merge_matched_phase_buckets(s_bucket, o_bucket));
                } else {
                    merged.push(s_bucket);
                }
            }
            merged.extend(other_by_idx.into_values());
            merged.sort_by_key(|b| b.step_index);
            self.stats.phases = merged;
        }

        // Append per-cgroup stats last: moving `other.stats.cgroups`
        // here consumes `other.stats`, so every scalar/map access
        // above goes through the `&other.stats` reference first.
        self.stats.cgroups.extend(other.stats.cgroups);

        // Fold structured measurements. Keys from `other` overwrite
        // existing keys from `self` because the merge protocol treats
        // the right-hand side as a more recent observation; a
        // duplicate-key write is a producer bug (two cgroups
        // measuring the same global metric) but the "later wins"
        // policy keeps the result deterministic for tests pinning
        // merge order. Producers that need additive accumulation
        // should use `stats.ext_metrics` (which has explicit polarity
        // semantics) rather than `measurements`.
        for (k, v) in other.measurements {
            self.measurements.insert(k, v);
        }
    }

    /// Attach a structured `(key, value)` measurement to the result.
    /// Writes into [`Self::measurements`] without altering
    /// the terminal verdict ([`Self::outcome`]) —
    /// pure context for stats tooling.
    ///
    /// Distinct from [`Self::note`]: `note` carries a free-form
    /// `String` for operator triage; `note_value` carries a typed
    /// `(key, NoteValue)` pair for programmatic consumption (sidecar
    /// parsers, `stats compare` regression dashboards). Producers
    /// commonly call BOTH — they occupy independent buffers and
    /// neither overwrites the other.
    ///
    /// Key collision policy: a second write with the same `key`
    /// overwrites the first. The intended call site shape is "one
    /// producer per key" (one site computes `max_wchar`, one site
    /// computes `psi_some_total_usec`); accidental key collision
    /// indicates a producer bug. The test
    /// `note_value_overwrites_on_duplicate_key` pins this last-
    /// write-wins semantics.
    ///
    /// ```
    /// # use ktstr::assert::{AssertResult, NoteValue};
    /// let mut r = AssertResult::pass();
    /// r.note_value("max_wchar", 12345i64);
    /// r.note_value("psi_available", true);
    /// assert_eq!(r.measurements["max_wchar"], NoteValue::Int(12345));
    /// assert_eq!(r.measurements["psi_available"], NoteValue::Bool(true));
    /// ```
    pub fn note_value(&mut self, key: impl Into<String>, value: impl Into<NoteValue>) -> &mut Self {
        self.measurements.insert(key.into(), value.into());
        self
    }

    /// Builder-style sibling of [`Self::note_value`] returning the
    /// owned result so a scenario can chain
    /// `AssertResult::pass().with_note_value("max_wchar", 12345u64)`
    /// at the return site. Equivalent to calling [`Self::note_value`]
    /// on a mutable binding. Mirrors [`Self::with_note`].
    pub fn with_note_value(mut self, key: impl Into<String>, value: impl Into<NoteValue>) -> Self {
        self.note_value(key, value);
        self
    }

    /// Fold a sequence of [`AssertResult`]s with OR semantics: the
    /// returned result passes iff at least one branch passes. Use
    /// when a test author expresses "either of these two checks
    /// suffices" — a kernel-version-fork case where one path is
    /// expected on 6.16 and another on 7.1, or a topology probe
    /// where any of several detection methods landing is enough.
    ///
    /// Outcomes:
    /// - **At least one branch passes**: returned result is passing.
    ///   `info_notes` carries the union of every passing branch's
    ///   info_notes, each prefix-stamped with `any_of[<branch-idx>]:`
    ///   so an operator can attribute every note to the emitting
    ///   branch. The synthesized "any_of: branch N satisfied the
    ///   disjunction" arbiter annotation is appended last, bare
    ///   (it's not from any branch — it IS the disposition).
    ///   Failed-branch details and info_notes are dropped (they would
    ///   only confuse the operator with messages from the not-taken
    ///   paths). `stats` adopts the first passing branch's `stats`.
    ///   `measurements` union all passing branches' measurements
    ///   (last write wins on key collision, matching `merge`).
    ///   `outcomes` follows the first passing branch (typically
    ///   empty per the Pass identity).
    /// - **No branch passes; at least one fails**: returned result
    ///   is failing. Every branch's recorded outcomes are re-emitted
    ///   with each payload's message prefixed by
    ///   `"any_of[<branch-idx>]: "` so an operator can identify
    ///   which branch produced which outcome. `stats` and
    ///   `measurements` adopt the FIRST branch's values (an
    ///   arbitrary choice but deterministic). A synthesized summary
    ///   record is appended last carrying the per-disposition
    ///   counts `(F failed, I inconclusive, S skipped of N branches)`.
    /// - **No branch passes or fails; at least one is Inconclusive**:
    ///   returned result is Inconclusive (the disjunction itself
    ///   could not be evaluated — every branch either was inconclusive
    ///   or skipped, and per the lattice
    ///   `Fail > Inconclusive > Pass > Skip` Inconclusive dominates
    ///   Skip). Same re-emission + summary shape as the fail case,
    ///   but the synthesized record is [`Outcome::Inconclusive`].
    ///   Critical: without this arm, all-zero-denominator branches
    ///   would silently MISCLASSIFY as Fail, defeating the
    ///   Inconclusive primitive's purpose of preserving "couldn't
    ///   evaluate" signal.
    /// - **All branches skipped**: returned result is Skip. Same
    ///   re-emission + summary shape, with [`Outcome::Skip`] as the
    ///   synthesized record (every alternative check's precondition
    ///   was unmet — the disjunction itself didn't run).
    /// - **Empty input**: returned result is failing with a single
    ///   Fail outcome explaining the empty `any_of`. An empty
    ///   disjunction is logically false; this surfaces a producer
    ///   bug as a nameable failure rather than a vacuous pass.
    ///
    /// Doc: a trivial two-branch test with the second branch passing
    /// and the first branch failing — pinning that the verdict
    /// chooses the passer.
    ///
    /// ```
    /// # use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
    /// let r = AssertResult::any_of([
    ///     {
    ///         let mut a = AssertResult::pass();
    ///         a.record_fail(AssertDetail::new(DetailKind::Other, "branch 0 boom"));
    ///         a
    ///     },
    ///     AssertResult::pass(),
    /// ]);
    /// assert!(r.is_pass());
    /// ```
    pub fn any_of(branches: impl IntoIterator<Item = AssertResult>) -> AssertResult {
        let branches: Vec<AssertResult> = branches.into_iter().collect();
        if branches.is_empty() {
            return AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                "any_of: empty branch list — a disjunction of zero alternatives is logically false",
            ));
        }

        let first_pass_idx = branches.iter().position(|b| b.is_pass());
        if let Some(idx) = first_pass_idx {
            // At least one branch passes. Take the first passing
            // branch as the "chosen" narrative: keep its stats /
            // outcomes, union measurements AND info_notes across
            // every passing branch (failed branches' content is
            // dropped — they would only confuse the operator with
            // messages from not-taken paths), and prefix every
            // surviving info_note with `any_of[<branch-idx>]:` for
            // operator-visible provenance.
            let mut chosen: Option<AssertResult> = None;
            let mut union_measurements: std::collections::BTreeMap<String, NoteValue> =
                std::collections::BTreeMap::new();
            let mut union_info_notes: Vec<InfoNote> = Vec::new();
            for (orig_idx, b) in branches.into_iter().enumerate() {
                if orig_idx == idx {
                    let mut b = b;
                    let pre_notes = std::mem::take(&mut b.info_notes);
                    let pre_meas = std::mem::take(&mut b.measurements);
                    chosen = Some(b);
                    for n in pre_notes {
                        union_info_notes
                            .push(InfoNote::new(format!("any_of[{orig_idx}]: {}", n.message)));
                    }
                    for (k, v) in pre_meas {
                        union_measurements.insert(k, v);
                    }
                } else if b.is_pass() {
                    for n in b.info_notes {
                        union_info_notes
                            .push(InfoNote::new(format!("any_of[{orig_idx}]: {}", n.message)));
                    }
                    for (k, v) in b.measurements {
                        union_measurements.insert(k, v);
                    }
                }
                // Failed/skipped non-chosen branches: contents are
                // dropped (would confuse the operator with not-taken
                // path messages).
            }
            let mut chosen = chosen.expect("first_pass_idx matched a branch");
            chosen.measurements = union_measurements;
            chosen.info_notes = union_info_notes;
            chosen.info_notes.push(InfoNote::new(format!(
                "any_of: branch {idx} satisfied the disjunction"
            )));
            chosen
        } else {
            // No branch passes. Re-emit every branch's outcome
            // stream with branch-index prefixes; adopt the first
            // branch's stats / measurements / passes (deterministic
            // but arbitrary). Variants keep their kind discriminant
            // so the operator narrative differentiates "this branch
            // failed" from "this branch was inconclusive" from
            // "this branch skipped". The synthesized summary record
            // appended last carries the per-disposition counts AND
            // its discriminant follows the precedence
            // `Fail > Inconclusive > Pass > Skip`:
            //   - any Fail branch → synth Fail
            //   - else any Inconclusive → synth Inconclusive
            //   - else (all Skip) → synth Skip
            // The Inconclusive arm is load-bearing: without it, a
            // disjunction of zero-denominator branches would silently
            // MISCLASSIFY as Fail (or, before that arm existed at
            // all, surface as a misleading "all branches failed"
            // verdict on data that simply couldn't be evaluated).
            let total_branches = branches.len();
            let (n_fail, n_inc, n_skip) =
                branches
                    .iter()
                    .fold((0usize, 0usize, 0usize), |(f, i, s), b| {
                        if b.is_fail() {
                            (f + 1, i, s)
                        } else if b.is_inconclusive() {
                            (f, i + 1, s)
                        } else if b.is_skip() {
                            (f, i, s + 1)
                        } else {
                            (f, i, s)
                        }
                    });
            let mut iter = branches.into_iter().enumerate();
            let (_, first) = iter.next().expect("non-empty checked above");
            let mut acc = AssertResult {
                outcomes: Vec::new(),
                passes: first.passes,
                stats: first.stats,
                measurements: first.measurements,
                info_notes: Vec::new(),
            };
            fn reemit_with_prefix(
                acc: &mut AssertResult,
                idx: usize,
                outcomes: Vec<Outcome>,
                info_notes: Vec<InfoNote>,
            ) {
                for o in outcomes {
                    match o {
                        Outcome::Pass => acc.outcomes.push(Outcome::Pass),
                        Outcome::Fail(d) => acc.outcomes.push(Outcome::Fail(AssertDetail::new(
                            d.kind,
                            format!("any_of[{idx}]: {}", d.message),
                        ))),
                        Outcome::Inconclusive(d) => acc.outcomes.push(Outcome::Inconclusive(
                            AssertDetail::new(d.kind, format!("any_of[{idx}]: {}", d.message)),
                        )),
                        Outcome::Skip(d) => acc.outcomes.push(Outcome::Skip(AssertDetail::new(
                            d.kind,
                            format!("any_of[{idx}]: {}", d.message),
                        ))),
                    }
                }
                for n in info_notes {
                    acc.info_notes
                        .push(InfoNote::new(format!("any_of[{idx}]: {}", n.message)));
                }
            }
            reemit_with_prefix(&mut acc, 0, first.outcomes, first.info_notes);
            for (idx, b) in iter {
                reemit_with_prefix(&mut acc, idx, b.outcomes, b.info_notes);
            }
            let summary = format!(
                "any_of: no branch passed ({n_fail} failed, {n_inc} inconclusive, {n_skip} skipped of {total_branches} branches)"
            );
            let synth = if n_fail > 0 {
                Outcome::Fail(AssertDetail::new(DetailKind::Other, summary))
            } else if n_inc > 0 {
                Outcome::Inconclusive(AssertDetail::new(DetailKind::Other, summary))
            } else {
                Outcome::Skip(AssertDetail::new(DetailKind::Skip, summary))
            };
            acc.outcomes.push(synth);
            acc
        }
    }

    /// Fold a sequence of [`AssertResult`]s with AND semantics:
    /// equivalent to `branches.into_iter().fold(pass(),
    /// |acc, b| { acc.merge(b); acc })`. Returns a passing result iff
    /// every branch passes.
    ///
    /// Distinct from [`Self::merge`] in API shape only: `merge`
    /// folds one external result into an existing accumulator;
    /// `all_of` folds an iterator of branches into a fresh result.
    /// Same semantics for `outcomes` (concatenated; Fail dominates
    /// per `Outcome::merge` precedence), `stats` (worst-per-dimension),
    /// `measurements` (union with last-write-wins). An empty input
    /// yields the passing identity (`AssertResult::pass()`) — the
    /// AND of an empty set is logically true, mirroring
    /// `Iterator::all`.
    ///
    /// Use when the test reads more naturally as "every check
    /// must hold" than as a merge chain — e.g. when the checks
    /// are dynamically generated from a slice and the call site
    /// would otherwise need an explicit `for` loop with `merge`.
    ///
    /// ```
    /// # use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
    /// let r = AssertResult::all_of([
    ///     AssertResult::pass(),
    ///     AssertResult::pass(),
    /// ]);
    /// assert!(r.is_pass());
    ///
    /// let r = AssertResult::all_of([
    ///     AssertResult::pass(),
    ///     AssertResult::fail(AssertDetail::new(DetailKind::Other, "boom")),
    /// ]);
    /// assert!(r.is_fail());
    /// ```
    pub fn all_of(branches: impl IntoIterator<Item = AssertResult>) -> AssertResult {
        let mut acc = AssertResult::pass();
        for b in branches {
            acc.merge(b);
        }
        acc
    }
}

/// Worker-side assertion plan (crate-internal). Specifies which checks
/// to run on worker reports after collection.
///
/// External users should use [`Assert`] and its `assert_cgroup()` method
/// instead.
#[derive(Clone, Debug, Default)]
pub(crate) struct AssertPlan {
    pub(crate) not_starved: bool,
    pub(crate) isolation: bool,
    pub(crate) max_gap_ms: Option<u64>,
    pub(crate) max_spread_pct: Option<f64>,
    pub(crate) max_throughput_cv: Option<f64>,
    pub(crate) min_work_rate: Option<f64>,
    pub(crate) max_p99_wake_latency_ns: Option<u64>,
    pub(crate) max_wake_latency_cv: Option<f64>,
    pub(crate) min_iteration_rate: Option<f64>,
    pub(crate) max_migration_ratio: Option<f64>,
    pub(crate) min_page_locality: Option<f64>,
    pub(crate) max_cross_node_migration_ratio: Option<f64>,
    pub(crate) max_slow_tier_ratio: Option<f64>,
}

impl AssertPlan {
    /// Construct an empty `AssertPlan` — equivalent to `AssertPlan::default()`.
    /// Kept as an alias for the existing test-suite call style.
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Run all configured checks against one cgroup's reports.
    ///
    /// `cpuset` is the expected CPU set for isolation checks. Pass `None`
    /// when there is no cpuset constraint (isolation check is skipped).
    ///
    /// `numa_nodes` is the NUMA node IDs covered by the cpuset (derived
    /// via `TestTopology::numa_nodes_for_cpuset`). Used for page locality
    /// and slow-tier ratio checks. Pass `None` when NUMA checks are not
    /// applicable.
    pub(crate) fn assert_cgroup(
        &self,
        reports: &[WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
        numa_nodes: Option<&BTreeSet<usize>>,
    ) -> AssertResult {
        // Per-cgroup TELEMETRY is built unconditionally — it is pure
        // measurement and must not be gated behind whether any worker
        // check is configured (see [`cgroup_stats`]). The opt-in checks
        // below only append fail/inconclusive outcomes onto `r`.
        let mut cg = cgroup_stats(reports);
        // NUMA page-locality telemetry: the expected-node set lives in
        // `numa_nodes` (cgroup_stats has no NUMA context), so compute
        // locality here whenever it is supplied — independent of whether
        // the min_page_locality CHECK is set — and store it on the
        // telemetry. The (locality, total, local) triple is reused by that
        // check below so the value is computed once.
        let page_locality = numa_nodes.map(|nodes| {
            let mut total: u64 = 0;
            let mut local: u64 = 0;
            for w in reports {
                for (&node, &count) in &w.numa_pages {
                    total += count;
                    if nodes.contains(&node) {
                        local += count;
                    }
                }
            }
            let locality = if total > 0 {
                local as f64 / total as f64
            } else {
                0.0
            };
            (locality, total, local)
        });
        if let Some((locality, _, _)) = page_locality {
            cg.page_locality = locality;
        }
        let mut r = AssertResult::pass();
        r.stats = scenario_stats_for_cgroup(&cg);

        // Opt-in CHECKS in fixed order — each only appends fail /
        // inconclusive outcomes onto `r`; the order they run is the order
        // outcomes land in `r.outcomes`, which is verdict-significant.
        self.eval_fairness(&mut r, &cg, reports);
        self.eval_isolation(&mut r, reports, cpuset);
        self.eval_throughput_latency(&mut r, reports);
        self.eval_migration_locality(&mut r, reports, page_locality);
        self.eval_numa_migration_slow_tier(&mut r, reports, numa_nodes);
        r
    }

    /// Fairness CHECKS (Starved / Unfair / Stuck), in order: the
    /// default-threshold `not_starved` arm, then the custom
    /// `max_spread_pct` and `max_gap_ms` arms. Each appends onto `r`.
    fn eval_fairness(&self, r: &mut AssertResult, cg: &CgroupStats, reports: &[WorkerReport]) {
        // `not_starved`: default-threshold fairness (Starved / Unfair /
        // Stuck) on top of the telemetry already built above.
        if self.not_starved {
            record_default_fairness(r, cg, reports);
        }
        // Custom spread threshold — independent of `not_starved` (it gates
        // on its own field). Strip any default-threshold Unfair outcome
        // (present only when `not_starved` also ran) before re-evaluating
        // against the caller's limit.
        if let Some(spread_limit) = self.max_spread_pct {
            r.outcomes
                .retain(|o| !matches!(o, Outcome::Fail(d) if d.kind == DetailKind::Unfair));
            // `num_workers >= 2` matches the pre-decouple custom-spread
            // path. It is equivalent to the default path's
            // `measurable >= 2` gate here: a non-zero `spread` requires at
            // least two workers with measurable wall time (fewer collapses
            // min==max -> spread==0), so the `spread > limit` guard already
            // implies >= 2 measurable.
            // `spread` is None when off-CPU% was not measured (no
            // worker with positive wall time) — inconclusive, never
            // flagged unfair. A measured `Some(spread)` over the
            // limit on >= 2 workers is the real violation.
            if let Some(spread) = cg.spread
                && spread > spread_limit
                && cg.num_workers >= 2
            {
                r.record_fail(AssertDetail::new(
                    DetailKind::Unfair,
                    format!(
                        "unfair cgroup: spread={:.0}% ({:.0}-{:.0}%) {} workers on {} cpus (threshold {:.0}%)",
                        spread,
                        cg.min_off_cpu_pct.unwrap_or(0.0),
                        cg.max_off_cpu_pct.unwrap_or(0.0),
                        cg.num_workers, cg.num_cpus, spread_limit
                    ),
                ));
            }
        }
        // Custom gap threshold — independent of `not_starved`. Strip any
        // default-threshold Stuck outcome before re-evaluating.
        if let Some(threshold) = self.max_gap_ms {
            r.outcomes
                .retain(|o| !matches!(o, Outcome::Fail(d) if d.kind == DetailKind::Stuck));
            for w in reports {
                if w.max_gap_ms > threshold {
                    r.record_fail(AssertDetail::new(
                        DetailKind::Stuck,
                        format!(
                            "tid {} stuck {}ms on cpu{} at +{}ms (threshold {}ms)",
                            w.tid, w.max_gap_ms, w.max_gap_cpu, w.max_gap_at_ms, threshold,
                        ),
                    ));
                }
            }
        }
    }

    /// Isolation CHECK: run [`assert_isolation`] only when the
    /// `isolation` flag is set and a `cpuset` constraint is supplied.
    fn eval_isolation(
        &self,
        r: &mut AssertResult,
        reports: &[WorkerReport],
        cpuset: Option<&BTreeSet<usize>>,
    ) {
        if self.isolation
            && let Some(cs) = cpuset
        {
            r.merge(assert_isolation(reports, cs));
        }
    }

    /// Throughput-parity and wake-latency / iteration-rate CHECKS, in
    /// order: [`assert_throughput_parity`] (gated on
    /// `max_throughput_cv` / `min_work_rate`) then [`assert_benchmarks`]
    /// (gated on `max_p99_wake_latency_ns` / `max_wake_latency_cv` /
    /// `min_iteration_rate`).
    fn eval_throughput_latency(&self, r: &mut AssertResult, reports: &[WorkerReport]) {
        if self.max_throughput_cv.is_some() || self.min_work_rate.is_some() {
            r.merge(assert_throughput_parity(
                reports,
                self.max_throughput_cv,
                self.min_work_rate,
            ));
        }
        if self.max_p99_wake_latency_ns.is_some()
            || self.max_wake_latency_cv.is_some()
            || self.min_iteration_rate.is_some()
        {
            r.merge(assert_benchmarks(
                reports,
                self.max_p99_wake_latency_ns,
                self.max_wake_latency_cv,
                self.min_iteration_rate,
            ));
        }
    }

    /// Migration-ratio then page-locality CHECKS, in order. The
    /// locality arm reuses the `page_locality` triple computed once in
    /// the telemetry head so the value is not recomputed.
    fn eval_migration_locality(
        &self,
        r: &mut AssertResult,
        reports: &[WorkerReport],
        page_locality: Option<(f64, u64, u64)>,
    ) {
        if let Some(max_ratio) = self.max_migration_ratio {
            let total_mig: u64 = reports.iter().map(|w| w.migration_count).sum();
            let total_iters: u64 = reports.iter().map(|w| w.iterations).sum();
            if total_iters == 0 {
                r.record_inconclusive(AssertDetail::new(
                    DetailKind::Migration,
                    format!(
                        "migration ratio inconclusive: 0 iterations across {} workers — \
                         denominator is zero, ratio cannot be computed; threshold {:.4} \
                         neither pass nor fail (was the workload able to run?)",
                        reports.len(),
                        max_ratio,
                    ),
                ));
            } else {
                let ratio = total_mig as f64 / total_iters as f64;
                if ratio > max_ratio {
                    r.record_fail(AssertDetail::new(
                        DetailKind::Migration,
                        format!(
                            "migration ratio {:.4} exceeds threshold {:.4} ({} migrations / {} iterations)",
                            ratio, max_ratio, total_mig, total_iters,
                        ),
                    ));
                }
            }
        }
        if let Some(min_locality) = self.min_page_locality
            && let Some((locality, total, local)) = page_locality
        {
            // Reuse the page-locality telemetry computed in the head (the
            // cgroup-wide local/total aggregate, evaluating the cgroup as a
            // whole rather than skipping zero-page workers or summing
            // misleading per-worker fractions).
            //
            // POLICY-derived denominator: this gate is only reachable when
            // the caller supplied a `numa_nodes` set — i.e. the test set a
            // NUMA policy (typically `MemPolicy::Bind`) declaring that the
            // workload WILL allocate pages on the expected nodes. Zero
            // observed pages is therefore a policy violation, not an
            // instrumentation gap, and stays Fail (the `0.0` locality the
            // head computed then fails the threshold) per the [`Outcome`]
            // doc's INSTRUMENT-vs-POLICY carve-out. The Inconclusive
            // primitive does NOT apply here — see the sibling
            // `max_migration_ratio` / `max_slow_tier_ratio` /
            // `assert_cross_node_migration` arms for the INSTRUMENT-derived
            // counterparts.
            r.merge(assert_page_locality(
                locality,
                Some(min_locality),
                total,
                local,
            ));
        }
    }

    /// Cross-node migration then slow-tier CHECKS, in order. Both are
    /// NUMA-derived: the cross-node arm gates on
    /// `max_cross_node_migration_ratio`; the slow-tier arm gates on
    /// `max_slow_tier_ratio` and a present `numa_nodes` set.
    fn eval_numa_migration_slow_tier(
        &self,
        r: &mut AssertResult,
        reports: &[WorkerReport],
        numa_nodes: Option<&BTreeSet<usize>>,
    ) {
        if let Some(max_ratio) = self.max_cross_node_migration_ratio {
            // `vmstat_numa_pages_migrated` is the delta of the
            // system-wide `/proc/vmstat numa_pages_migrated` counter
            // captured by each worker over its own work loop. With
            // concurrent workers the deltas overlap heavily — every
            // worker observes roughly the same system-wide migration
            // count, so summing them inflates the numerator by the
            // worker count. Take the maximum delta across the cgroup
            // as the closest approximation of total migrations
            // observed during the run, then divide once by the
            // cgroup-wide total of allocated pages.
            let total_pages: u64 = reports
                .iter()
                .map(|w| w.numa_pages.values().sum::<u64>())
                .sum();
            let migrated_pages: u64 = reports
                .iter()
                .map(|w| w.vmstat_numa_pages_migrated)
                .max()
                .unwrap_or(0);
            r.merge(assert_cross_node_migration(
                migrated_pages,
                total_pages,
                Some(max_ratio),
            ));
        }
        if let Some(max_ratio) = self.max_slow_tier_ratio
            && numa_nodes.is_some()
        {
            // Skip workers with no NUMA signal (empty numa_pages or
            // zero total) but count them: if every worker dropped
            // out, the gate had no data to evaluate and previously
            // silent-passed. Record Inconclusive instead so a
            // workload that produced no NUMA allocations at all
            // doesn't masquerade as meeting the slow-tier ratio.
            let mut evaluated = 0usize;
            for w in reports {
                if w.numa_pages.is_empty() {
                    continue;
                }
                let total: u64 = w.numa_pages.values().sum();
                if total > 0 {
                    evaluated += 1;
                    r.merge(assert_slow_tier_ratio(
                        &w.numa_pages,
                        max_ratio,
                        total,
                        numa_nodes,
                    ));
                }
            }
            if evaluated == 0 {
                r.record_inconclusive(AssertDetail::new(
                    DetailKind::SlowTier,
                    format!(
                        "slow-tier ratio inconclusive: no worker reported any NUMA pages \
                         (across {} workers) — denominator is zero, ratio cannot be computed; \
                         threshold {max_ratio:.4} neither pass nor fail \
                         (did the workload allocate any memory?)",
                        reports.len(),
                    ),
                ));
            }
        }
    }
}

/// Check slow-tier page ratio against threshold.
///
/// "Slow tier" nodes are NUMA nodes NOT in the cpuset's NUMA node set.
/// For CXL memory-only nodes, these are the nodes without CPUs.
pub(crate) fn assert_slow_tier_ratio(
    numa_pages: &BTreeMap<usize, u64>,
    max_ratio: f64,
    total_pages: u64,
    numa_nodes: Option<&BTreeSet<usize>>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    let Some(cpu_nodes) = numa_nodes else {
        return r;
    };
    let slow_pages: u64 = numa_pages
        .iter()
        .filter(|(node, _)| !cpu_nodes.contains(node))
        .map(|(_, count)| count)
        .sum();
    let ratio = slow_pages as f64 / total_pages as f64;
    if ratio > max_ratio {
        r.record_fail(AssertDetail::new(
            DetailKind::SlowTier,
            format!(
                "slow-tier page ratio {ratio:.4} ({pct:.2}%) exceeds threshold {max_ratio:.4} ({thr_pct:.2}%) \
                 ({slow_pages}/{total_pages} pages on non-CPU nodes)",
                pct = ratio * 100.0,
                thr_pct = max_ratio * 100.0,
            ),
        ));
    }
    r
}

/// Check NUMA page locality against threshold.
///
/// `observed` is the fraction of pages on expected nodes (0.0-1.0).
/// `total_pages` and `local_pages` are included in diagnostics.
pub fn assert_page_locality(
    observed: f64,
    min_locality: Option<f64>,
    total_pages: u64,
    local_pages: u64,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if let Some(threshold) = min_locality
        && observed < threshold
    {
        r.record_fail(AssertDetail::new(
            DetailKind::PageLocality,
            format!(
                "page locality {observed:.4} ({pct:.2}%) below threshold {threshold:.4} ({thr_pct:.2}%) ({local_pages}/{total_pages} pages local)",
                pct = observed * 100.0,
                thr_pct = threshold * 100.0,
            ),
        ));
    }
    r
}

/// Check cross-node page migration ratio against threshold.
///
/// `migrated_pages` is the delta of `/proc/vmstat` `numa_pages_migrated`
/// between pre- and post-workload snapshots. `total_pages` is the total
/// allocated pages from numa_maps.
///
/// Inconsistent inputs (`migrated_pages > 0` while `total_pages == 0`)
/// fail loudly: vmstat saw migrations the workload's numa_maps did not
/// account for, which is either a measurement gap or an instrumentation
/// bug, and silently coercing the ratio to 0.0 would let the assertion
/// pass on data the operator should not trust.
///
/// When both inputs are zero (`migrated_pages == 0 && total_pages == 0`)
/// the gate records Inconclusive — the denominator is zero and the
/// check has no signal to evaluate; neither Pass (would silently green
/// a workload that produced no NUMA pages) nor Fail (no actual ratio
/// violation observed) is truthful.
pub fn assert_cross_node_migration(
    migrated_pages: u64,
    total_pages: u64,
    max_ratio: Option<f64>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    if let Some(threshold) = max_ratio {
        if total_pages == 0 {
            if migrated_pages > 0 {
                r.record_fail(AssertDetail::new(
                    DetailKind::CrossNodeMigration,
                    format!(
                        "cross-node migration inconsistent: {migrated_pages} pages migrated but 0 pages observed in numa_maps (threshold {threshold:.4})",
                    ),
                ));
            } else {
                r.record_inconclusive(AssertDetail::new(
                    DetailKind::CrossNodeMigration,
                    format!(
                        "cross-node migration inconclusive: 0 pages observed in numa_maps and 0 pages migrated — \
                         denominator is zero, ratio cannot be computed; threshold {threshold:.4} \
                         neither pass nor fail (did the workload allocate any memory?)",
                    ),
                ));
            }
            return r;
        }
        let ratio = migrated_pages as f64 / total_pages as f64;
        if ratio > threshold {
            r.record_fail(AssertDetail::new(
                DetailKind::CrossNodeMigration,
                format!(
                    "cross-node migration ratio {ratio:.4} ({pct:.2}%) exceeds threshold {threshold:.4} ({thr_pct:.2}%) ({migrated_pages}/{total_pages} pages migrated)",
                    pct = ratio * 100.0,
                    thr_pct = threshold * 100.0,
                ),
            ));
        }
    }
    r
}

#[cfg(test)]
impl AssertPlan {
    pub(crate) fn check_not_starved(mut self) -> Self {
        self.not_starved = true;
        self
    }

    pub(crate) fn check_isolation(mut self) -> Self {
        self.isolation = true;
        self
    }

    pub(crate) fn max_gap_ms(mut self, ms: u64) -> Self {
        self.max_gap_ms = Some(ms);
        self
    }

    pub(crate) fn max_spread_pct(mut self, pct: f64) -> Self {
        self.max_spread_pct = Some(pct);
        self
    }
}
