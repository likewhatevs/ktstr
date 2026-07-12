use super::*;

/// Category tag for an [`AssertDetail`]. Enables structural filtering
/// (e.g. by `AssertPlan`) without matching on substrings of
/// human-readable messages, which is fragile if wording changes.
///
/// Notes previously lived as a `DetailKind::Note` variant on
/// [`AssertDetail`]; they now live on [`AssertResult::info_notes`] as
/// [`InfoNote`] values. See [`AssertResult::note`] /
/// [`AssertResult::with_note`] for the producer-side migration and
/// [`InfoNote`] for the rationale (structurally-separate context
/// stream so sidecar consumers iterating `details` count only real
/// failures without a "forgot to filter `kind == Note`" miscount
/// class of bug).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DetailKind {
    /// A worker made zero progress.
    NoProgress,
    /// A worker was stuck off-CPU longer than the gap threshold.
    Stuck,
    /// Spread between best and worst worker exceeded the fairness threshold.
    Unfair,
    /// A worker ran on a CPU outside its expected cpuset.
    Isolation,
    /// Throughput / benchmarking threshold failure (p99, CV, rate).
    Benchmark,
    /// Migration-ratio threshold failure (migrations per iteration).
    Migration,
    /// NUMA page locality threshold failure.
    PageLocality,
    /// Cross-node migration threshold failure.
    CrossNodeMigration,
    /// Slow-tier (memory tier) threshold failure.
    SlowTier,
    /// Monitor-subsystem anomaly (imbalance, DSQ depth, rq_clock stuck).
    /// Use one of [`DetailKind::SchedulerCrashed`] /
    /// [`DetailKind::SchedulerExitedCleanly`] /
    /// [`DetailKind::SchedulerDiedUnknownReason`] for scheduler-liveness failures.
    Monitor,
    /// Scheduler process observed to have died (via `sched_pid`
    /// probe returning ESRCH or wait on the leader) AND the BPF
    /// probe observed a non-clean `trace_sched_ext_exit` event
    /// before the liveness check fired. The crash classification
    /// covers SCX_EXIT_ERROR, SCX_EXIT_ERROR_STALL, watchdog kick,
    /// and BPF-side error paths — every kernel exit that latches
    /// `ktstr_err_exit_detected` in the probe BSS.
    ///
    /// Distinguished from [`DetailKind::SchedulerExitedCleanly`]
    /// (`SCX_EXIT_NONE` clean teardown) so the console-dump gate
    /// and downstream triage can tell a real crash from a benign
    /// completion. Consumers wanting to gate on "any scheduler
    /// exit" should match both variants via
    /// `matches!(d.kind, SchedulerCrashed | SchedulerExitedCleanly)`.
    SchedulerCrashed,
    /// Scheduler process observed to have died with the probe BSS
    /// `ktstr_err_exit_detected` latch unset — the kernel ran the
    /// `SCX_EXIT_NONE` clean-teardown path (sysrq, explicit
    /// unregister) without latching an error. Surfaces alongside
    /// `SchedulerCrashed` because both are "scheduler exited"
    /// signals; splitting them lets the operator distinguish a
    /// benign shutdown from a real fault without re-parsing
    /// console output.
    SchedulerExitedCleanly,
    /// Scheduler process observed to have died but the BPF probe
    /// has no classification yet — either the probe never armed
    /// for this run (no scheduler attached, host-only test) or
    /// the poll thread has not completed a first iteration since
    /// the prior reset. Operators triaging this variant should
    /// check whether the probe pipeline was wired before
    /// concluding "scheduler-exit classification is broken".
    SchedulerDiedUnknownReason,
    /// SCX event-counter threshold failure. An error-class
    /// `SCX_EV_*` counter (e.g. `enq_skip_exiting`,
    /// `enq_skip_migration_disabled`, `dispatch_local_dsq_offline`) crossed
    /// the configured bound. Distinct from the process-liveness
    /// variants ([`DetailKind::SchedulerCrashed`] /
    /// [`DetailKind::SchedulerExitedCleanly`] /
    /// [`DetailKind::SchedulerDiedUnknownReason`]) and
    /// [`DetailKind::Monitor`] (imbalance / DSQ-depth /
    /// rq_clock-stuck): this kind flags individual event-counter
    /// regressions surfaced by [`assert_scx_events_clean`]. The
    /// counters themselves originate in the kernel's per-task
    /// `scx_event_stats` (see `kernel/sched/ext.c` —
    /// `SCX_EV_*` macros); ktstr reads aggregated deltas via
    /// `monitor::ScxEventDeltas` and presents them to the
    /// assertion as `(name, count)` pairs.
    SchedulerEvent,
    /// Temporal assertion failure on a periodic-capture
    /// [`SampleSeries`](crate::scenario::sample::SampleSeries).
    /// One of the seven built-in patterns
    /// (`nondecreasing` / `strictly_increasing`, `rate_within`,
    /// `steady_within`, `converges_to`, `always_true`,
    /// `ratio_within`) or a per-sample scalar comparator
    /// invoked via `.each(...)` reported a violation. The
    /// detail message names the pattern, the offending sample
    /// tag(s), and the observed-vs-expected values; the
    /// stdout `--- temporal assertions ---` summary in
    /// `test_support::output` aggregates the same kind into
    /// per-assertion pass/fail rows.
    Temporal,
    /// Host-mode stuck worker detected by
    /// [`crate::scenario::host_stuck`]. The polling thread
    /// observed `Δnr_switches == 0` AND `Δsum_exec_runtime == 0`
    /// across the configured window for a worker pid — the task
    /// neither got picked nor preempted for at least
    /// `STUCK_WINDOW * poll_interval` ms. Distinct from
    /// [`DetailKind::Stuck`] (worker-side report: a worker was
    /// off-CPU longer than the in-test gap threshold): this kind
    /// fires from the host-side polling thread when running
    /// host-mode (no VM boot) and is the only stuck-worker signal
    /// available in that mode.
    WorkerStuck,
    /// Skip notification (scenario could not run under this topology/flags).
    Skip,
    /// Uncategorized — falls through when a detail has no specific kind.
    Other,
    /// Performance-mode host-isolation fault: a `performance_mode` cell's
    /// Body-phase dilation `D` exceeded the perf-isolation ceiling
    /// (`PERF_ISOLATION_D_MAX`, in `vmm::freeze_coord::latency_verdict`),
    /// so its 1:1-pinned vCPUs lost their dedicated host CPUs. Recorded
    /// HOST-side at the eval seam (never crosses the guest→host wire), so
    /// its postcard variant index is unobserved; still appended LAST to
    /// preserve the append-only wire ordering the other variants rely on.
    /// This is an infra/isolation fault (the run's timing verdicts are
    /// untrustworthy), NOT a scheduler-under-test fault.
    PerfIsolation,
}

/// Message prefix emitted by every scenario-runner site that
/// detects the scheduler process has died — whether through a
/// post-ops liveness probe or an inter-step liveness check. Both
/// paths share this single prefix as the operator-visible
/// message format so someone grepping stderr for the canonical
/// "scheduler process died" string hits every emission site.
/// Structural routing (the console-dump gate in
/// `test_support::eval`) goes through the `DetailKind::Scheduler*`
/// variants ([`DetailKind::SchedulerCrashed`] /
/// [`DetailKind::SchedulerExitedCleanly`] /
/// [`DetailKind::SchedulerDiedUnknownReason`]),
/// NOT this prefix — the prefix is a human-readability contract,
/// not a detection mechanism. Exposed as `pub(crate)` so emitters
/// reference the same literal; renaming the prefix is a one-site
/// edit instead of a grep-and-hope across `scenario::*`.
///
/// Vocabulary history: prior versions of this module used two
/// prefixes (`SCHED_EXITED_PREFIX` = "scheduler process exited"
/// and `SCHED_NO_LONGER_RUNNING_PREFIX` = "scheduler process no
/// longer running") for in-workload vs post-ops detection. The
/// distinction carried no downstream semantics — every consumer
/// treated both as equivalent scheduler-death signals — so the
/// wording was unified onto "died" (shorter, matches the prior
/// `SchedulerDied` variant name, and closes a class of "which
/// wording does this site use?" drift bugs).
pub(crate) const SCHED_DIED_PREFIX: &str = "scheduler process died";

/// Format the scheduler-died detail message for an inter-step
/// liveness-probe failure (the scheduler was alive after step
/// `step_idx - 1` but ESRCH'd before step `step_idx` ran).
///
/// Begins with [`SCHED_DIED_PREFIX`] verbatim, followed by
/// "unexpectedly after completing step N of M (X.Xs into test)".
/// The prefix is the operator-visible stderr anchor (see the
/// prefix doc); structural routing is via one of
/// [`DetailKind::SchedulerCrashed`] /
/// [`DetailKind::SchedulerExitedCleanly`] /
/// [`DetailKind::SchedulerDiedUnknownReason`] on the emitted `AssertDetail`.
/// Centralized so ops.rs and any future emitter share a single
/// format.
pub(crate) fn format_sched_died_after_step(
    step_idx: usize,
    total_steps: usize,
    elapsed_s: f64,
) -> String {
    format!(
        "{SCHED_DIED_PREFIX} unexpectedly after completing step {step_idx} of {total_steps} ({elapsed_s:.1}s into test)",
    )
}

/// Format the scheduler-died detail message for the post-loop
/// liveness probe (the scheduler was alive throughout the step loop
/// but ESRCH'd after the last step completed).
///
/// Begins with [`SCHED_DIED_PREFIX`] verbatim; shares the prefix
/// invariant documented on [`format_sched_died_after_step`].
/// Structural routing is via one of [`DetailKind::SchedulerCrashed`] /
/// [`DetailKind::SchedulerExitedCleanly`] /
/// [`DetailKind::SchedulerDiedUnknownReason`] on the emitted detail.
pub(crate) fn format_sched_died_after_all_steps(total_steps: usize, elapsed_s: f64) -> String {
    format!(
        "{SCHED_DIED_PREFIX} unexpectedly (detected after all {total_steps} steps completed, {elapsed_s:.1}s elapsed)",
    )
}

/// Format the scheduler-died detail message for the in-step
/// liveness probe (the scheduler ESRCH'd during a step's hold-period
/// sleep, before the step completed).
///
/// Begins with [`SCHED_DIED_PREFIX`] verbatim; shares the prefix
/// invariant documented on [`format_sched_died_after_step`].
/// Structural routing is via one of [`DetailKind::SchedulerCrashed`] /
/// [`DetailKind::SchedulerExitedCleanly`] /
/// [`DetailKind::SchedulerDiedUnknownReason`] on the emitted detail. Emitted by `run_scenario` when the
/// liveness-poll inside `run_step`'s hold sleep observes
/// `process_alive(sched_pid) == false`, replacing the prior
/// behavior that waited for the post-loop probe to fire (which
/// stamped the message with the full scenario duration even when
/// the scheduler had died seconds earlier).
pub(crate) fn format_sched_died_during_workload(elapsed_s: f64) -> String {
    format!("{SCHED_DIED_PREFIX} unexpectedly during workload ({elapsed_s:.1}s into test)")
}

/// Format the scheduler-died message for the guest-side `survives_storm`
/// post-function liveness probe: the test asserted the scheduler survives, but
/// after the test function returned it was found dead or down (the leader pid
/// ESRCH'd, or the scx state went `disabling`/`disabled`) while still expected
/// to run. Fires for scenarios that do NOT drive an `execute_*` in-hold probe
/// (which records its own scheduler-death detail via the `format_sched_died_*`
/// helpers above). Begins with [`SCHED_DIED_PREFIX`] verbatim; structural
/// routing is via one of [`DetailKind::SchedulerCrashed`] /
/// [`DetailKind::SchedulerExitedCleanly`] /
/// [`DetailKind::SchedulerDiedUnknownReason`] on the emitted detail.
pub(crate) fn format_sched_died_survives_storm() -> String {
    format!(
        "{SCHED_DIED_PREFIX} during a survives_storm workload (detected after the test function returned)"
    )
}

/// Structured wall-latency-gate evidence carried on a failing
/// [`AssertDetail`] so the host-side seam can re-run the
/// contention-aware tri-state verdict (the `latency_verdict` core in
/// `vmm::freeze_coord`).
///
/// The wall-latency ceilings fire IN-GUEST (`assert_benchmarks` runs
/// inside the VM), but the per-phase contention witness — the Body
/// dilation `D` and the peak-window run-delay series that bound
/// `W(measured)` — is measured HOST-side (in `vmm::freeze_coord`) and is
/// not available where the gate tripped. This struct rides the failing
/// detail across the wire so the host can pair the guest's `measured_ns`
/// / `threshold_ns` with its own witness and decide Pass / FailConfirmed
/// / ContentionIndeterminate.
///
/// Set ONLY on ns-denominated latency ceilings where the tri-state is
/// sound (`max_p99_wake_latency_ns`): the p99 exemplar is a single
/// guest-wall interval whose contamination `W` bounds. NOT set on the CV
/// or off-CPU-spread gates — those are dimensionless (a ratio / a
/// percentage), not ns intervals, so the ns-denominated `W` bound cannot
/// score them; they stay annotate-only via `dilation_annotation`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LatencyGate {
    /// The guest-measured wall-latency exemplar that tripped the gate (ns).
    pub measured_ns: u64,
    /// The ceiling it exceeded (ns).
    pub threshold_ns: u64,
}

/// A single diagnostic message from an assertion, paired with a
/// structural [`DetailKind`] so filtering is robust to wording changes.
///
/// Access the message text via `detail.message`; format-string probes
/// (`format!("{detail}")`) work via the `Display` impl. New code that
/// needs to filter by category should match on `kind` rather than
/// substring-match the message text — wording can change without
/// notice but the variant tag is the structural contract.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssertDetail {
    pub kind: DetailKind,
    pub message: String,
    /// Structured wall-latency-gate evidence for the host contention
    /// seam ([`LatencyGate`]); `None` on every detail that is not an
    /// in-scope ns-latency-ceiling failure. Stamped at the gate site via
    /// [`Self::with_latency_gate`].
    ///
    /// `#[serde(default)]` keeps sidecar JSON written before this field
    /// existed readable (an absent field deserializes to `None`). NO
    /// `skip_serializing_if`: [`AssertDetail`] crosses the guest→host
    /// boundary via postcard, which is not self-describing and always
    /// expects every field in declaration order — a conditionally-elided
    /// field would desync the decoder. The `Option` is encoded as its
    /// 1-byte tag either way, matching the sibling `phase` field.
    #[serde(default)]
    pub latency_gate: Option<LatencyGate>,
    /// Scenario phase the detail was emitted under. Mirrors
    /// [`PassDetail::phase`]: `None` outside any [`PhaseGuard`] scope
    /// (boot, BASELINE settle, non-scenario test fixtures), `Some`
    /// when an active guard has installed a label. Carried on every
    /// detail so consumers (auto-repro renderer, sidecar parsers)
    /// see a uniform phase field across pass + fail records.
    /// Producers that already know the active phase can stamp via
    /// [`Self::with_phase`].
    ///
    /// [`Cow`](std::borrow::Cow)`<'static, str>` mirrors [`PassDetail::phase`] for the
    /// same zero-allocation reason: the common case is the per-step
    /// RAII guard's static `&'static str` label staying as
    /// `Cow::Borrowed` (zero alloc); runtime-built `String`s become
    /// `Cow::Owned`.
    pub phase: Option<std::borrow::Cow<'static, str>>,
}

impl AssertDetail {
    pub fn new(kind: DetailKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            phase: current_phase_label(),
            latency_gate: None,
        }
    }

    /// Builder-style setter for [`Self::latency_gate`]. Consumes self,
    /// attaches the guest-measured `measured_ns` / `threshold_ns` so the
    /// host seam can re-run the contention-aware tri-state verdict, and
    /// returns the updated detail. Chain at the ns-latency-ceiling gate
    /// sites: `AssertDetail::new(Benchmark, msg).with_latency_gate(p99,
    /// limit)`. Only the in-scope wall-latency ceilings set this — see
    /// [`LatencyGate`] for why CV / spread do not.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn with_latency_gate(mut self, measured_ns: u64, threshold_ns: u64) -> Self {
        self.latency_gate = Some(LatencyGate {
            measured_ns,
            threshold_ns,
        });
        self
    }

    /// Builder-style setter for [`Self::phase`]. Consumes self,
    /// stamps the phase label, returns the updated value. Matches
    /// the [`PassDetail::with_phase`] shape so producers can chain
    /// `AssertDetail::new(...).with_phase(...)` uniformly across
    /// pass and fail records.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn with_phase(mut self, phase: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        self.phase = Some(phase.into());
        self
    }

    /// Borrow this detail as a kind-prefixed [`std::fmt::Display`]
    /// adapter. The default [`Display`](std::fmt::Display) impl on
    /// `AssertDetail` writes only `message` so terminal output reads
    /// as bare prose; structured-log consumers that want to bucket
    /// failures by category without re-checking [`Self::kind`] reach
    /// for this helper instead.
    ///
    /// Renders as `[<DetailKind variant name>] <message>` — debug-form
    /// for the kind so the variant token is grep-stable across renames
    /// (a regression that drops a `DetailKind` variant breaks the
    /// match arms that produce it; the rendered token follows). Zero-
    /// allocation: the wrapper holds a `&AssertDetail` and writes
    /// straight into the formatter.
    ///
    /// ```
    /// # use ktstr::assert::{AssertDetail, DetailKind};
    /// let d = AssertDetail::new(DetailKind::Stuck, "tid 7 stuck 1500ms on cpu3");
    /// assert_eq!(d.to_string(), "tid 7 stuck 1500ms on cpu3");
    /// assert_eq!(
    ///     d.display_with_kind().to_string(),
    ///     "[Stuck] tid 7 stuck 1500ms on cpu3",
    /// );
    /// ```
    pub fn display_with_kind(&self) -> AssertDetailWithKind<'_> {
        AssertDetailWithKind { detail: self }
    }
}

/// Structured record of a single passing claim — the positive
/// counterpart to [`AssertDetail`]. Populated by [`Verdict`]'s
/// `record_pass_unary` / `record_pass_binary` helpers at every
/// comparator's pass arm so the auto-repro renderer (and any other
/// consumer that wants per-claim fidelity) can iterate passes
/// alongside fails.
///
/// Carries the same shape primitives every comparator naturally has
/// at the pass site: the claim's `name`, a short `comparator`
/// token (`"eq"`, `"ge"`, `"is_finite"`, …), the `value` that was
/// compared (formatted via the comparator's `Display`), and an
/// optional `expected` for binary comparators. Unary comparators
/// (e.g. `is_finite`, `set_is_empty`) leave `expected = None`.
///
/// `comparator` is a **wire-canonical token** from
/// [`COMPARATOR_VOCABULARY`], NOT a string derived from the builder
/// method name. Operator-named comparators map to operator-canonical
/// tokens (`eq`/`ne`/`ge`/`le`/`lt`/`gt`) regardless of whether the
/// invoking builder method is `eq` or `at_least` — tokens are the
/// stable wire vocabulary, methods are the ergonomic surface. A
/// renderer that wants pretty operators can map `ge → >=` on output.
///
/// Container-bound comparators prefix their tokens with the
/// container type name (`set_*`, `sequence_*`) to disambiguate same-
/// named operations across surfaces (`contains` is ambiguous between
/// sets and sequences, so prefix; `is_finite` is scalar-only, so
/// bare). The prefix policy is part of the vocabulary contract.
///
/// `comparator` is a [`Cow`](std::borrow::Cow)`<'static, str>` so call sites passing a
/// `&'static str` literal — the universal case for built-in
/// comparators — pay zero allocation; runtime-built comparator
/// labels store as `Cow::Owned`. The same `Cow` shape applies to
/// `phase` (set by the per-step RAII guard's static label in the
/// common case).
///
/// **Structurally distinct from [`AssertDetail`]**: PassDetail
/// carries a uniform per-claim shape (every comparator emits
/// `name + comparator + value + expected`), while AssertDetail
/// uses a `kind: DetailKind` category enum because failure /
/// note / warning shapes diverge. Forcing them to one mold would
/// either lose comparator-typed slots (collapse to kind+message)
/// or invent a Pass variant of DetailKind that doesn't carry the
/// typed slots cleanly. Keeping them separate is a deliberate
/// design choice, not an inconsistency.
///
/// Distinct from a one-line tracing log — the structured form is
/// the data path the auto-repro renderer reads to compose the
/// bracketed phase output that surfaces passing context alongside
/// failing assertions. The tracing log path remains the
/// operator-facing surface for `--nocapture` runs.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PassDetail {
    pub name: String,
    pub comparator: std::borrow::Cow<'static, str>,
    pub value: String,
    pub expected: Option<String>,
    /// Scenario phase the claim was made under. `None` outside any
    /// [`PhaseGuard`] scope; `Some(label)` when the active-phase
    /// thread-local has been installed at the scenario-driver step
    /// loop entry. The auto-repro renderer groups passes by this
    /// field to compose the bracketed `==== PHASE N: <label> ====`
    /// output. [`Cow`](std::borrow::Cow)`<'static, str>` so the common case (the RAII
    /// guard's static `&'static str` label) pays zero allocation.
    pub phase: Option<std::borrow::Cow<'static, str>>,
}

impl PassDetail {
    /// Construct a binary-comparator pass record (e.g. `eq`, `ge`,
    /// `in_range`). Both `value` and `expected` are stringified via
    /// [`std::fmt::Display`] at the call site so the struct is
    /// `T`-agnostic on the wire. See [`COMPARATOR_VOCABULARY`] for
    /// the full set of canonical tokens.
    pub fn binary(
        name: impl Into<String>,
        comparator: impl Into<std::borrow::Cow<'static, str>>,
        value: impl Into<String>,
        expected: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            comparator: comparator.into(),
            value: value.into(),
            expected: Some(expected.into()),
            phase: current_phase_label(),
        }
    }

    /// Construct a unary-comparator pass record (e.g. `is_finite`,
    /// `set_is_empty`). `expected` is left None — the comparator
    /// name alone carries the meaning. See [`COMPARATOR_VOCABULARY`]
    /// for the full set of canonical tokens.
    pub fn unary(
        name: impl Into<String>,
        comparator: impl Into<std::borrow::Cow<'static, str>>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            comparator: comparator.into(),
            value: value.into(),
            expected: None,
            phase: current_phase_label(),
        }
    }

    /// Builder-style setter for [`Self::phase`]. Consumes self,
    /// stamps the phase label, returns the updated value so
    /// per-phase test fixtures and the [`PhaseGuard`] RAII helper can chain
    /// `PassDetail::binary(...).with_phase("step_0")`.
    /// `&'static str` literals stay `Cow::Borrowed` (zero alloc);
    /// runtime-built `String` becomes `Cow::Owned`.
    pub fn with_phase(mut self, phase: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        self.phase = Some(phase.into());
        self
    }
}

/// Wire-canonical vocabulary of `PassDetail.comparator` tokens.
///
/// Every comparator implementation in [`crate::assert::claim`] emits
/// one of these tokens when it records a passing claim. The vocabulary
/// is the **stable wire contract** — renderers, sidecar consumers,
/// and the auto-repro pipeline match against these exact
/// strings. A token rename in `claim.rs` without a parallel update
/// here breaks every downstream consumer; the regression test in
/// `tests/claim_comparator_tokens_canonical.rs` pins this.
///
/// One synthetic token is NOT in this vocabulary: the cap-overflow
/// sentinel record (see [`PASSES_TRUNCATION_SENTINEL_NAME`]) carries
/// `comparator = "truncated"` to indicate the slot is metadata, not
/// a real claim. Renderers that filter passes by vocabulary should
/// also handle the sentinel as a distinct category.
///
/// Tokens follow three style rules:
///
/// 1. **Operator-canonical**: comparison operators map to short
///    operator names (`eq`, `ne`, `ge`, `le`, `lt`, `gt`) regardless
///    of whether the builder method is `eq` or `at_least`. The
///    vocabulary is independent of method naming.
/// 2. **Container-prefixed**: comparators bound to a specific
///    container type prefix their token with the container name
///    (`set_*`, `sequence_*`) to disambiguate same-named operations
///    across surfaces. Scalar tokens are unprefixed.
/// 3. **Snake-case ASCII**: every token is lower-snake-case, no
///    Unicode, no spaces — survives shell escapes, IDE regex search,
///    and log-mining pipelines without transformation.
///
/// Asymmetries are intentional: `sequence_*` does not carry
/// `subset_of` / `disjoint_from` because sequences have order and
/// duplicates that set semantics don't model.
///
/// Categorization below groups by SEMANTIC AXIS (comparison /
/// predicate / cardinality / membership / relation), not by call-
/// site arity. Every `len_*` cardinality token (`set_len_eq` /
/// `set_len_le` / `set_len_ge` and the sequence peers) records
/// via the binary helper so renderer-side handling is uniform:
/// each pass surfaces both the actual length and the expected
/// bound. The previous unary-on-eq + binary-on-le/ge asymmetry
/// was a micro-optimization (eq's `actual == expected` makes the
/// actual redundant on the pass arm) that traded uniform output
/// for one elided field. The `*_is_non_empty` and `*_is_empty`
/// predicates remain unary by design — the comparator token IS
/// the predicate, and there is no separate expected bound to
/// surface. `*_is_non_empty` records the observed length
/// (evidence the container was non-empty); `*_is_empty` records
/// no value (the predicate token alone carries the meaning —
/// emptiness is self-evident from the comparator).
pub const COMPARATOR_VOCABULARY: &[&str] = &[
    // Scalar comparisons
    "eq",
    "ne",
    "ge",
    "le",
    "lt",
    "gt",
    "in_range",
    "near_within",
    // Scalar predicates
    "is_finite",
    // Set predicates (emptiness)
    "set_is_empty",
    "set_is_non_empty",
    // Set cardinality
    "set_len_eq",
    "set_len_le",
    "set_len_ge",
    // Set membership / relations
    "set_contains",
    "subset_of",
    "disjoint_from",
    // Sequence predicates (emptiness)
    "sequence_is_empty",
    "sequence_is_non_empty",
    // Sequence cardinality
    "sequence_len_eq",
    "sequence_len_le",
    "sequence_len_ge",
    // Sequence membership
    "sequence_contains",
];

/// Cap on `AssertResult.passes` (and the matching truncation sentinel)
/// so a pathological test that fires millions of claims doesn't
/// balloon the wire-formatted result. Mirrors SnapshotBridge's
/// `MAX_STORED_EVENTS` truncation pattern: when the cap is hit,
/// the cap-th record is replaced with a synthetic
/// `PassDetail { name: PASSES_TRUNCATION_SENTINEL_NAME, … }` carrying
/// the dropped-count, and further pushes are no-ops.
///
/// 10_000 is comfortably above the steady-state claim count of every
/// existing test (typical test fires <100 claims; pathological hot-
/// loop tests in the tree fire under 5_000) while bounding the
/// worst-case wire size to ~3 MB — well under the 16 MiB
/// `MAX_BULK_FRAME_PAYLOAD` per vmm/bulk.rs:56.
pub const MAX_RECORDED_PASSES: usize = 10_000;

/// Sentinel `PassDetail.name` value used by the truncation record
/// that replaces the `MAX_RECORDED_PASSES`-th slot when a test
/// over-runs the cap. Consumers (the auto-repro renderer) match on
/// this string to render `[N passes truncated]` instead of treating
/// it as a real claim.
///
/// **Truncation-check idiom**: a caller checking
/// `result.passes.len() == MAX_RECORDED_PASSES` MISSES the truncated
/// state because the truncation sentinel pushes the vec to
/// `MAX_RECORDED_PASSES + 1`. The correct check is
/// `result.passes.last().map(|p| p.name == PASSES_TRUNCATION_SENTINEL_NAME)
/// .unwrap_or(false)` — i.e. inspect the tail entry's name.
pub const PASSES_TRUNCATION_SENTINEL_NAME: &str = "__ktstr_passes_truncated__";

/// Comparator-string value used by the truncation sentinel record
/// alone. Out-of-vocabulary by design — not in [`COMPARATOR_VOCABULARY`]
/// — so the runtime debug_assert in `record_pass_inner` allows it
/// explicitly without polluting the wire-canonical token set.
pub const PASSES_TRUNCATION_SENTINEL_COMPARATOR: &str = "truncated";

/// `Display` adapter returned by [`AssertDetail::display_with_kind`].
/// Renders the detail as `[<kind>] <message>`. Held by reference so
/// the helper allocates nothing on the formatting path; the lifetime
/// is the borrow of the source `AssertDetail`.
#[must_use = "AssertDetailWithKind only renders when formatted"]
pub struct AssertDetailWithKind<'a> {
    detail: &'a AssertDetail,
}

impl std::fmt::Display for AssertDetailWithKind<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.detail.kind, self.detail.message)
    }
}

impl From<String> for AssertDetail {
    /// Conversion for uncategorized messages; defaults `kind` to
    /// [`DetailKind::Other`]. Prefer [`AssertDetail::new`] when the
    /// detail has a meaningful category — the `DetailKind` is serialized
    /// into the sidecar JSON and consumed by stats tooling to bucket
    /// failures, so losing the category bucket makes post-run
    /// categorization rely on free-text regex against `message`.
    fn from(message: String) -> Self {
        Self::new(DetailKind::Other, message)
    }
}

/// Informational annotation that does NOT contribute to the failure
/// verdict — the structural counterpart to [`AssertDetail`] for
/// "context surfaced alongside a result" emissions. Lives in its own
/// type (not as a `DetailKind` variant of `AssertDetail`) so the
/// "details = failures" mental model holds at the type level:
/// `AssertResult::details` is the failure stream, `AssertResult::info_notes`
/// is the context stream. Producers can no longer accidentally tag a
/// note as a failure (the prior `DetailKind::Note` variant on
/// `AssertDetail` made misclassification a one-character bug — every
/// sidecar consumer that read `details` needed to remember to filter
/// `kind == Note` to count real failures, and forgetting silently
/// misreported failure counts).
///
/// Carries the same `phase` field as [`AssertDetail`] so the auto-repro
/// renderer can attribute notes to the scenario phase they were emitted
/// under, mirroring the per-step grouping already used for failures
/// and passes.
///
/// `PartialEq + Eq` mirror the derive set on [`AssertDetail`] and
/// [`PassDetail`] so test authors can compose `AssertResult` fixtures
/// across the three record types with uniform structural-equality
/// affordances. Test authors should still prefer
/// `result.info_notes.iter().any(|n| n.message.contains(...))` over
/// `assert_eq!(result.info_notes, expected)` so pins survive note
/// wording adjustments without churn.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InfoNote {
    pub message: String,
    /// Scenario phase the note was emitted under. Mirrors
    /// [`AssertDetail::phase`] and [`PassDetail::phase`] so the
    /// renderer threads pass / fail / note records through one
    /// per-phase grouping.
    pub phase: Option<std::borrow::Cow<'static, str>>,
}

impl InfoNote {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            phase: current_phase_label(),
        }
    }

    /// Builder-style setter for [`Self::phase`]. Matches the
    /// [`AssertDetail::with_phase`] shape so producers can chain
    /// `InfoNote::new(...).with_phase(...)` uniformly with the
    /// failure-detail builder.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn with_phase(mut self, phase: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        self.phase = Some(phase.into());
        self
    }
}

impl std::fmt::Display for InfoNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl From<&str> for AssertDetail {
    /// Conversion for uncategorized messages; defaults `kind` to
    /// [`DetailKind::Other`]. Prefer [`AssertDetail::new`] when the
    /// detail has a meaningful category — the `DetailKind` is serialized
    /// into the sidecar JSON and consumed by stats tooling to bucket
    /// failures, so losing the category bucket makes post-run
    /// categorization rely on free-text regex against `message`.
    fn from(s: &str) -> Self {
        Self::new(DetailKind::Other, s)
    }
}

impl std::fmt::Display for AssertDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// Result of checking a scenario run.
///
/// Contains pass/fail status, human-readable detail messages, and
/// aggregated statistics. Multiple results can be combined with
/// [`merge()`](AssertResult::merge).
///
/// ```
/// # use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
/// let mut a = AssertResult::pass();
/// assert!(a.is_pass());
///
/// let mut b = AssertResult::pass();
/// b.record_fail(AssertDetail::new(DetailKind::NoProgress, "worker made no progress"));
///
/// a.merge(b);
/// assert!(a.is_fail());
/// assert!(a.failure_details().any(|d| d.kind == DetailKind::NoProgress));
/// ```
/// Structured measurement value attached via
/// [`AssertResult::note_value`] / [`Verdict::note_value`].
///
/// The variants cover every primitive shape stats tooling consumes:
/// signed and unsigned 64-bit ints, 64-bit floats, booleans, and
/// owned strings. A test that wants to surface "max_wchar=12345"
/// alongside a passing IO_ACCOUNTING reachability check writes
/// `verdict.note_value("max_wchar", 12345i64)` and downstream stats
/// tooling reads `result.measurements["max_wchar"]` as
/// `NoteValue::Int(12345)`.
///
/// Distinct from [`AssertResult::info_notes`]'s free-form
/// [`InfoNote`] messages: an `InfoNote` carries a single human-
/// readable string (formatted via its `Display` impl), the
/// structured map carries typed `(key, NoteValue)` pairs for
/// programmatic consumption (sidecar parsers, `perf-delta`,
/// regression dashboards). Producers can call BOTH `note(msg)`
/// and `note_value(key, val)` on the same result — they occupy
/// independent buffers (`info_notes` vs `measurements`).
///
/// Conversion via the `From` impls below: any
/// `i64`/`u64`/`f64`/`bool`/`String`/`&str` literal flows into
/// `note_value` without explicit variant naming. Integer types
/// narrower than 64-bit (`i32`, `u32`, etc.) need an explicit cast
/// at the call site rather than a blanket impl, so the call site
/// reads honestly about the value's resolution.
///
/// Derives `PartialEq` but NOT `Eq`: the `Float(f64)` variant holds
/// IEEE-754 doubles where `NaN != NaN`, which violates the
/// reflexivity requirement on `Eq`. Equality on `NoteValue` is
/// partial-equivalence semantics for the same reason `f64` is.
///
/// Uses serde's externally-tagged default (no `#[serde(untagged)]`).
/// Like [`Outcome`], NoteValue is wire-encoded as part of
/// [`AssertResult::measurements`] via postcard's TLV transport from
/// guest to host. Postcard is not a self-describing format and cannot
/// decode `#[serde(untagged)]` enums (returns `WontImplement`) — pre-fix
/// the decode silently failed when any test populated measurements
/// before its result crossed the wire. The externally-tagged default
/// (JSON form `{"Int": 42}` / `{"Text": "x"}`) is what postcard's
/// externally-tagged enum decoder expects. The
/// `assert_result_postcard_roundtrip` test pins this contract so a
/// regression that re-adds `#[serde(untagged)]` trips at test time
/// rather than as a silent data drop at runtime.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum NoteValue {
    /// 64-bit signed integer — pid_t, exit codes, signed counters.
    Int(i64),
    /// 64-bit unsigned integer — work_units, byte counts, durations.
    Uint(u64),
    /// 64-bit float — ratios, rates, percentiles in microseconds.
    Float(f64),
    /// Boolean — completion flags, feature-detect results.
    Bool(bool),
    /// Owned string — categorical labels, environment tokens.
    Text(String),
}

impl From<i64> for NoteValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<u64> for NoteValue {
    fn from(v: u64) -> Self {
        Self::Uint(v)
    }
}
impl From<f64> for NoteValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<bool> for NoteValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<String> for NoteValue {
    fn from(v: String) -> Self {
        Self::Text(v)
    }
}
impl From<&str> for NoteValue {
    fn from(v: &str) -> Self {
        Self::Text(v.to_string())
    }
}

/// Terminal verdict for a single test scenario or merge fold —
/// strict four-state enum that replaces the `(passed, skipped)`
/// bool-pair encoding on [`AssertResult`].
///
/// Precedence under [`AssertResult::merge`]:
/// **`Fail > Inconclusive > Pass > Skip`**.
/// A merge that contains any `Fail` resolves to `Fail`; absent a
/// `Fail`, any `Inconclusive` resolves to `Inconclusive`; absent
/// both, a `Pass + Skip` mix resolves to `Pass` (Pass dominates
/// Skip — a check that actually ran and passed overrides a
/// sibling check whose precondition was unmet, so the merge does
/// not falsely demote to Skip on the strength of an unrelated
/// missing-precondition sibling). Skip-only merges stay Skip.
/// Pass-only merges stay Pass. Inconclusive sits between Fail
/// and Pass because "couldn't evaluate" is not a real Pass (an
/// Inconclusive run must not satisfy `is_pass()`-keyed CI gates)
/// but also not a hard Fail (no claim was made that the system
/// did the wrong thing).
///
/// `Inconclusive` exists for ratio assertions whose denominator
/// is an INSTRUMENT-derived measurement (iteration count, sample
/// count, wall-clock interval) that legitimately reached zero —
/// the gate has no signal to evaluate against. Distinguish from
/// `Fail`: a POLICY-derived denominator (e.g. NUMA pages under
/// `MemPolicy::Bind`, where the policy specifies pages will
/// exist) staying at zero IS a defect signal and stays as `Fail`
/// per the existing semantic — see `assert_page_locality` /
/// `AssertPlan::assert_cgroup` for the policy-derived carve-out.
///
/// Note: Notes do NOT belong here. [`AssertResult::info_notes`]
/// is the structurally-separate context stream; re-encoding Note
/// as an `Outcome` variant would re-mix the failure / verdict
/// surface with the context surface and erase the separation.
/// Outcome is strictly terminal verdict; notes are non-verdict
/// context.
///
/// `Skip`, `Inconclusive`, and `Fail` carry an [`AssertDetail`]
/// payload so the match arm has the diagnostic in hand without
/// re-walking `details`. `Pass` carries no payload — there is no
/// failure to describe.
///
/// Outcomes are stored as [`AssertResult::outcomes`] and the
/// [`AssertResult::outcome`] accessor folds the vec via this enum's
/// [`Self::merge`] (identity = `Outcome::Pass`). Callers query via
/// [`AssertResult::is_pass`] / [`AssertResult::is_fail`] /
/// [`AssertResult::is_skip`] / [`AssertResult::is_inconclusive`]
/// (bool checks), [`AssertResult::record_fail`] /
/// [`AssertResult::record_skip`] / [`AssertResult::record_pass`] /
/// [`AssertResult::record_inconclusive`] (atomic mutators), or
/// [`AssertResult::failure_details`] / [`AssertResult::skip_details`] /
/// [`AssertResult::inconclusive_details`] (per-variant payload
/// iterators).
///
/// **Skip is not Pass**: `is_pass()` returns `false` on skip — a
/// skipped scenario is "couldn't run", not "passed". Stats tooling
/// and gate callers that want to count "not a failure" must test
/// `r.is_pass() || r.is_skip()` rather than bare `r.is_pass()`.
/// **Inconclusive is not Pass either**: `is_pass()` returns `false`
/// when any Inconclusive is recorded, so a zero-denominator ratio
/// gate cannot silently satisfy an `is_pass()`-keyed CI check.
/// Uses serde's externally-tagged default (no `#[serde(tag,
/// content)]`). Most ktstr enums adopt the adjacently-tagged
/// `#[serde(tag = "kind", content = "data")]` style for JSON
/// readability, but `Outcome` is uniquely wire-encoded via
/// postcard as part of [`AssertResult`]'s TLV transport from
/// guest to host (see
/// `crate::test_support::output::parse_assert_result_from_drain`
/// and `crate::test_support::test_helpers::assert_result_tlv_entry`).
/// Postcard is not a self-describing format and cannot decode
/// adjacently-tagged enums — pre-fix the decode silently failed and
/// surfaced as `ERR_NO_TEST_FUNCTION_OUTPUT`. The externally-tagged
/// default is what postcard's externally-tagged enum decoder
/// expects. `tests_assert.rs::outcome_serde_externally_tagged_*`
/// pins both the JSON shape and the postcard roundtrip so a
/// refactor that re-adds adjacent tagging trips loudly at test
/// time rather than at runtime.
///
/// # Wire-format stability (postcard variant index)
///
/// Postcard encodes externally-tagged enums by **variant index**,
/// not variant name — the integer position in the `enum` body
/// becomes part of the wire format. The current encoding is:
/// `Pass=0`, `Skip=1`, `Inconclusive=2`, `Fail=3`.
///
/// **Append-only:** new variants MUST be added at the END of the
/// variant list. Re-ordering, removing, or inserting a variant
/// shifts the index of every variant after it and silently
/// reinterprets in-flight bytes from guest payloads as a
/// different variant on the host — the failure mode is a `Pass`
/// reading as `Skip` (or vice versa) with no decode error.
///
/// Any change to the variant order or list MUST be accompanied
/// by an update to `tests_assert.rs::outcome_serde_externally_tagged_*`
/// (which pins the JSON shape and the postcard roundtrip) and
/// `tests_assert.rs::outcome_postcard_variant_index_byte_sentinel`
/// (which pins the leading variant-index byte) so a silent-shift
/// regression trips at test time.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Outcome {
    // Wire-format-stable: variant indices encode into postcard
    // bytes (Pass=0, Skip=1, Inconclusive=2, Fail=3). Append new
    // variants ONLY at the end of this list — see the enum doc's
    // "Wire-format stability" section for the silent-shift hazard
    // a reorder introduces.
    Pass,
    Skip(AssertDetail),
    Inconclusive(AssertDetail),
    Fail(AssertDetail),
}

impl Outcome {
    /// True iff `self == Outcome::Pass`.
    ///
    /// Part of the `is_pass` / `is_fail` / `is_inconclusive` /
    /// `is_skip` vocabulary uniform across the verdict surfaces:
    /// [`crate::assert::AssertResult::is_pass`] /
    /// [`crate::test_support::SidecarResult::is_pass`] /
    /// [`Self::is_pass`] / `MonitorVerdict::is_pass` (in the
    /// `monitor` module, which is `pub(crate)`) / `Verdict::is_pass`
    /// (re-exported at [`crate::assert::Verdict`]) /
    /// `GauntletRow::is_pass` (in the `stats` module, which is
    /// `pub(crate)`). [`OutcomeRef::is_pass`] is a borrowed-view
    /// twin of this method on the borrowed [`OutcomeRef`] enum and
    /// is intentionally NOT counted as a peer surface — it shares
    /// the boolean semantic for naming parity but is a `&self`
    /// projection over [`Outcome`], not an independent verdict
    /// shape.
    pub fn is_pass(&self) -> bool {
        matches!(self, Outcome::Pass)
    }

    /// True iff `self == Outcome::Skip(_)`.
    pub fn is_skip(&self) -> bool {
        matches!(self, Outcome::Skip(_))
    }

    /// True iff `self == Outcome::Fail(_)`.
    pub fn is_fail(&self) -> bool {
        matches!(self, Outcome::Fail(_))
    }

    /// True iff `self == Outcome::Inconclusive(_)`.
    pub fn is_inconclusive(&self) -> bool {
        matches!(self, Outcome::Inconclusive(_))
    }

    /// Merge two outcomes per the precedence
    /// `Fail > Inconclusive > Pass > Skip`.
    ///
    /// Discriminant-commutative: the merged Pass/Skip/Inconclusive/Fail
    /// kind is the same regardless of operand order. Idempotent on
    /// Pass (`Pass.merge(Pass) == Pass`).
    ///
    /// Payload semantic (NOT commutative):
    /// - Same-variant ties (Fail+Fail, Inconclusive+Inconclusive,
    ///   Skip+Skip): the LEFT operand's payload wins, so caller-
    ///   controlled merge ordering produces deterministic detail
    ///   content.
    /// - Cross-variant Fail+{Inconclusive,Skip}: the merged outcome is
    ///   Fail and the payload comes from whichever side carries the
    ///   Fail (the dominated side's payload is dropped — the merged
    ///   verdict is Fail, so the dominated narrative is irrelevant to
    ///   the failure record).
    /// - Cross-variant Inconclusive+{Pass,Skip}: merged outcome is
    ///   Inconclusive and the payload comes from whichever side
    ///   carries the Inconclusive.
    pub fn merge(self, other: Outcome) -> Outcome {
        use Outcome::*;
        match (self, other) {
            (Fail(d), _) | (_, Fail(d)) => Fail(d),
            (Inconclusive(d), _) | (_, Inconclusive(d)) => Inconclusive(d),
            (Pass, _) | (_, Pass) => Pass,
            (Skip(d), Skip(_)) => Skip(d),
        }
    }

    /// Borrow this outcome's payload as an [`OutcomeRef`]. Zero-
    /// allocation projection — `Pass` carries no payload; `Skip`,
    /// `Inconclusive`, and `Fail` borrow their [`AssertDetail`] in
    /// place. Used by
    /// the verdict-read fast path
    /// ([`AssertResult::outcome_ref`]) and any caller that wants
    /// to inspect the terminal verdict without cloning the
    /// detail (e.g. error-message formatting where the detail
    /// outlives the formatter, or sidecar emission that already
    /// owns the source `Outcome`).
    pub fn as_ref(&self) -> OutcomeRef<'_> {
        match self {
            Outcome::Pass => OutcomeRef::Pass,
            Outcome::Skip(d) => OutcomeRef::Skip(d),
            Outcome::Inconclusive(d) => OutcomeRef::Inconclusive(d),
            Outcome::Fail(d) => OutcomeRef::Fail(d),
        }
    }
}

/// Borrowed view of an [`Outcome`]: same four discriminants but
/// the `Skip`, `Inconclusive`, and `Fail` payloads borrow their
/// [`AssertDetail`] in place. Returned by [`Outcome::as_ref`] and
/// the zero-clone verdict-read fast path
/// [`AssertResult::outcome_ref`].
///
/// Use when the caller wants the terminal verdict shape (or its
/// payload) WITHOUT taking ownership — typical sites are
/// formatter and sidecar paths that already hold the source
/// `AssertResult` and want to avoid the per-call
/// `AssertDetail::clone` the owned [`Outcome`] accessor performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeRef<'a> {
    Pass,
    Skip(&'a AssertDetail),
    Inconclusive(&'a AssertDetail),
    Fail(&'a AssertDetail),
}

impl OutcomeRef<'_> {
    /// True iff `self == OutcomeRef::Pass`. Matches the boolean
    /// shape of [`Outcome::is_pass`] for naming parity.
    pub fn is_pass(&self) -> bool {
        matches!(self, OutcomeRef::Pass)
    }
    /// True iff `self == OutcomeRef::Skip(_)`. Matches the
    /// boolean shape of [`Outcome::is_skip`].
    pub fn is_skip(&self) -> bool {
        matches!(self, OutcomeRef::Skip(_))
    }
    /// True iff `self == OutcomeRef::Fail(_)`. Matches the
    /// boolean shape of [`Outcome::is_fail`].
    pub fn is_fail(&self) -> bool {
        matches!(self, OutcomeRef::Fail(_))
    }
    /// True iff `self == OutcomeRef::Inconclusive(_)`. Matches the
    /// boolean shape of [`Outcome::is_inconclusive`].
    pub fn is_inconclusive(&self) -> bool {
        matches!(self, OutcomeRef::Inconclusive(_))
    }
    /// Promote a borrowed [`OutcomeRef`] into an owned [`Outcome`]
    /// by cloning the borrowed [`AssertDetail`] (when present).
    /// `OutcomeRef::Pass` carries no payload so the conversion is
    /// allocation-free. Pairs with [`Outcome::as_ref`] for the
    /// borrow ↔ own round-trip; [`AssertResult::outcome`] delegates
    /// here so the fold logic stays single-sourced in
    /// [`AssertResult::outcome_ref`] and any future fold-rule
    /// change (e.g. a new terminal arm) lands in one place.
    pub fn to_owned(&self) -> Outcome {
        match self {
            OutcomeRef::Pass => Outcome::Pass,
            OutcomeRef::Skip(d) => Outcome::Skip((*d).clone()),
            OutcomeRef::Inconclusive(d) => Outcome::Inconclusive((*d).clone()),
            OutcomeRef::Fail(d) => Outcome::Fail((*d).clone()),
        }
    }
}
